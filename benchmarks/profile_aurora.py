"""AURORA Systems Profiling & Bottleneck Breakdown Tool.

Profiles the complete online training pipeline of AURORA:
1. Real environment interaction (env.step, action selection, buffer addition)
2. Ensemble world model training (Gaussian NLL loss backward & step)
3. Adaptive uncertainty imagination rollouts (dynamics prediction, horizon truncation)
4. Policy & Critic optimization (pessimistic Bellman targets, SAC updates)
5. Periodic deterministic policy evaluation

Computes Amdahl's law time fractions (phi_m), latency distributions (p50, p90, p99),
and exports cProfile flamegraph traces and structured JSON reports.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import json
import pstats
import time
from pathlib import Path
from typing import Any

import numpy as np
from aurora.algorithm.aurora_agent import AURORAAgent
from aurora.environments import Pendulum


def profile_aurora_pipeline(
    env_steps: int = 150,
    ensemble_size: int = 3,
    rollout_batch_size: int = 32,
    train_dynamics_interval: int = 50,
    policy_updates_per_step: int = 1,
    seed: int = 42,
    output_json: str | None = None,
    prof_file: str | None = None,
) -> dict[str, Any]:
    """Execute rigorous systems profiling across all pipeline components."""
    np.random.seed(seed)
    env = Pendulum(max_steps=200)

    agent = AURORAAgent(
        obs_dim=3,
        action_dim=1,
        ensemble_size=ensemble_size,
        ensemble_hidden_dims=[48, 48],
        actor_hidden_dims=[48, 48],
        critic_hidden_dims=[48, 48],
        horizon_max=8,
        horizon_min=1,
        tau_base=0.4,
        beta_pess=0.5,
        eta_max=0.85,
        env_buffer_capacity=10_000,
        model_buffer_capacity=20_000,
    )

    latencies: dict[str, list[float]] = {
        "env_step": [],
        "world_model_train": [],
        "adaptive_imagination": [],
        "policy_update": [],
        "evaluation": [],
    }

    # 1. Warmup real buffer with 50 transitions
    obs, _ = env.reset(seed=seed)
    for _ in range(50):
        act = np.random.uniform(-2.0, 2.0, size=(1,))
        next_obs, rew, terminated, truncated, _ = env.step(act)
        agent.add_experience(obs, act, rew, next_obs, terminated or truncated)
        obs = next_obs if not (terminated or truncated) else env.reset()[0]

    agent.train_dynamics(batch_size=32, num_epochs=2)

    # 2. Online Profiling Loop with cProfile and High-Precision Timers
    profiler = cProfile.Profile()
    profiler.enable()

    t_start_total = time.perf_counter_ns()

    for step in range(env_steps):
        # A. Environment Interaction
        t0 = time.perf_counter_ns()
        action = agent.select_action(obs, evaluate=False)
        next_obs, rew, terminated, truncated, _ = env.step(action)
        agent.add_experience(obs, action, rew, next_obs, terminated or truncated)
        obs = next_obs if not (terminated or truncated) else env.reset()[0]
        t1 = time.perf_counter_ns()
        latencies["env_step"].append((t1 - t0) / 1000.0)

        # B. World Model Periodic Training
        if step > 0 and step % train_dynamics_interval == 0:
            t0 = time.perf_counter_ns()
            agent.train_dynamics(batch_size=32, num_epochs=3)
            t1 = time.perf_counter_ns()
            latencies["world_model_train"].append((t1 - t0) / 1000.0)

        # C. Adaptive Imagination Rollouts
        if step % 5 == 0:
            t0 = time.perf_counter_ns()
            agent.generate_adaptive_rollouts(num_rollouts=rollout_batch_size)
            t1 = time.perf_counter_ns()
            latencies["adaptive_imagination"].append((t1 - t0) / 1000.0)

        # D. Policy & Critic Updates
        t0 = time.perf_counter_ns()
        agent.train_policy(num_updates=policy_updates_per_step, batch_size=32)
        t1 = time.perf_counter_ns()
        latencies["policy_update"].append((t1 - t0) / 1000.0)

        # E. Periodic Evaluation
        if (step + 1) % 50 == 0:
            t0 = time.perf_counter_ns()
            eval_obs, _ = env.reset(seed=999)
            for _ in range(50):
                eval_act = agent.select_action(eval_obs, evaluate=True)
                eval_obs, _, eval_term, eval_trunc, _ = env.step(eval_act)
                if eval_term or eval_trunc:
                    break
            t1 = time.perf_counter_ns()
            latencies["evaluation"].append((t1 - t0) / 1000.0)

    t_end_total = time.perf_counter_ns()
    profiler.disable()

    total_wall_sec = (t_end_total - t_start_total) / 1e9

    # 3. Statistical Analysis
    component_stats: dict[str, dict[str, float]] = {}
    total_component_time_us = sum(sum(vals) for vals in latencies.values())

    for name, times in latencies.items():
        if not times:
            component_stats[name] = {
                "count": 0,
                "total_sec": 0.0,
                "fraction_pct": 0.0,
                "mean_us": 0.0,
                "p50_us": 0.0,
                "p90_us": 0.0,
                "p99_us": 0.0,
                "throughput_per_sec": 0.0,
            }
            continue

        arr = np.array(times)
        total_sec = float(np.sum(arr) / 1e6)
        if total_component_time_us > 0:
            frac = (np.sum(arr) / total_component_time_us) * 100.0
        else:
            frac = 0.0

        component_stats[name] = {
            "count": len(arr),
            "total_sec": total_sec,
            "fraction_pct": float(frac),
            "mean_us": float(np.mean(arr)),
            "p50_us": float(np.percentile(arr, 50)),
            "p90_us": float(np.percentile(arr, 90)),
            "p99_us": float(np.percentile(arr, 99)),
            "throughput_per_sec": float(len(arr) / total_sec) if total_sec > 0 else 0.0,
        }

    report = {
        "metadata": {
            "env_steps": env_steps,
            "ensemble_size": ensemble_size,
            "rollout_batch_size": rollout_batch_size,
            "total_wall_clock_sec": total_wall_sec,
            "overall_env_steps_per_sec": float(env_steps / total_wall_sec),
        },
        "components": component_stats,
    }

    # 4. Print Summary Table
    print("\n" + "=" * 90)
    print("                     AURORA SYSTEMS PROFILING REPORT                      ")
    print("=" * 90)
    print(f"Steps: {env_steps} | Ensemble: {ensemble_size} | Wall-clock: {total_wall_sec:.2f}s")
    print(f"Overall Throughput: {env_steps / total_wall_sec:.2f} env steps/sec")
    print("-" * 90)
    header = (
        f"{'Component':<22} {'Time (s)':<10} {'Share (%)':<10} "
        f"{'Mean (us)':<12} {'p50 (us)':<12} {'p99 (us)':<12} {'Rate (/s)':<10}"
    )
    print(header)
    print("-" * 90)
    for name, c in component_stats.items():
        row = (
            f"{name:<22} {c['total_sec']:<10.3f} {c['fraction_pct']:<10.1f} "
            f"{c['mean_us']:<12.1f} {c['p50_us']:<12.1f} {c['p99_us']:<12.1f} "
            f"{c['throughput_per_sec']:<10.1f}"
        )
        print(row)
    print("=" * 90 + "\n")

    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
    ps.print_stats(15)
    print("Top 15 Call-Stack Bottlenecks (cProfile):")
    print("-" * 90)
    print(s.getvalue())

    if prof_file:
        Path(prof_file).parent.mkdir(parents=True, exist_ok=True)
        profiler.dump_stats(prof_file)
        print(f"cProfile binary stats saved to: {prof_file}")

    if output_json:
        Path(output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Structured systems profile exported to: {output_json}")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="AURORA Systems Profiling Tool")
    parser.add_argument("--env-steps", type=int, default=100, help="Env steps to profile")
    parser.add_argument("--ensemble-size", type=int, default=3, help="Dynamics ensemble size")
    parser.add_argument("--rollout-batch", type=int, default=32, help="Rollout batch size")
    parser.add_argument(
        "--output-json",
        type=str,
        default="results/systems_profile.json",
        help="Path for JSON output",
    )
    parser.add_argument(
        "--prof-file",
        type=str,
        default="results/aurora_profile.prof",
        help="Path for cProfile dump",
    )
    args = parser.parse_args()

    profile_aurora_pipeline(
        env_steps=args.env_steps,
        ensemble_size=args.ensemble_size,
        rollout_batch_size=args.rollout_batch,
        output_json=args.output_json,
        prof_file=args.prof_file,
    )


if __name__ == "__main__":
    main()
