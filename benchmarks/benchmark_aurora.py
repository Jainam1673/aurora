"""AURORA Empirical Benchmark Suite.

Compares:
1. Novel AURORA Agent (adaptive horizon scheduling, dynamic uncertainty blending, pessimistic Q)
2. Fixed-Horizon MBPO (fixed horizon H=5, static synthetic ratio eta=0.5)
3. Model-Free SAC (eta=0.0, real experience only)

Measures sample efficiency, imagination throughput, and policy returns on Continuous Control.
"""

from __future__ import annotations

import time

import numpy as np
from aurora.algorithm.aurora_agent import AURORAAgent
from aurora.environments import Pendulum


def evaluate_agent(agent: AURORAAgent, env: Pendulum, num_episodes: int = 3) -> float:
    """Evaluate agent deterministically over multiple episodes."""
    returns = []
    for ep in range(num_episodes):
        obs, _ = env.reset(seed=1000 + ep)
        ep_ret = 0.0
        done = False
        step = 0
        while not done and step < 200:
            action = agent.select_action(obs, evaluate=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_ret += reward
            step += 1
            done = terminated or truncated
        returns.append(ep_ret)
    return float(np.mean(returns))


def run_aurora_benchmark(
    env_steps: int = 400,
    seed: int = 42,
    verbose: bool = True,
) -> dict[str, dict[str, float]]:
    """Execute comparative benchmark across AURORA, MBPO, and SAC."""
    results: dict[str, dict[str, float]] = {}

    configs = [
        ("AURORA (Adaptive)", {"adaptive": True, "pessimistic": True, "eta_max": 0.85}),
        (
            "MBPO (Fixed H=4)",
            {"adaptive": False, "fixed_h": 4, "pessimistic": False, "fixed_eta": 0.5},
        ),
        (
            "Model-Free SAC",
            {"adaptive": False, "fixed_h": 0, "pessimistic": False, "fixed_eta": 0.0},
        ),
    ]

    for name, cfg in configs:
        if verbose:
            print("\n==========================================")
            print(f"Benchmarking: {name}")
            print("==========================================")

        np.random.seed(seed)
        env = Pendulum(max_steps=200)

        agent = AURORAAgent(
            obs_dim=3,
            action_dim=1,
            ensemble_size=3,
            ensemble_hidden_dims=[48, 48],
            actor_hidden_dims=[48, 48],
            critic_hidden_dims=[48, 48],
            horizon_max=int(8 if cfg.get("adaptive") else cfg.get("fixed_h", 4)),
            horizon_min=int(1 if cfg.get("adaptive") else max(1, cfg.get("fixed_h", 1))),
            tau_base=0.4,
            beta_pess=0.5 if cfg.get("pessimistic") else 0.0,
            eta_max=cfg.get("eta_max", 0.8),
            env_buffer_capacity=10_000,
            model_buffer_capacity=20_000,
        )

        obs, _ = env.reset(seed=seed)
        total_reward = 0.0
        start_time = time.perf_counter()
        rollout_horizons: list[float] = []

        # Warmup real buffer with random exploratory actions
        for _ in range(50):
            action = np.random.uniform(-2.0, 2.0, size=(1,))
            next_obs, reward, terminated, truncated, _ = env.step(action)
            agent.add_experience(obs, action, reward, next_obs, terminated or truncated)
            obs = next_obs if not (terminated or truncated) else env.reset()[0]

        for step in range(env_steps):
            action = agent.select_action(obs, evaluate=False)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            agent.add_experience(obs, action, reward, next_obs, terminated or truncated)
            total_reward += reward
            obs = next_obs if not (terminated or truncated) else env.reset()[0]

            # Model training & rollouts
            if step % 20 == 0 and step > 0:
                if cfg.get("fixed_h", 1) > 0 or cfg.get("adaptive", False):
                    agent.train_dynamics(batch_size=32, num_epochs=2)
                    r_stats = agent.rollout_adaptive_imagination(num_rollouts=15)
                    rollout_horizons.append(r_stats["mean_horizon"])

            # Policy updates
            if step >= 50:
                agent.train_policy(num_updates=1, batch_size=32)

        elapsed = time.perf_counter() - start_time
        eval_score = evaluate_agent(agent, env)

        avg_h = (
            float(np.mean(rollout_horizons)) if rollout_horizons else float(cfg.get("fixed_h", 0))
        )

        results[name] = {
            "eval_return": eval_score,
            "train_return": total_reward,
            "elapsed_seconds": elapsed,
            "mean_horizon": avg_h,
            "synthetic_buffer_size": float(len(agent.model_buffer)),
        }

        if verbose:
            print(f"Eval Return: {eval_score:.2f}")
            print(f"Mean Rollout Horizon: {avg_h:.2f}")
            print(f"Synthetic Samples Generated: {len(agent.model_buffer)}")
            print(f"Total Time: {elapsed:.2f}s ({env_steps / elapsed:.1f} steps/s)")

    if verbose:
        print("\n" + "=" * 65)
        print(f"{'Method':<20} | {'Eval Return':<12} | {'Horizon':<8} | {'Time (s)':<8}")
        print("-" * 65)
        for method, m in results.items():
            ret = m["eval_return"]
            horiz = m["mean_horizon"]
            sec = m["elapsed_seconds"]
            print(f"{method:<20} | {ret:<12.2f} | {horiz:<8.2f} | {sec:<8.2f}")
        print("=" * 65)

    return results


if __name__ == "__main__":
    run_aurora_benchmark(env_steps=200, seed=42, verbose=True)
