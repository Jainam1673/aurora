"""Reproducible multi-seed experiment execution harness."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from aurora.algorithm.aurora_agent import AURORAAgent
from aurora.environments import Pendulum

from evaluation.manifest import create_experiment_manifest, save_manifest
from evaluation.metrics import compute_statistical_summary


def evaluate_policy(
    agent: AURORAAgent,
    env: Pendulum,
    num_episodes: int = 5,
    base_seed: int = 1000,
) -> list[float]:
    """Evaluate agent deterministically over multiple test episodes."""
    returns = []
    for ep in range(num_episodes):
        obs, _ = env.reset(seed=base_seed + ep)
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
    return returns


def run_experiment(
    config: dict[str, Any] | str | Path,
    seed: int = 42,
    output_dir: str | Path | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """Execute a single experimental run with declarative configuration and manifest logging."""
    if isinstance(config, (str, Path)):
        with open(config, encoding="utf-8") as f:
            cfg = json.load(f)
    else:
        cfg = config

    exp_name = cfg.get("experiment_name", "aurora_experiment")
    agent_cfg = cfg.get("agent", {})
    train_cfg = cfg.get("training", {})
    eval_cfg = cfg.get("evaluation", {})

    np.random.seed(seed)
    env = Pendulum(max_steps=cfg.get("environment", {}).get("max_steps", 200))

    # Configure agent
    agent = AURORAAgent(
        obs_dim=3,
        action_dim=1,
        ensemble_size=agent_cfg.get("ensemble_size", 3),
        ensemble_hidden_dims=agent_cfg.get("ensemble_hidden_dims", [48, 48]),
        actor_hidden_dims=agent_cfg.get("actor_hidden_dims", [48, 48]),
        critic_hidden_dims=agent_cfg.get("critic_hidden_dims", [48, 48]),
        horizon_max=agent_cfg.get("horizon_max", 8),
        horizon_min=agent_cfg.get("horizon_min", 1),
        tau_base=agent_cfg.get("tau_base", 0.4),
        kappa=agent_cfg.get("kappa", 1.0),
        budget_max=agent_cfg.get("budget_max", 2.0),
        beta_pess=agent_cfg.get("beta_pess", 0.5),
        eta_max=agent_cfg.get("eta_max", 0.85),
        u_target=agent_cfg.get("u_target", 0.4),
        env_buffer_capacity=10_000,
        model_buffer_capacity=20_000,
    )

    # Disable adaptive mechanisms if ablation flags are set
    if not agent_cfg.get("dynamic_blending", True):
        fixed_eta = agent_cfg.get("fixed_eta", 0.5)
        agent.blending_controller.current_eta = fixed_eta
        agent.blending_controller.compute_ratio = lambda _: fixed_eta  # type: ignore

    warmup_steps = train_cfg.get("warmup_steps", 50)
    env_steps = train_cfg.get("env_steps", 300)
    model_freq = train_cfg.get("model_train_freq", 20)
    model_epochs = train_cfg.get("model_epochs", 2)
    num_rollouts = train_cfg.get("num_rollouts", 15)
    batch_size = train_cfg.get("policy_batch_size", 32)
    eval_freq = eval_cfg.get("eval_freq", 100)
    eval_episodes = eval_cfg.get("eval_episodes", 3)

    obs, _ = env.reset(seed=seed)
    total_reward = 0.0

    # 1. Warmup
    for _ in range(warmup_steps):
        action = np.random.uniform(-2.0, 2.0, size=(1,))
        next_obs, reward, terminated, truncated, _ = env.step(action)
        agent.add_experience(obs, action, reward, next_obs, terminated or truncated)
        obs = next_obs if not (terminated or truncated) else env.reset()[0]

    eval_history: list[dict[str, Any]] = []

    # 2. Main Training Loop
    for step in range(1, env_steps + 1):
        action = agent.select_action(obs, evaluate=False)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        agent.add_experience(obs, action, reward, next_obs, terminated or truncated)
        total_reward += reward
        obs = next_obs if not (terminated or truncated) else env.reset()[0]

        # Model updates & imagination rollouts
        if step % model_freq == 0:
            agent.train_dynamics(batch_size=batch_size, num_epochs=model_epochs)
            agent.rollout_adaptive_imagination(num_rollouts=num_rollouts)

        # Policy training step
        agent.train_policy(num_updates=1, batch_size=batch_size)

        # Periodic Evaluation
        if step % eval_freq == 0 or step == env_steps:
            eval_rets = evaluate_policy(
                agent, env, num_episodes=eval_episodes, base_seed=seed * 100
            )
            mean_eval = float(np.mean(eval_rets))
            eval_history.append({"step": step, "returns": eval_rets, "mean_return": mean_eval})
            if verbose:
                msg = (
                    f"[{exp_name} | Seed {seed}] Step {step:4d} | "
                    f"Eval Mean Return: {mean_eval:8.2f}"
                )
                print(msg)

    # Final evaluation with larger test sample
    final_eval_returns = evaluate_policy(agent, env, num_episodes=10, base_seed=seed * 1000)
    summary = compute_statistical_summary(final_eval_returns, seed=seed)

    metrics = {
        "final_returns": final_eval_returns,
        "statistical_summary": summary.to_dict(),
        "train_return": total_reward,
        "eval_history": eval_history,
        "synthetic_buffer_size": len(agent.model_buffer),
        "mean_horizon": agent.last_mean_horizon,
    }

    manifest = create_experiment_manifest(
        experiment_id=f"{exp_name}_seed{seed}",
        environment_name="Pendulum",
        algorithm_name=agent_cfg.get("type", "AURORA"),
        hyperparameters=cfg,
        metrics=metrics,
        seed=seed,
    )

    if output_dir is not None:
        out_p = Path(output_dir) / f"{exp_name}_seed{seed}"
        save_manifest(manifest, out_p / "manifest.json")

    return manifest
