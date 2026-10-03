"""Unit and integration tests for AURORA Model-Based RL reproductions."""

from __future__ import annotations

import numpy as np
import pytest
from aurora.tensor import tensor

from reproductions.decision_transformer import DecisionTransformer, DecisionTransformerTrainer
from reproductions.dreamer import DreamerActor, DreamerAgent, DreamerCritic, compute_lambda_returns
from reproductions.mbpo import MBPO
from reproductions.muzero import MuZeroAgent
from reproductions.tdmpc import TDMPCAgent


def test_mbpo_workflow() -> None:
    """Test MBPO dynamics training, model branching rollout, and hybrid SAC updates."""
    obs_dim = 4
    action_dim = 2
    mbpo = MBPO(
        obs_dim=obs_dim,
        action_dim=action_dim,
        ensemble_size=3,
        rollout_horizon=2,
        model_ratio=0.5,
        env_buffer_capacity=1000,
        model_buffer_capacity=1000,
    )

    # Populate real environment buffer
    rng = np.random.RandomState(42)
    for _ in range(50):
        s = rng.randn(obs_dim)
        a = np.clip(rng.randn(action_dim), -1.0, 1.0)
        r = float(rng.randn())
        s_next = s + 0.1 * a.sum()
        done = False
        mbpo.env_buffer.add(s, a, r, s_next, done)

    assert len(mbpo.env_buffer) == 50

    # 1. Train dynamics ensemble
    loss = mbpo.train_dynamics(batch_size=16, num_epochs=2)
    assert np.isfinite(loss)

    # 2. Branch rollouts
    num_rollouts = 10
    transitions_added = mbpo.rollout_model(num_rollouts=num_rollouts, horizon=2)
    assert transitions_added == num_rollouts * 2
    assert len(mbpo.model_buffer) == num_rollouts * 2

    # 3. Hybrid batch sampling
    mixed_batch = mbpo.sample_mixed_batch(batch_size=16)
    assert mixed_batch.obs.shape == (16, obs_dim)
    assert mixed_batch.actions.shape == (16, action_dim)

    # 4. Train policy
    metrics = mbpo.train_policy(num_updates=2, batch_size=16)
    assert "critic_loss" in metrics
    assert "actor_loss" in metrics
    assert np.isfinite(metrics["critic_loss"])
    assert np.isfinite(metrics["actor_loss"])


def test_dreamer_lambda_returns_and_actor_critic() -> None:
    """Verify Dreamer lambda-returns formula and actor-critic networks."""
    # 1. Exact mathematical check on lambda-returns
    # 2 steps: H=2
    # rewards: r0=1.0, r1=2.0
    # discounts: g0=0.9, g1=0.9
    # values: v0=0.5, v1=1.5, v2=2.5
    # lambda = 0.8
    # V_2^lambda = v2 = 2.5
    # V_1^lambda = r1 + g1 * ((1 - lambda) * v2 + lambda * V_2^lambda) = 4.25
    # V_0^lambda = r0 + g0 * ((1 - lambda) * v1 + lambda * V_1^lambda) = 4.33
    rewards = np.array([[1.0], [2.0]], dtype=np.float64)  # (2, 1)
    discounts = np.array([[0.9], [0.9]], dtype=np.float64)  # (2, 1)
    values = np.array([[0.5], [1.5], [2.5]], dtype=np.float64)  # (3, 1)

    returns = compute_lambda_returns(rewards, discounts, values, lambda_=0.8)
    assert returns.shape == (2, 1)
    np.testing.assert_allclose(returns[1, 0], 4.25, atol=1e-10)
    np.testing.assert_allclose(returns[0, 0], 4.33, atol=1e-10)

    # 2. Dreamer Actor & Critic modules
    latent_dim = 16
    action_dim = 2
    actor = DreamerActor(latent_dim=latent_dim, action_dim=action_dim, hidden_dims=[32])
    critic = DreamerCritic(latent_dim=latent_dim, hidden_dims=[32])

    feat = tensor(np.random.randn(4, latent_dim))
    act, logp = actor(feat)
    assert act.shape == (4, action_dim)
    assert logp.shape == (4,)

    v = critic(feat)
    assert v.shape == (4, 1)

    # 3. Full DreamerAgent check
    agent = DreamerAgent(
        obs_dim=6,
        action_dim=2,
        deter_dim=16,
        stoch_dim=8,
        hidden_dim=32,
        imagination_horizon=3,
    )
    obs = np.random.randn(6)
    action = agent.select_action(obs)
    assert action.shape == (2,)
    assert agent.prev_state is not None
    assert agent.prev_state.h.shape == (1, 16)
    assert agent.prev_state.z.shape == (1, 8)


def test_tdmpc_cem_planner_and_training() -> None:
    """Test TD-MPC latent trajectory planning and multi-task learning."""
    obs_dim = 4
    action_dim = 2
    latent_dim = 16

    agent = TDMPCAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        latent_dim=latent_dim,
        hidden_dims=[32, 32],
        planning_horizon=3,
        num_samples=15,
        num_elites=5,
        cem_iterations=3,
    )

    # Test CEM action planning
    obs = np.random.randn(obs_dim).astype(np.float32)
    action = agent.plan(obs)
    assert action.shape == (action_dim,)
    assert np.all(action >= -1.0) and np.all(action <= 1.0)

    # Test model & critic & policy multi-task update
    batch = {
        "obs": tensor(np.random.randn(8, obs_dim)),
        "actions": tensor(np.random.randn(8, action_dim)),
        "rewards": tensor(np.random.randn(8, 1)),
        "next_obs": tensor(np.random.randn(8, obs_dim)),
        "dones": tensor(np.zeros((8, 1))),
    }

    losses = agent.update(batch)
    assert "dyn_loss" in losses
    assert "reward_loss" in losses
    assert "critic_loss" in losses
    assert "policy_loss" in losses
    for v in losses.values():
        assert np.isfinite(v)


def test_muzero_mcts_and_unroll() -> None:
    """Test MuZero latent representation, dynamics, PUCT MCTS, and unroll training."""
    obs_dim = 4
    action_dim = 3  # discrete actions: 0, 1, 2
    latent_dim = 16

    agent = MuZeroAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        latent_dim=latent_dim,
        hidden_dims=[32, 32],
    )

    # Test MCTS planning
    obs = np.random.randn(obs_dim).astype(np.float32)
    best_act, pi_probs = agent.plan_and_act(obs, num_simulations=15)
    assert 0 <= best_act < action_dim
    assert pi_probs.shape == (action_dim,)
    assert pytest.approx(1.0, rel=1e-5) == float(np.sum(pi_probs))

    # Test multi-task update step
    obs_t = tensor(obs.reshape(1, -1))
    actions = [0, 1]
    target_rewards = [1.0, 0.5]
    target_values = [2.0, 1.5]
    target_policies = [np.array([0.7, 0.2, 0.1]), np.array([0.1, 0.8, 0.1])]

    info = agent.update_step(obs_t, actions, target_rewards, target_values, target_policies)
    assert "loss" in info
    assert np.isfinite(info["loss"])


def test_decision_transformer_forward_and_trainer() -> None:
    """Test Decision Transformer sequence processing, causal masking, and trainer."""
    state_dim = 4
    act_dim = 2
    d_model = 32
    seq_len = 5
    batch_size = 2

    dt = DecisionTransformer(
        state_dim=state_dim,
        act_dim=act_dim,
        d_model=d_model,
        n_heads=2,
        n_layers=2,
        max_length=10,
        max_ep_len=100,
        dropout=0.0,
    )

    states = tensor(np.random.randn(batch_size, seq_len, state_dim))
    actions = tensor(np.random.randn(batch_size, seq_len, act_dim))
    rtgs = tensor(np.random.randn(batch_size, seq_len, 1))
    timesteps = tensor(np.tile(np.arange(seq_len), (batch_size, 1)), dtype=np.int64)

    # 1. Forward pass
    preds = dt(states, actions, rtgs, timesteps)
    assert preds.shape == (batch_size, seq_len, act_dim)
    # Output must be tanh bounded
    assert np.all(preds.data >= -1.0) and np.all(preds.data <= 1.0)

    # 2. Autoregressive inference: get_action
    s_hist = np.random.randn(seq_len, state_dim)
    a_hist = np.random.randn(seq_len, act_dim)
    r_hist = np.random.randn(seq_len, 1)
    t_hist = np.arange(seq_len)
    next_action = dt.get_action(s_hist, a_hist, r_hist, t_hist)
    assert next_action.shape == (act_dim,)
    assert np.all(next_action >= -1.0) and np.all(next_action <= 1.0)

    # 3. Trainer step
    trainer = DecisionTransformerTrainer(dt, lr=1e-3)
    loss = trainer.train_step(states, actions, rtgs, timesteps)
    assert np.isfinite(loss)
