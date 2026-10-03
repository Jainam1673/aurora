"""Unit and integration tests for the novel AURORA algorithm."""

from __future__ import annotations

import math

import numpy as np
import pytest
from aurora.algorithm.aurora_agent import AURORAAgent
from aurora.algorithm.scheduler import AdaptiveHorizonScheduler, DynamicBlendingController
from aurora.environments import Pendulum


class TestAdaptiveHorizonScheduler:
    """Tests for uncertainty-aware adaptive rollout horizon scheduling."""

    def test_threshold_decay(self) -> None:
        scheduler = AdaptiveHorizonScheduler(tau_base=0.8, kappa=2.0)
        assert scheduler.current_tau == 0.8

        # 0.0 validation loss -> tau_base
        tau0 = scheduler.update_threshold(0.0)
        assert pytest.approx(tau0, abs=1e-12) == 0.8

        # 0.5 validation loss -> 0.8 * exp(-1.0)
        tau_half = scheduler.update_threshold(0.5)
        assert pytest.approx(tau_half, abs=1e-12) == 0.8 * math.exp(-1.0)

        # >= 1.0 clamped validation loss -> 0.8 * exp(-2.0)
        tau1 = scheduler.update_threshold(2.5)
        assert pytest.approx(tau1, abs=1e-12) == 0.8 * math.exp(-2.0)

    def test_horizon_truncation_rules(self) -> None:
        scheduler = AdaptiveHorizonScheduler(
            horizon_min=2,
            horizon_max=8,
            tau_base=0.5,
            kappa=0.0,
            budget_max=1.0,
            gamma=1.0,
        )

        # Step 0: below horizon_min -> never truncated even with huge uncertainty
        trunc0, b0 = scheduler.should_truncate(
            step=0, epistemic_uncertainty=10.0, cumulative_budget=0.0
        )
        assert not trunc0
        assert pytest.approx(b0, abs=1e-12) == 10.0

        # Step 2: above horizon_min, peak uncertainty exceeded (> 0.5)
        trunc2, b2 = scheduler.should_truncate(
            step=2, epistemic_uncertainty=0.6, cumulative_budget=0.2
        )
        assert trunc2
        assert pytest.approx(b2, abs=1e-12) == 0.2

        # Step 3: peak uncertainty below tau, but cumulative budget exceeded (> 1.0)
        trunc3, b3 = scheduler.should_truncate(
            step=3, epistemic_uncertainty=0.4, cumulative_budget=0.8
        )
        assert trunc3
        assert pytest.approx(b3, abs=1e-12) == 1.2

        # Step 8: reaches horizon_max -> truncated
        trunc8, _ = scheduler.should_truncate(
            step=8, epistemic_uncertainty=0.1, cumulative_budget=0.2
        )
        assert trunc8

    def test_compute_adaptive_horizon(self) -> None:
        scheduler = AdaptiveHorizonScheduler(
            horizon_min=1,
            horizon_max=10,
            tau_base=0.5,
            budget_max=1.5,
            gamma=0.9,
        )

        # Low uncertainty trajectory
        low_u = [0.05] * 10
        assert scheduler.compute_adaptive_horizon(low_u) == 10

        # Trajectory with spike at step 3
        spike_u = [0.1, 0.1, 0.1, 0.9, 0.1, 0.1]
        assert scheduler.compute_adaptive_horizon(spike_u) == 3


class TestDynamicBlendingController:
    """Tests for dynamic experience replay blending."""

    def test_blending_ratio_and_smoothing(self) -> None:
        controller = DynamicBlendingController(
            eta_max=0.8,
            eta_min=0.1,
            u_target=0.4,
            momentum=0.5,
        )
        assert controller.current_eta == 0.1

        # Zero uncertainty -> raw eta = 0.8
        # eta_1 = 0.5 * 0.1 + 0.5 * 0.8 = 0.45
        eta1 = controller.compute_ratio(0.0)
        assert pytest.approx(eta1, abs=1e-12) == 0.45

        # Another zero uncertainty: eta_2 = 0.5 * 0.45 + 0.5 * 0.8 = 0.625
        eta2 = controller.compute_ratio(0.0)
        assert pytest.approx(eta2, abs=1e-12) == 0.625

        # Extreme uncertainty (1.0 > u_target): raw eta = 0.1
        # eta_3 = 0.5 * 0.625 + 0.5 * 0.1 = 0.3625
        eta3 = controller.compute_ratio(1.0)
        assert pytest.approx(eta3, abs=1e-12) == 0.3625


class TestAURORAAgent:
    """Integration tests for AURORAAgent end-to-end workflow."""

    @pytest.fixture
    def agent(self) -> AURORAAgent:
        return AURORAAgent(
            obs_dim=3,
            action_dim=1,
            ensemble_size=3,
            ensemble_hidden_dims=[32, 32],
            actor_hidden_dims=[32, 32],
            critic_hidden_dims=[32, 32],
            horizon_max=4,
            horizon_min=1,
            tau_base=0.5,
            beta_pess=0.5,
            env_buffer_capacity=1000,
            model_buffer_capacity=1000,
        )

    def test_action_selection(self, agent: AURORAAgent) -> None:
        obs = np.array([0.5, -0.2, 0.1], dtype=np.float32)

        # Deterministic action
        action_det = agent.select_action(obs, evaluate=True)
        assert action_det.shape == (1,)
        assert -1.0 <= action_det[0] <= 1.0

        # Stochastic action
        action_stoch = agent.select_action(obs, evaluate=False)
        assert action_stoch.shape == (1,)
        assert -1.0 <= action_stoch[0] <= 1.0

    def test_add_experience_and_training_step(self, agent: AURORAAgent) -> None:
        env = Pendulum()
        obs, _ = env.reset(seed=42)

        # Seed real buffer
        for _ in range(50):
            action = agent.select_action(obs, evaluate=False)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            agent.add_experience(obs, action, reward, next_obs, terminated or truncated)
            obs = next_obs if not (terminated or truncated) else env.reset()[0]

        assert len(agent.env_buffer) == 50

        # Train dynamics ensemble
        dynamics_metrics = agent.train_dynamics(batch_size=16, epochs=2)
        assert "dynamics_loss" in dynamics_metrics
        assert "tau_threshold" in dynamics_metrics
        assert not math.isnan(dynamics_metrics["dynamics_loss"])

        # Generate adaptive imagination rollouts
        rollout_metrics = agent.generate_adaptive_rollouts(num_rollouts=10, batch_size=10)
        assert "mean_horizon" in rollout_metrics
        assert "mean_epistemic_uncertainty" in rollout_metrics
        assert "blending_eta" in rollout_metrics
        assert len(agent.model_buffer) > 0

        # Sample mixed batch
        batch = agent.sample_mixed_batch(batch_size=16)
        assert batch.obs.shape[0] == 16
        assert batch.actions.shape[0] == 16

        # Train policy and critic step
        policy_metrics = agent.train_policy_step(batch_size=16)
        assert "critic_loss" in policy_metrics
        assert "actor_loss" in policy_metrics
        assert not math.isnan(policy_metrics["critic_loss"])
        assert not math.isnan(policy_metrics["actor_loss"])
