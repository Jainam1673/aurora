"""Unit tests for latent world models, deep ensembles, uncertainty, and imagination."""

from __future__ import annotations

import numpy as np
from aurora.rl.buffers import ReplayBuffer
from aurora.tensor import tensor
from aurora.world_model.ensemble import EnsembleDynamicsModel
from aurora.world_model.imagination import ImaginationEngine
from aurora.world_model.rssm import RSSM, GRUCell
from aurora.world_model.uncertainty import UncertaintyEstimator


class TestEnsembleDynamicsModel:
    def test_initialization_and_forward(self) -> None:
        obs_dim = 4
        action_dim = 2
        ensemble_size = 5
        model = EnsembleDynamicsModel(
            obs_dim=obs_dim,
            action_dim=action_dim,
            ensemble_size=ensemble_size,
            hidden_dims=[32, 32],
            predict_reward=True,
        )

        assert len(model.members) == ensemble_size
        assert len(model.parameters()) > 0

        batch_size = 8
        obs = tensor(np.random.randn(batch_size, obs_dim), requires_grad=False)
        action = tensor(np.random.randn(batch_size, action_dim), requires_grad=False)

        means, log_vars = model.forward(obs, action)
        assert len(means) == ensemble_size
        assert len(log_vars) == ensemble_size

        for mu, lv in zip(means, log_vars, strict=True):
            assert mu.shape == (batch_size, obs_dim + 1)
            assert lv.shape == (batch_size, obs_dim + 1)

    def test_nll_loss_and_backward(self) -> None:
        obs_dim = 3
        action_dim = 1
        model = EnsembleDynamicsModel(
            obs_dim=obs_dim,
            action_dim=action_dim,
            ensemble_size=3,
            hidden_dims=[16, 16],
            predict_reward=True,
        )

        batch_size = 4
        obs = tensor(np.random.randn(batch_size, obs_dim), requires_grad=False)
        action = tensor(np.random.randn(batch_size, action_dim), requires_grad=False)
        next_obs = tensor(np.random.randn(batch_size, obs_dim), requires_grad=False)
        reward = tensor(np.random.randn(batch_size, 1), requires_grad=False)

        loss = model.compute_loss(obs, action, next_obs, reward)
        assert loss.numpy().item() > 0.0

        model.zero_grad()
        loss.backward()

        # Verify that all member parameters received gradients
        for member in model.members:
            for p in member.parameters():
                assert p.grad is not None

    def test_predict_and_step(self) -> None:
        obs_dim = 2
        action_dim = 1
        model = EnsembleDynamicsModel(
            obs_dim=obs_dim,
            action_dim=action_dim,
            ensemble_size=4,
            hidden_dims=[16],
            predict_reward=True,
        )

        obs = np.array([[1.0, -0.5], [0.2, 0.8]])
        action = np.array([[0.5], [-0.5]])

        next_states, rewards, variances = model.predict(obs, action)
        assert next_states.shape == (4, 2, obs_dim)
        assert rewards.shape == (4, 2, 1)
        assert variances.shape == (4, 2, obs_dim + 1)

        # Step single transition
        s_single = np.array([1.0, -0.5])
        a_single = np.array([0.5])
        s_next, r = model.step(s_single, a_single, member_idx=0)
        assert s_next.shape == (obs_dim,)
        assert isinstance(r, float)


class TestUncertaintyEstimator:
    def test_perfect_agreement(self) -> None:
        # If all ensemble members predict identical mean, epistemic uncertainty must be zero
        identical_mean = np.array([[1.0, 2.0], [3.0, 4.0]])
        var_data = np.array([[0.1, 0.2], [0.3, 0.4]])

        means = [identical_mean.copy() for _ in range(5)]
        variances = [var_data.copy() for _ in range(5)]

        unc = UncertaintyEstimator.decompose_np(means, variances)
        np.testing.assert_allclose(unc.mean, identical_mean, atol=1e-12)
        np.testing.assert_allclose(unc.epistemic, 0.0, atol=1e-12)
        np.testing.assert_allclose(unc.aleatoric, var_data, atol=1e-12)
        np.testing.assert_allclose(unc.total, unc.aleatoric, atol=1e-12)
        np.testing.assert_allclose(unc.disagreement, 0.0, atol=1e-12)

    def test_epistemic_disagreement(self) -> None:
        m1 = np.array([[0.0, 1.0]])
        m2 = np.array([[2.0, 1.0]])  # Disagrees along dimension 0
        v = np.array([[0.5, 0.5]])

        unc = UncertaintyEstimator.decompose_np([m1, m2], [v, v])
        # Mean along dim 0 is 1.0; squared diff is (0-1)^2 = 1 and (2-1)^2 = 1 -> epistemic = 1.0
        # Mean along dim 1 is 1.0; squared diff is 0 -> epistemic = 0.0
        np.testing.assert_allclose(unc.mean, [[1.0, 1.0]], atol=1e-12)
        np.testing.assert_allclose(unc.epistemic, [[1.0, 0.0]], atol=1e-12)
        np.testing.assert_allclose(unc.aleatoric, [[0.5, 0.5]], atol=1e-12)
        np.testing.assert_allclose(unc.total, [[1.5, 0.5]], atol=1e-12)
        np.testing.assert_allclose(unc.disagreement, [1.0], atol=1e-12)

    def test_tensor_differentiable_decomposition(self) -> None:
        t1 = tensor([[1.0, 2.0]], requires_grad=True)
        t2 = tensor([[3.0, 4.0]], requires_grad=True)
        v1 = tensor([[0.2, 0.3]], requires_grad=True)
        v2 = tensor([[0.4, 0.5]], requires_grad=True)

        _mean, _aleatoric, _epistemic, total = UncertaintyEstimator.decompose_tensor(
            [t1, t2], [v1, v2]
        )

        loss = total.sum()
        loss.backward()

        assert t1.grad is not None
        assert t2.grad is not None
        assert v1.grad is not None
        assert v2.grad is not None


class TestImaginationEngine:
    def test_adaptive_truncation(self) -> None:
        obs_dim = 2
        action_dim = 1
        model = EnsembleDynamicsModel(
            obs_dim=obs_dim,
            action_dim=action_dim,
            ensemble_size=3,
            hidden_dims=[16],
            predict_reward=True,
        )

        class DummyPolicy:
            def act(self, obs: np.ndarray) -> np.ndarray:
                return np.zeros(1)

        policy = DummyPolicy()

        # Test with very low threshold: must truncate before max_horizon
        engine_trunc = ImaginationEngine(
            dynamics=model,
            policy=policy,
            max_horizon=20,
            uncertainty_threshold=1e-6,  # Artificially low threshold to guarantee truncation
            adaptive_truncation=True,
        )

        initial_states = np.random.randn(5, obs_dim)
        res_trunc = engine_trunc.generate_rollouts(initial_states)

        assert res_trunc.truncated_trajectories > 0
        assert res_trunc.mean_horizon < 20

        # Test with high threshold: runs full horizon
        engine_full = ImaginationEngine(
            dynamics=model,
            policy=policy,
            max_horizon=5,
            uncertainty_threshold=100.0,
            adaptive_truncation=True,
        )
        res_full = engine_full.generate_rollouts(initial_states)
        assert res_full.truncated_trajectories == 0
        assert res_full.mean_horizon == 5.0

    def test_inject_into_replay_buffer(self) -> None:
        obs_dim = 3
        action_dim = 1
        model = EnsembleDynamicsModel(
            obs_dim=obs_dim,
            action_dim=action_dim,
            ensemble_size=3,
            hidden_dims=[16],
            predict_reward=True,
        )

        engine = ImaginationEngine(
            dynamics=model,
            policy=lambda s: np.array([0.1]),
            max_horizon=4,
            uncertainty_threshold=10.0,
        )

        buf = ReplayBuffer(capacity=100, obs_shape=(obs_dim,), action_shape=(action_dim,))
        res = engine.inject_into_buffer(buf, np.random.randn(2, obs_dim))

        assert res.total_transitions == 8
        assert len(buf) == 8


class TestRSSM:
    def test_gru_cell(self) -> None:
        cell = GRUCell(input_dim=4, hidden_dim=8)
        x = tensor(np.random.randn(2, 4), requires_grad=True)
        h = tensor(np.random.randn(2, 8), requires_grad=True)

        h_next = cell(x, h)
        assert h_next.shape == (2, 8)

        loss = h_next.sum()
        loss.backward()
        assert x.grad is not None
        assert h.grad is not None

    def test_rssm_step_and_decode(self) -> None:
        obs_dim = 4
        action_dim = 2
        deter_dim = 16
        stoch_dim = 8
        rssm = RSSM(
            obs_dim=obs_dim,
            action_dim=action_dim,
            deter_dim=deter_dim,
            stoch_dim=stoch_dim,
            hidden_dim=16,
        )

        state = rssm.initial_state(batch_size=2)
        assert state.h.shape == (2, deter_dim)
        assert state.z.shape == (2, stoch_dim)

        action = tensor(np.random.randn(2, action_dim))
        obs = tensor(np.random.randn(2, obs_dim))

        prior_state, post_state = rssm.observe_step(state, action, obs)
        assert prior_state.h.shape == (2, deter_dim)
        assert post_state.h.shape == (2, deter_dim)
        assert prior_state.z.shape == (2, stoch_dim)
        assert post_state.z.shape == (2, stoch_dim)

        # Decoding
        obs_hat, r_hat, gamma_hat = rssm.decode(post_state)
        assert obs_hat.shape == (2, obs_dim)
        assert r_hat.shape == (2, 1)
        assert gamma_hat.shape == (2, 1)

        # Continuation probability in [0, 1]
        assert np.all(gamma_hat.numpy() >= 0.0)
        assert np.all(gamma_hat.numpy() <= 1.0)

    def test_kl_divergence_zero_when_identical(self) -> None:
        mu = tensor([[1.0, -2.0, 0.5]], requires_grad=False)
        log_std = tensor([[0.0, 0.2, -0.5]], requires_grad=False)

        kl = RSSM.kl_divergence(mu, log_std, mu, log_std)
        np.testing.assert_allclose(kl.numpy(), 0.0, atol=1e-12)

    def test_rssm_loss_and_backward(self) -> None:
        rssm = RSSM(obs_dim=3, action_dim=1, deter_dim=8, stoch_dim=4, hidden_dim=8)
        state0 = rssm.initial_state(batch_size=2)
        a = tensor(np.random.randn(2, 1), requires_grad=False)
        o = tensor(np.random.randn(2, 3), requires_grad=False)
        r = tensor(np.random.randn(2, 1), requires_grad=False)
        d = tensor(np.zeros((2, 1)), requires_grad=False)

        prior_s, post_s = rssm.observe_step(state0, a, o)
        loss, metrics = rssm.compute_loss(prior_s, post_s, o, r, d)

        assert "total_loss" in metrics
        assert "kl_loss" in metrics
        assert metrics["total_loss"] > 0.0

        rssm.zero_grad()
        loss.backward()

        # Check gradients in submodules
        for p in rssm.parameters():
            if p.requires_grad and p.grad is not None:
                assert not np.isnan(p.grad.numpy()).any()
