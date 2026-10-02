"""Unit and integration tests for AURORA reinforcement learning primitives."""

from __future__ import annotations

import numpy as np
from aurora.distributions.categorical import Categorical
from aurora.distributions.normal import Normal
from aurora.distributions.tanh_normal import TanhNormal
from aurora.environments.classic_control import CartPole, Pendulum
from aurora.environments.spaces import Box, Discrete
from aurora.rl.buffers import ReplayBuffer, RolloutBuffer
from aurora.rl.policies import ActorCriticPolicy, SquashedGaussianActor, TwinCritic
from aurora.rl.ppo import PPO
from aurora.rl.sac import SAC
from aurora.tensor import tensor


class TestDistributions:
    def test_categorical_distribution(self) -> None:
        # Uniform logits
        logits = tensor([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
        dist = Categorical(logits=logits)
        probs = dist.probs.numpy()

        assert probs.shape == (2, 3)
        np.testing.assert_allclose(probs[0], [1.0 / 3, 1.0 / 3, 1.0 / 3], atol=1e-6)
        np.testing.assert_allclose(probs.sum(axis=-1), [1.0, 1.0], atol=1e-6)

        actions = tensor([0, 2])
        lp = dist.log_prob(actions).numpy()
        assert lp.shape == (2,)
        np.testing.assert_allclose(lp[0], np.log(1.0 / 3), atol=1e-6)

        # Entropy of uniform distribution = ln(3)
        ent = dist.entropy().numpy()
        np.testing.assert_allclose(ent[0], np.log(3.0), atol=1e-6)

        # Sample test
        samples = dist.sample()
        assert samples.shape == (2,)
        assert 0 <= samples.numpy()[0] < 3

    def test_normal_distribution(self) -> None:
        loc = tensor([[0.0, 1.0], [2.0, -1.0]], requires_grad=True)
        scale = tensor([[1.0, 2.0], [0.5, 1.5]], requires_grad=True)
        dist = Normal(loc=loc, scale=scale)

        # Log prob at mean: -0.5 * ln(2*pi*sigma^2)
        val = loc
        lp = dist.log_prob(val)
        expected_lp = -np.log(scale.numpy() * np.sqrt(2.0 * np.pi))
        np.testing.assert_allclose(lp.numpy(), expected_lp.sum(axis=-1), atol=1e-6)

        # Entropy: 0.5 + 0.5 * ln(2*pi*sigma^2)
        ent = dist.entropy().numpy()
        expected_ent = (0.5 + 0.5 * np.log(2.0 * np.pi * (scale.numpy() ** 2))).sum(axis=-1)
        np.testing.assert_allclose(ent, expected_ent, atol=1e-6)

        # Reparameterized sample gradient check
        sample = dist.rsample()
        loss = sample.sum()
        loss.backward()
        assert loc.grad is not None
        assert scale.grad is not None
        np.testing.assert_allclose(loc.grad.numpy(), np.ones_like(loc.numpy()))

    def test_tanh_normal_distribution(self) -> None:
        loc = tensor([[0.0, 0.5]], requires_grad=True)
        scale = tensor([[1.0, 1.0]], requires_grad=True)
        dist = TanhNormal(loc=loc, scale=scale)

        action, pre_tanh = dist.rsample_with_pre_tanh()
        assert action.shape == (1, 2)
        assert pre_tanh.shape == (1, 2)
        # Bounded in (-1, 1)
        assert np.all(np.abs(action.numpy()) <= 1.0)

        # Log prob with and without pre_tanh value should match
        lp1 = dist.log_prob(action, pre_tanh_value=pre_tanh).numpy()
        lp2 = dist.log_prob(action).numpy()
        np.testing.assert_allclose(lp1, lp2, atol=1e-5)

        # Pathwise gradient flow through rsample
        loss = action.sum()
        loss.backward()
        assert loc.grad is not None
        assert scale.grad is not None


class TestSpacesAndEnvironments:
    def test_discrete_space(self) -> None:
        space = Discrete(5)
        assert space.shape == ()
        for _ in range(20):
            s = space.sample()
            assert space.contains(s)
            assert 0 <= s < 5
        assert not space.contains(-1)
        assert not space.contains(5)

    def test_box_space(self) -> None:
        space = Box(low=-2.0, high=2.0, shape=(3,))
        assert space.shape == (3,)
        s = space.sample()
        assert s.shape == (3,)
        assert space.contains(s)
        assert not space.contains(np.array([3.0, 0.0, 0.0]))

    def test_cartpole_env(self) -> None:
        env = CartPole()
        obs, _ = env.reset(seed=42)
        assert obs.shape == (4,)
        assert env.observation_space.contains(obs)

        done = False
        steps = 0
        while not done and steps < 200:
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            assert next_obs.shape == (4,)
            assert reward == 1.0
            done = terminated or truncated
            steps += 1
        assert steps > 0

    def test_pendulum_env(self) -> None:
        env = Pendulum()
        obs, _ = env.reset(seed=123)
        assert obs.shape == (3,)
        assert env.observation_space.contains(obs)

        for _ in range(50):
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            assert next_obs.shape == (3,)
            assert isinstance(reward, float)
            assert not terminated  # Pendulum never terminates early
            assert not truncated


class TestBuffers:
    def test_rollout_buffer_gae(self) -> None:
        buf = RolloutBuffer(buffer_size=3, obs_shape=(2,), action_shape=(), batch_size=1)

        # Add 3 transitions
        buf.add(
            np.array([[1.0, 2.0]]), np.array([0]), 1.0, False, 0.5, -0.2
        )
        buf.add(
            np.array([[2.0, 3.0]]), np.array([1]), 2.0, False, 1.0, -0.3
        )
        buf.add(
            np.array([[3.0, 4.0]]), np.array([0]), 3.0, True, 1.5, -0.4
        )

        last_val = np.array([0.0])
        last_done = np.array([1.0])
        buf.compute_returns_and_advantages(last_val, last_done, gamma=0.99, gae_lambda=0.95)

        # Step 2 (t=2, done=1.0):
        # delta = r + gamma * last_val * (1 - done) - V = 3.0 + 0 - 1.5 = 1.5
        # gae = 1.5
        assert np.isclose(buf.advantages[2, 0], 1.5, atol=1e-5)
        assert np.isclose(buf.returns[2, 0], 1.5 + 1.5, atol=1e-5)

        # Generator check
        batches = list(buf.get_generator(batch_size=2))
        assert len(batches) == 2  # 3 elements with batch_size 2 yields 2 batches (2, 1)

    def test_replay_buffer(self) -> None:
        buf = ReplayBuffer(capacity=10, obs_shape=(2,), action_shape=(1,))
        assert len(buf) == 0

        for i in range(5):
            buf.add(
                obs=np.array([float(i), float(i)]),
                action=np.array([float(i)]),
                reward=float(i),
                next_obs=np.array([float(i + 1), float(i + 1)]),
                done=False,
            )
        assert len(buf) == 5

        batch = buf.sample(batch_size=4)
        assert batch.obs.shape == (4, 2)
        assert batch.actions.shape == (4, 1)
        assert batch.rewards.shape == (4,)
        assert batch.next_obs.shape == (4, 2)
        assert batch.dones.shape == (4,)


class TestAlgorithms:
    def test_ppo_step_discrete(self) -> None:
        env = CartPole()
        policy = ActorCriticPolicy(obs_dim=4, action_dim=2, is_discrete=True, hidden_dims=[32, 32])
        ppo = PPO(policy=policy, lr=1e-3, n_steps=32, batch_size=16, n_epochs=2)

        buffer = RolloutBuffer(buffer_size=32, obs_shape=(4,), action_shape=(), batch_size=1)
        obs, _ = env.reset(seed=42)
        _obs, _ep_rewards = ppo.collect_rollouts(env, buffer, obs)

        metrics = ppo.train(buffer)
        assert "policy_loss" in metrics
        assert "value_loss" in metrics
        assert "entropy" in metrics
        assert not np.isnan(metrics["policy_loss"])
        assert not np.isnan(metrics["value_loss"])

    def test_sac_step_continuous(self) -> None:
        env = Pendulum()
        actor = SquashedGaussianActor(obs_dim=3, action_dim=1, hidden_dims=[32, 32])
        critic = TwinCritic(obs_dim=3, action_dim=1, hidden_dims=[32, 32])
        sac = SAC(actor=actor, critic=critic, lr=1e-3, auto_alpha=True)

        buffer = ReplayBuffer(capacity=100, obs_shape=(3,), action_shape=(1,))
        obs, _ = env.reset(seed=42)
        for _ in range(50):
            action = actor.act(obs)
            next_obs, reward, terminated, _truncated, _ = env.step(action)
            buffer.add(obs, action, reward, next_obs, terminated)
            obs = next_obs

        metrics = sac.train_step(buffer, batch_size=16)
        assert "critic_loss" in metrics
        assert "actor_loss" in metrics
        assert "alpha" in metrics
        assert not np.isnan(metrics["critic_loss"])
        assert not np.isnan(metrics["actor_loss"])
