"""Dreamer (Hafner et al., 2020) latent imagination actor-critic reproduction."""

from __future__ import annotations

import numpy as np
from aurora.distributions.tanh_normal import TanhNormal
from aurora.nn.layers import MLP, Linear
from aurora.nn.module import Module
from aurora.optim.adamw import AdamW
from aurora.tensor import Tensor, concat, tensor, zeros
from aurora.world_model.rssm import RSSM, RSSMState


class DreamerActor(Module):
    """Latent Actor network outputting squashed Gaussian continuous actions."""

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        hidden_dims: list[int] | None = None,
        activation: str = "silu",
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256]

        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.net = MLP(latent_dim, hidden_dims, hidden_dims[-1], activation=activation)
        self.mu_head = Linear(hidden_dims[-1], action_dim)
        self.log_std_head = Linear(hidden_dims[-1], action_dim)

    def forward(self, feat: Tensor) -> tuple[Tensor, Tensor]:
        """Compute action and log probability from latent features."""
        h = self.net(feat)
        mu = self.mu_head(h)
        log_std = self.log_std_head(h).clamp(self.log_std_min, self.log_std_max)
        scale = log_std.exp()

        dist = TanhNormal(loc=mu, scale=scale)
        action, pre_tanh = dist.rsample_with_pre_tanh()
        log_prob = dist.log_prob(action, pre_tanh_value=pre_tanh)
        return action, log_prob

    def select_action(self, feat: Tensor, deterministic: bool = False) -> np.ndarray:
        """Sample action for environment execution."""
        h = self.net(feat)
        mu = self.mu_head(h)
        if deterministic:
            dist = TanhNormal(loc=mu, scale=zeros(mu.shape).exp())
            return np.asarray(dist.mean.numpy())
        log_std = self.log_std_head(h).clamp(self.log_std_min, self.log_std_max)
        dist = TanhNormal(loc=mu, scale=log_std.exp())
        act, _ = dist.rsample_with_pre_tanh()
        return np.asarray(act.numpy())


class DreamerCritic(Module):
    """Latent Critic / Value network predicting state value V(h, z)."""

    def __init__(
        self,
        latent_dim: int,
        hidden_dims: list[int] | None = None,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256]

        self.net = MLP(latent_dim, hidden_dims, 1, activation=activation)

    def forward(self, feat: Tensor) -> Tensor:
        out: Tensor = self.net(feat)
        return out


def compute_lambda_returns(
    rewards: list[Tensor] | np.ndarray,
    discounts: list[Tensor] | np.ndarray,
    values: list[Tensor] | np.ndarray,
    lambda_: float = 0.95,
) -> np.ndarray:
    """Compute generalized lambda-returns backwards through imagined latent trajectories.

    V_H^lambda = v_H
    V_t^lambda = r_t + gamma_t * ((1 - lambda) * v_{t+1} + lambda * V_{t+1}^lambda)
    """
    # Convert inputs to numpy arrays of shape (H, B)
    if isinstance(rewards, list):
        r_arr = np.array([r.numpy().squeeze() for r in rewards])
    else:
        r_arr = np.array(rewards)

    if isinstance(discounts, list):
        d_arr = np.array([d.numpy().squeeze() for d in discounts])
    else:
        d_arr = np.array(discounts)

    if isinstance(values, list):
        v_arr = np.array([v.numpy().squeeze() for v in values])
    else:
        v_arr = np.array(values)

    horizon = r_arr.shape[0]
    returns = np.zeros_like(r_arr)

    # Base case at horizon H - 1
    last_v = v_arr[horizon]
    last_return = last_v

    for t in reversed(range(horizon)):
        # target = r_t + discount_t * ((1 - lambda) * v_{t+1} + lambda * returns_{t+1})
        discount_t = d_arr[t]
        next_val = v_arr[t + 1]
        target = r_arr[t] + discount_t * ((1.0 - lambda_) * next_val + lambda_ * last_return)
        returns[t] = target
        last_return = target

    return returns


class DreamerAgent:
    """Dreamer latent imagination actor-critic agent."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        deter_dim: int = 200,
        stoch_dim: int = 30,
        hidden_dim: int = 200,
        imagination_horizon: int = 15,
        gamma: float = 0.99,
        lambda_: float = 0.95,
        world_model_lr: float = 1e-3,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        entropy_coeff: float = 1e-3,
    ) -> None:
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = deter_dim + stoch_dim
        self.imagination_horizon = imagination_horizon
        self.gamma = gamma
        self.lambda_ = lambda_
        self.entropy_coeff = entropy_coeff

        # 1. World Model (RSSM)
        self.rssm = RSSM(
            obs_dim=obs_dim,
            action_dim=action_dim,
            deter_dim=deter_dim,
            stoch_dim=stoch_dim,
            hidden_dim=hidden_dim,
        )
        self.wm_optimizer = AdamW(self.rssm.parameters(), lr=world_model_lr)

        # 2. Actor and Critic
        self.actor = DreamerActor(self.latent_dim, action_dim)
        self.critic = DreamerCritic(self.latent_dim)

        self.actor_optimizer = AdamW(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = AdamW(self.critic.parameters(), lr=critic_lr)

        self.prev_state: RSSMState | None = None

    def reset_state(self) -> None:
        """Reset internal recurrent state at episode boundaries."""
        self.prev_state = None

    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Observe environment step and select action conditioned on latent posterior."""
        obs_t = tensor(obs.reshape(1, -1), requires_grad=False)
        if self.prev_state is None:
            self.prev_state = self.rssm.initial_state(batch_size=1)
            action_t = zeros((1, self.action_dim))
        else:
            action_t = zeros((1, self.action_dim))

        _, post_state = self.rssm.observe_step(self.prev_state, action_t, obs_t)
        self.prev_state = post_state

        feat = concat([post_state.h, post_state.z], axis=-1)
        action = self.actor.select_action(feat, deterministic=deterministic)
        return action.squeeze(0)

    def imagine_rollout(
        self,
        initial_state: RSSMState,
    ) -> tuple[list[Tensor], list[Tensor], list[Tensor], list[Tensor]]:
        """Roll out policy in latent space for imagination_horizon steps.

        Returns:
            latent_features: list of (H + 1) feature tensors
            rewards: list of H reward tensors
            discounts: list of H discount factor tensors
            actions: list of H action tensors
        """
        curr_state = initial_state
        features: list[Tensor] = []
        rewards: list[Tensor] = []
        discounts: list[Tensor] = []
        actions: list[Tensor] = []

        feat = concat([curr_state.h, curr_state.z], axis=-1)
        features.append(feat)

        for _ in range(self.imagination_horizon):
            action, _ = self.actor(feat)
            actions.append(action)

            # Prior imagine step
            next_state = self.rssm.imagine_step(curr_state, action)
            _, rew, cont = self.rssm.decode(next_state)

            rewards.append(rew)
            discounts.append(cont * self.gamma)

            curr_state = next_state
            feat = concat([curr_state.h, curr_state.z], axis=-1)
            features.append(feat)

        return features, rewards, discounts, actions

    def update(
        self,
        obs: Tensor,
        action: Tensor,
        reward: Tensor,
        done: Tensor,
    ) -> dict[str, float]:
        """Update world model, actor, and critic using a batch of transitions."""
        # 1. Update World Model (RSSM)
        batch_size = obs.shape[0]
        init_s = self.rssm.initial_state(batch_size=batch_size)

        self.wm_optimizer.zero_grad()
        prior_s, post_s = self.rssm.observe_step(init_s, action, obs)
        wm_loss, wm_metrics = self.rssm.compute_loss(prior_s, post_s, obs, reward, done)
        wm_loss.backward()
        self.wm_optimizer.step()

        # 2. Latent Imagination Rollout starting from detached posterior
        post_detached = RSSMState(
            h=tensor(post_s.h.numpy(), requires_grad=False),
            z=tensor(post_s.z.numpy(), requires_grad=False),
            mu=tensor(post_s.mu.numpy(), requires_grad=False),
            log_std=tensor(post_s.log_std.numpy(), requires_grad=False),
        )
        features, rewards, discounts, _actions = self.imagine_rollout(post_detached)

        # 3. Compute Value Estimates and Lambda-Returns
        values: list[Tensor] = [self.critic(f) for f in features]
        lambda_returns = compute_lambda_returns(
            rewards=rewards,
            discounts=discounts,
            values=values,
            lambda_=self.lambda_,
        )

        # 4. Critic Update
        self.critic_optimizer.zero_grad()
        critic_loss = zeros((1,))
        for t in range(self.imagination_horizon):
            val_pred = values[t]
            target_t = tensor(lambda_returns[t], requires_grad=False)
            diff = val_pred.reshape(-1) - target_t.reshape(-1)
            critic_loss = critic_loss + (diff * diff * 0.5).mean()

        critic_loss = critic_loss * (1.0 / float(self.imagination_horizon))
        critic_loss.backward()
        self.critic_optimizer.step()

        # 5. Actor Update
        self.actor_optimizer.zero_grad()
        actor_loss = zeros((1,))
        for t in range(self.imagination_horizon):
            f_detached = tensor(features[t].numpy(), requires_grad=False)
            _act, log_prob = self.actor(f_detached)
            val_est = self.critic(f_detached)
            # Maximize value + entropy bonus
            actor_loss = actor_loss - (val_est + (log_prob * (-self.entropy_coeff))).mean()

        actor_loss = actor_loss * (1.0 / float(self.imagination_horizon))
        actor_loss.backward()
        self.actor_optimizer.step()

        return {
            "wm_loss": wm_metrics["total_loss"],
            "kl_loss": wm_metrics["kl_loss"],
            "critic_loss": float(critic_loss.numpy().item()),
            "actor_loss": float(actor_loss.numpy().item()),
        }
