"""Recurrent State-Space Models (RSSM) with KL Balancing and Variational Inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np

from aurora.nn.layers import MLP, Linear
from aurora.nn.module import Module
from aurora.tensor import Tensor, concat, tensor, zeros


class GRUCell(Module):
    """Gated Recurrent Unit cell implemented with first-principles linear transformations:

    r_t = sigmoid(W_ir x_t + W_hr h_{t-1} + b_r)
    z_t = sigmoid(W_iz x_t + W_hz h_{t-1} + b_z)
    n_t = tanh(W_in x_t + r_t * (W_hn h_{t-1} + b_n))
    h_t = (1 - z_t) * n_t + z_t * h_{t-1}
    """

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.w_ir = Linear(input_dim, hidden_dim)
        self.w_hr = Linear(hidden_dim, hidden_dim)
        self.w_iz = Linear(input_dim, hidden_dim)
        self.w_hz = Linear(hidden_dim, hidden_dim)
        self.w_in = Linear(input_dim, hidden_dim)
        self.w_hn = Linear(hidden_dim, hidden_dim)

    def forward(self, x: Tensor, h: Tensor) -> Tensor:
        r = (self.w_ir(x) + self.w_hr(h)).sigmoid()
        z = (self.w_iz(x) + self.w_hz(h)).sigmoid()
        cand = (self.w_in(x) + r * self.w_hn(h)).tanh()
        one_minus_z = 1.0 - z
        return cast(Tensor, one_minus_z * cand + z * h)


@dataclass(slots=True)
class RSSMState:
    """State tuple holding deterministic and stochastic latent states."""

    h: Tensor  # Deterministic recurrent state: (B, deter_dim)
    z: Tensor  # Stochastic sample: (B, stoch_dim)
    mu: Tensor  # Latent mean: (B, stoch_dim)
    log_std: Tensor  # Latent log standard deviation: (B, stoch_dim)


class RSSM(Module):
    """Recurrent State-Space Model for partially observable and latent dynamics.

    Maintains:
    - Deterministic recurrent state: h_t in R^{d_h}
    - Stochastic latent state: z_t in R^{d_z}

    Components:
    - Transition cell: h_t = GRUCell([z_{t-1}, a_{t-1}], h_{t-1})
    - Prior network: p(z_t | h_t) = N(mu^prior, (sigma^prior)^2)
    - Posterior network: q(z_t | h_t, x_t) = N(mu^post, (sigma^post)^2)
    - Observation decoder: x_hat_t = g_x(h_t, z_t)
    - Reward decoder: r_hat_t = g_r(h_t, z_t)
    - Continuation decoder: gamma_hat_t = sigmoid(g_gamma(h_t, z_t))
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        deter_dim: int = 200,
        stoch_dim: int = 30,
        hidden_dim: int = 200,
        kl_alpha: float = 0.8,
        kl_scale: float = 1.0,
        min_std: float = 0.1,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.deter_dim = deter_dim
        self.stoch_dim = stoch_dim
        self.kl_alpha = kl_alpha
        self.kl_scale = kl_scale
        self.min_std = min_std

        # 1. Recurrent Cell
        self.cell = GRUCell(input_dim=stoch_dim + action_dim, hidden_dim=deter_dim)

        # 2. Prior Network: h_t -> (mu_prior, log_std_prior)
        self.prior_mlp = MLP(deter_dim, [hidden_dim], hidden_dim, activation="silu")
        self.prior_mu = Linear(hidden_dim, stoch_dim)
        self.prior_log_std = Linear(hidden_dim, stoch_dim)

        # 3. Posterior Network: [h_t, x_t] -> (mu_post, log_std_post)
        self.post_mlp = MLP(deter_dim + obs_dim, [hidden_dim], hidden_dim, activation="silu")
        self.post_mu = Linear(hidden_dim, stoch_dim)
        self.post_log_std = Linear(hidden_dim, stoch_dim)

        # 4. Decoders
        self.obs_decoder = MLP(deter_dim + stoch_dim, [hidden_dim], obs_dim, activation="silu")
        self.reward_decoder = MLP(deter_dim + stoch_dim, [hidden_dim], 1, activation="silu")
        self.cont_decoder = MLP(deter_dim + stoch_dim, [hidden_dim], 1, activation="silu")

    def initial_state(self, batch_size: int = 1) -> RSSMState:
        """Create zero-initialized deterministic and stochastic state."""
        h0 = zeros((batch_size, self.deter_dim))
        z0 = zeros((batch_size, self.stoch_dim))
        mu0 = zeros((batch_size, self.stoch_dim))
        log_std0 = zeros((batch_size, self.stoch_dim))
        return RSSMState(h=h0, z=z0, mu=mu0, log_std=log_std0)

    def _sample_gaussian(self, mu: Tensor, log_std: Tensor) -> Tensor:
        """Reparameterized Gaussian sampling: z = mu + sigma * eps."""
        std = log_std.exp().clamp(self.min_std, 10.0)
        eps = tensor(np.random.randn(*mu.shape), requires_grad=False)
        return mu + std * eps

    def observe_step(
        self,
        prev_state: RSSMState,
        action: Tensor,
        obs: Tensor,
    ) -> tuple[RSSMState, RSSMState]:
        """Compute prior and posterior states for one time step with observation."""
        # 1. Deterministic update
        cell_in = concat([prev_state.z, action], axis=-1)
        h = self.cell(cell_in, prev_state.h)

        # 2. Prior distribution
        prior_feats = self.prior_mlp(h)
        prior_mu = self.prior_mu(prior_feats)
        prior_log_std = self.prior_log_std(prior_feats).clamp(-5.0, 2.0)
        prior_z = self._sample_gaussian(prior_mu, prior_log_std)
        prior_state = RSSMState(h=h, z=prior_z, mu=prior_mu, log_std=prior_log_std)

        # 3. Posterior distribution (conditioning on observation)
        post_in = concat([h, obs], axis=-1)
        post_feats = self.post_mlp(post_in)
        post_mu = self.post_mu(post_feats)
        post_log_std = self.post_log_std(post_feats).clamp(-5.0, 2.0)
        post_z = self._sample_gaussian(post_mu, post_log_std)
        post_state = RSSMState(h=h, z=post_z, mu=post_mu, log_std=post_log_std)

        return prior_state, post_state

    def imagine_step(
        self,
        prev_state: RSSMState,
        action: Tensor,
    ) -> RSSMState:
        """Pure prior rollout step without environment observations."""
        cell_in = concat([prev_state.z, action], axis=-1)
        h = self.cell(cell_in, prev_state.h)

        prior_feats = self.prior_mlp(h)
        prior_mu = self.prior_mu(prior_feats)
        prior_log_std = self.prior_log_std(prior_feats).clamp(-5.0, 2.0)
        prior_z = self._sample_gaussian(prior_mu, prior_log_std)
        return RSSMState(h=h, z=prior_z, mu=prior_mu, log_std=prior_log_std)

    def decode(self, state: RSSMState) -> tuple[Tensor, Tensor, Tensor]:
        """Decode observation, reward, and continuation probability from latent state."""
        feat = concat([state.h, state.z], axis=-1)
        obs_hat = self.obs_decoder(feat)
        r_hat = self.reward_decoder(feat)
        gamma_hat = self.cont_decoder(feat).sigmoid()
        return obs_hat, r_hat, gamma_hat

    @staticmethod
    def kl_divergence(
        mu_q: Tensor,
        log_std_q: Tensor,
        mu_p: Tensor,
        log_std_p: Tensor,
    ) -> Tensor:
        """Analytical KL divergence D_KL(N(mu_q, sigma_q^2) || N(mu_p, sigma_p^2)).

        D_KL = sum [ log(sigma_p / sigma_q) + (sigma_q^2 + (mu_q - mu_p)^2)/(2*sigma_p^2) - 0.5 ]
        """
        var_q = (log_std_q * 2.0).exp()
        inv_var_p = (-log_std_p * 2.0).exp()

        diff = mu_q - mu_p
        diff_sq = diff * diff

        term1 = log_std_p - log_std_q
        term2 = (var_q + diff_sq) * inv_var_p * 0.5
        kl_elem = term1 + term2 - 0.5
        return kl_elem.sum(axis=-1)

    def compute_loss(
        self,
        prior_state: RSSMState,
        post_state: RSSMState,
        target_obs: Tensor,
        target_reward: Tensor,
        target_done: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, float]]:
        """Compute ELBO loss with KL balancing:

        L = L_obs + L_reward + L_cont + beta_KL * L_KL
        """
        obs_hat, r_hat, gamma_hat = self.decode(post_state)

        # 1. Observation reconstruction loss (Gaussian NLL / MSE)
        obs_diff = target_obs - obs_hat
        obs_loss = (obs_diff * obs_diff * 0.5).sum(axis=-1).mean()

        # 2. Reward reconstruction loss
        r_target = target_reward if target_reward.ndim > 1 else target_reward.reshape(-1, 1)
        r_diff = r_target - r_hat
        reward_loss = (r_diff * r_diff * 0.5).sum(axis=-1).mean()

        # 3. Continuation loss (BCE)
        if target_done is not None:
            done = target_done if target_done.ndim > 1 else target_done.reshape(-1, 1)
            # gamma_hat is probability of continuation: target is (1 - done)
            cont_target = 1.0 - done
            eps = 1e-7
            pos_term = cont_target * (gamma_hat + eps).log()
            neg_term = (1.0 - cont_target) * (1.0 - gamma_hat + eps).log()
            bce = -(pos_term + neg_term)
            cont_loss = bce.sum(axis=-1).mean()
        else:
            cont_loss = zeros((1,))

        # 4. KL Balancing with stop-gradient
        # D_KL( q || stop_grad(p) )
        prior_mu_detached = tensor(prior_state.mu.numpy(), requires_grad=False)
        prior_log_std_detached = tensor(prior_state.log_std.numpy(), requires_grad=False)
        kl_lhs = self.kl_divergence(
            post_state.mu, post_state.log_std, prior_mu_detached, prior_log_std_detached
        ).mean()

        # D_KL( stop_grad(q) || p )
        post_mu_detached = tensor(post_state.mu.numpy(), requires_grad=False)
        post_log_std_detached = tensor(post_state.log_std.numpy(), requires_grad=False)
        kl_rhs = self.kl_divergence(
            post_mu_detached, post_log_std_detached, prior_state.mu, prior_state.log_std
        ).mean()

        kl_loss = self.kl_alpha * kl_lhs + (1.0 - self.kl_alpha) * kl_rhs

        total_loss = obs_loss + reward_loss + cont_loss + (kl_loss * self.kl_scale)

        metrics = {
            "obs_loss": float(obs_loss.numpy()),
            "reward_loss": float(reward_loss.numpy()),
            "cont_loss": float(cont_loss.numpy()),
            "kl_loss": float(kl_loss.numpy()),
            "total_loss": float(total_loss.numpy()),
        }
        return total_loss, metrics
