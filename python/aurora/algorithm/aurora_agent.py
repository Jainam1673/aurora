"""AURORA: Adaptive Uncertainty-calibrated Rollouts and Optimization Agent."""

from __future__ import annotations

import numpy as np

from aurora.algorithm.scheduler import AdaptiveHorizonScheduler, DynamicBlendingController
from aurora.nn.parameter import Parameter
from aurora.optim.adamw import AdamW
from aurora.rl.buffers import ReplayBatch, ReplayBuffer
from aurora.rl.policies import SquashedGaussianActor, TwinCritic
from aurora.tensor import Tensor, concat, tensor
from aurora.world_model.ensemble import EnsembleDynamicsModel


class AURORAAgent:
    """Adaptive Uncertainty-calibrated Rollouts and Optimization for Reinforcement Agents.

    Dynamically schedules imagination horizons, modulates real-to-synthetic data blending,
    and applies epistemic risk-sensitive (pessimistic) policy updates.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        ensemble_size: int = 5,
        ensemble_hidden_dims: list[int] | None = None,
        actor_hidden_dims: list[int] | None = None,
        critic_hidden_dims: list[int] | None = None,
        horizon_max: int = 15,
        horizon_min: int = 1,
        tau_base: float = 0.5,
        kappa: float = 1.0,
        budget_max: float = 2.0,
        eta_max: float = 0.85,
        u_target: float = 0.4,
        beta_pess: float = 0.5,
        tau_active: float = 1.0,
        gamma: float = 0.99,
        tau: float = 0.005,
        lr: float = 3e-4,
        model_lr: float = 1e-3,
        initial_alpha: float = 0.2,
        env_buffer_capacity: int = 100_000,
        model_buffer_capacity: int = 100_000,
    ) -> None:
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.beta_pess = beta_pess
        self.tau_active = tau_active

        # 1. World Model (Probabilistic Dynamics Ensemble)
        self.dynamics = EnsembleDynamicsModel(
            obs_dim=obs_dim,
            action_dim=action_dim,
            ensemble_size=ensemble_size,
            hidden_dims=ensemble_hidden_dims or [200, 200],
            predict_reward=True,
        )
        self.dynamics_opt = AdamW(self.dynamics.parameters(), lr=model_lr, weight_decay=1e-4)

        # 2. Policy Optimization (Squashed Gaussian Actor & Twin Critic)
        self.actor = SquashedGaussianActor(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dims=actor_hidden_dims or [256, 256],
        )
        self.critic = TwinCritic(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dims=critic_hidden_dims or [256, 256],
        )
        self.critic_target = TwinCritic(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dims=critic_hidden_dims or [256, 256],
        )
        self._copy_weights(self.critic, self.critic_target)

        self.actor_opt = AdamW(self.actor.parameters(), lr=lr)
        self.critic_opt = AdamW(self.critic.parameters(), lr=lr)

        # Automatic entropy temperature: alpha = exp(log_alpha)
        self.target_entropy = -float(action_dim)
        self.log_alpha = Parameter(np.array([float(np.log(initial_alpha))], dtype=np.float64))
        self.alpha_opt = AdamW([self.log_alpha], lr=lr)

        # 3. AURORA Uncertainty-Calibrated Controllers
        self.scheduler = AdaptiveHorizonScheduler(
            horizon_max=horizon_max,
            horizon_min=horizon_min,
            tau_base=tau_base,
            kappa=kappa,
            budget_max=budget_max,
            gamma=gamma,
        )
        self.blending_controller = DynamicBlendingController(
            eta_max=eta_max,
            u_target=u_target,
        )

        # 4. Experience Buffers
        self.env_buffer = ReplayBuffer(
            capacity=env_buffer_capacity,
            obs_shape=(obs_dim,),
            action_shape=(action_dim,),
        )
        self.model_buffer = ReplayBuffer(
            capacity=model_buffer_capacity,
            obs_shape=(obs_dim,),
            action_shape=(action_dim,),
        )

        # Tracking metrics
        self.last_mean_horizon: float = float(horizon_min)
        self.last_mean_uncertainty: float = 0.0

    @property
    def alpha(self) -> float:
        return float(np.exp(self.log_alpha.numpy()[0]))

    def _copy_weights(self, src: TwinCritic, dst: TwinCritic) -> None:
        src_p = src.parameters()
        dst_p = dst.parameters()
        for s, d in zip(src_p, dst_p, strict=True):
            d.data = np.copy(s.data)

    def _polyak_update(self) -> None:
        for p, tp in zip(self.critic.parameters(), self.critic_target.parameters(), strict=True):
            tp.data = self.tau * p.data + (1.0 - self.tau) * tp.data

    def select_action(
        self,
        obs: np.ndarray,
        deterministic: bool = False,
        active_exploration: bool = True,
        evaluate: bool | None = None,
    ) -> np.ndarray:
        if evaluate is not None:
            deterministic = evaluate
        """Select action with epistemic active exploration trigger."""
        action = self.actor.act(obs, deterministic=deterministic)

        # Active exploration check: if epistemic uncertainty is excessive, perturb action
        if active_exploration and not deterministic:
            u_epi = self.compute_epistemic_uncertainty_numpy(
                obs.reshape(1, -1), action.reshape(1, -1)
            )
            if float(u_epi[0]) > self.tau_active:
                noise = np.random.randn(self.action_dim) * 0.2
                action = np.clip(action + noise, -1.0, 1.0)

        return action

    def compute_epistemic_uncertainty_numpy(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
    ) -> np.ndarray:
        """Compute scalar epistemic uncertainty (max dimension variance) using numpy."""
        next_states, _, _ = self.dynamics.predict(obs, actions)
        # next_states shape: (ensemble_size, batch_size, obs_dim)
        epistemic_var = np.var(next_states, axis=0)  # (batch_size, obs_dim)
        return np.max(epistemic_var, axis=-1)  # (batch_size,)

    def compute_epistemic_uncertainty_tensor(
        self,
        obs: Tensor,
        actions: Tensor,
    ) -> Tensor:
        inputs = concat([obs, actions], axis=-1)
        means: list[Tensor] = []
        for member in self.dynamics.members:
            mu, _ = member(inputs)
            means.append(mu)

        # Stack ensemble means along axis 0: (E, B, obs_dim)
        from aurora.tensor import stack

        stacked = stack(means, axis=0)
        mean_ens = stacked.mean(axis=0, keepdims=True)
        diff = stacked - mean_ens
        epistemic_var = (diff * diff).mean(axis=0)  # (B, obs_dim)
        # Max over dimensions
        return epistemic_var.mean(axis=-1, keepdims=True)

    def add_experience(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """Add transition to real environment replay buffer."""
        self.env_buffer.add(obs, action, reward, next_obs, done)

    def train_dynamics(
        self,
        batch_size: int = 64,
        num_epochs: int = 5,
        epochs: int | None = None,
    ) -> dict[str, float]:
        """Train ensemble dynamics on real experience and update horizon threshold."""
        if epochs is not None:
            num_epochs = epochs
        if len(self.env_buffer) < batch_size:
            return {
                "loss": 0.0,
                "dynamics_loss": 0.0,
                "tau_threshold": self.scheduler.current_tau,
            }

        total_loss = 0.0
        steps = 0
        for _ in range(num_epochs):
            batch = self.env_buffer.sample(batch_size)
            self.dynamics_opt.zero_grad()
            loss = self.dynamics.compute_loss(
                batch.obs, batch.actions, batch.next_obs, batch.rewards
            )
            loss.backward()
            self.dynamics_opt.step()

            total_loss += float(loss.numpy().item())
            steps += 1

        avg_loss = total_loss / float(max(1, steps))
        tau = self.scheduler.update_threshold(avg_loss)

        return {
            "loss": avg_loss,
            "dynamics_loss": avg_loss,
            "tau_threshold": tau,
        }

    def rollout_adaptive_imagination(
        self,
        num_rollouts: int = 100,
    ) -> dict[str, float]:
        """Execute state-specific adaptive horizon rollouts with uncertainty termination."""
        if len(self.env_buffer) == 0:
            return {"transitions_added": 0.0, "mean_horizon": 0.0, "mean_uncertainty": 0.0}

        # Sample initial states from real buffer
        init_batch = self.env_buffer.sample(num_rollouts)
        curr_states = np.copy(init_batch.obs.numpy())

        branch_active = np.ones(num_rollouts, dtype=bool)
        branch_budgets = np.zeros(num_rollouts, dtype=np.float64)
        branch_horizons = np.zeros(num_rollouts, dtype=np.int32)
        total_uncertainties: list[float] = []

        transitions_added = 0

        for h in range(self.scheduler.horizon_max):
            active_indices = np.where(branch_active)[0]
            if len(active_indices) == 0:
                break

            active_states = curr_states[active_indices]

            # Generate actions under actor policy
            actions = np.zeros((len(active_indices), self.action_dim), dtype=np.float64)
            for i, s in enumerate(active_states):
                actions[i] = self.actor.act(s, deterministic=False)

            # Predict dynamics and compute epistemic uncertainty
            next_states, rewards, _ = self.dynamics.predict(active_states, actions)
            # Epistemic variance across ensemble: (N_active, obs_dim)
            epi_var = np.var(next_states, axis=0)
            scalar_epi = np.max(epi_var, axis=-1)  # (N_active,)

            # TS1 trajectory sampling: random ensemble member per branch
            member_indices = np.random.randint(
                0, self.dynamics.ensemble_size, size=len(active_indices)
            )

            for local_idx, global_idx in enumerate(active_indices):
                u_val = float(scalar_epi[local_idx])
                total_uncertainties.append(u_val)

                # Check adaptive horizon termination
                trunc, new_b = self.scheduler.should_truncate(h, u_val, branch_budgets[global_idx])
                branch_budgets[global_idx] = new_b

                if trunc:
                    branch_active[global_idx] = False
                    continue

                # Add transition to model buffer
                m_idx = member_indices[local_idx]
                s_next = next_states[m_idx, local_idx]
                r = float(rewards[m_idx, local_idx, 0])
                done = False

                self.model_buffer.add(active_states[local_idx], actions[local_idx], r, s_next, done)
                curr_states[global_idx] = s_next
                branch_horizons[global_idx] += 1
                transitions_added += 1

        self.last_mean_horizon = (
            float(np.mean(branch_horizons)) if len(branch_horizons) > 0 else 1.0
        )
        self.last_mean_uncertainty = (
            float(np.mean(total_uncertainties)) if total_uncertainties else 0.0
        )

        return {
            "transitions_added": float(transitions_added),
            "mean_horizon": self.last_mean_horizon,
            "mean_uncertainty": self.last_mean_uncertainty,
            "mean_epistemic_uncertainty": self.last_mean_uncertainty,
        }

    def generate_adaptive_rollouts(
        self,
        num_rollouts: int = 100,
        batch_size: int | None = None,
    ) -> dict[str, float]:
        """Alias for rollout_adaptive_imagination with dynamic blending stats."""
        n = batch_size if batch_size is not None else num_rollouts
        res = self.rollout_adaptive_imagination(num_rollouts=n)
        return {
            "transitions_added": res["transitions_added"],
            "mean_horizon": res["mean_horizon"],
            "mean_uncertainty": res["mean_uncertainty"],
            "mean_epistemic_uncertainty": res["mean_uncertainty"],
            "blending_eta": self.blending_controller.current_eta,
        }

    def sample_mixed_batch(self, batch_size: int = 64) -> ReplayBatch:
        """Sample mini-batch mixing real and synthetic experience using dynamic blending."""
        eta = self.blending_controller.compute_ratio(self.last_mean_uncertainty)
        model_size = int(batch_size * eta)
        env_size = batch_size - model_size

        if len(self.model_buffer) < model_size or model_size == 0:
            return self.env_buffer.sample(batch_size)

        if len(self.env_buffer) < env_size or env_size == 0:
            return self.model_buffer.sample(batch_size)

        env_batch = self.env_buffer.sample(env_size)
        model_batch = self.model_buffer.sample(model_size)

        return ReplayBatch(
            obs=concat([env_batch.obs, model_batch.obs], axis=0),
            actions=concat([env_batch.actions, model_batch.actions], axis=0),
            rewards=concat([env_batch.rewards, model_batch.rewards], axis=0),
            next_obs=concat([env_batch.next_obs, model_batch.next_obs], axis=0),
            dones=concat([env_batch.dones, model_batch.dones], axis=0),
        )

    def train_policy(self, num_updates: int = 1, batch_size: int = 64) -> dict[str, float]:
        """Update policy with Bellman targets and epistemic risk-sensitive pessimistic penalty."""
        if len(self.env_buffer) < batch_size:
            return {
                "critic_loss": 0.0,
                "actor_loss": 0.0,
                "alpha_loss": 0.0,
                "alpha": self.alpha,
                "eta": self.blending_controller.current_eta,
            }

        metrics_sum: dict[str, float] = {
            "critic_loss": 0.0,
            "actor_loss": 0.0,
            "alpha_loss": 0.0,
            "pessimism_penalty": 0.0,
        }

        for _ in range(num_updates):
            batch = self.sample_mixed_batch(batch_size)
            obs = batch.obs
            actions = batch.actions
            rewards = batch.rewards.reshape(-1, 1)
            next_obs = batch.next_obs
            dones = batch.dones.reshape(-1, 1)

            alpha = self.alpha

            # 1. Critic Update
            next_actions, next_log_probs = self.actor(next_obs)
            next_log_probs = next_log_probs.reshape(-1, 1)

            q1_target, q2_target = self.critic_target(next_obs, next_actions)
            mask_target_np = (q1_target.numpy() <= q2_target.numpy()).astype(np.float64)
            mask_target = tensor(mask_target_np, requires_grad=False)
            min_q_target = mask_target * q1_target + (1.0 - mask_target) * q2_target

            soft_target_v = min_q_target - next_log_probs * alpha
            target_q_data = (
                rewards.numpy() + self.gamma * (1.0 - dones.numpy()) * soft_target_v.numpy()
            )
            target_q = tensor(target_q_data, requires_grad=False)

            q1, q2 = self.critic(obs, actions)
            diff1 = q1 - target_q
            diff2 = q2 - target_q
            critic_loss = 0.5 * (diff1 * diff1).mean() + 0.5 * (diff2 * diff2).mean()

            self.critic_opt.zero_grad()
            critic_loss.backward()
            self.critic_opt.step()

            # 2. Pessimistic Actor Update
            new_actions, log_probs = self.actor(obs)
            log_probs = log_probs.reshape(-1, 1)

            q1_new, q2_new = self.critic(obs, new_actions)
            mask_new_np = (q1_new.numpy() <= q2_new.numpy()).astype(np.float64)
            mask_new = tensor(mask_new_np, requires_grad=False)
            min_q_new = mask_new * q1_new + (1.0 - mask_new) * q2_new

            # Epistemic risk-sensitive penalty: - beta_pess * u_epi
            u_epi_val = self.compute_epistemic_uncertainty_numpy(obs.numpy(), new_actions.numpy())
            pess_penalty = tensor(u_epi_val.reshape(-1, 1) * self.beta_pess, requires_grad=False)
            pessimistic_q = min_q_new - pess_penalty

            actor_loss = (log_probs * alpha - pessimistic_q).mean()

            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

            # 3. Temperature Update
            lp_np = log_probs.numpy()
            target_diff = tensor(lp_np + self.target_entropy, requires_grad=False)
            alpha_loss = -(self.log_alpha * target_diff).mean()

            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()

            self._polyak_update()

            metrics_sum["critic_loss"] += float(critic_loss.numpy().item())
            metrics_sum["actor_loss"] += float(actor_loss.numpy().item())
            metrics_sum["alpha_loss"] += float(alpha_loss.numpy().item())
            metrics_sum["pessimism_penalty"] += float(pess_penalty.numpy().mean().item())

        res = {k: v / float(num_updates) for k, v in metrics_sum.items()}
        res["alpha"] = self.alpha
        res["eta"] = self.blending_controller.current_eta
        res["mean_horizon"] = self.last_mean_horizon
        return res

    def train_policy_step(self, batch_size: int = 64) -> dict[str, float]:
        """Convenience alias for train_policy with a single update step."""
        return self.train_policy(num_updates=1, batch_size=batch_size)
