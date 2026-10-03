# AURORA Algorithm Specification

## 1. Purpose & Scientific Rationale
The AURORA algorithm (**Adaptive Uncertainty-calibrated Rollouts and Optimization for Reinforcement Agents**) provides an uncertainty-grounded model-based reinforcement learning framework. By dynamically adjusting the rollout horizon for each imagined trajectory, modulating the real-to-synthetic data mixture ratio, penalizing policy evaluation in high-disagreement regions, and actively probing epistemic blind spots, AURORA breaks the sample efficiency / compounding error bottleneck.

## 2. Inputs & Outputs
- **Inputs**:
  - Environment transition tuple $(s_t, a_t, r_t, s_{t+1}, d_t)$
  - Deep probabilistic dynamics ensemble $\mathcal{E} = \{f_{\theta_1}, \dots, f_{\theta_E}\}$
  - Twin critic networks $Q_{\phi_1}, Q_{\phi_2}$, policy network $\pi_\psi$, entropy parameter $\alpha$
  - Horizon scheduler parameters: $H_{\max}$, $\tau_{\text{base}}$, $\kappa$, $\mathcal{B}_{\max}$
  - Blending controller parameters: $\eta_{\max}$, $u_{\text{target}}$
  - Pessimism coefficient $\beta_{\text{pess}}$
- **Outputs**:
  - Planned continuous action $a_t \in \mathcal{A}$
  - Calibrated adaptive rollout batch $\mathcal{B}_{\text{model}}$ of varying trajectory lengths
  - Diagnostic metrics: mean rollout horizon $\bar{H}$, mean epistemic uncertainty $\bar{u}$, synthetic ratio $\eta_t$

## 3. Algorithm Pseudocode
```text
Initialize ensemble dynamics {f_theta_e}, actor pi_psi, critics Q_phi_1, Q_phi_2
Initialize real buffer D_env, model buffer D_model
Initialize adaptive horizon scheduler and blending controller

For step t = 1 to Total_Steps:
    Select action a_t = pi_psi(s_t) (+ exploration noise if u_epi(s_t, a_t) > tau_active)
    Execute a_t in environment, observe r_t, s_{t+1}, d_t
    Store (s_t, a_t, r_t, s_{t+1}, d_t) into D_env

    If t % model_train_freq == 0:
        Train ensemble {f_theta_e} on D_env via Gaussian NLL
        Update calibration threshold tau_threshold based on validation NLL

    If t >= min_train_steps:
        // 1. Adaptive Rollout
        For i = 1 to num_rollout_branches:
            Sample s_0 ~ D_env
            Roll out policy pi_psi under ensemble:
                At step h:
                    Predict next_s, r, and epistemic uncertainty u_epi
                    Check adaptive termination:
                        If u_epi > tau_threshold or cumulative_u > B_max:
                            Truncate rollout branch at horizon h
                    Store transition in D_model

        // 2. Dynamic Blending & Policy Optimization
        Compute rolling mean uncertainty u_bar across D_model
        Compute dynamic mixing ratio eta_t = eta_max * (1 - min(1, u_bar / u_target))

        For g = 1 to num_gradient_steps:
            Sample mixed batch B ~ (1 - eta_t) D_env + eta_t D_model
            Update critics Q_phi via Bellman target with target policy smoothing
            Update actor pi_psi with pessimistic value target:
                Loss_pi = E[alpha * log pi - (min(Q1, Q2) - beta_pess * u_epi)]
            Polyak-update target networks
```

## 4. Design Decisions & Numerical Stability Considerations
1. **Variance Clamping**: Log-variance is clamped to $[-10, 2]$ to prevent loss explosion or zero-variance degeneracy.
2. **Horizon Discretization**: Rollout length is bounded in $[1, H_{\max}]$ with $H_{\max} \le 25$, preventing unbounded graph depth and memory overhead.
3. **Pessimism Normalization**: Epistemic disagreement $u_{\text{epi}}$ is scaled by the reward/return scale or normalized across the batch to ensure consistent gradient weighting.
4. **Smooth Ratio Transitions**: The blending ratio $\eta_t$ uses momentum/exponential moving averages to avoid abrupt batch composition swings between training iterations.
