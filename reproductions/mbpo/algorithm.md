# Model-Based Policy Optimization (MBPO)

## 1. Purpose & Background
Model-Based Policy Optimization (MBPO; Janner et al., 2019) is a foundational Dyna-style model-based reinforcement learning algorithm. It demonstrates that training off-policy model-free algorithms (specifically Soft Actor-Critic) on a combination of real experience and short-horizon imagined rollouts from a deep probabilistic dynamics ensemble substantially increases sample efficiency while avoiding compounding model errors.

## 2. Mathematical Equations
1. **Ensemble Dynamics Training**:
   $$\mathcal{L}_{\text{NLL}}(\theta_e) = \frac{1}{2 B} \sum_{i=1}^B \sum_{j=1}^{d_y} \left[ \frac{(y_{i, j} - \mu_{e, j})^2}{\sigma_{e, j}^2} + \log \sigma_{e, j}^2 + \log(2\pi) \right]$$
2. **Branching Rollout**:
   Start from real state $s_0 \sim \mathcal{D}_{\text{env}}$. For step $h = 0 \dots k-1$:
   - Sample action: $a_h \sim \pi_\psi(\cdot \mid s_h)$
   - Sample transition from random ensemble member $e \sim \text{Uniform}(1..E)$:
     $$s_{h+1} = s_h + \boldsymbol{\mu}_{e, \Delta s}(s_h, a_h), \quad r_h = \mu_{e, r}(s_h, a_h)$$
   - Store transition $(s_h, a_h, r_h, s_{h+1}, d_h)$ into model buffer $\mathcal{D}_{\text{model}}$.
3. **Hybrid Replay Sampling**:
   For each gradient step, form minibatch:
   $$\mathcal{B} = \mathcal{B}_{\text{env}} \cup \mathcal{B}_{\text{model}}$$
   where $|\mathcal{B}_{\text{model}}| = \lfloor \eta \cdot B \rfloor$ and $|\mathcal{B}_{\text{env}}| = B - |\mathcal{B}_{\text{model}}|$.

## 3. Algorithm Pseudocode
```text
Initialize ensemble dynamics {f_theta_e}, actor pi_psi, critics Q_phi_1, Q_phi_2
Initialize real replay buffer D_env, model buffer D_model
For step t = 1 to T_total:
    Execute action in environment, add (s, a, r, s', d) to D_env
    If t % model_train_freq == 0:
        Train ensemble {f_theta_e} on D_env via Gaussian NLL
    If t >= min_train_steps:
        For i = 1 to num_rollouts:
            Sample s_0 ~ D_env
            Roll out policy pi under ensemble for k steps -> add to D_model
        For g = 1 to num_gradient_steps:
            Sample mixed batch B ~ (1 - eta) D_env + eta D_model
            Update SAC critics and actor on B
```

## 4. Key Design Decisions & Stability Considerations
- Log-variance clipping $[-10, 2]$ prevents infinite loss or variance collapse.
- Short rollout horizon $k \in [1, 5]$ prevents early policy exploitation before the model has gathered sufficient data.
- Trajectory sampling 1 (TS1): sampling a distinct model member at each step prevents single-model path overfitting.
