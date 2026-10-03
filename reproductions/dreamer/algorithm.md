# Dreamer: Latent Imagination Actor-Critic

## 1. Purpose & Background
Dreamer (Hafner et al., 2020, 2021) establishes that reinforcement learning can be conducted entirely within the learned latent space of a world model. Instead of decoding observations to generate synthetic environment transitions, Dreamer unrolls an actor and a value model purely through the Recurrent State-Space Model (RSSM), optimizing policy parameters via multi-step generalized $\lambda$-returns.

## 2. Mathematical Equations
1. **RSSM Dynamics Unroll**:
   At imagination step $\tau$:
   - $\hat{a}_\tau \sim \pi_\psi(\cdot \mid \hat{h}_\tau, \hat{z}_\tau)$
   - $\hat{h}_{\tau+1} = \text{GRUCell}([\hat{z}_\tau, \hat{a}_\tau], \hat{h}_\tau)$
   - $\hat{z}_{\tau+1} \sim p_\theta(\cdot \mid \hat{h}_{\tau+1}) = \mathcal{N}(\boldsymbol{\mu}_{\tau+1}, \boldsymbol{\sigma}_{\tau+1}^2)$
   - $\hat{r}_\tau = g_\theta^r(\hat{h}_\tau, \hat{z}_\tau), \quad \hat{\gamma}_\tau = \sigma(g_\theta^\gamma(\hat{h}_\tau, \hat{z}_\tau))$
2. **Latent Generalized Advantage Estimation ($\lambda$-Returns)**:
   Recursive calculation from horizon $H$:
   $$V_H^\lambda = v_\xi(\hat{h}_H, \hat{z}_H)$$
   $$V_t^\lambda = \hat{r}_t + \gamma \hat{\gamma}_t \left[ (1 - \lambda) v_\xi(\hat{h}_{t+1}, \hat{z}_{t+1}) + \lambda V_{t+1}^\lambda \right], \quad t = H-1 \dots 0$$
3. **Critic Loss**:
   $$\mathcal{L}_v(\xi) = \frac{1}{2 B H} \sum_{t=0}^{H-1} \left( v_\xi(\hat{h}_t, \hat{z}_t) - \text{stop\_grad}(V_t^\lambda) \right)^2$$
4. **Actor Loss**:
   $$\mathcal{L}_\pi(\psi) = - \frac{1}{B H} \sum_{t=0}^{H-1} \left( V_t^\lambda + \eta_{\text{ent}} \mathcal{H}(\pi_\psi(\cdot \mid \hat{h}_t, \hat{z}_t)) \right)$$

## 3. Algorithm Pseudocode
```text
Initialize RSSM, Actor pi_psi, Critic v_xi, ReplayBuffer D
For step t = 1 to T_total:
    Observe x_t, infer posterior state (h_t, z_t)
    Sample action a_t ~ pi_psi(h_t, z_t), step environment
    Add transition (x_t, a_t, r_t, d_t) to D
    Sample trajectory batch from D:
        Update RSSM parameters on observation, reward, continuation, and KL loss
        Extract posterior latent states (h_0, z_0)
        Unroll imagination horizon H under actor pi_psi
        Compute lambda-returns V_t^lambda backwards
        Update critic v_xi towards V_t^lambda
        Update actor pi_psi to maximize V_t^lambda
```
