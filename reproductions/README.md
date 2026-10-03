# MBRL Research Lineage & Reproductions

This directory houses faithful, modular reproductions of foundational Model-Based Reinforcement Learning (MBRL) algorithms, implemented from mathematical first principles using AURORA's native tensor, autograd, and neural network engines.

---

## 1. Research Lineages & Architectures

| Method | Key Reference | Core Mechanism | AURORA Subsystem Integration |
| :--- | :--- | :--- | :--- |
| **MBPO** | Janner et al. (NeurIPS 2019) | $k$-step branched ensemble rollouts + hybrid SAC replay | `world_model.EnsembleDynamicsModel`, `rl.SAC` |
| **Dreamer** | Hafner et al. (ICLR 2020, 2021) | RSSM latent imagination + analytical $\lambda$-returns | `world_model.RSSM`, `world_model.GRUCell` |
| **TD-MPC** | Hansen et al. (ICML 2022) | Latent trajectory shooting via Cross-Entropy Method (CEM) | `nn.MLP`, `tensor.Tensor` |
| **MuZero** | Schrittwieser et al. (Nature 2020) | Latent Monte Carlo Tree Search (MCTS) with PUCT | `nn.Linear`, `nn.Sequential` |
| **Decision Transformer** | Chen et al. (NeurIPS 2021) | Return-to-Go (RTG) conditioned causal sequence modeling | `nn.TransformerDecoder`, `nn.Embedding` |

---

## 2. Directory Layout

```text
reproductions/
├── README.md
├── mbpo/
│   ├── algorithm.md
│   └── mbpo.py
├── dreamer/
│   ├── algorithm.md
│   └── dreamer.py
├── tdmpc/
│   ├── algorithm.md
│   └── tdmpc.py
├── muzero/
│   ├── algorithm.md
│   └── muzero.py
└── decision_transformer/
    ├── algorithm.md
    └── decision_transformer.py
```

---

## 3. Comparative Analysis & Scientific Objective

Each reproduction serves as an empirical and conceptual anchor for AURORA:
1. **MBPO** demonstrates the power and vulnerability of fixed-horizon ensemble rollouts, motivating AURORA's dynamic uncertainty truncation.
2. **Dreamer** demonstrates value optimization purely in continuous/stochastic latent space, motivating AURORA's latent dynamics module.
3. **TD-MPC** highlights sample-efficient test-time planning with terminal Q-value bootstrapping without pixel decoder overhead.
4. **MuZero** showcases discrete search in learned dynamics models.
5. **Decision Transformer** demonstrates offline conditional policy synthesis using autoregressive transformers.

---

## 4. Verification Protocol
Each algorithm is accompanied by:
- Mathematical documentation (`algorithm.md`).
- Deterministic unit tests in `tests/python/test_reproductions.py`.
- Native C++23 implementations in `cpp/include/aurora/reproductions.hpp` and `cpp/src/reproductions.cpp`.
- Cross-language numerical parity tests asserting $< 10^{-10}$ error in `tests/parity/test_reproductions_parity.py`.
