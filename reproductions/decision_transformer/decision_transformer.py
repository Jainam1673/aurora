"""Decision Transformer (Chen et al., 2021) sequence modeling reproduction."""

from __future__ import annotations

import numpy as np
from aurora.nn.layers import Embedding, LayerNorm, Linear, ModuleList
from aurora.nn.module import Module
from aurora.nn.transformer import TransformerBlock
from aurora.optim.adamw import AdamW
from aurora.tensor import Tensor, stack, tensor


class DecisionTransformer(Module):
    """Decision Transformer autoregressive sequence model for RL."""

    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: int | None = None,
        max_length: int = 30,
        max_ep_len: int = 1000,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.d_model = d_model
        self.max_length = max_length
        self.max_ep_len = max_ep_len

        # Token embedding projections
        self.embed_timestep = Embedding(max_ep_len, d_model)
        self.embed_rtg = Linear(1, d_model)
        self.embed_state = Linear(state_dim, d_model)
        self.embed_action = Linear(act_dim, d_model)

        # Pre-LN Transformer blocks
        self.blocks = ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=n_heads,
                    d_ff=d_ff,
                    norm_type="layernorm",
                    activation="gelu",
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )
        self.final_norm = LayerNorm(d_model)

        # Action prediction head
        self.predict_action = Linear(d_model, act_dim)

    def forward(
        self,
        states: Tensor,
        actions: Tensor,
        returns_to_go: Tensor,
        timesteps: Tensor,
    ) -> Tensor:
        """Forward pass for trajectory sequence.

        Args:
            states: (B, T, state_dim)
            actions: (B, T, act_dim)
            returns_to_go: (B, T, 1)
            timesteps: (B, T) integer timestep indices

        Returns:
            action_preds: (B, T, act_dim) predicted actions at state positions.
        """
        batch_size, seq_len, _ = states.shape

        time_emb = self.embed_timestep(timesteps)  # (B, T, d_model)

        # Embed each modality with shared timestep embedding
        rtg_emb = self.embed_rtg(returns_to_go) + time_emb  # (B, T, d_model)
        state_emb = self.embed_state(states) + time_emb  # (B, T, d_model)
        action_emb = self.embed_action(actions) + time_emb  # (B, T, d_model)

        # Interleave tokens into sequence: (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # Shape: (B, T, 3, d_model) -> reshape to (B, 3 * T, d_model)
        stacked = stack([rtg_emb, state_emb, action_emb], dim=2)
        sequence = stacked.reshape(batch_size, 3 * seq_len, self.d_model)

        # Causal attention mask: upper triangular masked
        total_len = 3 * seq_len
        causal_mask = np.tril(np.ones((total_len, total_len), dtype=np.float64))
        mask_t = tensor(causal_mask)

        # Forward through transformer blocks
        h = sequence
        for block in self.blocks:
            h = block(h, mask=mask_t, is_causal=True)

        h = self.final_norm(h)

        # Reshape to (B, T, 3, d_model) and select state token positions (index 1)
        h_interleaved = h.reshape(batch_size, seq_len, 3, self.d_model)
        h_states = h_interleaved[:, :, 1, :]  # (B, T, d_model)

        # Predict actions from state token representations
        action_preds: Tensor = self.predict_action(h_states).tanh()
        return action_preds

    def get_action(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        returns_to_go: np.ndarray,
        timesteps: np.ndarray,
    ) -> np.ndarray:
        """Autoregressive inference: predict next action given context window."""
        s = states[-self.max_length :]
        a = actions[-self.max_length :]
        r = returns_to_go[-self.max_length :]
        t = timesteps[-self.max_length :]

        t_states = tensor(s.reshape(1, len(s), self.state_dim))
        t_actions = tensor(a.reshape(1, len(a), self.act_dim))
        t_rtgs = tensor(r.reshape(1, len(r), 1))
        t_steps = tensor(t.reshape(1, len(t)), dtype=np.int64)

        preds = self.forward(t_states, t_actions, t_rtgs, t_steps)
        # Return action prediction for the latest state
        return np.asarray(preds.data[0, -1])


class DecisionTransformerTrainer:
    """Trainer for offline Decision Transformer optimization."""

    def __init__(
        self,
        model: DecisionTransformer,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
    ) -> None:
        self.model = model
        self.optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    def train_step(
        self,
        states: Tensor,
        actions: Tensor,
        returns_to_go: Tensor,
        timesteps: Tensor,
    ) -> float:
        """Perform a single gradient step on offline trajectory batch."""
        action_preds = self.model(states, actions, returns_to_go, timesteps)
        loss = ((action_preds - actions) ** 2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return float(loss.data.item())
