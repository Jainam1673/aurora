"""MuZero (Schrittwieser et al., 2020) latent MCTS planning reproduction."""

from __future__ import annotations

import math

import numpy as np
from aurora.nn.layers import MLP
from aurora.nn.module import Module
from aurora.optim.adamw import AdamW
from aurora.tensor import Tensor, concat, tensor


class RepresentationNetwork(Module):
    """Encodes observations into root latent states."""

    def __init__(
        self,
        obs_dim: int,
        latent_dim: int,
        hidden_dims: list[int] | None = None,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 128]
        self.net = MLP(obs_dim, hidden_dims, latent_dim, activation=activation)

    def forward(self, obs: Tensor) -> Tensor:
        out: Tensor = self.net(obs)
        return out


class DynamicsNetwork(Module):
    """Predicts next latent state and immediate scalar reward given action."""

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        hidden_dims: list[int] | None = None,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 128]
        self.state_head = MLP(
            latent_dim + action_dim, hidden_dims, latent_dim, activation=activation
        )
        self.reward_head = MLP(latent_dim + action_dim, hidden_dims, 1, activation=activation)

    def forward(self, state: Tensor, action_one_hot: Tensor) -> tuple[Tensor, Tensor]:
        x = concat([state, action_one_hot], axis=-1)
        next_state: Tensor = self.state_head(x)
        reward: Tensor = self.reward_head(x)
        return next_state, reward


class PredictionNetwork(Module):
    """Predicts policy logits and value from latent state."""

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        hidden_dims: list[int] | None = None,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 128]
        self.action_dim = action_dim
        self.policy_head = MLP(latent_dim, hidden_dims, action_dim, activation=activation)
        self.value_head = MLP(latent_dim, hidden_dims, 1, activation=activation)

    def forward(self, state: Tensor) -> tuple[Tensor, Tensor]:
        logits: Tensor = self.policy_head(state)
        value: Tensor = self.value_head(state)
        return logits, value


class MinMaxStats:
    """Tracks empirical minimum and maximum Q-values for normalization."""

    def __init__(self) -> None:
        self.minimum = float("inf")
        self.maximum = float("-inf")

    def update(self, value: float) -> None:
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)

    def normalize(self, value: float) -> float:
        if self.maximum > self.minimum:
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value


class MCTSNode:
    """Monte Carlo Tree Search node in latent space."""

    def __init__(self, prior: float = 0.0) -> None:
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        self.reward = 0.0
        self.latent_state: np.ndarray | None = None
        self.children: dict[int, MCTSNode] = {}

    @property
    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def expanded(self) -> bool:
        return len(self.children) > 0


class MCTS:
    """PUCT MCTS planner operating on learned latent dynamics."""

    def __init__(
        self,
        dynamics: DynamicsNetwork,
        prediction: PredictionNetwork,
        action_dim: int,
        discount: float = 0.99,
        c1: float = 1.25,
        c2: float = 19652.0,
    ) -> None:
        self.dynamics = dynamics
        self.prediction = prediction
        self.action_dim = action_dim
        self.discount = discount
        self.c1 = c1
        self.c2 = c2

    def run(self, root_state: np.ndarray, num_simulations: int = 50) -> tuple[int, np.ndarray]:
        """Perform MCTS search from root latent state."""
        root = MCTSNode()
        root.latent_state = root_state

        # Expand root
        t_state = tensor(root_state.reshape(1, -1))
        logits, root_val = self.prediction(t_state)
        probs = np.exp(logits.data[0] - np.max(logits.data[0]))
        probs = probs / np.sum(probs)

        for a in range(self.action_dim):
            root.children[a] = MCTSNode(prior=float(probs[a]))

        min_max = MinMaxStats()
        min_max.update(float(root_val.data.item()))

        for _ in range(num_simulations):
            node = root
            search_path: list[tuple[MCTSNode, int]] = []

            # 1. Selection
            while node.expanded():
                action, next_node = self._select_child(node, min_max)
                search_path.append((node, action))
                node = next_node

            parent, last_action = search_path[-1]
            parent_state = parent.latent_state
            assert parent_state is not None

            # 2. Dynamics step (Expansion)
            action_one_hot = np.zeros((1, self.action_dim), dtype=np.float32)
            action_one_hot[0, last_action] = 1.0

            next_latent, pred_reward = self.dynamics(
                tensor(parent_state.reshape(1, -1)),
                tensor(action_one_hot),
            )
            node.latent_state = next_latent.data[0]
            node.reward = float(pred_reward.data.item())

            # 3. Prediction
            leaf_logits, leaf_value = self.prediction(next_latent)
            leaf_probs = np.exp(leaf_logits.data[0] - np.max(leaf_logits.data[0]))
            leaf_probs = leaf_probs / np.sum(leaf_probs)
            v_val = float(leaf_value.data.item())
            min_max.update(v_val)

            for a in range(self.action_dim):
                node.children[a] = MCTSNode(prior=float(leaf_probs[a]))

            # 4. Backup
            g = v_val
            for p_node, act in reversed(search_path):
                child = p_node.children[act]
                g = child.reward + self.discount * g
                child.value_sum += g
                child.visit_count += 1
                min_max.update(child.value)

        # Form visit count distribution
        visits = np.array(
            [root.children[a].visit_count for a in range(self.action_dim)], dtype=np.float32
        )
        total_visits = np.sum(visits)
        if total_visits > 0:
            pi_probs = visits / total_visits
        else:
            pi_probs = np.ones(self.action_dim, dtype=np.float32) / self.action_dim

        best_action = int(np.argmax(visits))
        return best_action, pi_probs

    def _select_child(self, node: MCTSNode, min_max: MinMaxStats) -> tuple[int, MCTSNode]:
        total_visits = sum(child.visit_count for child in node.children.values())
        best_score = float("-inf")
        best_action = 0
        best_child = None

        for action, child in node.children.items():
            # PUCT score
            q_val = child.value if child.visit_count > 0 else 0.0
            norm_q = min_max.normalize(q_val)

            pb_c = (math.log((total_visits + self.c2 + 1) / self.c2) + self.c1) * (
                math.sqrt(total_visits) / (1 + child.visit_count)
            )
            u_score = child.prior * pb_c
            score = norm_q + u_score

            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        assert best_child is not None
        return best_action, best_child


class MuZeroAgent(Module):
    """Full MuZero agent integrating representation, dynamics, prediction, and MCTS."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        latent_dim: int = 64,
        hidden_dims: list[int] | None = None,
        discount: float = 0.99,
        lr: float = 3e-4,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 128]

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.discount = discount

        self.representation = RepresentationNetwork(obs_dim, latent_dim, hidden_dims)
        self.dynamics = DynamicsNetwork(latent_dim, action_dim, hidden_dims)
        self.prediction = PredictionNetwork(latent_dim, action_dim, hidden_dims)

        self.optimizer = AdamW(
            list(self.representation.parameters())
            + list(self.dynamics.parameters())
            + list(self.prediction.parameters()),
            lr=lr,
        )

    def plan_and_act(
        self,
        obs: np.ndarray,
        num_simulations: int = 50,
    ) -> tuple[int, np.ndarray]:
        """Perform MCTS planning from raw observation."""
        root_latent = self.representation(tensor(obs.reshape(1, -1))).data[0]
        planner = MCTS(self.dynamics, self.prediction, self.action_dim, self.discount)
        return planner.run(root_latent, num_simulations=num_simulations)

    def update_step(
        self,
        obs: Tensor,
        actions: list[int],
        target_rewards: list[float],
        target_values: list[float],
        target_policies: list[np.ndarray],
    ) -> dict[str, float]:
        """Train unrolled trajectory on sequence of transitions."""
        s = self.representation(obs)
        total_loss = tensor(0.0)

        for _k, (action, target_r, target_v, target_pi) in enumerate(
            zip(actions, target_rewards, target_values, target_policies, strict=True)
        ):
            logits, val = self.prediction(s)

            # Policy cross-entropy: - sum(pi * log_softmax(logits))
            l_sm = logits.log_softmax(dim=-1)
            p_loss = -(l_sm * tensor(target_pi.reshape(1, -1))).sum(dim=-1).mean()

            # Value loss
            v_loss = ((val - tensor([[target_v]])) ** 2).mean()

            # Step dynamics
            a_one_hot = np.zeros((1, self.action_dim), dtype=np.float32)
            a_one_hot[0, action] = 1.0
            next_s, pred_r = self.dynamics(s, tensor(a_one_hot))

            # Reward loss
            r_loss = ((pred_r - tensor([[target_r]])) ** 2).mean()

            step_loss = p_loss + v_loss + r_loss
            total_loss = total_loss + step_loss
            s = next_s

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return {"loss": float(total_loss.data.item())}
