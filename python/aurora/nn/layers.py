"""Neural network layers, activations, and composable modules for AURORA."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, cast

import numpy as np

from aurora.nn.module import Module
from aurora.nn.parameter import Parameter
from aurora.tensor import Tensor, tensor


class Linear(Module):
    """Linear transformation: Y = X W^T + b."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Kaiming uniform initialization
        k = 1.0 / math.sqrt(float(in_features))
        w_data = np.random.uniform(-k, k, size=(out_features, in_features))
        self.weight = Parameter(w_data)

        if bias:
            b_data = np.random.uniform(-k, k, size=(out_features,))
            self.bias: Parameter | None = Parameter(b_data)
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        out = x @ self.weight.transpose()
        if self.bias is not None:
            out = out + self.bias
        return out


class Embedding(Module):
    """Lookup table for discrete token embeddings."""

    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # Standard normal initialization
        w_data = np.random.randn(num_embeddings, embedding_dim)
        self.weight = Parameter(w_data)

    def forward(self, indices: Sequence[int] | np.ndarray | Tensor) -> Tensor:
        if isinstance(indices, Tensor):
            idx_arr = indices.numpy().astype(np.int64)
        elif isinstance(indices, np.ndarray):
            idx_arr = indices.astype(np.int64)
        else:
            idx_arr = np.array(indices, dtype=np.int64)

        selected_data = self.weight.numpy()[idx_arr]
        out = tensor(selected_data, requires_grad=self.weight.requires_grad)

        # Autograd backward node for embedding lookup
        if self.weight.requires_grad:
            from aurora.autograd import Function

            class EmbeddingBackward(Function):
                def __init__(self, weight: Parameter, idx: np.ndarray) -> None:
                    super().__init__(weight)
                    self.idx = idx

                def backward(self, grad_output: np.ndarray) -> tuple[np.ndarray | None, ...]:
                    (w,) = self.inputs
                    if not w.requires_grad:
                        return (None,)
                    gw = np.zeros_like(w.data)
                    np.add.at(gw, self.idx, grad_output)
                    return (gw,)

            out.creator = EmbeddingBackward(self.weight, idx_arr)

        return out


class LayerNorm(Module):
    """Layer Normalization over the last dimension."""

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
    ) -> None:
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if elementwise_affine:
            self.weight: Parameter | None = Parameter(np.ones(normalized_shape))
            self.bias: Parameter | None = Parameter(np.zeros(normalized_shape))
        else:
            self.weight = None
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        return x.layer_norm(gamma=self.weight, beta=self.bias, eps=self.eps, axis=-1)


class RMSNorm(Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = Parameter(np.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        # RMS(x) = sqrt(mean(x^2) + eps)
        x2 = x * x
        mean_x2 = x2.mean(axis=-1, keepdims=True)
        rms = (mean_x2 + self.eps).sqrt()
        x_norm = x / rms
        return x_norm * self.weight


class Dropout(Module):
    """Inverted Dropout module."""

    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError(f"Dropout probability must be in [0, 1), got {p}")
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0.0:
            return x
        q = 1.0 - self.p
        mask_data = (np.random.rand(*x.shape) < q).astype(x.dtype) / q
        mask_t = tensor(mask_data, requires_grad=False)
        return x * mask_t


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.relu()


class GELU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.gelu()


class SiLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.silu()


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.tanh()


class Sigmoid(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.sigmoid()


class Softmax(Module):
    def __init__(self, axis: int = -1) -> None:
        super().__init__()
        self.axis = axis

    def forward(self, x: Tensor) -> Tensor:
        return x.softmax(axis=self.axis)


class LogSoftmax(Module):
    def __init__(self, axis: int = -1) -> None:
        super().__init__()
        self.axis = axis

    def forward(self, x: Tensor) -> Tensor:
        return x.log_softmax(axis=self.axis)


class Sequential(Module):
    """Sequential container of modules."""

    def __init__(self, *modules: Module) -> None:
        super().__init__()
        for i, mod in enumerate(modules):
            setattr(self, f"layer_{i}", mod)

    def forward(self, x: Any) -> Any:
        out = x
        for mod in self._modules.values():
            out = mod(out)
        return out


class MLP(Module):
    """Multi-Layer Perceptron architecture."""

    def __init__(
        self,
        in_features: int,
        hidden_dims: Sequence[int],
        out_features: int,
        activation: str = "relu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers: list[Module] = []
        curr_in = in_features

        def get_act(name: str) -> Module:
            match name.lower():
                case "relu":
                    return ReLU()
                case "gelu":
                    return GELU()
                case "silu":
                    return SiLU()
                case "tanh":
                    return Tanh()
                case "sigmoid":
                    return Sigmoid()
                case _:
                    raise ValueError(f"Unknown activation: {name}")

        for h_dim in hidden_dims:
            layers.append(Linear(curr_in, h_dim))
            layers.append(get_act(activation))
            if dropout > 0.0:
                layers.append(Dropout(p=dropout))
            curr_in = h_dim

        layers.append(Linear(curr_in, out_features))
        self.net = Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        out: Tensor = self.net(x)
        return out


class ResidualBlock(Module):
    """Residual skip-connection block: y = shortcut(x) + block(x)."""

    def __init__(self, block: Module, shortcut: Module | None = None) -> None:
        super().__init__()
        self.block = block
        self.shortcut = shortcut

    def forward(self, x: Tensor) -> Tensor:
        res = cast(Tensor, self.block(x))
        sc = cast(Tensor, self.shortcut(x)) if self.shortcut is not None else x
        return sc + res


class ModuleList(Module):
    """Holds submodules in a list and properly registers them with the Module system."""

    def __init__(self, modules: Sequence[Module] | None = None) -> None:
        super().__init__()
        self._module_list: list[Module] = []
        if modules is not None:
            for mod in modules:
                self.append(mod)

    def append(self, module: Module) -> None:
        idx = len(self._module_list)
        self._module_list.append(module)
        setattr(self, str(idx), module)

    def extend(self, modules: Sequence[Module]) -> None:
        for mod in modules:
            self.append(mod)

    def __len__(self) -> int:
        return len(self._module_list)

    def __iter__(self) -> Any:
        return iter(self._module_list)

    def __getitem__(self, idx: int) -> Module:
        return self._module_list[idx]
