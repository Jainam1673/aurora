"""Tensor abstraction and automatic differentiation engine for AURORA."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from aurora.autograd import (
    AddBackward,
    ClampBackward,
    ConcatBackward,
    DivBackward,
    ExpBackward,
    Function,
    GELUBackward,
    LayerNormBackward,
    LogBackward,
    LogSoftmaxBackward,
    MatmulBackward,
    MeanBackward,
    MulBackward,
    PowBackward,
    ReLUBackward,
    ReshapeBackward,
    SigmoidBackward,
    SiLUBackward,
    SliceBackward,
    SoftmaxBackward,
    SqrtBackward,
    SubBackward,
    SumBackward,
    TanhBackward,
    TransposeBackward,
)


class Tensor:
    """N-dimensional tensor supporting reverse-mode automatic differentiation."""

    __slots__ = ("_data", "_grad", "creator", "requires_grad")

    def __init__(
        self,
        data: Any,
        requires_grad: bool = False,
        dtype: np.dtype | type = np.float64,
    ) -> None:
        if isinstance(data, np.ndarray):
            self._data = data.astype(dtype, copy=False)
        else:
            self._data = np.array(data, dtype=dtype)

        self.requires_grad = requires_grad
        self._grad: Tensor | None = None
        self.creator: Function | None = None

    @property
    def data(self) -> np.ndarray:
        return self._data

    @data.setter
    def data(self, value: np.ndarray) -> None:
        self._data = value

    @property
    def grad(self) -> Tensor | None:
        return self._grad

    @grad.setter
    def grad(self, value: Tensor | np.ndarray | None) -> None:
        if value is None:
            self._grad = None
        elif isinstance(value, Tensor):
            self._grad = value
        else:
            self._grad = Tensor(value, dtype=self._data.dtype)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(int(s) for s in self._data.shape)

    @property
    def strides(self) -> tuple[int, ...]:
        return tuple(int(s) for s in self._data.strides)

    @property
    def ndim(self) -> int:
        return self._data.ndim

    @property
    def size(self) -> int:
        return self._data.size

    @property
    def dtype(self) -> np.dtype:
        return self._data.dtype

    def numpy(self) -> np.ndarray:
        return self._data

    def item(self) -> float | int:
        val = self._data.item()
        return val if isinstance(val, (float, int)) else float(val)

    def zero_grad(self) -> None:
        self._grad = None

    def __repr__(self) -> str:
        grad_str = f", requires_grad={self.requires_grad}" if self.requires_grad else ""
        return f"aurora.Tensor({self._data}{grad_str})"

    def __getitem__(self, key: Any) -> Tensor:
        out_data = self._data[key]
        out = Tensor(out_data, requires_grad=self.requires_grad, dtype=self.dtype)
        if self.requires_grad:
            out.creator = SliceBackward(self, key)
        return out

    # --- Factory Helpers ---
    @staticmethod
    def _ensure_tensor(other: Any, dtype: np.dtype | type = np.float64) -> Tensor:
        if isinstance(other, Tensor):
            return other
        return Tensor(other, dtype=dtype)

    # --- Elementwise Arithmetic ---
    def __add__(self, other: Any) -> Tensor:
        other_t = self._ensure_tensor(other, self.dtype)
        req_grad = self.requires_grad or other_t.requires_grad
        out = Tensor(self._data + other_t._data, requires_grad=req_grad, dtype=self.dtype)
        if req_grad:
            out.creator = AddBackward(self, other_t)
        return out

    def __radd__(self, other: Any) -> Tensor:
        return self.__add__(other)

    def __sub__(self, other: Any) -> Tensor:
        other_t = self._ensure_tensor(other, self.dtype)
        req_grad = self.requires_grad or other_t.requires_grad
        out = Tensor(self._data - other_t._data, requires_grad=req_grad, dtype=self.dtype)
        if req_grad:
            out.creator = SubBackward(self, other_t)
        return out

    def __rsub__(self, other: Any) -> Tensor:
        other_t = self._ensure_tensor(other, self.dtype)
        return other_t.__sub__(self)

    def __mul__(self, other: Any) -> Tensor:
        other_t = self._ensure_tensor(other, self.dtype)
        req_grad = self.requires_grad or other_t.requires_grad
        out = Tensor(self._data * other_t._data, requires_grad=req_grad, dtype=self.dtype)
        if req_grad:
            out.creator = MulBackward(self, other_t)
        return out

    def __rmul__(self, other: Any) -> Tensor:
        return self.__mul__(other)

    def __truediv__(self, other: Any) -> Tensor:
        other_t = self._ensure_tensor(other, self.dtype)
        req_grad = self.requires_grad or other_t.requires_grad
        out = Tensor(self._data / other_t._data, requires_grad=req_grad, dtype=self.dtype)
        if req_grad:
            out.creator = DivBackward(self, other_t)
        return out

    def __rtruediv__(self, other: Any) -> Tensor:
        other_t = self._ensure_tensor(other, self.dtype)
        return other_t.__truediv__(self)

    def __neg__(self) -> Tensor:
        return self * -1.0

    def __matmul__(self, other: Any) -> Tensor:
        other_t = self._ensure_tensor(other, self.dtype)
        req_grad = self.requires_grad or other_t.requires_grad
        out = Tensor(np.matmul(self._data, other_t._data), requires_grad=req_grad, dtype=self.dtype)
        if req_grad:
            out.creator = MatmulBackward(self, other_t)
        return out

    def __pow__(self, power: float | int) -> Tensor:
        out = Tensor(self._data ** float(power), requires_grad=self.requires_grad, dtype=self.dtype)
        if self.requires_grad:
            out.creator = PowBackward(self, power)
        return out

    # --- Shape Manipulation ---
    def reshape(self, *shape: int | Sequence[int]) -> Tensor:
        target_shape = (
            shape[0] if (len(shape) == 1 and isinstance(shape[0], (tuple, list))) else shape
        )
        out = Tensor(
            self._data.reshape(target_shape),
            requires_grad=self.requires_grad,
            dtype=self.dtype,
        )
        if self.requires_grad:
            out.creator = ReshapeBackward(self, self.shape)
        return out

    def transpose(self, *axes: int) -> Tensor:
        if not axes:
            axes = tuple(range(self.ndim))[::-1]
        out = Tensor(
            np.transpose(self._data, axes),
            requires_grad=self.requires_grad,
            dtype=self.dtype,
        )
        if self.requires_grad:
            out.creator = TransposeBackward(self, axes)
        return out

    def swapaxes(self, axis1: int, axis2: int) -> Tensor:
        a1 = axis1 if axis1 >= 0 else self.ndim + axis1
        a2 = axis2 if axis2 >= 0 else self.ndim + axis2
        axes = list(range(self.ndim))
        axes[a1], axes[a2] = axes[a2], axes[a1]
        return self.transpose(*axes)

    @property
    def mT(self) -> Tensor:
        """Matrix transpose swapping the last two dimensions."""
        if self.ndim < 2:
            raise ValueError(f"mT requires at least 2 dimensions, got {self.ndim}")
        return self.swapaxes(-1, -2)

    # --- Reductions ---
    def sum(
        self,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
        dim: int | tuple[int, ...] | None = None,
    ) -> Tensor:
        if dim is not None:
            axis = dim
        out_data = (
            self._data.sum(axis=axis, keepdims=True)
            if keepdims
            else self._data.sum(axis=axis, keepdims=False)
        )
        out = Tensor(out_data, requires_grad=self.requires_grad, dtype=self.dtype)
        if self.requires_grad:
            out.creator = SumBackward(self, axis, keepdims, self.shape)
        return out

    def mean(
        self,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
        dim: int | tuple[int, ...] | None = None,
    ) -> Tensor:
        if dim is not None:
            axis = dim
        out_data = (
            self._data.mean(axis=axis, keepdims=True)
            if keepdims
            else self._data.mean(axis=axis, keepdims=False)
        )
        out = Tensor(out_data, requires_grad=self.requires_grad, dtype=self.dtype)
        if self.requires_grad:
            out.creator = MeanBackward(self, axis, keepdims, self.shape)
        return out

    # --- Nonlinearities ---
    def exp(self) -> Tensor:
        out_data = np.exp(self._data)
        out = Tensor(out_data, requires_grad=self.requires_grad, dtype=self.dtype)
        if self.requires_grad:
            out.creator = ExpBackward(self, out_data)
        return out

    def log(self) -> Tensor:
        out_data = np.log(self._data)
        out = Tensor(out_data, requires_grad=self.requires_grad, dtype=self.dtype)
        if self.requires_grad:
            out.creator = LogBackward(self)
        return out

    def sqrt(self) -> Tensor:
        out_data = np.sqrt(self._data)
        out = Tensor(out_data, requires_grad=self.requires_grad, dtype=self.dtype)
        if self.requires_grad:
            out.creator = SqrtBackward(self, out_data)
        return out

    def relu(self) -> Tensor:
        out_data = np.maximum(0.0, self._data)
        out = Tensor(out_data, requires_grad=self.requires_grad, dtype=self.dtype)
        if self.requires_grad:
            out.creator = ReLUBackward(self)
        return out

    def gelu(self) -> Tensor:
        xd = self._data
        cdf = 0.5 * (1.0 + np.vectorize(math.erf)(xd / math.sqrt(2.0)))
        out_data = xd * cdf
        out = Tensor(out_data, requires_grad=self.requires_grad, dtype=self.dtype)
        if self.requires_grad:
            out.creator = GELUBackward(self)
        return out

    def silu(self) -> Tensor:
        xd = self._data
        sig = 1.0 / (1.0 + np.exp(-xd))
        out_data = xd * sig
        out = Tensor(out_data, requires_grad=self.requires_grad, dtype=self.dtype)
        if self.requires_grad:
            out.creator = SiLUBackward(self)
        return out

    def tanh(self) -> Tensor:
        out_data = np.tanh(self._data)
        out = Tensor(out_data, requires_grad=self.requires_grad, dtype=self.dtype)
        if self.requires_grad:
            out.creator = TanhBackward(self, out_data)
        return out

    def sigmoid(self) -> Tensor:
        out_data = 1.0 / (1.0 + np.exp(-self._data))
        out = Tensor(out_data, requires_grad=self.requires_grad, dtype=self.dtype)
        if self.requires_grad:
            out.creator = SigmoidBackward(self, out_data)
        return out

    def clamp(self, min_val: float | None = None, max_val: float | None = None) -> Tensor:
        out_data = np.clip(
            self._data,
            a_min=min_val if min_val is not None else -np.inf,
            a_max=max_val if max_val is not None else np.inf,
        )
        out = Tensor(out_data, requires_grad=self.requires_grad, dtype=self.dtype)
        if self.requires_grad:
            out.creator = ClampBackward(self, min_val, max_val)
        return out

    def clip(self, min_val: float | None = None, max_val: float | None = None) -> Tensor:
        return self.clamp(min_val, max_val)

    def softmax(self, axis: int = -1, dim: int | None = None) -> Tensor:
        if dim is not None:
            axis = dim
        shifted = self._data - np.max(self._data, axis=axis, keepdims=True)
        exp_data = np.exp(shifted)
        out_data = exp_data / np.sum(exp_data, axis=axis, keepdims=True)
        out = Tensor(out_data, requires_grad=self.requires_grad, dtype=self.dtype)
        if self.requires_grad:
            out.creator = SoftmaxBackward(self, out_data, axis)
        return out

    def log_softmax(self, axis: int = -1, dim: int | None = None) -> Tensor:
        if dim is not None:
            axis = dim
        max_val = np.max(self._data, axis=axis, keepdims=True)
        shifted = self._data - max_val
        exp_data = np.exp(shifted)
        sum_exp = np.sum(exp_data, axis=axis, keepdims=True)
        out_data = shifted - np.log(sum_exp)
        out = Tensor(out_data, requires_grad=self.requires_grad, dtype=self.dtype)
        if self.requires_grad:
            s = exp_data / sum_exp
            out.creator = LogSoftmaxBackward(self, s, axis)
        return out

    def layer_norm(
        self,
        gamma: Tensor | None = None,
        beta: Tensor | None = None,
        eps: float = 1e-5,
        axis: int = -1,
    ) -> Tensor:
        mean = np.mean(self._data, axis=axis, keepdims=True)
        var = np.var(self._data, axis=axis, keepdims=True)
        std_inv = 1.0 / np.sqrt(var + eps)
        x_hat = (self._data - mean) * std_inv

        out_data = x_hat
        req_grad = self.requires_grad
        if gamma is not None:
            out_data = out_data * gamma._data
            req_grad = req_grad or gamma.requires_grad
        if beta is not None:
            out_data = out_data + beta._data
            req_grad = req_grad or beta.requires_grad

        out = Tensor(out_data, requires_grad=req_grad, dtype=self.dtype)
        if req_grad:
            out.creator = LayerNormBackward(self, gamma, beta, x_hat, std_inv, axis)
        return out

    # --- Backpropagation Tape ---
    def backward(self, gradient: Tensor | np.ndarray | None = None) -> None:
        """Run reverse-mode automatic differentiation starting from this tensor."""
        if not self.requires_grad:
            raise RuntimeError("Called backward() on a tensor with requires_grad=False")

        if gradient is None:
            if self.size != 1:
                raise RuntimeError("Grad can only be implicitly created for scalar outputs")
            grad_val = np.ones_like(self._data)
        elif isinstance(gradient, Tensor):
            grad_val = gradient._data
        else:
            grad_val = np.array(gradient, dtype=self.dtype)

        # Topological sort of the computation graph DAG
        topo: list[Tensor] = []
        visited: set[int] = set()

        def build_topo(node: Tensor) -> None:
            node_id = id(node)
            if node_id not in visited:
                visited.add(node_id)
                if node.creator is not None:
                    for parent in node.creator.inputs:
                        build_topo(parent)
                topo.append(node)

        build_topo(self)

        # Adjoint dictionary
        grads: dict[int, np.ndarray] = {id(self): grad_val}

        # Backpropagation traversal in reverse topological order
        for node in reversed(topo):
            node_id = id(node)
            g = grads.get(node_id)
            if g is None:
                continue

            # Assign accumulated grad to tensor
            if node._grad is None:
                node._grad = Tensor(g, dtype=node.dtype)
            else:
                node._grad._data += g

            if node.creator is not None:
                parent_grads = node.creator.backward(g)
                for parent, p_grad in zip(node.creator.inputs, parent_grads, strict=False):
                    if p_grad is not None and parent.requires_grad:
                        pid = id(parent)
                        if pid not in grads:
                            grads[pid] = p_grad.copy()
                        else:
                            grads[pid] += p_grad


# --- Factory Functions ---
def tensor(data: Any, requires_grad: bool = False, dtype: np.dtype | type = np.float64) -> Tensor:
    return Tensor(data, requires_grad=requires_grad, dtype=dtype)


def zeros(
    shape: Sequence[int] | int,
    requires_grad: bool = False,
    dtype: np.dtype | type = np.float64,
) -> Tensor:
    return Tensor(np.zeros(shape, dtype=dtype), requires_grad=requires_grad, dtype=dtype)


def ones(
    shape: Sequence[int] | int,
    requires_grad: bool = False,
    dtype: np.dtype | type = np.float64,
) -> Tensor:
    return Tensor(np.ones(shape, dtype=dtype), requires_grad=requires_grad, dtype=dtype)


def randn(*shape: int, requires_grad: bool = False, dtype: np.dtype | type = np.float64) -> Tensor:
    return Tensor(np.random.randn(*shape), requires_grad=requires_grad, dtype=dtype)


def arange(
    start: float,
    stop: float | None = None,
    step: float = 1.0,
    requires_grad: bool = False,
) -> Tensor:
    return Tensor(np.arange(start, stop, step, dtype=np.float64), requires_grad=requires_grad)


def concat(tensors: Sequence[Tensor], axis: int = -1, dim: int | None = None) -> Tensor:
    """Concatenate sequence of tensors along specified axis with autograd support."""
    if dim is not None:
        axis = dim
    if not tensors:
        raise ValueError("Cannot concatenate empty sequence of tensors")

    first = tensors[0]
    ax = axis if axis >= 0 else first.ndim + axis
    split_indices = [int(x) for x in np.cumsum([t.shape[ax] for t in tensors[:-1]])]

    out_data = np.concatenate([t.data for t in tensors], axis=ax)
    req = any(t.requires_grad for t in tensors)
    out = Tensor(out_data, requires_grad=req, dtype=first.dtype)
    if req:
        out.creator = ConcatBackward(tensors, ax, split_indices)
    return out


def stack(tensors: Sequence[Tensor], axis: int = 0, dim: int | None = None) -> Tensor:
    """Stack sequence of tensors along a new axis with autograd support."""
    if dim is not None:
        axis = dim
    if not tensors:
        raise ValueError("Cannot stack empty sequence of tensors")

    first = tensors[0]
    ax = axis if axis >= 0 else first.ndim + 1 + axis

    expanded: list[Tensor] = []
    for t in tensors:
        new_shape = list(t.shape)
        new_shape.insert(ax, 1)
        expanded.append(t.reshape(*new_shape))
    return concat(expanded, axis=ax)
