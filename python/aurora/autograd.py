"""Reverse-mode automatic differentiation tape and backward nodes for AURORA."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from aurora.tensor import Tensor


def unbroadcast(grad: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    """Sum gradients along broadcasted axes to match the target shape."""
    if grad.shape == target_shape:
        return grad

    # Handle prepended axes (rank differences)
    ndim_diff = grad.ndim - len(target_shape)
    for _ in range(ndim_diff):
        grad = grad.sum(axis=0)

    # Handle axes that were expanded from size 1
    for i, dim in enumerate(target_shape):
        if dim == 1 and grad.shape[i] > 1:
            grad = grad.sum(axis=i, keepdims=True)

    return grad.reshape(target_shape)


class Function(ABC):
    """Abstract base class for autograd operations."""

    def __init__(self, *inputs: Tensor) -> None:
        self.inputs = inputs

    @abstractmethod
    def backward(self, grad_output: np.ndarray) -> tuple[np.ndarray | None, ...]:
        """Compute vector-jacobian products for all inputs."""
        raise NotImplementedError


class AddBackward(Function):
    def backward(self, grad_output: np.ndarray) -> tuple[np.ndarray | None, ...]:
        x, y = self.inputs
        gx = unbroadcast(grad_output, x.shape) if x.requires_grad else None
        gy = unbroadcast(grad_output, y.shape) if y.requires_grad else None
        return gx, gy


class SubBackward(Function):
    def backward(self, grad_output: np.ndarray) -> tuple[np.ndarray | None, ...]:
        x, y = self.inputs
        gx = unbroadcast(grad_output, x.shape) if x.requires_grad else None
        gy = unbroadcast(-grad_output, y.shape) if y.requires_grad else None
        return gx, gy


class MulBackward(Function):
    def backward(self, grad_output: np.ndarray) -> tuple[np.ndarray | None, ...]:
        x, y = self.inputs
        gx = unbroadcast(grad_output * y.data, x.shape) if x.requires_grad else None
        gy = unbroadcast(grad_output * x.data, y.shape) if y.requires_grad else None
        return gx, gy


class DivBackward(Function):
    def backward(self, grad_output: np.ndarray) -> tuple[np.ndarray | None, ...]:
        x, y = self.inputs
        gx = unbroadcast(grad_output / y.data, x.shape) if x.requires_grad else None
        gy = unbroadcast(-grad_output * x.data / (y.data**2), y.shape) if y.requires_grad else None
        return gx, gy


class MatmulBackward(Function):
    def backward(self, grad_output: np.ndarray) -> tuple[np.ndarray | None, ...]:
        a, b = self.inputs
        ga = None
        gb = None
        # Handle 2D and batched matmul
        if a.requires_grad:
            # dL/dA = grad @ B^T
            b_t = np.swapaxes(b.data, -1, -2)
            ga = np.matmul(grad_output, b_t)
            ga = unbroadcast(ga, a.shape)
        if b.requires_grad:
            # dL/dB = A^T @ grad
            a_t = np.swapaxes(a.data, -1, -2)
            gb = np.matmul(a_t, grad_output)
            gb = unbroadcast(gb, b.shape)
        return ga, gb


class ReshapeBackward(Function):
    def __init__(self, x: Tensor, orig_shape: tuple[int, ...]) -> None:
        super().__init__(x)
        self.orig_shape = orig_shape

    def backward(self, grad_output: np.ndarray) -> tuple[np.ndarray | None, ...]:
        (x,) = self.inputs
        gx = grad_output.reshape(self.orig_shape) if x.requires_grad else None
        return (gx,)


class TransposeBackward(Function):
    def __init__(self, x: Tensor, axes: tuple[int, ...]) -> None:
        super().__init__(x)
        # Compute inverse permutation
        inv_axes = [0] * len(axes)
        for i, ax in enumerate(axes):
            inv_axes[ax] = i
        self.inv_axes = tuple(inv_axes)

    def backward(self, grad_output: np.ndarray) -> tuple[np.ndarray | None, ...]:
        (x,) = self.inputs
        gx = np.transpose(grad_output, self.inv_axes) if x.requires_grad else None
        return (gx,)


class SumBackward(Function):
    def __init__(
        self,
        x: Tensor,
        axis: int | tuple[int, ...] | None,
        keepdims: bool,
        orig_shape: tuple[int, ...],
    ) -> None:
        super().__init__(x)
        self.axis = axis
        self.keepdims = keepdims
        self.orig_shape = orig_shape

    def backward(self, grad_output: np.ndarray) -> tuple[np.ndarray | None, ...]:
        (x,) = self.inputs
        if not x.requires_grad:
            return (None,)

        if not self.keepdims and self.axis is not None:
            # Expand dimensions that were reduced
            axes = (self.axis,) if isinstance(self.axis, int) else self.axis
            # Normalize negative axes
            axes = tuple(ax % len(self.orig_shape) for ax in axes)
            expanded_grad = grad_output
            for ax in sorted(axes):
                expanded_grad = np.expand_dims(expanded_grad, axis=ax)
        else:
            expanded_grad = grad_output

        # Broadcast gradient across original shape
        gx = np.broadcast_to(expanded_grad, self.orig_shape).copy()
        return (gx,)


class MeanBackward(Function):
    def __init__(
        self,
        x: Tensor,
        axis: int | tuple[int, ...] | None,
        keepdims: bool,
        orig_shape: tuple[int, ...],
    ) -> None:
        super().__init__(x)
        self.axis = axis
        self.keepdims = keepdims
        self.orig_shape = orig_shape

    def backward(self, grad_output: np.ndarray) -> tuple[np.ndarray | None, ...]:
        (x,) = self.inputs
        if not x.requires_grad:
            return (None,)

        if self.axis is None:
            reduced_elements = x.data.size
        elif isinstance(self.axis, int):
            reduced_elements = self.orig_shape[self.axis]
        else:
            reduced_elements = int(np.prod([self.orig_shape[ax] for ax in self.axis]))
        scale = 1.0 / float(reduced_elements)

        if not self.keepdims and self.axis is not None:
            axes = (self.axis,) if isinstance(self.axis, int) else self.axis
            axes = tuple(ax % len(self.orig_shape) for ax in axes)
            expanded_grad = grad_output
            for ax in sorted(axes):
                expanded_grad = np.expand_dims(expanded_grad, axis=ax)
        else:
            expanded_grad = grad_output

        gx = np.broadcast_to(expanded_grad * scale, self.orig_shape).copy()
        return (gx,)


class ExpBackward(Function):
    def __init__(self, x: Tensor, out_data: np.ndarray) -> None:
        super().__init__(x)
        self.out_data = out_data

    def backward(self, grad_output: np.ndarray) -> tuple[np.ndarray | None, ...]:
        (x,) = self.inputs
        gx = grad_output * self.out_data if x.requires_grad else None
        return (gx,)


class LogBackward(Function):
    def backward(self, grad_output: np.ndarray) -> tuple[np.ndarray | None, ...]:
        (x,) = self.inputs
        gx = grad_output / x.data if x.requires_grad else None
        return (gx,)


class SqrtBackward(Function):
    def __init__(self, x: Tensor, out_data: np.ndarray) -> None:
        super().__init__(x)
        self.out_data = out_data

    def backward(self, grad_output: np.ndarray) -> tuple[np.ndarray | None, ...]:
        (x,) = self.inputs
        gx = grad_output / (2.0 * self.out_data) if x.requires_grad else None
        return (gx,)


class PowBackward(Function):
    def __init__(self, x: Tensor, exponent: float | int) -> None:
        super().__init__(x)
        self.exponent = float(exponent)

    def backward(self, grad_output: np.ndarray) -> tuple[np.ndarray | None, ...]:
        (x,) = self.inputs
        if not x.requires_grad:
            return (None,)
        p = self.exponent
        gx = grad_output * (p * (x.data ** (p - 1.0)))
        return (gx,)


class ReLUBackward(Function):
    def backward(self, grad_output: np.ndarray) -> tuple[np.ndarray | None, ...]:
        (x,) = self.inputs
        gx = grad_output * (x.data > 0.0) if x.requires_grad else None
        return (gx,)


class GELUBackward(Function):
    def backward(self, grad_output: np.ndarray) -> tuple[np.ndarray | None, ...]:
        (x,) = self.inputs
        if not x.requires_grad:
            return (None,)
        xd = x.data
        inv_sqrt2 = 1.0 / math.sqrt(2.0)
        inv_sqrt_2pi = 1.0 / math.sqrt(2.0 * math.pi)
        cdf = 0.5 * (1.0 + np.vectorize(math.erf)(xd * inv_sqrt2))
        pdf = inv_sqrt_2pi * np.exp(-0.5 * (xd**2))
        grad_x = cdf + xd * pdf
        return (grad_output * grad_x,)


class SiLUBackward(Function):
    def backward(self, grad_output: np.ndarray) -> tuple[np.ndarray | None, ...]:
        (x,) = self.inputs
        if not x.requires_grad:
            return (None,)
        xd = x.data
        sig = 1.0 / (1.0 + np.exp(-xd))
        grad_x = sig * (1.0 + xd * (1.0 - sig))
        return (grad_output * grad_x,)


class TanhBackward(Function):
    def __init__(self, x: Tensor, out_data: np.ndarray) -> None:
        super().__init__(x)
        self.out_data = out_data

    def backward(self, grad_output: np.ndarray) -> tuple[np.ndarray | None, ...]:
        (x,) = self.inputs
        if not x.requires_grad:
            return (None,)
        dx = grad_output * (1.0 - self.out_data**2)
        return (dx,)


class SigmoidBackward(Function):
    def __init__(self, x: Tensor, out_data: np.ndarray) -> None:
        super().__init__(x)
        self.out_data = out_data

    def backward(self, grad_output: np.ndarray) -> tuple[np.ndarray | None, ...]:
        (x,) = self.inputs
        if not x.requires_grad:
            return (None,)
        dx = grad_output * self.out_data * (1.0 - self.out_data)
        return (dx,)


class ClampBackward(Function):
    def __init__(self, x: Tensor, min_val: float | None, max_val: float | None) -> None:
        super().__init__(x)
        self.min_val = min_val
        self.max_val = max_val

    def backward(self, grad_output: np.ndarray) -> tuple[np.ndarray | None, ...]:
        (x,) = self.inputs
        if not x.requires_grad:
            return (None,)
        mask = np.ones_like(x.data, dtype=bool)
        if self.min_val is not None:
            mask &= x.data >= self.min_val
        if self.max_val is not None:
            mask &= x.data <= self.max_val
        dx = np.where(mask, grad_output, 0.0)
        return (dx,)


class SoftmaxBackward(Function):
    def __init__(self, x: Tensor, out_data: np.ndarray, axis: int) -> None:
        super().__init__(x)
        self.out_data = out_data
        self.axis = axis

    def backward(self, grad_output: np.ndarray) -> tuple[np.ndarray | None, ...]:
        (x,) = self.inputs
        if not x.requires_grad:
            return (None,)
        # dL/dx_i = s_i * (dL/ds_i - sum_k(dL/ds_k * s_k))
        s = self.out_data
        dot = np.sum(grad_output * s, axis=self.axis, keepdims=True)
        gx = s * (grad_output - dot)
        return (gx,)


class LogSoftmaxBackward(Function):
    def __init__(self, x: Tensor, softmax_data: np.ndarray, axis: int) -> None:
        super().__init__(x)
        self.softmax_data = softmax_data
        self.axis = axis

    def backward(self, grad_output: np.ndarray) -> tuple[np.ndarray | None, ...]:
        (x,) = self.inputs
        if not x.requires_grad:
            return (None,)
        # dL/dx_i = dL/dy_i - s_i * sum_k(dL/dy_k)
        sum_grad = np.sum(grad_output, axis=self.axis, keepdims=True)
        gx = grad_output - self.softmax_data * sum_grad
        return (gx,)


class LayerNormBackward(Function):
    def __init__(
        self,
        x: Tensor,
        gamma: Tensor | None,
        beta: Tensor | None,
        x_hat: np.ndarray,
        std_inv: np.ndarray,
        axis: int = -1,
    ) -> None:
        inputs = [x]
        if gamma is not None:
            inputs.append(gamma)
        if beta is not None:
            inputs.append(beta)
        super().__init__(*inputs)
        self.has_gamma = gamma is not None
        self.has_beta = beta is not None
        self.x_hat = x_hat
        self.std_inv = std_inv
        self.axis = axis

    def backward(self, grad_output: np.ndarray) -> tuple[np.ndarray | None, ...]:
        x = self.inputs[0]
        gamma = self.inputs[1] if self.has_gamma else None
        beta = (
            self.inputs[2]
            if (self.has_gamma and self.has_beta)
            else (self.inputs[1] if self.has_beta else None)
        )

        d = x.shape[self.axis]
        gamma_data = gamma.data if gamma is not None else 1.0

        dy_gamma = grad_output * gamma_data
        sum_dy_gamma = np.sum(dy_gamma, axis=self.axis, keepdims=True)
        sum_dy_gamma_xhat = np.sum(dy_gamma * self.x_hat, axis=self.axis, keepdims=True)

        gx = None
        if x.requires_grad:
            gx = (self.std_inv / float(d)) * (
                float(d) * dy_gamma - sum_dy_gamma - self.x_hat * sum_dy_gamma_xhat
            )

        ggamma = None
        if gamma is not None and gamma.requires_grad:
            sum_gamma = np.sum(grad_output * self.x_hat, axis=0, keepdims=True)
            ggamma = unbroadcast(sum_gamma, gamma.shape)

        gbeta = None
        if beta is not None and beta.requires_grad:
            gbeta = unbroadcast(np.sum(grad_output, axis=0, keepdims=True), beta.shape)

        grads: list[np.ndarray | None] = [gx]
        if self.has_gamma:
            grads.append(ggamma)
        if self.has_beta:
            grads.append(gbeta)
        return tuple(grads)


class ConcatBackward(Function):
    def __init__(self, tensors: Sequence[Tensor], axis: int, split_indices: list[int]) -> None:
        super().__init__(*tensors)
        self.axis = axis
        self.split_indices = split_indices

    def backward(self, grad_output: np.ndarray) -> tuple[np.ndarray | None, ...]:
        grads = np.split(grad_output, self.split_indices, axis=self.axis)
        return tuple(
            g if inp.requires_grad else None for inp, g in zip(self.inputs, grads, strict=True)
        )


class SliceBackward(Function):
    def __init__(self, x: Tensor, key: Any) -> None:
        super().__init__(x)
        self.key = key

    def backward(self, grad_output: np.ndarray) -> tuple[np.ndarray | None, ...]:
        (x,) = self.inputs
        if not x.requires_grad:
            return (None,)
        dx = np.zeros_like(x.data)
        dx[self.key] = grad_output
        return (dx,)
