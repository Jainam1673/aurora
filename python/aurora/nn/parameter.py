"""Trainable model parameter abstraction for AURORA."""

from __future__ import annotations

from typing import Any

import numpy as np

from aurora.tensor import Tensor


class Parameter(Tensor):
    """A Tensor intended to be considered a trainable module parameter."""

    def __init__(
        self,
        data: Any,
        requires_grad: bool = True,
        dtype: np.dtype | type = np.float64,
    ) -> None:
        super().__init__(data, requires_grad=requires_grad, dtype=dtype)
