from __future__ import annotations

import math
from typing import Tuple
import numpy as np

from .module import Module, Parameter


def _kaiming_uniform(fan_in: int, fan_out: int, rng: np.random.Generator) -> np.ndarray:
    bound = math.sqrt(6.0 / fan_in)
    w = rng.uniform(low=-bound, high=bound, size=(fan_in, fan_out)).astype(np.float32)
    return w


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, seed: int | None = None) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features and out_features must be positive")
        rng = np.random.default_rng(seed)
        self.W = Parameter(_kaiming_uniform(in_features, out_features, rng))
        self.b = Parameter(np.zeros((out_features,), dtype=np.float32))
        self._cache_x: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        if x.ndim != 2:
            raise ValueError("Linear expects input of shape (N, D)")
        if x.shape[1] != self.W.data.shape[0]:
            raise ValueError("Input feature size mismatch in Linear")
        if x.dtype != np.float32:
            x = x.astype(np.float32)
        self._cache_x = x
        y = x @ self.W.data + self.b.data
        return y

    def backward(self, dy: np.ndarray) -> np.ndarray:
        if self._cache_x is None:
            raise RuntimeError("Linear backward called before forward")
        x = self._cache_x
        if dy.shape != (x.shape[0], self.W.data.shape[1]):
            raise ValueError("Upstream grad shape mismatch in Linear")
        self.W.grad += x.T @ dy
        self.b.grad += dy.sum(axis=0)
        dx = dy @ self.W.data.T
        return dx
