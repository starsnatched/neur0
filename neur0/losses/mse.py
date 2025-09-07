from __future__ import annotations

import numpy as np


class MSELoss:
    def __init__(self) -> None:
        self._cache_y: np.ndarray | None = None
        self._cache_t: np.ndarray | None = None

    def forward(self, y: np.ndarray, t: np.ndarray) -> np.ndarray:
        if y.shape != t.shape:
            raise ValueError("Predictions and targets must have the same shape")
        if y.dtype != np.float32:
            y = y.astype(np.float32)
        if t.dtype != np.float32:
            t = t.astype(np.float32)
        self._cache_y = y
        self._cache_t = t
        diff = y - t
        loss = np.mean(diff * diff)
        return loss

    def backward(self) -> np.ndarray:
        if self._cache_y is None or self._cache_t is None:
            raise RuntimeError("MSE backward called before forward")
        n = np.prod(self._cache_y.shape)
        d = (2.0 / float(n)) * (self._cache_y - self._cache_t)
        return d
