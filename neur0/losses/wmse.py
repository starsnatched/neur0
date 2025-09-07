from __future__ import annotations

import numpy as np


class WeightedMSELoss:
    def __init__(self) -> None:
        self._y: np.ndarray | None = None
        self._t: np.ndarray | None = None
        self._w: np.ndarray | None = None

    def forward(self, y: np.ndarray, t: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
        if y.shape != t.shape:
            raise ValueError("predictions and targets must have the same shape")
        if y.dtype != np.float32:
            y = y.astype(np.float32)
        if t.dtype != np.float32:
            t = t.astype(np.float32)
        if w is None:
            self._y = y
            self._t = t
            self._w = None
            d = y - t
            return np.mean(d * d)
        if w.ndim == 1:
            if w.shape[0] != y.shape[0]:
                raise ValueError("weight length must equal batch size")
            ww = w.astype(np.float32)[:, None]
            ww = np.broadcast_to(ww, y.shape)
        else:
            if w.shape != y.shape:
                raise ValueError("weight shape must be (N,) or match predictions")
            ww = w.astype(np.float32)
        self._y = y
        self._t = t
        self._w = ww
        d = y - t
        num = np.sum(ww * d * d)
        den = float(np.sum(ww))
        if den <= 0.0:
            return np.mean(d * d)
        return num / den

    def backward(self) -> np.ndarray:
        if self._y is None or self._t is None:
            raise RuntimeError("backward called before forward")
        d = self._y - self._t
        if self._w is None:
            n = float(np.prod(self._y.shape))
            return (2.0 / n) * d
        den = float(np.sum(self._w))
        if den <= 0.0:
            n = float(np.prod(self._y.shape))
            return (2.0 / n) * d
        return (2.0 / den) * (self._w * d)

