from __future__ import annotations

from typing import List
import numpy as np


class TemporalContext:
    def __init__(self, dim: int, alphas: List[float] | None = None, window: int = 8) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = int(dim)
        self.alphas = alphas or [0.9, 0.99, 0.999]
        for a in self.alphas:
            if not (0.0 < a < 1.0):
                raise ValueError("alpha must be in (0,1)")
        self.window = int(window)
        if self.window <= 0:
            raise ValueError("window must be positive")
        self.reset()

    def reset(self) -> None:
        self.ema = [np.zeros((self.dim,), dtype=np.float32) for _ in self.alphas]
        self.buf = np.zeros((self.window, self.dim), dtype=np.float32)
        self.n = 0
        self.prev = np.zeros((self.dim,), dtype=np.float32)

    def augment(self, x: np.ndarray) -> np.ndarray:
        if x.ndim != 1 or x.shape[0] != self.dim:
            raise ValueError("x must be (D)")
        x = x.astype(np.float32)
        for i, a in enumerate(self.alphas):
            self.ema[i] = a * self.ema[i] + (1.0 - a) * x
        idx = self.n % self.window
        self.buf[idx] = x
        self.n += 1
        k = min(self.n, self.window)
        avg = self.buf[:k].mean(axis=0)
        std = self.buf[:k].std(axis=0)
        delta = x - self.prev
        self.prev = x
        parts = [x] + self.ema + [avg, std, delta]
        z = np.concatenate(parts, axis=0)
        z = np.clip(z, -10.0, 10.0)
        return z

    def augmented_dim(self) -> int:
        return self.dim * (len(self.alphas) + 4)


class ContextProjector:
    def __init__(self, in_dim: int, out_dim: int, seed: int | None = None) -> None:
        if in_dim <= 0 or out_dim <= 0:
            raise ValueError("dims must be positive")
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        rng = np.random.default_rng(seed)
        s = np.sqrt(1.0 / float(in_dim))
        self.W = rng.uniform(-s, s, size=(in_dim, out_dim)).astype(np.float32)

    def forward(self, z: np.ndarray) -> np.ndarray:
        if z.ndim == 1:
            if z.shape[0] != self.in_dim:
                raise ValueError("z dim mismatch")
            return z @ self.W
        if z.ndim == 2:
            if z.shape[1] != self.in_dim:
                raise ValueError("z dim mismatch")
            return z @ self.W
        raise ValueError("z must be (D) or (N,D)")
