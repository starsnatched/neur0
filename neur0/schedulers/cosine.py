from __future__ import annotations

import math


class CosineWarmup:
    def __init__(self, base_lr: float, warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.1) -> None:
        if base_lr <= 0:
            raise ValueError("base_lr must be positive")
        if warmup_steps < 0 or total_steps <= 0 or warmup_steps >= total_steps:
            raise ValueError("invalid warmup/total steps")
        if not (0.0 < min_lr_ratio <= 1.0):
            raise ValueError("min_lr_ratio must be in (0,1]")
        self.base_lr = float(base_lr)
        self.warmup = int(warmup_steps)
        self.total = int(total_steps)
        self.min_lr = self.base_lr * float(min_lr_ratio)

    def lr(self, step: int) -> float:
        if step <= 0:
            return self.min_lr
        if step <= self.warmup:
            return self.base_lr * (step / max(1, self.warmup))
        if step >= self.total:
            return self.min_lr
        p = (step - self.warmup) / (self.total - self.warmup)
        return self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1.0 + math.cos(math.pi * p))
