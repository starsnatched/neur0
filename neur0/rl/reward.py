from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class RewardShaperConfig:
    base_scale: float = 1.0
    w_x: float = 0.5
    w_y: float = 0.0
    w_vx: float = 0.5
    w_vy: float = 0.2
    w_theta: float = 0.4
    w_vtheta: float = 0.1
    w_action: float = 0.02
    w_leg: float = 0.5
    clip: float = 10.0


class LunarRewardShaper:
    def __init__(self, cfg: RewardShaperConfig | None = None) -> None:
        self.cfg = cfg or RewardShaperConfig()

    def shape(self, obs: np.ndarray, act: np.ndarray, raw_r: float) -> float:
        if obs.ndim != 1 or obs.shape[0] < 8:
            raise ValueError("unexpected observation shape for LunarLanderContinuous")
        if act.ndim != 1 or act.shape[0] < 2:
            raise ValueError("unexpected action shape for LunarLanderContinuous")
        x, y, vx, vy, theta, vtheta, c1, c2 = [float(v) for v in obs[:8]]
        a_main = float(act[0])
        a_side = float(act[1])
        cost = 0.0
        cost += self.cfg.w_x * abs(x)
        cost += self.cfg.w_y * max(0.0, 0.0 - y)
        cost += self.cfg.w_vx * abs(vx)
        cost += self.cfg.w_vy * abs(vy)
        cost += self.cfg.w_theta * abs(theta)
        cost += self.cfg.w_vtheta * abs(vtheta)
        cost += self.cfg.w_action * (a_main * a_main + a_side * a_side)
        bonus = self.cfg.w_leg * (float(c1) + float(c2))
        shaped = self.cfg.base_scale * float(raw_r) - cost + bonus
        if self.cfg.clip > 0.0:
            lo = -abs(self.cfg.clip)
            hi = abs(self.cfg.clip)
            shaped = float(np.clip(shaped, lo, hi))
        return shaped


def discounted_returns(rewards: np.ndarray, gamma: float = 0.99) -> np.ndarray:
    if rewards.ndim != 1:
        raise ValueError("rewards must be (T)")
    if not (0.0 <= gamma <= 1.0):
        raise ValueError("gamma must be in [0,1]")
    T = rewards.shape[0]
    out = np.zeros_like(rewards, dtype=np.float32)
    acc = 0.0
    for t in range(T - 1, -1, -1):
        acc = float(rewards[t]) + gamma * acc
        out[t] = acc
    return out


def rwr_weights(returns: np.ndarray, beta: float = 1.0, eps: float = 1e-6) -> np.ndarray:
    if returns.ndim != 1:
        raise ValueError("returns must be (N)")
    if beta <= 0.0:
        return np.ones_like(returns, dtype=np.float32)
    r = returns.astype(np.float32)
    r = (r - float(np.mean(r))) / float(np.std(r) + eps)
    z = beta * r
    z = z - float(np.max(z))
    w = np.exp(z).astype(np.float32)
    s = float(np.sum(w))
    if s <= eps:
        return np.ones_like(returns, dtype=np.float32)
    w /= s
    scale = float(w.shape[0])
    return w * scale

