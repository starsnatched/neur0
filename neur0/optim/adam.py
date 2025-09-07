from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np

from ..nn.module import Parameter


class AdamW:
    def __init__(self, params: List[Parameter], lr: float = 1e-3, betas: Tuple[float, float] = (0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0.0) -> None:
        if lr <= 0:
            raise ValueError("lr must be positive")
        if not (0 < betas[0] < 1 and 0 < betas[1] < 1):
            raise ValueError("betas must be in (0,1)")
        if eps <= 0:
            raise ValueError("eps must be positive")
        if weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        self.params = params
        self.lr = float(lr)
        self.b1 = float(betas[0])
        self.b2 = float(betas[1])
        self.eps = float(eps)
        self.wd = float(weight_decay)
        self.t = 0
        self.m: Dict[int, np.ndarray] = {}
        self.v: Dict[int, np.ndarray] = {}

    def step(self) -> None:
        self.t += 1
        b1t = 1.0 - self.b1 ** self.t
        b2t = 1.0 - self.b2 ** self.t
        for p in self.params:
            pid = id(p)
            if pid not in self.m:
                self.m[pid] = np.zeros_like(p.data)
                self.v[pid] = np.zeros_like(p.data)
            g = p.grad
            if self.wd != 0.0:
                g = g + self.wd * p.data
            m = self.m[pid] = self.b1 * self.m[pid] + (1.0 - self.b1) * g
            v = self.v[pid] = self.b2 * self.v[pid] + (1.0 - self.b2) * (g * g)
            mhat = m / b1t
            vhat = v / b2t
            p.data -= self.lr * mhat / (np.sqrt(vhat) + self.eps)

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad[...] = 0

    def state_dict(self):
        m_list = []
        v_list = []
        for p in self.params:
            pid = id(p)
            if pid not in self.m:
                self.m[pid] = np.zeros_like(p.data)
                self.v[pid] = np.zeros_like(p.data)
            m_list.append(self.m[pid].copy())
            v_list.append(self.v[pid].copy())
        return {
            "t": self.t,
            "lr": self.lr,
            "b1": self.b1,
            "b2": self.b2,
            "eps": self.eps,
            "wd": self.wd,
            "m_list": m_list,
            "v_list": v_list,
        }

    def load_state_dict(self, state) -> None:
        self.t = int(state.get("t", 0))
        self.lr = float(state.get("lr", self.lr))
        self.b1 = float(state.get("b1", self.b1))
        self.b2 = float(state.get("b2", self.b2))
        self.eps = float(state.get("eps", self.eps))
        self.wd = float(state.get("wd", self.wd))
        m_list = state.get("m_list", [])
        v_list = state.get("v_list", [])
        for i, p in enumerate(self.params):
            pid = id(p)
            if i < len(m_list):
                self.m[pid] = m_list[i].astype(p.data.dtype, copy=True)
            else:
                self.m[pid] = np.zeros_like(p.data)
            if i < len(v_list):
                self.v[pid] = v_list[i].astype(p.data.dtype, copy=True)
            else:
                self.v[pid] = np.zeros_like(p.data)
