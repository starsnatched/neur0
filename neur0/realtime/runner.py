from __future__ import annotations

import os
from typing import Optional, Tuple
import numpy as np

from ..nn.lstm import LSTM, LSTMP
from ..nn.layers import Linear
from .context import TemporalContext, ContextProjector


class LanderRealtime:
    def __init__(self, core, head: Optional[Linear], context: TemporalContext, projector: Optional[ContextProjector] = None) -> None:
        self.core = core
        self.head = head
        self.context = context
        self.projector = projector
        self.h = None
        self.c = None

    def reset(self, batch_size: int = 1) -> None:
        self.h, self.c = self.core.init_state(batch_size)
        self.context.reset()

    def step(self, feat: np.ndarray) -> np.ndarray:
        if self.h is None or self.c is None:
            self.reset(batch_size=feat.shape[0] if feat.ndim == 2 else 1)
        if feat.ndim == 1:
            z = self.context.augment(feat)
            z = z[None, :]
        else:
            out = []
            for i in range(feat.shape[0]):
                out.append(self.context.augment(feat[i]))
            z = np.stack(out, axis=0)
        if self.projector is not None:
            z = self.projector.forward(z)
        h, c = self.core.step(z, self.h, self.c)
        self.h, self.c = h, c
        if self.head is not None:
            y = self.head.forward(h)
        else:
            y = h
        return y

    @staticmethod
    def load_from_checkpoint(core, head: Optional[Linear], ckpt_path: str) -> None:
        pass
