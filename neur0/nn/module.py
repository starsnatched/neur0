from __future__ import annotations

from typing import Iterable, List, Tuple
import numpy as np


class Parameter:
    def __init__(self, data: np.ndarray):
        if not isinstance(data, np.ndarray):
            raise TypeError("Parameter data must be a numpy.ndarray")
        if data.dtype not in (np.float32, np.float64):
            raise TypeError("Parameter dtype must be float32 or float64")
        self.data = data
        self.grad = np.zeros_like(data)

    def zero_grad(self) -> None:
        self.grad[...] = 0


class Module:
    def __init__(self) -> None:
        self._training = True

    def parameters(self) -> List[Parameter]:
        ps: List[Parameter] = []
        for v in self.__dict__.values():
            if isinstance(v, Parameter):
                ps.append(v)
            elif isinstance(v, (list, tuple)):
                for x in v:
                    if isinstance(x, Parameter):
                        ps.append(x)
        return ps

    def train(self) -> None:
        self._training = True

    def eval(self) -> None:
        self._training = False

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.zero_grad()

    def forward(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        raise NotImplementedError

    def backward(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        raise NotImplementedError

    def __call__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        return self.forward(*args, **kwargs)
