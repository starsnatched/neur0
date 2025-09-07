from __future__ import annotations

from typing import List, Optional, Tuple
import numpy as np

from .module import Module, Parameter
from .layers import Linear


class RMSNormSeq(Module):
    def __init__(self, dim: int, eps: float = 1e-5, affine: bool = True) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = int(dim)
        self.eps = float(eps)
        if affine:
            self.gamma = Parameter(np.ones((dim,), dtype=np.float32))
            self.beta = Parameter(np.zeros((dim,), dtype=np.float32))
        else:
            self.gamma = Parameter(np.ones((dim,), dtype=np.float32))
            self.beta = None
        self._cache = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        if x.ndim not in (2, 3):
            raise ValueError("RMSNormSeq expects (N,D) or (N,T,D)")
        if x.shape[-1] != self.dim:
            raise ValueError("feature size mismatch in RMSNormSeq")
        if x.dtype != np.float32:
            x = x.astype(np.float32)
        ms = (x * x).mean(axis=-1, keepdims=True)
        inv = 1.0 / np.sqrt(ms + self.eps)
        xnh = x * inv
        if self.beta is None:
            y = self.gamma.data * xnh
        else:
            y = self.gamma.data * xnh + self.beta.data
        self._cache = (x, inv, xnh)
        return y

    def backward(self, dy: np.ndarray) -> np.ndarray:
        if self._cache is None:
            raise RuntimeError("RMSNormSeq backward called before forward")
        x, inv, xnh = self._cache
        if dy.shape != x.shape:
            raise ValueError("Upstream grad shape mismatch in RMSNormSeq")
        dxhat = dy * self.gamma.data
        D = float(self.dim)
        t = (dxhat * x).sum(axis=-1, keepdims=True)
        dx = inv * dxhat - (inv ** 3) * x * (t / D)
        dgamma = (dy * xnh).sum(axis=tuple(range(dy.ndim - 1)), keepdims=False)
        self.gamma.grad += dgamma
        if self.beta is not None:
            dbeta = dy.sum(axis=tuple(range(dy.ndim - 1)), keepdims=False)
            self.beta.grad += dbeta
        return dx


def _layer_io_dims(layer) -> Tuple[int, int]:
    if hasattr(layer, "input_size") and hasattr(layer, "hidden_size"):
        return int(layer.input_size), int(layer.hidden_size)
    if hasattr(layer, "input_size") and hasattr(layer, "proj_size"):
        return int(layer.input_size), int(layer.proj_size)
    raise TypeError("Unsupported layer type for LSTMStack")


class LSTMStack(Module):
    def __init__(self, layers: List[Module], pre_norm: bool = True, residual: bool = True, eps: float = 1e-5) -> None:
        super().__init__()
        if not layers:
            raise ValueError("layers must be non-empty")
        self.layers = layers
        self.pre_norm = bool(pre_norm)
        self.residual = bool(residual)
        self.eps = float(eps)
        self.norms: List[Optional[RMSNormSeq]] = []
        self.res_projs: List[Optional[Linear]] = []
        prev_out = None
        for i, layer in enumerate(self.layers):
            lin, lout = _layer_io_dims(layer)
            if prev_out is not None and prev_out != lin:
                raise ValueError("Layer input sizes must match previous layer output sizes")
            prev_out = lout
            if self.pre_norm:
                self.norms.append(RMSNormSeq(lin, eps=self.eps, affine=True))
            else:
                self.norms.append(None)
            if self.residual:
                self.res_projs.append(None if lin == lout else Linear(lin, lout, seed=1234 + i))
            else:
                self.res_projs.append(None)
        self._cache = None
        self._return_sequence = False

    def parameters(self):
        ps: List[Parameter] = []
        for l in self.layers:
            ps.extend(l.parameters())
        for n in self.norms:
            if n is not None:
                ps.extend(n.parameters())
        for rp in self.res_projs:
            if rp is not None:
                ps.extend(rp.parameters())
        return ps

    def zero_grad(self) -> None:
        for l in self.layers:
            l.zero_grad()
        for n in self.norms:
            if n is not None:
                n.zero_grad()
        for rp in self.res_projs:
            if rp is not None:
                rp.zero_grad()

    def _seq_linear_forward(self, lin: Linear, x: np.ndarray) -> np.ndarray:
        N, T, D = x.shape
        y = lin.forward(x.reshape(N * T, D)).reshape(N, T, -1)
        return y

    def _seq_linear_backward(self, lin: Linear, dy: np.ndarray) -> np.ndarray:
        N, T, D = dy.shape
        dx = lin.backward(dy.reshape(N * T, D)).reshape(N, T, -1)
        return dx

    def forward(self, x: np.ndarray, return_sequence: bool = False) -> np.ndarray:
        if x.ndim != 3:
            raise ValueError("LSTMStack expects input of shape (N,T,D)")
        N, T, D0 = x.shape
        caches = []
        h = x
        for idx, layer in enumerate(self.layers):
            norm = self.norms[idx]
            res_proj = self.res_projs[idx]
            last = idx == len(self.layers) - 1
            if norm is not None:
                hn = norm.forward(h)
            else:
                hn = h
            out = layer.forward(hn, return_sequence=(not last) or bool(return_sequence))
            if (not last) or return_sequence:
                if self.residual:
                    if res_proj is not None:
                        r = self._seq_linear_forward(res_proj, h)
                        y = out + r
                    else:
                        y = out + h
                else:
                    y = out
                caches.append((h, hn, out, y, norm, res_proj, True))
                h = y
            else:
                caches.append((h, hn, out, out, norm, res_proj, False))
                h = out
        self._cache = caches
        self._return_sequence = bool(return_sequence)
        return h

    def backward(self, dy: np.ndarray) -> np.ndarray:
        if self._cache is None:
            raise RuntimeError("LSTMStack backward called before forward")
        upstream = dy
        for idx in range(len(self.layers) - 1, -1, -1):
            h, hn, out, y, norm, res_proj, produced_seq = self._cache[idx]
            if produced_seq:
                d_out = upstream
                d_layer_out = d_out
                d_res_in = d_out if self.residual else 0.0
                dx_from_layer = self.layers[idx].backward(d_layer_out)
                if self.residual:
                    if res_proj is not None:
                        d_res = self._seq_linear_backward(res_proj, d_res_in)
                    else:
                        d_res = d_res_in
                else:
                    d_res = 0.0
                if norm is not None:
                    d_h = norm.backward(dx_from_layer) + d_res
                else:
                    d_h = dx_from_layer + d_res
                upstream = d_h
            else:
                dx_from_layer = self.layers[idx].backward(upstream)
                if norm is not None:
                    d_h = norm.backward(dx_from_layer)
                else:
                    d_h = dx_from_layer
                upstream = d_h
        return upstream

