from __future__ import annotations

import numpy as np

from neur0.nn.lstm import LSTM
from neur0.nn.layers import Linear
from neur0.losses.mse import MSELoss


def finite_diff_grad(param: np.ndarray, f, eps: float = 1e-3, samples: int = 10, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    idxs = [tuple(rng.integers(0, s) for s in param.shape) for _ in range(samples)]
    grads = {}
    for idx in idxs:
        orig = param[idx]
        param[idx] = orig + eps
        lp = f()
        param[idx] = orig - eps
        lm = f()
        param[idx] = orig
        grads[idx] = (lp - lm) / (2 * eps)
    return grads


def test_lstm_backward_grads_close():
    rng = np.random.default_rng(0)
    N, T, D, H = 3, 4, 2, 3
    X = rng.standard_normal((N, T, D)).astype(np.float32) * 0.1
    target = np.zeros((N, H), dtype=np.float32)
    lstm = LSTM(D, H, seed=1, norm="none", p_in=0.0, p_hidden=0.0)
    lstm.eval()

    def forward_loss():
        y = lstm(X, return_sequence=False)
        loss = float(np.mean((y - target) ** 2))
        return loss

    y = lstm(X, return_sequence=False)
    loss = np.mean((y - target) ** 2)
    dy = (2.0 / float(N * H)) * (y - target)
    _ = lstm.backward(dy)

    wx_fd = finite_diff_grad(lstm.Wx.data, forward_loss, eps=1e-3, samples=12, rng=rng)
    wh_fd = finite_diff_grad(lstm.Wh.data, forward_loss, eps=1e-3, samples=12, rng=rng)
    b_fd = finite_diff_grad(lstm.b.data, forward_loss, eps=1e-3, samples=12, rng=rng)

    max_abs_diff = 0.0
    for idx, val in wx_fd.items():
        max_abs_diff = max(max_abs_diff, float(abs(val - lstm.Wx.grad[idx])))
    for idx, val in wh_fd.items():
        max_abs_diff = max(max_abs_diff, float(abs(val - lstm.Wh.grad[idx])))
    for idx, val in b_fd.items():
        max_abs_diff = max(max_abs_diff, float(abs(val - lstm.b.grad[idx])))
    assert max_abs_diff < 2e-2


def test_small_lstm_net_shapes():
    from neur0.nn.layers import Linear

    lstm = LSTM(1, 5, seed=0, norm="none", p_in=0.0, p_hidden=0.0)
    head = Linear(5, 1, seed=1)
    X = np.zeros((2, 3, 1), dtype=np.float32)
    y = lstm(X, return_sequence=False)
    out = head(y)
    assert out.shape == (2, 1)
