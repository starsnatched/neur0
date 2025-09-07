from __future__ import annotations

import numpy as np

from .module import Module, Parameter


def _sigmoid(x: np.ndarray) -> np.ndarray:
    y = np.empty_like(x)
    m = x >= 0
    y[m] = 1.0 / (1.0 + np.exp(-x[m]))
    e = np.exp(x[~m])
    y[~m] = e / (1.0 + e)
    return y


def _orthogonal(h: int, rng: np.random.Generator) -> np.ndarray:
    a = rng.standard_normal((h, h)).astype(np.float32)
    q, r = np.linalg.qr(a)
    d = np.sign(np.diag(r))
    q = q * d
    return q.astype(np.float32)


def _glorot_uniform(fan_in: int, fan_out: int, rng: np.random.Generator) -> np.ndarray:
    s = np.sqrt(6.0 / float(fan_in + fan_out))
    return rng.uniform(-s, s, size=(fan_in, fan_out)).astype(np.float32)


def _layer_norm_forward(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float):
    mu = x.mean(axis=1, keepdims=True)
    xc = x - mu
    var = (xc * xc).mean(axis=1, keepdims=True)
    inv = 1.0 / np.sqrt(var + eps)
    xnh = xc * inv
    y = gamma * xnh + beta
    cache = (x, xc, inv, xnh, gamma)
    return y, cache


def _layer_norm_backward(dy: np.ndarray, cache):
    x, xc, inv, xnh, gamma = cache
    dgamma = (dy * xnh).sum(axis=0)
    dbeta = dy.sum(axis=0)
    dxhat = dy * gamma
    m1 = dxhat.mean(axis=1, keepdims=True)
    m2 = (dxhat * xnh).mean(axis=1, keepdims=True)
    dx = (dxhat - xnh * m2 - m1) * inv
    return dx, dgamma, dbeta


def _rms_norm_forward(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray | None, eps: float):
    ms = (x * x).mean(axis=1, keepdims=True)
    inv = 1.0 / np.sqrt(ms + eps)
    xnh = x * inv
    if beta is None:
        y = gamma * xnh
    else:
        y = gamma * xnh + beta
    cache = (x, inv, gamma, beta)
    return y, cache


def _rms_norm_backward(dy: np.ndarray, cache):
    x, inv, gamma, beta = cache
    H = x.shape[1]
    dxhat = dy * gamma
    t = (dxhat * x).sum(axis=1, keepdims=True)
    dx = inv * dxhat - (inv ** 3) * x * (t / H)
    dgamma = (dy * (x * inv)).sum(axis=0)
    dbeta = dy.sum(axis=0) if beta is not None else None
    return dx, dgamma, dbeta


class LSTM(Module):
    def __init__(self, input_size: int, hidden_size: int, seed: int | None = None, norm: str = "none", eps: float = 1e-5, p_in: float = 0.0, p_hidden: float = 0.0, cifg: bool = False, peepholes: bool = False, p_weightdrop: float = 0.0, p_zoneout: float = 0.0) -> None:
        super().__init__()
        if input_size <= 0 or hidden_size <= 0:
            raise ValueError("input_size and hidden_size must be positive")
        if norm not in ("none", "layernorm", "rmsnorm"):
            raise ValueError("invalid norm")
        if not (0.0 <= p_in < 1.0 and 0.0 <= p_hidden < 1.0):
            raise ValueError("dropout probabilities must be in [0,1)")
        self.rng = np.random.default_rng(seed)
        H = hidden_size
        D = input_size
        wx = _glorot_uniform(D, 4 * H, self.rng)
        wh = np.concatenate([_orthogonal(H, self.rng) for _ in range(4)], axis=1)
        b = np.zeros((4 * H,), dtype=np.float32)
        b[H:2 * H] = 1.0
        self.Wx = Parameter(wx)
        self.Wh = Parameter(wh.astype(np.float32))
        self.b = Parameter(b)
        self.hidden_size = H
        self.input_size = D
        self.norm = norm
        self.eps = float(eps)
        self.p_in = float(p_in)
        self.p_hidden = float(p_hidden)
        self.cifg = bool(cifg)
        self.peepholes = bool(peepholes)
        self.p_weightdrop = float(p_weightdrop)
        self.p_zoneout = float(p_zoneout)
        if norm != "none":
            self.gamma_g = Parameter(np.ones((4, H), dtype=np.float32))
            self.beta_g = Parameter(np.zeros((4, H), dtype=np.float32))
            self.gamma_c = Parameter(np.ones((H,), dtype=np.float32))
            self.beta_c = Parameter(np.zeros((H,), dtype=np.float32))
        else:
            self.gamma_g = None
            self.beta_g = None
            self.gamma_c = None
            self.beta_c = None
        if self.peepholes:
            self.pi = Parameter(np.zeros((H,), dtype=np.float32))
            self.pf = Parameter(np.zeros((H,), dtype=np.float32))
            self.po = Parameter(np.zeros((H,), dtype=np.float32))
        else:
            self.pi = None
            self.pf = None
            self.po = None
        self._cache = None
        self._return_sequence = False

    def forward(self, x: np.ndarray, h0: np.ndarray | None = None, c0: np.ndarray | None = None, return_sequence: bool = False) -> np.ndarray:
        if x.ndim != 3:
            raise ValueError("LSTM expects input of shape (N, T, D)")
        N, T, D = x.shape
        if D != self.input_size:
            raise ValueError("Input feature size mismatch in LSTM")
        H = self.hidden_size
        if x.dtype != np.float32:
            x = x.astype(np.float32)
        if h0 is None:
            h_prev = np.zeros((N, H), dtype=np.float32)
        else:
            if h0.shape != (N, H):
                raise ValueError("h0 shape mismatch")
            h_prev = h0.astype(np.float32)
        if c0 is None:
            c_prev = np.zeros((N, H), dtype=np.float32)
        else:
            if c0.shape != (N, H):
                raise ValueError("c0 shape mismatch")
            c_prev = c0.astype(np.float32)
        if self._training and self.p_in > 0.0:
            mx = (self.rng.random((N, D), dtype=np.float32) >= self.p_in).astype(np.float32) / (1.0 - self.p_in)
        else:
            mx = np.ones((N, D), dtype=np.float32)
        if self._training and self.p_hidden > 0.0:
            mh = (self.rng.random((N, H), dtype=np.float32) >= self.p_hidden).astype(np.float32) / (1.0 - self.p_hidden)
        else:
            mh = np.ones((N, H), dtype=np.float32)
        if self._training and self.p_weightdrop > 0.0:
            Mw = (self.rng.random(self.Wh.data.shape, dtype=np.float32) >= self.p_weightdrop).astype(np.float32)
        else:
            Mw = np.ones_like(self.Wh.data)
        if self._training and self.p_zoneout > 0.0:
            mz = (self.rng.random((N, H), dtype=np.float32) >= self.p_zoneout).astype(np.float32)
        else:
            mz = np.ones((N, H), dtype=np.float32)
        hs = np.empty((N, T, H), dtype=np.float32)
        xs_m = np.empty((N, T, D), dtype=np.float32)
        cs = np.empty((N, T, H), dtype=np.float32)
        cs_hat = np.empty((N, T, H), dtype=np.float32) if self.norm != "none" else None
        i_s = np.empty((N, T, H), dtype=np.float32)
        f_s = np.empty((N, T, H), dtype=np.float32)
        g_s = np.empty((N, T, H), dtype=np.float32)
        o_s = np.empty((N, T, H), dtype=np.float32)
        caches_gates = []
        caches_cell = []
        for t in range(T):
            xt = x[:, t, :] * mx
            xs_m[:, t, :] = xt
            hp = h_prev * mh
            z = xt @ self.Wx.data + hp @ (self.Wh.data * Mw) + self.b.data
            zi = z[:, :H]
            zf = z[:, H:2 * H]
            zg = z[:, 2 * H:3 * H]
            zo = z[:, 3 * H:]
            if self.peepholes:
                zi = zi + c_prev * self.pi.data
                zf = zf + c_prev * self.pf.data
            if self.norm == "layernorm":
                yi, ci = _layer_norm_forward(zi, self.gamma_g.data[0], self.beta_g.data[0], self.eps)
                yf, cf = _layer_norm_forward(zf, self.gamma_g.data[1], self.beta_g.data[1], self.eps)
                yg, cg = _layer_norm_forward(zg, self.gamma_g.data[2], self.beta_g.data[2], self.eps)
                yo, co = _layer_norm_forward(zo, self.gamma_g.data[3], self.beta_g.data[3], self.eps)
                caches_gates.append((ci, cf, cg, co))
            elif self.norm == "rmsnorm":
                yi, ci = _rms_norm_forward(zi, self.gamma_g.data[0], self.beta_g.data[0], self.eps)
                yf, cf = _rms_norm_forward(zf, self.gamma_g.data[1], self.beta_g.data[1], self.eps)
                yg, cg = _rms_norm_forward(zg, self.gamma_g.data[2], self.beta_g.data[2], self.eps)
                yo, co = _rms_norm_forward(zo, self.gamma_g.data[3], self.beta_g.data[3], self.eps)
                caches_gates.append((ci, cf, cg, co))
            else:
                yi, yf, yg, yo = zi, zf, zg, zo
                caches_gates.append(None)
            f = _sigmoid(yf)
            if self.cifg:
                i = 1.0 - f
            else:
                i = _sigmoid(yi)
            g = np.tanh(yg)
            o = _sigmoid(yo)
            c = f * c_prev + i * g
            if self.norm == "layernorm":
                ch, cc = _layer_norm_forward(c, self.gamma_c.data, self.beta_c.data, self.eps)
                caches_cell.append(cc)
                ct = ch
                cs_hat[:, t, :] = ch
            elif self.norm == "rmsnorm":
                ch, cc = _rms_norm_forward(c, self.gamma_c.data, self.beta_c.data, self.eps)
                caches_cell.append(cc)
                ct = ch
                cs_hat[:, t, :] = ch
            else:
                ct = c
                caches_cell.append(None)
            if self.peepholes:
                yo_peep = yo + c * self.po.data
                if self.norm == "layernorm":
                    yo, co = _layer_norm_forward(yo_peep, self.gamma_g.data[3], self.beta_g.data[3], self.eps)
                    caches_gates[-1] = (caches_gates[-1][0], caches_gates[-1][1], caches_gates[-1][2], co)
                elif self.norm == "rmsnorm":
                    yo, co = _rms_norm_forward(yo_peep, self.gamma_g.data[3], self.beta_g.data[3], self.eps)
                    caches_gates[-1] = (caches_gates[-1][0], caches_gates[-1][1], caches_gates[-1][2], co)
                else:
                    yo = yo_peep
                o = _sigmoid(yo)
            h_new = o * np.tanh(ct)
            if self._training and self.p_zoneout > 0.0:
                h = mz * h_new + (1.0 - mz) * h_prev
            else:
                h = h_new
            i_s[:, t, :] = i
            f_s[:, t, :] = f
            g_s[:, t, :] = g
            o_s[:, t, :] = o
            cs[:, t, :] = c
            hs[:, t, :] = h
            h_prev = h
            c_prev = c
        self._cache = {
            "x_m": xs_m,
            "mx": mx,
            "mh": mh,
            "Mw": Mw,
            "mz": mz,
            "hs": hs,
            "cs": cs,
            "cs_hat": cs_hat,
            "i": i_s,
            "f": f_s,
            "g": g_s,
            "o": o_s,
            "gates_caches": caches_gates,
            "cell_caches": caches_cell,
        }
        self._return_sequence = return_sequence
        if return_sequence:
            return hs
        return hs[:, -1, :]

    def backward(self, dh: np.ndarray) -> np.ndarray:
        if self._cache is None:
            raise RuntimeError("LSTM backward called before forward")
        x_m = self._cache["x_m"]
        mx = self._cache["mx"]
        mh = self._cache["mh"]
        Mw = self._cache["Mw"]
        mz = self._cache["mz"]
        hs = self._cache["hs"]
        cs = self._cache["cs"]
        cs_hat = self._cache["cs_hat"]
        i_s = self._cache["i"]
        f_s = self._cache["f"]
        g_s = self._cache["g"]
        o_s = self._cache["o"]
        gates_caches = self._cache["gates_caches"]
        cell_caches = self._cache["cell_caches"]
        N, T, D = x_m.shape
        H = self.hidden_size
        if self._return_sequence:
            if dh.shape != (N, T, H):
                raise ValueError("Upstream grad shape mismatch for sequence output")
            dhs = dh.copy()
        else:
            if dh.shape != (N, H):
                raise ValueError("Upstream grad shape mismatch for last output")
            dhs = np.zeros((N, T, H), dtype=np.float32)
            dhs[:, -1, :] = dh
        dWx = np.zeros_like(self.Wx.data)
        dWh = np.zeros_like(self.Wh.data)
        db = np.zeros_like(self.b.data)
        if self.norm != "none":
            dgamma_g = np.zeros_like(self.gamma_g.data)
            dbeta_g = np.zeros_like(self.beta_g.data)
            dgamma_c = np.zeros_like(self.gamma_c.data)
            dbeta_c = np.zeros_like(self.beta_c.data)
        dx = np.zeros((N, T, D), dtype=np.float32)
        dh_next = np.zeros((N, H), dtype=np.float32)
        dc_next = np.zeros((N, H), dtype=np.float32)
        for t in range(T - 1, -1, -1):
            h_prev = hs[:, t - 1, :] if t > 0 else np.zeros((N, H), dtype=np.float32)
            c_prev = cs[:, t - 1, :] if t > 0 else np.zeros((N, H), dtype=np.float32)
            ct = cs_hat[:, t, :] if self.norm != "none" else cs[:, t, :]
            tanh_ct = np.tanh(ct)
            dout = dhs[:, t, :] + dh_next
            if self._training and self.p_zoneout > 0.0:
                dht = dout * mz
                dh_to_prev = dout * (1.0 - mz)
            else:
                dht = dout
                dh_to_prev = 0.0
            do_pre = dht * tanh_ct * o_s[:, t, :] * (1.0 - o_s[:, t, :])
            dct = dht * o_s[:, t, :] * (1.0 - tanh_ct * tanh_ct) + dc_next
            if self.norm == "layernorm":
                dc, dgc, dbc = _layer_norm_backward(dct, cell_caches[t])
                dgamma_c += dgc
                dbeta_c += dbc
            elif self.norm == "rmsnorm":
                dc, dgc, dbc = _rms_norm_backward(dct, cell_caches[t])
                dgamma_c += dgc
                if dbc is not None:
                    dbeta_c += dbc
            else:
                dc = dct
            if self.peepholes:
                do_pre += dc * self.po.data * o_s[:, t, :] * (1.0 - o_s[:, t, :])
            if self.cifg:
                di_pre = None
            else:
                di_pre = dc * g_s[:, t, :] * i_s[:, t, :] * (1.0 - i_s[:, t, :])
            df_pre = dc * c_prev * f_s[:, t, :] * (1.0 - f_s[:, t, :])
            dg_pre = dc * i_s[:, t, :] * (1.0 - g_s[:, t, :] * g_s[:, t, :])
            if self.peepholes:
                df_pre += dc * self.pf.data * f_s[:, t, :] * (1.0 - f_s[:, t, :])
            dc_next = dc * f_s[:, t, :]
            ci, cf, cg, co = gates_caches[t] if gates_caches[t] is not None else (None, None, None, None)
            if self.norm == "layernorm":
                if di_pre is not None:
                    di, dgi, dbi = _layer_norm_backward(di_pre, ci)
                else:
                    di, dgi, dbi = None, 0.0, 0.0
                df, dgf, dbf = _layer_norm_backward(df_pre, cf)
                dg, dgg, dbg = _layer_norm_backward(dg_pre, cg)
                do, dgo, dbo = _layer_norm_backward(do_pre, co)
                dgamma_g[0] += dgi
                dgamma_g[1] += dgf
                dgamma_g[2] += dgg
                dgamma_g[3] += dgo
                dbeta_g[0] += dbi
                dbeta_g[1] += dbf
                dbeta_g[2] += dbg
                dbeta_g[3] += dbo
            elif self.norm == "rmsnorm":
                if di_pre is not None:
                    di, dgi, dbi = _rms_norm_backward(di_pre, ci)
                else:
                    di, dgi, dbi = None, 0.0, None
                df, dgf, dbf = _rms_norm_backward(df_pre, cf)
                dg, dgg, dbg = _rms_norm_backward(dg_pre, cg)
                do, dgo, dbo = _rms_norm_backward(do_pre, co)
                dgamma_g[0] += dgi
                dgamma_g[1] += dgf
                dgamma_g[2] += dgg
                dgamma_g[3] += dgo
                if dbi is not None:
                    dbeta_g[0] += dbi
                if dbf is not None:
                    dbeta_g[1] += dbf
                if dbg is not None:
                    dbeta_g[2] += dbg
                if dbo is not None:
                    dbeta_g[3] += dbo
            else:
                di, df, dg, do = di_pre, df_pre, dg_pre, do_pre
            if di is None:
                da = np.concatenate([df, dg, do], axis=1)
                pad_left = np.zeros((N, H), dtype=np.float32)
                da_full = np.concatenate([pad_left, da], axis=1)
            else:
                da_full = np.concatenate([di, df, dg, do], axis=1)
            if self.peepholes:
                if di is not None:
                    self.pi.grad += (di * c_prev).sum(axis=0)
                self.pf.grad += (df * c_prev).sum(axis=0)
                self.po.grad += (do * cs[:, t, :]).sum(axis=0)
                dc_next += (di * self.pi.data).astype(np.float32) if di is not None else 0.0
                dc_next += (df * self.pf.data).astype(np.float32)
                dc_next += (do * self.po.data).astype(np.float32)
            dWx += x_m[:, t, :].T @ da_full
            dWh += (h_prev * mh).T @ da_full
            db += da_full.sum(axis=0)
            dx_t = da_full @ self.Wx.data.T
            dh_prev_m = da_full @ (self.Wh.data * Mw).T
            dx[:, t, :] = dx_t * mx
            dh_next = dh_prev_m * mh + dh_to_prev
        self.Wx.grad += dWx
        self.Wh.grad += dWh
        self.b.grad += db
        if self.norm != "none":
            self.gamma_g.grad += dgamma_g
            self.beta_g.grad += dbeta_g
            self.gamma_c.grad += dgamma_c
            self.beta_c.grad += dbeta_c
        return dx

    def init_state(self, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        h = np.zeros((batch_size, self.hidden_size), dtype=np.float32)
        c = np.zeros((batch_size, self.hidden_size), dtype=np.float32)
        return h, c

    def step(self, x_t: np.ndarray, h_prev: np.ndarray, c_prev: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if x_t.ndim != 2:
            raise ValueError("x_t must be (N,D)")
        N, D = x_t.shape
        if D != self.input_size:
            raise ValueError("feature size mismatch in LSTM.step")
        if h_prev.shape != (N, self.hidden_size) or c_prev.shape != (N, self.hidden_size):
            raise ValueError("state shape mismatch in LSTM.step")
        H = self.hidden_size
        xt = x_t.astype(np.float32)
        z = xt @ self.Wx.data + h_prev @ self.Wh.data + self.b.data
        zi = z[:, :H]
        zf = z[:, H:2 * H]
        zg = z[:, 2 * H:3 * H]
        zo = z[:, 3 * H:]
        if self.peepholes:
            zi = zi + c_prev * self.pi.data
            zf = zf + c_prev * self.pf.data
        if self.norm == "layernorm":
            yi, _ = _layer_norm_forward(zi, self.gamma_g.data[0], self.beta_g.data[0], self.eps)
            yf, _ = _layer_norm_forward(zf, self.gamma_g.data[1], self.beta_g.data[1], self.eps)
            yg, _ = _layer_norm_forward(zg, self.gamma_g.data[2], self.beta_g.data[2], self.eps)
            yo, _ = _layer_norm_forward(zo, self.gamma_g.data[3], self.beta_g.data[3], self.eps)
        elif self.norm == "rmsnorm":
            yi, _ = _rms_norm_forward(zi, self.gamma_g.data[0], self.beta_g.data[0], self.eps)
            yf, _ = _rms_norm_forward(zf, self.gamma_g.data[1], self.beta_g.data[1], self.eps)
            yg, _ = _rms_norm_forward(zg, self.gamma_g.data[2], self.beta_g.data[2], self.eps)
            yo, _ = _rms_norm_forward(zo, self.gamma_g.data[3], self.beta_g.data[3], self.eps)
        else:
            yi, yf, yg, yo = zi, zf, zg, zo
        f = _sigmoid(yf)
        i = (1.0 - f) if self.cifg else _sigmoid(yi)
        g = np.tanh(yg)
        if self.peepholes:
            yo = yo + (f * c_prev + i * g) * self.po.data
        o = _sigmoid(yo)
        c = f * c_prev + i * g
        if self.norm == "layernorm":
            ct, _ = _layer_norm_forward(c, self.gamma_c.data, self.beta_c.data, self.eps)
        elif self.norm == "rmsnorm":
            ct, _ = _rms_norm_forward(c, self.gamma_c.data, self.beta_c.data, self.eps)
        else:
            ct = c
        h = o * np.tanh(ct)
        return h, c


class LSTMP(Module):
    def __init__(self, input_size: int, cell_size: int, proj_size: int, seed: int | None = None, norm: str = "none", eps: float = 1e-5, p_in: float = 0.0, p_hidden: float = 0.0, cifg: bool = False, peepholes: bool = False, p_weightdrop: float = 0.0, p_zoneout: float = 0.0) -> None:
        super().__init__()
        if input_size <= 0 or cell_size <= 0 or proj_size <= 0:
            raise ValueError("input_size, cell_size, proj_size must be positive")
        if proj_size > 4_000_000 or cell_size > 4_000_000:
            raise ValueError("unreasonable sizes")
        if norm not in ("none", "layernorm", "rmsnorm"):
            raise ValueError("invalid norm")
        if not (0.0 <= p_in < 1.0 and 0.0 <= p_hidden < 1.0):
            raise ValueError("dropout probabilities must be in [0,1)")
        self.rng = np.random.default_rng(seed)
        Hc = int(cell_size)
        Hp = int(proj_size)
        D = int(input_size)
        self.Wx = Parameter(_glorot_uniform(D, 4 * Hc, self.rng))
        self.Wr = Parameter(_glorot_uniform(Hp, 4 * Hc, self.rng))
        b = np.zeros((4 * Hc,), dtype=np.float32)
        b[Hc:2 * Hc] = 1.0
        self.b = Parameter(b)
        self.Wp = Parameter(_glorot_uniform(Hc, Hp, self.rng))
        self.input_size = D
        self.cell_size = Hc
        self.proj_size = Hp
        self.norm = norm
        self.eps = float(eps)
        self.p_in = float(p_in)
        self.p_hidden = float(p_hidden)
        self.cifg = bool(cifg)
        self.peepholes = bool(peepholes)
        self.p_weightdrop = float(p_weightdrop)
        self.p_zoneout = float(p_zoneout)
        if norm != "none":
            self.gamma_g = Parameter(np.ones((4, Hc), dtype=np.float32))
            self.beta_g = Parameter(np.zeros((4, Hc), dtype=np.float32))
            self.gamma_c = Parameter(np.ones((Hc,), dtype=np.float32))
            self.beta_c = Parameter(np.zeros((Hc,), dtype=np.float32))
        else:
            self.gamma_g = None
            self.beta_g = None
            self.gamma_c = None
            self.beta_c = None
        if self.peepholes:
            self.pi = Parameter(np.zeros((Hc,), dtype=np.float32))
            self.pf = Parameter(np.zeros((Hc,), dtype=np.float32))
            self.po = Parameter(np.zeros((Hc,), dtype=np.float32))
        else:
            self.pi = None
            self.pf = None
            self.po = None
        self._cache = None
        self._return_sequence = False

    def forward(self, x: np.ndarray, h0: np.ndarray | None = None, c0: np.ndarray | None = None, return_sequence: bool = False) -> np.ndarray:
        if x.ndim != 3:
            raise ValueError("LSTMP expects input of shape (N, T, D)")
        N, T, D = x.shape
        if D != self.input_size:
            raise ValueError("Input feature size mismatch in LSTMP")
        Hc = self.cell_size
        Hp = self.proj_size
        if x.dtype != np.float32:
            x = x.astype(np.float32)
        if h0 is None:
            h_prev = np.zeros((N, Hp), dtype=np.float32)
        else:
            if h0.shape != (N, Hp):
                raise ValueError("h0 shape mismatch for LSTMP")
            h_prev = h0.astype(np.float32)
        if c0 is None:
            c_prev = np.zeros((N, Hc), dtype=np.float32)
        else:
            if c0.shape != (N, Hc):
                raise ValueError("c0 shape mismatch for LSTMP")
            c_prev = c0.astype(np.float32)
        if self._training and self.p_in > 0.0:
            mx = (self.rng.random((N, D), dtype=np.float32) >= self.p_in).astype(np.float32) / (1.0 - self.p_in)
        else:
            mx = np.ones((N, D), dtype=np.float32)
        if self._training and self.p_hidden > 0.0:
            mh = (self.rng.random((N, Hp), dtype=np.float32) >= self.p_hidden).astype(np.float32) / (1.0 - self.p_hidden)
        else:
            mh = np.ones((N, Hp), dtype=np.float32)
        if self._training and self.p_weightdrop > 0.0:
            Mw = (self.rng.random(self.Wr.data.shape, dtype=np.float32) >= self.p_weightdrop).astype(np.float32)
        else:
            Mw = np.ones_like(self.Wr.data)
        if self._training and self.p_zoneout > 0.0:
            mz = (self.rng.random((N, Hp), dtype=np.float32) >= self.p_zoneout).astype(np.float32)
        else:
            mz = np.ones((N, Hp), dtype=np.float32)
        hs = np.empty((N, T, Hp), dtype=np.float32)
        hs_unproj = np.empty((N, T, Hc), dtype=np.float32)
        xs_m = np.empty((N, T, D), dtype=np.float32)
        cs = np.empty((N, T, Hc), dtype=np.float32)
        cs_hat = np.empty((N, T, Hc), dtype=np.float32) if self.norm != "none" else None
        i_s = np.empty((N, T, Hc), dtype=np.float32)
        f_s = np.empty((N, T, Hc), dtype=np.float32)
        g_s = np.empty((N, T, Hc), dtype=np.float32)
        o_s = np.empty((N, T, Hc), dtype=np.float32)
        caches_gates = []
        caches_cell = []
        for t in range(T):
            xt = x[:, t, :] * mx
            xs_m[:, t, :] = xt
            hp = h_prev * mh
            z = xt @ self.Wx.data + hp @ (self.Wr.data * Mw) + self.b.data
            zi = z[:, :Hc]
            zf = z[:, Hc:2 * Hc]
            zg = z[:, 2 * Hc:3 * Hc]
            zo = z[:, 3 * Hc:]
            if self.peepholes:
                zi = zi + c_prev * self.pi.data
                zf = zf + c_prev * self.pf.data
            if self.norm == "layernorm":
                yi, ci = _layer_norm_forward(zi, self.gamma_g.data[0], self.beta_g.data[0], self.eps)
                yf, cf = _layer_norm_forward(zf, self.gamma_g.data[1], self.beta_g.data[1], self.eps)
                yg, cg = _layer_norm_forward(zg, self.gamma_g.data[2], self.beta_g.data[2], self.eps)
                yo, co = _layer_norm_forward(zo, self.gamma_g.data[3], self.beta_g.data[3], self.eps)
                caches_gates.append((ci, cf, cg, co))
            elif self.norm == "rmsnorm":
                yi, ci = _rms_norm_forward(zi, self.gamma_g.data[0], self.beta_g.data[0], self.eps)
                yf, cf = _rms_norm_forward(zf, self.gamma_g.data[1], self.beta_g.data[1], self.eps)
                yg, cg = _rms_norm_forward(zg, self.gamma_g.data[2], self.beta_g.data[2], self.eps)
                yo, co = _rms_norm_forward(zo, self.gamma_g.data[3], self.beta_g.data[3], self.eps)
                caches_gates.append((ci, cf, cg, co))
            else:
                yi, yf, yg, yo = zi, zf, zg, zo
                caches_gates.append(None)
            f = _sigmoid(yf)
            if self.cifg:
                i = 1.0 - f
            else:
                i = _sigmoid(yi)
            g = np.tanh(yg)
            o = _sigmoid(yo)
            c = f * c_prev + i * g
            if self.norm == "layernorm":
                ch, cc = _layer_norm_forward(c, self.gamma_c.data, self.beta_c.data, self.eps)
                caches_cell.append(cc)
                ct = ch
                cs_hat[:, t, :] = ch
            elif self.norm == "rmsnorm":
                ch, cc = _rms_norm_forward(c, self.gamma_c.data, self.beta_c.data, self.eps)
                caches_cell.append(cc)
                ct = ch
                cs_hat[:, t, :] = ch
            else:
                ct = c
                caches_cell.append(None)
            if self.peepholes:
                yo_peep = yo + c * self.po.data
                if self.norm == "layernorm":
                    yo, co = _layer_norm_forward(yo_peep, self.gamma_g.data[3], self.beta_g.data[3], self.eps)
                    caches_gates[-1] = (caches_gates[-1][0], caches_gates[-1][1], caches_gates[-1][2], co)
                elif self.norm == "rmsnorm":
                    yo, co = _rms_norm_forward(yo_peep, self.gamma_g.data[3], self.beta_g.data[3], self.eps)
                    caches_gates[-1] = (caches_gates[-1][0], caches_gates[-1][1], caches_gates[-1][2], co)
                else:
                    yo = yo_peep
                o = _sigmoid(yo)
            h_unproj = o * np.tanh(ct)
            h_new = h_unproj @ self.Wp.data
            if self._training and self.p_zoneout > 0.0:
                h = mz * h_new + (1.0 - mz) * h_prev
            else:
                h = h_new
            i_s[:, t, :] = i
            f_s[:, t, :] = f
            g_s[:, t, :] = g
            o_s[:, t, :] = o
            cs[:, t, :] = c
            hs[:, t, :] = h
            hs_unproj[:, t, :] = h_unproj
            h_prev = h
            c_prev = c
        self._cache = {
            "x_m": xs_m,
            "mx": mx,
            "mh": mh,
            "Mw": Mw,
            "mz": mz,
            "hs": hs,
            "hs_unproj": hs_unproj,
            "cs": cs,
            "cs_hat": cs_hat,
            "i": i_s,
            "f": f_s,
            "g": g_s,
            "o": o_s,
            "gates_caches": caches_gates,
            "cell_caches": caches_cell,
        }
        self._return_sequence = return_sequence
        if return_sequence:
            return hs
        return hs[:, -1, :]

    def backward(self, dh: np.ndarray) -> np.ndarray:
        if self._cache is None:
            raise RuntimeError("LSTMP backward called before forward")
        x_m = self._cache["x_m"]
        mx = self._cache["mx"]
        mh = self._cache["mh"]
        Mw = self._cache["Mw"]
        mz = self._cache["mz"]
        hs = self._cache["hs"]
        hs_unproj = self._cache["hs_unproj"]
        cs = self._cache["cs"]
        cs_hat = self._cache["cs_hat"]
        i_s = self._cache["i"]
        f_s = self._cache["f"]
        g_s = self._cache["g"]
        o_s = self._cache["o"]
        gates_caches = self._cache["gates_caches"]
        cell_caches = self._cache["cell_caches"]
        N, T, D = x_m.shape
        Hc = self.cell_size
        Hp = self.proj_size
        if self._return_sequence:
            if dh.shape != (N, T, Hp):
                raise ValueError("Upstream grad shape mismatch for sequence output in LSTMP")
            dhs = dh.copy()
        else:
            if dh.shape != (N, Hp):
                raise ValueError("Upstream grad shape mismatch for last output in LSTMP")
            dhs = np.zeros((N, T, Hp), dtype=np.float32)
            dhs[:, -1, :] = dh
        dWx = np.zeros_like(self.Wx.data)
        dWr = np.zeros_like(self.Wr.data)
        dWp = np.zeros_like(self.Wp.data)
        db = np.zeros_like(self.b.data)
        if self.norm != "none":
            dgamma_g = np.zeros_like(self.gamma_g.data)
            dbeta_g = np.zeros_like(self.beta_g.data)
            dgamma_c = np.zeros_like(self.gamma_c.data)
            dbeta_c = np.zeros_like(self.beta_c.data)
        dx = np.zeros((N, T, D), dtype=np.float32)
        dh_next = np.zeros((N, Hp), dtype=np.float32)
        dc_next = np.zeros((N, Hc), dtype=np.float32)
        for t in range(T - 1, -1, -1):
            h_prev = hs[:, t - 1, :] if t > 0 else np.zeros((N, Hp), dtype=np.float32)
            c_prev = cs[:, t - 1, :] if t > 0 else np.zeros((N, Hc), dtype=np.float32)
            ct = cs_hat[:, t, :] if self.norm != "none" else cs[:, t, :]
            tanh_ct = np.tanh(ct)
            dout_proj = dhs[:, t, :] + dh_next
            if self._training and self.p_zoneout > 0.0:
                dht_proj = dout_proj * mz
                dh_to_prev = dout_proj * (1.0 - mz)
            else:
                dht_proj = dout_proj
                dh_to_prev = 0.0
            dWp += hs_unproj[:, t, :].T @ dht_proj
            dh_unproj = dht_proj @ self.Wp.data.T
            do_pre = dh_unproj * tanh_ct * o_s[:, t, :] * (1.0 - o_s[:, t, :])
            dct = dh_unproj * o_s[:, t, :] * (1.0 - tanh_ct * tanh_ct) + dc_next
            if self.norm == "layernorm":
                dc, dgc, dbc = _layer_norm_backward(dct, cell_caches[t])
                dgamma_c += dgc
                dbeta_c += dbc
            elif self.norm == "rmsnorm":
                dc, dgc, dbc = _rms_norm_backward(dct, cell_caches[t])
                dgamma_c += dgc
                if dbc is not None:
                    dbeta_c += dbc
            else:
                dc = dct
            if self.peepholes:
                do_pre += dc * self.po.data * o_s[:, t, :] * (1.0 - o_s[:, t, :])
            if self.cifg:
                di_pre = None
            else:
                di_pre = dc * g_s[:, t, :] * i_s[:, t, :] * (1.0 - i_s[:, t, :])
            df_pre = dc * c_prev * f_s[:, t, :] * (1.0 - f_s[:, t, :])
            dg_pre = dc * i_s[:, t, :] * (1.0 - g_s[:, t, :] * g_s[:, t, :])
            if self.peepholes:
                df_pre += dc * self.pf.data * f_s[:, t, :] * (1.0 - f_s[:, t, :])
            dc_next = dc * f_s[:, t, :]
            ci, cf, cg, co = gates_caches[t] if gates_caches[t] is not None else (None, None, None, None)
            if self.norm == "layernorm":
                if di_pre is not None:
                    di, dgi, dbi = _layer_norm_backward(di_pre, ci)
                else:
                    di, dgi, dbi = None, 0.0, 0.0
                df, dgf, dbf = _layer_norm_backward(df_pre, cf)
                dg, dgg, dbg = _layer_norm_backward(dg_pre, cg)
                do, dgo, dbo = _layer_norm_backward(do_pre, co)
                dgamma_g[0] += dgi
                dgamma_g[1] += dgf
                dgamma_g[2] += dgg
                dgamma_g[3] += dgo
                dbeta_g[0] += dbi
                dbeta_g[1] += dbf
                dbeta_g[2] += dbg
                dbeta_g[3] += dbo
            elif self.norm == "rmsnorm":
                if di_pre is not None:
                    di, dgi, dbi = _rms_norm_backward(di_pre, ci)
                else:
                    di, dgi, dbi = None, 0.0, None
                df, dgf, dbf = _rms_norm_backward(df_pre, cf)
                dg, dgg, dbg = _rms_norm_backward(dg_pre, cg)
                do, dgo, dbo = _rms_norm_backward(do_pre, co)
                dgamma_g[0] += dgi
                dgamma_g[1] += dgf
                dgamma_g[2] += dgg
                dgamma_g[3] += dgo
                if dbi is not None:
                    dbeta_g[0] += dbi
                if dbf is not None:
                    dbeta_g[1] += dbf
                if dbg is not None:
                    dbeta_g[2] += dbg
                if dbo is not None:
                    dbeta_g[3] += dbo
            else:
                di, df, dg, do = di_pre, df_pre, dg_pre, do_pre
            if di is None:
                da = np.concatenate([df, dg, do], axis=1)
                pad_left = np.zeros((N, Hc), dtype=np.float32)
                da_full = np.concatenate([pad_left, da], axis=1)
            else:
                da_full = np.concatenate([di, df, dg, do], axis=1)
            if self.peepholes:
                if di is not None:
                    self.pi.grad += (di * c_prev).sum(axis=0)
                self.pf.grad += (df * c_prev).sum(axis=0)
                self.po.grad += (do * cs[:, t, :]).sum(axis=0)
                dc_next += (di * self.pi.data).astype(np.float32) if di is not None else 0.0
                dc_next += (df * self.pf.data).astype(np.float32)
                dc_next += (do * self.po.data).astype(np.float32)
            dWx += x_m[:, t, :].T @ da_full
            dWr += (h_prev * mh).T @ da_full
            db += da_full.sum(axis=0)
            dx_t = da_full @ self.Wx.data.T
            dh_prev_m = da_full @ (self.Wr.data * Mw).T
            dx[:, t, :] = dx_t * mx
            dh_next = dh_prev_m * mh + dh_to_prev
        self.Wx.grad += dWx
        self.Wr.grad += dWr
        self.Wp.grad += dWp
        self.b.grad += db
        if self.norm != "none":
            self.gamma_g.grad += dgamma_g
            self.beta_g.grad += dbeta_g
            self.gamma_c.grad += dgamma_c
            self.beta_c.grad += dbeta_c
        return dx

    def init_state(self, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        h = np.zeros((batch_size, self.proj_size), dtype=np.float32)
        c = np.zeros((batch_size, self.cell_size), dtype=np.float32)
        return h, c

    def step(self, x_t: np.ndarray, h_prev: np.ndarray, c_prev: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if x_t.ndim != 2:
            raise ValueError("x_t must be (N,D)")
        N, D = x_t.shape
        if D != self.input_size:
            raise ValueError("feature size mismatch in LSTMP.step")
        if h_prev.shape != (N, self.proj_size) or c_prev.shape != (N, self.cell_size):
            raise ValueError("state shape mismatch in LSTMP.step")
        Hc = self.cell_size
        xt = x_t.astype(np.float32)
        z = xt @ self.Wx.data + h_prev @ self.Wr.data + self.b.data
        zi = z[:, :Hc]
        zf = z[:, Hc:2 * Hc]
        zg = z[:, 2 * Hc:3 * Hc]
        zo = z[:, 3 * Hc:]
        if self.peepholes:
            zi = zi + c_prev * self.pi.data
            zf = zf + c_prev * self.pf.data
        if self.norm == "layernorm":
            yi, _ = _layer_norm_forward(zi, self.gamma_g.data[0], self.beta_g.data[0], self.eps)
            yf, _ = _layer_norm_forward(zf, self.gamma_g.data[1], self.beta_g.data[1], self.eps)
            yg, _ = _layer_norm_forward(zg, self.gamma_g.data[2], self.beta_g.data[2], self.eps)
            yo, _ = _layer_norm_forward(zo, self.gamma_g.data[3], self.beta_g.data[3], self.eps)
        elif self.norm == "rmsnorm":
            yi, _ = _rms_norm_forward(zi, self.gamma_g.data[0], self.beta_g.data[0], self.eps)
            yf, _ = _rms_norm_forward(zf, self.gamma_g.data[1], self.beta_g.data[1], self.eps)
            yg, _ = _rms_norm_forward(zg, self.gamma_g.data[2], self.beta_g.data[2], self.eps)
            yo, _ = _rms_norm_forward(zo, self.gamma_g.data[3], self.beta_g.data[3], self.eps)
        else:
            yi, yf, yg, yo = zi, zf, zg, zo
        f = _sigmoid(yf)
        i = (1.0 - f) if self.cifg else _sigmoid(yi)
        g = np.tanh(yg)
        if self.peepholes:
            yo = yo + (f * c_prev + i * g) * self.po.data
        o = _sigmoid(yo)
        c = f * c_prev + i * g
        if self.norm == "layernorm":
            ct, _ = _layer_norm_forward(c, self.gamma_c.data, self.beta_c.data, self.eps)
        elif self.norm == "rmsnorm":
            ct, _ = _rms_norm_forward(c, self.gamma_c.data, self.beta_c.data, self.eps)
        else:
            ct = c
        h_unproj = o * np.tanh(ct)
        h = h_unproj @ self.Wp.data
        return h, c
