from __future__ import annotations

import argparse
from typing import Tuple
import numpy as np

from neur0.nn.lstm import LSTM, LSTMP
from neur0.nn.layers import Linear
from neur0.nn.stack import LSTMStack
from neur0.losses.mse import MSELoss
from neur0.optim.adam import AdamW
from neur0.schedulers.cosine import CosineWarmup
from neur0.data.yolo import load_yolo_logits_dataset


class SmallLSTMNet:
    def __init__(self, input_size: int, hidden_size: int, output_size: int, seed: int | None = None, norm: str = "rmsnorm", p_in: float = 0.0, p_hidden: float = 0.0, eps: float = 1e-3, cifg: bool = False, peepholes: bool = False, p_weightdrop: float = 0.0, p_zoneout: float = 0.0, layers: int = 1, proj_size: int | None = None, residual: bool = True, prenorm: bool = True) -> None:
        rng = np.random.default_rng(seed)
        depth = int(layers)
        modules = []
        Din = int(input_size)
        for li in range(depth):
            if proj_size is not None and proj_size > 0:
                mod = LSTMP(Din, int(hidden_size), int(proj_size), seed=None if seed is None else seed + 31 + li, norm=norm, p_in=p_in, p_hidden=p_hidden, eps=eps, cifg=cifg, peepholes=peepholes, p_weightdrop=p_weightdrop, p_zoneout=p_zoneout)
                Din = mod.proj_size
            else:
                mod = LSTM(Din, int(hidden_size), seed=None if seed is None else seed + 31 + li, norm=norm, p_in=p_in, p_hidden=p_hidden, eps=eps, cifg=cifg, peepholes=peepholes, p_weightdrop=p_weightdrop, p_zoneout=p_zoneout)
                Din = mod.hidden_size
            modules.append(mod)
        self.stack = LSTMStack(modules, pre_norm=bool(prenorm), residual=bool(residual), eps=eps)
        self.head = Linear(Din, output_size, seed=0 if seed is None else seed + 1)

    def parameters(self):
        return [*self.stack.parameters(), *self.head.parameters()]

    def zero_grad(self) -> None:
        self.stack.zero_grad()
        self.head.zero_grad()

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = self.stack(x, return_sequence=False)
        y = self.head(h)
        return y

    def backward(self, dy: np.ndarray) -> None:
        dh = self.head.backward(dy)
        _ = self.stack.backward(dh)


def make_sine_dataset(n_samples: int, seq_len: int, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    N = n_samples
    T = seq_len
    X = np.zeros((N, T, 1), dtype=np.float32)
    Y = np.zeros((N, 1), dtype=np.float32)
    for i in range(N):
        w = rng.uniform(0.5, 1.5)
        phi = rng.uniform(0, 2 * np.pi)
        t = np.arange(T + 1, dtype=np.float32) / T
        s = np.sin(w * 2 * np.pi * t + phi)
        s += 0.05 * rng.standard_normal(size=s.shape).astype(np.float32)
        X[i, :, 0] = s[:T]
        Y[i, 0] = s[T]
    return X, Y


def batch_iter(X: np.ndarray, Y: np.ndarray, batch_size: int, shuffle: bool, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(X.shape[0])
    if shuffle:
        rng.shuffle(idx)
    for start in range(0, X.shape[0], batch_size):
        end = min(start + batch_size, X.shape[0])
        bi = idx[start:end]
        yield X[bi], Y[bi]


def clip_grad_norm(parameters, max_norm: float) -> None:
    total = 0.0
    for p in parameters:
        total += float(np.sum(p.grad * p.grad))
    total = float(np.sqrt(total))
    if total > max_norm and total > 0:
        scale = max_norm / total
        for p in parameters:
            p.grad *= scale


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-len", type=int, default=24)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--train-samples", type=int, default=4096)
    ap.add_argument("--val-samples", type=int, default=512)
    ap.add_argument("--max-grad-norm", type=float, default=5.0)
    ap.add_argument("--norm", type=str, default="rmsnorm", choices=["none", "layernorm", "rmsnorm"])
    ap.add_argument("--p-in", type=float, default=0.0)
    ap.add_argument("--p-hidden", type=float, default=0.0)
    ap.add_argument("--warmup-steps", type=int, default=100)
    ap.add_argument("--min-lr-ratio", type=float, default=0.1)
    ap.add_argument("--checkpoint", type=str, default="checkpoint.npz")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--save-every", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=0)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--cifg", action="store_true")
    ap.add_argument("--peepholes", action="store_true")
    ap.add_argument("--p-weightdrop", type=float, default=0.0)
    ap.add_argument("--p-zoneout", type=float, default=0.0)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--proj", type=int, default=0)
    ap.add_argument("--residual", action="store_true")
    ap.add_argument("--prenorm", action="store_true")
    ap.add_argument("--yolo-logits", type=str, default="")
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--ema-decay", type=float, default=0.0)
    ap.add_argument("--ema-start", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    if args.yolo_logits:
        Xall, Yall = load_yolo_logits_dataset(args.yolo_logits)
        if Yall is None:
            raise ValueError("YOLO dataset missing targets 'Y' in npz; provide labels to train")
        ntr = min(args.train_samples, Xall.shape[0])
        nva = min(args.val_samples, Xall.shape[0] - ntr)
        Xtr, Ytr = Xall[:ntr], Yall[:ntr]
        Xva, Yva = Xall[ntr:ntr + nva], Yall[ntr:ntr + nva]
        input_size = Xtr.shape[-1]
        out_size = Ytr.shape[-1] if Ytr.ndim == 2 else 1
    else:
        Xtr, Ytr = make_sine_dataset(args.train_samples, args.seq_len, seed=args.seed)
        Xva, Yva = make_sine_dataset(args.val_samples, args.seq_len, seed=args.seed + 1)
        input_size = 1
        out_size = 1

    model = SmallLSTMNet(
        input_size=input_size,
        hidden_size=args.hidden,
        output_size=out_size,
        seed=args.seed,
        norm=args.norm,
        p_in=float(args.p_in),
        p_hidden=float(args.p_hidden),
        eps=1e-3,
        cifg=bool(args.cifg),
        peepholes=bool(args.peepholes),
        p_weightdrop=float(args.p_weightdrop),
        p_zoneout=float(args.p_zoneout),
        layers=int(args.layers),
        proj_size=int(args.proj) if args.proj > 0 else None,
        residual=bool(args.residual),
        prenorm=bool(args.prenorm),
    )
    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=float(args.wd))
    loss_fn = MSELoss()
    total_steps = args.epochs * (int(np.ceil(Xtr.shape[0] / args.batch_size)))
    if args.warmup_steps >= total_steps:
        args.warmup_steps = max(1, total_steps // 10)
    sched = CosineWarmup(base_lr=args.lr, warmup_steps=args.warmup_steps, total_steps=max(1, total_steps), min_lr_ratio=args.min_lr_ratio)

    def save_ckpt(path: str, epoch: int, step: int, best_val: float):
        ps = model.parameters()
        data = {f"w_{i}": p.data for i, p in enumerate(ps)}
        opt_state = opt.state_dict()
        for i, a in enumerate(opt_state["m_list"]):
            data[f"m_{i}"] = a
        for i, a in enumerate(opt_state["v_list"]):
            data[f"v_{i}"] = a
        data["n_params"] = np.array([len(ps)], dtype=np.int64)
        data["epoch"] = np.array([epoch], dtype=np.int64)
        data["step"] = np.array([step], dtype=np.int64)
        data["best_val"] = np.array([best_val], dtype=np.float32)
        data["t"] = np.array([opt_state["t"]], dtype=np.int64)
        data["lr"] = np.array([opt_state["lr"]], dtype=np.float32)
        data["b1"] = np.array([opt_state["b1"]], dtype=np.float32)
        data["b2"] = np.array([opt_state["b2"]], dtype=np.float32)
        data["eps"] = np.array([opt_state["eps"]], dtype=np.float32)
        data["wd"] = np.array([opt_state["wd"]], dtype=np.float32)
        np.savez_compressed(path, **data)

    def load_ckpt(path: str):
        z = np.load(path)
        ps = model.parameters()
        n_params = int(z["n_params"][0])
        for i in range(min(n_params, len(ps))):
            ps[i].data[...] = z[f"w_{i}"]
        opt_state = {
            "t": int(z["t"][0]),
            "lr": float(z["lr"][0]),
            "b1": float(z["b1"][0]),
            "b2": float(z["b2"][0]),
            "eps": float(z["eps"][0]),
            "wd": float(z["wd"][0]),
            "m_list": [z[f"m_{i}"] for i in range(n_params)],
            "v_list": [z[f"v_{i}"] for i in range(n_params)],
        }
        opt.load_state_dict(opt_state)
        return int(z["epoch"][0]), int(z["step"][0]), float(z["best_val"][0])

    start_epoch = 1
    global_step = 0
    best_val = float("inf")
    if args.ema_decay > 0.0:
        ema_shadow = [p.data.copy() for p in model.parameters()]
    else:
        ema_shadow = None
    if args.resume:
        try:
            e0, s0, bv = load_ckpt(args.checkpoint)
            start_epoch = e0
            global_step = s0
            best_val = bv
        except Exception:
            pass
    no_improve = 0
    for epoch in range(start_epoch, args.epochs + 1):
        model.stack.train()
        model.head.train()
        total_loss = 0.0
        n_batches = 0
        for xb, yb in batch_iter(Xtr, Ytr, args.batch_size, shuffle=True, seed=args.seed + epoch):
            model.zero_grad()
            yhat = model.forward(xb)
            loss = loss_fn.forward(yhat, yb)
            dy = loss_fn.backward()
            model.backward(dy)
            total_grad_sq = 0.0
            for p in model.parameters():
                total_grad_sq += float(np.sum(p.grad * p.grad))
            total_grad = float(np.sqrt(max(0.0, total_grad_sq)))
            clip_grad_norm(model.parameters(), args.max_grad_norm)
            global_step += 1
            opt.lr = sched.lr(global_step)
            opt.step()
            if ema_shadow is not None and global_step >= args.ema_start:
                for i, p in enumerate(model.parameters()):
                    ema_shadow[i] = args.ema_decay * ema_shadow[i] + (1.0 - args.ema_decay) * p.data
            total_loss += float(loss)
            n_batches += 1
            if args.save_every and (global_step % args.save_every == 0):
                save_ckpt(args.checkpoint, epoch, global_step, best_val)
        avg_loss = total_loss / max(1, n_batches)
        model.stack.eval()
        model.head.eval()
        with np.errstate(all="ignore"):
            if ema_shadow is not None and global_step >= args.ema_start:
                orig = [p.data.copy() for p in model.parameters()]
                for i, p in enumerate(model.parameters()):
                    p.data[...] = ema_shadow[i]
                yh = model.forward(Xva)
                vloss = float(loss_fn.forward(yh, Yva))
                for i, p in enumerate(model.parameters()):
                    p.data[...] = orig[i]
            else:
                yh = model.forward(Xva)
                vloss = float(loss_fn.forward(yh, Yva))
        if vloss < best_val:
            best_val = vloss
            no_improve = 0
            save_ckpt(args.checkpoint, epoch, global_step, best_val)
        else:
            no_improve += 1
        print(f"epoch={epoch} step={global_step} lr={opt.lr:.6f} grad_norm={total_grad:.6f} train_loss={avg_loss:.6f} val_loss={vloss:.6f}")
        if args.eval_every and (epoch % args.eval_every == 0) and args.resume:
            pass
        if no_improve >= args.patience:
            break

    idx = np.arange(min(5, Xva.shape[0]))
    preds = model.forward(Xva[idx])[:, 0]
    targets = Yva[idx, 0]
    for i in range(preds.shape[0]):
        print(f"pred={preds[i]:+.4f} target={targets[i]:+.4f}")


if __name__ == "__main__":
    main()
