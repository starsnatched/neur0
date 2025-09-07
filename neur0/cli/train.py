from __future__ import annotations

import argparse
import os
from typing import List
import numpy as np

from ..vision.onnx_yolo import YoloV8NanoONNX
from ..vision.gym_lander import collect_episodes
from ..realtime.context import TemporalContext
from ..nn.lstm import LSTM, LSTMP
from ..nn.layers import Linear
from ..optim.adam import AdamW
from ..losses.mse import MSELoss
from ..schedulers.cosine import CosineWarmup


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-id", type=str, default="LunarLanderContinuous-v3")
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=24)
    ap.add_argument("--max-steps", type=int, default=512)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--alphas", type=str, default="0.9,0.99,0.999")
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--proj", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--norm", type=str, default="rmsnorm", choices=["none","layernorm","rmsnorm"])
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--checkpoint", type=str, default="checkpoint.npz")
    ap.add_argument("--yolo-path", type=str, default="")
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--p-in", type=float, default=0.0)
    ap.add_argument("--p-hidden", type=float, default=0.0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    enc = YoloV8NanoONNX(model_path=(args.yolo_path or None))
    alphas = [float(x) for x in args.alphas.split(",") if x]
    def ctx_factory():
        # determine raw dim by a dummy frame run on demand
        return TemporalContext(dim=enc.encode(np.zeros((640,640,3), dtype=np.uint8)).shape[0], alphas=alphas, window=args.window)

    data = collect_episodes(args.episodes, args.seq_len, enc, ctx_factory, topk=args.topk, seed=args.seed, max_steps=args.max_steps, env_id=args.env_id)
    Xtr, Ytr = data.X, data.Y
    nva = max(1, int(0.1 * Xtr.shape[0]))
    Xva, Yva = Xtr[-nva:], Ytr[-nva:]
    Xtr, Ytr = Xtr[:-nva], Ytr[:-nva]

    in_dim = Xtr.shape[-1]
    out_dim = Ytr.shape[-1]
    modules: List = []
    Din = in_dim
    for li in range(args.layers):
        if args.proj > 0:
            m = LSTMP(Din, args.hidden, args.proj, seed=args.seed + 31 + li, norm=args.norm, p_in=args.p_in, p_hidden=args.p_hidden)
            Din = m.proj_size
        else:
            m = LSTM(Din, args.hidden, seed=args.seed + 31 + li, norm=args.norm, p_in=args.p_in, p_hidden=args.p_hidden)
            Din = m.hidden_size
        modules.append(m)
    # simple sequential stack without residuals for training
    head = Linear(Din, out_dim, seed=args.seed + 7)

    params = []
    for m in modules:
        params.extend(m.parameters())
    params.extend(head.parameters())
    opt = AdamW(params, lr=args.lr, weight_decay=float(args.wd))
    loss_fn = MSELoss()
    steps_per_epoch = int(np.ceil(Xtr.shape[0] / args.batch_size))
    sched = CosineWarmup(base_lr=args.lr, warmup_steps=max(1, steps_per_epoch), total_steps=max(1, args.epochs * steps_per_epoch))

    def save_ckpt(path: str, epoch: int, step: int, best_val: float):
        data = {}
        p = []
        for m in modules:
            p.extend(m.parameters())
        p.extend(head.parameters())
        for i, par in enumerate(p):
            data[f"w_{i}"] = par.data
        data["n_params"] = np.array([len(p)], dtype=np.int64)
        data["epoch"] = np.array([epoch], dtype=np.int64)
        data["step"] = np.array([step], dtype=np.int64)
        data["best_val"] = np.array([best_val], dtype=np.float32)
        data["in_dim"] = np.array([in_dim], dtype=np.int64)
        data["out_dim"] = np.array([out_dim], dtype=np.int64)
        data["layers"] = np.array([args.layers], dtype=np.int64)
        data["hidden"] = np.array([args.hidden], dtype=np.int64)
        data["proj"] = np.array([args.proj], dtype=np.int64)
        np.savez_compressed(path, **data)

    best_val = float("inf")
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        idx = np.arange(Xtr.shape[0])
        rng.shuffle(idx)
        tot = 0.0
        nb = 0
        for bi in range(0, Xtr.shape[0], args.batch_size):
            be = min(bi + args.batch_size, Xtr.shape[0])
            bidx = idx[bi:be]
            xb = Xtr[bidx]
            yb = Ytr[bidx]
            for m in modules:
                m.train()
            head.train()
            for m in modules:
                h = m(xb, return_sequence=(m is not modules[-1]))
                xb = h
            yhat = head(xb)
            loss = loss_fn.forward(yhat, yb)
            dy = loss_fn.backward()
            dh = head.backward(dy)
            for m in reversed(modules):
                dh = m.backward(dh)
            for m in modules:
                pass
            opt.lr = sched.lr(global_step + 1)
            opt.step()
            global_step += 1
            tot += float(loss)
            nb += 1
        for m in modules:
            m.eval()
        head.eval()
        with np.errstate(all="ignore"):
            xb = Xva
            for m in modules:
                h = m(xb, return_sequence=(m is not modules[-1]))
                xb = h
            yh = head(xb)
            vloss = float(loss_fn.forward(yh, Yva))
        if vloss < best_val:
            best_val = vloss
            save_ckpt(args.checkpoint, epoch, global_step, best_val)
        print(f"epoch={epoch} step={global_step} lr={opt.lr:.6f} train_loss={tot/max(1,nb):.6f} val_loss={vloss:.6f}")


if __name__ == "__main__":
    main()
