from __future__ import annotations

import argparse
import time
from typing import List

import numpy as np

from neur0.vision.onnx_yolo import YoloV8NanoONNX
from neur0.realtime.context import TemporalContext, ContextProjector
from neur0.realtime.runner import LanderRealtime
from neur0.nn.lstm import LSTM, LSTMP
from neur0.nn.layers import Linear


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--video", type=str, default="")
    ap.add_argument("--providers", type=str, default="CPUExecutionProvider")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--alphas", type=str, default="0.9,0.99,0.999")
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--proj", type=int, default=128)
    ap.add_argument("--out", type=int, default=3)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--checkpoint", type=str, default="")
    ap.add_argument("--project-dim", type=int, default=0)
    ap.add_argument("--fps", type=float, default=30.0)
    args = ap.parse_args()

    try:
        import cv2  # type: ignore
    except Exception as e:
        raise RuntimeError("opencv-python is required for realtime video") from e

    enc = YoloV8NanoONNX(providers=[p.strip() for p in args.providers.split(",")])
    alphas = [float(x) for x in args.alphas.split(",") if x]

    cap = cv2.VideoCapture(args.camera if not args.video else args.video)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("failed to read initial frame")
    d_raw = enc.encode(frame, topk=args.topk).shape[0]
    ctx = TemporalContext(dim=d_raw, alphas=alphas, window=args.window)
    in_dim = ctx.augmented_dim()
    projector = None
    if args.project_dim > 0:
        projector = ContextProjector(in_dim, args.project_dim, seed=args.seed)
        in_dim = args.project_dim

    if args.proj > 0:
        core = LSTMP(in_dim, args.hidden, args.proj, seed=args.seed, norm="rmsnorm")
        Din_head = args.proj
    else:
        core = LSTM(in_dim, args.hidden, seed=args.seed, norm="rmsnorm")
        Din_head = args.hidden
    head = Linear(Din_head, args.out, seed=args.seed + 1)
    if args.checkpoint:
        z = np.load(args.checkpoint)
        params = [*core.parameters(), *head.parameters()]
        n_params = int(z["n_params"][0]) if "n_params" in z else len(params)
        for i in range(min(n_params, len(params))):
            params[i].data[...] = z[f"w_{i}"]
    rt = LanderRealtime(core, head, ctx, projector)
    rt.reset(1)

    interval = 1.0 / max(1e-6, args.fps)
    last = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        feats = enc.encode(frame, topk=args.topk)
        y = rt.step(feats)
        now = time.time()
        if now - last >= interval:
            last = now
            print("output:", y[0])
        if args.video == "":
            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    if args.video == "":
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
