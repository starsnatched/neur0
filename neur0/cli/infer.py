from __future__ import annotations

import argparse
import time
import numpy as np

from ..vision.onnx_yolo import YoloV8NanoONNX
from ..realtime.context import TemporalContext
from ..nn.lstm import LSTM, LSTMP
from ..nn.layers import Linear


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-id", type=str, default="LunarLanderContinuous-v3")
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--alphas", type=str, default="0.9,0.99,0.999")
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--yolo-path", type=str, default="")
    args = ap.parse_args()

    import gymnasium as gym  # type: ignore
    import cv2  # type: ignore

    z = np.load(args.checkpoint)
    in_dim = int(z["in_dim"][0])
    out_dim = int(z["out_dim"][0])
    layers = int(z["layers"][0])
    hidden = int(z["hidden"][0])
    proj = int(z["proj"][0])

    enc = YoloV8NanoONNX(model_path=(args.yolo_path or None))
    alphas = [float(x) for x in args.alphas.split(",") if x]
    ctx = TemporalContext(dim=enc.encode(np.zeros((640,640,3), dtype=np.uint8)).shape[0], alphas=alphas, window=args.window)
    # Build network
    modules = []
    Din = in_dim
    for li in range(layers):
        if proj > 0:
            m = LSTMP(Din, hidden, proj, seed=li, norm="rmsnorm")
            Din = m.proj_size
        else:
            m = LSTM(Din, hidden, seed=li, norm="rmsnorm")
            Din = m.hidden_size
        modules.append(m)
    head = Linear(Din, out_dim, seed=999)
    params = []
    for m in modules: params.extend(m.parameters())
    params.extend(head.parameters())
    for i, p in enumerate(params):
        p.data[...] = z[f"w_{i}"]

    env = gym.make(args.env_id, render_mode="rgb_array")
    obs, info = env.reset()
    h_states = []
    c_states = []
    bs = 1
    for m in modules:
        h, c = m.init_state(bs)
        h_states.append(h)
        c_states.append(c)
    interval = 1.0 / max(1e-6, args.fps)
    last = time.time()
    while True:
        frame = env.render()
        if frame is None:
            break
        f = enc.encode(frame, topk=args.topk)
        zf = ctx.augment(f)[None, :]
        x = zf
        for i, m in enumerate(modules):
            h, c = m.step(x, h_states[i], c_states[i])
            h_states[i], c_states[i] = h, c
            x = h
        y = head.forward(x)
        act = np.clip(y[0], -1.0, 1.0)
        obs, reward, terminated, truncated, info = env.step(act)
        frame_vis = frame.copy()
        cv2.putText(frame_vis, f"act: {act[0]:+.2f}, {act[1]:+.2f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.imshow("Lander", frame_vis)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if terminated or truncated:
            obs, info = env.reset()
            ctx.reset()
            for i, m in enumerate(modules):
                h_states[i], c_states[i] = m.init_state(bs)
        now = time.time()
        if now - last < interval:
            time.sleep(interval - (now - last))
        last = time.time()
    env.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
