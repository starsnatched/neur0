# neur0 — NumPy Neural Network + Real‑Time Lander

neur0 is a production‑grade, NumPy‑only neural network library and training pipeline with manual backprop. It includes clean modules for LSTM/LSTMP, normalization, optimizers, schedulers, losses, checkpointing, and configuration. A real‑time vision stack plugs in a pre‑trained YOLOv8‑nano ONNX encoder and feeds its latent vectors, augmented with temporal context, directly into an LSTM controller for OpenAI Gymnasium’s LunarLanderContinuous environment.

Core goals:
- NumPy‑only model execution with explicit forward/backward passes
- Stable, scalable LSTM and projected LSTM (LSTMP)
- Pre‑norm + residual support for deep stacks
- Deterministic training with seeding, robust checkpointing, and simple CLIs
- Real‑time inference that visualizes actions while the environment runs

## Quick Start

Prerequisites: Python 3.13+, uv, and a working display for visualization. Optional dependencies are installed automatically by uv via `pyproject.toml`.

Install uv if needed: https://github.com/astral-sh/uv

Place `yolov8n.onnx` in the project root (recommended), or pass `--yolo-path` to the CLIs. If not provided and no local file is found, the ONNX model is downloaded to `~/.cache/neur0/yolov8n.onnx`.

Train a controller from expert demonstrations generated online:

```
uv run train \
  --episodes 6 \
  --seq-len 24 \
  --layers 2 \
  --hidden 512 \
  --proj 128 \
  --epochs 5 \
  --yolo-path yolov8n.onnx
```

Run inference with visualization from the saved checkpoint:

```
uv run infer --checkpoint checkpoint.npz --yolo-path yolov8n.onnx
```

The infer window overlays the predicted continuous actions (main and side engines) and steps the `LunarLanderContinuous-v3` environment in real time.

## What Gets Trained

- A stack of `LSTM` or `LSTMP` layers predicts the 2D continuous action `[main, side]` directly from vision features.
- YOLOv8‑nano is used as a fixed encoder. Its logits/latent features are NOT trained and are fed directly into the LSTM. No pre‑LSTM projection layers are used.
- A temporal context module augments the YOLO feature at each step with multi‑timescale EMAs, a short window mean/std, and a delta term to provide memory without changing weights of the encoder or the LSTM.
- The head is a linear layer to the action dimension. Outputs are clipped to `[-1, 1]` during inference.

## YOLO Weights

The system looks for `yolov8n.onnx` in:
- `./yolov8n.onnx`
- `./models/yolov8n.onnx`
- `~/.cache/neur0/yolov8n.onnx`

You can also pass `--yolo-path /path/to/yolov8n.onnx` to both `train` and `infer`.

## CLI Reference

`uv run train` collects episodes from Gymnasium’s `LunarLanderContinuous-v3`, builds a dataset of `(N, T, D)` sequences from YOLO+context features, and trains an LSTM/LSTMP to imitate a heuristic controller.

Important flags:
- `--episodes`: number of episodes to collect
- `--seq-len`: unroll length for training windows
- `--layers`, `--hidden`, `--proj`: model depth, cell size, and LSTMP projection size (set `--proj 0` for plain LSTM)
- `--alphas`, `--window`: temporal context parameters
- `--topk`: detections used from YOLO output
- `--checkpoint`: output file (npz)
- `--yolo-path`: path to `yolov8n.onnx`

`uv run infer` loads a checkpoint, reconstructs the network, and runs the real‑time environment with visualization.

Important flags:
- `--checkpoint`: path to a saved `npz`
- `--alphas`, `--window`, `--topk`: must match training for feature shapes to align
- `--fps`: visualization cadence
- `--yolo-path`: path to `yolov8n.onnx`

Run `uv run train -h` and `uv run infer -h` for full options.

## Checkpoints

Saved to `checkpoint.npz` by default and include:
- All parameter arrays in layer order
- Model metadata: `in_dim`, `out_dim`, `layers`, `hidden`, `proj`
- Training progress fields: epoch, step, best validation loss

To resume inference, pass the checkpoint to `uv run infer`.
