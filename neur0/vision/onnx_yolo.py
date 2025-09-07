from __future__ import annotations

import os
import pathlib
import urllib.request
from typing import Tuple

import numpy as np


_DEFAULT_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.onnx"


def _download(url: str, dst: str) -> None:
    pathlib.Path(os.path.dirname(dst)).mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dst)


def _letterbox(im: np.ndarray, new_shape: Tuple[int, int] = (640, 640)) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    h, w = im.shape[:2]
    r = min(new_shape[0] / h, new_shape[1] / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = np.zeros((new_shape[0], new_shape[1], 3), dtype=im.dtype)
    if nh > 0 and nw > 0:
        import cv2  # type: ignore
        tmp = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    else:
        tmp = np.zeros((nh, nw, 3), dtype=im.dtype)
    top = (new_shape[0] - nh) // 2
    left = (new_shape[1] - nw) // 2
    resized[top:top + nh, left:left + nw] = tmp
    return resized, r, (left, top)


class YoloV8NanoONNX:
    def __init__(self, model_path: str | None = None, providers: list[str] | None = None) -> None:
        try:
            import onnxruntime as ort  # type: ignore
        except Exception as e:
            raise RuntimeError("onnxruntime is required for YoloV8NanoONNX") from e
        if model_path is None:
            candidates = [
                os.path.join(os.getcwd(), "yolov8n.onnx"),
                os.path.join(os.getcwd(), "models", "yolov8n.onnx"),
                os.path.join(os.path.expanduser("~"), ".cache", "neur0", "yolov8n.onnx"),
            ]
            found = None
            for c in candidates:
                if os.path.exists(c):
                    found = c
                    break
            if found is None:
                cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "neur0")
                found = os.path.join(cache_dir, "yolov8n.onnx")
                if not os.path.exists(found):
                    _download(_DEFAULT_URL, found)
            model_path = found
        else:
            if not os.path.exists(model_path):
                raise FileNotFoundError(model_path)
        self.model_path = model_path
        self.ort = ort.InferenceSession(model_path, providers=providers or ["CPUExecutionProvider"])  # type: ignore
        self.input_name = self.ort.get_inputs()[0].name
        self.input_hw = (self.ort.get_inputs()[0].shape[2], self.ort.get_inputs()[0].shape[3])
        self.output_name = self.ort.get_outputs()[0].name

    def preprocess(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError("frame must be HxWx3")
        img, r, (l, t) = _letterbox(frame_bgr, (int(self.input_hw[0]), int(self.input_hw[1])))
        x = img[:, :, ::-1].astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))
        x = np.expand_dims(x, 0).astype(np.float32)
        return x, r, (l, t)

    def infer(self, frame_bgr: np.ndarray) -> np.ndarray:
        x, _, _ = self.preprocess(frame_bgr)
        out = self.ort.run([self.output_name], {self.input_name: x})[0]
        return out

    def encode(self, frame_bgr: np.ndarray, topk: int = 5) -> np.ndarray:
        out = self.infer(frame_bgr)
        if out.ndim == 3 and out.shape[0] == 1:
            out = out[0]
        if out.shape[-1] < 85:
            raise RuntimeError("unexpected YOLOv8n output shape")
        conf = out[:, 4:5]
        cls = out[:, 5:]
        scores = conf * cls
        class_max = scores.max(axis=0)
        idx = np.argsort(scores.max(axis=1))[::-1][:topk]
        sel = out[idx]
        bx = sel[:, :4]
        sc = sel[:, 4:5]
        bc = sel[:, 5:]
        bc_top = bc.argmax(axis=1).astype(np.float32)[:, None]
        bx_norm = bx / np.array([[640.0, 640.0, 640.0, 640.0]], dtype=np.float32)
        top_feats = np.concatenate([bx_norm, sc, bc_top], axis=1).reshape(-1)
        feats = np.concatenate([class_max, top_feats], axis=0).astype(np.float32)
        return feats
