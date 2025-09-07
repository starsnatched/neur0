from __future__ import annotations

import os
import numpy as np


def load_yolo_logits_dataset(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if path.endswith('.npz'):
        z = np.load(path)
        if 'X' not in z:
            raise KeyError("expected key 'X' in npz")
        X = z['X']
        Y = z['Y'] if 'Y' in z else None
    elif path.endswith('.npy'):
        X = np.load(path)
        Y = None
    else:
        raise ValueError("unsupported file extension; use .npz or .npy")
    if X.ndim != 3:
        raise ValueError("YOLO logits dataset must have shape (N,T,D)")
    if X.dtype != np.float32:
        X = X.astype(np.float32)
    if Y is not None and Y.dtype != np.float32:
        Y = Y.astype(np.float32)
    return X, Y

