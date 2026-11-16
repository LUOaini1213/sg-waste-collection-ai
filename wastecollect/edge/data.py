"""TrashNet file list, deterministic split and torch-free preprocessing for ONNX evaluation.

The preprocessing mirrors torchvision's eval transform used in training
(Resize(256) with PIL bilinear, CenterCrop(224), ImageNet mean/std) so that ONNX Runtime
sees exactly what the PyTorch model saw.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CLASSES = ("cardboard", "glass", "metal", "paper", "plastic", "trash")
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
FRACTIONS = (0.70, 0.13, 0.17)  # train / val / test, as in the TrashNet README


def list_images(root: Path) -> pd.DataFrame:
    rows = []
    for k, c in enumerate(CLASSES):
        for p in sorted((root / c).glob("*.jpg")):
            rows.append({"file": f"{c}/{p.name}", "label": k, "class": c})
    return pd.DataFrame(rows)


def make_split(df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Stratified 70/13/17 split, deterministic for a seed."""
    rng = np.random.default_rng(seed)
    out = []
    for k in range(len(CLASSES)):
        sub = df[df["label"] == k].sort_values("file").reset_index(drop=True)
        perm = rng.permutation(len(sub))
        n_tr = int(round(FRACTIONS[0] * len(sub)))
        n_va = int(round(FRACTIONS[1] * len(sub)))
        split = np.empty(len(sub), dtype=object)
        split[perm[:n_tr]] = "train"
        split[perm[n_tr:n_tr + n_va]] = "val"
        split[perm[n_tr + n_va:]] = "test"
        out.append(sub.assign(split=split))
    return pd.concat(out, ignore_index=True)


def preprocess(path, size: int = 224, resize: int = 256) -> np.ndarray:
    """(3, size, size) float32, normalised; same geometry as torchvision Resize + CenterCrop."""
    from PIL import Image
    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        if w <= h:
            nw, nh = resize, int(resize * h / w)
        else:
            nw, nh = int(resize * w / h), resize
        im = im.resize((nw, nh), Image.BILINEAR)
        top = int(round((nh - size) / 2.0))
        left = int(round((nw - size) / 2.0))
        im = im.crop((left, top, left + size, top + size))
        x = np.asarray(im, dtype=np.float32) / 255.0
    x = (x - MEAN) / STD
    return np.ascontiguousarray(x.transpose(2, 0, 1))


def macro_f1(y_true, y_pred, n_classes: int = len(CLASSES)) -> float:
    f1 = []
    for k in range(n_classes):
        tp = np.sum((y_pred == k) & (y_true == k))
        fp = np.sum((y_pred == k) & (y_true != k))
        fn = np.sum((y_pred != k) & (y_true == k))
        f1.append(2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0)
    return float(np.mean(f1))


def confusion(y_true, y_pred, n_classes: int = len(CLASSES)) -> list:
    m = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        m[int(t), int(p)] += 1
    return m.tolist()
