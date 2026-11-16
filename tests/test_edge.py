"""Edge classifier: split, preprocessing, metrics, and the committed ONNX models (no images, no torch)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wastecollect.edge.data import CLASSES, MEAN, STD, confusion, macro_f1, make_split, preprocess

ROOT = Path(__file__).resolve().parents[1]


def test_split_is_stratified_disjoint_and_deterministic():
    df = pd.DataFrame({"file": [f"{c}/{c}{i}.jpg" for c in CLASSES for i in range(100)],
                       "label": [k for k in range(len(CLASSES)) for _ in range(100)],
                       "class": [c for c in CLASSES for _ in range(100)]})
    a, b = make_split(df, seed=0), make_split(df, seed=0)
    assert a.equals(b)
    assert not a["split"].equals(make_split(df, seed=1)["split"])
    counts = a.groupby(["label", "split"]).size().unstack()
    assert (counts["train"] == 70).all() and (counts["val"] == 13).all() and (counts["test"] == 17).all()
    assert a["file"].is_unique and len(a) == 600


def test_committed_split_matches_trashnet_counts():
    s = pd.read_csv(ROOT / "data" / "inputs" / "trashnet_split.csv")
    assert len(s) == 2527 and s["file"].is_unique
    assert s.groupby("class").size().to_dict() == {"cardboard": 403, "glass": 501, "metal": 410, "paper": 594,
                                                    "plastic": 482, "trash": 137}
    frac = s["split"].value_counts(normalize=True)
    assert frac["train"] == pytest.approx(0.70, abs=0.005) and frac["test"] == pytest.approx(0.17, abs=0.005)
    assert set(s.loc[s["split"] == "train", "file"]).isdisjoint(s.loc[s["split"] == "test", "file"])


def test_preprocess_constant_colour_and_geometry(tmp_path):
    from PIL import Image
    p = tmp_path / "flat.png"
    Image.new("RGB", (512, 384), (128, 64, 32)).save(p)
    x = preprocess(p)
    assert x.shape == (3, 224, 224) and x.dtype == np.float32
    want = (np.array([128, 64, 32], dtype=np.float32) / 255.0 - MEAN) / STD
    assert np.allclose(x[:, 0, 0], want, atol=1e-6) and np.allclose(x[:, 200, 17], want, atol=1e-6)
    # left half black, right half white: 512x384 -> 341x256, centre crop starts at column 58,
    # so the edge (column 170.5 of 341) lands between crop columns 112 and 113
    arr = np.zeros((384, 512, 3), dtype=np.uint8)
    arr[:, 256:] = 255
    q = tmp_path / "edge.png"
    Image.fromarray(arr).save(q)
    y = preprocess(q)
    black, white = (0.0 - MEAN) / STD, (1.0 - MEAN) / STD
    assert np.allclose(y[:, 100, 109], black, atol=1e-5) and np.allclose(y[:, 100, 116], white, atol=1e-5)
    # portrait input: the short side (width) is resized to 256
    Image.new("RGB", (384, 512), (0, 0, 0)).save(tmp_path / "tall.png")
    assert preprocess(tmp_path / "tall.png").shape == (3, 224, 224)


def test_macro_f1_and_confusion_hand_worked():
    y = np.array([0, 0, 1, 1, 2, 2])
    p = np.array([0, 1, 1, 1, 2, 0])
    # class 0: tp 1 fp 1 fn 1 -> 0.5; class 1: tp 2 fp 1 fn 0 -> 0.8; class 2: tp 1 fp 0 fn 1 -> 2/3
    assert macro_f1(y, p, 3) == pytest.approx((0.5 + 0.8 + 2 / 3) / 3)
    assert confusion(y, p, 3) == [[1, 1, 0], [0, 2, 0], [1, 0, 1]]
    assert macro_f1(y, y, 3) == 1.0


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_committed_models_match_results_and_size_limit():
    res = json.loads((ROOT / "results" / "edge_model.json").read_text(encoding="utf-8"))
    assert res["edge_hardware_tested"] is False
    for name in ("fp32", "int8"):
        m = res["models"][name]
        path = ROOT / m["file"]
        assert path.stat().st_size == m["bytes"] < 20 * 1024 * 1024
        assert _sha(path) == m["sha256"]
        assert sum(sum(r) for r in m["confusion"]) == res["test_images"]
        correct = sum(m["confusion"][k][k] for k in range(len(CLASSES)))
        assert m["accuracy"] == pytest.approx(correct / res["test_images"], abs=1e-6)
    assert res["models"]["int8"]["bytes"] < 0.5 * res["models"]["fp32"]["bytes"]


def test_onnx_models_reproduce_reference_outputs():
    ort = pytest.importorskip("onnxruntime")
    ref = json.loads((ROOT / "results" / "edge_reference_outputs.json").read_text(encoding="utf-8"))
    z = np.random.default_rng(ref["seed"]).standard_normal(ref["shape"]).astype(np.float32)
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    for name, atol in (("fp32", 2e-3), ("int8", 1.5)):
        path = ROOT / "models" / f"trashnet_mnv3s_{name}.onnx"
        assert _sha(path) == ref[name]["sha256"]
        out = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"]).run(None, {"input": z})[0]
        want = np.array(ref[name]["logits"])
        assert out.shape == (4, len(CLASSES))
        # INT8 kernels differ slightly between CPU instruction sets, hence the loose tolerance there
        assert np.max(np.abs(out - want)) <= atol
