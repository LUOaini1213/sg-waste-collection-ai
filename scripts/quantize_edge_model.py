"""INT8 static quantisation of the TrashNet ONNX model, accuracy and CPU latency, FP32 vs INT8.

Needs onnx + onnxruntime + Pillow (requirements-edge.txt) and the images from
scripts/fetch_trashnet.py. Calibration: 300 training-split images (fixed selection); QDQ
format, per-channel symmetric INT8 weights, UINT8 activations, MinMax calibration.
Evaluation: the 17 % test split of data/inputs/trashnet_split.csv.
Latency: ONNX Runtime CPU execution provider on THIS machine (not an edge device), batch 1,
20 warm-up runs then 300 timed runs over test images, 1 and 4 intra-op threads.
Writes models/trashnet_mnv3s_int8.onnx, results/edge_model.json,
results/edge_reference_outputs.json (logits on fixed synthetic inputs, used by a test) and a figure.

usage: python scripts/quantize_edge_model.py
"""
from __future__ import annotations

import hashlib
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wastecollect.edge.data import CLASSES, confusion, macro_f1, preprocess  # noqa: E402
from wastecollect.io import write_json  # noqa: E402

IMG = ROOT / "data" / "external" / "trashnet" / "dataset-resized"
FP32 = ROOT / "models" / "trashnet_mnv3s_fp32.onnx"
INT8 = ROOT / "models" / "trashnet_mnv3s_int8.onnx"
N_CALIB, N_WARM, N_TIMED = 300, 20, 300


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def cpu_name() -> str:
    try:
        import winreg
        k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
        return winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
    except Exception:
        return platform.processor()


def session(path: Path, threads: int):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def quantize(calib: np.ndarray) -> None:
    from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType,
                                          quantize_static)
    from onnxruntime.quantization.shape_inference import quant_pre_process

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.it = iter(calib)

        def get_next(self):
            x = next(self.it, None)
            return None if x is None else {"input": x[None]}

    pre = ROOT / "artifacts" / "trashnet_fp32_pre.onnx"
    pre.parent.mkdir(exist_ok=True)
    quant_pre_process(str(FP32), str(pre))
    quantize_static(str(pre), str(INT8), Reader(), quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                    calibrate_method=CalibrationMethod.MinMax)


def predict(sess, X: np.ndarray, batch: int = 32) -> np.ndarray:
    out = [sess.run(None, {"input": X[i:i + batch]})[0] for i in range(0, len(X), batch)]
    return np.concatenate(out)


def latency(sess, X: np.ndarray) -> dict:
    for i in range(N_WARM):
        sess.run(None, {"input": X[i % len(X)][None]})
    ts = []
    for i in range(N_TIMED):
        x = X[i % len(X)][None]
        t0 = time.perf_counter()
        sess.run(None, {"input": x})
        ts.append((time.perf_counter() - t0) * 1000.0)
    ts = np.array(ts)
    return {"p50_ms": float(np.percentile(ts, 50)), "p95_ms": float(np.percentile(ts, 95)),
            "mean_ms": float(ts.mean()), "runs": N_TIMED}


def main() -> int:
    import onnxruntime as ort
    split = pd.read_csv(ROOT / "data" / "inputs" / "trashnet_split.csv")
    train = split[split["split"] == "train"].reset_index(drop=True)
    test = split[split["split"] == "test"].reset_index(drop=True)
    rng = np.random.default_rng(0)
    calib_files = train["file"].iloc[np.sort(rng.choice(len(train), size=N_CALIB, replace=False))]
    calib = np.stack([preprocess(IMG / f) for f in calib_files])
    quantize(calib)
    X = np.stack([preprocess(IMG / f) for f in test["file"]])
    y = test["label"].to_numpy()
    res = {"test_images": int(len(y)), "calibration_images": N_CALIB, "classes": list(CLASSES),
           "onnxruntime": ort.__version__, "cpu": cpu_name(), "python": platform.python_version(),
           "edge_hardware_tested": False, "models": {}}
    preds = {}
    for name, path in (("fp32", FP32), ("int8", INT8)):
        s1 = session(path, 1)
        logits = predict(s1, X)
        p = logits.argmax(1)
        preds[name] = p
        res["models"][name] = {
            "file": f"models/{path.name}", "bytes": path.stat().st_size, "sha256": sha256(path),
            "accuracy": float((p == y).mean()), "macro_f1": macro_f1(y, p),
            "per_class_recall": {c: float(((p == k) & (y == k)).sum() / max((y == k).sum(), 1)) for k, c in enumerate(CLASSES)},
            "confusion": confusion(y, p),
            "latency_1_thread": latency(s1, X),
            "latency_4_threads": latency(session(path, 4), X),
        }
    res["int8_fp32_prediction_agreement"] = float((preds["fp32"] == preds["int8"]).mean())
    res["size_ratio_int8_over_fp32"] = res["models"]["int8"]["bytes"] / res["models"]["fp32"]["bytes"]
    write_json(ROOT / "results" / "edge_model.json", res)
    # reference outputs on fixed synthetic inputs (for a replay test that needs no images)
    ref_rng = np.random.default_rng(2026)
    Z = ref_rng.standard_normal((4, 3, 224, 224)).astype(np.float32)
    ref = {"seed": 2026, "shape": [4, 3, 224, 224]}
    for name, path in (("fp32", FP32), ("int8", INT8)):
        ref[name] = {"sha256": sha256(path), "logits": predict(session(path, 1), Z).round(5).tolist()}
    write_json(ROOT / "results" / "edge_reference_outputs.json", ref)
    figure(res)
    for name in ("fp32", "int8"):
        m = res["models"][name]
        print(name, f"acc {m['accuracy']:.4f} f1 {m['macro_f1']:.4f} size {m['bytes'] / 1e6:.2f} MB",
              f"p50 {m['latency_1_thread']['p50_ms']:.2f} ms p95 {m['latency_1_thread']['p95_ms']:.2f} ms (1 thread)")
    return 0


def figure(res: dict) -> None:
    from wastecollect import plotstyle as ps
    import matplotlib.pyplot as plt
    ps.apply()
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.9))
    col = {"fp32": ps.SERIES[0], "int8": ps.SERIES[1]}
    for ax, (key, title, fmt) in zip(axes, [("accuracy", "Test accuracy", "{:.1%}"), ("bytes", "Model size (MB)", "{:.2f}"),
                                            ("p50", "CPU latency p50 / p95 (ms, 1 thread)", "{:.1f}")]):
        for k, name in enumerate(("fp32", "int8")):
            m = res["models"][name]
            if key == "bytes":
                v = m["bytes"] / 1e6
            elif key == "p50":
                v = m["latency_1_thread"]["p50_ms"]
                ax.errorbar(k, v, yerr=[[0], [m["latency_1_thread"]["p95_ms"] - v]], color=ps.INK2, capsize=3, lw=1)
            else:
                v = m[key]
            ax.bar(k, v, 0.6, color=col[name])
            ax.text(k, v, fmt.format(v), ha="center", va="bottom", fontsize=8, color=ps.INK2)
        ax.set_xticks([0, 1], ["FP32", "INT8"])
        ax.set_title(title, loc="left", fontsize=10)
    fig.suptitle("MobileNetV3-Small on TrashNet, ONNX Runtime on a laptop CPU (not edge hardware)", x=0.01,
                 ha="left", fontsize=11, fontweight="bold")
    ps.save(fig, ROOT / "figures" / "edge_fp32_vs_int8.png")


if __name__ == "__main__":
    sys.exit(main())
