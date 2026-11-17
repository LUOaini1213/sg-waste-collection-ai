"""INT8 static quantisation of the TrashNet ONNX model; accuracy, size and CPU latency vs FP32.

Needs onnx + onnxruntime + Pillow (requirements-edge.txt) and the images from
scripts/fetch_trashnet.py. All choices are made on the VALIDATION split; the test split is used
once, for the reported numbers.

Steps
  1. Full-model static quantisation (QDQ, per-channel INT8 weights, UINT8 activations, all
     supported operators) with three calibration methods: MinMax, 99.99th percentile, entropy.
  2. Sensitivity scan: quantise the Conv/Gemm nodes of one block at a time (MinMax) and record
     the validation accuracy.
  3. Mixed-precision model: Conv/Gemm nodes of every block whose solo quantisation costs at most
     SENS_DROP validation accuracy are quantised; the sensitive blocks and the activation
     operators stay in FP32. This is the committed models/trashnet_mnv3s_int8.onnx.
  4. Test accuracy, macro F1, file size and batch-1 latency (ONNX Runtime CPU execution provider
     on THIS machine, not an edge device; 20 warm-up runs, then three rounds of 300 timed runs,
     the round with the lowest median is reported; 1 and 4 intra-op threads) for FP32, the best
     full-INT8 model and the mixed model.
Calibration uses 300 training-split images (fixed selection).
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
ART = ROOT / "artifacts"
FP32 = ROOT / "models" / "trashnet_mnv3s_fp32.onnx"
INT8 = ROOT / "models" / "trashnet_mnv3s_int8.onnx"
PRE = ART / "trashnet_fp32_pre.onnx"
N_CALIB, N_WARM, N_TIMED, N_ROUNDS = 300, 20, 300, 3
FULL_METHODS = ("minmax", "percentile", "entropy")
SENS_DROP = 0.02          # a block is "sensitive" if quantising it alone loses more than 2 points (validation)
WEIGHT_OPS = ("Conv", "Gemm")


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


def block_of(node_name: str) -> str:
    """'/features/features.4/block/...' -> 'features.4'; classifier nodes -> 'classifier'."""
    parts = node_name.split("/")
    return parts[2] if node_name.startswith("/features/") and len(parts) > 2 else "classifier"


def weight_blocks() -> dict:
    import onnx
    model = onnx.load(str(PRE))
    blocks = {}
    for n in model.graph.node:
        if n.op_type in WEIGHT_OPS:
            blocks.setdefault(block_of(n.name), []).append(n.name)
    return blocks


def quantize(calib: np.ndarray, out: Path, method: str = "minmax", nodes=None) -> None:
    """Static QDQ quantisation; `nodes` restricts it to the given node names (None = every supported operator)."""
    from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType,
                                          quantize_static)

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.it = iter(calib)

        def get_next(self):
            x = next(self.it, None)
            return None if x is None else {"input": x[None]}

    cal = {"minmax": CalibrationMethod.MinMax, "percentile": CalibrationMethod.Percentile,
           "entropy": CalibrationMethod.Entropy}[method]
    extra = {"CalibPercentile": 99.99} if method == "percentile" else {}
    kw = {"nodes_to_quantize": list(nodes)} if nodes is not None else {}
    quantize_static(str(PRE), str(out), Reader(), quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                    calibrate_method=cal, extra_options=extra, **kw)


def predict(sess, X: np.ndarray, batch: int = 32) -> np.ndarray:
    out = [sess.run(None, {"input": X[i:i + batch]})[0] for i in range(0, len(X), batch)]
    return np.concatenate(out)


def accuracy(path: Path, X: np.ndarray, y: np.ndarray) -> float:
    return float((predict(session(path, 1), X).argmax(1) == y).mean())


def latency(sess, X: np.ndarray) -> dict:
    """Batch-1 latency. Three rounds of N_TIMED runs; the round with the lowest median is reported
    (other programs may be running on the machine), all round medians are kept."""
    for i in range(N_WARM):
        sess.run(None, {"input": X[i % len(X)][None]})
    rounds = []
    for _ in range(N_ROUNDS):
        ts = []
        for i in range(N_TIMED):
            x = X[i % len(X)][None]
            t0 = time.perf_counter()
            sess.run(None, {"input": x})
            ts.append((time.perf_counter() - t0) * 1000.0)
        rounds.append(np.array(ts))
    best = min(rounds, key=lambda r: float(np.percentile(r, 50)))
    return {"p50_ms": float(np.percentile(best, 50)), "p95_ms": float(np.percentile(best, 95)),
            "mean_ms": float(best.mean()), "runs": N_TIMED, "rounds": N_ROUNDS,
            "round_p50_ms": [float(np.percentile(r, 50)) for r in rounds]}


def load_arrays():
    split = pd.read_csv(ROOT / "data" / "inputs" / "trashnet_split.csv")
    part = {s: split[split["split"] == s].reset_index(drop=True) for s in ("train", "val", "test")}
    rng = np.random.default_rng(0)
    calib_files = part["train"]["file"].iloc[np.sort(rng.choice(len(part["train"]), size=N_CALIB, replace=False))]
    stack = lambda files: np.stack([preprocess(IMG / f) for f in files])  # noqa: E731
    return (stack(calib_files), stack(part["val"]["file"]), part["val"]["label"].to_numpy(),
            stack(part["test"]["file"]), part["test"]["label"].to_numpy())


def evaluate(path: Path, X, y, committed: bool) -> dict:
    s1 = session(path, 1)
    p = predict(s1, X).argmax(1)
    return {
        "file": f"models/{path.name}" if committed else None, "bytes": path.stat().st_size, "sha256": sha256(path),
        "accuracy": float((p == y).mean()), "macro_f1": macro_f1(y, p),
        "per_class_recall": {c: float(((p == k) & (y == k)).sum() / max((y == k).sum(), 1)) for k, c in enumerate(CLASSES)},
        "confusion": confusion(y, p),
        "latency_1_thread": latency(s1, X),
        "latency_4_threads": latency(session(path, 4), X),
    }, p


def main() -> int:
    import onnxruntime as ort
    from onnxruntime.quantization.shape_inference import quant_pre_process
    ART.mkdir(exist_ok=True)
    quant_pre_process(str(FP32), str(PRE))
    calib, Xv, yv, Xt, yt = load_arrays()
    fp32_val = accuracy(FP32, Xv, yv)
    # 1. full-model quantisation, three calibration methods
    full = {}
    for method in FULL_METHODS:
        out = ART / f"trashnet_int8_full_{method}.onnx"
        quantize(calib, out, method)
        full[method] = accuracy(out, Xv, yv)
        print("full", method, f"{full[method]:.4f}", flush=True)
    best_full = max(FULL_METHODS, key=lambda k: (full[k], -FULL_METHODS.index(k)))
    # 2. sensitivity scan, one block at a time
    blocks = weight_blocks()
    scan = {}
    tmp = ART / "trashnet_int8_scan.onnx"
    for name, nodes in blocks.items():
        quantize(calib, tmp, "minmax", nodes)
        scan[name] = accuracy(tmp, Xv, yv)
        print("scan", name, len(nodes), f"{scan[name]:.4f}", flush=True)
    sensitive = [b for b in blocks if fp32_val - scan[b] > SENS_DROP]
    # 3. mixed precision: quantise the weight operators of the non-sensitive blocks only
    keep = [n for b, nodes in blocks.items() if b not in sensitive for n in nodes]
    quantize(calib, INT8, "minmax", keep)
    mixed_val = accuracy(INT8, Xv, yv)
    print("mixed", f"{mixed_val:.4f}", "sensitive blocks", sensitive, flush=True)
    # 4. test metrics and latency
    res = {"test_images": int(len(yt)), "validation_images": int(len(yv)), "calibration_images": N_CALIB,
           "classes": list(CLASSES), "onnxruntime": ort.__version__, "cpu": cpu_name(),
           "python": platform.python_version(), "edge_hardware_tested": False,
           "selection_on_validation": {
               "fp32": fp32_val, "full_int8": full, "full_int8_best": best_full,
               "block_scan": scan, "sensitivity_threshold": SENS_DROP, "sensitive_blocks": sensitive,
               "weight_nodes_total": sum(len(v) for v in blocks.values()), "weight_nodes_quantised": len(keep),
               "mixed_int8": mixed_val},
           "models": {}}
    preds = {}
    for key, path, committed in (("fp32", FP32, True), ("int8_full", ART / f"trashnet_int8_full_{best_full}.onnx", False),
                                 ("int8", INT8, True)):
        res["models"][key], preds[key] = evaluate(path, Xt, yt, committed)
    res["models"]["int8"]["description"] = "mixed precision: Conv/Gemm of non-sensitive blocks in INT8, the rest FP32"
    res["models"]["int8_full"]["description"] = f"every supported operator in INT8, {best_full} calibration (not committed)"
    res["int8_fp32_prediction_agreement"] = float((preds["fp32"] == preds["int8"]).mean())
    res["size_ratio_int8_over_fp32"] = res["models"]["int8"]["bytes"] / res["models"]["fp32"]["bytes"]
    write_json(ROOT / "results" / "edge_model.json", res)
    # reference outputs on fixed synthetic inputs (for a replay test that needs no images)
    Z = np.random.default_rng(2026).standard_normal((4, 3, 224, 224)).astype(np.float32)
    ref = {"seed": 2026, "shape": [4, 3, 224, 224]}
    for name, path in (("fp32", FP32), ("int8", INT8)):
        ref[name] = {"sha256": sha256(path), "logits": predict(session(path, 1), Z).round(5).tolist()}
    write_json(ROOT / "results" / "edge_reference_outputs.json", ref)
    figure(res)
    for name in ("fp32", "int8_full", "int8"):
        m = res["models"][name]
        print(name, f"acc {m['accuracy']:.4f} f1 {m['macro_f1']:.4f} size {m['bytes'] / 1e6:.2f} MB",
              f"p50 {m['latency_1_thread']['p50_ms']:.2f} ms p95 {m['latency_1_thread']['p95_ms']:.2f} ms (1 thread)")
    return 0


def figure(res: dict) -> None:
    from wastecollect import plotstyle as ps
    import matplotlib.pyplot as plt
    ps.apply()
    names = [("fp32", "FP32"), ("int8_full", "INT8, all ops"), ("int8", "INT8, mixed")]
    col = {"fp32": ps.SERIES[0], "int8_full": ps.SERIES[1], "int8": ps.SERIES[2]}
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.0))
    for ax, (key, title, fmt) in zip(axes, [("accuracy", "Test accuracy", "{:.1%}"), ("bytes", "Model size (MB)", "{:.2f}"),
                                            ("p50", "CPU latency p50, whisker to p95 (ms, 1 thread)", "{:.1f}")]):
        for k, (name, _) in enumerate(names):
            m = res["models"][name]
            if key == "bytes":
                v = m["bytes"] / 1e6
            elif key == "p50":
                v = m["latency_1_thread"]["p50_ms"]
                ax.errorbar(k, v, yerr=[[0], [m["latency_1_thread"]["p95_ms"] - v]], color=ps.INK2, capsize=3, lw=1)
            else:
                v = m[key]
            ax.bar(k, v, 0.6, color=col[name])
            ax.text(k - 0.28 if key == "p50" else k, v, fmt.format(v), ha="left" if key == "p50" else "center", va="bottom",
                    fontsize=8, color=ps.INK2)
        ax.set_xticks(range(len(names)), [lab for _, lab in names], fontsize=8)
        ax.set_title(title, loc="left", fontsize=10)
    fig.suptitle("MobileNetV3-Small on TrashNet, ONNX Runtime on a laptop CPU (not edge hardware)", x=0.01, y=1.04,
                 ha="left", fontsize=11, fontweight="bold")
    ps.save(fig, ROOT / "figures" / "edge_fp32_vs_int8.png")


if __name__ == "__main__":
    sys.exit(main())
