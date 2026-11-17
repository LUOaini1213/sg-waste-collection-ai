"""Fill-level forecasting backtest on the committed Trnava inputs (offline).

usage: python scripts/run_forecast.py            full backtest -> results/forecast_backtest.json + figures
       python scripts/run_forecast.py --ci       reduced run (every 5th bin, 4 origins, 100 trees)
                                                 -> results/forecast_backtest_ci.json (checked by a test)
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wastecollect.forecast.backtest import (HORIZONS, METHODS, TYPES, Config, cluster_bootstrap_diff,  # noqa: E402
                                            metrics, run_backtest)
from wastecollect.io import write_json  # noqa: E402

INPUTS = ROOT / "data" / "inputs"


def ci_config() -> Config:
    bins = pd.read_csv(INPUTS / "fill_bins.csv")
    return Config(bins=tuple(int(b) for b in bins["bin"] if b % 5 == 1), max_origins=4, n_estimators=100)


def run(cfg: Config) -> tuple[dict, pd.DataFrame]:
    levels = pd.read_csv(INPUTS / "fill_levels_daily.csv")
    bins = pd.read_csv(INPUTS / "fill_bins.csv")
    pred, imps, info = run_backtest(levels, bins, cfg)
    overall = metrics(pred, cfg.full_threshold)
    best_base = min(("persistence", "seasonal_naive", "mean_rate"), key=lambda m: overall[m]["mae"])
    imp = pd.DataFrame(imps).mean()
    imp = (imp / imp.sum()).sort_values(ascending=False)
    res = {
        "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(cfg).items() if k != "bins"},
        "n_bins_config": None if cfg.bins is None else len(cfg.bins),
        "n_forecasts": int(len(pred)), "n_bins": int(pred["bin"].nunique()),
        "n_candidate_samples": info["n_samples_total"], "origins": info["origins"],
        "first_test_day": str(pred["date"].min()), "last_test_day": str(pred["date"].max()),
        "overall": overall,
        "by_horizon": {str(h): metrics(pred[pred["h"] == h], cfg.full_threshold) for h in HORIZONS},
        "by_type": {TYPES[t]: metrics(pred[pred["type"] == t], cfg.full_threshold)
                    for t in sorted(pred["type"].unique()) if (pred["type"] == t).sum() >= 50},
        "best_baseline": best_base,
        "lightgbm_beats_best_baseline": bool(overall["lightgbm"]["mae"] < overall[best_base]["mae"]),
        "bootstrap": [cluster_bootstrap_diff(pred, "lightgbm", best_base),
                      cluster_bootstrap_diff(pred, "lightgbm", "persistence"),
                      cluster_bootstrap_diff(pred, best_base, "persistence")],
        "feature_importance_gain_share": {k: float(v) for k, v in imp.items()},
    }
    return res, pred


def figures(res: dict, pred: pd.DataFrame) -> None:
    from wastecollect import plotstyle as ps
    import matplotlib.pyplot as plt
    ps.apply()
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    w = 0.19
    x = np.arange(len(HORIZONS))
    for k, m in enumerate(METHODS):
        vals = [res["by_horizon"][str(h)][m]["mae"] for h in HORIZONS]
        bars = ax.bar(x + (k - 1.5) * w, vals, w * 0.92, color=ps.METHOD_COLOR[m], label=ps.METHOD_LABEL[m])
        if m in ("lightgbm", "persistence"):
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.3, f"{v:.1f}", ha="center", va="bottom",
                        fontsize=8, color=ps.INK2)
    ax.set_xticks(x, [f"{h} day{'s' if h > 1 else ''} ahead" for h in HORIZONS])
    ax.set_ylabel("MAE (percentage points of fill)")
    ax.set_title("Fill-level forecast error, Trnava smart bins (no collection in between)", loc="left")
    ax.legend(ncol=2, fontsize=8, loc="upper left")
    ax.set_ylim(0, max(res["by_horizon"][str(h)]["persistence"]["mae"] for h in HORIZONS) * 1.25)
    ps.save(fig, ROOT / "figures" / "forecast_mae_by_horizon.png")
    # example bin trajectory: a paper or plastic bin with several logged collections in the window
    levels = pd.read_csv(INPUTS / "fill_levels_daily.csv")
    lv = levels.assign(date=pd.to_datetime(levels["date"]))
    lv = lv[(lv["date"] >= "2026-09-01") & (lv["date"] <= "2026-10-15")]
    ncoll = lv.groupby("bin")["coll"].sum()
    cand = pred[pred["type"].isin([TYPES.index("paper"), TYPES.index("plastic")]) & (pred["h"] == 1)]
    cand = cand[cand["date"].between("2026-09-01", "2026-10-15")]
    counts = cand.groupby("bin").size()
    counts = counts[counts.index.map(lambda b: 3 <= ncoll.get(b, 0) <= 8)]
    b = int(counts.sort_values(ascending=False, kind="stable").index[0])
    s = lv[lv["bin"] == b].copy()
    p1 = pred[(pred["bin"] == b) & (pred["h"] == 1)].copy()
    p1["tdate"] = pd.to_datetime(p1["date"]) + pd.Timedelta(days=1)
    p1 = p1[(p1["tdate"] >= "2026-09-01") & (p1["tdate"] <= "2026-10-15")]
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ax.plot(s["date"], s["eod"], color=ps.INK2, lw=1.6, label="Observed end-of-day fill")
    for m in ("mean_rate", "lightgbm"):
        ax.plot(p1["tdate"], p1[m], ls="none", marker="o", ms=4.5, color=ps.METHOD_COLOR[m],
                label=f"{ps.METHOD_LABEL[m]} (1 day ahead)")
    cd = s[s["coll"] > 0]["date"]
    for d in cd:
        ax.axvline(d, color=ps.MUTED, lw=0.9, ls=":", zorder=0)
    ax.set_ylabel("Fill level (%)")
    ax.set_ylim(0, 105)
    wt = TYPES[int(pred.loc[pred["bin"] == b, "type"].iloc[0])]
    ax.set_title(f"Example {wt} bin (id {b}); dotted lines mark logged collections", loc="left")
    ax.legend(fontsize=8, loc="upper left", ncol=3)
    fig.autofmt_xdate()
    ps.save(fig, ROOT / "figures" / "forecast_example_bin.png")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ci", action="store_true")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--figures-only", action="store_true", help="redraw figures from artifacts/forecast_pred.pkl")
    args = ap.parse_args(argv)
    cache = ROOT / "artifacts" / "forecast_pred.pkl"
    if args.figures_only:
        res = __import__("json").loads((ROOT / "results" / "forecast_backtest.json").read_text(encoding="utf-8"))
        figures(res, pd.read_pickle(cache))
        return 0
    cfg = ci_config() if args.ci else Config()
    cfg.threads = args.threads
    res, pred = run(cfg)
    if not args.ci:
        cache.parent.mkdir(exist_ok=True)
        pred.to_pickle(cache)
    out = ROOT / "results" / ("forecast_backtest_ci.json" if args.ci else "forecast_backtest.json")
    write_json(out, res)
    if not args.ci:
        figures(res, pred)
    o = res["overall"]
    print({m: round(o[m]["mae"], 3) for m in METHODS}, "->", out.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
