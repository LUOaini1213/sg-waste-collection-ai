"""Singapore desk study: fixed schedule vs "collect when needed" for Bishan's HDB recycling bins.

Reads the committed inputs (data/inputs/sg_casestudy_*, fill_levels_daily.csv) and writes
results/casestudy.json, results/casestudy_daily.csv and two figures. Offline.

usage: python scripts/run_casestudy.py [--workers 4]
       python scripts/run_casestudy.py --ci     (reduced: 7 warm-up + 3 measured days, 20 ALNS
                                                  iterations, base scenario, seed 1 -> results/casestudy_ci.json)
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wastecollect.casestudy.sim import SimConfig, simulate  # noqa: E402
from wastecollect.io import write_csv, write_json  # noqa: E402

INPUTS = ROOT / "data" / "inputs"
POLICIES = ("fixed", "threshold", "forecast")
METRICS = ("km", "lifts", "truck_days", "max_trucks_per_day", "unloads", "overflow_bin_days",
           "overflow_volume", "mean_fill_at_collection", "co2_kg", "loads_collected", "work_hours")


def load_inputs():
    nodes = pd.read_csv(INPUTS / "sg_casestudy_nodes.csv")
    D = pd.read_csv(INPUTS / "sg_casestudy_dist_km.csv").to_numpy()
    T = pd.read_csv(INPUTS / "sg_casestudy_time_s.csv").to_numpy()
    return nodes, D, T


def trnava_shape() -> dict:
    """Weekday shape and day-to-day variability of deposits in Trnava paper/plastic bins (ILLUSTRATIVE)."""
    from wastecollect.forecast.backtest import Config, daily_increments, load_panel
    levels = pd.read_csv(INPUTS / "fill_levels_daily.csv")
    bins = pd.read_csv(INPUTS / "fill_bins.csv")
    keep = tuple(bins.loc[bins["waste_type"].isin(["paper", "plastic"]), "bin"].astype(int))
    P = load_panel(levels, bins, Config(bins=keep))
    inc = daily_increments(P)
    dow = pd.to_datetime(P["days"]).day_of_week.to_numpy()
    with np.errstate(invalid="ignore"):
        by_dow = np.array([np.nanmean(inc[:, dow == k]) for k in range(7)])
        mu = np.nanmean(inc, axis=1)
        sd = np.nanstd(inc, axis=1)
        n = np.sum(~np.isnan(inc), axis=1)
    ok = (mu > 0.5) & (n >= 30)
    cv = float(np.median(sd[ok] / mu[ok]))
    return {"dow_factors": [float(v) for v in by_dow / by_dow.mean()], "noise_cv": cv,
            "n_bins": int(ok.sum()), "mean_daily_increment_pp": float(np.nanmean(mu[ok]))}


def scenarios(shape: dict) -> dict:
    # weekday shape from Trnava (ILLUSTRATIVE); day-to-day CV 0.5 is an ASSUMPTION because the Trnava
    # increments also contain sensor noise (their CV is run as a sensitivity case)
    base = SimConfig(dow_factors=tuple(shape["dow_factors"]), noise_cv=0.5, alns_iters=60)
    return {
        "base": (base, [1, 2, 3], None),
        "trnava_variability": (replace(base, noise_cv=shape["noise_cv"]), [1], None),
        "low_fill_rate": (replace(base, base_rate=0.15), [1], None),
        "high_fill_rate": (replace(base, base_rate=0.30), [1], None),
        "bigger_truck": (replace(base, truck_capacity=80.0), [1], None),
        "facility_in_town": (base, [1], "central_bin"),
        "more_search": (replace(base, alns_iters=200), [1], None),
    }


def run_job(args):
    name, cfg, seed, facility_mode = args
    nodes, D, T = load_inputs()
    b = nodes[nodes["kind"] == "bin"]
    nd = {"depot": 0, "facility": 1, "bins": b["idx"].to_numpy()}
    if facility_mode == "central_bin":
        c = np.array([b["lat"].mean(), b["lon"].mean()])
        k = int(np.argmin((b["lat"] - c[0]) ** 2 + (b["lon"] - c[1]) ** 2))
        nd["depot"] = nd["facility"] = int(b["idx"].iloc[k])
    cfg = replace(cfg, seed=seed)
    res = simulate(nd, D, T, b["dwelling_units"].to_numpy(dtype=float), b["lon"].to_numpy(), cfg)
    days = pd.DataFrame([asdict(d) for d in res.days])
    days.insert(0, "scenario", name)
    days.insert(1, "seed", seed)
    return name, seed, res.summary(cfg.days_warmup), days, asdict(cfg)


def aggregate(results) -> dict:
    out = {}
    for name in dict.fromkeys(r[0] for r in results):
        runs = [r for r in results if r[0] == name]
        pol = {}
        for p in POLICIES:
            vals = {m: [r[2][p][m] for r in runs] for m in METRICS}
            pol[p] = {m: {"mean": float(np.mean(v)), "min": float(np.min(v)), "max": float(np.max(v))} for m, v in vals.items()}
            pol[p]["all_routes_feasible"] = all(r[2][p]["all_routes_feasible"] for r in runs)
        rel = {p: {m: 100.0 * (pol[p][m]["mean"] - pol["fixed"][m]["mean"]) / pol["fixed"][m]["mean"]
                   if pol["fixed"][m]["mean"] else None for m in ("km", "lifts", "truck_days", "overflow_bin_days", "co2_kg", "work_hours")}
               for p in POLICIES if p != "fixed"}
        out[name] = {"seeds": [r[1] for r in runs], "config": runs[0][4], "policies": pol, "change_vs_fixed_pct": rel}
    return out


def figures(res: dict, nodes: pd.DataFrame, daily: pd.DataFrame) -> None:
    from matplotlib.colors import LinearSegmentedColormap
    from wastecollect import plotstyle as ps
    import matplotlib.pyplot as plt
    ps.apply()
    colors = {"fixed": ps.SERIES[0], "threshold": ps.SERIES[1], "forecast": ps.SERIES[2]}
    labels = {"fixed": "Fixed, 3x a week", "threshold": "Reactive, reading >= 70 %", "forecast": "Forecast-based"}
    base = res["scenarios"]["base"]["policies"]
    fig, axes = plt.subplots(1, 4, figsize=(10.5, 3.3))
    for ax, (m, title) in zip(axes, [("km", "Truck km"), ("lifts", "Bin lifts"),
                                     ("overflow_bin_days", "Overflow bin-days"), ("work_hours", "Crew hours")]):
        for k, p in enumerate(POLICIES):
            v = base[p][m]
            ax.bar(k, v["mean"], 0.7, color=colors[p], label=labels[p])
            ax.errorbar(k, v["mean"], yerr=[[v["mean"] - v["min"]], [v["max"] - v["mean"]]], color=ps.INK2,
                        capsize=3, lw=1)
            ax.text(k, v["max"] * 1.02, f"{v['mean']:.0f}", ha="center", va="bottom", fontsize=8, color=ps.INK2)
        ax.set_xticks([])
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_ylim(0, max(base[p][m]["max"] for p in POLICIES) * 1.18)
    handles, lab = axes[0].get_legend_handles_labels()
    fig.legend(handles, lab, fontsize=9, loc="lower left", bbox_to_anchor=(0.04, -0.06), ncol=3)
    fig.suptitle("Bishan HDB recycling bins, six simulated weeks, base scenario (mean of 3 seeds; whiskers = min-max)",
                 x=0.045, y=1.04, ha="left", fontsize=11, fontweight="bold")
    fig.subplots_adjust(wspace=0.38)
    ps.save(fig, ROOT / "figures" / "casestudy_kpis.png")
    b = nodes[nodes["kind"] == "bin"]
    cmap = LinearSegmentedColormap.from_list("blues", ["#b7d3f6", "#5598e7", "#256abf", "#0d366b"])
    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    sc = ax.scatter(b["lon"], b["lat"], c=b["dwelling_units"], cmap=cmap, s=22, edgecolors=ps.SURFACE, linewidths=0.5)
    cb = fig.colorbar(sc, ax=ax, shrink=0.8)
    cb.set_label("Dwelling units of the block")
    cb.outline.set_visible(False)
    ax.ticklabel_format(useOffset=False, style="plain")
    ax.set_aspect(1.0 / np.cos(np.radians(float(b["lat"].mean()))))
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"{len(b)} recycling bins in Bishan HDB estates (NEA layer)", loc="left")
    ax.text(0.0, -0.2, "Unloading point used in the model: about 16 km north-west by road (Senoko).", transform=ax.transAxes,
            fontsize=8, color=ps.INK2)
    ps.save(fig, ROOT / "figures" / "casestudy_bins_map.png")
    # scenario comparison: change of the forecast-based policy against the fixed schedule
    names = list(res["scenarios"])
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), sharey=True)
    ypos = np.arange(len(names))[::-1]
    for ax, (m, title) in zip(axes, [("km", "Truck km"), ("work_hours", "Crew hours"), ("overflow_bin_days", "Overflow bin-days")]):
        vals = [res["scenarios"][n]["change_vs_fixed_pct"]["forecast"][m] for n in names]
        ax.barh(ypos, vals, 0.6, color=colors["forecast"])
        ax.axvline(0, color=ps.AXIS, lw=1)
        for y, v in zip(ypos, vals):
            ax.text(v + (1.5 if v >= 0 else -1.5), y, f"{v:+.0f} %", va="center", ha="left" if v >= 0 else "right",
                    fontsize=8, color=ps.INK2)
        lim = max(abs(v) for v in vals) * 1.35 + 5
        ax.set_xlim(-lim, lim)
        ax.set_title(title, loc="left", fontsize=10)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(ypos, [n.replace("_", " ") for n in names])
    fig.suptitle("Forecast-based collection vs the fixed schedule, change per scenario", x=0.045, y=1.03, ha="left",
                 fontsize=11, fontweight="bold")
    ps.save(fig, ROOT / "figures" / "casestudy_scenarios.png")


def ci_result() -> dict:
    """Reduced run replayed by tests: base scenario, seed 1, 7 warm-up + 3 measured days, 20 ALNS iterations."""
    shape = trnava_shape()
    cfg = replace(scenarios(shape)["base"][0], days_warmup=7, days_measured=3, alns_iters=20)
    return {"trnava_shape": shape, "scenarios": aggregate([run_job(("base", cfg, 1, None))])}


def plausibility() -> dict:
    """Upper bound on recyclables per HDB dwelling from published statistics (see data/inputs/sg_facts.json)."""
    import json
    facts = {f["id"]: f for f in json.loads((INPUTS / "sg_facts.json").read_text(encoding="utf-8"))["facts"]}
    disposed = facts["domestic_waste_disposed_2025_mt"]["value"] * 1e9            # kg per year
    rate = facts["domestic_recycling_rate_2025"]["value"]
    recycled = disposed * rate / (1.0 - rate)
    units = facts["hdb_dwelling_units_total"]["value"]
    return {"domestic_recycled_t_per_day": recycled / 365.0 / 1000.0,
            "upper_bound_kg_per_hdb_dwelling_day": recycled / 365.0 / units,
            "note": "all domestic recycling channels divided by HDB dwellings only: an upper bound for blue bins"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ci", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args(argv)
    if args.ci:
        write_json(ROOT / "results" / "casestudy_ci.json", ci_result())
        return 0
    shape = trnava_shape()
    jobs = [(name, cfg, s, fm) for name, (cfg, seeds, fm) in scenarios(shape).items() for s in seeds]
    if args.workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            results = list(ex.map(run_job, jobs))
    else:
        results = [run_job(j) for j in jobs]
    res = {"trnava_shape": shape, "plausibility": plausibility(), "scenarios": aggregate(results)}
    write_json(ROOT / "results" / "casestudy.json", res)
    daily = pd.concat([r[3] for r in results], ignore_index=True)
    write_csv(ROOT / "results" / "casestudy_daily.csv", daily, float_format="%.4f")
    nodes, _, _ = load_inputs()
    figures(res, nodes, daily)
    base = res["scenarios"]["base"]
    for p in POLICIES:
        print(p, {m: round(base["policies"][p][m]["mean"], 1) for m in ("km", "lifts", "overflow_bin_days", "work_hours")})
    return 0


if __name__ == "__main__":
    sys.exit(main())
