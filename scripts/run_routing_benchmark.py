"""ALNS on the ten Kim et al. (2006) instances; compare with published values.

Needs the instance files: run scripts/fetch_wcvrptw.py first (they are not redistributed).
Each (instance, seed) run is written to results/routing/runs/ as soon as it finishes, then
results/routing_benchmark.json is rebuilt from all run files.

usage: python scripts/run_routing_benchmark.py [--instances 102 277] [--seeds 1 2 3]
       [--iters N] [--workers 4] [--summary-only]
"""
from __future__ import annotations

import argparse
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wastecollect.io import read_json, write_json  # noqa: E402

INSTANCES = ["102", "277", "335", "444", "804", "1051", "1351", "1599", "1932", "2100"]
RUNS = ROOT / "results" / "routing" / "runs"
DATA = ROOT / "data" / "external" / "wcvrptw"
DEFAULT_ITERS = 10000
SEEDS = [1, 2, 3]


def run_one(name: str, seed: int, iters: int) -> dict:
    from wastecollect.routing.alns import solve
    from wastecollect.routing.checker import evaluate
    from wastecollect.routing.instance import parse_kim
    inst = parse_kim(DATA / f"{name}_stop.txt")
    t0 = time.perf_counter()
    best, info = solve(inst, iters, seed)
    ev = evaluate(inst, best.as_lists())
    ids = inst.ids
    out = {
        "instance": name, "seed": seed, "iters": iters, "iters_used": info["iters_used"],
        "distance": ev.distance, "n_routes": ev.n_routes, "n_unloads": ev.n_unloads,
        "feasible": ev.feasible, "violations": ev.violations[:10],
        "initial_distance": info["initial_distance"], "seconds": time.perf_counter() - t0,
        "history": info["history"], "destroy_usage": info["destroy_usage"], "repair_usage": info["repair_usage"],
        "routes_stop_ids": [[ids[v] for v in r] for r in best.as_lists()],
    }
    write_json(RUNS / f"{name}_s{seed}.json", out)
    return out


def summarise() -> dict:
    ref = read_json(ROOT / "data" / "inputs" / "wcvrptw_reference.json")
    rows = []
    for name in INSTANCES:
        runs = [read_json(p) for p in sorted(RUNS.glob(f"{name}_s*.json"))]
        if not runs:
            continue
        r = ref["instances"][name]
        dists = [x["distance"] for x in runs]
        best = min(runs, key=lambda x: (x["distance"], x["seed"]))
        cands = [(r["benjamin2011"][0], "Benjamin 2011")]
        if "literature_bks_2016" in r:
            cands.append((r["literature_bks_2016"], "BKS listed by Markov et al. 2016"))
        if "markov2016" in r:
            cands.append((r["markov2016"][0], "Markov et al. 2016"))
        ref_val, ref_src = min(cands)
        rows.append({
            "instance": name, "customers": r["customers"], "facilities": r["facilities"],
            "seeds": [x["seed"] for x in runs], "iters": runs[0]["iters"],
            "all_feasible": all(x["feasible"] for x in runs),
            "best": best["distance"], "mean": sum(dists) / len(dists), "worst": max(dists),
            "best_routes": best["n_routes"], "best_unloads": best["n_unloads"],
            "initial_nn": best["initial_distance"],
            "kim2006": r["kim2006"][0], "benjamin2011": r["benjamin2011"][0],
            "reference": ref_val, "reference_source": ref_src,
            "reference_vehicles": (r["markov2016"][1] if ref_src.startswith("Markov et al.") else r["benjamin2011"][1]),
            "gap_best_pct": 100.0 * (best["distance"] - ref_val) / ref_val,
            "gap_mean_pct": 100.0 * (sum(dists) / len(dists) - ref_val) / ref_val,
            "gap_vs_kim2006_pct": 100.0 * (best["distance"] - r["kim2006"][0]) / r["kim2006"][0],
            "seconds_mean": sum(x["seconds"] for x in runs) / len(runs),
        })
    out = {"rows": rows,
           "n_instances": len(rows),
           "mean_gap_best_pct": sum(x["gap_best_pct"] for x in rows) / len(rows) if rows else None,
           "mean_gap_mean_pct": sum(x["gap_mean_pct"] for x in rows) / len(rows) if rows else None,
           "machine": {"python": platform.python_version(), "platform": platform.platform()},
           "note": ref["not_verified"]}
    write_json(ROOT / "results" / "routing_benchmark.json", out)
    figures(out)
    return out


def figures(out: dict) -> None:
    import numpy as np
    from wastecollect import plotstyle as ps
    import matplotlib.pyplot as plt
    ps.apply()
    rows = out["rows"]
    if not rows:
        return
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(8.2, 3.4))
    ax.bar(x - 0.2, [r["gap_best_pct"] for r in rows], 0.38, color=ps.SERIES[0], label="Best of 3 seeds")
    ax.bar(x + 0.2, [r["gap_mean_pct"] for r in rows], 0.38, color=ps.SERIES[1], label="Mean of 3 seeds")
    for k, r in enumerate(rows):
        ax.text(k - 0.2, r["gap_best_pct"] + (0.25 if r["gap_best_pct"] >= 0 else -0.25), f"{r['gap_best_pct']:+.1f}",
                ha="center", va="bottom" if r["gap_best_pct"] >= 0 else "top", fontsize=8, color=ps.INK2)
    ax.axhline(0, color=ps.AXIS, lw=1)
    ax.set_xticks(x, [f"{r['instance']}" for r in rows])
    ax.set_xlabel("Instance (number of stops in the file)")
    ax.set_ylabel("Gap to lowest published distance (%)")
    ax.set_title("ALNS vs published results, Kim et al. (2006) waste-collection instances", loc="left")
    ax.legend(fontsize=8, loc="upper left")
    ps.save(fig, ROOT / "figures" / "routing_gap.png")
    path = DATA / "102_stop.txt"
    run = RUNS / "102_s1.json"
    if not (path.exists() and run.exists()):
        return
    from wastecollect.routing.instance import FACILITY, parse_kim
    inst = parse_kim(path)
    best = min((read_json(p) for p in sorted(RUNS.glob("102_s*.json"))), key=lambda r: (r["distance"], r["seed"]))
    pos = {sid: k for k, sid in enumerate(inst.ids)}
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for k, route in enumerate(best["routes_stop_ids"]):
        idx = [pos[s] for s in route]
        xs, ys = inst.x[idx] / 5280.0, inst.y[idx] / 5280.0
        ax.step(xs, ys, where="post", color=ps.SERIES[k % 8], lw=1.2, label=f"Route {k + 1}")
        ax.plot(xs[1:-1], ys[1:-1], ls="none", marker="o", ms=3.5, color=ps.SERIES[k % 8])
    fac = inst.kind == FACILITY
    ax.plot(inst.x[fac] / 5280.0, inst.y[fac] / 5280.0, ls="none", marker="s", ms=9, color=ps.INK, label="Disposal facility")
    ax.plot(inst.x[0] / 5280.0, inst.y[0] / 5280.0, ls="none", marker="*", ms=14, color=ps.INK2, label="Depot")
    ax.set_aspect("equal")
    ax.ticklabel_format(useOffset=False, style="plain")
    ax.set_xlabel("x (miles)")
    ax.set_ylabel("y (miles)")
    ax.set_title(f"Instance 102: best ALNS run, {best['distance']:.1f} miles, {best['n_routes']} routes (Manhattan legs)", loc="left")
    ax.legend(fontsize=8, loc="best")
    ps.save(fig, ROOT / "figures" / "routing_102_routes.png")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", nargs="*", default=INSTANCES)
    ap.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    ap.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("--skip-existing", action="store_true", help="keep run files already written (continue an interrupted run)")
    args = ap.parse_args(argv)
    if not args.summary_only:
        missing = [n for n in args.instances if not (DATA / f"{n}_stop.txt").exists()]
        if missing:
            print("missing instance files; run scripts/fetch_wcvrptw.py first:", missing)
            return 2
        jobs = [(n, s) for n in sorted(args.instances, key=int, reverse=True) for s in args.seeds
                if not (args.skip_existing and (RUNS / f"{n}_s{s}.json").exists()
                        and read_json(RUNS / f"{n}_s{s}.json").get("iters") == args.iters)]
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(run_one, n, s, args.iters): (n, s) for n, s in jobs}
            for f in as_completed(futs):
                r = f.result()
                print(f"{r['instance']:>5} seed {r['seed']}: {r['distance']:.2f} mi, {r['n_routes']} routes, "
                      f"feasible={r['feasible']}, {r['seconds']:.0f}s", flush=True)
    s = summarise()
    for row in s["rows"]:
        print(f"{row['instance']:>5} best {row['best']:.2f} mean {row['mean']:.2f} ref {row['reference']:.2f} "
              f"gap {row['gap_best_pct']:+.2f}% / {row['gap_mean_pct']:+.2f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
