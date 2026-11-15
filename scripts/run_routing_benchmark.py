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
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", nargs="*", default=INSTANCES)
    ap.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    ap.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("--skip-existing", action="store_true", help="keep run files already written (resume)")
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
