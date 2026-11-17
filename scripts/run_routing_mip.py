"""Exact MIP (HiGHS) vs ALNS on small WCVRP-IF instances.

Two groups:
  synthetic  SYNTHETIC instances from wastecollect.routing.instance.synthetic (offline, replayed in CI)
  kim102     the first n customers of Kim et al.'s instance 102 (needs scripts/fetch_wcvrptw.py), with
             the original vehicle capacity and with capacity cut to 40 yd^3 so that several unloads
             are needed; derived instances are not committed, only the results
For every instance: MIP status, optimum and bound; the MIP routes re-checked by the independent
checker; ALNS (5 seeds x 300 iterations) best and per-seed distances.

usage: python scripts/run_routing_mip.py [--group synthetic|kim102|all] [--ci]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wastecollect.io import read_json, write_json  # noqa: E402
from wastecollect.routing.alns import solve  # noqa: E402
from wastecollect.routing.checker import evaluate  # noqa: E402
from wastecollect.routing.instance import parse_kim, subset, synthetic  # noqa: E402
from wastecollect.routing.mip import solve_mip  # noqa: E402

SYNTH = [(seed, n) for n in (5, 6, 7) for seed in range(4)]
SYNTH_CI = [(11, 5), (12, 5)]
KIM_N = (6, 8, 10)
ALNS_SEEDS = (1, 2, 3, 4, 5)
TIME_LIMIT = 900.0   # seconds per MIP; instances not solved in time are reported as not proved optimal


def check(inst, vehicles, time_limit, iters):
    r = solve_mip(inst, n_vehicles=vehicles, time_limit=time_limit)
    ev = evaluate(inst, r["routes"]) if r["routes"] else None
    alns = []
    for s in ALNS_SEEDS:
        best, _ = solve(inst, iters, s)
        e = evaluate(inst, best.as_lists())
        alns.append({"seed": s, "distance": e.distance, "feasible": e.feasible, "routes": e.n_routes})
    best_alns = min(a["distance"] for a in alns)
    row = {"instance": inst.name, "customers": inst.n_customers, "facilities": int(len(inst.facilities)),
           "capacity": inst.capacity, "vehicles_in_mip": vehicles, "mip_status": r["status"], "mip_optimal": r["optimal"],
           "mip_objective": r["objective"], "mip_bound": r["bound"], "mip_seconds": r["seconds"],
           "mip_routes_checked": bool(ev and ev.feasible), "mip_checked_distance": ev.distance if ev else None,
           "alns_best": best_alns, "alns_all_feasible": all(a["feasible"] for a in alns),
           "alns_hits_optimum_any": bool(r["optimal"] and abs(best_alns - r["objective"]) <= 1e-6),
           "alns_hits_optimum_all": bool(r["optimal"] and all(abs(a["distance"] - r["objective"]) <= 1e-6 for a in alns)),
           "alns_below_proven_optimum": bool(r["optimal"] and best_alns < r["objective"] - 1e-6),
           "alns_gap_pct": (100.0 * (best_alns - r["objective"]) / r["objective"]) if r["objective"] else None,
           "alns_worst_seed_gap_pct": (100.0 * (max(a["distance"] for a in alns) - r["objective"]) / r["objective"]) if r["objective"] else None,
           "mip_copies_per_facility": r["copies"],
           "alns": alns}
    print(f"{inst.name}: {r['status']} mip {r['objective']} alns {best_alns:.4f} checked={row['mip_routes_checked']}", flush=True)
    return row


def ci_rows() -> list:
    rows = [check(synthetic(s, n_customers=n, capacity=20.0), 2, 120, 100) for s, n in SYNTH_CI]
    for r in rows:
        r.pop("mip_seconds")  # wall time is machine-dependent
    return rows


def _job(spec):
    group, name, n, cap, vehicles = spec
    if group == "synthetic":
        inst = synthetic(name, n_customers=n, capacity=cap)
    else:
        full = parse_kim(ROOT / "data" / "external" / "wcvrptw" / "102_stop.txt")
        over = {} if cap is None else {"capacity": cap}
        inst = subset(full, full.customers[:n], name=name, **over)
    return group, check(inst, vehicles, TIME_LIMIT, 300)


def main(argv=None) -> int:
    from concurrent.futures import ProcessPoolExecutor
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="all", choices=["synthetic", "kim102", "all"])
    ap.add_argument("--ci", action="store_true", help="two tiny synthetic instances -> results/routing_mip_ci.json")
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args(argv)
    if args.ci:
        write_json(ROOT / "results" / "routing_mip_ci.json", {"rows": ci_rows()})
        return 0
    out_path = ROOT / "results" / "routing_mip.json"
    out = read_json(out_path) if out_path.exists() else {}
    jobs = []
    if args.group in ("synthetic", "all"):
        jobs += [("synthetic", s, n, 20.0, 3) for s, n in SYNTH]
    if args.group in ("kim102", "all"):
        if not (ROOT / "data" / "external" / "wcvrptw" / "102_stop.txt").exists():
            print("instance 102 missing; run scripts/fetch_wcvrptw.py")
        else:
            for n in KIM_N:
                jobs.append(("kim102", f"102-first{n}", n, None, 2))
                jobs.append(("kim102", f"102-first{n}-Q40", n, 40.0, 3))
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        results = list(ex.map(_job, jobs))
    for group in dict.fromkeys(g for g, _ in results):
        out[group] = [row for g, row in results if g == group]
    out["time_limit_s"] = TIME_LIMIT
    write_json(out_path, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
