"""Mutation testing: plant one defect at a time in the core logic and require a test to fail.

Each mutation is an exact string replacement in one source file, applied to a temporary copy of
the repository; only the listed (fast) test files are run for it. A mutation counts as killed
only when pytest exits with code 1 (test failures). Collection errors, timeouts or a mutation
whose target text is missing make the script fail. Before the mutations, the unmutated copy and
a comment-only control must pass. The report goes to results/mutation_report.json.

usage: python scripts/mutation_test.py [--workers 3] [--only N]
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wastecollect.io import write_json  # noqa: E402

R, F, C, K, E = ("tests/test_routing.py", "tests/test_forecast.py", "tests/test_casestudy.py",
                 "tests/test_pipeline.py", "tests/test_edge.py")
RC, RR, RI, RA, RM = ("wastecollect/routing/checker.py", "wastecollect/routing/routes.py",
                      "wastecollect/routing/instance.py", "wastecollect/routing/alns.py", "wastecollect/routing/mip.py")
FB, CS, SQL, ED, IO = ("wastecollect/forecast/backtest.py", "wastecollect/casestudy/sim.py",
                       "wastecollect/pipeline/kpi.sql", "wastecollect/edge/data.py", "wastecollect/io.py")
EDGE_FAST = "tests/test_edge.py::test_preprocess_constant_colour_and_geometry tests/test_edge.py::test_macro_f1_and_confusion_hand_worked tests/test_edge.py::test_split_is_stratified_disjoint_and_deterministic"

# (file, old, new, tests, what the defect is)
MUTATIONS = [
    (RC, "if load > inst.capacity + TOL:", "if load > 2 * inst.capacity + TOL:", R, "checker: vehicle capacity doubled"),
    (RC, "            load = 0.0\n            unloads += 1", "            unloads += 1", R, "checker: load not reset at a facility"),
    (RC, "if not any(kind[v] == FACILITY for v in nodes[last_customer + 1:-1]):", "if False:", R, "checker: return to depot without unloading allowed"),
    (RC, "arrival = b + dur + float(T[i, j])", "arrival = b + float(T[i, j])", R, "checker: break (option A) takes no time"),
    (RC, "t = max(arrival, float(ready[j]))", "t = arrival", R, "checker: service may start before the window opens"),
    (RC, "if t > float(due[j]) + TOL:", "if False:", R, "checker: latest start time ignored"),
    (RC, "                if v in seen:", "                if False:", R, "checker: duplicate visits not detected"),
    (RC, "if total > inst.route_capacity + TOL:", "if False:", R, "checker: route (daily) capacity ignored"),
    (RC, "for k in range(len(nodes) - 1))\n    load = 0.0", "for k in range(len(nodes) - 2))\n    load = 0.0", R, "checker: last leg missing from the distance"),
    (RR, "    if b <= r2 + EPS:\n        return b + d\n    b = f if f > r1 else r1", "    if b <= r2 + EPS:\n        return b\n    b = f if f > r1 else r1", R, "search: break on arrival takes no time"),
    (RR, "                if cur > inst.capacity + EPS:\n                    trip_ok = False", "                if cur > inst.capacity + EPS:\n                    trip_ok = True", R, "search: trip overload accepted"),
    (RR, "    ok &= (col(\"tl\") + ud.q[None, :]) <= inst.capacity + EPS\n", "", R, "search: plain insertion ignores trip capacity"),
    (RR, "    cost = D[gi, U] + D[U, gj] - col(\"dij\")", "    cost = D[gi, U] + D[U, gj]", R, "search: insertion cost forgets the removed arc"),
    (RR, "ok &= (np.where(final, 0.0, pre) + uq <= inst.capacity + EPS) & (post <= inst.capacity + EPS) & route_ok", "ok &= (np.where(final, 0.0, pre) <= inst.capacity + EPS) & (post <= inst.capacity + EPS) & route_ok", R, "search: [stop, unload] insertion ignores the stop's load"),
    (RR, "        ok = np.maximum(ej, x1) <= zj + EPS", "        ok = np.maximum(ej, x1) <= zj + 3600.0", R, "search: one hour of lateness tolerated after an insertion"),
    (RR, "if kind[v] == FACILITY and kind[out[-1]] != CUSTOMER:", "if kind[v] == FACILITY and kind[out[-1]] == CUSTOMER:", R, "search: drops the wrong facility visits"),
    (RR, "            if load_a + load_b <= inst.capacity + EPS:", "            if load_a + load_b <= 0 * inst.capacity + EPS:", R, "search: trips never merged"),
    (RI, "feet = np.abs(x[:, None] - x[None, :]) + np.abs(y[:, None] - y[None, :])", "feet = np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :])", R, "parser: Euclidean instead of Manhattan distance"),
    (RI, "return float((v // 100) * 3600 + (v % 100) * 60)", "return float((v // 100) * 3600 + (v % 100) * 36)", R, "parser: HHMM minutes read as hundredths of an hour"),
    (RI, "time = dist / mph * 3600.0", "time = dist / mph * 60.0", R, "parser: travel time in minutes instead of seconds"),
    (RA, "q_hi = max(min(n, params.q_min), min(params.q_max_abs, int(math.ceil(params.xi * n))), 1)", "q_hi = max(1, min(params.q_max_abs, int(math.ceil(params.xi * n))))", R, "ALNS: removal size back to ceil(0.4 n) on tiny instances"),
    (RA, "rng = np.random.default_rng(seed)", "rng = np.random.default_rng()", R, "ALNS: seed ignored"),
    (RM, "if j == E and not is_fac[i]:", "if False and not is_fac[i]:", R, "MIP: may return to the depot loaded"),
    (RM, "h.addConstr(w[j, k] - w[i, k] - Q * x[i, j, k] >= qj - Q)", "h.addConstr(w[j, k] - w[i, k] - Q * x[i, j, k] >= -Q)", R, "MIP: load does not accumulate along a trip"),
    (RM, "min(inst.n_customers, 2 * int(math.floor(total / inst.capacity + 1e-9)) + n_vehicles)", "min(inst.n_customers, 4)", R, "MIP: at most four unloads per facility (the earlier cap)"),
    (FB, "    bad = (coll > 0) | stale | np.isnan(inc)", "    bad = stale | np.isnan(inc)", F, "forecast: increments on collection days kept"),
    (FB, "        bad |= inc <= -DROP_PP\n", "        pass\n", F, "forecast: unlogged emptyings kept as increments"),
    (FB, "    bad[:, 1:] |= stale[:, :-1]\n", "", F, "forecast: increment after a stale day kept"),
    (FB, "ok &= ~stale[:, seg].any(axis=1) & (coll[:, seg].sum(axis=1) == 0)", "ok &= ~stale[:, seg].any(axis=1)", F, "forecast: evaluation windows may contain a collection"),
    (FB, "                v = inc[:, t + k - 7]", "                v = inc[:, t + k - 1]", F, "forecast: seasonal naive uses yesterday (and future days) instead of last week"),
    (FB, "S[\"y_t\"].to_numpy() + S[\"h\"].to_numpy() * np.nan_to_num(", "S[\"y_t\"].to_numpy() + np.nan_to_num(", F, "forecast: mean-rate baseline ignores the horizon"),
    (FB, "        e = pred[m].to_numpy() - y\n", "        e = y - pred[m].to_numpy()\n", F, "forecast: bias sign flipped"),
    (FB, "tp = np.sum((pred[m] >= threshold) & (pred[\"target\"] >= threshold))", "tp = np.sum((pred[m] >= threshold) | (pred[\"target\"] >= threshold))", F, "forecast: true positives counted with OR"),
    (FB, "\"rmse\": float(np.sqrt(np.mean(e ** 2)))", "\"rmse\": float(np.mean(e ** 2))", F, "forecast: RMSE without the square root"),
    (CS, "            if dow != 6:  # no collection on Sundays", "            if True:  # no collection on Sundays", C, "simulation: collections on Sundays"),
    (CS, "over = np.maximum(level - 1.0, 0.0)", "over = np.maximum(level - 0.9, 0.0)", C, "simulation: overflow counted from 90 %"),
    (CS, "            level = np.minimum(level, 1.0)\n", "", C, "simulation: bins hold more than their volume"),
    (CS, "sel = reading >= cfg.threshold", "sel = reading >= cfg.threshold + 0.2", C, "simulation: reactive threshold shifted"),
    (CS, "ahead = 2 if dow == 5 else 1", "ahead = 1", C, "simulation: forecast policy forgets the Sunday gap"),
    (CS, " | (d - last_coll >= cfg.max_gap_days)", "", C, "simulation: maximum gap between collections not enforced"),
    (CS, "np.clip(units / units.mean(), *cfg.units_clip)", "(units / units.mean())", C, "simulation: fill-rate clip removed"),
    (CS, "\"co2_kg\": km * self.config[\"co2_kg_per_km\"]", "\"co2_kg\": km", C, "simulation: CO2 factor not applied"),
    (CS, "g[order[len(lon) // 2:]] = 1", "g[order[len(lon) // 2 + 1:]] = 1", C, "simulation: uneven fixed-schedule groups"),
    (CS, "if d.policy == pol and d.day >= measured_from]", "if d.policy == pol and d.day > measured_from]", C, "simulation: first measured day dropped"),
    (CS, "            level[sel] = 0.0\n", "", C, "simulation: collected bins are not emptied"),
    (SQL, "    FROM daily_ops d CROSS JOIN sim_meta m\n    WHERE d.day >= m.warmup_days\n    GROUP BY d.scenario, d.seed, d.policy", "    FROM daily_ops d CROSS JOIN sim_meta m\n    WHERE d.day >= 0\n    GROUP BY d.scenario, d.seed, d.policy", K, "KPI SQL: warm-up days included"),
    (SQL, "AVG(km) AS km,", "SUM(km) AS km,", K, "KPI SQL: km summed over seeds instead of averaged"),
    (SQL, "100.0 * (a.km - f.km) / NULLIF(f.km, 0)         AS km_change_pct", "100.0 * (f.km - a.km) / NULLIF(f.km, 0)         AS km_change_pct", K, "KPI SQL: change sign flipped"),
    (SQL, "AVG(CASE WHEN stale = 0 THEN eod END)                                  AS mean_level_pct", "AVG(eod)                                  AS mean_level_pct", K, "KPI SQL: stale sensor days included in the mean level"),
    (CS, "hours = (drive + sum(float(inst.service[v]) for r in lists for v in r)) / H", "hours = drive / H", C, "simulation: crew hours without lifting and unloading time"),
    (SQL, "7.0 * SUM(coll)", "SUM(coll)", K, "KPI SQL: collections per week not scaled to a week"),
    (SQL, "AVG(work_hours) AS work_hours", "MAX(work_hours) AS work_hours", K, "KPI SQL: crew hours take the worst seed"),
    (SQL, "SUM(d.km) * MAX(m.co2_kg_per_km)       AS co2_kg", "SUM(d.km)       AS co2_kg", K, "KPI SQL: CO2 factor not applied"),
    (ED, "    x = (x - MEAN) / STD\n", "    x = x - MEAN\n", EDGE_FAST, "edge: no division by the channel standard deviation"),
    (ED, "left = int(round((nw - size) / 2.0))", "left = 0", EDGE_FAST, "edge: crop taken from the left edge instead of the centre"),
    (ED, "f1.append(2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0)", "f1.append(tp / (tp + fp + fn) if tp + fp + fn else 0.0)", EDGE_FAST, "edge: Jaccard index reported as F1"),
    (ED, "n_tr = int(round(FRACTIONS[0] * len(sub)))", "n_tr = int(round(FRACTIONS[1] * len(sub)))", EDGE_FAST, "edge: training share swapped with the validation share"),
    (IO, ".replace(b\"\\r\\n\", b\"\\n\")", "", "tests/test_repo.py::test_sha256_lf_ignores_crlf", "io: CRLF not normalised before hashing"),
]
CONTROL = (RC, "TOL = 1e-6", "TOL = 1e-6  # control: comment only", R, "control (must pass)")
COPY = ("wastecollect", "tests", "scripts", "data/inputs", "results", "models", "dashboard", "pyproject.toml", "README.md",
        "requirements.txt", "docs")


def make_copy(dst: Path) -> None:
    for name in COPY:
        src = ROOT / name
        if not src.exists():
            continue
        if src.is_dir():
            shutil.copytree(src, dst / name, ignore=shutil.ignore_patterns("__pycache__", "runs"))
        else:
            (dst / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst / name)


def run_one(idx: int, mut, timeout: int) -> dict:
    file, old, new, tests, what = mut
    with tempfile.TemporaryDirectory(prefix="wc_mut_") as tmp:
        tmp = Path(tmp)
        make_copy(tmp)
        path = tmp / file
        text = path.read_text(encoding="utf-8")
        n = text.count(old)
        if n != 1:
            return {"id": idx, "file": file, "defect": what, "status": "invalid", "detail": f"target text found {n} times"}
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text.replace(old, new))
        t0 = time.perf_counter()
        try:
            p = subprocess.run([sys.executable, "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider", *tests.split()],
                               cwd=tmp, capture_output=True, text=True, timeout=timeout)
            lines = p.stdout.strip().splitlines() or [""]
            failed = [ln for ln in lines if ln.startswith("FAILED ")]
            code, tail = p.returncode, (failed[0].split(" - ")[0] if failed else lines[-1])
        except subprocess.TimeoutExpired:
            code, tail = -1, "timeout"
        status = {1: "killed", 0: "survived"}.get(code, "invalid")
        return {"id": idx, "file": file, "defect": what, "tests": tests, "status": status, "pytest_exit": code,
                "last_line": tail, "seconds": round(time.perf_counter() - t0, 1)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--only", type=int, nargs="*")
    args = ap.parse_args(argv)
    base = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--collect-only"], cwd=ROOT,
                          capture_output=True, text=True)
    n_tests = 0
    for line in base.stdout.splitlines():   # "tests/x.py::name" (-q) or "tests/x.py: 12" (-qq)
        if "::" in line:
            n_tests += 1
        elif line.startswith("tests") and line.rsplit(": ", 1)[-1].isdigit():
            n_tests += int(line.rsplit(": ", 1)[-1])
    control = run_one(0, CONTROL, args.timeout)
    if control["status"] != "survived":
        print("control run did not pass:", control)
        return 2
    muts = [(i + 1, m) for i, m in enumerate(MUTATIONS) if not args.only or i + 1 in args.only]
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        rows = list(ex.map(lambda im: run_one(im[0], im[1], args.timeout), muts))
    for r in rows:
        print(f"{r['id']:>3} {r['status']:<8} {r['defect']}  [{r.get('last_line', r.get('detail', ''))}]")
    killed = sum(r["status"] == "killed" for r in rows)
    report = {"tests_collected": n_tests, "mutations": len(rows), "killed": killed,
              "survived": [r["id"] for r in rows if r["status"] == "survived"],
              "invalid": [r["id"] for r in rows if r["status"] == "invalid"],
              "control": control["status"], "rows": [{k: v for k, v in r.items() if k != "seconds"} for r in rows]}
    if not args.only:
        write_json(ROOT / "results" / "mutation_report.json", report)
    print(f"killed {killed}/{len(rows)}; tests collected {n_tests}")
    return 0 if killed == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
