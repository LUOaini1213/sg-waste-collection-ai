"""Fill the generated blocks of README.md from results/*.json.

Blocks are delimited by <!-- BEGIN:name --> and <!-- END:name -->. `--check` exits 1 when the
README on disk differs from what the results produce (a test calls this).

usage: python scripts/render_readme.py [--check]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
METHOD = {"persistence": "Last value", "seasonal_naive": "Seasonal naive (same weekday last week)",
          "mean_rate": "14-day mean fill rate", "lightgbm": "LightGBM"}
POLICY = {"fixed": "Fixed, 3x a week", "threshold": "Reactive (reading >= 70 %)", "forecast": "Forecast-based"}
SCEN = {"base": "Base (3 seeds)", "trnava_variability": "Trnava-like day-to-day variability", "low_fill_rate": "Fill rate 0.15 bin/day",
        "high_fill_rate": "Fill rate 0.30 bin/day", "bigger_truck": "Truck holds 80 bin volumes",
        "facility_in_town": "Depot and unloading point in town", "more_search": "200 ALNS iterations per day"}


def load(name):
    return json.loads((RES / name).read_text(encoding="utf-8"))


def pct(x, d=1):
    return f"{x:+.{d}f} %"


def block_routing() -> str:
    r = load("routing_benchmark.json")
    ref = json.loads((ROOT / "data" / "inputs" / "wcvrptw_reference.json").read_text(encoding="utf-8"))["instances"]
    out = ["| Instance | Stops | IFs | Kim 2006 | Benjamin 2011 | Markov 2016 | ALNS best | ALNS mean | Routes (ALNS / ref.) | Gap of best | Gap of mean |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for row in r["rows"]:
        x = ref[row["instance"]]
        mk = f"{x['markov2016'][0]:.2f}" if "markov2016" in x else "not reported"
        out.append(f"| {row['instance']} | {row['customers']} | {row['facilities']} | {x['kim2006'][0]:.1f} | {x['benjamin2011'][0]:.1f} | {mk} | "
                   f"{row['best']:.2f} | {row['mean']:.2f} | {row['best_routes']} / {row['reference_vehicles']} | "
                   f"{pct(row['gap_best_pct'], 2)} | {pct(row['gap_mean_pct'], 2)} |")
    first5 = [x for x in r["rows"] if x["customers"] < 1000]
    last5 = [x for x in r["rows"] if x["customers"] >= 1000]
    avg = lambda rows, k: sum(x[k] for x in rows) / len(rows)  # noqa: E731
    n_seeds = len(r["rows"][0]["seeds"])
    out.append("")
    out.append(f"Distances in miles; {r['rows'][0]['iters']:,} ALNS iterations, {n_seeds} seeds per instance; every run passed the "
               f"independent checker ({'yes' if all(x['all_feasible'] for x in r['rows']) else 'NO'}). Gaps are against the lowest "
               "published value that could be read in an accessible source: Markov et al. (2016) for the first five instances, "
               "Benjamin (2011) for the other five.")
    out.append(f"Mean gap of the best run: {pct(avg(first5, 'gap_best_pct'), 2)} on the five instances below 1,000 stops, "
               f"{pct(avg(last5, 'gap_best_pct'), 2)} on the five larger ones, {pct(r['mean_gap_best_pct'], 2)} over all ten "
               f"(mean of all runs: {pct(r['mean_gap_mean_pct'], 2)}). Against Kim et al. (2006) the best run is "
               f"{pct(avg(r['rows'], 'gap_vs_kim2006_pct'), 1)} on average.")
    more = [x["instance"] for x in r["rows"] if x["best_routes"] > x["reference_vehicles"]]
    if more:
        out.append(f"The objective here is distance only; on instances {', '.join(more)} the best run uses more routes than the reference solution.")
    return "\n".join(out)


def block_mip() -> str:
    m = load("routing_mip.json")
    out = ["| Group | Instances | MIP proved optimal | MIP routes pass the checker | ALNS = optimum (best of 5 seeds) | ALNS = optimum (all 5 seeds) | ALNS below a proven optimum | Largest gap, best seed | Largest gap, any seed |",
           "|---|---|---|---|---|---|---|---|---|"]
    names = {"synthetic": "SYNTHETIC, 5-7 stops, 2 IFs", "kim102": "First 6-10 stops of instance 102 (two capacities)"}
    for g in ("synthetic", "kim102"):
        rows = m.get(g, [])
        if not rows:
            continue
        opt = [x for x in rows if x["mip_optimal"]]
        gap = abs(max((x["alns_gap_pct"] for x in opt), default=0.0))
        worst = abs(max((x["alns_worst_seed_gap_pct"] for x in opt), default=0.0))
        out.append(f"| {names[g]} | {len(rows)} | {len(opt)} | {sum(x['mip_routes_checked'] for x in rows)} | "
                   f"{sum(x['alns_hits_optimum_any'] for x in opt)}/{len(opt)} | {sum(x['alns_hits_optimum_all'] for x in opt)}/{len(opt)} | "
                   f"{sum(x['alns_below_proven_optimum'] for x in opt)} | {gap:.2f} % | {worst:.2f} % |")
    out += ["", f"MIP time limit: {m['time_limit_s']:.0f} s per instance (solve times are in `results/routing_mip.json`; they depend on the machine)."]
    lim = [x for g in ("synthetic", "kim102") for x in m.get(g, []) if not x["mip_optimal"]]
    if lim:
        out += ["", f"Not proved optimal within the time limit of {m['time_limit_s']:.0f} s (ALNS compared with the MIP's best solution instead): "
                + ", ".join(f"{x['instance']} (MIP {x['mip_objective']:.3f}, bound {x['mip_bound']:.3f}, ALNS {x['alns_best']:.3f})" for x in lim) + "."]
    return "\n".join(out)


def block_forecast() -> str:
    f = load("forecast_backtest.json")
    o = f["overall"]
    out = [f"{f['n_forecasts']:,} forecasts, {f['n_bins']} bins, {len(f['origins'])} weekly origins from {f['origins'][0]} to {f['origins'][-1]} "
           "(the model is retrained at each origin on earlier data only). Errors in percentage points of fill level.", "",
           "| Method | MAE | RMSE | Bias | MAE 1 day | MAE 2 days | MAE 3 days | Full-bin (>= 80 %) precision / recall |", "|---|---|---|---|---|---|---|---|"]
    for m in ("persistence", "seasonal_naive", "mean_rate", "lightgbm"):
        h = [f["by_horizon"][str(k)][m]["mae"] for k in (1, 2, 3)]
        out.append(f"| {METHOD[m]} | {o[m]['mae']:.2f} | {o[m]['rmse']:.2f} | {o[m]['bias']:+.2f} | {h[0]:.2f} | {h[1]:.2f} | {h[2]:.2f} | "
                   f"{o[m]['full_precision']:.2f} / {o[m]['full_recall']:.2f} |")
    out += ["", "| Waste type | Forecasts | Last value | Seasonal naive | 14-day mean rate | LightGBM |", "|---|---|---|---|---|---|"]
    for t, v in f["by_type"].items():
        out.append(f"| {t} | {v['lightgbm']['n']:,} | {v['persistence']['mae']:.2f} | {v['seasonal_naive']['mae']:.2f} | {v['mean_rate']['mae']:.2f} | {v['lightgbm']['mae']:.2f} |")
    b = f["bootstrap"][0]
    verdict = "beats" if f["lightgbm_beats_best_baseline"] and b["ci95"][1] < 0 else "does not clearly beat"
    out += ["", f"LightGBM {verdict} the best baseline ({METHOD[f['best_baseline']].lower()}): MAE difference {b['mae_diff']:+.2f} pp, "
            f"95 % interval {b['ci95'][0]:+.2f} to {b['ci95'][1]:+.2f} (bootstrap over {b['n_bins']} bins)."]
    worse = [t for t, v in f["by_type"].items() if v["lightgbm"]["mae"] >= min(v[m]["mae"] for m in ("persistence", "seasonal_naive", "mean_rate"))]
    if worse:
        out.append(f"It is not better than the best baseline for: {', '.join(worse)}.")
    sn = [t for t, v in f["by_type"].items() if v["seasonal_naive"]["mae"] > v["persistence"]["mae"]]
    if sn:
        out.append(f"The weekly seasonal naive is worse than simply repeating the last value for: {', '.join(sn)}.")
    return "\n".join(out)


def block_casestudy() -> str:
    c = load("casestudy.json")
    out = ["| Scenario | Policy | Truck km | Bin lifts | Crew hours | Truck shifts | Max trucks/day | Overflow bin-days | Mean fill at lift | CO2e (kg) | km vs fixed | Lifts vs fixed | Hours vs fixed | Overflow vs fixed |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, sc in c["scenarios"].items():
        for p in ("fixed", "threshold", "forecast"):
            v = sc["policies"][p]
            ch = sc["change_vs_fixed_pct"].get(p)
            cells = [pct(ch["km"]), pct(ch["lifts"]), pct(ch["work_hours"]), pct(ch["overflow_bin_days"])] if ch else ["-", "-", "-", "-"]
            out.append(f"| {SCEN.get(name, name) if p == 'fixed' else ''} | {POLICY[p]} | {v['km']['mean']:.0f} | {v['lifts']['mean']:.0f} | {v['work_hours']['mean']:.0f} | "
                       f"{v['truck_days']['mean']:.1f} | {v['max_trucks_per_day']['max']:.0f} | {v['overflow_bin_days']['mean']:.0f} | "
                       f"{v['mean_fill_at_collection']['mean']:.2f} | {v['co2_kg']['mean']:.0f} | {' | '.join(cells)} |")
    base = c["scenarios"]["base"]["policies"]
    rng = lambda p, m: f"{base[p][m]['min']:.0f}-{base[p][m]['max']:.0f}"  # noqa: E731
    cfg = c["scenarios"]["base"]["config"]
    out += ["", f"Six simulated weeks after a {cfg['days_warmup']}-day warm-up, {cfg['alns_iters']} ALNS iterations per daily plan; all daily routes passed the checker "
            f"({'yes' if all(sc['policies'][p]['all_routes_feasible'] for sc in c['scenarios'].values() for p in sc['policies']) else 'NO'}). "
            f"Base-scenario range over the three seeds: km {rng('fixed', 'km')} (fixed) vs {rng('forecast', 'km')} (forecast-based); "
            f"overflow bin-days {rng('fixed', 'overflow_bin_days')} vs {rng('forecast', 'overflow_bin_days')}."]
    return "\n".join(out)


def block_edge() -> str:
    e = load("edge_model.json")
    t = load("edge_training.json")
    s = e["selection_on_validation"]
    label = {"fp32": "FP32", "int8_full": "INT8, every operator (not committed)", "int8": "INT8, mixed precision (committed)"}
    out = ["| Model | Test accuracy | Macro F1 | File size | p50 / p95 latency, 1 thread | p50 / p95 latency, 4 threads |", "|---|---|---|---|---|---|"]
    for k in ("fp32", "int8_full", "int8"):
        m = e["models"][k]
        out.append(f"| {label[k]} | {m['accuracy']:.1%} | {m['macro_f1']:.3f} | {m['bytes'] / 1e6:.2f} MB | "
                   f"{m['latency_1_thread']['p50_ms']:.1f} / {m['latency_1_thread']['p95_ms']:.1f} ms | "
                   f"{m['latency_4_threads']['p50_ms']:.1f} / {m['latency_4_threads']['p95_ms']:.1f} ms |")
    f, q, full = e["models"]["fp32"], e["models"]["int8"], e["models"]["int8_full"]
    speed = f["latency_1_thread"]["p50_ms"] / q["latency_1_thread"]["p50_ms"]
    out += ["", f"{e['test_images']} test images; calibration on {e['calibration_images']} training images; every choice below was made on the "
            f"{e['validation_images']} validation images (FP32 validation accuracy {s['fp32']:.1%}).", "",
            "- **Quantising every operator fails.** Validation accuracy "
            + ", ".join(f"{v:.1%} ({k})" for k, v in s["full_int8"].items())
            + f"; on the test split the best of these reaches {full['accuracy']:.1%}.",
            "- **Sensitivity scan.** Quantising the convolutions of one block at a time: "
            + ", ".join(f"{b} {v:.1%}" for b, v in s["block_scan"].items() if b in s["sensitive_blocks"])
            + f"; every other block stays at {min(v for b, v in s['block_scan'].items() if b not in s['sensitive_blocks']):.1%} or above.",
            f"- **Mixed precision.** Keeping {', '.join(s['sensitive_blocks'])} and the activation operators in FP32 and quantising "
            f"{s['weight_nodes_quantised']} of {s['weight_nodes_total']} convolution / linear nodes gives {s['mixed_int8']:.1%} on validation and "
            f"{q['accuracy']:.1%} on test ({(q['accuracy'] - f['accuracy']) * 100:+.1f} points against FP32), a file {f['bytes'] / q['bytes']:.1f}x smaller, "
            f"and {'no speed-up' if speed < 1.05 else f'a {speed:.2f}x speed-up'} on this CPU (p50 {f['latency_1_thread']['p50_ms']:.1f} ms FP32 vs "
            f"{q['latency_1_thread']['p50_ms']:.1f} ms with one thread; with four threads {f['latency_4_threads']['p50_ms']:.1f} ms vs {q['latency_4_threads']['p50_ms']:.1f} ms). The two models agree on {e['int8_fp32_prediction_agreement']:.1%} of test images.",
            "", "Per-class recall on test (FP32 / mixed INT8): "
            + ", ".join(f"{c} {f['per_class_recall'][c]:.2f} / {q['per_class_recall'][c]:.2f}" for c in e["classes"]) + ".",
            f"Latency: ONNX Runtime {e['onnxruntime']}, CPU execution provider, batch 1, best of three rounds of 300 runs, on {e['cpu']} "
            "(a laptop; **not measured on edge hardware**, where integer kernels may behave differently). "
            f"Training: {t['epochs']} epochs, {t['train_seconds'] / 60:.0f} min wall time on a {t['device']} shared with other jobs; "
            f"best validation accuracy {t['best_val_acc']:.1%} at epoch {t['best_epoch']}."]
    return "\n".join(out)


def block_tests() -> str:
    m = load("mutation_report.json")
    s = f"`python -m pytest -q` collects {m['tests_collected']} tests. Mutation testing (`python scripts/mutation_test.py`): {m['killed']}/{m['mutations']} planted defects killed"
    s += "." if not m["survived"] else f"; survivors: {m['survived']} (see results/mutation_report.json)."
    return s


def block_findings() -> str:
    r, f, c, e = load("routing_benchmark.json"), load("forecast_backtest.json"), load("casestudy.json"), load("edge_model.json")
    first5 = [x for x in r["rows"] if x["customers"] < 1000]
    last5 = [x for x in r["rows"] if x["customers"] >= 1000]
    avg = lambda rows, k: sum(x[k] for x in rows) / len(rows)  # noqa: E731
    o = f["overall"]
    b = c["scenarios"]["base"]
    ch = b["change_vs_fixed_pct"]["forecast"]
    town = c["scenarios"]["facility_in_town"]["change_vs_fixed_pct"]["forecast"]
    nbins = json.loads((ROOT / "data" / "inputs" / "sg_casestudy_meta.json").read_text(encoding="utf-8"))["bins_after_dedup"]
    if ch["lifts"] < -5 and ch["km"] > -1:
        verdict = "Under these assumptions collecting when needed saves handling, not distance."
    elif ch["km"] <= -1:
        verdict = "Under these assumptions collecting when needed saves both handling and distance."
    else:
        verdict = "Under these assumptions collecting when needed brings no clear saving."
    fp, q = e["models"]["fp32"], e["models"]["int8"]
    return "\n".join([
        f"- **Routing.** ALNS with unloading trips and a driver break on the ten Kim et al. (2006) instances: best of {len(r['rows'][0]['seeds'])} seeds is "
        f"{pct(avg(first5, 'gap_best_pct'), 1)} from the lowest published value on the five instances below 1,000 stops and {pct(avg(last5, 'gap_best_pct'), 1)} on the five larger ones "
        f"({pct(avg(r['rows'], 'gap_vs_kim2006_pct'), 1)} against the original 2006 results). It is not state of the art.",
        f"- **Forecasting.** On {f['n_bins']} real smart bins, LightGBM has an MAE of {o['lightgbm']['mae']:.1f} points of fill level against {o[f['best_baseline']]['mae']:.1f} for the best baseline "
        f"({METHOD[f['best_baseline']].lower()}) and {o['persistence']['mae']:.1f} for the last value.",
        f"- **Singapore case study (simulation).** For the {nbins} recycling bins of Bishan's HDB estates, forecast-based collection changes bin lifts by {pct(ch['lifts'], 0)}, "
        f"crew hours by {pct(ch['work_hours'], 0)}, overflow bin-days by {pct(ch['overflow_bin_days'], 0)} and truck km by {pct(ch['km'], 1)} against the three-times-a-week schedule "
        f"(truck km {pct(town['km'], 0)} when depot and unloading point are in town). {verdict}",
        f"- **Edge classifier.** MobileNetV3-Small on TrashNet: {fp['accuracy']:.1%} test accuracy in FP32. Static INT8 quantisation of the whole network "
        f"collapses to {e['models']['int8_full']['accuracy']:.1%}; a mixed-precision model ({len(e['selection_on_validation']['sensitive_blocks'])} sensitive blocks kept in FP32) reaches {q['accuracy']:.1%} at "
        f"{fp['bytes'] / 1e6:.1f} MB -> {q['bytes'] / 1e6:.1f} MB, with CPU p50 latency {fp['latency_1_thread']['p50_ms']:.1f} -> {q['latency_1_thread']['p50_ms']:.1f} ms "
        "(laptop CPU, 1 thread; no edge device was used).",
    ])


def block_inputs() -> str:
    import csv
    inp = ROOT / "data" / "inputs"
    meta = json.loads((inp / "sg_casestudy_meta.json").read_text(encoding="utf-8"))
    with open(inp / "fill_bins.csv", encoding="utf-8") as fh:
        bins = list(csv.DictReader(fh))
    days, n, stale, coll = set(), 0, 0, 0
    with open(inp / "fill_levels_daily.csv", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            days.add(row["date"])
            n += 1
            stale += int(row["stale"])
            coll += int(row["coll"])
    types = {}
    for b in bins:
        types[b["waste_type"]] = types.get(b["waste_type"], 0) + 1
    with open(inp / "trashnet_split.csv", encoding="utf-8") as fh:
        split = list(csv.DictReader(fh))
    return "\n".join([
        "Committed inputs (`data/inputs/`):", "",
        f"- Trnava smart bins: {len(bins)} bins ({', '.join(f'{v} {k}' for k, v in sorted(types.items()))}), {len(days)} days "
        f"({min(days)} to {max(days)}), {n:,} bin-days, {coll:,} logged collections, {100 * stale / n:.1f} % of bin-days flagged as stale.",
        f"- Bishan case study: {meta['bins_after_dedup']} recycling bins in HDB estates, {meta['bins_matched_to_hdb_block']} matched to an HDB block "
        f"({meta['town_bins_dwelling_units_total']:,} dwelling units in total), a {meta['bins_after_dedup'] + 2} x {meta['bins_after_dedup'] + 2} road distance and "
        f"travel-time matrix from an OSM graph of {meta['road_graph']['nodes']:,} nodes; bins snapped to the road network at {meta['mean_snap_m']:.0f} m on average.",
        f"- TrashNet split: {len(split):,} file names ({sum(r['split'] == 'train' for r in split):,} train, {sum(r['split'] == 'val' for r in split)} validation, {sum(r['split'] == 'test' for r in split)} test).",
        "- Published reference distances for the routing benchmark, cited Singapore facts (`sg_facts.json`).",
    ])


BLOCKS = {"inputs": block_inputs, "findings": block_findings, "routing": block_routing, "mip": block_mip, "forecast": block_forecast,
          "casestudy": block_casestudy, "edge": block_edge, "tests": block_tests}


def render(text: str) -> str:
    for name, fn in BLOCKS.items():
        pat = re.compile(rf"(<!-- BEGIN:{name} -->\n)(?:.*?\n)?(<!-- END:{name} -->)", re.S)
        if not pat.search(text):
            raise SystemExit(f"README block {name} missing")
        body = fn()
        text = pat.sub(lambda m: m.group(1) + body + "\n" + m.group(2), text)
    return text


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    path = ROOT / "README.md"
    old = path.read_text(encoding="utf-8")
    new = render(old)
    if args.check:
        if new != old:
            print("README.md is out of date; run python scripts/render_readme.py")
            return 1
        return 0
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(new)
    return 0


if __name__ == "__main__":
    sys.exit(main())
