"""Rebuild every committed result that depends only on committed inputs (offline).

Order: forecasting backtest (reduced + full), exact-MIP validation on synthetic instances,
Singapore case study (reduced + full), benchmark summary from the stored runs, KPI tables,
dashboard, README blocks. Not included, because they need downloads or a GPU:
  scripts/run_routing_benchmark.py (Kim et al. instances), scripts/run_routing_mip.py --group kim102,
  scripts/train_edge_model.py and scripts/quantize_edge_model.py (TrashNet images).
Takes roughly one and a half hours on a 4-core laptop.

usage: python scripts/run_all.py [--workers 3]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args(argv)
    steps = [
        ["scripts/run_forecast.py", "--ci"],
        ["scripts/run_forecast.py"],
        ["scripts/run_routing_mip.py", "--ci"],
        ["scripts/run_routing_mip.py", "--group", "synthetic"],
        ["scripts/run_casestudy.py", "--ci"],
        ["scripts/run_casestudy.py", "--workers", str(args.workers)],
        ["scripts/run_routing_benchmark.py", "--summary-only"],
        ["scripts/build_kpis.py"],
        ["scripts/render_readme.py"],
    ]
    for step in steps:
        print(">>", " ".join(step), flush=True)
        code = subprocess.run([sys.executable, *step], cwd=ROOT).returncode
        if code:
            return code
    return 0


if __name__ == "__main__":
    sys.exit(main())
