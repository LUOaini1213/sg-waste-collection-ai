"""DuckDB KPI tables (results/kpi/*.csv) and the static dashboard (dashboard/index.html). Offline.

usage: python scripts/build_kpis.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wastecollect.io import write_csv, write_text  # noqa: E402
from wastecollect.pipeline.dashboard import build  # noqa: E402
from wastecollect.pipeline.kpis import raw_from_repo, run_kpis  # noqa: E402


def main() -> int:
    kpi = run_kpis(raw_from_repo(ROOT))
    for name, df in kpi.items():
        write_csv(ROOT / "results" / "kpi" / f"{name}.csv", df, float_format="%.6g")
    write_text(ROOT / "dashboard" / "index.html", build(ROOT, kpi))
    print({k: len(v) for k, v in kpi.items()}, "-> results/kpi/, dashboard/index.html")
    return 0


if __name__ == "__main__":
    sys.exit(main())
