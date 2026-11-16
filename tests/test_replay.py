"""Re-run reduced versions of the pipeline and compare with the committed results.

The committed result files were produced on Windows 11 / Python 3.11 with the pinned
requirements; these tests re-compute them (also on Linux in CI). Stored values are rounded to
6 decimals, hence the absolute tolerance of 2e-6.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from conftest import assert_close

ROOT = Path(__file__).resolve().parents[1]
TOL = dict(rel=1e-9, abs_tol=2e-6)


def load(name):
    return json.loads((ROOT / "results" / name).read_text(encoding="utf-8"))


def clean(obj):
    """Round-trip through the repository's JSON writer conventions."""
    from wastecollect.io import _clean
    return json.loads(json.dumps(_clean(obj, 6)))


def test_forecast_reduced_backtest_matches_committed():
    import run_forecast
    res, pred = run_forecast.run(run_forecast.ci_config())
    want = load("forecast_backtest_ci.json")
    got = clean(res)
    assert got["n_forecasts"] == want["n_forecasts"] and got["origins"] == want["origins"]
    for block in ("overall", "by_horizon", "by_type"):
        for key, methods in (want[block].items() if block != "overall" else [("", want[block])]):
            g = got[block][key] if key else got[block]
            for m in ("persistence", "seasonal_naive", "mean_rate"):
                assert_close(g[m], methods[m], **TOL, path=f"{block}.{key}.{m}")
            # gradient boosting: allow tiny platform differences in tree building
            for k in ("mae", "rmse", "bias"):
                assert g["lightgbm"][k] == pytest.approx(methods["lightgbm"][k], abs=0.02), (block, key, k)
    assert got["best_baseline"] == want["best_baseline"]
    assert got["lightgbm_beats_best_baseline"] == want["lightgbm_beats_best_baseline"]


def test_casestudy_reduced_run_matches_committed():
    import run_casestudy
    assert_close(clean(run_casestudy.ci_result()), load("casestudy_ci.json"), **TOL)


def test_mip_validation_matches_committed():
    import run_routing_mip
    got = clean(run_routing_mip.ci_rows())
    want = load("routing_mip_ci.json")["rows"]
    assert_close(got, want, rel=1e-7, abs_tol=2e-6)
    for r in got:
        assert r["mip_optimal"] and r["mip_routes_checked"] and r["alns_all_feasible"]
        assert not r["alns_below_proven_optimum"]


def test_kpi_tables_match_committed():
    from wastecollect.pipeline.kpis import TABLES, raw_from_repo, run_kpis
    kpi = run_kpis(raw_from_repo(ROOT))
    for name in TABLES:
        want = pd.read_csv(ROOT / "results" / "kpi" / f"{name}.csv")
        got = kpi[name]
        assert list(got.columns) == list(want.columns) and len(got) == len(want), name
        for c in want.columns:
            if pd.api.types.is_numeric_dtype(want[c]) and not pd.api.types.is_bool_dtype(want[c]):
                g = pd.to_numeric(got[c]).to_numpy(dtype=float)
                w = want[c].to_numpy(dtype=float)
                for a, b in zip(g, w):
                    assert (pd.isna(a) and pd.isna(b)) or a == pytest.approx(b, rel=1e-5, abs=1e-9), (name, c)
            else:
                assert [str(v) for v in got[c]] == [str(v) for v in want[c]], (name, c)


def test_dashboard_is_current_and_self_contained():
    from wastecollect.pipeline.dashboard import build
    from wastecollect.pipeline.kpis import raw_from_repo, run_kpis
    html = (ROOT / "dashboard" / "index.html").read_text(encoding="utf-8")
    assert build(ROOT, run_kpis(raw_from_repo(ROOT))) == html
    assert html.count("<script src=") == 1 and "cdn.plot.ly/plotly-" in html
    for div in ("scen-km", "scen-overflow_bin_days", "weekly-overflow", "forecast-mae", "trnava-map"):
        assert f'id="{div}"' in html
