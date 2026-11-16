"""KPI SQL on a hand-worked toy dataset (every expected number computed by hand)."""
from __future__ import annotations

import pandas as pd
import pytest

from wastecollect.pipeline.kpis import run_kpis, statements


def toy_raw():
    rows = []
    # scenario, seed, policy, day, km, bins, routes, overflow_bins, overflow_volume, load
    data = [
        ("fixed", 1, 0, 99, 9, 1, 5, 2.0, 9.0),      # warm-up day: must be ignored
        ("fixed", 1, 2, 10, 5, 1, 1, 0.5, 3.0), ("fixed", 1, 3, 20, 5, 2, 0, 0.0, 4.0),
        ("forecast", 1, 2, 8, 3, 1, 0, 0.0, 2.5), ("forecast", 1, 3, 12, 2, 1, 1, 0.2, 2.0),
        ("fixed", 2, 2, 14, 6, 1, 2, 1.0, 3.5), ("fixed", 2, 3, 16, 4, 1, 0, 0.0, 3.0),
        ("forecast", 2, 2, 6, 2, 1, 0, 0.0, 1.6), ("forecast", 2, 3, 10, 2, 1, 0, 0.0, 1.8),
    ]
    for pol, seed, day, km, bins, routes, ob, ov, load in data:
        rows.append({"scenario": "base", "seed": seed, "policy": pol, "day": day, "dow": day % 7, "bins": bins,
                     "load": load, "routes": routes, "unloads": routes, "km": float(km), "feasible": True,
                     "overflow_bins": ob, "overflow_volume": ov, "mean_fill_at_collection": 0.5})
    fill = pd.DataFrame({"bin": [1] * 4 + [2] * 4, "date": ["2026-01-0%d" % d for d in range(1, 5)] * 2,
                         "eod": [10, 30, 0, 20, 5, 5, 5, 5], "max": [12, 35, 95, 25, 5, 5, 5, 5],
                         "mean": [0.0] * 8, "changes": [3] * 4 + [0] * 4, "coll": [0, 0, 1, 0, 0, 0, 0, 0],
                         "stale": [0] * 4 + [1] * 4})
    return {
        "daily_ops": pd.DataFrame(rows),
        "sim_meta": pd.DataFrame([{"warmup_days": 2, "co2_kg_per_km": 0.5}]),
        "fill_levels": fill,
        "fill_bins": pd.DataFrame({"bin": [1, 2], "waste_type": ["paper", "glass"], "lat": [0.0, 0.0], "lon": [0.0, 0.0]}),
        "forecast_metrics": pd.DataFrame({"method": ["persistence", "lightgbm", "persistence", "lightgbm"],
                                          "horizon": [1, 1, 2, 2], "mae": [3.0, 2.0, 4.0, 5.0], "rmse": [1.0] * 4,
                                          "bias": [0.0] * 4, "n": [10] * 4}),
        "routing": pd.DataFrame({"instance": ["a", "b", "c"], "customers": [10, 20, 30], "best": [1.0] * 3,
                                 "mean": [1.0] * 3, "reference": [1.0] * 3, "reference_source": ["x"] * 3,
                                 "gap_best_pct": [1.5, -0.3, 7.0], "gap_mean_pct": [2.0] * 3}),
    }


@pytest.fixture(scope="module")
def kpi():
    return run_kpis(toy_raw())


def test_policy_totals_and_changes(kpi):
    t = kpi["kpi_policy"].set_index("policy")
    # fixed: per seed km 30 and 30, lifts 10 and 10, shifts 3 and 2, overflow 1 and 2
    assert t.loc["fixed", "km"] == pytest.approx(30.0)
    assert t.loc["fixed", "lifts"] == pytest.approx(10.0)
    assert t.loc["fixed", "truck_shifts"] == pytest.approx(2.5)
    assert t.loc["fixed", "max_trucks_per_day"] == 2
    assert t.loc["fixed", "overflow_bin_days"] == pytest.approx(1.5)
    assert t.loc["fixed", "co2_kg"] == pytest.approx(15.0)
    # forecast: km 20 and 16 -> 18, lifts 5 and 4 -> 4.5, overflow 1 and 0 -> 0.5, loads 4.5 and 3.4
    assert t.loc["forecast", "km"] == pytest.approx(18.0)
    assert t.loc["forecast", "km_change_pct"] == pytest.approx(-40.0)
    assert t.loc["forecast", "lifts_change_pct"] == pytest.approx(-55.0)
    assert t.loc["forecast", "overflow_change_pct"] == pytest.approx(-200.0 / 3.0)
    assert t.loc["forecast", "co2_change_pct"] == pytest.approx(-40.0)
    assert t.loc["forecast", "km_per_lift"] == pytest.approx(4.0)
    assert t.loc["forecast", "mean_fill_at_lift"] == pytest.approx(3.95 / 4.5)
    assert t.loc["fixed", "km_change_pct"] == pytest.approx(0.0)
    assert bool(t.loc["forecast", "all_feasible"])
    assert int(t.loc["forecast", "seeds"]) == 2


def test_weekly_series(kpi):
    w = kpi["kpi_weekly"].set_index("policy")
    assert w.loc["fixed", "week"] == 1
    assert w.loc["fixed", "km"] == pytest.approx(30.0)
    assert w.loc["forecast", "lifts"] == pytest.approx(4.5)
    assert w.loc["fixed", "overflow_bin_days"] == pytest.approx(1.5)


def test_observed_bins(kpi):
    b = kpi["kpi_bins_observed"].set_index("waste_type")
    assert b.loc["paper", "mean_level_pct"] == pytest.approx(15.0)
    assert b.loc["paper", "mean_daily_increment_pp"] == pytest.approx(20.0)   # +20, +20; the -30 collection day is excluded
    assert b.loc["paper", "collections_per_bin_week"] == pytest.approx(1.75)
    assert b.loc["paper", "share_days_max_ge_90"] == pytest.approx(0.25)
    assert b.loc["glass", "stale_share"] == pytest.approx(1.0)
    assert pd.isna(b.loc["glass", "mean_level_pct"])


def test_forecast_and_routing_tables(kpi):
    f = kpi["kpi_forecast"]
    best = f[f["best_in_horizon"]].set_index("horizon")["method"].to_dict()
    assert best == {1: "lightgbm", 2: "persistence"}
    r = kpi["kpi_routing"].set_index("instance")["band"].to_dict()
    assert r == {"a": "within 2%", "b": "at or below reference", "c": "above 5%"}


def test_statement_splitter_ignores_comments():
    assert statements("-- a; b\nSELECT 1;\n-- c\nSELECT 2;") == ["SELECT 1", "SELECT 2"]
