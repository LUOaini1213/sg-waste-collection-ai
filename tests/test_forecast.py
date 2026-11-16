"""Forecast backtest: hand-worked increments, sample rules, baselines and metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wastecollect.forecast.backtest import (Config, baselines, build_samples, cluster_bootstrap_diff,
                                            daily_increments, load_panel, metrics)


def toy_panel(n_days=40):
    """Bin 1 rises 5 pp a day and is emptied (logged) every 10th day; bin 2 is flat with one unlogged drop."""
    days = pd.date_range("2026-01-05", periods=n_days, freq="D").strftime("%Y-%m-%d")  # starts on a Monday
    rows = []
    lvl = 0.0
    for k, d in enumerate(days):
        coll = 1 if k % 10 == 0 and k > 0 else 0
        lvl = 5.0 if coll else lvl + 5.0
        rows.append({"bin": 1, "date": d, "eod": lvl, "coll": coll, "stale": 0})
    for k, d in enumerate(days):
        rows.append({"bin": 2, "date": d, "eod": 10.0 if k >= 33 else 50.0, "coll": 0, "stale": 1 if 20 <= k < 23 else 0})
    for k, d in enumerate(days):
        # bin 3: flat at 10, one logged collection with a small drop (10 -> 2) on day 31
        rows.append({"bin": 3, "date": d, "eod": 2.0 if k >= 31 else 10.0, "coll": 1 if k == 31 else 0, "stale": 0})
    lvl = 0.0
    for k, d in enumerate(days):
        # bin 4: never emptied, daily rise depends on the weekday (Mon 0.5 ... Sun 3.5)
        lvl += 0.5 * (k % 7 + 1)
        rows.append({"bin": 4, "date": d, "eod": lvl, "coll": 0, "stale": 0})
    levels = pd.DataFrame(rows)
    bins = pd.DataFrame({"bin": [1, 2, 3, 4], "waste_type": ["paper", "glass", "plastic", "residual"]})
    return load_panel(levels, bins, Config())


def test_daily_increments_mask_collections_drops_and_stale_days():
    P = toy_panel()
    inc = daily_increments(P)
    assert np.isnan(inc[0, 0])                       # no previous day
    assert inc[0, 1] == 5.0 and inc[0, 9] == 5.0
    assert np.isnan(inc[0, 10])                      # logged collection day
    assert inc[0, 11] == 5.0
    assert np.isnan(inc[1, 20]) and np.isnan(inc[1, 22])   # stale days
    assert np.isnan(inc[1, 23])                      # day after a stale day (previous reading unreliable)
    assert inc[1, 24] == 0.0
    assert np.isnan(inc[1, 33])                      # unlogged drop of 40 pp >= 20 pp
    assert inc[1, 34] == 0.0
    assert np.isnan(inc[2, 31]) and inc[2, 30] == 0.0 and inc[2, 32] == 0.0   # logged collection, small drop
    assert inc[3, 7] == 0.5 and inc[3, 13] == 3.5                             # Monday, Sunday


def test_samples_skip_windows_with_collections_drops_or_stale_days():
    P = toy_panel()
    S = build_samples(P, daily_increments(P))
    s1 = S[S["bin"] == 1]
    # day 30 is a collection: no window may contain it (t+1..t+h), but t = 30 itself is allowed
    for _, r in s1.iterrows():
        assert not (r["t"] < 30 <= r["t"] + r["h"])
        assert r["target"] == pytest.approx(r["y_t"] + 5.0 * r["h"])
    assert set(s1[s1["t"] == 29]["h"]) == set()
    assert set(s1[s1["t"] == 28]["h"]) == {1}
    assert set(s1[s1["t"] == 30]["h"]) == {1, 2, 3}
    s2 = S[S["bin"] == 2]
    assert set(s2[s2["t"] == 32]["h"]) == set()      # the drop on day 33 hides every window from day 32
    assert set(s2[s2["t"] == 31]["h"]) == {1}
    assert S["t"].min() == 28                        # 28 days of history needed
    s3 = S[S["bin"] == 3]
    assert set(s3[s3["t"] == 30]["h"]) == set() and set(s3[s3["t"] == 31]["h"]) == {1, 2, 3}
    assert set(s3[s3["t"] == 29]["h"]) == {1}


def test_baselines_on_hand_values():
    S = pd.DataFrame({"y_t": [40.0, 97.0, 10.0], "h": [1, 2, 3], "m14": [5.0, 5.0, np.nan], "sn_inc": [4.0, 12.0, -30.0]})
    b = baselines(S)
    assert b["persistence"].tolist() == [40.0, 97.0, 10.0]
    assert b["mean_rate"].tolist() == [45.0, 100.0, 10.0]          # 97 + 2*5 clipped to 100; nan rate -> 0
    assert b["seasonal_naive"].tolist() == [44.0, 100.0, 0.0]       # clipped to [0, 100]


def test_seasonal_naive_uses_same_weekday_increment():
    P = toy_panel()
    S = build_samples(P, daily_increments(P))
    r = S[(S["bin"] == 1) & (S["t"] == 35) & (S["h"] == 2)].iloc[0]
    assert r["sn_inc"] == pytest.approx(10.0)        # increments of days 29 and 30-> day 30 is masked, replaced by m14 = 5
    r = S[(S["bin"] == 1) & (S["t"] == 36) & (S["h"] == 1)].iloc[0]
    assert r["sn_inc"] == pytest.approx(5.0) and r["m14"] == pytest.approx(5.0) and r["days_since_coll"] == 6
    # bin 4, origin day 29 (a Tuesday): Wednesday rises 1.5, Thursday 2.0, Friday 2.5
    for h, want in ((1, 1.5), (2, 3.5), (3, 6.0)):
        r = S[(S["bin"] == 4) & (S["t"] == 29) & (S["h"] == h)].iloc[0]
        assert r["sn_inc"] == pytest.approx(want) and r["target"] - r["y_t"] == pytest.approx(want)
        assert r["m7"] == pytest.approx(2.0) and r["lag0"] == pytest.approx(1.0) and r["lag1"] == pytest.approx(0.5)
        assert r["dow_next"] == 2


def test_metrics_hand_worked():
    pred = pd.DataFrame({"target": [50.0, 90.0, 85.0, 20.0], "persistence": [40.0, 90.0, 70.0, 30.0],
                         "seasonal_naive": [50.0, 90.0, 85.0, 20.0], "mean_rate": [55.0, 95.0, 85.0, 85.0],
                         "lightgbm": [48.0, 79.0, 86.0, 21.0]})
    m = metrics(pred, threshold=80.0)
    assert m["persistence"]["mae"] == pytest.approx((10 + 0 + 15 + 10) / 4)
    assert m["persistence"]["rmse"] == pytest.approx(np.sqrt((100 + 0 + 225 + 100) / 4))
    assert m["persistence"]["bias"] == pytest.approx((-10 + 0 - 15 + 10) / 4)
    assert m["persistence"]["full_recall"] == pytest.approx(0.5) and m["persistence"]["full_precision"] == 1.0
    assert m["mean_rate"]["full_precision"] == pytest.approx(2 / 3) and m["mean_rate"]["full_recall"] == 1.0
    assert m["lightgbm"]["mae"] == pytest.approx((2 + 11 + 1 + 1) / 4)
    assert m["lightgbm"]["mae_vs_persistence"] == pytest.approx(3.75 / 8.75)
    assert m["seasonal_naive"]["mae"] == 0.0 and m["seasonal_naive"]["n"] == 4


def test_cluster_bootstrap_sign_and_determinism():
    rng = np.random.default_rng(0)
    n = 400
    pred = pd.DataFrame({"bin": np.repeat(np.arange(40), 10), "target": rng.uniform(0, 100, n)})
    pred["a"] = pred["target"] + rng.normal(0, 2, n)
    pred["b"] = pred["target"] + rng.normal(0, 8, n)
    d1 = cluster_bootstrap_diff(pred, "a", "b", n_boot=200, seed=1)
    d2 = cluster_bootstrap_diff(pred, "a", "b", n_boot=200, seed=1)
    assert d1 == d2 and d1["n_bins"] == 40
    assert d1["ci95"][0] < d1["mae_diff"] < d1["ci95"][1] < 0
    same = cluster_bootstrap_diff(pred, "a", "a", n_boot=50)
    assert same["mae_diff"] == 0.0 and same["ci95"] == [0.0, 0.0]
