"""Case-study simulator on a hand-worked toy town (no noise, four identical bins)."""
from __future__ import annotations

import numpy as np
import pytest

from wastecollect.casestudy.sim import SimConfig, _fixed_groups, make_instance, simulate
from wastecollect.routing.instance import CUSTOMER, DEPOT, FACILITY

H = 3600.0


def toy():
    """Depot 0 and facility 1 at x = 0; bins 2..5 at x = 10, 11, 20, 21 km on a line (30 km/h)."""
    xs = np.array([0.0, 0.0, 10.0, 11.0, 20.0, 21.0])
    dist = np.abs(xs[:, None] - xs[None, :])
    nodes = {"depot": 0, "facility": 1, "bins": np.array([2, 3, 4, 5])}
    return nodes, dist, dist / 30.0 * H, np.full(4, 100.0), xs[2:]


def cfg(**kw):
    base = dict(days_warmup=0, days_measured=14, base_rate=0.4, noise_cv=0.0, sensor_sd=0.0, truck_capacity=40.0,
                alns_iters=20, seed=1)
    base.update(kw)
    return SimConfig(**base)


def by_day(res, policy):
    return {d.day: d for d in res.days if d.policy == policy}


def test_fixed_groups_split_west_east():
    assert _fixed_groups(np.array([103.9, 103.1, 103.5, 103.7])).tolist() == [1, 0, 0, 1]
    assert _fixed_groups(np.array([1.0, 2.0, 3.0])).tolist() == [0, 1, 1]


def test_fixed_schedule_hand_worked():
    nodes, dist, time, units, lon = toy()
    res = simulate(nodes, dist, time, units, lon, cfg(policies=("fixed",)))
    d = by_day(res, "fixed")
    # west pair (x = 10, 11) on Mon/Wed/Fri, east pair on Tue/Thu/Sat, nothing on Sunday
    assert [d[k].bins for k in range(7)] == [2, 2, 2, 2, 2, 2, 0]
    # west bins: 0.4 a day; emptied Mon(0), Wed(0.8), Fri(0.8); Sat 0.8, Sun 1.2 -> overflow 0.2 each
    assert d[0].load == pytest.approx(0.0) and d[2].load == pytest.approx(1.6) and d[4].load == pytest.approx(1.6)
    assert d[6].overflow_bins == 2 and d[6].overflow_volume == pytest.approx(0.4)
    # east bins: first emptied Tue at 0.4 each; they overflow on Mondays from the second week on
    assert d[1].load == pytest.approx(0.8)
    assert d[0].overflow_bins == 0 and d[7].overflow_bins == 2 and d[7].overflow_volume == pytest.approx(0.4)
    assert d[7].load == pytest.approx(2.0)           # west bins were capped at full
    assert d[7].mean_fill_at_collection == pytest.approx(1.0)
    # routes: depot -> 10 -> 11 -> facility(0) -> depot = 22 km; east pair 42 km
    assert d[2].km == pytest.approx(22.0) and d[1].km == pytest.approx(42.0)
    assert d[2].routes == 1 and d[2].unloads == 1 and d[2].feasible
    s = res.summary(7)["fixed"]
    assert s["lifts"] == 12 and s["km"] == pytest.approx(3 * 22 + 3 * 42)
    assert s["overflow_bin_days"] == 4 and s["truck_days"] == 6 and s["max_trucks_per_day"] == 1
    assert s["co2_kg"] == pytest.approx(192 * 0.99156)
    assert s["mean_fill_at_collection"] == pytest.approx((2 * 1.0 + 4 * 0.8 + 2 * 1.0 + 4 * 0.8) / 12)


def test_threshold_policy_hand_worked():
    nodes, dist, time, units, lon = toy()
    res = simulate(nodes, dist, time, units, lon, cfg(policies=("threshold",)))
    d = by_day(res, "threshold")
    # readings 0, 0.4, 0.8 -> first collection on Wednesday, then Fri; Sat reading 0.4; Sunday overflow; Monday full
    assert [d[k].bins for k in range(8)] == [0, 0, 4, 0, 4, 0, 0, 4]
    assert d[2].load == pytest.approx(3.2) and d[7].load == pytest.approx(4.0)
    assert d[6].overflow_bins == 4 and d[6].overflow_volume == pytest.approx(0.8)
    assert d[2].km == pytest.approx(42.0)


def test_forecast_policy_hand_worked():
    nodes, dist, time, units, lon = toy()
    res = simulate(nodes, dist, time, units, lon, cfg(policies=("forecast",)))
    d = by_day(res, "forecast")
    # Mon: no history. Tue: 0.4 + 0.4 < 0.9. Wed: 0.8 + 0.4 -> collect. Fri: collect. Sat: 0.4 + 2 x 0.4 -> collect
    assert [d[k].bins for k in range(8)] == [0, 0, 4, 0, 4, 4, 0, 4]
    assert d[5].load == pytest.approx(1.6) and d[7].load == pytest.approx(3.2)
    assert sum(x.overflow_bins for x in res.days) == 0
    # a higher trigger level waits one more day and the bins overflow
    late = simulate(nodes, dist, time, units, lon, cfg(policies=("forecast",), forecast_level=1.3))
    assert sum(x.overflow_bins for x in late.days) > 0


def test_max_gap_forces_collection():
    nodes, dist, time, units, lon = toy()
    res = simulate(nodes, dist, time, units, lon, cfg(policies=("forecast",), base_rate=0.01, max_gap_days=7))
    d = by_day(res, "forecast")
    assert [k for k in range(14) if d[k].bins] == [7]    # day 7 is the first day with a 7-day gap (a Monday)


def test_noise_is_shared_by_policies_and_seeded():
    nodes, dist, time, units, lon = toy()
    c = cfg(noise_cv=0.5, sensor_sd=0.03, days_measured=7)
    a = simulate(nodes, dist, time, units, lon, c)
    b = simulate(nodes, dist, time, units, lon, c)
    assert [x.__dict__ for x in a.days] == [x.__dict__ for x in b.days]
    other = simulate(nodes, dist, time, units, lon, cfg(noise_cv=0.5, sensor_sd=0.03, days_measured=7, seed=2))
    assert [x.load for x in a.days] != [x.load for x in other.days]


def test_units_scale_fill_rate_with_clip():
    nodes, dist, time, _, lon = toy()
    units = np.array([10.0, 100.0, 100.0, 1000.0])   # mean 302.5 -> ratios clipped to 0.5 and 2.0
    res = simulate(nodes, dist, time, units, lon, cfg(policies=("fixed",), base_rate=0.1, days_measured=7))
    d = by_day(res, "fixed")
    # Wednesday: west pair (bins at x = 10, 11) after two days: 2 x 0.1 x (0.5 + 0.5)
    assert d[2].load == pytest.approx(2 * 0.1 * 0.5 + 2 * 0.1 * 0.5)
    # Thursday: east pair after two days: 2 x 0.1 x (0.5 + 2.0)
    assert d[3].load == pytest.approx(2 * 0.1 * 0.5 + 2 * 0.1 * 2.0)


def test_make_instance_layout():
    nodes, dist, time, units, lon = toy()
    c = cfg()
    inst = make_instance(nodes, dist, time, np.array([1, 3]), np.array([0.5, 0.25]), c)
    assert inst.kind.tolist() == [DEPOT, FACILITY, CUSTOMER, CUSTOMER]
    assert inst.demand.tolist() == [0.0, 0.0, 0.5, 0.25]
    assert inst.dist[2, 3] == 10.0 and inst.dist[0, 3] == 21.0
    assert inst.service.tolist() == [0.0, c.unload_s, c.service_s, c.service_s]
    assert inst.ready[0] == c.shift[0] and inst.due[0] == c.shift[1] and inst.rest == c.rest
