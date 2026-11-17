"""Routing: hand-worked cases, checker vs search-side consistency, ALNS and MIP cross-checks."""
from __future__ import annotations

import numpy as np
import pytest

from wastecollect.routing.alns import Params, solve
from wastecollect.routing.checker import check_route, evaluate
from wastecollect.routing.instance import CUSTOMER, DEPOT, FACILITY, Instance, parse_kim, subset, synthetic
from wastecollect.routing.routes import (INF, Route, UData, apply_move, brk_ready, drop_empty_trips,
                                         merge_trips, plain_insertion, split_insertion)

H = 3600.0


def line_instance(capacity=12.0, route_capacity=float("inf"), l3=61200.0, e3=28800.0, rest=(11 * H, 12 * H, H),
                  e2=28800.0, l2=61200.0):
    """Depot x=0, facility x=10, customers at x=4 (q=5) and x=6 (q=7); 1 km per minute."""
    xs = np.array([0.0, 10.0, 4.0, 6.0])
    dist = np.abs(xs[:, None] - xs[None, :])
    return Instance("line", np.array([DEPOT, FACILITY, CUSTOMER, CUSTOMER]),
                    ready=np.array([8 * H, 0.0, e2, e3]), due=np.array([17 * H, 24 * H, l2, l3]),
                    service=np.array([0.0, 600.0, 300.0, 300.0]), demand=np.array([0.0, 0.0, 5.0, 7.0]),
                    dist=dist, time=dist * 60.0, capacity=capacity, route_capacity=route_capacity,
                    max_stops=None, rest=rest)


# ---------------------------------------------------------------- hand-worked checker cases

def test_hand_worked_distance_and_schedule():
    inst = line_instance()
    rep = check_route(inst, [0, 2, 3, 1, 0])
    assert rep.feasible, rep.violations
    assert rep.distance == pytest.approx(4 + 2 + 4 + 10)
    # depart 08:00, x=4 at 08:04, x=6 at 08:11, facility at 08:20, back at 08:40; the route ends
    # before 11:00, so the break is taken at the depot after the return (gap 3, option B, 11:00-12:00)
    # and the last entry is the end of that break
    assert rep.starts == pytest.approx([28800, 29040, 29460, 30000, 43200])
    assert rep.rest == (3, "B", 39600.0)
    assert check_route(inst.with_rest(None), [0, 2, 3, 1, 0]).starts[-1] == pytest.approx(31200)
    assert rep.load == 12 and rep.stops == 2 and rep.unloads == 1


def test_break_absorbs_waiting_and_moves_after_tight_window():
    inst = line_instance(e3=11.5 * H, l3=12 * H)
    rep = check_route(inst, [0, 2, 3, 1, 0])
    assert rep.feasible
    # waiting for x=6 (opens 11:30): break 11:00-12:00 on arrival there, service at exactly 12:00
    assert rep.rest == (1, "B", 39600.0)
    assert rep.starts[2] == pytest.approx(43200.0)
    inst2 = line_instance(e3=11.5 * H, l3=43100.0)
    rep2 = check_route(inst2, [0, 2, 3, 1, 0])
    assert rep2.feasible
    # a break before x=6 would push it past 11:58:20, so the break follows it: 11:35 + 4 min drive
    assert rep2.rest == (2, "B", 41940.0)


def test_break_option_a_and_no_slot():
    # break must start exactly at 11:00; x=6 is served 10:54-10:59 (window 10:54-11:00)
    inst = line_instance(e3=10.9 * H, l3=11.0 * H, rest=(11 * H, 11 * H, H))
    rep = check_route(inst, [0, 3, 2, 1, 0])
    assert rep.feasible
    # on arrival at x=4 (11:01) is too late, so the break is taken at x=6 after service (option A)
    assert rep.rest == (1, "A", 39600.0)
    assert rep.starts[2] == pytest.approx(12 * H + 120)
    # if x=4 must also start by 11:05, no break placement works
    inst2 = line_instance(e3=10.9 * H, l3=11.0 * H, rest=(11 * H, 11 * H, H), l2=11 * H + 300)
    rep2 = check_route(inst2, [0, 3, 2, 1, 0])
    assert not rep2.feasible and any("no feasible break" in v for v in rep2.violations)
    assert not Route(inst2, [0, 3, 2, 1, 0]).feasible(inst2)
    assert Route(inst, [0, 3, 2, 1, 0]).feasible(inst)


def test_capacity_unload_and_route_limits():
    inst = line_instance(capacity=10.0)
    bad = check_route(inst, [0, 2, 3, 1, 0])
    assert not bad.feasible and any("exceeds capacity" in v for v in bad.violations)
    ok = check_route(inst, [0, 2, 1, 3, 1, 0])
    assert ok.feasible and ok.distance == pytest.approx(4 + 6 + 4 + 4 + 10)
    no_unload = check_route(inst, [0, 2, 0])
    assert not no_unload.feasible and any("without unloading" in v for v in no_unload.violations)
    inst2 = line_instance(capacity=10.0, route_capacity=11.0)
    assert not check_route(inst2, [0, 2, 1, 3, 1, 0]).feasible
    ev = evaluate(inst, [[0, 2, 1, 0]])
    assert not ev.feasible and ev.unassigned == [3]
    ev = evaluate(inst, [[0, 2, 1, 0], [0, 2, 3, 1, 0]])
    assert any("visited by routes" in v for v in ev.violations)


def test_brk_ready_options():
    rest = (11 * H, 12 * H, H)
    assert brk_ready(10 * H, 600, rest) == 12 * H            # arrive 10:10, wait, break 11-12 (B)
    assert brk_ready(11.5 * H, 600, rest) == 11.5 * H + 600 + H  # B: break on arrival 11:40
    assert brk_ready(11.95 * H, 600, rest) == 11.95 * H + H + 600  # B too late, A: break first
    assert brk_ready(12.1 * H, 0, rest) == INF


# ---------------------------------------------------------------- parser

KIM_SAMPLE = """280.0 \t\t\t\t// vehicle capacity (yards), real number
400.0\t\t\t\t// total yardage allowed for a vehicle per day, real number
500\t\t\t\t// total number of regular stops for a vehicle allowed per day
3600\t\t\t\t// lunch time duration in second, lunch time should start between 11:00am and 12:00 pm
40\t\t\t\t// speed: MPH
Stop_ID\tX(feet)\tY(feet)\tEarly_St_Time (HHMM)\tLate_St_Time (HHMM)\tSvice_Time(Sec)\tLoad (Yards)\t"Stop_Type(0: Depot, 1: regular stop, 2: Landfill)"
0   1215029.00   3461398.00  0400 1500 0  9999.00 0
1   1229125.00   3460107.00  0000 2400 1600  9999.00 2
7   1208344.00   3469904.00  0700 2400 180    12.00 1
"""


def test_parse_kim_format(tmp_path):
    p = tmp_path / "3_stop.txt"
    p.write_text(KIM_SAMPLE.replace("\n", "\r\n"), encoding="utf-8", newline="")
    inst = parse_kim(p)
    assert list(inst.kind) == [DEPOT, FACILITY, CUSTOMER]
    assert inst.capacity == 280 and inst.route_capacity == 400 and inst.max_stops == 500
    assert inst.rest == (39600.0, 43200.0, 3600.0)
    # benchmark page example: 15387 ft Manhattan = 2.914 mi, 262.3 s at 40 mph
    assert inst.dist[0, 1] == pytest.approx(15387 / 5280)
    assert inst.time[0, 1] == pytest.approx(262.3, abs=0.05)
    assert inst.ready[0] == 4 * H and inst.due[0] == 15 * H and inst.due[1] == 24 * H
    assert inst.demand.tolist() == [0.0, 0.0, 12.0] and inst.ids == (0, 1, 7)
    from wastecollect.routing.instance import hhmm_to_seconds
    assert hhmm_to_seconds(1445) == 14 * H + 45 * 60 and hhmm_to_seconds("0730") == 7.5 * H
    tabbed = KIM_SAMPLE.replace("280.0 \t\t\t\t// vehicle capacity (yards), real number",
                                '280\t//\tvehicle\tcapacity\t"(yards),"\treal\tnumber')
    p.write_text(tabbed, encoding="utf-8")
    assert parse_kim(p).capacity == 280


def test_subset_keeps_facilities():
    inst = synthetic(3, n_customers=6)
    sub = subset(inst, inst.customers[:3])
    assert sub.n_customers == 3 and len(sub.facilities) == len(inst.facilities)
    assert sub.dist[0, 1] == inst.dist[0, 1]


# ---------------------------------------------------------------- search side vs checker

def _random_routes(inst, rng, n_routes=3, unload_p=0.35):
    cust = list(rng.permutation(inst.customers))
    cuts = sorted(rng.choice(np.arange(1, len(cust)), size=min(n_routes - 1, len(cust) - 1), replace=False))
    routes = []
    for part in np.split(np.array(cust), cuts):
        nodes = [0]
        for c in part:
            nodes.append(int(c))
            if rng.random() < unload_p:
                nodes.append(int(rng.choice(inst.facilities)))
        if rng.random() < 0.9 and inst.kind[nodes[-1]] != FACILITY:
            nodes.append(int(rng.choice(inst.facilities)))
        nodes.append(0)
        routes.append(nodes)
    return routes


@pytest.mark.parametrize("seed", range(12))
def test_route_cache_agrees_with_checker(seed):
    rng = np.random.default_rng(seed)
    inst = synthetic(seed, n_customers=9, n_facilities=2, capacity=25.0,
                     rest=None if seed % 4 == 3 else (11 * H, 12 * H, H))
    agree = feasible = 0
    for _ in range(40):
        for nodes in _random_routes(inst, rng):
            r = Route(inst, nodes)
            rep = check_route(inst, nodes)
            assert r.feasible(inst) == rep.feasible, (nodes, rep.violations)
            assert r.dist == pytest.approx(rep.distance)
            agree += 1
            feasible += rep.feasible
    assert agree > 0 and feasible > 0


def _feasible_routes(inst, seed):
    best, _ = solve(inst, 60, seed)
    return [r for r in best.routes]


@pytest.mark.parametrize("seed", range(6))
def test_insertion_costs_match_checker(seed):
    inst = synthetic(100 + seed, n_customers=10, n_facilities=2, capacity=30.0)
    routes = _feasible_routes(inst, seed)
    from wastecollect.routing.alns import Context
    ctx = Context(inst, Params())
    rng = np.random.default_rng(seed)
    checked = 0
    for ri, r in enumerate(routes):
        cust = [v for v in r.nodes if inst.kind[v] == CUSTOMER]
        others = [int(v) for v in rng.choice(cust, size=max(1, len(cust) // 2), replace=False)]
        others += [v for k, o in enumerate(routes) if k != ri for v in o.nodes if inst.kind[v] == CUSTOMER]
        base = Route(inst, drop_empty_trips(inst, [v for v in r.nodes if v not in others]))
        if not base.feasible(inst):
            continue
        ud = UData(inst, others, ctx)
        cost = plain_insertion(inst, base, ud)
        for p in range(cost.shape[0]):
            for k, u in enumerate(others):
                nodes = apply_move(base.nodes, u, p, -1, 0)
                rep = check_route(inst, nodes)
                if np.isfinite(cost[p, k]):
                    assert rep.feasible, (nodes, rep.violations)
                    assert cost[p, k] == pytest.approx(rep.distance - base.dist)
                else:
                    assert not rep.feasible or p == len(base.nodes) - 2, (nodes, p)
                checked += 1
        gap_idx = np.repeat(np.arange(len(base.nodes) - 1)[:, None], len(others), axis=1)
        fac = np.repeat(inst.facilities[:, None], len(others), axis=1)
        sc = split_insertion(inst, base, ud, gap_idx, fac)
        for o in range(2):
            for p in range(gap_idx.shape[0]):
                for kf in range(fac.shape[0]):
                    for k, u in enumerate(others):
                        nodes = apply_move(base.nodes, u, p, int(fac[kf, k]), o + 1)
                        rep = check_route(inst, nodes)
                        val = sc[o, p, kf, k]
                        if np.isfinite(val):
                            assert rep.feasible, (nodes, rep.violations)
                            assert val == pytest.approx(rep.distance - base.dist)
                        elif rep.feasible:
                            # allowed only where the move creates a pointless double unload
                            assert o == 1 and p == len(base.nodes) - 2, (nodes, o, p)
                        checked += 1
    assert checked > 50


def test_drop_empty_trips_and_merge():
    inst = line_instance(capacity=12.0)
    assert drop_empty_trips(inst, [0, 1, 2, 1, 1, 0]) == [0, 2, 1, 0]
    assert drop_empty_trips(inst, [0, 1, 0]) == [0, 0]
    r = merge_trips(inst, Route(inst, [0, 2, 1, 3, 1, 0]))
    assert r.nodes == [0, 2, 3, 1, 0]
    inst10 = line_instance(capacity=10.0)
    assert merge_trips(inst10, Route(inst10, [0, 2, 1, 3, 1, 0])).nodes == [0, 2, 1, 3, 1, 0]


# ---------------------------------------------------------------- ALNS and MIP

def test_alns_feasible_and_deterministic():
    inst = synthetic(7, n_customers=25, n_facilities=2, capacity=40.0)
    a, ia = solve(inst, 150, 3)
    b, ib = solve(inst, 150, 3)
    ev = evaluate(inst, a.as_lists())
    assert ev.feasible, ev.violations
    assert a.as_lists() == b.as_lists()
    assert ev.distance <= ia["initial_distance"] + 1e-9


@pytest.mark.parametrize("seed", [11, 12])
def test_mip_optimum_checked_and_matched_by_alns(seed):
    from wastecollect.routing.mip import solve_mip
    inst = synthetic(seed, n_customers=5, n_facilities=2, capacity=20.0)
    res = solve_mip(inst, n_vehicles=2, time_limit=60)
    assert res["optimal"], res["status"]
    ev = evaluate(inst, res["routes"])
    assert ev.feasible, ev.violations
    assert ev.distance == pytest.approx(res["objective"], abs=1e-6)
    best, _ = solve(inst, 200, 1)
    assert best.distance() >= res["objective"] - 1e-6
    assert best.distance() == pytest.approx(res["objective"], abs=1e-6)


def test_alns_reaches_proven_optimum_on_five_customer_case():
    """Regression: with only ceil(0.4 n) = 2 removals per iteration ALNS stalled at 50.52 / 50.76 here.

    50.062357 is the HiGHS optimum (results/routing_mip.json, instance syn0-5).
    """
    inst = synthetic(0, n_customers=5, n_facilities=2, capacity=20.0)
    for seed in (1, 2, 3, 4, 5):
        best, _ = solve(inst, 300, seed)
        ev = evaluate(inst, best.as_lists())
        assert ev.feasible and ev.distance == pytest.approx(50.062357, abs=1e-5), seed


def test_mip_allows_as_many_unloads_as_an_optimum_may_need():
    """Regression: a fixed cap of four copies per facility cut off optima with five unloads at one facility."""
    from wastecollect.routing.mip import solve_mip, unload_bound
    xs = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    dist = np.abs(xs[:, None] - xs[None, :])
    n = len(xs)
    inst = Instance("five-unloads", np.array([DEPOT, FACILITY] + [CUSTOMER] * 5), ready=np.zeros(n), due=np.full(n, 24 * H),
                    service=np.zeros(n), demand=np.array([0, 0, 13, 13, 13, 13, 13.0]), dist=dist, time=dist * 60.0,
                    capacity=20.0, rest=None)
    assert unload_bound(inst, 1) == 5            # min(5 customers, 2 * floor(65 / 20) + 1 = 7)
    assert unload_bound(inst.with_rest(None), 3) == 5
    assert unload_bound(line_instance(capacity=12.0), 2) == 2   # 12 collected: min(2 customers, 2 * 1 + 2)
    res = solve_mip(inst, n_vehicles=1, time_limit=120)
    assert res["optimal"] and res["copies"] == 5
    # every stop needs its own trip: 2 (x - 1) per stop from the facility, +1 for leaving the depot, +1 to return
    assert res["objective"] == pytest.approx(2 * (1 + 2 + 3 + 4 + 5) + 2)
    ev = evaluate(inst, res["routes"])
    assert ev.feasible and ev.n_unloads == 5 and ev.n_routes == 1
    best, _ = solve(inst, 100, 1)
    assert best.distance() == pytest.approx(32.0)
