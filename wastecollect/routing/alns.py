"""Adaptive Large Neighbourhood Search for the WCVRP-IF (time windows, unloads, driver break).

Adapted from vrptw/alns.py of the author's sibling repository delivery-vrptw (MIT licence),
which follows Ropke & Pisinger (2006, Transportation Science 40(4)). Kept: the destroy operators
(random, worst, Shaw, route removal), greedy / noisy / regret-k repair over a cost matrix,
roulette-wheel adaptive weights (scores 33/9/13, reaction 0.1) and simulated-annealing
acceptance. Changed for waste collection:
  * objective is total distance (unlimited fleet, as in the benchmark), no vehicle-count stage;
  * an extra destroy operator removes one whole trip (the stops between two unloads);
  * repair moves are plain insertion into a trip, insertion together with a new unload
    ([u, f] or [f, u], f among the nearest facilities of u), or a new route [depot, u, f, depot];
  * after destroy, empty trips are dropped; after repair, adjacent trips that fit in one
    vehicle load are merged; the returned best solution is polished by re-choosing facilities.
Every move is checked with the search-side caches in routes.py; the caller re-checks the final
answer with the independent checker.py. Budgets are in iterations (deterministic per seed).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from .construct import nearest_neighbour
from .instance import CUSTOMER
from .routes import (INF, Route, Solution, UData, apply_move, best_insertion, drop_empty_trips,
                     empty_route, merge_trips, reselect_facilities, removal_savings)

DESTROY = ("random", "worst", "shaw", "route", "trip")
REPAIR = ("greedy", "greedy_noise", "regret2", "regret3")


@dataclass
class Params:
    xi: float = 0.4
    q_min: int = 4
    q_max_abs: int = 40
    p_worst: float = 3.0
    p_shaw: float = 6.0
    noise: float = 0.025
    segment: int = 100
    reaction: float = 0.1
    scores: tuple = (33.0, 9.0, 13.0)
    t_start: float = 0.05
    t_end_ratio: float = 0.002
    n_if_cand: int = 3
    n_split_gaps: int = 8
    history_every: int = 50


@dataclass
class OpStats:
    w_d: np.ndarray = field(default_factory=lambda: np.ones(len(DESTROY)))
    w_r: np.ndarray = field(default_factory=lambda: np.ones(len(REPAIR)))
    s_d: np.ndarray = field(default_factory=lambda: np.zeros(len(DESTROY)))
    s_r: np.ndarray = field(default_factory=lambda: np.zeros(len(REPAIR)))
    n_d: np.ndarray = field(default_factory=lambda: np.zeros(len(DESTROY)))
    n_r: np.ndarray = field(default_factory=lambda: np.zeros(len(REPAIR)))
    used_d: np.ndarray = field(default_factory=lambda: np.zeros(len(DESTROY), dtype=int))
    used_r: np.ndarray = field(default_factory=lambda: np.zeros(len(REPAIR), dtype=int))


class Context:
    def __init__(self, inst, params: Params):
        self.inst = inst
        self.p = params
        D = inst.dist
        C = inst.customers
        self.customers = C
        maxd = float(D.max()) if D.size and D.max() > 0 else 1.0
        span = max(float(inst.due.max() - inst.ready.min()), 1.0)
        maxq = max(float(inst.demand.max()), 1.0)
        self.rel = (9.0 * D / maxd
                    + 3.0 * (np.abs(inst.ready[:, None] - inst.ready[None, :])
                             + np.abs(inst.due[:, None] - inst.due[None, :])) / span
                    + 2.0 * np.abs(inst.demand[:, None] - inst.demand[None, :]) / maxq)
        F = inst.facilities
        k = min(params.n_if_cand, len(F))
        order = np.argsort(D[:, F], axis=1, kind="stable")[:, :k]
        self.if_cand = F[order]
        self.all_if = np.repeat(F[None, :], inst.n_nodes, axis=0)
        self.empty = empty_route(inst)
        self.max_arc = float(D.max()) if D.size else 1.0
        self.noise_amp = params.noise * self.max_arc
        self.penalty = 10.0 * max(self.max_arc, 1.0)


# ----------------------------------------------------------------------------- destroy

def _remove(ctx, sol: Solution, removed: set) -> None:
    inst = ctx.inst
    removed = set(removed)
    new_routes = []
    for r in sol.routes:
        if not any(v in removed for v in r.nodes):
            new_routes.append(r)
            continue
        nodes = drop_empty_trips(inst, [v for v in r.nodes if v not in removed])
        if len(nodes) <= 2:
            continue
        nr = Route(inst, nodes)
        if nr.feasible(inst):
            new_routes.append(nr)
        else:  # rare: the break no longer fits after the removal -> dissolve the route
            removed.update(v for v in nodes if inst.kind[v] == CUSTOMER)
    sol.routes = new_routes
    sol.unassigned = sorted(set(sol.unassigned) | removed)


def _assigned(ctx, sol):
    kind = ctx.inst.kind
    return [v for r in sol.routes for v in r.nodes if kind[v] == CUSTOMER]


def destroy_random(ctx, sol, q, rng):
    cust = _assigned(ctx, sol)
    if not cust:
        return
    q = min(q, len(cust))
    pick = rng.choice(len(cust), size=q, replace=False)
    _remove(ctx, sol, {cust[i] for i in pick})


def destroy_worst(ctx, sol, q, rng):
    cust, sav = [], []
    for r in sol.routes:
        c, s = removal_savings(ctx.inst, r)
        cust.extend(c.tolist())
        sav.extend(s.tolist())
    if not cust:
        return
    cu = np.array(cust)
    sv = np.array(sav)
    order = list(np.lexsort((cu, -sv)))
    removed = set()
    q = min(q, len(order))
    while len(removed) < q and order:
        k = int(math.floor(rng.random() ** ctx.p.p_worst * len(order)))
        removed.add(int(cu[order.pop(k)]))
    _remove(ctx, sol, removed)


def destroy_shaw(ctx, sol, q, rng):
    cust = _assigned(ctx, sol)
    if not cust:
        return
    q = min(q, len(cust))
    rest = np.array(sorted(cust))
    first = int(rest[int(rng.integers(len(rest)))])
    removed = [first]
    rest = rest[rest != first]
    while len(removed) < q and len(rest):
        r = removed[int(rng.integers(len(removed)))]
        order = np.lexsort((rest, ctx.rel[r, rest]))
        k = int(math.floor(rng.random() ** ctx.p.p_shaw * len(order)))
        c = int(rest[order[k]])
        removed.append(c)
        rest = rest[rest != c]
    _remove(ctx, sol, set(removed))


def destroy_route(ctx, sol, q, rng):
    if not sol.routes:
        return
    kind = ctx.inst.kind
    w = np.array([1.0 / max(r.stops, 1) for r in sol.routes])
    k = int(rng.choice(len(sol.routes), p=w / w.sum()))
    _remove(ctx, sol, {v for v in sol.routes[k].nodes if kind[v] == CUSTOMER})


def destroy_trip(ctx, sol, q, rng):
    kind = ctx.inst.kind
    trips = []
    for r in sol.routes:
        cur = []
        for v in r.nodes[1:-1]:
            if kind[v] == CUSTOMER:
                cur.append(v)
            elif cur:
                trips.append(cur)
                cur = []
    if not trips:
        return
    k = int(rng.integers(len(trips)))
    _remove(ctx, sol, set(trips[k]))


DESTROY_FN = {"random": destroy_random, "worst": destroy_worst, "shaw": destroy_shaw,
              "route": destroy_route, "trip": destroy_trip}


# ----------------------------------------------------------------------------- repair

def _column(ctx, route, ud, rng, noise):
    c, (g, f, o) = best_insertion(ctx.inst, route, ud, ctx.p.n_split_gaps)
    if noise:
        eps = rng.uniform(-1.0, 1.0, size=c.shape) * ctx.noise_amp
        c = np.where(np.isfinite(c), np.maximum(c + eps, 0.0), INF)
    return c, g, f, o


def repair(ctx, sol: Solution, rng, k_regret: int = 1, noise: bool = False) -> None:
    """Insert all unassigned customers (greedy if k_regret == 1, else regret-k). In place."""
    inst = ctx.inst
    U = np.array(sorted(sol.unassigned), dtype=int)
    nU = len(U)
    if nU == 0:
        return
    routes = list(sol.routes)
    R0 = len(routes)
    ncol = R0 + nU + 1
    C = np.full((nU, ncol), INF)
    MG = np.zeros((nU, ncol), dtype=int)
    MF = np.full((nU, ncol), -1, dtype=int)
    MO = np.zeros((nU, ncol), dtype=int)
    ud = UData(inst, U, ctx)
    for ri, r in enumerate(routes):
        C[:, ri], MG[:, ri], MF[:, ri], MO[:, ri] = _column(ctx, r, ud, rng, noise)
    ud_all = UData(inst, U, ctx)
    ud_all.cand = ctx.all_if[U]
    newc = ncol - 1
    C[:, newc], MG[:, newc], MF[:, newc], MO[:, newc] = _column(ctx, ctx.empty, ud_all, rng, noise)
    col_route = list(range(R0)) + [None] * nU
    next_col = R0
    active = np.ones(nU, dtype=bool)
    touched = set()
    BIG = 1e12
    while active.any():
        idx = np.flatnonzero(active)
        sub = C[idx]
        c1 = sub.min(axis=1)
        if k_regret == 1:
            pick = int(np.lexsort((U[idx], c1))[0])
        else:
            k = min(k_regret, sub.shape[1])
            part = np.sort(np.partition(sub, k - 1, axis=1)[:, :k], axis=1)
            fin = np.isfinite(part)
            with np.errstate(invalid="ignore"):
                diff = np.where(fin, part - part[:, :1], BIG)
            reg = diff[:, 1:].sum(axis=1)
            reg = np.where(np.isfinite(c1), reg, -INF)
            pick = int(np.lexsort((U[idx], c1, -reg))[0])
        row = idx[pick]
        if not np.isfinite(c1[pick]):
            break
        u = int(U[row])
        col = int(np.argmin(C[row]))
        gap, fac, order = int(MG[row, col]), int(MF[row, col]), int(MO[row, col])
        active[row] = False
        if col == newc:
            new = Route(inst, apply_move([0, 0], u, gap, fac, order))
            routes.append(new)
            ri = len(routes) - 1
            col_route[next_col] = ri
            tcol = next_col
            next_col += 1
        else:
            ri = col_route[col]
            new = Route(inst, apply_move(routes[ri].nodes, u, gap, fac, order))
            routes[ri] = new
            tcol = col
        touched.add(ri)
        rem = np.flatnonzero(active)
        if len(rem):
            sub_ud = UData(inst, U[rem], ctx)
            C[rem, tcol], MG[rem, tcol], MF[rem, tcol], MO[rem, tcol] = _column(ctx, new, sub_ud, rng, noise)
    for ri in touched:
        routes[ri] = merge_trips(inst, routes[ri])
    sol.routes = [r for r in routes if len(r.nodes) > 2]
    sol.unassigned = [int(U[i]) for i in np.flatnonzero(active)]


REPAIR_ARGS = {"greedy": (1, False), "greedy_noise": (1, True), "regret2": (2, False), "regret3": (3, False)}


# ----------------------------------------------------------------------------- search

def _roulette(w, rng):
    return int(rng.choice(len(w), p=w / w.sum()))


def _update_weights(st: OpStats, p: Params):
    for w, s, n in ((st.w_d, st.s_d, st.n_d), (st.w_r, st.s_r, st.n_r)):
        m = n > 0
        w[m] = (1 - p.reaction) * w[m] + p.reaction * s[m] / n[m]
        w[:] = np.maximum(w, 0.05)
        s[:] = 0
        n[:] = 0


def polish(inst, sol: Solution) -> Solution:
    routes = [reselect_facilities(inst, merge_trips(inst, r)) for r in sol.routes]
    return Solution(routes, sol.unassigned)


def solve(inst, iters: int, seed: int, params: Params | None = None, start: Solution | None = None,
          time_limit: float | None = None):
    """Run ALNS from the nearest-neighbour start (or `start`). Returns (best Solution, info)."""
    params = params or Params()
    ctx = Context(inst, params)
    rng = np.random.default_rng(seed)
    stats = OpStats()
    t0 = time.perf_counter()
    deadline = t0 + time_limit if time_limit else None
    cur = start.copy() if start is not None else nearest_neighbour(inst, ctx)
    if cur.unassigned:
        repair(ctx, cur, rng, 2, False)
    init_cost = cur.cost(0.0)
    n = inst.n_customers
    # remove at least min(n, q_min) customers: with fewer, tiny instances cannot regroup their trips
    q_hi = max(min(n, params.q_min), min(params.q_max_abs, int(math.ceil(params.xi * n))), 1)
    q_lo = min(params.q_min, q_hi)
    cur_cost = cur.cost(ctx.penalty)
    best = cur.copy()
    best_key = (len(best.unassigned), best.cost(0.0))
    T = params.t_start * max(init_cost, 1e-9) / math.log(2.0)
    cool = params.t_end_ratio ** (1.0 / max(iters, 1))
    history = [(0, best_key[1])]
    it = 0
    for it in range(1, iters + 1):
        if deadline is not None and time.perf_counter() > deadline:
            it -= 1
            break
        di = _roulette(stats.w_d, rng)
        ri = _roulette(stats.w_r, rng)
        q = int(rng.integers(q_lo, q_hi + 1))
        cand = cur.copy()
        DESTROY_FN[DESTROY[di]](ctx, cand, q, rng)
        k, noise = REPAIR_ARGS[REPAIR[ri]]
        repair(ctx, cand, rng, k, noise)
        c = cand.cost(ctx.penalty)
        key = (len(cand.unassigned), cand.cost(0.0))
        score = 0.0
        if key[0] < best_key[0] or (key[0] == best_key[0] and key[1] < best_key[1] - 1e-9):
            best, best_key = cand.copy(), key
            cur, cur_cost = cand, c
            score = params.scores[0]
        elif c < cur_cost - 1e-9:
            cur, cur_cost = cand, c
            score = params.scores[1]
        elif T > 0 and rng.random() < math.exp(-(c - cur_cost) / T):
            cur, cur_cost = cand, c
            score = params.scores[2]
        stats.s_d[di] += score
        stats.s_r[ri] += score
        stats.n_d[di] += 1
        stats.n_r[ri] += 1
        stats.used_d[di] += 1
        stats.used_r[ri] += 1
        if it % params.segment == 0:
            _update_weights(stats, params)
        if it % params.history_every == 0:
            history.append((it, best_key[1]))
        T *= cool
    best = polish(inst, best)
    info = {"iters": iters, "iters_used": it, "seed": seed, "initial_distance": init_cost,
            "seconds": time.perf_counter() - t0, "history": history,
            "destroy_usage": dict(zip(DESTROY, stats.used_d.tolist())),
            "repair_usage": dict(zip(REPAIR, stats.used_r.tolist()))}
    return best, info
