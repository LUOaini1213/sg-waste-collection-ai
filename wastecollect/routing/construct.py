"""Nearest-neighbour route building for the WCVRP-IF (initial solution for ALNS).

Routes are built one at a time. From the last collection stop the vehicle goes to the nearest
unrouted stop that can still be appended to the current trip; when none fits, it unloads and
starts a new trip at the nearest stop reachable from the facility; when no new trip fits either,
the route is closed. Every intermediate route is kept closed ([..., facility, depot]) so
feasibility (time windows, break, capacities) is checked with the search-side insertion code.
Similar in spirit to the initial-solution procedure of Benjamin (2011, ch. 3), not a copy of it.
"""
from __future__ import annotations

import numpy as np

from .routes import INF, Route, Solution, UData, empty_route, plain_insertion, split_insertion


def nearest_neighbour(inst, ctx) -> Solution:
    D = inst.dist
    unrouted = np.array(sorted(int(c) for c in inst.customers), dtype=int)
    routes = []
    unassigned = []
    while len(unrouted):
        route = empty_route(inst)
        ud = UData(inst, unrouted, ctx)
        # seed: nearest stop to the depot that can open a route on its own
        sc = split_insertion(inst, route, ud, np.zeros((1, len(unrouted)), dtype=int), ud.cand.T)
        open_cost = sc[0, 0].min(axis=0)
        ok = np.isfinite(open_cost)
        if not ok.any():
            unassigned.extend(unrouted.tolist())
            break
        bad = unrouted[~ok]
        if len(bad):
            unassigned.extend(bad.tolist())
            unrouted = unrouted[ok]
            ud = UData(inst, unrouted, ctx)
            sc = sc[:, :, :, ok]
        u_idx = int(np.lexsort((unrouted, D[0, unrouted]))[0])
        k = int(np.argmin(sc[0, 0, :, u_idx]))
        f = int(ud.cand[u_idx, k])
        route = Route(inst, [0, int(unrouted[u_idx]), f, 0])
        unrouted = np.delete(unrouted, u_idx)
        while len(unrouted):
            v = route.nodes
            m = len(v)
            ud = UData(inst, unrouted, ctx)
            last = v[m - 3]  # last collection stop (v[m-2] is the closing facility)
            # (a) append to the current trip: gap between `last` and the closing facility
            gap = np.array([m - 3])
            c = plain_insertion(inst, route, ud, gaps=gap)[0]
            if np.isfinite(c).any():
                score = np.where(np.isfinite(c), D[last, unrouted], INF)
                u_idx = int(np.lexsort((unrouted, score))[0])
                route = Route(inst, v[: m - 2] + [int(unrouted[u_idx])] + v[m - 2:])
                unrouted = np.delete(unrouted, u_idx)
                continue
            # (b) unload, then start a new trip: [u, f] after the closing facility
            fac_now = v[m - 2]
            sc = split_insertion(inst, route, ud, np.full((1, len(unrouted)), m - 2, dtype=int), ud.cand.T)
            best = sc[0, 0].min(axis=0)
            if np.isfinite(best).any():
                score = np.where(np.isfinite(best), D[fac_now, unrouted], INF)
                u_idx = int(np.lexsort((unrouted, score))[0])
                k = int(np.argmin(sc[0, 0, :, u_idx]))
                f = int(ud.cand[u_idx, k])
                route = Route(inst, v[: m - 1] + [int(unrouted[u_idx]), f] + v[m - 1:])
                unrouted = np.delete(unrouted, u_idx)
                continue
            break
        routes.append(route)
    return Solution(routes, unassigned)
