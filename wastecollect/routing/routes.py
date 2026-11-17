"""Search-side route representation with vectorised insertion checks (WCVRP-IF with a break).

Adapted from vrptw/routes.py of the author's sibling repository delivery-vrptw (MIT licence):
the forward/backward time-window caches are extended with a second "break already taken"
label, and trip (unload-to-unload) loads replace the single route load.

For a route v_0 .. v_m (v_0 = v_m = depot) we cache, per position k,
  a[k]   earliest service start at v_k when the break has NOT been taken yet
  aB[k]  earliest service start at v_k when the break HAS been taken before v_k
  z[k]   latest service start at v_k so that the rest of the route is feasible, break already taken
  zB[k]  latest service start at v_k so that the rest of the route is feasible and includes the break
(inf / -inf mark impossible states). The break may be taken in the gap between two consecutive
nodes i -> j, either at i after service (option A) or at j on arrival (option B); given the finish
time f at i and travel time t, the earliest time ready to serve j is
  max(f + t, r1) + d          if max(f + t, r1) <= r2      (B)
  max(f, r1) + d + t          elif max(f, r1) <= r2        (A)
  infeasible                  otherwise.
Inserting a segment between v_p and v_p+1 is time-feasible iff, propagating both labels through
the segment, (break label reaches v_p+1 by z[p+1]) or (no-break label reaches it by zB[p+1]).
"""
from __future__ import annotations

import numpy as np

from .instance import CUSTOMER, FACILITY

EPS = 1e-6
INF = float("inf")
NEG = -INF


def brk_ready(f, t, rest):
    """Earliest ready time at the next node when the break is taken in this gap (scalar)."""
    r1, r2, d = rest
    b = f + t
    if b < r1:
        b = r1
    if b <= r2 + EPS:
        return b + d
    b = f if f > r1 else r1
    if b <= r2 + EPS:
        return b + d + t
    return INF


def brk_ready_np(f, t, rest):
    r1, r2, d = rest
    with np.errstate(invalid="ignore"):
        bB = np.maximum(f + t, r1)
        bA = np.maximum(f, r1)
        return np.where(bB <= r2 + EPS, bB + d, np.where(bA <= r2 + EPS, bA + d + t, INF))


class Lists:
    """Python-list snapshots of the instance arrays (fast scalar access in the refresh loop)."""

    def __init__(self, inst):
        self.e = inst.ready.tolist()
        self.l = inst.due.tolist()
        self.s = inst.service.tolist()
        self.q = inst.demand.tolist()
        self.kind = inst.kind.tolist()
        self.T = inst.time.tolist()
        self.D = inst.dist.tolist()


def lists(inst) -> Lists:
    L = inst.meta.get("_lists")
    if L is None:
        L = Lists(inst)
        inst.meta["_lists"] = L
    return L


class Route:
    __slots__ = ("nodes", "dist", "load", "stops", "a", "aB", "z", "zB", "pre", "tl", "trip_ok",
                 "time_ok", "_gap")

    def __init__(self, inst, nodes):
        self.nodes = [int(v) for v in nodes]
        self._gap = None
        self._refresh(inst)

    # ------------------------------------------------------------------ caches
    def _refresh(self, inst):
        L = lists(inst)
        e, l, s, q, kind, T, D = L.e, L.l, L.s, L.q, L.kind, L.T, L.D
        rest = inst.rest
        v = self.nodes
        m = len(v)
        a = [INF] * m
        aB = [INF] * m
        a[0] = e[v[0]]
        dist = 0.0
        for k in range(1, m):
            i, j = v[k - 1], v[k]
            t = T[i][j]
            dist += D[i][j]
            f0 = a[k - 1] + s[i]
            f1 = aB[k - 1] + s[i]
            x0 = f0 + t
            st0 = x0 if x0 > e[j] else e[j]
            a[k] = st0 if st0 <= l[j] + EPS else INF
            if rest is not None:
                x1 = f1 + t
                if f0 < INF:
                    rb = brk_ready(f0, t, rest)
                    if rb < x1:
                        x1 = rb
                st1 = x1 if x1 > e[j] else e[j]
                aB[k] = st1 if st1 <= l[j] + EPS else INF
        if rest is None:
            aB = list(a)
        z = [NEG] * m
        zB = [NEG] * m
        z[m - 1] = l[v[m - 1]] if l[v[m - 1]] >= e[v[m - 1]] else NEG
        for k in range(m - 2, -1, -1):
            i, j = v[k], v[k + 1]
            t = T[i][j]
            Z = z[k + 1]
            zz = NEG
            if Z > NEG and Z >= e[j] - EPS:
                zz = Z - t - s[i]
                if zz > l[i]:
                    zz = l[i]
                if zz < e[i] - EPS:
                    zz = NEG
            z[k] = zz
            if rest is None:
                continue
            best = NEG
            ZB = zB[k + 1]
            if ZB > NEG and ZB >= e[j] - EPS:
                best = ZB - t - s[i]
            if Z > NEG and Z >= e[j] - EPS:
                r1, r2, d = rest
                w = r2 if r2 < Z - d else Z - d            # option B: break start <= w
                if r1 <= w + EPS:
                    cand = w - t - s[i]
                    if cand > best:
                        best = cand
                w = r2 if r2 < Z - d - t else Z - d - t    # option A
                if r1 <= w + EPS:
                    cand = w - s[i]
                    if cand > best:
                        best = cand
            if best > NEG:
                if best > l[i]:
                    best = l[i]
                if best < e[i] - EPS:
                    best = NEG
            zB[k] = best
        # trips: pre[k] = load collected since the last unload, after serving v_k
        pre = [0.0] * m
        cur = 0.0
        load = 0.0
        stops = 0
        trip_ok = True
        for k in range(m):
            node = v[k]
            if kind[node] == CUSTOMER:
                cur += q[node]
                load += q[node]
                stops += 1
                if cur > inst.capacity + EPS:
                    trip_ok = False
            else:
                cur = 0.0
            pre[k] = cur
        # tl[p]: load of the trip containing gap p (between v_p and v_p+1), i.e. the load collected
        # before the next unload; inf when no unload follows (the final gap into the depot)
        tl = [INF] * m
        close = INF
        for k in range(m - 2, -1, -1):
            nj = v[k + 1]
            if k + 1 == m - 1:
                close = INF
            elif kind[nj] == FACILITY:
                close = pre[k]
            tl[k] = close
        if stops and kind[v[m - 2]] != FACILITY:
            trip_ok = False
        self.a, self.aB, self.z, self.zB = a, aB, z, zB
        self.pre, self.tl = pre, tl
        self.dist, self.load, self.stops = dist, load, stops
        self.trip_ok = trip_ok
        self.time_ok = (aB[m - 1] < INF) if rest is not None else (a[m - 1] < INF)
        self._gap = None

    def feasible(self, inst) -> bool:
        if not (self.time_ok and self.trip_ok):
            return False
        if self.load > inst.route_capacity + EPS:
            return False
        if inst.max_stops is not None and self.stops > inst.max_stops:
            return False
        return True

    def customers(self, inst):
        kind = lists(inst).kind
        return [v for v in self.nodes if kind[v] == CUSTOMER]

    def n_unloads(self, inst):
        kind = lists(inst).kind
        return sum(1 for v in self.nodes if kind[v] == FACILITY)

    def gaps(self, inst):
        """Numpy arrays describing the m-1 gaps (cached)."""
        if self._gap is None:
            v = np.asarray(self.nodes)
            gi, gj = v[:-1], v[1:]
            a = np.asarray(self.a[:-1])
            aB = np.asarray(self.aB[:-1])
            s = inst.service[gi]
            g = {
                "gi": gi, "gj": gj,
                "f0": a + s, "f1": aB + s,
                "zj": np.asarray(self.z[1:]), "zBj": np.asarray(self.zB[1:]),
                "ej": inst.ready[gj],
                "pre": np.asarray(self.pre[:-1]), "tl": np.asarray(self.tl[:-1]),
                "dij": inst.dist[gi, gj],
            }
            g["final"] = np.zeros(len(gi), dtype=bool)
            g["final"][-1] = True
            self._gap = g
        return self._gap


def empty_route(inst):
    return Route(inst, [0, 0])


# ---------------------------------------------------------------------- vectorised checks

def _through(inst, f0, f1, t_in, node_e, node_l, node_s):
    """Propagate (no-break, break-taken) finish times through one inserted node."""
    rest = inst.rest
    with np.errstate(invalid="ignore"):
        st0 = np.maximum(node_e, f0 + t_in)
        st0 = np.where(st0 <= node_l + EPS, st0, INF)
        if rest is not None:
            x1 = np.minimum(f1 + t_in, brk_ready_np(f0, t_in, rest))
        else:
            x1 = f1 + t_in
        st1 = np.maximum(node_e, x1)
        st1 = np.where(st1 <= node_l + EPS, st1, INF)
    return st0 + node_s, st1 + node_s


def _arrive_ok(inst, f0, f1, t_last, ej, zj, zBj):
    rest = inst.rest
    with np.errstate(invalid="ignore"):
        if rest is not None:
            x1 = np.minimum(f1 + t_last, brk_ready_np(f0, t_last, rest))
        else:
            x1 = f1 + t_last
        ok = np.maximum(ej, x1) <= zj + EPS
        if rest is not None:
            ok |= np.maximum(ej, f0 + t_last) <= zBj + EPS
    return ok


class UData:
    __slots__ = ("U", "e", "l", "s", "q", "cand")

    def __init__(self, inst, U, ctx=None):
        U = np.asarray(U, dtype=int)
        self.U = U
        self.e = inst.ready[U]
        self.l = inst.due[U]
        self.s = inst.service[U]
        self.q = inst.demand[U]
        self.cand = ctx.if_cand[U] if ctx is not None else None


def plain_insertion(inst, route: Route, ud: UData, gaps=None):
    """cost[P, U] of inserting each customer at each gap (inf where infeasible).

    `gaps` optionally restricts the rows to the given gap indices.
    """
    g = route.gaps(inst)
    sel = slice(None) if gaps is None else np.asarray(gaps, dtype=int)
    col = lambda name: g[name][sel][:, None]  # noqa: E731
    gi, gj = col("gi"), col("gj")
    U = ud.U[None, :]
    T, D = inst.time, inst.dist
    t_iu = T[gi, U]
    t_uj = T[U, gj]
    f0, f1 = _through(inst, col("f0"), col("f1"), t_iu, ud.e, ud.l, ud.s)
    ok = _arrive_ok(inst, f0, f1, t_uj, col("ej"), col("zj"), col("zBj"))
    ok &= (col("tl") + ud.q[None, :]) <= inst.capacity + EPS
    ok &= ~col("final")
    ok &= route.load + ud.q[None, :] <= inst.route_capacity + EPS
    if inst.max_stops is not None and route.stops + 1 > inst.max_stops:
        ok[:] = False
    cost = D[gi, U] + D[U, gj] - col("dij")
    return np.where(ok, cost, INF)


def split_insertion(inst, route: Route, ud: UData, gap_idx, fac):
    """Insert [u, f] (order 1) or [f, u] (order 2) at the given gaps.

    gap_idx: int array (G, U) of gap indices per customer; fac: int array (K, U) of facility nodes.
    Returns cost arrays (2, G, K, U) for the two orders.
    """
    g = route.gaps(inst)
    T, D, E, Lt, S = inst.time, inst.dist, inst.ready, inst.due, inst.service
    G = gap_idx.shape[0]
    K = fac.shape[0]
    U = ud.U[None, None, :]
    gi = g["gi"][gap_idx][:, None, :]          # (G,1,U)
    gj = g["gj"][gap_idx][:, None, :]
    F = fac[None, :, :]                        # (1,K,U)
    f0 = g["f0"][gap_idx][:, None, :]
    f1 = g["f1"][gap_idx][:, None, :]
    ej = g["ej"][gap_idx][:, None, :]
    zj = g["zj"][gap_idx][:, None, :]
    zBj = g["zBj"][gap_idx][:, None, :]
    pre = g["pre"][gap_idx][:, None, :]
    tl = g["tl"][gap_idx][:, None, :]
    final = g["final"][gap_idx][:, None, :]
    dij = g["dij"][gap_idx][:, None, :]
    ue, ul, us, uq = ud.e[None, None, :], ud.l[None, None, :], ud.s[None, None, :], ud.q[None, None, :]
    fe, fl, fs = E[F], Lt[F], S[F]
    route_ok = route.load + uq <= inst.route_capacity + EPS
    if inst.max_stops is not None and route.stops + 1 > inst.max_stops:
        route_ok = np.zeros_like(route_ok)
    out = np.full((2, G, K, U.shape[2]), INF)
    # order 1: i -> u -> f -> j
    a0, a1 = _through(inst, f0, f1, T[gi, U], ue, ul, us)
    b0, b1 = _through(inst, a0, a1, T[U, F], fe, fl, fs)
    ok = _arrive_ok(inst, b0, b1, T[F, gj], ej, zj, zBj)
    post = np.where(final, 0.0, tl - pre)
    ok &= (np.where(final, 0.0, pre) + uq <= inst.capacity + EPS) & (post <= inst.capacity + EPS) & route_ok
    cost = D[gi, U] + D[U, F] + D[F, gj] - dij
    out[0] = np.where(ok, cost, INF)
    # order 2: i -> f -> u -> j (not allowed in the final gap: u would follow the last unload)
    a0, a1 = _through(inst, f0, f1, T[gi, F], fe, fl, fs)
    b0, b1 = _through(inst, a0, a1, T[F, U], ue, ul, us)
    ok = _arrive_ok(inst, b0, b1, T[U, gj], ej, zj, zBj)
    ok &= (tl - pre + uq <= inst.capacity + EPS) & ~final & route_ok
    cost = D[gi, F] + D[F, U] + D[U, gj] - dij
    out[1] = np.where(ok, cost, INF)
    return out


def best_insertion(inst, route: Route, ud: UData, n_split_gaps: int = 8):
    """Best insertion move of every customer in ud into route.

    Returns (cost[U], move) where move = (gap[U], facility[U], order[U]); order 0 = plain,
    1 = [u, f], 2 = [f, u]; cost inf where no feasible move exists.
    """
    nU = len(ud.U)
    P = len(route.nodes) - 1
    cols = np.arange(nU)
    plain = plain_insertion(inst, route, ud) if P > 1 else np.full((1, nU), INF)
    p_best = plain.argmin(axis=0)
    c_best = plain[p_best, cols]
    best_gap, best_fac, best_ord = p_best.copy(), np.full(nU, -1), np.zeros(nU, dtype=int)
    # candidate gaps for split moves: cheapest plain detours (ignoring feasibility) + final gap
    D = inst.dist
    g = route.gaps(inst)
    det = D[g["gi"][:, None], ud.U[None, :]] + D[ud.U[None, :], g["gj"][:, None]] - g["dij"][:, None]
    G = min(n_split_gaps, P)
    if G < P:
        idx = np.argpartition(det, G - 1, axis=0)[:G]
    else:
        idx = np.repeat(np.arange(P)[:, None], nU, axis=1)
    idx = np.vstack([idx, np.full((1, nU), P - 1)])
    fac = ud.cand.T  # (K, U)
    sc = split_insertion(inst, route, ud, idx, fac)   # (2, G+1, K, U)
    flat = sc.reshape(-1, nU)
    j = flat.argmin(axis=0)
    c_split = flat[j, cols]
    o, gk = np.divmod(j, sc.shape[1] * sc.shape[2])
    gg, kk = np.divmod(gk, sc.shape[2])
    better = c_split < c_best - 1e-12
    c_best = np.where(better, c_split, c_best)
    best_gap = np.where(better, idx[gg, cols], best_gap)
    best_fac = np.where(better, fac[kk, cols], best_fac)
    best_ord = np.where(better, o + 1, best_ord)
    return c_best, (best_gap, best_fac, best_ord)


def apply_move(nodes, u, gap, fac, order):
    if order == 0:
        seg = [u]
    elif order == 1:
        seg = [u, fac]
    else:
        seg = [fac, u]
    return nodes[: gap + 1] + seg + nodes[gap + 1:]


def removal_savings(inst, route: Route):
    """(customer nodes, distance saved by removing each one alone)."""
    v = np.asarray(route.nodes)
    k = np.flatnonzero(inst.kind[v] == CUSTOMER)
    if len(k) == 0:
        return np.empty(0, dtype=int), np.empty(0)
    D = inst.dist
    prev, cur, nxt = v[k - 1], v[k], v[k + 1]
    return cur, D[prev, cur] + D[cur, nxt] - D[prev, nxt]


# ---------------------------------------------------------------------- normalisation

def drop_empty_trips(inst, nodes):
    """Remove facility visits that close an empty trip; [0, ..., 0] without customers -> [0, 0]."""
    kind = inst.kind
    out = [nodes[0]]
    for v in nodes[1:-1]:
        if kind[v] == FACILITY and kind[out[-1]] != CUSTOMER:
            continue
        out.append(v)
    out.append(nodes[-1])
    if not any(kind[v] == CUSTOMER for v in out):
        return [0, 0]
    return out


def merge_trips(inst, route: Route) -> Route:
    kind = inst.kind
    changed = True
    while changed:
        changed = False
        v = route.nodes
        facs = [k for k in range(1, len(v) - 1) if kind[v[k]] == FACILITY]
        for idx in range(len(facs) - 1):
            k, k2 = facs[idx], facs[idx + 1]
            load_a = route.pre[k - 1]
            load_b = route.pre[k2 - 1]
            if load_a + load_b <= inst.capacity + EPS:
                cand = Route(inst, v[:k] + v[k + 1:])
                if cand.feasible(inst):
                    route = cand
                    changed = True
                    break
    return route


def reselect_facilities(inst, route: Route) -> Route:
    """For each unload, switch to the facility with the smallest detour that keeps the route feasible."""
    facs_all = inst.facilities
    if len(facs_all) < 2:
        return route
    D = inst.dist
    k = 1
    while k < len(route.nodes) - 1:
        v = route.nodes
        if inst.kind[v[k]] != FACILITY:
            k += 1
            continue
        i, j, cur = v[k - 1], v[k + 1], v[k]
        det = D[i, facs_all] + D[facs_all, j]
        order = np.argsort(det, kind="stable")
        cur_det = D[i, cur] + D[cur, j]
        for f in facs_all[order]:
            if D[i, f] + D[f, j] >= cur_det - 1e-12:
                break
            cand = Route(inst, v[:k] + [int(f)] + v[k + 1:])
            if cand.feasible(inst):
                route = cand
                break
        k += 1
    return route


class Solution:
    __slots__ = ("routes", "unassigned")

    def __init__(self, routes, unassigned=()):
        self.routes = [r for r in routes if len(r.nodes) > 2]
        self.unassigned = list(unassigned)

    def copy(self):
        return Solution(list(self.routes), list(self.unassigned))

    def distance(self):
        return sum(r.dist for r in self.routes)

    def cost(self, penalty=0.0):
        return self.distance() + penalty * len(self.unassigned)

    def as_lists(self):
        return [list(r.nodes) for r in self.routes]
