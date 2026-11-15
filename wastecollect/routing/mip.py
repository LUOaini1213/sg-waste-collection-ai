"""Exact vehicle-indexed MIP for small WCVRP-IF instances, solved with HiGHS (highspy).

Graph: start depot s, end depot e, the customers, and `copies` copies of every facility (each
copy visited at most once, so a solution can unload up to copies x |F| times in total).
Variables per vehicle k: x[i,j,k] arc binaries, start time t[i,k], load after service w[i,k],
break start b[k], and two break binaries per arc (option A: break at i after service, then drive;
option B: drive, then break at j before service). Constraints: each customer served once,
flow conservation, load reset at facilities, vehicle and route capacity, stop limit, time
windows (big-M), exactly one break per vehicle starting in [r1, r2], and vehicle-use symmetry
breaking. Arcs s->facility, facility->facility and customer->e are excluded (empty trips and
returning loaded are not allowed). Objective: total distance.
The decoded routes are re-checked with checker.evaluate by the caller.
"""
from __future__ import annotations

import math
import time

import highspy
import numpy as np

from .instance import CUSTOMER


def solve_mip(inst, n_vehicles: int, copies: int | None = None, time_limit: float = 60.0,
              threads: int = 1, verbose: bool = False) -> dict:
    C = [int(c) for c in inst.customers]
    F = [int(f) for f in inst.facilities]
    if copies is None:
        copies = min(4, int(math.ceil(inst.demand.sum() / inst.capacity)) + 1)
    # MIP node list: 0 = start depot, then customers, facility copies, last = end depot
    nodes = [0] + C + [f for f in F for _ in range(copies)] + [0]
    is_fac = [False] + [False] * len(C) + [True] * (len(F) * copies) + [False]
    N = len(nodes)
    S, E = 0, N - 1
    cust_idx = list(range(1, 1 + len(C)))
    fac_idx = list(range(1 + len(C), N - 1))
    D, T = inst.dist, inst.time
    e, l, s, q = inst.ready, inst.due, inst.service, inst.demand
    K = n_vehicles
    arcs = [(S, E)]  # an unused vehicle goes straight back
    for i in range(N - 1):
        for j in range(1, N):
            if i == j or (i == S and j == E):
                continue
            if i == S and is_fac[j]:          # no empty first trip
                continue
            if is_fac[i] and is_fac[j]:       # no two unloads in a row
                continue
            if j == E and not is_fac[i]:      # must unload before returning
                continue
            arcs.append((i, j))
    h = highspy.Highs()
    h.silent(not verbose)
    h.setOptionValue("time_limit", float(time_limit))
    h.setOptionValue("threads", int(threads))
    h.setOptionValue("mip_rel_gap", 0.0)
    h.setOptionValue("random_seed", 0)
    x = {}
    for k in range(K):
        for (i, j) in arcs:
            x[i, j, k] = h.addBinary(obj=float(D[nodes[i], nodes[j]]))
    tt, w = {}, {}
    for k in range(K):
        for v in range(N):
            nv = nodes[v]
            lo, hi = float(e[nv]), float(l[nv])
            tt[v, k] = h.addVariable(lb=lo, ub=hi)
            if v in cust_idx:
                w[v, k] = h.addVariable(lb=float(q[nv]), ub=float(inst.capacity))
    out = {i: [] for i in range(N)}
    inn = {j: [] for j in range(N)}
    for (i, j) in arcs:
        out[i].append(j)
        inn[j].append(i)
    for c in cust_idx:
        h.addConstr(sum(x[i, c, k] for k in range(K) for i in inn[c]) == 1)
    for f in fac_idx:
        h.addConstr(sum(x[i, f, k] for k in range(K) for i in inn[f]) <= 1)
    for k in range(K):
        h.addConstr(sum(x[S, j, k] for j in out[S]) == 1)
        h.addConstr(sum(x[i, E, k] for i in inn[E]) == 1)
        for v in cust_idx + fac_idx:
            h.addConstr(sum(x[i, v, k] for i in inn[v]) - sum(x[v, j, k] for j in out[v]) == 0)
        served = [(c, sum(x[i, c, k] for i in inn[c])) for c in cust_idx]
        h.addConstr(sum(float(q[nodes[c]]) * y for c, y in served) <= float(min(inst.route_capacity, 1e12)))
        if inst.max_stops is not None:
            h.addConstr(sum(y for _, y in served) <= int(inst.max_stops))
        if k + 1 < K:
            h.addConstr(sum(x[S, j, k] for j in out[S] if j != E) - sum(x[S, j, k + 1] for j in out[S] if j != E) >= 0)
    # loads (reset to zero after a facility or at the start)
    Q = float(inst.capacity)
    for k in range(K):
        for (i, j) in arcs:
            if j in cust_idx:
                qj = float(q[nodes[j]])
                if i in cust_idx:
                    h.addConstr(w[j, k] - w[i, k] - Q * x[i, j, k] >= qj - Q)
                # from start / facility: w_j >= q_j holds by its lower bound
    # times (big-M) and the break
    rest = inst.rest
    bvar = {}
    if rest is not None:
        r1, r2, dur = rest
        for k in range(K):
            bvar[k] = h.addVariable(lb=float(r1), ub=float(r2))
    for k in range(K):
        brk_terms = []
        for (i, j) in arcs:
            ni, nj = nodes[i], nodes[j]
            si, tij = float(s[ni]), float(T[ni, nj])
            M = float(l[ni]) + si + tij - float(e[nj])
            if M > 0:
                h.addConstr(tt[j, k] - tt[i, k] - M * x[i, j, k] >= si + tij - M)
            if rest is None:
                continue
            r1, r2, dur = rest
            bA = h.addBinary()
            bB = h.addBinary()
            brk_terms += [bA, bB]
            h.addConstr(bA + bB - x[i, j, k] <= 0)
            MA = float(l[ni]) + si - float(r1)
            if MA > 0:  # option A: b >= t_i + s_i
                h.addConstr(bvar[k] - tt[i, k] - MA * bA >= si - MA)
            MA2 = float(r2) + dur + tij - float(e[nj])
            if MA2 > 0:  # t_j >= b + dur + t_ij
                h.addConstr(tt[j, k] - bvar[k] - MA2 * bA >= dur + tij - MA2)
            MB = float(l[ni]) + si + tij - float(r1)
            if MB > 0:  # option B: b >= t_i + s_i + t_ij
                h.addConstr(bvar[k] - tt[i, k] - MB * bB >= si + tij - MB)
            MB2 = float(r2) + dur - float(e[nj])
            if MB2 > 0:  # t_j >= b + dur
                h.addConstr(tt[j, k] - bvar[k] - MB2 * bB >= dur - MB2)
        if rest is not None:
            h.addConstr(sum(brk_terms) == 1)
    t0 = time.perf_counter()
    h.run()
    secs = time.perf_counter() - t0
    status = h.getModelStatus()
    info = h.getInfo()
    res = {"status": h.modelStatusToString(status), "seconds": secs, "n_vehicles": K, "copies": copies,
           "n_binaries": len(x) * (3 if rest is not None else 1), "routes": None, "objective": None,
           "bound": float(info.mip_dual_bound) if np.isfinite(info.mip_dual_bound) else None,
           "gap": float(info.mip_gap) if np.isfinite(info.mip_gap) else None, "optimal": False}
    if info.primal_solution_status != 2:  # no feasible solution
        return res
    vals = h.getSolution().col_value
    routes = []
    for k in range(K):
        succ = {}
        for (i, j) in arcs:
            if vals[x[i, j, k].index] > 0.5:
                succ[i] = j
        path, cur, guard = [0], S, 0
        while cur != E and guard <= N:
            cur = succ.get(cur, E)
            path.append(nodes[cur])
            guard += 1
        if any(inst.kind[v] == CUSTOMER for v in path):
            routes.append(path)
    res["routes"] = routes
    res["objective"] = float(info.objective_function_value)
    res["optimal"] = status == highspy.HighsModelStatus.kOptimal
    return res
