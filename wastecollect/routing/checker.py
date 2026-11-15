"""Independent route checker for the WCVRP-IF (see instance.py for the rules).

Deliberately written as plain sequential loops that share no code with the search side
(routes.py / alns.py): for every route it tries every possible break placement explicitly.
Tests compare the two implementations on random routes.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .instance import CUSTOMER, DEPOT, FACILITY

TOL = 1e-6


@dataclass
class RouteReport:
    nodes: list
    distance: float
    load: float
    stops: int
    unloads: int
    feasible: bool
    violations: list
    starts: list = field(default_factory=list)   # service start per node (with the break used)
    rest: tuple | None = None                     # (gap index, option, start time)


@dataclass
class Evaluation:
    feasible: bool
    distance: float
    n_routes: int
    n_unloads: int
    violations: list
    routes: list
    unassigned: list

    def summary(self) -> dict:
        return {"feasible": self.feasible, "distance": self.distance, "routes": self.n_routes,
                "unloads": self.n_unloads, "violations": len(self.violations)}


def _simulate(inst, nodes, gap, option):
    """Service-start times along `nodes`; break in `gap` with `option` ('A' or 'B'), or none if gap is None.

    Returns (starts, None) when every time window holds, else (partial starts, message).
    """
    ready, due, service, T = inst.ready, inst.due, inst.service, inst.time
    t = float(ready[nodes[0]])
    starts = [t]
    for k in range(1, len(nodes)):
        i, j = nodes[k - 1], nodes[k]
        finish = t + float(service[i])
        if gap == k - 1:
            r1, r2, dur = inst.rest
            if option == "A":
                b = max(finish, r1)
                if b > r2 + TOL:
                    return starts, f"break cannot start by {r2:.0f}s at position {k - 1}"
                arrival = b + dur + float(T[i, j])
            else:
                b = max(finish + float(T[i, j]), r1)
                if b > r2 + TOL:
                    return starts, f"break cannot start by {r2:.0f}s at position {k}"
                arrival = b + dur
        else:
            arrival = finish + float(T[i, j])
        t = max(arrival, float(ready[j]))
        if t > float(due[j]) + TOL:
            return starts, f"node {j} served at {t:.1f}s after its latest start {float(due[j]):.1f}s"
        starts.append(t)
    return starts, None


def check_route(inst, nodes) -> RouteReport:
    nodes = [int(v) for v in nodes]
    viol = []
    kind, q, D = inst.kind, inst.demand, inst.dist
    n = inst.n_nodes
    if len(nodes) < 2 or nodes[0] != 0 or nodes[-1] != 0:
        viol.append("route must start and end at the depot (node 0)")
    for v in nodes:
        if not 0 <= v < n:
            viol.append(f"unknown node {v}")
            return RouteReport(nodes, float("nan"), 0.0, 0, 0, False, viol)
    inner = nodes[1:-1]
    if any(kind[v] == DEPOT for v in inner):
        viol.append("depot visited in the middle of a route")
    dist = sum(float(D[nodes[k], nodes[k + 1]]) for k in range(len(nodes) - 1))
    load = 0.0
    total = 0.0
    stops = 0
    unloads = 0
    for v in inner:
        if kind[v] == CUSTOMER:
            load += float(q[v])
            total += float(q[v])
            stops += 1
            if load > inst.capacity + TOL:
                viol.append(f"vehicle load {load:.2f} exceeds capacity {inst.capacity:g} at node {v}")
        elif kind[v] == FACILITY:
            load = 0.0
            unloads += 1
    customers = [v for v in inner if kind[v] == CUSTOMER]
    if customers:
        last_customer = max(k for k, v in enumerate(nodes) if kind[v] == CUSTOMER)
        if not any(kind[v] == FACILITY for v in nodes[last_customer + 1:-1]):
            viol.append("route returns to the depot without unloading after its last collection")
    if total > inst.route_capacity + TOL:
        viol.append(f"route total {total:.2f} exceeds route capacity {inst.route_capacity:g}")
    if inst.max_stops is not None and stops > inst.max_stops:
        viol.append(f"{stops} stops exceed the limit {inst.max_stops}")
    starts, rest_used = [], None
    if inst.rest is None:
        starts, msg = _simulate(inst, nodes, None, None)
        if msg:
            viol.append(msg)
    else:
        first_msg = None
        for gap in range(len(nodes) - 1):
            for option in ("B", "A"):
                st, msg = _simulate(inst, nodes, gap, option)
                if msg is None:
                    starts = st
                    i = nodes[gap]
                    fin = st[gap] + float(inst.service[i])
                    b = max(fin, inst.rest[0]) if option == "A" else max(fin + float(inst.time[i, nodes[gap + 1]]), inst.rest[0])
                    rest_used = (gap, option, b)
                    break
                first_msg = first_msg or msg
            if rest_used is not None:
                break
        if rest_used is None:
            viol.append("no feasible break placement: " + (first_msg or "route too short"))
    return RouteReport(nodes, dist, total, stops, unloads, not viol, viol, starts, rest_used)


def evaluate(inst, routes) -> Evaluation:
    """Check a full solution: every customer exactly once, every route feasible."""
    reports = [check_route(inst, r) for r in routes]
    viol = [f"route {k}: {m}" for k, rep in enumerate(reports) for m in rep.violations]
    seen = {}
    for k, rep in enumerate(reports):
        for v in rep.nodes[1:-1]:
            if 0 <= v < inst.n_nodes and inst.kind[v] == CUSTOMER:
                if v in seen:
                    viol.append(f"customer {v} visited by routes {seen[v]} and {k}")
                seen[v] = k
    missing = [int(c) for c in inst.customers if int(c) not in seen]
    if missing:
        viol.append(f"{len(missing)} customers not visited (e.g. {missing[:5]})")
    used = [rep for rep in reports if rep.stops > 0]
    dist = sum(rep.distance for rep in reports if rep.distance == rep.distance)
    return Evaluation(not viol, dist, len(used), sum(rep.unloads for rep in reports), viol, reports, missing)
