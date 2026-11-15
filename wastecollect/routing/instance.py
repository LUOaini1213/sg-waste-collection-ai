"""Problem data for the waste-collection VRP with intermediate facilities (WCVRP-IF).

Node layout used everywhere in this package: index 0 is the depot, indices 1..F are the
intermediate (disposal) facilities, F+1..N-1 are the collection stops ("customers").

Rules (the WCVRPTW of Kim, Kim & Sahoo 2006, as used by Benjamin 2011):
  * a route starts at the depot, collects at customers, unloads at an IF whenever needed
    (the load between two unloads must not exceed `capacity`), and ends at the depot empty,
    i.e. the last stop before the depot is an IF;
  * the total amount collected by one route is at most `route_capacity` and the number of
    customers at most `max_stops`;
  * service at node i must start within [ready_i, due_i] (waiting is allowed); the route must
    leave the depot no earlier than ready_0 and be back no later than due_0;
  * if `rest` = (r1, r2, duration) is set, every route takes one break of `duration` seconds that
    starts within [r1, r2]. The break is taken at a node: either after service at a node, before
    driving on ("option A"), or on arrival at the next node, before its service ("option B").
    Taking it at the depot after the return counts (Benjamin 2011, sec. 3.7.1);
  * the fleet is homogeneous and unlimited; the objective is total distance.
Times are seconds after midnight, loads in the instance's own unit (cubic yards for Kim et al.).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

DEPOT, CUSTOMER, FACILITY = 0, 1, 2
FEET_PER_MILE = 5280.0


def _ro(a, dtype=float):
    a = np.array(a, dtype=dtype)
    a.setflags(write=False)
    return a


@dataclass(frozen=True)
class Instance:
    name: str
    kind: np.ndarray          # 0 depot, 1 customer, 2 facility (index 0 must be the depot)
    ready: np.ndarray         # earliest service start (s)
    due: np.ndarray           # latest service start (s); due[0] = latest return to depot
    service: np.ndarray       # service duration (s)
    demand: np.ndarray        # amount collected (0 for depot / facilities)
    dist: np.ndarray          # N x N distance matrix (objective unit)
    time: np.ndarray          # N x N travel-time matrix (s)
    capacity: float           # vehicle capacity between two unloads
    route_capacity: float = float("inf")   # total collected per route (per driver-day)
    max_stops: int | None = None            # customers per route
    rest: tuple | None = None               # (r1, r2, duration) or None
    x: np.ndarray | None = None
    y: np.ndarray | None = None
    ids: tuple = ()                          # original stop ids, aligned with node index
    meta: dict = field(default_factory=dict, compare=False)

    def __post_init__(self):
        for name in ("kind", "ready", "due", "service", "demand", "dist", "time"):
            arr = getattr(self, name)
            ro = _ro(arr, int if name == "kind" else float)
            object.__setattr__(self, name, ro)
        self.validate()

    # ------------------------------------------------------------------ helpers
    @property
    def n_nodes(self) -> int:
        return len(self.kind)

    @property
    def facilities(self) -> np.ndarray:
        return np.flatnonzero(self.kind == FACILITY)

    @property
    def customers(self) -> np.ndarray:
        return np.flatnonzero(self.kind == CUSTOMER)

    @property
    def n_customers(self) -> int:
        return int((self.kind == CUSTOMER).sum())

    def validate(self) -> None:
        n = len(self.kind)
        if n < 1 or self.kind[0] != DEPOT or (self.kind == DEPOT).sum() != 1:
            raise ValueError("node 0 must be the only depot")
        nf = int((self.kind == FACILITY).sum())
        if not np.all(self.kind[1:1 + nf] == FACILITY) or not np.all(self.kind[1 + nf:] == CUSTOMER):
            raise ValueError("layout must be depot, facilities, customers")
        if self.n_customers and nf == 0:
            raise ValueError("customers need at least one facility")
        for name in ("ready", "due", "service", "demand"):
            a = getattr(self, name)
            if a.shape != (n,) or not np.all(np.isfinite(a)):
                raise ValueError(f"{name} must be a finite vector of length {n}")
        for name in ("dist", "time"):
            a = getattr(self, name)
            if a.shape != (n, n) or not np.all(np.isfinite(a)) or np.any(a < 0):
                raise ValueError(f"{name} must be a finite non-negative {n}x{n} matrix")
        if np.any(self.service < 0) or np.any(self.demand < 0):
            raise ValueError("service and demand must be non-negative")
        if np.any(self.demand[self.kind != CUSTOMER] != 0):
            raise ValueError("depot and facilities must have zero demand")
        if np.any(self.ready > self.due):
            raise ValueError("ready must not exceed due")
        if not (self.capacity > 0):
            raise ValueError("capacity must be positive")
        if self.rest is not None:
            r1, r2, d = self.rest
            if not (r1 <= r2 and d >= 0):
                raise ValueError("rest must be (r1 <= r2, duration >= 0)")

    def with_rest(self, rest):
        """Copy with a different break rule (None disables the break)."""
        return Instance(self.name, self.kind, self.ready, self.due, self.service, self.demand,
                        self.dist, self.time, self.capacity, self.route_capacity, self.max_stops,
                        rest, self.x, self.y, self.ids, dict(self.meta))


# ---------------------------------------------------------------------- Kim et al. format

def hhmm_to_seconds(v: float) -> float:
    v = int(round(float(v)))
    return float((v // 100) * 3600 + (v % 100) * 60)


def parse_kim(path) -> Instance:
    """Read one Kim, Kim & Sahoo (2006) WCVRPTW benchmark file (e.g. ``102_stop.txt``).

    File layout: five header lines (vehicle capacity, route capacity per day, max stops per day,
    lunch duration in seconds, default speed in mph), one column-header line, then one line per
    stop: id, x (feet), y (feet), earliest and latest service start (HHMM), service time (s),
    load, type (0 depot, 1 regular stop, 2 landfill / transfer station). Some files are
    tab-separated with quoted words; both forms are accepted.
    Distances follow the benchmark's instructions: Manhattan distance on the feet coordinates,
    reported in miles; travel time = distance / speed. The lunch break must start between
    11:00 and 12:00 (header comment in every file).
    """
    path = Path(path)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    head = []
    for line in lines[:5]:
        tok = re.split(r'[\s"]+', line.strip())
        head.append(float(tok[0]))
    cap, route_cap, max_stops, lunch, mph = head
    rows = []
    for line in lines[6:]:
        parts = line.replace('"', " ").split()
        if len(parts) < 8:
            continue
        rows.append([float(p) for p in parts[:8]])
    a = np.array(rows)
    typ = a[:, 7].astype(int)
    if (typ == 0).sum() != 1:
        raise ValueError(f"{path.name}: expected exactly one depot")
    order = np.concatenate([np.flatnonzero(typ == 0), np.flatnonzero(typ == 2), np.flatnonzero(typ == 1)])
    a = a[order]
    kind = np.where(a[:, 7] == 0, DEPOT, np.where(a[:, 7] == 2, FACILITY, CUSTOMER))
    x, y = a[:, 1], a[:, 2]
    feet = np.abs(x[:, None] - x[None, :]) + np.abs(y[:, None] - y[None, :])
    dist = feet / FEET_PER_MILE
    time = dist / mph * 3600.0
    ready = np.array([hhmm_to_seconds(v) for v in a[:, 3]])
    due = np.array([hhmm_to_seconds(v) for v in a[:, 4]])
    demand = np.where(kind == CUSTOMER, a[:, 6], 0.0)
    service = a[:, 5].copy()
    return Instance(
        name=path.name.split("_")[0], kind=kind, ready=ready, due=due, service=service,
        demand=demand, dist=dist, time=time, capacity=cap, route_capacity=route_cap,
        max_stops=int(max_stops), rest=(11 * 3600.0, 12 * 3600.0, lunch), x=x, y=y,
        ids=tuple(int(v) for v in a[:, 0]), meta={"speed_mph": mph, "source_file": path.name})


def subset(inst: Instance, customers, name=None, **overrides) -> Instance:
    """Instance restricted to the depot, all facilities and the given customer node indices."""
    customers = [int(c) for c in customers]
    if any(inst.kind[c] != CUSTOMER for c in customers):
        raise ValueError("subset() takes customer indices")
    keep = np.array([0] + list(inst.facilities) + customers, dtype=int)
    ix = np.ix_(keep, keep)
    kw = dict(name=name or f"{inst.name}-sub{len(customers)}", kind=inst.kind[keep],
              ready=inst.ready[keep], due=inst.due[keep], service=inst.service[keep],
              demand=inst.demand[keep], dist=inst.dist[ix], time=inst.time[ix],
              capacity=inst.capacity, route_capacity=inst.route_capacity, max_stops=inst.max_stops,
              rest=inst.rest, x=None if inst.x is None else inst.x[keep],
              y=None if inst.y is None else inst.y[keep],
              ids=tuple(inst.ids[k] for k in keep) if inst.ids else (), meta=dict(inst.meta))
    kw.update(overrides)
    return Instance(**kw)


def synthetic(seed: int, n_customers: int = 8, n_facilities: int = 2, capacity: float = 40.0,
              route_capacity: float = float("inf"), rest=(11 * 3600.0, 12 * 3600.0, 3600.0),
              horizon=(6 * 3600.0, 16 * 3600.0), name=None) -> Instance:
    """Small SYNTHETIC instance (km, 30 km/h) for tests and MIP cross-checks."""
    rng = np.random.default_rng(seed)
    n = 1 + n_facilities + n_customers
    xy = rng.uniform(0, 10, size=(n, 2))
    xy[0] = (5.0, 5.0)
    kind = np.array([DEPOT] + [FACILITY] * n_facilities + [CUSTOMER] * n_customers)
    dist = np.abs(xy[:, None, :] - xy[None, :, :]).sum(axis=2)
    time = dist / 30.0 * 3600.0
    ready = np.full(n, horizon[0])
    due = np.full(n, horizon[1])
    c = kind == CUSTOMER
    starts = rng.uniform(horizon[0], horizon[1] - 4 * 3600.0, size=n)
    width = rng.uniform(2 * 3600.0, 6 * 3600.0, size=n)
    ready[c] = np.round(starts[c] / 60) * 60
    due[c] = np.minimum(horizon[1] - 1800.0, np.round((starts[c] + width[c]) / 60) * 60)
    service = np.where(c, np.round(rng.uniform(120, 600, size=n)), 0.0)
    service[kind == FACILITY] = 900.0
    demand = np.where(c, np.round(rng.uniform(2, 15, size=n)), 0.0)
    return Instance(name or f"syn{seed}-{n_customers}", kind, ready, due, service, demand, dist, time,
                    capacity, route_capacity, None, rest, xy[:, 0], xy[:, 1],
                    tuple(range(n)), {"synthetic": True, "seed": seed})
