"""Desk-study simulation: recycling-bin collection in one HDB town, fixed schedule vs "when needed".

Everything here is a model. Bin locations, dwelling units per block and road distances come
from public data (see data/inputs/sg_casestudy_*); fill rates, truck size, shift times and
policy rules are ASSUMPTIONS documented in DEFAULTS and in the README provenance table.

Timeline per day d (Mon..Sun): the planner decides in the morning, using sensor readings
taken at the end of day d-1, which bins to empty on day d (no collection on Sundays). Emptied
bins are assumed emptied before the day's deposits arrive. Deposits of day d are then added;
if a bin's content exceeds its volume at the end of the day, that bin-day counts as an
overflow and the excess is lost (spilled / taken elsewhere).

Policies
  fixed      NEA-style minimum service: every bin three times a week. Bins are split into two
             geographic halves served Mon/Wed/Fri and Tue/Thu/Sat (the split is an assumption).
  threshold  reactive: empty every bin whose last reading is >= 70 % of its volume.
  forecast   predictive: empty a bin when its reading plus the forecast deposits (14-day mean
             daily increment, the best baseline in the Trnava backtest) up to the end of the
             next service day would reach 90 % of its volume; also empty any bin not emptied
             for 7 days.
Daily routes are planned with the ALNS of wastecollect.routing (depot and unloading facility,
time windows of the shift, one-hour break starting 11:00-12:00, truck capacity in bin volumes),
using the true volumes in the selected bins (perfect knowledge of the day's amounts).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from ..routing.alns import Params, solve
from ..routing.checker import evaluate
from ..routing.instance import CUSTOMER, DEPOT, FACILITY, Instance

H = 3600.0


@dataclass
class SimConfig:
    days_warmup: int = 14
    days_measured: int = 42                 # six weeks
    start_dow: int = 0                      # day 0 is a Monday
    base_rate: float = 0.20                 # bin volumes per day for a block with the town-mean dwelling units (ASSUMPTION)
    units_clip: tuple = (0.5, 2.0)          # per-bin rate = base x clip(units / mean units) (ASSUMPTION)
    dow_factors: tuple = (1.0,) * 7         # weekday shape (ILLUSTRATIVE, from Trnava recyclables bins)
    noise_cv: float = 0.5                   # day-to-day variability of deposits (ILLUSTRATIVE, Trnava)
    sensor_sd: float = 0.03                 # sensor reading noise, fraction of volume (ASSUMPTION)
    truck_capacity: float = 40.0            # bin volumes per load, after compaction (ASSUMPTION)
    service_s: float = 180.0                # per bin lift (ASSUMPTION)
    unload_s: float = 1200.0                # at the facility (ASSUMPTION)
    shift: tuple = (7 * H, 17 * H)          # leave depot / back by (ASSUMPTION)
    rest: tuple = (11 * H, 12 * H, H)       # break start window and duration (ASSUMPTION)
    threshold: float = 0.70
    forecast_level: float = 0.90
    max_gap_days: int = 7
    rate_window: int = 14
    alns_iters: int = 150
    seed: int = 1
    policies: tuple = ("fixed", "threshold", "forecast")
    co2_kg_per_km: float = 0.99156          # DESNZ 2025, HGV rigid >17 t, average laden (REAL factor)


@dataclass
class DayLog:
    policy: str
    day: int
    dow: int
    bins: int
    load: float
    routes: int
    unloads: int
    km: float
    feasible: bool
    overflow_bins: int
    overflow_volume: float
    mean_fill_at_collection: float
    work_hours: float = 0.0   # driving + lifting + unloading time of the day's routes (break excluded)


@dataclass
class SimResult:
    config: dict
    days: list = field(default_factory=list)

    def summary(self, measured_from: int) -> dict:
        out = {}
        for pol in self.config["policies"]:
            rows = [d for d in self.days if d.policy == pol and d.day >= measured_from]
            lifts = sum(d.bins for d in rows)
            km = sum(d.km for d in rows)
            coll_days = [d for d in rows if d.bins > 0]
            out[pol] = {
                "km": km, "lifts": lifts, "loads_collected": sum(d.load for d in rows),
                "truck_days": sum(d.routes for d in rows),
                "max_trucks_per_day": max((d.routes for d in rows), default=0),
                "mean_trucks_per_service_day": float(np.mean([d.routes for d in coll_days])) if coll_days else 0.0,
                "unloads": sum(d.unloads for d in rows),
                "overflow_bin_days": sum(d.overflow_bins for d in rows),
                "overflow_volume": sum(d.overflow_volume for d in rows),
                "work_hours": sum(d.work_hours for d in rows),
                "mean_fill_at_collection": (sum(d.mean_fill_at_collection * d.bins for d in rows) / lifts) if lifts else 0.0,
                "co2_kg": km * self.config["co2_kg_per_km"],
                "all_routes_feasible": all(d.feasible for d in rows),
            }
        return out


def make_instance(nodes: dict, dist, time, bins, amounts, cfg: SimConfig) -> Instance:
    """Day instance: depot, facility, and the selected bins (indices into the bin list)."""
    idx = np.concatenate([[nodes["depot"], nodes["facility"]], nodes["bins"][bins]])
    n = len(idx)
    kind = np.array([DEPOT, FACILITY] + [CUSTOMER] * len(bins))
    ready = np.full(n, cfg.shift[0])
    due = np.full(n, cfg.shift[1])
    due[1] = 24 * H
    ready[1] = 0.0
    service = np.where(kind == CUSTOMER, cfg.service_s, 0.0)
    service[1] = cfg.unload_s
    demand = np.zeros(n)
    demand[2:] = amounts
    ix = np.ix_(idx, idx)
    return Instance("day", kind, ready, due, service, demand, dist[ix], time[ix], cfg.truck_capacity,
                    rest=cfg.rest)


def _fixed_groups(lon: np.ndarray) -> np.ndarray:
    """0 = Mon/Wed/Fri, 1 = Tue/Thu/Sat: split bins into west and east halves by longitude."""
    order = np.argsort(lon, kind="stable")
    g = np.zeros(len(lon), dtype=int)
    g[order[len(lon) // 2:]] = 1
    return g


def simulate(nodes: dict, dist, time, units: np.ndarray, lon: np.ndarray, cfg: SimConfig) -> SimResult:
    rng = np.random.default_rng(cfg.seed)
    nb = len(units)
    days = cfg.days_warmup + cfg.days_measured
    mean_rate = cfg.base_rate * np.clip(units / units.mean(), *cfg.units_clip)
    # common random numbers: the same deposits for every policy; rounded so that last-bit
    # differences of exp/log between C libraries cannot change a decision
    sigma = np.sqrt(np.log(1 + cfg.noise_cv ** 2))
    eps = rng.lognormal(-sigma ** 2 / 2, sigma, size=(days, nb))
    dowf = np.array(cfg.dow_factors) / np.mean(cfg.dow_factors)
    deposits = np.round(np.array([mean_rate * dowf[(cfg.start_dow + d) % 7] for d in range(days)]) * eps, 6)
    sensor_noise = np.round(rng.normal(0.0, cfg.sensor_sd, size=(days, nb)), 6)
    groups = _fixed_groups(lon)
    res = SimResult(asdict(cfg))
    for pol in cfg.policies:
        level = np.zeros(nb)
        reading = np.zeros(nb)
        last_coll = np.zeros(nb, dtype=int)
        hist = []  # observed daily increments (nan on collection days)
        for d in range(days):
            dow = (cfg.start_dow + d) % 7
            sel = np.zeros(nb, dtype=bool)
            if dow != 6:  # no collection on Sundays
                if pol == "fixed":
                    sel = groups == (0 if dow in (0, 2, 4) else 1)
                elif pol == "threshold":
                    sel = reading >= cfg.threshold
                elif pol == "forecast":
                    inc = np.array(hist[-cfg.rate_window:]) if hist else np.zeros((1, nb))
                    with np.errstate(invalid="ignore"):
                        cnt = np.sum(~np.isnan(inc), axis=0)
                        r = np.where(cnt > 0, np.nansum(inc, axis=0) / np.maximum(cnt, 1), 0.0)
                    # deposit days until the next chance to collect (Saturday -> Monday: two)
                    ahead = 2 if dow == 5 else 1
                    h_eff = sum(dowf[(dow + j) % 7] for j in range(ahead))
                    sel = (reading + r * h_eff >= cfg.forecast_level) | (d - last_coll >= cfg.max_gap_days)
            amounts = np.minimum(level[sel], 1.0)
            day_bins = np.flatnonzero(sel)
            km = hours = 0.0
            routes = unloads = 0
            feasible = True
            if len(day_bins) and d >= cfg.days_warmup:
                inst = make_instance(nodes, dist, time, day_bins, np.maximum(amounts, 1e-6), cfg)
                sol, _ = solve(inst, cfg.alns_iters, cfg.seed * 1000 + d, Params())
                ev = evaluate(inst, sol.as_lists())
                km, routes, unloads, feasible = ev.distance, ev.n_routes, ev.n_unloads, ev.feasible
                lists = sol.as_lists()
                drive = sum(float(inst.time[a, b]) for r in lists for a, b in zip(r[:-1], r[1:]))
                hours = (drive + sum(float(inst.service[v]) for r in lists for v in r)) / H
            fill_at_coll = float(level[sel].mean()) if sel.any() else 0.0
            level[sel] = 0.0
            last_coll[sel] = d
            level = level + deposits[d]
            over = np.maximum(level - 1.0, 0.0)
            n_over = int((over > 0).sum())
            level = np.minimum(level, 1.0)
            new_reading = np.clip(level + sensor_noise[d], 0.0, 1.0)
            inc_obs = new_reading - reading
            inc_obs[sel] = np.nan
            hist.append(inc_obs)
            reading = new_reading
            res.days.append(DayLog(pol, d, dow, int(sel.sum()), float(amounts.sum()), routes, unloads, km,
                                   feasible, n_over, float(over.sum()), fill_at_coll, hours))
    return res
