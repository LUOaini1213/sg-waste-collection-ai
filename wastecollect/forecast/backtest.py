"""Fill-level forecasting: baselines vs gradient boosting in a rolling-origin backtest.

Question asked of every method: at the end of day t, how full will a bin be at the end of
day t+h (h = 1..3) if it is NOT emptied in between? That is the quantity a "collect when
needed" planner needs. Evaluation therefore uses only collection-free windows: no logged
collection and no daily drop of DROP_PP percentage points or more on days t+1..t+h (unlogged
emptyings), and no stale day (sensor repeating its last value) on days t..t+h.

Methods (all clipped to [0, 100]):
  persistence     y_t
  seasonal_naive  y_t + sum of the same-weekday daily increments one week earlier
                  (falls back to the recent mean rate when such an increment is not valid)
  mean_rate       y_t + h * mean valid daily increment over the last 14 days
  lightgbm        y_t + predicted increment; one LightGBM model over all bins and horizons,
                  retrained at every origin on samples whose target lies before the origin
Increments are only "valid" on collection-free, non-stale days.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

DROP_PP = 20.0
HORIZONS = (1, 2, 3)
METHODS = ("persistence", "seasonal_naive", "mean_rate", "lightgbm")
TYPES = ("residual", "paper", "plastic", "glass", "unknown")


@dataclass
class Config:
    first_origin: str = "2026-06-01"
    last_day: str = "2026-11-12"
    origin_step: int = 7
    bins: tuple | None = None          # restrict to these bin ids (reduced test runs)
    max_origins: int | None = None
    n_estimators: int = 300
    learning_rate: float = 0.05
    num_leaves: int = 31
    min_child_samples: int = 50
    threads: int = 1
    seed: int = 0
    full_threshold: float = 80.0


def load_panel(levels: pd.DataFrame, bins: pd.DataFrame, cfg: Config):
    """Wide arrays (bin x day) from the long daily table."""
    df = levels.copy()
    if cfg.bins is not None:
        df = df[df["bin"].isin(cfg.bins)]
    days = pd.date_range(df["date"].min(), df["date"].max(), freq="D").strftime("%Y-%m-%d")
    ids = np.sort(df["bin"].unique())
    bi = {b: k for k, b in enumerate(ids)}
    di = {d: k for k, d in enumerate(days)}
    shape = (len(ids), len(days))
    y = np.full(shape, np.nan)
    coll = np.zeros(shape)
    stale = np.ones(shape, dtype=bool)
    r, c = df["bin"].map(bi).to_numpy(), df["date"].map(di).to_numpy()
    y[r, c] = df["eod"].to_numpy()
    coll[r, c] = df["coll"].to_numpy()
    stale[r, c] = df["stale"].to_numpy().astype(bool)
    wt = bins.set_index("bin")["waste_type"].reindex(ids).fillna("unknown").to_numpy()
    return {"ids": ids, "days": np.array(days), "y": y, "coll": coll, "stale": stale, "type": wt}


def daily_increments(P):
    """Valid daily increments (nan where the day had a collection, a big drop or is stale)."""
    y, coll, stale = P["y"], P["coll"], P["stale"]
    inc = np.full(y.shape, np.nan)
    inc[:, 1:] = y[:, 1:] - y[:, :-1]
    bad = (coll > 0) | stale | np.isnan(inc)
    bad[:, 1:] |= stale[:, :-1]
    with np.errstate(invalid="ignore"):
        bad |= inc <= -DROP_PP
    inc[bad] = np.nan
    return inc


def _rolling_mean(inc, t, w):
    win = inc[:, max(0, t - w + 1): t + 1]
    with np.errstate(invalid="ignore"):
        cnt = np.sum(~np.isnan(win), axis=1)
        s = np.nansum(win, axis=1)
    return np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)


def _nanstd(win):
    cnt = np.sum(~np.isnan(win), axis=1)
    mu = np.nansum(win, axis=1) / np.maximum(cnt, 1)
    var = np.nansum((win - mu[:, None]) ** 2, axis=1) / np.maximum(cnt, 1)
    return np.where(cnt > 1, np.sqrt(var), np.nan)


def build_samples(P, inc):
    """One row per (bin, origin day t, horizon h) with a collection-free, non-stale target window."""
    y, coll, stale = P["y"], P["coll"], P["stale"]
    nb, nd = y.shape
    days = pd.to_datetime(P["days"])
    dow = days.dayofweek.to_numpy()
    rows = []
    type_code = np.array([TYPES.index(t) if t in TYPES else len(TYPES) - 1 for t in P["type"]])
    last_coll = np.full(nb, -999)
    for t in range(nd):
        last_coll = np.where(coll[:, t] > 0, t, last_coll)
        if t < 28:
            continue
        m7 = _rolling_mean(inc, t, 7)
        m14 = _rolling_mean(inc, t, 14)
        m28 = _rolling_mean(inc, t, 28)
        sd28 = _nanstd(inc[:, t - 27: t + 1])
        lags = [inc[:, t - k] for k in range(7)]
        for h in HORIZONS:
            if t + h >= nd:
                continue
            seg = slice(t + 1, t + h + 1)
            with np.errstate(invalid="ignore"):
                ok = (~stale[:, t]) & ~np.isnan(y[:, t]) & ~np.isnan(y[:, t + h])
                ok &= ~stale[:, seg].any(axis=1) & (coll[:, seg].sum(axis=1) == 0)
                steps = np.diff(y[:, t: t + h + 1], axis=1)
                ok &= ~(steps <= -DROP_PP).any(axis=1)
            if not ok.any():
                continue
            sn = np.zeros(nb)
            for k in range(1, h + 1):
                v = inc[:, t + k - 7]
                sn += np.where(np.isnan(v), np.nan_to_num(m14, nan=0.0), v)
            b = np.flatnonzero(ok)
            rows.append(pd.DataFrame({
                "bin": P["ids"][b], "t": t, "h": h, "date": P["days"][t],
                "y_t": y[b, t], "target": y[b, t + h],
                "m7": m7[b], "m14": m14[b], "m28": m28[b], "sd28": sd28[b],
                **{f"lag{k}": lags[k][b] for k in range(7)},
                "sn_inc": sn[b], "dow_next": dow[t + 1] if t + 1 < nd else 0,
                "days_since_coll": np.minimum(t - last_coll[b], 60),
                "type": type_code[b],
            }))
    return pd.concat(rows, ignore_index=True)


FEATURES = ["y_t", "h", "m7", "m14", "m28", "sd28"] + [f"lag{k}" for k in range(7)] + \
    ["sn_inc", "dow_next", "days_since_coll", "type", "bin_rate"]


def baselines(S):
    out = {}
    out["persistence"] = S["y_t"].to_numpy()
    out["seasonal_naive"] = S["y_t"].to_numpy() + S["sn_inc"].to_numpy()
    out["mean_rate"] = S["y_t"].to_numpy() + S["h"].to_numpy() * np.nan_to_num(S["m14"].to_numpy(), nan=0.0)
    return {k: np.clip(v, 0, 100) for k, v in out.items()}


def fit_predict_lgbm(train, test, cfg: Config):
    import lightgbm as lgb
    rate = train.assign(r=(train["target"] - train["y_t"]) / train["h"]).groupby("bin")["r"].mean()
    train = train.assign(bin_rate=train["bin"].map(rate))
    test = test.assign(bin_rate=test["bin"].map(rate))
    model = lgb.LGBMRegressor(
        n_estimators=cfg.n_estimators, learning_rate=cfg.learning_rate, num_leaves=cfg.num_leaves,
        min_child_samples=cfg.min_child_samples, subsample=1.0, colsample_bytree=1.0,
        reg_lambda=1.0, random_state=cfg.seed, deterministic=True, force_row_wise=True,
        n_jobs=cfg.threads, verbose=-1)
    model.fit(train[FEATURES], train["target"] - train["y_t"], categorical_feature=["type"])
    pred = model.predict(test[FEATURES])
    imp = dict(zip(FEATURES, model.booster_.feature_importance(importance_type="gain").tolist()))
    return np.clip(test["y_t"].to_numpy() + pred, 0, 100), imp


def run_backtest(levels: pd.DataFrame, bins: pd.DataFrame, cfg: Config | None = None):
    cfg = cfg or Config()
    P = load_panel(levels, bins, cfg)
    inc = daily_increments(P)
    S = build_samples(P, inc)
    days = P["days"]
    di = {d: k for k, d in enumerate(days)}
    o0 = di[cfg.first_origin]
    last = di.get(cfg.last_day, len(days) - 1)
    origins = list(range(o0, last, cfg.origin_step))
    if cfg.max_origins:
        origins = origins[: cfg.max_origins]
    preds = []
    importances = []
    for o in origins:
        train = S[S["t"] + S["h"] < o]
        test = S[(S["t"] >= o) & (S["t"] < o + cfg.origin_step) & (S["t"] + S["h"] <= last)]
        if len(test) == 0 or len(train) == 0:
            continue
        p = baselines(test)
        p["lightgbm"], imp = fit_predict_lgbm(train, test, cfg)
        importances.append(imp)
        frame = test[["bin", "t", "h", "date", "y_t", "target", "type"]].copy()
        frame["origin"] = days[o]
        for k in METHODS:
            frame[k] = p[k]
        preds.append(frame)
    pred = pd.concat(preds, ignore_index=True)
    return pred, importances, {"n_samples_total": int(len(S)), "origins": [days[o] for o in origins]}


def metrics(pred: pd.DataFrame, threshold: float = 80.0) -> dict:
    out = {}
    y = pred["target"].to_numpy()
    for m in METHODS:
        e = pred[m].to_numpy() - y
        tp = np.sum((pred[m] >= threshold) & (pred["target"] >= threshold))
        fp = np.sum((pred[m] >= threshold) & (pred["target"] < threshold))
        fn = np.sum((pred[m] < threshold) & (pred["target"] >= threshold))
        out[m] = {"mae": float(np.mean(np.abs(e))), "rmse": float(np.sqrt(np.mean(e ** 2))),
                  "bias": float(np.mean(e)), "n": int(len(e)),
                  "full_precision": float(tp / (tp + fp)) if tp + fp else None,
                  "full_recall": float(tp / (tp + fn)) if tp + fn else None}
    base = out["persistence"]["mae"]
    for m in METHODS:
        out[m]["mae_vs_persistence"] = out[m]["mae"] / base if base else None
    return out


def cluster_bootstrap_diff(pred: pd.DataFrame, a: str, b: str, n_boot: int = 1000, seed: int = 0) -> dict:
    """95% interval for MAE(a) - MAE(b), resampling whole bins (errors within a bin are correlated)."""
    rng = np.random.default_rng(seed)
    g = pred.assign(ea=(pred[a] - pred["target"]).abs(), eb=(pred[b] - pred["target"]).abs()).groupby("bin")
    sa, sb, n = g["ea"].sum().to_numpy(), g["eb"].sum().to_numpy(), g.size().to_numpy()
    k = len(n)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, k, size=k)
        diffs[i] = (sa[idx].sum() - sb[idx].sum()) / n[idx].sum()
    point = (sa.sum() - sb.sum()) / n.sum()
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"a": a, "b": b, "mae_diff": float(point), "ci95": [float(lo), float(hi)], "n_bins": int(k)}
