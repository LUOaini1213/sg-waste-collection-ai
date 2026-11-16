"""DuckDB KPI pipeline: raw result tables -> KPI tables (kpi.sql) -> CSV files for the dashboard."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

SQL = Path(__file__).with_name("kpi.sql")
TABLES = ("kpi_policy", "kpi_weekly", "kpi_bins_observed", "kpi_forecast", "kpi_routing")
METHODS = ("persistence", "seasonal_naive", "mean_rate", "lightgbm")


def statements(sql: str) -> list[str]:
    body = "\n".join(line for line in sql.splitlines() if not line.strip().startswith("--"))
    return [s.strip() for s in body.split(";") if s.strip()]


def run_kpis(raw: dict) -> dict:
    """raw: name -> DataFrame for daily_ops, sim_meta, fill_levels, fill_bins, forecast_metrics, routing."""
    import duckdb
    con = duckdb.connect()
    try:
        for name, df in raw.items():
            con.register(f"{name}_df", df)
            con.execute(f"CREATE TABLE {name} AS SELECT * FROM {name}_df")
        for st in statements(SQL.read_text(encoding="utf-8")):
            con.execute(st)
        return {t: con.execute(f"SELECT * FROM {t}").df() for t in TABLES}
    finally:
        con.close()


def forecast_table(res: dict) -> pd.DataFrame:
    rows = []
    for h, block in res["by_horizon"].items():
        for m in METHODS:
            v = block[m]
            rows.append({"method": m, "horizon": int(h), "mae": v["mae"], "rmse": v["rmse"], "bias": v["bias"], "n": v["n"]})
    return pd.DataFrame(rows)


def routing_table(res: dict | None) -> pd.DataFrame:
    cols = ["instance", "customers", "best", "mean", "reference", "reference_source", "gap_best_pct", "gap_mean_pct"]
    if not res:
        return pd.DataFrame({c: pd.Series(dtype="float64" if c not in ("instance", "reference_source") else "object") for c in cols})
    return pd.DataFrame([{c: r[c] for c in cols} for r in res["rows"]])


def raw_from_repo(root: Path) -> dict:
    res_dir = root / "results"
    cs = json.loads((res_dir / "casestudy.json").read_text(encoding="utf-8"))
    cfg = cs["scenarios"]["base"]["config"]
    routing = res_dir / "routing_benchmark.json"
    fc = json.loads((res_dir / "forecast_backtest.json").read_text(encoding="utf-8"))
    return {
        "daily_ops": pd.read_csv(res_dir / "casestudy_daily.csv"),
        "sim_meta": pd.DataFrame([{"warmup_days": int(cfg["days_warmup"]), "co2_kg_per_km": float(cfg["co2_kg_per_km"])}]),
        "fill_levels": pd.read_csv(root / "data" / "inputs" / "fill_levels_daily.csv"),
        "fill_bins": pd.read_csv(root / "data" / "inputs" / "fill_bins.csv")[["bin", "waste_type", "lat", "lon"]],
        "forecast_metrics": forecast_table(fc),
        "routing": routing_table(json.loads(routing.read_text(encoding="utf-8")) if routing.exists() else None),
    }
