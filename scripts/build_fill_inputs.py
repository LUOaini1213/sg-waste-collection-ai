"""Build the committed fill-level inputs from the raw Trnava downloads.

Reads data/external/trnava/ (scripts/fetch_fill_levels.py) and writes
  data/inputs/fill_bins.csv          one row per bin: short id, waste type, station, lat, lon, volume
  data/inputs/fill_levels_daily.csv  one row per bin and local day (Europe/Bratislava):
      eod      fill level (%) in the last hourly value of the day
      max/mean daily maximum / mean of the hourly values
      changes  number of hourly values that differ from the previous hour (0 = no new reading)
      coll     number of logged collections (emptyings) that day
      stale    1 if the day lies in a run of >= 3 consecutive days without any change
               (the API repeats the last value when a sensor stops reporting)
Source: City of Trnava open data, CC BY 4.0 (see scripts/fetch_fill_levels.py).
The vendor field and street names of the source are not copied.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wastecollect.io import write_csv  # noqa: E402

RAW = ROOT / "data" / "external" / "trnava"
OUT = ROOT / "data" / "inputs"
TZ = "Europe/Bratislava"
FIRST, LAST = "2026-03-18", "2026-11-12"
TYPES = (("Komun", "residual"), ("Pap", "paper"), ("Plast", "plastic"), ("Skl", "glass"))  # Slovak bin names
TYPES_EN = {"Residual": "residual", "Paper": "paper", "Plastic": "plastic", "Glass": "glass"}
STALE_RUN = 3


def read_hourly() -> pd.DataFrame:
    parts = []
    for path in sorted(RAW.glob("hourly_*.csv")):
        wide = pd.read_csv(path, dtype=str)
        n = (len(wide.columns) - 1) // 2
        for k in range(n):
            sub = wide[["timestamp", f"items.{k}.id", f"items.{k}.level"]].dropna()
            sub.columns = ["ts", "bin", "level"]
            parts.append(sub)
    df = pd.concat(parts, ignore_index=True)
    df["level"] = pd.to_numeric(df["level"], errors="coerce")
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.dropna().drop_duplicates(["ts", "bin"]).sort_values(["bin", "ts"])
    return df


def main() -> int:
    hourly = read_hourly()
    hourly["date"] = hourly["ts"].dt.tz_convert(TZ).dt.strftime("%Y-%m-%d")
    hourly = hourly[(hourly["date"] >= FIRST) & (hourly["date"] <= LAST)].copy()
    hourly["chg"] = (hourly.groupby("bin")["level"].diff().fillna(0) != 0).astype(int)
    g = hourly.groupby(["bin", "date"])
    daily = pd.DataFrame({
        "eod": g["level"].last(), "max": g["level"].max(), "mean": g["level"].mean(),
        "changes": g["chg"].sum(), "hours": g["level"].size(),
    }).reset_index()
    # collections
    coll = pd.read_csv(RAW / "collections.csv", dtype=str)
    coll["ts"] = pd.to_datetime(coll["timestamp"], utc=True)
    coll["date"] = coll["ts"].dt.tz_convert(TZ).dt.strftime("%Y-%m-%d")
    meta = coll.dropna(subset=["item.id"]).drop_duplicates("item.id", keep="last")
    word = meta["item.name"].fillna("").str.split().str[1].fillna("")
    bin_type = {b: next((t for pre, t in TYPES if w.startswith(pre)), None) for b, w in zip(meta["item.id"], word)}
    bin_station = dict(zip(meta["item.id"], meta["item.parentId"]))
    ncoll = coll[(coll["date"] >= FIRST) & (coll["date"] <= LAST)].groupby(["item.id", "date"]).size()
    ncoll.index.names = ["bin", "date"]
    daily = daily.merge(ncoll.rename("coll").reset_index(), on=["bin", "date"], how="left")
    daily["coll"] = daily["coll"].fillna(0).astype(int)
    # stale runs: >= STALE_RUN consecutive days with no change at all
    daily = daily.sort_values(["bin", "date"]).reset_index(drop=True)
    still = (daily["changes"] == 0).to_numpy()
    stale = np.zeros(len(daily), dtype=int)
    bins = daily["bin"].to_numpy()
    k = 0
    while k < len(daily):
        j = k
        while j < len(daily) and still[j] and bins[j] == bins[k]:
            j += 1
        if j - k >= STALE_RUN:
            stale[k:j] = 1
        k = max(j, k + 1)
    daily["stale"] = stale
    # stations
    geo = json.loads((RAW / "stations.geojson").read_text(encoding="utf-8"))
    feats = geo["features"] if isinstance(geo, dict) else geo
    st = {}
    for f in feats:
        p = f["properties"]
        lon, lat = f["geometry"]["coordinates"]
        vols = {TYPES_EN.get(b.get("type"), b.get("type")): b.get("volume") for b in p.get("bins", [])}
        st[p["id"]] = (lat, lon, vols)
    ids = sorted(daily["bin"].unique(), key=lambda b: (str(bin_station.get(b, "~")), str(bin_type.get(b, "~")), b))
    rows = []
    for k, b in enumerate(ids, start=1):
        sid = bin_station.get(b)
        lat, lon, vols = st.get(sid, (np.nan, np.nan, {}))
        wt = bin_type.get(b) or "unknown"
        rows.append({"bin": k, "source_id": b, "waste_type": wt, "station": sid or "",
                     "lat": lat, "lon": lon, "volume_l": vols.get(wt, np.nan)})
    bins_df = pd.DataFrame(rows)
    short = dict(zip(bins_df["source_id"], bins_df["bin"]))
    daily["bin"] = daily["bin"].map(short)
    daily = daily.sort_values(["bin", "date"])
    daily["mean"] = daily["mean"].round(2)
    out = daily[["bin", "date", "eod", "max", "mean", "changes", "coll", "stale"]]
    write_csv(OUT / "fill_bins.csv", bins_df, float_format="%.7f")
    write_csv(OUT / "fill_levels_daily.csv", out, float_format="%.2f")
    print(f"bins {len(bins_df)} ({bins_df['waste_type'].value_counts().to_dict()}), "
          f"rows {len(out)}, days {out['date'].nunique()} {out['date'].min()}..{out['date'].max()}, "
          f"stale share {out['stale'].mean():.3f}, collections {int(out['coll'].sum())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
