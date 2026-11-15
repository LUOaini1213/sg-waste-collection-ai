"""Download the City of Trnava (Slovakia) smart-bin open data used for the fill-level backtest.

Dataset: "Smart bin fill level history" (historia-naplnenosti-smart-kontajnerov) and the
smart-bin collection log, City of Trnava open-data portal, licence CC BY 4.0
(https://opendata.trnava.sk/en/dataset/historia-naplnenosti-smart-kontajnerov ,
terms of use https://opendata.trnava.sk/en/terms-of-use , attribution "City of Trnava").
The portal API is public (no key). period=day returns hourly values.

Window: local days 2026-03-18 .. 2026-11-12 (the feed was frozen from 2024-02-01 to 2026-03-17,
and the last two days are revised after publication, so they are left out).
Raw files go to data/external/trnava/ (not committed); scripts/build_fill_inputs.py turns them
into the committed data/inputs/fill_*.csv. A manifest with SHA-256 values is written next to
the raw files; the live API may revise values, so the committed inputs are the reference.

usage: python scripts/fetch_fill_levels.py
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "external" / "trnava"
API = "https://opendata.trnava.sk/api/datasets"
START_UTC = dt.datetime(2026, 3, 17, 22, 0)   # local (CET/CEST) day 2026-03-18 starts 23:00Z; 22:00Z keeps one hour margin
END_UTC = dt.datetime(2026, 11, 12, 23, 0)    # local day 2026-11-12 ends 23:00Z (CET)


def _get(url: str, path: Path) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "sg-waste-collection-ai fetch"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                data = resp.read()
            path.write_bytes(data)
            return {"url": url, "file": path.name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        except Exception as exc:  # network hiccup: retry
            if attempt == 3:
                raise
            print("retry", url, exc)
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("unreachable")


def _iso(t: dt.datetime) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%S.000Z")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {"accessed": dt.date.today().isoformat(), "licence": "CC BY 4.0 (City of Trnava)", "files": []}
    t = START_UTC
    k = 0
    while t < END_UTC:
        t2 = min(t + dt.timedelta(days=31), END_UTC)
        url = f"{API}/v2/smart-bins/items-history?outputType=csv&period=day&from={_iso(t)}&to={_iso(t2 - dt.timedelta(milliseconds=1))}"
        info = _get(url, OUT / f"hourly_{k:02d}.csv")
        manifest["files"].append(info)
        print(info["file"], info["bytes"])
        t = t2
        k += 1
    url = f"{API}/v1/smart-bins/collection/list?outputType=csv&from={_iso(START_UTC)}&to={_iso(END_UTC)}&limit=100000"
    manifest["files"].append(_get(url, OUT / "collections.csv"))
    url = f"{API}/v1/city-dashboard/smart-bins/geojson/list"
    manifest["files"].append(_get(url, OUT / "stations.geojson"))
    with open(OUT / "MANIFEST.json", "w", encoding="utf-8", newline="\n") as fh:
        json.dump(manifest, fh, indent=1)
        fh.write("\n")
    print("wrote", OUT / "MANIFEST.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
