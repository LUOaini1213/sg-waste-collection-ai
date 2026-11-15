"""Download the public inputs of the Singapore case study (network access; run once).

data.gov.sg (Singapore Open Data Licence v1.0, https://data.gov.sg/open-data-licence):
  d_4dde14826642f49eefff48b7832b90db  Recycling Bins (NEA)                           GeoJSON
  d_17f5382f26140b1fdae0ba2ef6239d2f  HDB Property Information (HDB)                 CSV
  d_4765db0e87b9c86336792efe8a1f7a66  Master Plan 2019 Planning Area Boundary (URA)  GeoJSON
  d_4b6f69760d43000af97708ca7367c334  Waste Disposal Site (NEA)                      GeoJSON
OpenStreetMap data: BBBike's daily Singapore extract (.osm.pbf), (c) OpenStreetMap
contributors, ODbL 1.0 (https://download.bbbike.org/osm/bbbike/Singapore/). The public Overpass
endpoints refused automated queries when this was written (HTTP 502/504), hence the extract.

Files go to data/external/sg/ (not committed); scripts/build_sg_inputs.py derives the committed
data/inputs/sg_*. The extract changes daily, so its SHA-256 is recorded in the manifest only.

usage: python scripts/fetch_sg_inputs.py
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
OUT = ROOT / "data" / "external" / "sg"
API = "https://api-open.data.gov.sg/v1/public/api/datasets/{}/{}"
DATASETS = {
    "recycling_bins.geojson": "d_4dde14826642f49eefff48b7832b90db",
    "hdb_property_information.csv": "d_17f5382f26140b1fdae0ba2ef6239d2f",
    "planning_area_2019.geojson": "d_4765db0e87b9c86336792efe8a1f7a66",
    "waste_disposal_site.geojson": "d_4b6f69760d43000af97708ca7367c334",
}
OSM_URL = "https://download.bbbike.org/osm/bbbike/Singapore/Singapore.osm.pbf"
UA = {"User-Agent": "sg-waste-collection-ai fetch"}


def _open(url, timeout=60):
    """GET with retries; data.gov.sg answers 429 when polled too fast."""
    import urllib.error
    for attempt in range(8):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as exc:
            if exc.code != 429 or attempt == 7:
                raise
            time.sleep(15 * (attempt + 1))
    raise RuntimeError("unreachable")


def _json(url):
    return json.loads(_open(url).decode("utf-8"))


def fetch_dataset(did: str, path: Path) -> dict:
    _json(API.format(did, "initiate-download"))
    for _ in range(20):
        time.sleep(6)
        resp = _json(API.format(did, "poll-download")).get("data", {})
        url = resp.get("url")
        if url:
            data = _open(url, timeout=600)
            path.write_bytes(data)
            return {"dataset": did, "file": path.name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    raise RuntimeError(f"{did}: download not ready")


def fetch_osm(path: Path) -> dict:
    data = _open(OSM_URL, timeout=900)
    path.write_bytes(data)
    return {"url": OSM_URL, "file": path.name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {"accessed": dt.date.today().isoformat(), "files": []}
    for name, did in DATASETS.items():
        if (OUT / name).exists():
            data = (OUT / name).read_bytes()
            manifest["files"].append({"dataset": did, "file": name, "bytes": len(data),
                                      "sha256": hashlib.sha256(data).hexdigest(), "note": "already present"})
            continue
        info = fetch_dataset(did, OUT / name)
        print(info)
        manifest["files"].append(info)
        time.sleep(10)
    info = fetch_osm(OUT / "singapore.osm.pbf")
    print(info)
    manifest["files"].append(info)
    with open(OUT / "MANIFEST.json", "w", encoding="utf-8", newline="\n") as fh:
        json.dump(manifest, fh, indent=1)
        fh.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
