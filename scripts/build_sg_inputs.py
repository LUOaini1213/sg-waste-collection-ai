"""Derive the committed Singapore case-study inputs from data/external/sg/ (run with .venv-build).

Writes
  data/inputs/sg_casestudy_nodes.csv   depot, unloading facility and the NEA recycling bins in
                                       HDB estates inside the Bishan planning area (MP2019),
                                       with dwelling units of the matching HDB block
  data/inputs/sg_casestudy_dist_km.csv road distance of the fastest path (OSM), square matrix
  data/inputs/sg_casestudy_time_s.csv  truck travel time of that path, square matrix
  data/inputs/sg_casestudy_meta.json   counts, match rates, assumptions, source hashes

Assumptions made here (also in the README provenance table): truck speed = min(road speed, 70)
x 0.8, road speeds from the maxspeed tag or a class default; the depot is co-located with the
unloading facility; bins are snapped to the nearest drivable OSM node.
The OD matrices are a derived database of OpenStreetMap data (ODbL 1.0).
"""
from __future__ import annotations

import html
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wastecollect.io import write_csv, write_json  # noqa: E402

RAW = ROOT / "data" / "external" / "sg"
OUT = ROOT / "data" / "inputs"
TOWN = "BISHAN"
BBOX = (103.770, 1.330, 103.875, 1.475)  # west, south, east, north: town + corridor to Senoko
FACILITY_STREET = "30 Attap Valley Road"
DRIVE = {"motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link", "secondary",
         "secondary_link", "tertiary", "tertiary_link", "unclassified", "residential", "living_street", "service"}
DEFAULT_KMH = {"motorway": 90, "trunk": 70, "primary": 60, "secondary": 50, "tertiary": 50, "unclassified": 40,
               "residential": 30, "living_street": 15, "service": 15}
TRUCK_FACTOR, TRUCK_CAP_KMH = 0.8, 70.0
ABBREV = {"ST": "STREET", "AVE": "AVENUE", "RD": "ROAD", "DR": "DRIVE", "CRES": "CRESCENT", "CTRL": "CENTRAL",
          "NTH": "NORTH", "STH": "SOUTH", "UPP": "UPPER", "LOR": "LORONG", "PL": "PLACE", "CL": "CLOSE",
          "TER": "TERRACE", "BT": "BUKIT", "KG": "KAMPONG", "JLN": "JALAN", "PK": "PARK", "HTS": "HEIGHTS",
          "GDNS": "GARDENS", "CTR": "CENTRE", "TG": "TANJONG", "SQ": "SQUARE", "WALK": "WALK"}


def attrs(desc: str) -> dict:
    return {k.strip(): html.unescape(v.strip()) for k, v in re.findall(r"<th>(.*?)</th>\s*<td>(.*?)</td>", desc or "", re.S)}


def norm_street(s: str) -> str:
    return " ".join(ABBREV.get(w, w) for w in re.sub(r"[^A-Z0-9' ]", " ", str(s).upper()).split())


def town_polygon():
    from shapely.geometry import shape
    g = json.loads((RAW / "planning_area_2019.geojson").read_text(encoding="utf-8"))
    for f in g["features"]:
        p = f["properties"]
        name = p.get("PLN_AREA_N") or attrs(p.get("Description", "")).get("PLN_AREA_N")
        if name == TOWN:
            return shape(f["geometry"])
    raise KeyError(TOWN)


def load_bins(poly):
    from shapely.geometry import Point
    g = json.loads((RAW / "recycling_bins.geojson").read_text(encoding="utf-8"))
    rows = []
    for f in g["features"]:
        a = attrs(f["properties"].get("Description", ""))
        lon, lat = f["geometry"]["coordinates"][:2]
        if a.get("NAME") != "Recycling Bins in HDB Estates" or not poly.contains(Point(lon, lat)):
            continue
        rows.append({"lat": lat, "lon": lon, "block": a.get("ADDRESSBLOCKHOUSENUMBER", ""),
                     "street": a.get("ADDRESSSTREETNAME", ""),
                     "feature_updated": a.get("FMEL_UPD_D", "")[:8]})
    df = pd.DataFrame(rows).drop_duplicates(["lat", "lon"]).sort_values(["street", "block", "lat", "lon"])
    return df.reset_index(drop=True), len(rows)


def facility():
    g = json.loads((RAW / "waste_disposal_site.geojson").read_text(encoding="utf-8"))
    for f in g["features"]:
        p = f["properties"]
        if p.get("ADDRESSSTREETNAME") == FACILITY_STREET:
            lon, lat = f["geometry"]["coordinates"][:2]
            return lat, lon, p.get("DESCRIPTION", "")
    raise KeyError(FACILITY_STREET)


def haversine_m(lat1, lon1, lat2, lon2):
    r = 6371008.8
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def road_graph():
    import osmium
    w0, s0, e0, n0 = BBOX

    class H(osmium.SimpleHandler):
        def __init__(self):
            super().__init__()
            self.edges = []
            self.loc = {}

        def way(self, w):
            t = w.tags
            hw = t.get("highway")
            if hw not in DRIVE or t.get("access") in ("private", "no") or t.get("area") == "yes":
                return
            nodes = []
            for n in w.nodes:
                if not n.location.valid():
                    return
                nodes.append((n.ref, n.location.lat, n.location.lon))
            if not any(w0 <= lon <= e0 and s0 <= lat <= n0 for _, lat, lon in nodes):
                return
            base = hw.replace("_link", "")
            kmh = DEFAULT_KMH.get(base, 30) if not hw.endswith("_link") else min(DEFAULT_KMH.get(base, 30), 50)
            m = re.match(r"^\s*(\d+)", t.get("maxspeed", "") or "")
            if m:
                kmh = float(m.group(1))
            ow = t.get("oneway", "")
            one = ow in ("yes", "true", "1") or hw in ("motorway", "motorway_link") or t.get("junction") == "roundabout"
            rev = ow == "-1"
            for (a, la, lo), (b, lb, lob) in zip(nodes[:-1], nodes[1:]):
                self.loc[a] = (la, lo)
                self.loc[b] = (lb, lob)
                self.edges.append((a, b, kmh, one, rev))

    h = H()
    h.apply_file(str(RAW / "singapore.osm.pbf"), locations=True)
    ids = np.array(sorted(h.loc))
    pos = {n: k for k, n in enumerate(ids)}
    lat = np.array([h.loc[n][0] for n in ids])
    lon = np.array([h.loc[n][1] for n in ids])
    src, dst, length, ttime = [], [], [], []
    for a, b, kmh, one, rev in h.edges:
        ia, ib = pos[a], pos[b]
        d = float(haversine_m(lat[ia], lon[ia], lat[ib], lon[ib]))
        v = min(kmh, TRUCK_CAP_KMH) * TRUCK_FACTOR / 3.6
        pairs = [(ib, ia)] if rev else ([(ia, ib)] if one else [(ia, ib), (ib, ia)])
        for i, j in pairs:
            src.append(i)
            dst.append(j)
            length.append(d)
            ttime.append(d / v)
    return ids, lat, lon, np.array(src), np.array(dst), np.array(length), np.array(ttime)


def main() -> int:
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components, dijkstra
    poly = town_polygon()
    bins, n_raw = load_bins(poly)
    hdb = pd.read_csv(RAW / "hdb_property_information.csv", dtype=str)
    hdb = hdb[hdb["residential"] == "Y"].copy()
    hdb["key"] = hdb["blk_no"].str.upper().str.strip() + "|" + hdb["street"].map(norm_street)
    units = dict(zip(hdb["key"], pd.to_numeric(hdb["total_dwelling_units"], errors="coerce")))
    bins["key"] = bins["block"].str.upper().str.strip() + "|" + bins["street"].map(norm_street)
    bins["dwelling_units"] = bins["key"].map(units)
    matched = bins["dwelling_units"].notna()
    med = float(bins.loc[matched, "dwelling_units"].median())
    bins["units_source"] = np.where(matched, "HDB Property Information", "imputed: median of matched bins")
    bins["dwelling_units"] = bins["dwelling_units"].fillna(med)
    flat, flon, fdesc = facility()
    ids, lat, lon, src, dst, length, ttime = road_graph()
    n = len(ids)
    T = csr_matrix((ttime, (src, dst)), shape=(n, n))
    L = csr_matrix((length, (src, dst)), shape=(n, n))
    ncomp, lab = connected_components(T, directed=True, connection="strong")
    big = np.bincount(lab).argmax()
    keep = np.flatnonzero(lab == big)
    pts_lat = np.concatenate([[flat, flat], bins["lat"].to_numpy()])
    pts_lon = np.concatenate([[flon, flon], bins["lon"].to_numpy()])
    snap, snap_m = [], []
    for la, lo in zip(pts_lat, pts_lon):
        d = haversine_m(la, lo, lat[keep], lon[keep])
        k = int(np.argmin(d))
        snap.append(int(keep[k]))
        snap_m.append(float(d[k]))
    snap = np.array(snap)
    tt, pred = dijkstra(T, directed=True, indices=np.unique(snap), return_predecessors=True)
    row_of = {s: r for r, s in enumerate(np.unique(snap))}
    m = len(snap)
    time_s = np.zeros((m, m))
    dist_km = np.zeros((m, m))
    Lc = L.tocsr()
    for i in range(m):
        r = row_of[snap[i]]
        for j in range(m):
            if snap[i] == snap[j]:
                continue
            time_s[i, j] = tt[r, snap[j]]
            total, v = 0.0, snap[j]
            while v != snap[i]:
                u = pred[r, v]
                total += Lc[u, v]
                v = u
            dist_km[i, j] = total / 1000.0
    if not np.all(np.isfinite(time_s)):
        raise RuntimeError("unreachable pairs in the OD matrix")
    nodes = pd.DataFrame({
        "idx": np.arange(m), "kind": ["depot", "facility"] + ["bin"] * len(bins),
        "lat": pts_lat, "lon": pts_lon,
        "block": ["", ""] + bins["block"].tolist(), "street": ["", FACILITY_STREET] + bins["street"].tolist(),
        "dwelling_units": [0, 0] + bins["dwelling_units"].round(0).astype(int).tolist(),
        "units_source": ["", ""] + bins["units_source"].tolist(),
        "feature_updated": ["", ""] + bins["feature_updated"].tolist(),
        "osm_node": ids[snap], "snap_m": np.round(snap_m, 1),
    })
    write_csv(OUT / "sg_casestudy_nodes.csv", nodes, float_format="%.7f")
    write_csv(OUT / "sg_casestudy_dist_km.csv", pd.DataFrame(dist_km, columns=[str(k) for k in range(m)]), float_format="%.4f")
    write_csv(OUT / "sg_casestudy_time_s.csv", pd.DataFrame(time_s, columns=[str(k) for k in range(m)]), float_format="%.1f")
    manifest = json.loads((RAW / "MANIFEST.json").read_text(encoding="utf-8"))
    meta = {
        "town": TOWN, "bins_in_town_hdb_estates_raw": n_raw, "bins_after_dedup": int(len(bins)),
        "bins_matched_to_hdb_block": int(matched.sum()), "median_units_used_for_unmatched": med,
        "town_bins_dwelling_units_total": int(bins["dwelling_units"].sum()),
        "facility": {"street": FACILITY_STREET, "nea_description": fdesc, "lat": flat, "lon": flon},
        "depot": "ASSUMPTION: co-located with the facility",
        "road_graph": {"bbox_wsen": BBOX, "nodes": int(n), "directed_edges": int(len(src)),
                       "largest_scc_nodes": int(len(keep)), "truck_speed": f"min(road speed, {TRUCK_CAP_KMH}) x {TRUCK_FACTOR}",
                       "default_kmh": DEFAULT_KMH},
        "max_snap_m": float(np.max(snap_m)), "mean_snap_m": float(np.mean(snap_m)),
        "sources": manifest,
    }
    write_json(OUT / "sg_casestudy_meta.json", meta)
    print(json.dumps({k: v for k, v in meta.items() if k != "sources"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
