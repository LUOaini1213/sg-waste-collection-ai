"""Download the Kim, Kim & Sahoo (2006) WCVRPTW benchmark instances (not redistributed here).

Source: GitHub mirror israelpereira55/WCVRPTW-benchmark, pinned to one commit. The mirror's
README states that the benchmark author permitted the mirror and asks users to cite
Kim, Kim & Sahoo (2006); it carries no licence file, so this repository does not commit the
instance files. Files land in data/external/wcvrptw/ and are checked against SHA-256 values
recorded when this script was written (2026-11-15).

usage: python scripts/fetch_wcvrptw.py [--verify]
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "external" / "wcvrptw"
COMMIT = "ad4b010c096349bc3e333c0327de79e533ffac27"
BASE = f"https://raw.githubusercontent.com/israelpereira55/WCVRPTW-benchmark/{COMMIT}/instances/"
SHA256 = {
    "102_stop.txt": "9112ec7e4a12b62ee6efc174be99b5d4f0f655629e9fcc4d9c8a7a7abe598c4c",
    "277_stop.txt": "6a929fe6f4b857adb1ce55af7bbee9fb0706ad8a3f081e198da8daa4a1192290",
    "335_stop.txt": "0e20225c8069c9b3e8cef2f6bfe5b979be9a89fbf026f27004b6964789f52209",
    "444_stop.txt": "e4759c6a5befe63b7a7f67a18ccd66e5dc6f555dd9c71e15c76550cb8ae0f568",
    "804_stop.txt": "3d9c4c5b5b3f18e07fc277942fd68b675df3bfcdf6c9ef4ac0c2e18e0fd17f0b",
    "1051_stop.txt": "07f1a58e87c262cfd4d4655af842b6be11dfbd6c8f773bebb183d8b4567e36d9",
    "1351_stop.txt": "5b28d891215c0214285999e1a0739ee58c68c3cb2f6a331b0e40d85966f2a37e",
    "1599_stop.txt": "7922b2762d3dd328e56462a462add2859816d3db881a9f2f0effd1fb5af38414",
    "1932_stop.txt": "37c5b43d0e1cf6051e5ab0a5e3216ac5e1daa22f1716d36adc6a3cdb77945188",
    "2100_stop.txt": "e204317e85301f08fd52fd95dea47eb72345fac92164f8c6b5717a81a072c8ce",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true", help="only check files already downloaded")
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    bad = 0
    for name, want in SHA256.items():
        path = OUT / name
        if not args.verify and not path.exists():
            req = urllib.request.Request(BASE + name, headers={"User-Agent": "sg-waste-collection-ai fetch"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                path.write_bytes(resp.read())
        if not path.exists():
            print(f"missing  {name}")
            bad += 1
            continue
        got = sha256(path)
        status = "ok" if want is None or got == want else "MISMATCH"
        if status != "ok":
            bad += 1
        print(f"{status:8s} {name} {got}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
