"""Download TrashNet (Thung & Yang, 2016) for the edge classifier. Images are not committed.

Source: https://github.com/garythung/trashnet (MIT License, Copyright (c) 2017 Gary Thung),
file data/dataset-resized.zip at a pinned commit: 2,527 images (512 x 384) in six classes
(glass, paper, cardboard, plastic, metal, trash). The repository asks users to cite it.
Unzips to data/external/trashnet/dataset-resized/<class>/*.jpg and checks the SHA-256.

usage: python scripts/fetch_trashnet.py
"""
from __future__ import annotations

import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "external" / "trashnet"
COMMIT = "6fa2b878c6c1b4304b91109070ce0edf9279bb31"
URL = f"https://raw.githubusercontent.com/garythung/trashnet/{COMMIT}/data/dataset-resized.zip"
SHA256 = "0bf472790f8b20e5c950d5b5012a9d38af0d3392efd65f8ce171334fc16b07c2"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    z = OUT / "dataset-resized.zip"
    if not z.exists():
        req = urllib.request.Request(URL, headers={"User-Agent": "sg-waste-collection-ai fetch"})
        with urllib.request.urlopen(req, timeout=600) as r:
            z.write_bytes(r.read())
    got = hashlib.sha256(z.read_bytes()).hexdigest()
    print("sha256", got)
    if SHA256 is not None and got != SHA256:
        print("SHA-256 mismatch")
        return 1
    with zipfile.ZipFile(z) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".jpg") and "__MACOSX" not in n]
        for n in names:
            target = OUT / n
            if not target.resolve().is_relative_to(OUT.resolve()):
                raise ValueError(f"unsafe path in zip: {n}")
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(zf.read(n))
    print(len(names), "images under", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
