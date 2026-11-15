"""Small I/O helpers: deterministic JSON/CSV writing with LF line endings."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _clean(obj, ndigits):
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return round(obj, ndigits)
    if isinstance(obj, dict):
        return {str(k): _clean(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v, ndigits) for v in obj]
    if hasattr(obj, "item") and callable(obj.item):  # numpy scalar
        return _clean(obj.item(), ndigits)
    return obj


def write_json(path, obj, ndigits: int = 6) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(_clean(obj, ndigits), indent=1, sort_keys=False, ensure_ascii=False) + "\n"
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


def read_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def write_text(path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


def write_csv(path, df, float_format="%.6g") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        df.to_csv(fh, index=False, float_format=float_format, lineterminator="\n")


def sha256_lf(path) -> str:
    """SHA-256 of a text file with CRLF normalised to LF (stable across Windows/Linux checkouts)."""
    data = Path(path).read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()
