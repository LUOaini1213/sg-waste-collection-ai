import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def assert_close(got, want, rel=1e-9, abs_tol=1e-9, path="$"):
    """Recursive comparison of JSON-like structures; numbers with a tolerance, everything else exactly."""
    if isinstance(want, dict):
        assert isinstance(got, dict) and set(got) == set(want), f"{path}: keys {sorted(got)} != {sorted(want)}"
        for k in want:
            assert_close(got[k], want[k], rel, abs_tol, f"{path}.{k}")
    elif isinstance(want, (list, tuple)):
        assert isinstance(got, (list, tuple)) and len(got) == len(want), f"{path}: length differs"
        for i, (g, w) in enumerate(zip(got, want)):
            assert_close(g, w, rel, abs_tol, f"{path}[{i}]")
    elif isinstance(want, bool) or want is None or isinstance(want, str):
        assert got == want, f"{path}: {got!r} != {want!r}"
    else:
        assert got is not None and abs(float(got) - float(want)) <= abs_tol + rel * abs(float(want)), f"{path}: {got} != {want}"
