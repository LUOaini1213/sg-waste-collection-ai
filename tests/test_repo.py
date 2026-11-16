"""Repository hygiene: generated README blocks, pinned requirements, line endings, provenance labels."""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

import pytest

from wastecollect.io import sha256_lf, write_json, write_text

ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".py", ".md", ".txt", ".json", ".csv", ".sql", ".yml", ".toml", ".html"}
TEXT_NAMES = {".gitattributes", ".gitignore", "LICENSE"}


def tracked_files():
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True)
    if out.returncode != 0 or not out.stdout.strip():
        pytest.skip("not a git checkout")
    return [ROOT / line for line in out.stdout.splitlines()]


def test_readme_blocks_are_generated_from_results():
    import render_readme
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert render_readme.render(text) == text, "run python scripts/render_readme.py"
    for name in render_readme.BLOCKS:
        assert f"<!-- BEGIN:{name} -->" in text and f"<!-- END:{name} -->" in text


def test_readme_provenance_table_labels_every_input():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    section = text.split("## Data provenance")[1].split("\n## ")[0]
    rows = [line for line in section.splitlines() if line.startswith("|") and not set(line) <= set("|- ")][1:]
    assert len(rows) >= 15
    for line in rows:
        assert re.search(r"\b(REAL|ASSUMPTION|ILLUSTRATIVE|SYNTHETIC)\b", line), line
    for label in ("REAL", "ASSUMPTION", "ILLUSTRATIVE", "SYNTHETIC"):
        assert label in section
    assert "not verified" in text


def test_requirements_are_pinned():
    for name in ("requirements.txt", "requirements-edge.txt", "requirements-build.txt"):
        lines = [x.strip() for x in (ROOT / name).read_text(encoding="utf-8").splitlines()
                 if x.strip() and not x.startswith("#")]
        assert lines and all(re.fullmatch(r"[A-Za-z0-9_.\-]+==[0-9][A-Za-z0-9.+]*", x) for x in lines), name


def test_text_files_use_lf_and_gitattributes_rules():
    ga = (ROOT / ".gitattributes").read_text(encoding="utf-8")
    assert "* text=auto eol=lf" in ga and "*.png binary" in ga and "*.onnx binary" in ga
    bad = [str(p.relative_to(ROOT)) for p in tracked_files()
           if (p.suffix in TEXT_SUFFIXES or p.name in TEXT_NAMES) and p.exists() and b"\r\n" in p.read_bytes()]
    assert not bad, f"CRLF line endings in {bad[:5]}"


def test_no_raw_downloads_or_large_files_are_tracked():
    files = tracked_files()
    rel = [str(p.relative_to(ROOT)).replace("\\", "/") for p in files]
    assert not [r for r in rel if r.startswith("data/external/")]
    assert not [r for r in rel if r.lower().endswith((".jpg", ".jpeg", ".pbf", ".zip", ".pt", ".pth"))]
    assert all(p.stat().st_size < 20 * 1024 * 1024 for p in files if p.exists())


def test_licence_and_facts_file():
    lic = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert lic.startswith("MIT License") and "Copyright (c) 2026 Wenjie Luo (LUOaini1213)" in lic
    facts = json.loads((ROOT / "data" / "inputs" / "sg_facts.json").read_text(encoding="utf-8"))
    assert all(f["url"].startswith("https://") and f["source"] for f in facts["facts"]) and facts["not_verified"]


def test_sha256_lf_ignores_crlf(tmp_path):
    a, b = tmp_path / "a.txt", tmp_path / "b.txt"
    a.write_bytes(b"x,y\n1,2\n")
    b.write_bytes(b"x,y\r\n1,2\r\n")
    assert sha256_lf(a) == sha256_lf(b) == hashlib.sha256(b"x,y\n1,2\n").hexdigest()


def test_writers_emit_lf_and_round(tmp_path):
    p = tmp_path / "x.json"
    write_json(p, {"a": 1.23456789, "b": [float("nan"), 2], "c": {"d": 1e-9}})
    raw = p.read_bytes()
    assert b"\r" not in raw and raw.endswith(b"\n")
    assert json.loads(raw) == {"a": 1.234568, "b": [None, 2], "c": {"d": 0.0}}
    q = tmp_path / "y.txt"
    write_text(q, "l1\nl2\n")
    assert q.read_bytes() == b"l1\nl2\n"


def test_stored_benchmark_routes_recheck_when_instances_are_present():
    """The 30 stored runs exist; with the (fetched, not committed) instance files they are re-checked."""
    data = ROOT / "data" / "external" / "wcvrptw"
    runs = sorted((ROOT / "results" / "routing" / "runs").glob("*.json"))
    assert len(runs) == 30
    if not (data / "102_stop.txt").exists():
        pytest.skip("benchmark instances not downloaded (scripts/fetch_wcvrptw.py)")
    from wastecollect.routing.checker import evaluate
    from wastecollect.routing.instance import parse_kim
    cache = {}
    for path in runs:
        run = json.loads(path.read_text(encoding="utf-8"))
        if run["instance"] not in cache:
            cache[run["instance"]] = parse_kim(data / f"{run['instance']}_stop.txt")
        inst = cache[run["instance"]]
        pos = {sid: k for k, sid in enumerate(inst.ids)}
        ev = evaluate(inst, [[pos[s] for s in route] for route in run["routes_stop_ids"]])
        assert ev.feasible, (path.name, ev.violations[:3])
        assert ev.distance == pytest.approx(run["distance"], abs=1e-5) and ev.n_routes == run["n_routes"]
