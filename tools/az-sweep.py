#!/usr/bin/env python3
"""Run every sleep A-Z harness on one HEAD and diff what it prints against what the documents claim.

    python whoop-rs/tools/az-sweep.py                 # build, run everything, report mismatches
    python whoop-rs/tools/az-sweep.py --reuse          # skip the runs, re-check the cached output
    python whoop-rs/tools/az-sweep.py --only emit_wake # one harness
    python whoop-rs/tools/az-sweep.py --list           # every figure the manifest pins, with its run

The pairing of figure to command lives in `az-manifest.toml`, never here: a figure names the run that
prints it, the dataset that run reads, the document and section it appears in, the documented value, and
the regex that extracts the actual one. A figure whose `pattern` is absent is UNREPRODUCIBLE by design -
it is reported as such rather than silently passed, because a number no command regenerates is how this
project accumulated facts that turned out to be wrong.

Exit status is 1 when any figure mismatches, so this can gate a step.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    print("needs Python 3.11+ for tomllib", file=sys.stderr)
    raise SystemExit(2)

HERE = Path(__file__).resolve().parent
CARGO_ROOT = HERE.parent
WHOOP = CARGO_ROOT.parent
MANIFEST = HERE / "az-manifest.toml"
OUT = CARGO_ROOT / "dev-docs" / "az_sweep_out"


def load() -> dict:
    with MANIFEST.open("rb") as fh:
        return tomllib.load(fh)


def head_sha(repo: Path) -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=repo, capture_output=True, text=True)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True)
        return r.stdout.strip() + ("+dirty" if dirty.stdout.strip() else "")
    except OSError:
        return "?"


def build(cargo_root: Path) -> bool:
    print(f"building examples in {cargo_root} ...", flush=True)
    r = subprocess.run(
        ["cargo", "build", "--release", "-p", "physio-algo", "--examples"],
        cwd=cargo_root, capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(r.stdout[-4000:] + r.stderr[-4000:], file=sys.stderr)
        return False
    print("  build ok", flush=True)
    return True


def run_one(run: dict, cargo_root: Path) -> tuple[int, float]:
    """Execute one harness, writing stdout to the cache. Returns (exit code, seconds)."""
    OUT.mkdir(parents=True, exist_ok=True)
    exe = cargo_root / "target" / "release" / "examples" / f"{run['example']}.exe"
    if not exe.exists():
        exe = cargo_root / "target" / "release" / "examples" / run["example"]
    env = dict(os.environ)
    env["WHOOP_SLEEP_FIXTURES"] = str(WHOOP / run["fixtures"])
    t0 = time.time()
    r = subprocess.run([str(exe), *run.get("args", [])], capture_output=True, text=True, env=env)
    (OUT / f"{run['id']}.txt").write_text(r.stdout, encoding="utf-8")
    if r.stderr.strip():
        (OUT / f"{run['id']}.err").write_text(r.stderr, encoding="utf-8")
    return r.returncode, time.time() - t0


def norm(v: str) -> str:
    """Compare figures the way a reader does: sign, thousands separators and unit noise are not the value."""
    v = v.strip().replace(",", "").replace("−", "-").replace("–", "-")
    for unit in ("%", "ms", "min", "pp", "h"):
        v = v.replace(unit, "")
    v = v.strip().lstrip("+")
    if v[-1:] in ("x", "s") and v[:-1].strip().replace(".", "").replace("-", "").isdigit():
        v = v[:-1].strip()
    try:
        f = float(v)
        return f"{f:.10g}"
    except ValueError:
        return v


def check(fig: dict, text: str) -> tuple[str, str]:
    """Returns (verdict, actual). Verdict is MATCH, MISMATCH, NOT-FOUND or UNREPRODUCIBLE."""
    pat = fig.get("pattern")
    if not pat:
        return "UNREPRODUCIBLE", ""
    m = re.search(pat, text, re.MULTILINE)
    if not m:
        return "NOT-FOUND", ""
    actual = m.group(1)
    return ("MATCH" if norm(actual) == norm(str(fig["documented"])) else "MISMATCH"), actual


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse", action="store_true", help="do not run; re-check the cached output")
    ap.add_argument("--only", action="append", default=[], help="run id (repeatable)")
    ap.add_argument("--list", action="store_true", help="print every pinned figure and exit")
    ap.add_argument("--no-build", action="store_true")
    args = ap.parse_args()

    man = load()
    cargo_root = WHOOP / man["meta"]["cargo_root"]
    runs = {r["id"]: r for r in man["run"]}
    figs = man["figure"]

    if args.list:
        for f in figs:
            print(f"{f['run']:<18} {f['doc']:<24} {f.get('step','-'):<3} {f['label']:<62} = {f['documented']}")
        print(f"\n{len(figs)} figures over {len(runs)} runs")
        return 0

    wanted = [r for r in man["run"] if not args.only or r["id"] in args.only]

    print(f"whoop-rs HEAD {head_sha(cargo_root)}   manifest {man['meta']['version']}")
    if not args.reuse:
        if not args.no_build and not build(cargo_root):
            return 2
        for r in wanted:
            if r.get("skip"):
                print(f"  {r['id']:<18} SKIPPED - {r['skip']}", flush=True)
                continue
            rc, secs = run_one(r, cargo_root)
            print(f"  {r['id']:<18} rc={rc} {secs:6.1f}s   {r['fixtures'].split('/')[-1]}", flush=True)

    cache: dict[str, str] = {}
    for r in runs.values():
        p = OUT / f"{r['id']}.txt"
        cache[r["id"]] = p.read_text(encoding="utf-8") if p.exists() else ""

    tally = {"MATCH": 0, "MISMATCH": 0, "NOT-FOUND": 0, "UNREPRODUCIBLE": 0}
    problems = []
    for f in figs:
        if args.only and f["run"] not in args.only:
            continue
        verdict, actual = check(f, cache.get(f["run"], ""))
        tally[verdict] += 1
        if verdict != "MATCH":
            problems.append((verdict, f, actual))

    print("\n" + "=" * 100)
    for verdict, f, actual in problems:
        run = runs[f["run"]]
        print(f"\n{verdict}  [{f.get('step','-')}] {f['doc']} - {f['section']}")
        print(f"   figure      {f['label']}")
        print(f"   documented  {f['documented']}")
        print(f"   actual      {actual or '(no command reproduces this)'}")
        print(f"   command     WHOOP_SLEEP_FIXTURES={run['fixtures']} cargo run --release -p physio-algo "
              f"--example {run['example']}")
        if f.get("note"):
            print(f"   note        {f['note']}")
    print("\n" + "=" * 100)
    print(f"{tally['MATCH']} matched | {tally['MISMATCH']} MISMATCHED | {tally['NOT-FOUND']} not found in output "
          f"| {tally['UNREPRODUCIBLE']} unreproducible by any command")
    return 1 if tally["MISMATCH"] or tally["NOT-FOUND"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
