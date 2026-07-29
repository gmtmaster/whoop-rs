"""Sweep `SEAM_SLACK_MS` and report what each value does to nightly RMSSD.

The constant lives inside `report_seam_breaks`, which is private, so the sweep cannot be driven from
outside the crate. This patches the literal, rebuilds, runs `hrv_seam`, and restores the file in a
finally block, which keeps the recorded sensitivity table re-runnable in one command instead of a
hand-edited experiment nobody can repeat.

    python tools/seam-slack-sweep.py [fixture-root]

With no argument it sweeps the default fixtures. Pass `sleep-benchmark/fixtures_multi_clean` to
sweep the de-duplicated corpus.
"""
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "crates/physio-algo/src/hrv.rs"
PATTERN = re.compile(r"(const SEAM_SLACK_MS: u64 = )([\d_]+)(;)")
VALUES = [0, 1_000, 2_000, 3_000, 4_000, 60_000, 300_000]


def run(values, env):
    out = []
    for v in values:
        text = SRC.read_text(encoding="utf-8", newline="")
        SRC.write_text(PATTERN.sub(rf"\g<1>{v}\g<3>", text), encoding="utf-8", newline="")
        subprocess.run(
            ["cargo", "build", "--release", "-p", "physio-algo", "--example", "hrv_seam"],
            cwd=ROOT, check=True, capture_output=True,
        )
        r = subprocess.run(
            [str(ROOT / "target/release/examples/hrv_seam.exe")],
            cwd=ROOT, check=True, capture_output=True, text=True, env=env,
        )
        wearers = {}
        for line in r.stdout.splitlines():
            f = line.split()
            if len(f) == 4 and f[1].isdigit():
                wearers[f[0]] = f[3]
        shift = next((l for l in r.stdout.splitlines() if l.startswith("per-night shift")), "")
        out.append((v, shift.split("median ")[1].split()[0] if "median " in shift else "?", wearers))
    return out


def main():
    env = dict(os.environ)
    if len(sys.argv) > 1:
        env["WHOOP_SLEEP_FIXTURES"] = str(Path(sys.argv[1]).resolve())
    print(f"fixtures: {env.get('WHOOP_SLEEP_FIXTURES', '(default)')}")
    original = SRC.read_text(encoding="utf-8", newline="")
    try:
        rows = run(VALUES, env)
    finally:
        SRC.write_text(original, encoding="utf-8", newline="")
        subprocess.run(
            ["cargo", "build", "--release", "-p", "physio-algo", "--example", "hrv_seam"],
            cwd=ROOT, check=False, capture_output=True,
        )

    names = sorted({k for _, _, w in rows for k in w})
    print(f"\n{'slack':>8} {'median shift':>13} " + " ".join(f"{n:>13}" for n in names))
    for v, shift, w in rows:
        print(f"{v:>8} {shift:>13} " + " ".join(f"{w.get(n, '-'):>13}" for n in names))
    print("\nRMSSD after the seam rule, per wearer, at each slack. A flat column is an insensitive value.")


if __name__ == "__main__":
    main()
