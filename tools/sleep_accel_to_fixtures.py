#!/usr/bin/env python3
"""Build deterministic physio-algo fixtures from SleepAccel v1.0.0.

The source timestamps are seconds relative to PSG start. PSG labels are on a
30-second grid beginning at zero.  The fixture keeps that clock unchanged and
crops Apple Watch HR and acceleration to the PSG window.  Truth is never used
to alter either production input stream.

PSG mapping: 0 -> Wake, 1/2 -> Light, 3/4 -> Deep, 5 -> REM; -1 is unscored
and omitted from truth.csv.  SleepAccel has no beat-to-beat/R-R stream, so an
empty rr.csv is generated deliberately.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

DATASET = "Motion and heart rate from a wrist-worn wearable and labeled sleep from polysomnography"
VERSION = "1.0.0"
DOI = "10.13026/hmhs-py35"
STAGE_MAP = {0: 0, 1: 1, 2: 1, 3: 2, 4: 2, 5: 3}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8", newline="\n")


def convert_stream(source: Path, target: Path, start: float, end: float, columns: int) -> int:
    count = 0
    with source.open("r", encoding="utf-8") as inp, target.open("w", encoding="utf-8", newline="\n") as out:
        for line_number, line in enumerate(inp, 1):
            cells = line.replace(",", " ").split()
            if len(cells) != columns:
                raise ValueError(f"{source}:{line_number}: expected {columns} columns, got {len(cells)}")
            timestamp = float(cells[0])
            if start <= timestamp < end:
                out.write(",".join(cells) + "\n")
                count += 1
    return count


def build(source_root: Path, fixture_root: Path) -> dict:
    labels_dir = source_root / "labels"
    subjects = sorted(path.name.removesuffix("_labeled_sleep.txt") for path in labels_dir.glob("*_labeled_sleep.txt"))
    if not subjects:
        raise ValueError(f"no SleepAccel labels under {labels_dir}")

    output_root = fixture_root / "sleep-accel"
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "dataset": DATASET,
        "dataset_version": VERSION,
        "doi": DOI,
        "source_root": str(source_root.resolve()),
        "fixture_root": str(output_root.resolve()),
        "stage_mapping": {"0": "Wake", "1": "Light", "2": "Light", "3": "Deep", "4": "Deep", "5": "REM", "-1": "excluded_unscored"},
        "alignment": "source-relative PSG clock; truth epoch k covers [30k,30k+30); evaluation probes midpoint",
        "missing_data": "raw HR/acceleration samples retained when inside PSG window; no interpolation or imputation; -1 PSG epochs omitted",
        "subjects": [],
        "excluded": [],
    }

    for subject in subjects:
        label_path = labels_dir / f"{subject}_labeled_sleep.txt"
        hr_path = source_root / "heart_rate" / f"{subject}_heartrate.txt"
        accel_path = source_root / "motion" / f"{subject}_acceleration.txt"
        missing = [str(path) for path in (hr_path, accel_path) if not path.is_file()]
        if missing:
            manifest["excluded"].append({"subject": subject, "reason": "missing required source", "paths": missing})
            continue

        rows = []
        with label_path.open("r", encoding="utf-8") as labels:
            for line_number, line in enumerate(labels, 1):
                cells = line.split()
                if len(cells) != 2:
                    raise ValueError(f"{label_path}:{line_number}: expected 2 columns")
                timestamp, raw_stage = float(cells[0]), int(cells[1])
                rows.append((timestamp, raw_stage))
        if not rows or rows[0][0] != 0.0:
            raise ValueError(f"{label_path}: expected a non-empty grid starting at zero")
        for index, (timestamp, _) in enumerate(rows):
            expected = index * 30.0
            if abs(timestamp - expected) > 1e-6:
                raise ValueError(f"{label_path}: timestamp {timestamp} is not epoch {index} ({expected})")

        start = 0.0
        end = len(rows) * 30.0
        night = output_root / subject
        night.mkdir(parents=True, exist_ok=True)
        write_text(night / "meta.txt", f"{subject} 0 {int(end)} {len(rows)}\n")
        truth_lines = []
        raw_counts: dict[str, int] = {}
        mapped_counts = {str(i): 0 for i in range(4)}
        for index, (_, raw_stage) in enumerate(rows):
            raw_counts[str(raw_stage)] = raw_counts.get(str(raw_stage), 0) + 1
            if raw_stage in STAGE_MAP:
                mapped = STAGE_MAP[raw_stage]
                mapped_counts[str(mapped)] += 1
                truth_lines.append(f"{index},{mapped}\n")
            elif raw_stage != -1:
                raise ValueError(f"{label_path}: unsupported PSG stage {raw_stage}")
        write_text(night / "truth.csv", "".join(truth_lines))
        hr_rows = convert_stream(hr_path, night / "hr.csv", start, end, 2)
        accel_rows = convert_stream(accel_path, night / "gravity.csv", start, end, 4)
        write_text(night / "rr.csv", "")

        generated = {name: sha256(night / name) for name in ("meta.txt", "gravity.csv", "hr.csv", "rr.csv", "truth.csv")}
        manifest["subjects"].append({
            "subject": subject,
            "fixture": str(night.relative_to(fixture_root)),
            "psg_epochs": len(rows),
            "evaluable_epochs": len(truth_lines),
            "excluded_unscored_epochs": raw_counts.get("-1", 0),
            "truth_counts": mapped_counts,
            "hr_samples": hr_rows,
            "accel_samples": accel_rows,
            "rr_available": False,
            "source_sha256": {"labels": sha256(label_path), "heart_rate": sha256(hr_path), "motion": sha256(accel_path)},
            "fixture_sha256": generated,
        })

    manifest["included_subjects"] = len(manifest["subjects"])
    manifest["excluded_subjects"] = len(manifest["excluded"])
    write_text(output_root / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="SleepAccel 1.0.0 directory containing labels/, heart_rate/, motion/")
    parser.add_argument("output", type=Path, help="fixture root; sleep-accel/ is created below it")
    args = parser.parse_args()
    manifest = build(args.source, args.output)
    print(f"built {manifest['included_subjects']} SleepAccel fixtures; excluded {manifest['excluded_subjects']}")
    print(Path(manifest["fixture_root"]) / "manifest.json")


if __name__ == "__main__":
    main()
