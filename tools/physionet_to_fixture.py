#!/usr/bin/env python3
"""Build the PhysioNet ECG reference corpus that the irregular-rhythm and ECG-quality gates measure
against. Stdlib only (urllib, zipfile, struct) so it runs on a bare VPS with no pip.

Two stages, both idempotent:

  fetch    pull the source files into a staging dir. Only headers and annotations for afdb/nsrdb/mitdb
           (the `.dat` signal files are 1.2 GB across the three and are never downloaded in full), plus
           the 2017 Challenge training archive and its reference labels. 60 s waveform windows are
           pulled from `.dat` by HTTP range request, a few tens of KB each.
  convert  write one folder per database under the datasets root: R-R series with per-interval rhythm
           and beat labels, waveform windows, and for the Challenge the full single-lead corpus.

Every one of these databases is chest or handheld ECG. None is wrist PPG, so a result here is
necessary and not sufficient for this project's hardware - the fixture headers repeat that.

Usage:
  python tools/physionet_to_fixture.py fetch   [STAGING_DIR]
  python tools/physionet_to_fixture.py convert [STAGING_DIR] [DATASETS_DIR]
"""

import json
import os
import shutil
import struct
import sys
import urllib.request
import zipfile

from physionet_wfdb import (
    beats,
    decode_212,
    parse_annotations,
    parse_header,
    read_mat_v4,
    rhythm_at,
    rhythm_changes,
)

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_STAGING = os.path.normpath(
    os.path.join(HERE, "..", "..", "whoop-data", "datasets", "_staging")
)
DEFAULT_DATASETS = os.path.normpath(os.path.join(HERE, "..", "..", "whoop-data", "datasets"))

PHYSIONET = "https://physionet.org/files/%s/1.0.0/%s"
CHALLENGE = "challenge-2017"

# Beat marks and rhythm markers do not always live in the same file. afdb keeps rhythm spans in `.atr`
# and beats in `.qrs`; mitdb and nsrdb keep both in `.atr`.
ANNOTATED = {
    "afdb": {
        "out": "physionet-afdb",
        "beat_ext": ["qrsc", "qrs"],
        "rhythm_ext": "atr",
        "lead": "chest, 2-channel ambulatory Holter ECG - NOT wrist PPG",
    },
    "nsrdb": {
        "out": "physionet-nsrdb",
        "beat_ext": ["atr"],
        "rhythm_ext": "atr",
        "lead": "chest, 2-channel ambulatory Holter ECG - NOT wrist PPG",
    },
    "mitdb": {
        "out": "physionet-mitdb",
        "beat_ext": ["atr"],
        "rhythm_ext": "atr",
        "lead": "chest, 2-channel ambulatory Holter ECG - NOT wrist PPG",
    },
}

WINDOW_S = 60.0
CHALLENGE_OUT = "physionet-challenge2017"
# One human-readable text copy per reference class, so the binary corpus is inspectable without tooling.
CHALLENGE_TEXT_SAMPLES = 1


# --------------------------------------------------------------------------------------------- fetch


def _get(url, timeout=300, headers=None):
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _save(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(data)


def fetch(staging):
    os.makedirs(staging, exist_ok=True)
    total = 0
    for db, spec in ANNOTATED.items():
        recs = _get(PHYSIONET % (db, "RECORDS")).decode().split()
        _save(os.path.join(staging, db + ".RECORDS"), "\n".join(recs).encode())
        wanted = ["hea", spec["rhythm_ext"]] + spec["beat_ext"]
        for rec in recs:
            for ext in dict.fromkeys(wanted):
                dst = os.path.join(staging, db, "%s.%s" % (rec, ext))
                if os.path.exists(dst) and os.path.getsize(dst) > 0:
                    total += os.path.getsize(dst)
                    continue
                try:
                    data = _get(PHYSIONET % (db, "%s.%s" % (rec, ext)), timeout=120)
                except Exception:
                    continue  # optional per-record extras (.qrsc exists for two afdb records only)
                _save(dst, data)
                total += len(data)
        print("fetched %s" % db)
    for name in ("training2017.zip", "REFERENCE-v3.csv"):
        dst = os.path.join(staging, name)
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            total += os.path.getsize(dst)
            continue
        data = _get(PHYSIONET % (CHALLENGE, name), timeout=1800)
        _save(dst, data)
        total += len(data)
        print("fetched %s (%d bytes)" % (name, len(data)))
    print("staging total %.1f MB" % (total / 1e6))


def fetch_window(db, record, header, start_frame, n_frames):
    """Range-request one window of channel 0 from a format-212 `.dat`. Returns baseline-subtracted counts."""
    sig = header.signals[0]
    if sig.fmt != 212:
        raise ValueError("%s/%s is format %d, only 212 is handled" % (db, record, sig.fmt))
    nsig = header.nsig
    k0 = start_frame * nsig
    if k0 % 2:
        start_frame += 1
        k0 = start_frame * nsig
    n_samples = n_frames * nsig
    byte0 = k0 * 3 // 2
    nbytes = ((n_samples + 1) // 2) * 3
    url = PHYSIONET % (db, sig.filename)
    buf = _get(url, timeout=300, headers={"Range": "bytes=%d-%d" % (byte0, byte0 + nbytes - 1)})
    raw = decode_212(buf, n_samples)
    return start_frame, [raw[i] - sig.baseline for i in range(0, n_samples, nsig)]


# ------------------------------------------------------------------------------------------- convert


def _read(path):
    with open(path, "rb") as fh:
        return fh.read()


def _write_lines(path, lines):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="\n") as fh:
        fh.write("".join(lines))


def rr_series(beat_list, changes, fs_hz):
    """`(rr_ms, ending-beat symbol, rhythm)` per interval. Non-positive intervals are dropped and counted."""
    rows = []
    dropped = 0
    for i in range(1, len(beat_list)):
        dt = beat_list[i][0] - beat_list[i - 1][0]
        if dt <= 0:
            dropped += 1
            continue
        rows.append((dt * 1000.0 / fs_hz, beat_list[i][1], rhythm_at(changes, beat_list[i][0])))
    return rows, dropped


def convert_annotated(db, staging, datasets):
    spec = ANNOTATED[db]
    src = os.path.join(staging, db)
    out = os.path.join(datasets, spec["out"])
    recs = open(os.path.join(staging, db + ".RECORDS")).read().split()

    index = []
    stats = {"database": db, "records": [], "rhythm_totals": {}, "beat_totals": {}}
    for rec in recs:
        hea = os.path.join(src, rec + ".hea")
        if not os.path.exists(hea):
            print("  %s/%s: no header, skipped" % (db, rec))
            continue
        header = parse_header(_read(hea).decode("ascii", "replace"))

        beat_file = None
        for ext in spec["beat_ext"]:
            p = os.path.join(src, "%s.%s" % (rec, ext))
            if os.path.exists(p):
                beat_file = (p, ext)
                break
        if beat_file is None:
            print("  %s/%s: no beat annotations, skipped" % (db, rec))
            continue
        beat_list = beats(parse_annotations(_read(beat_file[0])))

        rhythm_path = os.path.join(src, "%s.%s" % (rec, spec["rhythm_ext"]))
        changes = rhythm_changes(parse_annotations(_read(rhythm_path))) if os.path.exists(rhythm_path) else []
        changes.sort(key=lambda c: c[0])

        rows, dropped = rr_series(beat_list, changes, header.fs_hz)
        if not rows:
            print("  %s/%s: no intervals, skipped" % (db, rec))
            continue

        rhythms = {}
        symbols = {}
        for _, sym, rhy in rows:
            rhythms[rhy] = rhythms.get(rhy, 0) + 1
            symbols[sym] = symbols.get(sym, 0) + 1
        for k, v in rhythms.items():
            stats["rhythm_totals"][k] = stats["rhythm_totals"].get(k, 0) + v
        for k, v in symbols.items():
            stats["beat_totals"][k] = stats["beat_totals"].get(k, 0) + v

        corrected = beat_file[1] == "qrsc"
        head = [
            "# physionet %s r-r series\n" % db,
            "# database %s\n" % db,
            "# record %s\n" % rec,
            "# fs_hz %s\n" % header.fs_hz,
            "# beat_source %s (%s)\n"
            % (
                os.path.basename(beat_file[0]),
                "manually corrected" if corrected else ("audited by the database's cardiologists" if db != "afdb" else "automated detector, NOT manually corrected"),
            ),
            "# rhythm_source %s\n" % os.path.basename(rhythm_path),
            "# beats %d\n" % len(beat_list),
            "# intervals %d\n" % len(rows),
            "# dropped_nonpositive %d\n" % dropped,
            "# columns rr_ms beat rhythm\n",
            "# units rr_ms=milliseconds, exact from beat sample index / fs_hz\n",
            "# rhythm_counts %s\n" % " ".join("%s:%d" % kv for kv in sorted(rhythms.items())),
            "# beat_counts %s\n" % " ".join("%s:%d" % kv for kv in sorted(symbols.items())),
            "# lead %s\n" % spec["lead"],
        ]
        body = ["%.3f %s %s\n" % r for r in rows]
        _write_lines(os.path.join(out, "rr", rec + ".txt"), head + body)

        index.append(
            {
                "record": rec,
                "fs_hz": header.fs_hz,
                "duration_s": round(header.nsamp / header.fs_hz, 1) if header.nsamp else None,
                "beats": len(beat_list),
                "intervals": len(rows),
                "beat_source": os.path.basename(beat_file[0]),
                "beats_corrected": corrected,
                "rhythms": rhythms,
                "calibrated": bool(header.signals) and header.signals[0].calibrated,
            }
        )
        stats["records"].append(index[-1])
        print("  %s/%s  %d intervals  %s" % (db, rec, len(rows), sorted(rhythms)))

    _write_index_tsv(
        os.path.join(out, "index.tsv"),
        ["record", "fs_hz", "duration_s", "beats", "intervals", "beat_source", "beats_corrected", "rhythms"],
        [
            [
                r["record"],
                r["fs_hz"],
                r["duration_s"],
                r["beats"],
                r["intervals"],
                r["beat_source"],
                str(r["beats_corrected"]).lower(),
                ",".join("%s:%d" % kv for kv in sorted(r["rhythms"].items())),
            ]
            for r in index
        ],
    )
    return stats


def copy_source_annotations(db, staging, datasets):
    """Keep the headers and annotation files beside the fixtures they produced.

    They are a few MB in total and they are the entire derivation input, so `convert` stays re-runnable
    without a network. The `.dat` signal files are 1.2 GB across the three databases and are not copied;
    the waveform windows are all this corpus keeps of them.
    """
    src = os.path.join(staging, db)
    dst = os.path.join(datasets, ANNOTATED[db]["out"], "source")
    os.makedirs(dst, exist_ok=True)
    n = 0
    for name in sorted(os.listdir(src)):
        if name.rsplit(".", 1)[-1] not in ("hea", "atr", "qrs", "qrsc"):
            continue
        shutil.copyfile(os.path.join(src, name), os.path.join(dst, name))
        n += 1
    shutil.copyfile(os.path.join(staging, db + ".RECORDS"), os.path.join(dst, "RECORDS"))
    return n


def _write_index_tsv(path, columns, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="\n") as fh:
        fh.write("\t".join(columns) + "\n")
        for r in rows:
            fh.write("\t".join("" if v is None else str(v) for v in r) + "\n")


def window_plan(db, rec, header, changes, total_frames):
    """Which 60 s windows to pull for a record, chosen by a fixed rule so nothing is cherry-picked.

    afdb: the first full window inside the LONGEST span of each rhythm the record contains, so both an
    AF and a non-AF window come from every record that has both. Everything else: the middle window.
    """
    n = int(round(WINDOW_S * header.fs_hz))
    if total_frames < n:
        return []
    if db != "afdb" or not changes:
        return [((total_frames - n) // 2, n, rhythm_at(changes, (total_frames - n) // 2) if changes else "?")]
    spans = []
    for i, (start, token) in enumerate(changes):
        end = changes[i + 1][0] if i + 1 < len(changes) else total_frames
        spans.append((token, start, end))
    plan = []
    for token in sorted(set(s[0] for s in spans)):
        best = max((s for s in spans if s[0] == token), key=lambda s: s[2] - s[1])
        if best[2] - best[1] >= n:
            plan.append((best[1], n, token))
    return plan


def convert_windows(db, staging, datasets, stats):
    spec = ANNOTATED[db]
    src = os.path.join(staging, db)
    out = os.path.join(datasets, spec["out"], "raw")
    recs = open(os.path.join(staging, db + ".RECORDS")).read().split()
    written = 0
    for rec in recs:
        hea = os.path.join(src, rec + ".hea")
        if not os.path.exists(hea):
            continue
        header = parse_header(_read(hea).decode("ascii", "replace"))
        if not header.signals or not header.nsamp:
            continue  # afdb 00735 and 03665 keep their annotations; the signal files were lost
        sig = header.signals[0]
        rhythm_path = os.path.join(src, "%s.%s" % (rec, spec["rhythm_ext"]))
        changes = rhythm_changes(parse_annotations(_read(rhythm_path))) if os.path.exists(rhythm_path) else []
        changes.sort(key=lambda c: c[0])
        for start, n, token in window_plan(db, rec, header, changes, header.nsamp):
            slug = "".join(c for c in (token or "").lower() if c.isalnum()) or "unlabelled"
            name = "%s_%s.txt" % (rec, slug)
            dst = os.path.join(out, name)
            if os.path.exists(dst) and os.path.getsize(dst) > 0:
                written += 1
                continue
            try:
                start, samples = fetch_window(db, rec, header, start, n)
            except Exception as e:
                print("  %s/%s window: %s" % (db, rec, e))
                continue
            scale = (1.0 / sig.gain) if sig.calibrated else 1.0
            head = [
                "# physionet %s raw window, one sample per line\n" % db,
                "# database %s\n" % db,
                "# record %s\n" % rec,
                "# channel 0 (%s)\n" % (sig.desc or "?"),
                "# fs_hz %s\n" % header.fs_hz,
                "# samples %d\n" % len(samples),
                "# start_sample %d\n" % start,
                "# duration_s %.1f\n" % (len(samples) / header.fs_hz),
                "# calibrated %s\n" % ("true" if sig.calibrated else "false"),
                "# scale %s  (%s)\n"
                % (
                    repr(scale),
                    "value = count * scale, mV" if sig.calibrated else "raw ADC counts, amplitude UNCALIBRATED - the header gain is 0",
                ),
                "# rhythm %s\n" % (token or "?"),
                "# lead %s\n" % spec["lead"],
            ]
            _write_lines(dst, head + ["%d\n" % v for v in samples])
            written += 1
            print("  %s/%s window %s at %d" % (db, rec, token, start))
    stats["raw_windows"] = written
    return stats


def convert_challenge(staging, datasets):
    """Convert the whole 2017 Challenge training set: one concatenated int16 blob plus an index.

    No R-R series is emitted. The Challenge ships no beat annotations, so R-R could only come from a
    QRS detector - and writing one here would put a second, untested detector in the project beside the
    two in `physio-algo::ecg`. The raw lead ships instead, and R-R is derived by the code under test.
    """
    out = os.path.join(datasets, CHALLENGE_OUT)
    os.makedirs(out, exist_ok=True)
    ref = {}
    for line in open(os.path.join(staging, "REFERENCE-v3.csv")):
        line = line.strip()
        if not line:
            continue
        rec, cls = line.split(",")
        ref[rec] = cls

    z = zipfile.ZipFile(os.path.join(staging, "training2017.zip"))
    names = sorted(n for n in z.namelist() if n.endswith(".mat"))
    rows = []
    per_class = {}
    text_written = {}
    offset = 0
    blob_path = os.path.join(out, "signals.i16")
    with open(blob_path, "wb") as blob:
        for name in names:
            rec = os.path.basename(name)[:-4]
            header = parse_header(z.read(name[:-4] + ".hea").decode("ascii", "replace"))
            sig = header.signals[0]
            _, vals, data_off = read_mat_v4(z.read(name))
            if data_off != sig.offset:
                raise ValueError("%s: .hea says offset %d, MAT header says %d" % (rec, sig.offset, data_off))
            if header.nsamp and len(vals) != header.nsamp:
                raise ValueError("%s: .hea says %d samples, MAT has %d" % (rec, header.nsamp, len(vals)))
            cls = ref.get(rec, "?")
            blob.write(struct.pack("<%dh" % len(vals), *vals))
            rows.append([rec, cls, len(vals), offset, header.fs_hz, sig.gain, sig.baseline])
            offset += len(vals) * 2
            per_class[cls] = per_class.get(cls, 0) + 1
            if text_written.get(cls, 0) < CHALLENGE_TEXT_SAMPLES:
                _write_challenge_text(out, rec, cls, header, sig, vals)
                text_written[cls] = text_written.get(cls, 0) + 1

    _write_index_tsv(
        os.path.join(out, "index.tsv"),
        ["record", "class", "samples", "offset_bytes", "fs_hz", "adu_per_mv", "baseline"],
        rows,
    )
    with open(os.path.join(out, "REFERENCE-v3.csv"), "w", newline="\n") as fh:
        for rec, cls in sorted(ref.items()):
            fh.write("%s,%s\n" % (rec, cls))
    return {
        "database": CHALLENGE,
        "records": len(rows),
        "classes": per_class,
        "blob_bytes": offset,
        "total_samples": offset // 2,
    }


def _write_challenge_text(out, rec, cls, header, sig, vals):
    head = [
        "# physionet challenge-2017 record, one sample per line\n",
        "# database challenge-2017\n",
        "# record %s\n" % rec,
        "# reference_class %s\n" % cls,
        "# fs_hz %s\n" % header.fs_hz,
        "# samples %d\n" % len(vals),
        "# calibrated true\n",
        "# scale %s  (value = count * scale, mV)\n" % repr(1.0 / sig.gain),
        "# lead handheld single lead I (AliveCor, thumbs on two electrodes) - NOT wrist PPG\n",
        "# note this text copy exists so the format is readable without tooling; the corpus is signals.i16\n",
    ]
    _write_lines(os.path.join(out, "samples", "%s.txt" % rec), head + ["%d\n" % v for v in vals])


def dir_bytes(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    return total


def convert(staging, datasets):
    report = {}
    for db in ANNOTATED:
        s = convert_annotated(db, staging, datasets)
        s = convert_windows(db, staging, datasets, s)
        s["source_files"] = copy_source_annotations(db, staging, datasets)
        s["bytes"] = dir_bytes(os.path.join(datasets, ANNOTATED[db]["out"]))
        report[db] = s
    c = convert_challenge(staging, datasets)
    c["bytes"] = dir_bytes(os.path.join(datasets, CHALLENGE_OUT))
    report[CHALLENGE] = c
    report["total_bytes"] = sum(v["bytes"] for v in report.values() if isinstance(v, dict))
    with open(os.path.join(datasets, "physionet_ecg_corpus_stats.json"), "w", newline="\n") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
    print("\ntotal on disk %.1f MB" % (report["total_bytes"] / 1e6))
    return report


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "convert"
    stage = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_STAGING
    if cmd == "fetch":
        fetch(stage)
    elif cmd == "convert":
        convert(stage, sys.argv[3] if len(sys.argv) > 3 else DEFAULT_DATASETS)
    else:
        raise SystemExit(__doc__)
