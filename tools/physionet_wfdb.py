#!/usr/bin/env python3
"""Minimal stdlib readers for the PhysioNet/WFDB formats this project pulls.

No pip, no numpy: everything here is `struct` and `array` so the converter runs on a bare VPS. Covers
the four formats the ECG reference corpus needs and nothing else:

  * `.hea`   WFDB header  - sample rate, signal count, storage format, gain/baseline.
  * `.atr` / `.qrs` / `.qrsc`  MIT annotation files - beat marks and rhythm-change markers.
  * format 212 signal data  - two 12-bit samples packed into three bytes (afdb, nsrdb, mitdb `.dat`).
  * MATLAB v4 numeric matrix - the container the 2017 Challenge ships its single lead in.

Used by `physionet_to_fixture.py`.
"""

import struct

# MIT annotation type code -> symbol, from the WFDB code table. Index is the 6-bit code.
SYMBOLS = [
    " ", "N", "L", "R", "a", "V", "F", "J", "A", "S", "E", "j", "/", "Q", "~", "?",
    "|", "?", "s", "T", "*", "D", '"', "=", "p", "B", "^", "t", "+", "u", "?", "!",
    "[", "]", "e", "n", "@", "x", "f", "(", ")", "r", "?", "?", "?", "?", "?", "?",
    "?", "?",
]

# Codes that mark a heartbeat (WFDB `isqrs`). Everything else is a rhythm marker, a wave boundary,
# a quality note or an artefact flag, and never contributes an R-R interval.
BEAT_CODES = frozenset([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 25, 34, 35, 38, 41])

CODE_SKIP = 59
CODE_NUM = 60
CODE_SUB = 61
CODE_CHN = 62
CODE_AUX = 63
CODE_RHYTHM = 28  # '+', carries the rhythm token in its aux field, e.g. "(AFIB".


class Signal(object):
    """One signal line of a `.hea`."""

    def __init__(self, filename, fmt, offset, gain, units, adcres, adczero, baseline, desc):
        self.filename = filename
        self.fmt = fmt
        self.offset = offset
        self.gain = gain
        self.units = units
        self.adcres = adcres
        self.adczero = adczero
        self.baseline = baseline
        self.desc = desc

    @property
    def calibrated(self):
        """A gain of 0 in a `.hea` means the recording carries no amplitude calibration at all."""
        return self.gain > 0


class Header(object):
    def __init__(self, name, nsig, fs_hz, nsamp, signals):
        self.name = name
        self.nsig = nsig
        self.fs_hz = fs_hz
        self.nsamp = nsamp
        self.signals = signals


class Annotation(object):
    __slots__ = ("sample", "code", "sub", "chan", "num", "aux")

    def __init__(self, sample, code, sub, chan, num, aux):
        self.sample = sample
        self.code = code
        self.sub = sub
        self.chan = chan
        self.num = num
        self.aux = aux

    @property
    def symbol(self):
        return SYMBOLS[self.code] if self.code < len(SYMBOLS) else "?"


def parse_header(text):
    """Parse a `.hea`. Returns a `Header`; raises ValueError on anything it cannot represent."""
    lines = [l for l in text.splitlines() if l.strip() and not l.startswith("#")]
    if not lines:
        raise ValueError("empty header")
    f = lines[0].split()
    name = f[0].split("/")[0]
    nsig = int(f[1])
    fs_hz = float(f[2].split("/")[0]) if len(f) > 2 else 250.0
    nsamp = int(f[3]) if len(f) > 3 else 0
    signals = []
    for line in lines[1 : 1 + nsig]:
        signals.append(_parse_signal_line(line))
    return Header(name, nsig, fs_hz, nsamp, signals)


def _parse_signal_line(line):
    g = line.split()
    fmt_field = g[1]
    offset = 0
    if "+" in fmt_field:
        fmt_field, off = fmt_field.split("+", 1)
        offset = int(off)
    fmt = int(fmt_field.split("x")[0].split(":")[0])

    gain_field = g[2] if len(g) > 2 else "200"
    units = "mV"
    if "/" in gain_field:
        gain_field, units = gain_field.split("/", 1)
    baseline = None
    if "(" in gain_field:
        gain_field, b = gain_field.split("(", 1)
        baseline = int(b.rstrip(")"))
    gain = float(gain_field)

    adcres = int(g[3]) if len(g) > 3 else 12
    adczero = int(g[4]) if len(g) > 4 else 0
    if baseline is None:
        baseline = adczero
    desc = " ".join(g[8:]) if len(g) > 8 else ""
    return Signal(g[0], fmt, offset, gain, units, adcres, adczero, baseline, desc)


def parse_annotations(data):
    """Parse a MIT annotation file.

    The stream is 16-bit little-endian words: the top 6 bits are the type code, the low 10 the sample
    interval since the previous annotation. Code 59 escapes to a 32-bit interval in the next two words;
    codes 60-63 attach NUM/SUB/CHN/AUX to the annotation just read; code 0 with a 0 interval ends it.
    """
    nwords = len(data) // 2
    words = struct.unpack("<%dH" % nwords, data[: nwords * 2])
    out = []
    i = 0
    t = 0
    while i < nwords:
        w = words[i]
        code = w >> 10
        dt = w & 0x3FF
        if code == 0 and dt == 0:
            break
        if code == CODE_SKIP:
            if i + 3 >= nwords:
                break
            dt = (words[i + 1] << 16) | words[i + 2]
            code = words[i + 3] >> 10
            i += 4
        else:
            i += 1
        t += dt
        sub = chan = num = 0
        aux = None
        while i < nwords:
            w2 = words[i]
            c2 = w2 >> 10
            lo = w2 & 0xFF
            if c2 == CODE_SUB:
                sub = lo - 256 if lo > 127 else lo
                i += 1
            elif c2 == CODE_CHN:
                chan = lo
                i += 1
            elif c2 == CODE_NUM:
                num = lo - 256 if lo > 127 else lo
                i += 1
            elif c2 == CODE_AUX:
                start = (i + 1) * 2
                aux = data[start : start + lo]
                i += 1 + (lo + 1) // 2
            else:
                break
        out.append(Annotation(t, code, sub, chan, num, aux))
    return out


def beats(anns):
    """Sample indices and symbols of the annotations that mark a heartbeat."""
    return [(a.sample, a.symbol) for a in anns if a.code in BEAT_CODES]


def rhythm_changes(anns):
    """`(sample, token)` for every rhythm-change marker, token stripped of its leading '('.

    A marker with an empty aux ends the current rhythm without naming a new one; that is emitted as
    the explicit token "?" rather than being dropped, so a gap can never read as the previous rhythm.
    """
    out = []
    for a in anns:
        if a.code != CODE_RHYTHM:
            continue
        aux = (a.aux or b"").split(b"\x00")[0].decode("ascii", "replace").strip()
        if not aux.startswith("("):
            continue
        token = aux[1:].strip()
        out.append((a.sample, token if token else "?"))
    return out


def rhythm_at(changes, sample):
    """Rhythm token in force at `sample`; "?" before the first marker. `changes` must be sorted."""
    lo, hi = 0, len(changes)
    while lo < hi:
        mid = (lo + hi) // 2
        if changes[mid][0] <= sample:
            lo = mid + 1
        else:
            hi = mid
    return changes[lo - 1][1] if lo else "?"


def decode_212(buf, count):
    """Decode `count` consecutive 12-bit samples from a format-212 buffer.

    Three bytes hold two samples: the first is `b1[3:0] << 8 | b0`, the second `b1[7:4] << 8 | b2`,
    each two's complement. `buf` must start on a 3-byte group, i.e. at an even sample index.
    """
    out = []
    n_groups = (count + 1) // 2
    if len(buf) < n_groups * 3:
        raise ValueError("format-212 buffer short: %d bytes for %d samples" % (len(buf), count))
    for g in range(n_groups):
        b0, b1, b2 = buf[g * 3], buf[g * 3 + 1], buf[g * 3 + 2]
        a = ((b1 & 0x0F) << 8) | b0
        b = ((b1 >> 4) << 8) | b2
        out.append(a - 4096 if a > 2047 else a)
        if len(out) < count:
            out.append(b - 4096 if b > 2047 else b)
    return out


def read_mat_v4(data):
    """Read a MATLAB v4 numeric matrix. Returns `(name, values, data_offset)`.

    The 2017 Challenge stores each single-lead record as one v4 int16 row vector named `val`, which is
    why its `.hea` declares format `16+24`: 20 header bytes plus the 4-byte name.
    """
    if len(data) < 20:
        raise ValueError("not a MATLAB v4 file: %d bytes" % len(data))
    mopt, mrows, ncols, imagf, namlen = struct.unpack("<5i", data[:20])
    if not (0 <= mopt < 10000) or imagf not in (0, 1) or namlen <= 0:
        raise ValueError("not a MATLAB v4 numeric matrix (mopt=%d)" % mopt)
    prec = (mopt // 10) % 10
    if prec != 3:
        raise ValueError("unsupported v4 precision %d (want 3 = int16)" % prec)
    name = data[20 : 20 + namlen].split(b"\x00")[0].decode("ascii", "replace")
    off = 20 + namlen
    n = mrows * ncols
    vals = struct.unpack("<%dh" % n, data[off : off + n * 2])
    return name, vals, off
