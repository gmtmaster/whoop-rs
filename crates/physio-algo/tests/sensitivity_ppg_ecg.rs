//! Negative control for the `ppg_ecg` metric family: it falsifies the claim that the shipped PPG and
//! ECG gates would notice if their algorithms broke.
//!
//! Every gate in this family is green today. Green means one of two things and the gates cannot tell
//! them apart: the algorithm is right, or the gate cannot see the algorithm. This file separates the
//! two by re-running each shipped gate — its exact target and tolerance copied into a `const` here,
//! with the `file:line` it came from — against deliberately damaged arms:
//!
//!   NULL        a scorer that does no work (constant output, shuffled input, noise). The gate MUST
//!               fail. A null that passes means the gate is fake, and that is a CRITICAL finding.
//!   STRUCTURAL  a wrong SHAPE (swapped classes, a shifted series, a reversal, a truncation). These
//!               teach what the headline number is actually made of.
//!   PARAMETER   every tunable the algorithm reads, moved ±10%, plus one +0.5% floor probe. Whether
//!               these are caught is the MEASUREMENT, not the requirement — a gate that passes a 10%
//!               drift is a reproduction check, not a regression check, and that is worth knowing.
//!
//! Only two things are asserted: the baseline reproduces the shipped figure, and the null arm fails
//! the shipped gate. A parameter arm passing is recorded, never asserted, and NOTHING here may be used
//! as grounds to move a shipped constant.
//!
//! Where a constant is a private literal with no config struct, it cannot be reached from an
//! integration test. Visibility is NOT widened. Such arms either re-apply the threshold to the value
//! the algorithm already reports (exact, labelled `re-applied`) or perturb the input in the equivalent
//! direction (labelled `proxy`), and the difference is stated per row.
//!
//! Health numbers named here are wellness estimates, never medical and never diagnostic.
//!
//! Reads the committed AAUWSS, rhythm-R-R and real-frame fixtures, so every test is `#[ignore]`d and
//! none of it enters CI. A missing fixture PANICS with instructions; it never skips.
//!
//!   cargo test --release -p physio-algo --test sensitivity_ppg_ecg -- --ignored --nocapture

mod ecg_corpus;
#[path = "ecg_corpus/encode.rs"]
#[allow(dead_code)]
mod encode;

use std::collections::{BTreeMap, HashSet};
use std::fs;
use std::path::PathBuf;

use ecg_corpus::{
    ecg_fixtures, gaussian_like, mean, ppg_fixtures, resample, sawtooth_like, shuffled,
    Fixture, Rng,
};
use encode::{detector_consensus, encode_stream, pulse_train};

use physio_algo::ecg::{self, mains, morphology, score, sweep};
use physio_algo::ppg;
use physio_algo::rr_irregularity::{self as rr, cosen, quality, screen};
use physio_algo::stats::median;
use whoop_protocol::bytes::from_hex;
use whoop_protocol::family::Family;
use whoop_protocol::framing;
use whoop_protocol::records::{decode, Record};

// ---------------------------------------------------------------------------------------------
// The shared arm table.
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Baseline,
    Null,
    Structural,
    Parameter,
}

impl Kind {
    fn tag(self) -> &'static str {
        match self {
            Kind::Baseline => "base",
            Kind::Null => "NULL",
            Kind::Structural => "strc",
            Kind::Parameter => "parm",
        }
    }
}

struct Row {
    arm: String,
    kind: Kind,
    value: f64,
    note: String,
    gate: bool,
}

struct Table {
    metric: &'static str,
    claim: &'static str,
    /// Set when the shipped gate is a range/property check carrying no threshold that any input can
    /// fail. A null passing such a gate is the finding rather than a harness fault, so the null
    /// assertion is skipped and the fact is printed instead.
    property_only: bool,
    rows: Vec<Row>,
}

impl Table {
    fn new(metric: &'static str, claim: &'static str) -> Self {
        Table { metric, claim, property_only: false, rows: Vec::new() }
    }

    fn add(
        &mut self,
        arm: impl Into<String>,
        kind: Kind,
        value: f64,
        gate: bool,
        note: impl Into<String>,
    ) {
        self.rows.push(Row { arm: arm.into(), kind, value, note: note.into(), gate });
    }

    /// Print the table and the `caught N, missed M` headline, then assert the only two things that
    /// have to hold for the harness to be worth reading.
    fn finish(&self) {
        let base = self.rows.first().expect("a table opens with its baseline");
        assert_eq!(base.kind, Kind::Baseline, "{}: the first arm must be the baseline", self.metric);

        println!("\n================ {} ================", self.metric);
        println!("shipped gate: {}", self.claim);
        println!(
            "{:<56} {:>5} {:>11} {:>11}  {:<5} note",
            "arm", "kind", "value", "delta", "gate"
        );
        for r in &self.rows {
            let delta = r.value - base.value;
            let mark = match (r.kind, r.gate) {
                (Kind::Baseline, _) => "",
                (_, true) => "  <-- MISSED",
                (_, false) => "  <-- caught",
            };
            println!(
                "{:<56} {:>5} {:>11.4} {:>11.4}  {:<5} {}{}",
                r.arm,
                r.kind.tag(),
                r.value,
                delta,
                if r.gate { "PASS" } else { "FAIL" },
                r.note,
                mark
            );
        }

        let caught = self.rows.iter().skip(1).filter(|r| !r.gate).count();
        let missed = self.rows.iter().skip(1).filter(|r| r.gate).count();
        let floor = self
            .rows
            .iter()
            .skip(1)
            .filter(|r| !r.gate)
            .map(|r| (r.value - base.value).abs())
            .filter(|v| v.is_finite())
            .fold(f64::INFINITY, f64::min);
        println!("caught {caught}, missed {missed}");
        if floor.is_finite() {
            println!("sensitivity floor: the smallest delta this gate catches is {floor:.4}");
        } else {
            println!("sensitivity floor: no caught arm carries a finite delta");
        }

        assert!(base.gate, "{}: the baseline does NOT reproduce the shipped figure", self.metric);
        if self.property_only {
            println!(
                "PROPERTY GATE: this claim has no threshold, so a null passes it by construction. \
                 Recorded, not asserted."
            );
        } else {
            let null_caught = self.rows.iter().any(|r| r.kind == Kind::Null && !r.gate);
            assert!(
                null_caught,
                "{}: EVERY null arm passed the shipped gate — the gate is fake",
                self.metric
            );
        }

        let probes: Vec<(&str, f64)> = self
            .rows
            .iter()
            .filter(|r| matches!(r.kind, Kind::Null | Kind::Structural))
            .map(|r| (r.arm.as_str(), r.value))
            .collect();
        enforce_floors(self.metric, base.value, &probes);
    }
}

fn pct(base: f64, frac: f64) -> f64 {
    base * (1.0 + frac)
}

fn median_opt(v: &[Option<f64>]) -> Option<f64> {
    let x: Vec<f64> = v.iter().flatten().copied().collect();
    (!x.is_empty()).then(|| median(&x))
}

// ---------------------------------------------------------------------------------------------
// Metric 1 — PPG-derived heart rate over the real v26 optical bursts.
// Gate: crates/physio-algo/tests/ppg_hr_real.rs:32, :36-38, :43-44, golden values in
// crates/whoop-protocol/tests/fixtures/real_frames.json under `ppg_hr`.
// ---------------------------------------------------------------------------------------------

/// ppg_hr_real.rs:32 — `real_frames.json` `ppg_hr.estimate_count`.
const PPG_ESTIMATE_COUNT: usize = 10;
/// ppg_hr_real.rs:36-38 — `ppg_hr.first`.
const PPG_FIRST_TS: i64 = 1_783_955_687;
const PPG_FIRST_BPM: i32 = 78;
const PPG_FIRST_CONF: f64 = 0.733;
const PPG_FIRST_CONF_TOL: f64 = 1e-9;
/// ppg_hr_real.rs:43-44 — `ppg_hr.confident_bpm_range`, applied to estimates at or above this conf.
const PPG_CONFIDENT_FLOOR: f64 = 0.7;
const PPG_CONFIDENT_LO: i32 = 60;
const PPG_CONFIDENT_HI: i32 = 100;

fn ppg_samples() -> Vec<ppg::Sample> {
    let oracle: serde_json::Value = serde_json::from_str(include_str!(
        "../../whoop-protocol/tests/fixtures/real_frames.json"
    ))
    .unwrap();
    let mut out = Vec::new();
    for f in oracle["ppg_frames"].as_array().unwrap() {
        let wire = from_hex(f["hex"].as_str().unwrap()).unwrap();
        let frame = framing::decode(Family::Gen5, &wire).unwrap();
        let p = match decode(&frame) {
            Some(Record::Ppg(p)) => p,
            other => panic!("expected a Ppg record, got {other:?}"),
        };
        for v in p.samples {
            out.push(ppg::Sample { ts: i64::from(p.unix), value: i32::from(v) });
        }
    }
    out
}

fn ppg_gate_with(est: &[ppg::Estimate], conf_floor: f64, lo: i32, hi: i32) -> bool {
    est.len() == PPG_ESTIMATE_COUNT
        && est.first().is_some_and(|f| {
            f.ts == PPG_FIRST_TS
                && f.bpm == PPG_FIRST_BPM
                && (f.conf - PPG_FIRST_CONF).abs() < PPG_FIRST_CONF_TOL
        })
        && est.iter().filter(|e| e.conf >= conf_floor).all(|e| (lo..=hi).contains(&e.bpm))
}

fn ppg_gate(est: &[ppg::Estimate]) -> bool {
    ppg_gate_with(est, PPG_CONFIDENT_FLOOR, PPG_CONFIDENT_LO, PPG_CONFIDENT_HI)
}

fn ppg_note(est: &[ppg::Estimate]) -> String {
    match est.first() {
        Some(f) => format!("first ts {} bpm {} conf {:.3}", f.ts, f.bpm, f.conf),
        None => "no estimate at all".to_string(),
    }
}

fn ppg_map(src: &[ppg::Sample], f: impl Fn(i32) -> i32) -> Vec<ppg::Sample> {
    src.iter().map(|s| ppg::Sample { ts: s.ts, value: f(s.value) }).collect()
}

fn shuffled_i32(src: &[i32], seed: u64) -> Vec<i32> {
    let mut out = src.to_vec();
    let mut rng = Rng(seed);
    for i in (1..out.len()).rev() {
        let j = (rng.next_u64() % (i as u64 + 1)) as usize;
        out.swap(i, j);
    }
    out
}

/// Re-lay each second's samples onto `n` linearly interpolated ones. `estimate` still believes the
/// rate is `ppg::SAMPLE_RATE_HZ`, so the rate it effectively applies becomes `24 * 24 / n` — the exact
/// equivalent of moving that constant, reached from the input side because it is a bare `pub const`
/// with no config struct behind it.
fn ppg_resecond(src: &[ppg::Sample], n: usize) -> Vec<ppg::Sample> {
    let mut by_sec: BTreeMap<i64, Vec<f64>> = BTreeMap::new();
    for s in src {
        by_sec.entry(s.ts).or_default().push(f64::from(s.value));
    }
    let mut out = Vec::new();
    for (ts, v) in by_sec {
        let last = v.len().saturating_sub(1) as f64;
        for k in 0..n {
            let t = if n > 1 { k as f64 * last / (n - 1) as f64 } else { 0.0 };
            let i = (t.floor() as usize).min(v.len() - 1);
            let frac = t - i as f64;
            let b = v[(i + 1).min(v.len() - 1)];
            out.push(ppg::Sample { ts, value: (v[i] + (b - v[i]) * frac).round() as i32 });
        }
    }
    out
}

// ── Sensitivity floors ─────────────────────────────────────────────────────────────────────────

/// `(metric, arm, minimum |delta| from the baseline)`. A floor asserts the arm still MOVES the number,
/// which is what catches an algorithm that stopped being reached; each is 0.45x the delta measured
/// 2026-08-02, so it sits well below the observed move and well above zero.
const FLOORS: &[(&str, &str, f64)] = &[
    ("ECG atrial-band power ratio", "input: gaussian noise", 5.85),
    ("ECG atrial-band power ratio", "input: samples shuffled", 5.85),
    ("ECG decode sweep (recover layout + sample rate from unknown bytes)", "input: gaussian noise encoded at the true layout", 0.9),
    ("ECG decode sweep (recover layout + sample rate from unknown bytes)", "input: samples shuffled, encoded at the true layout", 0.9),
    ("ECG decode sweep (recover layout + sample rate from unknown bytes)", "input: constant encoded at the true layout", 0.9),
    ("ECG decode sweep (recover layout + sample rate from unknown bytes)", "bytes: buffer cut one byte late", 0.45),
    ("ECG decode sweep (recover layout + sample rate from unknown bytes)", "bytes: buffer reversed", 0.9),
    ("ECG decode sweep (recover layout + sample rate from unknown bytes)", "beats: optical times stretched 10%", 0.9),
    ("ECG decode sweep (recover layout + sample rate from unknown bytes)", "beats: optical channel removed", 0.9),
    ("Mains anchor (recover the sample rate from 50 Hz line interference)", "input: gaussian noise (no line peak)", 4.5),
    ("Mains anchor (recover the sample rate from 50 Hz line interference)", "input: constant", 4.5),
    ("Mains anchor (recover the sample rate from 50 Hz line interference)", "input: samples shuffled (spectrum whitened)", 4.5),
    ("Mains anchor (recover the sample rate from 50 Hz line interference)", "input: 50 Hz notched out by a two-sample comb", 4.5),
    ("ECG P-wave morphology (present/absent + deflection ratio)", "input: gaussian noise", 5.85),
    ("ECG P-wave morphology (present/absent + deflection ratio)", "input: samples shuffled", 5.85),
    ("ECG P-wave morphology (present/absent + deflection ratio)", "input: PR segment replaced by a line plus noise", 5.85),
    ("ECG QRS detector pair (Pan-Tompkins + wavelet) and bSQI agreement", "input: gaussian noise matched to each subject", 0.441),
    ("ECG QRS detector pair (Pan-Tompkins + wavelet) and bSQI agreement", "input: constant (dead channel)", 0.441),
    ("ECG QRS detector pair (Pan-Tompkins + wavelet) and bSQI agreement", "input: samples shuffled (morphology destroyed)", 0.441),
    ("ECG QRS detector pair (Pan-Tompkins + wavelet) and bSQI agreement", "input: sawtooth at each subject's own heart rate", 0.441),
    ("ECG QRS detector pair (Pan-Tompkins + wavelet) and bSQI agreement", "input: time-reversed", 0.138),
    ("ECG QRS detector pair (Pan-Tompkins + wavelet) and bSQI agreement", "input: last 10% dropped", 0.000675),
    ("ECG QRS detector pair (Pan-Tompkins + wavelet) and bSQI agreement", "input: decimated x2, rate still declared 200 Hz", 0.123),
    ("ECG signal-quality gate (bSQI/kSQI/pSQI/basSQI/templateSQI + HR)", "input: positives replaced by gaussian noise", 0.45),
    ("ECG signal-quality gate (bSQI/kSQI/pSQI/basSQI/templateSQI + HR)", "input: positives replaced by a constant", 0.45),
    ("ECG signal-quality gate (bSQI/kSQI/pSQI/basSQI/templateSQI + HR)", "input: positives read at 2.5x the true rate", 0.415),
    ("PPG-HR bucket aggregate, poor-signal derate, span trust verdict", "input: every confidence identical (weights carry nothing)", 24.7),
    ("PPG-HR bucket aggregate, poor-signal derate, span trust verdict", "input: every confidence zero (fallback path)", 24.7),
    ("PPG-HR bucket aggregate, poor-signal derate, span trust verdict", "output: the confident and the noisy bpm swapped", 32.8),
    ("PPG-HR bucket aggregate, poor-signal derate, span trust verdict", "output: shifted -41 seconds (crosses the bucket edge)", 5.85),
    ("PPG-derived heart rate (v26 optical, 24 Hz autocorrelation)", "output: every sample the series mean", 4.5),
    ("PPG-derived heart rate (v26 optical, 24 Hz autocorrelation)", "input: samples shuffled (same distribution)", 4.5),
    ("PPG-derived heart rate (v26 optical, 24 Hz autocorrelation)", "input: last 10% dropped", 0.9),
    ("Irregular-rhythm screen (COSEn episodes + ectopy veto)", "output: every interval the stretch's own median", 0.262),
    ("Irregular-rhythm screen (COSEn episodes + ectopy veto)", "output: every interval a fixed 800 ms", 0.262),
    ("Irregular-rhythm screen (COSEn episodes + ectopy veto)", "input: intervals shuffled (same distribution, no order)", 0.104),
    ("Irregular-rhythm screen (COSEn episodes + ectopy veto)", "input: each stretch time-reversed", 0.00751),
    ("Irregular-rhythm screen (COSEn episodes + ectopy veto)", "input: every interval x1.10 (a 10% slower rhythm)", 0.03),
    ("Irregular-rhythm screen (COSEn episodes + ectopy veto)", "oracle: fibrillation and sinus labels swapped", 0.262),
    ("R-R stream quality (coverage, duplication, 1000/1024 rescale)", "input: every beat stored twice", 0.9),
    ("R-R stream quality (coverage, duplication, 1000/1024 rescale)", "input: a 1000/1024 second copy of every beat", 0.9),
    ("R-R stream quality (coverage, duplication, 1000/1024 rescale)", "input: timeline compressed 4x", 0.45),
    ("R-R stream quality (coverage, duplication, 1000/1024 rescale)", "input: half the beats below any physiological floor", 0.45),
    ("R-R stream quality (coverage, duplication, 1000/1024 rescale)", "input: only 4 beats", 0.45),
];

/// `(metric, arm, why)`. Probe arms that cannot carry a floor, because the mutation does not move the
/// number at all. Their blindness is the finding, not a defect to assert away.
const NO_FLOOR: &[(&str, &str, &str)] = &[
    ("ECG atrial-band power ratio", "input: sawtooth", "measured delta is exactly zero: this mutation does not move the number"),
    ("ECG atrial-band power ratio", "input: time-reversed", "measured delta is exactly zero: this mutation does not move the number"),
    ("ECG atrial-band power ratio", "input: lead inverted", "measured delta is exactly zero: this mutation does not move the number"),
    ("Mains anchor (recover the sample rate from 50 Hz line interference)", "input: time-reversed", "measured delta is exactly zero: this mutation does not move the number"),
    ("Mains anchor (recover the sample rate from 50 Hz line interference)", "input: first half only", "measured delta is exactly zero: this mutation does not move the number"),
    ("ECG P-wave morphology (present/absent + deflection ratio)", "input: lead inverted", "measured delta is exactly zero: this mutation does not move the number"),
    ("ECG P-wave morphology (present/absent + deflection ratio)", "input: time-reversed (the T wave lands in the PR window)", "measured delta is exactly zero: this mutation does not move the number"),
    ("ECG P-wave morphology (present/absent + deflection ratio)", "input: matched PPG pulse waveform", "measured delta is exactly zero: this mutation does not move the number"),
    ("ECG QRS detector pair (Pan-Tompkins + wavelet) and bSQI agreement", "input: lead inverted", "measured delta is exactly zero: this mutation does not move the number"),
    ("ECG signal-quality gate (bSQI/kSQI/pSQI/basSQI/templateSQI + HR)", "input: positives time-reversed", "measured delta is exactly zero: this mutation does not move the number"),
    ("ECG signal-quality gate (bSQI/kSQI/pSQI/basSQI/templateSQI + HR)", "input: positives inverted", "measured delta is exactly zero: this mutation does not move the number"),
    ("ECG signal-quality gate (bSQI/kSQI/pSQI/basSQI/templateSQI + HR)", "gate: kSQI removed from the conjunction", "measured delta is exactly zero: this mutation does not move the number"),
    ("PPG-HR bucket aggregate, poor-signal derate, span trust verdict", "input: the strap's poor-signal set emptied", "measured delta is exactly zero: this mutation does not move the number"),
    ("PPG-HR bucket aggregate, poor-signal derate, span trust verdict", "output: shifted +1 second", "measured delta is exactly zero: this mutation does not move the number"),
    ("PPG-HR bucket aggregate, poor-signal derate, span trust verdict", "input: estimate order reversed", "measured delta is exactly zero: this mutation does not move the number"),
    ("PPG-derived heart rate (v26 optical, 24 Hz autocorrelation)", "output: shifted +1 second", "measured delta is exactly zero: this mutation does not move the number"),
    ("PPG-derived heart rate (v26 optical, 24 Hz autocorrelation)", "input: optical sign inverted", "measured delta is exactly zero: this mutation does not move the number"),
    ("PPG-derived heart rate (v26 optical, 24 Hz autocorrelation)", "input: waveform time-reversed", "measured delta is exactly zero: this mutation does not move the number"),
    ("R-R stream quality (coverage, duplication, 1000/1024 rescale)", "input: interval order reversed", "measured delta is exactly zero: this mutation does not move the number"),
];

/// Assert one metric's floors, and require every NULL/STRUCTURAL arm to be classified.
fn enforce_floors(metric: &str, base: f64, probes: &[(&str, f64)]) {
    let (mut asserted, mut waived) = (0usize, 0usize);
    let mut breached: Vec<String> = Vec::new();
    let mut unclassified: Vec<&str> = Vec::new();
    for &(arm, value) in probes {
        let floor = FLOORS.iter().find(|(m, a, _)| *m == metric && *a == arm).map(|t| t.2);
        let waiver = NO_FLOOR.iter().find(|(m, a, _)| *m == metric && *a == arm).map(|t| t.2);
        match (floor, waiver) {
            (Some(_), Some(_)) => breached.push(format!("'{arm}' carries both a floor and a waiver")),
            (Some(d), None) => {
                asserted += 1;
                let moved = (value - base).abs();
                if moved.is_nan() || moved < d {
                    breached.push(format!("'{arm}' moved {moved} against a floor of {d}"));
                }
            }
            (None, Some(w)) => {
                waived += 1;
                println!("   no floor: {arm} — {w}");
            }
            (None, None) => unclassified.push(arm),
        }
    }
    let orphans: Vec<&str> = FLOORS
        .iter()
        .filter(|(m, _, _)| *m == metric)
        .map(|t| t.1)
        .chain(NO_FLOOR.iter().filter(|(m, _, _)| *m == metric).map(|t| t.1))
        .filter(|a| !probes.iter().any(|(p, _)| *p == *a))
        .collect();
    println!("   floors: {asserted} asserted, {waived} un-floorable");
    assert!(
        unclassified.is_empty(),
        "{metric}: probe arms carry neither a floor nor a waiver — classify them: {unclassified:?}"
    );
    assert!(orphans.is_empty(), "{metric}: floor rows match no arm — stale or misspelt: {orphans:?}");
    assert!(breached.is_empty(), "{metric}: SENSITIVITY FLOOR BREACHED — {}", breached.join(" | "));
}

#[test]
#[ignore = "negative control over committed fixtures; run with --release --ignored"]
fn control_ppg_hr_golden() {
    let src = ppg_samples();
    let values: Vec<i32> = src.iter().map(|s| s.value).collect();
    let mut t = Table::new(
        "PPG-derived heart rate (v26 optical, 24 Hz autocorrelation)",
        "ppg_hr_real.rs:32/:36-38/:44 — 10 estimates, first {ts 1783955687, bpm 78, conf 0.733}, \
         every conf>=0.7 estimate inside [60,100] bpm",
    );

    let base = ppg::estimate(&src);
    t.add("baseline (unmutated)", Kind::Baseline, base.len() as f64, ppg_gate(&base), ppg_note(&base));

    // NULL arms.
    let flat = mean(&values.iter().map(|&v| f64::from(v)).collect::<Vec<f64>>()).round() as i32;
    let e = ppg::estimate(&ppg_map(&src, |_| flat));
    t.add("output: every sample the series mean", Kind::Null, e.len() as f64, ppg_gate(&e), ppg_note(&e));

    let sh = shuffled_i32(&values, 0x9_9001);
    let shuf: Vec<ppg::Sample> =
        src.iter().zip(&sh).map(|(s, &v)| ppg::Sample { ts: s.ts, value: v }).collect();
    let e = ppg::estimate(&shuf);
    t.add("input: samples shuffled (same distribution)", Kind::Null, e.len() as f64, ppg_gate(&e), ppg_note(&e));

    // STRUCTURAL arms.
    let shifted: Vec<ppg::Sample> =
        src.iter().map(|s| ppg::Sample { ts: s.ts + 1, value: s.value }).collect();
    let e = ppg::estimate(&shifted);
    t.add("output: shifted +1 second", Kind::Structural, e.len() as f64, ppg_gate(&e), ppg_note(&e));

    let e = ppg::estimate(&ppg_map(&src, |v| -v));
    t.add("input: optical sign inverted", Kind::Structural, e.len() as f64, ppg_gate(&e), ppg_note(&e));

    let mut rev = values.clone();
    rev.reverse();
    let reversed: Vec<ppg::Sample> =
        src.iter().zip(&rev).map(|(s, &v)| ppg::Sample { ts: s.ts, value: v }).collect();
    let e = ppg::estimate(&reversed);
    t.add("input: waveform time-reversed", Kind::Structural, e.len() as f64, ppg_gate(&e), ppg_note(&e));

    let keep = src.len() * 9 / 10;
    let e = ppg::estimate(&src[..keep]);
    t.add("input: last 10% dropped", Kind::Structural, e.len() as f64, ppg_gate(&e), ppg_note(&e));

    // PARAMETER arms. SAMPLE_RATE_HZ / WINDOW_SECONDS / MIN_BPM / MAX_BPM / MIN_CONFIDENCE are bare
    // `pub const`s that `estimate` reads directly, so none is reachable as an argument.
    for n in [22usize, 27] {
        let eff = f64::from(ppg::SAMPLE_RATE_HZ as u32) * ppg::SAMPLE_RATE_HZ as f64 / n as f64;
        let e = ppg::estimate(&ppg_resecond(&src, n));
        t.add(
            format!("param proxy: SAMPLE_RATE_HZ 24 -> {eff:.2}"),
            Kind::Parameter,
            e.len() as f64,
            ppg_gate(&e),
            format!("proxy: input re-laid at {n} samples/s. {}", ppg_note(&e)),
        );
    }

    for f in [0.10, -0.10, 0.005] {
        let cut = pct(ppg::MIN_CONFIDENCE, f);
        let kept: Vec<ppg::Estimate> = base.iter().copied().filter(|e| e.conf >= cut).collect();
        t.add(
            format!("param: MIN_CONFIDENCE 0.30 -> {cut:.4}"),
            Kind::Parameter,
            kept.len() as f64,
            ppg_gate(&kept),
            format!("re-applied to the emitted conf (the emission gate tests the ACF max). {}", ppg_note(&kept)),
        );
    }

    for f in [0.10, -0.10, 0.005] {
        let floor = pct(PPG_CONFIDENT_FLOOR, f);
        let ok = ppg_gate_with(&base, floor, PPG_CONFIDENT_LO, PPG_CONFIDENT_HI);
        t.add(
            format!("param: gate conf floor 0.70 -> {floor:.4}"),
            Kind::Parameter,
            base.len() as f64,
            ok,
            "re-applied: the gate's own filter at ppg_hr_real.rs:43",
        );
    }

    for f in [0.10, -0.10] {
        let (lo, hi) = (
            pct(f64::from(PPG_CONFIDENT_LO), f).round() as i32,
            pct(f64::from(PPG_CONFIDENT_HI), f).round() as i32,
        );
        let ok = ppg_gate_with(&base, PPG_CONFIDENT_FLOOR, lo, hi);
        t.add(
            format!("param: confident_bpm_range [60,100] -> [{lo},{hi}]"),
            Kind::Parameter,
            base.len() as f64,
            ok,
            "re-applied: the golden band at ppg_hr_real.rs:44",
        );
    }

    let e = ppg::estimate(&ppg_map(&src, |v| ((f64::from(v)) * 1.10).round() as i32));
    t.add(
        "param proxy: amplitude x1.10",
        Kind::Parameter,
        e.len() as f64,
        ppg_gate(&e),
        format!("proxy for any amplitude-scaled constant. {}", ppg_note(&e)),
    );

    t.finish();
}

// ---------------------------------------------------------------------------------------------
// Metric 2 — PPG-HR bucket aggregate, poor-signal derate, span trust verdict.
// Gate: crates/physio-algo/src/ppg.rs:361-364, :394-395, :417-435.
// ---------------------------------------------------------------------------------------------

/// ppg.rs:361-363.
const AGG_TS: i64 = 60;
const AGG_BPM: i32 = 73;
const AGG_CONF: f64 = 0.2625;
const AGG_CONF_TOL: f64 = 1e-3;
/// ppg.rs:394.
const DERATE_CONFS: [f64; 3] = [0.9, 0.0, 0.9];

fn agg_input() -> Vec<ppg::Estimate> {
    vec![
        ppg::Estimate { ts: 100, bpm: 60, conf: 0.9 },
        ppg::Estimate { ts: 101, bpm: 150, conf: 0.05 },
        ppg::Estimate { ts: 102, bpm: 150, conf: 0.05 },
        ppg::Estimate { ts: 103, bpm: 150, conf: 0.05 },
    ]
}

fn derate_input() -> Vec<ppg::Estimate> {
    vec![
        ppg::Estimate { ts: 10, bpm: 60, conf: 0.9 },
        ppg::Estimate { ts: 11, bpm: 150, conf: 0.9 },
        ppg::Estimate { ts: 12, bpm: 61, conf: 0.9 },
    ]
}

fn clean_span(clean: usize) -> Vec<ppg::Estimate> {
    (0..clean as i64).map(|t| ppg::Estimate { ts: t, bpm: 60, conf: 0.9 }).collect()
}

/// `ppg::signal_check` with its three thresholds exposed, so a threshold arm is an exact
/// re-application rather than a proxy. Pinned against the shipped function at the baseline.
fn check_with(
    est: &[ppg::Estimate],
    start: i64,
    end: i64,
    good_conf: f64,
    fair_frac: f64,
    good_frac: f64,
) -> ppg::SignalCheck {
    if end < start {
        return ppg::SignalCheck::Poor;
    }
    let span = (end - start + 1) as f64;
    let clean: HashSet<i64> = est
        .iter()
        .filter(|e| e.ts >= start && e.ts <= end && e.conf > good_conf)
        .map(|e| e.ts)
        .collect();
    let frac = clean.len() as f64 / span;
    if frac > good_frac {
        ppg::SignalCheck::Good
    } else if frac > fair_frac {
        ppg::SignalCheck::Fair
    } else {
        ppg::SignalCheck::Poor
    }
}

struct AggArm {
    out: Vec<ppg::Estimate>,
    confs: Vec<f64>,
    levels: [ppg::SignalCheck; 3],
}

fn agg_gate(a: &AggArm) -> bool {
    a.out.len() == 1
        && a.out[0].ts == AGG_TS
        && a.out[0].bpm == AGG_BPM
        && (a.out[0].conf - AGG_CONF).abs() < AGG_CONF_TOL
        && a.confs.len() == 3
        && a.confs.iter().zip(DERATE_CONFS.iter()).all(|(x, y)| (x - y).abs() < 1e-12)
        && a.levels == [ppg::SignalCheck::Good, ppg::SignalCheck::Fair, ppg::SignalCheck::Poor]
}

fn agg_arm(
    est: &[ppg::Estimate],
    derate: &[ppg::Estimate],
    poor: &HashSet<i64>,
    spans: [usize; 3],
    good_conf: f64,
    fair_frac: f64,
    good_frac: f64,
) -> AggArm {
    AggArm {
        out: ppg::aggregate(est, 60),
        confs: ppg::derate_poor_seconds(derate, poor).iter().map(|e| e.conf).collect(),
        levels: spans
            .map(|n| check_with(&clean_span(n), 0, 99, good_conf, fair_frac, good_frac)),
    }
}

fn agg_default(est: &[ppg::Estimate], derate: &[ppg::Estimate], poor: &HashSet<i64>) -> AggArm {
    agg_arm(
        est,
        derate,
        poor,
        [80, 30, 10],
        ppg::GOOD_CONFIDENCE,
        ppg::SIGNAL_CHECK_FAIR_FRACTION,
        ppg::SIGNAL_CHECK_GOOD_FRACTION,
    )
}

#[test]
#[ignore = "negative control; run with --release --ignored"]
fn control_ppg_aggregate_derate_and_signal_check() {
    let est = agg_input();
    let derate = derate_input();
    let poor: HashSet<i64> = [11i64].into_iter().collect();
    let mut t = Table::new(
        "PPG-HR bucket aggregate, poor-signal derate, span trust verdict",
        "ppg.rs:361-364 bucket {ts 60, bpm 73, conf 0.2625}; :394 derate confs [0.9,0.0,0.9]; \
         :417-435 span(80)/span(30)/span(10) over [0,99] = Good/Fair/Poor",
    );

    // The re-implemented verdict must agree with the shipped one before any threshold arm is read.
    for n in [80usize, 30, 10] {
        assert_eq!(
            check_with(
                &clean_span(n),
                0,
                99,
                ppg::GOOD_CONFIDENCE,
                ppg::SIGNAL_CHECK_FAIR_FRACTION,
                ppg::SIGNAL_CHECK_GOOD_FRACTION
            ),
            ppg::signal_check(&clean_span(n), 0, 99),
            "the re-applied signal_check diverges from the shipped one at span({n})"
        );
    }

    let base = agg_default(&est, &derate, &poor);
    t.add("baseline (unmutated)", Kind::Baseline, f64::from(base.out[0].bpm), agg_gate(&base), "");

    // NULL arms.
    let flat: Vec<ppg::Estimate> =
        est.iter().map(|e| ppg::Estimate { conf: 0.9, ..*e }).collect();
    let a = agg_default(&flat, &derate, &poor);
    t.add(
        "input: every confidence identical (weights carry nothing)",
        Kind::Null,
        f64::from(a.out[0].bpm),
        agg_gate(&a),
        "a flat weight makes the weighted mean the plain mean",
    );

    let zero: Vec<ppg::Estimate> = est.iter().map(|e| ppg::Estimate { conf: 0.0, ..*e }).collect();
    let a = agg_default(&zero, &derate, &poor);
    t.add(
        "input: every confidence zero (fallback path)",
        Kind::Null,
        f64::from(a.out[0].bpm),
        agg_gate(&a),
        "the zero-weight fallback is the plain mean",
    );

    let a = agg_default(&est, &derate, &HashSet::new());
    t.add(
        "input: the strap's poor-signal set emptied",
        Kind::Null,
        f64::from(a.out[0].bpm),
        agg_gate(&a),
        "derate becomes a no-op",
    );

    // STRUCTURAL arms.
    let mut swapped = est.clone();
    let (b0, b1) = (swapped[0].bpm, swapped[1].bpm);
    swapped[0].bpm = b1;
    swapped[1].bpm = b0;
    let a = agg_default(&swapped, &derate, &poor);
    t.add(
        "output: the confident and the noisy bpm swapped",
        Kind::Structural,
        f64::from(a.out[0].bpm),
        agg_gate(&a),
        "the weights now back the wrong second",
    );

    let plus1: Vec<ppg::Estimate> = est.iter().map(|e| ppg::Estimate { ts: e.ts + 1, ..*e }).collect();
    let a = agg_default(&plus1, &derate, &poor);
    t.add(
        "output: shifted +1 second",
        Kind::Structural,
        f64::from(a.out[0].bpm),
        agg_gate(&a),
        "still inside the same 60 s bucket",
    );

    let back: Vec<ppg::Estimate> = est.iter().map(|e| ppg::Estimate { ts: e.ts - 41, ..*e }).collect();
    let a = agg_default(&back, &derate, &poor);
    t.add(
        "output: shifted -41 seconds (crosses the bucket edge)",
        Kind::Structural,
        f64::from(a.out[0].bpm),
        agg_gate(&a),
        "splits into two buckets",
    );

    let mut order = est.clone();
    order.reverse();
    let a = agg_default(&order, &derate, &poor);
    t.add(
        "input: estimate order reversed",
        Kind::Structural,
        f64::from(a.out[0].bpm),
        agg_gate(&a),
        "aggregation is order-free",
    );

    // PARAMETER arms.
    for f in [0.10, -0.10, 0.005] {
        let g = pct(ppg::GOOD_CONFIDENCE, f);
        let a = agg_arm(
            &est,
            &derate,
            &poor,
            [80, 30, 10],
            g,
            ppg::SIGNAL_CHECK_FAIR_FRACTION,
            ppg::SIGNAL_CHECK_GOOD_FRACTION,
        );
        t.add(
            format!("param: GOOD_CONFIDENCE 0.50 -> {g:.4}"),
            Kind::Parameter,
            f64::from(a.out[0].bpm),
            agg_gate(&a),
            "re-applied",
        );
    }
    for f in [0.10, -0.10] {
        let g = pct(ppg::SIGNAL_CHECK_GOOD_FRACTION, f);
        let a = agg_arm(&est, &derate, &poor, [80, 30, 10], ppg::GOOD_CONFIDENCE, ppg::SIGNAL_CHECK_FAIR_FRACTION, g);
        t.add(
            format!("param: SIGNAL_CHECK_GOOD_FRACTION 0.50 -> {g:.4}"),
            Kind::Parameter,
            f64::from(a.out[0].bpm),
            agg_gate(&a),
            "re-applied",
        );
        let fa = pct(ppg::SIGNAL_CHECK_FAIR_FRACTION, f);
        let a = agg_arm(&est, &derate, &poor, [80, 30, 10], ppg::GOOD_CONFIDENCE, fa, ppg::SIGNAL_CHECK_GOOD_FRACTION);
        t.add(
            format!("param: SIGNAL_CHECK_FAIR_FRACTION 0.20 -> {fa:.4}"),
            Kind::Parameter,
            f64::from(a.out[0].bpm),
            agg_gate(&a),
            "re-applied",
        );
    }
    for e in [1.10f64, 0.90] {
        let bent: Vec<ppg::Estimate> =
            est.iter().map(|x| ppg::Estimate { conf: x.conf.powf(e), ..*x }).collect();
        let a = agg_default(&bent, &derate, &poor);
        t.add(
            format!("param proxy: confidence-weight exponent 1.0 -> {e:.2}"),
            Kind::Parameter,
            f64::from(a.out[0].bpm),
            agg_gate(&a),
            "proxy: the blend has no exponent to reach, so the weights are bent instead",
        );
    }

    t.finish();
}

// ---------------------------------------------------------------------------------------------
// Metric 3 — ECG QRS detection pair and bSQI agreement.
// Gate: crates/physio-algo/tests/ecg_qrs_agreement.rs:124-127, :133-134.
// ---------------------------------------------------------------------------------------------

/// ecg_qrs_agreement.rs:133.
const QRS_WORST_F1_MIN: f64 = 0.95;
/// ecg_qrs_agreement.rs:134.
const QRS_EXCESS_MIN: f64 = 0.70;
/// ecg_qrs_agreement.rs:124-125.
const QRS_HR_LO: f64 = 30.0;
const QRS_HR_HI: f64 = 220.0;
/// ecg_qrs_agreement.rs:127.
const QRS_RATE_TOL: f64 = 0.05;

struct QrsResult {
    worst_f1: f64,
    worst_excess: f64,
    ok: bool,
}

fn qrs_eval(sigs: &[(f64, Vec<f64>)], window_ms: f64) -> QrsResult {
    let (mut worst_f1, mut worst_excess, mut ok) = (f64::INFINITY, f64::INFINITY, true);
    for (fs, x) in sigs {
        let a = ecg::detect_pan_tompkins(x, *fs);
        let b = ecg::detect_wavelet(x, *fs);
        let g = ecg::beat_agreement(&a, &b, x.len(), *fs, window_ms);
        let bpm = |n: usize| n as f64 / (x.len() as f64 / fs) * 60.0;
        let (ha, hb) = (bpm(a.len()), bpm(b.len()));
        ok &= (QRS_HR_LO..=QRS_HR_HI).contains(&ha)
            && (QRS_HR_LO..=QRS_HR_HI).contains(&hb)
            && (ha - hb).abs() / ha <= QRS_RATE_TOL;
        worst_f1 = worst_f1.min(g.f1);
        worst_excess = worst_excess.min(g.excess);
    }
    ok &= worst_f1 >= QRS_WORST_F1_MIN && worst_excess >= QRS_EXCESS_MIN;
    QrsResult { worst_f1, worst_excess, ok }
}

fn qrs_row(t: &mut Table, arm: &str, kind: Kind, sigs: &[(f64, Vec<f64>)], window_ms: f64, note: &str) {
    let r = qrs_eval(sigs, window_ms);
    t.add(arm, kind, r.worst_f1, r.ok, format!("worst excess {:.3}. {note}", r.worst_excess));
}

#[test]
#[ignore = "negative control over the AAUWSS corpus; run with --release --ignored"]
fn control_ecg_qrs_agreement() {
    let fx = ecg_fixtures();
    assert_eq!(fx.len(), 13, "AAUWSS ships 13 ECG subjects; found {}", fx.len());
    let base: Vec<(f64, Vec<f64>)> = fx.iter().map(|f| (f.fs_hz, f.samples.clone())).collect();
    let mut t = Table::new(
        "ECG QRS detector pair (Pan-Tompkins + wavelet) and bSQI agreement",
        "ecg_qrs_agreement.rs:133 worst real F1 >= 0.95; :134 every excess >= 0.70; \
         :124-127 both detectors in 30..220 bpm and within 5% of each other",
    );

    qrs_row(&mut t, "baseline (unmutated)", Kind::Baseline, &base, ecg::DEFAULT_MATCH_WINDOW_MS, "");

    // NULL arms.
    let g: Vec<(f64, Vec<f64>)> = fx
        .iter()
        .enumerate()
        .map(|(k, f)| (f.fs_hz, gaussian_like(&f.samples, 0xEC6_5000 + k as u64)))
        .collect();
    qrs_row(&mut t, "input: gaussian noise matched to each subject", Kind::Null, &g, ecg::DEFAULT_MATCH_WINDOW_MS, "");

    let c: Vec<(f64, Vec<f64>)> =
        fx.iter().map(|f| (f.fs_hz, vec![mean(&f.samples); f.samples.len()])).collect();
    qrs_row(&mut t, "input: constant (dead channel)", Kind::Null, &c, ecg::DEFAULT_MATCH_WINDOW_MS, "");

    let s: Vec<(f64, Vec<f64>)> = fx
        .iter()
        .enumerate()
        .map(|(k, f)| (f.fs_hz, shuffled(&f.samples, 0xEC6_5000 + k as u64)))
        .collect();
    qrs_row(&mut t, "input: samples shuffled (morphology destroyed)", Kind::Null, &s, ecg::DEFAULT_MATCH_WINDOW_MS, "");

    // STRUCTURAL arms.
    let saw: Vec<(f64, Vec<f64>)> = fx
        .iter()
        .map(|f| {
            let n = ecg::detect_pan_tompkins(&f.samples, f.fs_hz).len().max(2);
            (f.fs_hz, sawtooth_like(&f.samples, f.samples.len() as f64 / n as f64))
        })
        .collect();
    qrs_row(&mut t, "input: sawtooth at each subject's own heart rate", Kind::Structural, &saw, ecg::DEFAULT_MATCH_WINDOW_MS, "");

    let inv: Vec<(f64, Vec<f64>)> =
        fx.iter().map(|f| (f.fs_hz, f.samples.iter().map(|v| -v).collect())).collect();
    qrs_row(&mut t, "input: lead inverted", Kind::Structural, &inv, ecg::DEFAULT_MATCH_WINDOW_MS, "");

    let rev: Vec<(f64, Vec<f64>)> = fx
        .iter()
        .map(|f| {
            let mut x = f.samples.clone();
            x.reverse();
            (f.fs_hz, x)
        })
        .collect();
    qrs_row(&mut t, "input: time-reversed", Kind::Structural, &rev, ecg::DEFAULT_MATCH_WINDOW_MS, "");

    let cut: Vec<(f64, Vec<f64>)> =
        fx.iter().map(|f| (f.fs_hz, f.samples[..f.samples.len() * 9 / 10].to_vec())).collect();
    qrs_row(&mut t, "input: last 10% dropped", Kind::Structural, &cut, ecg::DEFAULT_MATCH_WINDOW_MS, "");

    let dec: Vec<(f64, Vec<f64>)> = fx
        .iter()
        .map(|f| (f.fs_hz, f.samples.iter().step_by(2).copied().collect::<Vec<f64>>()))
        .collect();
    qrs_row(&mut t, "input: decimated x2, rate still declared 200 Hz", Kind::Structural, &dec, ecg::DEFAULT_MATCH_WINDOW_MS, "implied rate doubles");

    // PARAMETER arms. DEFAULT_MATCH_WINDOW_MS is an argument, so those three are exact.
    for f in [0.10, -0.10, 0.005] {
        let w = pct(ecg::DEFAULT_MATCH_WINDOW_MS, f);
        qrs_row(&mut t, &format!("param: DEFAULT_MATCH_WINDOW_MS 100 -> {w:.3}"), Kind::Parameter, &base, w, "exact: it is an argument");
    }
    // MAX_QRS_MS / REFRACTORY_MS and every internal ms window are bare consts. Declaring the rate
    // k times high scales all of them by 1/k at once, which is the only equivalent an integration
    // test can reach without widening visibility.
    for f in [0.10, -0.10] {
        let scaled: Vec<(f64, Vec<f64>)> =
            fx.iter().map(|x| (pct(x.fs_hz, f), x.samples.clone())).collect();
        qrs_row(
            &mut t,
            &format!("param proxy: every ms window scaled {:+.0}%", -f * 100.0),
            Kind::Parameter,
            &scaled,
            ecg::DEFAULT_MATCH_WINDOW_MS,
            "proxy: declared rate moved, which scales MAX_QRS_MS and REFRACTORY_MS together",
        );
    }
    println!(
        "param: MIN_FS_HZ {} / MAX_FS_HZ {} are structurally inert on this corpus — every subject is \
         at 200 Hz, far inside both bounds, so no ±10% move can change a single sample.",
        ecg::MIN_FS_HZ,
        ecg::MAX_FS_HZ
    );

    t.finish();
}

// ---------------------------------------------------------------------------------------------
// Metric 4 — ECG signal-quality gate (bSQI/kSQI/pSQI/basSQI/templateSQI + HR verdict).
// Gate: crates/physio-algo/tests/ecg_sqi_and_mains.rs:182, :184, :190.
// ---------------------------------------------------------------------------------------------

const SQI_WINDOW_S: f64 = 10.0;
/// ecg_sqi_and_mains.rs:182 — no negative may be accepted.
const SQI_NEG_ACCEPT_MAX: usize = 0;
/// ecg_sqi_and_mains.rs:184 — `pos_accept * 10 >= pos_windows * 9`.
const SQI_POS_NUM: usize = 9;
const SQI_POS_DEN: usize = 10;

#[derive(Clone, Copy)]
struct SqiThresholds {
    b: f64,
    k: f64,
    p_lo: f64,
    p_hi: f64,
    tmpl: f64,
    hr_lo: f64,
    hr_hi: f64,
    /// Cleared to drop kSQI from the conjunction — the composition arm, not a threshold move.
    use_k: bool,
}

impl SqiThresholds {
    fn shipped() -> Self {
        SqiThresholds {
            b: score::B_SQI_MIN,
            k: score::K_SQI_MIN,
            p_lo: score::P_SQI_MIN,
            p_hi: score::P_SQI_MAX,
            tmpl: score::TEMPLATE_SQI_MIN,
            hr_lo: score::MIN_HR_BPM,
            hr_hi: score::MAX_HR_BPM,
            use_k: true,
        }
    }
}

/// Which of the five gated indices FAILED, in the order the shipped verdict reports them.
fn sqi_failures(s: &score::EcgScore, t: &SqiThresholds) -> [bool; 5] {
    let b_ok = s.b_sqi >= t.b;
    [
        !b_ok,
        t.use_k && !s.k_sqi.is_some_and(|v| v >= t.k),
        !s.p_sqi.is_some_and(|v| (t.p_lo..=t.p_hi).contains(&v)),
        !s.template_sqi.is_some_and(|v| v >= t.tmpl),
        !s.mean_hr_bpm.is_some_and(|v| (t.hr_lo..=t.hr_hi).contains(&v)),
    ]
}

fn sqi_accepted(s: &score::EcgScore, t: &SqiThresholds) -> bool {
    sqi_failures(s, t).iter().all(|f| !f)
}

fn chunks(samples: &[f64], fs_hz: f64, seconds: f64) -> Vec<Vec<f64>> {
    let n = (seconds * fs_hz) as usize;
    samples.chunks(n).filter(|c| c.len() == n).map(|c| c.to_vec()).collect()
}

fn sqi_negatives(f: &Fixture, ppg_ref: &Fixture, seed: u64) -> Vec<Vec<f64>> {
    let beats = score(&f.samples, f.fs_hz).beats.max(1);
    let period = f.samples.len() as f64 / beats as f64;
    let up = resample(&ppg_ref.samples, ppg_ref.fs_hz, f.fs_hz);
    vec![
        gaussian_like(&f.samples, seed),
        shuffled(&f.samples, seed),
        sawtooth_like(&f.samples, period),
        ecg_corpus::matched_to(&up, &f.samples),
    ]
}

fn sqi_gate(
    pos: &[score::EcgScore],
    neg: &[score::EcgScore],
    t: &SqiThresholds,
) -> (bool, f64) {
    let pos_accept = pos.iter().filter(|s| sqi_accepted(s, t)).count();
    let neg_accept = neg.iter().filter(|s| sqi_accepted(s, t)).count();
    let mut rejects = [0usize; 5];
    for s in neg {
        for (i, failed) in sqi_failures(s, t).iter().enumerate() {
            rejects[i] += usize::from(*failed);
        }
    }
    // ecg_sqi_and_mains.rs:190 exempts nothing, but a dropped index cannot reject and is not asked to.
    let each_works = rejects
        .iter()
        .enumerate()
        .all(|(i, c)| *c > 0 || (i == 1 && !t.use_k));
    let frac = if pos.is_empty() { f64::NAN } else { pos_accept as f64 / pos.len() as f64 };
    let ok = neg_accept == SQI_NEG_ACCEPT_MAX
        && pos_accept * SQI_POS_DEN >= pos.len() * SQI_POS_NUM
        && each_works;
    (ok, frac)
}

fn score_all(sigs: &[(f64, Vec<f64>)]) -> Vec<score::EcgScore> {
    sigs.iter()
        .flat_map(|(fs, x)| chunks(x, *fs, SQI_WINDOW_S).into_iter().map(|w| score(&w, *fs)))
        .collect()
}

#[test]
#[ignore = "negative control over the AAUWSS corpus; run with --release --ignored"]
fn control_ecg_sqi_gate() {
    let fx = ecg_fixtures();
    let ppg_ref = ppg_fixtures().pop().expect("one published PPG subject");
    let shipped = SqiThresholds::shipped();

    let positives: Vec<(f64, Vec<f64>)> = fx.iter().map(|f| (f.fs_hz, f.samples.clone())).collect();
    let neg_sigs: Vec<(f64, Vec<f64>)> = fx
        .iter()
        .enumerate()
        .flat_map(|(k, f)| {
            sqi_negatives(f, &ppg_ref, 0x5C1_0000u64.wrapping_add(k as u64))
                .into_iter()
                .map(move |s| (f.fs_hz, s))
        })
        .collect();

    let pos = score_all(&positives);
    let neg = score_all(&neg_sigs);

    // The re-applied verdict must reproduce the shipped one before any threshold arm is read.
    for s in pos.iter().chain(neg.iter()) {
        assert_eq!(
            sqi_accepted(s, &shipped),
            s.verdict.accepted,
            "the re-applied SQI verdict diverges from the shipped one: {s:?}"
        );
    }

    let mut t = Table::new(
        "ECG signal-quality gate (bSQI/kSQI/pSQI/basSQI/templateSQI + HR)",
        "ecg_sqi_and_mains.rs:182 no negative accepted; :184 at least 9 of every 10 real windows \
         accepted; :190 each of bSQI/kSQI/pSQI/tmplSQI/HR rejects at least one negative",
    );

    let (ok, frac) = sqi_gate(&pos, &neg, &shipped);
    t.add("baseline (unmutated)", Kind::Baseline, frac, ok, format!("{} positive windows", pos.len()));

    // NULL arms: the positives are replaced by something that carries no ECG.
    let alt = |name: &str, kind: Kind, sigs: &[(f64, Vec<f64>)], note: &str, t: &mut Table| {
        let p = score_all(sigs);
        let (ok, frac) = sqi_gate(&p, &neg, &shipped);
        t.add(name, kind, frac, ok, note);
    };

    let g: Vec<(f64, Vec<f64>)> = fx
        .iter()
        .enumerate()
        .map(|(k, f)| (f.fs_hz, gaussian_like(&f.samples, 0x5C1_9000 + k as u64)))
        .collect();
    alt("input: positives replaced by gaussian noise", Kind::Null, &g, "", &mut t);

    let c: Vec<(f64, Vec<f64>)> =
        fx.iter().map(|f| (f.fs_hz, vec![mean(&f.samples); f.samples.len()])).collect();
    alt("input: positives replaced by a constant", Kind::Null, &c, "", &mut t);

    // STRUCTURAL arms.
    let rev: Vec<(f64, Vec<f64>)> = fx
        .iter()
        .map(|f| {
            let mut x = f.samples.clone();
            x.reverse();
            (f.fs_hz, x)
        })
        .collect();
    alt("input: positives time-reversed", Kind::Structural, &rev, "", &mut t);

    let inv: Vec<(f64, Vec<f64>)> =
        fx.iter().map(|f| (f.fs_hz, f.samples.iter().map(|v| -v).collect())).collect();
    alt("input: positives inverted", Kind::Structural, &inv, "", &mut t);

    let wrong: Vec<(f64, Vec<f64>)> = fx.iter().map(|f| (f.fs_hz * 2.5, f.samples.clone())).collect();
    alt("input: positives read at 2.5x the true rate", Kind::Structural, &wrong, "", &mut t);

    let drop_k = SqiThresholds { use_k: false, ..shipped };
    let (ok, frac) = sqi_gate(&pos, &neg, &drop_k);
    t.add(
        "gate: kSQI removed from the conjunction",
        Kind::Structural,
        frac,
        ok,
        "kSQI is the only index that rejects the sawtooth",
    );

    // PARAMETER arms — exact, because every index value is on `EcgScore` and only the comparison moves.
    let param = |name: String, th: SqiThresholds, t: &mut Table| {
        let (ok, frac) = sqi_gate(&pos, &neg, &th);
        t.add(name, Kind::Parameter, frac, ok, "re-applied");
    };
    for f in [0.10, -0.10] {
        param(format!("param: B_SQI_MIN 0.60 -> {:.4}", pct(shipped.b, f)), SqiThresholds { b: pct(shipped.b, f), ..shipped }, &mut t);
        param(format!("param: K_SQI_MIN 2.00 -> {:.4}", pct(shipped.k, f)), SqiThresholds { k: pct(shipped.k, f), ..shipped }, &mut t);
        param(format!("param: P_SQI_MIN 0.50 -> {:.4}", pct(shipped.p_lo, f)), SqiThresholds { p_lo: pct(shipped.p_lo, f), ..shipped }, &mut t);
        param(format!("param: P_SQI_MAX 0.80 -> {:.4}", pct(shipped.p_hi, f)), SqiThresholds { p_hi: pct(shipped.p_hi, f), ..shipped }, &mut t);
        param(format!("param: TEMPLATE_SQI_MIN 0.70 -> {:.4}", pct(shipped.tmpl, f)), SqiThresholds { tmpl: pct(shipped.tmpl, f), ..shipped }, &mut t);
        param(format!("param: MIN_HR_BPM 30 -> {:.2}", pct(shipped.hr_lo, f)), SqiThresholds { hr_lo: pct(shipped.hr_lo, f), ..shipped }, &mut t);
        param(format!("param: MAX_HR_BPM 220 -> {:.2}", pct(shipped.hr_hi, f)), SqiThresholds { hr_hi: pct(shipped.hr_hi, f), ..shipped }, &mut t);
    }
    param(
        format!("param: K_SQI_MIN 2.00 -> {:.4}", pct(shipped.k, 0.005)),
        SqiThresholds { k: pct(shipped.k, 0.005), ..shipped },
        &mut t,
    );
    println!(
        "param: BAS_SQI_MIN {} is REPORTED, not gated (score.rs:29-32), so no move of it can change \
         a verdict; GATED_INDEX_COUNT {} is a count, covered by the kSQI-removal arm above.",
        score::BAS_SQI_MIN,
        score::GATED_INDEX_COUNT
    );

    t.finish();
}

// ---------------------------------------------------------------------------------------------
// Metric 5 — mains anchor (recover the sample rate from 50 Hz line interference).
// Gate: crates/physio-algo/tests/ecg_sqi_and_mains.rs:274, :285, :301.
// ---------------------------------------------------------------------------------------------

/// ecg_sqi_and_mains.rs:274 — every anchored subject must land within a hertz of its true rate.
const MAINS_TOL_HZ: f64 = 1.0;

fn mains_eval(fx: &[Fixture], cfg: mains::MainsConfig, xf: impl Fn(&Fixture) -> Vec<f64>) -> (usize, usize) {
    let (mut right, mut found) = (0usize, 0usize);
    for f in fx {
        if let mains::MainsAnchor::Found(fix) = ecg::mains_anchor_with(&xf(f), cfg) {
            found += 1;
            right += usize::from((fix.fs_hz - f.fs_hz).abs() < MAINS_TOL_HZ);
        }
    }
    (right, found)
}

fn mains_row(
    t: &mut Table,
    arm: impl Into<String>,
    kind: Kind,
    fx: &[Fixture],
    cfg: mains::MainsConfig,
    note: &str,
    xf: impl Fn(&Fixture) -> Vec<f64>,
) {
    let (right, found) = mains_eval(fx, cfg, xf);
    // :285 more than half anchored, :274 none of them wrong.
    let ok = right * 2 > fx.len() && right == found;
    t.add(arm, kind, right as f64, ok, format!("{found} anchored, {right} within 1 Hz. {note}"));
}

#[test]
#[ignore = "negative control over the AAUWSS corpus; run with --release --ignored"]
fn control_ecg_mains_anchor() {
    let fx = ecg_fixtures();
    let d = mains::MainsConfig::default();
    let mut t = Table::new(
        "Mains anchor (recover the sample rate from 50 Hz line interference)",
        "ecg_sqi_and_mains.rs:285 more than half the subjects anchor; :274 every anchored rate is \
         within 1.0 Hz of the true 200 Hz",
    );

    mains_row(&mut t, "baseline (unmutated)", Kind::Baseline, &fx, d, "", |f| f.samples.clone());

    // NULL arms.
    mains_row(&mut t, "input: gaussian noise (no line peak)", Kind::Null, &fx, d, "", |f| {
        gaussian_like(&f.samples, 0x5A1_0000)
    });
    mains_row(&mut t, "input: constant", Kind::Null, &fx, d, "", |f| vec![mean(&f.samples); f.samples.len()]);
    mains_row(&mut t, "input: samples shuffled (spectrum whitened)", Kind::Null, &fx, d, "", |f| {
        shuffled(&f.samples, 0x5A1_0000)
    });

    // STRUCTURAL arms.
    mains_row(&mut t, "input: 50 Hz notched out by a two-sample comb", Kind::Structural, &fx, d, "the shipped negative at :292", |f| {
        (2..f.samples.len()).map(|i| f.samples[i] + f.samples[i - 2]).collect()
    });
    mains_row(&mut t, "input: time-reversed", Kind::Structural, &fx, d, "", |f| {
        let mut x = f.samples.clone();
        x.reverse();
        x
    });
    mains_row(&mut t, "input: first half only", Kind::Structural, &fx, d, "coarser bin", |f| {
        f.samples[..f.samples.len() / 2].to_vec()
    });

    // PARAMETER arms — exact, `mains_anchor_with` takes the whole config.
    for f in [0.10, -0.10] {
        mains_row(
            &mut t,
            format!("param: mains_hz 50 -> {:.3}", pct(d.mains_hz, f)),
            Kind::Parameter,
            &fx,
            mains::MainsConfig { mains_hz: pct(d.mains_hz, f), ..d },
            "exact",
            |x| x.samples.clone(),
        );
        mains_row(
            &mut t,
            format!("param: fs_min_hz 110 -> {:.2}", pct(d.fs_min_hz, f)),
            Kind::Parameter,
            &fx,
            mains::MainsConfig { fs_min_hz: pct(d.fs_min_hz, f), ..d },
            "exact",
            |x| x.samples.clone(),
        );
        mains_row(
            &mut t,
            format!("param: fs_max_hz 1000 -> {:.2}", pct(d.fs_max_hz, f)),
            Kind::Parameter,
            &fx,
            mains::MainsConfig { fs_max_hz: pct(d.fs_max_hz, f), ..d },
            "exact",
            |x| x.samples.clone(),
        );
        mains_row(
            &mut t,
            format!("param: min_prominence_db 15 -> {:.3}", pct(d.min_prominence_db, f)),
            Kind::Parameter,
            &fx,
            mains::MainsConfig { min_prominence_db: pct(d.min_prominence_db, f), ..d },
            "exact",
            |x| x.samples.clone(),
        );
        mains_row(
            &mut t,
            format!("param: min_margin_db 3 -> {:.3}", pct(d.min_margin_db, f)),
            Kind::Parameter,
            &fx,
            mains::MainsConfig { min_margin_db: pct(d.min_margin_db, f), ..d },
            "exact",
            |x| x.samples.clone(),
        );
    }
    mains_row(
        &mut t,
        format!("param: min_prominence_db 15 -> {:.4}", pct(d.min_prominence_db, 0.005)),
        Kind::Parameter,
        &fx,
        mains::MainsConfig { min_prominence_db: pct(d.min_prominence_db, 0.005), ..d },
        "exact, floor probe",
        |x| x.samples.clone(),
    );
    println!(
        "note: the test-side MAINS_WINDOW_S 10.0 (ecg_sqi_and_mains.rs:28) is not exercised here — \
         this control anchors on the whole 30 s epoch, exactly as the shipped :262 test does."
    );

    t.finish();
}

// ---------------------------------------------------------------------------------------------
// Metric 6 — ECG P-wave morphology.
// Gate: crates/physio-algo/tests/ecg_morphology.rs:48-49, :59, :60.
// ---------------------------------------------------------------------------------------------

/// ecg_morphology.rs:48 — every one of the 13 subjects reads Present.
const P_PRESENT_REQUIRED: usize = 13;
/// ecg_morphology.rs:59 — `worst_amp > 2.0 * P_MIN_DEFLECTION_RATIO`.
const P_WORST_AMP_MULT: f64 = 2.0;
/// ecg_morphology.rs:60.
const P_WORST_FRACTION_MIN: f64 = 0.90;

struct PParams {
    present_fraction_min: f64,
    min_deflection_ratio: f64,
    min_beats: usize,
    worst_fraction_min: f64,
    amp_mult: f64,
}

impl PParams {
    fn shipped() -> Self {
        PParams {
            present_fraction_min: morphology::p_wave::P_PRESENT_FRACTION_MIN,
            min_deflection_ratio: morphology::p_wave::P_MIN_DEFLECTION_RATIO,
            min_beats: morphology::p_wave::P_MIN_BEATS,
            worst_fraction_min: P_WORST_FRACTION_MIN,
            amp_mult: P_WORST_AMP_MULT,
        }
    }
}

/// Read the P-wave evidence for every subject under `xf`, then apply the shipped gate with `p`'s
/// thresholds substituted. `present_fraction >= present_fraction_min` together with the deflection
/// floor is the shipped `Present` rule (p_wave.rs:196-199) re-applied to the values the reading
/// already reports; the noise-floor and ambiguity arms of that rule are NOT reachable this way and
/// are covered by the input-side proxies instead.
fn p_eval(fx: &[Fixture], p: &PParams, xf: impl Fn(&Fixture) -> Vec<f64>) -> (usize, bool, f64, f64) {
    let (mut present, mut worst_frac, mut worst_amp, mut beats_ok) =
        (0usize, f64::INFINITY, f64::INFINITY, true);
    for f in fx {
        let x = xf(f);
        let peaks = ecg::detect_pan_tompkins(&x, f.fs_hz);
        let r = morphology::p_wave(&x, f.fs_hz, &peaks);
        let frac = r.present_fraction.unwrap_or(f64::NAN);
        let amp = r.amplitude_ratio.unwrap_or(f64::NAN);
        present += usize::from(amp >= p.min_deflection_ratio && frac >= p.present_fraction_min);
        beats_ok &= r.beats_examined >= p.min_beats;
        worst_frac = worst_frac.min(frac);
        worst_amp = worst_amp.min(amp);
    }
    let ok = present == P_PRESENT_REQUIRED
        && beats_ok
        && worst_amp > p.amp_mult * p.min_deflection_ratio
        && worst_frac >= p.worst_fraction_min;
    (present, ok, worst_frac, worst_amp)
}

fn p_row(
    t: &mut Table,
    arm: impl Into<String>,
    kind: Kind,
    fx: &[Fixture],
    p: &PParams,
    note: &str,
    xf: impl Fn(&Fixture) -> Vec<f64>,
) {
    let (present, ok, frac, amp) = p_eval(fx, p, xf);
    t.add(arm, kind, present as f64, ok, format!("worst frac {frac:.3}, worst amp {amp:.4}. {note}"));
}

/// Replace each PR window with a straight line between its own end points plus the record's own
/// high-frequency noise, leaving the QRS and the T wave alone — the one manipulation that targets
/// exactly what this measure claims to see.
fn flatten_pr(x: &[f64], fs_hz: f64, peaks: &[usize], seed: u64) -> Vec<f64> {
    let mut out = x.to_vec();
    let lo = (0.260 * fs_hz).round() as usize;
    let hi = (0.060 * fs_hz).round() as usize;
    let mut rng = Rng(seed);
    for &p in peaks {
        if p < lo || p >= x.len() {
            continue;
        }
        let (start, end) = (p - lo, p - hi);
        let steps: Vec<f64> = x[start..end].windows(2).map(|w| w[1] - w[0]).collect();
        let scale = if steps.is_empty() { 0.0 } else { ecg_corpus::sd(&steps) / 2f64.sqrt() };
        let (a, b) = (x[start], x[end]);
        let span = (end - start) as f64;
        for (k, v) in out[start..end].iter_mut().enumerate() {
            *v = a + (b - a) * k as f64 / span + scale * rng.gaussian();
        }
    }
    out
}

#[test]
#[ignore = "negative control over the AAUWSS corpus; run with --release --ignored"]
fn control_ecg_p_wave_morphology() {
    let fx = ecg_fixtures();
    let ppg_ref = ppg_fixtures().pop().expect("one published PPG subject");
    let d = PParams::shipped();
    let mut t = Table::new(
        "ECG P-wave morphology (present/absent + deflection ratio)",
        "ecg_morphology.rs:48 all 13 subjects Present; :49 beats_examined >= P_MIN_BEATS; \
         :59 worst deflection > 2 x P_MIN_DEFLECTION_RATIO; :60 worst present fraction >= 0.90",
    );

    // The re-applied Present rule must agree with the shipped finding at the shipped thresholds.
    for f in &fx {
        let peaks = ecg::detect_pan_tompkins(&f.samples, f.fs_hz);
        let r = morphology::p_wave(&f.samples, f.fs_hz, &peaks);
        let re = r.amplitude_ratio.unwrap_or(f64::NAN) >= d.min_deflection_ratio
            && r.present_fraction.unwrap_or(f64::NAN) >= d.present_fraction_min;
        assert_eq!(
            re,
            r.finding == morphology::PWaveFinding::Present,
            "subject {}: the re-applied Present rule diverges from the shipped finding: {r:?}",
            f.subject
        );
    }

    p_row(&mut t, "baseline (unmutated)", Kind::Baseline, &fx, &d, "", |f| f.samples.clone());

    // NULL arms.
    p_row(&mut t, "input: gaussian noise", Kind::Null, &fx, &d, "", |f| gaussian_like(&f.samples, 0xEC6_7000));
    p_row(&mut t, "input: samples shuffled", Kind::Null, &fx, &d, "", |f| shuffled(&f.samples, 0xEC6_7000));

    // STRUCTURAL arms.
    p_row(&mut t, "input: PR segment replaced by a line plus noise", Kind::Structural, &fx, &d, "the shipped manipulation at :153", |f| {
        let peaks = ecg::detect_pan_tompkins(&f.samples, f.fs_hz);
        flatten_pr(&f.samples, f.fs_hz, &peaks, 0xEC6_2000)
    });
    p_row(&mut t, "input: lead inverted", Kind::Structural, &fx, &d, "", |f| f.samples.iter().map(|v| -v).collect());
    p_row(&mut t, "input: time-reversed (the T wave lands in the PR window)", Kind::Structural, &fx, &d, "", |f| {
        let mut x = f.samples.clone();
        x.reverse();
        x
    });
    p_row(&mut t, "input: matched PPG pulse waveform", Kind::Structural, &fx, &d, "the documented limitation at :71", |f| {
        ecg_corpus::matched_to(&resample(&ppg_ref.samples, ppg_ref.fs_hz, f.fs_hz), &f.samples)
    });

    // PARAMETER arms. The thresholds that compare against a reported value are exact; the ms windows
    // and the SNR rule are bare consts with no argument, so they are reached from the input side.
    for f in [0.10, -0.10, 0.005] {
        let p = PParams { present_fraction_min: pct(d.present_fraction_min, f), ..PParams::shipped() };
        p_row(&mut t, format!("param: P_PRESENT_FRACTION_MIN 0.60 -> {:.4}", p.present_fraction_min), Kind::Parameter, &fx, &p, "re-applied", |x| x.samples.clone());
    }
    for f in [0.10, -0.10] {
        let p = PParams { min_deflection_ratio: pct(d.min_deflection_ratio, f), ..PParams::shipped() };
        p_row(&mut t, format!("param: P_MIN_DEFLECTION_RATIO 0.005 -> {:.5}", p.min_deflection_ratio), Kind::Parameter, &fx, &p, "re-applied", |x| x.samples.clone());

        let p = PParams { min_beats: pct(d.min_beats as f64, f).round() as usize, ..PParams::shipped() };
        p_row(&mut t, format!("param: P_MIN_BEATS 8 -> {}", p.min_beats), Kind::Parameter, &fx, &p, "re-applied", |x| x.samples.clone());

        let p = PParams { worst_fraction_min: pct(d.worst_fraction_min, f), ..PParams::shipped() };
        p_row(&mut t, format!("param: the :60 fraction gate 0.90 -> {:.4}", p.worst_fraction_min), Kind::Parameter, &fx, &p, "re-applied", |x| x.samples.clone());

        let p = PParams { amp_mult: pct(d.amp_mult, f), ..PParams::shipped() };
        p_row(&mut t, format!("param: the :59 deflection multiple 2.0 -> {:.3}", p.amp_mult), Kind::Parameter, &fx, &p, "re-applied", |x| x.samples.clone());

        let scale = 1.0 + f;
        p_row(
            &mut t,
            format!("param proxy: every ms window scaled {:+.0}%", -f * 100.0),
            Kind::Parameter,
            &fx,
            &d,
            "proxy: PR_GUARD_MS, P_SEARCH_MS and P_LOWPASS_NULL_HZ move together with the rate",
            move |x| resample(&x.samples, x.fs_hz, x.fs_hz * scale),
        );
    }
    println!(
        "unreachable exactly: P_BEAT_CORRELATION_MIN {}, P_DETECT_SNR {}, P_RESOLVABLE_RATIO {}, \
         P_ABSENT_FRACTION_MAX {}, P_FULL_EVIDENCE_BEATS {} — each is consumed inside p_wave and none \
         of them is a field of PWaveEvidence, so no integration test can re-apply them without \
         widening visibility. Covered only indirectly by the input-side proxies above.",
        morphology::p_wave::P_BEAT_CORRELATION_MIN,
        morphology::p_wave::P_DETECT_SNR,
        morphology::p_wave::P_RESOLVABLE_RATIO,
        morphology::p_wave::P_ABSENT_FRACTION_MAX,
        morphology::p_wave::P_FULL_EVIDENCE_BEATS
    );

    t.finish();
}

// ---------------------------------------------------------------------------------------------
// Metric 7 — ECG atrial-band power ratio. PROPERTY GATE, no threshold.
// Gate: crates/physio-algo/tests/ecg_morphology.rs:134-135 (and :126 states there is no threshold).
// ---------------------------------------------------------------------------------------------

fn atrial_eval(fx: &[Fixture], xf: impl Fn(&Fixture) -> Vec<f64>) -> (usize, bool) {
    let mut measured = 0usize;
    let mut ok = true;
    for f in fx {
        let x = xf(f);
        let peaks = ecg::detect_pan_tompkins(&x, f.fs_hz);
        if let morphology::AtrialBandEvidence::Measured(a) = morphology::atrial_band(&x, f.fs_hz, &peaks) {
            measured += 1;
            ok &= (0.0..=1.0).contains(&a.ratio) && a.median_segment_ms > 0.0 && a.segments > 0;
        }
        // :139 — an empty peak list must always be Indeterminate.
        ok &= matches!(
            morphology::atrial_band(&x, f.fs_hz, &[]),
            morphology::AtrialBandEvidence::Indeterminate(_)
        );
    }
    (measured, ok)
}

#[test]
#[ignore = "negative control over the AAUWSS corpus; run with --release --ignored"]
fn control_ecg_atrial_band() {
    let fx = ecg_fixtures();
    let mut t = Table::new(
        "ECG atrial-band power ratio",
        "ecg_morphology.rs:134-135 ratio in 0..=1 with segments > 0 — the file's own :126 comment \
         states no threshold is asserted, by decision",
    );
    t.property_only = true;

    let row = |t: &mut Table, arm: &str, kind: Kind, xf: &dyn Fn(&Fixture) -> Vec<f64>| {
        let (measured, ok) = atrial_eval(&fx, xf);
        t.add(arm, kind, measured as f64, ok, "subjects returning a ratio");
    };

    row(&mut t, "baseline (unmutated)", Kind::Baseline, &|f: &Fixture| f.samples.clone());
    row(&mut t, "input: gaussian noise", Kind::Null, &|f: &Fixture| gaussian_like(&f.samples, 0xA7B_0000));
    row(&mut t, "input: samples shuffled", Kind::Null, &|f: &Fixture| shuffled(&f.samples, 0xA7B_0000));
    row(&mut t, "input: sawtooth", Kind::Null, &|f: &Fixture| sawtooth_like(&f.samples, 160.0));
    row(&mut t, "input: time-reversed", Kind::Structural, &|f: &Fixture| {
        let mut x = f.samples.clone();
        x.reverse();
        x
    });
    row(&mut t, "input: lead inverted", Kind::Structural, &|f: &Fixture| f.samples.iter().map(|v| -v).collect());
    row(&mut t, "param proxy: every ms window scaled -9%", Kind::Parameter, &|f: &Fixture| {
        resample(&f.samples, f.fs_hz, f.fs_hz * 1.1)
    });

    t.finish();
}

// ---------------------------------------------------------------------------------------------
// Metric 8 — ECG decode sweep (recover layout + sample rate from an unknown byte stream).
// Gate: crates/physio-algo/tests/ecg_sweep.rs:126, :130-131.
// ---------------------------------------------------------------------------------------------

const SWEEP_EPOCH_MS: f64 = 30_000.0;
const SWEEP_WINDOWS: usize = 3;
const SWEEP_PTT_MS: f64 = 220.0;
const SWEEP_PTT_JITTER_MS: f64 = 18.0;
/// ecg_sweep.rs:57-61 — the first truth: 16-bit signed LE, dense, no header, 400 Hz, no hum.
const SWEEP_TRUE_FS_HZ: f64 = 400.0;

fn sweep_layout() -> sweep::Layout {
    sweep::Layout {
        bits: 16,
        signed: true,
        order: sweep::BitOrder::LsbFirst,
        start_bit: 0,
        stride_bits: 16,
    }
}

fn sweep_stream(f: &Fixture, wave: &[f64]) -> (Vec<u8>, Vec<f64>) {
    let filler = shuffled(&f.samples, 0xA5A5);
    let bytes = encode_stream(wave, f.fs_hz, SWEEP_TRUE_FS_HZ, &sweep_layout(), Some(&filler));
    let beats = pulse_train(
        &detector_consensus(&f.samples, f.fs_hz),
        f.fs_hz,
        SWEEP_PTT_MS,
        SWEEP_PTT_JITTER_MS,
        7,
    );
    (bytes, beats)
}

fn sweep_gate(r: &sweep::SweepReport, cfg: &sweep::SweepConfig) -> bool {
    let recovered = match &r.outcome {
        sweep::SweepOutcome::Converged { shape, fs_hz, .. } => {
            *fs_hz == SWEEP_TRUE_FS_HZ
                && (*shape == sweep_layout().shape() || r.alias_shapes.contains(&sweep_layout().shape()))
        }
        _ => false,
    };
    recovered && r.margin >= cfg.min_margin && r.windows_agreed == SWEEP_WINDOWS
}

fn sweep_row(
    t: &mut Table,
    arm: impl Into<String>,
    kind: Kind,
    cases: &[(Vec<u8>, Vec<f64>)],
    cfg: &sweep::SweepConfig,
    note: &str,
) {
    let mut ok = true;
    let mut recovered = 0usize;
    let mut last = String::new();
    for (bytes, beats) in cases {
        let r = sweep::sweep_split(bytes, beats, SWEEP_EPOCH_MS, SWEEP_WINDOWS, cfg);
        let pass = sweep_gate(&r, cfg);
        recovered += usize::from(pass);
        ok &= pass;
        last = format!("{:?}", r.outcome);
    }
    last.truncate(60);
    t.add(arm, kind, recovered as f64, ok, format!("{last}. {note}"));
}

#[test]
#[ignore = "negative control; the sweep is expensive, run with --release --ignored"]
fn control_ecg_decode_sweep() {
    let fx = ecg_fixtures();
    let d = sweep::SweepConfig::default();
    let mut t = Table::new(
        "ECG decode sweep (recover layout + sample rate from unknown bytes)",
        "ecg_sweep.rs:126 the truth is recovered; :130 fs comes back exactly 400 Hz; \
         :131 margin >= min_margin and every one of 3 windows agrees",
    );

    // ecg_sweep.rs:107 RECOVERED_CASES pairs truth 0 with subjects 0, 2 and 3.
    let base: Vec<(Vec<u8>, Vec<f64>)> =
        [0usize, 2].iter().map(|&i| sweep_stream(&fx[i], &fx[i].samples)).collect();
    let one: Vec<(Vec<u8>, Vec<f64>)> = vec![sweep_stream(&fx[0], &fx[0].samples)];

    sweep_row(&mut t, "baseline (subjects 01 and 03)", Kind::Baseline, &base, &d, "");

    // NULL arms — the same layout and rate, carrying something that is not an ECG.
    let f0 = &fx[0];
    for (name, wave) in [
        ("input: gaussian noise encoded at the true layout", gaussian_like(&f0.samples, 11)),
        ("input: samples shuffled, encoded at the true layout", shuffled(&f0.samples, 13)),
        ("input: constant encoded at the true layout", vec![f0.samples[0]; f0.samples.len()]),
    ] {
        let case = vec![sweep_stream(f0, &wave)];
        sweep_row(&mut t, name, Kind::Null, &case, &d, "");
    }

    // STRUCTURAL arms.
    let (bytes, beats) = sweep_stream(f0, &f0.samples);
    let shifted: Vec<u8> = bytes.iter().skip(1).copied().collect();
    sweep_row(&mut t, "bytes: buffer cut one byte late", Kind::Structural, &[(shifted, beats.clone())], &d, "");
    let mut reversed = bytes.clone();
    reversed.reverse();
    sweep_row(&mut t, "bytes: buffer reversed", Kind::Structural, &[(reversed, beats.clone())], &d, "");
    let scrambled: Vec<f64> = beats.iter().map(|b| b * 1.10).collect();
    sweep_row(&mut t, "beats: optical times stretched 10%", Kind::Structural, &[(bytes.clone(), scrambled)], &d, "");
    sweep_row(&mut t, "beats: optical channel removed", Kind::Structural, &[(bytes.clone(), Vec::new())], &d, "");

    // PARAMETER arms — exact, SweepConfig is the whole tunable surface.
    let p = |name: String, cfg: sweep::SweepConfig, t: &mut Table| {
        sweep_row(t, name, Kind::Parameter, &one, &cfg, "exact");
    };
    for f in [0.10, -0.10] {
        p(format!("param: max_roughness 0.60 -> {:.4}", pct(d.max_roughness, f)), sweep::SweepConfig { max_roughness: pct(d.max_roughness, f), ..d.clone() }, &mut t);
        p(format!("param: min_kurtosis 2.00 -> {:.4}", pct(d.min_kurtosis, f)), sweep::SweepConfig { min_kurtosis: pct(d.min_kurtosis, f), ..d.clone() }, &mut t);
        p(format!("param: min_samples 1024 -> {}", pct(d.min_samples as f64, f).round() as usize), sweep::SweepConfig { min_samples: pct(d.min_samples as f64, f).round() as usize, ..d.clone() }, &mut t);
        p(format!("param: class_min_r 0.99 -> {:.4}", pct(d.class_min_r, f)), sweep::SweepConfig { class_min_r: pct(d.class_min_r, f), ..d.clone() }, &mut t);
        p(format!("param: min_b_excess 0.40 -> {:.4}", pct(d.min_b_excess, f)), sweep::SweepConfig { min_b_excess: pct(d.min_b_excess, f), ..d.clone() }, &mut t);
        p(format!("param: min_ppg_match 0.70 -> {:.4}", pct(d.min_ppg_match, f)), sweep::SweepConfig { min_ppg_match: pct(d.min_ppg_match, f), ..d.clone() }, &mut t);
        p(format!("param: hr_prune_tolerance 0.20 -> {:.4}", pct(d.hr_prune_tolerance, f)), sweep::SweepConfig { hr_prune_tolerance: pct(d.hr_prune_tolerance, f), ..d.clone() }, &mut t);
        p(format!("param: min_margin 0.10 -> {:.4}", pct(d.min_margin, f)), sweep::SweepConfig { min_margin: pct(d.min_margin, f), ..d.clone() }, &mut t);
        p(format!("param: rate_agreement_tolerance 0.02 -> {:.4}", pct(d.rate_agreement_tolerance, f)), sweep::SweepConfig { rate_agreement_tolerance: pct(d.rate_agreement_tolerance, f), ..d.clone() }, &mut t);
        p(format!("param: unit_error_tolerance 0.01 -> {:.4}", pct(d.unit_error_tolerance, f)), sweep::SweepConfig { unit_error_tolerance: pct(d.unit_error_tolerance, f), ..d.clone() }, &mut t);
        p(format!("param: top_n 8 -> {}", pct(d.top_n as f64, f).round() as usize), sweep::SweepConfig { top_n: pct(d.top_n as f64, f).round() as usize, ..d.clone() }, &mut t);
    }
    p(
        format!("param: min_margin 0.10 -> {:.5}", pct(d.min_margin, 0.005)),
        sweep::SweepConfig { min_margin: pct(d.min_margin, 0.005), ..d.clone() },
        &mut t,
    );
    p(
        "param: DEFAULT_RATES_HZ all +10% (the truth is no longer searched)".to_string(),
        sweep::SweepConfig { rates_hz: d.rates_hz.iter().map(|r| r * 1.1).collect(), ..d.clone() },
        &mut t,
    );

    t.finish();
}

// ---------------------------------------------------------------------------------------------
// Metric 10 — irregular-rhythm screen (COSEn episode detection + ectopy veto).
// Gate: crates/physio-algo/tests/rr_irregularity_rhythm.rs:124-125, :128, :141, :236.
// ---------------------------------------------------------------------------------------------

/// rr_irregularity_rhythm.rs:124-125.
const RHY_AFIB_AUDITED_MIN: f64 = 0.55;
const RHY_AFIB_MACHINE_MIN: f64 = 0.50;
/// rr_irregularity_rhythm.rs:128.
const RHY_SINUS_MAX: f64 = 0.02;
/// rr_irregularity_rhythm.rs:141.
const RHY_ECTOPY_MAX: f64 = 0.10;
/// rr_irregularity_rhythm.rs:236.
const RHY_FLUTTER_MAX: f64 = 0.15;

struct Stretch {
    class: String,
    rr: Vec<u16>,
}

fn rhythm_fixtures() -> Vec<Stretch> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/rhythm_rr");
    let entries = fs::read_dir(&dir).unwrap_or_else(|e| {
        panic!(
            "rhythm R-R fixture directory unusable: {} ({e}). It is TRACKED, so a clean checkout \
             carries it — restore it from git rather than skipping.",
            dir.display()
        )
    });
    let mut paths: Vec<PathBuf> = entries
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "txt"))
        .collect();
    paths.sort();
    assert!(!paths.is_empty(), "{} holds no .txt fixture", dir.display());
    let mut out = Vec::new();
    for path in &paths {
        let text = fs::read_to_string(path).unwrap();
        let mut class = String::new();
        for line in text.lines() {
            if let Some(rest) = line.strip_prefix('#') {
                let mut it = rest.split_whitespace();
                if let (Some("class"), Some(v)) = (it.next(), it.next()) {
                    class = v.to_string();
                }
                continue;
            }
            let mut it = line.split_whitespace();
            let (Some(_record), Some(_start), Some(_prem)) = (it.next(), it.next(), it.next()) else {
                continue;
            };
            out.push(Stretch { class: class.clone(), rr: it.map(|v| v.parse().unwrap()).collect() });
        }
    }
    assert!(out.len() > 300, "only {} stretches loaded", out.len());
    out
}

/// One beat a second, exactly as the shipped test stamps them, so the duplication and coverage gates
/// inside `assess` stay inert and the rhythm logic is what is under test.
fn stamp(rr: &[u16]) -> Vec<(u32, u16)> {
    rr.iter().enumerate().map(|(i, &v)| (i as u32, v)).collect()
}

fn class_shares(
    all: &[Stretch],
    reported: impl Fn(&[u16]) -> bool,
) -> BTreeMap<String, (usize, usize)> {
    let mut out: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    for s in all {
        let e = out.entry(s.class.clone()).or_default();
        e.1 += 1;
        e.0 += usize::from(reported(&s.rr));
    }
    out
}

fn share_of(sh: &BTreeMap<String, (usize, usize)>, c: &str) -> f64 {
    sh.get(c).map_or(f64::NAN, |&(hit, n)| if n == 0 { f64::NAN } else { hit as f64 / n as f64 })
}

fn rhythm_gate(sh: &BTreeMap<String, (usize, usize)>) -> bool {
    share_of(sh, "AfibAudited") >= RHY_AFIB_AUDITED_MIN
        && share_of(sh, "AfibMachine") >= RHY_AFIB_MACHINE_MIN
        && ["SinusNsr", "SinusAfdb", "SinusMit"].iter().all(|c| share_of(sh, c) <= RHY_SINUS_MAX)
        && ["EctopyLight", "EctopyHeavy", "EctopyBigeminal"]
            .iter()
            .all(|c| share_of(sh, c) <= RHY_ECTOPY_MAX)
        && share_of(sh, "Flutter") <= RHY_FLUTTER_MAX
}

fn rhythm_row(t: &mut Table, arm: impl Into<String>, kind: Kind, sh: &BTreeMap<String, (usize, usize)>, note: &str) {
    t.add(
        arm,
        kind,
        share_of(sh, "AfibMachine"),
        rhythm_gate(sh),
        format!(
            "audited {:.3}, sinusNsr {:.3}, bigeminal {:.3}. {note}",
            share_of(sh, "AfibAudited"),
            share_of(sh, "SinusNsr"),
            share_of(sh, "EctopyBigeminal")
        ),
    );
}

struct RhythmParams {
    window: usize,
    step: usize,
    min_windows: usize,
    cosen_floor: f64,
    residual_floor: f64,
    min_assessed: f64,
    cosen_m: usize,
    cosen_r_ms: f64,
    /// Take COSEn from `cosen_with` on the ranged beats instead of from `assess`, which is the only
    /// way to move COSEN_M / COSEN_R_MS without widening visibility.
    own_cosen: bool,
}

impl RhythmParams {
    fn shipped() -> Self {
        RhythmParams {
            window: screen::WINDOW_BEATS,
            step: screen::STEP_BEATS,
            min_windows: screen::MIN_EPISODE_WINDOWS,
            cosen_floor: screen::COSEN_IRREGULAR_FLOOR,
            residual_floor: screen::EPISODE_RESIDUAL_COSEN_FLOOR,
            min_assessed: screen::MIN_ASSESSED_FRACTION,
            cosen_m: cosen::COSEN_M,
            cosen_r_ms: cosen::COSEN_R_MS,
            own_cosen: false,
        }
    }
}

/// `screen` with its constants exposed, following screen.rs:178-235 step for step: window, assess,
/// the assessed-fraction refusal, the COSEn floor, the run length, and the episode's median residual.
/// A proxy, and its agreement with the shipped function is printed before any arm is read.
fn proxy_reported(rr: &[u16], p: &RhythmParams) -> bool {
    let beats = stamp(rr);
    if beats.len() < p.window + p.min_windows.saturating_sub(1) * p.step {
        return false;
    }
    let (mut cos, mut res) = (Vec::new(), Vec::new());
    let mut s = 0usize;
    while s + p.window <= beats.len() {
        let w = &beats[s..s + p.window];
        match rr::assess(w) {
            rr::IrregularityReading::Assessed(i) => {
                cos.push(if p.own_cosen {
                    cosen::cosen_with(&quality::ranged(w), p.cosen_m, p.cosen_r_ms)
                } else {
                    i.cosen
                });
                res.push(i.ectopy.and_then(|e| e.residual_cosen));
            }
            rr::IrregularityReading::Inconclusive { .. } => {
                cos.push(None);
                res.push(None);
            }
        }
        s += p.step;
    }
    let assessed = cos.iter().filter(|c| c.is_some()).count();
    if (assessed as f64) < p.min_assessed * cos.len() as f64 {
        return false;
    }
    let flags: Vec<bool> = cos.iter().map(|c| c.is_some_and(|v| v >= p.cosen_floor)).collect();
    let mut i = 0usize;
    while i < flags.len() {
        if !flags[i] {
            i += 1;
            continue;
        }
        let start = i;
        while i < flags.len() && flags[i] {
            i += 1;
        }
        if i - start < p.min_windows {
            continue;
        }
        if median_opt(&res[start..i]).is_some_and(|m| m >= p.residual_floor) {
            return true;
        }
    }
    false
}

#[test]
#[ignore = "negative control over the committed rhythm R-R corpus; run with --release --ignored"]
fn control_rr_irregularity_screen() {
    let all = rhythm_fixtures();
    let d = RhythmParams::shipped();
    let mut t = Table::new(
        "Irregular-rhythm screen (COSEn episodes + ectopy veto)",
        "rr_irregularity_rhythm.rs:124 AfibAudited >= 0.55; :125 AfibMachine >= 0.50; \
         :128 each sinus class <= 0.02; :141 each ectopy class <= 0.10; :236 flutter <= 0.15",
    );

    let base = class_shares(&all, |rr| {
        matches!(rr::screen(&stamp(rr)), rr::ScreenState::IrregularEpisodes { .. })
    });
    rhythm_row(&mut t, "baseline (unmutated)", Kind::Baseline, &base, "");

    // NULL arms.
    let flat = class_shares(&all, |rr| {
        let m = median(&rr.iter().map(|&v| f64::from(v)).collect::<Vec<f64>>()).round() as u16;
        matches!(
            rr::screen(&stamp(&vec![m; rr.len()])),
            rr::ScreenState::IrregularEpisodes { .. }
        )
    });
    rhythm_row(&mut t, "output: every interval the stretch's own median", Kind::Null, &flat, "perfectly regular");

    let fixed = class_shares(&all, |rr| {
        matches!(
            rr::screen(&stamp(&vec![800u16; rr.len()])),
            rr::ScreenState::IrregularEpisodes { .. }
        )
    });
    rhythm_row(&mut t, "output: every interval a fixed 800 ms", Kind::Null, &fixed, "");

    // STRUCTURAL arms.
    let shuf = class_shares(&all, |rr| {
        let x: Vec<f64> = rr.iter().map(|&v| f64::from(v)).collect();
        let y: Vec<u16> = shuffled(&x, 0x88_0001).iter().map(|v| v.round() as u16).collect();
        matches!(rr::screen(&stamp(&y)), rr::ScreenState::IrregularEpisodes { .. })
    });
    rhythm_row(&mut t, "input: intervals shuffled (same distribution, no order)", Kind::Structural, &shuf, "");

    let rev = class_shares(&all, |rr| {
        let mut y = rr.to_vec();
        y.reverse();
        matches!(rr::screen(&stamp(&y)), rr::ScreenState::IrregularEpisodes { .. })
    });
    rhythm_row(&mut t, "input: each stretch time-reversed", Kind::Structural, &rev, "");

    let scaled = class_shares(&all, |rr| {
        let y: Vec<u16> = rr.iter().map(|&v| (f64::from(v) * 1.10).round() as u16).collect();
        matches!(rr::screen(&stamp(&y)), rr::ScreenState::IrregularEpisodes { .. })
    });
    rhythm_row(&mut t, "input: every interval x1.10 (a 10% slower rhythm)", Kind::Structural, &scaled, "");

    let mut swapped: BTreeMap<String, (usize, usize)> = base.clone();
    for (a, b) in [("AfibAudited", "SinusAfdb"), ("AfibMachine", "SinusMit")] {
        if let (Some(x), Some(y)) = (base.get(a).copied(), base.get(b).copied()) {
            swapped.insert(a.to_string(), y);
            swapped.insert(b.to_string(), x);
        }
    }
    rhythm_row(&mut t, "oracle: fibrillation and sinus labels swapped", Kind::Structural, &swapped, "the labels, not the screen");

    // PARAMETER arms, on a proxy of `screen` whose agreement with the shipped one is printed first.
    let proxy_base = class_shares(&all, |rr| proxy_reported(rr, &d));
    println!(
        "\nproxy fidelity — shipped vs proxy at the shipped constants: AfibMachine {:.3}/{:.3}, \
         AfibAudited {:.3}/{:.3}, SinusNsr {:.3}/{:.3}, EctopyBigeminal {:.3}/{:.3}",
        share_of(&base, "AfibMachine"),
        share_of(&proxy_base, "AfibMachine"),
        share_of(&base, "AfibAudited"),
        share_of(&proxy_base, "AfibAudited"),
        share_of(&base, "SinusNsr"),
        share_of(&proxy_base, "SinusNsr"),
        share_of(&base, "EctopyBigeminal"),
        share_of(&proxy_base, "EctopyBigeminal")
    );
    rhythm_row(&mut t, "proxy at the shipped constants (fidelity reference)", Kind::Parameter, &proxy_base, "proxy");

    let param = |name: String, p: RhythmParams, t: &mut Table| {
        let sh = class_shares(&all, |rr| proxy_reported(rr, &p));
        rhythm_row(t, name, Kind::Parameter, &sh, "proxy");
    };
    for f in [0.10, -0.10] {
        param(format!("param: WINDOW_BEATS 32 -> {}", pct(d.window as f64, f).round() as usize), RhythmParams { window: pct(d.window as f64, f).round() as usize, ..RhythmParams::shipped() }, &mut t);
        param(format!("param: STEP_BEATS 8 -> {}", pct(d.step as f64, f).round() as usize), RhythmParams { step: pct(d.step as f64, f).round() as usize, ..RhythmParams::shipped() }, &mut t);
        param(format!("param: MIN_EPISODE_WINDOWS 24 -> {}", pct(d.min_windows as f64, f).round() as usize), RhythmParams { min_windows: pct(d.min_windows as f64, f).round() as usize, ..RhythmParams::shipped() }, &mut t);
        param(format!("param: COSEN_IRREGULAR_FLOOR -1.28 -> {:.4}", pct(d.cosen_floor, f)), RhythmParams { cosen_floor: pct(d.cosen_floor, f), ..RhythmParams::shipped() }, &mut t);
        param(format!("param: EPISODE_RESIDUAL_COSEN_FLOOR -1.10 -> {:.4}", pct(d.residual_floor, f)), RhythmParams { residual_floor: pct(d.residual_floor, f), ..RhythmParams::shipped() }, &mut t);
        param(format!("param: MIN_ASSESSED_FRACTION 0.50 -> {:.4}", pct(d.min_assessed, f)), RhythmParams { min_assessed: pct(d.min_assessed, f), ..RhythmParams::shipped() }, &mut t);
        param(format!("param: COSEN_R_MS 30 -> {:.3}", pct(d.cosen_r_ms, f)), RhythmParams { cosen_r_ms: pct(d.cosen_r_ms, f), own_cosen: true, ..RhythmParams::shipped() }, &mut t);
    }
    param(
        format!("param: COSEN_IRREGULAR_FLOOR -1.28 -> {:.5}", pct(d.cosen_floor, 0.005)),
        RhythmParams { cosen_floor: pct(d.cosen_floor, 0.005), ..RhythmParams::shipped() },
        &mut t,
    );
    param(
        "param: COSEN_M 1 -> 2".to_string(),
        RhythmParams { cosen_m: 2, own_cosen: true, ..RhythmParams::shipped() },
        &mut t,
    );
    println!(
        "structurally inert on this fixture: MAX_BEAT_GAP_S {} (the stretches are stamped one beat a \
         second, so no window ever holds a gap) and MIN_SCREEN_BEATS {} (derived from WINDOW_BEATS, \
         STEP_BEATS and MIN_EPISODE_WINDOWS, already moved above). Unreachable without widening \
         visibility: COSEN_CONFIDENT {}, RESIDUAL_COSEN_CONFIDENT {}, COSEN_MODERATE {} (they band a \
         reported episode and never decide one), and the ectopy / poincare / indices constants.",
        screen::MAX_BEAT_GAP_S,
        screen::MIN_SCREEN_BEATS,
        screen::COSEN_CONFIDENT,
        screen::RESIDUAL_COSEN_CONFIDENT,
        screen::COSEN_MODERATE
    );

    t.finish();
}

// ---------------------------------------------------------------------------------------------
// Metric 11 — R-R stream quality (coverage, duplication, 1000/1024 rescale detection).
// There is NO integration gate for this metric: quality.rs carries threshold-edge unit tests only
// and no golden verdict against a real stream. The gate re-created here is the conjunction of the
// module's own published limits, so the arms measure headroom rather than a shipped claim.
// ---------------------------------------------------------------------------------------------

struct QualityLimits {
    coverage: f64,
    duplicate: f64,
    range_rejected: f64,
    rescaled: f64,
    min_beats: usize,
}

impl QualityLimits {
    fn shipped() -> Self {
        QualityLimits {
            coverage: quality::MAX_COVERAGE,
            duplicate: quality::MAX_DUPLICATE_FRACTION,
            range_rejected: quality::MAX_RANGE_REJECTED_FRACTION,
            rescaled: quality::MAX_RESCALED_FRACTION,
            min_beats: quality::MIN_QUALITY_BEATS,
        }
    }
}

fn quality_clean_series() -> Vec<(u32, u16)> {
    (0..500u32).map(|i| (i, 780 + u16::try_from(i % 41).unwrap())).collect()
}

fn quality_eval(beats: &[(u32, u16)], l: &QualityLimits) -> (f64, bool, String) {
    let q = quality::measure(beats);
    let clauses = [
        q.coverage <= l.coverage,
        q.duplicate_fraction <= l.duplicate,
        q.range_rejected_fraction <= l.range_rejected,
        q.rescaled_fraction <= l.rescaled,
        beats.len() >= l.min_beats,
    ];
    let satisfied = clauses.iter().filter(|c| **c).count();
    let note = format!(
        "cov {:.3}/{:.2} dup {:.3}/{:.2} rng {:.3}/{:.2} resc {:.3}/{:.2} n {}",
        q.coverage,
        l.coverage,
        q.duplicate_fraction,
        l.duplicate,
        q.range_rejected_fraction,
        l.range_rejected,
        q.rescaled_fraction,
        l.rescaled,
        beats.len()
    );
    (satisfied as f64, satisfied == clauses.len(), note)
}

fn quality_row(
    t: &mut Table,
    arm: impl Into<String>,
    kind: Kind,
    beats: &[(u32, u16)],
    l: &QualityLimits,
    extra: &str,
) {
    let (value, ok, note) = quality_eval(beats, l);
    t.add(arm, kind, value, ok, format!("{note}. {extra}"));
}

/// Second copies of every beat, each `round(v * ratio)` and `lag` seconds after its original.
fn rescaled_copies(beats: &[(u32, u16)], ratio: f64, lag: u32) -> Vec<(u32, u16)> {
    let mut out = beats.to_vec();
    for &(ts, v) in beats {
        out.push((ts + lag, (f64::from(v) * ratio).round() as u16));
    }
    out.sort_by_key(|&(ts, v)| (ts, v));
    out
}

#[test]
#[ignore = "negative control; run with --release --ignored"]
fn control_rr_stream_quality() {
    let clean = quality_clean_series();
    let d = QualityLimits::shipped();
    let mut t = Table::new(
        "R-R stream quality (coverage, duplication, 1000/1024 rescale)",
        "NO shipped integration gate — quality.rs carries threshold-edge unit tests only. \
         Reconstructed here as: coverage <= 1.15, duplicate <= 0.02, range-rejected <= 0.20, \
         rescaled <= 0.20, beats >= 8 (quality.rs:15-29)",
    );

    quality_row(&mut t, "baseline (clean 500-beat series)", Kind::Baseline, &clean, &d, "");

    // NULL arms.
    let mut doubled = clean.clone();
    doubled.extend(clean.iter().copied());
    doubled.sort_by_key(|&(ts, v)| (ts, v));
    quality_row(&mut t, "input: every beat stored twice", Kind::Null, &doubled, &d, "");

    let rescaled = rescaled_copies(&clean, quality::RESCALE_RATIO, 0);
    quality_row(&mut t, "input: a 1000/1024 second copy of every beat", Kind::Null, &rescaled, &d, "");

    // STRUCTURAL arms.
    let mut rev = clean.clone();
    let stamps: Vec<u32> = clean.iter().map(|&(ts, _)| ts).collect();
    rev.reverse();
    let rev: Vec<(u32, u16)> =
        stamps.iter().zip(rev.iter()).map(|(&ts, &(_, v))| (ts, v)).collect();
    quality_row(&mut t, "input: interval order reversed", Kind::Structural, &rev, &d, "");

    let squeezed: Vec<(u32, u16)> = clean.iter().map(|&(ts, v)| (ts / 4, v)).collect();
    quality_row(&mut t, "input: timeline compressed 4x", Kind::Structural, &squeezed, &d, "");

    let out_of_range: Vec<(u32, u16)> =
        clean.iter().enumerate().map(|(i, &(ts, v))| (ts, if i % 2 == 0 { 50 } else { v })).collect();
    quality_row(&mut t, "input: half the beats below any physiological floor", Kind::Structural, &out_of_range, &d, "");

    quality_row(&mut t, "input: only 4 beats", Kind::Structural, &clean[..4], &d, "");

    // PARAMETER arms.
    let param = |name: String, l: QualityLimits, beats: &[(u32, u16)], t: &mut Table| {
        quality_row(t, name, Kind::Parameter, beats, &l, "re-applied");
    };
    for f in [0.10, -0.10] {
        param(format!("param: MAX_COVERAGE 1.15 -> {:.4}", pct(d.coverage, f)), QualityLimits { coverage: pct(d.coverage, f), ..QualityLimits::shipped() }, &clean, &mut t);
        param(format!("param: MAX_DUPLICATE_FRACTION 0.02 -> {:.4}", pct(d.duplicate, f)), QualityLimits { duplicate: pct(d.duplicate, f), ..QualityLimits::shipped() }, &clean, &mut t);
        param(format!("param: MAX_RANGE_REJECTED_FRACTION 0.20 -> {:.4}", pct(d.range_rejected, f)), QualityLimits { range_rejected: pct(d.range_rejected, f), ..QualityLimits::shipped() }, &clean, &mut t);
        param(format!("param: MAX_RESCALED_FRACTION 0.20 -> {:.4}", pct(d.rescaled, f)), QualityLimits { rescaled: pct(d.rescaled, f), ..QualityLimits::shipped() }, &clean, &mut t);
        param(format!("param: MIN_QUALITY_BEATS 8 -> {}", pct(d.min_beats as f64, f).round() as usize), QualityLimits { min_beats: pct(d.min_beats as f64, f).round() as usize, ..QualityLimits::shipped() }, &clean, &mut t);
    }
    param(
        format!("param: MAX_DUPLICATE_FRACTION 0.02 -> {:.5}", pct(d.duplicate, 0.005)),
        QualityLimits { duplicate: pct(d.duplicate, 0.005), ..QualityLimits::shipped() },
        &clean,
        &mut t,
    );
    // RESCALE_RATIO and RESCALE_LAG_S live inside the detector, so they are moved on the copy side:
    // if a ratio 10% off is no longer seen, the detector is narrower than the failure it screens for.
    for f in [0.10, -0.10] {
        let r = pct(quality::RESCALE_RATIO, f);
        let beats = rescaled_copies(&clean, r, 0);
        quality_row(&mut t, format!("param proxy: copies written at ratio {r:.5}"), Kind::Parameter, &beats, &d, "proxy: the detector's own ratio cannot be passed in");
    }
    for lag in [0u32, 2] {
        let beats = rescaled_copies(&clean, quality::RESCALE_RATIO, lag);
        quality_row(&mut t, format!("param proxy: copies written {lag} s after the original"), Kind::Parameter, &beats, &d, "proxy: RESCALE_LAG_S is 1");
    }
    println!(
        "measured for the record: rescaled_copy_fraction on the clean series is {:.4}, on the \
         1000/1024 copy series {:.4}.",
        quality::rescaled_copy_fraction(&clean),
        quality::rescaled_copy_fraction(&rescaled_copies(&clean, quality::RESCALE_RATIO, 0))
    );

    t.finish();
}

// ---------------------------------------------------------------------------------------------
// What this file could NOT control, and why.
// ---------------------------------------------------------------------------------------------

#[test]
#[ignore = "prints the family's coverage gaps; run with --ignored"]
fn control_ppg_ecg_coverage_notes() {
    println!("\n================ ppg_ecg: what is NOT controlled here ================");

    let gt = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/ecg_ground_truth.rs");
    match fs::read_to_string(&gt) {
        Ok(text) => println!(
            "ECG sweep ground truth: {} has {} lines and {} occurrences of `assert`. A metric with \
             no assertion has no gate, so there is nothing for a negative control to falsify — every \
             arm would pass by construction. Recorded, not tested.",
            gt.display(),
            text.lines().count(),
            text.matches("assert").count()
        ),
        Err(e) => println!("could not read {} ({e}) — re-derive the assert count by hand", gt.display()),
    }

    println!(
        "\nECG terminal render scale (25 mm/s, 10 mm/mV, braille dot pitch): the gate is \
         crates/whoopctl/src/ecg_oracle/tests.rs:141-171, a unit test inside another crate. \
         physio-algo does not depend on whoopctl and this task owns exactly one file, so it is out \
         of reach from here and wants its own control under crates/whoopctl/tests/. Nothing about \
         it is presumed working."
    );

    println!(
        "\nAtrial-band ratio: its gate is a range check with no threshold (ecg_morphology.rs:126 \
         says so explicitly), so a null passes it. That is a documented decision, not a fake gate, \
         and the control records it rather than asserting against it."
    );

    println!(
        "\nR-R stream quality: no golden integration gate exists at all. The limits in the table \
         above were reconstructed from quality.rs:15-29 so the arms had something to be measured \
         against; they are not a shipped claim."
    );

    println!(
        "\nEvery number this family produces is a wellness estimate. Nothing here is medical, \
         diagnostic, or a basis for changing a shipped constant."
    );
}
