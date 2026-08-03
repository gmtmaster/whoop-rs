//! PPG-derived HR on real optical data, two ways. The v26 half pins every estimate off the recorded
//! bursts in the shared protocol fixture. The `aauwss` half scores a wrist PPG epoch against the R
//! peaks of the ECG recorded at the same instant, which is the only independent HR reference in tree.

use physio_algo::ppg::{estimate, Estimate, Sample, SAMPLE_RATE_HZ};
use serde_json::Value;
use whoop_protocol::bytes::from_hex;
use whoop_protocol::family::Family;
use whoop_protocol::framing;
use whoop_protocol::records::{decode, Record};

/// Every estimate `physio_algo::ppg::estimate` returns over the concatenated `ppg_frames`, as
/// `(ts, bpm, conf)`. The fixture JSON carries only the first, which left a bias on the rest unseen.
const V26_GOLDEN: [(i64, i32, f64); 10] = [
    (1_783_955_687, 78, 0.733),
    (1_783_955_688, 78, 0.757),
    (1_783_955_689, 78, 0.764),
    (1_783_955_690, 78, 0.802),
    (1_783_955_691, 78, 0.797),
    (1_783_955_692, 78, 0.789),
    (1_783_955_693, 78, 0.603),
    (1_783_955_694, 39, 0.371),
    (1_783_955_725, 44, 0.300),
    (1_783_955_726, 44, 0.309),
];

fn v26_estimates() -> Vec<Estimate> {
    let oracle: Value =
        serde_json::from_str(include_str!("../../whoop-protocol/tests/fixtures/real_frames.json")).unwrap();
    let frames = oracle["ppg_frames"].as_array().unwrap();
    let mut samples = Vec::new();
    for f in frames {
        let wire = from_hex(f["hex"].as_str().unwrap()).unwrap();
        let frame = framing::decode(Family::Gen5, &wire).unwrap();
        let p = match decode(&frame) {
            Some(Record::Ppg(p)) => p,
            other => panic!("expected a Ppg record, got {other:?}"),
        };
        for v in p.samples {
            samples.push(Sample { ts: i64::from(p.unix), value: i32::from(v) });
        }
    }
    estimate(&samples)
}

/// Every v26 estimate is pinned, not only the first: a bias applied from the second second onward
/// used to leave the whole series inside a 60-100 bpm band check and stay green.
#[test]
fn real_v26_ppg_hr_matches_golden_at_every_second() {
    let est = v26_estimates();
    let got: Vec<(i64, i32, f64)> = est.iter().map(|e| (e.ts, e.bpm, e.conf)).collect();
    assert_eq!(got.len(), V26_GOLDEN.len(), "estimate count: {got:?}");
    for (i, (want, have)) in V26_GOLDEN.iter().zip(&got).enumerate() {
        assert_eq!((want.0, want.1), (have.0, have.1), "estimate {i}: ts/bpm");
        assert!((want.2 - have.2).abs() < 1e-9, "estimate {i}: conf {} vs {}", want.2, have.2);
    }
    // The fixture's own first-estimate anchor still has to agree with the series.
    let oracle: Value =
        serde_json::from_str(include_str!("../../whoop-protocol/tests/fixtures/real_frames.json")).unwrap();
    let g = &oracle["ppg_hr"]["first"];
    assert_eq!((got[0].0, got[0].1), (g["ts"].as_i64().unwrap(), g["bpm"].as_i64().unwrap() as i32));
}

/// The null arm for the v26 series: a uniform bias, or any constant, now moves it off the golden.
#[test]
fn a_biased_or_constant_v26_series_no_longer_matches() {
    let est = v26_estimates();
    let matches = |s: &[Estimate]| {
        s.len() == V26_GOLDEN.len() && s.iter().zip(&V26_GOLDEN).all(|(e, g)| e.bpm == g.1 && e.ts == g.0)
    };
    assert!(matches(&est), "the shipped series must match");
    for bias in [-10i32, -1, 1, 10] {
        let shifted: Vec<Estimate> = est.iter().map(|e| Estimate { bpm: e.bpm + bias, ..*e }).collect();
        assert!(!matches(&shifted), "a uniform {bias} bpm bias still matched");
    }
    // A bias from the second estimate on, which the old first-plus-band gate could not see.
    let tail: Vec<Estimate> = est
        .iter()
        .enumerate()
        .map(|(i, e)| Estimate { bpm: if i == 0 { e.bpm } else { e.bpm + 10 }, ..*e })
        .collect();
    assert!(!matches(&tail), "a bias sparing only the first estimate still matched");
    for bpm in [39i32, 44, 78] {
        let flat: Vec<Estimate> = est.iter().map(|e| Estimate { bpm, ..*e }).collect();
        assert!(!matches(&flat), "the constant {bpm} still matched");
    }
}

// ─────────────────────────── the ECG-referenced half ───────────────────────────

/// One fixture epoch: its `# fs_hz`, its subject/row identity, and the samples. Panics when absent -
/// a skip on a missing fixture reports a pass.
fn load_epoch(path: &str) -> (f64, String, Vec<f64>) {
    let raw = std::fs::read_to_string(path).unwrap_or_else(|e| {
        panic!("{path}: {e}. Regenerate it with the converter named in tests/fixtures/README.md")
    });
    let (mut fs, mut id, mut v) = (0.0, String::new(), Vec::new());
    for line in raw.lines() {
        if let Some(r) = line.strip_prefix("# fs_hz ") {
            fs = r.trim().parse().unwrap();
        } else if let Some(r) = line.strip_prefix("# subject ").or_else(|| line.strip_prefix("# source_row ")) {
            id.push_str(r.trim());
            id.push('/');
        } else if !line.starts_with('#') && !line.trim().is_empty() {
            v.push(line.trim().parse::<f64>().unwrap());
        }
    }
    assert!(fs > 0.0 && !v.is_empty(), "{path} carries no samples");
    (fs, id, v)
}

/// R-peak sample indices by first-difference magnitude over `mult` x the 99th percentile, with a
/// 250 ms refractory. Deliberately not our own QRS detector: this is the reference, not a result.
fn r_peaks(ecg: &[f64], fs: f64, mult: f64) -> Vec<usize> {
    let d: Vec<f64> = (1..ecg.len()).map(|i| (ecg[i] - ecg[i - 1]).abs()).collect();
    let mut sorted = d.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let thr = sorted[(sorted.len() as f64 * 0.99) as usize] * mult;
    let refractory = (0.25 * fs) as usize;
    let (mut peaks, mut i) = (Vec::new(), 0usize);
    while i < d.len() {
        if d[i] >= thr {
            let hi = (i + refractory).min(d.len());
            let best = (i..hi).max_by(|&a, &b| d[a].partial_cmp(&d[b]).unwrap()).unwrap();
            peaks.push(best);
            i = best + refractory;
        } else {
            i += 1;
        }
    }
    peaks
}

/// R-R intervals (ms) of the reference beat train.
fn reference_rr(ecg: &[f64], fs: f64, mult: f64) -> Vec<f64> {
    r_peaks(ecg, fs, mult).windows(2).map(|w| (w[1] - w[0]) as f64 / fs * 1000.0).collect()
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// Linear resample of a fixture epoch onto the 24 Hz grid `estimate` reads, stamped one whole second
/// per 24 samples so the whole epoch is a single consecutive-second run.
fn to_24hz(x: &[f64], fs: f64) -> Vec<Sample> {
    let n = (x.len() as f64 / fs * SAMPLE_RATE_HZ as f64).round() as usize;
    (0..n)
        .map(|k| {
            let t = k as f64 / SAMPLE_RATE_HZ as f64 * fs;
            let i = (t.floor() as usize).min(x.len() - 1);
            let next = x[(i + 1).min(x.len() - 1)];
            let value = (x[i] + (t - t.floor()) * (next - x[i])).round() as i32;
            Sample { ts: (k / SAMPLE_RATE_HZ) as i64, value }
        })
        .collect()
}

/// The reference itself, before it is used on anything: the R-peak count is unchanged across a 2x
/// sweep of the detector's one threshold, and every interval is a single beat, so it is not a fit.
#[test]
fn the_ecg_reference_beat_train_is_stable() {
    let (fs, _, ecg) = load_epoch("tests/fixtures/aauwss_ecg/subject_01.txt");
    let rr = reference_rr(&ecg, fs, 0.40);
    assert_eq!(rr.len(), 28, "R-R count");
    for mult in [0.30f64, 0.35, 0.45, 0.50, 0.60] {
        assert_eq!(reference_rr(&ecg, fs, mult).len(), rr.len(), "threshold x{mult} changed the count");
    }
    assert!(rr.iter().all(|&v| (600.0..=1500.0).contains(&v)), "a dropped or doubled beat: {rr:?}");
    assert!((median(rr) - 1060.0).abs() < 1e-9, "median R-R");
}

/// PPG-derived HR against the R peaks of the ECG recorded at the same instant: median 57 bpm against
/// a reference of 56.60 bpm, a +0.40 bpm difference over one 30 s epoch on one subject.
#[test]
fn ppg_hr_agrees_with_the_simultaneous_ecg_reference() {
    let (fs_ecg, ecg_id, ecg) = load_epoch("tests/fixtures/aauwss_ecg/subject_01.txt");
    let (fs_ppg, ppg_id, ppg) = load_epoch("tests/fixtures/aauwss_ppg/subject_01.txt");
    assert_eq!(ecg_id, ppg_id, "the two epochs must be the same subject and row to be simultaneous");

    let reference_bpm = 60_000.0 / median(reference_rr(&ecg, fs_ecg, 0.40));
    assert!((reference_bpm - 56.604).abs() < 0.01, "reference {reference_bpm}");

    let est = estimate(&to_24hz(&ppg, fs_ppg));
    assert_eq!(est.len(), 25, "estimate count over the 30 s epoch");
    let derived = median(est.iter().map(|e| f64::from(e.bpm)).collect());
    assert!((derived - reference_bpm).abs() <= 3.0, "PPG {derived} vs ECG {reference_bpm}");

    // The null arm: a systematic 10 bpm bias, and any constant, now breaks the agreement.
    for bias in [-10.0f64, 10.0] {
        assert!((derived + bias - reference_bpm).abs() > 3.0, "a {bias} bpm bias still agreed");
    }
    // No constant satisfies this and the 78 bpm v26 series above at the same time.
    assert!((derived - 78.0).abs() > 3.0, "the v26 level would also satisfy this reference");
}

/// Recorded, not a target: the high-confidence subset reads ~3 bpm ABOVE the reference on this epoch,
/// where the full set reads +0.40. One subject over 30 s cannot say whether that generalises.
#[test]
fn the_high_confidence_subset_sits_above_the_reference_here() {
    let (fs_ppg, _, ppg) = load_epoch("tests/fixtures/aauwss_ppg/subject_01.txt");
    let est = estimate(&to_24hz(&ppg, fs_ppg));
    let confident: Vec<f64> = est.iter().filter(|e| e.conf >= 0.7).map(|e| f64::from(e.bpm)).collect();
    assert_eq!(confident.len(), 14, "confident count");
    assert!((median(confident) - 60.0).abs() < 1e-9, "confident median");
}
