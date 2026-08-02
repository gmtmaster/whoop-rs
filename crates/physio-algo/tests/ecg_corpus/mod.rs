//! Shared corpus loader and matched negatives for the ECG integration tests.
//!
//! Compiled into every test binary that declares it, and no one binary uses all of it, so the unused
//! warning here would be about the OTHER tests rather than about this file.
//!
//! Fixtures are plain text, one 30 s epoch per subject, converted from the AAUWSS aligned-sleep
//! pickles by `tools/aauwss_ecg_to_fixture.py` and `tools/aauwss_ppg_to_fixture.py` (middle epoch, no
//! cherry-picking). Every negative is matched to the positive it came from — same length, same mean,
//! same standard deviation — so nothing here can separate on amplitude or duration alone.

#![allow(dead_code)]

use std::fs;
use std::path::{Path, PathBuf};

/// A subject's converted epoch, already scaled out of its stored integer counts.
pub struct Fixture {
    pub subject: String,
    pub fs_hz: f64,
    pub samples: Vec<f64>,
}

/// Every ECG subject, ascending by subject id.
pub fn ecg_fixtures() -> Vec<Fixture> {
    load_dir("tests/fixtures/aauwss_ecg")
}

/// Every PPG subject. Only subject 01 has a published PPG pickle, so this is one entry by
/// availability, not by selection.
pub fn ppg_fixtures() -> Vec<Fixture> {
    load_dir("tests/fixtures/aauwss_ppg")
}

fn load_dir(rel: &str) -> Vec<Fixture> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(rel);
    let entries = fs::read_dir(&dir).unwrap_or_else(|e| panic!("{}", unusable(&dir, &e.to_string())));
    let mut paths: Vec<PathBuf> = entries
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "txt"))
        .collect();
    paths.sort();
    assert!(!paths.is_empty(), "{}", unusable(&dir, "it holds no .txt fixture"));
    paths.iter().map(read_fixture).collect()
}

/// The failure message for an absent corpus: these fixtures are tracked, so a miss means an incomplete
/// checkout, and the tests that read them have no skip path.
fn unusable(dir: &Path, why: &str) -> String {
    format!(
        "AAUWSS fixture directory unusable: {} ({why}).\n\
         It is TRACKED, so a clean checkout carries it - restore it from git rather than skipping.\n\
         To rebuild from source: fetch AAUWSS v1.1 from https://doi.org/10.5281/zenodo.16919071 \
         (open access, CC-BY-4.0), unpack `aligned_sleep_data_set` under \
         whoop-data/datasets/AAUWSS/extracted/, then run tools/aauwss_ecg_to_fixture.py and \
         tools/aauwss_ppg_to_fixture.py.",
        dir.display()
    )
}

fn read_fixture(path: &PathBuf) -> Fixture {
    let text = fs::read_to_string(path).unwrap();
    let (mut fs_hz, mut scale, mut subject, mut samples) = (0.0f64, 1.0f64, String::new(), Vec::new());
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix('#') {
            let mut it = rest.split_whitespace();
            match (it.next(), it.next()) {
                (Some("fs_hz"), Some(v)) => fs_hz = v.parse().unwrap(),
                (Some("scale"), Some(v)) => scale = v.parse().unwrap(),
                (Some("subject"), Some(v)) => subject = v.to_string(),
                _ => {}
            }
        } else if !line.trim().is_empty() {
            samples.push(line.trim().parse::<f64>().unwrap() * scale);
        }
    }
    assert!(fs_hz > 0.0 && !samples.is_empty(), "unusable fixture {}", path.display());
    Fixture { subject, fs_hz, samples }
}

/// Splitmix64 — negatives must be reproducible, so nothing here draws from the OS.
pub struct Rng(pub u64);

impl Rng {
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    pub fn gaussian(&mut self) -> f64 {
        let u1 = self.unit().max(1e-12);
        let u2 = self.unit();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

pub fn mean(x: &[f64]) -> f64 {
    x.iter().sum::<f64>() / x.len() as f64
}

pub fn sd(x: &[f64]) -> f64 {
    let m = mean(x);
    (x.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / x.len() as f64).sqrt()
}

pub fn min_of(x: &[f64]) -> f64 {
    x.iter().copied().fold(f64::INFINITY, f64::min)
}

pub fn max_of(x: &[f64]) -> f64 {
    x.iter().copied().fold(f64::NEG_INFINITY, f64::max)
}

pub fn median_of(x: &[f64]) -> f64 {
    let mut s = x.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if s.is_empty() {
        f64::NAN
    } else {
        s[s.len() / 2]
    }
}

/// Gaussian noise with the source's length, mean and standard deviation.
pub fn gaussian_like(src: &[f64], seed: u64) -> Vec<f64> {
    let (m, s) = (mean(src), sd(src));
    let mut rng = Rng(seed);
    (0..src.len()).map(|_| m + s * rng.gaussian()).collect()
}

/// Sawtooth at `period` samples, scaled to the source's spread. Sharp, broadband and perfectly
/// periodic — the negative built to fool anything that only looks for repeating sharp events.
pub fn sawtooth_like(src: &[f64], period: f64) -> Vec<f64> {
    let (m, s) = (mean(src), sd(src));
    let period = period.max(2.0);
    let gain = s * 12f64.sqrt(); // a unit sawtooth has SD 1/sqrt(12)
    (0..src.len()).map(|i| m + gain * ((i as f64 / period).fract() - 0.5)).collect()
}

/// The same samples in a deterministic Fisher-Yates permutation: identical amplitude distribution,
/// every trace of beat morphology destroyed.
pub fn shuffled(src: &[f64], seed: u64) -> Vec<f64> {
    let mut out = src.to_vec();
    let mut rng = Rng(seed);
    for i in (1..out.len()).rev() {
        let j = (rng.next_u64() % (i as u64 + 1)) as usize;
        out.swap(i, j);
    }
    out
}

/// Rescale to another signal's length, mean and standard deviation, so a comparison cannot turn on
/// amplitude. Used to match the PPG negative to the ECG subject it is scored against.
pub fn matched_to(src: &[f64], target: &[f64]) -> Vec<f64> {
    let (ms, ss) = (mean(src), sd(src));
    let (mt, st) = (mean(target), sd(target));
    let gain = if ss > 0.0 { st / ss } else { 0.0 };
    let n = target.len().min(src.len());
    src[..n].iter().map(|v| mt + (v - ms) * gain).collect()
}

/// Linear resample from `fs_in` to `fs_out`, anti-alias filtered first when decimating. Linear
/// interpolation is its own mild low-pass; the moving average ahead of it is what stops real content
/// above the new Nyquist from folding back in and being mistaken for a result.
pub fn resample(x: &[f64], fs_in: f64, fs_out: f64) -> Vec<f64> {
    if x.len() < 2 || fs_in <= 0.0 || fs_out <= 0.0 {
        return Vec::new();
    }
    let src: Vec<f64> = if fs_out < fs_in {
        let len = ((fs_in / fs_out).round() as usize).max(2);
        physio_algo::signal::moving_average_centred(x, len | 1)
    } else {
        x.to_vec()
    };
    let ratio = fs_in / fs_out;
    let n_out = ((src.len() as f64 - 1.0) / ratio).floor() as usize;
    (0..n_out)
        .map(|i| {
            let t = i as f64 * ratio;
            let k = t.floor() as usize;
            let frac = t - k as f64;
            src[k] * (1.0 - frac) + src[(k + 1).min(src.len() - 1)] * frac
        })
        .collect()
}

/// Add mains at `mains_hz` (with its 2nd and 3rd harmonic at 1/2 and 1/3 amplitude) at
/// `amp_ratio` times the signal's own standard deviation.
pub fn add_hum(x: &[f64], fs_hz: f64, mains_hz: f64, amp_ratio: f64) -> Vec<f64> {
    let amp = amp_ratio * sd(x);
    x.iter()
        .enumerate()
        .map(|(i, v)| {
            let t = i as f64 / fs_hz;
            let mut out = *v;
            for (k, scale) in [(1.0, 1.0), (2.0, 0.5), (3.0, 1.0 / 3.0)] {
                out += amp * scale * (2.0 * std::f64::consts::PI * k * mains_hz * t).sin();
            }
            out
        })
        .collect()
}
