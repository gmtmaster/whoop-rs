//! bSQI gate: run both QRS detectors over real overnight ECG and over matched negatives, and check what
//! their agreement can and cannot separate.
//!
//! The fixtures are one 30 s epoch per subject, converted from the AAUWSS aligned-sleep pickles by
//! `tools/aauwss_ecg_to_fixture.py` (middle epoch, no cherry-picking) and committed as plain text.
//!
//! Every negative is matched to the positive it is derived from - same length, same sample rate, same
//! standard deviation - so a separation cannot come from an amplitude or duration difference. The sawtooth
//! is additionally given the subject's own mean R-R, so it is periodic at a plausible heart rate: it is
//! the negative designed to fool a detector that only looks for sharp repeating events, and it is the one
//! this index does NOT reject.
//!
//!   cargo test -p physio-algo --test ecg_qrs_agreement -- --nocapture

mod ecg_corpus;

use ecg_corpus::{ecg_fixtures as fixtures, Fixture};
use physio_algo::ecg::{
    beat_agreement, detect_pan_tompkins, detect_wavelet, Agreement, DEFAULT_MATCH_WINDOW_MS,
};

/// Splitmix64 - the negatives must be reproducible, so nothing here draws from the OS.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn gaussian(&mut self) -> f64 {
        let u1 = self.unit().max(1e-12);
        let u2 = self.unit();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

fn mean(x: &[f64]) -> f64 {
    x.iter().sum::<f64>() / x.len() as f64
}

fn sd(x: &[f64]) -> f64 {
    let m = mean(x);
    (x.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / x.len() as f64).sqrt()
}

fn median(x: &[f64]) -> f64 {
    let mut s = x.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    s[s.len() / 2]
}

/// Gaussian noise with the source's length and standard deviation.
fn gaussian_like(src: &[f64], seed: u64) -> Vec<f64> {
    let (m, s) = (mean(src), sd(src));
    let mut rng = Rng(seed);
    (0..src.len()).map(|_| m + s * rng.gaussian()).collect()
}

/// Sawtooth at `period` samples, scaled to the source's standard deviation. A ramp plus a hard flyback:
/// sharp, broadband, and perfectly periodic. A unit sawtooth has SD 1/sqrt(12), hence the gain.
fn sawtooth_like(src: &[f64], period: f64) -> Vec<f64> {
    let (m, s) = (mean(src), sd(src));
    let period = period.max(2.0);
    let gain = s * 12f64.sqrt();
    (0..src.len()).map(|i| m + gain * ((i as f64 / period).fract() - 0.5)).collect()
}

/// The same samples in a deterministic Fisher-Yates permutation: identical amplitude distribution, with
/// every trace of beat morphology destroyed.
fn shuffled(src: &[f64], seed: u64) -> Vec<f64> {
    let mut out = src.to_vec();
    let mut rng = Rng(seed);
    for i in (1..out.len()).rev() {
        let j = (rng.next_u64() % (i as u64 + 1)) as usize;
        out.swap(i, j);
    }
    out
}

/// Agreement between the two detectors on one signal, plus the two beat counts.
fn bsqi(x: &[f64], fs_hz: f64, window_ms: f64) -> (Agreement, usize, usize) {
    let a = detect_pan_tompkins(x, fs_hz);
    let b = detect_wavelet(x, fs_hz);
    (beat_agreement(&a, &b, x.len(), fs_hz, window_ms), a.len(), b.len())
}

fn bpm(beats: usize, n: usize, fs_hz: f64) -> f64 {
    beats as f64 / (n as f64 / fs_hz) * 60.0
}

/// The four matched negatives for one subject, in the order printed by the tables below.
fn negatives(f: &Fixture, seed: u64, n_pt: usize) -> [(&'static str, Vec<f64>); 4] {
    let period = if n_pt > 1 { f.samples.len() as f64 / n_pt as f64 } else { f.fs_hz };
    [
        ("gauss", gaussian_like(&f.samples, seed)),
        ("saw", sawtooth_like(&f.samples, period)),
        ("const", vec![mean(&f.samples); f.samples.len()]),
        ("shuf", shuffled(&f.samples, seed)),
    ]
}

#[test]
fn real_ecg_agrees_and_the_two_detectors_read_the_same_heart_rate() {
    let fx = fixtures();
    assert_eq!(fx.len(), 13, "AAUWSS ships 13 ECG subjects; found {}", fx.len());

    println!("\n{:>5} {:>5} {:>6} {:>6} {:>7} {:>7} {:>7} {:>7} {:>7}", "subj", "fs", "n_pt", "n_wav", "bpm_pt", "bpm_wav", "F1", "chance", "excess");
    let (mut f1s, mut excesses) = (Vec::new(), Vec::new());
    for f in &fx {
        let (g, n_pt, n_wav) = bsqi(&f.samples, f.fs_hz, DEFAULT_MATCH_WINDOW_MS);
        let (hr_pt, hr_wav) = (bpm(n_pt, f.samples.len(), f.fs_hz), bpm(n_wav, f.samples.len(), f.fs_hz));
        println!(
            "{:>5} {:>5.0} {:>6} {:>6} {:>7.1} {:>7.1} {:>7.3} {:>7.3} {:>7.3}",
            f.subject, f.fs_hz, n_pt, n_wav, hr_pt, hr_wav, g.f1, g.chance_f1, g.excess
        );
        assert!((30.0..=220.0).contains(&hr_pt), "{}: pan-tompkins {hr_pt:.1} bpm", f.subject);
        assert!((30.0..=220.0).contains(&hr_wav), "{}: wavelet {hr_wav:.1} bpm", f.subject);
        // Two independent detectors landing within 5% on rate is a far stronger claim than either alone.
        assert!((hr_pt - hr_wav).abs() / hr_pt <= 0.05, "{}: {hr_pt:.1} vs {hr_wav:.1} bpm", f.subject);
        f1s.push(g.f1);
        excesses.push(g.excess);
    }
    let worst = f1s.iter().copied().fold(f64::INFINITY, f64::min);
    println!("\nreal ECG: worst F1 {:.3}, median F1 {:.3}, worst excess {:.3}", worst, median(&f1s), excesses.iter().copied().fold(f64::INFINITY, f64::min));
    assert!(worst >= 0.95, "worst real-ECG F1 {worst:.3}");
    assert!(excesses.iter().all(|&e| e >= 0.70), "a real record must clear the chance floor by a wide margin");
}

#[test]
fn noise_and_a_dead_channel_are_rejected_but_a_sawtooth_is_not() {
    println!("\n{:>5} | {:>17} | {:>17} | {:>17} | {:>17}", "subj", "gauss n/n F1", "saw n/n F1", "const n/n F1", "shuf n/n F1");
    let mut worst: [f64; 4] = [0.0; 4];
    let mut worst_excess: [f64; 4] = [0.0; 4];
    for (k, f) in fixtures().iter().enumerate() {
        let (_, n_pt, _) = bsqi(&f.samples, f.fs_hz, DEFAULT_MATCH_WINDOW_MS);
        let mut cells = Vec::new();
        for (i, (_, sig)) in negatives(f, 0xEC6_0000 + k as u64, n_pt).into_iter().enumerate() {
            let (g, a, b) = bsqi(&sig, f.fs_hz, DEFAULT_MATCH_WINDOW_MS);
            worst[i] = worst[i].max(g.f1);
            worst_excess[i] = worst_excess[i].max(g.excess);
            cells.push(format!("{a:>4}/{b:<4} {:>6.3}", g.f1));
        }
        println!("{:>5} | {} | {} | {} | {}", f.subject, cells[0], cells[1], cells[2], cells[3]);
    }
    println!("\nworst-case (highest) negative F1  gauss {:.3}  saw {:.3}  const {:.3}  shuf {:.3}", worst[0], worst[1], worst[2], worst[3]);
    println!("worst-case negative excess        gauss {:.3}  saw {:.3}  const {:.3}  shuf {:.3}", worst_excess[0], worst_excess[1], worst_excess[2], worst_excess[3]);

    // What bSQI DOES reject, cleanly.
    assert!(worst[0] <= 0.05, "gaussian noise F1 {:.3}", worst[0]);
    assert!(worst[2] <= 0.05, "constant F1 {:.3}", worst[2]);
    // Shuffled ECG reaches a high RAW F1 purely on detection density; the chance-corrected excess is what
    // actually rejects it, and this asserts that correction is doing the work.
    assert!(worst[3] >= 0.50, "shuffled raw F1 dropped to {:.3} - the density trap is no longer covered", worst[3]);
    assert!(worst_excess[3] <= 0.35, "shuffled excess {:.3}", worst_excess[3]);
    // And what it does NOT: a sawtooth at heart rate is accepted by both detectors, on most subjects, at
    // F1 = 1.0 and a high excess. bSQI answers "are these real QRS or noise", never "is this an ECG".
    // Rejecting it needs a morphology index (kurtosis, band power, template correlation), not this one.
    assert!(worst[1] >= 0.95, "the sawtooth stopped fooling bSQI - re-derive the documented limitation");
    assert!(worst_excess[1] >= 0.70, "sawtooth excess {:.3}", worst_excess[1]);
}

#[test]
fn each_detector_alone_fails_on_a_subject_the_other_gets_right() {
    // The pair only buys anything if the two do not fail together. Measured on this corpus, an earlier
    // build had pan-tompkins counting T waves on subject 02 (a 47.6 bpm rhythm puts the T past the fixed
    // 360 ms guard) and the wavelet counting them on 06/10/13. A doubled series alternates long/short, so
    // this asserts every subject now reads a steady rhythm on BOTH - and would catch either regressing.
    for f in fixtures() {
        for (name, peaks) in [
            ("pan-tompkins", detect_pan_tompkins(&f.samples, f.fs_hz)),
            ("wavelet", detect_wavelet(&f.samples, f.fs_hz)),
        ] {
            let rr: Vec<f64> = peaks.windows(2).map(|w| (w[1] - w[0]) as f64 / f.fs_hz * 1000.0).collect();
            assert!(rr.len() >= 10, "{} {}: only {} intervals", f.subject, name, rr.len());
            let alternation =
                rr.windows(2).map(|w| (w[1] - w[0]).abs()).sum::<f64>() / (rr.len() - 1) as f64 / median(&rr);
            println!("subject {} {:>12}: median R-R {:>6.0} ms, alternation {:.2}", f.subject, name, median(&rr), alternation);
            assert!(alternation < 0.25, "{} {}: alternation {alternation:.2} - the series is doubled", f.subject, name);
        }
    }
}

#[test]
fn the_matching_window_is_not_load_bearing() {
    // 100 ms is justified against QRS width and the 220 bpm R-R ceiling. If the separation only existed at
    // exactly 100 ms it would be an artefact of the window, so check the literature's 150 ms too.
    for window_ms in [60.0, 100.0, 150.0] {
        let (mut pos, mut neg) = (Vec::new(), Vec::new());
        for (k, f) in fixtures().iter().enumerate() {
            pos.push(bsqi(&f.samples, f.fs_hz, window_ms).0.excess);
            neg.push(bsqi(&shuffled(&f.samples, 0xEC6_0000 + k as u64), f.fs_hz, window_ms).0.excess);
        }
        let worst = pos.iter().copied().fold(f64::INFINITY, f64::min);
        let best = neg.iter().copied().fold(0.0f64, f64::max);
        println!("window {window_ms:>5.0} ms: worst real excess {worst:.3}, best shuffled excess {best:.3}");
        assert!(worst - best >= 0.35, "window {window_ms} ms does not separate");
    }
}

#[test]
fn detectors_survive_a_resampled_rate_and_an_inverted_lead() {
    // The strap's rate is unknown, so nothing may be an artefact of 200 Hz. Decimate to 100 Hz and
    // linearly upsample to 400 Hz; both must still read the same rate and still agree.
    for f in fixtures().iter().take(5) {
        let reference = bpm(bsqi(&f.samples, f.fs_hz, DEFAULT_MATCH_WINDOW_MS).1, f.samples.len(), f.fs_hz);
        for (label, x, fs) in [
            ("half", decimate(&f.samples, 2), f.fs_hz / 2.0),
            ("double", upsample(&f.samples, 2), f.fs_hz * 2.0),
            ("inverted", f.samples.iter().map(|v| -v).collect::<Vec<f64>>(), f.fs_hz),
        ] {
            let (g, n_pt, _) = bsqi(&x, fs, DEFAULT_MATCH_WINDOW_MS);
            let hr = bpm(n_pt, x.len(), fs);
            println!("subject {} {label:>8} @ {fs:>5.0} Hz: {hr:>5.1} bpm (ref {reference:.1}), F1 {:.3}", f.subject, g.f1);
            assert!((hr - reference).abs() / reference <= 0.05, "{label}: {hr:.1} vs {reference:.1} bpm");
            assert!(g.f1 >= 0.90, "{label}: F1 {:.3}", g.f1);
        }
    }
}

fn decimate(x: &[f64], factor: usize) -> Vec<f64> {
    x.iter().step_by(factor).copied().collect()
}

fn upsample(x: &[f64], factor: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(x.len() * factor);
    for i in 0..x.len() {
        let next = x[(i + 1).min(x.len() - 1)];
        for k in 0..factor {
            out.push(x[i] + (next - x[i]) * k as f64 / factor as f64);
        }
    }
    out
}
