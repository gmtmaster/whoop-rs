//! One pass over a candidate decode: every signal-quality index, each one still visible, plus a
//! per-index verdict. A single fused number would say a candidate failed without saying which
//! assumption broke, and the sweep that consumes this needs to know which.
//!
//! `score` never converts counts to millivolts and never reports an amplitude. Every index it carries
//! is invariant to the amplitude scale, so the result is as meaningful on raw counts as on mV.

use crate::ecg::spectrum::Periodogram;
use crate::ecg::sqi::{bas_sqi, beat_template, k_sqi, p_sqi};
use crate::ecg::{
    DEFAULT_MATCH_WINDOW_MS, beat_agreement, detect_pan_tompkins, detect_wavelet, usable_rate,
};
use crate::stats::median;

/// Detector-agreement floor. DERIVED, not published. Measured over 39 ten-second windows from 13
/// AAUWSS subjects and their matched negatives: real 0.667-1.000; gaussian noise 0.00-0.05, matched
/// PPG 0.00-0.00. This is the index that kills unstructured noise and a non-ECG pulse waveform.
pub const B_SQI_MIN: f64 = 0.60;
/// Kurtosis floor. Li, Mark & Clifford (2008) take kurtosis > 5 as clean single-lead ECG; that
/// rejects 6 of the 39 real windows here, so the value is DERIVED instead. Measured: real 2.19-29.6,
/// sawtooth 1.75-1.86 (a uniform amplitude distribution sits at 1.8 analytically). **This is the only
/// index that rejects the sawtooth, and its margin is 0.33 — the thinnest in the set.** A permutation
/// of real ECG keeps the same amplitude distribution, so this index cannot see the shuffled negative
/// at all; pSQI and the template are what handle that one.
pub const K_SQI_MIN: f64 = 2.00;
/// QRS-band fraction, Behar et al. (2013): 0.5-0.8 for clean adult ECG, used unchanged. Measured real
/// range 0.581-0.792 sits inside it; the upper bound is what rejects the PPG negative, whose power is
/// entirely below 15 Hz and reads 1.00.
pub const P_SQI_MIN: f64 = 0.50;
pub const P_SQI_MAX: f64 = 0.80;
/// Baseline-wander floor, Behar et al. (2013): basSQI > 0.95 as clean. REPORTED, NOT GATED. Measured
/// real 0.148-0.997 against negatives 0.374-0.992 — sleeping ECG drifts and the ranges overlap
/// completely, so this index separates nothing here and would only cost real windows.
pub const BAS_SQI_MIN: f64 = 0.95;
/// Beat-template floor. The conventional figure is ~0.9; DERIVED to 0.70 from the measured ranges:
/// real 0.745-0.996, matched PPG 0.43-0.66, gaussian -0.04-0.06, shuffled -0.02-0.28.
pub const TEMPLATE_SQI_MIN: f64 = 0.70;
/// Plausible heart rate (bpm). The sharpest constraint the candidate sample rate has to satisfy: a
/// wrong rate scales every R-R interval and pushes the implied rate straight out of this band.
pub const MIN_HR_BPM: f64 = 30.0;
pub const MAX_HR_BPM: f64 = 220.0;
/// Indices that must all pass for [`EcgVerdict::accepted`]: bSQI, kSQI, pSQI, the beat template and
/// the implied heart rate. basSQI is computed and reported but does not gate — see its note.
pub const GATED_INDEX_COUNT: usize = 5;

/// Which indices passed. An index that could not be computed counts as a failure: it cannot vouch for
/// the candidate either way, and a sweep that treated "unknown" as "fine" would accept short or
/// low-rate windows on the strength of nothing.
///
/// `k_ok` and `bas_ok` are reported against their published thresholds but do not enter `accepted`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EcgVerdict {
    pub b_ok: bool,
    pub k_ok: bool,
    pub p_ok: bool,
    pub bas_ok: bool,
    pub template_ok: bool,
    pub hr_ok: bool,
    /// How many of the [`GATED_INDEX_COUNT`] gated indices passed — the debugging figure.
    pub passed: usize,
    pub accepted: bool,
}

/// Every index for one candidate decode, plus what the beat detectors saw.
#[derive(Clone, Debug, PartialEq)]
pub struct EcgScore {
    /// The candidate rate this was scored at. Every time-domain figure below is conditional on it.
    pub fs_hz: f64,
    pub samples: usize,
    /// bSQI: one-to-one agreement (F1) between the energy and the shape QRS detectors.
    pub b_sqi: f64,
    /// The same agreement minus the F1 two independent detectors reach at these densities. `b_sqi`
    /// alone rises with detection density, so a candidate whose peaks are dense and spurious can carry
    /// a high one; this is the part of the agreement density does not explain. Reported, not gated.
    pub b_excess: f64,
    pub k_sqi: Option<f64>,
    pub p_sqi: Option<f64>,
    pub bas_sqi: Option<f64>,
    /// Mean leave-one-out beat-to-template correlation.
    pub template_sqi: Option<f64>,
    /// Beats the energy detector found.
    pub beats: usize,
    /// Median-R-R heart rate implied by `fs_hz`; `None` on fewer than two beats.
    pub mean_hr_bpm: Option<f64>,
    pub verdict: EcgVerdict,
}

/// Score `samples` as if they were sampled at `fs_candidate`. Pure; no amplitude calibration anywhere.
pub fn score(samples: &[f64], fs_candidate: f64) -> EcgScore {
    let n = samples.len();
    if !usable_rate(fs_candidate) {
        return EcgScore {
            fs_hz: fs_candidate,
            samples: n,
            b_sqi: 0.0,
            b_excess: 0.0,
            k_sqi: None,
            p_sqi: None,
            bas_sqi: None,
            template_sqi: None,
            beats: 0,
            mean_hr_bpm: None,
            verdict: verdict(0.0, None, None, None, None, None),
        };
    }

    let energy = detect_pan_tompkins(samples, fs_candidate);
    let shape = detect_wavelet(samples, fs_candidate);
    let agreement = beat_agreement(&energy, &shape, n, fs_candidate, DEFAULT_MATCH_WINDOW_MS);
    let b = agreement.f1;

    let spectrum = Periodogram::new(samples);
    let k = k_sqi(samples);
    let p = p_sqi(&spectrum, fs_candidate);
    let bas = bas_sqi(&spectrum, fs_candidate);
    let template = beat_template(samples, fs_candidate, &energy).map(|t| t.correlation);
    let hr = median_hr_bpm(&energy, fs_candidate);

    EcgScore {
        fs_hz: fs_candidate,
        samples: n,
        b_sqi: b,
        b_excess: agreement.excess,
        k_sqi: k,
        p_sqi: p,
        bas_sqi: bas,
        template_sqi: template,
        beats: energy.len(),
        mean_hr_bpm: hr,
        verdict: verdict(b, k, p, bas, template, hr),
    }
}

/// Heart rate from the median R-R of a peak set at `fs_hz`; `None` on fewer than two peaks.
fn median_hr_bpm(peaks: &[usize], fs_hz: f64) -> Option<f64> {
    if peaks.len() < 2 {
        return None;
    }
    let gaps: Vec<f64> = peaks.windows(2).map(|w| (w[1] - w[0]) as f64).collect();
    let m = median(&gaps);
    (m > 0.0).then(|| 60.0 * fs_hz / m)
}

fn verdict(
    b: f64,
    k: Option<f64>,
    p: Option<f64>,
    bas: Option<f64>,
    template: Option<f64>,
    hr: Option<f64>,
) -> EcgVerdict {
    let b_ok = b >= B_SQI_MIN;
    let k_ok = k.is_some_and(|v| v >= K_SQI_MIN);
    let p_ok = p.is_some_and(|v| (P_SQI_MIN..=P_SQI_MAX).contains(&v));
    let bas_ok = bas.is_some_and(|v| v >= BAS_SQI_MIN);
    let template_ok = template.is_some_and(|v| v >= TEMPLATE_SQI_MIN);
    let hr_ok = hr.is_some_and(|v| (MIN_HR_BPM..=MAX_HR_BPM).contains(&v));
    let passed = [b_ok, k_ok, p_ok, template_ok, hr_ok]
        .iter()
        .filter(|v| **v)
        .count();
    EcgVerdict {
        b_ok,
        k_ok,
        p_ok,
        bas_ok,
        template_ok,
        hr_ok,
        passed,
        accepted: passed == GATED_INDEX_COUNT,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// A crude but repeatable beat train: a narrow gaussian R spike with a broad T bump after it.
    fn beats(n: usize, fs: f64, bpm: f64) -> Vec<f64> {
        let period = 60.0 / bpm;
        (0..n)
            .map(|i| {
                let t = i as f64 / fs;
                let phase = t % period;
                let g = |c: f64, s: f64| (-((phase - c) / s).powi(2) / 2.0).exp();
                g(0.10, 0.008) - 0.25 * g(0.13, 0.010) + 0.30 * g(0.40, 0.060)
            })
            .collect()
    }

    #[test]
    fn a_clean_beat_train_scores_and_a_flat_line_does_not() {
        let s = score(&beats(4000, 200.0, 60.0), 200.0);
        assert!(s.b_sqi > B_SQI_MIN, "{s:?}");
        assert!(s.template_sqi.unwrap() > TEMPLATE_SQI_MIN, "{s:?}");
        assert!((s.mean_hr_bpm.unwrap() - 60.0).abs() < 3.0, "{s:?}");

        let flat = score(&vec![0.0; 4000], 200.0);
        assert!(
            !flat.verdict.accepted && flat.verdict.passed == 0,
            "{flat:?}"
        );
        assert_eq!(flat.b_sqi, 0.0, "two empty peak sets are not agreement");
    }

    #[test]
    fn an_unusable_rate_yields_no_index_at_all() {
        let s = score(&beats(4000, 200.0, 60.0), 5.0);
        assert!(s.k_sqi.is_none() && s.p_sqi.is_none() && s.template_sqi.is_none());
        assert_eq!(s.verdict.passed, 0);
        assert!(!s.verdict.accepted);
    }

    #[test]
    fn the_wrong_candidate_rate_moves_the_implied_heart_rate_out_of_band() {
        // The same samples read at 5x the true rate imply a 5x heart rate: 60 bpm becomes 300.
        let x = beats(4000, 200.0, 60.0);
        let wrong = score(&x, 1000.0);
        assert!(
            !wrong.verdict.hr_ok,
            "300 bpm must be refused: {:?}",
            wrong.mean_hr_bpm
        );
        assert!(!wrong.verdict.accepted);
    }

    #[test]
    fn a_pure_tone_has_the_kurtosis_of_a_tone_and_is_refused() {
        // A 3 Hz sine has kurtosis 1.5, far under the ECG floor, however periodic it looks.
        let tone: Vec<f64> = (0..4000)
            .map(|i| (2.0 * PI * 3.0 * i as f64 / 200.0).sin())
            .collect();
        let s = score(&tone, 200.0);
        assert!(s.k_sqi.unwrap() < K_SQI_MIN, "kurtosis {:?}", s.k_sqi);
        assert!(!s.verdict.accepted, "{s:?}");
    }
}
