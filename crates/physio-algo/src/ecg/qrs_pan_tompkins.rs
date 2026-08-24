//! Pan-Tompkins QRS detection: bandpass, derivative, square, moving-window integrate, dual adaptive
//! threshold with searchback. The ENERGY half of the detector pair (see the module header).
//!
//! Rate adaptation: the published integer filters are fixed at 200 Hz, so the passband is rebuilt here
//! from two centred moving averages sized in Hz. Centred means zero phase, so no group delay has to be
//! subtracted back off the peak positions.

use super::{REFRACTORY_MS, band_limited, qt_end_ms, samples_for_ms, sanitized, usable_rate};
use crate::signal::{find_peaks, moving_average_centred};
use crate::stats::mean;

/// Bandpass corners, as the first null of each moving average. Low-pass null at 30 Hz gives roughly a
/// 13 Hz -3 dB corner; high-pass null at 5 Hz removes baseline wander and the P/T waves.
const LOWPASS_NULL_HZ: f64 = 30.0;
const HIGHPASS_NULL_HZ: f64 = 5.0;
/// Integrator width (ms) — the nominal QRS duration, so one complex fills one window.
const INTEGRATION_MS: f64 = 150.0;
/// Below this gap since the last beat a candidate is slope-tested as a possible T wave. The fixed 360 ms
/// is the published floor; at a slow rate the QT interval outruns it and the T wave is counted as a beat,
/// so the guard also tracks the rhythm through Bazett's `QT = 0.4 * sqrt(R-R)`.
const TWAVE_GUARD_MS: f64 = 360.0;
/// Learning window (s) used only to seed the running peak/noise levels.
const LEARNING_S: f64 = 2.0;
/// Running-estimate weights, and the searchback trigger as a multiple of the recent mean R-R.
const SIGNAL_WEIGHT: f64 = 0.125;
const SEARCHBACK_WEIGHT: f64 = 0.25;
const THRESHOLD_FRACTION: f64 = 0.25;
const SEARCHBACK_RR_MULTIPLE: f64 = 1.66;
/// Beats kept in the running R-R mean that drives searchback.
const RR_HISTORY: usize = 8;

/// R-peak sample indices from a single-lead ECG, ascending and strictly increasing.
///
/// Returns empty (never panics) on an empty or too-short input, an unsupported `fs_hz`, or a signal with
/// no local structure at all. Non-finite samples are flattened to 0.0 first.
pub fn detect_pan_tompkins(samples: &[f64], fs_hz: f64) -> Vec<usize> {
    if !usable_rate(fs_hz) {
        return Vec::new();
    }
    let refractory = samples_for_ms(REFRACTORY_MS, fs_hz, 1);
    let integration = samples_for_ms(INTEGRATION_MS, fs_hz, 1);
    let x = sanitized(samples);
    if x.len() < integration * 4 {
        return Vec::new();
    }

    let band = bandpass(&x, fs_hz);
    let energy = integrated_energy(&band, fs_hz, integration);
    let candidates = find_peaks(&energy, 1, f64::NEG_INFINITY);
    if candidates.is_empty() {
        return Vec::new();
    }

    let accepted = adaptive_threshold(&energy, &candidates, fs_hz, refractory);
    // Report the fiducial on the bandpassed trace, not on the rectified envelope: the integrator's
    // plateau centre can sit up to half a window off the true R peak.
    let mut out: Vec<usize> = accepted
        .iter()
        .map(|&i| refine_fiducial(&band, i, integration / 2))
        .collect();
    out.sort_unstable();
    out.dedup();
    out
}

/// Zero-phase 5-15 Hz bandpass, at this detector's corners.
fn bandpass(x: &[f64], fs_hz: f64) -> Vec<f64> {
    band_limited(x, fs_hz, LOWPASS_NULL_HZ, HIGHPASS_NULL_HZ)
}

/// Derivative, square, moving-window integrate — the rectifying stage that discards sign and phase.
fn integrated_energy(band: &[f64], fs_hz: f64, integration: usize) -> Vec<f64> {
    let n = band.len();
    let mut squared = vec![0.0f64; n];
    for i in 0..n {
        // 5-point centred derivative, edge-clamped so the ends stay finite instead of being dropped.
        let m2 = band[i.saturating_sub(2)];
        let m1 = band[i.saturating_sub(1)];
        let p1 = band[(i + 1).min(n - 1)];
        let p2 = band[(i + 2).min(n - 1)];
        let d = (-m2 - 2.0 * m1 + 2.0 * p1 + p2) / 8.0 * fs_hz;
        squared[i] = d * d;
    }
    moving_average_centred(&squared, integration)
}

/// Dual-threshold scan over the energy peaks: running signal / noise levels, a 200 ms refractory, a
/// T-wave slope test inside 360 ms, and a searchback at 1.66x the recent mean R-R.
fn adaptive_threshold(
    energy: &[f64],
    candidates: &[usize],
    fs_hz: f64,
    refractory: usize,
) -> Vec<usize> {
    let learn = ((LEARNING_S * fs_hz) as usize).min(energy.len()).max(1);
    let head = &energy[..learn];
    let mut signal_level =
        head.iter().copied().fold(f64::MIN, f64::max).max(0.0) * THRESHOLD_FRACTION;
    let mut noise_level = mean(head) * 0.5;

    let mut accepted: Vec<usize> = Vec::new();
    let mut rr: Vec<f64> = Vec::new();
    let mut ci = 0usize;
    while ci < candidates.len() {
        let idx = candidates[ci];
        let value = energy[idx];
        let threshold = noise_level + THRESHOLD_FRACTION * (signal_level - noise_level);

        if let Some(&last) = accepted.last() {
            if idx.saturating_sub(last) < refractory {
                ci += 1;
                continue;
            }
        }
        // Searchback: a gap far longer than the recent rhythm means a real beat was thresholded out, so
        // rescan it at the halved threshold. Runs before the normal test so the beat lands in order.
        if let (Some(&last), true) = (accepted.last(), rr.len() >= 2) {
            let expected = mean(&rr) * SEARCHBACK_RR_MULTIPLE;
            if (idx - last) as f64 > expected {
                if let Some(found) = search_back(
                    energy,
                    candidates,
                    ci,
                    last,
                    refractory,
                    threshold * 0.5,
                    threshold,
                ) {
                    signal_level = SEARCHBACK_WEIGHT * energy[found]
                        + (1.0 - SEARCHBACK_WEIGHT) * signal_level;
                    push_beat(&mut accepted, &mut rr, found);
                    continue; // re-test the current candidate against the new refractory
                }
            }
        }

        if value >= threshold {
            let guard = twave_guard(&rr, fs_hz);
            let is_twave = accepted
                .last()
                .is_some_and(|&last| idx - last < guard && value < energy[last] * 0.5);
            if is_twave {
                noise_level = SIGNAL_WEIGHT * value + (1.0 - SIGNAL_WEIGHT) * noise_level;
            } else {
                signal_level = SIGNAL_WEIGHT * value + (1.0 - SIGNAL_WEIGHT) * signal_level;
                push_beat(&mut accepted, &mut rr, idx);
            }
        } else {
            noise_level = SIGNAL_WEIGHT * value + (1.0 - SIGNAL_WEIGHT) * noise_level;
        }
        ci += 1;
    }
    accepted
}

/// T-wave guard in samples: the published 360 ms, widened to Bazett's QT once there is a rhythm to
/// measure. A 48 bpm rhythm puts the T at ~450 ms, outside the fixed guard and counted as a second beat.
fn twave_guard(rr: &[f64], fs_hz: f64) -> usize {
    let qt_ms = if rr.is_empty() {
        0.0
    } else {
        qt_end_ms(mean(rr) / fs_hz * 1000.0)
    };
    samples_for_ms(TWAVE_GUARD_MS.max(qt_ms), fs_hz, 1)
}

fn push_beat(accepted: &mut Vec<usize>, rr: &mut Vec<f64>, idx: usize) {
    if let Some(&last) = accepted.last() {
        rr.push((idx - last) as f64);
        if rr.len() > RR_HISTORY {
            rr.remove(0);
        }
    }
    accepted.push(idx);
}

/// Largest candidate strictly between the last beat and the current one that clears the halved threshold
/// but not the full one, honouring the refractory on both sides.
fn search_back(
    energy: &[f64],
    candidates: &[usize],
    upto: usize,
    last: usize,
    refractory: usize,
    low: f64,
    high: f64,
) -> Option<usize> {
    let mut best: Option<usize> = None;
    for &c in candidates[..upto].iter().rev() {
        if c <= last + refractory {
            break;
        }
        let v = energy[c];
        if v >= low && v < high && best.is_none_or(|b| v > energy[b]) {
            best = Some(c);
        }
    }
    best
}

/// Index of the largest `|band|` within `radius` of `centre` — the R peak itself.
fn refine_fiducial(band: &[f64], centre: usize, radius: usize) -> usize {
    let lo = centre.saturating_sub(radius);
    let hi = (centre + radius).min(band.len() - 1);
    let mut best = centre.min(band.len() - 1);
    let mut best_v = band[best].abs();
    for (i, v) in band.iter().enumerate().take(hi + 1).skip(lo) {
        if v.abs() > best_v {
            best_v = v.abs();
            best = i;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::super::test_signals::{constant, synthetic_ecg};
    use super::*;

    #[test]
    fn recovers_a_known_beat_count_at_every_supported_rate() {
        for fs in [100.0, 200.0, 250.0, 500.0, 1000.0] {
            let (x, truth) = synthetic_ecg(fs, 20.0, 60.0, 1.0, 0.0, 1);
            let peaks = detect_pan_tompkins(&x, fs);
            assert_eq!(
                peaks.len(),
                truth.len(),
                "fs {fs}: got {} want {}",
                peaks.len(),
                truth.len()
            );
            for (p, t) in peaks.iter().zip(&truth) {
                let err_ms = (*p as f64 - *t as f64).abs() / fs * 1000.0;
                assert!(err_ms <= 30.0, "fs {fs}: fiducial off by {err_ms:.1} ms");
            }
        }
    }

    #[test]
    fn degenerate_input_is_empty_never_a_panic() {
        assert!(detect_pan_tompkins(&[], 200.0).is_empty());
        assert!(detect_pan_tompkins(&constant(2000, 3.0), 200.0).is_empty());
        assert!(detect_pan_tompkins(&[f64::NAN; 2000], 200.0).is_empty());
        assert!(detect_pan_tompkins(&[f64::INFINITY; 2000], 200.0).is_empty());
        // Unsupported rates and a too-short buffer decline rather than guess.
        let (x, _) = synthetic_ecg(200.0, 10.0, 60.0, 1.0, 0.0, 2);
        assert!(detect_pan_tompkins(&x, 50.0).is_empty());
        assert!(detect_pan_tompkins(&x, 2000.0).is_empty());
        assert!(detect_pan_tompkins(&x, f64::NAN).is_empty());
        assert!(detect_pan_tompkins(&x[..10], 200.0).is_empty());
    }

    #[test]
    fn a_lone_nan_does_not_destroy_the_rest_of_the_record() {
        let (mut x, truth) = synthetic_ecg(200.0, 20.0, 60.0, 1.0, 0.0, 3);
        x[1234] = f64::NAN;
        let peaks = detect_pan_tompkins(&x, 200.0);
        assert!(
            peaks.len() >= truth.len() - 1,
            "got {} want ~{}",
            peaks.len(),
            truth.len()
        );
    }

    #[test]
    fn output_is_deterministic_and_strictly_increasing() {
        let (x, _) = synthetic_ecg(250.0, 20.0, 72.0, 1.0, 0.05, 4);
        let a = detect_pan_tompkins(&x, 250.0);
        assert_eq!(a, detect_pan_tompkins(&x, 250.0));
        assert!(a.windows(2).all(|w| w[1] > w[0]));
    }

    #[test]
    fn amplitude_scale_does_not_change_the_answer() {
        // Every threshold is relative, so counts-per-mV cannot shift the beat set. This is what lets the
        // detector run on an uncalibrated y-axis.
        let (x, _) = synthetic_ecg(200.0, 20.0, 66.0, 1.0, 0.02, 5);
        let base = detect_pan_tompkins(&x, 200.0);
        for gain in [1e-3, 37.0, 5000.0] {
            let scaled: Vec<f64> = x.iter().map(|v| v * gain).collect();
            assert_eq!(detect_pan_tompkins(&scaled, 200.0), base, "gain {gain}");
        }
    }
}
