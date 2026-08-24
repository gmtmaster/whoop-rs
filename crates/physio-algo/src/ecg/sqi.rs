//! Published single-lead ECG signal-quality indices: kSQI, pSQI, basSQI and a beat-template
//! correlation. Definitions follow Li, Mark & Clifford, *Physiol Meas* 29(1):15-32 (2008) and Behar,
//! Oster, Li & Clifford, *IEEE Trans Biomed Eng* 60(6):1660-1666 (2013); the template index follows
//! Clifford, Behar, Li & Rezek, *Physiol Meas* 33(9):1419-1433 (2012). Each constant below records
//! whether its value is the published one or one derived here, and from what.
//!
//! Nothing here converts counts to millivolts. Every index is scale- and offset-invariant by
//! construction (a kurtosis is standardised, a band ratio divides the scale out, a Pearson r is
//! invariant to both), which is exactly why they can gate a stream whose amplitude units are unknown.

use crate::ecg::spectrum::Periodogram;
use crate::ecg::{sanitized, usable_rate};
use crate::signal::robust_sigma;
use crate::stats::{least_squares_line, mean, pearson};

/// Behar (2013) band edges, in Hz. pSQI is the 5-15 Hz QRS band over the 5-40 Hz in-band total;
/// basSQI is the fraction of 0-40 Hz power that is NOT baseline wander (0-1 Hz).
pub const QRS_BAND_LO_HZ: f64 = 5.0;
pub const QRS_BAND_HI_HZ: f64 = 15.0;
pub const IN_BAND_HI_HZ: f64 = 40.0;
pub const BASELINE_HI_HZ: f64 = 1.0;

/// The 40 Hz edge only exists below Nyquist at 80 Hz and up, so pSQI and basSQI are undefined below
/// this rate rather than silently clamped to a narrower band that would mean something else.
pub const MIN_FS_FOR_BANDS_HZ: f64 = 2.0 * IN_BAND_HI_HZ;

/// Beat window around each detected R peak, in ms. Covers the QRS and the start of the T wave, which
/// is the span that repeats beat to beat at a stable heart rate.
pub const TEMPLATE_PRE_MS: f64 = 100.0;
pub const TEMPLATE_POST_MS: f64 = 200.0;
/// Fewest beats a template may be built from. Two beats always correlate perfectly against a template
/// that is their own average, so three is the smallest count that can fail.
pub const TEMPLATE_MIN_BEATS: usize = 3;
/// Fewest samples a template window may span. A correlation over three points is dominated by its own
/// endpoints, so a window this narrow is refused rather than scored — the case a low sample rate and a
/// short window (`ecg::morphology`'s PR segment) reach first.
pub const TEMPLATE_MIN_WIDTH: usize = 4;

/// Kurtosis (Pearson's b2, `m4 / m2²`; the excess form is this minus 3). ECG is sharply peaked and
/// strongly non-Gaussian because of the QRS; gaussian noise sits at 3. `None` on fewer than four
/// samples or on a constant signal.
pub fn k_sqi(samples: &[f64]) -> Option<f64> {
    let x = sanitized(samples);
    if x.len() < 4 {
        return None;
    }
    let m = mean(&x);
    let n = x.len() as f64;
    let m2 = x.iter().map(|v| (v - m).powi(2)).sum::<f64>() / n;
    if m2 <= 0.0 {
        return None;
    }
    let m4 = x.iter().map(|v| (v - m).powi(4)).sum::<f64>() / n;
    Some(m4 / (m2 * m2))
}

/// pSQI: QRS-band power (5-15 Hz) over in-band power (5-40 Hz). `None` when `fs_hz` is unusable, below
/// [`MIN_FS_FOR_BANDS_HZ`], or when the in-band power is zero.
pub fn p_sqi(spectrum: &Periodogram, fs_hz: f64) -> Option<f64> {
    if !usable_rate(fs_hz) || fs_hz < MIN_FS_FOR_BANDS_HZ || spectrum.is_empty() {
        return None;
    }
    let qrs = spectrum.band_power(QRS_BAND_LO_HZ / fs_hz, QRS_BAND_HI_HZ / fs_hz);
    let in_band = spectrum.band_power(QRS_BAND_LO_HZ / fs_hz, IN_BAND_HI_HZ / fs_hz);
    (in_band > 0.0).then(|| qrs / in_band)
}

/// basSQI: the fraction of 0-40 Hz power that is not baseline wander, `1 − P(0-1) / P(0-40)`. A value
/// near 1 means little drift; the raw wander ratio the name suggests is `1 − bas_sqi`.
pub fn bas_sqi(spectrum: &Periodogram, fs_hz: f64) -> Option<f64> {
    if !usable_rate(fs_hz) || fs_hz < MIN_FS_FOR_BANDS_HZ || spectrum.is_empty() {
        return None;
    }
    // The integral starts one bin above DC: the buffer is already linearly detrended, so the DC bin
    // carries only the window's own leakage and would otherwise dominate the baseline term.
    let lo = spectrum.bin_width();
    let total = spectrum.band_power(lo, IN_BAND_HI_HZ / fs_hz);
    if total <= 0.0 {
        return None;
    }
    let wander = spectrum.band_power(lo, BASELINE_HI_HZ / fs_hz);
    Some((1.0 - wander / total).clamp(0.0, 1.0))
}

/// Whether each beat window is linearly detrended before it joins the template.
///
/// `AsIs` keeps the window as sampled. `Detrended` removes each window's own OLS line first, which is
/// what a low-amplitude feature needs: over a window a fraction of a second wide, baseline wander and
/// the slow flank of a neighbouring complex are both close to linear, and a Pearson r is invariant to
/// an offset and a scale but NOT to a slope, so an undetrended pair of windows can correlate on a
/// shared drift alone.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WindowBaseline {
    AsIs,
    Detrended,
}

/// A beat template and how well the individual beats match it.
#[derive(Clone, Debug, PartialEq)]
pub struct BeatTemplate {
    /// Mean of [`correlations`](Self::correlations).
    pub correlation: f64,
    /// Beats that fit wholly inside the buffer and were used.
    pub beats: usize,
    /// Leave-one-out Pearson correlation of each beat against the average of the others, in beat order.
    /// Shorter than `beats` when a beat had no variance to correlate.
    pub correlations: Vec<f64>,
    /// The average beat, `hi_ms - lo_ms` long, in the caller's own units.
    pub template: Vec<f64>,
    /// Robust spread of every beat's departure from the template, pooled — the part of a beat the
    /// template does not explain, in the caller's own units.
    pub residual_sigma: f64,
}

/// Beat-template correlation over the default QRS-and-T window, `-`[`TEMPLATE_PRE_MS`] to
/// `+`[`TEMPLATE_POST_MS`] around each R peak, windows as sampled.
pub fn beat_template(samples: &[f64], fs_hz: f64, peaks: &[usize]) -> Option<BeatTemplate> {
    beat_template_window(
        samples,
        fs_hz,
        peaks,
        -TEMPLATE_PRE_MS,
        TEMPLATE_POST_MS,
        WindowBaseline::AsIs,
    )
}

/// Beat-template correlation over an arbitrary window `lo_ms..=hi_ms` relative to each R peak, in ms;
/// negative is before the peak, so a window wholly ahead of it (the PR segment) is expressible.
///
/// Beats are aligned on their peaks, averaged into a template, and each beat is scored against the
/// average of the OTHERS — a beat never contributes to the template it is compared with, so a handful
/// of beats cannot inflate the score toward 1 by self-inclusion. `None` on an unusable rate, an
/// inverted or sub-[`TEMPLATE_MIN_WIDTH`] window, or fewer than [`TEMPLATE_MIN_BEATS`] whole beats.
pub fn beat_template_window(
    samples: &[f64],
    fs_hz: f64,
    peaks: &[usize],
    lo_ms: f64,
    hi_ms: f64,
    baseline: WindowBaseline,
) -> Option<BeatTemplate> {
    if !usable_rate(fs_hz) || !lo_ms.is_finite() || !hi_ms.is_finite() || hi_ms <= lo_ms {
        return None;
    }
    let lo = (lo_ms / 1000.0 * fs_hz).round() as isize;
    let hi = (hi_ms / 1000.0 * fs_hz).round() as isize;
    let width = (hi - lo + 1) as usize;
    if width < TEMPLATE_MIN_WIDTH {
        return None;
    }
    let x = sanitized(samples);
    let beats: Vec<Vec<f64>> = peaks
        .iter()
        .filter_map(|&p| {
            let (s, e) = (p as isize + lo, p as isize + hi);
            (s >= 0 && e < x.len() as isize)
                .then(|| levelled(&x[s as usize..=e as usize], baseline))
        })
        .collect();
    if beats.len() < TEMPLATE_MIN_BEATS {
        return None;
    }

    let mut sum = vec![0.0; width];
    for b in &beats {
        for (s, v) in sum.iter_mut().zip(b.iter()) {
            *s += v;
        }
    }
    let n = beats.len() as f64;
    let template: Vec<f64> = sum.iter().map(|s| s / n).collect();

    let mut rs = Vec::with_capacity(beats.len());
    let mut residual = Vec::with_capacity(beats.len() * width);
    for b in &beats {
        let others: Vec<f64> = sum
            .iter()
            .zip(b.iter())
            .map(|(s, v)| (s - v) / (n - 1.0))
            .collect();
        if let Some(r) = pearson(b, &others) {
            rs.push(r);
        }
        residual.extend(b.iter().zip(template.iter()).map(|(v, m)| v - m));
    }
    (!rs.is_empty()).then(|| BeatTemplate {
        correlation: mean(&rs),
        beats: beats.len(),
        correlations: rs,
        template,
        residual_sigma: robust_sigma(&residual),
    })
}

/// The window as sampled, or with its own OLS line removed.
fn levelled(window: &[f64], baseline: WindowBaseline) -> Vec<f64> {
    match baseline {
        WindowBaseline::AsIs => window.to_vec(),
        WindowBaseline::Detrended => {
            let (slope, intercept) = least_squares_line(window);
            window
                .iter()
                .enumerate()
                .map(|(i, &v)| v - (slope * i as f64 + intercept))
                .collect()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// A deterministic pseudo-gaussian: sum of 12 uniforms minus 6 (Irwin-Hall), from a LCG.
    fn gaussian(n: usize, seed: u64) -> Vec<f64> {
        let mut s = seed;
        let mut next = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (s >> 11) as f64 / (1u64 << 53) as f64
        };
        (0..n)
            .map(|_| (0..12).map(|_| next()).sum::<f64>() - 6.0)
            .collect()
    }

    #[test]
    fn kurtosis_separates_gaussian_noise_from_a_spiky_train() {
        let noise = k_sqi(&gaussian(4000, 7)).unwrap();
        assert!(
            (noise - 3.0).abs() < 0.5,
            "gaussian kurtosis must sit near 3, got {noise}"
        );
        // One spike every 200 samples: sharply peaked, so kurtosis climbs far above 3.
        let spiky: Vec<f64> = (0..4000)
            .map(|i| if i % 200 == 0 { 10.0 } else { 0.0 })
            .collect();
        assert!(k_sqi(&spiky).unwrap() > 20.0);
        assert!(k_sqi(&[1.0, 1.0, 1.0, 1.0]).is_none()); // constant has no scale
        assert!(k_sqi(&[1.0, 2.0]).is_none());
    }

    #[test]
    fn p_sqi_moves_with_where_the_power_is() {
        let fs = 200.0;
        let tone = |hz: f64| -> Vec<f64> {
            (0..2000)
                .map(|i| (2.0 * PI * hz * i as f64 / fs).sin())
                .collect()
        };
        let inside = p_sqi(&Periodogram::new(&tone(10.0)), fs).unwrap();
        let outside = p_sqi(&Periodogram::new(&tone(30.0)), fs).unwrap();
        assert!(inside > 0.95, "a 10 Hz tone is all QRS-band, got {inside}");
        assert!(outside < 0.05, "a 30 Hz tone is none of it, got {outside}");
        // Below 80 Hz the 40 Hz edge is above Nyquist, so the index is refused rather than clamped.
        assert!(p_sqi(&Periodogram::new(&tone(10.0)), 64.0).is_none());
    }

    #[test]
    fn bas_sqi_falls_when_baseline_wander_is_added() {
        let fs = 200.0;
        let clean: Vec<f64> = (0..4000)
            .map(|i| (2.0 * PI * 10.0 * i as f64 / fs).sin())
            .collect();
        let drifted: Vec<f64> = clean
            .iter()
            .enumerate()
            .map(|(i, v)| v + 20.0 * (2.0 * PI * 0.3 * i as f64 / fs).sin())
            .collect();
        let a = bas_sqi(&Periodogram::new(&clean), fs).unwrap();
        let b = bas_sqi(&Periodogram::new(&drifted), fs).unwrap();
        assert!(a > 0.99, "clean must be near 1, got {a}");
        assert!(b < 0.2, "0.3 Hz wander must dominate, got {b}");
        assert!(bas_sqi(&Periodogram::new(&clean), 50.0).is_none());
    }

    #[test]
    fn template_correlation_is_high_for_repeats_and_low_for_noise() {
        let fs = 200.0;
        let width = 61usize; // one arbitrary but repeated shape
        let shape: Vec<f64> = (0..width)
            .map(|i| (PI * i as f64 / width as f64).sin().powi(3))
            .collect();
        let (mut real, mut peaks) = (vec![0.0; 4000], Vec::new());
        for k in 0..15 {
            let start = 100 + k * 200;
            for (j, v) in shape.iter().enumerate() {
                real[start + j] = *v;
            }
            peaks.push(start + width / 2);
        }
        let t = beat_template(&real, fs, &peaks).unwrap();
        assert!(
            t.correlation > 0.99,
            "identical beats must correlate, got {}",
            t.correlation
        );
        assert_eq!(t.beats, 15);

        // Same peak positions over noise: no stable shape, so the leave-one-out template collapses.
        let noise = gaussian(4000, 11);
        let n = beat_template(&noise, fs, &peaks).unwrap();
        assert!(
            n.correlation < 0.3,
            "noise must not build a template, got {}",
            n.correlation
        );
    }

    #[test]
    fn an_arbitrary_window_reads_the_span_asked_for_and_a_detrend_kills_a_shared_slope() {
        let fs = 200.0;
        let peaks: Vec<usize> = (0..12).map(|k| 400 + k * 200).collect();
        // A bump 150 ms BEFORE each peak and nothing at the peak: only a window ahead of the peak sees it.
        let mut x = vec![0.0; 4000];
        for &p in &peaks {
            for j in 0..21usize {
                x[p - 25 - 10 + j] += (std::f64::consts::PI * j as f64 / 20.0).sin();
            }
        }
        let ahead =
            beat_template_window(&x, fs, &peaks, -250.0, -50.0, WindowBaseline::AsIs).unwrap();
        assert_eq!(ahead.template.len(), 41, "200 ms at 200 Hz is 40 steps");
        assert!(ahead.template.iter().fold(0.0f64, |a, &v| a.max(v.abs())) > 0.9);
        assert!(
            ahead.residual_sigma < 1e-12,
            "identical beats leave no residual"
        );

        // A steep shared ramp with independent noise on it: undetrended the windows correlate almost
        // perfectly on the slope alone, which is a drift agreeing with itself, not a repeating feature.
        let noise = gaussian(4000, 21);
        let ramp: Vec<f64> = noise
            .iter()
            .enumerate()
            .map(|(i, v)| i as f64 * 2.0 + v)
            .collect();
        let sloped =
            beat_template_window(&ramp, fs, &peaks, -250.0, -50.0, WindowBaseline::AsIs).unwrap();
        let flat =
            beat_template_window(&ramp, fs, &peaks, -250.0, -50.0, WindowBaseline::Detrended)
                .unwrap();
        assert!(
            sloped.correlation > 0.99,
            "the slope alone must inflate it, got {}",
            sloped.correlation
        );
        assert!(
            flat.correlation.abs() < 0.3,
            "detrended, only the noise is left, got {}",
            flat.correlation
        );
        // An inverted window and one too narrow to correlate over are refused, not guessed at.
        assert!(
            beat_template_window(&x, fs, &peaks, -50.0, -250.0, WindowBaseline::AsIs).is_none()
        );
        assert!(beat_template_window(&x, fs, &peaks, -60.0, -50.0, WindowBaseline::AsIs).is_none());
    }

    #[test]
    fn template_refuses_too_few_beats_and_bad_rates() {
        let x = vec![0.0; 4000];
        assert!(beat_template(&x, 200.0, &[500, 1000]).is_none()); // two beats cannot fail, so refuse
        assert!(beat_template(&x, 5.0, &[500, 1000, 1500]).is_none());
        assert!(beat_template(&x, 200.0, &[1, 2, 3]).is_none()); // no whole window fits
    }
}
