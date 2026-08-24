//! P-wave presence in the PR segment — a three-way finding, never a boolean.
//!
//! The atrial deflection is the ECG feature R-R scatter cannot reach, and the honest difficulty is that
//! a P wave is roughly a tenth of an R wave. At the amplitude a wrist electrode delivers it can sit
//! UNDER the noise, and "I could not see it" is not "it was not there". So the measure separates three
//! states: a consistent deflection was found, no consistent deflection was found AND the ensemble was
//! quiet enough that one this size would have shown, or neither could be established.
//!
//! Consistency, not amplitude, is what is measured. Any deflection is easy to find in the PR segment;
//! what marks a P wave is that it repeats — same place, same shape, beat after beat. Every window is
//! scored against the average of the OTHER windows ([`beat_template_window`]), so incoherent activity
//! in the same band does not read as a P wave, and no beat can vouch for itself.
//!
//! Every figure is a RATIO against the record's own R amplitude, so nothing here needs counts-per-mV.

use super::evidence_fraction;
use crate::ecg::sqi::{TEMPLATE_MIN_WIDTH, WindowBaseline, beat_template_window};
use crate::ecg::{qt_end_ms, sanitized, usable_rate};
use crate::signal::moving_average_centred;
use crate::stats::median;

/// Search window for the P wave, in ms before the R peak: it ends [`PR_GUARD_MS`] early to stay clear
/// of Q, and runs [`P_SEARCH_MS`] back from there. 60-260 ms covers a 120-200 ms PR interval and the
/// ~100 ms the wave itself occupies.
pub const PR_GUARD_MS: f64 = 60.0;
pub const P_SEARCH_MS: f64 = 200.0;

/// Low-pass null applied before the search: it keeps the P wave (a ~100 ms bump, so its energy sits
/// near 5 Hz) while dropping sensor and muscle noise. There is deliberately NO high-pass — subtracting
/// a long moving average leaves a broad shadow of the QRS reaching back over the PR segment, and that
/// shadow is identical on every beat, so it reads as a perfectly consistent deflection. Baseline drift
/// is removed per window instead, by [`WindowBaseline::Detrended`].
pub const P_LOWPASS_NULL_HZ: f64 = 30.0;

/// Eligible beats needed before any finding is returned.
pub const P_MIN_BEATS: usize = 8;
/// Eligible beats at which the reading has its full evidence budget; [`PWaveEvidence::confidence`] is
/// the fraction of it that was met. DERIVED: 30 beats is ~30 s at a resting rate and drops the
/// ensemble noise floor by 5.5x, which is what makes the absence claim reachable at all.
pub const P_FULL_EVIDENCE_BEATS: usize = 30;

/// A beat counts as carrying a P wave when its PR window correlates with the ensemble of the others at
/// least this well. DERIVED, not published.
pub const P_BEAT_CORRELATION_MIN: f64 = 0.50;
/// Share of eligible beats that must carry one for [`PWaveFinding::Present`], and the share below which
/// [`PWaveFinding::Absent`] may be claimed. Between them the finding is `Indeterminate`, and on a
/// labelled corpus that middle band is where a good deal of irregular rhythm lands — a measured cost of
/// refusing to force a two-way answer, not a defect.
pub const P_PRESENT_FRACTION_MIN: f64 = 0.60;
pub const P_ABSENT_FRACTION_MAX: f64 = 0.35;

/// Anti-dust floor: an ensemble deflection under this fraction of the record's own R amplitude is not
/// called a P wave however consistent it is. DERIVED, and deliberately far below physiology — its ONLY
/// job is to stop windows agreeing perfectly on structure orders of magnitude too small to be a wave.
///
/// It is not a discriminator and must not be used as one. Measured deflection over R across two
/// independent corpora: annotated sinus 0.015-0.080, annotated irregular rhythm 0.007-0.089. Those
/// OVERLAP, so any amplitude threshold placed between them would separate the corpora rather than the
/// physiology — an earlier 0.02 sat inside the sinus range and reported two real P waves as absent.
/// Consistency is what separates here; amplitude only says the deflection is real at all.
pub const P_MIN_DEFLECTION_RATIO: f64 = 0.005;

/// The smallest P wave a claim of ABSENCE must have been able to see, as a fraction of R. DERIVED: the
/// smallest deflection measured on a record annotated as sinus, over both corpora. Absence is only
/// allowed when the ensemble noise floor sits [`P_DETECT_SNR`] under this — i.e. when a wave as small
/// as the smallest real one on record would have shown.
pub const P_RESOLVABLE_RATIO: f64 = 0.015;
/// Deflection-over-noise a claim of absence requires.
pub const P_DETECT_SNR: f64 = 3.0;

/// What the PR segment showed. `Absent` is a positive claim about a quiet ensemble; `Indeterminate` is
/// the answer whenever that claim cannot be supported, and it is a first-class result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PWaveFinding {
    /// A consistent deflection sits in the PR segment across the record.
    Present,
    /// No consistent deflection, and the ensemble was quiet enough that one would have shown.
    Absent,
    /// Neither could be established. The variant says which limit was hit.
    Indeterminate(PWaveLimit),
}

/// Why a finding could not be made. Every variant is a measured property of the INPUT.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PWaveLimit {
    /// Outside the supported sample-rate span.
    UnusableRate,
    /// The sample rate makes the PR window too few samples to correlate over.
    WindowTooNarrow { samples: usize, need: usize },
    /// Too few beats survived eligibility (an R-R too short for the window to clear the previous
    /// T wave, or a window running off either end of the buffer).
    TooFewBeats { have: usize, need: usize },
    /// The beats carry no amplitude to measure a deflection against.
    NoReferenceAmplitude,
    /// Nothing consistent was found, but the noise floor is above the smallest wave worth claiming —
    /// so the wave may simply be under it. This is the variant that keeps the measure honest.
    BelowNoiseFloor,
    /// Between the present and absent shares: the evidence does not decide.
    Ambiguous,
}

/// The P-wave reading. Every ratio is against the record's own R amplitude and so is unit-free.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PWaveEvidence {
    pub finding: PWaveFinding,
    /// Beats with a usable PR window.
    pub beats_examined: usize,
    /// Beats dropped for a short preceding R-R or a window off the end of the buffer.
    pub beats_excluded: usize,
    /// Share of examined beats whose PR window matched the ensemble of the others.
    pub present_fraction: Option<f64>,
    /// Mean leave-one-out correlation across the PR windows.
    pub consistency: Option<f64>,
    /// Peak ensemble deflection over median R amplitude.
    pub amplitude_ratio: Option<f64>,
    /// Ensemble noise floor over median R amplitude — the detectability limit this reading ran into.
    pub noise_ratio: Option<f64>,
    /// Eligible beats over [`P_FULL_EVIDENCE_BEATS`], capped at 1.0. How much of the evidence budget
    /// the reading had — NOT a probability that the finding is right.
    pub confidence: f64,
}

impl PWaveEvidence {
    fn refused(limit: PWaveLimit, examined: usize, excluded: usize) -> Self {
        PWaveEvidence {
            finding: PWaveFinding::Indeterminate(limit),
            beats_examined: examined,
            beats_excluded: excluded,
            present_fraction: None,
            consistency: None,
            amplitude_ratio: None,
            noise_ratio: None,
            confidence: evidence_fraction(examined, P_FULL_EVIDENCE_BEATS),
        }
    }
}

/// Search the PR segment before each R peak for a consistent low-amplitude deflection.
///
/// `peaks` are R-peak sample indices in ascending order, as either QRS detector in this module returns
/// them. Pure and deterministic; never panics on empty, constant or non-finite input.
pub fn p_wave(samples: &[f64], fs_hz: f64, peaks: &[usize]) -> PWaveEvidence {
    if !usable_rate(fs_hz) {
        return PWaveEvidence::refused(PWaveLimit::UnusableRate, 0, peaks.len());
    }
    let (lo_ms, hi_ms) = (-(PR_GUARD_MS + P_SEARCH_MS), -PR_GUARD_MS);
    let lo = (lo_ms / 1000.0 * fs_hz).round() as isize;
    let hi = (hi_ms / 1000.0 * fs_hz).round() as isize;
    let width = (hi - lo + 1) as usize;
    if width < TEMPLATE_MIN_WIDTH {
        let limit = PWaveLimit::WindowTooNarrow {
            samples: width,
            need: TEMPLATE_MIN_WIDTH,
        };
        return PWaveEvidence::refused(limit, 0, peaks.len());
    }

    let x = sanitized(samples);
    let (eligible, excluded) = eligible_beats(&x, fs_hz, peaks, lo, hi);
    if eligible.len() < P_MIN_BEATS {
        let limit = PWaveLimit::TooFewBeats {
            have: eligible.len(),
            need: P_MIN_BEATS,
        };
        return PWaveEvidence::refused(limit, eligible.len(), excluded);
    }

    let smooth_len = (fs_hz / P_LOWPASS_NULL_HZ).round().max(1.0) as usize;
    let band = moving_average_centred(&x, smooth_len);
    let baseline = WindowBaseline::Detrended;
    let Some(t) = beat_template_window(&band, fs_hz, &eligible, lo_ms, hi_ms, baseline) else {
        let limit = PWaveLimit::TooFewBeats {
            have: eligible.len(),
            need: P_MIN_BEATS,
        };
        return PWaveEvidence::refused(limit, eligible.len(), excluded);
    };

    // R amplitude comes off the UNFILTERED trace against the PR window's own median, while the
    // deflection comes off the low-passed ensemble. The filter takes more off a sharp R than off a slow
    // P, so pairing them this way keeps the denominator large and the ratio conservative.
    let r_amplitudes: Vec<f64> = eligible
        .iter()
        .map(|&p| {
            let start = (p as isize + lo) as usize;
            (x[p] - median(&x[start..start + width])).abs()
        })
        .collect();
    let r_amplitude = median(&r_amplitudes);
    // Every sample is finite by `sanitized`, so a non-positive median is a flat trace, not a NaN.
    if r_amplitude <= 0.0 {
        return PWaveEvidence::refused(PWaveLimit::NoReferenceAmplitude, eligible.len(), excluded);
    }

    // Averaging n windows drops incoherent noise by sqrt(n); that is what an absence claim rests on.
    let ensemble_noise = t.residual_sigma / (eligible.len() as f64).sqrt();
    let noise_ratio = ensemble_noise / r_amplitude;
    let deflection = t.template.iter().fold(0.0f64, |a, &v| a.max(v.abs()));
    let amplitude_ratio = deflection / r_amplitude;
    let matched = t
        .correlations
        .iter()
        .filter(|r| **r >= P_BEAT_CORRELATION_MIN)
        .count();
    let present_fraction = matched as f64 / t.correlations.len() as f64;

    // Order is the integrity of this measure. A deflection that was SEEN can be reported however noisy
    // the record; absence may only be claimed once the ensemble floor is low enough that the smallest
    // real wave on record would have shown. Correlation without amplitude is not a P wave — windows can
    // agree perfectly on structure orders of magnitude too small — so the deflection also has to clear
    // the anti-dust floor to be asserted, and failing it is itself grounds to rule one out.
    let real_deflection = amplitude_ratio >= P_MIN_DEFLECTION_RATIO;
    let finding = if real_deflection && present_fraction >= P_PRESENT_FRACTION_MIN {
        PWaveFinding::Present
    } else if noise_ratio * P_DETECT_SNR > P_RESOLVABLE_RATIO {
        PWaveFinding::Indeterminate(PWaveLimit::BelowNoiseFloor)
    } else if !real_deflection || present_fraction <= P_ABSENT_FRACTION_MAX {
        PWaveFinding::Absent
    } else {
        PWaveFinding::Indeterminate(PWaveLimit::Ambiguous)
    };

    PWaveEvidence {
        finding,
        beats_examined: eligible.len(),
        beats_excluded: excluded,
        present_fraction: Some(present_fraction),
        consistency: Some(t.correlation),
        amplitude_ratio: Some(amplitude_ratio),
        noise_ratio: Some(noise_ratio),
        confidence: evidence_fraction(eligible.len(), P_FULL_EVIDENCE_BEATS),
    }
}

/// Beats whose PR window fits in the buffer AND starts after the previous beat's T wave has ended.
/// Returns the eligible peaks and how many were dropped. The first beat is always dropped: with no
/// preceding interval there is no way to know whether its window sits on a T wave.
fn eligible_beats(
    x: &[f64],
    fs_hz: f64,
    peaks: &[usize],
    lo: isize,
    hi: isize,
) -> (Vec<usize>, usize) {
    let mut eligible = Vec::with_capacity(peaks.len());
    let mut excluded = 0usize;
    for (i, &p) in peaks.iter().enumerate() {
        // `hi` is negative, so a window can fit entirely while the peak itself sits past the end of the
        // buffer; the R amplitude is read AT the peak, so that has to be excluded too.
        let fits = p < x.len() && p as isize + lo >= 0 && p as isize + hi < x.len() as isize;
        let clear_of_t = i > 0 && peaks[i - 1] < p && {
            let rr_ms = (p - peaks[i - 1]) as f64 / fs_hz * 1000.0;
            let t_end = peaks[i - 1] as f64 + qt_end_ms(rr_ms) / 1000.0 * fs_hz;
            t_end <= (p as isize + lo) as f64
        };
        if fits && clear_of_t {
            eligible.push(p);
        } else {
            excluded += 1;
        }
    }
    (eligible, excluded)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecg::test_signals::{constant, synthetic_ecg};

    /// The synthetic generator's beat carries a P wave at 0.15 of R; strip it by rebuilding without it.
    fn with_and_without_p(
        fs: f64,
        seconds: f64,
        noise_sd: f64,
        seed: u64,
    ) -> (Vec<f64>, Vec<f64>, Vec<usize>) {
        let (with, truth) = synthetic_ecg(fs, seconds, 60.0, 1.0, noise_sd, seed);
        // Subtract the P wave the generator adds at -180 ms, sigma 25 ms, gain 0.15.
        let mut without = with.clone();
        for &centre in &truth {
            let mu = centre as f64 - 0.180 * fs;
            let sigma = 0.025 * fs;
            let span = (4.0 * sigma).ceil() as isize;
            for i in
                (mu as isize - span).max(0)..((mu as isize + span).min(without.len() as isize - 1))
            {
                let d = (i as f64 - mu) / sigma;
                without[i as usize] -= 0.15 * (-0.5 * d * d).exp();
            }
        }
        (with, without, truth)
    }

    #[test]
    fn a_clean_p_wave_is_found_and_a_stripped_one_is_reported_absent() {
        // A trace of noise, deliberately: a perfectly flat PR segment has no variance to correlate, so
        // the noise-free case is the one input this measure genuinely cannot score.
        let fs = 250.0;
        let (with, without, truth) = with_and_without_p(fs, 40.0, 0.01, 1);
        let a = p_wave(&with, fs, &truth);
        assert_eq!(a.finding, PWaveFinding::Present, "{a:?}");
        assert!(a.amplitude_ratio.unwrap() > P_MIN_DEFLECTION_RATIO, "{a:?}");
        assert!(
            a.beats_examined >= P_MIN_BEATS && a.confidence > 0.0,
            "{a:?}"
        );

        let b = p_wave(&without, fs, &truth);
        assert_eq!(b.finding, PWaveFinding::Absent, "{b:?}");
        assert!(
            b.present_fraction.unwrap() <= P_ABSENT_FRACTION_MAX,
            "{b:?}"
        );
    }

    #[test]
    fn a_noise_floor_over_the_wave_gives_indeterminate_not_absent() {
        // The same stripped record, buried in noise at 0.9 of the R amplitude. "Not seen" must not
        // become "not there". Measured on the way here: at 0.3 the ensemble of 38 beats still resolves
        // a 5%-of-R wave and correctly answers Absent, so the floor is genuinely hard to reach — which
        // is the point of averaging, and why the threshold is on the ENSEMBLE noise, not the per-beat.
        let fs = 250.0;
        let (_, without, truth) = with_and_without_p(fs, 40.0, 0.90, 2);
        let r = p_wave(&without, fs, &truth);
        assert_eq!(
            r.finding,
            PWaveFinding::Indeterminate(PWaveLimit::BelowNoiseFloor),
            "{r:?}"
        );
        assert!(
            r.noise_ratio.unwrap() * P_DETECT_SNR > P_RESOLVABLE_RATIO,
            "{r:?}"
        );
    }

    #[test]
    fn degenerate_input_is_indeterminate_never_a_panic() {
        let (x, truth) = synthetic_ecg(250.0, 40.0, 60.0, 1.0, 0.0, 3);
        assert!(matches!(
            p_wave(&x, f64::NAN, &truth).finding,
            PWaveFinding::Indeterminate(PWaveLimit::UnusableRate)
        ));
        assert!(matches!(
            p_wave(&x, 250.0, &[]).finding,
            PWaveFinding::Indeterminate(PWaveLimit::TooFewBeats { have: 0, .. })
        ));
        // A flat line has beats nowhere and no amplitude to reference.
        let flat = p_wave(&constant(10_000, 3.0), 250.0, &truth);
        assert!(
            matches!(flat.finding, PWaveFinding::Indeterminate(_)),
            "{flat:?}"
        );
        // Non-finite samples are flattened, not propagated.
        assert!(matches!(
            p_wave(&[f64::NAN; 10_000], 250.0, &truth).finding,
            PWaveFinding::Indeterminate(_)
        ));
        // Peaks past the end of the buffer are excluded, not indexed. The PR window sits BEFORE the
        // peak, so it can fit while the peak itself does not — the one arrangement that would panic.
        let past: Vec<usize> = truth.iter().map(|p| p + x.len()).collect();
        assert!(matches!(
            p_wave(&x, 250.0, &past).finding,
            PWaveFinding::Indeterminate(PWaveLimit::TooFewBeats { have: 0, .. })
        ));
        let mut mixed = truth.clone();
        mixed.push(x.len() + 5);
        assert_eq!(
            p_wave(&x, 250.0, &mixed).beats_examined,
            p_wave(&x, 250.0, &truth).beats_examined
        );
    }

    #[test]
    fn the_pr_window_must_be_wide_enough_to_correlate_over() {
        // 200 ms of search at 100 Hz is 21 samples; at the floor of the supported span it still fits, so
        // the narrow-window refusal is reached by a window the caller narrows, not by a plausible rate.
        let (x, truth) = synthetic_ecg(100.0, 60.0, 60.0, 1.0, 0.0, 4);
        assert!(!matches!(
            p_wave(&x, 100.0, &truth).finding,
            PWaveFinding::Indeterminate(PWaveLimit::WindowTooNarrow { .. })
        ));
    }

    #[test]
    fn a_beat_after_a_short_interval_is_excluded_rather_than_scored_over_a_t_wave() {
        // At 150 bpm the previous T wave runs into the PR window, so every beat is ineligible and the
        // reading refuses instead of measuring repolarisation and calling it a P wave.
        let fs = 250.0;
        let (x, truth) = synthetic_ecg(fs, 40.0, 150.0, 1.0, 0.0, 5);
        let r = p_wave(&x, fs, &truth);
        assert!(r.beats_excluded > 0, "{r:?}");
        assert!(
            matches!(
                r.finding,
                PWaveFinding::Indeterminate(PWaveLimit::TooFewBeats { .. })
            ),
            "{r:?}"
        );
    }

    #[test]
    fn the_answer_does_not_depend_on_the_amplitude_scale() {
        // Every figure is a ratio, which is what lets this run on uncalibrated counts.
        let fs = 250.0;
        let (x, truth) = synthetic_ecg(fs, 40.0, 60.0, 1.0, 0.02, 6);
        let base = p_wave(&x, fs, &truth);
        for gain in [1e-3, 37.0, 5000.0] {
            let scaled: Vec<f64> = x.iter().map(|v| v * gain).collect();
            let r = p_wave(&scaled, fs, &truth);
            assert_eq!(r.finding, base.finding, "gain {gain}");
            assert!(
                (r.amplitude_ratio.unwrap() - base.amplitude_ratio.unwrap()).abs() < 1e-9,
                "gain {gain}"
            );
        }
    }
}
