//! Single-component cosinor (Halberg cosine fit) over a rest-activity rhythm → MESOR, amplitude and
//! acrophase, plus a body-clock phase estimate vs the habitual wake time. Pure. Feeds bodyClock + the
//! biological-age (CosinorAge) model. The nightly skin-temperature minimum can corroborate the phase.
//!
//! The activity input is the strap's own on-chip gravity-removed motion magnitude (ENMO, g) at 1 Hz, not
//! a derived |delta gravity| jerk. The 24 h fundamental sits far below that 0.5 Hz Nyquist, so the rate
//! costs this fit nothing.

use std::f64::consts::PI;

/// Radians per clock hour for the 24 h fundamental.
const W_HOURS: f64 = 2.0 * PI / 24.0;

/// Rhythm-strength floor: below this relative amplitude the rhythm is treated as unreadable.
pub const MIN_RELATIVE_AMPLITUDE: f64 = 0.10;
/// Days of coverage needed for a usable fit, and for a "solid" (vs "wide") confidence.
pub const MIN_DAYS_FOR_FIT: u32 = 7;
pub const GOOD_DAYS_FOR_FIT: u32 = 14;
/// Habitual CBTmin sits this many hours before wake; the acrophase sits this many hours after CBTmin.
const CBT_MIN_BEFORE_WAKE_HOURS: f64 = 2.5;
const ACROPHASE_AFTER_CBT_MIN_HOURS: f64 = 12.0;

/// One per-hour rest-activity sample: local clock hour (0..24, may be fractional) + mean motion magnitude.
/// Units are the caller's: MESOR and amplitude carry that scale, while acrophase and relative amplitude are
/// scale-invariant — so the phase path needs no unit conversion and the age path applies its own.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ActivityBin {
    pub hour: f64,
    pub activity: f64,
}

/// A single-component cosinor fit: y ≈ mesor + amplitude·cos(2π(hour − acrophase_hours)/24).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CosinorFit {
    pub mesor: f64,
    pub amplitude: f64,
    pub acrophase_hours: f64,
}

impl CosinorFit {
    /// Relative amplitude (amplitude / |mesor|); 0 when the mesor is 0. The rhythm-strength gate.
    pub fn relative_amplitude(&self) -> f64 {
        if self.mesor != 0.0 {
            self.amplitude / self.mesor.abs()
        } else {
            0.0
        }
    }

    /// Acrophase in radians, CosinorPy convention (−atan2(γ,β), wrapped to (−2π, 0]) — the phase input the
    /// biological-age model expects. `acrophase_hours ∈ [0,24)` so `−hour·ω` already lands in (−2π, 0].
    pub fn acrophase_radians(&self) -> f64 {
        -(self.acrophase_hours * W_HOURS)
    }
}

/// Fit a single 24 h cosine to the (hour, activity) bins by ordinary least squares (Cramer's rule on the
/// 3×3 normal equations for MESOR/β/γ). `None` when fewer than 3 bins or the system is degenerate.
pub fn cosinor(bins: &[ActivityBin]) -> Option<CosinorFit> {
    if bins.len() < 3 {
        return None;
    }
    let n = bins.len() as f64;
    let (mut sum_y, mut sum_c, mut sum_s) = (0.0, 0.0, 0.0);
    let (mut sum_cc, mut sum_ss, mut sum_cs) = (0.0, 0.0, 0.0);
    let (mut sum_yc, mut sum_ys) = (0.0, 0.0);
    for b in bins {
        let c = (W_HOURS * b.hour).cos();
        let s = (W_HOURS * b.hour).sin();
        let y = b.activity;
        sum_y += y;
        sum_c += c;
        sum_s += s;
        sum_cc += c * c;
        sum_ss += s * s;
        sum_cs += c * s;
        sum_yc += y * c;
        sum_ys += y * s;
    }

    let (a11, a12, a13) = (n, sum_c, sum_s);
    let (a21, a22, a23) = (sum_c, sum_cc, sum_cs);
    let (a31, a32, a33) = (sum_s, sum_cs, sum_ss);
    let det = a11 * (a22 * a33 - a23 * a32) - a12 * (a21 * a33 - a23 * a31)
        + a13 * (a21 * a32 - a22 * a31);
    if det.abs() <= 1e-12 {
        return None;
    }

    let det_m = sum_y * (a22 * a33 - a23 * a32) - a12 * (sum_yc * a33 - a23 * sum_ys)
        + a13 * (sum_yc * a32 - a22 * sum_ys);
    let det_b = a11 * (sum_yc * a33 - a23 * sum_ys) - sum_y * (a21 * a33 - a23 * a31)
        + a13 * (a21 * sum_ys - sum_yc * a31);
    let det_g = a11 * (a22 * sum_ys - sum_yc * a32) - a12 * (a21 * sum_ys - sum_yc * a31)
        + sum_y * (a21 * a32 - a22 * a31);

    let mesor = det_m / det;
    let beta = det_b / det;
    let gamma = det_g / det;
    let amplitude = (beta * beta + gamma * gamma).sqrt();
    let mut acrophase_hours = gamma.atan2(beta) / W_HOURS % 24.0;
    if acrophase_hours < 0.0 {
        acrophase_hours += 24.0;
    }
    Some(CosinorFit { mesor, amplitude, acrophase_hours })
}

/// Pool raw `(unix seconds, activity)` samples into per-hour rest-activity bins: the mean activity over each
/// LOCAL clock hour, so a multi-day window collapses to at most 24 bins (ascending; empty hours omitted).
/// `tz_offset_seconds` maps each unix to the user's local hour.
pub fn hourly_bins(samples: &[(i64, f64)], tz_offset_seconds: i64) -> Vec<ActivityBin> {
    let mut sum = [0.0f64; 24];
    let mut count = [0u32; 24];
    for &(unix, activity) in samples {
        let hour = ((unix + tz_offset_seconds).rem_euclid(86_400) / 3600) as usize;
        sum[hour] += activity;
        count[hour] += 1;
    }
    (0..24)
        .filter(|&h| count[h] > 0)
        .map(|h| ActivityBin { hour: h as f64, activity: sum[h] / count[h] as f64 })
        .collect()
}

/// Confidence in a phase estimate, widening as coverage shrinks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PhaseConfidence {
    Unreadable,
    Wide,
    Solid,
}

/// Which way the body clock leans versus the user's schedule (the UI renders the sentence).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PhaseLean {
    Earlier,
    Aligned,
    Later,
}

/// A body-clock phase estimate. Numbers + classification only; the note text is generated by the frontend.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PhaseEstimate {
    pub temp_min_hour: f64,
    pub acrophase_hours: f64,
    pub offset_vs_schedule_minutes: f64,
    pub confidence: PhaseConfidence,
    pub lean: PhaseLean,
}

/// Estimate the body-clock phase from a pooled activity profile and the habitual wake hour. An observed
/// nightly skin-temp minimum (local hour) corroborates the acrophase-derived CBTmin when supplied.
pub fn estimate_phase(
    bins: &[ActivityBin],
    days_observed: u32,
    habitual_wake_hour: f64,
    observed_temp_min_hour: Option<f64>,
) -> Option<PhaseEstimate> {
    let fit = cosinor(bins)?;
    let derived_temp_min = wrap24(fit.acrophase_hours - ACROPHASE_AFTER_CBT_MIN_HOURS);

    if days_observed < MIN_DAYS_FOR_FIT || fit.relative_amplitude() < MIN_RELATIVE_AMPLITUDE {
        let tmin = observed_temp_min_hour.unwrap_or(derived_temp_min);
        return Some(PhaseEstimate {
            temp_min_hour: tmin,
            acrophase_hours: fit.acrophase_hours,
            offset_vs_schedule_minutes: 0.0,
            confidence: PhaseConfidence::Unreadable,
            lean: PhaseLean::Aligned,
        });
    }

    let temp_min_hour = observed_temp_min_hour.unwrap_or(derived_temp_min);
    let ideal_temp_min = wrap24(habitual_wake_hour - CBT_MIN_BEFORE_WAKE_HOURS);
    let offset_minutes = signed_hour_delta(ideal_temp_min, temp_min_hour) * 60.0;

    let confidence = if days_observed >= GOOD_DAYS_FOR_FIT {
        PhaseConfidence::Solid
    } else {
        PhaseConfidence::Wide
    };
    let lean = if offset_minutes > 20.0 {
        PhaseLean::Later
    } else if offset_minutes < -20.0 {
        PhaseLean::Earlier
    } else {
        PhaseLean::Aligned
    };

    Some(PhaseEstimate {
        temp_min_hour,
        acrophase_hours: fit.acrophase_hours,
        offset_vs_schedule_minutes: offset_minutes,
        confidence,
        lean,
    })
}

/// Wrap an hour value into [0, 24).
pub fn wrap24(h: f64) -> f64 {
    let mut x = h % 24.0;
    if x < 0.0 {
        x += 24.0;
    }
    x
}

/// Signed shortest delta in hours from `a` to `b` on the 24 h clock, in (−12, 12].
fn signed_hour_delta(a: f64, b: f64) -> f64 {
    let mut d = (b - a) % 24.0;
    if d > 12.0 {
        d -= 24.0;
    }
    if d <= -12.0 {
        d += 24.0;
    }
    d
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build 24 hourly bins from a known mesor/amplitude/acrophase, then confirm the fit recovers them.
    fn synth(mesor: f64, amp: f64, acro_hours: f64) -> Vec<ActivityBin> {
        (0..24)
            .map(|h| {
                let hour = h as f64;
                let activity = mesor + amp * (W_HOURS * (hour - acro_hours)).cos();
                ActivityBin { hour, activity }
            })
            .collect()
    }

    #[test]
    fn cosinor_recovers_injected_parameters() {
        let fit = cosinor(&synth(100.0, 40.0, 14.0)).unwrap();
        assert!((fit.mesor - 100.0).abs() < 1e-6, "mesor {}", fit.mesor);
        assert!((fit.amplitude - 40.0).abs() < 1e-6, "amp {}", fit.amplitude);
        assert!((fit.acrophase_hours - 14.0).abs() < 1e-6, "acro {}", fit.acrophase_hours);
    }

    #[test]
    fn relative_amplitude_and_radians() {
        let fit = cosinor(&synth(50.0, 5.0, 6.0)).unwrap();
        assert!((fit.relative_amplitude() - 0.1).abs() < 1e-9);
        // acrophase 6h → −6·(2π/24) = −π/2, inside (−2π, 0].
        assert!((fit.acrophase_radians() - (-PI / 2.0)).abs() < 1e-6);
        assert!(fit.acrophase_radians() > -2.0 * PI && fit.acrophase_radians() <= 0.0);
    }

    /// The phase path feeds raw g and the age path feeds mg-ENMO through the SAME fit, which is only sound
    /// because a positive rescale moves MESOR/amplitude together and leaves acrophase + relative amplitude fixed.
    #[test]
    fn a_positive_rescale_leaves_phase_and_relative_amplitude_fixed() {
        let base = cosinor(&synth(0.062, 0.057, 15.0)).unwrap();
        let scaled_bins: Vec<ActivityBin> = synth(0.062, 0.057, 15.0)
            .into_iter()
            .map(|b| ActivityBin { hour: b.hour, activity: b.activity * 1000.0 })
            .collect();
        let scaled = cosinor(&scaled_bins).unwrap();
        assert!((scaled.acrophase_hours - base.acrophase_hours).abs() < 1e-9);
        assert!((scaled.relative_amplitude() - base.relative_amplitude()).abs() < 1e-9);
        assert!((scaled.mesor - base.mesor * 1000.0).abs() < 1e-6);
        assert!((scaled.amplitude - base.amplitude * 1000.0).abs() < 1e-6);
    }

    #[test]
    fn cosinor_needs_three_bins() {
        assert!(cosinor(&[]).is_none());
        assert!(cosinor(&[ActivityBin { hour: 0.0, activity: 1.0 }]).is_none());
    }

    #[test]
    fn hourly_bins_pools_local_hours() {
        // Two samples in local hour 0 (mean), one in hour 1.
        let bins = hourly_bins(&[(0, 0.01), (1800, 0.03), (3600, 0.10)], 0);
        assert_eq!(bins.len(), 2);
        assert!((bins[0].hour - 0.0).abs() < 1e-9 && (bins[0].activity - 0.02).abs() < 1e-9);
        assert!((bins[1].hour - 1.0).abs() < 1e-9 && (bins[1].activity - 0.10).abs() < 1e-9);
        // A +2 h tz offset shifts unix 0 into local hour 2.
        let shifted = hourly_bins(&[(0, 0.05)], 2 * 3600);
        assert_eq!(shifted.len(), 1);
        assert!((shifted[0].hour - 2.0).abs() < 1e-9);
        assert!(hourly_bins(&[], 0).is_empty());
    }

    #[test]
    fn low_amplitude_or_short_history_is_unreadable() {
        let flat = synth(100.0, 1.0, 12.0); // relative amp 0.01 < 0.10
        let e = estimate_phase(&flat, 14, 7.0, None).unwrap();
        assert_eq!(e.confidence, PhaseConfidence::Unreadable);

        let strong = synth(100.0, 40.0, 16.0);
        let short = estimate_phase(&strong, 3, 7.0, None).unwrap();
        assert_eq!(short.confidence, PhaseConfidence::Unreadable);
    }

    #[test]
    fn solid_when_well_covered_and_rhythmic() {
        // acrophase 16h → CBTmin 4h; wake 7h → ideal CBTmin 4.5h; offset small → aligned.
        let e = estimate_phase(&synth(100.0, 40.0, 16.0), 14, 7.0, None).unwrap();
        assert_eq!(e.confidence, PhaseConfidence::Solid);
        assert!((e.acrophase_hours - 16.0).abs() < 1e-6);
    }
}
