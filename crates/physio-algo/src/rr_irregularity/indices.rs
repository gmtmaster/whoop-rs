//! The Dash trio: normalised RMSSD, binned Shannon entropy of the successive R-R differences, and the
//! turning point ratio. Dash, Chon, Lu & Raeder, "Automatic real time detection of atrial fibrillation",
//! Annals of Biomedical Engineering 37(9):1701-1709, 2009 — three statistics over one short R-R segment,
//! each reported here on its own so it can be inspected without the other two.
//!
//! Their segment is 128 beats. Each function below states the shorter floor it will still answer at, and
//! returns `None` under it rather than a number the length cannot support.

use crate::hrv::HrvReadiness;

/// Beats needed for a normalised RMSSD: two successive differences.
pub const RMSSD_RATIO_MIN_BEATS: usize = 3;
/// Beats needed for the binned entropy. Below this the 16-bin histogram holds under 1.5 values a bin and
/// the estimate is dominated by bin-count bias rather than by the rhythm.
pub const SHANNON_MIN_BEATS: usize = 26;
/// Beats needed for a turning point ratio.
pub const TPR_MIN_BEATS: usize = 8;
/// Equal-width bins the trimmed difference series is binned into.
pub const SHANNON_BINS: usize = 16;
/// Largest and smallest differences dropped before binning, each. Dash trim 8 either side of a 128-beat
/// segment; at the lengths here that would delete most of it, so one either side carries the same role.
pub const SHANNON_TRIM: usize = 1;
/// Turning point ratio of an independent identically distributed series: 2 of every 3 interior points
/// turn. The reference a measured ratio is read against; a smooth trend sits below it.
pub const TPR_RANDOM_EXPECTED: f64 = 2.0 / 3.0;

/// RMSSD divided by the mean R-R — the scatter of successive differences as a fraction of the beat
/// period, so it is comparable across heart rates. `None` under [`RMSSD_RATIO_MIN_BEATS`] or on a
/// non-positive mean.
///
/// The RMSSD is [`HrvReadiness::rmssd_plain`], NOT the artifact-corrected `rmssd`: that one drops any
/// beat-to-beat change over 200 ms as an ectopic, which is the change this index exists to measure.
pub fn rmssd_over_mean_rr(rr_ms: &[u16]) -> Option<f64> {
    if rr_ms.len() < RMSSD_RATIO_MIN_BEATS {
        return None;
    }
    let mean = mean_rr_ms(rr_ms)?;
    HrvReadiness::rmssd_plain(rr_ms).map(|r| r / mean)
}

/// Mean R-R (ms); `None` when empty or non-positive.
pub fn mean_rr_ms(rr_ms: &[u16]) -> Option<f64> {
    if rr_ms.is_empty() {
        return None;
    }
    let m = rr_ms.iter().map(|&v| f64::from(v)).sum::<f64>() / rr_ms.len() as f64;
    (m > 0.0).then_some(m)
}

/// Normalised Shannon entropy of the successive R-R differences, in `0.0..=1.0`.
///
/// The binning, which decides the answer: differences `rr[i+1] - rr[i]`; the [`SHANNON_TRIM`] largest and
/// smallest dropped; the survivors binned into [`SHANNON_BINS`] equal-width bins spanning their own min
/// to max; entropy `-sum p ln p` over the bins, divided by `ln(SHANNON_BINS)` so a flat spread across all
/// bins is 1.0. A trimmed set with no spread is 0.0, which is the right answer for a metronomic series.
/// `None` under [`SHANNON_MIN_BEATS`].
pub fn shannon_entropy_drr(rr_ms: &[u16]) -> Option<f64> {
    if rr_ms.len() < SHANNON_MIN_BEATS {
        return None;
    }
    let mut diffs: Vec<f64> = rr_ms.windows(2).map(|w| f64::from(w[1]) - f64::from(w[0])).collect();
    diffs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let keep = diffs.len().checked_sub(2 * SHANNON_TRIM)?;
    if keep == 0 {
        return None;
    }
    let trimmed = &diffs[SHANNON_TRIM..SHANNON_TRIM + keep];

    let (lo, hi) = (trimmed[0], trimmed[keep - 1]);
    let span = hi - lo;
    let mut counts = [0usize; SHANNON_BINS];
    for &d in trimmed {
        let bin = if span > 0.0 {
            (((d - lo) / span * SHANNON_BINS as f64) as usize).min(SHANNON_BINS - 1)
        } else {
            0
        };
        counts[bin] += 1;
    }
    let n = keep as f64;
    let h: f64 = counts
        .iter()
        .filter(|&&c| c > 0)
        .map(|&c| {
            let p = c as f64 / n;
            -p * p.ln()
        })
        .sum();
    Some(h / (SHANNON_BINS as f64).ln())
}

/// Turning point ratio: interior R-R values that are a strict local maximum or minimum, over the
/// `n - 2` interior values. [`TPR_RANDOM_EXPECTED`] is the value for an independent series; a series
/// carrying a smooth trend turns less often. Equal neighbours are not a turn, which matters on
/// millisecond-quantised input. `None` under [`TPR_MIN_BEATS`].
pub fn turning_point_ratio(rr_ms: &[u16]) -> Option<f64> {
    if rr_ms.len() < TPR_MIN_BEATS {
        return None;
    }
    let turns = rr_ms
        .windows(3)
        .filter(|w| {
            let (a, b, c) = (i32::from(w[0]), i32::from(w[1]), i32::from(w[2]));
            (b > a && b > c) || (b < a && b < c)
        })
        .count();
    Some(turns as f64 / (rr_ms.len() - 2) as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A metronome: every interval identical.
    fn flat(n: usize, rr: u16) -> Vec<u16> {
        vec![rr; n]
    }

    /// Alternating +/- `d` around `rr` — maximally regular scatter, every interior point a turn.
    fn zigzag(n: usize, rr: u16, d: u16) -> Vec<u16> {
        (0..n).map(|i| if i % 2 == 0 { rr - d } else { rr + d }).collect()
    }

    /// `n` differences taking 40 distinct values evenly across +/-20 ms, sign alternating so the series
    /// does not drift.
    fn spread_diffs(n: usize) -> Vec<u16> {
        let mut rr = vec![800u16];
        for i in 0..n {
            let mag = 1 + (i / 2) % 20;
            let d = if i % 2 == 0 { mag as i32 } else { -(mag as i32) };
            rr.push((i32::from(*rr.last().unwrap()) + d) as u16);
        }
        rr
    }

    #[test]
    fn rmssd_ratio_is_scatter_as_a_fraction_of_the_beat_period() {
        // +/-10 ms about 800: every successive |d| is 20, so RMSSD = 20 and the ratio is 20/800.
        let z = zigzag(40, 800, 10);
        assert!((rmssd_over_mean_rr(&z).unwrap() - 20.0 / 800.0).abs() < 1e-12);
        // A metronome has no scatter at all.
        assert_eq!(rmssd_over_mean_rr(&flat(40, 800)), Some(0.0));
        assert_eq!(rmssd_over_mean_rr(&[800, 810]), None);
        // A 200 ms jump must COUNT here; the artifact-corrected RMSSD would have dropped it.
        let mut jump = flat(40, 800);
        jump[20] = 1400;
        assert!(rmssd_over_mean_rr(&jump).unwrap() > 0.1, "a large jump must not be filtered away");
    }

    #[test]
    fn shannon_entropy_is_zero_on_a_metronome_and_high_on_a_spread() {
        assert_eq!(shannon_entropy_drr(&flat(60, 800)), Some(0.0));
        // A two-valued difference series occupies two of sixteen bins: ln 2 / ln 16 = 0.25.
        let z = shannon_entropy_drr(&zigzag(60, 800, 10)).unwrap();
        assert!((z - 0.25).abs() < 1e-3, "got {z}");
        // Differences spread evenly over the bins approach 1.0. Sign alternates each step so the series
        // has no drift; magnitude walks 1..20, giving 40 distinct differences across the same span.
        let spread = spread_diffs(200);
        let s = shannon_entropy_drr(&spread).unwrap();
        assert!(s > 0.9, "an even spread should approach 1.0, got {s}");
        assert!(shannon_entropy_drr(&flat(SHANNON_MIN_BEATS - 1, 800)).is_none());
    }

    #[test]
    fn turning_point_ratio_reads_a_zigzag_high_and_a_ramp_low() {
        assert_eq!(turning_point_ratio(&zigzag(40, 800, 10)), Some(1.0));
        // A monotone ramp never turns.
        let ramp: Vec<u16> = (0..40).map(|i| 600 + i as u16).collect();
        assert_eq!(turning_point_ratio(&ramp), Some(0.0));
        // A metronome has no strict turns either — equal neighbours are not a turn.
        assert_eq!(turning_point_ratio(&flat(40, 800)), Some(0.0));
        assert_eq!(turning_point_ratio(&[800, 810, 805]), None);
    }

    #[test]
    fn degenerate_inputs_return_none_not_a_number() {
        type Index = fn(&[u16]) -> Option<f64>;
        let all: [Index; 3] = [rmssd_over_mean_rr, shannon_entropy_drr, turning_point_ratio];
        for f in all {
            assert_eq!(f(&[]), None);
            assert_eq!(f(&[800]), None);
        }
        assert_eq!(mean_rr_ms(&[]), None);
        assert_eq!(mean_rr_ms(&[0u16, 0]), None);
        // A long all-zero series is physiologically impossible but must still not divide by zero.
        assert_eq!(rmssd_over_mean_rr(&flat(60, 0)), None);
    }
}
