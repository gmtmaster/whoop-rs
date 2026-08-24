//! Input quality for the irregularity indices: what the range filter removed, what Malik ectopic
//! rejection WOULD remove, and the three ways an R-R series can be a duplicate of itself.
//!
//! Every index in this module measures beat-to-beat scatter, so a repeated or rescaled beat reads as
//! irregularity the heart never produced. The detectors here are reused from `hrv` where they already
//! exist ([`rr_coverage`], [`duplicate_beat_count`], [`clean_counts`]); only the rescaled-copy test is
//! new, because an exact-repeat count cannot see a copy stored at a different value.

use std::collections::HashMap;

use crate::hrv::{HrvReadiness, clean_counts, duplicate_beat_count, rr_coverage};

/// Beat-time over elapsed time. Above 1.0 is physically impossible. The slack absorbs whole-second
/// stamping, which lets a true 1.0 read high on a short window.
pub const MAX_COVERAGE: f64 = 1.15;
/// Exact `(second, value)` repeats, as a fraction of input beats.
pub const MAX_DUPLICATE_FRACTION: f64 = 0.02;
/// The range filter may drop at most this fraction before a reading is refused.
pub const MAX_RANGE_REJECTED_FRACTION: f64 = 0.20;
/// A second copy of each beat, rescaled by this ratio, is the shape one storage path in this project
/// produced: `round(v * 1000/1024)`, 2.34 % short of the original.
pub const RESCALE_RATIO: f64 = 1000.0 / 1024.0;
/// A rescaled copy carries the same second as its original or up to this many seconds later.
pub const RESCALE_LAG_S: u32 = 1;
/// Rescaled-copy fraction at or above which a series is refused. Set from the measured gap between a
/// clean synthetic series and a known-duplicated real one, not from a published figure.
pub const MAX_RESCALED_FRACTION: f64 = 0.20;
/// Below this many beats the quality fractions are too coarse to act on.
pub const MIN_QUALITY_BEATS: usize = 8;

/// What one R-R series looks like before any index is computed. `ectopic_rejected_fraction` is MEASURED
/// and deliberately NOT applied: Malik rejection drops any beat over 20 % from its local median, which is
/// exactly the beat an irregularity index exists to see.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RrQuality {
    pub n_input: u32,
    /// Survivors of the physiological range filter — the beats the indices are computed over.
    pub n_ranged: u32,
    pub range_rejected_fraction: f64,
    pub ectopic_rejected_fraction: f64,
    /// Exact `(second, value)` repeats over input beats.
    pub duplicate_fraction: f64,
    /// Beats that are `round(other * RESCALE_RATIO)` of another beat in the same second or the next.
    pub rescaled_fraction: f64,
    /// Total beat-time over the wall-clock span of the same beats.
    pub coverage: f64,
}

/// Measure one timestamped R-R series. Never panics; an empty input reports zeros.
pub fn measure(beats: &[(u32, u16)]) -> RrQuality {
    let values: Vec<u16> = beats.iter().map(|&(_, v)| v).collect();
    let counts = clean_counts(&values);
    let n = beats.len();
    let ts: Vec<i64> = beats.iter().map(|&(t, _)| i64::from(t)).collect();
    let ms: Vec<f64> = values.iter().map(|&v| f64::from(v)).collect();
    let frac = |num: u32| {
        if n == 0 {
            0.0
        } else {
            f64::from(num) / n as f64
        }
    };
    RrQuality {
        n_input: counts.n_input,
        n_ranged: counts.n_ranged,
        range_rejected_fraction: frac(counts.n_input - counts.n_ranged),
        ectopic_rejected_fraction: frac(counts.n_ranged - counts.n_clean),
        duplicate_fraction: frac(duplicate_beat_count(&ts, &ms)),
        rescaled_fraction: rescaled_copy_fraction(beats),
        coverage: rr_coverage(&ts, &ms),
    }
}

/// Fraction of beats that are `round(other * RESCALE_RATIO)` of a DIFFERENT beat in the same second or
/// up to [`RESCALE_LAG_S`] seconds earlier. An exact-repeat count is blind to this: a rescaled copy is a
/// different value, so it is a different row and a distinct beat everywhere downstream.
pub fn rescaled_copy_fraction(beats: &[(u32, u16)]) -> f64 {
    if beats.is_empty() {
        return 0.0;
    }
    let mut by_sec: HashMap<u32, Vec<(usize, u16)>> = HashMap::new();
    for (i, &(t, v)) in beats.iter().enumerate() {
        by_sec.entry(t).or_default().push((i, v));
    }
    let mut copies = 0usize;
    for (i, &(t, v)) in beats.iter().enumerate() {
        let mut hit = false;
        for lag in 0..=RESCALE_LAG_S {
            let Some(sources) = t.checked_sub(lag).and_then(|s| by_sec.get(&s)) else {
                continue;
            };
            hit |= sources
                .iter()
                .any(|&(j, src)| j != i && src != v && rescaled(src) == i64::from(v));
            if hit {
                break;
            }
        }
        copies += usize::from(hit);
    }
    copies as f64 / beats.len() as f64
}

/// The value a mis-scaled second copy of `v` would carry.
fn rescaled(v: u16) -> i64 {
    (f64::from(v) * RESCALE_RATIO).round() as i64
}

/// Beats that survive the physiological range filter, in input order. The one cleaning step the indices
/// apply; see [`RrQuality::ectopic_rejected_fraction`] for the one they deliberately do not.
pub fn ranged(beats: &[(u32, u16)]) -> Vec<u16> {
    HrvReadiness::range_filter(&beats.iter().map(|&(_, v)| v).collect::<Vec<u16>>())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One beat a second at `rr` ms, so beat-time tracks the clock.
    fn steady(n: u32, rr: u16) -> Vec<(u32, u16)> {
        (0..n).map(|i| (i, rr)).collect()
    }

    #[test]
    fn a_clean_steady_series_is_clean_on_every_axis() {
        let q = measure(&steady(60, 1000));
        assert_eq!((q.n_input, q.n_ranged), (60, 60));
        assert_eq!(q.range_rejected_fraction, 0.0);
        assert_eq!(q.duplicate_fraction, 0.0);
        assert_eq!(q.rescaled_fraction, 0.0);
        assert!(
            (q.coverage - 60.0 / 59.0).abs() < 1e-9,
            "coverage {}",
            q.coverage
        );
    }

    #[test]
    fn exact_repeats_and_impossible_coverage_are_both_seen() {
        // Every beat stored twice on its own second: half the rows are exact repeats and beat-time is 2x.
        let dup: Vec<(u32, u16)> = (0..30u32).flat_map(|i| [(i, 1000u16), (i, 1000)]).collect();
        let q = measure(&dup);
        assert!((q.duplicate_fraction - 0.5).abs() < 1e-9, "{q:?}");
        assert!(q.coverage > MAX_COVERAGE, "{q:?}");
    }

    #[test]
    fn a_rescaled_copy_is_invisible_to_the_exact_repeat_count() {
        // The real shape: each beat plus round(v * 1000/1024) a second later. Distinct values, so the
        // exact-repeat count sees nothing at all.
        let mut beats: Vec<(u32, u16)> = Vec::new();
        for i in 0..40u32 {
            let v = 800u16 + (i % 7) as u16;
            beats.push((i, v));
            beats.push((i + 1, rescaled(v) as u16));
        }
        let q = measure(&beats);
        assert_eq!(
            q.duplicate_fraction, 0.0,
            "exact repeats cannot see a rescaled copy"
        );
        assert!(
            q.rescaled_fraction > MAX_RESCALED_FRACTION,
            "rescaled {}",
            q.rescaled_fraction
        );
    }

    #[test]
    fn ectopic_rejection_is_measured_and_never_applied() {
        // One beat 40 % off its neighbours: Malik would drop it, the range filter keeps it.
        let mut beats = steady(30, 800);
        beats[15].1 = 1200;
        let q = measure(&beats);
        assert_eq!(
            q.n_ranged, 30,
            "the range filter keeps a physiologically possible beat"
        );
        assert!(
            q.ectopic_rejected_fraction > 0.0,
            "and the ectopic count still reports it"
        );
        assert!(ranged(&beats).contains(&1200));
    }

    #[test]
    fn degenerate_inputs_never_panic() {
        let q = measure(&[]);
        assert_eq!((q.n_input, q.coverage, q.rescaled_fraction), (0, 0.0, 0.0));
        assert_eq!(rescaled_copy_fraction(&[(0, 800)]), 0.0);
        assert_eq!(measure(&[(0, 0)]).n_ranged, 0);
        // A zero-second series cannot underflow the lag lookup; only the copy (781) is the copy.
        assert_eq!(rescaled_copy_fraction(&[(0, 800), (0, 781)]), 0.5);
    }
}
