//! Agreement between two QRS detectors' peak sets — the raw material of bSQI.
//!
//! Matching is one-to-one. A many-to-one match would let one detector's burst of spurious peaks each
//! claim the same real beat and report near-perfect agreement on noise, which is precisely the failure
//! this index exists to catch.

use super::usable_rate;

/// Default matching tolerance (ms). Bounded from below by fiducial dispersion: the two detectors place
/// the R peak by different rules (largest bandpassed excursion vs. the derivative's zero crossing) and can
/// differ by most of a QRS, which is at most 140 ms wide. Bounded from above by beat separation: at the
/// 220 bpm plausible ceiling the R-R is 273 ms, so a window under half of that (136 ms) cannot let a
/// detection match the neighbouring beat. 100 ms sits inside both bounds.
pub const DEFAULT_MATCH_WINDOW_MS: f64 = 100.0;

/// One-to-one agreement between two peak sets.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Agreement {
    /// Beats matched within the window (the true positives of either detector against the other).
    pub matched: usize,
    /// Peaks only the first detector found.
    pub only_a: usize,
    /// Peaks only the second detector found.
    pub only_b: usize,
    /// `2M / (2M + only_a + only_b)`. Symmetric in the two arguments. This is the raw bSQI value, and it
    /// must be read against `chance_f1` — on its own it rises with detection density alone.
    pub f1: f64,
    /// `M / (M + only_a + only_b)`. Same ordering as `f1`, harsher on disagreement.
    pub jaccard: f64,
    /// The `f1` two INDEPENDENT detectors would score at these densities, from coincidence alone.
    pub chance_f1: f64,
    /// `f1 - chance_f1`: how much of the agreement is not explained by density. Never negative.
    pub excess: f64,
}

/// Agreement between two ascending peak-index sets over a `span_samples`-long record at `fs_hz`, matching
/// within `window_ms`.
///
/// **Two empty sets score 0.0, not 1.0.** Vacuous agreement is the state a flat line, a dead channel and a
/// wrong packet layout all produce; scoring it as perfect would make the index pass exactly the inputs it
/// is there to reject. An unusable rate, a non-positive window or a non-positive span also scores 0.0.
///
/// `span_samples` is only used for `chance_f1`, and it is not optional: two detectors both firing every
/// 250 ms match most of each other's peaks by coincidence, which is how noise reaches an `f1` near 0.8.
pub fn beat_agreement(
    a: &[usize],
    b: &[usize],
    span_samples: usize,
    fs_hz: f64,
    window_ms: f64,
) -> Agreement {
    let empty = Agreement {
        matched: 0,
        only_a: a.len(),
        only_b: b.len(),
        f1: 0.0,
        jaccard: 0.0,
        chance_f1: 0.0,
        excess: 0.0,
    };
    if !usable_rate(fs_hz) || !window_ms.is_finite() || window_ms <= 0.0 || span_samples == 0 {
        return empty;
    }
    let tolerance = (window_ms / 1000.0 * fs_hz).round() as usize;
    if a.is_empty() || b.is_empty() {
        return empty;
    }

    let matched = matched_pairs(a, b, tolerance).len();
    let only_a = a.len() - matched;
    let only_b = b.len() - matched;
    let f1 = f1_of(matched as f64, a.len() as f64, b.len() as f64);
    let union = matched + only_a + only_b;
    let jaccard = if union == 0 {
        0.0
    } else {
        matched as f64 / union as f64
    };
    let chance_f1 = chance_f1(a.len(), b.len(), span_samples, tolerance);
    Agreement {
        matched,
        only_a,
        only_b,
        f1,
        jaccard,
        chance_f1,
        excess: (f1 - chance_f1).max(0.0),
    }
}

/// The one-to-one pairing itself, as `(index into a, index into b)`.
///
/// Two-pointer over two ascending lists: consume a pair when it is inside `tolerance` samples, otherwise
/// drop whichever side is behind. Each index is consumed once, so the match is one-to-one by
/// construction — a many-to-one match would let one detector's burst of spurious peaks each claim the
/// same real beat. The single source of the matching rule; [`beat_agreement`] counts what this returns.
pub fn matched_pairs(a: &[usize], b: &[usize], tolerance: usize) -> Vec<(usize, usize)> {
    let (mut i, mut j) = (0usize, 0usize);
    let mut out = Vec::new();
    while i < a.len() && j < b.len() {
        if a[i].abs_diff(b[j]) <= tolerance {
            out.push((i, j));
            i += 1;
            j += 1;
        } else if a[i] < b[j] {
            i += 1;
        } else {
            j += 1;
        }
    }
    out
}

/// `2M / (|A| + |B|)` — the F1 identity, written for real-valued expected match counts too.
fn f1_of(matched: f64, n_a: f64, n_b: f64) -> f64 {
    let denom = n_a + n_b;
    if denom <= 0.0 {
        0.0
    } else {
        2.0 * matched / denom
    }
}

/// F1 expected from coincidence alone at these densities.
///
/// Each peak of one set covers `2 * tolerance + 1` samples of the record, and a refractory keeps peaks
/// apart, so the covered fraction is `n * (2t + 1) / span`. A peak of the other set falls in it by
/// accident at that rate. The one-to-one match caps the count at `min(|A|, |B|)`, and the smaller of the
/// two directional expectations is taken so neither side's density alone sets the floor.
fn chance_f1(n_a: usize, n_b: usize, span_samples: usize, tolerance: usize) -> f64 {
    let (na, nb) = (n_a as f64, n_b as f64);
    let covered = (2 * tolerance + 1) as f64 / span_samples as f64;
    let coverage_a = (na * covered).min(1.0);
    let coverage_b = (nb * covered).min(1.0);
    let expected = (nb * coverage_a).min(na * coverage_b).min(na).min(nb);
    f1_of(expected, na, nb)
}

#[cfg(test)]
mod tests {
    use super::*;

    const FS: f64 = 200.0;
    /// 30 s at 200 Hz - the capture length the detectors are aimed at.
    const SPAN: usize = 6000;

    fn agree(a: &[usize], b: &[usize]) -> Agreement {
        beat_agreement(a, b, SPAN, FS, DEFAULT_MATCH_WINDOW_MS)
    }

    #[test]
    fn identical_sets_agree_completely() {
        let a = [100usize, 300, 500, 700];
        let g = agree(&a, &a);
        assert_eq!((g.matched, g.only_a, g.only_b), (4, 0, 0));
        assert!((g.f1 - 1.0).abs() < 1e-12 && (g.jaccard - 1.0).abs() < 1e-12);
    }

    #[test]
    fn a_shift_inside_the_window_still_matches_and_outside_does_not() {
        let a = [100usize, 300, 500];
        // 100 ms at 200 Hz = 20 samples.
        let inside: Vec<usize> = a.iter().map(|v| v + 19).collect();
        let outside: Vec<usize> = a.iter().map(|v| v + 21).collect();
        assert_eq!(agree(&a, &inside).matched, 3);
        assert_eq!(agree(&a, &outside).matched, 0);
    }

    #[test]
    fn matching_is_one_to_one_so_a_burst_cannot_inflate_the_score() {
        // Six detections crowded around one real beat may claim it once, not six times.
        let a = [500usize];
        let burst = [495usize, 496, 497, 498, 499, 501];
        let g = agree(&a, &burst);
        assert_eq!((g.matched, g.only_a, g.only_b), (1, 0, 5));
        assert!(g.f1 < 0.3, "f1 {} should be poor", g.f1);
    }

    #[test]
    fn empty_sets_score_zero_not_one() {
        assert_eq!(agree(&[], &[]).f1, 0.0);
        assert_eq!(agree(&[100], &[]).f1, 0.0);
        assert_eq!(agree(&[], &[100]).only_b, 1);
    }

    #[test]
    fn is_symmetric_and_rejects_a_bad_rate_window_or_span() {
        let a = [100usize, 300, 500, 900];
        let b = [105usize, 500, 505, 1200];
        let ab = agree(&a, &b);
        let ba = agree(&b, &a);
        assert_eq!(ab.matched, ba.matched);
        assert!((ab.f1 - ba.f1).abs() < 1e-12);
        assert!((ab.chance_f1 - ba.chance_f1).abs() < 1e-12);
        assert_eq!(
            beat_agreement(&a, &b, SPAN, 10.0, DEFAULT_MATCH_WINDOW_MS).f1,
            0.0
        );
        assert_eq!(beat_agreement(&a, &b, SPAN, FS, 0.0).f1, 0.0);
        assert_eq!(beat_agreement(&a, &b, SPAN, FS, f64::NAN).f1, 0.0);
        assert_eq!(
            beat_agreement(&a, &b, 0, FS, DEFAULT_MATCH_WINDOW_MS).f1,
            0.0
        );
    }

    #[test]
    fn half_the_beats_missed_scores_two_thirds() {
        // 4 matched, 0 extra on a, 4 extra on b -> 8 / (8 + 0 + 4) = 0.667.
        let a: Vec<usize> = (0..4).map(|k| 200 + k * 200).collect();
        let b: Vec<usize> = (0..8).map(|k| 200 + k * 200).collect();
        let g = agree(&a, &b);
        assert_eq!((g.matched, g.only_a, g.only_b), (4, 0, 4));
        assert!((g.f1 - 2.0 / 3.0).abs() < 1e-12);
        assert!((g.jaccard - 0.5).abs() < 1e-12);
    }

    #[test]
    fn the_chance_floor_rises_with_density_and_leaves_a_sparse_match_alone() {
        // 30 beats in 30 s: a peak covers 41 of 6000 samples, so coincidence explains almost nothing.
        let sparse: Vec<usize> = (0..30).map(|k| 100 + k * 200).collect();
        let g = agree(&sparse, &sparse);
        assert!(
            g.chance_f1 < 0.25,
            "sparse chance floor {} too high",
            g.chance_f1
        );
        assert!(
            g.excess > 0.75,
            "a real match must clear the floor by a lot"
        );

        // 120 peaks in the same 30 s is 240 bpm - past physiology, and now over half of a perfect score
        // is coincidence. An index that did not say so would report noise as good agreement.
        let dense: Vec<usize> = (0..120).map(|k| 25 + k * 50).collect();
        let d = agree(&dense, &dense);
        assert!(
            d.f1 == 1.0 && d.chance_f1 > 0.7,
            "dense: f1 {} chance {}",
            d.f1,
            d.chance_f1
        );
        assert!(
            d.excess < g.excess,
            "the dense case must not look better than the sparse one"
        );
    }

    #[test]
    fn excess_is_clamped_at_zero_and_chance_is_capped_at_one() {
        // Saturating density: every sample is within tolerance of some peak, so nothing is informative.
        let all: Vec<usize> = (0..300).map(|k| k * 20).collect();
        let g = agree(&all, &all);
        assert!(g.chance_f1 <= 1.0 && g.excess >= 0.0);
        assert!(
            g.excess < 0.05,
            "saturated density must leave no excess, got {}",
            g.excess
        );
    }
}
