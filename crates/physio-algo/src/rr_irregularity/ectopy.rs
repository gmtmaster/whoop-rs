//! Telling an ectopy burst apart from a sustained irregular rhythm.
//!
//! Premature beats — atrial or ventricular — are common in healthy people and produce large R-R scatter,
//! so every scatter statistic in this module ranks them ABOVE fibrillation. Measured on this project's
//! corpus at 32 beats a segment: RMSSD over mean R-R reads 0.457 on bigeminal ectopy against 0.289 on
//! fibrillation, and SD1 234 ms against 131 ms. A screen built on scatter alone would fire hardest on the
//! people who least need it.
//!
//! What separates them is not how big the scatter is but how it is SHAPED. A premature beat is an
//! excursion from a rhythm that is otherwise steady, and the rhythm returns to where it was; fibrillation
//! has no level to return to. Three views of that, each measurable on a short segment:
//!
//! - [`EctopyProfile::origin_fraction`] — the steady background an excursion departs from.
//! - [`EctopyProfile::alternation_fraction`] — matched short-then-long pairs, which is what bigeminy and
//!   trigeminy are made of and what leaves them with no steady background at all.
//! - [`EctopyProfile::residual_sample_entropy`] — the entropy left once the premature beats are removed.
//!   Ectopy collapses to sinus; a sustained irregular rhythm does not.
//!
//! The removal reuses `hrv`'s Malik rejection, which the indices themselves deliberately never apply.
//! Here it is the point: applying it and re-measuring is the discriminator.

use crate::hrv::{HrvReadiness, ECTOPIC_THRESHOLD};
use crate::rr_irregularity::cosen::{cosen, sample_entropy, COSEN_M, COSEN_R_MS};

/// Beat-to-beat change treated as no change at all — the same 50 ms boundary pNN50 is defined on.
pub const ORIGIN_MS: f64 = 50.0;
/// Beat-to-beat change large enough to be an excursion rather than sinus sway. Sinus segments in this
/// project's corpus carry a median successive difference near 12 ms; fibrillation near 108 ms.
pub const EXCURSION_MS: f64 = 100.0;
/// Two opposite excursions are a matched pair when the smaller is at least this share of the larger, so a
/// short beat and the long beat compensating for it count together.
pub const PAIR_MATCH: f64 = 0.5;
/// Differences needed before the shape fractions mean anything.
pub const PROFILE_MIN_BEATS: usize = 12;

/// The shape of one segment's scatter. Every field is a share in `0.0..=1.0` except the entropies, and
/// each is reported on its own: a fused number that hides which view fired cannot be argued with.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EctopyProfile {
    /// Successive differences within [`ORIGIN_MS`] of zero, over all differences. The steady background:
    /// high when a few beats depart from an otherwise even rhythm, low when no two beats agree.
    pub origin_fraction: f64,
    /// Excursions over [`EXCURSION_MS`] that belong to an opposite-signed neighbour of matching size, over
    /// all excursions. Near 1.0 for bigeminy, whose every beat is one half of such a pair.
    pub alternation_fraction: f64,
    /// Beats Malik rejection removes, over input beats.
    pub ectopic_fraction: f64,
    /// Sample entropy of the series with those beats removed. `None` when too few survive, or when the
    /// survivors match nowhere.
    pub residual_sample_entropy: Option<f64>,
    /// Sample entropy of the series as given, for the same `m` and `r`, so the drop is visible.
    pub sample_entropy: Option<f64>,
    /// COSEn of the series with the premature beats removed. The residual view of the index the screen
    /// leans on, so the two are read on the same scale.
    pub residual_cosen: Option<f64>,
}

/// Profile one R-R series (ms) in beat order. `None` under [`PROFILE_MIN_BEATS`].
///
/// The two shape fractions read the successive-difference series; the two entropies read the beats before
/// and after Malik rejection. Removal splices the survivors together, which is deliberate: splicing out a
/// premature beat and the pause after it is exactly what should leave a plain sinus series behind.
pub fn profile(rr_ms: &[u16]) -> Option<EctopyProfile> {
    if rr_ms.len() < PROFILE_MIN_BEATS {
        return None;
    }
    let d: Vec<f64> = rr_ms.windows(2).map(|w| f64::from(w[1]) - f64::from(w[0])).collect();
    let origin = d.iter().filter(|v| v.abs() <= ORIGIN_MS).count() as f64 / d.len() as f64;

    let big: Vec<usize> = (0..d.len()).filter(|&i| d[i].abs() > EXCURSION_MS).collect();
    let paired = big
        .iter()
        .filter(|&&i| {
            let matched = |j: usize| {
                d[i] * d[j] < 0.0 && d[i].abs().min(d[j].abs()) >= PAIR_MATCH * d[i].abs().max(d[j].abs())
            };
            (i > 0 && d[i - 1].abs() > EXCURSION_MS && matched(i - 1))
                || (i + 1 < d.len() && d[i + 1].abs() > EXCURSION_MS && matched(i + 1))
        })
        .count();
    let alternation = if big.is_empty() { 0.0 } else { paired as f64 / big.len() as f64 };

    let clean = HrvReadiness::clean_rr(rr_ms);
    Some(EctopyProfile {
        origin_fraction: origin,
        alternation_fraction: alternation,
        ectopic_fraction: 1.0 - clean.len() as f64 / rr_ms.len() as f64,
        residual_sample_entropy: sample_entropy(&clean, COSEN_M, COSEN_R_MS),
        sample_entropy: sample_entropy(rr_ms, COSEN_M, COSEN_R_MS),
        residual_cosen: cosen(&clean),
    })
}

/// The Malik tolerance the removal uses, re-exported so a caller reporting the profile can say what
/// "ectopic" meant here without reaching into `hrv`.
pub const MALIK_TOLERANCE: f64 = ECTOPIC_THRESHOLD;

#[cfg(test)]
mod tests {
    use super::*;

    /// A steady rhythm at `rr` ms with a premature beat every `period` beats, each followed by a
    /// compensatory pause that returns the series to `rr`.
    fn ectopic(n: usize, rr: u16, period: usize, short_ms: u16) -> Vec<u16> {
        (0..n)
            .map(|i| match i % period {
                0 if period < n => rr - short_ms,
                1 => rr + short_ms,
                _ => rr,
            })
            .collect()
    }

    /// Every interval drawn independently over a wide band — no level to return to.
    fn scattered(n: usize, lo: u16, span: u64, seed: u64) -> Vec<u16> {
        let mut x = seed;
        (0..n)
            .map(|_| {
                x = x.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
                lo + ((x >> 33) % span) as u16
            })
            .collect()
    }

    #[test]
    fn isolated_ectopy_keeps_a_steady_background_and_a_scattered_rhythm_does_not() {
        let e = profile(&ectopic(64, 800, 8, 250)).unwrap();
        let s = profile(&scattered(64, 500, 700, 99)).unwrap();
        assert!(e.origin_fraction > 0.6, "ectopy must keep a steady background, got {}", e.origin_fraction);
        // A wide independent draw still lands two beats within 50 ms of each other now and then, so the
        // claim is the gap, not a floor near zero.
        assert!(
            e.origin_fraction > s.origin_fraction + 0.3,
            "ectopy {} against scatter {}",
            e.origin_fraction,
            s.origin_fraction
        );
    }

    #[test]
    fn bigeminy_is_matched_alternation_and_scatter_is_not() {
        // Every other beat premature: the pure alternating case, which has no steady background at all.
        let b = profile(&ectopic(64, 800, 2, 250)).unwrap();
        assert!(b.origin_fraction < 0.1, "bigeminy has no steady background: {}", b.origin_fraction);
        assert_eq!(b.alternation_fraction, 1.0, "every excursion is half a matched pair");
        let s = profile(&scattered(64, 500, 700, 7)).unwrap();
        assert!(s.alternation_fraction < b.alternation_fraction, "scatter {s:?} vs bigeminy {b:?}");
    }

    #[test]
    fn removing_the_premature_beats_collapses_ectopy_and_leaves_scatter_alone() {
        let e = profile(&ectopic(64, 800, 8, 250)).unwrap();
        assert!(e.ectopic_fraction > 0.0, "the premature beats must be found");
        assert_eq!(e.residual_sample_entropy, Some(0.0), "what is left is a metronome");
        let s = profile(&scattered(64, 500, 700, 31)).unwrap();
        assert!(
            s.residual_sample_entropy.unwrap() > 1.0,
            "scatter survives its own cleaning: {:?}",
            s.residual_sample_entropy
        );
    }

    #[test]
    fn short_input_returns_none_and_a_metronome_returns_zeros() {
        assert_eq!(profile(&[]), None);
        assert_eq!(profile(&[800u16; PROFILE_MIN_BEATS - 1]), None);
        let flat = profile(&[800u16; 40]).unwrap();
        assert_eq!((flat.origin_fraction, flat.alternation_fraction), (1.0, 0.0));
        assert_eq!(flat.ectopic_fraction, 0.0);
    }
}
