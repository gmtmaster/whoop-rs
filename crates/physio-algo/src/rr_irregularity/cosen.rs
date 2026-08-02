//! Sample entropy and COSEn, the Coefficient of Sample Entropy. Lake & Moorman, "Accurate estimation of
//! entropy in very short physiological time series: the problem of atrial fibrillation detection in
//! implanted ventricular devices", Am J Physiol Heart Circ Physiol 300:H319-H325, 2011.
//!
//! COSEn is sample entropy plus two corrections, and it is the corrections that make it work on a
//! 12-beat record where plain sample entropy does not:
//!
//! ```text
//! COSEn = SampEn(m, r, N) + ln(2r) - ln(mean R-R)
//! ```
//!
//! `+ ln(2r)` turns a conditional PROBABILITY into a probability DENSITY (for `m = 1` the matched region
//! in the added dimension is `2r` wide), which is what removes the estimate's dependence on the tolerance.
//! `- ln(mean R-R)` then normalises the density by the beat period, making it a coefficient rather than
//! an absolute — the same move a coefficient of variation makes. Implementing only `-ln(A/B)` gives plain
//! sample entropy, which is a different and much length-sensitive number; the two are separate functions
//! here so the difference is visible.
//!
//! `r` is ABSOLUTE in milliseconds, not a multiple of the series SD: over 12 beats an SD estimate is too
//! unstable to set a tolerance from.

use crate::rr_irregularity::indices::mean_rr_ms;

/// Embedding dimension. Lake & Moorman use 1 for very short R-R records.
pub const COSEN_M: usize = 1;
/// Matching tolerance (ms), absolute.
pub const COSEN_R_MS: f64 = 30.0;
/// Beats needed. Lake & Moorman's short record; under it neither count has enough template pairs.
pub const COSEN_MIN_BEATS: usize = 12;
/// Longest record either function will answer on. COSEn is a SHORT-record statistic by construction, so
/// beyond this the number would no longer be the published index; the pair count also grows as N^2, and a
/// whole night would cost billions of comparisons to produce it. Refused rather than approximated.
pub const SAMPEN_MAX_BEATS: usize = 1_000;

/// Sample entropy `-ln(A/B)` (Richman & Moorman) over an R-R series (ms), with tolerance `r_ms` absolute
/// and embedding `m`. `B` counts template pairs matching over `m` samples, `A` over `m + 1`; both range
/// over the same `N - m` template positions, so the ratio is a conditional probability.
///
/// `None` when the series is shorter than `m + 2` or longer than [`SAMPEN_MAX_BEATS`], when `r_ms` is not
/// positive and finite, or when either count is zero — no match at all is an absence of evidence, not an
/// infinite entropy.
pub fn sample_entropy(rr_ms: &[u16], m: usize, r_ms: f64) -> Option<f64> {
    if m == 0 || !r_ms.is_finite() || r_ms <= 0.0 || rr_ms.len() < m + 2 || rr_ms.len() > SAMPEN_MAX_BEATS
    {
        return None;
    }
    let templates = rr_ms.len() - m;
    let matches = |len: usize, i: usize, j: usize| {
        (0..len).all(|k| (f64::from(rr_ms[i + k]) - f64::from(rr_ms[j + k])).abs() <= r_ms)
    };
    let (mut b, mut a) = (0usize, 0usize);
    for i in 0..templates {
        for j in (i + 1)..templates {
            if matches(m, i, j) {
                b += 1;
                if matches(m + 1, i, j) {
                    a += 1;
                }
            }
        }
    }
    (a > 0 && b > 0).then(|| -((a as f64) / (b as f64)).ln())
}

/// COSEn over an R-R series (ms) at [`COSEN_M`] / [`COSEN_R_MS`]. `None` under [`COSEN_MIN_BEATS`], on a
/// non-positive mean R-R, or when sample entropy itself is undefined.
///
/// Higher is more irregular. It is reported as a bare index: this project has no labelled rhythm corpus
/// to set a decision threshold from, so none is defined here.
pub fn cosen(rr_ms: &[u16]) -> Option<f64> {
    cosen_with(rr_ms, COSEN_M, COSEN_R_MS)
}

/// [`cosen`] with the embedding and tolerance supplied, for a sweep over them.
pub fn cosen_with(rr_ms: &[u16], m: usize, r_ms: f64) -> Option<f64> {
    if rr_ms.len() < COSEN_MIN_BEATS {
        return None;
    }
    let mean = mean_rr_ms(rr_ms)?;
    let sampen = sample_entropy(rr_ms, m, r_ms)?;
    Some(sampen + (2.0 * r_ms).ln() - mean.ln())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `n` R-R values uniform over `[lo, lo + span)` ms from a seeded LCG — deterministic, so a failure
    /// is always reproducible.
    fn scatter(n: usize, lo: u16, span: u64, seed: u64) -> Vec<u16> {
        let mut x = seed;
        (0..n)
            .map(|_| {
                x = x.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
                lo + ((x >> 33) % span) as u16
            })
            .collect()
    }

    #[test]
    fn sample_entropy_matches_a_hand_counted_series() {
        // m = 1, r = 5. Values 100, 200, 100, 200, 100: templates 0..4 (N - m = 4).
        // 1-matches: (0,2) (1,3) — 2. Of those, 2-matches: (0,2) has [100,200] vs [100,200] -> yes;
        // (1,3) has [200,100] vs [200,100] -> yes. So A = B = 2 and SampEn = -ln(1) = 0.
        let s = &[100u16, 200, 100, 200, 100];
        assert_eq!(sample_entropy(s, 1, 5.0), Some(0.0));
        // Six values, so templates 0..4 hold 100, 200, 100, 200, 100. 1-matches: the three pairs of 100
        // plus (1,3) -> B = 4. Extending by one kills (0,4) and (2,4), whose successors are 100 and 900
        // -> A = 2. SampEn = -ln(2/4) = ln 2.
        let broken = &[100u16, 200, 100, 200, 100, 900];
        let v = sample_entropy(broken, 1, 5.0).unwrap();
        assert!((v - 2.0f64.ln()).abs() < 1e-12, "got {v}");
    }

    #[test]
    fn sample_entropy_refuses_rather_than_returning_infinity() {
        // No template pair matches at all: A and B are both zero, which is not zero entropy.
        assert_eq!(sample_entropy(&[100u16, 500, 900, 1400], 1, 5.0), None);
        assert_eq!(sample_entropy(&[800u16, 800], 1, 5.0), None); // shorter than m + 2
        assert_eq!(sample_entropy(&[800u16; 10], 1, 0.0), None); // non-positive tolerance
        assert_eq!(sample_entropy(&[800u16; 10], 0, 5.0), None); // zero embedding
        assert_eq!(sample_entropy(&[800u16; 10], 1, f64::NAN), None);
        // A whole night is not a short record: refused at the cap, answered just under it.
        assert_eq!(sample_entropy(&[800u16; SAMPEN_MAX_BEATS + 1], 1, 5.0), None);
        assert_eq!(cosen(&[800u16; SAMPEN_MAX_BEATS + 1]), None);
        assert!(sample_entropy(&[800u16; SAMPEN_MAX_BEATS], 1, 5.0).is_some());
    }

    #[test]
    fn cosen_is_sample_entropy_plus_the_two_stated_corrections() {
        // The identity is the whole point of the index; a plain-SampEn implementation would fail here.
        let rr: Vec<u16> = (0..40).map(|i| 800 + (i % 5) as u16 * 6).collect();
        let sampen = sample_entropy(&rr, COSEN_M, COSEN_R_MS).unwrap();
        let mean = mean_rr_ms(&rr).unwrap();
        let expected = sampen + (2.0 * COSEN_R_MS).ln() - mean.ln();
        assert!((cosen(&rr).unwrap() - expected).abs() < 1e-12);
        // The corrections are not cosmetic: they move it by ln(60) - ln(~812) = -2.6.
        assert!((cosen(&rr).unwrap() - sampen + 2.6).abs() < 0.1, "correction size");
    }

    #[test]
    fn cosen_separates_a_metronome_from_a_scattered_series() {
        let flat: Vec<u16> = vec![800; 60];
        // A perfectly regular series matches everywhere: SampEn 0, so COSEn is just the correction.
        let regular = cosen(&flat).unwrap();
        // A wide scatter breaks the m+1 matches and pushes entropy, hence COSEn, up.
        let irregular = cosen(&scatter(120, 600, 400, 12345)).unwrap();
        assert!(irregular > regular, "regular {regular}, scattered {irregular}");
    }

    #[test]
    fn cosen_refuses_short_and_degenerate_input() {
        assert_eq!(cosen(&[800u16; COSEN_MIN_BEATS - 1]), None);
        assert_eq!(cosen(&[]), None);
        assert_eq!(cosen(&[0u16; 20]), None); // mean R-R of zero
        // Every beat further apart than the tolerance: no match, so no entropy rather than a fake one.
        let sparse: Vec<u16> = (0..20).map(|i| 400 + i as u16 * 70).collect();
        assert_eq!(cosen(&sparse), None);
    }
}
