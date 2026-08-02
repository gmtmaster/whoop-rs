//! R-R interval irregularity indices — the published beat-to-beat scatter statistics, computed from R-R
//! alone. No ECG morphology is involved, so this path works on the optical channel of every generation
//! rather than only on a strap with an electrode.
//!
//! **This is a wellness screen. It returns numbers, not a verdict.** Nothing here classifies a rhythm,
//! names a condition, or carries a decision threshold: this project has no labelled rhythm corpus to set
//! one from, and a threshold set by feel would be a diagnosis wearing a number. A caller that wants to
//! report irregular-rhythm episodes supplies its own separation, states where it came from, and says
//! plainly that the result is not a diagnosis.
//!
//! Every index is a measure of scatter, so a duplicated, rescaled or mis-stamped beat reads as
//! irregularity the heart never produced. [`assess`] gates on input quality FIRST and returns
//! [`IrregularityReading::Inconclusive`] often and by design; that is the result, not a failure.
//!
//! Cleaning is deliberately asymmetric with the rest of `hrv`. The physiological range filter is applied;
//! Malik ectopic rejection is only MEASURED, never applied, because it drops any beat over 20 % from its
//! local median — precisely the beat these indices exist to see.

pub mod cosen;
pub mod ectopy;
pub mod indices;
pub mod poincare;
pub mod quality;
pub mod screen;

pub use cosen::{cosen, sample_entropy, COSEN_M, COSEN_MIN_BEATS, COSEN_R_MS, SAMPEN_MAX_BEATS};
pub use ectopy::{profile, EctopyProfile};
pub use screen::{screen, Episode, EpisodeConfidence, ScreenRefusal, ScreenState};
pub use indices::{
    mean_rr_ms, rmssd_over_mean_rr, shannon_entropy_drr, turning_point_ratio, TPR_RANDOM_EXPECTED,
};
pub use poincare::{poincare, Poincare, POINCARE_CELL_MS};
pub use quality::{rescaled_copy_fraction, RrQuality};

/// Clean beats needed before [`assess`] computes anything. The shortest record COSEn is defined for; the
/// longer indices report `None` on their own until their own floor is met.
pub const ASSESS_MIN_BEATS: usize = COSEN_MIN_BEATS;

/// Beats in one segment. Every index here is defined over a SHORT segment — Dash et al. use 128 beats,
/// Lake & Moorman 12 — so a whole night is assessed as a run of segments, never as one series.
pub const SEGMENT_BEATS: usize = 128;

/// Why a reading was refused. Every variant is a measured condition of the INPUT, so a caller can say
/// which one blocked it rather than only that nothing came back.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refusal {
    /// Fewer beats survived the range filter than any index needs.
    TooFewBeats { have: usize, need: usize },
    /// The series repeats beats exactly — the same second and the same value.
    RepeatedBeats,
    /// The series carries rescaled second copies of its own beats.
    RescaledCopies,
    /// More beat-time than elapsed time: the beats cannot all be real.
    ImpossibleCoverage,
    /// The range filter dropped too much of the input to trust what is left.
    TooManyOutOfRange,
}

/// Every index over one R-R series, with the input quality that produced them. An index that its own
/// beat-count floor rules out is `None`; the reading as a whole still stands.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IrregularityIndices {
    pub quality: RrQuality,
    pub mean_rr_ms: f64,
    /// RMSSD over the mean R-R.
    pub rmssd_over_mean: Option<f64>,
    /// Normalised Shannon entropy of the successive differences, `0.0..=1.0`.
    pub shannon_entropy: Option<f64>,
    /// Turning point ratio; [`TPR_RANDOM_EXPECTED`] is the independent-series reference.
    pub turning_point_ratio: Option<f64>,
    /// Sample entropy at [`COSEN_M`] / [`COSEN_R_MS`], reported beside COSEn so the correction is visible.
    pub sample_entropy: Option<f64>,
    pub cosen: Option<f64>,
    pub poincare: Option<Poincare>,
    /// The shape of the scatter, which is what tells an ectopy burst from a sustained irregular rhythm.
    pub ectopy: Option<EctopyProfile>,
}

/// The result of one assessment.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum IrregularityReading {
    /// The input cannot support an honest index. Carries what was measured, so the refusal is legible.
    Inconclusive { reason: Refusal, quality: RrQuality },
    Assessed(IrregularityIndices),
}

/// Assess one timestamped R-R series (unix second, interval in ms), in the caller's chronological order.
///
/// Gate order: too few beats first (nothing else can be judged under it), then the three duplication
/// tests, then the out-of-range share. Only a series that clears all of them is scored.
pub fn assess(beats: &[(u32, u16)]) -> IrregularityReading {
    let quality = quality::measure(beats);
    let ranged = quality::ranged(beats);
    let refuse = |reason| IrregularityReading::Inconclusive { reason, quality };

    if ranged.len() < ASSESS_MIN_BEATS {
        return refuse(Refusal::TooFewBeats { have: ranged.len(), need: ASSESS_MIN_BEATS });
    }
    if beats.len() >= quality::MIN_QUALITY_BEATS {
        if quality.duplicate_fraction > quality::MAX_DUPLICATE_FRACTION {
            return refuse(Refusal::RepeatedBeats);
        }
        if quality.rescaled_fraction > quality::MAX_RESCALED_FRACTION {
            return refuse(Refusal::RescaledCopies);
        }
        if quality.coverage > quality::MAX_COVERAGE {
            return refuse(Refusal::ImpossibleCoverage);
        }
    }
    if quality.range_rejected_fraction > quality::MAX_RANGE_REJECTED_FRACTION {
        return refuse(Refusal::TooManyOutOfRange);
    }

    IrregularityReading::Assessed(IrregularityIndices {
        quality,
        mean_rr_ms: mean_rr_ms(&ranged).unwrap_or(0.0),
        rmssd_over_mean: rmssd_over_mean_rr(&ranged),
        shannon_entropy: shannon_entropy_drr(&ranged),
        turning_point_ratio: turning_point_ratio(&ranged),
        sample_entropy: sample_entropy(&ranged, COSEN_M, COSEN_R_MS),
        cosen: cosen(&ranged),
        poincare: poincare(&ranged),
        ectopy: profile(&ranged),
    })
}

/// Assess consecutive non-overlapping runs of `segment_beats` input beats, each tagged with the first
/// second in it. This is how a night is read: as a run of short segments, since every index here is
/// defined over one. Nothing is returned for a zero-length request or an empty input; a trailing partial
/// segment is assessed like any other and will refuse itself if it is too short.
pub fn assess_segments(beats: &[(u32, u16)], segment_beats: usize) -> Vec<(u32, IrregularityReading)> {
    if segment_beats == 0 {
        return Vec::new();
    }
    beats
        .chunks(segment_beats)
        .filter_map(|c| c.first().map(|&(t, _)| (t, assess(c))))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A regular series: one beat a second at ~1000 ms with a few ms of respiratory sway.
    fn regular(n: u32) -> Vec<(u32, u16)> {
        (0..n)
            .map(|i| {
                let sway = (f64::from(i) * 0.25 * std::f64::consts::TAU / 4.0).sin() * 12.0;
                (i, (1000.0 + sway).round() as u16)
            })
            .collect()
    }

    /// An irregular series: the same mean rate, but each interval drawn independently over a wide band.
    fn irregular(n: u32, seed: u64) -> Vec<(u32, u16)> {
        let mut x = seed;
        let mut out = Vec::new();
        let mut acc = 0.0f64;
        for _ in 0..n {
            x = x.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            let rr = 600 + ((x >> 33) % 800) as u16;
            acc += f64::from(rr) / 1000.0;
            out.push((acc as u32, rr));
        }
        out
    }

    fn indices(beats: &[(u32, u16)]) -> IrregularityIndices {
        match assess(beats) {
            IrregularityReading::Assessed(i) => i,
            other => panic!("expected an assessed reading, got {other:?}"),
        }
    }

    #[test]
    fn every_index_separates_a_regular_series_from_an_irregular_one() {
        let r = indices(&regular(120));
        let i = indices(&irregular(120, 4242));
        assert!(i.rmssd_over_mean > r.rmssd_over_mean, "rmssd/mean {:?} vs {:?}", r.rmssd_over_mean, i.rmssd_over_mean);
        assert!(i.shannon_entropy > r.shannon_entropy, "entropy {:?} vs {:?}", r.shannon_entropy, i.shannon_entropy);
        assert!(i.cosen > r.cosen, "cosen {:?} vs {:?}", r.cosen, i.cosen);
        let (rp, ip) = (r.poincare.unwrap(), i.poincare.unwrap());
        assert!(ip.sd1 > rp.sd1 && ip.normalised_area > rp.normalised_area, "poincare {rp:?} vs {ip:?}");
        assert!(ip.cell_occupancy > rp.cell_occupancy, "occupancy {} vs {}", rp.cell_occupancy, ip.cell_occupancy);
    }

    #[test]
    fn a_duplicated_series_is_refused_rather_than_scored() {
        // Each beat stored twice on its own second — the shape that would read as pure irregularity.
        let doubled: Vec<(u32, u16)> = regular(120).into_iter().flat_map(|b| [b, b]).collect();
        match assess(&doubled) {
            IrregularityReading::Inconclusive { reason: Refusal::RepeatedBeats, .. } => {}
            other => panic!("a repeated series must be refused, got {other:?}"),
        }
        // And the rescaled shape, which the exact-repeat count cannot see at all.
        let mut rescaled: Vec<(u32, u16)> = Vec::new();
        for (t, v) in regular(120) {
            rescaled.push((t, v));
            rescaled.push((t + 1, (f64::from(v) * quality::RESCALE_RATIO).round() as u16));
        }
        match assess(&rescaled) {
            IrregularityReading::Inconclusive { reason: Refusal::RescaledCopies, quality } => {
                assert_eq!(quality.duplicate_fraction, 0.0, "not one exact repeat, and still duplicated");
            }
            other => panic!("a rescaled series must be refused, got {other:?}"),
        }
    }

    #[test]
    fn short_and_junk_input_is_inconclusive_never_a_number() {
        assert!(matches!(
            assess(&[]),
            IrregularityReading::Inconclusive { reason: Refusal::TooFewBeats { have: 0, .. }, .. }
        ));
        // Every beat physiologically impossible: nothing survives the range filter.
        let junk: Vec<(u32, u16)> = (0..60).map(|i| (i, 5)).collect();
        assert!(matches!(
            assess(&junk),
            IrregularityReading::Inconclusive { reason: Refusal::TooFewBeats { .. }, .. }
        ));
        // Enough survivors to score, but a fifth of the input was out of range.
        let mut mixed: Vec<(u32, u16)> = (0..40u32).map(|i| (i, 1000u16)).collect();
        mixed.extend((40..55u32).map(|i| (i, 5u16)));
        assert!(matches!(
            assess(&mixed),
            IrregularityReading::Inconclusive { reason: Refusal::TooManyOutOfRange, .. }
        ));
    }

    #[test]
    fn segments_are_assessed_one_at_a_time_and_a_short_tail_refuses_itself() {
        let segs = assess_segments(&regular(300), SEGMENT_BEATS);
        assert_eq!(segs.len(), 3, "128 + 128 + a 44-beat tail");
        assert_eq!(segs[0].0, 0);
        assert!(matches!(segs[0].1, IrregularityReading::Assessed(_)));
        // A tail of 44 beats is still over the floor, so it is assessed rather than dropped.
        assert!(matches!(segs[2].1, IrregularityReading::Assessed(_)));
        // A tail under the floor refuses itself instead of being silently merged or discarded.
        let short_tail = assess_segments(&regular(133), SEGMENT_BEATS);
        assert!(matches!(
            short_tail[1].1,
            IrregularityReading::Inconclusive { reason: Refusal::TooFewBeats { have: 5, .. }, .. }
        ));
        assert!(assess_segments(&regular(300), 0).is_empty());
        assert!(assess_segments(&[], SEGMENT_BEATS).is_empty());
    }

    #[test]
    fn a_night_length_series_refuses_the_short_record_indices_but_still_reports_the_rest() {
        // COSEn and sample entropy are short-record statistics; over a long series they are refused
        // rather than approximated, while the length-tolerant indices still answer.
        let long = indices(&regular(SAMPEN_MAX_BEATS as u32 + 200));
        assert_eq!((long.cosen, long.sample_entropy), (None, None));
        assert!(long.rmssd_over_mean.is_some() && long.shannon_entropy.is_some());
    }

    #[test]
    fn no_wording_this_module_can_emit_is_clinical() {
        // The same gate `format_hr_watch` carries in the CLI, applied here to the Debug rendering, which
        // is the only text this crate produces and the text that reaches a log.
        let banned = [
            "afib", "fibrillat", "arrhythm", "cardiac", "diagnos", "disease", "patient", "clinical",
            "alarm", "emergency", "abnormal",
        ];
        let q = quality::measure(&regular(60));
        let mut renderings: Vec<String> = vec![format!("{:?}", assess(&regular(120)))];
        for reason in [
            Refusal::TooFewBeats { have: 3, need: ASSESS_MIN_BEATS },
            Refusal::RepeatedBeats,
            Refusal::RescaledCopies,
            Refusal::ImpossibleCoverage,
            Refusal::TooManyOutOfRange,
        ] {
            renderings.push(format!("{:?}", IrregularityReading::Inconclusive { reason, quality: q }));
        }
        for text in renderings {
            let low = text.to_lowercase();
            for term in banned {
                assert!(!low.contains(term), "clinical term '{term}' in: {text}");
            }
        }
    }
}
