//! ECG waveform morphology — the evidence R-R scatter cannot reach, and the reason a strap with an
//! electrode can say more than one without.
//!
//! Three measures over the same samples and R peaks, each computed and reported ON ITS OWN:
//! [`p_wave`] (is there a consistent deflection in the PR segment), [`atrial_band`] (the 4-9 Hz share
//! of the quiet segment between beats) and beat-template consistency (do the beats keep one shape).
//!
//! **They are deliberately not fused, and they are not fused with `rr_irregularity` either.** A single
//! number would say something different depending on which strap produced it — morphology needs an
//! electrode, R-R scatter does not — and a caller would have no way to tell which. Keep the streams
//! apart, present them as separate lines, and let the reader see which evidence was available.
//!
//! **This is a wellness screen. It returns measurements, not a verdict.** Nothing here classifies a
//! rhythm or names a condition: that is a regulated claim, this is not a regulated device, and both
//! error directions are real — a false positive causes needless worry, a false negative is false
//! reassurance. So every measure can return `Indeterminate`, does so often, and that is the result
//! rather than a failure.
//!
//! Nothing converts counts to millivolts. Every figure is a ratio against the record's own amplitude,
//! which is what lets these run on a stream whose scale is unknown.
//!
//! Measured over 116 labelled 60 s chest-ECG windows (`examples/ecg_morphology_corpus.rs`), the P-wave
//! finding came out: 73 windows annotated sinus → 71 Present, 0 Absent, 2 Indeterminate; 25 annotated
//! irregular → 4 Present, 8 Absent, 13 Indeterminate. Zero real P waves were called absent, which is
//! the direction the thresholds are set for, and the price is that half the irregular windows decline
//! to answer. The atrial band shifted with the label but overlapped (median 0.35 against 0.25 on the
//! same database), which is why it carries no threshold. NONE of it is wrist ECG.

pub mod atrial_band;
pub mod p_wave;

pub use atrial_band::{AtrialBand, AtrialBandEvidence, AtrialBandLimit, atrial_band};
pub use p_wave::{PWaveEvidence, PWaveFinding, PWaveLimit, p_wave};

use crate::ecg::sqi::beat_template;

/// Beats at which the consistency reading has its full evidence budget. Matches the P-wave budget so
/// the two confidences on one report mean the same thing.
pub const CONSISTENCY_FULL_EVIDENCE_BEATS: usize = p_wave::P_FULL_EVIDENCE_BEATS;

/// The one confidence in this module: how much of a measure's evidence budget it actually had, capped
/// at 1.0. NOT a probability and not a calibrated confidence — a reading with everything it wanted can
/// still be wrong, and this number says nothing about that.
pub fn evidence_fraction(have: usize, full: usize) -> f64 {
    if full == 0 {
        0.0
    } else {
        (have as f64 / full as f64).min(1.0)
    }
}

/// Do the beats keep one shape? A thin reading over [`beat_template`], which the sweep's scorer already
/// builds; nothing here re-implements the correlator.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BeatConsistency {
    /// Mean leave-one-out correlation of each beat against the average of the others.
    Measured {
        correlation: f64,
        beats: usize,
        confidence: f64,
    },
    /// Too few whole beat windows fit the buffer to build a template from.
    Indeterminate,
}

/// All three morphology measures over one record, side by side and unfused.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EcgMorphology {
    /// The rate every time-domain figure below is conditional on.
    pub fs_hz: f64,
    /// R peaks the caller supplied.
    pub beats: usize,
    pub p_wave: PWaveEvidence,
    pub atrial_band: AtrialBandEvidence,
    pub beat_consistency: BeatConsistency,
}

/// Run the three measures over `samples` at `fs_hz`, using R peaks the caller has already detected
/// (either QRS detector in this module produces them, ascending).
///
/// Pure and deterministic. Never panics on empty, constant, non-finite or peak-free input; each measure
/// refuses on its own terms and says which limit it hit.
pub fn morphology(samples: &[f64], fs_hz: f64, peaks: &[usize]) -> EcgMorphology {
    let consistency = match beat_template(samples, fs_hz, peaks) {
        Some(t) => BeatConsistency::Measured {
            correlation: t.correlation,
            beats: t.beats,
            confidence: evidence_fraction(t.beats, CONSISTENCY_FULL_EVIDENCE_BEATS),
        },
        None => BeatConsistency::Indeterminate,
    };
    EcgMorphology {
        fs_hz,
        beats: peaks.len(),
        p_wave: p_wave(samples, fs_hz, peaks),
        atrial_band: atrial_band(samples, fs_hz, peaks),
        beat_consistency: consistency,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecg::test_signals::{constant, synthetic_ecg};

    #[test]
    fn a_clean_record_measures_all_three_and_a_flat_line_refuses_all_three() {
        let fs = 250.0;
        let (x, truth) = synthetic_ecg(fs, 40.0, 60.0, 1.0, 0.01, 1);
        let m = morphology(&x, fs, &truth);
        assert_eq!(m.beats, truth.len());
        assert_eq!(m.p_wave.finding, PWaveFinding::Present, "{m:?}");
        assert!(
            matches!(m.atrial_band, AtrialBandEvidence::Measured(_)),
            "{m:?}"
        );
        assert!(
            matches!(m.beat_consistency, BeatConsistency::Measured { .. }),
            "{m:?}"
        );

        let flat = morphology(&constant(10_000, 0.0), fs, &[]);
        assert!(
            matches!(flat.p_wave.finding, PWaveFinding::Indeterminate(_)),
            "{flat:?}"
        );
        assert!(
            matches!(flat.atrial_band, AtrialBandEvidence::Indeterminate(_)),
            "{flat:?}"
        );
        assert_eq!(flat.beat_consistency, BeatConsistency::Indeterminate);
    }

    #[test]
    fn no_measure_can_stand_in_for_another() {
        // The three are independent by construction: strip the P wave and the other two are unmoved.
        // A fused number would hide exactly this, which is why there is not one.
        let fs = 250.0;
        let (x, truth) = synthetic_ecg(fs, 40.0, 60.0, 1.0, 0.0, 2);
        let full = morphology(&x, fs, &truth);
        let mut stripped = x.clone();
        for &c in &truth {
            let (mu, sigma) = (c as f64 - 0.180 * fs, 0.025 * fs);
            let span = (4.0 * sigma) as isize;
            for i in
                (mu as isize - span).max(0)..(mu as isize + span).min(stripped.len() as isize - 1)
            {
                let d = (i as f64 - mu) / sigma;
                stripped[i as usize] -= 0.15 * (-0.5 * d * d).exp();
            }
        }
        let cut = morphology(&stripped, fs, &truth);
        assert_eq!(cut.p_wave.finding, PWaveFinding::Absent, "{cut:?}");
        assert_eq!(
            cut.beat_consistency, full.beat_consistency,
            "the QRS did not change"
        );
    }

    #[test]
    fn no_wording_this_module_can_emit_is_clinical() {
        // The gate `format_hr_watch` carries in the CLI, applied to the Debug rendering — the only text
        // this crate produces, and the text that reaches a log. Same list as `rr_irregularity`.
        let banned = [
            "afib",
            "fibrillat",
            "arrhythm",
            "cardiac",
            "diagnos",
            "disease",
            "patient",
            "clinical",
            "alarm",
            "emergency",
            "abnormal",
        ];
        let (x, truth) = synthetic_ecg(250.0, 40.0, 60.0, 1.0, 0.01, 3);
        let mut renderings = vec![
            format!("{:?}", morphology(&x, 250.0, &truth)),
            format!("{:?}", morphology(&constant(10_000, 0.0), 250.0, &[])),
            format!(
                "{BeatConsistency:?}",
                BeatConsistency = BeatConsistency::Indeterminate
            ),
        ];
        for finding in [
            PWaveFinding::Present,
            PWaveFinding::Absent,
            PWaveFinding::Indeterminate(PWaveLimit::UnusableRate),
            PWaveFinding::Indeterminate(PWaveLimit::WindowTooNarrow {
                samples: 2,
                need: 4,
            }),
            PWaveFinding::Indeterminate(PWaveLimit::TooFewBeats { have: 1, need: 8 }),
            PWaveFinding::Indeterminate(PWaveLimit::NoReferenceAmplitude),
            PWaveFinding::Indeterminate(PWaveLimit::BelowNoiseFloor),
            PWaveFinding::Indeterminate(PWaveLimit::Ambiguous),
        ] {
            renderings.push(format!("{finding:?}"));
        }
        for limit in [
            AtrialBandLimit::UnusableRate,
            AtrialBandLimit::TooFewSegments { have: 0, need: 5 },
        ] {
            renderings.push(format!("{:?}", AtrialBandEvidence::Indeterminate(limit)));
        }
        for text in renderings {
            let low = text.to_lowercase();
            for term in banned {
                assert!(!low.contains(term), "clinical term '{term}' in: {text}");
            }
        }
    }
}
