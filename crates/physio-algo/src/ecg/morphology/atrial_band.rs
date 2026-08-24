//! Atrial-band energy in the quiet segment between beats — the 4-9 Hz share of a TP segment.
//!
//! Between the end of one T wave and the start of the next P wave the ventricles contribute nothing, so
//! whatever sits there is atrial activity or noise. Continuous atrial activity lands around 4-9 Hz,
//! which is above the T wave and below the QRS, and that is the only reason the band is interesting.
//!
//! **This returns a ratio, not a name and not a verdict.** No threshold is defined here: the band is
//! shared with muscle noise and with electrode movement, and a cut set by feel would turn an artefact
//! into a finding. A caller that wants to separate on it supplies its own separation and says where it
//! came from.
//!
//! The stated weakness, because it is structural rather than fixable by tuning: a TP segment is a few
//! hundred milliseconds, and a Hann-windowed buffer of length `T` resolves no finer than `4/T` Hz — at
//! 250 ms that is 16 Hz, wider than the 5 Hz band being measured. So this index reads the band's share
//! of a smeared spectrum, not a resolved peak, and it gets blunter as the heart rate rises and the
//! segments shorten. [`AtrialBand::median_segment_ms`] is reported for exactly that reason.

use crate::ecg::spectrum::Periodogram;
use crate::ecg::{qt_end_ms, sanitized, usable_rate};
use crate::stats::median;

use super::evidence_fraction;
use super::p_wave::{P_SEARCH_MS, PR_GUARD_MS};

/// The atrial band, and the in-band reference it is expressed as a share of. The reference starts at
/// 1 Hz so residual baseline drift does not inflate the denominator, and stops at 30 Hz because above
/// that a TP segment carries only noise.
pub const ATRIAL_BAND_LO_HZ: f64 = 4.0;
pub const ATRIAL_BAND_HI_HZ: f64 = 9.0;
pub const REFERENCE_BAND_LO_HZ: f64 = 1.0;
pub const REFERENCE_BAND_HI_HZ: f64 = 30.0;

/// The 30 Hz reference edge only exists below Nyquist at 60 Hz and up. The module's supported span
/// already starts at 100 Hz, so every rate that reaches here clears it — this constant records the
/// dependency rather than guarding one that cannot fire.
pub const MIN_FS_FOR_BANDS_HZ: f64 = 2.0 * REFERENCE_BAND_HI_HZ;

/// Shortest TP segment used. Below this the band occupies under a third of the resolvable width and the
/// ratio stops carrying information; the segment is dropped rather than measured.
pub const TP_MIN_MS: f64 = 120.0;
/// Segments needed before a ratio is reported at all.
pub const MIN_SEGMENTS: usize = 5;
/// Segments at which the reading has its full evidence budget, mirroring the P-wave beat budget.
pub const FULL_EVIDENCE_SEGMENTS: usize = 20;

/// The atrial-band reading over one record.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AtrialBandEvidence {
    Measured(AtrialBand),
    /// No ratio could be measured. First-class, not a failure.
    Indeterminate(AtrialBandLimit),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AtrialBand {
    /// Median across segments of `P(4-9 Hz) / P(1-30 Hz)`.
    pub ratio: f64,
    /// Segments the median was taken over.
    pub segments: usize,
    /// Median segment length (ms) — the spectral resolution this ratio was measured at.
    pub median_segment_ms: f64,
    /// Segments over [`FULL_EVIDENCE_SEGMENTS`], capped at 1.0. How much of the evidence budget the
    /// reading had — NOT a probability.
    pub confidence: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AtrialBandLimit {
    /// Outside the supported sample-rate span.
    UnusableRate,
    /// Too few TP segments long enough to measure. At a fast rate the quiet segment closes entirely,
    /// which is a property of the rhythm, not a fault.
    TooFewSegments { have: usize, need: usize },
}

/// Median 4-9 Hz share of the TP segments between consecutive R peaks.
///
/// `peaks` are R-peak sample indices in ascending order. Pure and deterministic; never panics on empty,
/// constant or non-finite input.
pub fn atrial_band(samples: &[f64], fs_hz: f64, peaks: &[usize]) -> AtrialBandEvidence {
    if !usable_rate(fs_hz) {
        return AtrialBandEvidence::Indeterminate(AtrialBandLimit::UnusableRate);
    }
    let x = sanitized(samples);
    let min_len = (TP_MIN_MS / 1000.0 * fs_hz).round() as usize;
    // The next beat's P search starts here, so the segment stops before it: TP, not TP plus P.
    let p_onset = ((PR_GUARD_MS + P_SEARCH_MS) / 1000.0 * fs_hz).round() as usize;

    let mut ratios = Vec::new();
    let mut lengths = Vec::new();
    for w in peaks.windows(2) {
        let (prev, next) = (w[0], w[1]);
        if next <= prev || next > x.len() {
            continue;
        }
        let rr_ms = (next - prev) as f64 / fs_hz * 1000.0;
        let start = prev + (qt_end_ms(rr_ms) / 1000.0 * fs_hz).round() as usize;
        let Some(end) = next.checked_sub(p_onset) else {
            continue;
        };
        if end <= start || end - start < min_len {
            continue;
        }
        let segment = &x[start..end];
        let p = Periodogram::new(segment);
        let reference = p.band_power(REFERENCE_BAND_LO_HZ / fs_hz, REFERENCE_BAND_HI_HZ / fs_hz);
        if reference <= 0.0 {
            continue;
        }
        ratios.push(p.band_power(ATRIAL_BAND_LO_HZ / fs_hz, ATRIAL_BAND_HI_HZ / fs_hz) / reference);
        lengths.push(segment.len() as f64 / fs_hz * 1000.0);
    }

    if ratios.len() < MIN_SEGMENTS {
        let limit = AtrialBandLimit::TooFewSegments {
            have: ratios.len(),
            need: MIN_SEGMENTS,
        };
        return AtrialBandEvidence::Indeterminate(limit);
    }
    AtrialBandEvidence::Measured(AtrialBand {
        ratio: median(&ratios),
        segments: ratios.len(),
        median_segment_ms: median(&lengths),
        confidence: evidence_fraction(ratios.len(), FULL_EVIDENCE_SEGMENTS),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecg::test_signals::{constant, synthetic_ecg};
    use std::f64::consts::PI;

    fn measured(e: AtrialBandEvidence) -> AtrialBand {
        match e {
            AtrialBandEvidence::Measured(m) => m,
            other => panic!("expected a measured ratio, got {other:?}"),
        }
    }

    #[test]
    fn a_tone_inside_the_band_raises_the_ratio_and_one_outside_lowers_it() {
        let fs = 250.0;
        let (x, truth) = synthetic_ecg(fs, 40.0, 60.0, 1.0, 0.0, 1);
        let add = |hz: f64, amp: f64| -> Vec<f64> {
            x.iter()
                .enumerate()
                .map(|(i, v)| v + amp * (2.0 * PI * hz * i as f64 / fs).sin())
                .collect()
        };
        let inside = measured(atrial_band(&add(6.0, 0.05), fs, &truth)).ratio;
        let outside = measured(atrial_band(&add(20.0, 0.05), fs, &truth)).ratio;
        assert!(
            inside > 0.7,
            "a 6 Hz tone must dominate the band, got {inside:.3}"
        );
        assert!(outside < 0.2, "a 20 Hz tone must not, got {outside:.3}");
        assert!(
            inside > 3.0 * outside,
            "inside {inside:.3} outside {outside:.3}"
        );
    }

    #[test]
    fn the_segments_used_and_their_length_are_reported() {
        let fs = 250.0;
        let (x, truth) = synthetic_ecg(fs, 40.0, 60.0, 1.0, 0.01, 2);
        let m = measured(atrial_band(&x, fs, &truth));
        assert!(
            m.segments >= MIN_SEGMENTS && m.segments < truth.len(),
            "{m:?}"
        );
        assert!(m.median_segment_ms >= TP_MIN_MS, "{m:?}");
        assert!(
            (0.0..=1.0).contains(&m.confidence) && (0.0..=1.0).contains(&m.ratio),
            "{m:?}"
        );
    }

    #[test]
    fn a_fast_rhythm_closes_the_segment_and_the_reading_refuses() {
        // At 150 bpm the T wave runs into the next P: there is no quiet segment left to measure.
        let fs = 250.0;
        let (x, truth) = synthetic_ecg(fs, 40.0, 150.0, 1.0, 0.0, 3);
        assert!(matches!(
            atrial_band(&x, fs, &truth),
            AtrialBandEvidence::Indeterminate(AtrialBandLimit::TooFewSegments { .. })
        ));
    }

    #[test]
    fn degenerate_input_refuses_rather_than_panics() {
        let (x, truth) = synthetic_ecg(250.0, 40.0, 60.0, 1.0, 0.0, 4);
        assert_eq!(
            atrial_band(&x, f64::NAN, &truth),
            AtrialBandEvidence::Indeterminate(AtrialBandLimit::UnusableRate)
        );
        // The supported span starts at 100 Hz, comfortably over the 60 Hz the reference band needs.
        assert_eq!(
            atrial_band(&x, 50.0, &truth),
            AtrialBandEvidence::Indeterminate(AtrialBandLimit::UnusableRate)
        );
        const { assert!(crate::ecg::MIN_FS_HZ >= MIN_FS_FOR_BANDS_HZ) };
        assert!(matches!(
            atrial_band(&x, 250.0, &[]),
            AtrialBandEvidence::Indeterminate(AtrialBandLimit::TooFewSegments { have: 0, .. })
        ));
        // A flat segment has no power at all, so it contributes no ratio rather than a 0/0.
        assert!(matches!(
            atrial_band(&constant(10_000, 3.0), 250.0, &truth),
            AtrialBandEvidence::Indeterminate(AtrialBandLimit::TooFewSegments { .. })
        ));
        assert!(matches!(
            atrial_band(&[f64::NAN; 10_000], 250.0, &truth),
            AtrialBandEvidence::Indeterminate(AtrialBandLimit::TooFewSegments { .. })
        ));
    }

    #[test]
    fn the_ratio_does_not_depend_on_the_amplitude_scale() {
        let fs = 250.0;
        let (x, truth) = synthetic_ecg(fs, 40.0, 60.0, 1.0, 0.02, 5);
        let base = measured(atrial_band(&x, fs, &truth)).ratio;
        for gain in [1e-3, 37.0, 5000.0] {
            let scaled: Vec<f64> = x.iter().map(|v| v * gain).collect();
            assert!(
                (measured(atrial_band(&scaled, fs, &truth)).ratio - base).abs() < 1e-9,
                "gain {gain}"
            );
        }
    }
}
