//! Single-lead ECG: two structurally different QRS detectors and the agreement between them, the
//! published signal-quality indices built on top ([`sqi`], [`score`]), and the mains anchor that reads
//! an unknown sample rate straight out of the power-line interference ([`mains`]). [`spectrum`] is the
//! uniform-grid periodogram the last two share.
//!
//! The pair is the point. [`detect_pan_tompkins`] asks an ENERGY question (how much 5-15 Hz power sits in
//! this 150 ms window) and is sign- and phase-blind; [`detect_wavelet`] asks a SHAPE question (is there a
//! signed slope reversal inside one QRS width, coherent across an octave of dyadic scales). They fail in
//! opposite directions — the energy detector accepts any sharp broadband transient, the shape detector
//! misses smooth low-amplitude beats and accepts slow baseline swings — so their agreement
//! ([`beat_agreement`]) is evidence about the SIGNAL rather than about either detector.
//!
//! Both take raw samples plus a sample rate, work over [`MIN_FS_HZ`]..=[`MAX_FS_HZ`], never panic on
//! empty / constant / non-finite input, and are deterministic. Amplitude units are whatever the caller
//! passes; nothing here converts counts to mV, and no threshold depends on the amplitude scale.

mod agreement;
pub mod mains;
pub mod morphology;
mod qrs_pan_tompkins;
mod qrs_wavelet;
pub mod score;
pub mod spectrum;
pub mod sqi;
pub mod sweep;
#[cfg(test)]
mod test_signals;

pub use agreement::{Agreement, DEFAULT_MATCH_WINDOW_MS, beat_agreement};
pub use mains::{
    MainsAnchor, MainsConfig, MainsFix, MainsUnavailable, mains_anchor, mains_anchor_with,
};
pub use morphology::{EcgMorphology, morphology};
pub use qrs_pan_tompkins::detect_pan_tompkins;
pub use qrs_wavelet::detect_wavelet;
pub use score::{EcgScore, EcgVerdict, score};
pub use spectrum::Periodogram;
pub use sqi::{BeatTemplate, bas_sqi, beat_template, beat_template_window, k_sqi, p_sqi};

/// Supported sample-rate span. The strap's ECG rate is unknown, so every stage sizes its windows from
/// `fs` in seconds rather than in samples; outside this span the detectors return no peaks. The upper
/// end is 1024 rather than 1000 because a converter's rate register selects a binary decimation, so
/// 1024 Hz is a rate the sweep in [`sweep`] has to be able to reach.
pub const MIN_FS_HZ: f64 = 100.0;
pub const MAX_FS_HZ: f64 = 1024.0;

/// Widest QRS a detector will treat as one complex (ms). The upper physiological bound for a wide
/// complex; anything broader is a different event, and both detectors use it to bound their own windows.
pub const MAX_QRS_MS: f64 = 140.0;

/// Absolute refractory after an accepted beat (ms). 220 bpm is the plausible upper HR, i.e. a 273 ms R-R,
/// so 200 ms cannot suppress a real beat while it does suppress a T-wave or a re-detection of the same QRS.
pub const REFRACTORY_MS: f64 = 200.0;

/// `true` when `fs_hz` is finite and inside the supported span.
pub(crate) fn usable_rate(fs_hz: f64) -> bool {
    fs_hz.is_finite() && (MIN_FS_HZ..=MAX_FS_HZ).contains(&fs_hz)
}

/// Copy with every non-finite sample replaced by 0.0. NaN and infinity come off a real link (a dropped
/// packet, a divide in a caller's decode); a detector that propagates them returns garbage indices or
/// panics on a `partial_cmp`, so they are flattened once here and never again downstream.
pub(crate) fn sanitized(samples: &[f64]) -> Vec<f64> {
    samples
        .iter()
        .map(|&v| if v.is_finite() { v } else { 0.0 })
        .collect()
}

/// Window length in samples for a duration in ms, at least `min`.
pub(crate) fn samples_for_ms(ms: f64, fs_hz: f64, min: usize) -> usize {
    ((ms / 1000.0 * fs_hz).round() as usize).max(min)
}

/// Bazett end of the T wave after an R peak (ms), for a preceding interval of `rr_ms`: `QT = 0.4 * √RR`.
/// The one rate-aware boundary between a beat's repolarisation and the quiet segment after it, shared by
/// the T-wave guard in `qrs_pan_tompkins` and the segment geometry in [`morphology`].
pub(crate) const BAZETT_QT_COEFFICIENT: f64 = 0.4;

pub(crate) fn qt_end_ms(rr_ms: f64) -> f64 {
    BAZETT_QT_COEFFICIENT * (rr_ms / 1000.0).sqrt() * 1000.0
}

/// Zero-phase band limit: a centred moving average whose first null sits at `lowpass_null_hz` keeps the
/// low side, and subtracting a second centred average nulling at `highpass_null_hz` removes the slow
/// side. Centred means no group delay to subtract back off any position measured on the result.
pub(crate) fn band_limited(
    x: &[f64],
    fs_hz: f64,
    lowpass_null_hz: f64,
    highpass_null_hz: f64,
) -> Vec<f64> {
    let lp_len = (fs_hz / lowpass_null_hz).round().max(1.0) as usize;
    let hp_len = (fs_hz / highpass_null_hz).round().max(3.0) as usize;
    let low = crate::signal::moving_average_centred(x, lp_len);
    let baseline = crate::signal::moving_average_centred(&low, hp_len);
    (0..low.len()).map(|i| low[i] - baseline[i]).collect()
}
