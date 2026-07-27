//! Blood oxygen from the 4.0 raw red/IR pair, and its rolling nightly reading.

use crate::*;

/// SpO2 (%) from a 4.0 paired red/IR window (ratio-of-ratios). `None` if not pulsatile.
#[uniffi::export]
pub fn spo2_from_paired(red: Vec<f64>, ir: Vec<f64>) -> Option<f64> {
    Spo2::from_paired(&red, &ir)
}

/// One in-bed span (`[start, end]` unix seconds) for the nightly SpO2 raw-means gate.
#[derive(uniffi::Record)]
pub struct Spo2Span {
    pub start: i64,
    pub end: i64,
}

/// One 4.0 raw SpO2 sample: red/IR PPG ADC at unix second `ts`.
#[derive(uniffi::Record)]
pub struct Spo2RawSample {
    pub ts: i64,
    pub red: i32,
    pub ir: i32,
}

/// Integer-truncated nightly means of the raw red/IR ADC over the detected in-bed spans.
#[derive(uniffi::Record)]
pub struct Spo2RawMeans {
    pub red: i32,
    pub ir: i32,
}

/// Nightly integer-truncated means of the 4.0 raw red/IR PPG ADC over the detected in-bed `spans`, the
/// app's stored `DailyMetric.spo2Red`/`spo2Ir`. A sample counts when its `ts` lies inside any span.
/// `None` when either input is empty or no sample landed in-span. Raw ADC only, never a calibrated percent.
#[uniffi::export]
pub fn nightly_spo2_raw_means(spans: Vec<Spo2Span>, samples: Vec<Spo2RawSample>) -> Option<Spo2RawMeans> {
    let spans: Vec<(i64, i64)> = spans.into_iter().map(|s| (s.start, s.end)).collect();
    let samples: Vec<(i64, i32, i32)> = samples.into_iter().map(|s| (s.ts, s.red, s.ir)).collect();
    Spo2::nightly_raw_means(&spans, &samples).map(|(red, ir)| Spo2RawMeans { red, ir })
}

/// A smoothed multi-night SpO2 readout: `pct` once calibrated, else `calibrating_nights` carries the
/// night count so far.
#[derive(uniffi::Record)]
pub struct Spo2Rolling {
    pub pct: Option<f64>,
    pub calibrating_nights: Option<u32>,
}

/// The 4.0 display value. Its ratio-of-ratios percent carries an uncalibrated per-device DC offset, so
/// the absolute number means nothing on its own — this anchors the 30-night median to a plausible
/// baseline and reports the 7-night median at that offset, keeping the night-to-night movement.
/// `recent_nightly` is oldest to newest. 5.0/MG does not use this: its percent comes off the strap.
#[uniffi::export]
pub fn spo2_rolling_reading(recent_nightly: Vec<f64>) -> Spo2Rolling {
    let r = Spo2::rolling_reading(&recent_nightly);
    Spo2Rolling { pct: r.pct, calibrating_nights: r.calibrating_nights.map(|n| n as u32) }
}
