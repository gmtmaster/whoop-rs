//! Type-47 HISTORICAL_DATA decoders, dispatched by the version byte (version → generation is
//! many-to-one). Offsets are INNER-relative (frame-absolute − inner_start), which is why the GEN4/GEN5
//! +4 delta disappears. Every field is range-gated so a wrong offset on unmapped firmware yields `None`.

mod gen4;
mod gen5;

use crate::bytes::f32_at;
use crate::family::Family;
use crate::packet::Frame;

/// A per-second summary record (v18 on 5.0/MG, v5/v24 on 4.0). Absent fields = not carried by that
/// version; constructors set only what they carry and `..Default::default()` fills the rest with None.
#[derive(Clone, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct HistoryRecord {
    pub version: u8,
    pub unix: u32,
    pub heart_rate: Option<u8>,
    pub rr_intervals: Vec<u16>,
    pub gravity: Option<[f32; 3]>,
    pub skin_temp_c: Option<f32>,
    pub spo2: Option<(u16, u16)>, // 4.0 v24 raw red/IR ADC
    pub resp_raw: Option<u16>,    // 4.0 v24 raw respiration ADC
    pub steps: Option<u16>,       // 5.0 v18 cumulative counter
    pub activity_class: Option<u8>, // 5.0 v18 (0 still / 1 walk / 2 run)
    pub sleep_state: Option<u8>,  // 5.0 v18 (0 wake / 1 still / 2 asleep / 3 up)
}

/// A raw 24 Hz optical buffer (v26 on 5.0/MG). One record = one strap second.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct PpgRecord {
    pub version: u8,
    pub unix: u32,
    pub record_id: Option<u16>, // strap record counter (monotonic within a v26 burst)
    pub samples: Vec<i16>,      // AC-coupled raw optical waveform (single wavelength, no red/IR pair)
}

/// A raw 6-axis IMU offload buffer (v21 on 5.0/MG): one strap-second of 100 Hz accel + gyro, shipped in
/// the R22 deep buffers on the historical path. Samples are raw i16 LSB; scale by the constants below.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct ImuRecord {
    pub version: u8,
    pub unix: u32,
    pub sample_rate_hz: u16,
    pub accel: Vec<[i16; 3]>, // per-sample [ax, ay, az] raw LSB (× IMU_ACCEL_SCALE_G = g)
    pub gyro: Vec<[i16; 3]>,  // per-sample [gx, gy, gz] raw LSB (× IMU_GYRO_SCALE_DPS = deg/s)
}

/// Accel scale (g per LSB) and gyro scale (deg/s per LSB, ±2000 dps) for `ImuRecord` samples.
pub const IMU_ACCEL_SCALE_G: f32 = 1.0 / 4096.0;
pub const IMU_GYRO_SCALE_DPS: f32 = 2000.0 / 32768.0;

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
#[cfg_attr(feature = "serde", serde(tag = "kind"))]
pub enum Record {
    History(HistoryRecord),
    Ppg(PpgRecord),
    Imu(ImuRecord),
}

/// Decode a CRC-checked type-47 frame. Version-keyed, except the IMU deep buffer is identified by its
/// length + in-packet counts (not the version byte), so it's tried first for GEN5 — a normal v18/v26
/// frame is too short to pass its gate. `None` = unknown version or a failed range gate.
pub fn decode(frame: &Frame) -> Option<Record> {
    if frame.family == Family::Gen5 {
        if let Some(imu) = gen5::v21_imu(frame) {
            return Some(Record::Imu(imu));
        }
    }
    match (frame.family, frame.version()) {
        (Family::Gen5, 18) => gen5::v18(frame).map(Record::History),
        (Family::Gen5, 26) => gen5::v26(frame).map(Record::Ppg),
        (Family::Gen4, 12) | (Family::Gen4, 24) => gen4::v24(frame).map(Record::History),
        (Family::Gen4, 5) | (Family::Gen4, 7) | (Family::Gen4, 9) => gen4::v5(frame).map(Record::History),
        (Family::Gen4, 25) => gen4::v25(frame).map(Record::History),
        // Unmapped 4.0 version → try the v24 layout, accept only if it passes a strict plausibility gate.
        (Family::Gen4, _) => gen4::v24(frame).filter(plausible_fallback).map(Record::History),
        // Unmapped 5.0/MG version → skip (the IMU deep buffer was already tried above).
        (Family::Gen5, _) => None,
    }
}

/// The strict gate for accepting an unmapped 4.0 version via the v24 fallback: HR 25..230 AND |g| ≈ 1.
fn plausible_fallback(r: &HistoryRecord) -> bool {
    let hr_ok = r.heart_rate.is_some_and(|h| (25..=230).contains(&h));
    let g_ok = r.gravity.is_some_and(|g| (0.8..1.2).contains(&magnitude(g)));
    hr_ok && g_ok
}

/// Euclidean magnitude of a gravity vector.
fn magnitude(g: [f32; 3]) -> f32 {
    (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]).sqrt()
}

/// Accept a gravity vector only if finite and physically plausible (|g| ≈ 1), so a wrong offset drops.
pub(crate) fn accept_gravity(g: [f32; 3]) -> Option<[f32; 3]> {
    if !g.iter().all(|v| v.is_finite()) {
        return None;
    }
    (0.5..1.5).contains(&magnitude(g)).then_some(g)
}

/// Read three consecutive f32 as a gravity vector, gated by `accept_gravity`.
pub(crate) fn gravity3(b: &[u8], off: usize) -> Option<[f32; 3]> {
    accept_gravity([f32_at(b, off)?, f32_at(b, off + 4)?, f32_at(b, off + 8)?])
}

#[cfg(all(test, feature = "serde"))]
mod serde_tests {
    use super::*;

    #[test]
    fn record_serializes_kind_tagged_snake_case() {
        let r = Record::History(HistoryRecord {
            version: 18,
            unix: 1_784_000_000,
            heart_rate: Some(96),
            rr_intervals: vec![600, 610],
            ..Default::default()
        });
        let j = serde_json::to_string(&r).unwrap();
        assert!(j.contains(r#""kind":"History""#));
        assert!(j.contains(r#""heart_rate":96"#));
        assert!(j.contains(r#""rr_intervals":[600,610]"#));
    }
}
