//! Type-47 HISTORICAL_DATA decoders, dispatched by the version byte (version → generation is
//! many-to-one). Offsets are INNER-relative (frame-absolute − inner_start), which is why the GEN4/GEN5
//! +4 delta disappears. Every field is range-gated so a wrong offset on unmapped firmware yields `None`.

mod gen4;
mod gen5;

use crate::bytes::f32_at;
use crate::family::Family;
use crate::packet::Frame;

/// Inner offsets of the v18 per-second bytes that carry information but have no established meaning,
/// in the order [`HistoryRecord::unpinned`] packs them. Every other varying byte in the record is
/// already a named field.
pub const UNPINNED_OFFSETS: [usize; 13] = [2, 11, 12, 24, 51, 67, 68, 69, 70, 71, 72, 95, 97];

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
    pub skin_temp_raw: Option<u16>, // raw register; the consumer applies the family/device-specific °C scale
    pub spo2: Option<(u16, u16)>, // 4.0 v24 raw red/IR ADC
    pub spo2_pct: Option<u8>,     // 5.0 v18 computed SpO2 %; sleep-only tri-mode byte, %-range only
    pub resp_raw: Option<u16>,    // 4.0 v24 raw respiration ADC
    pub steps: Option<u16>,       // 5.0 v18 cumulative counter
    pub activity_class: Option<u8>, // 5.0 v18 (0 still / 1 walk / 2 run)
    pub sleep_state: Option<u8>,  // 5.0 v18 packed state, provisional {0 still / 1 wake / 2 asleep / 3 up}
    pub signal_flags: Option<u8>, // 5.0 v18 PPG SIGPROC status bitfield (bit 4 = off-wrist); empirical
    pub signal_quality: Option<u8>, // 5.0 v18 PPG confidence, 255 = clean; empirical
    pub dynamic_acceleration_g: Option<f32>, // 5.0 v18 on-chip gravity-removed motion magnitude (g, 0..8); empirical
    // 5.0 v18 optical front-end telemetry. Two independent u8 baselines and two u8 amplitudes, NOT
    // two u16s: the high byte steps without a low-byte carry, so a u16 read invents deltas.
    pub optical_baseline_a: Option<u8>, // 0 marks off-wrist; magnitudes are device-specific
    pub optical_baseline_b: Option<u8>,
    pub optical_amp_a: Option<u8>, // 128 is a quality sentinel, not a reading; see optical_signal_poor
    pub optical_amp_b: Option<u8>,
    /// The record-level 128 sentinel on both amplitude channels: the band's own beat detection is
    /// unreliable for this second. Fires on both or neither, never one alone.
    pub optical_signal_poor: Option<bool>,
    /// 5.0 v18 dense per-second emission counter, one step per unix second. Orders and de-duplicates
    /// records independently of arrival, which no timestamp can do within a second.
    pub record_index: Option<u32>,
    /// 5.0 v18 auxiliary thermal registers in deci-degrees C, raw like [`Self::skin_temp_raw`] which is
    /// centi-degrees. Both read below the skin channel worn, and all three converge off-wrist.
    pub temp_aux_1_raw: Option<u16>,
    pub temp_aux_2_raw: Option<u16>,
    /// 5.0 v18 whole packed state byte at inner 73; [`Self::sleep_state`] is bits 4-5 of it.
    pub sleep_state_raw: Option<u8>,
    // 5.0 v18 fields carried every second whose meaning is unpinned. Named for their inner offset so
    // the name claims nothing, and gated only on being readable so nothing real is filtered away.
    // Inner 28/29 are a pair; reading them as one u16 scaled by 256 tracks the rate on some straps
    // and not others, so neither byte is promoted here.
    pub raw_u8_28: Option<u8>,
    pub raw_u8_29: Option<u8>,
    pub raw_u16_30: Option<u16>,
    pub raw_f32_105: Option<f32>,
    /// The widest-ranging unpinned channel: thousands of distinct values over a night, against tens
    /// for the bytes beside it.
    pub raw_u16_26: Option<u16>,
    /// Every remaining per-second byte that carries information, packed in [`UNPINNED_OFFSETS`] order.
    /// Opaque by design — the consumer stores it whole, and a later census unpacks it here.
    pub unpinned: Option<Vec<u8>>,
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

/// A raw multi-wavelength optical offload buffer (v20 on 5.0/MG): ~25 Hz across 6 photodiode
/// channels, shipped in the R22 deep buffers. `channels` is 6 × 25 raw 20-bit signed ADC counts.
/// Channel roles (upstream RE, partly inferred): [0,1] green (carry the rest cardiac pulse),
/// [2,3] ambient/dark reference, [4,5] further optical (red/IR inferred, unproven). SpO2 % is NOT
/// derived here — the strap's own computed value rides v18 (`HistoryRecord.spo2_pct`).
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct OpticalRecord {
    pub version: u8,
    pub unix: u32,
    pub sample_rate_hz: u16,
    pub channels: Vec<Vec<i32>>,
}

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
#[cfg_attr(feature = "serde", serde(tag = "kind"))]
pub enum Record {
    History(HistoryRecord),
    Ppg(PpgRecord),
    Imu(ImuRecord),
    Optical(OpticalRecord),
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
        (Family::Gen5, 20) => gen5::v20_optical(frame).map(Record::Optical),
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

/// Single-frame v26 24 Hz PPG waveform decode (5.0/MG). `None` off-generation or on a non-v26 frame —
/// the offload path yields these via `decode`, this is the single-frame door the FFI needs.
pub fn decode_ppg(frame: &Frame) -> Option<PpgRecord> {
    (frame.family == Family::Gen5 && frame.version() == 26).then(|| gen5::v26(frame)).flatten()
}

/// Single-frame v21 100 Hz 6-axis IMU decode (5.0/MG). `None` off-generation or if the sample-count gate
/// fails. Version-independent (gated on the in-packet counts), matching `decode`'s IMU-first dispatch.
pub fn decode_imu(frame: &Frame) -> Option<ImuRecord> {
    (frame.family == Family::Gen5).then(|| gen5::v21_imu(frame)).flatten()
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
