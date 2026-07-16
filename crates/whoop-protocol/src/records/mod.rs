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
    pub skin_temp_raw: Option<u16>, // raw register; the consumer applies the family/device-specific °C scale
    pub spo2: Option<(u16, u16)>, // 4.0 v24 raw red/IR ADC
    pub spo2_pct: Option<u8>,     // 5.0 v18 computed SpO2 %; sleep-only tri-mode byte, %-range only
    pub resp_raw: Option<u16>,    // 4.0 v24 raw respiration ADC
    pub steps: Option<u16>,       // 5.0 v18 cumulative counter
    pub activity_class: Option<u8>, // 5.0 v18 (0 still / 1 walk / 2 run)
    pub sleep_state: Option<u8>,  // 5.0 v18 packed state, provisional {0 still / 1 wake / 2 asleep / 3 up}
    pub signal_flags: Option<u8>, // 5.0 v18 PPG SIGPROC status bitfield (bit 4 = off-wrist); empirical
    pub signal_quality: Option<u8>, // 5.0 v18 PPG confidence, 255 = clean; empirical
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

// Real captured type-47 frames decoded through the public dispatcher and checked against pinned values —
// the cross-decoder drift guard on real hardware bytes. A type-47 record carries no serial/name/token.
#[cfg(test)]
mod real_frames {
    use crate::bytes::from_hex;
    use crate::family::Family;
    use crate::framing;
    use crate::records::{decode, HistoryRecord, Record};

    fn hist(family: Family, hex: &str) -> HistoryRecord {
        let frame = framing::decode(family, &from_hex(hex).unwrap()).expect("crc-valid frame");
        match decode(&frame) {
            Some(Record::History(h)) => h,
            other => panic!("expected History, got {other:?}"),
        }
    }

    fn g_mag(r: &HistoryRecord) -> f32 {
        let g = r.gravity.expect("gravity");
        (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]).sqrt()
    }

    #[test]
    fn v24_real_whoop4_worn() {
        let r = hist(Family::Gen4, "aa6400a12f18054c1c0a023ed0266a5037805418016d022b0234020000000000006b07ff00\
            85593c1f65cebed7b3e63eb85a5f3f000080401f65cebed7b3e63eb85a5f3f500264025d03640229014009010c020c00000000000f0001c4020000000000008fdeb278");
        assert_eq!(r.version, 24);
        assert_eq!(r.unix, 1_780_928_574);
        assert_eq!(r.heart_rate, Some(109));
        assert_eq!(r.rr_intervals, vec![555, 564]);
        assert!((0.9..1.1).contains(&g_mag(&r)), "|g| = {}", g_mag(&r));
        assert_eq!(r.skin_temp_raw, Some(861));
        assert_eq!(r.spo2, Some((592, 612)));
    }

    #[test]
    fn v25_real_whoop4_frames() {
        for (hex, unix) in [
            ("aa50000c2f1900006800007dff2a6a20430900433103007e026502ba026c022eff70f996f879fad6fd8300d6017e0267027201be00290258030e05c507f00c030ead11cb15791500d2553c9003000000d6393716", 1_781_202_813),
            ("aa50000c2f1900016800007eff2a6a283e0900a0ad03007a0e880698018bfff5fb61eee9f2a7fa2bfe1af5fdf618fdf0f9c2fb0804510a14046a004dffd0ff6dfdddfd670183014e071a3f9003000000587bbabf", 1_781_202_814),
            ("aa50000c2f1900026800007fff2a6a38390900729103003608a2fd0104850d4f1bd21aa60f080d850edb116b0f160b7d063f06ab04d5041704a4045f04f003f5ffd7ff7efe73ffa8b2333e9003010000fa54e5e9", 1_781_202_815),
        ] {
            let r = hist(Family::Gen4, hex);
            assert_eq!(r.version, 25);
            assert_eq!(r.unix, unix);
            assert_eq!(r.heart_rate, None); // v25 carries no per-second HR
            assert!((0.9..1.1).contains(&g_mag(&r)), "|g| = {}", g_mag(&r));
        }
    }

    #[test]
    fn v18_real_whoop5_worn() {
        let r = hist(Family::Gen5, "aa01740001003fb12f1280733d8401b69f266a66460066025a0265020000000000007b0a8d656463ff00\
            12163cf6a439bf2924fd3ed763fe3e3200aa000000000000000000f7000901f10b0007010c020c00000000000000000000000000000000000000000000000100656f1e1e0000009d61a7c00000003e862817");
        assert_eq!(r.version, 18);
        assert_eq!(r.unix, 1_780_916_150);
        assert_eq!(r.heart_rate, Some(102));
        assert_eq!(r.rr_intervals, vec![602, 613]);
        assert_eq!(r.skin_temp_raw, Some(3057)); // /100 = 30.57 °C
        assert_eq!(r.activity_class, Some(0)); // still
        assert_eq!(r.steps, Some(50));
        assert_eq!(r.sleep_state, Some(0));
    }

    #[test]
    fn v18_real_whoop5_offwrist_and_second_device() {
        let off = hist(Family::Gen5, "aa01740001003fb12f12803a3d84018889266a3d0a0000000000000000000000000000000000000000\
            0064c33b52b47d3fe1ba1dbda470ecbd000064000000000000000000e500e200c708000c010c020c0000000000000000000000000000000000000000000000010000008080000000000000000000009ffafe6c");
        assert_eq!(off.version, 18);
        assert_eq!(off.unix, 1_780_910_472);
        assert_eq!(off.heart_rate, None); // off-wrist, hr byte 0
        assert_eq!(off.skin_temp_raw, Some(2247));

        let dev2 = hist(Family::Gen5, "aa01740001003fb12f128093c47c006dbc296a00600039000000000000000000006137020b610000\
            e1e04c063d8fce36bf7b08233f8fea993e38a50000000000000000000019012101920b5002010c020c0100000000000000000000000000000000000000000005010085808080000000a5538ec000000016d0680d");
        assert_eq!(dev2.version, 18);
        assert_eq!(dev2.unix, 1_781_120_109);
        assert_eq!(dev2.heart_rate, Some(57));
        assert_eq!(dev2.skin_temp_raw, Some(2962));
    }
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
