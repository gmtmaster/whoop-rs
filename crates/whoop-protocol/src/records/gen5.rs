//! WHOOP 5.0/MG historical records, decoded inner-relative (frame-absolute − 8). v18 carries a sleep-only
//! SpO2 % byte at inner 74 (tri-mode: %-range kept, sentinels/codes dropped).

use super::{gravity3, HistoryRecord, ImuRecord, PpgRecord};
use crate::bytes::{i16_at, nonzero_u8_at, rr_intervals, u16_at, u8_at, unix_at};
use crate::packet::Frame;

/// Samples per axis in a v21 IMU buffer (both the accel and gyro count fields carry this). Reused as
/// `sample_rate_hz`: the buffer's time span is unconfirmed, so 100 Hz is inferred from the count, not measured.
const IMU_SAMPLES: usize = 100;

pub fn v18(f: &Frame) -> Option<HistoryRecord> {
    let b = f.inner();
    let unix = unix_at(b)?;
    Some(HistoryRecord {
        version: 18,
        unix,
        heart_rate: nonzero_u8_at(b, 14),
        rr_intervals: rr_intervals(b, 15, 16, 4),
        gravity: gravity3(b, 37),
        skin_temp_c: u16_at(b, 65).map(|r| r as f32 / 100.0).filter(|c| (5.0..45.0).contains(c)),
        // Sleep-only tri-mode byte: a %-range value is a real SpO2; bit-7 sentinels and sub-70 codes → None.
        spo2_pct: u8_at(b, 74).filter(|&v| (70..=100).contains(&v)),
        steps: u16_at(b, 49),
        activity_class: u8_at(b, 55),
        sleep_state: u8_at(b, 73).map(|v| (v >> 4) & 3),
        signal_flags: u8_at(b, 25),
        signal_quality: u8_at(b, 32),
        ..Default::default()
    })
}

pub fn v26(f: &Frame) -> Option<PpgRecord> {
    let b = f.inner();
    let unix = unix_at(b)?;
    let record_id = u16_at(b, 3); // strap record counter, not a wavelength channel
    let mut samples = Vec::with_capacity(24);
    for i in 0..24 {
        samples.push(i16_at(b, 19 + i * 2)?);
    }
    Some(PpgRecord { version: 26, unix, record_id, samples })
}

/// v21 — the 100 Hz raw 6-axis IMU offload buffer. Columnar i16 LE inner offsets: unix@7, count_a@16,
/// ax@20 ay@220 az@420, count_b@622, gx@632 gy@832 gz@1032. Gated on both counts (=100), not the version
/// byte, so it can't misfire; a short buffer fails the trailing reads.
pub fn v21_imu(f: &Frame) -> Option<ImuRecord> {
    let b = f.inner();
    if u16_at(b, 16)? != IMU_SAMPLES as u16 || u16_at(b, 622)? != IMU_SAMPLES as u16 {
        return None;
    }
    let unix = unix_at(b)?;
    let mut accel = Vec::with_capacity(IMU_SAMPLES);
    let mut gyro = Vec::with_capacity(IMU_SAMPLES);
    for i in 0..IMU_SAMPLES {
        let o = i * 2;
        accel.push([i16_at(b, 20 + o)?, i16_at(b, 220 + o)?, i16_at(b, 420 + o)?]);
        gyro.push([i16_at(b, 632 + o)?, i16_at(b, 832 + o)?, i16_at(b, 1032 + o)?]);
    }
    Some(ImuRecord { version: f.version(), unix, sample_rate_hz: IMU_SAMPLES as u16, accel, gyro })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::family::Family;
    use crate::framing;

    #[test]
    fn v18_decodes_hr_rr_gravity() {
        // Build a synthetic v18: type 47, version 18. Inner-relative: unix@7, hr@14, rr_count@15, rr@16,
        // gravity@37. Payload = inner[3..], so payload index = inner offset - 3.
        let mut payload = vec![0u8; 80];
        payload[7 - 3..11 - 3].copy_from_slice(&1_784_000_000u32.to_le_bytes()); // unix @ inner 7
        payload[14 - 3] = 96; // hr @ inner 14
        payload[15 - 3] = 1; // rr_count @ inner 15
        payload[16 - 3..18 - 3].copy_from_slice(&600u16.to_le_bytes()); // rr[0] @ inner 16
        payload[37 - 3..41 - 3].copy_from_slice(&0.1f32.to_le_bytes()); // gx
        payload[41 - 3..45 - 3].copy_from_slice(&0.0f32.to_le_bytes()); // gy
        payload[45 - 3..49 - 3].copy_from_slice(&0.99f32.to_le_bytes()); // gz  (|g| ≈ 1)
        payload[74 - 3] = 97; // spo2 @ inner 74 (a valid %)

        let wire = framing::encode(Family::Gen5, 47, 18, 0, &payload);
        let frame = framing::decode(Family::Gen5, &wire).unwrap();
        let r = v18(&frame).unwrap();
        assert_eq!(r.heart_rate, Some(96));
        assert_eq!(r.rr_intervals, vec![600]);
        assert!(r.gravity.is_some());
        assert_eq!(r.unix, 1_784_000_000);
        assert_eq!(r.spo2_pct, Some(97));
    }

    #[test]
    fn v18_spo2_tri_mode_gates_sentinels_and_codes() {
        let decode = |byte74: u8| {
            let mut payload = vec![0u8; 80];
            payload[74 - 3] = byte74;
            let wire = framing::encode(Family::Gen5, 47, 18, 0, &payload);
            v18(&framing::decode(Family::Gen5, &wire).unwrap()).unwrap().spo2_pct
        };
        assert_eq!(decode(98), Some(98)); // real percentage
        assert_eq!(decode(8), None); // low diagnostic code
        assert_eq!(decode(0xA8), None); // bit-7 saturation sentinel
        assert_eq!(decode(0), None); // no reading
    }

    #[test]
    fn v18_decodes_signal_flags_and_quality() {
        let mut payload = vec![0u8; 80];
        payload[25 - 3] = 0x10; // signal_flags @ inner 25 (bit 4 = off-wrist)
        payload[32 - 3] = 255; // signal_quality @ inner 32 (clean)
        let wire = framing::encode(Family::Gen5, 47, 18, 0, &payload);
        let r = v18(&framing::decode(Family::Gen5, &wire).unwrap()).unwrap();
        assert_eq!(r.signal_flags, Some(0x10));
        assert_eq!(r.signal_quality, Some(255));
    }

    #[test]
    fn v21_imu_decodes_100_sample_6axis() {
        // payload index = inner offset − 3. inner = 3 + 1229 = 1232 = gz end (already 4-aligned).
        let mut payload = vec![0u8; 1229];
        payload[4..8].copy_from_slice(&1_784_000_000u32.to_le_bytes()); // unix @ inner 7
        payload[13..15].copy_from_slice(&100u16.to_le_bytes()); // count_a @ inner 16
        payload[17..19].copy_from_slice(&4096i16.to_le_bytes()); // ax[0] @ inner 20 (= 1 g)
        payload[619..621].copy_from_slice(&100u16.to_le_bytes()); // count_b @ inner 622
        payload[629..631].copy_from_slice(&250i16.to_le_bytes()); // gx[0] @ inner 632

        let wire = framing::encode(Family::Gen5, 47, 21, 0, &payload);
        let frame = framing::decode(Family::Gen5, &wire).unwrap();

        let r = v21_imu(&frame).unwrap();
        assert_eq!(r.unix, 1_784_000_000);
        assert_eq!(r.sample_rate_hz, 100);
        assert_eq!(r.accel.len(), 100);
        assert_eq!(r.gyro.len(), 100);
        assert_eq!(r.accel[0], [4096, 0, 0]);
        assert_eq!(r.gyro[0], [250, 0, 0]);

        // An unmapped GEN5 version routes here through the public dispatcher.
        assert!(matches!(crate::records::decode(&frame), Some(crate::records::Record::Imu(_))));
    }

    #[test]
    fn v21_imu_rejects_wrong_sample_count() {
        let mut payload = vec![0u8; 1229];
        payload[13..15].copy_from_slice(&99u16.to_le_bytes()); // count_a != 100
        payload[619..621].copy_from_slice(&100u16.to_le_bytes());
        let wire = framing::encode(Family::Gen5, 47, 21, 0, &payload);
        let frame = framing::decode(Family::Gen5, &wire).unwrap();
        assert!(v21_imu(&frame).is_none());
    }

    #[test]
    fn imu_buffer_decodes_regardless_of_version_byte() {
        // A buffer whose version byte collides with v18 still routes to Imu via the count gate, not v18.
        let mut payload = vec![0u8; 1229];
        payload[13..15].copy_from_slice(&100u16.to_le_bytes()); // count_a @ inner 16
        payload[619..621].copy_from_slice(&100u16.to_le_bytes()); // count_b @ inner 622
        let wire = framing::encode(Family::Gen5, 47, 18, 0, &payload); // version byte = 18
        let frame = framing::decode(Family::Gen5, &wire).unwrap();
        assert!(matches!(crate::records::decode(&frame), Some(crate::records::Record::Imu(_))));
    }
}
