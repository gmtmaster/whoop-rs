//! WHOOP 5.0/MG historical records, decoded inner-relative (frame-absolute − 8). v18 carries a sleep-only
//! SpO2 % byte at inner 74 (tri-mode: %-range kept, sentinels/codes dropped).

use super::{gravity3, HistoryRecord, ImuRecord, OpticalRecord, PpgRecord};
use crate::bytes::{f32_at, i16_at, nonzero_u8_at, rr_intervals, u16_at, u32_at, u8_at, unix_at};
use crate::packet::Frame;

/// Samples per axis in a v21 IMU buffer (both the accel and gyro count fields carry this). Reused as
/// `sample_rate_hz`: the buffer's time span is unconfirmed, so 100 Hz is inferred from the count, not measured.
const IMU_SAMPLES: usize = 100;

/// v20 optical buffer: 25 samples per channel, 6 channels, at these inner offsets (frame − 8).
const OPTICAL_SAMPLES: usize = 25;
const OPTICAL_CHANNELS: [usize; 6] = [39, 239, 1305, 1505, 1727, 1927];

pub fn v18(f: &Frame) -> Option<HistoryRecord> {
    let b = f.inner();
    let unix = unix_at(b)?;
    // Raw skin-temp register, kept only in the physiological band (°C = raw/100 in 5..45); the consumer
    // owns the final scale.
    let skin_temp_raw = u16_at(b, 65).filter(|&r| (5.0..45.0).contains(&(r as f32 / 100.0)));
    Some(HistoryRecord {
        version: 18,
        unix,
        heart_rate: nonzero_u8_at(b, 14),
        rr_intervals: rr_intervals(b, 15, 16, 4),
        gravity: gravity3(b, 37),
        skin_temp_c: skin_temp_raw.map(|r| r as f32 / 100.0),
        skin_temp_raw,
        // Sleep-only tri-mode byte: a %-range value is a real SpO2; bit-7 sentinels and sub-70 codes → None.
        spo2_pct: u8_at(b, 74).filter(|&v| (70..=100).contains(&v)),
        steps: u16_at(b, 49),
        // Only the mapped {0,1,2} codes; a wrong offset / invalid sentinel (0xFF) stores nothing.
        activity_class: u8_at(b, 55).filter(|&v| v <= 2),
        sleep_state: u8_at(b, 73).map(|v| (v >> 4) & 3),
        signal_flags: u8_at(b, 25),
        signal_quality: u8_at(b, 32),
        // On-chip gravity-removed motion magnitude (g), 1 Hz; gated finite and in 0..8 g so a wrong
        // offset stores nothing. Feeds the circadian activity series (CosinorAge).
        dynamic_acceleration_g: f32_at(b, 33).filter(|v| v.is_finite() && (0.0..=8.0).contains(v)),
        // Optical front-end telemetry. Baselines read 0 together when the strap is off the wrist;
        // 128 on both amplitudes is a quality sentinel rather than a magnitude, so it is reported
        // as `optical_signal_poor` and withheld from the amplitude channels.
        optical_baseline_a: u8_at(b, 98).filter(|&v| v != 0),
        optical_baseline_b: u8_at(b, 99).filter(|&v| v != 0),
        optical_amp_a: u8_at(b, 100).filter(|_| !amps_sentinel(b)),
        optical_amp_b: u8_at(b, 101).filter(|_| !amps_sentinel(b)),
        optical_signal_poor: u8_at(b, 100).zip(u8_at(b, 101)).map(|_| amps_sentinel(b)),
        record_index: u32_at(b, 3),
        temp_aux_1_raw: u16_at(b, 61),
        temp_aux_2_raw: u16_at(b, 63),
        sleep_state_raw: u8_at(b, 73),
        raw_u8_28: u8_at(b, 28),
        raw_u8_29: u8_at(b, 29),
        raw_u16_30: u16_at(b, 30),
        raw_f32_105: f32_at(b, 105).filter(|v| v.is_finite()),
        raw_u16_26: u16_at(b, 26),
        unpinned: pack_unpinned(b),
        ..Default::default()
    })
}

/// The unpinned bytes gathered in [`UNPINNED_OFFSETS`] order, or None when the record is too short to
/// carry them all — a partial pack would silently shift every slot.
fn pack_unpinned(b: &[u8]) -> Option<Vec<u8>> {
    super::UNPINNED_OFFSETS.iter().map(|&o| u8_at(b, o)).collect()
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

/// v20 — the ~25 Hz raw 6-channel optical offload buffer. Each channel is 25 samples × 4-byte LE,
/// 20-bit signed. Gated on the config anchor that holds for every real buffer: the 2×-green-LED echo
/// (inner 23 == 2 × inner 20, green nonzero), so a wrong-length or misrouted frame drops.
pub fn v20_optical(f: &Frame) -> Option<OpticalRecord> {
    let b = f.inner();
    let unix = unix_at(b)?;
    let green = u16_at(b, 20)?;
    if green == 0 || u16_at(b, 23)? != green.wrapping_mul(2) {
        return None;
    }
    let mut channels = Vec::with_capacity(OPTICAL_CHANNELS.len());
    for &off in &OPTICAL_CHANNELS {
        let mut ch = Vec::with_capacity(OPTICAL_SAMPLES);
        for s in 0..OPTICAL_SAMPLES {
            ch.push(sign_extend_20(u32_at(b, off + s * 4)?));
        }
        channels.push(ch);
    }
    Some(OpticalRecord { version: 20, unix, sample_rate_hz: OPTICAL_SAMPLES as u16, channels })
}

/// Sign-extend a 20-bit ADC sample carried in a 4-byte LE word.
fn sign_extend_20(v: u32) -> i32 {
    ((v << 12) as i32) >> 12
}

/// The v18 optical amplitude sentinel: both channels reading 128 marks a second whose beat detection
/// the band could not trust. It is record-level — the two bytes carry it together or not at all.
fn amps_sentinel(b: &[u8]) -> bool {
    matches!((u8_at(b, 100), u8_at(b, 101)), (Some(128), Some(128)))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The amplitude sentinel is record-level: both bytes or neither. The two channels do vary apart
    /// (64% of captured records), yet exactly one reading 128 has never occurred in 82,185 of them.
    #[test]
    fn amps_sentinel_needs_both_channels() {
        let mut b = [7u8; 110];
        b[100] = 128;
        b[101] = 128;
        assert!(amps_sentinel(&b), "both 128 is the sentinel");

        b[101] = 30;
        assert!(!amps_sentinel(&b), "one channel alone must not fire it");

        b[100] = 30;
        b[101] = 128;
        assert!(!amps_sentinel(&b), "the other channel alone must not fire it either");
    }
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
    fn v18_skin_temp_raw_kept_in_band_dropped_out_of_band() {
        let decode = |raw: u16| {
            let mut payload = vec![0u8; 80];
            payload[65 - 3..67 - 3].copy_from_slice(&raw.to_le_bytes()); // skin temp @ inner 65
            let wire = framing::encode(Family::Gen5, 47, 18, 0, &payload);
            v18(&framing::decode(Family::Gen5, &wire).unwrap()).unwrap()
        };
        let worn = decode(3345); // 33.45 °C
        assert_eq!(worn.skin_temp_raw, Some(3345));
        assert_eq!(worn.skin_temp_c, Some(33.45));
        assert_eq!(decode(300).skin_temp_raw, None); // 3.0 °C, below the 5 °C floor
        assert_eq!(decode(5000).skin_temp_raw, None); // 50 °C, above the 45 °C ceiling
    }

    #[test]
    fn v18_activity_class_gates_out_invalid_codes() {
        let decode = |b: u8| {
            let mut payload = vec![0u8; 80];
            payload[55 - 3] = b; // activity_class @ inner 55
            let wire = framing::encode(Family::Gen5, 47, 18, 0, &payload);
            v18(&framing::decode(Family::Gen5, &wire).unwrap()).unwrap().activity_class
        };
        assert_eq!(decode(0), Some(0)); // still
        assert_eq!(decode(2), Some(2)); // run
        assert_eq!(decode(0xFF), None); // invalid sentinel
        assert_eq!(decode(7), None); // unmapped code
    }

    #[test]
    fn v18_decodes_dynamic_acceleration_gated() {
        // dynamic_acceleration f32 @ inner 33 (frame 41); gravity is a separate vector @ inner 37/41/45.
        let decode = |g: f32| {
            let mut payload = vec![0u8; 80];
            payload[33 - 3..37 - 3].copy_from_slice(&g.to_le_bytes());
            let wire = framing::encode(Family::Gen5, 47, 18, 0, &payload);
            v18(&framing::decode(Family::Gen5, &wire).unwrap()).unwrap().dynamic_acceleration_g
        };
        assert_eq!(decode(0.014), Some(0.014)); // real rest-floor motion
        assert_eq!(decode(3.2), Some(3.2)); // active
        assert_eq!(decode(9.0), None); // above the 0..8 g band
        assert_eq!(decode(-1.0), None); // impossible negative
        assert!(decode(f32::NAN).is_none()); // non-finite
    }

    /// One real offloaded v18 second, pinning every per-second field the record carries. Inner 28 and
    /// 29 are two separate bytes, not one 8.8 fixed-point rate: read as a u16 scaled by 256 the whole
    /// part is just inner 29 and the fraction just inner 28, so the pairing adds nothing to either.
    #[test]
    fn v18_decodes_a_real_frame_end_to_end() {
        const WIRE: &str = "aa01740001003fb12f128066b7760180fc546aeb710056011d0200000000000000002c0481555700\
ffb063f13d852b853db87ef43d298e803f7a01a100000000000000000051015901db0d6006010c020c000000000000000000\
00000000000000000000000000000100adb18080000000c8d69bc00000000d27d7e3";
        let wire: Vec<u8> = (0..WIRE.len() / 2)
            .map(|i| u8::from_str_radix(&WIRE[i * 2..i * 2 + 2], 16).unwrap())
            .collect();
        let r = v18(&framing::decode(Family::Gen5, &wire).unwrap()).unwrap();

        assert_eq!(r.unix, 1_783_954_560);
        assert_eq!(r.record_index, Some(24_557_414));
        assert_eq!(r.heart_rate, Some(86));
        assert_eq!(r.rr_intervals, vec![541]);
        assert_eq!(r.raw_u8_28, Some(129));
        assert_eq!(r.raw_u8_29, Some(85));
        assert_eq!(r.raw_u16_30, Some(87));
        assert_eq!(r.temp_aux_1_raw, Some(337));
        assert_eq!(r.temp_aux_2_raw, Some(345));
        assert_eq!(r.skin_temp_raw, Some(3547));
        assert_eq!(r.skin_temp_c, Some(35.47));
        assert_eq!(r.sleep_state_raw, Some(0));
        assert_eq!(r.sleep_state, Some(0));
        assert_eq!(r.raw_f32_105, Some(-4.869_968_4));
        assert_eq!(r.raw_u16_26, Some(1068));
        // Packed in offset order, so slot i is UNPINNED_OFFSETS[i]'s byte.
        let packed = r.unpinned.clone().unwrap();
        assert_eq!(packed.len(), super::super::UNPINNED_OFFSETS.len());
        for (slot, &off) in packed.iter().zip(super::super::UNPINNED_OFFSETS.iter()) {
            assert_eq!(*slot, framing::decode(Family::Gen5, &wire).unwrap().inner()[off]);
        }

        // The 8.8 reading decomposes exactly back into the two bytes, on any record: no third value
        // exists to recover, which is why neither byte is exposed as a rate.
        let fixed_8_8 = (u16::from(r.raw_u8_29.unwrap()) * 256 + u16::from(r.raw_u8_28.unwrap())) as f32 / 256.0;
        assert_eq!(fixed_8_8, 85.503_91);
        assert_eq!(fixed_8_8.trunc() as u8, r.raw_u8_29.unwrap());
        assert_eq!(((fixed_8_8.fract() * 256.0).round()) as u8, r.raw_u8_28.unwrap());
    }

    /// The auxiliary channels are ungated, but a truncated record must not invent them.
    #[test]
    fn v18_aux_fields_absent_on_a_short_record() {
        let payload = vec![0u8; 40];
        let wire = framing::encode(Family::Gen5, 47, 18, 0, &payload);
        let r = v18(&framing::decode(Family::Gen5, &wire).unwrap()).unwrap();
        assert_eq!(r.temp_aux_1_raw, None);
        assert_eq!(r.sleep_state_raw, None);
        assert_eq!(r.raw_f32_105, None);
        assert_eq!(r.record_index, Some(0)); // inner 3 is within even a short record
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

    #[test]
    fn v20_optical_decodes_6_channels_25_samples() {
        // payload index = inner offset − 3; inner = 2128 (frame 2140 − 8 header − 4 CRC).
        let mut payload = vec![0u8; 2125];
        payload[4..8].copy_from_slice(&1_784_000_000u32.to_le_bytes()); // unix @ inner 7
        payload[17..19].copy_from_slice(&1400u16.to_le_bytes()); // green LED @ inner 20
        payload[20..22].copy_from_slice(&2800u16.to_le_bytes()); // 2×green echo @ inner 23
        payload[36..40].copy_from_slice(&12345u32.to_le_bytes()); // ch0[0] @ inner 39
        payload[40..44].copy_from_slice(&0x000F_FFFBu32.to_le_bytes()); // ch0[1] = −5 (20-bit signed)

        let wire = framing::encode(Family::Gen5, 47, 20, 0, &payload);
        let frame = framing::decode(Family::Gen5, &wire).unwrap();
        let r = v20_optical(&frame).unwrap();
        assert_eq!(r.unix, 1_784_000_000);
        assert_eq!(r.sample_rate_hz, 25);
        assert_eq!(r.channels.len(), 6);
        assert!(r.channels.iter().all(|c| c.len() == 25));
        assert_eq!(r.channels[0][0], 12345);
        assert_eq!(r.channels[0][1], -5);
        assert!(matches!(crate::records::decode(&frame), Some(crate::records::Record::Optical(_))));
    }

    #[test]
    fn v20_optical_rejects_without_led_anchor() {
        let mut payload = vec![0u8; 2125];
        payload[4..8].copy_from_slice(&1_784_000_000u32.to_le_bytes());
        payload[17..19].copy_from_slice(&1400u16.to_le_bytes()); // green
        payload[20..22].copy_from_slice(&999u16.to_le_bytes()); // != 2×green → reject
        let wire = framing::encode(Family::Gen5, 47, 20, 0, &payload);
        let frame = framing::decode(Family::Gen5, &wire).unwrap();
        assert!(v20_optical(&frame).is_none());
    }
}

