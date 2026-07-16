//! WHOOP 4.0 historical records, decoded inner-relative (frame-absolute − 4). UNVERIFIED: not yet
//! exercised against a real 4.0 offload (the connect/bond path is 5/MG-only so far).

use super::{accept_gravity, gravity3, HistoryRecord};
use crate::bytes::{i16_at, nonzero_u8_at, rr_intervals, u16_at, unix_at};
use crate::packet::Frame;

/// v24 / v12 — full DSP block (HR, R-R, gravity, SpO2 red/IR, skin-temp, respiration).
pub fn v24(f: &Frame) -> Option<HistoryRecord> {
    let b = f.inner();
    let unix = unix_at(b)?;
    Some(HistoryRecord {
        version: f.version(),
        unix,
        heart_rate: nonzero_u8_at(b, 17),
        rr_intervals: rr_intervals(b, 18, 19, 4),
        gravity: gravity3(b, 36),
        skin_temp_c: i16_at(b, 68).map(|r| r as f32 * 0.04).filter(|c| (20.0..45.0).contains(c)),
        // Raw register, ungated: the °C scale is per-device on this generation, so the consumer owns it.
        skin_temp_raw: u16_at(b, 68),
        spo2: match (u16_at(b, 64), u16_at(b, 66)) {
            (Some(red), Some(ir)) => Some((red, ir)),
            _ => None,
        },
        resp_raw: u16_at(b, 76),
        ..Default::default()
    })
}

/// v5 / v7 / v9 — generic HR + R-R only, no DSP block.
pub fn v5(f: &Frame) -> Option<HistoryRecord> {
    let b = f.inner();
    let unix = unix_at(b)?;
    Some(HistoryRecord {
        version: f.version(),
        unix,
        heart_rate: nonzero_u8_at(b, 17),
        rr_intervals: rr_intervals(b, 18, 19, 4),
        ..Default::default()
    })
}

/// v25 — PPG waveform + gravity as i16/16384, no per-second HR.
pub fn v25(f: &Frame) -> Option<HistoryRecord> {
    let b = f.inner();
    let unix = unix_at(b)?;
    let g = [
        i16_at(b, 69)? as f32 / 16384.0,
        i16_at(b, 71)? as f32 / 16384.0,
        i16_at(b, 73)? as f32 / 16384.0,
    ];
    Some(HistoryRecord {
        version: 25,
        unix,
        gravity: accept_gravity(g),
        ..Default::default()
    })
}

#[cfg(test)]
mod tests {
    use crate::bytes::from_hex;
    use crate::family::Family;
    use crate::framing;
    use crate::records::{decode, Record};

    // A real on-wrist type-47 v24 record from a WHOOP 4.0: HR 109, two R-R, gravity ~1 g, skin-temp
    // register 861. A type-47 record carries no serial/name/token, so the frame is a safe fixture.
    const REAL_V24: &str = "aa6400a12f18054c1c0a023ed0266a5037805418016d022b0234020000000000006b07ff00\
        85593c1f65cebed7b3e63eb85a5f3f000080401f65cebed7b3e63eb85a5f3f500264025d03\
        640229014009010c020c00000000000f0001c4020000000000008fdeb278";

    #[test]
    fn v24_decodes_real_whoop4_hardware_record() {
        let wire = from_hex(REAL_V24).unwrap();
        let frame = framing::decode(Family::Gen4, &wire).unwrap();
        let r = match decode(&frame) {
            Some(Record::History(h)) => h,
            other => panic!("expected History, got {other:?}"),
        };
        assert_eq!(r.version, 24);
        assert_eq!(r.unix, 1_780_928_574);
        assert_eq!(r.heart_rate, Some(109));
        assert_eq!(r.rr_intervals, vec![555, 564]);
        let g = r.gravity.expect("gravity");
        let mag = (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]).sqrt();
        assert!((0.9..1.1).contains(&mag), "|g| = {mag}");
        // Raw register the app stores; the °C scale (per-device anchor) is applied app-side.
        assert_eq!(r.skin_temp_raw, Some(861));
        assert_eq!(r.spo2, Some((592, 612)));
    }
}
