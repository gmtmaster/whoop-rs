//! WHOOP 4.0 historical records, decoded inner-relative (frame-absolute − 4). UNVERIFIED: not yet
//! exercised against a real 4.0 offload (the connect/bond path is 5/MG-only so far).

use super::{accept_gravity, gravity3, HistoryRecord};
use crate::bytes::{i16_at, rr_intervals, u16_at, u32_at, u8_at};
use crate::packet::Frame;

/// v24 / v12 — full DSP block (HR, R-R, gravity, SpO2 red/IR, skin-temp, respiration).
pub fn v24(f: &Frame) -> Option<HistoryRecord> {
    let b = f.inner();
    let unix = u32_at(b, 7)?;
    Some(HistoryRecord {
        version: f.version(),
        unix,
        heart_rate: u8_at(b, 17).filter(|&h| h > 0),
        rr_intervals: rr_intervals(b, 18, 19, 4),
        gravity: gravity3(b, 36),
        // 4.0 skin-temp scale not pinned — keep raw only (skin_temp_c defaults None), don't fabricate Celsius.
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
    let unix = u32_at(b, 7)?;
    Some(HistoryRecord {
        version: f.version(),
        unix,
        heart_rate: u8_at(b, 17).filter(|&h| h > 0),
        rr_intervals: rr_intervals(b, 18, 19, 4),
        ..Default::default()
    })
}

/// v25 — PPG waveform + gravity as i16/16384, no per-second HR.
pub fn v25(f: &Frame) -> Option<HistoryRecord> {
    let b = f.inner();
    let unix = u32_at(b, 7)?;
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
