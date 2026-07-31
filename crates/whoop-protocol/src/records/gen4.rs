//! WHOOP 4.0 historical records, decoded inner-relative (frame-absolute − 4). v24 and v25 are pinned to
//! real captured 4.0 frames (`tests/fixtures/real_frames.json`); the desktop connect/bond path is still
//! 5/MG-only, so these frames reached the decoder from an app offload rather than from `whoop-client`.

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
