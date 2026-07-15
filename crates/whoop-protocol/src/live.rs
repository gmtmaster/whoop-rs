//! Live (non-historical) decoders: REALTIME_DATA, EVENT, and the METADATA end_data the offload ACK
//! echoes. Inner-relative offsets, so one decoder serves both generations.

use crate::bytes::{f32_at, nonzero_u8_at, rr_intervals, u16_at, u32_at, u8_at, unix_at};
use crate::packet::Frame;

#[derive(Clone, Debug, PartialEq)]
pub struct Realtime {
    pub timestamp: u32,
    pub heart_rate: u8,
    pub rr_intervals: Vec<u16>,
}

/// REALTIME_DATA (type 40): timestamp (device epoch) @ inner 2, hr @ 8, rr_count @ 9, rr @ 10+.
/// The realtime burst is unbounded (unlike the 4-slot historical layout), so no slot cap.
pub fn realtime(f: &Frame) -> Option<Realtime> {
    let b = f.inner();
    let timestamp = u32_at(b, 2)?;
    let heart_rate = u8_at(b, 8)?;
    Some(Realtime { timestamp, heart_rate, rr_intervals: rr_intervals(b, 9, 10, usize::MAX) })
}

#[derive(Clone, Debug, PartialEq)]
pub struct R22Live {
    pub timestamp: u32,
    pub hr_ch1: u8,
    pub hr_ch2: Option<u8>,
    pub accel: [f32; 3],
}

/// Live r22 biometric (type 47, cmd 0x80/0x82, 112-byte): device ts @ inner 7, HR ch1 @ 14, HR ch2
/// (hr_ch_switching) @ 29, accel x/y/z f32 @ 37/41/45. Forked on the cmd byte, not the version byte, and
/// only flows on-wrist — off the historical (version-keyed) path.
pub fn r22_live(f: &Frame) -> Option<R22Live> {
    let b = f.inner();
    Some(R22Live {
        timestamp: unix_at(b)?,
        hr_ch1: u8_at(b, 14)?,
        hr_ch2: nonzero_u8_at(b, 29),
        accel: [f32_at(b, 37)?, f32_at(b, 41)?, f32_at(b, 45)?],
    })
}

#[derive(Clone, Debug, PartialEq)]
pub struct Event {
    pub number: u8,
    pub timestamp: u32,
}

/// EVENT (type 48): event number is the inner cmd byte, timestamp @ inner 4 (real unix).
pub fn event(f: &Frame) -> Option<Event> {
    Some(Event { number: f.cmd(), timestamp: u32_at(f.inner(), 4)? })
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BatteryEvent {
    pub soc_percent: f32,
    pub millivolts: u16,
    pub charging: bool,
}

/// The BATTERY_LEVEL event body: soc deci-% @ inner 13, mV @ inner 17, charging bit0 @ inner 22.
pub fn battery_event(f: &Frame) -> Option<BatteryEvent> {
    if f.cmd() != crate::event::BATTERY_LEVEL {
        return None;
    }
    let b = f.inner();
    Some(BatteryEvent {
        soc_percent: u16_at(b, 13)? as f32 / 10.0,
        millivolts: u16_at(b, 17)?,
        charging: u8_at(b, 22).is_some_and(|v| v & 1 != 0),
    })
}

/// METADATA HISTORY_END 8-byte end_data (trim u32 + next u32) @ inner 13..21 — echoed verbatim in the ACK.
pub fn history_end_data(f: &Frame) -> Option<[u8; 8]> {
    let s = f.inner().get(13..21)?;
    let mut ed = [0u8; 8];
    ed.copy_from_slice(s);
    Some(ed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::family::Family;
    use crate::framing;
    use crate::packet::PacketType;

    #[test]
    fn r22_live_reads_hr_and_accel() {
        // payload = inner[3..]; so inner 7/14/29/37 map to payload 4/11/26/34.
        let mut payload = vec![0u8; 60];
        payload[4..8].copy_from_slice(&1_784_000_000u32.to_le_bytes());
        payload[11] = 98;
        payload[26] = 99;
        payload[34..38].copy_from_slice(&0.10f32.to_le_bytes());
        payload[38..42].copy_from_slice(&0.20f32.to_le_bytes());
        payload[42..46].copy_from_slice(&0.98f32.to_le_bytes());
        let wire = framing::encode(Family::Gen5, PacketType::HistoricalData.to_u8(), 0, 0x80, &payload);
        let f = framing::decode(Family::Gen5, &wire).unwrap();
        let r = r22_live(&f).unwrap();
        assert_eq!(r.timestamp, 1_784_000_000);
        assert_eq!(r.hr_ch1, 98);
        assert_eq!(r.hr_ch2, Some(99));
        assert!((r.accel[2] - 0.98).abs() < 1e-6);
    }
}
