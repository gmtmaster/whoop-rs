//! Haptics. The 5/MG one-shot buzz is the maverick opcode with the "notify" preset body. The Haptic
//! Clock turns a wall time into a deterministic buzz schedule (long pulse = a ten, short = a unit).

use crate::command;
use crate::family::Family;
use crate::framing;

/// A ready-to-write 5/MG one-shot buzz (RUN_HAPTIC_PATTERN_MAVERICK 0x13, the notification preset).
pub fn maverick_buzz_frame(seq: u8) -> Vec<u8> {
    let body = [0x01u8, 47, 152, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    framing::command(Family::Gen5, seq, command::RUN_HAPTIC_PATTERN_MAVERICK, &body)
}

/// RUN_HAPTICS_PATTERN body (5 bytes): `[pattern_id][loops][0][0][0]`. The 4.0 preset buzz (pattern 2 =
/// the graduated alarm buzz); on 5/MG the client remaps cmd 79 to the maverick opcode instead.
pub fn run_haptics_pattern(pattern_id: u8, loops: u8) -> [u8; 5] {
    [pattern_id, loops, 0, 0, 0]
}

// Haptic-Clock pulse/gap timing (ms). Long = a "tens" pulse, short = a "units" pulse.
const LONG_MS: u32 = 550;
const SHORT_MS: u32 = 200;
const INTRA_GAP_MS: u32 = 450;
const GROUP_GAP_MS: u32 = 900;
const BLOCK_GAP_MS: u32 = 1500;

/// One buzz instruction: buzz for `duration_ms`, then silence for `gap_ms`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pulse {
    pub duration_ms: u32,
    pub gap_ms: u32,
}

impl Pulse {
    pub fn is_long(&self) -> bool {
        self.duration_ms >= LONG_MS
    }
}

/// 24h hour → 12-hour dial reading (0→12, 13→1 … 23→11).
pub fn twelve_hour(h24: u32) -> u32 {
    let h = h24 % 12;
    if h == 0 { 12 } else { h }
}

/// Encode `hour:minute` into the buzz schedule (order: hour-tens, hour-units, minute-tens, minute-units).
pub fn pulses(hour: u32, minute: u32, is_24h: bool) -> Vec<Pulse> {
    let h24 = hour.min(23);
    let m = minute.min(59);
    let display_hour = if is_24h { h24 } else { twelve_hour(h24) };

    let mut out = Vec::new();
    append_group(&mut out, display_hour / 10, LONG_MS);
    close_group(&mut out, GROUP_GAP_MS);
    append_group(&mut out, display_hour % 10, SHORT_MS);
    close_group(&mut out, BLOCK_GAP_MS);
    append_group(&mut out, m / 10, LONG_MS);
    close_group(&mut out, GROUP_GAP_MS);
    append_group(&mut out, m % 10, SHORT_MS);

    if let Some(last) = out.last_mut() {
        last.gap_ms = 0; // end on a buzz
    }
    out
}

fn append_group(out: &mut Vec<Pulse>, count: u32, duration_ms: u32) {
    for _ in 0..count {
        out.push(Pulse { duration_ms, gap_ms: INTRA_GAP_MS });
    }
}

fn close_group(out: &mut [Pulse], gap_ms: u32) {
    if let Some(last) = out.last_mut() {
        last.gap_ms = last.gap_ms.max(gap_ms);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn three_twentyfive_24h() {
        // 3:25 → hour-units 3 (short×3), minute-tens 2 (long×2), minute-units 5 (short×5) = 10 pulses.
        let p = pulses(3, 25, true);
        assert_eq!(p.len(), 3 + 2 + 5);
        assert_eq!(p.last().unwrap().gap_ms, 0);
    }

    #[test]
    fn midnight_24h_is_silent() {
        assert!(pulses(0, 0, true).is_empty());
    }

    #[test]
    fn run_haptics_pattern_body() {
        assert_eq!(run_haptics_pattern(2, 3), [2, 3, 0, 0, 0]);
    }
}
