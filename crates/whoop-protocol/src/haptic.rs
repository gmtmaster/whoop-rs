//! Haptics. The 5/MG one-shot buzz is the maverick opcode with the "notify" preset body; the 4.0 preset
//! buzz rides RUN_HAPTICS_PATTERN.

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_haptics_pattern_body() {
        assert_eq!(run_haptics_pattern(2, 3), [2, 3, 0, 0, 0]);
    }
}
