//! WHOOP 5.0/MG wake-alarm (SET_ALARM_TIME) REVISION_4 body. EXPERIMENTAL/UNCONFIRMED — not confirmed to
//! wake a strap. All multi-byte fields little-endian; timezone→epoch resolution stays in the client.

const OVERALL_LOOP: u8 = 7;
const DURATION_SECONDS: u8 = 30;
/// The canonical WHOOP wake waveform-effect pair (the same 47/152 the notification buzz uses).
const WAVEFORM_EFFECTS: [u8; 8] = [47, 152, 0, 0, 0, 0, 0, 0];

/// SET_ALARM_TIME REVISION_4 body (20 bytes):
///   [0]=0x04  [1]=alarmId  [2..6]=u32 LE epoch s  [6..8]=u16 LE subseconds (ms*32768/1000)
///   [8..16]=8 effects  [16..18]=u16 LE loopControl(0)  [18]=overallLoop(7)  [19]=duration(30)
pub fn build(wake_epoch_ms: u64, alarm_id: u8) -> [u8; 20] {
    let seconds = (wake_epoch_ms / 1000) as u32;
    let subseconds = (((wake_epoch_ms % 1000) * 32768) / 1000) as u16;
    let mut out = [0u8; 20];
    out[0] = 4;
    out[1] = alarm_id;
    out[2..6].copy_from_slice(&seconds.to_le_bytes());
    out[6..8].copy_from_slice(&subseconds.to_le_bytes());
    out[8..16].copy_from_slice(&WAVEFORM_EFFECTS);
    // [16..18] loopControl = 0 (already zero)
    out[18] = OVERALL_LOOP;
    out[19] = DURATION_SECONDS;
    out
}

/// DISABLE_ALARM (cmd 69) REVISION_2 body `[0x02, 0xFF]` (the 5/MG form).
pub fn disable_rev2() -> [u8; 2] {
    [0x02, 0xFF]
}

/// WHOOP 4.0 SET_ALARM_TIME body (9 bytes): `[0x01][u32 LE epoch s][2 zero subsec][2 zero haptic-mode]`.
/// EXPERIMENTAL. Minute-precision, so subseconds are always zero; the trailing pair is the haptic-mode
/// field the strap needs to actually buzz.
pub fn whoop4_build(epoch_secs: u32) -> [u8; 9] {
    let mut out = [0u8; 9];
    out[0] = 0x01;
    out[1..5].copy_from_slice(&epoch_secs.to_le_bytes());
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rev4_body_layout() {
        // 1s past the epoch, alarm 1, subseconds = 0.
        let b = build(1_784_000_000_000, 1);
        assert_eq!(b[0], 4);
        assert_eq!(b[1], 1);
        assert_eq!(&b[2..6], &1_784_000_000u32.to_le_bytes());
        assert_eq!(&b[8..16], &WAVEFORM_EFFECTS);
        assert_eq!(b[18], OVERALL_LOOP);
        assert_eq!(b[19], DURATION_SECONDS);
    }

    #[test]
    fn whoop4_body_layout() {
        let b = whoop4_build(1_784_000_000);
        assert_eq!(b[0], 0x01);
        assert_eq!(&b[1..5], &1_784_000_000u32.to_le_bytes());
        assert_eq!(&b[5..], &[0, 0, 0, 0]);
    }
}
