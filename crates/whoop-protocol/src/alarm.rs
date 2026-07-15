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
