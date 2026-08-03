//! Device-epoch → wall-clock correction for a strap whose RTC has wandered. A healthy 5/MG RTC already
//! carries real unix, so a small offset is trusted as-is; a gross offset (> 1 day) is snapped to a
//! 5-minute grid so the same record re-syncs to the same corrected timestamp (dedup safety), and a
//! correction that overshoots wall time is discarded. `wall_now` is supplied by the caller (pure fn).

const DAY: i64 = 86_400;
const SNAP_GRID_SECS: i64 = 300; // 5-minute snap

/// Correct a device timestamp to wall-clock seconds. Trusted (|offset| ≤ 1 day) → returned unchanged.
pub fn to_wall(device_unix: u32, wall_now: i64) -> i64 {
    let device = device_unix as i64;
    let offset = wall_now - device;
    if offset.abs() <= DAY {
        return device;
    }
    let snapped = (offset / SNAP_GRID_SECS) * SNAP_GRID_SECS;
    let corrected = device + snapped;
    if corrected > wall_now {
        device
    } else {
        corrected
    }
}

/// SET_CLOCK payload — the 8-byte form `[u32 LE seconds][4 zero]` newer 4.0 and 5.0/MG firmware latches.
pub fn set_clock_payload(now_unix: u32) -> [u8; 8] {
    let mut out = [0u8; 8];
    out[..4].copy_from_slice(&now_unix.to_le_bytes());
    out
}

/// SET_CLOCK payload — the legacy 9-byte form `[u32 LE seconds][5 zero]` older 4.0 firmware needs (a no-op
/// on newer firmware). Sent alongside the 8-byte form so either firmware latches; both carry the same value.
pub fn set_clock_payload_legacy(now_unix: u32) -> [u8; 9] {
    let mut out = [0u8; 9];
    out[..4].copy_from_slice(&now_unix.to_le_bytes());
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trusted_rtc_unchanged() {
        assert_eq!(to_wall(1_784_000_000, 1_784_000_030), 1_784_000_000);
    }

    #[test]
    fn stale_rtc_snapped_to_grid() {
        // Device stuck far in the past; correct it to within one 5-min grid step of wall time.
        let wall = 1_784_000_000;
        let out = to_wall(60_000_000, wall);
        assert_ne!(out, 60_000_000); // not the raw stale ts
        assert!(out <= wall && (wall - out) < SNAP_GRID_SECS);
    }

    #[test]
    fn set_clock_payloads_carry_seconds_le() {
        let p8 = set_clock_payload(1_784_000_000);
        assert_eq!(p8.len(), 8);
        assert_eq!(&p8[..4], &1_784_000_000u32.to_le_bytes());
        assert_eq!(&p8[4..], &[0, 0, 0, 0]);
        let p9 = set_clock_payload_legacy(1_784_000_000);
        assert_eq!(p9.len(), 9);
        assert_eq!(&p9[..4], &1_784_000_000u32.to_le_bytes());
        assert_eq!(&p9[4..], &[0, 0, 0, 0, 0]);
    }
}
