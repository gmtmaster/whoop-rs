//! Device-epoch → wall-clock correction for a strap whose RTC has wandered. A healthy 5/MG RTC already
//! carries real unix, so a small offset is trusted as-is; a gross offset (> 1 day) is snapped to a
//! 5-minute grid so the same record re-syncs to the same corrected timestamp (dedup safety), and a
//! correction that overshoots wall time is discarded. `wall_now` is supplied by the caller (pure fn).

const DAY: i64 = 86_400;
const GRID: i64 = 300; // 5-minute snap

/// Correct a device timestamp to wall-clock seconds. Trusted (|offset| ≤ 1 day) → returned unchanged.
pub fn to_wall(device_unix: u32, wall_now: i64) -> i64 {
    let device = device_unix as i64;
    let offset = wall_now - device;
    if offset.abs() <= DAY {
        return device;
    }
    let snapped = (offset / GRID) * GRID;
    let corrected = device + snapped;
    if corrected > wall_now {
        device
    } else {
        corrected
    }
}

/// Plausibility gate for a type-47 / event unix: reject a 2023-11 floor or a timestamp over a day ahead.
pub fn is_plausible(unix: u32, wall_now: i64) -> bool {
    let t = unix as i64;
    t >= 1_700_000_000 && t <= wall_now + DAY
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
        assert!(out <= wall && (wall - out) < GRID);
    }
}
