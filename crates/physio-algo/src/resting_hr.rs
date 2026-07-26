//! Resting heart rate: the WHOOP-style floor, i.e. the lowest sustained 5-min in-bed level, not the
//! night mean. `session_resting_hr` is the per-session floor; `daily_resting_hr` folds the day's
//! sessions to their minimum; `floor_mean_log_line` formats the floor-vs-night-mean diagnostic line.
//! Plain HR samples in, bpm out. Absent signal returns `None`. Wellness estimate, never medical.

const WINDOW_SECONDS: i64 = 5 * 60;

pub use crate::hr_sample::HrSample;

/// Round to nearest, ties toward positive infinity.
fn round_half_up(x: f64) -> i32 {
    (x + 0.5).floor() as i32
}

/// The session floor: the lowest 5-min tumbling-window mean bpm over `[start, end]`. Falls back to the
/// whole-segment mean when no window holds a sample, and to `None` when the segment is empty.
pub fn session_resting_hr(start: i64, end: i64, hr: &[HrSample]) -> Option<i32> {
    let seg: Vec<&HrSample> = hr.iter().filter(|s| s.ts >= start && s.ts <= end).collect();
    if seg.is_empty() {
        return None;
    }
    let mut means: Vec<f64> = Vec::new();
    let mut t = start;
    while t < end {
        let win: Vec<&&HrSample> = seg.iter().filter(|s| s.ts >= t && s.ts < t + WINDOW_SECONDS).collect();
        if !win.is_empty() {
            let sum: i64 = win.iter().map(|s| s.bpm as i64).sum();
            means.push(sum as f64 / win.len() as f64);
        }
        t += WINDOW_SECONDS;
    }
    if let Some(m) = means.into_iter().reduce(f64::min) {
        return Some(round_half_up(m));
    }
    let all: i64 = seg.iter().map(|s| s.bpm as i64).sum();
    Some(round_half_up(all as f64 / seg.len() as f64))
}

/// The day's resting HR: the minimum session floor across the day's matched sessions, or `None`.
pub fn daily_resting_hr(session_floors: &[Option<i32>]) -> Option<i32> {
    session_floors.iter().filter_map(|f| *f).min()
}

/// The floor-vs-night-mean diagnostic line for one day. `floor` is the WHOOP-style RHR; the mean over
/// `in_bed_bpms` (the samples in the same span) is the sleeping-HR-app number, or `nil` when empty.
pub fn floor_mean_log_line(day: &str, floor: i32, in_bed_bpms: &[i32]) -> String {
    let mean_log = if in_bed_bpms.is_empty() {
        "nil".to_string()
    } else {
        let sum: i64 = in_bed_bpms.iter().map(|&b| b as i64).sum();
        round_half_up(sum as f64 / in_bed_bpms.len() as f64).to_string()
    };
    format!(
        "rhr day={day} floor={floor} nightMean={mean_log} inBedSamples={} \
         (floor = WHOOP-style lowest-sustained = NOOP RHR; mean = sleeping-HR-app number)",
        in_bed_bpms.len()
    )
}
