//! Resting heart rate: the median in-bed level. `session_resting_hr` is the per-session value;
//! `session_resting_hr_floor` is the lowest-sustained-5-min floor, kept as a comparison instrument;
//! `daily_resting_hr` folds the day's sessions to their minimum; `floor_mean_log_line` formats the
//! floor-vs-night-mean diagnostic line.
//! Plain HR samples in, bpm out. Absent signal returns `None`. Wellness estimate, never medical.
//!
//! The median, not the floor, is what ships. Measured against WHOOP's own published resting HR over
//! 16 nights of one wearer, the floor reads 10.84 bpm low (MAE 10.84) while the median lands at
//! +0.94 (MAE 1.06, sd 0.93); a second capture of the same wearer gives -8.79 and +0.43. One wearer,
//! so the floor stays callable and the sweep is worth re-running when a second wearer's paired
//! export and raw HR exist.

const WINDOW_SECONDS: i64 = 5 * 60;

pub use crate::hr_sample::HrSample;

/// Round to nearest, ties toward positive infinity.
fn round_half_up(x: f64) -> i32 {
    (x + 0.5).floor() as i32
}

/// The session resting HR: the MEDIAN in-bed bpm over `[start, end]`, `None` when the segment is
/// empty. An even count averages the two middle samples and rounds half up, so the value is a bpm.
/// Robust to the one-sample dips the floor is built to survive, because half the samples have to move
/// before it does.
pub fn session_resting_hr(start: i64, end: i64, hr: &[HrSample]) -> Option<i32> {
    let mut bpms: Vec<i32> = hr.iter().filter(|s| s.ts >= start && s.ts <= end).map(|s| s.bpm).collect();
    if bpms.is_empty() {
        return None;
    }
    bpms.sort_unstable();
    let n = bpms.len();
    let mid = if n % 2 == 1 {
        bpms[n / 2] as f64
    } else {
        (bpms[n / 2 - 1] as f64 + bpms[n / 2] as f64) / 2.0
    };
    Some(round_half_up(mid))
}

/// The session floor: the lowest 5-min tumbling-window mean bpm over `[start, end]`. Falls back to the
/// whole-segment mean when no window holds a sample, and to `None` when the segment is empty.
/// No longer the shipped resting HR (see the module note) — kept as the comparison instrument the
/// diagnostic line prints, and as the thing to re-measure against when a second wearer's data lands.
pub fn session_resting_hr_floor(start: i64, end: i64, hr: &[HrSample]) -> Option<i32> {
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

/// The day's resting HR: the minimum session value across the day's matched sessions, or `None`.
/// A min over MEDIANS picks the day's calmest session, where a min over floors picked its deepest dip.
pub fn daily_resting_hr(session_floors: &[Option<i32>]) -> Option<i32> {
    session_floors.iter().filter_map(|f| *f).min()
}

/// The floor-vs-night-mean diagnostic line for one day. `floor` is the lowest-sustained instrument;
/// the mean over `in_bed_bpms` (the samples in the same span) is the sleeping-HR-app number, or `nil`
/// when empty. Neither is the shipped resting HR any more — that is the median.
pub fn floor_mean_log_line(day: &str, floor: i32, in_bed_bpms: &[i32]) -> String {
    let mean_log = if in_bed_bpms.is_empty() {
        "nil".to_string()
    } else {
        let sum: i64 = in_bed_bpms.iter().map(|&b| b as i64).sum();
        round_half_up(sum as f64 / in_bed_bpms.len() as f64).to_string()
    };
    format!(
        "rhr day={day} floor={floor} nightMean={mean_log} inBedSamples={} \
         (floor = lowest-sustained instrument; mean = sleeping-HR-app number; NOOP RHR = the median)",
        in_bed_bpms.len()
    )
}
