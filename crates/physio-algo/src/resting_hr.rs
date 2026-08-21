//! Resting-heart-rate observations and comparison instruments.
//! Plain HR samples in, bpm out. Absent signal returns `None`.
//! Wellness estimate, never medical.

const WINDOW_SECONDS: i64 = 5 * 60;
pub const RHR_MIN_BPM: i32 = 25;
pub const RHR_MAX_BPM: i32 = 220;
pub const RHR_MIN_WINDOW_COVERAGE_SECONDS: usize = 4 * 60;
pub const RHR_MIN_WEIGHTED_DEEP_WINDOWS: usize = 3;
pub const RHR_TRIM_FRACTION: f64 = 0.10;

pub use crate::hr_sample::HrSample;

/// Round to nearest, ties toward positive infinity.
fn round_half_up(x: f64) -> i32 {
    (x + 0.5).floor() as i32
}

/// Median in-bed bpm over `[start, end]`, or `None` for an empty segment.
/// Even counts average the middle pair and round half up.
pub fn session_resting_hr(start: i64, end: i64, hr: &[HrSample]) -> Option<i32> {
    let mut bpms: Vec<i32> = hr
        .iter()
        .filter(|s| s.ts >= start && s.ts <= end)
        .map(|s| s.bpm)
        .collect();
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

/// Stage-aware nightly RHR from complete, quality-gated Deep windows with recency weighting.
/// Falls back to the Deep median, then the whole-session median. Spans are half-open.
pub fn session_resting_hr_stage_aware(
    start: i64,
    end: i64,
    hr: &[HrSample],
    deep_spans: &[(i64, i64)],
) -> Option<i32> {
    let valid = |s: &&HrSample| {
        s.ts >= start && s.ts <= end && (RHR_MIN_BPM..=RHR_MAX_BPM).contains(&s.bpm)
    };
    let session: Vec<&HrSample> = hr.iter().filter(valid).collect();
    if session.is_empty() {
        return None;
    }

    let mut windows = Vec::new();
    for &(span_start, span_end) in deep_spans {
        let mut window_start = span_start.max(start);
        let span_end = span_end.min(end);
        while window_start + WINDOW_SECONDS <= span_end {
            let samples: Vec<i32> = session
                .iter()
                .filter(|s| s.ts >= window_start && s.ts < window_start + WINDOW_SECONDS)
                .map(|s| s.bpm)
                .collect();
            let covered: std::collections::HashSet<i64> = session
                .iter()
                .filter(|s| s.ts >= window_start && s.ts < window_start + WINDOW_SECONDS)
                .map(|s| s.ts)
                .collect();
            if covered.len() >= RHR_MIN_WINDOW_COVERAGE_SECONDS {
                windows.push(trimmed_mean(samples, RHR_TRIM_FRACTION));
            }
            window_start += WINDOW_SECONDS;
        }
    }

    if windows.len() >= RHR_MIN_WEIGHTED_DEEP_WINDOWS {
        let (weighted_sum, weight): (f64, f64) = windows
            .iter()
            .enumerate()
            .map(|(i, value)| (*value * (i + 1) as f64, (i + 1) as f64))
            .fold((0.0, 0.0), |(sv, sw), (v, w)| (sv + v, sw + w));
        return Some(round_half_up(weighted_sum / weight));
    }

    let mut deep: Vec<i32> = session
        .iter()
        .filter(|s| deep_spans.iter().any(|&(a, b)| s.ts >= a && s.ts < b))
        .map(|s| s.bpm)
        .collect();
    let deep_seconds: std::collections::HashSet<i64> = session
        .iter()
        .filter(|s| deep_spans.iter().any(|&(a, b)| s.ts >= a && s.ts < b))
        .map(|s| s.ts)
        .collect();
    if deep_seconds.len() >= WINDOW_SECONDS as usize {
        return median_bpm(&mut deep);
    }

    let mut all: Vec<i32> = session.iter().map(|s| s.bpm).collect();
    median_bpm(&mut all)
}

fn trimmed_mean(mut values: Vec<i32>, fraction: f64) -> f64 {
    values.sort_unstable();
    let trim = ((values.len() as f64 * fraction).floor() as usize).min(values.len() / 2);
    let kept = &values[trim..values.len() - trim];
    kept.iter().map(|&v| v as f64).sum::<f64>() / kept.len() as f64
}

fn median_bpm(values: &mut [i32]) -> Option<i32> {
    if values.is_empty() {
        return None;
    }
    values.sort_unstable();
    let n = values.len();
    let value = if n % 2 == 1 {
        values[n / 2] as f64
    } else {
        (values[n / 2 - 1] + values[n / 2]) as f64 / 2.0
    };
    Some(round_half_up(value))
}

/// The session floor: the lowest 5-min tumbling-window mean bpm over `[start, end]`. Falls back to the
/// whole-segment mean when no window holds a sample, and to `None` when the segment is empty.
/// Not the primary resting HR (see the module note) — kept as the comparison instrument the
/// diagnostic line prints, and as the thing to re-measure against when a second wearer's data lands.
pub fn session_resting_hr_floor(start: i64, end: i64, hr: &[HrSample]) -> Option<i32> {
    let seg: Vec<&HrSample> = hr.iter().filter(|s| s.ts >= start && s.ts <= end).collect();
    if seg.is_empty() {
        return None;
    }
    let mut means: Vec<f64> = Vec::new();
    let mut t = start;
    while t < end {
        let win: Vec<&&HrSample> = seg
            .iter()
            .filter(|s| s.ts >= t && s.ts < t + WINDOW_SECONDS)
            .collect();
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
/// when empty. Neither is the primary resting HR; the stage-aware observation is.
pub fn floor_mean_log_line(day: &str, floor: i32, in_bed_bpms: &[i32]) -> String {
    let mean_log = if in_bed_bpms.is_empty() {
        "nil".to_string()
    } else {
        let sum: i64 = in_bed_bpms.iter().map(|&b| b as i64).sum();
        round_half_up(sum as f64 / in_bed_bpms.len() as f64).to_string()
    };
    format!(
        "rhr day={day} floor={floor} nightMean={mean_log} inBedSamples={} \
         (floor = lowest-sustained instrument; mean = sleeping-HR-app number; NOOP RHR = stage-aware Deep/SWS)",
        in_bed_bpms.len()
    )
}
