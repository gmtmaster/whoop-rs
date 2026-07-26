//! Parity gate for resting_hr, over the crate's own module. The session-floor tests recover multiple
//! injected floors so the method is shown to track a varying input, not one lucky match.

use physio_algo::resting_hr::{daily_resting_hr, floor_mean_log_line, session_resting_hr, HrSample};

const SUFFIX: &str = "(floor = WHOOP-style lowest-sustained = NOOP RHR; mean = sleeping-HR-app number)";

#[test]
fn log_line_floor_below_mean() {
    let bpms = [48, 50, 52, 55, 58, 60, 62]; // mean 55
    let line = floor_mean_log_line("2026-06-12", 48, &bpms);
    assert_eq!(
        line,
        format!("rhr day=2026-06-12 floor=48 nightMean=55 inBedSamples=7 {SUFFIX}")
    );
}

#[test]
fn log_line_mean_rounds_to_nearest() {
    // 50,51,52,54 → 207/4 = 51.75 → 52.
    let line = floor_mean_log_line("2026-06-13", 50, &[50, 51, 52, 54]);
    assert!(line.contains("floor=50 nightMean=52 inBedSamples=4"), "{line}");
}

#[test]
fn log_line_empty_in_bed_is_nil() {
    let line = floor_mean_log_line("2026-06-12", 47, &[]);
    assert_eq!(
        line,
        format!("rhr day=2026-06-12 floor=47 nightMean=nil inBedSamples=0 {SUFFIX}")
    );
}

#[test]
fn log_line_carries_no_em_dash() {
    let line = floor_mean_log_line("2026-06-12", 48, &[48, 60]);
    assert!(!line.contains('\u{2014}'));
}

/// A night at `high` bpm with one sustained 5-min window pinned at `floor`, offset into the session.
fn night_with_floor(start: i64, floor: i32, high: i32) -> Vec<HrSample> {
    let mut v = Vec::new();
    for w in 0..6i64 {
        let bpm = if w == 3 { floor } else { high };
        let base = start + w * 5 * 60;
        for s in 0..5i64 {
            v.push(HrSample::new(base + s * 60, bpm));
        }
    }
    v
}

#[test]
fn session_floor_recovers_multiple_injected_values() {
    // Tracks a varying input: three different injected floors, each recovered exactly.
    for &(floor, high) in &[(48, 60), (44, 58), (52, 70)] {
        let hr = night_with_floor(1000, floor, high);
        assert_eq!(session_resting_hr(1000, 1000 + 6 * 5 * 60, &hr), Some(floor));
    }
}

#[test]
fn session_floor_is_the_lowest_window_mean_not_the_global_min() {
    // Window A mean = (50+60)/2 = 55; window B mean = (58+58)/2 = 58 → floor 55, below neither raw min.
    let hr = [
        HrSample::new(0, 50),
        HrSample::new(60, 60),
        HrSample::new(300, 58),
        HrSample::new(360, 58),
    ];
    assert_eq!(session_resting_hr(0, 600, &hr), Some(55));
}

#[test]
fn session_floor_rounds_window_mean_half_up() {
    // (50+51+52)/3 = 51.0 in one window vs a higher window → 51.
    let hr = [
        HrSample::new(0, 50),
        HrSample::new(60, 51),
        HrSample::new(120, 52),
        HrSample::new(300, 80),
    ];
    assert_eq!(session_resting_hr(0, 600, &hr), Some(51));
}

#[test]
fn session_floor_empty_segment_is_none() {
    assert_eq!(session_resting_hr(0, 600, &[]), None);
    // Samples all outside the window → still none.
    assert_eq!(session_resting_hr(0, 600, &[HrSample::new(5000, 60)]), None);
}

#[test]
fn session_floor_falls_back_to_segment_mean_when_no_window_holds_a_sample() {
    // A lone sample exactly at `end` is in-segment (inclusive) but in no tumbling window → segment mean.
    assert_eq!(session_resting_hr(0, 600, &[HrSample::new(600, 63)]), Some(63));
}

#[test]
fn daily_resting_hr_is_the_min_session_floor() {
    assert_eq!(daily_resting_hr(&[Some(52), None, Some(48), Some(55)]), Some(48));
    assert_eq!(daily_resting_hr(&[None, None]), None);
    assert_eq!(daily_resting_hr(&[]), None);
}
