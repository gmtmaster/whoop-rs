//! Parity gate for resting_hr, over the crate's own module. These pin the SEMANTICS of the
//! lowest-sustained-window floor - that it tracks a varying input, prefers a sustained low over a
//! transient dip, and is neither the global minimum nor a mean. No external resting-HR reference is
//! read anywhere in this crate, so nothing here measures agreement with a clinical resting HR.

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

/// A night at 70 bpm with a sustained 5-min window at 52 and a single-sample dip to 40 elsewhere.
/// The dip is the global minimum and lies inside a window whose mean is 64.
fn night_with_a_transient_dip() -> Vec<HrSample> {
    let mut v = Vec::new();
    for w in 0..6i64 {
        for s in 0..5i64 {
            let bpm = if w == 3 { 52 } else if w == 1 && s == 2 { 40 } else { 70 };
            v.push(HrSample::new(1000 + w * 300 + s * 60, bpm));
        }
    }
    v
}

/// Every fixture the floor gates run, as `(samples, start, end, expected floor)`.
fn floor_cases() -> Vec<(Vec<HrSample>, i64, i64, i32)> {
    let mut v: Vec<(Vec<HrSample>, i64, i64, i32)> = [(48, 60), (44, 58), (52, 70)]
        .iter()
        .map(|&(floor, high)| (night_with_floor(1000, floor, high), 1000, 2800, floor))
        .collect();
    v.push((night_with_a_transient_dip(), 1000, 2800, 52));
    // Window A mean (50+60)/2 = 55, window B (58+58)/2 = 58 - the floor is below neither raw sample.
    v.push((
        vec![HrSample::new(0, 50), HrSample::new(60, 60), HrSample::new(300, 58), HrSample::new(360, 58)],
        0,
        600,
        55,
    ));
    v
}

/// Anything that turns a session window of HR samples into a floor: the shipped function, or a null.
type FloorScorer<'a> = &'a dyn Fn(i64, i64, &[HrSample]) -> Option<i32>;

/// Cases a scorer gets wrong. Empty = it reproduces every floor.
fn floor_misses(scorer: FloorScorer) -> Vec<(i32, Option<i32>)> {
    floor_cases()
        .iter()
        .filter_map(|(hr, start, end, want)| {
            let got = scorer(*start, *end, hr);
            (got != Some(*want)).then_some((*want, got))
        })
        .collect()
}

#[test]
fn session_floor_recovers_multiple_injected_values() {
    // Tracks a varying input: three different injected floors, each recovered exactly.
    for &(floor, high) in &[(48, 60), (44, 58), (52, 70)] {
        let hr = night_with_floor(1000, floor, high);
        assert_eq!(session_resting_hr(1000, 1000 + 6 * 5 * 60, &hr), Some(floor));
    }
}

/// A one-sample dip to 40 is not a resting HR: the floor takes the sustained 52 instead, so what is
/// measured is "lowest SUSTAINED window" and not "lowest reading".
#[test]
fn session_floor_prefers_a_sustained_low_over_a_transient_dip() {
    let hr = night_with_a_transient_dip();
    assert_eq!(hr.iter().map(|s| s.bpm).min(), Some(40), "the dip is the global minimum");
    assert_eq!(session_resting_hr(1000, 2800, &hr), Some(52));
}

/// The null arm: the three do-nothing scorers that would otherwise look right. The global minimum
/// reproduces all three injected floors on its own, and only the dip and window-mean cases reject it.
#[test]
fn no_do_nothing_scorer_reproduces_the_floors() {
    let nulls: [(&str, FloorScorer); 5] = [
        ("global minimum", &|_, _, hr: &[HrSample]| hr.iter().map(|s| s.bpm).min()),
        ("segment mean", &|_, _, hr: &[HrSample]| {
            (!hr.is_empty()).then(|| (hr.iter().map(|s| s.bpm).sum::<i32>() as f64 / hr.len() as f64).round() as i32)
        }),
        ("first sample", &|_, _, hr: &[HrSample]| hr.first().map(|s| s.bpm)),
        ("constant 48", &|_, _, _: &[HrSample]| Some(48)),
        ("refuses", &|_, _, _: &[HrSample]| None),
    ];
    for (name, f) in nulls {
        assert!(!floor_misses(f).is_empty(), "the {name} scorer reproduced every floor");
    }
    assert!(floor_misses(&session_resting_hr).is_empty(), "{:?}", floor_misses(&session_resting_hr));
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
