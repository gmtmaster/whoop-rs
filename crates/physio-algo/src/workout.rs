//! Retroactive workout detection from the 1 Hz HR + gravity store. A workout is a sustained window
//! of elevated HR AND sustained motion. Per detected bout: avg/peak HR, Edwards zone time-%, mean %HRR,
//! strain and estimated calories (Keytel 2005 + Harris-Benedict BMR). APPROXIMATE, not medical advice.

use crate::calories::{self, MERGE_GAP_S};
use crate::strain;

// ── Constants ──────────────────────────────────────────────────────────────────

const MIN_EXERCISE_MIN: f64 = 5.0;
const HR_MARGIN_BPM: f64 = 15.0;
const MOTION_THRESHOLD: f64 = 0.20;
const MOTION_SMOOTH_S: f64 = 10.0;
const MIN_INTENSITY_Z2PLUS: f64 = 0.50;
const ALIGN_TOLERANCE_S: f64 = 5.0;
const RESTING_PERCENTILE: f64 = 10.0;
/// Second-pass bridge window: merge adjacent runs across a gap while HR stays elevated.
const BRIDGE_GAP_S: f64 = 300.0;

// ── Input / output types ──────────────────────────────────────────────────────

pub use crate::hr_sample::HrSample;

/// One 3-axis gravity / accelerometer sample.
#[derive(Clone, Copy, Debug)]
pub struct GravitySample {
    pub ts: i64,
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

/// Per-record motion intensity sample.
#[derive(Clone, Copy, Debug)]
pub struct ActivityPoint {
    pub ts: i64,
    pub intensity: f64,
}

/// One detected workout session.
#[derive(Clone, Debug)]
pub struct ExerciseSession {
    pub start: i64,
    pub end: i64,
    pub avg_hr: f64,
    pub peak_hr: i32,
    pub strain: Option<f64>,
    pub duration_s: f64,
    pub zone_time_pct: Vec<(i32, f64)>,
    pub avg_hrr_pct: Option<f64>,
    pub hrmax: Option<f64>,
    pub hrmax_source: String,
    pub calories_kcal: Option<f64>,
    pub calories_kj: Option<f64>,
}

// ── Activity series ────────────────────────────────────────────────────────────

/// Per-record motion-intensity series: L2 magnitude of the gravity change vs the previous record.
pub fn activity_series(gravity: &[GravitySample]) -> Vec<ActivityPoint> {
    if gravity.is_empty() {
        return Vec::new();
    }
    let mut sorted: Vec<&GravitySample> = gravity.iter().collect();
    sorted.sort_by_key(|g| g.ts);

    let mut series = Vec::with_capacity(sorted.len());
    let mut prev: Option<&GravitySample> = None;
    for (i, row) in sorted.iter().enumerate() {
        let intensity = if i == 0 {
            0.0
        } else if let Some(p) = prev {
            let dx = row.x - p.x;
            let dy = row.y - p.y;
            let dz = row.z - p.z;
            (dx * dx + dy * dy + dz * dz).sqrt()
        } else {
            0.0
        };
        series.push(ActivityPoint { ts: row.ts, intensity });
        prev = Some(row);
    }
    series
}

// ── Helpers ────────────────────────────────────────────────────────────────────

/// Day resting-HR baseline = nearest-rank RESTING_PERCENTILE of bpm values.
pub fn derive_resting_hr(hr_seg: &[HrSample]) -> f64 {
    let mut bpms: Vec<f64> = hr_seg.iter().map(|h| h.bpm as f64).collect();
    bpms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let rank = ((RESTING_PERCENTILE / 100.0 * bpms.len() as f64).ceil() as usize).max(1);
    bpms[rank - 1]
}

/// Value whose ts is nearest to `ts` within `tol` seconds, else None.
pub fn nearest(sorted_ts: &[i64], values: &[f64], ts: i64, tol: f64) -> Option<f64> {
    if sorted_ts.is_empty() {
        return None;
    }
    // bisect_left
    let mut lo = 0;
    let mut hi = sorted_ts.len();
    while lo < hi {
        let mid = (lo + hi) / 2;
        if sorted_ts[mid] < ts {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    let i = lo;
    let mut best_v: Option<f64> = None;
    let mut best_d = tol;
    for j in [i.wrapping_sub(1), i] {
        if j < sorted_ts.len() {
            let d = (sorted_ts[j] - ts).abs() as f64;
            if d <= best_d {
                best_d = d;
                best_v = Some(values[j]);
            }
        }
    }
    best_v
}

/// Trailing rolling mean over `window_s` of all-finite intensities.
pub fn smoothed_intensity(motion: &[ActivityPoint], window_s: f64) -> Vec<f64> {
    let ts: Vec<i64> = motion.iter().map(|m| m.ts).collect();
    let raw: Vec<f64> = motion.iter().map(|m| if m.intensity.is_finite() { m.intensity } else { 0.0 }).collect();
    let mut out = Vec::with_capacity(motion.len());
    let mut lo = 0;
    let mut running = 0.0;
    for i in 0..motion.len() {
        running += raw[i];
        while (ts[i] - ts[lo]) as f64 > window_s {
            running -= raw[lo];
            lo += 1;
        }
        out.push(running / (i - lo + 1) as f64);
    }
    out
}

/// Per-bout Edwards zone breakdown (%) + mean %HRR.
pub fn bout_intensity(
    hr_series: &[HrSample],
    resting_hr: f64,
    max_hr: f64,
) -> (Vec<(i32, f64)>, Option<f64>) {
    if hr_series.is_empty() || max_hr <= resting_hr {
        return (vec![], None);
    }
    let hr_reserve = max_hr - resting_hr;
    let mut zone_counts = [0i32; 6];
    let mut hrr_vals = Vec::with_capacity(hr_series.len());
    for r in hr_series {
        let bpm = r.bpm as f64;
        let z = strain::zone_weight(bpm, resting_hr, hr_reserve);
        zone_counts[z as usize] += 1;
        hrr_vals.push(strain::pct_hrr(bpm, resting_hr, hr_reserve));
    }
    let n = hr_series.len() as f64;
    let zone_pct: Vec<(i32, f64)> = zone_counts
        .iter()
        .enumerate()
        .map(|(z, &c)| {
            let pct = (c as f64 / n * 100.0 * 10.0).round() / 10.0;
            (z as i32, pct)
        })
        .collect();
    let avg_hrr = Some((hrr_vals.iter().sum::<f64>() / n * 10.0).round() / 10.0);
    (zone_pct, avg_hrr)
}

/// Second-pass merge: bridge adjacent runs where HR stays elevated across the gap.
pub fn bridge_runs(
    runs: &[(i64, i64)],
    hr_seg: &[HrSample],
    hr_floor: f64,
) -> Vec<(i64, i64)> {
    if runs.len() <= 1 {
        return runs.to_vec();
    }
    let mut merged: Vec<(i64, i64)> = Vec::new();
    let mut cur_start = runs[0].0;
    let mut cur_end = runs[0].1;
    for &next in &runs[1..] {
        let gap = (next.0 - cur_end) as f64;
        let mut bridge = false;
        if gap <= BRIDGE_GAP_S {
            let gap_hr: Vec<f64> = hr_seg
                .iter()
                .filter(|h| h.ts > cur_end && h.ts < next.0)
                .map(|h| h.bpm as f64)
                .collect();
            bridge = if gap_hr.is_empty() {
                true // sensor dropout mid-effort
            } else {
                let mean_gap_hr = gap_hr.iter().sum::<f64>() / gap_hr.len() as f64;
                mean_gap_hr > hr_floor // still working
            };
        }
        if bridge {
            cur_end = cur_end.max(next.1);
        } else {
            merged.push((cur_start, cur_end));
            cur_start = next.0;
            cur_end = next.1;
        }
    }
    merged.push((cur_start, cur_end));
    merged
}

/// Back-date a confirmed run's start over the warm-up.
pub fn backdated_start(core_start: i64, motion_ts: &[i64], smooth: &[f64]) -> i64 {
    let mut i = motion_ts.iter().position(|&t| t >= core_start).unwrap_or(motion_ts.len());
    if i >= motion_ts.len() {
        return core_start;
    }
    let mut start = core_start;
    let mut prev_ts = motion_ts[i];
    while i > 0 {
        i -= 1;
        if smooth[i] <= MOTION_THRESHOLD {
            break;
        }
        if (prev_ts - motion_ts[i]) as f64 > MERGE_GAP_S {
            break;
        }
        start = motion_ts[i];
        prev_ts = motion_ts[i];
    }
    start
}

// ── Public API ─────────────────────────────────────────────────────────────────

/// Detect workouts from HR + gravity streams.
///
/// Returns `Vec<ExerciseSession>` — one per detected bout. Empty when no workout found.
///
/// * `hr` — heart-rate stream (required; empty → `vec![]`).
/// * `gravity` — gravity stream (required; empty → `vec![]`).
/// * `resting_hr` — day resting-HR baseline. `None` → derived as 10th percentile.
/// * `max_hr` — HRmax. `None` → estimated via Tanaka from `age`.
/// * `age` — used for Tanaka fallback when `max_hr` is `None`.
/// * `weight_kg`, `height_cm`, `sex` — user profile for calories. All zero → no calorie estimate.
#[allow(clippy::too_many_arguments)]
pub fn detect(
    hr: &[HrSample],
    gravity: &[GravitySample],
    resting_hr: Option<f64>,
    max_hr: Option<f64>,
    age: Option<f64>,
    weight_kg: f64,
    height_cm: f64,
    sex: &str,
) -> Vec<ExerciseSession> {
    let hr_seg: Vec<HrSample> = {
        let mut s: Vec<&HrSample> = hr.iter().collect();
        s.sort_by_key(|h| h.ts);
        s.into_iter().copied().collect()
    };
    let motion = activity_series(gravity);
    if hr_seg.is_empty() || motion.is_empty() {
        return vec![];
    }

    let rest_hr = resting_hr.unwrap_or_else(|| derive_resting_hr(&hr_seg));
    let hr_floor = rest_hr + HR_MARGIN_BPM;

    let (eff_max_hr, hrmax_source) = match max_hr {
        Some(m) => (Some(m), "caller".to_string()),
        None => {
            let bpms: Vec<f64> = hr_seg.iter().map(|h| h.bpm as f64).collect();
            let (est, src) = strain::estimate_hrmax(&bpms, age);
            if est == 0.0 {
                (None, src.to_string())
            } else {
                (Some(est), src.to_string())
            }
        }
    };

    let hr_ts: Vec<i64> = hr_seg.iter().map(|h| h.ts).collect();
    let hr_bpm: Vec<f64> = hr_seg.iter().map(|h| h.bpm as f64).collect();
    let smooth = smoothed_intensity(&motion, MOTION_SMOOTH_S);
    let motion_ts: Vec<i64> = motion.iter().map(|m| m.ts).collect();

    // Walk the gravity timeline; flag samples where BOTH gates hold.
    let mut active_ts = Vec::new();
    for (idx, p) in motion.iter().enumerate() {
        let inten = smooth[idx];
        if inten <= MOTION_THRESHOLD {
            continue;
        }
        let bpm = nearest(&hr_ts, &hr_bpm, p.ts, ALIGN_TOLERANCE_S);
        match bpm {
            Some(b) if b > hr_floor => active_ts.push(p.ts),
            _ => {}
        }
    }
    if active_ts.is_empty() {
        return vec![];
    }

    // Group contiguous active samples into runs, merging gaps < MERGE_GAP_S.
    let mut raw_runs: Vec<(i64, i64)> = Vec::new();
    let mut run_start = active_ts[0];
    let mut prev = active_ts[0];
    for &ts in &active_ts[1..] {
        if (ts - prev) as f64 > MERGE_GAP_S {
            raw_runs.push((run_start, prev));
            run_start = ts;
        }
        prev = ts;
    }
    raw_runs.push((run_start, prev));

    // Second pass: bridge adjacent runs across brief still-elevated-HR lulls.
    let runs = bridge_runs(&raw_runs, &hr_seg, hr_floor);

    let min_dur_s = MIN_EXERCISE_MIN * 60.0;
    let mut sessions = Vec::new();

    let has_profile = weight_kg > 0.0;
    let profile_sex = if sex.is_empty() { "nonbinary" } else { sex };

    for (idx, run) in runs.iter().enumerate() {
        let (start, end) = *run;
        if ((end - start) as f64) < min_dur_s - MOTION_SMOOTH_S {
            continue;
        }
        let core: Vec<HrSample> = hr_seg.iter().filter(|h| h.ts >= start && h.ts <= end).copied().collect();
        if core.is_empty() {
            continue;
        }

        let mut zone_pct: Vec<(i32, f64)> = vec![];
        let mut avg_hrr: Option<f64> = None;
        if let Some(m) = eff_max_hr {
            if m > rest_hr {
                let (zp, ah) = bout_intensity(
                    &core,
                    rest_hr,
                    m,
                );
                zone_pct = zp;
                avg_hrr = ah;
            }
        }

        // Intensity qualification: require >= MIN_INTENSITY_Z2PLUS in zone 2+.
        let z2plus: f64 = zone_pct.iter().filter(|(z, _)| *z >= 2).map(|(_, p)| p / 100.0).sum();
        if !zone_pct.is_empty() && z2plus < MIN_INTENSITY_Z2PLUS {
            continue;
        }

        // Qualified -> back-date the start over the warm-up.
        let floor_ts = if idx > 0 { runs[idx - 1].1 + 1 } else { i64::MIN };
        let eff_start = backdated_start(start, &motion_ts, &smooth).max(floor_ts);

        let window: Vec<&HrSample> = hr_seg.iter().filter(|h| h.ts >= eff_start && h.ts <= end).collect();
        if window.is_empty() {
            continue;
        }
        let bpms: Vec<f64> = window.iter().map(|h| h.bpm as f64).collect();

        let calories = has_profile.then(|| {
            let hr_for_cal: Vec<HrSample> = window.iter().copied().copied().collect();
            calories::estimate_bout_calories(
                &hr_for_cal,
                weight_kg,
                height_cm,
                age.unwrap_or(30.0),
                profile_sex,
                eff_max_hr.unwrap_or(220.0),
                rest_hr,
            )
        });

        let avg = bpms.iter().sum::<f64>() / bpms.len() as f64;
        let peak = window.iter().map(|h| h.bpm).max().unwrap_or(0);

        let session_strain = eff_max_hr.and_then(|m| {
            let core_copies: Vec<HrSample> = window.iter().copied().copied().collect();
            strain::strain(&core_copies, Some(m), rest_hr, strain::Method::Edwards, profile_sex, strain::STRAIN_DENOMINATOR)
        });


        sessions.push(ExerciseSession {
            start: eff_start,
            end,
            avg_hr: avg,
            peak_hr: peak,
            strain: session_strain,
            duration_s: (end - eff_start) as f64,
            zone_time_pct: zone_pct,
            avg_hrr_pct: avg_hrr,
            hrmax: eff_max_hr,
            hrmax_source: hrmax_source.clone(),
            calories_kcal: calories.map(|(kcal, _)| kcal),
            calories_kj: calories.map(|(_, kj)| kj),
        });
    }
    sessions
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calories;

    fn hr(ts: i64, bpm: i32) -> HrSample {
        HrSample { ts, bpm }
    }

    fn grav(ts: i64, x: f64, y: f64, z: f64) -> GravitySample {
        GravitySample { ts, x, y, z }
    }

    /// One hour at 1 Hz: `bout_len` seconds from ts 900 at `bout_bpm` with the gravity vector
    /// flipping by `amp` every other second, resting 60 bpm and still elsewhere. `amp = 0` is a
    /// motionless bout and `bout_bpm = 60` a still one, so either alone kills detection.
    fn day_with_one_bout(bout_len: i64, bout_bpm: i32, amp: f64) -> (Vec<HrSample>, Vec<GravitySample>) {
        let (start, end) = (900i64, 900 + bout_len);
        let mut hr_series = Vec::with_capacity(3600);
        let mut gravity = Vec::with_capacity(3600);
        for ts in 0..3600i64 {
            let inside = ts >= start && ts < end;
            hr_series.push(HrSample { ts, bpm: if inside { bout_bpm } else { 60 } });
            let x = if inside && ts % 2 == 0 { amp } else { 0.0 };
            gravity.push(GravitySample { ts, x, y: 0.0, z: 1.0 });
        }
        (hr_series, gravity)
    }

    /// The 80 kg / 180 cm / 35 y male profile, resting 60, HRmax 190 supplied by the caller.
    fn detect_default(hr_series: &[HrSample], gravity: &[GravitySample]) -> Vec<ExerciseSession> {
        detect(hr_series, gravity, Some(60.0), Some(190.0), Some(35.0), 80.0, 180.0, "male")
    }

    #[test]
    fn empty_inputs_return_empty() {
        let r = detect(&[], &[], None, None, None, 0.0, 0.0, "");
        assert!(r.is_empty());
    }

    /// Every number the Activity card shows for one bout, pinned on a 30-minute fixture, plus the
    /// two nulls that must produce no card at all. A detector that returns `vec![]` unconditionally
    /// fails the count; one that returns a constant session fails the fields; one that fires on any
    /// input fails the still-day and motionless-day arms.
    #[test]
    fn detect_pins_every_field_of_one_bout_and_refuses_a_still_or_motionless_day() {
        let (hr_series, gravity) = day_with_one_bout(1800, 150, 0.3);
        let sessions = detect_default(&hr_series, &gravity);
        assert_eq!(sessions.len(), 1, "one 30-minute bout in the fixture day");
        let b = &sessions[0];

        // Span: the run opens once the 10 s trailing motion mean clears MOTION_THRESHOLD.
        assert_eq!((b.start, b.end), (907, 2699), "bout span");
        assert!((b.duration_s - 1792.0).abs() < 1e-9, "duration {}", b.duration_s);

        // HR summary. A mean over the whole day would read 105, not 150.
        assert!((b.avg_hr - 150.0).abs() < 1e-9, "avg_hr {}", b.avg_hr);
        assert_eq!(b.peak_hr, 150);
        assert_eq!(b.hrmax, Some(190.0));
        assert_eq!(b.hrmax_source, "caller");

        // Zone-time %: 150 bpm is 69.2 %HRR against (60, 190), which is Edwards zone 2.
        let pct: Vec<(i32, f64)> = b.zone_time_pct.clone();
        assert_eq!(pct, vec![(0, 0.0), (1, 0.0), (2, 100.0), (3, 0.0), (4, 0.0), (5, 0.0)]);
        assert!((pct.iter().map(|(_, p)| p).sum::<f64>() - 100.0).abs() < 1e-9, "zone % must total 100");
        assert_eq!(b.avg_hrr_pct, Some(69.2));

        // Strain and calories, by value and then rebuilt from the two public engines, so a bout that
        // stops routing through them fails here rather than drifting silently.
        assert_eq!(b.strain, Some(46.24));
        let kcal = b.calories_kcal.expect("a profiled bout reports calories");
        assert!((kcal - 446.404432759732).abs() < 1e-9, "kcal {kcal}");
        assert!((b.calories_kj.unwrap() - kcal * 4.184).abs() < 1e-9, "kJ must be kcal x 4.184");

        let window: Vec<HrSample> =
            hr_series.iter().filter(|h| h.ts >= b.start && h.ts <= b.end).copied().collect();
        assert_eq!(window.len(), 1793);
        assert_eq!(
            b.strain,
            strain::strain(&window, Some(190.0), 60.0, strain::Method::Edwards, "male", strain::STRAIN_DENOMINATOR),
            "bout strain must be the Edwards engine over the bout window"
        );
        let (re_kcal, re_kj) = calories::estimate_bout_calories(&window, 80.0, 180.0, 35.0, "male", 190.0, 60.0);
        assert!((kcal - re_kcal).abs() < 1e-9, "bout calories must be the Keytel bout engine");
        assert!((b.calories_kj.unwrap() - re_kj).abs() < 1e-9);

        // Null arms: neither gate alone makes a workout.
        let (still_hr, still_g) = day_with_one_bout(1800, 60, 0.3);
        assert!(detect_default(&still_hr, &still_g).is_empty(), "moving at resting HR is not a workout");
        let (moveless_hr, moveless_g) = day_with_one_bout(1800, 150, 0.0);
        assert!(detect_default(&moveless_hr, &moveless_g).is_empty(), "elevated HR without motion is not a workout");

        // No profile, no calorie claim.
        let unprofiled = detect(&hr_series, &gravity, Some(60.0), Some(190.0), Some(35.0), 0.0, 0.0, "male");
        assert_eq!(unprofiled[0].calories_kcal, None);
        assert_eq!(unprofiled[0].calories_kj, None);
    }

    /// MIN_INTENSITY_Z2PLUS: half the bout must sit in Edwards zone 2+, which for (60, 190) is
    /// 60 %HRR = 138 bpm. A detector that skipped the check would report the 137 bpm bout.
    #[test]
    fn a_sustained_bout_under_the_zone_2_share_is_not_a_workout() {
        assert!((MIN_INTENSITY_Z2PLUS - 0.50).abs() < 1e-9, "shipped zone-2+ share");
        let edge_bpm: f64 = 60.0 + 0.60 * (190.0 - 60.0);
        assert!((edge_bpm - 138.0).abs() < 1e-9, "zone-2 floor {edge_bpm}");

        let (under_hr, under_g) = day_with_one_bout(1800, 137, 0.3);
        assert!(detect_default(&under_hr, &under_g).is_empty(), "137 bpm is all zone 1");
        let (over_hr, over_g) = day_with_one_bout(1800, 138, 0.3);
        assert_eq!(detect_default(&over_hr, &over_g).len(), 1, "138 bpm is zone 2");
    }

    /// MIN_EXERCISE_MIN is 5 minutes but the gate subtracts the smoothing window, so the run floor is
    /// 290 s. Under it the bout is ABSENT, not a zero-strain card: nothing tells the wearer it existed.
    #[test]
    fn a_run_under_the_290_s_floor_is_absent_not_a_zero_bout() {
        assert!((MIN_EXERCISE_MIN - 5.0).abs() < 1e-9, "shipped bout minimum, minutes");
        assert!((MOTION_SMOOTH_S - 10.0).abs() < 1e-9, "shipped smoothing window");
        let floor_s = MIN_EXERCISE_MIN * 60.0 - MOTION_SMOOTH_S;
        assert!((floor_s - 290.0).abs() < 1e-9, "effective run floor {floor_s} s");

        // The 10 s smoothing lead-in costs 8 s, so a 297 s bout yields a 289 s run and disappears.
        let (short_hr, short_g) = day_with_one_bout(297, 150, 0.3);
        assert!(detect_default(&short_hr, &short_g).is_empty(), "a 4.95-minute bout is reported absent");
        let (edge_hr, edge_g) = day_with_one_bout(298, 150, 0.3);
        let edge = detect_default(&edge_hr, &edge_g);
        assert_eq!(edge.len(), 1);
        assert!((edge[0].duration_s - 290.0).abs() < 1e-9, "duration {}", edge[0].duration_s);
    }

    /// RECORDED, not a repair: a qualifying bout carries calories but `strain: None` until it holds
    /// [`strain::MIN_READINGS`] samples, so every bout between the 290 s floor and 599 s shows the
    /// wearer an Activity card with a blank strain.
    #[test]
    fn a_bout_between_290_s_and_599_s_reports_calories_but_no_strain() {
        assert_eq!(strain::MIN_READINGS, 600, "shipped strain sample floor");
        assert_eq!(strain::MIN_SPAN_SECONDS, 600, "shipped strain span floor");

        for len in [298i64, 400, 605] {
            let (h, g) = day_with_one_bout(len, 150, 0.3);
            let b = &detect_default(&h, &g)[0];
            assert_eq!(b.strain, None, "{len} s bout still reports a strain");
            assert!(b.calories_kcal.unwrap() > 0.0, "{len} s bout reports calories");
        }
        let (h, g) = day_with_one_bout(607, 150, 0.3);
        let b = &detect_default(&h, &g)[0];
        assert!((b.duration_s - 599.0).abs() < 1e-9);
        assert_eq!(b.strain, Some(34.28), "600 samples is where strain starts");
    }

    /// The bout breakdown's "Zone N" is Edwards %HRR; the daily time-in-zone's is %HRmax. Same bpm,
    /// same profile, different number on 74 of the 131 bpm values between resting and HRmax.
    #[test]
    fn bout_zone_numbers_are_hrr_based_and_disagree_with_the_daily_hrmax_zones() {
        let (rest, max) = (60.0, 190.0);
        let reserve = max - rest;
        let daily = crate::hr_zones::zones_from_max(max, "manual");

        // 114 bpm is the widest gap: 41.5 %HRR is below every Edwards band, 60 %HRmax is Zone 2.
        assert_eq!(strain::zone_weight(114.0, rest, reserve), 0, "bout view");
        assert_eq!(daily.zone_number(114.0), 2, "daily view");

        let mut differ = 0;
        let mut worst = 0i64;
        for bpm in 60..=190 {
            let bout = strain::zone_weight(bpm as f64, rest, reserve);
            let day = daily.zone_number(bpm as f64) as i64;
            if bout != day {
                differ += 1;
            }
            worst = worst.max((day - bout).abs());
        }
        assert_eq!(differ, 74, "the two Zone N definitions disagree on 74 of 131 bpm");
        assert_eq!(worst, 2, "up to two zones apart");
    }

    /// RECORDED, not a repair: with `max_hr: None` and no age the denominator is estimated from the
    /// same stream about to be scored, so the stream's own peak is 100 %HRR and lands in zone 5. The
    /// caller-supplied HRmax puts the identical sample two zones lower.
    #[test]
    fn an_estimated_hrmax_scores_the_stream_against_its_own_peak() {
        let mut own: Vec<f64> = vec![110.0; 3400];
        own.extend(std::iter::repeat_n(150.0, 200));
        assert_eq!(strain::estimate_hrmax(&own, None), (150.0, "observed"));

        let self_ref = 150.0 - 60.0;
        let tanaka = strain::tanaka_hrmax(30.0);
        assert!((strain::pct_hrr(150.0, 60.0, self_ref) - 100.0).abs() < 1e-9, "its own peak maxes out");
        assert!((strain::pct_hrr(150.0, 60.0, tanaka - 60.0) - 70.86614173228347).abs() < 1e-9);
        assert_eq!(strain::zone_weight(150.0, 60.0, self_ref), 5);
        assert_eq!(strain::zone_weight(150.0, 60.0, tanaka - 60.0), 3, "same sample, two zones lower");
    }

    #[test]
    fn activity_series_sorts_by_ts() {
        let g = vec![grav(10, 0.0, 0.0, 0.0), grav(0, 1.0, 0.0, 0.0)];
        let s = activity_series(&g);
        assert_eq!(s.len(), 2);
        assert_eq!(s[0].ts, 0);
        assert_eq!(s[1].ts, 10);
    }

    #[test]
    fn derive_resting_hr_works() {
        let mut h = Vec::new();
        for bpm in [60, 65, 70, 72, 75, 80, 85, 90, 95, 100] {
            h.push(hr(0, bpm));
        }
        let r = derive_resting_hr(&h);
        assert!((r - 60.0).abs() < 1.0); // first element = 60
    }

    #[test]
    fn nearest_within_tolerance() {
        let ts = vec![10, 20, 30];
        let vals = vec![1.0, 2.0, 3.0];
        assert_eq!(nearest(&ts, &vals, 20, 2.0), Some(2.0));
        assert_eq!(nearest(&ts, &vals, 29, 2.0), Some(3.0));
        assert_eq!(nearest(&ts, &vals, 11, 2.0), Some(1.0));
        assert_eq!(nearest(&ts, &vals, 25, 2.0), None);
    }

    #[test]
    fn smoothed_intensity_smoothes() {
        let m = vec![
            ActivityPoint { ts: 0, intensity: 0.0 },
            ActivityPoint { ts: 1, intensity: 1.0 },
            ActivityPoint { ts: 2, intensity: 0.0 },
            ActivityPoint { ts: 3, intensity: 1.0 },
            ActivityPoint { ts: 11, intensity: 0.0 }, // outside 10s window
        ];
        let s = smoothed_intensity(&m, 5.0);
        // At ts=0: avg of [0.0] = 0.0; at ts=1: (0+1)/2=0.5; at ts=2: (0+1+0)/3≈0.33
        assert!((s[0] - 0.0).abs() < 1e-9);
        assert!((s[1] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn bout_intensity_produces_zones() {
        let h: Vec<HrSample> = (0..10).map(|i| hr(i as i64, 130)).collect();
        let (zp, avg_hrr) = bout_intensity(&h, 60.0, 160.0);
        assert!(!zp.is_empty());
        assert!(avg_hrr.is_some());
        // At HR=130, resting=60, HRR=100 → %HRR=70 → Edwards weight=3
        let z3 = zp.iter().find(|(z, _)| *z == 3);
        assert!(z3.is_some());
        assert!((z3.unwrap().1 - 100.0).abs() < 0.1);
    }

    #[test]
    fn bridge_runs_merges_elevated_gap() {
        let runs = vec![(0, 100), (200, 300)]; // 100s gap
        let hr_seg: Vec<HrSample> = (0..=300).map(|i| hr(i, 100)).collect();
        let merged = bridge_runs(&runs, &hr_seg, 75.0);
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0], (0, 300));
    }

    #[test]
    fn bridge_runs_separates_rest_gap() {
        let runs = vec![(0, 100), (200, 300)];
        let hr_seg: Vec<HrSample> = (0..100).map(|i| hr(i, 100)).collect();
        // gap HR at resting: hr 60
        let gap_hr: Vec<HrSample> = (101..200).map(|i| hr(i, 60)).collect();
        let after: Vec<HrSample> = (201..301).map(|i| hr(i, 100)).collect();
        let all: Vec<HrSample> = hr_seg.into_iter().chain(gap_hr).chain(after).collect();
        let merged = bridge_runs(&runs, &all, 75.0);
        assert_eq!(merged.len(), 2);
    }
}

