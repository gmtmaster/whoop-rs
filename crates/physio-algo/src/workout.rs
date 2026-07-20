//! Retroactive workout detection from the 1 Hz HR + gravity store. A workout is a sustained window
//! of elevated HR AND sustained motion. Per detected bout: avg/peak HR, Edwards zone time-%, mean %HRR,
//! strain and estimated calories (Keytel 2005 + Harris-Benedict BMR). APPROXIMATE, not medical advice.

use crate::calories;
use crate::strain;

// ── Constants ──────────────────────────────────────────────────────────────────

const MIN_EXERCISE_MIN: f64 = 5.0;
const HR_MARGIN_BPM: f64 = 15.0;
const MOTION_THRESHOLD: f64 = 0.20;
const MOTION_SMOOTH_S: f64 = 10.0;
const MERGE_GAP_S: f64 = 150.0;
const MIN_INTENSITY_Z2PLUS: f64 = 0.50;
const ALIGN_TOLERANCE_S: f64 = 5.0;
const RESTING_PERCENTILE: f64 = 10.0;
/// Second-pass bridge window: merge adjacent runs across a gap while HR stays elevated.
const BRIDGE_GAP_S: f64 = 300.0;

// ── Input / output types ──────────────────────────────────────────────────────

/// One heart-rate sample.
#[derive(Clone, Copy, Debug)]
pub struct HrSample {
    pub ts: i64,
    pub bpm: i32,
}

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

/// Edwards zone breakdown: map of zone number (0-5) to percentage.
pub type ZoneTimePct = [(i32, f64); 6];

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
        let z3plus: f64 = zone_pct.iter().filter(|(z, _)| *z >= 2).map(|(_, p)| p / 100.0).sum();
        if !zone_pct.is_empty() && z3plus < MIN_INTENSITY_Z2PLUS {
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

        let (kcal, kj) = if has_profile {
            let hr_for_cal: Vec<calories::HrSample> = window
                .iter()
                .map(|h| calories::HrSample { ts: h.ts, bpm: h.bpm })
                .collect();
            calories::estimate_bout_calories(
                &hr_for_cal,
                weight_kg,
                height_cm,
                age.unwrap_or(30.0),
                profile_sex,
                eff_max_hr.unwrap_or(220.0),
                rest_hr,
            )
        } else {
            (0.0, 0.0)
        };

        let avg = bpms.iter().sum::<f64>() / bpms.len() as f64;
        let peak = window.iter().map(|h| h.bpm).max().unwrap_or(0);

        let session_strain = eff_max_hr.and_then(|m| {
            let core_copies: Vec<strain::HrSample> = window
                .iter()
                .map(|h| strain::HrSample { ts: h.ts, bpm: h.bpm })
                .collect();
            strain::strain(&core_copies, Some(m), rest_hr, strain::Method::Edwards, profile_sex, strain::STRAIN_DENOMINATOR)
        });

        let kcal_to_store = if has_profile { Some(kcal) } else { None };
        let kj_to_store = if has_profile { Some(kj) } else { None };

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
            calories_kcal: kcal_to_store,
            calories_kj: kj_to_store,
        });
    }
    sessions
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hr(ts: i64, bpm: i32) -> HrSample {
        HrSample { ts, bpm }
    }

    fn grav(ts: i64, x: f64, y: f64, z: f64) -> GravitySample {
        GravitySample { ts, x, y, z }
    }

    #[test]
    fn empty_inputs_return_empty() {
        let r = detect(&[], &[], None, None, None, 0.0, 0.0, "");
        assert!(r.is_empty());
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
