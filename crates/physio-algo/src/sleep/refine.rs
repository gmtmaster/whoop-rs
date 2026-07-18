//! Motion-aware wake refinement — a post-pass over a staged hypnogram that reclassifies hot-but-still
//! wake to light. For each long wake segment with no walking cadence and posture stable outside a
//! minority of burst minutes, non-burst minutes become light; burst minutes (+/- a pad) stay wake, so
//! the pass only ever shrinks wake. Gated on a per-minute density self-check over the observed streams.

use std::collections::{HashMap, HashSet};

use super::detect::posture_variance_g2;
use super::input::{AccelSample, StepSample};
use super::{SleepStage, StageSegment};

const MIN_WAKE_SEGMENT_SECONDS: i64 = 5 * 60;
const SUSTAINED_WALK_TICKS_PER_MINUTE: i32 = 10;
const SUSTAINED_WALK_MIN_CONSECUTIVE_MINUTES: i32 = 2;
const SINGLE_MINUTE_WALK_TICKS: i32 = 40;
const STABLE_POSTURE_VARIANCE_G2: f64 = 0.05;
const MIN_STABLE_MINUTE_FRACTION: f64 = 0.80;
const BURST_PAD_MINUTES: i64 = 1;
const MIN_GRAVITY_SAMPLES_PER_MINUTE_FOR_VARIANCE: i32 = 2;
const MIN_STEP_SAMPLES_PER_MINUTE_FOR_DENSITY: i32 = 1;
const MIN_DENSE_MINUTE_COVERAGE_FRACTION: f64 = 0.80;

/// Reclassify non-burst minutes of eligible wake segments to light. `segments` must tile one contiguous
/// window in order. Byte-identical passthrough when empty, degenerate, or the density gate declines.
pub(super) fn refine(segments: &[StageSegment], grav: &[AccelSample], steps: &[StepSample]) -> Vec<StageSegment> {
    let (Some(first), Some(last)) = (segments.first(), segments.last()) else { return segments.to_vec() };
    let (window_start, window_end) = (first.start, last.end);
    if window_end <= window_start || !is_motion_dense(window_start, window_end, grav, steps) {
        return segments.to_vec();
    }
    let grav_by_minute = bucket_by_minute(grav);
    let ticks_by_minute = walk_class_ticks_per_minute(steps);
    let mut out: Vec<StageSegment> = Vec::new();
    for seg in segments {
        for piece in refine_segment(*seg, &grav_by_minute, &ticks_by_minute) {
            append_merging(piece, &mut out);
        }
    }
    out
}

/// True when both the gravity and step streams are dense enough over `[start, end)` to trust per-minute
/// locomotion/posture evidence. Judges the observed streams directly, never a strap model.
pub(super) fn is_motion_dense(start: i64, end: i64, grav: &[AccelSample], steps: &[StepSample]) -> bool {
    let grav_density = dense_minute_fraction(grav, start, end, MIN_GRAVITY_SAMPLES_PER_MINUTE_FOR_VARIANCE, |g| g.ts);
    let step_density = dense_minute_fraction(steps, start, end, MIN_STEP_SAMPLES_PER_MINUTE_FOR_DENSITY, |s| s.ts);
    grav_density >= MIN_DENSE_MINUTE_COVERAGE_FRACTION && step_density >= MIN_DENSE_MINUTE_COVERAGE_FRACTION
}

/// Fraction of the wall-clock minutes tiling `[start, end)` that carry at least `min_per_minute` samples.
fn dense_minute_fraction<T>(samples: &[T], start: i64, end: i64, min_per_minute: i32, ts: impl Fn(&T) -> i64) -> f64 {
    if end <= start {
        return 0.0;
    }
    let (first, last) = (start / 60, (end - 1) / 60);
    if last < first {
        return 0.0;
    }
    let mut counts: HashMap<i64, i32> = HashMap::new();
    for s in samples {
        let m = ts(s) / 60;
        if (first..=last).contains(&m) {
            *counts.entry(m).or_insert(0) += 1;
        }
    }
    let total = last - first + 1;
    let dense = (first..=last).filter(|m| counts.get(m).copied().unwrap_or(0) >= min_per_minute).count();
    dense as f64 / total as f64
}

/// Refine one segment. Non-wake, short, or well-evidenced-active segments pass through unchanged; only a
/// segment that reads "hot-but-still" (no locomotion AND stable posture) has its non-burst minutes lit.
fn refine_segment(
    seg: StageSegment,
    grav_by_minute: &HashMap<i64, Vec<AccelSample>>,
    ticks_by_minute: &HashMap<i64, i32>,
) -> Vec<StageSegment> {
    if seg.stage != SleepStage::Wake || seg.end - seg.start < MIN_WAKE_SEGMENT_SECONDS {
        return vec![seg];
    }
    let mins = minutes(seg.start, seg.end);
    if mins.is_empty() || has_locomotion(&mins, ticks_by_minute) {
        return vec![seg];
    }
    let Some(burst) = stable_burst_minutes(&mins, grav_by_minute) else { return vec![seg] };
    let (first_minute, last_minute) = (mins[0], mins[mins.len() - 1]);
    let mut keep_wake: HashSet<i64> = HashSet::new();
    for &m in &burst {
        let (lo, hi) = (first_minute.max(m - BURST_PAD_MINUTES), last_minute.min(m + BURST_PAD_MINUTES));
        for p in lo..=hi {
            keep_wake.insert(p);
        }
    }
    let mut result: Vec<StageSegment> = Vec::new();
    let n = mins.len();
    for (idx, &m) in mins.iter().enumerate() {
        let stage = if keep_wake.contains(&m) { SleepStage::Wake } else { SleepStage::Light };
        let start = if idx == 0 { seg.start } else { m * 60 };
        let end = if idx == n - 1 { seg.end } else { (m + 1) * 60 };
        append_merging(StageSegment { start, end, stage }, &mut result);
    }
    result
}

/// True when `mins` shows real walking: a single minute `>= SINGLE_MINUTE_WALK_TICKS`, or
/// `>= SUSTAINED_WALK_MIN_CONSECUTIVE_MINUTES` in a row each `>= SUSTAINED_WALK_TICKS_PER_MINUTE`.
fn has_locomotion(mins: &[i64], ticks_by_minute: &HashMap<i64, i32>) -> bool {
    let mut consecutive = 0;
    for m in mins {
        let ticks = ticks_by_minute.get(m).copied().unwrap_or(0);
        if ticks >= SINGLE_MINUTE_WALK_TICKS {
            return true;
        }
        if ticks >= SUSTAINED_WALK_TICKS_PER_MINUTE {
            consecutive += 1;
            if consecutive >= SUSTAINED_WALK_MIN_CONSECUTIVE_MINUTES {
                return true;
            }
        } else {
            consecutive = 0;
        }
    }
    false
}

/// The burst (not posture-stable) minutes when at least `MIN_STABLE_MINUTE_FRACTION` of `mins` are stable;
/// `None` when too few are stable to trust. A minute with too little gravity to judge counts as a burst.
fn stable_burst_minutes(mins: &[i64], grav_by_minute: &HashMap<i64, Vec<AccelSample>>) -> Option<HashSet<i64>> {
    let mut burst: HashSet<i64> = HashSet::new();
    let mut stable = 0i64;
    let empty: Vec<AccelSample> = Vec::new();
    for &m in mins {
        let samples = grav_by_minute.get(&m).unwrap_or(&empty);
        match posture_variance_g2(samples) {
            Some(v) if v < STABLE_POSTURE_VARIANCE_G2 => stable += 1,
            _ => {
                burst.insert(m);
            }
        }
    }
    if (stable as f64) / (mins.len() as f64) < MIN_STABLE_MINUTE_FRACTION {
        return None;
    }
    Some(burst)
}

/// Per-minute walk-class tick cadence: the wrap-aware u16 counter delta between consecutive samples,
/// attributed to the later sample's minute, kept only when its class is walk (1) or run (2).
fn walk_class_ticks_per_minute(steps: &[StepSample]) -> HashMap<i64, i32> {
    let mut sorted = steps.to_vec();
    sorted.sort_by_key(|s| s.ts);
    let mut out: HashMap<i64, i32> = HashMap::new();
    for w in sorted.windows(2) {
        let cur = w[1];
        let Some(cls) = cur.activity_class else { continue };
        if cls != 1 && cls != 2 {
            continue;
        }
        let delta = cur.counter.wrapping_sub(w[0].counter) as i32;
        *out.entry(cur.ts / 60).or_insert(0) += delta;
    }
    out
}

/// The wall-clock minute indices (unix seconds / 60) tiling `[start, end)`; empty for a degenerate window.
fn minutes(start: i64, end: i64) -> Vec<i64> {
    if end <= start {
        return Vec::new();
    }
    let (first, last) = (start / 60, (end - 1) / 60);
    if last < first {
        return Vec::new();
    }
    (first..=last).collect()
}

fn bucket_by_minute(grav: &[AccelSample]) -> HashMap<i64, Vec<AccelSample>> {
    let mut out: HashMap<i64, Vec<AccelSample>> = HashMap::new();
    for &g in grav {
        out.entry(g.ts / 60).or_default().push(g);
    }
    out
}

/// Append `piece`, merging into the previous segment when the stage matches and the two are contiguous.
fn append_merging(piece: StageSegment, out: &mut Vec<StageSegment>) {
    if let Some(last) = out.last_mut() {
        if last.stage == piece.stage && last.end == piece.start {
            last.end = piece.end;
            return;
        }
    }
    out.push(piece);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn a(ts: i64, x: f64, y: f64, z: f64) -> AccelSample {
        AccelSample { ts, x, y, z }
    }
    fn st(ts: i64, counter: u16, cls: Option<u8>) -> StepSample {
        StepSample { ts, counter, activity_class: cls }
    }

    #[test]
    fn hot_but_still_wake_reclassified_to_light() {
        let seg = vec![StageSegment { start: 0, end: 600, stage: SleepStage::Wake }];
        let (mut grav, mut steps) = (Vec::new(), Vec::new());
        for m in 0..10i64 {
            grav.push(a(m * 60, 0.0, 0.0, 1.0));
            grav.push(a(m * 60 + 30, 0.0, 0.0, 1.0));
            steps.push(st(m * 60, 100, Some(0))); // still class, no walk ticks
        }
        let out = refine(&seg, &grav, &steps);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].stage, SleepStage::Light);
        assert_eq!((out[0].start, out[0].end), (0, 600));
    }

    #[test]
    fn sparse_stream_declines_and_passes_through() {
        let seg = vec![StageSegment { start: 0, end: 600, stage: SleepStage::Wake }];
        let grav: Vec<_> = (0..10).map(|m| a(m * 60, 0.0, 0.0, 1.0)).collect(); // 1/min < required 2
        assert_eq!(refine(&seg, &grav, &[]), seg);
    }

    #[test]
    fn walk_ticks_and_locomotion_gate() {
        let steps = vec![st(120, 100, Some(1)), st(150, 140, Some(1))]; // +40 in minute 2
        let ticks = walk_class_ticks_per_minute(&steps);
        assert_eq!(ticks.get(&2).copied().unwrap_or(0), 40);
        assert!(has_locomotion(&[2], &ticks));
        let still = vec![st(120, 100, Some(0)), st(150, 200, Some(0))];
        assert!(walk_class_ticks_per_minute(&still).is_empty());
    }
}
