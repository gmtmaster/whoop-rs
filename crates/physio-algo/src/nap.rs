//! Short-nap detection from the 1 Hz gravity + HR store. A daytime nap is a short stretch where the wrist
//! goes quiet AND the heart rate settles into the resting band. Inferred, never a strap flag, so the verdict
//! is deliberately conservative: tri-state (NAP / NONE / INCONCLUSIVE) that only PROPOSES a review card,
//! never auto-writes a sleep session. Pure and deterministic. Approximate, non-clinical.

use crate::workout::{activity_series, smoothed_intensity, GravitySample, HrSample};

// Defaults.
pub const DEFAULT_MIN_NAP_MIN: i32 = 20;
pub const DEFAULT_MAX_NAP_MIN: i32 = 90;
pub const DEFAULT_STILL_THRESHOLD_G: f64 = 0.08;
pub const DEFAULT_HR_SETTLE_MARGIN_BPM: i32 = 8;
pub const DEFAULT_SMOOTH_WINDOW_S: f64 = 120.0;
/// Break a quiet run when the inter-record gap exceeds this (s) — a data hole isn't sleep.
pub const MAX_GAP_S: i64 = 10 * 60;
/// A verdict is only attempted with at least this many gravity rows in the window.
pub const DEFAULT_MIN_GRAVITY_SAMPLES: usize = 20;
/// ...and only when their median inter-sample gap is no larger than this (s).
pub const DEFAULT_MAX_MEDIAN_GAP_S: i64 = 90;

/// Tri-state verdict for one candidate window.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NapVerdict {
    Nap,
    None,
    Inconclusive,
}

/// A proposed nap to offer for review. `confidence` in 0..1 orders the UI only, never a medical claim.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NapCandidate {
    pub start: i64,
    pub end: i64,
    pub mean_hr: Option<i32>,
    pub confidence: f64,
}

/// The outcome of one [`evaluate`] pass: the verdict + (only when `Nap`) the candidate to review.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NapDecision {
    pub verdict: NapVerdict,
    pub candidate: Option<NapCandidate>,
}

/// User-tunable thresholds; defaults mirror the app's fixed config.
#[derive(Clone, Copy, Debug)]
pub struct NapConfig {
    pub enabled: bool,
    pub min_nap_minutes: i32,
    pub max_nap_minutes: i32,
    pub still_threshold_g: f64,
    pub hr_settle_margin_bpm: i32,
    pub smooth_window_seconds: f64,
}

impl Default for NapConfig {
    fn default() -> Self {
        NapConfig {
            enabled: false,
            min_nap_minutes: DEFAULT_MIN_NAP_MIN,
            max_nap_minutes: DEFAULT_MAX_NAP_MIN,
            still_threshold_g: DEFAULT_STILL_THRESHOLD_G,
            hr_settle_margin_bpm: DEFAULT_HR_SETTLE_MARGIN_BPM,
            smooth_window_seconds: DEFAULT_SMOOTH_WINDOW_S,
        }
    }
}

/// Dense enough to judge? True only with >= `min_samples` rows AND a median inter-sample gap
/// <= `max_median_gap_s`. A sparse window is [`NapVerdict::Inconclusive`], never `None`.
pub fn is_window_dense(gravity: &[GravitySample], min_samples: usize, max_median_gap_s: i64) -> bool {
    if gravity.len() < min_samples {
        return false;
    }
    let mut ts: Vec<i64> = gravity.iter().map(|g| g.ts).collect();
    ts.sort_unstable();
    let mut gaps: Vec<i64> = ts.windows(2).map(|w| w[1] - w[0]).filter(|&d| d >= 0).collect();
    if gaps.is_empty() {
        return false;
    }
    gaps.sort_unstable();
    gaps[gaps.len() / 2] <= max_median_gap_s
}

/// The longest stretch of sustained stillness (smoothed motion <= `still_threshold_g`, unbroken by
/// movement or a data gap > [`MAX_GAP_S`]). Reuses the shared workout motion primitives. `(start, end)`
/// or `None`.
pub fn longest_quiet_run(
    gravity: &[GravitySample],
    still_threshold_g: f64,
    smooth_window_seconds: f64,
) -> Option<(i64, i64)> {
    if gravity.len() < 2 {
        return None;
    }
    let motion = activity_series(gravity);
    let smoothed = smoothed_intensity(&motion, smooth_window_seconds);
    let ts: Vec<i64> = motion.iter().map(|m| m.ts).collect();
    let n = ts.len();

    let (mut best_start, mut best_end): (i64, i64) = (-1, -1);
    let mut run_start: i64 = -1;
    let close_run = |end_idx: i64, ts: &[i64], best_start: &mut i64, best_end: &mut i64, run_start: &mut i64| {
        if *run_start >= 0 && *run_start <= end_idx {
            let s = ts[*run_start as usize];
            let e = ts[end_idx as usize];
            if *best_start < 0 || (e - s) > (*best_end - *best_start) {
                *best_start = s;
                *best_end = e;
            }
        }
        *run_start = -1;
    };
    for i in 0..n {
        if i > 0 && ts[i] - ts[i - 1] > MAX_GAP_S {
            close_run(i as i64 - 1, &ts, &mut best_start, &mut best_end, &mut run_start); // data gap ends the run
        }
        if smoothed[i] > still_threshold_g {
            close_run(i as i64 - 1, &ts, &mut best_start, &mut best_end, &mut run_start); // movement ends the run
        } else if run_start < 0 {
            run_start = i as i64;
        }
    }
    close_run(n as i64 - 1, &ts, &mut best_start, &mut best_end, &mut run_start);
    if best_start < 0 {
        None
    } else {
        Some((best_start, best_end))
    }
}

/// Mean HR (bpm) over `[start, end]`, plausible beats only (25..=220), or `None` when none landed.
pub fn mean_hr_in(hr: &[HrSample], start: i64, end: i64) -> Option<i32> {
    let in_window: Vec<i32> = hr
        .iter()
        .filter(|h| h.ts >= start && h.ts <= end && (25..=220).contains(&h.bpm))
        .map(|h| h.bpm)
        .collect();
    if in_window.is_empty() {
        return None;
    }
    let mean = in_window.iter().map(|&b| b as f64).sum::<f64>() / in_window.len() as f64;
    Some(mean as i32)
}

/// 0..1 ordering confidence (not a probability). Longer + a known, well-settled HR band reads as more
/// confident; an unknown HR band caps the total at the duration term.
pub fn confidence_for(duration_min: f64, resting_hr: Option<i32>, mean_hr: Option<i32>, config: &NapConfig) -> f64 {
    let span = (config.max_nap_minutes - config.min_nap_minutes).max(1) as f64;
    let dur_term = 0.4 + 0.45 * (((duration_min - config.min_nap_minutes as f64) / span).clamp(0.0, 1.0));
    let (Some(resting), Some(mean)) = (resting_hr, mean_hr) else {
        return dur_term.clamp(0.0, 0.7);
    };
    let headroom = config.hr_settle_margin_bpm.max(1) as f64;
    let below = (resting + config.hr_settle_margin_bpm - mean).max(0) as f64;
    let hr_term = 0.15 * (below / headroom).clamp(0.0, 1.0);
    (dur_term + hr_term).clamp(0.0, 1.0)
}

/// Classify one candidate window. Tri-state, conservative: OFF or sparse -> Inconclusive; dense but moving
/// or a too-short quiet run -> None; a too-long quiet run -> Inconclusive; a quiet run in `[min,max]` with
/// HR not settled (when known) -> None; else -> Nap (offered as a review card).
pub fn evaluate(gravity: &[GravitySample], hr: &[HrSample], resting_hr: Option<i32>, config: &NapConfig) -> NapDecision {
    let inconclusive = NapDecision { verdict: NapVerdict::Inconclusive, candidate: None };
    let none = NapDecision { verdict: NapVerdict::None, candidate: None };
    if !config.enabled {
        return inconclusive;
    }
    if !is_window_dense(gravity, DEFAULT_MIN_GRAVITY_SAMPLES, DEFAULT_MAX_MEDIAN_GAP_S) {
        return inconclusive;
    }
    let Some((start, end)) = longest_quiet_run(gravity, config.still_threshold_g, config.smooth_window_seconds) else {
        return none;
    };
    let duration_min = (end - start) as f64 / 60.0;
    if duration_min < config.min_nap_minutes as f64 {
        return none;
    }
    if duration_min > config.max_nap_minutes as f64 {
        return inconclusive;
    }
    let mean_hr = mean_hr_in(hr, start, end);
    if let (Some(resting), Some(mean)) = (resting_hr, mean_hr) {
        if mean > resting + config.hr_settle_margin_bpm {
            return none;
        }
    }
    NapDecision {
        verdict: NapVerdict::Nap,
        candidate: Some(NapCandidate {
            start,
            end,
            mean_hr,
            confidence: confidence_for(duration_min, resting_hr, mean_hr, config),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Gravity rows every `step` s over `[t0, t0+dur)` with per-record motion `move_g` (0 = perfectly still).
    fn grav(t0: i64, dur: i64, step: i64, move_g: f64) -> Vec<GravitySample> {
        let mut out = Vec::new();
        let mut t = t0;
        let mut x = 0.0;
        let mut flip = 1.0;
        while t < t0 + dur {
            // Alternate x by move_g each record so the L2 per-record delta is exactly move_g.
            x += flip * move_g;
            flip = -flip;
            out.push(GravitySample { ts: t, x, y: 0.0, z: 0.0 });
            t += step;
        }
        out
    }

    fn hr_flat(t0: i64, dur: i64, bpm: i32) -> Vec<HrSample> {
        (0..dur).step_by(30).map(|d| HrSample { ts: t0 + d, bpm }).collect()
    }

    #[test]
    fn disabled_is_inconclusive() {
        let cfg = NapConfig { enabled: false, ..Default::default() };
        assert_eq!(evaluate(&grav(0, 3600, 30, 0.0), &[], None, &cfg).verdict, NapVerdict::Inconclusive);
    }

    #[test]
    fn sparse_window_is_inconclusive_not_none() {
        let cfg = NapConfig { enabled: true, ..Default::default() };
        // Dense enough count but a huge median gap (300 s > 90 s) -> can't judge.
        let g = grav(0, 40 * 300, 300, 0.0);
        assert_eq!(evaluate(&g, &[], Some(55), &cfg).verdict, NapVerdict::Inconclusive);
    }

    #[test]
    fn still_settled_run_is_a_nap() {
        let cfg = NapConfig { enabled: true, ..Default::default() };
        // 40 min perfectly still, HR settled at resting+2 (<= margin 8).
        let g = grav(1000, 40 * 60, 30, 0.0);
        let hr = hr_flat(1000, 40 * 60, 57);
        let d = evaluate(&g, &hr, Some(55), &cfg);
        assert_eq!(d.verdict, NapVerdict::Nap);
        let c = d.candidate.unwrap();
        assert!(c.confidence > 0.0 && c.confidence <= 1.0);
        assert_eq!(c.mean_hr, Some(57));
    }

    #[test]
    fn still_but_elevated_hr_is_none() {
        let cfg = NapConfig { enabled: true, ..Default::default() };
        let g = grav(1000, 40 * 60, 30, 0.0);
        let hr = hr_flat(1000, 40 * 60, 80); // resting 55 + 8 = 63 gate; 80 > gate -> awake
        assert_eq!(evaluate(&g, &hr, Some(55), &cfg).verdict, NapVerdict::None);
    }

    #[test]
    fn moving_window_is_none() {
        let cfg = NapConfig { enabled: true, ..Default::default() };
        // Constant 0.3 g per-record motion (> 0.08 still threshold) -> no quiet run.
        let g = grav(1000, 40 * 60, 30, 0.3);
        assert_eq!(evaluate(&g, &[], Some(55), &cfg).verdict, NapVerdict::None);
    }

    #[test]
    fn too_long_run_is_inconclusive() {
        let cfg = NapConfig { enabled: true, ..Default::default() };
        // 2 h still (> 90 min max) -> could be main sleep, don't mislabel.
        let g = grav(1000, 120 * 60, 30, 0.0);
        assert_eq!(evaluate(&g, &[], Some(55), &cfg).verdict, NapVerdict::Inconclusive);
    }
}
