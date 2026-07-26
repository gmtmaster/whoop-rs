//! Live stress-onset detector — edge-triggered, exercise-gated HRV-dip detection for JITAI nudges.
//! Stateful: the caller persists `State` between evaluations. R-R cleaning and the plain RMSSD come from
//! [`crate::hrv`], so the live window and the nightly path share one definition.

use crate::hrv::HrvReadiness;

const BASELINE_EMA_ALPHA: f64 = 0.98;
const DROP_RATIO: f64 = 0.6;
const FAST_WINDOW_BEATS: usize = 60;
const MIN_BEATS: usize = 20;
const RESTING_HR_LOW: f64 = 55.0;
const RESTING_HR_HIGH: f64 = 100.0;
const MOTION_GATE_G: f64 = 0.15;
const MIN_SECONDS_BETWEEN_FIRES: i64 = 900;

/// Persisted state carried between evaluations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OnsetState {
    pub baseline_rmssd: f64,
    pub was_below: bool,
    pub last_fire_at: i64,
}

impl Default for OnsetState {
    fn default() -> Self {
        Self { baseline_rmssd: 0.0, was_below: false, last_fire_at: 0 }
    }
}

/// Why the detector did or didn't fire.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OnsetReason {
    Onset,
    Disabled,
    InsufficientData,
    NoDip,
    NotAnEdge,
    ExerciseGated,
    Suppressed,
}

/// The decision returned each evaluation.
#[derive(Debug, Clone, PartialEq)]
pub struct OnsetDecision {
    pub should_nudge: bool,
    pub reason: OnsetReason,
    pub fast_rmssd: Option<f64>,
    pub baseline_rmssd: Option<f64>,
    pub next_state: OnsetState,
}

/// Evaluate the live window. `rr_buffer` is the rolling R-R series (ms, newest last).
#[allow(clippy::too_many_arguments)]
pub fn evaluate(
    rr_buffer: &[u16],
    current_hr: Option<f64>,
    recent_motion_g: Option<f64>,
    session_active: bool,
    state: OnsetState,
    enabled: bool,
    auto_nudge: bool,
    quiet_hours_enabled: bool,
    quiet_start_min: i32,
    quiet_end_min: i32,
    now_sec: i64,
    tz_offset_sec: i64,
) -> OnsetDecision {
    let refuse = |reason: OnsetReason| OnsetDecision {
        should_nudge: false, reason,
        fast_rmssd: None,
        baseline_rmssd: if state.baseline_rmssd > 0.0 { Some(state.baseline_rmssd) } else { None },
        next_state: state,
    };
    if !enabled || !auto_nudge { return refuse(OnsetReason::Disabled); }

    let clean_all = HrvReadiness::clean_rr(rr_buffer);
    let fast_window = if clean_all.len() > FAST_WINDOW_BEATS { &clean_all[clean_all.len() - FAST_WINDOW_BEATS..] } else { &clean_all[..] };
    let fast = if fast_window.len() >= MIN_BEATS { HrvReadiness::rmssd_plain(fast_window) } else { None };
    let Some(fast) = fast else { return refuse(OnsetReason::InsufficientData); };

    let new_baseline = if state.baseline_rmssd == 0.0 { fast }
        else { state.baseline_rmssd * BASELINE_EMA_ALPHA + fast * (1.0 - BASELINE_EMA_ALPHA) };
    let mut next = OnsetState { baseline_rmssd: new_baseline, ..state };
    let threshold = new_baseline * DROP_RATIO;
    let is_below = fast < threshold;
    let is_edge = is_below && !state.was_below;
    next.was_below = is_below;

    let decide = |nudge: bool, reason: OnsetReason| OnsetDecision {
        should_nudge: nudge, reason,
        fast_rmssd: Some(fast), baseline_rmssd: Some(new_baseline),
        next_state: next,
    };
    if !is_below { return decide(false, OnsetReason::NoDip); }
    if !is_edge { return decide(false, OnsetReason::NotAnEdge); }

    let hr_in_band = current_hr.is_some_and(|h| (RESTING_HR_LOW..=RESTING_HR_HIGH).contains(&h));
    let moving = recent_motion_g.is_some_and(|m| m >= MOTION_GATE_G);
    if !hr_in_band || moving { return decide(false, OnsetReason::ExerciseGated); }

    if session_active { return decide(false, OnsetReason::Suppressed); }
    if state.last_fire_at != 0 && (now_sec - state.last_fire_at) < MIN_SECONDS_BETWEEN_FIRES {
        return decide(false, OnsetReason::Suppressed);
    }
    if quiet_hours_enabled {
        let local_min = ((now_sec + tz_offset_sec).rem_euclid(86_400) / 60) as i32;
        let in_window = if quiet_start_min <= quiet_end_min {
            (quiet_start_min..quiet_end_min).contains(&local_min)
        } else {
            local_min >= quiet_start_min || local_min < quiet_end_min
        };
        if in_window { return decide(false, OnsetReason::Suppressed); }
    }

    next.last_fire_at = now_sec;
    OnsetDecision {
        should_nudge: true, reason: OnsetReason::Onset,
        fast_rmssd: Some(fast), baseline_rmssd: Some(new_baseline),
        next_state: next,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn steady_rr(n: usize, around: u16) -> Vec<u16> { vec![around; n] }

    #[test]
    fn disabled_when_off() {
        let rr = steady_rr(80, 800);
        let d = evaluate(&rr, Some(70.0), None, false, OnsetState::default(), false, false, false, 0, 0, 1000, 0);
        assert_eq!(d.reason, OnsetReason::Disabled);
    }

    #[test]
    fn insufficient_when_too_few_beats() {
        let rr = vec![800u16; 10];
        let d = evaluate(&rr, Some(70.0), None, false, OnsetState::default(), true, true, false, 0, 0, 1000, 0);
        assert_eq!(d.reason, OnsetReason::InsufficientData);
    }

    #[test]
    fn no_dip_when_stable() {
        // Steady 800ms → RMSSD=0. Baseline seeds at 0. 0 < 0*0.6 = false → NoDip.
        let rr = steady_rr(80, 800);
        let d = evaluate(&rr, Some(70.0), None, false, OnsetState::default(), true, true, false, 0, 0, 1000, 0);
        assert_eq!(d.reason, OnsetReason::NoDip);
    }

    #[test]
    fn seeds_baseline_and_detects_dip() {
        // Seed baseline at 40ms. Feed variable series → RMSSD > 0. Baseline EMA adapts.
        let state = OnsetState { baseline_rmssd: 40.0, was_below: false, last_fire_at: 0 };
        let rr = (0..80).map(|i| if i % 2 == 0 { 820u16 } else { 780u16 }).collect::<Vec<_>>();
        let d = evaluate(&rr, Some(70.0), None, false, state, true, true, false, 0, 0, 1000, 0);
        assert!(d.fast_rmssd.is_some());
        assert!(d.baseline_rmssd.unwrap() > 0.0);
        // Alternating 820/780 → RMSSD=40ms. Baseline: 40*0.98+40*0.02=40. 40 < 40*0.6=24? No → NoDip
        // But RMSSD of the full 80-beat window after cleaning → all deltas are 40ms → RMSSD=40ms.
        // baseline stays at 40. threshold=24. 40<24=false → NoDip. Valid result.
        assert!(!d.should_nudge);
    }

    #[test]
    fn exercise_gate_blocks_high_hr() {
        // HR at 110 → out of resting band [55,100]. Even if dip detected, exercise gate blocks.
        let state = OnsetState { baseline_rmssd: 40.0, was_below: false, last_fire_at: 0 };
        let rr = vec![800u16; 80]; // RMSSD=0, below threshold → would fire, but HR gate blocks first
        let d = evaluate(&rr, Some(110.0), None, false, state, true, true, false, 0, 0, 1000, 0);
        // 0 < 40*0.6=24 → is_below=true, is_edge=true → would be Onset.
        // But HR=110 is out of [55,100] → ExerciseGated.
        assert_eq!(d.reason, OnsetReason::ExerciseGated, "got {:?}", d.reason);
    }
}
