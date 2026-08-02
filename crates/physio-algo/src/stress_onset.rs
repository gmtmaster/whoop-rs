//! Live stress-onset detector — edge-triggered, exercise-gated HRV-dip detection for JITAI nudges.
//! Stateful: the caller persists `State` between evaluations. R-R cleaning and the plain RMSSD come from
//! [`crate::hrv`], so the live window and the nightly path share one definition.

use crate::hrv::HrvReadiness;
use crate::stress::ACTIVITY_GATE_G;

const BASELINE_EMA_ALPHA: f64 = 0.98;
const DROP_RATIO: f64 = 0.6;
const FAST_WINDOW_BEATS: usize = 60;
const MIN_BEATS: usize = 20;
const RESTING_HR_LOW: f64 = 55.0;
const RESTING_HR_HIGH: f64 = 100.0;
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
    let moving = recent_motion_g.is_some_and(|m| m >= ACTIVITY_GATE_G);
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

    /// 80 beats alternating `800` and `800 + spread` ms, all inside the R-R range and well inside
    /// the ectopic tolerance, so every beat survives cleaning and the fast window's RMSSD is
    /// exactly `spread`. Feeding the detector a known RMSSD is what makes the dip arithmetic
    /// checkable rather than merely present.
    fn buffer(spread: u16) -> Vec<u16> {
        (0..80).map(|i| if i % 2 == 0 { 800u16 } else { 800 + spread }).collect()
    }

    /// A baseline the detector has already learned, so a dip has something to fall from.
    const SEEDED: OnsetState =
        OnsetState { baseline_rmssd: 40.0, was_below: false, last_fire_at: 0 };
    /// The fixture clock: 100000 s past the epoch is 03:46 local at tz 0, the minute the
    /// quiet-hours rows straddle.
    const NOW: i64 = 100_000;
    const NOW_LOCAL_MIN: i32 = 226;
    /// A dip against [`SEEDED`]: 23 ms is under the crossing point, 24 ms is over it.
    const DIP_SPREAD: u16 = 23;
    const CALM_SPREAD: u16 = 24;
    /// Rows of [`cases`] on which the detector must nudge, asserted there and used by the null arm.
    const FIRING_ROWS: usize = 6;

    /// One evaluation and the decision it must produce. Every arm of [`evaluate`] appears here at
    /// least once, including the one that fires.
    struct Case {
        what: &'static str,
        rr: Vec<u16>,
        hr: Option<f64>,
        motion: Option<f64>,
        session_active: bool,
        state: OnsetState,
        enabled: bool,
        auto_nudge: bool,
        /// `[start, end)` in local minutes, or `None` for quiet hours off.
        quiet: Option<(i32, i32)>,
        tz_offset_sec: i64,
        reason: OnsetReason,
        nudge: bool,
    }

    impl Case {
        /// The shipped decision for this row, with `rr` swappable so a stand-in can be run against
        /// the same clock, state and gates.
        fn decide_on(&self, rr: &[u16]) -> OnsetDecision {
            let (qs, qe) = self.quiet.unwrap_or((0, 0));
            evaluate(
                rr, self.hr, self.motion, self.session_active, self.state, self.enabled,
                self.auto_nudge, self.quiet.is_some(), qs, qe, NOW, self.tz_offset_sec,
            )
        }

        fn decide(&self) -> OnsetDecision {
            self.decide_on(&self.rr)
        }
    }

    fn case(what: &'static str, reason: OnsetReason, nudge: bool) -> Case {
        Case {
            what,
            rr: buffer(DIP_SPREAD),
            hr: Some(70.0),
            motion: None,
            session_active: false,
            state: SEEDED,
            enabled: true,
            auto_nudge: true,
            quiet: None,
            tz_offset_sec: 0,
            reason,
            nudge,
        }
    }

    /// The decision table. Every row is a real evaluation of the shipped `evaluate`; the `reason`
    /// and `nudge` columns are what it must answer.
    fn cases() -> Vec<Case> {
        use OnsetReason::*;
        vec![
            Case { enabled: false, ..case("detector switched off", Disabled, false) },
            Case { auto_nudge: false, ..case("auto-nudge switched off", Disabled, false) },
            Case {
                rr: vec![800u16; 19],
                ..case("fewer clean beats than MIN_BEATS", InsufficientData, false)
            },
            Case {
                state: OnsetState::default(),
                ..case("first evaluation, no baseline yet", NoDip, false)
            },
            Case {
                rr: buffer(CALM_SPREAD),
                ..case("reading above the dip threshold", NoDip, false)
            },
            case("a dip below the learned baseline", Onset, true),
            Case {
                state: OnsetState { was_below: true, ..SEEDED },
                ..case("still below, so not a fresh edge", NotAnEdge, false)
            },
            Case { hr: None, ..case("no heart rate to gate on", ExerciseGated, false) },
            Case {
                hr: Some(RESTING_HR_HIGH + 0.1),
                ..case("heart rate above the resting band", ExerciseGated, false)
            },
            Case {
                hr: Some(RESTING_HR_LOW - 0.1),
                ..case("heart rate below the resting band", ExerciseGated, false)
            },
            Case {
                motion: Some(ACTIVITY_GATE_G),
                ..case("motion at the activity gate", ExerciseGated, false)
            },
            Case {
                motion: Some(ACTIVITY_GATE_G - 1e-4),
                ..case("motion just under the activity gate", Onset, true)
            },
            Case { session_active: true, ..case("a session is running", Suppressed, false) },
            Case {
                state: OnsetState { last_fire_at: NOW - MIN_SECONDS_BETWEEN_FIRES + 1, ..SEEDED },
                ..case("inside the refractory window", Suppressed, false)
            },
            Case {
                state: OnsetState { last_fire_at: NOW - MIN_SECONDS_BETWEEN_FIRES, ..SEEDED },
                ..case("refractory window just elapsed", Onset, true)
            },
            Case {
                quiet: Some((NOW_LOCAL_MIN, NOW_LOCAL_MIN + 1)),
                ..case("inside quiet hours", Suppressed, false)
            },
            Case {
                quiet: Some((NOW_LOCAL_MIN + 1, NOW_LOCAL_MIN + 2)),
                ..case("outside quiet hours", Onset, true)
            },
            Case {
                quiet: Some((23 * 60, 7 * 60)),
                ..case("inside quiet hours that wrap midnight", Suppressed, false)
            },
            Case {
                quiet: Some((23 * 60, 3 * 60)),
                ..case("outside quiet hours that wrap midnight", Onset, true)
            },
            Case {
                quiet: Some((NOW_LOCAL_MIN, NOW_LOCAL_MIN + 1)),
                tz_offset_sec: 3_600,
                ..case("the same quiet window, one time zone east", Onset, true)
            },
        ]
    }

    #[test]
    fn the_fixtures_are_derived_from_these_shipped_constants() {
        // Every fixture below is sized and timed from these values: 19 beats is one short of the
        // gate, 23 and 24 ms straddle the crossing point they set, the 60-beat tail is the whole
        // fast window, 03:46 sits inside the quiet windows. Move one and the fixtures stop meaning
        // what their names say, so they are pinned here rather than read back through the symbol.
        assert_eq!(BASELINE_EMA_ALPHA, 0.98);
        assert_eq!(DROP_RATIO, 0.6);
        assert_eq!(FAST_WINDOW_BEATS, 60);
        assert_eq!(MIN_BEATS, 20);
        assert_eq!(RESTING_HR_LOW, 55.0);
        assert_eq!(RESTING_HR_HIGH, 100.0);
        assert_eq!(MIN_SECONDS_BETWEEN_FIRES, 900);
        assert_eq!(ACTIVITY_GATE_G, 0.15);
        assert_eq!(NOW.rem_euclid(86_400) / 60, NOW_LOCAL_MIN as i64);
    }

    #[test]
    fn every_branch_of_the_decision_is_reached_with_its_reason() {
        use OnsetReason::*;
        let cases = cases();
        for c in &cases {
            let d = c.decide();
            assert_eq!(d.reason, c.reason, "{}", c.what);
            assert_eq!(d.should_nudge, c.nudge, "{}", c.what);
        }
        // Coverage, asserted rather than assumed: every reason the enum can carry is produced by
        // some row, and the firing rows exist. A table of refusals alone cannot see a detector
        // that never nudges.
        for r in [Onset, Disabled, InsufficientData, NoDip, NotAnEdge, ExerciseGated, Suppressed] {
            assert!(cases.iter().any(|c| c.reason == r), "no row reaches {r:?}");
        }
        assert_eq!(cases.iter().filter(|c| c.nudge).count(), FIRING_ROWS);
        assert_eq!(cases.len(), 20);
    }

    #[test]
    fn a_dip_below_the_learned_baseline_fires_the_nudge() {
        // The firing path, end to end and to the bit: the reading, the folded baseline, the
        // decision, and the state the caller must persist so the refractory window starts.
        let c = case("a dip below the learned baseline", OnsetReason::Onset, true);
        let d = c.decide();
        assert!(d.should_nudge);
        assert_eq!(d.reason, OnsetReason::Onset);
        assert_eq!(d.fast_rmssd.unwrap().to_bits(), (DIP_SPREAD as f64).to_bits());
        assert_eq!(d.baseline_rmssd.unwrap().to_bits(), 39.660000000000004f64.to_bits());
        assert_eq!(
            d.next_state,
            OnsetState { baseline_rmssd: 39.660000000000004, was_below: true, last_fire_at: NOW }
        );
        // A refusal leaves the refractory clock alone, so a suppressed evaluation cannot silently
        // start one.
        let calm = Case { rr: buffer(CALM_SPREAD), ..case("", OnsetReason::NoDip, false) };
        assert_eq!(calm.decide().next_state.last_fire_at, 0);
    }

    #[test]
    fn the_dip_threshold_is_the_drop_ratio_of_the_folded_baseline() {
        // The EMA folds the new reading in BEFORE the comparison, so the crossing point solves
        // `f = DROP_RATIO * (alpha*b + (1-alpha)*f)`. Derived from the shipped constants here, so
        // moving either one moves this bound and breaks the pair below it.
        let b = SEEDED.baseline_rmssd;
        let crossing = DROP_RATIO * BASELINE_EMA_ALPHA * b / (1.0 - DROP_RATIO * (1.0 - BASELINE_EMA_ALPHA));
        assert!(
            (DIP_SPREAD as f64) < crossing && crossing < (CALM_SPREAD as f64),
            "the fixture no longer straddles the crossing point at {crossing}"
        );
        let dip = case("", OnsetReason::Onset, true);
        assert_eq!(dip.decide().reason, OnsetReason::Onset);
        let calm = Case { rr: buffer(CALM_SPREAD), ..case("", OnsetReason::NoDip, false) };
        assert_eq!(calm.decide().reason, OnsetReason::NoDip);
    }

    #[test]
    fn the_fast_window_is_the_last_beats_only() {
        // A buffer whose first 20 beats are wildly variable and whose last 60 are the calm dip.
        // The whole-buffer RMSSD is far above the threshold; the fast window's is the dip, so the
        // detector fires only because it read the tail and not the head.
        let tail = buffer(DIP_SPREAD).into_iter().take(60).collect::<Vec<_>>();
        let mut rr: Vec<u16> = (0..20).map(|i| if i % 2 == 0 { 700u16 } else { 900 }).collect();
        rr.extend(tail.iter().copied());
        let clean = HrvReadiness::clean_rr(&rr);
        assert_eq!(clean[clean.len() - tail.len()..], tail[..], "the tail must survive cleaning");
        let whole = HrvReadiness::rmssd_plain(&clean).unwrap();
        assert!(whole > 4.0 * DIP_SPREAD as f64, "head and tail must differ, whole reads {whole}");

        let c = Case { rr, ..case("", OnsetReason::Onset, true) };
        let d = c.decide();
        assert_eq!(d.fast_rmssd.unwrap().to_bits(), (DIP_SPREAD as f64).to_bits());
        assert_eq!(d.reason, OnsetReason::Onset);
    }

    #[test]
    fn the_first_evaluation_seeds_the_baseline_and_cannot_dip() {
        // With no baseline the reading BECOMES the baseline, and a value can never be under its own
        // fraction, so the very first evaluation never fires however calm it is.
        for spread in [1u16, DIP_SPREAD, 40] {
            let c = Case {
                rr: buffer(spread),
                state: OnsetState::default(),
                ..case("", OnsetReason::NoDip, false)
            };
            let d = c.decide();
            assert_eq!(d.reason, OnsetReason::NoDip, "spread {spread}");
            assert_eq!(d.baseline_rmssd.unwrap().to_bits(), (spread as f64).to_bits());
            assert_eq!(d.next_state.baseline_rmssd.to_bits(), (spread as f64).to_bits());
            assert!(!d.should_nudge);
        }
    }

    #[test]
    fn the_exercise_gate_is_inclusive_at_both_resting_edges() {
        for hr in [RESTING_HR_LOW, 70.0, RESTING_HR_HIGH] {
            let c = Case { hr: Some(hr), ..case("", OnsetReason::Onset, true) };
            assert_eq!(c.decide().reason, OnsetReason::Onset, "{hr} bpm is inside the band");
        }
        for hr in [RESTING_HR_LOW - 0.1, RESTING_HR_HIGH + 0.1] {
            let c = Case { hr: Some(hr), ..case("", OnsetReason::ExerciseGated, false) };
            assert_eq!(c.decide().reason, OnsetReason::ExerciseGated, "{hr} bpm is outside it");
        }
    }

    #[test]
    fn the_motion_gate_shares_the_windowed_constant_and_its_edge() {
        // The same `>=` on the same constant the windowed path uses, so both stress paths refuse
        // the same movement. Exercised here because the shipped Android caller passes no motion.
        let at = |m: f64| {
            Case { motion: Some(m), ..case("", OnsetReason::Onset, true) }.decide().reason
        };
        assert_eq!(at(ACTIVITY_GATE_G), OnsetReason::ExerciseGated);
        assert_eq!(at(ACTIVITY_GATE_G + 1.0), OnsetReason::ExerciseGated);
        assert_eq!(at(ACTIVITY_GATE_G - 1e-4), OnsetReason::Onset);
        assert_eq!(at(0.0), OnsetReason::Onset);
        // An absent channel is not movement: the gate is skipped, not failed.
        assert_eq!(case("", OnsetReason::Onset, true).decide().reason, OnsetReason::Onset);
    }

    #[test]
    fn the_refractory_window_is_open_at_its_far_edge() {
        let after = |elapsed: i64| {
            Case {
                state: OnsetState { last_fire_at: NOW - elapsed, ..SEEDED },
                ..case("", OnsetReason::Onset, true)
            }
            .decide()
            .should_nudge
        };
        assert!(!after(0));
        assert!(!after(MIN_SECONDS_BETWEEN_FIRES - 1));
        assert!(after(MIN_SECONDS_BETWEEN_FIRES));
        assert!(after(MIN_SECONDS_BETWEEN_FIRES + 1));
        // `last_fire_at == 0` means "never fired", not "fired at the epoch", so a fresh state is
        // not held off for the first fifteen minutes of 1970.
        assert!(case("", OnsetReason::Onset, true).decide().should_nudge);
    }

    #[test]
    fn quiet_hours_suppress_only_inside_the_window() {
        let at = |q: (i32, i32), tz: i64| {
            Case { quiet: Some(q), tz_offset_sec: tz, ..case("", OnsetReason::Onset, true) }
                .decide()
                .should_nudge
        };
        // `[start, end)`: the starting minute is quiet, the ending minute is not.
        assert!(!at((NOW_LOCAL_MIN, NOW_LOCAL_MIN + 1), 0));
        assert!(at((NOW_LOCAL_MIN + 1, NOW_LOCAL_MIN + 2), 0));
        assert!(at((NOW_LOCAL_MIN - 1, NOW_LOCAL_MIN), 0));
        // A window that wraps midnight is the union of its two halves, not an empty range.
        assert!(!at((23 * 60, 7 * 60), 0));
        assert!(at((23 * 60, 3 * 60), 0));
        // The window is in LOCAL minutes: one zone east moves the same clock reading out of it.
        assert!(at((NOW_LOCAL_MIN, NOW_LOCAL_MIN + 1), 3_600));
        assert!(!at((NOW_LOCAL_MIN + 60, NOW_LOCAL_MIN + 61), 3_600));
    }

    #[test]
    fn a_stand_in_that_does_no_work_fails_the_decision_table() {
        // The null arm. Three detectors that do not read their inputs, scored against the same
        // table. Each must disagree with the shipped decision somewhere, or the table cannot tell
        // the algorithm apart from a stand-in.
        let cases = cases();
        let missed = |f: &dyn Fn(&Case) -> (OnsetReason, bool), firing: bool| -> usize {
            cases.iter().filter(|c| c.nudge == firing && f(c) != (c.reason, c.nudge)).count()
        };
        let never = |_: &Case| (OnsetReason::NoDip, false);
        let always = |_: &Case| (OnsetReason::Onset, true);
        // Reads the settings, the clock and the state, but never the R-R buffer: it is handed a
        // reading equal to the baseline it already holds, so nothing can dip.
        let deaf = |c: &Case| {
            let d = c.decide_on(&buffer(SEEDED.baseline_rmssd as u16));
            (d.reason, d.should_nudge)
        };
        assert_eq!(missed(&never, true), FIRING_ROWS, "a detector that never nudges must miss every firing row");
        assert_eq!(
            missed(&always, false),
            cases.len() - FIRING_ROWS,
            "a detector that always nudges must miss every refusal"
        );
        assert_eq!(missed(&deaf, true), FIRING_ROWS, "a detector deaf to the R-R buffer must miss every dip");
    }
}
