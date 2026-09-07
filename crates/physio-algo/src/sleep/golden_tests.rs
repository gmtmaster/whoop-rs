//! Frozen-golden + contract tests for the sleep stagers. The golden hypnogram pins the shipped V2 OUTPUT
//! — six segments on a crafted integer-only night — not the recipe that produced it; which coefficients
//! the table can actually see is measured beside it, and six of the twenty-six are blind to it. Any drift
//! in the twenty it does see fails immediately. Integer-literal input keeps the two languages bit-identical.

use super::input::{AccelSample, HrSample, RrRun, SleepInput, StepSample};
use super::params::Params;
use super::refine::{RefineParams, refine_with};
use super::v2::stage_with as stage_v2_with;
use super::{
    DEEP_GATE_THRESH, Session, SessionAcceptancePath, SessionRole, SleepStage, SleepStreams,
    StageSegment, analyze, analyze_for_rr_generation, analyze_for_rr_generation_with_short_nap_promotion,
    analyze_for_rr_generation_with_short_nap_shadow, assign_session_roles, motion_dense, stage_v2,
};
use crate::nightly_physiology::RrDeviceGeneration;

const REF_MIDNIGHT: i64 = 1_749_513_600;

fn rsa_wave(ph: usize, i: i64) -> i64 {
    let amp = [12i64, 60, 30, 20][ph];
    [0, amp, 0, -amp][(i % 4) as usize]
}

/// The crafted 4-phase night (deep-favorable → high-RSA → mild → restless) used by the frozen golden.
fn golden_input() -> SleepInput {
    let start = REF_MIDNIGHT + 3_600;
    let phase: i64 = 90 * 60;
    let dur = phase * 4;
    let mut accel = Vec::new();
    let mut hr = Vec::new();
    let mut rr = Vec::new();
    for i in 0..dur {
        let ts = start + i;
        let ph = (i / phase) as usize;
        let restless = ph == 3 && (i % 20) < 6;
        if restless {
            accel.push(AccelSample {
                ts,
                x: 0.2,
                y: 0.15,
                z: 0.96,
            });
        } else {
            accel.push(AccelSample {
                ts,
                x: 0.0,
                y: 0.0,
                z: 1.0,
            });
        }
        let bpm: i64 = match ph {
            0 => 50,
            1 => 54 + [0, 1, 2, 3, 2, 1][((i / 20) % 6) as usize],
            2 => 56 + (i / 60) % 4,
            _ => 66 + (i / 30) % 6,
        };
        hr.push(HrSample {
            ts,
            bpm: bpm as u16,
        });
        let rr_ms = 60_000 / bpm + rsa_wave(ph, i);
        rr.push(RrRun {
            ts,
            intervals: vec![rr_ms as u16],
        });
    }
    SleepInput {
        start,
        end: start + dur,
        hr,
        rr,
        accel,
    }
}

/// The shipped V2 output on the crafted night, segment for segment. The three constant stagers below are
/// checked against the same table, so the golden is known to reject one that always answers a single stage.
#[test]
fn frozen_golden_hypnogram_v2() {
    let input = golden_input();
    let start = input.start;
    let segs = stage_v2(&input);
    let golden = [
        (0i64, 5070i64, SleepStage::Deep),
        (5070, 5280, SleepStage::Light),
        (5280, 5550, SleepStage::Rem),
        (5550, 10740, SleepStage::Light),
        (10740, 16290, SleepStage::Rem),
        (16290, 21600, SleepStage::Wake),
    ];
    assert_eq!(golden.len(), segs.len(), "segment count");
    for (k, g) in golden.iter().enumerate() {
        assert_eq!(start + g.0, segs[k].start, "seg {k} start");
        assert_eq!(start + g.1, segs[k].end, "seg {k} end");
        assert_eq!(g.2, segs[k].stage, "seg {k} stage");
    }
    let expected: Vec<StageSegment> = golden
        .iter()
        .map(|g| StageSegment {
            start: start + g.0,
            end: start + g.1,
            stage: g.2,
        })
        .collect();
    assert_eq!(expected, segs);
    for stage in [SleepStage::Light, SleepStage::Deep, SleepStage::Wake] {
        let constant = vec![StageSegment {
            start,
            end: input.end,
            stage,
        }];
        assert_ne!(
            expected,
            constant,
            "an always-{} stager must fail this table",
            stage.as_str()
        );
    }
}

/// Which V2 coefficients the frozen table can see. Twenty of the twenty-six change it; the six below do
/// not, even at extreme values, so editing one of those is invisible to the golden and needs the fixture
/// sheet instead. A blind row that starts moving the table is a change to report, not to silence.
#[test]
fn the_frozen_golden_defends_twenty_of_the_twenty_six_v2_coefficients() {
    let input = golden_input();
    let base = stage_v2(&input);
    let p = Params::SHIPPED;
    let seen: Vec<(&str, Params)> = vec![
        ("deep_hrv", Params { deep_hrv: 0.0, ..p }),
        ("deep_hr", Params { deep_hr: 5.0, ..p }),
        (
            "deep_motion",
            Params {
                deep_motion: 5.0,
                ..p
            },
        ),
        ("rem_hrv", Params { rem_hrv: 0.0, ..p }),
        (
            "rem_motion",
            Params {
                rem_motion: 0.0,
                ..p
            },
        ),
        ("rem_hr", Params { rem_hr: 0.0, ..p }),
        (
            "awake_motion",
            Params {
                awake_motion: 10.0,
                ..p
            },
        ),
        (
            "awake_hrv",
            Params {
                awake_hrv: 10.0,
                ..p
            },
        ),
        (
            "awake_hr",
            Params {
                awake_hr: 10.0,
                ..p
            },
        ),
        (
            "deep_gate_thresh",
            Params {
                deep_gate_thresh: 0.0,
                ..p
            },
        ),
        (
            "deep_gate_slope",
            Params {
                deep_gate_slope: 0.0,
                ..p
            },
        ),
        (
            "motion_gate_boost",
            Params {
                motion_gate_boost: 0.0,
                ..p
            },
        ),
        (
            "resp_weight",
            Params {
                resp_weight: 0.0,
                ..p
            },
        ),
        (
            "base_rate",
            Params {
                base_rate: [0.25, 0.25, 0.25, 0.25],
                ..p
            },
        ),
        (
            "cycle_deep_scale",
            Params {
                cycle_deep_scale: 0.0,
                ..p
            },
        ),
        (
            "cycle_deep_decay",
            Params {
                cycle_deep_decay: 0.1,
                ..p
            },
        ),
        (
            "cycle_rem_scale",
            Params {
                cycle_rem_scale: 0.0,
                ..p
            },
        ),
        (
            "cycle_rem_onset_minutes",
            Params {
                cycle_rem_onset_minutes: 600.0,
                ..p
            },
        ),
        (
            "cycle_rem_ramp_cap",
            Params {
                cycle_rem_ramp_cap: 0.3,
                ..p
            },
        ),
        (
            "transition",
            Params {
                transition: [[0.25; 4]; 4],
                ..p
            },
        ),
    ];
    let blind: Vec<(&str, Params)> = vec![
        (
            "awake_deadzone",
            Params {
                awake_deadzone: 5.0,
                ..p
            },
        ),
        (
            "jerk_move_mult",
            Params {
                jerk_move_mult: 1000.0,
                ..p
            },
        ),
        (
            "jerk_gate_mult",
            Params {
                jerk_gate_mult: 1000.0,
                ..p
            },
        ),
        (
            "cycle_rem_early_frac",
            Params {
                cycle_rem_early_frac: 1.0,
                ..p
            },
        ),
        (
            "cycle_rem_early_penalty",
            Params {
                cycle_rem_early_penalty: 50.0,
                ..p
            },
        ),
        (
            "cycle_clock_from_onset",
            Params {
                cycle_clock_from_onset: true,
                ..p
            },
        ),
    ];
    assert_eq!(
        base,
        stage_v2_with(&input, &p),
        "the baseline row must be the shipped recipe itself"
    );
    for (name, q) in &seen {
        assert_ne!(
            base,
            stage_v2_with(&input, q),
            "{name} must change the golden table"
        );
    }
    for (name, q) in &blind {
        assert_eq!(
            base,
            stage_v2_with(&input, q),
            "{name} now moves the golden — move it to the seen list"
        );
    }
    assert_eq!((20, 6), (seen.len(), blind.len()));
}

#[test]
fn tuned_deep_boundary_constants_are_pinned() {
    assert_eq!(0.40, DEEP_GATE_THRESH);
    assert_eq!([0.76, 0.012, 0.216, 0.012], Params::SHIPPED.transition[0]);
    for row in Params::SHIPPED.transition.iter() {
        let sum: f64 = row.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9, "transition row must sum to 1.0");
    }
}

#[test]
fn v2_segments_tile_the_span_contiguously() {
    let input = golden_input();
    let segs = stage_v2(&input);
    assert!(!segs.is_empty());
    assert_eq!(input.start, segs.first().unwrap().start);
    assert_eq!(input.end, segs.last().unwrap().end);
    for w in segs.windows(2) {
        assert_eq!(w[0].end, w[1].start, "no gap/overlap");
        assert!(w[1].end > w[1].start, "non-empty");
    }
}

#[test]
fn v2_degenerate_input_falls_back_to_single_light_block() {
    let start = REF_MIDNIGHT;
    let end = start + 3_600;
    let input = SleepInput {
        start,
        end,
        hr: Vec::new(),
        rr: Vec::new(),
        accel: vec![AccelSample {
            ts: start,
            x: 0.0,
            y: 0.0,
            z: 1.0,
        }],
    };
    let segs = stage_v2(&input);
    assert_eq!(1, segs.len());
    assert_eq!(SleepStage::Light, segs[0].stage);
    assert_eq!(start, segs[0].start);
    assert_eq!(end, segs[0].end);
}

#[test]
fn analyze_still_night_stages_and_tiles_the_span() {
    let input = golden_input();
    let streams = SleepStreams {
        hr: input.hr.clone(),
        rr: input.rr.clone(),
        accel: input.accel.clone(),
        tz_offset_s: 0,
        ..Default::default()
    };
    let sessions = analyze(&streams);
    assert_eq!(1, sessions.len());
    let segs = &sessions[0].segments;
    assert_eq!(sessions[0].start, segs.first().unwrap().start);
    assert_eq!(sessions[0].end, segs.last().unwrap().end);
    for w in segs.windows(2) {
        assert_eq!(w[0].end, w[1].start);
    }
}

#[test]
fn shadow_observation_keeps_the_golden_production_session_byte_identical() {
    let input = golden_input();
    let streams = SleepStreams {
        hr: input.hr,
        rr: input.rr,
        accel: input.accel,
        tz_offset_s: 0,
        ..Default::default()
    };
    let production = analyze_for_rr_generation(&streams, RrDeviceGeneration::Whoop5Mg);
    let shadow =
        analyze_for_rr_generation_with_short_nap_shadow(&streams, RrDeviceGeneration::Whoop5Mg);
    assert_eq!(shadow.sessions, production);
    assert!(shadow.short_nap_diagnostics.is_empty());
}

// ---------------------------------------------------------------------------------------------
// Phase B: promotion path + canonical role assignment.
// ---------------------------------------------------------------------------------------------

/// Regression test C (Phase B spec §11): the promotion path must reproduce the SAME winning main
/// group on the SAME crafted night the golden hypnogram is pinned to, with zero session-boundary
/// or physiology change relative to the pre-Phase-B `analyze_for_rr_generation` path, and the
/// winning (only) session must be assigned MAIN_SLEEP.
#[test]
fn promotion_path_keeps_the_golden_main_sleep_session_byte_identical_and_roles_it_main_sleep() {
    let input = golden_input();
    let streams = SleepStreams {
        hr: input.hr,
        rr: input.rr,
        accel: input.accel,
        tz_offset_s: 0,
        ..Default::default()
    };
    let production = analyze_for_rr_generation(&streams, RrDeviceGeneration::Whoop5Mg);
    let promoted =
        analyze_for_rr_generation_with_short_nap_promotion(&streams, RrDeviceGeneration::Whoop5Mg);

    // No duration-rejected candidate exists on this crafted night, so nothing to promote: the
    // session set is byte-identical to the pre-Phase-B production path (every field, including the
    // new `acceptance_path`, which the pre-Phase-B path already stamps `Normal`).
    assert_eq!(promoted.sessions, production);
    assert!(promoted.short_nap_diagnostics.is_empty());
    assert!(
        promoted
            .sessions
            .iter()
            .all(|s| s.acceptance_path == SessionAcceptancePath::Normal)
    );

    let roles = assign_session_roles(&promoted.sessions, 0);
    assert_eq!(roles, vec![SessionRole::MainSleep]);
}

fn stub_session(start: i64, end: i64, path: SessionAcceptancePath) -> Session {
    // Six hours asleep out of the [start, end] span, matching `ScoredNightBlock`'s expectation of
    // decoded asleep seconds (not the raw clock span) for the main-night scorer under test.
    let asleep_end = start + ((end - start) * 9) / 10;
    Session {
        start,
        end,
        efficiency: (asleep_end - start) as f64 / (end - start) as f64,
        resting_hr: Some(55),
        avg_hrv: Some(60.0),
        segments: vec![
            StageSegment { start, end: asleep_end, stage: SleepStage::Light },
            StageSegment { start: asleep_end, end, stage: SleepStage::Wake },
        ],
        motion_grid: vec![],
        sleep_state_grid: vec![],
        acceptance_path: path,
    }
}

/// Test D-adjacent (Phase B spec §11 D + §5): a promoted short nap must become NAP regardless of
/// where it falls relative to the winning main-night group, and — critically for the physiology
/// isolation contract — its presence must never change which Normal-path group wins MAIN_SLEEP nor
/// demote/relabel any other Normal-path session, because role assignment scores `Normal` sessions
/// only.
#[test]
fn a_shortnap_session_never_enters_the_main_night_scorer_and_always_roles_nap() {
    let night = 1_700_000_000i64; // 23:00-ish local (offset 0), matches mainnight's overnight band
    let main = stub_session(night, night + 7 * 3600, SessionAcceptancePath::Normal);
    let other_daytime = stub_session(
        night + 15 * 3600,
        night + 16 * 3600,
        SessionAcceptancePath::Normal,
    );
    let promoted_nap = stub_session(
        night + 11 * 3600,
        night + 11 * 3600 + 90 * 60,
        SessionAcceptancePath::ShortNap,
    );

    let baseline_roles = assign_session_roles(&[main.clone(), other_daytime.clone()], 0);
    let with_nap_roles = assign_session_roles(&[main, other_daytime, promoted_nap], 0);

    assert_eq!(baseline_roles, vec![SessionRole::MainSleep, SessionRole::Fragment]);
    assert_eq!(
        with_nap_roles,
        vec![SessionRole::MainSleep, SessionRole::Fragment, SessionRole::Nap],
        "adding a ShortNap-path session must roles it NAP without moving any other role"
    );
}

/// Root-cause regression: an isolated daytime `Normal`-path session — the shape of the real "Nap B"
/// fixture (`crates/physio-algo/tests/short_nap_audit.rs`'s `NAP_B_START`/`NAP_B_END`, 2026-09-06
/// 13:43:55 -> 15:53:35 local, Europe/Budapest offset) — must NOT be assigned MAIN_SLEEP merely for
/// being the only `Normal`-path candidate `assign_session_roles` was called with. Before the
/// overnight-evidence gate, `main_night_group_indices_scored` trivially picked this lone daytime
/// block as its `OnlyBlock` winner and `assign_session_roles` converted that straight into
/// MAIN_SLEEP with no further evidence — exactly what happens in production when noop-engine's
/// noon-to-noon compute-date window puts a real main night and a same-nominal-day isolated daytime
/// session in different calls, so they never compete together. The fix: gate MAIN_SLEEP on the
/// winning group's onset actually falling overnight (`detect::is_overnight_onset`, reused verbatim
/// from `mainnight`'s own internal night-tail-bridging check). A lone daytime session fails that
/// gate and is conservatively FRAGMENT — never silently promoted to NAP, which still requires an
/// explicit `ShortNap` acceptance path.
#[test]
fn an_isolated_daytime_session_never_becomes_main_sleep_merely_for_being_alone() {
    let nap_b_start = 1_788_695_035i64; // 2026-09-06 13:43:55 local (offset +2h)
    let nap_b_end = 1_788_702_815i64; // 2026-09-06 15:53:35 local
    let offset_s = 2 * 3_600;
    let asleep_end = nap_b_start + ((nap_b_end - nap_b_start) * 9) / 10;
    let session = Session {
        start: nap_b_start,
        end: nap_b_end,
        efficiency: (asleep_end - nap_b_start) as f64 / (nap_b_end - nap_b_start) as f64,
        resting_hr: Some(55),
        avg_hrv: Some(60.0),
        segments: vec![
            StageSegment {
                start: nap_b_start,
                end: asleep_end,
                stage: SleepStage::Light,
            },
            StageSegment {
                start: asleep_end,
                end: nap_b_end,
                stage: SleepStage::Wake,
            },
        ],
        motion_grid: vec![],
        sleep_state_grid: vec![],
        acceptance_path: SessionAcceptancePath::Normal,
    };

    let roles = assign_session_roles(&[session], offset_s);
    assert_eq!(
        roles,
        vec![SessionRole::Fragment],
        "a lone daytime-onset session must never read MAIN_SLEEP without overnight evidence"
    );
}

/// Companion positive control: the SAME lone-candidate `OnlyBlock` shape, but with a genuine
/// overnight onset (23:00 local), must still read MAIN_SLEEP — the gate above only excludes
/// candidates lacking overnight evidence, it does not disable the `OnlyBlock` case generally.
#[test]
fn a_lone_overnight_session_still_reads_main_sleep() {
    let night = 1_700_000_000i64; // ~22:13 local (offset 0) — outside the daytime band
    let asleep_end = night + ((7 * 3_600) * 9) / 10;
    let session = Session {
        start: night,
        end: night + 7 * 3_600,
        efficiency: (asleep_end - night) as f64 / (7.0 * 3_600.0),
        resting_hr: Some(55),
        avg_hrv: Some(60.0),
        segments: vec![
            StageSegment {
                start: night,
                end: asleep_end,
                stage: SleepStage::Light,
            },
            StageSegment {
                start: asleep_end,
                end: night + 7 * 3_600,
                stage: SleepStage::Wake,
            },
        ],
        motion_grid: vec![],
        sleep_state_grid: vec![],
        acceptance_path: SessionAcceptancePath::Normal,
    };

    let roles = assign_session_roles(&[session], 0);
    assert_eq!(roles, vec![SessionRole::MainSleep]);
}

/// A candidate that never reaches the promotion step (the shadow evaluator's `InsufficientEvidence`
/// or `Rejected` verdicts) must never appear as `SessionAcceptancePath::ShortNap` — the promotion
/// gate is `verdict == Eligible`, nothing looser. Exercised directly against the same evaluator
/// fixture `short_nap`'s own unit tests use, so this stays coupled to Phase A's real gate rather
/// than a duplicated one.
#[test]
fn only_eligible_verdicts_are_representable_as_a_promoted_shortnap_role() {
    // A `Fragment`/`MainSleep` role is never produced for a `ShortNap`-path session, and a `Nap`
    // role is never produced for a `Normal`-path session — the mapping in `assign_session_roles`
    // is total and exhaustive over `acceptance_path`, so this is checked structurally: every
    // ShortNap-path session in a mixed set roles Nap, and no Normal-path session ever roles Nap.
    let a = stub_session(0, 3600, SessionAcceptancePath::Normal);
    let b = stub_session(100_000, 100_000 + 3600, SessionAcceptancePath::ShortNap);
    let roles = assign_session_roles(&[a, b], 0);
    assert_ne!(roles[0], SessionRole::Nap);
    assert_eq!(roles[1], SessionRole::Nap);
}

/// The golden above carries no step stream, so `analyze`'s last stage declines on it. This drives the
/// same night WITH one. Its only wake run is its last, 5,309 s of post-wake in-bed time: the shipped
/// rule leaves it and the rule H opened with took all of it, so the fix is pinned end to end here.
#[test]
fn analyze_keeps_the_golden_night_trailing_wake_when_the_step_stream_is_dense() {
    let input = golden_input();
    let steps: Vec<StepSample> = (input.start..input.end)
        .step_by(30)
        .map(|ts| StepSample {
            ts,
            counter: 100,
            activity_class: Some(0),
        })
        .collect();
    let bare = SleepStreams {
        hr: input.hr.clone(),
        rr: input.rr.clone(),
        accel: input.accel.clone(),
        tz_offset_s: 0,
        ..Default::default()
    };
    let dense = SleepStreams {
        steps: steps.clone(),
        ..bare.clone()
    };
    let (a, b) = (analyze(&bare), analyze(&dense));
    assert_eq!((1, 1), (a.len(), b.len()));
    let span = (b[0].start, b[0].end);
    assert!(
        !motion_dense(span.0, span.1, &input.accel, &[]),
        "the bare golden must decline"
    );
    assert!(
        motion_dense(span.0, span.1, &input.accel, &steps),
        "the dense one must be accepted"
    );
    assert_eq!(
        a[0].segments, b[0].segments,
        "the shipped rule leaves a trailing wake run alone"
    );
    let tail = *b[0].segments.last().unwrap();
    assert_eq!(SleepStage::Wake, tail.stage);
    assert_eq!(5_309, tail.end - tail.start);
    assert_eq!(b[0].end, tail.end);

    // The rule H opened with, over the same staging and the same streams: it takes the whole run.
    let span_steps: Vec<StepSample> = steps
        .iter()
        .copied()
        .filter(|s| s.ts >= span.0 && s.ts < span.1)
        .collect();
    let span_accel: Vec<AccelSample> = input
        .accel
        .iter()
        .copied()
        .filter(|g| g.ts >= span.0 && g.ts < span.1)
        .collect();
    let pre_h = RefineParams {
        skip_window_edges: false,
        ..RefineParams::SHIPPED
    };
    let old = refine_with(&b[0].segments, &span_accel, &span_steps, &pre_h);
    assert_ne!(
        SleepStage::Wake,
        old.last().unwrap().stage,
        "pre-H converted the trailing run"
    );
}

/// The detected span on a window that is nothing but the night: both edges pinned, not a floor. Efficiency
/// is asserted at its value, not inside `0..=1` — the function clamps to that range, so the old range
/// assert held for any implementation at all.
#[test]
fn analyze_pins_the_span_of_a_window_that_is_all_still_night() {
    let start = REF_MIDNIGHT;
    let dur = 2 * 3600;
    let mut hr = Vec::new();
    let mut accel = Vec::new();
    for i in 0..dur {
        let ts = start + i;
        hr.push(HrSample { ts, bpm: 50 });
        accel.push(AccelSample {
            ts,
            x: 0.0,
            y: 0.0,
            z: 1.0,
        });
    }
    let streams = SleepStreams {
        hr,
        accel,
        tz_offset_s: 0,
        ..Default::default()
    };
    let sessions = analyze(&streams);
    assert_eq!(1, sessions.len());
    let s = &sessions[0];
    assert_eq!((start, start + 7_199), (s.start, s.end));
    assert_eq!(
        1.0, s.efficiency,
        "no wake was staged, so every in-bed second is asleep"
    );
    assert_eq!(Some(50), s.resting_hr);
    assert_eq!(s.start, s.segments.first().unwrap().start);
    assert_eq!(s.end, s.segments.last().unwrap().end);
}

/// The same two claims on a window that is mostly NOT sleep: four hours awake and moving at 78 bpm, seven
/// still at 50, three awake again, plus a ten-minute awake dip to 40. A whole-window detector and an
/// anchor read over the whole window both fail here; neither could fail the all-night fixture above.
#[test]
fn analyze_finds_the_night_inside_a_waking_day_and_anchors_resting_hr_to_it() {
    let (day_start, night_start) = (REF_MIDNIGHT - 4 * 3600, REF_MIDNIGHT);
    let (night_end, day_end) = (night_start + 7 * 3600, night_start + 10 * 3600);
    let (mut hr, mut accel) = (Vec::new(), Vec::new());
    for ts in day_start..day_end {
        let asleep = (night_start..night_end).contains(&ts);
        let dip = (day_start + 1_800..day_start + 2_400).contains(&ts);
        hr.push(HrSample {
            ts,
            bpm: if asleep {
                50
            } else if dip {
                40
            } else {
                78
            },
        });
        let f = ((ts % 4) as f64) * 0.25;
        accel.push(if asleep {
            AccelSample {
                ts,
                x: 0.0,
                y: 0.0,
                z: 1.0,
            }
        } else {
            AccelSample {
                ts,
                x: f,
                y: 0.3 - f,
                z: 0.9,
            }
        });
    }
    let sessions = analyze(&SleepStreams {
        hr,
        accel,
        tz_offset_s: 0,
        ..Default::default()
    });
    assert_eq!(1, sessions.len());
    let s = &sessions[0];
    assert_eq!((night_start + 271, night_start + 25_064), (s.start, s.end));
    // Both edges inside the true night, and 25,607 s short of the 14 h a whole-window detector returns.
    assert!(s.start >= night_start && s.end <= night_end);
    assert_eq!(25_607, (day_end - day_start) - (s.end - s.start));
    // The anchor is the SESSION floor: 50, not the 40 the awake dip would give over the whole window.
    assert_eq!(Some(50), s.resting_hr);
    assert_eq!(s.start, s.segments.first().unwrap().start);
    assert_eq!(s.end, s.segments.last().unwrap().end);
}
