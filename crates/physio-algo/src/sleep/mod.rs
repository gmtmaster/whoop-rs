//! Sleep staging — per-30 s-epoch hypnogram over a detected in-bed span.
//!
//! The V2 (cardiorespiratory) recipe stages every strap: z-scored HR / HR-variability / motion emissions, a
//! deep gate on HR-flatness, a soft sleep-cycle prior, a self-calibrating jerk wake gate, an R-R RSA
//! respiration term, and Viterbi transition smoothing.
//!
//! Inputs are protocol-free (see [`input`]). [`analyze`] runs the whole pipeline (detect → stage → refine)
//! over a night's streams, and each stage is exported on its own so a caller can run just one:
//! [`detect_sessions`] (or [`detect_sessions_with`] under a chosen [`DetectParams`]) returns the in-bed
//! windows without staging them, [`stage_v2`] stages one already-detected `[start, end]` span,
//! [`refine_wake`] runs the wake refinement over a staging the caller already has, and
//! [`stage_refined`] is the two together. Staging splits once more: [`emissions_v2`] is the per-epoch
//! log-evidence and [`decode_v2`] the path search over it, so the two can be scored apart. Pure +
//! deterministic.
//!
//! [`refine_wake`]'s density gate declines silently on a sparse or absent step stream, so it is exported
//! as [`motion_dense`]: ask it first and a caller can never report an unrefined span as a refined one.

mod common;
mod detect;
mod input;
mod mainnight;
pub mod params;
mod refine;
mod v2;

use crate::hrv::HrvReadiness;

pub use input::{AccelSample, HrSample, RrRun, SleepInput, StepSample};
pub use params::Params;
pub use detect::{detect_sessions, detect_sessions_with, DetectParams, DetectedSpan};
pub use v2::{emissions_prepared as emissions_v2, epoch_starts as epoch_starts_v2, prepare as prepare_v2,
    segments_of as segments_v2, stage as stage_v2, stage_prepared as stage_v2_prepared,
    stage_with as stage_v2_with, viterbi as decode_v2, Prepared, DEEP_GATE_THRESH, STAGE_ORDER};
pub use refine::{
    is_motion_dense as motion_dense, motion_density, refine as refine_wake, refine_with as refine_wake_with,
    RefineParams, MIN_DENSE_FRACTION,
};
pub use mainnight::{
    bridge_adjacent, bridged_night_groups, habitual_midsleep_sec, main_night_group_indices,
    main_night_group_indices_scored, main_night_index, main_night_index_scored, main_night_selection,
    main_night_selection_scored, BridgedNightGroup, HistoryBlock, MainNightReason, MainNightSelection, NightBlock,
    ScoredNightBlock, HABITUAL_MIN_DAYS,
};

/// The stream bundle for one detection window: raw per-sample signals plus the wrist-off intervals, band
/// sleep-state `(ts, state)`, and the local UTC offset the daytime guard needs.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SleepStreams {
    pub hr: Vec<HrSample>,
    pub rr: Vec<RrRun>,
    pub accel: Vec<AccelSample>,
    pub steps: Vec<StepSample>,
    pub tz_offset_s: i64,
    pub wrist_off: Vec<(i64, i64)>,
    pub band_sleep_state: Vec<(i64, i32)>,
}

/// One detected sleep session: its span, the motion-refined V2 hypnogram, the derived scalars, and the
/// per-30 s-epoch motion + band sleep-state grids the caller persists beside the hypnogram.
#[derive(Debug, Clone, PartialEq)]
pub struct Session {
    pub start: i64,
    pub end: i64,
    pub efficiency: f64,
    pub resting_hr: Option<i32>,
    pub avg_hrv: Option<f64>,
    pub segments: Vec<StageSegment>,
    pub motion_grid: Vec<f64>,
    pub sleep_state_grid: Vec<i32>,
}

/// Detect in-bed spans from a window's streams and stage each with the V2 recipe + motion-aware wake
/// refinement, returning one [`Session`] per accepted span with its resting HR and windowed average HRV.
pub fn analyze(streams: &SleepStreams) -> Vec<Session> {
    let spans = detect::detect_sessions(
        &streams.hr,
        &streams.accel,
        streams.tz_offset_s,
        &streams.wrist_off,
        &streams.band_sleep_state,
        None,
    );
    let mut beats: Vec<(u32, u16)> = Vec::new();
    for run in &streams.rr {
        for &ms in &run.intervals {
            beats.push((run.ts as u32, ms));
        }
    }
    beats.sort_by_key(|b| b.0);

    let mut out = Vec::with_capacity(spans.len());
    for span in spans {
        let input = SleepInput {
            start: span.start,
            end: span.end,
            hr: streams.hr.clone(),
            rr: streams.rr.clone(),
            accel: streams.accel.clone(),
        };
        let segments = refine::refine(&v2::stage(&input), &streams.accel, &streams.steps);
        let efficiency = detect::efficiency(span.start, span.end, &segments);
        let avg_hrv = HrvReadiness::windowed_avg_hrv(span.start as u32, span.end as u32, &beats);
        let motion_grid = detect::session_epoch_motion(span.start, span.end, &streams.accel);
        let sleep_state_grid = detect::session_epoch_sleep_state(span.start, span.end, &streams.band_sleep_state);
        out.push(Session {
            start: span.start,
            end: span.end,
            efficiency,
            resting_hr: span.resting_hr,
            avg_hrv,
            segments,
            motion_grid,
            sleep_state_grid,
        });
    }
    out
}

/// Stage a single detected span with the V2 recipe + the motion-aware wake refinement — the single-span
/// re-stage a caller runs after editing a session's bounds.
pub fn stage_refined(input: &SleepInput, steps: &[StepSample]) -> Vec<StageSegment> {
    refine::refine(&v2::stage(input), &input.accel, steps)
}

/// A sleep stage. String forms are `"wake" | "light" | "deep" | "rem"` for cross-platform parity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SleepStage {
    Wake,
    Light,
    Deep,
    Rem,
}

impl SleepStage {
    /// The JSON label, identical across the platform twins.
    pub fn as_str(self) -> &'static str {
        match self {
            SleepStage::Wake => "wake",
            SleepStage::Light => "light",
            SleepStage::Deep => "deep",
            SleepStage::Rem => "rem",
        }
    }
}

/// A contiguous run of one stage. Times are wall-clock unix seconds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StageSegment {
    pub start: i64,
    pub end: i64,
    pub stage: SleepStage,
}

#[cfg(test)]
mod golden_tests;
