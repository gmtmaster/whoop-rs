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
mod short_nap;
mod v2;

pub use detect::{
    DAYTIME_BAND_END_HOUR, DAYTIME_BAND_START_HOUR, DetectParams, DetectedSpan, MAX_GAP_MIN,
    SPARSE_GRAVITY_SPAN_FRAC, detect_sessions, detect_sessions_with, efficiency,
};
pub use input::{AccelSample, HrSample, RrRun, SleepInput, StepSample};
pub use mainnight::{
    BridgedNightGroup, HABITUAL_MIN_DAYS, HABITUAL_WINDOW_DAYS, HistoryBlock, MainNightReason,
    MainNightSelection, NightBlock, OVERNIGHT_END_HOUR, OVERNIGHT_START_HOUR, ScoredNightBlock,
    bridge_adjacent, bridged_night_groups, habitual_midsleep_sec, habitual_midsleep_series,
    main_night_group_indices, main_night_group_indices_scored, main_night_index,
    main_night_index_scored, main_night_selection, main_night_selection_scored,
};
pub use params::Params;
pub use refine::{
    MIN_DENSE_FRACTION, RefineParams, is_motion_dense as motion_dense, motion_density,
    refine as refine_wake, refine_with as refine_wake_with,
};
pub use short_nap::{
    ShortNapCorroborators, ShortNapOriginalRejection, ShortNapShadowDiagnostic,
    ShortNapShadowVerdict,
};
pub use v2::{
    DEEP_GATE_THRESH, EpochDiagnostic, Prepared, STAGE_ORDER, STAGING_ALGORITHM_VERSION,
    diagnostics_prepared as diagnostics_v2, emissions_prepared as emissions_v2,
    epoch_starts as epoch_starts_v2, prepare as prepare_v2, segments_of as segments_v2,
    stage as stage_v2, stage_prepared as stage_v2_prepared, stage_with as stage_v2_with,
    viterbi as decode_v2,
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

/// How a session reached the canonical set. `Normal` is the existing accepted-candidate path
/// (detector + duration policy), unchanged since before Phase B. `ShortNap` is a Phase-A ELIGIBLE
/// short-nap candidate promoted into a canonical session by Phase B's [`analyze_for_rr_generation_with_short_nap_promotion`].
///
/// Downstream physiology aggregation (respiratory rate, skin temperature, main-night grouping) MUST
/// filter to `Normal` only — a promoted short nap must never contribute to those. See
/// `sleep::short_nap`'s module doc and the Phase B physiology-isolation contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionAcceptancePath {
    Normal,
    ShortNap,
}

impl SessionAcceptancePath {
    /// The stable, explicit serialized value. Never re-derived downstream.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Normal => "NORMAL",
            Self::ShortNap => "SHORT_NAP",
        }
    }
}

/// Canonical session role. Assigned only by [`assign_session_roles`], which needs the FULL day's
/// session set (including any promoted short nap) and runs the existing main-night grouping over
/// it — never inferred from time-of-day, never re-derived independently in Java/Swift.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionRole {
    /// Belongs to the winning main-night group (existing `mainnight` selection) AND that group's
    /// onset carries real overnight evidence — see [`assign_session_roles`]'s doc for why the raw
    /// "winning group" alone is not sufficient.
    MainSleep,
    /// An independently accepted daytime nap: either a Phase-A promoted short nap, or (not yet
    /// implemented — see Phase B report) an existing Normal-path session with enough evidence to
    /// be conservatively distinguished from a detached night fragment.
    Nap,
    /// An accepted sleep session that is neither main sleep nor an accepted canonical nap.
    Fragment,
}

impl SessionRole {
    /// The stable, explicit serialized value. Never re-derived downstream.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::MainSleep => "MAIN_SLEEP",
            Self::Nap => "NAP",
            Self::Fragment => "FRAGMENT",
        }
    }
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
    /// How this session reached the canonical set. `Normal` for every session accepted before
    /// Phase B; `ShortNap` only for a Phase-A ELIGIBLE candidate promoted by Phase B.
    pub acceptance_path: SessionAcceptancePath,
}

/// Production sessions plus isolated, non-authoritative observations of duration-rejected candidates.
#[derive(Debug, Clone, PartialEq)]
pub struct SleepAnalysisWithShadow {
    pub sessions: Vec<Session>,
    pub short_nap_diagnostics: Vec<ShortNapShadowDiagnostic>,
}

/// Phase B: production sessions, WITH any Phase-A ELIGIBLE short nap promoted in
/// (`acceptance_path == ShortNap`), plus the same shadow diagnostics the Phase A path exposes.
/// See [`analyze_for_rr_generation_with_short_nap_promotion`].
#[derive(Debug, Clone, PartialEq)]
pub struct SleepAnalysisWithRoles {
    pub sessions: Vec<Session>,
    pub short_nap_diagnostics: Vec<ShortNapShadowDiagnostic>,
}

/// Detect in-bed spans from a window's streams and stage each with the V2 recipe + motion-aware wake
/// refinement, returning one [`Session`] per accepted span with its resting HR and windowed average HRV.
pub fn analyze(streams: &SleepStreams) -> Vec<Session> {
    analyze_for_rr_generation(streams, crate::nightly_physiology::RrDeviceGeneration::Whoop5Mg)
}

pub fn analyze_for_rr_generation(
    streams: &SleepStreams,
    rr_generation: crate::nightly_physiology::RrDeviceGeneration,
) -> Vec<Session> {
    let spans = detect::detect_sessions(
        &streams.hr,
        &streams.accel,
        streams.tz_offset_s,
        &streams.wrist_off,
        &streams.band_sleep_state,
        None,
    );
    stage_sessions(streams, rr_generation, spans)
}

/// Runs the normal pipeline unchanged and separately observes candidates rejected only by duration policy.
pub fn analyze_for_rr_generation_with_short_nap_shadow(
    streams: &SleepStreams,
    rr_generation: crate::nightly_physiology::RrDeviceGeneration,
) -> SleepAnalysisWithShadow {
    let observed = detect::detect_sessions_with_duration_rejections(
        &streams.hr, &streams.accel, streams.tz_offset_s, &streams.wrist_off,
        &streams.band_sleep_state, None,
    );
    let short_nap_diagnostics = observed.duration_rejected.into_iter().map(|candidate| {
        short_nap::evaluate(
            candidate, &streams.hr, &streams.rr, &streams.accel, &streams.steps,
            &streams.band_sleep_state,
        )
    }).collect();
    SleepAnalysisWithShadow {
        sessions: stage_sessions(streams, rr_generation, observed.accepted),
        short_nap_diagnostics,
    }
}

fn stage_sessions(
    streams: &SleepStreams,
    rr_generation: crate::nightly_physiology::RrDeviceGeneration,
    spans: Vec<DetectedSpan>,
) -> Vec<Session> {
    spans
        .into_iter()
        .map(|span| {
            stage_one_span(
                streams,
                rr_generation,
                span.start,
                span.end,
                SessionAcceptancePath::Normal,
            )
        })
        .collect()
}

/// Stage one span with the V2 recipe + wake refinement and its OWN isolated per-session physiology
/// (never a cross-session aggregate — safe to call for a promoted short nap without leaking into
/// any other day's or session's physiology). `acceptance_path` is stamped verbatim onto the result.
fn stage_one_span(
    streams: &SleepStreams,
    rr_generation: crate::nightly_physiology::RrDeviceGeneration,
    start: i64,
    end: i64,
    acceptance_path: SessionAcceptancePath,
) -> Session {
    let input = SleepInput {
        start,
        end,
        hr: streams.hr.clone(),
        rr: streams.rr.clone(),
        accel: streams.accel.clone(),
    };
    let segments = refine::refine(&v2::stage(&input), &streams.accel, &streams.steps);
    let physiology = crate::nightly_physiology::nightly_physiology_for_generation(
        start,
        end,
        &streams.hr,
        &streams.accel,
        &streams.rr,
        &streams.wrist_off,
        &segments,
        rr_generation,
    );
    // Authoritative per-session RHR: v4 quality-weighted (`quality_v4`), matching the v4
    // promotion in noop-engine's grouped/authoritative nightly RHR. `physiology.rhr` (v3) is
    // still computed on `physiology` for research/parity but no longer read here.
    let resting_hr = physiology.rhr_v4.quality_v4;
    let efficiency = detect::efficiency(start, end, &segments);
    let avg_hrv = physiology.hrv.rmssd_ms;
    let motion_grid = detect::session_epoch_motion(start, end, &streams.accel);
    let sleep_state_grid = detect::session_epoch_sleep_state(start, end, &streams.band_sleep_state);
    Session {
        start,
        end,
        efficiency,
        resting_hr,
        avg_hrv,
        segments,
        motion_grid,
        sleep_state_grid,
        acceptance_path,
    }
}

/// Production sessions (unchanged Normal path) plus any Phase-A ELIGIBLE short-nap candidate
/// promoted into the canonical set. Additive over [`analyze_for_rr_generation_with_short_nap_shadow`]:
/// the normal accepted-candidate path is byte-identical; a duration-only-rejected candidate is
/// promoted into a canonical `Session` with `acceptance_path = ShortNap` IF AND ONLY IF the SAME
/// Phase A evaluator (`short_nap::evaluate`, unmodified) returns `Eligible` for it — no new
/// detector, no changed candidate boundaries, no rerun of the legacy `nap` module. A candidate
/// that is `InsufficientEvidence` or `Rejected` is never promoted and is not part of `sessions`.
///
/// Callers MUST exclude `acceptance_path == ShortNap` sessions from every physiology aggregation
/// input (respiratory rate, skin temperature, main-night grouping) — see [`SessionAcceptancePath`]'s
/// doc. `short_nap_diagnostics` is unchanged from the shadow path: still one diagnostic per
/// duration-rejected candidate, promoted or not, so existing shadow observability is preserved.
pub fn analyze_for_rr_generation_with_short_nap_promotion(
    streams: &SleepStreams,
    rr_generation: crate::nightly_physiology::RrDeviceGeneration,
) -> SleepAnalysisWithRoles {
    let observed = detect::detect_sessions_with_duration_rejections(
        &streams.hr, &streams.accel, streams.tz_offset_s, &streams.wrist_off,
        &streams.band_sleep_state, None,
    );
    let short_nap_diagnostics: Vec<ShortNapShadowDiagnostic> = observed
        .duration_rejected
        .iter()
        .map(|candidate| {
            short_nap::evaluate(
                *candidate, &streams.hr, &streams.rr, &streams.accel, &streams.steps,
                &streams.band_sleep_state,
            )
        })
        .collect();

    let mut sessions = stage_sessions(streams, rr_generation, observed.accepted);
    for (candidate, diagnostic) in observed.duration_rejected.iter().zip(short_nap_diagnostics.iter()) {
        if diagnostic.verdict == ShortNapShadowVerdict::Eligible {
            sessions.push(stage_one_span(
                streams,
                rr_generation,
                candidate.period.start,
                candidate.period.end,
                SessionAcceptancePath::ShortNap,
            ));
        }
    }
    sessions.sort_by(|a, b| a.start.cmp(&b.start).then(a.end.cmp(&b.end)));

    SleepAnalysisWithRoles { sessions, short_nap_diagnostics }
}

/// Assign each session's canonical [`SessionRole`] from the FULL set of this day's sessions
/// (including any promoted short nap) plus the local UTC offset main-night grouping needs.
///
/// MAIN_SLEEP: the winning main-night GROUP, scored the existing way
/// (`mainnight::main_night_group_indices_scored`, unchanged) over `Normal`-path sessions only — a
/// promoted short nap is never a main-night candidate, so it cannot distort the group the way
/// including it in the scorer's input would — AND ONLY IF that winning group's own onset carries
/// real overnight evidence (`detect::is_overnight_onset`, the SAME signal `mainnight`'s own
/// `bridged_night_groups` already reads internally for night-tail bridging — see its
/// `substantial_night_tail` check). This gate exists because `main_night_index_scored`/
/// `main_night_group_indices_scored` are RANKERS, not classifiers: given any non-empty candidate
/// list they always return a winner (`None` only for an empty list — see their own doc), including
/// a list holding exactly one isolated daytime session with nothing to rank against. Before Phase B
/// this never mattered because their only caller always fed a genuine main-night candidate — with
/// Phase B invoking them per compute-date over the noon-to-noon window (see noop-engine's
/// `DailyComputeRequest::sleep_*` doc and the canonical-daily-convergence-plan's noon-to-noon
/// section), a real main night and a same-"day" isolated daytime session (e.g. "Nap B") can land in
/// DIFFERENT compute-date buckets and so never compete in the same call — the isolated daytime one
/// would otherwise trivially "win" its own one-candidate group and be mislabeled MAIN_SLEEP. The
/// gate does NOT add a new heuristic or threshold: it reuses the exact overnight/daytime boundary
/// (`DAYTIME_BAND_START_HOUR`/`DAYTIME_BAND_END_HOUR`, the same window `OVERNIGHT_START_HOUR`/
/// `OVERNIGHT_END_HOUR` describe from the other side) `mainnight.rs` itself already depends on. A
/// winning group that fails the gate is NOT reclassified NAP — it stays FRAGMENT, since NAP still
/// requires the explicit, evidence-corroborated `ShortNap` promotion path below; this keeps the
/// existing "additional session != NAP" contract intact.
///
/// FRAGMENT: every other `Normal`-path session, plus a winning group that fails the overnight-
/// evidence gate above. Phase B does not have architecture evidence to separate a real existing
/// daytime nap from a detached night fragment on the Normal path without inventing a heuristic, so
/// this remains deliberately conservative (see Phase B report for what evidence a follow-up would
/// need). NAP: every `ShortNap`-path session, i.e. exactly the Phase-A promoted candidates — those
/// have explicit, already-proven provenance (Phase A's evidence-corroborated shadow evaluation) and
/// so may safely be canonical NAP.
///
/// Returned in the same order as `sessions`.
pub fn assign_session_roles(sessions: &[Session], offset_s: i64) -> Vec<SessionRole> {
    let mut roles = vec![SessionRole::Fragment; sessions.len()];

    let normal_indices: Vec<usize> = sessions
        .iter()
        .enumerate()
        .filter(|(_, s)| s.acceptance_path == SessionAcceptancePath::Normal)
        .map(|(i, _)| i)
        .collect();
    if !normal_indices.is_empty() {
        let blocks: Vec<ScoredNightBlock> = normal_indices
            .iter()
            .map(|&i| {
                let s = &sessions[i];
                ScoredNightBlock {
                    onset: s.start,
                    asleep_s: session_asleep_seconds(s),
                    in_bed_s: (s.end - s.start).max(0) as f64,
                }
            })
            .collect();
        if let Some(group) = main_night_group_indices_scored(&blocks, offset_s, None) {
            // The scorer always picks a winner from whatever it's given (see doc above) — gate on
            // the winning group's own onset actually falling overnight before trusting it as
            // MAIN_SLEEP, using the SAME signal `bridged_night_groups` already reads internally.
            let group_onset = group
                .iter()
                .map(|&local_idx| sessions[normal_indices[local_idx]].start)
                .min();
            let has_overnight_evidence = group_onset
                .map(|onset| detect::is_overnight_onset(onset, offset_s))
                .unwrap_or(false);
            if has_overnight_evidence {
                for local_idx in group {
                    roles[normal_indices[local_idx]] = SessionRole::MainSleep;
                }
            }
        }
    }

    for (i, s) in sessions.iter().enumerate() {
        if s.acceptance_path == SessionAcceptancePath::ShortNap {
            roles[i] = SessionRole::Nap;
        }
    }

    roles
}

/// Sum of a session's asleep stage time (everything but `Wake`) — the same "decoded asleep time"
/// `mainnight`'s scored grouping already expects (see `ScoredNightBlock`'s doc), just read off a
/// `Session`'s own segments rather than re-derived.
fn session_asleep_seconds(s: &Session) -> f64 {
    s.segments
        .iter()
        .filter(|seg| seg.stage != SleepStage::Wake)
        .map(|seg| (seg.end - seg.start).max(0) as f64)
        .sum()
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
