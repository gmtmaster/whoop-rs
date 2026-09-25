//! V2 (cardiorespiratory) sleep stager — the 5.0/MG default. Per-night z-scored HR / HR-variability /
//! motion emissions, a deep gate on the 11-min HR-flatness percentile, a soft sleep-cycle prior, a
//! self-calibrating jerk wake gate, an R-R RSA respiration term, and Viterbi transition smoothing. The
//! per-epoch coefficients are fixed a-priori from sleep physiology + population base rates.
//!
//! Mirrors the shipped app's V2 recipe. Absent signal scores the neutral centre, so a sparse channel never
//! blocks a stage. Outputs are wellness estimates, never medical advice.

use std::collections::HashMap;
use std::f64::consts::PI;

use super::common::{ZScore, flatten_rr, median};
use super::input::{AccelSample, HrSample, SleepInput};
use super::params::Params;
use super::{SleepStage, StageSegment};

/// Consecutive non-wake epochs that mark sleep onset for the onset-anchored priors.
const SUSTAINED_ONSET_EPOCHS: usize = 10;
/// One epoch in minutes, the unit the onset-anchored REM guard grades over.
const EPOCH_MIN: f64 = 0.5;
const LATE_DEEP_INTERACTION_START: f64 = 0.65;
const LATE_DEEP_INTERACTION_SUPPORT: f64 = 0.4;

// Stage indices for the emission/transition arrays: order [deep, rem, light, awake].
const DEEP: usize = 0;
const REM: usize = 1;
const LIGHT: usize = 2;
const AWAKE: usize = 3;

/// Column order of every emission row and of both transition-matrix axes.
pub const STAGE_ORDER: [SleepStage; 4] = [
    SleepStage::Deep,
    SleepStage::Rem,
    SleepStage::Light,
    SleepStage::Wake,
];

/// Code-level identity of this module's shipped emission/decoding recipe, independent of
/// `shadow_metrics::ALGORITHM_VERSION_V4` (which stamps the nightly-physiology/RHR bundle, not
/// staging). Bump only when `Params::SHIPPED`'s staging-relevant fields change; `sleep-staging-v2`
/// was the recipe documented in `docs/algorithms.md` prior to this constant's introduction.
/// `sleep-staging-v3` was the pre-H-ABC final V8 Deep emission. Bumped to `sleep-staging-v4-habc`
/// now that the frozen Architecture H-ABC / ">=2-of-3" candidate replaces the Deep emission slot
/// (see `h_abc_deep_value`) -- the smallest in-crate identity change that lets a newly computed
/// sleep session be told apart from one staged by the prior recipe. This constant alone does not
/// reach `sleep_sessions.algorithm_version`; that column is currently stamped by a process outside
/// this crate (noop-backend's `ReplayArtifactSet`/derived-artifact import path) that does not read
/// this constant today -- closing that gap is a separate, out-of-crate change.
pub const STAGING_ALGORITHM_VERSION: &str = "sleep-staging-v4-habc";

/// Deep-eligibility HR-flatness percentile gate of the shipped recipe.
pub const DEEP_GATE_THRESH: f64 = Params::SHIPPED.deep_gate_thresh;

// === Architecture H-ABC / frozen ">=2-of-3" Deep candidate -- FROZEN research constants ===============
// Recovered verbatim from the validated offline candidate: `noop-backend` repo,
// `research/frozen_holdout_validation_2026-09-22/scripts/frozen_constants.py` (channel formulas + the
// fusion rule) and `.../scripts/build_epoch_table.py` (motion_mod + trailing-window semantics), with the
// decision record at `research/final_deep_production_candidate_2026-09-23/README.md`. Every constant
// below is a reference-population median/IQR or a bisection-solved scale factor fixed BEFORE this
// architecture was scored against any outcome -- do not tune any of them without a new validation cycle.

/// Channel B (HR-stability) trailing-window feature sizes, in epochs (30 s each): 10 min / 2 min.
const HABC_HR600_EPOCHS: usize = 20;
const HABC_HR120_EPOCHS: usize = 4;
const HABC_MOVE600_EPOCHS: usize = 20;

/// Channel B reference medians/IQRs (TRUE_LIGHT-anchored) and its scale factor.
const HABC_V9_HR600_MED: f64 = 2.3339;
const HABC_V9_HR600_IQR: f64 = 3.0434;
const HABC_V9_HR120_MED: f64 = 0.9087;
const HABC_V9_HR120_IQR: f64 = 0.7861;
const HABC_V9_MOVE600_MED: f64 = 0.0086;
const HABC_V9_MOVE600_IQR: f64 = 0.0293;
const HABC_V9_K_SCALE: f64 = 1.7795;

/// Channel C (RR/HRV) trailing-window widths, in whole seconds, and gap-aware RMSSD's max beat gap.
const HABC_RR_SDNN_MAD_WINDOW_SEC: i64 = 600;
const HABC_RR_RMSSD_WINDOW_SEC: i64 = 300;
const HABC_RR_MAX_GAP_SEC: i64 = 5;

/// Channel C reference medians/IQRs (TRUE_LIGHT-anchored) and its scale factor.
const HABC_RR_SDNN10_MED: f64 = 122.848;
const HABC_RR_SDNN10_IQR: f64 = 32.578;
const HABC_RR_MAD10_MED: f64 = 61.0;
const HABC_RR_MAD10_IQR: f64 = 17.25;
const HABC_RR_RMSSD5_MED: f64 = 108.456;
const HABC_RR_RMSSD5_IQR: f64 = 39.713;
const HABC_K_RR: f64 = 0.689;

/// Fusion: Deep is only replaced by the fused value when at least this many of {A, B, C} individually
/// clear their own reference zero (`em_light`). 2 is what makes this the ">=2-of-3" candidate rather than
/// the looser "H-ABC" (`min_agree=1`) variant that was NOT selected for production.
const HABC_MIN_AGREE: usize = 2;

/// Channel E (motion veto/bonus) constants -- unchanged from the frozen offline candidate.
const HABC_IMMOB_MED_MIN: f64 = 4.5;
const HABC_IMMOB_IQR_MIN: f64 = 6.0;
const HABC_MOTION_BONUS_CAP: f64 = 0.15;
const HABC_MOTION_VETO: f64 = -3.0;
const HABC_NEARZERO_MOVE_FRAC: f64 = 0.01;
const HABC_MOVE_VETO_THRESH: f64 = 0.15;

// === Challenger B + boundary/edge confidence backstop ("B+1") -- FROZEN research constants ============
// Recovered verbatim from `noop-backend` repo,
// `docs/habc-residual-false-deep-forensics-2026-09-25.md` (Part 8's frozen pseudocode + Part 9's holdout/
// forward validation) and its sandbox replay `crates/physio-algo/examples/print_bplus.rs`. Applies only
// to "added" epochs (V8-only-decoded label != Deep AND preliminary H-ABC-fused-decoded label == Deep);
// every other epoch is untouched. Do not tune any of these without a new validation cycle.
/// Challenger B: an added epoch is reverted to A (`em_deep_v8`) when it sits more than this many minutes
/// (by nearest V8-only-decoded Deep epoch) from any V8-trusted Deep AND its `second_strongest` is weak.
const HABC_BPLUS_DIST_MIN_MINUTES: f64 = 30.0;
/// Challenger B's own confidence half of the AND above.
const HABC_BPLUS_WEAK_SECOND_STRONGEST: f64 = 0.5;
/// The B+1 boundary backstop: for an added epoch Challenger B did NOT already catch, revert it anyway
/// when it sits at the entry/exit edge of its own contiguous preliminary-fused-Deep run AND
/// `second_strongest` is below this (stricter than Challenger B's own 0.5) threshold.
const HABC_BPLUS_EDGE_SECOND_STRONGEST: f64 = 0.0;

/// One 30 s epoch's recipe features. `None` means "no measurement" (scored neutral).
struct Epoch {
    start: i64,
    hr: Option<f64>,
    hr_var: Option<f64>,
    hr_flat11: Option<f64>,
    move_frac: Option<f64>,
    jerk_max: f64,
    resp_reg: Option<f64>,
    clock: f64,
    jerk_scale: f64,
    /// Frozen H-ABC channel B inputs: trailing (causal, index-windowed) population std of per-epoch mean
    /// HR over the preceding 20 / 4 epochs, and trailing mean `move_frac` over the preceding 20 epochs.
    hr_trail_std_600s: Option<f64>,
    hr_trail_std_120s: Option<f64>,
    move_trail_mean_600s: Option<f64>,
    /// Frozen H-ABC channel C inputs: trailing (causal, time-windowed) RR/HRV statistics ending at this
    /// epoch's END (`start + 30`), computed from the UNCLAMPED flattened RR beats (deliberately not the
    /// [300,2000]ms-clamped beats `resp_regularity` uses -- the frozen research channel never clamped).
    sdnn_10min: Option<f64>,
    rr_mad_10min: Option<f64>,
    rmssd_5min: Option<f64>,
    /// Frozen H-ABC channel E input: minutes of the current unbroken run of near-zero `move_frac`
    /// (`<= HABC_NEARZERO_MOVE_FRAC`) ending at and including this epoch, rounded to 0.1 min exactly as
    /// the frozen offline candidate does.
    low_motion_run_min: Option<f64>,
}

/// Stage a detected in-bed span with the shipped V2 recipe; segments tile `[start, end]`.
pub fn stage(input: &SleepInput) -> Vec<StageSegment> {
    stage_with(input, &Params::SHIPPED)
}

/// One night's epoch features, already extracted. Only `jerk_move_mult` changes these, so a sweep over
/// the emission/prior/transition axes can extract once and re-label many times.
pub struct Prepared {
    feats: Vec<Epoch>,
    start: i64,
    end: i64,
}

/// Extract a night's epoch features under `p`. Pair with [`stage_prepared`].
pub fn prepare(input: &SleepInput, p: &Params) -> Prepared {
    let mut grav = input.accel.clone();
    grav.sort_by_key(|g| g.ts);
    let mut hr = input.hr.clone();
    hr.sort_by_key(|h| h.ts);
    let mut rr = flatten_rr(&input.rr);
    rr.sort_by_key(|a| a.0);
    let feats = features(input.start, input.end, &grav, &hr, &rr, p);
    Prepared {
        feats,
        start: input.start,
        end: input.end,
    }
}

/// Label already-extracted features under `p`; segments tile the prepared span.
pub fn stage_prepared(prep: &Prepared, p: &Params) -> Vec<StageSegment> {
    if prep.feats.is_empty() {
        return vec![StageSegment {
            start: prep.start,
            end: prep.end,
            stage: SleepStage::Light,
        }];
    }
    let labels = stage_epochs(&prep.feats, p);
    segments_from(&prep.feats, &labels, prep.start, prep.end)
}

/// The per-epoch log-emissions a prepared night hands the decoder, in [`STAGE_ORDER`] columns.
/// [`viterbi`] over these reproduces [`stage_prepared`]'s labels exactly, so a caller can score the path
/// search and the emissions feeding it apart.
pub fn emissions_prepared(prep: &Prepared, p: &Params) -> Vec<[f64; 4]> {
    final_emissions(&prep.feats, p)
}

/// The unix second each prepared epoch opens at, index-for-index with [`emissions_prepared`]. An epoch
/// with neither HR nor gravity is dropped, so the sequence can skip and a reference has to be aligned by
/// time rather than by position.
pub fn epoch_starts(prep: &Prepared) -> Vec<i64> {
    prep.feats.iter().map(|f| f.start).collect()
}

/// One epoch's full diagnostic trace: every raw/derived feature, z-score, gate and prior term
/// [`emissions`] computes internally, plus the final per-stage log-emission it hands the decoder.
/// Forensic-only export — mirrors `emissions()`'s arithmetic exactly (same anchor selection via
/// `final_emissions`), so `[em_deep, em_rem, em_light, em_awake]` here equals the corresponding row of
/// [`emissions_prepared`]. Adds no behaviour; changes nothing production reads.
#[derive(Debug, Clone, Copy)]
pub struct EpochDiagnostic {
    pub start: i64,
    /// Raw per-epoch mean HR (bpm), and the windowed HR std/HR-flatness features feeding the deep gate.
    pub hr: Option<f64>,
    pub hr_var: Option<f64>,
    pub hr_flat11: Option<f64>,
    /// This epoch's HR-flatness percentile rank within the night (0..1) — what `deep_gate_slope` reads.
    pub hr_flat11_pct: f64,
    pub move_frac: Option<f64>,
    pub jerk_max: f64,
    pub jerk_scale: f64,
    pub resp_reg: Option<f64>,
    /// Fraction of the WINDOW (or, under the onset anchor, of the sleep period) this epoch sits at.
    pub clock: f64,
    /// Per-night z-scores of HR, HR-variability, motion fraction and RSA regularity.
    pub z_hr: f64,
    pub z_hr_var: f64,
    pub z_move: f64,
    pub z_resp_reg: f64,
    /// The deep-eligibility HR-flatness gate penalty (subtracted from the deep emission).
    pub deep_gate: f64,
    /// The sleep-cycle prior's deep/REM terms this epoch received (already net of the REM guard).
    pub cycle_prior_deep: f64,
    pub cycle_prior_rem: f64,
    /// The early-REM suppression subtracted from `cycle_prior_rem` above, reported separately so a
    /// caller can see the guard's own size.
    pub rem_guard: f64,
    /// True when this epoch's peak jerk cleared the motion gate, adding `motion_gate_boost` to awake.
    pub motion_gate_boost_applied: bool,
    /// The late-deep-interaction bonus folded into `em_deep_v8` when RSA + motion + HRV all agree (0
    /// else). Named `_v8` here (not `late_deep_bonus` alone) only to sit next to the channel it belongs
    /// to; nothing about its own computation changed.
    pub late_deep_bonus: f64,
    /// Frozen H-ABC channel B/C trailing inputs, mirrored 1:1 from `Epoch` — see its doc comments.
    pub hr_trail_std_600s: Option<f64>,
    pub hr_trail_std_120s: Option<f64>,
    pub move_trail_mean_600s: Option<f64>,
    pub sdnn_10min: Option<f64>,
    pub rr_mad_10min: Option<f64>,
    pub rmssd_5min: Option<f64>,
    pub low_motion_run_min: Option<f64>,
    /// A: the complete, unmodified legacy V8 Deep emission (every existing term through
    /// `late_deep_bonus`) -- what `em_deep` used to be, and would still be under `min_agree` architectures
    /// this candidate does not use.
    pub em_deep_v8: f64,
    /// B: frozen HR_STABILITY_DEEP channel value.
    pub em_deep_v9: f64,
    /// C: frozen RR_HRV_DEEP channel value.
    pub em_deep_rr: f64,
    /// E: the frozen motion veto/bonus actually applied inside the fusion (0 unless the fusion's
    /// `>=2-of-3` gate fired; the raw veto/bonus value is reported regardless of whether it was used, so
    /// a caller can see it either way).
    pub habc_motion_mod: f64,
    /// The frozen ">=2-of-3" fusion's decision: how many of {A, B, C} individually cleared `em_light`.
    pub habc_agree_count: usize,
    /// True when the frozen "B+1" gate (Challenger B's distance/confidence test, or its boundary-edge
    /// backstop) reverted this epoch's Deep emission back to `em_deep_v8` after the preliminary H-ABC
    /// fusion -- forensic only, never true for an epoch that was not "added" (V8-only decode already
    /// Deep, or the preliminary H-ABC decode was not Deep here).
    pub habc_bplus_reverted: bool,
    /// Final per-stage log-emissions handed to the decoder — identical to `emissions_prepared`'s row.
    /// `em_deep` is the FUSED H-ABC value (`em_deep_v8` when fewer than `HABC_MIN_AGREE` channels agree,
    /// otherwise `max(qualifying channels) + habc_motion_mod`), further reverted to `em_deep_v8` when the
    /// frozen "B+1" gate above fired -- see `em_deep_v8` above for A's own always-unmodified value.
    pub em_deep: f64,
    pub em_rem: f64,
    pub em_light: f64,
    pub em_awake: f64,
}

/// Every intermediate term [`emissions_prepared`] computes but does not expose, one row per prepared
/// epoch, under the same onset-anchor selection `final_emissions` uses. Forensic replay only.
pub fn diagnostics_prepared(prep: &Prepared, p: &Params) -> Vec<EpochDiagnostic> {
    if prep.feats.is_empty() {
        return Vec::new();
    }
    let anchor = if p.cycle_rem_onset_minutes > 0.0 || p.cycle_clock_from_onset {
        let probe = viterbi(&emissions(&prep.feats, p, Anchor::Probe), &p.transition);
        Anchor::Onset(sustained_onset(&probe).unwrap_or(0))
    } else {
        Anchor::Window
    };
    diagnostics(&prep.feats, p, anchor)
}

/// The diagnostic trace under an explicit anchor — [`diagnostics_prepared`]'s inner loop, duplicated from
/// `emissions()` term-for-term so a forensic row and the real emission it explains can never drift apart
/// silently: `tests::diagnostics_matches_emissions` pins the two identical.
fn diagnostics(feats: &[Epoch], p: &Params, anchor: Anchor) -> Vec<EpochDiagnostic> {
    let blp = p.base_log_prior();
    let zhr = ZScore::build(&feats.iter().map(|f| f.hr).collect::<Vec<_>>());
    let zhv = ZScore::build(&feats.iter().map(|f| f.hr_var).collect::<Vec<_>>());
    let zmv = ZScore::build(&feats.iter().map(|f| f.move_frac).collect::<Vec<_>>());
    let zrg = ZScore::build(&feats.iter().map(|f| f.resp_reg).collect::<Vec<_>>());

    let mut fsorted: Vec<f64> = feats.iter().filter_map(|f| f.hr_flat11).collect();
    fsorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let fpct = |value: Option<f64>| -> f64 {
        match value {
            Some(v) if !fsorted.is_empty() => {
                let mut lo = 0usize;
                let mut hi = fsorted.len();
                while lo < hi {
                    let mid = (lo + hi) / 2;
                    if fsorted[mid] <= v {
                        lo = mid + 1;
                    } else {
                        hi = mid;
                    }
                }
                lo as f64 / fsorted.len() as f64
            }
            _ => 0.5,
        }
    };

    let mut out = Vec::with_capacity(feats.len());
    for (i, f) in feats.iter().enumerate() {
        let zhrv = zhr.apply(f.hr);
        let zhvv = zhv.apply(f.hr_var);
        let zmvv = zmv.apply(f.move_frac);
        let hr_flat11_pct = fpct(f.hr_flat11);
        let gate = p.deep_gate_slope * (hr_flat11_pct - p.deep_gate_thresh).max(0.0);
        let awake_cardiac0 =
            p.awake_hrv * dz(zhvv, p.awake_deadzone) + p.awake_hr * dz(zhrv, p.awake_deadzone);
        let awake_cardiac = if motion_quiescent(f, p) {
            awake_cardiac0.min(0.0)
        } else {
            awake_cardiac0
        };

        let mut em = [0.0f64; 4];
        em[DEEP] = p.deep_hrv * zhvv + p.deep_hr * zhrv + p.deep_motion * zmvv - gate + blp[DEEP];
        em[REM] = p.rem_hrv * zhvv + p.rem_motion * zmvv + p.rem_hr * zhrv + blp[REM];
        em[LIGHT] = blp[LIGHT];
        em[AWAKE] = p.awake_motion * zmvv + awake_cardiac + blp[AWAKE];

        let clock = cycle_clock(f.clock, feats, anchor, p);
        let guard = rem_guard(i, f.clock, anchor, p);
        let pr = cycle_prior(clock, guard, p);
        for (s, pv) in pr.iter().enumerate() {
            em[s] += pv;
        }
        let motion_gate_boost_applied = f.jerk_max > f.jerk_scale * p.jerk_gate_mult;
        if motion_gate_boost_applied {
            em[AWAKE] += p.motion_gate_boost;
        }
        let mut late_deep_bonus = 0.0;
        if let Some(rg) = f.resp_reg {
            let z = zrg.apply(Some(rg));
            em[DEEP] += p.resp_weight * z;
            em[REM] -= p.resp_weight * z;
            late_deep_bonus = late_deep_interaction(
                f.clock,
                zhrv,
                zhvv,
                f.move_frac.map(|_| zmvv),
                Some(z),
                p.deep_hr,
            );
            em[DEEP] += late_deep_bonus;
        }

        // === Architecture H-ABC / frozen ">=2-of-3" Deep candidate ===================================
        // A is em[DEEP] as computed above, complete and untouched by anything below. See the frozen
        // constants block near the top of this file for provenance; do not tune anything here.
        let habc_a = em[DEEP];
        let habc_b = em_deep_v9_of(f, blp[DEEP]);
        let habc_c = em_deep_rr_of(f, blp[DEEP]);
        let habc_mm = habc_motion_mod(f.move_frac, f.low_motion_run_min);
        let habc_agree_count = habc_agreeing(habc_a, habc_b, habc_c, em[LIGHT]);
        em[DEEP] = h_abc_deep_value(habc_a, habc_b, habc_c, em[LIGHT], habc_mm);

        out.push(EpochDiagnostic {
            start: f.start,
            hr: f.hr,
            hr_var: f.hr_var,
            hr_flat11: f.hr_flat11,
            hr_flat11_pct,
            move_frac: f.move_frac,
            jerk_max: f.jerk_max,
            jerk_scale: f.jerk_scale,
            resp_reg: f.resp_reg,
            clock: f.clock,
            z_hr: zhrv,
            z_hr_var: zhvv,
            z_move: zmvv,
            z_resp_reg: f.resp_reg.map(|rg| zrg.apply(Some(rg))).unwrap_or(0.0),
            deep_gate: gate,
            cycle_prior_deep: pr[DEEP],
            cycle_prior_rem: pr[REM],
            rem_guard: guard,
            motion_gate_boost_applied,
            late_deep_bonus,
            hr_trail_std_600s: f.hr_trail_std_600s,
            hr_trail_std_120s: f.hr_trail_std_120s,
            move_trail_mean_600s: f.move_trail_mean_600s,
            sdnn_10min: f.sdnn_10min,
            rr_mad_10min: f.rr_mad_10min,
            rmssd_5min: f.rmssd_5min,
            low_motion_run_min: f.low_motion_run_min,
            em_deep_v8: habc_a,
            em_deep_v9: habc_b,
            em_deep_rr: habc_c,
            habc_motion_mod: habc_mm,
            habc_agree_count,
            habc_bplus_reverted: false,
            em_deep: em[DEEP],
            em_rem: em[REM],
            em_light: em[LIGHT],
            em_awake: em[AWAKE],
        });
    }

    // === Challenger B + boundary/edge confidence backstop ("B+1") ====================================
    // Applied once over the whole night, after every epoch's preliminary H-ABC-fused emission is known --
    // see `bplus_gate`'s own doc comment for the two-pass structure. `out[i].em_deep` above is still the
    // PRELIMINARY (pre-B+1) fused value at this point; this overwrites it with the final, gated one, so
    // it matches exactly what `emissions_prepared`/`final_emissions` hands the decoder.
    let starts: Vec<i64> = feats.iter().map(|f| f.start).collect();
    let em_v8: Vec<[f64; 4]> = out
        .iter()
        .map(|d| [d.em_deep_v8, d.em_rem, d.em_light, d.em_awake])
        .collect();
    let em_fused: Vec<[f64; 4]> = out
        .iter()
        .map(|d| [d.em_deep, d.em_rem, d.em_light, d.em_awake])
        .collect();
    let habc_b: Vec<f64> = out.iter().map(|d| d.em_deep_v9).collect();
    let habc_c: Vec<f64> = out.iter().map(|d| d.em_deep_rr).collect();
    let (gated, reverted) = bplus_gate(&starts, &em_v8, &em_fused, &habc_b, &habc_c, &p.transition);
    for ((d, g), r) in out.iter_mut().zip(gated.iter()).zip(reverted.iter()) {
        d.em_deep = g[DEEP];
        d.habc_bplus_reverted = *r;
    }

    out
}

/// Tile the prepared span from one label per epoch — staging's last step, so a caller that decoded its own
/// path gets the segments [`stage_prepared`] would have returned. `labels` must be one per prepared epoch.
pub fn segments_of(prep: &Prepared, labels: &[SleepStage]) -> Vec<StageSegment> {
    if prep.feats.is_empty() {
        return vec![StageSegment {
            start: prep.start,
            end: prep.end,
            stage: SleepStage::Light,
        }];
    }
    segments_from(&prep.feats, labels, prep.start, prep.end)
}

/// Stage with an explicit recipe. The tuning path; `stage` is what everything else calls.
pub fn stage_with(input: &SleepInput, p: &Params) -> Vec<StageSegment> {
    let start = input.start;
    let end = input.end;

    let mut grav = input.accel.clone();
    grav.sort_by_key(|g| g.ts);
    let mut hr = input.hr.clone();
    hr.sort_by_key(|h| h.ts);
    let mut rr = flatten_rr(&input.rr);
    rr.sort_by_key(|a| a.0);

    let feats = features(start, end, &grav, &hr, &rr, p);
    if feats.is_empty() {
        return vec![StageSegment {
            start,
            end,
            stage: SleepStage::Light,
        }];
    }
    let labels = stage_epochs(&feats, p);
    segments_from(&feats, &labels, start, end)
}

/// Tile `[start, end]` from one label per epoch, merging equal-stage neighbours.
fn segments_from(
    feats: &[Epoch],
    labels: &[SleepStage],
    start: i64,
    end: i64,
) -> Vec<StageSegment> {
    let mut segments: Vec<StageSegment> = Vec::new();
    let n = feats.len();
    for (i, f) in feats.iter().enumerate() {
        let stage = labels[i];
        let seg_start = if i == 0 { start } else { f.start };
        let seg_end = if i == n - 1 { end } else { feats[i + 1].start };
        match segments.last_mut() {
            Some(last) if last.stage == stage => last.end = seg_end,
            _ => segments.push(StageSegment {
                start: seg_start,
                end: seg_end,
                stage,
            }),
        }
    }
    segments
}

fn features(
    start: i64,
    end: i64,
    grav: &[AccelSample],
    hr: &[HrSample],
    rr: &[(i64, f64)],
    p: &Params,
) -> Vec<Epoch> {
    if end <= start {
        return Vec::new();
    }
    let span = (end - start).max(1) as f64;

    // Per-second HR mean.
    let mut hr_sum: HashMap<i64, f64> = HashMap::new();
    let mut hr_cnt: HashMap<i64, i64> = HashMap::new();
    for s in hr {
        *hr_sum.entry(s.ts).or_insert(0.0) += s.bpm as f64;
        *hr_cnt.entry(s.ts).or_insert(0) += 1;
    }
    let mut sec_hr: HashMap<i64, f64> = HashMap::with_capacity(hr_sum.len());
    for (k, v) in &hr_sum {
        sec_hr.insert(*k, v / hr_cnt[k] as f64);
    }

    // Per-second gravity mean (x, y, z).
    let mut gx: HashMap<i64, f64> = HashMap::new();
    let mut gy: HashMap<i64, f64> = HashMap::new();
    let mut gz: HashMap<i64, f64> = HashMap::new();
    let mut gc: HashMap<i64, i64> = HashMap::new();
    for g in grav {
        *gx.entry(g.ts).or_insert(0.0) += g.x;
        *gy.entry(g.ts).or_insert(0.0) += g.y;
        *gz.entry(g.ts).or_insert(0.0) += g.z;
        *gc.entry(g.ts).or_insert(0) += 1;
    }
    let mut sec_g: HashMap<i64, (f64, f64, f64)> = HashMap::with_capacity(gc.len());
    for (k, c) in &gc {
        let d = *c as f64;
        sec_g.insert(*k, (gx[k] / d, gy[k] / d, gz[k] / d));
    }

    // R-R bucketed by second (for the RSA window).
    let mut rr_by: HashMap<i64, Vec<f64>> = HashMap::new();
    for (ts, ms) in rr {
        rr_by.entry(*ts).or_default().push(*ms);
    }

    // Prefix sums over the integer-second HR axis → O(1) windowed population std.
    let (axis_lo, sum_px, sum_sq_px, cnt_px) = if sec_hr.is_empty() {
        (0i64, vec![0.0], vec![0.0], vec![0i64])
    } else {
        let lo = *sec_hr.keys().min().unwrap();
        let hi = *sec_hr.keys().max().unwrap();
        let len = (hi - lo + 1) as usize;
        let mut sp = vec![0.0; len + 1];
        let mut sqp = vec![0.0; len + 1];
        let mut cp = vec![0i64; len + 1];
        for i in 0..len {
            let v = sec_hr.get(&(lo + i as i64)).copied();
            sp[i + 1] = sp[i] + v.unwrap_or(0.0);
            sqp[i + 1] = sqp[i] + v.map(|x| x * x).unwrap_or(0.0);
            cp[i + 1] = cp[i] + if v.is_some() { 1 } else { 0 };
        }
        (lo, sp, sqp, cp)
    };
    let axis_hi_excl = axis_lo + (cnt_px.len() as i64 - 1);

    let std_of_seconds = |lo: i64, hi: i64| -> Option<f64> {
        if cnt_px.len() <= 1 {
            return None;
        }
        let q_lo = lo.clamp(axis_lo, axis_hi_excl);
        let q_hi = hi.clamp(axis_lo, axis_hi_excl);
        if q_hi <= q_lo {
            return None;
        }
        let a = (q_lo - axis_lo) as usize;
        let b = (q_hi - axis_lo) as usize;
        let n = cnt_px[b] - cnt_px[a];
        if n < 2 {
            return None;
        }
        let sum = sum_px[b] - sum_px[a];
        let sum_sq = sum_sq_px[b] - sum_sq_px[a];
        let mean = sum / n as f64;
        let variance = sum_sq / n as f64 - mean * mean;
        Some(if variance < 0.0 { 0.0 } else { variance }.sqrt())
    };

    // PASS 1 — per-epoch quantities except moveFrac; pool every per-second jerk.
    struct Raw {
        start: i64,
        hr: Option<f64>,
        hr_var: Option<f64>,
        hr_flat11: Option<f64>,
        jerks: Vec<f64>,
        gap_sec: i64,
        jerk_max: f64,
        resp_reg: Option<f64>,
        clock: f64,
    }
    let mut raws: Vec<Raw> = Vec::new();
    let mut all_jerks: Vec<f64> = Vec::new();
    let first_e = ((start + 29) / 30) * 30;
    let mut e = first_e;
    while e < end {
        let mut hrs: Vec<f64> = Vec::new();
        let mut gseq: Vec<(f64, f64, f64)> = Vec::new();
        let mut s = e;
        while s < e + 30 {
            if let Some(v) = sec_hr.get(&s) {
                hrs.push(*v);
            }
            if let Some(v) = sec_g.get(&s) {
                gseq.push(*v);
            }
            s += 1;
        }
        if hrs.is_empty() && gseq.is_empty() {
            e += 30;
            continue;
        }

        let mut jerks: Vec<f64> = Vec::new();
        let mut i = 1usize;
        while i < gseq.len().max(1) {
            let a = gseq[i - 1];
            let b = gseq[i];
            let dx = a.0 - b.0;
            let dy = a.1 - b.1;
            let dz = a.2 - b.2;
            jerks.push((dx * dx + dy * dy + dz * dz).sqrt());
            i += 1;
        }
        all_jerks.extend_from_slice(&jerks);
        let jerk_max = jerks.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let jerk_max = if jerks.is_empty() { 0.0 } else { jerk_max };

        let hr_mean = if hrs.is_empty() {
            None
        } else {
            Some(hrs.iter().sum::<f64>() / hrs.len() as f64)
        };
        let hr_var = std_of_seconds(e - 150, e + 30 + 150);
        let hr_flat11 = std_of_seconds(e - 330, e + 30 + 360);

        let mut beats: Vec<(f64, f64)> = Vec::new();
        let mut bs = e - 90;
        while bs < e + 120 {
            if let Some(vs) = rr_by.get(&bs) {
                for v in vs {
                    beats.push((bs as f64, v.clamp(300.0, 2000.0)));
                }
            }
            bs += 1;
        }
        beats.sort_by(|a, b| {
            a.0.partial_cmp(&b.0)
                .unwrap()
                .then(a.1.partial_cmp(&b.1).unwrap())
        });
        let resp_reg = resp_regularity(&beats);

        raws.push(Raw {
            start: e,
            hr: hr_mean,
            hr_var,
            hr_flat11,
            gap_sec: (gseq.len() as i64 - 1).max(1),
            jerk_max,
            resp_reg,
            clock: (e + 15 - start) as f64 / span,
            jerks,
        });
        e += 30;
    }

    let jerk_scale = if all_jerks.is_empty() {
        1e-6
    } else {
        median(&all_jerks)
    };
    let move_thr = jerk_scale * p.jerk_move_mult;

    // PASS 2 — move fraction against the night-relative threshold.
    let mut epochs: Vec<Epoch> = raws
        .into_iter()
        .map(|r| {
            // No gravity in the epoch = no motion evidence; absent, not "perfectly still".
            let observed = !r.jerks.is_empty();
            let moves = r.jerks.iter().filter(|&&j| j > move_thr).count();
            Epoch {
                start: r.start,
                hr: r.hr,
                hr_var: r.hr_var,
                hr_flat11: r.hr_flat11,
                move_frac: observed.then(|| moves as f64 / r.gap_sec as f64),
                jerk_max: r.jerk_max,
                resp_reg: r.resp_reg,
                clock: r.clock,
                jerk_scale,
                hr_trail_std_600s: None,
                hr_trail_std_120s: None,
                move_trail_mean_600s: None,
                sdnn_10min: None,
                rr_mad_10min: None,
                rmssd_5min: None,
                low_motion_run_min: None,
            }
        })
        .collect();

    // PASS 3 — frozen Architecture H-ABC trailing features (causal; see the frozen-constants block near
    // the top of this file). Index-windowed over this already-gap-dropped epoch sequence for B, and
    // time-windowed over the raw unclamped `rr` slice (ending at each epoch's own end, `start + 30`) for
    // C. `low_motion_run_min` is a forward-running streak, chronological in this same epoch-index order.
    let hr_seq: Vec<Option<f64>> = epochs.iter().map(|ep| ep.hr).collect();
    let move_seq: Vec<Option<f64>> = epochs.iter().map(|ep| ep.move_frac).collect();
    let mut low_run_epochs: i64 = 0;
    for (i, ep) in epochs.iter_mut().enumerate() {
        ep.hr_trail_std_600s = trailing_pstdev(&hr_seq, i, HABC_HR600_EPOCHS);
        ep.hr_trail_std_120s = trailing_pstdev(&hr_seq, i, HABC_HR120_EPOCHS);
        ep.move_trail_mean_600s = trailing_mean(&move_seq, i, HABC_MOVE600_EPOCHS);

        let near_zero = ep.move_frac.is_some_and(|m| m <= HABC_NEARZERO_MOVE_FRAC);
        low_run_epochs = if near_zero { low_run_epochs + 1 } else { 0 };
        ep.low_motion_run_min = Some(((low_run_epochs as f64 * 0.5) * 10.0).round() / 10.0);

        let e_end = ep.start + 30;
        let w10 = rr_window_stats(
            rr,
            e_end - HABC_RR_SDNN_MAD_WINDOW_SEC,
            e_end,
            HABC_RR_MAX_GAP_SEC,
        );
        let w5 = rr_window_stats(
            rr,
            e_end - HABC_RR_RMSSD_WINDOW_SEC,
            e_end,
            HABC_RR_MAX_GAP_SEC,
        );
        ep.sdnn_10min = w10.sdnn;
        ep.rr_mad_10min = w10.rr_mad;
        ep.rmssd_5min = w5.rmssd;
    }

    epochs
}

/// RSA respiration regularity: tachogram → 4 Hz resample → detrend → band-limited DFT peak/sum over the
/// 0.15–0.40 Hz band. Higher = more regular breathing. `None` when there are too few beats.
fn resp_regularity(beats: &[(f64, f64)]) -> Option<f64> {
    if beats.len() < 12 {
        return None;
    }
    let t0 = beats[0].0;
    let t_n = beats[beats.len() - 1].0;
    if t_n <= t0 {
        return None;
    }
    let n = ((t_n - t0) / 0.25 - 1e-9).ceil() as i64;
    if n < 16 {
        return None;
    }
    let n = n as usize;

    let mut y = vec![0.0f64; n];
    let mut seg = 0usize;
    for (i, yi) in y.iter_mut().enumerate() {
        let t = t0 + 0.25 * i as f64;
        while seg < beats.len() - 2 && beats[seg + 1].0 < t {
            seg += 1;
        }
        let ta = beats[seg].0;
        let tb = beats[seg + 1].0;
        let va = beats[seg].1;
        let vb = beats[seg + 1].1;
        *yi = if tb <= ta {
            va
        } else {
            va + ((t - ta) / (tb - ta)).clamp(0.0, 1.0) * (vb - va)
        };
    }
    let mean = y.iter().sum::<f64>() / n as f64;
    for v in y.iter_mut() {
        *v -= mean;
    }

    let k_lo = (0.15 * 0.25 * n as f64).ceil() as i64;
    let k_hi = (0.40 * 0.25 * n as f64).floor() as i64;
    if k_hi < k_lo || k_lo < 0 {
        return None;
    }
    let mut max_p = 0.0;
    let mut sum_p = 0.0;
    for k in k_lo..=k_hi {
        let mut re = 0.0;
        let mut im = 0.0;
        let w = -2.0 * PI * k as f64 / n as f64;
        for (j, &yj) in y.iter().enumerate() {
            let a = w * j as f64;
            re += yj * a.cos();
            im += yj * a.sin();
        }
        let p = re * re + im * im;
        sum_p += p;
        if p > max_p {
            max_p = p;
        }
    }
    if sum_p == 0.0 {
        None
    } else {
        Some(max_p / sum_p)
    }
}

/// Where the time-of-night priors are measured from. `Probe` is the first pass of a two-pass staging: it
/// has no labels yet, so the early-REM guard is off and cannot bias the onset it is looking for.
#[derive(Clone, Copy)]
enum Anchor {
    Probe,
    /// Read from the window, which is what a one-pass staging has.
    Window,
    /// Read from the epoch sleep began at.
    Onset(usize),
}

/// Soft sleep-cycle prior: deep concentrated early (decays), REM raised through the night less `guard`.
fn cycle_prior(c: f64, guard: f64, p: &Params) -> [f64; 4] {
    let mut pr = [0.0; 4];
    pr[DEEP] = p.cycle_deep_scale * (1.0 - c / p.cycle_deep_decay).max(0.0);
    pr[REM] = p.cycle_rem_scale * c.min(p.cycle_rem_ramp_cap) - guard;
    pr
}

/// Time-of-night for the cycle prior. `clock` is a fraction of the WINDOW, so rebasing the pair against
/// the onset epoch's own value re-expresses it as a fraction of the sleep period in the same units.
fn cycle_clock(c: f64, feats: &[Epoch], anchor: Anchor, p: &Params) -> f64 {
    let Anchor::Onset(o) = anchor else { return c };
    if !p.cycle_clock_from_onset {
        return c;
    }
    let c0 = feats[o].clock;
    if c0 >= 1.0 {
        c
    } else {
        ((c - c0) / (1.0 - c0)).clamp(0.0, 1.0)
    }
}

/// The early-REM suppression for one epoch. Zero `cycle_rem_onset_minutes` keeps the step at a fraction
/// of the session; a positive one grades the same magnitude to zero over that many minutes past onset,
/// clamped so a pre-onset epoch is never penalised harder than onset itself.
fn rem_guard(idx: usize, c: f64, anchor: Anchor, p: &Params) -> f64 {
    match anchor {
        Anchor::Probe => 0.0,
        Anchor::Onset(o) if p.cycle_rem_onset_minutes > 0.0 => {
            let minutes = (idx as f64 - o as f64) * EPOCH_MIN;
            p.cycle_rem_early_penalty * (1.0 - minutes / p.cycle_rem_onset_minutes).clamp(0.0, 1.0)
        }
        _ if c < p.cycle_rem_early_frac => p.cycle_rem_early_penalty,
        _ => 0.0,
    }
}

/// First epoch of the earliest run of at least `SUSTAINED_ONSET_EPOCHS` consecutive non-wake labels.
fn sustained_onset(labels: &[SleepStage]) -> Option<usize> {
    let mut run = 0usize;
    for (i, l) in labels.iter().enumerate() {
        run = if *l == SleepStage::Wake { 0 } else { run + 1 };
        if run == SUSTAINED_ONSET_EPOCHS {
            return Some(i + 1 - SUSTAINED_ONSET_EPOCHS);
        }
    }
    None
}

/// Motion-quiescent: movement was OBSERVED and was none, with peak jerk at/below the night floor × the
/// gate multiplier. An epoch with no gravity cannot be quiescent — absence is not stillness.
fn motion_quiescent(f: &Epoch, p: &Params) -> bool {
    f.move_frac.is_some_and(|m| m <= 0.0) && f.jerk_max <= f.jerk_scale * p.jerk_gate_mult
}

fn dz(z: f64, deadzone: f64) -> f64 {
    if deadzone <= 0.0 {
        z
    } else if z > deadzone {
        z - deadzone
    } else if z < -deadzone {
        z + deadzone
    } else {
        0.0
    }
}

/// Viterbi most-likely path over the per-epoch log-emissions with the sticky transition matrix and a
/// uniform start. Columns are [`STAGE_ORDER`]; ties resolve to the earlier stage. A zero transition is
/// floored at 1e-9 rather than forbidden, so no path is unreachable.
#[allow(clippy::needless_range_loop)]
pub fn viterbi(em_seq: &[[f64; 4]], transition: &[[f64; 4]; 4]) -> Vec<SleepStage> {
    if em_seq.is_empty() {
        return Vec::new();
    }
    let mut log_t = [[0.0f64; 4]; 4];
    for (fi, row) in transition.iter().enumerate() {
        for (ti, &v) in row.iter().enumerate() {
            log_t[fi][ti] = v.max(1e-9).ln();
        }
    }
    let mut v = em_seq[0];
    let mut back: Vec<[usize; 4]> = Vec::new();
    for em in &em_seq[1..] {
        let mut new_v = [0.0f64; 4];
        let mut bp = [0usize; 4];
        for s in 0..4 {
            let mut best_prev = 0usize;
            let mut best_val = v[0] + log_t[0][s];
            for p in 1..4 {
                let value = v[p] + log_t[p][s];
                if value > best_val {
                    best_val = value;
                    best_prev = p;
                }
            }
            new_v[s] = best_val + em[s];
            bp[s] = best_prev;
        }
        v = new_v;
        back.push(bp);
    }
    let mut last = 0usize;
    let mut last_v = v[0];
    for s in 1..4 {
        if v[s] > last_v {
            last_v = v[s];
            last = s;
        }
    }
    let mut path = vec![last];
    for bp in back.iter().rev() {
        last = bp[last];
        path.push(last);
    }
    path.reverse();
    path.into_iter().map(idx_to_stage).collect()
}

fn idx_to_stage(i: usize) -> SleepStage {
    match i {
        DEEP => SleepStage::Deep,
        REM => SleepStage::Rem,
        LIGHT => SleepStage::Light,
        _ => SleepStage::Wake,
    }
}

fn late_deep_interaction(
    clock: f64,
    zhr: f64,
    zhrv: f64,
    zmotion: Option<f64>,
    zrsa: Option<f64>,
    deep_hr_weight: f64,
) -> f64 {
    if clock < LATE_DEEP_INTERACTION_START
        || zhrv >= 0.0
        || !zmotion.is_some_and(|z| z <= 0.0)
        || !zrsa.is_some_and(|z| z > 0.0)
    {
        return 0.0;
    }
    -deep_hr_weight * zhr.min(0.0) + LATE_DEEP_INTERACTION_SUPPORT
}

/// The log-emissions the decoder is handed, under whichever anchor `p` selects. An onset-anchored prior
/// needs a staging to find the onset, so it stages once with the guard off first.
fn final_emissions(feats: &[Epoch], p: &Params) -> Vec<[f64; 4]> {
    let anchor = if p.cycle_rem_onset_minutes > 0.0 || p.cycle_clock_from_onset {
        let probe = viterbi(&emissions(feats, p, Anchor::Probe), &p.transition);
        Anchor::Onset(sustained_onset(&probe).unwrap_or(0))
    } else {
        Anchor::Window
    };
    // === Challenger B + boundary/edge confidence backstop ("B+1") ================================
    // The anchor (above) is resolved exactly as before -- B+1 never touches onset detection. Only the
    // REAL pass (under the resolved anchor) is gated; see `bplus_gate`'s own doc comment.
    let em_fused = emissions(feats, p, anchor);
    let (em_v8, habc_b, habc_c) = emissions_v8_and_habc(feats, p, anchor);
    let starts: Vec<i64> = feats.iter().map(|f| f.start).collect();
    let (gated, _reverted) = bplus_gate(&starts, &em_v8, &em_fused, &habc_b, &habc_c, &p.transition);
    gated
}

/// Per-epoch V8-only emissions (A alone, before the H-ABC fusion) alongside the H-ABC channel B/C raw
/// values -- duplicated term-for-term from `emissions()`'s own identical pre-fusion block (the same
/// duplication discipline `diagnostics()` already uses, pinned by `bplus_v8_matches_emissions`), so the
/// B+1 gate can see the pre-fusion state it needs without re-deriving it a third, possibly-drifting way.
fn emissions_v8_and_habc(feats: &[Epoch], p: &Params, anchor: Anchor) -> (Vec<[f64; 4]>, Vec<f64>, Vec<f64>) {
    let blp = p.base_log_prior();
    let zhr = ZScore::build(&feats.iter().map(|f| f.hr).collect::<Vec<_>>());
    let zhv = ZScore::build(&feats.iter().map(|f| f.hr_var).collect::<Vec<_>>());
    let zmv = ZScore::build(&feats.iter().map(|f| f.move_frac).collect::<Vec<_>>());
    let zrg = ZScore::build(&feats.iter().map(|f| f.resp_reg).collect::<Vec<_>>());

    let mut fsorted: Vec<f64> = feats.iter().filter_map(|f| f.hr_flat11).collect();
    fsorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let fpct = |value: Option<f64>| -> f64 {
        match value {
            Some(v) if !fsorted.is_empty() => {
                let mut lo = 0usize;
                let mut hi = fsorted.len();
                while lo < hi {
                    let mid = (lo + hi) / 2;
                    if fsorted[mid] <= v {
                        lo = mid + 1;
                    } else {
                        hi = mid;
                    }
                }
                lo as f64 / fsorted.len() as f64
            }
            _ => 0.5,
        }
    };

    let mut em_v8 = Vec::with_capacity(feats.len());
    let mut habc_b = Vec::with_capacity(feats.len());
    let mut habc_c = Vec::with_capacity(feats.len());
    for (i, f) in feats.iter().enumerate() {
        let zhrv = zhr.apply(f.hr);
        let zhvv = zhv.apply(f.hr_var);
        let zmvv = zmv.apply(f.move_frac);
        let gate = p.deep_gate_slope * (fpct(f.hr_flat11) - p.deep_gate_thresh).max(0.0);
        let awake_cardiac0 =
            p.awake_hrv * dz(zhvv, p.awake_deadzone) + p.awake_hr * dz(zhrv, p.awake_deadzone);
        let awake_cardiac = if motion_quiescent(f, p) {
            awake_cardiac0.min(0.0)
        } else {
            awake_cardiac0
        };

        let mut em = [0.0f64; 4];
        em[DEEP] = p.deep_hrv * zhvv + p.deep_hr * zhrv + p.deep_motion * zmvv - gate + blp[DEEP];
        em[REM] = p.rem_hrv * zhvv + p.rem_motion * zmvv + p.rem_hr * zhrv + blp[REM];
        em[LIGHT] = blp[LIGHT];
        em[AWAKE] = p.awake_motion * zmvv + awake_cardiac + blp[AWAKE];

        let pr = cycle_prior(
            cycle_clock(f.clock, feats, anchor, p),
            rem_guard(i, f.clock, anchor, p),
            p,
        );
        for (s, pv) in pr.iter().enumerate() {
            em[s] += pv;
        }
        if f.jerk_max > f.jerk_scale * p.jerk_gate_mult {
            em[AWAKE] += p.motion_gate_boost;
        }
        if let Some(rg) = f.resp_reg {
            let z = zrg.apply(Some(rg));
            em[DEEP] += p.resp_weight * z;
            em[REM] -= p.resp_weight * z;
            em[DEEP] += late_deep_interaction(
                f.clock,
                zhrv,
                zhvv,
                f.move_frac.map(|_| zmvv),
                Some(z),
                p.deep_hr,
            );
        }
        em_v8.push(em);
        habc_b.push(em_deep_v9_of(f, blp[DEEP]));
        habc_c.push(em_deep_rr_of(f, blp[DEEP]));
    }
    (em_v8, habc_b, habc_c)
}

/// Distance in minutes from epoch `i` to the nearest `Deep`-labeled epoch in `labels`, via one forward
/// and one backward sweep over the already time-sorted epoch starts -- O(n), no pairwise search.
/// `f64::INFINITY` when `labels` contains no `Deep` epoch at all, matching the frozen research replay's
/// own convention (`print_bplus.rs::dist_to_v8_deep_min`) exactly, including that no-Deep-anywhere case.
fn dist_to_nearest_deep_min(starts: &[i64], labels: &[SleepStage]) -> Vec<f64> {
    let n = starts.len();
    let mut dist = vec![f64::INFINITY; n];
    let mut last_deep: Option<i64> = None;
    for i in 0..n {
        if labels[i] == SleepStage::Deep {
            last_deep = Some(starts[i]);
        }
        if let Some(t) = last_deep {
            dist[i] = dist[i].min((starts[i] - t).abs() as f64 / 60.0);
        }
    }
    let mut next_deep: Option<i64> = None;
    for i in (0..n).rev() {
        if labels[i] == SleepStage::Deep {
            next_deep = Some(starts[i]);
        }
        if let Some(t) = next_deep {
            dist[i] = dist[i].min((starts[i] - t).abs() as f64 / 60.0);
        }
    }
    dist
}

/// Whether epoch `i` is the first or last epoch of its own contiguous run of `Deep` in `labels` --
/// `false` for a non-`Deep` epoch. A single-epoch island is both first and last, so it is always an
/// edge. O(n), one pass. Always computed from the PRELIMINARY (pre-B+1) fused decode, per the frozen
/// spec -- never re-derived from the gated result (no recursion/iteration to convergence).
fn fused_deep_run_edges(labels: &[SleepStage]) -> Vec<bool> {
    let n = labels.len();
    let mut edge = vec![false; n];
    for i in 0..n {
        if labels[i] != SleepStage::Deep {
            continue;
        }
        let entry = i == 0 || labels[i - 1] != SleepStage::Deep;
        let exit = i + 1 >= n || labels[i + 1] != SleepStage::Deep;
        edge[i] = entry || exit;
    }
    edge
}

/// The second-highest of {A, B, C} among the channels that individually, strictly, clear `em_light` --
/// the same eligibility test `h_abc_deep_value`/`habc_agreeing` use. `NEG_INFINITY` when zero channels
/// qualify (sequence context, not this epoch's own emission, made the decoder choose Deep here), so the
/// gate conditions below still evaluate consistently. Mirrors `print_bplus.rs::second_strongest` exactly,
/// including this edge case -- not invented for this port.
fn habc_second_strongest(em_deep_v8: f64, em_deep_v9: f64, em_deep_rr: f64, em_light: f64) -> f64 {
    let mut qual: Vec<f64> = Vec::with_capacity(3);
    if em_deep_v8 > em_light {
        qual.push(em_deep_v8);
    }
    if em_deep_v9 > em_light {
        qual.push(em_deep_v9);
    }
    if em_deep_rr > em_light {
        qual.push(em_deep_rr);
    }
    qual.sort_by(|a, b| b.partial_cmp(a).unwrap());
    match qual.len() {
        0 => f64::NEG_INFINITY,
        1 => qual[0],
        _ => qual[1],
    }
}

/// Apply the frozen B+1 gate to a full night's preliminary H-ABC-fused emissions. Two-pass by
/// construction: PASS0 (`em_v8`, a V8-only decode) gives the distance-to-V8-Deep every epoch needs;
/// PASS1 (`em_fused`, the preliminary H-ABC decode) gives the run-edge structure. Matches the frozen
/// research replay (`print_bplus.rs`) exactly -- edges are read once from the PASS1 decode and never
/// re-derived from the gated result. Returns the gated per-epoch emissions (what the caller's own final
/// decode -- PASS2 -- should score) and, per epoch, whether B+1 reverted it (forensic only).
fn bplus_gate(
    starts: &[i64],
    em_v8: &[[f64; 4]],
    em_fused: &[[f64; 4]],
    habc_b: &[f64],
    habc_c: &[f64],
    transition: &[[f64; 4]; 4],
) -> (Vec<[f64; 4]>, Vec<bool>) {
    let n = em_v8.len();
    if n == 0 {
        return (Vec::new(), Vec::new());
    }
    let labels_v8 = viterbi(em_v8, transition);
    let labels_fused = viterbi(em_fused, transition);
    let dist = dist_to_nearest_deep_min(starts, &labels_v8);
    let is_edge = fused_deep_run_edges(&labels_fused);

    let mut gated = em_fused.to_vec();
    let mut reverted = vec![false; n];
    for i in 0..n {
        let added = labels_fused[i] == SleepStage::Deep && labels_v8[i] != SleepStage::Deep;
        if !added {
            continue;
        }
        let second = habc_second_strongest(em_v8[i][DEEP], habc_b[i], habc_c[i], em_v8[i][LIGHT]);
        let gate_b =
            dist[i] > HABC_BPLUS_DIST_MIN_MINUTES && second < HABC_BPLUS_WEAK_SECOND_STRONGEST;
        let gate_edge = !gate_b && is_edge[i] && second < HABC_BPLUS_EDGE_SECOND_STRONGEST;
        if gate_b || gate_edge {
            gated[i][DEEP] = em_v8[i][DEEP];
            reverted[i] = true;
        }
    }
    (gated, reverted)
}

/// Run the full recipe over a night's epochs and return one stage label per epoch. All normalisation
/// (z-scores, the HR-flatness percentile) is within the night.
fn stage_epochs(feats: &[Epoch], p: &Params) -> Vec<SleepStage> {
    if feats.is_empty() {
        return Vec::new();
    }
    viterbi(&final_emissions(feats, p), &p.transition)
}

/// Per-epoch log-emissions under `p`, with the time-of-night priors read from `anchor`.
fn emissions(feats: &[Epoch], p: &Params, anchor: Anchor) -> Vec<[f64; 4]> {
    let blp = p.base_log_prior();
    let zhr = ZScore::build(&feats.iter().map(|f| f.hr).collect::<Vec<_>>());
    let zhv = ZScore::build(&feats.iter().map(|f| f.hr_var).collect::<Vec<_>>());
    let zmv = ZScore::build(&feats.iter().map(|f| f.move_frac).collect::<Vec<_>>());
    let zrg = ZScore::build(&feats.iter().map(|f| f.resp_reg).collect::<Vec<_>>());

    let mut fsorted: Vec<f64> = feats.iter().filter_map(|f| f.hr_flat11).collect();
    fsorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let fpct = |value: Option<f64>| -> f64 {
        match value {
            Some(v) if !fsorted.is_empty() => {
                // bisect_right / n
                let mut lo = 0usize;
                let mut hi = fsorted.len();
                while lo < hi {
                    let mid = (lo + hi) / 2;
                    if fsorted[mid] <= v {
                        lo = mid + 1;
                    } else {
                        hi = mid;
                    }
                }
                lo as f64 / fsorted.len() as f64
            }
            _ => 0.5,
        }
    };

    let mut seq: Vec<[f64; 4]> = Vec::with_capacity(feats.len());
    for (i, f) in feats.iter().enumerate() {
        let zhrv = zhr.apply(f.hr);
        let zhvv = zhv.apply(f.hr_var);
        let zmvv = zmv.apply(f.move_frac);
        let gate = p.deep_gate_slope * (fpct(f.hr_flat11) - p.deep_gate_thresh).max(0.0);
        let awake_cardiac0 =
            p.awake_hrv * dz(zhvv, p.awake_deadzone) + p.awake_hr * dz(zhrv, p.awake_deadzone);
        let awake_cardiac = if motion_quiescent(f, p) {
            awake_cardiac0.min(0.0)
        } else {
            awake_cardiac0
        };

        let mut em = [0.0f64; 4];
        em[DEEP] = p.deep_hrv * zhvv + p.deep_hr * zhrv + p.deep_motion * zmvv - gate + blp[DEEP];
        em[REM] = p.rem_hrv * zhvv + p.rem_motion * zmvv + p.rem_hr * zhrv + blp[REM];
        em[LIGHT] = blp[LIGHT];
        em[AWAKE] = p.awake_motion * zmvv + awake_cardiac + blp[AWAKE];

        let pr = cycle_prior(
            cycle_clock(f.clock, feats, anchor, p),
            rem_guard(i, f.clock, anchor, p),
            p,
        );
        for (s, p) in pr.iter().enumerate() {
            em[s] += p;
        }
        if f.jerk_max > f.jerk_scale * p.jerk_gate_mult {
            em[AWAKE] += p.motion_gate_boost;
        }
        if let Some(rg) = f.resp_reg {
            let z = zrg.apply(Some(rg));
            em[DEEP] += p.resp_weight * z;
            em[REM] -= p.resp_weight * z;
            em[DEEP] += late_deep_interaction(
                f.clock,
                zhrv,
                zhvv,
                f.move_frac.map(|_| zmvv),
                Some(z),
                p.deep_hr,
            );
        }

        // === Architecture H-ABC / frozen ">=2-of-3" Deep candidate ===================================
        // A is em[DEEP] as computed above, complete and untouched by anything below. Duplicated
        // term-for-term against `diagnostics()`'s identical block (see `diagnostics_matches_emissions`).
        let habc_a = em[DEEP];
        let habc_b = em_deep_v9_of(f, blp[DEEP]);
        let habc_c = em_deep_rr_of(f, blp[DEEP]);
        let habc_mm = habc_motion_mod(f.move_frac, f.low_motion_run_min);
        em[DEEP] = h_abc_deep_value(habc_a, habc_b, habc_c, em[LIGHT], habc_mm);

        seq.push(em);
    }
    seq
}

/// Channel B (`em_deep_v9`) from an epoch's already-computed trailing HR/motion features.
fn em_deep_v9_of(f: &Epoch, blp_deep: f64) -> f64 {
    em_deep_v9(f.hr_trail_std_600s, f.hr_trail_std_120s, f.move_trail_mean_600s, blp_deep)
}

/// Channel C (`em_deep_rr`) from an epoch's already-computed trailing RR/HRV features.
fn em_deep_rr_of(f: &Epoch, blp_deep: f64) -> f64 {
    em_deep_rr(f.sdnn_10min, f.rr_mad_10min, f.rmssd_5min, blp_deep)
}

/// Population standard deviation over the trailing window of up to `n_ep` epochs ending at and including
/// index `i` (fewer at the start of the sequence). Missing (`None`) values inside the window are skipped,
/// not zero-filled. `None` when fewer than 2 present values fall in the window. The window is by EPOCH
/// INDEX in this already-gap-dropped sequence (an epoch with neither HR nor gravity was already dropped
/// upstream in `features()`), not by wall-clock time -- mirrors
/// `research/frozen_holdout_validation_2026-09-22/scripts/build_epoch_table.py::trailing_pstdev` exactly.
fn trailing_pstdev(vals: &[Option<f64>], i: usize, n_ep: usize) -> Option<f64> {
    let lo = i.saturating_sub(n_ep.saturating_sub(1));
    let w: Vec<f64> = vals[lo..=i].iter().filter_map(|v| *v).collect();
    if w.len() < 2 {
        return None;
    }
    Some(crate::stats::population_sd(&w))
}

/// Trailing mean over the same window semantics as [`trailing_pstdev`]; `None` only when the window has
/// no present values at all. Mirrors `build_epoch_table.py::trailing_mean` exactly.
fn trailing_mean(vals: &[Option<f64>], i: usize, n_ep: usize) -> Option<f64> {
    let lo = i.saturating_sub(n_ep.saturating_sub(1));
    let w: Vec<f64> = vals[lo..=i].iter().filter_map(|v| *v).collect();
    if w.is_empty() {
        None
    } else {
        Some(w.iter().sum::<f64>() / w.len() as f64)
    }
}

/// One RR/HRV window's SDNN / RR-MAD / gap-aware RMSSD over the beats in `[lo, hi)` (wall-clock seconds),
/// read from the already-flattened, second-sorted, UNCLAMPED `rr` slice `features()` builds via
/// `flatten_rr`. Mirrors `build_epoch_table.py::rr_window_stats` exactly: population SDNN and
/// median-absolute-deviation require >=2 beats (else `None` for both), RMSSD sums only over consecutive
/// beat pairs -- in `rr`'s own already-second-sorted, intra-second-emission-preserved order -- whose gap
/// is <= `max_gap_sec`, and requires >=1 such pair (else `None`).
struct RrWindowStats {
    sdnn: Option<f64>,
    rr_mad: Option<f64>,
    rmssd: Option<f64>,
}

fn rr_window_stats(rr: &[(i64, f64)], lo: i64, hi: i64, max_gap_sec: i64) -> RrWindowStats {
    let a = rr.partition_point(|&(ts, _)| ts < lo);
    let b = rr.partition_point(|&(ts, _)| ts < hi);
    let beats = &rr[a..b];
    if beats.is_empty() {
        return RrWindowStats {
            sdnn: None,
            rr_mad: None,
            rmssd: None,
        };
    }
    let vals: Vec<f64> = beats.iter().map(|&(_, v)| v).collect();
    let (sdnn, rr_mad) = if vals.len() >= 2 {
        let med = median(&vals);
        let sd = crate::stats::population_sd(&vals);
        let devs: Vec<f64> = vals.iter().map(|v| (v - med).abs()).collect();
        (Some(sd), Some(median(&devs)))
    } else {
        (None, None)
    };
    let mut sq_sum = 0.0;
    let mut nd = 0usize;
    for w in beats.windows(2) {
        let gap = w[1].0 - w[0].0;
        if gap <= max_gap_sec {
            let d = w[1].1 - w[0].1;
            sq_sum += d * d;
            nd += 1;
        }
    }
    let rmssd = if nd >= 1 {
        Some((sq_sum / nd as f64).sqrt())
    } else {
        None
    };
    RrWindowStats { sdnn, rr_mad, rmssd }
}

/// Channel B: frozen HR_STABILITY_DEEP. `blp_deep` is `p.base_log_prior()[DEEP]` -- identically
/// `ln(0.15)` under `Params::SHIPPED`, matching the frozen research's own `BLP_DEEP = ln(0.15)` exactly
/// (proven equal by construction, not merely similar, so it is reused rather than re-declared).
fn em_deep_v9(
    hr_trail_std_600s: Option<f64>,
    hr_trail_std_120s: Option<f64>,
    move_trail_mean_600s: Option<f64>,
    blp_deep: f64,
) -> f64 {
    let mut combined = 0.0;
    if let Some(v) = hr_trail_std_600s {
        combined -= (v - HABC_V9_HR600_MED) / HABC_V9_HR600_IQR;
    }
    if let Some(v) = hr_trail_std_120s {
        combined -= (v - HABC_V9_HR120_MED) / HABC_V9_HR120_IQR;
    }
    if let Some(v) = move_trail_mean_600s {
        combined -= (v - HABC_V9_MOVE600_MED) / HABC_V9_MOVE600_IQR;
    }
    blp_deep + HABC_V9_K_SCALE * combined
}

/// Channel C: frozen RR_HRV_DEEP. See [`em_deep_v9`] on `blp_deep`.
fn em_deep_rr(
    sdnn_10min: Option<f64>,
    rr_mad_10min: Option<f64>,
    rmssd_5min: Option<f64>,
    blp_deep: f64,
) -> f64 {
    let mut combined = 0.0;
    if let Some(v) = sdnn_10min {
        combined -= (v - HABC_RR_SDNN10_MED) / HABC_RR_SDNN10_IQR;
    }
    if let Some(v) = rr_mad_10min {
        combined -= (v - HABC_RR_MAD10_MED) / HABC_RR_MAD10_IQR;
    }
    if let Some(v) = rmssd_5min {
        combined -= (v - HABC_RR_RMSSD5_MED) / HABC_RR_RMSSD5_IQR;
    }
    blp_deep + HABC_K_RR * combined
}

/// How many of {A, B, C} individually, strictly, exceed `em_light` -- the frozen fusion's own agreement
/// count, exposed separately so a diagnostic caller doesn't have to reimplement the comparison.
fn habc_agreeing(em_deep_v8: f64, em_deep_v9: f64, em_deep_rr: f64, em_light: f64) -> usize {
    [em_deep_v8, em_deep_v9, em_deep_rr]
        .iter()
        .filter(|&&v| v > em_light)
        .count()
}

/// The frozen ">=2-of-3" fusion. Candidates are the values among {A, B, C} that individually, strictly,
/// exceed `em_light` (each channel's own reference-zero eligibility test); when at least
/// `HABC_MIN_AGREE` (2) qualify, the fused Deep emission is their max plus the motion modifier;
/// otherwise it falls back to A (`em_deep_v8`) UNCHANGED -- never `-inf`, never `em_light` itself.
/// Mirrors `frozen_constants.py::h_abc_deep_value(..., min_agree=2)` exactly.
fn h_abc_deep_value(em_deep_v8: f64, em_deep_v9: f64, em_deep_rr: f64, em_light: f64, motion_mod: f64) -> f64 {
    let mut candidates: Vec<f64> = Vec::with_capacity(3);
    if em_deep_v8 > em_light {
        candidates.push(em_deep_v8);
    }
    if em_deep_v9 > em_light {
        candidates.push(em_deep_v9);
    }
    if em_deep_rr > em_light {
        candidates.push(em_deep_rr);
    }
    if candidates.len() >= HABC_MIN_AGREE {
        candidates.iter().copied().fold(f64::NEG_INFINITY, f64::max) + motion_mod
    } else {
        em_deep_v8
    }
}

/// Channel E: the frozen motion veto/bonus. A hard veto when the CURRENT epoch's `move_frac` clears the
/// veto threshold; else a capped bonus once the CAUSAL running low-motion streak (in minutes) exceeds the
/// TRUE_LIGHT-anchored reference median. Mirrors `build_epoch_table.py::motion_mod` exactly.
fn habc_motion_mod(move_frac: Option<f64>, low_motion_run_min: Option<f64>) -> f64 {
    if let Some(m) = move_frac {
        if m >= HABC_MOVE_VETO_THRESH {
            return HABC_MOTION_VETO;
        }
    }
    if let Some(run_min) = low_motion_run_min {
        let z = (run_min - HABC_IMMOB_MED_MIN) / HABC_IMMOB_IQR_MIN;
        if z > 0.0 {
            return HABC_MOTION_BONUS_CAP.min(0.05 * z);
        }
    }
    0.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sleep::input::RrRun;

    /// One minute of HR with gravity over only the first half: the epochs that saw the accelerometer
    /// carry a motion reading, the epochs that did not carry `None`.
    fn half_blind_epochs() -> Vec<Epoch> {
        let start = 1_749_513_600i64;
        let hr: Vec<HrSample> = (0..60)
            .map(|i| HrSample {
                ts: start + i,
                bpm: 55,
            })
            .collect();
        let accel: Vec<AccelSample> = (0..30)
            .map(|i| AccelSample {
                ts: start + i,
                x: 0.0,
                y: 0.0,
                z: 1.0,
            })
            .collect();
        let input = SleepInput {
            start,
            end: start + 60,
            hr,
            rr: Vec::<RrRun>::new(),
            accel,
        };
        let mut grav = input.accel.clone();
        grav.sort_by_key(|g| g.ts);
        features(
            input.start,
            input.end,
            &grav,
            &input.hr,
            &[],
            &Params::SHIPPED,
        )
    }

    #[test]
    fn late_deep_interaction_removes_the_low_hr_penalty_when_all_channels_agree() {
        let got = late_deep_interaction(0.80, -1.2, -0.5, Some(-0.1), Some(0.2), 0.5);
        assert!((got - 1.0).abs() < 1e-12);
    }

    #[test]
    fn late_deep_interaction_requires_favorable_rsa() {
        assert_eq!(
            0.0,
            late_deep_interaction(0.80, -1.2, -0.5, Some(-0.1), Some(-0.1), 0.5)
        );
    }

    #[test]
    fn late_deep_interaction_requires_low_observed_motion() {
        assert_eq!(
            0.0,
            late_deep_interaction(0.80, -1.2, -0.5, Some(0.1), Some(0.2), 0.5)
        );
    }

    #[test]
    fn late_deep_interaction_requires_rsa_measurement() {
        assert_eq!(
            0.0,
            late_deep_interaction(0.80, -1.2, -0.5, Some(-0.1), None, 0.5)
        );
    }

    #[test]
    fn late_deep_interaction_does_not_apply_before_sixty_five_percent() {
        assert_eq!(
            0.0,
            late_deep_interaction(0.649, -1.2, -0.5, Some(-0.1), Some(0.2), 0.5)
        );
    }

    #[test]
    fn late_rem_like_hr_variability_does_not_receive_deep_support() {
        assert_eq!(
            0.0,
            late_deep_interaction(0.80, -1.2, 0.5, Some(-0.1), Some(0.2), 0.5)
        );
    }

    #[test]
    fn an_epoch_without_gravity_reports_no_motion_reading() {
        let feats = half_blind_epochs();
        assert_eq!(feats.len(), 2, "two 30 s epochs");
        assert_eq!(
            feats[0].move_frac,
            Some(0.0),
            "gravity present and still -> a measured zero"
        );
        assert_eq!(
            feats[1].move_frac, None,
            "no gravity -> no reading, not a measured zero"
        );
    }

    #[test]
    fn an_epoch_without_gravity_is_not_motion_quiescent() {
        // The quiescent gate clamps the awake-cardiac term, so asserting it on absent motion would let a
        // dropout suppress wake. Only an epoch that actually SAW a still wrist may be quiescent.
        let feats = half_blind_epochs();
        assert!(
            motion_quiescent(&feats[0], &Params::SHIPPED),
            "an observed still wrist is quiescent"
        );
        assert!(
            !motion_quiescent(&feats[1], &Params::SHIPPED),
            "an absent accelerometer cannot assert stillness"
        );
    }

    #[test]
    fn sustained_onset_is_the_first_epoch_of_the_first_long_non_wake_run() {
        let w = SleepStage::Wake;
        let l = SleepStage::Light;
        // A 9-epoch run is too short; the 10-epoch run that follows it is the onset.
        let mut labels = vec![w; 3];
        labels.extend(vec![l; 9]);
        labels.push(w);
        labels.extend(vec![l; 10]);
        assert_eq!(Some(13), sustained_onset(&labels));
        assert_eq!(None, sustained_onset(&[l; SUSTAINED_ONSET_EPOCHS - 1]));
    }

    #[test]
    fn the_onset_anchored_guard_grades_from_onset_not_from_the_window() {
        let p = Params::SHIPPED;
        assert!(
            p.cycle_rem_onset_minutes > 0.0,
            "the shipped guard is onset-anchored"
        );
        let onset = 40usize;
        let full = p.cycle_rem_early_penalty;
        // Full magnitude at onset, nothing at onset + the grading width, and never negative before onset.
        assert_eq!(full, rem_guard(onset, 0.9, Anchor::Onset(onset), &p));
        let past = onset + (p.cycle_rem_onset_minutes / EPOCH_MIN) as usize;
        assert_eq!(0.0, rem_guard(past, 0.9, Anchor::Onset(onset), &p));
        assert_eq!(full, rem_guard(0, 0.9, Anchor::Onset(onset), &p));
        // A window anchor keeps the old step, and the probe pass carries no guard at all.
        assert_eq!(full, rem_guard(0, 0.01, Anchor::Window, &p));
        assert_eq!(0.0, rem_guard(0, 0.9, Anchor::Window, &p));
        assert_eq!(0.0, rem_guard(0, 0.01, Anchor::Probe, &p));
    }

    #[test]
    fn the_onset_anchored_clock_rebases_only_when_asked() {
        let feats = half_blind_epochs();
        let shipped = Params::SHIPPED;
        let from_onset = Params {
            cycle_clock_from_onset: true,
            ..shipped
        };
        let c0 = feats[0].clock;
        assert_eq!(
            0.75,
            cycle_clock(0.75, &feats, Anchor::Onset(0), &shipped),
            "off by default"
        );
        assert_eq!(
            0.75,
            cycle_clock(0.75, &feats, Anchor::Window, &from_onset),
            "no anchor, no rebase"
        );
        assert_eq!(
            (0.75 - c0) / (1.0 - c0),
            cycle_clock(0.75, &feats, Anchor::Onset(0), &from_onset),
            "rebased onto the onset epoch's own fraction"
        );
        assert_eq!(
            0.0,
            cycle_clock(0.0, &feats, Anchor::Onset(1), &from_onset),
            "clamped below onset"
        );
    }

    // ── the decoder, isolated from the emissions feeding it ───────────────────────────────────────

    /// Column index of a stage — the inverse of [`STAGE_ORDER`], so a test can talk in indices.
    fn idx_of(s: SleepStage) -> usize {
        STAGE_ORDER.iter().position(|&x| x == s).unwrap()
    }

    /// Total log-score of one path: its emissions plus every transition it takes. Written independently
    /// of the decoder, so a test can rank paths without using the thing under test.
    fn path_score(em: &[[f64; 4]], t: &[[f64; 4]; 4], path: &[usize]) -> f64 {
        let mut s = em[0][path[0]];
        for i in 1..path.len() {
            s += t[path[i - 1]][path[i]].max(1e-9).ln() + em[i][path[i]];
        }
        s
    }

    /// Best path and its score by exhaustive enumeration of all 4^n. The oracle the decoder is checked
    /// against; feasible only for a handful of epochs, which is why it lives in a test.
    fn brute_force(em: &[[f64; 4]], t: &[[f64; 4]; 4]) -> (Vec<usize>, f64) {
        let n = em.len();
        let mut best = (Vec::new(), f64::NEG_INFINITY);
        for code in 0..4usize.pow(n as u32) {
            let path: Vec<usize> = (0..n).map(|i| (code >> (2 * i)) & 3).collect();
            let s = path_score(em, t, &path);
            if s > best.1 {
                best = (path, s);
            }
        }
        best
    }

    /// Deterministic LCG in [-4, 4) — the same emission sequences on every run, at a spread comparable to
    /// the log-transitions so neither term trivially dominates.
    fn lcg(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((*state >> 33) as f64) / ((1u64 << 31) as f64) * 8.0 - 4.0
    }

    /// A second matrix beside the shipped one, with a hard zero row so the 1e-9 floor is exercised.
    const PROBE_T: [[f64; 4]; 4] = [
        [0.50, 0.20, 0.20, 0.10],
        [0.10, 0.40, 0.40, 0.10],
        [0.25, 0.25, 0.25, 0.25],
        [0.00, 0.00, 0.50, 0.50],
    ];

    #[test]
    fn the_decoder_returns_the_maximum_likelihood_path_on_every_crafted_sequence() {
        // 80 sequences of 7 epochs, alternating between the shipped matrix and one with a zero row, each
        // checked against exhaustive enumeration of all 16,384 paths. A decoder that matched once would
        // prove nothing.
        let mut state = 0x5eed_1234_u64;
        let mut checked = 0usize;
        for case in 0..80 {
            let t = if case % 2 == 0 {
                Params::SHIPPED.transition
            } else {
                PROBE_T
            };
            let em: Vec<[f64; 4]> = (0..7)
                .map(|_| {
                    [
                        lcg(&mut state),
                        lcg(&mut state),
                        lcg(&mut state),
                        lcg(&mut state),
                    ]
                })
                .collect();
            let got: Vec<usize> = viterbi(&em, &t).into_iter().map(idx_of).collect();
            let (want, best) = brute_force(&em, &t);
            assert_eq!(
                path_score(&em, &t, &got),
                best,
                "case {case}: decoded path is not optimal"
            );
            assert_eq!(
                got, want,
                "case {case}: decoded path differs from the enumerated best"
            );
            checked += 1;
        }
        assert_eq!(80, checked);
    }

    #[test]
    fn every_injected_path_is_recovered_when_the_emissions_name_it() {
        // All 1,024 paths over 5 epochs, injected as an emission margin large enough to outweigh any
        // transition (the 1e-9 floor is worth 20.7 log units, and a deviation can gain at most two of
        // them). Recovery of one path is an anecdote; this is the whole space.
        let t = Params::SHIPPED.transition;
        let mut recovered = 0usize;
        for code in 0..1024usize {
            let want: Vec<usize> = (0..5).map(|i| (code >> (2 * i)) & 3).collect();
            let em: Vec<[f64; 4]> = want
                .iter()
                .map(|&s| {
                    let mut row = [0.0; 4];
                    row[s] = 200.0;
                    row
                })
                .collect();
            let got: Vec<usize> = viterbi(&em, &t).into_iter().map(idx_of).collect();
            assert_eq!(got, want, "path {code} not recovered");
            recovered += 1;
        }
        assert_eq!(1024, recovered);
    }

    #[test]
    fn the_transitions_override_a_one_epoch_emission_blip() {
        // Hand-computable: REM-REM-REM scores 10 + 2*ln(0.92) = 9.83, while following the middle epoch's
        // own argmax into deep scores 11 + ln(0.00333) + ln(0.012) = 0.87. Without the transition term the
        // decoder would be a per-epoch argmax, so this is what separates the two.
        let t = Params::SHIPPED.transition;
        let mut em = [[0.0f64; 4]; 3];
        em[0][REM] = 5.0;
        em[1][DEEP] = 1.0;
        em[2][REM] = 5.0;
        let got: Vec<usize> = viterbi(&em, &t).into_iter().map(idx_of).collect();
        assert_eq!(
            vec![REM, REM, REM],
            got,
            "the sticky diagonal must smooth a single blip"
        );
        assert_eq!(
            DEEP,
            (0..4)
                .max_by(|a, b| em[1][*a].total_cmp(&em[1][*b]))
                .unwrap(),
            "argmax disagrees"
        );
        // And it is not unconditional: widen the blip's margin past the two transitions it must pay for
        // and the decoder follows the evidence instead.
        em[1][DEEP] = 12.0;
        assert_eq!(
            DEEP,
            idx_of(viterbi(&em, &t)[1]),
            "a large enough margin must win"
        );
    }

    #[test]
    fn the_decoder_is_total_on_degenerate_input() {
        let t = Params::SHIPPED.transition;
        assert!(viterbi(&[], &t).is_empty(), "no epochs, no labels");
        // One epoch has no transition to pay, so it is the emission argmax alone.
        assert_eq!(SleepStage::Wake, viterbi(&[[0.0, 1.0, 2.0, 3.0]], &t)[0]);
        // Everything equal under a uniform matrix: every path ties and the earliest column wins.
        let uniform = [[0.25; 4]; 4];
        assert!(
            viterbi(&[[0.0; 4]; 6], &uniform)
                .iter()
                .all(|&s| s == SleepStage::Deep)
        );
    }

    /// Twenty minutes of HR and gravity, still for the first half and jittering in the second, so the
    /// staging has a path to search rather than one flat answer.
    fn crafted_night() -> Prepared {
        let start = 1_749_513_600i64;
        let (mut hr, mut accel) = (Vec::new(), Vec::new());
        for i in 0..1200i64 {
            let bpm = if i < 600 {
                52 + (i % 3) as u16
            } else {
                66 + (i % 5) as u16
            };
            hr.push(HrSample { ts: start + i, bpm });
            let jitter = if i < 600 { 0.0 } else { 0.02 * (i % 7) as f64 };
            accel.push(AccelSample {
                ts: start + i,
                x: jitter,
                y: 0.0,
                z: 1.0 - jitter,
            });
        }
        let input = SleepInput {
            start,
            end: start + 1200,
            hr,
            rr: Vec::<RrRun>::new(),
            accel,
        };
        prepare(&input, &Params::SHIPPED)
    }

    #[test]
    fn diagnostics_matches_emissions() {
        // The forensic export must never drift from the emissions the decoder actually scores.
        let prep = crafted_night();
        let p = Params::SHIPPED;
        let em = emissions_prepared(&prep, &p);
        let diag = diagnostics_prepared(&prep, &p);
        assert_eq!(em.len(), diag.len());
        for (row, d) in em.iter().zip(&diag) {
            assert_eq!(row[DEEP], d.em_deep);
            assert_eq!(row[REM], d.em_rem);
            assert_eq!(row[LIGHT], d.em_light);
            assert_eq!(row[AWAKE], d.em_awake);
        }
        assert_eq!(epoch_starts(&prep), diag.iter().map(|d| d.start).collect::<Vec<_>>());
    }

    #[test]
    fn the_exported_emissions_decode_to_the_labels_the_stager_ships() {
        // The exported pair has to BE the shipped staging, or a harness scoring them scores a
        // reconstruction of the pipeline instead of the pipeline.
        let prep = crafted_night();
        let p = Params::SHIPPED;
        let em = emissions_prepared(&prep, &p);
        assert_eq!(em.len(), prep.feats.len(), "one emission row per epoch");
        assert_eq!(
            em.len(),
            epoch_starts(&prep).len(),
            "one epoch start per emission row"
        );
        assert!(
            em.len() >= 30,
            "a night long enough for the path search to matter"
        );
        let via_decoder = segments_of(&prep, &viterbi(&em, &p.transition));
        assert_eq!(stage_prepared(&prep, &p), via_decoder);
    }

    #[test]
    fn every_transition_row_is_a_distribution_and_a_zero_is_only_a_floor() {
        // The rows are read as probabilities and logged, so a row that does not sum to 1 would weight one
        // stage's whole outgoing mass. A 0 cell is floored, not forbidden.
        for (i, row) in Params::SHIPPED.transition.iter().enumerate() {
            assert!(
                (row.iter().sum::<f64>() - 1.0).abs() < 1e-9,
                "row {i} is not a distribution"
            );
        }
        let t = Params::SHIPPED.transition;
        assert_eq!(
            0.0, t[AWAKE][DEEP],
            "the awake row's zeros are what the floor has to cover"
        );
        let em = [[0.0, 0.0, 0.0, 50.0], [50.0, 0.0, 0.0, 0.0]];
        assert_eq!(
            vec![SleepStage::Wake, SleepStage::Deep],
            viterbi(&em, &t),
            "a zero cell is traversable"
        );
    }

    #[test]
    fn an_absent_motion_reading_z_scores_neutral() {
        // The z-scorer must place "no reading" at the centre, not at the bottom of the night's motion
        // distribution, so a dropout never reads as the stillest stretch of the night.
        let z = ZScore::build(&[Some(0.0), Some(4.0), Some(8.0), None]);
        assert_eq!(z.apply(None), 0.0);
        assert!(
            z.apply(Some(0.0)) < 0.0,
            "a measured zero sits below the night mean"
        );
        assert!(
            z.apply(None) > z.apply(Some(0.0)),
            "absence must not out-still a measured zero"
        );
    }

    #[test]
    fn staging_algorithm_version_is_the_shipped_sleep_staging_v4_habc_identity() {
        // Pins the code-level staging identity so a future edit here is a deliberate, visible bump,
        // not a silent drift. Distinct from `shadow_metrics::ALGORITHM_VERSION_V4`, which stamps the
        // nightly-physiology/RHR bundle and must NOT move when only staging changes. Bumped from
        // `sleep-staging-v3` to `sleep-staging-v4-habc` when the frozen H-ABC Deep candidate shipped.
        assert_eq!(STAGING_ALGORITHM_VERSION, "sleep-staging-v4-habc");
    }

    #[test]
    fn shipped_deep_hrv_is_the_approved_minus_one_point_zero() {
        // Pins the one approved staging-coefficient change (was -0.8) so it cannot silently regress
        // or drift further without a visible test failure.
        assert_eq!(Params::SHIPPED.deep_hrv, -1.0);
    }

    #[test]
    fn only_deep_hrv_moved_from_its_prior_shipped_value() {
        // Every other SHIPPED staging coefficient must be byte-for-byte the value this task found
        // and was explicitly told to leave alone. Catches an accidental edit to any neighbor while
        // touching deep_hrv.
        let p = Params::SHIPPED;
        assert_eq!(p.deep_hr, 0.5);
        assert_eq!(p.deep_motion, -0.1);
        assert_eq!(p.rem_hrv, 0.8);
        assert_eq!(p.rem_motion, -0.4);
        assert_eq!(p.rem_hr, 0.4);
        assert_eq!(p.awake_motion, 1.0);
        assert_eq!(p.awake_hrv, 0.5);
        assert_eq!(p.awake_hr, 0.6);
        assert_eq!(p.awake_deadzone, 0.30);
        assert_eq!(p.deep_gate_thresh, 0.40);
        assert_eq!(p.deep_gate_slope, 5.0);
        assert_eq!(p.resp_weight, 0.6);
        assert_eq!(p.base_rate, [0.15, 0.22, 0.50, 0.34]);
        assert_eq!(p.cycle_deep_scale, 1.2);
        assert_eq!(p.cycle_deep_decay, 0.55);
        assert_eq!(
            p.transition,
            [
                [0.76, 0.012, 0.216, 0.012],
                [0.00333, 0.92, 0.06667, 0.01],
                [0.08, 0.08, 0.80, 0.04],
                [0.0, 0.0, 0.10, 0.90],
            ]
        );
    }

    // ── Architecture H-ABC / frozen ">=2-of-3" Deep candidate ─────────────────────────────────────

    #[test]
    fn trailing_pstdev_windows_by_index_and_skips_missing() {
        let vals = vec![Some(1.0), Some(2.0), None, Some(4.0), Some(6.0)];
        // window of 3 ending at i=4 (indices 2,3,4): None skipped -> [4,6], pstdev = 1.0
        assert!((trailing_pstdev(&vals, 4, 3).unwrap() - 1.0).abs() < 1e-12);
        // window of 2 ending at i=1: [1,2], pstdev = 0.5
        assert!((trailing_pstdev(&vals, 1, 2).unwrap() - 0.5).abs() < 1e-12);
        // window of 3 ending at i=2 (indices 0,1,2): [1,2,None] -> [1,2], pstdev = 0.5
        assert!((trailing_pstdev(&vals, 2, 3).unwrap() - 0.5).abs() < 1e-12);
        // fewer than 2 present values in the window -> None
        assert_eq!(None, trailing_pstdev(&vals, 0, 1));
    }

    #[test]
    fn trailing_mean_windows_by_index_and_skips_missing() {
        let vals = vec![Some(0.0), None, Some(0.2), Some(0.4)];
        // window of 3 ending at i=3 (indices 1,2,3): None skipped -> [0.2,0.4], mean = 0.3
        assert!((trailing_mean(&vals, 3, 3).unwrap() - 0.3).abs() < 1e-12);
        // a single present value is enough for a mean (unlike pstdev's >=2 requirement)
        assert!((trailing_mean(&vals, 0, 5).unwrap() - 0.0).abs() < 1e-12);
        // entirely-missing window -> None
        assert_eq!(None, trailing_mean(&[None, None], 1, 2));
    }

    #[test]
    fn rr_window_stats_sdnn_and_mad_require_two_beats() {
        let one_beat = vec![(100i64, 800.0)];
        let s = rr_window_stats(&one_beat, 0, 200, 5);
        assert_eq!(None, s.sdnn);
        assert_eq!(None, s.rr_mad);

        // three beats: 700, 800, 900 -- mean=800, population sdnn = sqrt(20000/3); median=800,
        // abs-devs=[100,0,100], median-abs-dev = 100.
        let three = vec![(100i64, 700.0), (101, 800.0), (102, 900.0)];
        let s2 = rr_window_stats(&three, 0, 200, 5);
        assert!((s2.sdnn.unwrap() - (20_000.0f64 / 3.0).sqrt()).abs() < 1e-9);
        assert!((s2.rr_mad.unwrap() - 100.0).abs() < 1e-12);
    }

    #[test]
    fn rr_window_stats_rmssd_is_gap_aware() {
        // beats at t=0,1,2 (successive gaps of 1s, <=5s, included), then a jump to t=20 (gap 18s, its
        // diff from the t=2 beat is excluded).
        let rr = vec![(0i64, 800.0), (1, 850.0), (2, 800.0), (20, 1000.0)];
        let s = rr_window_stats(&rr, 0, 30, 5);
        // included diffs: 850-800=50, 800-850=-50 -> squares 2500,2500 -> mean 2500 -> rmssd = 50
        assert!((s.rmssd.unwrap() - 50.0).abs() < 1e-9);
    }

    #[test]
    fn rr_window_stats_window_is_half_open() {
        let rr = vec![(100i64, 700.0), (200, 800.0)];
        // hi=200 excludes the beat AT ts=200 -> only one beat in [100,200) -> not enough for sdnn/mad
        let s = rr_window_stats(&rr, 100, 200, 5);
        assert_eq!(None, s.sdnn, "only one beat in the half-open window");
        // widen to [100,201) and both beats are included
        let s2 = rr_window_stats(&rr, 100, 201, 5);
        assert!(s2.sdnn.is_some(), "both beats now in window");
    }

    #[test]
    fn em_deep_v9_arithmetic_is_hand_computable() {
        let blp_deep = 0.15f64.ln();
        // all three features exactly at their reference medians -> combined = 0 -> em = blp_deep
        let at_median = em_deep_v9(
            Some(HABC_V9_HR600_MED),
            Some(HABC_V9_HR120_MED),
            Some(HABC_V9_MOVE600_MED),
            blp_deep,
        );
        assert!((at_median - blp_deep).abs() < 1e-12);

        // one feature present, one IQR BELOW its median (flatter HR = more Deep-typical) ->
        // combined = -((med - iqr) - med) / iqr = 1.0 -> em = blp_deep + K_SCALE * 1.0
        let one_iqr_below = em_deep_v9(Some(HABC_V9_HR600_MED - HABC_V9_HR600_IQR), None, None, blp_deep);
        assert!((one_iqr_below - (blp_deep + HABC_V9_K_SCALE)).abs() < 1e-9);

        // missing features contribute nothing -> all-None reduces to blp_deep exactly
        assert!((em_deep_v9(None, None, None, blp_deep) - blp_deep).abs() < 1e-12);
    }

    #[test]
    fn em_deep_rr_arithmetic_is_hand_computable() {
        let blp_deep = 0.15f64.ln();
        let at_median = em_deep_rr(
            Some(HABC_RR_SDNN10_MED),
            Some(HABC_RR_MAD10_MED),
            Some(HABC_RR_RMSSD5_MED),
            blp_deep,
        );
        assert!((at_median - blp_deep).abs() < 1e-12);

        let one_iqr_below =
            em_deep_rr(Some(HABC_RR_SDNN10_MED - HABC_RR_SDNN10_IQR), None, None, blp_deep);
        assert!((one_iqr_below - (blp_deep + HABC_K_RR)).abs() < 1e-9);

        assert!((em_deep_rr(None, None, None, blp_deep) - blp_deep).abs() < 1e-12);
    }

    #[test]
    fn h_abc_fusion_falls_back_to_a_when_zero_channels_qualify() {
        // A, B, C all <= em_light -> 0 candidates -> falls back to A verbatim, not -inf, not em_light.
        assert_eq!(-2.0, h_abc_deep_value(-2.0, -3.0, -5.0, -1.0, 0.7));
    }

    #[test]
    fn h_abc_fusion_falls_back_to_a_when_exactly_one_channel_qualifies() {
        // only B qualifies (5.0 > light=0.0); A and C do not -> 1 candidate, still falls back to A.
        assert_eq!(-2.0, h_abc_deep_value(-2.0, 5.0, -1.0, 0.0, 0.7));
    }

    #[test]
    fn h_abc_fusion_fires_when_exactly_two_channels_qualify() {
        // A and C qualify (both > light=0.0), B does not -> max(A,C) + motion_mod.
        let got = h_abc_deep_value(3.0, -1.0, 4.0, 0.0, 0.7);
        assert!((got - (4.0 + 0.7)).abs() < 1e-12);
    }

    #[test]
    fn h_abc_fusion_fires_when_all_three_channels_qualify() {
        let got = h_abc_deep_value(3.0, 9.0, 4.0, 0.0, -3.0); // motion veto applied too
        assert!((got - (9.0 - 3.0)).abs() < 1e-12);
    }

    #[test]
    fn h_abc_fusion_eligibility_is_a_strict_greater_than() {
        // A and B sit exactly AT em_light (do not qualify under strict >); only C qualifies -> 1
        // candidate -> falls back to A. Under a wrongly-lenient `>=` this would instead return
        // max(A,C)+motion_mod = 5.0, so the assertion below distinguishes the two.
        let got = h_abc_deep_value(0.0, 0.0, 5.0, 0.0, 0.0);
        assert_eq!(0.0, got);
    }

    #[test]
    fn h_abc_missing_b_or_c_is_just_blp_deep_and_can_still_fail_to_qualify() {
        // A channel with no trailing features at all reduces to blp_deep -- it can still fail to beat
        // em_light like any other candidate, it is never specially excluded from the vote.
        let blp_deep = 0.15f64.ln();
        let b = em_deep_v9(None, None, None, blp_deep);
        let c = em_deep_rr(None, None, None, blp_deep);
        assert!((b - blp_deep).abs() < 1e-12);
        assert!((c - blp_deep).abs() < 1e-12);
    }

    #[test]
    fn habc_motion_mod_vetoes_on_current_epoch_movement() {
        assert_eq!(HABC_MOTION_VETO, habc_motion_mod(Some(0.15), Some(100.0)));
        assert_eq!(HABC_MOTION_VETO, habc_motion_mod(Some(0.9), None));
    }

    #[test]
    fn habc_motion_mod_bonus_only_past_the_reference_median_and_capped() {
        // at/below the reference median -> no bonus
        assert_eq!(0.0, habc_motion_mod(Some(0.0), Some(HABC_IMMOB_MED_MIN)));
        // just above -> a small positive bonus
        let z = (5.0 - HABC_IMMOB_MED_MIN) / HABC_IMMOB_IQR_MIN;
        let got = habc_motion_mod(Some(0.0), Some(5.0));
        assert!((got - (0.05 * z).min(HABC_MOTION_BONUS_CAP)).abs() < 1e-12);
        // far above -> capped
        assert_eq!(HABC_MOTION_BONUS_CAP, habc_motion_mod(Some(0.0), Some(1000.0)));
    }

    #[test]
    fn habc_motion_mod_is_zero_when_no_motion_evidence_at_all() {
        assert_eq!(0.0, habc_motion_mod(None, None));
    }

    #[test]
    fn habc_diagnostic_fields_are_internally_consistent_with_the_fused_emission() {
        // The forensic export must be recomputable from its own A/B/C/E/light columns exactly, so a
        // caller (or a parity harness) never has to trust `em_deep` without being able to check it.
        // `em_deep` is the FINAL (post-B+1) value: it equals `h_abc_deep_value(...)` (the preliminary
        // fused value) UNLESS B+1 reverted this epoch, in which case it equals `em_deep_v8` instead --
        // `habc_bplus_reverted` says which, so this stays a exact (not just "close enough") check.
        let prep = crafted_night();
        let p = Params::SHIPPED;
        let diag = diagnostics_prepared(&prep, &p);
        for d in &diag {
            let prelim =
                h_abc_deep_value(d.em_deep_v8, d.em_deep_v9, d.em_deep_rr, d.em_light, d.habc_motion_mod);
            let want = if d.habc_bplus_reverted { d.em_deep_v8 } else { prelim };
            assert!(
                (want - d.em_deep).abs() < 1e-12,
                "fused em_deep must equal h_abc_deep_value(A,B,C,light,E), or em_deep_v8 if B+1 reverted it"
            );
            assert_eq!(
                habc_agreeing(d.em_deep_v8, d.em_deep_v9, d.em_deep_rr, d.em_light),
                d.habc_agree_count
            );
        }
    }

    // === Frozen "B+1" (Challenger B + boundary/edge confidence backstop) — unit tests over the small,
    // pure `bplus_gate`/`dist_to_nearest_deep_min`/`fused_deep_run_edges`/`habc_second_strongest`
    // primitives directly, so each frozen condition is pinned in isolation rather than only through a
    // full night's Viterbi behavior. `light` = -0.7, non-Deep slots = -5.0 (never win) throughout.

    #[test]
    fn bplus_challenger_b_removes_a_far_weak_added_epoch() {
        // (A) far-from-V8 + weak second evidence -> the boost IS removed.
        let starts = [0i64, 1860]; // epoch1 sits 31.0 min from the only V8-Deep epoch.
        let em_v8 = [[5.0, -5.0, -0.7, -5.0], [-5.0, -5.0, -0.7, -5.0]];
        let em_fused = [em_v8[0], [0.3, -5.0, -0.7, -5.0]];
        let habc_b = [0.0, -0.3]; // qualifies (> -0.7), qual = [-0.3] alone -> second_strongest = -0.3 < 0.5
        let habc_c = [0.0, -5.0]; // does not qualify
        let (gated, reverted) =
            bplus_gate(&starts, &em_v8, &em_fused, &habc_b, &habc_c, &Params::SHIPPED.transition);
        assert!(reverted[1], "far + weak second_strongest must be gated by Challenger B");
        assert_eq!(em_v8[1][DEEP], gated[1][DEEP]);
    }

    #[test]
    fn bplus_challenger_b_alone_does_not_remove_a_far_confident_added_epoch() {
        // (B) far-from-V8 + strong evidence -> B alone must NOT remove it.
        let starts = [0i64, 1860];
        let em_v8 = [[5.0, -5.0, -0.7, -5.0], [-5.0, -5.0, -0.7, -5.0]];
        let em_fused = [em_v8[0], [2.0, -5.0, -0.7, -5.0]];
        let habc_b = [0.0, 2.0];
        let habc_c = [0.0, 1.0]; // qual = [2.0, 1.0] -> second_strongest = 1.0, not < 0.5
        let (gated, reverted) =
            bplus_gate(&starts, &em_v8, &em_fused, &habc_b, &habc_c, &Params::SHIPPED.transition);
        assert!(!reverted[1], "far + confident agreement must survive Challenger B");
        assert_eq!(em_fused[1][DEEP], gated[1][DEEP]);
    }

    #[test]
    fn bplus_edge_backstop_fires_only_on_weak_edges_never_interior_never_strong() {
        // Epoch 0 = V8-Deep anchor; epoch 1 = a clear Light separator in BOTH decodes, so the anchor's
        // own Deep run never merges with the added run below; epochs 2..=5 = a 4-epoch preliminary
        // fused-Deep run (all V8-Light, all within 5 min of the anchor so Challenger B's own dist>30
        // term can never fire here -- isolates the edge backstop). Epoch 2 = entry edge (weak),
        // 3/4 = interior (weak, same evidence as 2), 5 = exit edge (strong).
        let starts = [0i64, 60, 120, 180, 240, 300];
        let light = -0.7;
        let em_v8 = [
            [5.0, -5.0, light, -5.0], // V8 Deep anchor
            [-5.0, -5.0, 2.0, -5.0],  // separator: clearly Light in both decodes
            [-5.0, -5.0, light, -5.0], // V8 Light (added candidates below)
            [-5.0, -5.0, light, -5.0],
            [-5.0, -5.0, light, -5.0],
            [-5.0, -5.0, light, -5.0],
        ];
        let em_fused = [
            em_v8[0],
            em_v8[1],
            [1.0, -5.0, light, -5.0],
            [1.0, -5.0, light, -5.0],
            [1.0, -5.0, light, -5.0],
            [1.0, -5.0, light, -5.0],
        ];
        // (C)/(D) weak (second_strongest = -0.3 < 0.0) on entry (2) and interior (3, 4).
        let habc_b = [0.0, 0.0, -0.3, -0.3, -0.3, 0.5];
        let habc_c = [0.0, 0.0, -5.0, -5.0, -5.0, 0.4]; // epoch 5: qual=[0.5,0.4] -> second_strongest=0.4
        let (gated, reverted) =
            bplus_gate(&starts, &em_v8, &em_fused, &habc_b, &habc_c, &Params::SHIPPED.transition);

        assert!(!reverted[0], "(F) a V8-Deep epoch is never eligible to be gated by B+1");
        assert!(!reverted[1], "the separator was never fused-Deep, so it is never 'added'");
        assert!(reverted[2], "(C) a weak entry-edge epoch must be reverted by the edge backstop");
        assert!(!reverted[3], "(D) a weak INTERIOR epoch must not be reverted");
        assert!(!reverted[4], "(D) a weak INTERIOR epoch must not be reverted");
        assert!(!reverted[5], "(E) a strong exit-edge epoch must not be reverted");

        assert_eq!(em_v8[0][DEEP], gated[0][DEEP]);
        assert_eq!(em_v8[2][DEEP], gated[2][DEEP]);
        assert_eq!(em_fused[3][DEEP], gated[3][DEEP]);
        assert_eq!(em_fused[4][DEEP], gated[4][DEEP]);
        assert_eq!(em_fused[5][DEEP], gated[5][DEEP]);

        // (J) REM/Light/Awake are never touched by the gate, reverted or not.
        for i in 0..starts.len() {
            assert_eq!(em_fused[i][REM], gated[i][REM]);
            assert_eq!(em_fused[i][LIGHT], gated[i][LIGHT]);
            assert_eq!(em_fused[i][AWAKE], gated[i][AWAKE]);
        }
    }

    #[test]
    fn bplus_run_edges_are_computed_in_one_pass_over_the_preliminary_decode() {
        // (G) run boundaries: two separate runs plus a gap, first/last of EACH run flagged independently.
        let l = SleepStage::Light;
        let d = SleepStage::Deep;
        let labels = [l, d, d, d, l, l, d, l];
        let edge = fused_deep_run_edges(&labels);
        assert_eq!(
            edge,
            vec![false, true, false, true, false, false, true, false],
            "first/last of each contiguous Deep run flagged, interior epochs and non-Deep epochs are not"
        );
    }

    #[test]
    fn bplus_a_single_epoch_deep_island_is_its_own_edge() {
        // (H) a one-epoch island is simultaneously the first and last epoch of its own run -> edge.
        let l = SleepStage::Light;
        let d = SleepStage::Deep;
        let edge = fused_deep_run_edges(&[l, d, l]);
        assert_eq!(edge, vec![false, true, false]);
    }

    #[test]
    fn bplus_distance_is_infinite_when_v8_never_decodes_deep_anywhere() {
        // (I) no-V8-Deep-anywhere: matches the frozen research replay's own convention exactly (never
        // invented for this port -- see `print_bplus.rs::dist_to_v8_deep_min`).
        let starts = [0i64, 60, 120];
        let labels = [SleepStage::Light, SleepStage::Rem, SleepStage::Wake];
        let dist = dist_to_nearest_deep_min(&starts, &labels);
        assert!(dist.iter().all(|d| d.is_infinite() && *d > 0.0));
    }
}
