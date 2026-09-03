//! Personal Cardiovascular Profile: a user-scoped, evidence-gated, versioned estimate of personal
//! Max HR, plus the HRR/relative-intensity primitive built from it and the chronic RHR baseline.
//!
//! This module is the implementation of the two prior specs:
//!   - `hr-intensity-healthspan-architecture.md` (codebase audit + architecture)
//!   - `hr-intensity-model-spec.md` (evidence-based physiological model, "Freeze this for implementation")
//! and of the superseding architecture note that made continuous %HRR the single physiological
//! source of truth, with workout Z0-5 and Healthspan light/moderate/vigorous as two independent
//! consumers of it (see `hrr_zones.rs` and `healthspan_intensity.rs`).
//!
//! Deliberately reuses rather than re-derives: `strain::pct_hrr` is THE %HRR formula (no second
//! implementation here); `strain::estimate_hrmax`'s percentile idea this module extends into a
//! persisted, multi-session form (see `SESSION_HRMAX_PERCENTILE`'s doc); `hr_recovery::calculate`
//! and `hr_gap` for evidence corroboration, unchanged. The cold-start/insufficient-evidence
//! population prior is Gellish(age) (`gellish_hrmax`, below) - deliberately NOT
//! `strain::tanaka_hrmax`, which remains that module's own separate, unrelated fallback (see
//! `gellish_hrmax`'s doc for why the two are not the same authority).
//!
//! CONFIDENCE NOTE (read before tuning constants): the recency window, decay half-life, and the
//! MEDIUM/HIGH corroboration thresholds below are ENGINEERING JUDGMENT, not literature-derived
//! numbers the way e.g. `ZONE_EDGES` in `healthspan_intensity.rs` are. Flagged LOW confidence
//! individually below; every one is a named constant, not an inline magic number, so a future
//! tuning pass has one place to look and one thing to change without touching the update logic.
//!
//! TWO SEPARATE CONFIDENCE QUESTIONS (conceptual correction, see `MaxHrProfile.confidence` and
//! `observed_lower_bound_confidence` for the full reasoning): a credible HR observation of X proves
//! roughly `MaxHr >= X`, not `MaxHr == X`. Repeated corroboration of the same value legitimately
//! earns HIGH confidence in the FIRST claim (`observed_lower_bound_confidence`, computed on demand
//! from `evidence`) - it must NOT by itself earn HIGH confidence in the SECOND
//! (`MaxHrProfile.confidence`, the point-estimate/ceiling question), because nothing in
//! `evaluate_session_evidence`'s gates distinguishes a hard-but-sustainable effort from a genuinely
//! maximal one. `MaxHrProfile.confidence` is therefore capped at `Medium` for `FieldInferred` in v1
//! (see `point_estimate_confidence`) regardless of corroboration strength - this is a deliberate,
//! conservative choice given the signals actually available in this codebase today, not an
//! oversight to be "fixed" by loosening the cap without a genuinely new near-maximal-effort signal.

pub use crate::hr_gap::{GapPosition, classify as gap_classify};
pub use crate::hr_sample::HrSample;
use crate::strain::percentile_pct;

/// Gellish et al. (2011) age-predicted Max HR: 207 - 0.7 x age. THE canonical cold-start/
/// insufficient-evidence prior for the Personal Cardiovascular Profile (product decision,
/// superseding the Tanaka formula `strain::tanaka_hrmax` still uses for its own, separate,
/// unrelated fallback path - see that function's doc; this module never calls it). Not claimed to
/// be WHOOP's own exact cold-start formula - WHOOP's public material confirms age-based
/// initialization with later personalization but does not establish its precise equation. Any
/// Swift/local mirror of this constant must stay in lockstep by hand (see `Profile.swift`'s doc) -
/// this Rust function is the one authority it mirrors.
pub fn gellish_hrmax(age: f64) -> f64 {
    207.0 - 0.7 * age
}

// ── HRR primitive (delegates to `strain::pct_hrr`; nothing new is computed here) ──────────────────

/// Heart-rate reserve: `max_hr - rhr_baseline`. `None` when non-physiological (reserve <= 0), which
/// callers should treat as "personalization unavailable, fall back to the population-prior path."
pub fn heart_rate_reserve(max_hr: f64, rhr_baseline: f64) -> Option<f64> {
    let hrr = max_hr - rhr_baseline;
    (hrr > 0.0).then_some(hrr)
}

/// Continuous personalized relative intensity in `[0, 100]`, i.e. %HRR. Thin wrapper over
/// `strain::pct_hrr` - the canonical HRR formula already used by strain's internal Edwards TRIMP
/// zone weighting; this module does not reimplement it, only exposes it under the name the zone
/// consumers (`hrr_zones`, `healthspan_intensity`) are built against.
pub fn relative_intensity_pct(bpm: f64, rhr_baseline: f64, hrr: f64) -> f64 {
    crate::strain::pct_hrr(bpm, rhr_baseline, hrr)
}

// ── Provenance / confidence ────────────────────────────────────────────────────────────────────

/// Where the current `max_hr` value came from. Renamed slightly from the spec's draft
/// (`FIELD_OBSERVED` -> `FieldInferred`) to make explicit this is an already-processed inference,
/// not a raw observed value (the raw observed value is `observed_peak`, tracked separately below).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MaxHrSource {
    /// Tanaka(age), no qualifying field evidence yet.
    AgeEstimate,
    /// Built from corroborated wearable evidence via `update_from_evidence`.
    FieldInferred,
    /// Explicit user entry.
    UserSupplied,
    /// A `UserSupplied` value present alongside field evidence that meaningfully disagrees with it.
    /// Surfaced, never silently resolved in either direction.
    Hybrid,
}

/// HIGH confidence never means "measured": see the terminology lock in `hr-intensity-model-spec.md`
/// Part 14 - a wearable-inferred Max HR remains an estimate at every tier. These tiers describe how
/// much corroborating evidence backs the current value, not how precise it is in absolute bpm terms.
///
/// CORRECTION (conceptual validation pass): this enum is used for TWO DIFFERENT QUESTIONS in this
/// module, and they must not be conflated - see `MaxHrProfile.confidence`'s doc and
/// `observed_lower_bound_confidence`'s doc for exactly which question each one answers. A repeated,
/// clean, hard-but-ordinary training peak can legitimately earn HIGH confidence as an OBSERVED LOWER
/// BOUND ("we're sure this bpm was genuinely reached") without that at all establishing HIGH
/// confidence that the value IS the physiological ceiling - nothing in `evaluate_session_evidence`'s
/// gates (sustain/continuity/effort-floor/recovery) distinguishes a hard sustainable effort from a
/// genuine near-maximal one (no plateau-despite-continued-effort signal, no RPE, no lactate, no
/// explicit maximal-test tagging available in this codebase today - see
/// `MaxHrProfile.confidence`'s doc for the full reasoning and why the point estimate is deliberately
/// capped for v1).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Confidence {
    Low,
    Medium,
    High,
}

// ── Evidence ────────────────────────────────────────────────────────────────────────────────────

/// One session's candidate Max HR evidence, already gated (constructing one of these means the
/// session passed the qualification checks in `evaluate_session_evidence` - a rejected session never
/// produces an instance of this type, so a caller cannot accidentally fold in a spike).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MaxHrEvidence {
    /// The candidate peak bpm this session supports.
    pub candidate_bpm: f64,
    /// Session end timestamp (unix seconds) - used for recency decay.
    pub observed_at: i64,
    /// Quality in `[0, 1]`: how corroborating this single session's evidence is on its own, from
    /// sustain duration, recovery-shape plausibility, sample continuity and modality confidence.
    /// A quality near 1.0 is "about as good as field evidence gets"; this is NOT a lab-test
    /// equivalence claim (see module doc), it is relative to other field sessions only.
    pub quality: f64,
}

/// LOW CONFIDENCE (engineering judgment, not literature): minimum sustained-near-peak duration
/// before a session's peak is even considered. Matches the order of magnitude already used
/// elsewhere in this crate for "sustained" (`hr_recovery::MINIMUM_HIGH_INTENSITY_SECONDS` = 120,
/// `workout::MIN_EXERCISE_MIN` = 5.0 min) rather than inventing an unrelated number.
pub const MIN_SUSTAIN_NEAR_PEAK_SECONDS: i64 = 90;

/// LOW CONFIDENCE: minimum number of HR samples in the session before a percentile-based candidate
/// peak is trusted at all. `strain::estimate_hrmax`'s own threshold (`HRMAX_MIN_SAMPLES` = 600, a
/// full day of 1Hz-equivalent coverage) is calibrated for a whole-day stream, not one workout, so a
/// materially lower session-scoped minimum is used here and called out as a deliberate departure.
pub const MIN_SESSION_SAMPLES: usize = 60;

/// Percentile used for the session's candidate peak - same idea as `strain::HRMAX_PERCENTILE`
/// (99.5th) but slightly relaxed for a shorter, session-scoped sample: a single-workout stream has
/// far fewer points than the 600+ `estimate_hrmax` expects, so a slightly lower percentile keeps the
/// "not literally the single highest sample" artifact-resistance property without starving a normal
/// workout of a crediting sample. LOW CONFIDENCE, not re-derived from the cited literature.
pub const SESSION_HRMAX_PERCENTILE: f64 = 98.0;

/// A session's peak only counts as evidence at all if its %HRR (against the CURRENT estimate) is at
/// least this high - i.e. it needs to look like a genuinely hard effort, not a light jog with one
/// noisy sample. HIGH CONFIDENCE that *some* floor belongs here (this is what the spec calls
/// "whether the effort appears genuinely maximal/submaximal"); the exact number is MEDIUM confidence,
/// set to match the vigorous-intensity floor already independently justified in
/// `healthspan_intensity.rs` (60% HRR, ACSM/WHO-sourced) rather than inventing a third number.
pub const MIN_EFFORT_HRR_PCT: f64 = 60.0;

/// Evaluate one completed session/bout for Max HR evidence. Returns `None` when the session does not
/// qualify at all (too few samples, too short a sustain, or the resulting recovery/continuity checks
/// fail) - a rejected session is simply absent from the caller's evidence history, never a zero-weight
/// entry, so it cannot silently accumulate influence.
///
/// `current_max_hr` is the profile's estimate BEFORE this session - used both as the recovery-
/// eligibility threshold (matching `hr_recovery`'s own contract) and as the denominator for the
/// effort-floor gate above. `rhr_baseline` is the chronic RHR baseline (never last night's raw
/// value - see `baselines.rs` / Part F of the model spec).
/// `modality_confidence` is `[0, 1]`, caller-supplied (e.g. from `workout.rs` bout classification):
/// 1.0 for a session logged/detected as a hard run/ride/HIIT effort, lower for an ambiguous or
/// low-expected-intensity modality (a "walk" producing a claimed near-maximal reading should not
/// receive full weight even if the raw numbers look clean).
pub fn evaluate_session_evidence(
    samples: &[HrSample],
    workout_start: i64,
    workout_end: i64,
    current_max_hr: f64,
    rhr_baseline: f64,
    modality_confidence: f64,
) -> Option<MaxHrEvidence> {
    if workout_start <= 0 || workout_end <= workout_start || current_max_hr <= rhr_baseline {
        return None;
    }
    let in_session: Vec<HrSample> = samples
        .iter()
        .copied()
        .filter(|s| s.ts >= workout_start && s.ts <= workout_end)
        .collect();
    if in_session.len() < MIN_SESSION_SAMPLES {
        return None;
    }

    // Candidate peak: a high percentile of the session's own samples, not the single max - the same
    // artifact-resistance idea as `strain::estimate_hrmax`, applied at session scope.
    let mut bpms: Vec<f64> = in_session.iter().map(|s| s.bpm as f64).collect();
    bpms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let candidate_bpm = percentile_pct(&bpms, SESSION_HRMAX_PERCENTILE);

    let hrr = heart_rate_reserve(current_max_hr, rhr_baseline)?;
    let effort_pct = relative_intensity_pct(candidate_bpm, rhr_baseline, hrr);
    if effort_pct < MIN_EFFORT_HRR_PCT {
        return None; // doesn't look like a genuinely hard effort against what we currently believe
    }

    // Sustained-near-peak: reuse the same idea as `hr_recovery`'s eligibility check (time spent at or
    // above a high fraction of the reference max), computed directly here so this module does not
    // depend on `hr_recovery`'s internal (non-pub) `sustained_seconds` helper.
    let near_peak_threshold = candidate_bpm - 5.0; // within 5bpm of the session's own candidate peak
    let mut sustained = 0i64;
    for w in in_session.windows(2) {
        let gap = w[1].ts - w[0].ts;
        if (1..=10).contains(&gap) && (w[0].bpm as f64) >= near_peak_threshold {
            sustained += gap;
        }
    }
    if sustained < MIN_SUSTAIN_NEAR_PEAK_SECONDS {
        return None;
    }

    // Continuity around the peak: find the sample nearest the candidate percentile value and confirm
    // its neighbouring gap is Cadence or a short Bridge, not a Refuse - an artifact sitting in a
    // refused span must not corroborate.
    let peak_idx = in_session
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            (a.bpm as f64 - candidate_bpm)
                .abs()
                .partial_cmp(&(b.bpm as f64 - candidate_bpm).abs())
                .unwrap()
        })
        .map(|(i, _)| i)?;
    let mut continuity_clean = true;
    if peak_idx > 0 {
        let gap = (in_session[peak_idx].ts - in_session[peak_idx - 1].ts) as f64;
        if gap_classify(gap, GapPosition::Interior) == crate::hr_gap::GapVerdict::Refuse {
            continuity_clean = false;
        }
    }
    if peak_idx + 1 < in_session.len() {
        let gap = (in_session[peak_idx + 1].ts - in_session[peak_idx].ts) as f64;
        if gap_classify(gap, GapPosition::Interior) == crate::hr_gap::GapVerdict::Refuse {
            continuity_clean = false;
        }
    }
    if !continuity_clean {
        return None;
    }

    // Recovery shape: opportunistic corroboration, not a hard requirement (many qualifying sessions
    // will have insufficient post-workout coverage - e.g. the user stopped recording - and the spec
    // is explicit recovery/RPE/plateau are not always available). When present and physiologically
    // plausible (a real, non-negative drop), it raises quality; when absent, quality is lower but the
    // session can still qualify on sustain + continuity + effort-floor alone.
    // BUG FIX (verification pass): this must be `samples` (the full, un-filtered stream), NOT
    // `in_session` (already filtered to `[workout_start, workout_end]`). `hr_recovery::calculate`
    // needs samples AFTER `workout_end` (up to +5min) to compute the 1/2/5-minute recovery deltas;
    // passing `in_session` meant no post-workout sample could ever reach it, so `recovery_quality`
    // was structurally always 0.0 regardless of what recovery data actually existed - silently
    // disabling this corroboration signal for every session ever evaluated.
    let recovery_quality = match crate::hr_recovery::calculate(
        samples,
        workout_start,
        workout_end,
        candidate_bpm,
    ) {
        Some(r) if r.after_1min.is_some_and(|d| d > 0) => 1.0,
        Some(r) if r.has_measurement() => 0.5, // measured but not a clean positive drop at 1min
        _ => 0.0,
    };

    let sustain_quality = ((sustained as f64) / (5.0 * MIN_SUSTAIN_NEAR_PEAK_SECONDS as f64)).min(1.0);
    let quality = (0.35 * sustain_quality
        + 0.35 * recovery_quality
        + 0.30 * modality_confidence.clamp(0.0, 1.0))
    .clamp(0.0, 1.0);

    Some(MaxHrEvidence {
        candidate_bpm,
        observed_at: workout_end,
        quality,
    })
}

// ── Profile state ───────────────────────────────────────────────────────────────────────────────

/// LOW CONFIDENCE: how long a piece of evidence keeps contributing to corroboration before its
/// weight has fully decayed. Chosen as the same order of magnitude as the model spec's proposed
/// recency window (~16 weeks), expressed as a half-life so old evidence fades gradually rather than
/// dropping off a cliff.
pub const EVIDENCE_HALF_LIFE_DAYS: f64 = 56.0;

/// LOW CONFIDENCE: cumulative quality-weighted corroboration score thresholds. Deliberately NOT
/// "exactly N workouts" (see module doc): a single quality~1.0 session can cross the raise/MEDIUM
/// threshold on its own; crossing HIGH generally wants either one truly exceptional session plus at
/// least a little agreement, or several good ones.
///
/// `MEDIUM_CONFIDENCE_THRESHOLD` gates two DIFFERENT things that must not be confused (conceptual
/// correction): (1) whether new evidence is strong enough to raise `max_hr` at all
/// (`update_from_evidence`), and (2) the low tier of `observed_lower_bound_confidence`.
/// `HIGH_CONFIDENCE_THRESHOLD` gates ONLY `observed_lower_bound_confidence`'s top tier - it no
/// longer gates `MaxHrProfile.confidence` (the point estimate), which is capped at `Medium` for
/// `FieldInferred` regardless of corroboration (see `point_estimate_confidence`).
pub const MEDIUM_CONFIDENCE_THRESHOLD: f64 = 0.55;
pub const HIGH_CONFIDENCE_THRESHOLD: f64 = 1.6;

/// LOW CONFIDENCE: how far a new session's candidate value may deviate from the current estimate and
/// still count as "agreeing" (full weight) rather than "novel" (reduced weight, since a single
/// wildly-higher reading disagreeing with everything else so far should raise the estimate cautiously,
/// not instantly promote to HIGH on its own say-so).
pub const AGREEMENT_TOLERANCE_BPM: f64 = 4.0;

/// LOW CONFIDENCE: maximum bpm a single evidence-driven update may raise `max_hr` by, regardless of
/// how strong the evidence is - the "don't bounce around" clamp from the model spec (Part 9 / Part L).
pub const MAX_RAISE_STEP_BPM: f64 = 5.0;

/// Confidence is downgraded one tier after this many days with no qualifying evidence at all -
/// mirrors `physio_algo::baselines::STALE_DAYS`'s pattern (seen-and-trusted decays to a lower-trust
/// state after a fixed inactivity window) applied to Max HR instead of a nightly metric. MEDIUM
/// CONFIDENCE: the pattern is a direct, deliberate reuse of existing precedent in this crate; the
/// specific day count (180, ~6 months) is the model spec's proposed number, not independently
/// re-derived here.
pub const STALE_AFTER_DAYS: i64 = 180;

/// The user-scoped Personal Cardiovascular Profile's Max HR component. RHR baseline is intentionally
/// NOT duplicated here - it stays owned by `baselines::BaselineState` (Part F of the model spec:
/// "reuse the existing chronic RHR baseline implementation... do not invent a second one"). A
/// caller assembles the full HRR primitive from `(profile.max_hr, baseline.baseline)` at the point
/// of use (`heart_rate_reserve`/`relative_intensity_pct` above).
#[derive(Clone, Debug, PartialEq)]
pub struct MaxHrProfile {
    pub max_hr: f64,
    pub source: MaxHrSource,
    /// Confidence in `max_hr` AS THE PHYSIOLOGICAL CEILING (the point estimate) - a DIFFERENT,
    /// DELIBERATELY MORE CONSERVATIVE question than "how sure are we this bpm was genuinely
    /// reached" (see `observed_lower_bound_confidence`, computed separately, not stored here).
    ///
    /// CORRECTION (conceptual validation pass): earlier versions of this module let repeated
    /// corroboration of the same observed value push THIS field to `High`, which overstated what
    /// the evidence proves - three clean, hard, sustained-but-submaximal sessions at ~194bpm prove
    /// `MaxHr >= ~194` with real confidence; they do not prove `MaxHr == 194`, because nothing in
    /// `evaluate_session_evidence` distinguishes "hard and sustainable" from "at the physiological
    /// limit". For `MaxHrSource::FieldInferred` this field is now DELIBERATELY CAPPED at `Medium`
    /// regardless of corroboration strength - see `point_estimate_confidence`. Raise that cap only
    /// alongside a real near-maximal-effort signal (HR plateau despite continued effort, an
    /// explicit maximal-test tag, etc.) becoming available - not by loosening the cap in isolation.
    /// `UserSupplied` stays `Medium` by design too (a self-report, never `High` by default - see
    /// `apply_user_override`). `AgeEstimate` is always `Low`.
    pub confidence: Confidence,
    /// The best single corroborated field-observed peak seen, independent of whether it currently
    /// drives `max_hr` - kept separate per the model spec's three-way terminology distinction
    /// (Part 6/14): this is "peak HR observed", never presented as the profile's max_hr claim itself.
    /// Call `observed_lower_bound_confidence` for how sure we are THIS value was genuinely reached -
    /// that confidence CAN legitimately be `High` even while `confidence` above stays `Medium`.
    pub observed_peak: Option<f64>,
    /// User-supplied override, if any - kept alongside the working estimate so `Hybrid` disagreement
    /// can be detected and surfaced without discarding either value.
    pub user_supplied: Option<f64>,
    /// Rolling evidence window, oldest first, pruned to `EVIDENCE_HALF_LIFE_DAYS`-scale relevance by
    /// the caller (see `prune_evidence`). Persisted so confidence/corroboration survive a restart.
    pub evidence: Vec<MaxHrEvidence>,
    pub last_meaningful_update_at: Option<i64>,
    /// Monotonic; incremented only when `max_hr` itself (the PUBLISHED value) changes - not on every
    /// internal evidence append. See "Zone stability" (model spec Part 9): the internal continuous
    /// estimate and the published value are deliberately different things.
    pub profile_version: i32,
}

impl MaxHrProfile {
    /// Cold start: Gellish(age), no evidence. `age` must be the user's real age (Part 3 of the
    /// implementation request: no more hardcoded 26).
    pub fn cold_start(age: f64) -> Self {
        Self {
            max_hr: gellish_hrmax(age),
            source: MaxHrSource::AgeEstimate,
            confidence: Confidence::Low,
            observed_peak: None,
            user_supplied: None,
            evidence: Vec::new(),
            last_meaningful_update_at: None,
            profile_version: 1,
        }
    }
}

/// Recency-decayed weight for one piece of evidence at `now`. Exponential half-life decay, matching
/// the shape (not the constants) `baselines.rs` uses elsewhere in this crate for time-weighted state.
fn decayed_weight(evidence: &MaxHrEvidence, now: i64) -> f64 {
    let age_days = ((now - evidence.observed_at).max(0) as f64) / 86_400.0;
    let decay = 0.5f64.powf(age_days / EVIDENCE_HALF_LIFE_DAYS);
    evidence.quality * decay
}

/// Drop evidence old enough to be immaterial (>~4 half-lives, <10% weight remaining) so the stored
/// list does not grow without bound. Callers persist the pruned list, not the raw append-only log.
pub fn prune_evidence(evidence: &[MaxHrEvidence], now: i64) -> Vec<MaxHrEvidence> {
    let cutoff_days = EVIDENCE_HALF_LIFE_DAYS * 4.0;
    evidence
        .iter()
        .copied()
        .filter(|e| ((now - e.observed_at).max(0) as f64) / 86_400.0 <= cutoff_days)
        .collect()
}

/// Fold one new session's evidence into the profile. Implements the asymmetric update rule from the
/// model spec Part 7/9: may only ever RAISE `max_hr` from evidence; a value below the current
/// estimate never lowers it (absence of a high reading is not evidence the ceiling is lower - see
/// module/spec discussion of the asymmetry). Lowering happens only via `apply_user_override` or
/// `apply_age_decay`, both explicit, both below.
pub fn update_from_evidence(profile: &MaxHrProfile, new_evidence: MaxHrEvidence, now: i64) -> MaxHrProfile {
    let mut evidence = profile.evidence.clone();
    evidence.push(new_evidence);
    let evidence = prune_evidence(&evidence, now);

    let observed_peak = profile
        .observed_peak
        .map(|p| p.max(new_evidence.candidate_bpm))
        .or(Some(new_evidence.candidate_bpm));

    // Corroboration score: sum of decayed weights for evidence that AGREES with the current best
    // candidate (within tolerance); evidence that is far below the current candidate still counts
    // (it's not disagreement, just a lighter effort) but does not itself push the estimate up.
    let best_candidate = evidence
        .iter()
        .map(|e| e.candidate_bpm)
        .fold(profile.max_hr, f64::max);
    let corroboration: f64 = evidence
        .iter()
        .filter(|e| (e.candidate_bpm - best_candidate).abs() <= AGREEMENT_TOLERANCE_BPM)
        .map(|e| decayed_weight(e, now))
        .sum();

    let mut next = profile.clone();
    next.evidence = evidence;
    next.observed_peak = observed_peak;

    if best_candidate > profile.max_hr && corroboration >= MEDIUM_CONFIDENCE_THRESHOLD {
        let raised = profile.max_hr + (best_candidate - profile.max_hr).min(MAX_RAISE_STEP_BPM);
        next.max_hr = raised;
        next.source = match profile.source {
            MaxHrSource::UserSupplied => MaxHrSource::Hybrid,
            MaxHrSource::Hybrid => MaxHrSource::Hybrid,
            _ => MaxHrSource::FieldInferred,
        };
        next.last_meaningful_update_at = Some(now);
        next.profile_version = profile.profile_version + 1;
    }
    // Point-estimate confidence is now a PURE FUNCTION of `source` alone (see
    // `point_estimate_confidence`'s doc) - never a function of `corroboration` magnitude. This is
    // the conceptual fix: corroboration strength legitimately drives `observed_lower_bound_confidence`
    // (below), which is a genuinely different question and is computed on demand, not stored here.
    next.confidence = point_estimate_confidence(next.source);

    next
}

/// Confidence in `max_hr` as the physiological ceiling, purely from its provenance - see
/// `MaxHrProfile.confidence`'s doc for why this is deliberately NOT sensitive to corroboration
/// strength for `FieldInferred` in v1.
fn point_estimate_confidence(source: MaxHrSource) -> Confidence {
    match source {
        MaxHrSource::AgeEstimate => Confidence::Low,
        // Capped at Medium: see MaxHrProfile.confidence's doc. No signal in this codebase today
        // (no plateau-despite-effort detection, no RPE, no lactate, no maximal-test tag) can
        // distinguish "hard and sustainable" from "at the physiological limit", so no amount of
        // corroboration of an ordinary hard effort is allowed to imply we know the true ceiling.
        MaxHrSource::FieldInferred => Confidence::Medium,
        // A self-report, not a lab measurement - never High by default (matches apply_user_override).
        MaxHrSource::UserSupplied => Confidence::Medium,
        MaxHrSource::Hybrid => Confidence::Medium,
    }
}

/// Confidence that `profile.observed_peak` (or, if higher, the best value still in
/// `profile.evidence`) was GENUINELY REACHED - i.e. confidence in the LOWER BOUND
/// `MaxHr >= this value`, a DIFFERENT and DELIBERATELY LESS CONSERVATIVE question than
/// `MaxHrProfile.confidence` (the point-estimate/ceiling question - see that field's doc). Computed
/// on demand from the same `evidence`/`observed_peak` state already on the profile, rather than
/// stored as a second field, so it can never drift out of sync with the evidence it's derived from
/// and so this correction did not require changing `MaxHrProfile`'s shape (kept in-scope to
/// `personal_cardio.rs` only, per this pass's constraint not to touch noop-engine/noop-backend).
///
/// Repeated credible observations clustered near the same high value SHOULD and DO drive this to
/// `High` - that is exactly what "we're confident the user has genuinely reached ~X bpm" means, and
/// corroboration (independent sessions agreeing) is good evidence for that claim specifically, even
/// though it is not good evidence that X is the ceiling (see `point_estimate_confidence`).
pub fn observed_lower_bound_confidence(profile: &MaxHrProfile, now: i64) -> Confidence {
    let Some(peak) = profile.observed_peak else {
        return Confidence::Low;
    };
    let corroboration: f64 = profile
        .evidence
        .iter()
        .filter(|e| (e.candidate_bpm - peak).abs() <= AGREEMENT_TOLERANCE_BPM)
        .map(|e| decayed_weight(e, now))
        .sum();
    if corroboration >= HIGH_CONFIDENCE_THRESHOLD {
        Confidence::High
    } else if corroboration >= MEDIUM_CONFIDENCE_THRESHOLD {
        Confidence::Medium
    } else {
        Confidence::Low
    }
}

/// Apply staleness: called at compute time (not per-evidence) with the days elapsed since
/// `last_meaningful_update_at` (or profile creation, if never updated). Downgrades confidence one
/// tier past `STALE_AFTER_DAYS`; never changes `max_hr` itself - staleness is about how much to
/// trust the number, not a reason to change it (see module doc / spec Part 7).
pub fn apply_staleness(profile: &MaxHrProfile, days_since_update: i64) -> MaxHrProfile {
    if days_since_update <= STALE_AFTER_DAYS {
        return profile.clone();
    }
    let mut next = profile.clone();
    next.confidence = match profile.confidence {
        Confidence::High => Confidence::Medium,
        Confidence::Medium => Confidence::Low,
        Confidence::Low => Confidence::Low,
    };
    next
}

/// Explicit user override. Per the model spec: always wins as the working value; `Hybrid` is used
/// (not silent replacement) when strong, contrary field evidence already exists, so the disagreement
/// is surfaced rather than thrown away in either direction.
pub fn apply_user_override(profile: &MaxHrProfile, user_value: f64, now: i64) -> MaxHrProfile {
    let mut next = profile.clone();
    let disagreement = profile
        .observed_peak
        .map(|p| (p - user_value).abs() > AGREEMENT_TOLERANCE_BPM * 2.0)
        .unwrap_or(false);
    next.user_supplied = Some(user_value);
    next.max_hr = user_value;
    next.source = if disagreement {
        MaxHrSource::Hybrid
    } else {
        MaxHrSource::UserSupplied
    };
    next.confidence = Confidence::Medium; // a self-report, not a lab measurement - never HIGH by default
    next.last_meaningful_update_at = Some(now);
    next.profile_version = profile.profile_version + 1;
    next
}

/// Bounded downward drift of the PRIOR only - applies solely while `source == AgeEstimate` (no field
/// evidence yet exists to protect from the asymmetry rule). Once evidence exists, age-based recompute
/// of the prior is irrelevant and must not be blended back in (model spec Part 7's explicit
/// instruction) - this function is a no-op for any other source.
pub fn apply_age_decay(profile: &MaxHrProfile, new_age: f64) -> MaxHrProfile {
    if profile.source != MaxHrSource::AgeEstimate {
        return profile.clone();
    }
    let mut next = profile.clone();
    next.max_hr = gellish_hrmax(new_age);
    next
}

// ── Daily-compute integration helper ───────────────────────────────────────────────────────────

/// LOW CONFIDENCE / EXPLICIT SIMPLIFICATION: finds candidate "session" spans directly from a day's
/// raw HR stream (contiguous runs at or above the effort floor), WITHOUT true motion-based workout
/// detection (`workout::detect`, which needs gravity/accel samples `noop-engine` currently receives
/// but discards - see `DailyComputeRequest.accel`'s `#[allow(dead_code)]`). Wiring full bout
/// detection into the daily-compute path is a materially bigger feature (motion/accel plumbing,
/// bout merging/bridging) - out of scope for this pass, matching the precedent already set in this
/// codebase for deferring `steps::fit_k`'s full calibration model out of the daily-compute path for
/// the same reason. This is a documented, versioned placeholder, not a silent shortcut: callers
/// evaluating evidence from spans returned here should pass a conservatively low
/// `modality_confidence` (e.g. 0.6, not 1.0) to `evaluate_session_evidence`, since a contiguous
/// high-HRR run is corroborated by HR shape alone, not by confirmed motion/exercise-type context.
/// Replace with real `workout::detect`-derived bouts (`modality_confidence` near 1.0 for a
/// confirmed run/ride/HIIT bout) as a follow-up, not a rewrite of this module's evidence logic.
pub fn candidate_session_spans(
    hr: &[HrSample],
    max_hr: f64,
    rhr_baseline: f64,
    min_span_seconds: i64,
) -> Vec<(i64, i64)> {
    let Some(hrr) = heart_rate_reserve(max_hr, rhr_baseline) else {
        return Vec::new();
    };
    let mut sorted: Vec<HrSample> = hr.to_vec();
    sorted.sort_by_key(|s| s.ts);

    let mut spans = Vec::new();
    let mut span_start: Option<i64> = None;
    let mut last_ts: Option<i64> = None;
    for s in &sorted {
        let pct = relative_intensity_pct(s.bpm as f64, rhr_baseline, hrr);
        let above = pct >= MIN_EFFORT_HRR_PCT;
        match (above, span_start, last_ts) {
            (true, None, _) => span_start = Some(s.ts),
            (true, Some(_), Some(prev)) if s.ts - prev > 10 => {
                // contiguity break (same 10s cap `hr_recovery::sustained_seconds` uses) - close out
                // the run so far, start a new one here.
                if let Some(start) = span_start {
                    if prev - start >= min_span_seconds {
                        spans.push((start, prev));
                    }
                }
                span_start = Some(s.ts);
            }
            (false, Some(start), Some(prev)) => {
                if prev - start >= min_span_seconds {
                    spans.push((start, prev));
                }
                span_start = None;
            }
            _ => {}
        }
        last_ts = Some(s.ts);
    }
    if let (Some(start), Some(end)) = (span_start, last_ts) {
        if end - start >= min_span_seconds {
            spans.push((start, end));
        }
    }
    spans
}

/// LOW CONFIDENCE placeholder used when calling `evaluate_session_evidence` on a
/// `candidate_session_spans` result rather than a true detected/classified workout bout - see that
/// function's doc for why this is conservative rather than 1.0.
pub const HR_ONLY_SPAN_MODALITY_CONFIDENCE: f64 = 0.6;

#[cfg(test)]
mod tests {
    use super::*;

    // ── Cold-start authority: Gellish, not Tanaka ─────────────────────────────────────────────

    #[test]
    fn gellish_hrmax_formula() {
        assert!((gellish_hrmax(30.0) - 186.0).abs() < 1e-9); // 207 - 0.7*30
        assert!((gellish_hrmax(26.0) - 188.8).abs() < 1e-9); // 207 - 0.7*26
    }

    #[test]
    fn cold_start_uses_gellish_not_tanaka() {
        let profile = MaxHrProfile::cold_start(26.0);
        assert!((profile.max_hr - gellish_hrmax(26.0)).abs() < 1e-9);
        // Must NOT match Tanaka(26) = 208 - 0.7*26 = 189.8 - the two priors differ by exactly 1.0.
        assert!((profile.max_hr - 189.8).abs() > 1e-6, "cold start must not use the Tanaka prior");
        assert_eq!(profile.source, MaxHrSource::AgeEstimate);
        assert_eq!(profile.confidence, Confidence::Low);
    }

    // Nonzero base, matching `hr_recovery`'s own test convention (`const END: i64 = 10_000`) -
    // `evaluate_session_evidence`'s `workout_start <= 0` sanity guard (mirroring
    // `hr_recovery::calculate`'s identical guard) correctly treats unix epoch 0 as an invalid/
    // sentinel timestamp, not a legitimate workout start. Fixtures below are written relative to
    // this so none of them accidentally trip that guard the way the pre-fix versions did.
    const START: i64 = 10_000;

    fn dense_session(start: i64, len_s: i64, bpm_fn: impl Fn(i64) -> i32) -> Vec<HrSample> {
        (0..len_s)
            .map(|i| HrSample {
                ts: start + i,
                bpm: bpm_fn(i),
            })
            .collect()
    }

    /// A ramp from 100bpm up to 195bpm over 300s, then a sustained plateau at 195bpm, then
    /// (optionally) a sharp recovery back down - the shared shape several tests below need, with the
    /// recovery block separable so tests can compare "with genuine recovery data" against "without".
    fn ramp_sustain_and_optionally_recover(include_recovery: bool) -> Vec<HrSample> {
        let mut samples = Vec::new();
        for i in 0..300 {
            samples.push(HrSample {
                ts: START + i,
                bpm: 100 + (i as i32 / 3).min(95),
            }); // ramp to ~195
        }
        for i in 300..420 {
            samples.push(HrSample {
                ts: START + i,
                bpm: 195,
            }); // sustained 120s near peak
        }
        if include_recovery {
            for i in 420..480 {
                samples.push(HrSample {
                    ts: START + i,
                    bpm: 195 - (i as i32 - 420),
                }); // sharp plausible recovery
            }
        }
        samples
    }

    // ── HRR primitive ──────────────────────────────────────────────────────────────────────────

    #[test]
    fn hrr_is_none_for_nonpositive_reserve() {
        assert_eq!(heart_rate_reserve(180.0, 180.0), None);
        assert_eq!(heart_rate_reserve(170.0, 180.0), None);
        assert_eq!(heart_rate_reserve(190.0, 52.0), Some(138.0));
    }

    #[test]
    fn relative_intensity_matches_strain_pct_hrr_exactly() {
        // This module must not diverge from strain's canonical formula.
        assert_eq!(
            relative_intensity_pct(140.0, 52.0, 138.0),
            crate::strain::pct_hrr(140.0, 52.0, 138.0)
        );
    }

    // ── Evidence gating ────────────────────────────────────────────────────────────────────────

    #[test]
    fn rejects_a_session_with_too_few_samples() {
        let samples = dense_session(START, 10, |_| 190);
        assert_eq!(
            evaluate_session_evidence(&samples, START, START + 9, 190.0, 52.0, 1.0),
            None
        );
    }

    #[test]
    fn rejects_a_light_effort_even_with_plenty_of_samples() {
        // 62 bpm above a 190/52 profile is well under the 60% HRR effort floor.
        let samples = dense_session(START, 600, |_| 100);
        assert_eq!(
            evaluate_session_evidence(&samples, START, START + 599, 190.0, 52.0, 1.0),
            None
        );
    }

    #[test]
    fn rejects_a_lone_spike_that_does_not_sustain() {
        let mut samples = dense_session(START, 600, |_| 100); // light throughout
        samples.push(HrSample { ts: START + 300, bpm: 210 }); // one artifact spike
        let result = evaluate_session_evidence(&samples, START, START + 599, 190.0, 52.0, 1.0);
        assert_eq!(result, None);
        // Confirm this actually exercises artifact resistance (the percentile-based candidate
        // discounting the lone spike), not just an incidental reject at an earlier gate for the
        // wrong reason - the 98th percentile of 601 samples where only 1 is the spike stays ~100,
        // which is what pushes it below the effort floor, not the sustain/continuity gates.
        let mut bpms: Vec<f64> = samples.iter().map(|s| s.bpm as f64).collect();
        bpms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let candidate = percentile_pct(&bpms, SESSION_HRMAX_PERCENTILE);
        assert!(candidate < 150.0, "the lone spike must not dominate the percentile: {candidate}");
    }

    // NOTE (verification pass): I attempted to add a dedicated "isolated/discontinuous cluster"
    // artifact test here and removed it - traced by hand, it did not actually exercise the
    // continuity gate the way it was meant to. The continuity check in `evaluate_session_evidence`
    // only inspects the immediate one-sample neighbours of the SINGLE sample nearest the candidate
    // percentile value; a cluster that is itself internally contiguous (even if it sits hours away
    // from an earlier, separate episode in the same `samples`/`(workout_start, workout_end)` window)
    // has locally-clean neighbours and is NOT caught by this check - the large gap between the two
    // episodes only fails to accumulate `sustained` seconds, which is a different gate and did not
    // reject in my constructed case once the isolated cluster was made long enough to pass sustain
    // on its own. This is a real, narrow gap (a genuinely separate high-HR episode accidentally
    // included in one evaluation window could be selected as evidence) but is NOT one of the two
    // reported failures and fixing it would mean strengthening continuity into a whole-window check
    // rather than a local one - out of scope for this narrow correction pass; flagged in the report
    // as a follow-up rather than silently patched here or silently dropped.

    #[test]
    fn accepts_a_sustained_hard_effort_with_recovery() {
        let samples = ramp_sustain_and_optionally_recover(true);
        let evidence =
            evaluate_session_evidence(&samples, START, START + 419, 190.0, 52.0, 1.0);
        assert!(evidence.is_some(), "a clean sustained effort should qualify");
        let e = evidence.unwrap();
        assert!(e.candidate_bpm > 190.0, "candidate should exceed the prior estimate: {}", e.candidate_bpm);
        assert!(e.quality > 0.3, "quality should reflect the clean sustain: {}", e.quality);
    }

    #[test]
    fn plausible_recovery_increases_quality_over_no_recovery_data() {
        // Verifies the `samples` (not `in_session`) bug fix: recovery data past `workout_end` must
        // actually reach `hr_recovery::calculate` and raise quality, not be silently invisible.
        let with_recovery = ramp_sustain_and_optionally_recover(true);
        let without_recovery = ramp_sustain_and_optionally_recover(false);
        let a = evaluate_session_evidence(&with_recovery, START, START + 419, 190.0, 52.0, 1.0)
            .expect("qualifies with recovery data");
        let b = evaluate_session_evidence(&without_recovery, START, START + 419, 190.0, 52.0, 1.0)
            .expect("qualifies without recovery data too - recovery is opportunistic, not required");
        assert!(
            a.quality > b.quality,
            "genuine post-workout recovery data must raise quality: with={} without={}",
            a.quality,
            b.quality
        );
    }

    #[test]
    fn low_modality_confidence_lowers_quality_but_does_not_by_itself_reject() {
        let samples = ramp_sustain_and_optionally_recover(false);
        let high_modality =
            evaluate_session_evidence(&samples, START, START + 419, 190.0, 52.0, 1.0).unwrap();
        let low_modality =
            evaluate_session_evidence(&samples, START, START + 419, 190.0, 52.0, 0.1).unwrap();
        assert!(low_modality.quality < high_modality.quality);
    }

    // ── Profile update: asymmetry and stability ───────────────────────────────────────────────

    #[test]
    fn a_single_strong_session_reaches_medium_not_high() {
        let profile = MaxHrProfile::cold_start(30.0); // Gellish(30) = 207 - 21 = 186.0
        let evidence = MaxHrEvidence { candidate_bpm: 192.0, observed_at: 1000, quality: 1.0 };
        let next = update_from_evidence(&profile, evidence, 1000);
        assert_eq!(next.confidence, Confidence::Medium);
        assert_eq!(next.source, MaxHrSource::FieldInferred);
        assert!(next.max_hr > profile.max_hr);
        assert_eq!(next.profile_version, profile.profile_version + 1);
    }

    #[test]
    fn raise_is_bounded_even_under_a_large_jump() {
        let profile = MaxHrProfile::cold_start(30.0); // Gellish(30) = 186.0
        let evidence = MaxHrEvidence { candidate_bpm: 230.0, observed_at: 1000, quality: 1.0 };
        let next = update_from_evidence(&profile, evidence, 1000);
        assert!(
            next.max_hr <= profile.max_hr + MAX_RAISE_STEP_BPM + 1e-9,
            "raise must be clamped: {} vs prior {}",
            next.max_hr,
            profile.max_hr
        );
    }

    /// CORRECTED (conceptual validation pass - was `repeated_agreeing_sessions_reach_high_confidence`,
    /// which asserted `profile.confidence == High` here; that was exactly the conflation being
    /// fixed). Repeated, ordinary hard-but-submaximal sessions at the same value are real, strong
    /// evidence of a LOWER BOUND - `observed_lower_bound_confidence` legitimately reaches `High`.
    /// They are NOT evidence that the value IS the physiological ceiling - the point estimate's own
    /// `confidence` must stay capped at `Medium` for `FieldInferred` regardless.
    #[test]
    fn repeated_agreeing_sessions_reach_high_lower_bound_confidence_but_estimate_confidence_stays_medium() {
        let mut profile = MaxHrProfile::cold_start(30.0);
        let mut t = 0;
        for _ in 0..4 {
            let evidence = MaxHrEvidence { candidate_bpm: 194.0, observed_at: t, quality: 0.9 };
            profile = update_from_evidence(&profile, evidence, t);
            t += 7 * 86_400; // weekly sessions
        }
        assert_eq!(
            observed_lower_bound_confidence(&profile, t),
            Confidence::High,
            "four corroborating sessions should give high confidence the user reached ~194bpm"
        );
        assert_eq!(
            profile.confidence,
            Confidence::Medium,
            "the SAME evidence must not claim high confidence that 194bpm is the true ceiling"
        );
        assert_eq!(profile.source, MaxHrSource::FieldInferred);
    }

    #[test]
    fn a_single_session_gives_low_or_medium_lower_bound_confidence_not_high() {
        let profile = MaxHrProfile::cold_start(30.0);
        let evidence = MaxHrEvidence { candidate_bpm: 192.0, observed_at: 1000, quality: 0.9 };
        let next = update_from_evidence(&profile, evidence, 1000);
        assert_ne!(
            observed_lower_bound_confidence(&next, 1000),
            Confidence::High,
            "one session alone should not yet be high lower-bound confidence"
        );
    }

    #[test]
    fn evidence_alone_never_lowers_max_hr() {
        let mut profile = MaxHrProfile::cold_start(30.0);
        profile = update_from_evidence(
            &profile,
            MaxHrEvidence { candidate_bpm: 195.0, observed_at: 0, quality: 1.0 },
            0,
        );
        let raised = profile.max_hr;
        // A later, lower-quality/lower-value session must not pull max_hr back down.
        let lower = update_from_evidence(
            &profile,
            MaxHrEvidence { candidate_bpm: 175.0, observed_at: 100, quality: 0.9 },
            100,
        );
        assert!(lower.max_hr >= raised, "max_hr must never decrease from evidence alone");
    }

    #[test]
    fn staleness_downgrades_confidence_but_not_the_value() {
        let mut profile = MaxHrProfile::cold_start(30.0);
        profile = update_from_evidence(
            &profile,
            MaxHrEvidence { candidate_bpm: 194.0, observed_at: 0, quality: 1.0 },
            0,
        );
        assert_eq!(profile.confidence, Confidence::Medium);
        let stale = apply_staleness(&profile, STALE_AFTER_DAYS + 1);
        assert_eq!(stale.confidence, Confidence::Low);
        assert_eq!(stale.max_hr, profile.max_hr, "staleness must not change the value");
    }

    #[test]
    fn user_override_wins_and_flags_hybrid_on_strong_disagreement() {
        let mut profile = MaxHrProfile::cold_start(30.0);
        profile.observed_peak = Some(205.0);
        let overridden = apply_user_override(&profile, 180.0, 0);
        assert_eq!(overridden.max_hr, 180.0);
        assert_eq!(overridden.source, MaxHrSource::Hybrid);
    }

    #[test]
    fn user_override_without_disagreement_is_plain_user_supplied() {
        let profile = MaxHrProfile::cold_start(30.0);
        let overridden = apply_user_override(&profile, 188.0, 0);
        assert_eq!(overridden.source, MaxHrSource::UserSupplied);
    }

    #[test]
    fn candidate_session_spans_finds_one_contiguous_hard_run() {
        // RHR 52, MaxHR 190 -> HRR 138. 60% HRR = 52 + 0.60*138 = 135.8bpm.
        let mut samples = Vec::new();
        for i in 0..200 {
            samples.push(HrSample { ts: i, bpm: 100 }); // light, below effort floor
        }
        for i in 200..320 {
            samples.push(HrSample { ts: i, bpm: 150 }); // hard span, 120s
        }
        for i in 320..500 {
            samples.push(HrSample { ts: i, bpm: 100 });
        }
        let spans = candidate_session_spans(&samples, 190.0, 52.0, 90);
        assert_eq!(spans, vec![(200, 319)]);
    }

    #[test]
    fn candidate_session_spans_drops_runs_shorter_than_the_minimum() {
        let mut samples: Vec<HrSample> = (0..200).map(|i| HrSample { ts: i, bpm: 100 }).collect();
        samples.extend((200..230).map(|i| HrSample { ts: i, bpm: 150 })); // only 30s, hard
        let spans = candidate_session_spans(&samples, 190.0, 52.0, 90);
        assert!(spans.is_empty());
    }

    #[test]
    fn age_decay_only_applies_before_field_evidence() {
        let profile = MaxHrProfile::cold_start(30.0);
        let decayed = apply_age_decay(&profile, 31.0);
        assert!((decayed.max_hr - gellish_hrmax(31.0)).abs() < 1e-9);

        let mut evidenced = profile.clone();
        evidenced = update_from_evidence(
            &evidenced,
            MaxHrEvidence { candidate_bpm: 195.0, observed_at: 0, quality: 1.0 },
            0,
        );
        let unaffected = apply_age_decay(&evidenced, 31.0);
        assert_eq!(unaffected.max_hr, evidenced.max_hr, "age decay must not touch an evidenced profile");
    }
}
