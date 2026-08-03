//! The headline daily scores: recovery, strain, stress, rest, and the heart-rate profile
//! (resting HR, zones, VO2 max, fitness age) they read from.

use crate::*;

/// One heart-rate tick (unix seconds + bpm) shared by strain, recovery, resting-HR and zones.
#[derive(uniffi::Record)]
pub struct HrTick {
    pub ts: i64,
    pub bpm: i32,
}

/// The one `HrTick` -> algo-crate `HrSample` conversion; every plain-value metric takes this series.
pub(crate) fn to_hr(ticks: Vec<HrTick>) -> Vec<physio_algo::HrSample> {
    ticks.into_iter().map(|h| physio_algo::HrSample { ts: h.ts, bpm: h.bpm }).collect()
}

/// A personal baseline driver (mean + spread) for recovery z-scoring.
#[derive(uniffi::Record)]
pub struct DriverBaselineInfo {
    pub mean: f64,
    pub spread: f64,
}

impl From<DriverBaselineInfo> for recovery::DriverBaseline {
    fn from(b: DriverBaselineInfo) -> Self {
        recovery::DriverBaseline { mean: b.mean, spread: b.spread }
    }
}

/// Nightly recovery drivers. Optional terms drop and the weights renormalise.
#[derive(uniffi::Record)]
pub struct RecoveryDrivers {
    pub hrv: f64,
    pub rhr: f64,
    pub resp: Option<f64>,
    pub hrv_baseline: Option<DriverBaselineInfo>,
    pub rhr_baseline: Option<DriverBaselineInfo>,
    pub resp_baseline: Option<DriverBaselineInfo>,
    pub sleep_perf: Option<f64>,
    pub skin_temp_dev: Option<f64>,
    pub hrv_baseline_usable: bool,
    pub recovery_index_slope: Option<f64>,
    pub effort_baseline: Option<DriverBaselineInfo>,
    pub prior_day_effort: Option<f64>,
}

impl From<RecoveryDrivers> for recovery::RecoveryInput {
    fn from(d: RecoveryDrivers) -> Self {
        recovery::RecoveryInput {
            hrv: d.hrv,
            rhr: d.rhr,
            resp: d.resp,
            hrv_baseline: d.hrv_baseline.map(Into::into),
            rhr_baseline: d.rhr_baseline.map(Into::into),
            resp_baseline: d.resp_baseline.map(Into::into),
            sleep_perf: d.sleep_perf,
            skin_temp_dev: d.skin_temp_dev,
            hrv_baseline_usable: d.hrv_baseline_usable,
            recovery_index_slope: d.recovery_index_slope,
            effort_baseline: d.effort_baseline.map(Into::into),
            prior_day_effort: d.prior_day_effort,
        }
    }
}

/// Recovery "Charge" score in [0, 100]. `None` at cold-start or when no driver is available.
#[uniffi::export]
pub fn recovery_score(d: RecoveryDrivers) -> Option<f64> {
    recovery::recovery(&d.into())
}

/// Recovery colour band ("red" | "yellow" | "green") for a score.
#[uniffi::export]
pub fn recovery_band(score: f64) -> String {
    recovery::band(score).to_string()
}

/// The one-word state band a recovery score falls in. The word and its colour are the caller's.
#[derive(uniffi::Enum)]
pub enum RecoveryState {
    Depleted,
    Low,
    Moderate,
    Primed,
    Peak,
}

impl From<recovery::RecoveryState> for RecoveryState {
    fn from(s: recovery::RecoveryState) -> Self {
        match s {
            recovery::RecoveryState::Depleted => RecoveryState::Depleted,
            recovery::RecoveryState::Low => RecoveryState::Low,
            recovery::RecoveryState::Moderate => RecoveryState::Moderate,
            recovery::RecoveryState::Primed => RecoveryState::Primed,
            recovery::RecoveryState::Peak => RecoveryState::Peak,
        }
    }
}

/// The state band of a recovery score in [0, 100] — finer than [`recovery_band`]'s three colours.
#[uniffi::export]
pub fn recovery_state(score: f64) -> RecoveryState {
    recovery::state(score).into()
}

/// Which signal a driver row describes; the caller owns its label, unit and value text.
#[derive(uniffi::Enum)]
pub enum DriverKind {
    Hrv,
    RestingHr,
    Sleep,
    Respiratory,
    SkinTemp,
    RecoveryIndex,
    ActivityBalance,
}

impl From<recovery_drivers::DriverKind> for DriverKind {
    fn from(k: recovery_drivers::DriverKind) -> Self {
        match k {
            recovery_drivers::DriverKind::Hrv => DriverKind::Hrv,
            recovery_drivers::DriverKind::RestingHr => DriverKind::RestingHr,
            recovery_drivers::DriverKind::Sleep => DriverKind::Sleep,
            recovery_drivers::DriverKind::Respiratory => DriverKind::Respiratory,
            recovery_drivers::DriverKind::SkinTemp => DriverKind::SkinTemp,
            recovery_drivers::DriverKind::RecoveryIndex => DriverKind::RecoveryIndex,
            recovery_drivers::DriverKind::ActivityBalance => DriverKind::ActivityBalance,
        }
    }
}

/// How a driver reads against its baseline. `LimitingHigh` / `LimitingLow` carry the side for the
/// symmetric skin-temp term; every single-sided driver yields only the first three.
#[derive(uniffi::Enum)]
pub enum DriverVerdict {
    Supporting,
    Neutral,
    Limiting,
    LimitingHigh,
    LimitingLow,
}

impl From<recovery_drivers::DriverVerdict> for DriverVerdict {
    fn from(v: recovery_drivers::DriverVerdict) -> Self {
        match v {
            recovery_drivers::DriverVerdict::Supporting => DriverVerdict::Supporting,
            recovery_drivers::DriverVerdict::Neutral => DriverVerdict::Neutral,
            recovery_drivers::DriverVerdict::Limiting => DriverVerdict::Limiting,
            recovery_drivers::DriverVerdict::LimitingHigh => DriverVerdict::LimitingHigh,
            recovery_drivers::DriverVerdict::LimitingLow => DriverVerdict::LimitingLow,
        }
    }
}

/// One driver row behind a Charge score: the signal, its marginal swing in whole points, and its
/// direction. `delta_points` is NaN when that driver's own value is.
#[derive(uniffi::Record)]
pub struct DriverRow {
    pub kind: DriverKind,
    pub delta_points: f64,
    pub verdict: DriverVerdict,
}

/// The per-driver breakdown behind [`recovery_score`], from the identical input: biggest mover first,
/// ties in emission order, empty exactly where the score returns `None`.
#[uniffi::export]
pub fn recovery_driver_rows(d: RecoveryDrivers) -> Vec<DriverRow> {
    recovery_drivers::driver_rows(&d.into())
        .into_iter()
        .map(|r| DriverRow {
            kind: r.kind.into(),
            delta_points: r.delta_points,
            verdict: r.verdict.into(),
        })
        .collect()
}

/// Overnight HR-decline slope (bpm/hour) — the recovery-index driver.
#[uniffi::export]
pub fn recovery_index_slope(hr: Vec<HrTick>, start: i64, end: i64) -> Option<f64> {
    let s = to_hr(hr);
    recovery::recovery_index_slope(&s, start, end)
}

/// Count of nights carrying a usable nightly HRV — the calibration-progress count.
#[uniffi::export]
pub fn recovery_banked_nights(nightly_hrv: Vec<Option<f64>>) -> u32 {
    recovery::banked_nights(&nightly_hrv, recovery::HRV_MIN_MS, recovery::HRV_MAX_MS) as u32
}

/// TRIMP accumulation method for strain.
#[derive(uniffi::Enum)]
pub enum StrainMethod {
    Edwards,
    Banister,
}

impl From<StrainMethod> for strain::Method {
    fn from(m: StrainMethod) -> Self {
        match m {
            StrainMethod::Edwards => strain::Method::Edwards,
            StrainMethod::Banister => strain::Method::Banister,
        }
    }
}

/// Cardiovascular Effort (0–100) from an HR series. `None` without enough data or when HRR ≤ 0.
#[uniffi::export]
pub fn strain_score(
    hr: Vec<HrTick>,
    max_hr: Option<f64>,
    resting_hr: f64,
    method: StrainMethod,
    sex: String,
    denominator: f64,
) -> Option<f64> {
    let s = to_hr(hr);
    strain::strain(&s, max_hr, resting_hr, method.into(), &sex, denominator)
}

/// The default strain denominator (log-map scale onto 0–100).
#[uniffi::export]
pub fn strain_default_denominator() -> f64 {
    strain::STRAIN_DENOMINATOR
}

/// Baevsky Stress Index histogram terms behind an SI.
#[derive(uniffi::Record)]
pub struct StressComponentsInfo {
    pub mo_sec: f64,
    pub amo_percent: f64,
    pub mxdmn_sec: f64,
    pub si: f64,
}

/// Baevsky Stress Index from a raw R-R series (ms). `None` on too-few beats or a degenerate range.
#[uniffi::export]
pub fn stress_index(rr_ms: Vec<f64>) -> Option<f64> {
    stress::stress_index_raw(&rr_ms)
}

/// Full SI components from a raw R-R series (ms).
#[uniffi::export]
pub fn stress_components(rr_ms: Vec<f64>) -> Option<StressComponentsInfo> {
    stress::components_raw(&rr_ms).map(|c| StressComponentsInfo {
        mo_sec: c.mo_sec,
        amo_percent: c.amo_percent,
        mxdmn_sec: c.mxdmn_sec,
        si: c.si,
    })
}

/// Lowest 5-min tumbling-window mean bpm floor over `[start, end]`. `None` with no samples.
#[uniffi::export]
pub fn session_resting_hr(start: i64, end: i64, hr: Vec<HrTick>) -> Option<i32> {
    let s = to_hr(hr);
    resting_hr::session_resting_hr(start, end, &s)
}

/// HR recovery: bpm drop 1/2/5 min after a sustained high-intensity bout. `None` when ineligible or
/// under-sampled; a HR rise stays signed.
#[derive(uniffi::Record)]
pub struct HrRecoveryInfo {
    pub end_hr: i32,
    pub after_1min: Option<i32>,
    pub after_2min: Option<i32>,
    pub after_5min: Option<i32>,
}

#[uniffi::export]
pub fn hr_recovery_calculate(hr: Vec<HrTick>, workout_start: i64, workout_end: i64, max_hr: f64) -> Option<HrRecoveryInfo> {
    let s = to_hr(hr);
    hr_recovery::calculate(&s, workout_start, workout_end, max_hr).map(|r| HrRecoveryInfo {
        end_hr: r.end_hr,
        after_1min: r.after_1min,
        after_2min: r.after_2min,
        after_5min: r.after_5min,
    })
}

/// Daily resting HR = min of the per-session floors.
#[uniffi::export]
pub fn daily_resting_hr(session_floors: Vec<Option<i32>>) -> Option<i32> {
    resting_hr::daily_resting_hr(&session_floors)
}

/// A single HR zone as a bpm interval `[lower, upper)` plus its %HRmax band.
#[derive(uniffi::Record)]
pub struct HrZoneInfo {
    pub number: u8,
    pub lower: f64,
    pub upper: f64,
    pub lower_pct: f64,
    pub upper_pct: f64,
}

/// Five HR zones, the max HR they were built from, and its source ("tanaka" | "manual").
#[derive(uniffi::Record)]
pub struct HrZoneSetInfo {
    pub zones: Vec<HrZoneInfo>,
    pub max_hr: f64,
    pub source: String,
}

/// Seconds in each of the five zones (index 0 == Zone 1) plus time below Zone 1.
#[derive(uniffi::Record)]
pub struct TimeInZoneInfo {
    pub seconds: Vec<f64>,
    pub below_zone1: f64,
}

pub(crate) fn zone_set_to_ffi(z: hr_zones::HrZoneSet) -> HrZoneSetInfo {
    HrZoneSetInfo {
        zones: z
            .zones
            .iter()
            .map(|zn| HrZoneInfo {
                number: zn.number,
                lower: zn.lower,
                upper: zn.upper,
                lower_pct: zn.lower_pct,
                upper_pct: zn.upper_pct,
            })
            .collect(),
        max_hr: z.max_hr,
        source: z.source,
    }
}

/// Age-derived (Tanaka) HR zones, or a manual max-HR override.
#[uniffi::export]
pub fn hr_zones_for_age(age: f64, max_hr_override: Option<f64>) -> HrZoneSetInfo {
    zone_set_to_ffi(hr_zones::zones_for_age(age, max_hr_override))
}

/// Seconds spent in each HR zone over an HR series, using age-derived (or override) zones.
#[uniffi::export]
pub fn hr_time_in_zone(hr: Vec<HrTick>, age: f64, max_hr_override: Option<f64>) -> TimeInZoneInfo {
    let zs = hr_zones::zones_for_age(age, max_hr_override);
    let s = to_hr(hr);
    let t = hr_zones::time_in_zone(&s, &zs);
    TimeInZoneInfo { seconds: t.seconds.to_vec(), below_zone1: t.below_zone1 }
}

/// A computed Fitness Age with the inputs to present it. `vo2max` is filled only with a waist.
/// `advance_years` is POSITIVE when older than chronological, matching `rhythm_age`'s convention.
#[derive(uniffi::Record)]
pub struct FitnessAgeInfo {
    pub vo2max: Option<f64>,
    pub fitness_age: f64,
    pub chrono_age: f64,
    pub advance_years: f64,
    pub band_years: f64,
    pub lower_confidence: bool,
}

/// Non-exercise VO2max estimate (ml/kg/min) from the waist-circumference model. Wellness only.
#[uniffi::export]
pub fn vo2max_estimate(age: f64, sex: String, waist_cm: f64, resting_hr: f64, pa_index: f64) -> f64 {
    vo2max::estimate_vo2max(age, &sex, waist_cm, resting_hr, pa_index)
}

/// Full Fitness Age. `None` only if RHR or age is missing.
#[uniffi::export]
pub fn fitness_age_compute(
    age: f64,
    sex: String,
    resting_hr: f64,
    pa_index: f64,
    waist_cm: Option<f64>,
    lower_confidence: bool,
) -> Option<FitnessAgeInfo> {
    vo2max::compute(age, &sex, resting_hr, pa_index, waist_cm, lower_confidence).map(|r| FitnessAgeInfo {
        vo2max: r.vo2max,
        fitness_age: r.fitness_age,
        chrono_age: r.chrono_age,
        advance_years: r.advance_years,
        band_years: r.band_years,
        lower_confidence: r.lower_confidence,
    })
}

/// Rest (sleep performance) composite [0, 100] from a night's aggregates. `None` when there is no asleep
/// time. Absent `sleep_need_hours` defaults to 8 h; absent `consistency` defaults to a neutral 0.5.
#[uniffi::export]
pub fn rest_score(
    asleep_seconds: f64,
    efficiency: f64,
    deep_seconds: f64,
    rem_seconds: f64,
    sleep_need_hours: Option<f64>,
    consistency: Option<f64>,
) -> Option<f64> {
    rest::rest(asleep_seconds, efficiency, deep_seconds, rem_seconds, sleep_need_hours, consistency)
}

// ── Sleep debt ledger ──────────────────────────────────────────────────────
