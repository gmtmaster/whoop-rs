//! Per-metric rolling baselines: configuration, one update, and folding a history in.

use crate::*;

/// One metric's baseline configuration: the validity band, the spread floor and the two EWMA
/// half-lives (centre, spread).
#[derive(uniffi::Record)]
pub struct MetricCfgInfo {
    pub min_val: f64,
    pub max_val: f64,
    pub floor_spread: f64,
    pub half_life_b: f64,
    pub half_life_s: f64,
}

/// Persisted baseline state for one metric.
#[derive(uniffi::Record)]
pub struct BaselineStateInfo {
    pub baseline: f64,
    pub spread: f64,
    pub n_valid: i32,
    pub nights_since_update: i32,
    pub status: String,
}

/// One metric's baseline configuration by name ("hrv" / "resting_hr" / "resp" / "skin_temp" /
/// "strain"). The tuning table lives here so the app cannot hold a second copy that drifts from it;
/// an unknown name yields `None`.
#[uniffi::export]
pub fn baseline_metric_cfg(metric: String) -> Option<MetricCfgInfo> {
    let c = match metric.as_str() {
        "hrv" => baselines::MetricCfg::hrv(),
        "resting_hr" => baselines::MetricCfg::resting_hr(),
        "resp" => baselines::MetricCfg::resp(),
        "skin_temp" => baselines::MetricCfg::skin_temp(),
        "strain" => baselines::MetricCfg::strain(),
        _ => return None,
    };
    Some(MetricCfgInfo { min_val: c.min_val, max_val: c.max_val, floor_spread: c.floor_spread,
        half_life_b: c.half_life_b, half_life_s: c.half_life_s })
}

#[uniffi::export]
pub fn baseline_update(state: Option<BaselineStateInfo>, value: Option<f64>, cfg: MetricCfgInfo) -> BaselineStateInfo {
    let s = state.map(|s| baselines::BaselineState {
        baseline: s.baseline, spread: s.spread, n_valid: s.n_valid,
        nights_since_update: s.nights_since_update,
        status: baselines::BaselineStatus::parse(&s.status),
    });
    let c = baselines::MetricCfg { min_val: cfg.min_val, max_val: cfg.max_val,
        floor_spread: cfg.floor_spread, half_life_b: cfg.half_life_b, half_life_s: cfg.half_life_s };
    let r = baselines::update(s, value, &c);
    BaselineStateInfo { baseline: r.baseline, spread: r.spread, n_valid: r.n_valid,
        nights_since_update: r.nights_since_update, status: r.status.as_str().to_string() }
}

#[uniffi::export]
pub fn baseline_fold_history(values: Vec<Option<f64>>, cfg: MetricCfgInfo) -> BaselineStateInfo {
    let c = baselines::MetricCfg { min_val: cfg.min_val, max_val: cfg.max_val,
        floor_spread: cfg.floor_spread, half_life_b: cfg.half_life_b, half_life_s: cfg.half_life_s };
    let r = baselines::fold_history(&values, &c);
    BaselineStateInfo { baseline: r.baseline, spread: r.spread, n_valid: r.n_valid,
        nights_since_update: r.nights_since_update, status: r.status.as_str().to_string() }
}

// ── Vital banding ──────────────────────────────────────────────────────────

/// An inclusive typical-adult window.
#[derive(uniffi::Record)]
pub struct TypicalRangeInfo {
    pub min: f64,
    pub max: f64,
}

/// One banded vital: `band` ∈ inRange/outOfRange/noData, `basis` ∈ personal/population, and the
/// valid-night count the basis was decided on.
#[derive(uniffi::Record)]
pub struct VitalBandInfo {
    pub band: String,
    pub basis: String,
    pub nights: i32,
}

/// The typical-adult window for a vital key ("resp" / "spo2" / "rhr" / "hrv" / "skin_abs" /
/// "skin_dev"); `None` for an unknown key.
#[uniffi::export]
pub fn vital_typical_range(vital: String) -> Option<TypicalRangeInfo> {
    vital_bands::typical_range(&vital).map(|r| TypicalRangeInfo { min: r.min, max: r.max })
}

/// Band one vital against the wearer's own baseline once trusted, else the typical window.
/// `history` is nightly values oldest first excluding the displayed day; a `None` `cfg` leaves the
/// window as the only yardstick.
#[uniffi::export]
pub fn vital_band(
    value: Option<f64>,
    history: Vec<Option<f64>>,
    population_range: TypicalRangeInfo,
    cfg: Option<MetricCfgInfo>,
) -> VitalBandInfo {
    let c = cfg.map(|c| baselines::MetricCfg { min_val: c.min_val, max_val: c.max_val,
        floor_spread: c.floor_spread, half_life_b: c.half_life_b, half_life_s: c.half_life_s });
    let r = vital_bands::band(
        value,
        &history,
        vital_bands::TypicalRange::new(population_range.min, population_range.max),
        c.as_ref(),
    );
    VitalBandInfo { band: r.band.as_str().to_string(), basis: r.basis.as_str().to_string(), nights: r.nights }
}

/// Whether a skin-temp reading is an absolute wrist temperature rather than a ±°C deviation.
#[uniffi::export]
pub fn skin_temp_is_absolute(value: f64) -> bool {
    vital_bands::is_absolute_skin_temp(value)
}

/// Blank every history entry of the other skin-temp kind, so a baseline is never folded across the
/// absolute and deviation scales at once.
#[uniffi::export]
pub fn skin_temp_history(value: f64, history: Vec<Option<f64>>) -> Vec<Option<f64>> {
    vital_bands::skin_temp_history(value, &history)
}

// ── Steps counter ──────────────────────────────────────────────────────────
