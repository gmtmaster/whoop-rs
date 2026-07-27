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

// ── Steps counter ──────────────────────────────────────────────────────────
