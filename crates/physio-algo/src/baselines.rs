//! Personal baselines — winsorized EWMA center + absolute-deviation spread per metric.
//! One `update` per nightly value; cold-start / young / steady-state / stale lifecycle.

pub const MIN_NIGHTS_SEED: i32 = 4;
pub const MIN_NIGHTS_TRUST: i32 = 14;
const STALE_DAYS: i32 = 14;
pub const EARLY_ADAPT_NIGHTS: i32 = 8;
const EARLY_HALF_LIFE_B: f64 = 3.0;
const HARD_OUTLIER_K: f64 = 5.0;
const WINSOR_K: f64 = 3.0;
const EARLY_SPREAD_INFLATE: f64 = 2.5;

fn lambda(half_life: f64) -> f64 { 1.0 - 0.5f64.powf(1.0 / half_life) }

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BaselineStatus { Calibrating, Provisional, Trusted, Stale }

impl BaselineStatus {
    pub fn as_str(self) -> &'static str {
        match self { Self::Calibrating => "calibrating", Self::Provisional => "provisional",
            Self::Trusted => "trusted", Self::Stale => "stale" }
    }
    pub fn parse(s: &str) -> Self {
        match s { "provisional" => Self::Provisional, "trusted" => Self::Trusted,
            "stale" => Self::Stale, _ => Self::Calibrating }
    }
}

fn compute_status(n_valid: i32, nights_since_update: i32) -> BaselineStatus {
    if nights_since_update > STALE_DAYS && n_valid >= MIN_NIGHTS_SEED { BaselineStatus::Stale }
    else if n_valid < MIN_NIGHTS_SEED { BaselineStatus::Calibrating }
    else if n_valid < MIN_NIGHTS_TRUST { BaselineStatus::Provisional }
    else { BaselineStatus::Trusted }
}

/// One metric's configuration.
#[derive(Debug, Clone, Copy)]
pub struct MetricCfg {
    pub min_val: f64, pub max_val: f64, pub floor_spread: f64,
    pub half_life_b: f64, pub half_life_s: f64,
}

impl MetricCfg {
    pub fn hrv() -> Self { Self { min_val: 5.0, max_val: 250.0, floor_spread: 5.0, half_life_b: 14.0, half_life_s: 21.0 } }
    pub fn resting_hr() -> Self { Self { min_val: 30.0, max_val: 120.0, floor_spread: 2.0, half_life_b: 14.0, half_life_s: 21.0 } }
    pub fn resp() -> Self { Self { min_val: 4.0, max_val: 40.0, floor_spread: 0.5, half_life_b: 14.0, half_life_s: 21.0 } }
    pub fn skin_temp() -> Self { Self { min_val: 20.0, max_val: 42.0, floor_spread: 0.3, half_life_b: 14.0, half_life_s: 21.0 } }
    pub fn strain() -> Self { Self { min_val: 0.0, max_val: 100.0, floor_spread: 5.0, half_life_b: 14.0, half_life_s: 21.0 } }
}

/// Persisted state for one metric's baseline.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BaselineState {
    pub baseline: f64,
    pub spread: f64,
    pub n_valid: i32,
    pub nights_since_update: i32,
    pub status: BaselineStatus,
}

impl BaselineState {
    pub fn usable(&self) -> bool { self.n_valid >= MIN_NIGHTS_SEED }
    pub fn trusted(&self) -> bool { self.n_valid >= MIN_NIGHTS_TRUST }
}

/// Incorporate one new nightly value. `None` state = first night.
pub fn update(state: Option<BaselineState>, value: Option<f64>, cfg: &MetricCfg) -> BaselineState {
    let lb = lambda(cfg.half_life_b);
    let ls = lambda(cfg.half_life_s);

    let Some(s) = state else {
        // First night ever.
        if let Some(v) = value {
            if v >= cfg.min_val && v <= cfg.max_val {
                return BaselineState { baseline: v, spread: cfg.floor_spread, n_valid: 1,
                    nights_since_update: 0, status: BaselineStatus::Calibrating };
            }
        }
        let seed = (cfg.min_val + cfg.max_val) / 2.0;
        return BaselineState { baseline: seed, spread: cfg.floor_spread, n_valid: 0,
            nights_since_update: 1, status: BaselineStatus::Calibrating };
    };

    // Missing or out-of-range → skip-and-hold.
    let Some(v) = value else {
        let m = s.nights_since_update + 1;
        return BaselineState { baseline: s.baseline, spread: s.spread, n_valid: s.n_valid,
            nights_since_update: m, status: compute_status(s.n_valid, m) };
    };
    if !(cfg.min_val <= v && v <= cfg.max_val) {
        let m = s.nights_since_update + 1;
        return BaselineState { baseline: s.baseline, spread: s.spread, n_valid: s.n_valid,
            nights_since_update: m, status: compute_status(s.n_valid, m) };
    }

    let is_young = s.n_valid < EARLY_ADAPT_NIGHTS;

    // Hard outlier: seen but not folded (only after seeded + settled).
    if s.n_valid >= MIN_NIGHTS_SEED && !is_young {
        let dev = (v - s.baseline).abs();
        if dev > HARD_OUTLIER_K * s.spread {
            return BaselineState { baseline: s.baseline, spread: s.spread, n_valid: s.n_valid,
                nights_since_update: 0, status: compute_status(s.n_valid, 0) };
        }
    }

    // First real value after a None-placeholder seed.
    if s.n_valid == 0 {
        return BaselineState { baseline: v, spread: cfg.floor_spread, n_valid: 1,
            nights_since_update: 0, status: BaselineStatus::Calibrating };
    }

    let eff_spread = if is_young { s.spread * EARLY_SPREAD_INFLATE } else { s.spread };
    let eff_lb = if is_young { lambda(EARLY_HALF_LIFE_B) } else { lb };
    let lo = s.baseline - WINSOR_K * eff_spread;
    let hi = s.baseline + WINSOR_K * eff_spread;
    let clamped = v.clamp(lo, hi);
    let new_center = eff_lb * clamped + (1.0 - eff_lb) * s.baseline;
    let dev = (v - new_center).abs();
    let new_spread = (ls * dev + (1.0 - ls) * s.spread).max(cfg.floor_spread);
    let n = s.n_valid + 1;
    BaselineState { baseline: new_center, spread: new_spread, n_valid: n,
        nights_since_update: 0, status: compute_status(n, 0) }
}

/// Build state from a chronological series of nightly values (oldest first).
pub fn fold_history(values: &[Option<f64>], cfg: &MetricCfg) -> BaselineState {
    let mut state: Option<BaselineState> = None;
    for v in values { state = Some(update(state, *v, cfg)); }
    state.unwrap_or_else(|| BaselineState {
        baseline: (cfg.min_val + cfg.max_val) / 2.0, spread: cfg.floor_spread,
        n_valid: 0, nights_since_update: 0, status: BaselineStatus::Calibrating,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cold_start_seeds_first_valid_value() {
        let s = update(None, Some(30.0), &MetricCfg::hrv());
        assert_eq!(s.baseline, 30.0);
        assert_eq!(s.n_valid, 1);
    }

    #[test]
    fn null_skip_and_hold() {
        let s = update(Some(BaselineState { baseline: 30.0, spread: 5.0, n_valid: 5,
            nights_since_update: 0, status: BaselineStatus::Provisional }), None, &MetricCfg::hrv());
        assert_eq!(s.baseline, 30.0);
        assert_eq!(s.n_valid, 5);
        assert_eq!(s.nights_since_update, 1);
    }

    #[test]
    fn out_of_range_skip() {
        let s = update(Some(BaselineState { baseline: 50.0, spread: 5.0, n_valid: 5,
            nights_since_update: 0, status: BaselineStatus::Provisional }),
            Some(300.0), &MetricCfg::hrv()); // HRV max=250
        assert_eq!(s.baseline, 50.0); // unchanged
    }

    #[test]
    fn normal_update_converges() {
        let cfg = MetricCfg::hrv();
        let mut s = update(None, Some(50.0), &cfg);
        for _ in 0..20 { s = update(Some(s), Some(45.0), &cfg); }
        assert!(s.baseline < 50.0, "should drift toward 45, got {}", s.baseline);
        assert!(s.n_valid >= 14, "should be trusted after 14+ nights");
        assert_eq!(s.status, BaselineStatus::Trusted);
    }
}
