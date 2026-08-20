//! Personal baselines — winsorized EWMA center + absolute-deviation spread per metric.
//! One `update` per nightly value; cold-start / young / steady-state / stale lifecycle.
//! Night validity is a conjunction across channels: an implausible night is missing for every metric.

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

// ── Deviation from baseline ──────────────────────────────────────────────────

/// Z-score scale: converts the mean-absolute-deviation `spread` this module tracks into an
/// approximate standard deviation (1 / sqrt(2/pi) ≈ 1.253), matching the agreed Swift semantics.
const Z_SPREAD_SCALE: f64 = 1.253;

/// One value's deviation from a `BaselineState`, as delta, z-score and ratio.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Deviation {
    /// `value - baseline`, in the metric's own units.
    pub delta: f64,
    /// `(value - baseline) / (1.253 * spread)`.
    pub z: f64,
    /// `value / baseline - 1`.
    pub ratio: f64,
}

/// A value's deviation from an established baseline: `delta = value - baseline`,
/// `z = (value - baseline) / (1.253 * spread)`, `ratio = value / baseline - 1`.
pub fn deviation(value: f64, state: &BaselineState) -> Deviation {
    let delta = value - state.baseline;
    let z = delta / (Z_SPREAD_SCALE * state.spread);
    let ratio = value / state.baseline - 1.0;
    Deviation { delta, z, ratio }
}

// ── Night validity as a conjunction across channels ─────────────────────────

/// The nightly metrics a night's validity is decided for. Daily Effort/strain is a daytime metric
/// and is deliberately absent: a short night must not invalidate the day's effort.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NightMetric { Hrv, RestingHr, Resp, SkinTemp }

impl NightMetric {
    pub fn cfg(self) -> MetricCfg {
        match self { Self::Hrv => MetricCfg::hrv(), Self::RestingHr => MetricCfg::resting_hr(),
            Self::Resp => MetricCfg::resp(), Self::SkinTemp => MetricCfg::skin_temp() }
    }
    fn value(self, n: &NightChannels) -> Option<f64> {
        match self { Self::Hrv => n.hrv_ms, Self::RestingHr => n.resting_hr_bpm,
            Self::Resp => n.resp_bpm, Self::SkinTemp => n.skin_temp_c }
    }
}

/// One night's channels as the validity conjunction sees them. `None` = not measured and never
/// rejects; only a channel that is present and implausible does.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct NightChannels {
    pub hrv_ms: Option<f64>,
    pub resting_hr_bpm: Option<f64>,
    pub resp_bpm: Option<f64>,
    pub skin_temp_c: Option<f64>,
    pub skin_temp_max_c: Option<f64>,
    pub total_sleep_secs: Option<f64>,
    pub quality: Option<f64>,
}

/// Cross-channel gate for one night. Carries only what `MetricCfg` cannot express: a slept-long-
/// enough duration, the worn-skin band, and an optional quality floor.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NightGate {
    pub min_sleep_secs: f64,
    pub worn_skin_temp_c: (f64, f64),
    pub min_quality: Option<f64>,
}

/// 4 h of sleep: below it a night's resting HR and HRV are a nap statistic, not a night one.
pub const MIN_SLEEP_SECS: f64 = 4.0 * 3600.0;
/// Worn-skin plausibility band in absolute degC — tighter than `MetricCfg::skin_temp`, which is a
/// storage band and admits an off-wrist strap at room temperature.
pub const WORN_SKIN_TEMP_C: (f64, f64) = (28.0, 40.0);

impl Default for NightGate {
    fn default() -> Self {
        Self { min_sleep_secs: MIN_SLEEP_SECS, worn_skin_temp_c: WORN_SKIN_TEMP_C, min_quality: None }
    }
}

/// Why a night is not a baseline night. Rust decides the case; the app words it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NightVerdict {
    Valid,
    SleepTooShort,
    SkinTempImplausible,
    RestingHrImplausible,
    HrvImplausible,
    RespImplausible,
    LowQuality,
}

impl NightVerdict {
    pub fn valid(self) -> bool { self == Self::Valid }
    pub fn as_str(self) -> &'static str {
        match self { Self::Valid => "valid", Self::SleepTooShort => "sleep_too_short",
            Self::SkinTempImplausible => "skin_temp_implausible",
            Self::RestingHrImplausible => "resting_hr_implausible",
            Self::HrvImplausible => "hrv_implausible", Self::RespImplausible => "resp_implausible",
            Self::LowQuality => "low_quality" }
    }
}

fn out_of_band(v: Option<f64>, lo: f64, hi: f64) -> bool {
    v.is_some_and(|x| !(x.is_finite() && lo <= x && x <= hi))
}

/// A present value that is non-finite or under `floor`. NaN fails rather than passing silently.
fn under_floor(v: Option<f64>, floor: f64) -> bool {
    v.is_some_and(|x| !x.is_finite() || x < floor)
}

/// One night's verdict: the conjunction of every present channel. Checked in enum order, so a night
/// failing two channels always reports the same one.
pub fn night_verdict(n: &NightChannels, gate: &NightGate) -> NightVerdict {
    if under_floor(n.total_sleep_secs, gate.min_sleep_secs) { return NightVerdict::SleepTooShort; }
    let (lo, hi) = gate.worn_skin_temp_c;
    if out_of_band(n.skin_temp_c, lo, hi) || out_of_band(n.skin_temp_max_c, lo, hi) {
        return NightVerdict::SkinTempImplausible;
    }
    let rhr = MetricCfg::resting_hr();
    if out_of_band(n.resting_hr_bpm, rhr.min_val, rhr.max_val) {
        return NightVerdict::RestingHrImplausible;
    }
    let hrv = MetricCfg::hrv();
    if out_of_band(n.hrv_ms, hrv.min_val, hrv.max_val) { return NightVerdict::HrvImplausible; }
    let resp = MetricCfg::resp();
    if out_of_band(n.resp_bpm, resp.min_val, resp.max_val) { return NightVerdict::RespImplausible; }
    if gate.min_quality.is_some_and(|f| under_floor(n.quality, f)) { return NightVerdict::LowQuality; }
    NightVerdict::Valid
}

/// Per-night verdicts, oldest first, so the caller can report how many nights were rejected and why.
pub fn night_verdicts(nights: &[NightChannels], gate: &NightGate) -> Vec<NightVerdict> {
    nights.iter().map(|n| night_verdict(n, gate)).collect()
}

/// Fold a night series (oldest first) into one metric's baseline under the conjunction: a night
/// rejected on ANY channel enters as missing, so it skip-and-holds and staleness still advances.
pub fn fold_history_nights(nights: &[NightChannels], metric: NightMetric, gate: &NightGate) -> BaselineState {
    let cfg = metric.cfg();
    let values: Vec<Option<f64>> = nights.iter()
        .map(|n| if night_verdict(n, gate).valid() { metric.value(n) } else { None })
        .collect();
    fold_history(&values, &cfg)
}

/// One night's skin temperature reduced for the validity check: a MEDIAN centre (not a mean, so one
/// bad window cannot move it) and the per-night MAX, over samples at or above `min_conf`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NightSkinTemp {
    pub median_c: f64,
    pub max_c: f64,
    pub n_kept: usize,
    pub n_total: usize,
}

impl NightSkinTemp {
    /// Fraction of the night's samples that cleared the confidence floor; feeds `NightChannels.quality`.
    pub fn coverage(&self) -> f64 {
        if self.n_total == 0 { 0.0 } else { self.n_kept as f64 / self.n_total as f64 }
    }
}

/// Reduce `(degC, confidence 0..1)` samples to a night's centre and max. Samples below `min_conf`
/// and non-finite ones are dropped first; `None` when nothing survives.
pub fn night_skin_temp(samples: &[(f64, f64)], min_conf: f64) -> Option<NightSkinTemp> {
    let kept: Vec<f64> = samples.iter()
        .filter(|(v, c)| v.is_finite() && *c >= min_conf)
        .map(|(v, _)| *v)
        .collect();
    if kept.is_empty() { return None; }
    let max_c = kept.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    Some(NightSkinTemp { median_c: crate::stats::median(&kept), max_c,
        n_kept: kept.len(), n_total: samples.len() })
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

    /// A named do-nothing scorer over input `I`: the null arm every gate below has to reject.
    type Scorer<'a, I> = (&'a str, &'a dyn Fn(I) -> f64);

    /// A state already past the cold start, for the limbs a converging series never reaches.
    fn seeded(baseline: f64, spread: f64, n_valid: i32) -> BaselineState {
        BaselineState { baseline, spread, n_valid, nights_since_update: 0,
            status: compute_status(n_valid, 0) }
    }

    /// The exact EWMA centre a fold produces, at the young/steady handover and at the end. Both targets
    /// are closed forms of the half-lives, so changing either half-life fails here rather than silently
    /// rescaling every driver's `(value - baseline) / spread`.
    #[test]
    fn fold_reproduces_the_ewma_centre_a_z_score_is_measured_against() {
        let cfg = MetricCfg::hrv();
        let mut s = update(None, Some(50.0), &cfg);
        assert_eq!((s.baseline, s.spread, s.n_valid), (50.0, cfg.floor_spread, 1));
        // Seven young folds run at EARLY_HALF_LIFE_B, so the 5 ms gap decays by 2^(-7/3).
        for _ in 0..7 { s = update(Some(s), Some(45.0), &cfg); }
        let young = 45.0 + 5.0 * 0.5f64.powf(7.0 / EARLY_HALF_LIFE_B);
        assert!((s.baseline - young).abs() < 1e-12, "after 7 young folds got {}, want {young}", s.baseline);
        assert_eq!(s.n_valid, EARLY_ADAPT_NIGHTS);
        // Thirteen steady folds run at half_life_b, so the remaining gap decays by 2^(-13/14).
        for _ in 0..13 { s = update(Some(s), Some(45.0), &cfg); }
        let steady = 45.0 + (young - 45.0) * 0.5f64.powf(13.0 / cfg.half_life_b);
        assert!((s.baseline - steady).abs() < 1e-12, "after 20 folds got {}, want {steady}", s.baseline);
        assert_eq!((s.n_valid, s.status), (21, BaselineStatus::Trusted));
        // A one-sided approach keeps every residual under the floor, so this series leaves the spread
        // limb pinned at `floor_spread`; `winsorised_fold_lifts_the_spread_off_its_floor` reaches it.
        assert_eq!(s.spread, cfg.floor_spread);
    }

    /// Four do-nothing centres, each asserted to miss the shipped fold on at least one series. The
    /// previous gate was `baseline < 50.0`, which "return the last value" and "return the mean" satisfy.
    #[test]
    fn a_do_nothing_centre_fails_the_fold() {
        let cfg = MetricCfg::hrv();
        let converging: Vec<Option<f64>> =
            std::iter::once(Some(50.0)).chain(std::iter::repeat_n(Some(45.0), 20)).collect();
        let swinging: Vec<Option<f64>> =
            (0..20).map(|i| Some(if i % 2 == 0 { 40.0 } else { 60.0 })).collect();
        let with_outlier: Vec<Option<f64>> =
            std::iter::repeat_n(Some(45.0), 12).chain(std::iter::once(Some(120.0))).collect();
        let series: [&[Option<f64>]; 3] = [&converging, &swinging, &with_outlier];

        let last_valid = |v: &[Option<f64>]| v.iter().flatten().next_back().copied().unwrap();
        let mean_valid = |v: &[Option<f64>]| {
            let xs: Vec<f64> = v.iter().flatten().copied().collect();
            xs.iter().sum::<f64>() / xs.len() as f64
        };
        let constant = |_: &[Option<f64>]| 45.0;
        let midpoint = |_: &[Option<f64>]| (cfg.min_val + cfg.max_val) / 2.0;
        let scorers: [Scorer<&[Option<f64>]>; 4] = [
            ("last valid value", &last_valid),
            ("mean of valid values", &mean_valid),
            ("constant 45", &constant),
            ("the cold-start midpoint", &midpoint),
        ];
        for (name, f) in scorers {
            let misses = series.iter().any(|s| (f(s) - fold_history(s, &cfg).baseline).abs() > 0.05);
            assert!(misses, "the do-nothing centre `{name}` reproduces the fold on every series");
        }
    }

    /// The hard-outlier limb: past `HARD_OUTLIER_K` spreads a night is SEEN (staleness restarts) but
    /// never folded. One millisecond inside the threshold it folds, winsorised to `WINSOR_K` spreads.
    #[test]
    fn a_hard_outlier_night_is_seen_but_never_folded() {
        let cfg = MetricCfg::hrv();
        let before = seeded(45.0, 5.0, 10);
        let held = update(Some(before), Some(71.0), &cfg); // dev 26 > 5 x 5
        assert_eq!((held.baseline, held.spread, held.n_valid), (45.0, 5.0, 10));
        assert_eq!(held.nights_since_update, 0, "the night was seen, so staleness restarts");
        let folded = update(Some(before), Some(70.0), &cfg); // dev 25 is not PAST the threshold
        let lb = lambda(cfg.half_life_b);
        let winsorised = 45.0 + WINSOR_K * 5.0 * lb;
        assert!((folded.baseline - winsorised).abs() < 1e-12, "got {}, want {winsorised}", folded.baseline);
        assert_eq!(folded.n_valid, 11);
        assert!(folded.baseline < 45.0 + 25.0 * lb, "winsorising is what caps the move");
    }

    /// The spread limb: `floor_spread` is a floor, not the value. A winsorised fold leaves a residual
    /// the spread EWMA absorbs, so the spread every z-score divides by rises off its floor.
    #[test]
    fn winsorised_fold_lifts_the_spread_off_its_floor() {
        let cfg = MetricCfg::hrv();
        let s = update(Some(seeded(45.0, 5.0, 10)), Some(70.0), &cfg);
        let ls = lambda(cfg.half_life_s);
        let want = ls * (70.0 - s.baseline).abs() + (1.0 - ls) * 5.0;
        assert!((s.spread - want).abs() < 1e-12, "got {}, want {want}", s.spread);
        assert!(s.spread > cfg.floor_spread, "spread stayed on its floor at {}", s.spread);
        // A steady series never leaves the floor, so a gate built on one cannot see this limb at all.
        assert_eq!(fold_history(&vec![Some(45.0); 21], &cfg).spread, cfg.floor_spread);
    }

    /// RECORDED, not desired: below `EARLY_ADAPT_NIGHTS` there is no hard-outlier limb and the winsor
    /// window is `EARLY_SPREAD_INFLATE` times wider, so one 200 ms night moves an HRV centre of 45 ms
    /// by 7.736 ms. The same night one fold later moves it by zero. The guard is a step, not a ramp,
    /// and it is open over exactly the nights the baseline is weakest.
    #[test]
    fn a_young_baseline_has_no_outlier_guard() {
        let cfg = MetricCfg::hrv();
        let young = update(Some(seeded(45.0, 5.0, EARLY_ADAPT_NIGHTS - 1)), Some(200.0), &cfg);
        let elb = lambda(EARLY_HALF_LIFE_B);
        let clamped = 45.0 + WINSOR_K * EARLY_SPREAD_INFLATE * 5.0;
        assert!((young.baseline - (elb * clamped + (1.0 - elb) * 45.0)).abs() < 1e-12, "got {}", young.baseline);
        assert_eq!(young.n_valid, EARLY_ADAPT_NIGHTS, "a young night is always folded");
        assert!((young.baseline - 52.736230276).abs() < 1e-9, "got {}", young.baseline);
        // The same night one fold later is refused outright: the guard is a step, not a ramp.
        assert_eq!(update(Some(seeded(45.0, 5.0, EARLY_ADAPT_NIGHTS)), Some(200.0), &cfg).baseline, 45.0);
    }

    #[test]
    fn deviation_matches_the_agreed_delta_z_ratio_formula() {
        let state = seeded(50.0, 4.0, 20);
        let d = deviation(58.0, &state);
        assert!((d.delta - 8.0).abs() < 1e-12, "got {}", d.delta);
        let want_z = 8.0 / (1.253 * 4.0);
        assert!((d.z - want_z).abs() < 1e-12, "got {}, want {want_z}", d.z);
        let want_ratio = 58.0 / 50.0 - 1.0;
        assert!((d.ratio - want_ratio).abs() < 1e-12, "got {}, want {want_ratio}", d.ratio);
        // A value exactly at baseline deviates by nothing on every axis.
        let zero = deviation(50.0, &state);
        assert_eq!((zero.delta, zero.z, zero.ratio), (0.0, 0.0, 0.0));
        // Below baseline: every axis flips sign.
        let below = deviation(42.0, &state);
        assert!(below.delta < 0.0 && below.z < 0.0 && below.ratio < 0.0);
    }

    /// The status lifecycle end to end: Calibrating, Provisional, Trusted, Stale after `STALE_DAYS`
    /// missed nights, and back on the next real one. An unseeded baseline never goes stale.
    #[test]
    fn the_status_lifecycle_reaches_stale_and_recovers() {
        let cfg = MetricCfg::hrv();
        let mut s = update(None, Some(45.0), &cfg);
        assert_eq!(s.status, BaselineStatus::Calibrating);
        for _ in 1..MIN_NIGHTS_SEED { s = update(Some(s), Some(45.0), &cfg); }
        assert_eq!((s.n_valid, s.status), (MIN_NIGHTS_SEED, BaselineStatus::Provisional));
        for _ in MIN_NIGHTS_SEED..MIN_NIGHTS_TRUST { s = update(Some(s), Some(45.0), &cfg); }
        assert_eq!((s.n_valid, s.status), (MIN_NIGHTS_TRUST, BaselineStatus::Trusted));
        // STALE_DAYS missed nights are still Trusted; the next one tips it.
        for _ in 0..STALE_DAYS { s = update(Some(s), None, &cfg); }
        assert_eq!((s.nights_since_update, s.status), (STALE_DAYS, BaselineStatus::Trusted));
        s = update(Some(s), None, &cfg);
        assert_eq!((s.nights_since_update, s.status), (STALE_DAYS + 1, BaselineStatus::Stale));
        assert_eq!((s.n_valid, s.baseline), (MIN_NIGHTS_TRUST, 45.0), "a stale baseline keeps its nights");
        s = update(Some(s), Some(45.0), &cfg);
        assert_eq!((s.nights_since_update, s.status), (0, BaselineStatus::Trusted));
        // Under MIN_NIGHTS_SEED there is nothing to go stale.
        let unseeded: Vec<Option<f64>> =
            std::iter::repeat_n(Some(45.0), 2).chain(std::iter::repeat_n(None, 40)).collect();
        let u = fold_history(&unseeded, &cfg);
        assert_eq!((u.n_valid, u.status), (2, BaselineStatus::Calibrating));
    }

    fn good_night() -> NightChannels {
        NightChannels { hrv_ms: Some(45.0), resting_hr_bpm: Some(55.0), resp_bpm: Some(14.0),
            skin_temp_c: Some(33.0), skin_temp_max_c: Some(34.2), total_sleep_secs: Some(7.0 * 3600.0),
            quality: Some(0.9) }
    }

    #[test]
    fn a_plausible_night_is_valid() {
        assert_eq!(night_verdict(&good_night(), &NightGate::default()), NightVerdict::Valid);
    }

    #[test]
    fn missing_channels_never_reject() {
        let n = NightChannels { hrv_ms: Some(45.0), ..Default::default() };
        assert_eq!(night_verdict(&n, &NightGate::default()), NightVerdict::Valid);
    }

    #[test]
    fn nightstand_night_rejects_every_metric() {
        // Strap off the wrist at room temperature: HRV and resting HR still look plausible on their own.
        let mut n = good_night();
        n.skin_temp_c = Some(22.4);
        n.skin_temp_max_c = Some(23.1);
        assert_eq!(night_verdict(&n, &NightGate::default()), NightVerdict::SkinTempImplausible);
        // The per-metric gate accepts it — this is the behaviour the conjunction fixes.
        assert!(MetricCfg::hrv().min_val <= 45.0 && 45.0 <= MetricCfg::hrv().max_val);
        assert!(MetricCfg::skin_temp().min_val <= 22.4, "the storage band admits an off-wrist night");
    }

    #[test]
    fn conjunction_changes_the_hrv_baseline_a_per_metric_gate_would_fold() {
        let mut nights = vec![good_night(); 12];
        // Three nightstand nights: implausible temperature, a plausible-looking low HRV.
        for i in [4usize, 5, 6] {
            nights[i].hrv_ms = Some(20.0);
            nights[i].skin_temp_c = Some(22.0);
            nights[i].skin_temp_max_c = Some(22.9);
        }
        let per_metric: Vec<Option<f64>> = nights.iter().map(|n| n.hrv_ms).collect();
        let old = fold_history(&per_metric, &MetricCfg::hrv());
        let new = fold_history_nights(&nights, NightMetric::Hrv, &NightGate::default());
        assert_eq!(old.n_valid, 12, "per-metric gate folds all twelve");
        assert_eq!(new.n_valid, 9, "the conjunction skips the three nightstand nights");
        assert!(new.baseline > old.baseline + 1.0,
            "old {} was dragged down by nights the strap was not worn, new {}", old.baseline, new.baseline);
        assert_eq!(night_verdicts(&nights, &NightGate::default()).iter()
            .filter(|v| **v == NightVerdict::SkinTempImplausible).count(), 3);
    }

    #[test]
    fn rejected_night_skip_and_holds_rather_than_disappearing() {
        let mut nights = vec![good_night(); 6];
        nights[5].total_sleep_secs = Some(2.0 * 3600.0);
        let s = fold_history_nights(&nights, NightMetric::RestingHr, &NightGate::default());
        assert_eq!(s.n_valid, 5);
        assert_eq!(s.nights_since_update, 1, "staleness must still advance on a rejected night");
    }

    #[test]
    fn each_channel_can_reject_alone() {
        let g = NightGate::default();
        let check = |mutate: &dyn Fn(&mut NightChannels), want: NightVerdict| {
            let mut n = good_night();
            mutate(&mut n);
            assert_eq!(night_verdict(&n, &g), want);
        };
        check(&|n| n.total_sleep_secs = Some(3.9 * 3600.0), NightVerdict::SleepTooShort);
        check(&|n| n.skin_temp_max_c = Some(41.0), NightVerdict::SkinTempImplausible);
        check(&|n| n.resting_hr_bpm = Some(180.0), NightVerdict::RestingHrImplausible);
        check(&|n| n.hrv_ms = Some(400.0), NightVerdict::HrvImplausible);
        check(&|n| n.resp_bpm = Some(60.0), NightVerdict::RespImplausible);
        check(&|n| n.skin_temp_c = Some(f64::NAN), NightVerdict::SkinTempImplausible);
    }

    #[test]
    fn sleep_gate_is_inclusive_at_four_hours() {
        let mut n = good_night();
        n.total_sleep_secs = Some(MIN_SLEEP_SECS);
        assert_eq!(night_verdict(&n, &NightGate::default()), NightVerdict::Valid);
        n.total_sleep_secs = Some(MIN_SLEEP_SECS - 1.0);
        assert_eq!(night_verdict(&n, &NightGate::default()), NightVerdict::SleepTooShort);
    }

    #[test]
    fn quality_floor_is_off_until_a_caller_supplies_one() {
        let mut n = good_night();
        n.quality = Some(0.05);
        assert_eq!(night_verdict(&n, &NightGate::default()), NightVerdict::Valid);
        let g = NightGate { min_quality: Some(0.5), ..Default::default() };
        assert_eq!(night_verdict(&n, &g), NightVerdict::LowQuality);
        // An unknown quality never rejects, even with a floor set.
        n.quality = None;
        assert_eq!(night_verdict(&n, &g), NightVerdict::Valid);
    }

    /// One night's samples and the centre/max/kept they reduce to. Six nights with SIX DIFFERENT
    /// centres: the previous gate pinned a single 33.4, which a reducer returning 33.4 also satisfies.
    struct TempNight {
        what: &'static str,
        samples: &'static [(f64, f64)],
        median_c: f64,
        max_c: f64,
        n_kept: usize,
    }

    const MIN_CONF: f64 = 0.3;

    const TEMP_NIGHTS: &[TempNight] = &[
        TempNight { what: "a low-confidence sample is dropped before the median",
            samples: &[(33.0, 0.9), (33.4, 0.8), (5.0, 0.05), (34.0, 0.7)],
            median_c: 33.4, max_c: 34.0, n_kept: 3 },
        TempNight { what: "an even count takes the mean of the two middles",
            samples: &[(32.0, 0.9), (32.5, 0.9), (33.5, 0.9), (34.0, 0.9)],
            median_c: 33.0, max_c: 34.0, n_kept: 4 },
        TempNight { what: "one hot window moves the max but not the centre",
            samples: &[(33.0, 0.9), (33.125, 0.9), (33.25, 0.9), (33.375, 0.9), (40.0, 0.9)],
            median_c: 33.25, max_c: 40.0, n_kept: 5 },
        TempNight { what: "the samples are sorted before the middle is taken",
            samples: &[(34.5, 0.9), (33.0, 0.9), (33.75, 0.9)],
            median_c: 33.75, max_c: 34.5, n_kept: 3 },
        TempNight { what: "confidence exactly at the floor is kept",
            samples: &[(33.0, MIN_CONF), (35.0, 0.9)],
            median_c: 34.0, max_c: 35.0, n_kept: 2 },
        TempNight { what: "a non-finite sample is dropped however confident",
            samples: &[(f64::INFINITY, 1.0), (33.0, 0.9), (33.5, 0.9), (34.5, 0.9)],
            median_c: 33.5, max_c: 34.5, n_kept: 3 },
    ];

    #[test]
    fn night_skin_temp_medians_the_confident_samples_of_every_night() {
        for n in TEMP_NIGHTS {
            let s = night_skin_temp(n.samples, MIN_CONF).unwrap_or_else(|| panic!("{}", n.what));
            assert!((s.median_c - n.median_c).abs() < 1e-12, "{}: centre {}", n.what, s.median_c);
            assert!((s.max_c - n.max_c).abs() < 1e-12, "{}: max {}", n.what, s.max_c);
            assert_eq!((s.n_kept, s.n_total), (n.n_kept, n.samples.len()), "{}", n.what);
            let want_coverage = n.n_kept as f64 / n.samples.len() as f64;
            assert!((s.coverage() - want_coverage).abs() < 1e-12, "{}", n.what);
        }
        // Nothing confident, and nothing at all, both refuse rather than inventing a centre.
        assert!(night_skin_temp(&[(33.0, 0.0)], MIN_CONF).is_none());
        assert!(night_skin_temp(&[], MIN_CONF).is_none());
    }

    /// Five do-nothing reducers, each asserted to miss at least one night above. Without this the
    /// single-night gate was satisfied by a reducer that returns 33.4 for every night ever recorded.
    #[test]
    fn a_do_nothing_night_reducer_fails_the_skin_temp_nights() {
        let constant = |_: &[(f64, f64)]| 33.4;
        let mean_of_confident = |s: &[(f64, f64)]| {
            let xs: Vec<f64> = s.iter().filter(|(v, c)| v.is_finite() && *c >= MIN_CONF).map(|(v, _)| *v).collect();
            xs.iter().sum::<f64>() / xs.len() as f64
        };
        let max_of_confident = |s: &[(f64, f64)]| {
            s.iter().filter(|(v, c)| v.is_finite() && *c >= MIN_CONF)
                .map(|(v, _)| *v).fold(f64::NEG_INFINITY, f64::max)
        };
        let median_ignoring_confidence = |s: &[(f64, f64)]| {
            let xs: Vec<f64> = s.iter().filter(|(v, _)| v.is_finite()).map(|(v, _)| *v).collect();
            crate::stats::median(&xs)
        };
        let first_confident = |s: &[(f64, f64)]| {
            s.iter().find(|(v, c)| v.is_finite() && *c >= MIN_CONF).map(|(v, _)| *v).unwrap()
        };
        let scorers: [Scorer<&[(f64, f64)]>; 5] = [
            ("constant 33.4", &constant),
            ("mean instead of median", &mean_of_confident),
            ("max instead of median", &max_of_confident),
            ("median without the confidence filter", &median_ignoring_confidence),
            ("the first confident sample", &first_confident),
        ];
        for (name, f) in scorers {
            let misses = TEMP_NIGHTS.iter()
                .any(|n| (f(n.samples) - n.median_c).abs() > 1e-9);
            assert!(misses, "the do-nothing reducer `{name}` reproduces every night's centre");
        }
    }

    /// The skin-temp chain the Vitals card and the skin-temp driver read: per-night median, into
    /// `NightChannels`, into the baseline and spread a nightly deviation is measured against. Ten
    /// nights with ten different medians, so a constant reducer lands on a different baseline.
    #[test]
    fn nightly_skin_temp_medians_drive_the_skin_temp_baseline() {
        let samples = |centre: f64| vec![(centre - 0.5, 0.9), (centre, 0.9), (centre + 0.75, 0.9), (10.0, 0.01)];
        let centres: Vec<f64> = (0..10).map(|i| 33.0 + i as f64 * 0.125).collect();
        let nights: Vec<NightChannels> = centres.iter()
            .map(|&c| {
                let t = night_skin_temp(&samples(c), MIN_CONF).unwrap();
                assert!((t.median_c - c).abs() < 1e-12, "the reducer must recover the planted centre");
                NightChannels { skin_temp_c: Some(t.median_c), skin_temp_max_c: Some(t.max_c),
                    total_sleep_secs: Some(7.0 * 3600.0), quality: Some(t.coverage()), ..Default::default() }
            })
            .collect();
        assert!(night_verdicts(&nights, &NightGate::default()).iter().all(|v| v.valid()),
            "every planted night must be a baseline night, or this measures rejection instead");

        let s = fold_history_nights(&nights, NightMetric::SkinTemp, &NightGate::default());
        // Closed form: the first centre seeds, the next seven fold young at EARLY_HALF_LIFE_B and the
        // last two at half_life_b. Steps of 0.125 stay inside the winsor window, so nothing is clamped.
        let (elb, lb) = (lambda(EARLY_HALF_LIFE_B), lambda(MetricCfg::skin_temp().half_life_b));
        let mut want = centres[0];
        for (i, &c) in centres.iter().enumerate().skip(1) {
            let rate = if i < EARLY_ADAPT_NIGHTS as usize { elb } else { lb };
            want = rate * c + (1.0 - rate) * want;
        }
        assert!((s.baseline - want).abs() < 1e-12, "got {}, want {want}", s.baseline);
        assert_eq!((s.n_valid, s.status), (10, BaselineStatus::Provisional));

        // A reducer that returns one fixed centre for every night lands somewhere else entirely.
        let flat: Vec<NightChannels> =
            nights.iter().map(|n| NightChannels { skin_temp_c: Some(33.4), ..*n }).collect();
        let f = fold_history_nights(&flat, NightMetric::SkinTemp, &NightGate::default());
        assert!((f.baseline - s.baseline).abs() > 0.1,
            "a constant nightly centre reaches the same baseline: {} vs {}", f.baseline, s.baseline);
    }
}
