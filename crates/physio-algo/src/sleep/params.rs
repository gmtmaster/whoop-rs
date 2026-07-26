//! The V2 recipe's tunable surface, gathered in one struct so the emission weights, gates and priors are
//! read (and swept) in one place instead of scattered as literals. [`Params::SHIPPED`] is the tuned recipe
//! the stager uses by default; `stage` with anything else is the tuning path only.

/// Every coefficient the V2 stager reads. Field order mirrors the emission it feeds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Params {
    /// Deep emission: HR-variability, HR and motion weights.
    pub deep_hrv: f64,
    pub deep_hr: f64,
    pub deep_motion: f64,
    /// REM emission: HR-variability, motion and HR weights.
    pub rem_hrv: f64,
    pub rem_motion: f64,
    pub rem_hr: f64,
    /// Awake emission: motion weight, plus the dead-zoned cardiac pair.
    pub awake_motion: f64,
    pub awake_hrv: f64,
    pub awake_hr: f64,
    /// Below this |z| the cardiac terms read as noise and contribute nothing to the awake emission.
    pub awake_deadzone: f64,
    /// Deep-eligibility HR-flatness percentile gate and the slope of the penalty past it.
    pub deep_gate_thresh: f64,
    pub deep_gate_slope: f64,
    /// Multiples of the night's median jerk: the per-sample movement threshold, and the ceiling under
    /// which an epoch counts as quiescent.
    pub jerk_move_mult: f64,
    pub jerk_gate_mult: f64,
    /// Added to the awake emission when peak jerk clears the gate multiple.
    pub motion_gate_boost: f64,
    /// Weight of the RSA respiration-regularity term (added to deep, subtracted from REM).
    pub resp_weight: f64,
    /// Stage base rates, in probability (logged before use), in [deep, rem, light, awake] order.
    pub base_rate: [f64; 4],
    /// Sleep-cycle prior: deep scale + the fraction of the night it decays over; REM scale, the early
    /// fraction it is suppressed in, and the size of that suppression.
    pub cycle_deep_scale: f64,
    pub cycle_deep_decay: f64,
    pub cycle_rem_scale: f64,
    pub cycle_rem_early_frac: f64,
    pub cycle_rem_early_penalty: f64,
    /// Sticky transition matrix (rows = from, cols = to) in [deep, rem, light, awake] order.
    pub transition: [[f64; 4]; 4],
}

impl Params {
    /// The tuned recipe the stager ships. Every value here is measured, not chosen: changing one moves
    /// the benchmark, so treat it as data and re-run the fixture sheet after any edit.
    pub const SHIPPED: Params = Params {
        deep_hrv: -0.8,
        deep_hr: 0.5,
        deep_motion: -0.1,
        rem_hrv: 0.8,
        rem_motion: -0.4,
        rem_hr: 0.4,
        awake_motion: 1.0,
        awake_hrv: 0.5,
        awake_hr: 0.6,
        awake_deadzone: 0.30,
        deep_gate_thresh: 0.40,
        deep_gate_slope: 5.0,
        jerk_move_mult: 75.0,
        jerk_gate_mult: 35.0,
        motion_gate_boost: 4.0,
        resp_weight: 0.6,
        base_rate: [0.15, 0.22, 0.50, 0.34],
        cycle_deep_scale: 1.2,
        cycle_deep_decay: 0.55,
        cycle_rem_scale: 1.0,
        cycle_rem_early_frac: 0.12,
        cycle_rem_early_penalty: 3.0,
        transition: [
            [0.76, 0.012, 0.216, 0.012],
            [0.00333, 0.92, 0.06667, 0.01],
            [0.08, 0.08, 0.80, 0.04],
            [0.0, 0.0, 0.10, 0.90],
        ],
    };

    /// Stage base rates in the log domain the emissions add into.
    pub(super) fn base_log_prior(&self) -> [f64; 4] {
        [
            self.base_rate[0].ln(),
            self.base_rate[1].ln(),
            self.base_rate[2].ln(),
            self.base_rate[3].ln(),
        ]
    }
}

impl Default for Params {
    fn default() -> Self {
        Params::SHIPPED
    }
}
