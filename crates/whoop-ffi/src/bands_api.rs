//! Bands, tiers and split percentages behind a displayed word, swatch or ramp position. A frontend
//! reads the reading here and then chooses the colour and the wording for it, instead of declaring
//! its own cut points or rounding a share on its own.

use crate::*;

/// How many equal tiers a sleep-performance driver's 0-100 is read in.
#[uniffi::export]
pub fn sleep_driver_tiers() -> u32 {
    rest::DRIVER_TIERS
}

/// Which tier a driver's 0-100 falls in, counting from 0. Off-scale values clamp into range.
#[uniffi::export]
pub fn sleep_driver_tier(percent: f64) -> u32 {
    rest::driver_tier(percent)
}

/// The tier a driver LIGHTS, mirrored when `higher_is_better` is false. Only the lit tier moves.
#[uniffi::export]
pub fn sleep_driver_tier_lit(percent: f64, higher_is_better: bool) -> u32 {
    rest::driver_tier_lit(percent, higher_is_better)
}

/// Where a tier sits on a 0..1 ramp, so its swatch samples the same scale the value does.
#[uniffi::export]
pub fn sleep_driver_tier_position(tier: u32) -> f64 {
    rest::driver_tier_position(tier)
}

/// How far behind a sleep-debt balance reads. The word and colour for each band are the caller's.
#[derive(uniffi::Enum)]
pub enum DebtSeverity {
    OnTarget,
    Moderate,
    Heavy,
}

impl From<sleep_debt::DebtSeverity> for DebtSeverity {
    fn from(s: sleep_debt::DebtSeverity) -> Self {
        match s {
            sleep_debt::DebtSeverity::OnTarget => DebtSeverity::OnTarget,
            sleep_debt::DebtSeverity::Moderate => DebtSeverity::Moderate,
            sleep_debt::DebtSeverity::Heavy => DebtSeverity::Heavy,
        }
    }
}

/// Severity band of a SIGNED ledger balance (minutes; negative = debt). Any surplus is on target.
#[uniffi::export]
pub fn sleep_debt_severity(balance_min: f64) -> DebtSeverity {
    sleep_debt::severity(balance_min).into()
}

/// How strong a correlation reads. Each surface words these its own way; none owns the cut points.
#[derive(uniffi::Enum)]
pub enum CorrelationStrength {
    Negligible,
    Weak,
    Moderate,
    Strong,
    VeryStrong,
}

impl From<physio_algo::stats::CorrelationStrength> for CorrelationStrength {
    fn from(s: physio_algo::stats::CorrelationStrength) -> Self {
        use physio_algo::stats::CorrelationStrength as S;
        match s {
            S::Negligible => CorrelationStrength::Negligible,
            S::Weak => CorrelationStrength::Weak,
            S::Moderate => CorrelationStrength::Moderate,
            S::Strong => CorrelationStrength::Strong,
            S::VeryStrong => CorrelationStrength::VeryStrong,
        }
    }
}

/// Strength band of a Pearson `r`, cut on `|r|`, so −0.62 and +0.62 read equally strong.
#[uniffi::export]
pub fn correlation_strength(r: f64) -> CorrelationStrength {
    physio_algo::stats::correlation_strength(r).into()
}

/// Fewest overlapping day pairs a correlation may be shown from.
#[uniffi::export]
pub fn correlation_min_pairs() -> u32 {
    physio_algo::stats::CORRELATION_MIN_PAIRS as u32
}

/// Anchor positions of the two reading ramps on a 0..1 scale. The colours at them are the caller's.
#[derive(uniffi::Record)]
pub struct RampStopsInfo {
    pub recovery: Vec<f64>,
    pub strain: Vec<f64>,
}

/// The ramp anchor positions, so a palette places its colours instead of naming where they sit.
#[uniffi::export]
pub fn ramp_stops() -> RampStopsInfo {
    RampStopsInfo {
        recovery: physio_algo::ramps::RECOVERY_STOPS.to_vec(),
        strain: physio_algo::ramps::STRAIN_STOPS.to_vec(),
    }
}

/// Where a 0-100 score sits on its ramp, clamped to the ends.
#[uniffi::export]
pub fn ramp_position_score(score: f64) -> f64 {
    physio_algo::ramps::score_position(score)
}

/// Where an already-normalised 0..1 fraction sits on its ramp.
#[uniffi::export]
pub fn ramp_position_fraction(fraction: f64) -> f64 {
    physio_algo::ramps::fraction_position(fraction)
}

/// Where a Pearson `r` sits on a ramp: −1 at the bottom, 0 at the middle, +1 at the top.
#[uniffi::export]
pub fn ramp_position_correlation(r: f64) -> f64 {
    physio_algo::ramps::correlation_position(r)
}

/// A split as whole percentages summing to exactly 100, so shares rounded one at a time can no
/// longer read 99 or 101 side by side. `None` when nothing was measured.
#[uniffi::export]
pub fn whole_percentages(parts: Vec<f64>) -> Option<Vec<u32>> {
    physio_algo::apportion::whole_percentages(&parts)
}
