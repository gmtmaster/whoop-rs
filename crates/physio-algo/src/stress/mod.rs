//! Autonomic stress on one shared 0–3 scale: the Baevsky Stress Index over an R-R histogram
//! (`index`), the day-over-day score against a personal baseline (`daily`), and the windowed
//! score that the day and the night both derive from (`window`). Every score in this family is a
//! z-sum squashed by [`squash`] and banded by [`band_of`], so the scale means one thing
//! everywhere. Wellness only, never medical.

mod daily;
mod index;
mod window;

pub use daily::{StressDay, daily_stress};
pub use index::{MIN_BEATS, StressComponents, components_raw, stress_index_raw};
pub use window::{
    ACTIVITY_GATE_G, HourPoint, ScoredHour, SpanMs, StressWindowCfg, StressWindows,
    SuppressedBucket, Suppression, daytime_stress, sleep_stress, windowed_stress,
};

/// Score ceiling. The three bands split it into thirds, and a zero z-sum lands mid-scale at 1.5.
pub const STRESS_MAX: f64 = 3.0;
/// Band floors on the 0–3 scale: low [0, 1) · medium [1, 2) · high [2, 3].
pub const MEDIUM_BAND_FLOOR: f64 = 1.0;
pub const HIGH_BAND_FLOOR: f64 = 2.0;
/// A spread at or below this reads as no spread, so a flat window scores neutral, not infinite.
const SD_FLOOR: f64 = 0.0001;

/// Where a score sits on the shared 0–3 scale.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StressBand {
    Low,
    Medium,
    High,
}

/// The band a 0–3 score falls in. The one place the two floors are compared.
pub fn band_of(score: f64) -> StressBand {
    if score < MEDIUM_BAND_FLOOR {
        StressBand::Low
    } else if score < HIGH_BAND_FLOOR {
        StressBand::Medium
    } else {
        StressBand::High
    }
}

/// Logistic squash of a z-sum onto 0–3: a zero sum is 1.5, positive is stressed. Every score in
/// this module ends here, so they share one scale and one set of bands.
pub fn squash(raw: f64) -> f64 {
    (STRESS_MAX / (1.0 + (-raw).exp())).clamp(0.0, STRESS_MAX)
}

/// Arithmetic mean, `None` on empty — distinct from [`crate::stats::mean`], which reads 0.0 there.
/// An absent reference must not become a zero one.
pub(crate) fn mean_opt(xs: &[f64]) -> Option<f64> {
    if xs.is_empty() {
        None
    } else {
        Some(xs.iter().sum::<f64>() / xs.len() as f64)
    }
}

/// Population standard deviation about `m`; 0.0 with no mean or fewer than two values.
pub(crate) fn population_std(xs: &[f64], m: Option<f64>) -> f64 {
    let Some(m) = m else { return 0.0 };
    if xs.len() <= 1 {
        return 0.0;
    }
    let var = xs.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / xs.len() as f64;
    var.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn squash_centres_a_zero_sum_mid_scale() {
        assert!((squash(0.0) - 1.5).abs() < 1e-12);
        assert!(squash(-40.0) >= 0.0 && squash(40.0) <= STRESS_MAX);
    }

    #[test]
    fn bands_split_the_scale_into_thirds() {
        assert_eq!(band_of(0.0), StressBand::Low);
        assert_eq!(band_of(0.999), StressBand::Low);
        assert_eq!(band_of(1.0), StressBand::Medium);
        assert_eq!(band_of(1.999), StressBand::Medium);
        assert_eq!(band_of(2.0), StressBand::High);
        assert_eq!(band_of(STRESS_MAX), StressBand::High);
    }

    #[test]
    fn mean_opt_refuses_an_empty_series() {
        assert_eq!(mean_opt(&[]), None);
        assert_eq!(mean_opt(&[1.0, 3.0]), Some(2.0));
        assert_eq!(crate::stats::mean(&[]), 0.0);
    }

    #[test]
    fn population_std_is_zero_without_spread() {
        assert_eq!(population_std(&[5.0; 4], Some(5.0)), 0.0);
        assert_eq!(population_std(&[1.0], Some(1.0)), 0.0);
        assert_eq!(population_std(&[1.0, 2.0], None), 0.0);
        assert!((population_std(&[2.0, 4.0], Some(3.0)) - 1.0).abs() < 1e-12);
    }
}
