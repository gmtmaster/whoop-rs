//! A provenance-qualified movement signal may establish a low daily Effort floor. The floor is
//! smooth, bounded below 25/100, and combines with cardiovascular Effort by maximum, never addition.
//!
//! ## Curve history (2026-08-26)
//! The first implementation reused the audited Edwards/TRIMP-style `100 * ln(load+1) / ln(7201)`
//! mapping (already used for cardiovascular Effort) to convert an exponentially-saturating step
//! "load" into a 0-25 floor. That composition is front-loaded: at confidence 1.0 it produced
//! ~7.6/100 at 1,000 steps and ~11.7/100 at 2,000 steps - already reaching roughly half the
//! 25-point ceiling within the first quarter of a plausible daily step range. That does not match
//! the product intent (0-1k negligible, ~2k very small, ~5k small-but-visible, ~8-10k meaningful,
//! ~15-20k substantial, all well short of the ceiling), so it has been replaced.
//!
//! The current curve is a simple rational (Hill-type, n=2) saturation:
//!   `floor(steps, confidence) = confidence * CEILING * steps^2 / (steps^2 + HALF_SATURATION^2)`
//! `HALF_SATURATION` is the step count at which the *unscaled* curve reaches half of `CEILING`
//! (by construction: `HALF_SATURATION^2 / (HALF_SATURATION^2 + HALF_SATURATION^2) = 0.5`), which
//! makes the one free parameter directly interpretable. It is smooth (C-infinity), strictly
//! monotonic increasing in `steps`, bounded strictly below `CEILING` for any finite step count, has
//! no cliff/threshold, and needs no coupling to the unrelated cardiovascular Strain/TRIMP mapping.

pub const MAX_EFFORT: f64 = 100.0;
pub const MOVEMENT_CEILING_EFFORT: f64 = 25.0;
/// Step count at which the unscaled curve reaches half of `MOVEMENT_CEILING_EFFORT`. Chosen so
/// that low/incidental daily movement (0-2k) stays negligible-to-small, a genuinely active day
/// (~8-10k) reads as a meaningful but clearly sub-workout floor, and only sustained high daily
/// step counts (~15-20k) approach (without reaching) the ceiling.
pub const HALF_SATURATION_STEPS: f64 = 6_000.0;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MovementEffort {
    pub hr_effort: Option<f64>,
    pub movement_floor: Option<f64>,
    pub final_effort: Option<f64>,
    pub movement_floor_active: bool,
}

/// Smooth, monotonic, bounded movement floor. Confidence linearly scales the floor (weak evidence
/// -> a proportionally weaker floor, never a fabricated strong one); zero steps or non-finite/
/// non-positive confidence yields no floor at all (`None`), leaving HR-derived Effort untouched.
pub fn movement_floor(steps: u32, confidence: f64) -> Option<f64> {
    if steps == 0 || !confidence.is_finite() || confidence <= 0.0 {
        return None;
    }
    let confidence = confidence.clamp(0.0, 1.0);
    let steps = steps as f64;
    let saturation = steps * steps / (steps * steps + HALF_SATURATION_STEPS * HALF_SATURATION_STEPS);
    Some(confidence * MOVEMENT_CEILING_EFFORT * saturation)
}

pub fn resolve(
    hr_effort: Option<f64>,
    steps: Option<u32>,
    confidence: Option<f64>,
) -> MovementEffort {
    let floor = steps
        .zip(confidence)
        .and_then(|(s, c)| movement_floor(s, c));
    let final_effort = match (hr_effort, floor) {
        (Some(hr), Some(movement)) => Some(hr.max(movement)),
        (Some(hr), None) => Some(hr),
        (None, Some(movement)) => Some(movement),
        (None, None) => None,
    };
    MovementEffort {
        hr_effort,
        movement_floor: floor,
        final_effort,
        movement_floor_active: floor.zip(hr_effort).map_or(floor.is_some(), |(m, h)| m > h),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reference_values_are_smooth_and_bounded() {
        let cases = [
            (1_000, 0.68),
            (2_000, 2.50),
            (5_000, 10.25),
            (8_000, 16.00),
            (10_000, 18.38),
            (15_000, 21.55),
            (20_000, 22.94),
        ];
        let mut previous = 0.0;
        for (steps, expected) in cases {
            let value = movement_floor(steps, 1.0).unwrap();
            assert!(
                (value - expected).abs() < 0.02,
                "steps={steps} value={value}"
            );
            assert!(value > previous);
            assert!(value < MOVEMENT_CEILING_EFFORT);
            previous = value;
        }
    }

    #[test]
    fn hr_is_authoritative_and_never_added() {
        let high = resolve(Some(22.0), Some(10_000), Some(1.0));
        assert_eq!(high.final_effort, Some(22.0));
        assert!(!high.movement_floor_active);

        let low = resolve(Some(2.0), Some(10_000), Some(1.0));
        assert!((low.final_effort.unwrap() - 18.38).abs() < 0.02);
        assert!(low.movement_floor_active);
    }

    #[test]
    fn zero_or_weak_low_step_evidence_is_negligible() {
        assert_eq!(movement_floor(0, 1.0), None);
        assert!(movement_floor(1_000, 0.05).unwrap() < 1.0);
    }

    #[test]
    fn untrusted_or_missing_steps_leave_hr_unchanged() {
        assert_eq!(resolve(Some(8.2), None, None).final_effort, Some(8.2));
        assert_eq!(
            resolve(Some(8.2), Some(10_000), Some(0.0)).final_effort,
            Some(8.2)
        );
    }

    /// A very low sedentary-day HR effort (e.g. a walk that never clears %HRR threshold) is
    /// raised to the movement floor at every audited step level, never above the 25-point
    /// ceiling, and never by addition (floor is always <= MOVEMENT_CEILING_EFFORT on its own).
    #[test]
    fn low_hr_day_is_raised_to_the_movement_floor_at_every_audited_step_level() {
        let low_hr = 1.5;
        for steps in [0u32, 1_000, 2_000, 5_000, 8_000, 10_000, 15_000, 20_000] {
            let result = resolve(Some(low_hr), Some(steps), Some(1.0));
            let expected_floor = movement_floor(steps, 1.0);
            assert_eq!(result.movement_floor, expected_floor);
            match expected_floor {
                Some(floor) => {
                    assert_eq!(result.final_effort, Some(floor.max(low_hr)));
                    assert!(floor < MOVEMENT_CEILING_EFFORT);
                }
                None => assert_eq!(result.final_effort, Some(low_hr)),
            }
        }
    }

    /// High cardiovascular load must never be reduced by a low step count, and must never be
    /// increased by adding movement on top of it (max, not sum).
    #[test]
    fn high_sedentary_hr_is_never_reduced_or_added_to_by_movement() {
        let high_hr = 45.0;
        let result = resolve(Some(high_hr), Some(20_000), Some(1.0));
        assert_eq!(result.final_effort, Some(high_hr));
        assert!(!result.movement_floor_active);
        assert_ne!(
            result.final_effort,
            Some(high_hr + result.movement_floor.unwrap()),
            "final effort must never be the sum of HR effort and the movement floor"
        );
    }

    /// `HALF_SATURATION_STEPS` is directly interpretable: the unscaled curve is exactly half of
    /// `MOVEMENT_CEILING_EFFORT` there, by construction of the rational saturation curve.
    #[test]
    fn half_saturation_point_is_exactly_half_the_ceiling() {
        let value = movement_floor(HALF_SATURATION_STEPS as u32, 1.0).unwrap();
        assert!((value - MOVEMENT_CEILING_EFFORT / 2.0).abs() < 1e-9);
    }
}
