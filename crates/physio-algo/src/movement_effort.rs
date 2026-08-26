//! A provenance-qualified movement signal may establish a low daily Effort floor. The floor is
//! smooth, bounded below 25/100, and combines with cardiovascular Effort by maximum, never addition.

pub const MAX_EFFORT: f64 = 100.0;
pub const STRAIN_DENOMINATOR: f64 = 7201.0;
pub const MOVEMENT_CEILING_EFFORT: f64 = 25.0;
pub const HALF_RESPONSE_STEPS: f64 = 8_000.0;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MovementEffort {
    pub hr_effort: Option<f64>,
    pub movement_floor: Option<f64>,
    pub final_effort: Option<f64>,
    pub movement_floor_active: bool,
}

/// Audited exponential response. Confidence scales movement load before the existing logarithmic
/// Effort mapping; the 25-point asymptote keeps movement evidence from resembling workout load.
pub fn movement_floor(steps: u32, confidence: f64) -> Option<f64> {
    if steps == 0 || !confidence.is_finite() || confidence <= 0.0 {
        return None;
    }
    let confidence = confidence.clamp(0.0, 1.0);
    let fraction = 1.0 - (-(steps as f64) / HALF_RESPONSE_STEPS).exp();
    let load_cap = (MOVEMENT_CEILING_EFFORT / MAX_EFFORT * STRAIN_DENOMINATOR.ln()).exp() - 1.0;
    let load = confidence * load_cap * fraction;
    Some(MAX_EFFORT * (load + 1.0).ln() / STRAIN_DENOMINATOR.ln())
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
            (1_000, 7.60),
            (2_000, 11.66),
            (5_000, 17.70),
            (8_000, 20.53),
            (10_000, 21.68),
            (15_000, 23.34),
            (20_000, 24.14),
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
        assert!((low.final_effort.unwrap() - 21.68).abs() < 0.02);
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
}
