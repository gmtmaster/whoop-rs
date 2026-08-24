//! Display-ramp banding: the anchor positions each reading ramp's stops sit at, and the value → 0..1
//! normalisation that samples them. A ramp is a banding with the steps smoothed out, so the positions
//! live here with the other cut points. Which colour sits at an anchor, and how two are blended, is the
//! frontend's.

/// Anchor positions of the Charge (recovery) ramp on a 0..1 scale.
pub const RECOVERY_STOPS: [f64; 5] = [0.00, 0.30, 0.55, 0.78, 1.00];

/// Anchor positions of the Effort (strain) ramp on a 0..1 scale.
pub const STRAIN_STOPS: [f64; 4] = [0.00, 0.33, 0.66, 1.00];

/// Full scale of a 0-100 score, the divisor in [`score_position`].
pub const SCORE_SCALE_MAX: f64 = 100.0;

/// Where a 0-100 score sits on its ramp, clamped to the ends.
pub fn score_position(score: f64) -> f64 {
    clamp01(score / SCORE_SCALE_MAX)
}

/// Where an already-normalised 0..1 fraction sits on its ramp: the identity, clamped.
pub fn fraction_position(fraction: f64) -> f64 {
    clamp01(fraction)
}

/// Where a Pearson `r` sits on a ramp: −1 at the bottom, 0 at the middle, +1 at the top.
pub fn correlation_position(r: f64) -> f64 {
    clamp01((r + 1.0) / 2.0)
}

/// Hold a position inside the ramp; a value that is not a number reads as the bottom.
fn clamp01(position: f64) -> f64 {
    if position.is_nan() {
        return 0.0;
    }
    position.clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_two_ramps_keep_their_anchor_positions() {
        assert_eq!(RECOVERY_STOPS, [0.00, 0.30, 0.55, 0.78, 1.00]);
        assert_eq!(STRAIN_STOPS, [0.00, 0.33, 0.66, 1.00]);
    }

    /// Both ramps span the whole scale and never step backwards, so sampling one is monotonic.
    #[test]
    fn anchor_positions_are_sorted_and_span_the_scale() {
        for stops in [&RECOVERY_STOPS[..], &STRAIN_STOPS[..]] {
            assert_eq!(stops.first().copied(), Some(0.0));
            assert_eq!(stops.last().copied(), Some(1.0));
            assert!(
                stops.windows(2).all(|w| w[1] > w[0]),
                "{stops:?} is not strictly increasing"
            );
        }
    }

    #[test]
    fn a_score_maps_to_its_own_hundredth() {
        assert_eq!(score_position(0.0), 0.0);
        assert_eq!(score_position(55.0), 0.55);
        assert_eq!(score_position(100.0), 1.0);
    }

    #[test]
    fn a_position_off_the_scale_reads_as_the_nearer_end() {
        assert_eq!(score_position(-20.0), 0.0);
        assert_eq!(score_position(140.0), 1.0);
        assert_eq!(fraction_position(-0.5), 0.0);
        assert_eq!(fraction_position(1.5), 1.0);
        assert_eq!(score_position(f64::NAN), 0.0);
        assert_eq!(fraction_position(f64::NAN), 0.0);
    }

    #[test]
    fn a_fraction_is_its_own_position() {
        assert_eq!(fraction_position(0.0), 0.0);
        assert_eq!(fraction_position(0.42), 0.42);
        assert_eq!(fraction_position(1.0), 1.0);
    }

    /// Zero lands mid-ramp and the two signs sit equally far from it.
    #[test]
    fn a_correlation_centres_on_zero() {
        assert_eq!(correlation_position(-1.0), 0.0);
        assert_eq!(correlation_position(0.0), 0.5);
        assert_eq!(correlation_position(1.0), 1.0);
        assert!((correlation_position(0.5) - 0.75).abs() < 1e-12);
        assert!((correlation_position(-0.5) - 0.25).abs() < 1e-12);
    }
}
