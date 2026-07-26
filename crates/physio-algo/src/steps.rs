//! Wrap-aware step-derivation from the strap's cumulative u16 `step_motion_counter`. Returns raw
//! motion-tick totals before the caller's per-user tick-per-step calibration. Daily total and
//! per-workout window share this kernel so they never disagree on the counter math.

use crate::sleep::StepSample;

/// Largest wrap-aware delta treated as real motion between two adjacent 1 Hz records.
/// A delta at/above this is a sync-gap or reboot boundary, not real steps.
pub const MAX_STEP_DELTA: u16 = 512;

/// Wrap-aware motion ticks between two adjacent counter samples, or `None` when the delta is zero,
/// backwards, or past [`MAX_STEP_DELTA`] (a sync gap / reboot boundary, not real steps). The one
/// counter-delta rule: the daily total, the per-workout window and the wake refinement all read it.
pub fn tick_delta(prev: &StepSample, next: &StepSample) -> Option<u16> {
    let delta = next.counter.wrapping_sub(prev.counter);
    (delta > 0 && delta < MAX_STEP_DELTA).then_some(delta)
}

/// Raw wrap-aware motion-tick total across `samples`. Sorts by `ts`. Returns `None` for
/// fewer than two samples or no positive forward movement (so "no data" stays distinct from zero).
/// The caller applies its `stepTicksPerStep` calibration to the returned ticks.
pub fn steps_in_window(samples: &[StepSample]) -> Option<u32> {
    if samples.len() < 2 {
        return None;
    }
    let mut sorted: Vec<&StepSample> = samples.iter().collect();
    sorted.sort_by_key(|s| s.ts);

    let mut total: u32 = 0;
    for pair in sorted.windows(2) {
        if let Some(delta) = tick_delta(pair[0], pair[1]) {
            total += delta as u32;
        }
    }
    if total > 0 { Some(total) } else { None }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn step(ts: i64, counter: u16) -> StepSample {
        StepSample { ts, counter, activity_class: None }
    }

    #[test]
    fn sums_positive_consecutive_deltas() {
        assert_eq!(steps_in_window(&[step(0, 100), step(60, 150), step(120, 220)]), Some(120));
    }

    #[test]
    fn sorts_unordered_input() {
        assert_eq!(steps_in_window(&[step(120, 220), step(0, 100), step(60, 150)]), Some(120));
    }

    #[test]
    fn handles_u16_wraparound() {
        // 65500 -> 20 wraps: wrapping_sub = 56; then 20 -> 80 => 60. Total 116.
        assert_eq!(steps_in_window(&[step(0, 65500), step(60, 20), step(120, 80)]), Some(116));
    }

    #[test]
    fn fewer_than_two_samples_is_null() {
        assert_eq!(steps_in_window(&[]), None);
        assert_eq!(steps_in_window(&[step(0, 100)]), None);
    }

    #[test]
    fn no_forward_movement_is_null() {
        assert_eq!(steps_in_window(&[step(0, 500), step(60, 500), step(120, 500)]), None);
    }

    #[test]
    fn drops_big_gap_delta_as_boundary() {
        assert_eq!(steps_in_window(&[step(0, 100), step(60, 140), step(120, 5000), step(180, 5030)]), Some(70));
    }

    #[test]
    fn tick_delta_rejects_gap_reboot_and_backwards() {
        assert_eq!(tick_delta(&step(0, 100), &step(1, 140)), Some(40));
        assert_eq!(tick_delta(&step(0, 65500), &step(1, 20)), Some(56)); // genuine wrap
        assert_eq!(tick_delta(&step(0, 100), &step(1, 100)), None); // no movement
        assert_eq!(tick_delta(&step(0, 100), &step(1, 5000)), None); // sync gap / reboot
        assert_eq!(tick_delta(&step(0, 0), &step(1, MAX_STEP_DELTA)), None); // ceiling is exclusive
    }

    #[test]
    fn max_step_delta_boundary_is_exclusive() {
        assert_eq!(steps_in_window(&[step(0, 0), step(60, 512)]), None);
        assert_eq!(steps_in_window(&[step(0, 0), step(60, 511)]), Some(511));
    }
}
