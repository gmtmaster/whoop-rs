//! Daily fluid goal: a sex baseline plus a bump for the day's Effort, rounded to a clean step.
//! Derived only from the body profile and the day's load — never from what was logged.

pub const BASELINE_MALE_ML: i32 = 3700;
pub const BASELINE_FEMALE_ML: i32 = 2700;
pub const BASELINE_OTHER_ML: i32 = 3200;

/// The most extra fluid a full-Effort day adds; the bump is clamped here.
pub const MAX_EFFORT_BUMP_ML: i32 = 700;

/// Goals land on a multiple of this, so the readout is a round number.
pub const ROUND_TO_ML: i32 = 50;

/// The Effort scale the bump is a fraction of.
const EFFORT_SCALE: f64 = 100.0;

/// Nearest integer, ties toward positive infinity.
fn round_half_up(v: f64) -> i32 {
    if !v.is_finite() { return 0; }
    (v + 0.5).floor() as i32
}

/// The baseline (ml) for a profile sex tag. Anything but male/female is the neutral baseline;
/// case- and whitespace-insensitive.
pub fn baseline_for_sex(sex: &str) -> i32 {
    match sex.trim().to_lowercase().as_str() {
        "male" | "m" => BASELINE_MALE_ML,
        "female" | "f" => BASELINE_FEMALE_ML,
        _ => BASELINE_OTHER_ML,
    }
}

/// The bump (ml) for an Effort score in 0–100, clamped into 0–[`MAX_EFFORT_BUMP_ML`] so an
/// out-of-range input cannot push the goal past the cap or under the baseline. `None` scores 0.
pub fn effort_bump_ml(effort: Option<f64>) -> i32 {
    let Some(e) = effort else { return 0 };
    round_half_up(e / EFFORT_SCALE * MAX_EFFORT_BUMP_ML as f64).clamp(0, MAX_EFFORT_BUMP_ML)
}

/// Round `value` to the nearest multiple of `step`, half up. A non-positive step is a no-op.
pub fn round_to_nearest(value: i32, step: i32) -> i32 {
    if step <= 0 { return value; }
    ((value + step / 2) / step) * step
}

/// The daily goal (ml): baseline + Effort bump on the [`ROUND_TO_ML`] grid. `effort` is today's
/// score in 0–100, or `None` when the day has not been scored.
pub fn daily_goal_ml(sex: &str, effort: Option<f64>) -> i32 {
    round_to_nearest(baseline_for_sex(sex) + effort_bump_ml(effort), ROUND_TO_ML)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn baselines_by_sex_tag() {
        assert_eq!(baseline_for_sex("male"), 3700);
        assert_eq!(baseline_for_sex("female"), 2700);
        assert_eq!(baseline_for_sex("nonbinary"), 3200);
        assert_eq!(baseline_for_sex("unspecified"), 3200);
        assert_eq!(baseline_for_sex(""), 3200);
    }

    #[test]
    fn sex_tag_is_case_and_whitespace_insensitive() {
        assert_eq!(baseline_for_sex("  MALE "), 3700);
        assert_eq!(baseline_for_sex("Female"), 2700);
        assert_eq!(baseline_for_sex("M"), 3700);
        assert_eq!(baseline_for_sex("f"), 2700);
    }

    #[test]
    fn effort_bump_spans_zero_to_the_cap() {
        assert_eq!(effort_bump_ml(None), 0);
        assert_eq!(effort_bump_ml(Some(0.0)), 0);
        assert_eq!(effort_bump_ml(Some(100.0)), 700);
        assert_eq!(effort_bump_ml(Some(50.0)), 350);
        assert_eq!(effort_bump_ml(Some(37.0)), 259);
        assert_eq!(effort_bump_ml(Some(1.0)), 7);
    }

    #[test]
    fn out_of_range_effort_is_clamped() {
        assert_eq!(effort_bump_ml(Some(-20.0)), 0);
        assert_eq!(effort_bump_ml(Some(150.0)), 700);
        assert_eq!(effort_bump_ml(Some(f64::NAN)), 0);
    }

    #[test]
    fn unscored_day_is_the_bare_baseline() {
        assert_eq!(daily_goal_ml("male", None), 3700);
        assert_eq!(daily_goal_ml("female", None), 2700);
        assert_eq!(daily_goal_ml("nonbinary", None), 3200);
    }

    #[test]
    fn full_effort_adds_the_cap() {
        assert_eq!(daily_goal_ml("male", Some(100.0)), 4400);
        assert_eq!(daily_goal_ml("female", Some(100.0)), 3400);
    }

    #[test]
    fn partial_effort_rounds_to_the_grid() {
        assert_eq!(daily_goal_ml("male", Some(37.0)), 3950);
        assert_eq!(daily_goal_ml("female", Some(63.0)), 3150);
        assert_eq!(daily_goal_ml("nonbinary", Some(13.0)), 3300);
    }

    #[test]
    fn rounding_step_is_half_up_and_a_non_positive_step_is_a_no_op() {
        assert_eq!(round_to_nearest(3725, 50), 3750);
        assert_eq!(round_to_nearest(3724, 50), 3700);
        assert_eq!(round_to_nearest(3700, 50), 3700);
        assert_eq!(round_to_nearest(3717, 0), 3717);
        assert_eq!(round_to_nearest(3717, -50), 3717);
    }
}
