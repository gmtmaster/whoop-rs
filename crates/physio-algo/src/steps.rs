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


// ── One steps model, both families ────────────────────────────────────────────────────────────────
//
// A 5.0/MG counts motion TICKS and a 4.0 has no counter at all, only movement volume. Both map to
// steps by the same through-origin line, `steps = k * raw`, so the model, the fit and the confidence
// are shared and only the INPUT differs. `StepsCfg` carries that difference and nothing else.

/// One calibration day: the strap's raw movement signal, and a reference step count for the SAME day.
#[derive(Clone, Copy, Debug)]
pub struct StepsPoint {
    pub raw: f64,
    pub steps: f64,
}

/// The fitted (or hand-set) personal model.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StepsCalibration {
    pub coefficient: f64,
    pub sample_days: u32,
    pub confidence: f64,
    pub manual: bool,
}

/// What differs per strap family: the scale of `raw`, and therefore the floor below which a day is
/// too still to fit or to estimate from. The MODEL either side of this is identical.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StepsCfg {
    pub min_raw_for_fit: f64,
    pub max_daily_steps: u32,
}

impl StepsCfg {
    /// Gravity-delta volume: a day must move at least one whole unit to say anything.
    pub const GEN4: StepsCfg = StepsCfg { min_raw_for_fit: 1.0, max_daily_steps: 60_000 };
    /// Motion ticks. One tick is already a movement event, so the floor is one tick.
    pub const GEN5: StepsCfg = StepsCfg { min_raw_for_fit: 1.0, max_daily_steps: 60_000 };
}

/// Fewest days carrying both a raw signal and a reference count before an auto-fit is offered. ONE:
/// the same schedule `calibration::STEPS_GEN4` already publishes. A single overlapping day pins the
/// through-origin line, and the confidence it earns says how much to trust it.
pub const MIN_CALIBRATION_DAYS: usize = 1;
/// Day count at which confidence saturates toward 1.
pub const GOOD_CALIBRATION_DAYS: f64 = 14.0;


/// Where a steps figure's `k` came from, as numbers only. The frontend words it; nothing here is a
/// string, so a locale change can never move a coefficient.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum StepsState {
    /// The user asserted `k` by hand.
    Manual { coefficient: f64 },
    /// Fitted from `days` days that carried both a raw signal and a reference count.
    Fitted { coefficient: f64, days: u32, confidence: f64 },
    /// Nothing usable to fit from yet. `have` is the count of usable days seen.
    Uncalibrated { have: u32, need: u32 },
}

impl StepsState {
    /// The `k` in force, or `None` when uncalibrated.
    pub fn coefficient(&self) -> Option<f64> {
        match self {
            StepsState::Manual { coefficient } | StepsState::Fitted { coefficient, .. } => Some(*coefficient),
            StepsState::Uncalibrated { .. } => None,
        }
    }

    /// 0..1. A hand-set `k` is 1.0: the user asserted it.
    pub fn confidence(&self) -> f64 {
        match self {
            StepsState::Manual { .. } => 1.0,
            StepsState::Fitted { confidence, .. } => *confidence,
            StepsState::Uncalibrated { .. } => 0.0,
        }
    }
}

/// The calibration state for a strap, from its overlapping days and any hand-set `k`.
pub fn state(points: &[StepsPoint], manual: Option<f64>, cfg: StepsCfg) -> StepsState {
    match calibrate(points, manual, cfg) {
        Some(c) if c.manual => StepsState::Manual { coefficient: c.coefficient },
        Some(c) => StepsState::Fitted {
            coefficient: c.coefficient,
            days: c.sample_days,
            confidence: c.confidence,
        },
        None => StepsState::Uncalibrated {
            have: usable_days(points, cfg),
            need: MIN_CALIBRATION_DAYS as u32,
        },
    }
}

/// Days that could enter a fit: enough movement to be worth reading, and a reference count to fit to.
pub fn usable_days(points: &[StepsPoint], cfg: StepsCfg) -> u32 {
    points.iter().filter(|p| p.raw >= cfg.min_raw_for_fit && p.steps > 0.0).count() as u32
}

/// Fit `k` as a raw-weighted median of per-day `steps / raw` ratios, so one odd day cannot drag it
/// and a busy day votes harder than a near-still one. A positive `manual` short-circuits the fit.
pub fn calibrate(points: &[StepsPoint], manual: Option<f64>, cfg: StepsCfg) -> Option<StepsCalibration> {
    if let Some(k) = manual {
        if k > 0.0 {
            return Some(StepsCalibration {
                coefficient: k,
                sample_days: points.len() as u32,
                confidence: 1.0,
                manual: true,
            });
        }
    }
    let usable: Vec<(f64, f64)> = points
        .iter()
        .filter(|p| p.raw >= cfg.min_raw_for_fit && p.steps > 0.0)
        .map(|p| (p.steps / p.raw, p.raw))
        .collect();
    if usable.len() < MIN_CALIBRATION_DAYS {
        return None;
    }
    let ratios: Vec<f64> = usable.iter().map(|r| r.0).collect();
    let weights: Vec<f64> = usable.iter().map(|r| r.1).collect();
    let k = weighted_median(&ratios, &weights);
    if k <= 0.0 {
        return None;
    }
    // Confidence grows with sample size and shrinks with relative spread, so a noisy fit is honestly
    // less trusted than a tight one. The MAD is weighted by the same days that drove `k`.
    let size_term = (usable.len() as f64 / GOOD_CALIBRATION_DAYS).min(1.0);
    let devs: Vec<f64> = ratios.iter().map(|r| (r - k).abs()).collect();
    let mad = weighted_median(&devs, &weights);
    let tightness = (1.0 - mad / k).max(0.0);
    Some(StepsCalibration {
        coefficient: k,
        sample_days: usable.len() as u32,
        confidence: (0.5 * size_term + 0.5 * tightness).clamp(0.0, 1.0),
        manual: false,
    })
}

/// Steps for one day from its raw signal. `None` below the family floor, so "too still to say" stays
/// distinct from a real zero and the UI can show a dash rather than a fabricated 0.
pub fn estimate(raw: f64, cal: &StepsCalibration, cfg: StepsCfg) -> Option<u32> {
    if raw < cfg.min_raw_for_fit || cal.coefficient <= 0.0 {
        return None;
    }
    let steps = (raw * cal.coefficient).round();
    Some(steps.clamp(0.0, cfg.max_daily_steps as f64) as u32)
}

/// Weighted median: sort by value, walk the cumulative weight, take the value where it first passes
/// half the total. On an exact half-mass boundary average the two straddling values, which reduces to
/// the plain even-count midpoint at equal weights. Degenerate weights fall back to the plain median.
pub fn weighted_median(xs: &[f64], weights: &[f64]) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    if weights.len() != xs.len() {
        return crate::stats::median(xs);
    }
    let total: f64 = weights.iter().sum();
    if total <= 0.0 {
        return crate::stats::median(xs);
    }
    let mut order: Vec<usize> = (0..xs.len()).collect();
    order.sort_by(|a, b| xs[*a].partial_cmp(&xs[*b]).unwrap_or(std::cmp::Ordering::Equal));
    let half = total / 2.0;
    let mut cum = 0.0;
    for pos in 0..order.len() {
        let idx = order[pos];
        cum += weights[idx].max(0.0);
        if cum > half {
            return xs[idx];
        }
        if cum == half {
            let next = if pos + 1 < order.len() { order[pos + 1] } else { idx };
            return (xs[idx] + xs[next]) / 2.0;
        }
    }
    xs[order[order.len() - 1]]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn step(ts: i64, counter: u16) -> StepSample {
        StepSample { ts, counter, activity_class: None }
    }

    #[test]
    fn sums_positive_consecutive_tick_deltas() {
        assert_eq!(steps_in_window(&[step(0, 100), step(60, 150), step(120, 220)]), Some(120));
    }

    #[test]
    fn sorts_unordered_input() {
        assert_eq!(steps_in_window(&[step(120, 220), step(0, 100), step(60, 150)]), Some(120));
    }

    /// Two stand-ins a bare "returns 120" gate would accept: counting samples, and subtracting the
    /// first counter from the last. Both agree with the kernel on a clean series and diverge as soon
    /// as a wrap or a reboot boundary appears, which is the whole reason the kernel exists.
    #[test]
    fn a_sample_count_and_a_first_to_last_subtraction_both_fail_the_real_series() {
        let clean = [step(0, 100), step(60, 150), step(120, 220)];
        let wrapped = [step(0, 65_500), step(60, 20), step(120, 80)];
        let rebooted = [step(0, 100), step(60, 140), step(120, 5000), step(180, 5030)];

        let count_scorer = |s: &[StepSample]| Some(s.len() as u32);
        let span_scorer = |s: &[StepSample]| Some(s[s.len() - 1].counter.wrapping_sub(s[0].counter) as u32);

        for series in [clean.as_slice(), wrapped.as_slice(), rebooted.as_slice()] {
            assert_ne!(steps_in_window(series), count_scorer(series), "a sample count is not a tick total");
        }
        // The span scorer only survives while nothing crosses a boundary.
        assert_eq!(steps_in_window(&clean), span_scorer(&clean));
        assert_eq!(steps_in_window(&wrapped), span_scorer(&wrapped));
        assert_eq!(steps_in_window(&rebooted), Some(70));
        assert_eq!(span_scorer(&rebooted), Some(4930), "a reboot jump billed as motion");
    }

    /// What this module does NOT measure: the caller's `stepTicksPerStep` divisor. Every figure here
    /// is raw motion ticks, so no gate in this crate can tell a right step count from a wrong one.
    #[test]
    fn nothing_here_converts_ticks_to_steps() {
        let ticks = steps_in_window(&[step(0, 0), step(60, 400), step(120, 800)]).unwrap();
        assert_eq!(ticks, 800, "the return value is ticks, before any per-user calibration");
        // Two plausible divisors give two step counts from the same ticks; neither is checked here.
        assert_eq!(ticks / 2, 400);
        assert_eq!(ticks / 4, 200);
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

    fn pt(raw: f64, steps: f64) -> StepsPoint {
        StepsPoint { raw, steps }
    }

    #[test]
    fn a_positive_manual_k_short_circuits_the_fit() {
        let cal = calibrate(&[pt(100.0, 500.0)], Some(7.5), StepsCfg::GEN4).unwrap();
        assert_eq!(cal.coefficient, 7.5);
        assert!(cal.manual);
        assert_eq!(cal.confidence, 1.0);
        // A zero or negative manual is not an override, it means "auto" - so the fit runs instead.
        let auto = calibrate(&[pt(100.0, 500.0)], Some(0.0), StepsCfg::GEN4).unwrap();
        assert!(!auto.manual);
        assert_eq!(auto.coefficient, 5.0);
    }

    /// ONE overlapping day is enough, matching the schedule `calibration::STEPS_GEN4` publishes. It is
    /// trusted only as far as its confidence says, which is what makes a one-day fit safe to offer.
    #[test]
    fn one_overlapping_day_fits_and_earns_only_middling_confidence() {
        let one = [pt(100.0, 500.0)];
        let cal = calibrate(&one, None, StepsCfg::GEN4).unwrap();
        assert_eq!(cal.coefficient, 5.0);
        assert_eq!(cal.sample_days, 1);
        assert!((0.5..0.6).contains(&cal.confidence), "one day: {}", cal.confidence);
        assert!(calibrate(&[], None, StepsCfg::GEN4).is_none(), "no days fits nothing");
    }

    #[test]
    fn the_state_reports_where_k_came_from_without_wording_it() {
        assert_eq!(
            state(&[pt(100.0, 500.0)], Some(7.5), StepsCfg::GEN4),
            StepsState::Manual { coefficient: 7.5 },
        );
        match state(&[pt(100.0, 500.0)], None, StepsCfg::GEN4) {
            StepsState::Fitted { coefficient, days, .. } => {
                assert_eq!((coefficient, days), (5.0, 1));
            }
            other => panic!("expected Fitted, got {other:?}"),
        }
        // A day too still to read is counted as seen but not as usable.
        assert_eq!(
            state(&[pt(0.5, 900.0)], None, StepsCfg::GEN4),
            StepsState::Uncalibrated { have: 0, need: 1 },
        );
        assert_eq!(state(&[], None, StepsCfg::GEN4).coefficient(), None);
        assert_eq!(state(&[], Some(3.0), StepsCfg::GEN4).confidence(), 1.0);
    }

    /// The weighting is the point: a busy day pins the ratio harder than a near-still one.
    #[test]
    fn the_fit_is_weighted_by_raw_so_a_still_day_cannot_drag_it() {
        let points = [pt(10.0, 100.0), pt(1000.0, 5000.0), pt(1000.0, 5000.0)];
        let cal = calibrate(&points, None, StepsCfg::GEN4).unwrap();
        assert_eq!(cal.coefficient, 5.0, "the two busy days at 5.0 outweigh the still day at 10.0");
    }

    #[test]
    fn a_day_below_the_family_floor_never_enters_the_fit_or_produces_an_estimate() {
        let points = [pt(0.5, 900.0), pt(100.0, 500.0), pt(200.0, 1000.0), pt(300.0, 1500.0)];
        let cal = calibrate(&points, None, StepsCfg::GEN4).unwrap();
        assert_eq!(cal.sample_days, 3, "the sub-floor day is excluded");
        assert_eq!(cal.coefficient, 5.0);
        assert_eq!(estimate(0.5, &cal, StepsCfg::GEN4), None, "too still to say, not zero");
    }

    #[test]
    fn confidence_rises_with_days_and_falls_with_spread() {
        let tight: Vec<StepsPoint> = (1..=14).map(|i| pt(100.0 * i as f64, 500.0 * i as f64)).collect();
        let tight_cal = calibrate(&tight, None, StepsCfg::GEN4).unwrap();
        assert!(tight_cal.confidence > 0.99, "14 days on an exact line: {}", tight_cal.confidence);

        let noisy = [pt(100.0, 200.0), pt(100.0, 500.0), pt(100.0, 1500.0)];
        let noisy_cal = calibrate(&noisy, None, StepsCfg::GEN4).unwrap();
        assert!(noisy_cal.confidence < tight_cal.confidence);
    }

    #[test]
    fn an_estimate_is_the_line_through_the_origin_and_is_clamped() {
        let cal = StepsCalibration { coefficient: 5.0, sample_days: 5, confidence: 1.0, manual: false };
        assert_eq!(estimate(1000.0, &cal, StepsCfg::GEN4), Some(5000));
        assert_eq!(estimate(1_000_000.0, &cal, StepsCfg::GEN4), Some(StepsCfg::GEN4.max_daily_steps));
    }

    /// The reason this module is shared: a 5.0's ticks and a 4.0's motion volume run the SAME line.
    /// Only the input differs, so the same points fit the same k under either family config.
    #[test]
    fn both_families_run_the_same_model() {
        let points = [pt(1000.0, 5000.0), pt(2000.0, 10000.0), pt(3000.0, 15000.0)];
        let g4 = calibrate(&points, None, StepsCfg::GEN4).unwrap();
        let g5 = calibrate(&points, None, StepsCfg::GEN5).unwrap();
        assert_eq!(g4.coefficient, g5.coefficient);
        assert_eq!(estimate(1500.0, &g4, StepsCfg::GEN4), estimate(1500.0, &g5, StepsCfg::GEN5));
    }

    /// A 5.0 whose divisor is the shipped 1.0 default reads its raw tick count as its step count.
    /// Pinned because that is the behaviour on a strap nobody has calibrated.
    #[test]
    fn the_gen5_default_divisor_is_a_pass_through() {
        let cal = calibrate(&[], Some(1.0), StepsCfg::GEN5).unwrap();
        assert_eq!(estimate(6953.0, &cal, StepsCfg::GEN5), Some(6953));
    }

    #[test]
    fn weighted_median_matches_the_plain_median_at_equal_weights() {
        let xs = [1.0, 2.0, 3.0, 4.0];
        assert_eq!(weighted_median(&xs, &[1.0; 4]), 2.5, "even count averages the middle pair");
        let odd = [1.0, 2.0, 3.0];
        assert_eq!(weighted_median(&odd, &[1.0; 3]), 2.0);
    }

    #[test]
    fn weighted_median_falls_back_when_the_weights_are_degenerate() {
        let xs = [1.0, 2.0, 3.0];
        assert_eq!(weighted_median(&xs, &[1.0, 2.0]), 2.0, "mismatched lengths");
        assert_eq!(weighted_median(&xs, &[0.0, 0.0, 0.0]), 2.0, "zero total");
        assert_eq!(weighted_median(&[], &[]), 0.0);
    }
}
