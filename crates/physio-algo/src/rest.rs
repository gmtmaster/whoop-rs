//! Rest (sleep performance) composite, 0–100. Weighted sum of duration vs personal need, efficiency,
//! restorative (deep+REM) share, and sleep/wake consistency. Pure; absent consistency defaults neutral.

pub const W_DURATION: f64 = 0.50;
pub const W_EFFICIENCY: f64 = 0.20;
pub const W_RESTORATIVE: f64 = 0.20;
pub const W_CONSISTENCY: f64 = 0.10;
pub const DEFAULT_SLEEP_NEED_HOURS: f64 = 8.0;
pub const RESTORATIVE_TARGET_SHARE: f64 = 0.50;
pub const DEEP_SHARE_TARGET: f64 = 0.13;
pub const DEEP_FLOOR_FACTOR: f64 = 0.5;
pub const NEUTRAL_CONSISTENCY: f64 = 0.5;

/// Rest composite, or `None` when there is no asleep time.
pub fn rest(
    asleep_seconds: f64,
    efficiency: f64,
    deep_seconds: f64,
    rem_seconds: f64,
    sleep_need_hours: Option<f64>,
    consistency: Option<f64>,
) -> Option<f64> {
    if asleep_seconds <= 0.0 {
        return None;
    }
    let asleep_hours = asleep_seconds / 3600.0;
    let need_hours = sleep_need_hours.unwrap_or(DEFAULT_SLEEP_NEED_HOURS).max(1e-9);

    let duration_score = (asleep_hours / need_hours * 100.0).min(100.0);
    let efficiency_score = (efficiency * 100.0).clamp(0.0, 100.0);
    let restorative_share = (deep_seconds + rem_seconds) / asleep_seconds;
    let deep_adequacy = ((deep_seconds / asleep_seconds) / DEEP_SHARE_TARGET).clamp(0.0, 1.0);
    let deep_factor = DEEP_FLOOR_FACTOR + (1.0 - DEEP_FLOOR_FACTOR) * deep_adequacy;
    let restorative_score =
        (restorative_share / RESTORATIVE_TARGET_SHARE * 100.0).min(100.0) * deep_factor;
    let consistency_score = ((consistency.unwrap_or(NEUTRAL_CONSISTENCY)) * 100.0).clamp(0.0, 100.0);

    let weighted = W_DURATION * duration_score
        + W_EFFICIENCY * efficiency_score
        + W_RESTORATIVE * restorative_score
        + W_CONSISTENCY * consistency_score;
    Some((weighted * 100.0).round() / 100.0)
}

/// Healthy floor for the personal sleep-need estimate (hours).
pub const MIN_SLEEP_NEED_HOURS: f64 = 7.5;

/// Personal sleep need (hours) = the mean of recent nightly asleep hours, floored at [MIN_SLEEP_NEED_HOURS].
/// Non-positive nights are ignored; an empty window returns the floor. Feeds `rest`'s `sleep_need_hours`.
pub fn personal_sleep_need_hours(recent_asleep_hours: &[f64]) -> f64 {
    let mut sum = 0.0;
    let mut n = 0u32;
    for &h in recent_asleep_hours {
        if h > 0.0 {
            sum += h;
            n += 1;
        }
    }
    if n == 0 {
        return MIN_SLEEP_NEED_HOURS;
    }
    (sum / n as f64).max(MIN_SLEEP_NEED_HOURS)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A stand-in sleep-need estimator, so a do-nothing one can be run against the same rows.
    type NullNeed = fn(&[f64]) -> f64;

    /// The need is the mean of the WORN nights only, floored. `[9, 0, 9]` separates that from a plain
    /// mean over the whole window: ignoring the blank gives 9.0, counting it gives 6.0 and floors to 7.5.
    #[test]
    fn personal_need_averages_worn_nights_only_and_floors_at_seven_and_a_half() {
        assert_eq!(personal_sleep_need_hours(&[]), MIN_SLEEP_NEED_HOURS);
        assert_eq!(personal_sleep_need_hours(&[6.0, 0.0, 6.5]), MIN_SLEEP_NEED_HOURS); // mean 6.25 < floor
        assert!((personal_sleep_need_hours(&[8.0, 9.0, 8.5]) - 8.5).abs() < 1e-9); // mean above the floor
        assert!((personal_sleep_need_hours(&[9.0, 0.0, 9.0]) - 9.0).abs() < 1e-9); // the blank is not a night

        // Every do-nothing need disagrees with at least one row above, so none of them can pass this test.
        let cases: [&[f64]; 4] = [&[], &[6.0, 0.0, 6.5], &[8.0, 9.0, 8.5], &[9.0, 0.0, 9.0]];
        let want = [MIN_SLEEP_NEED_HOURS, MIN_SLEEP_NEED_HOURS, 8.5, 9.0];
        let nulls: [(&str, NullNeed); 3] = [
            ("always the floor", |_| MIN_SLEEP_NEED_HOURS),
            ("always the default", |_| DEFAULT_SLEEP_NEED_HOURS),
            ("plain mean, blanks counted", |v| {
                if v.is_empty() {
                    MIN_SLEEP_NEED_HOURS
                } else {
                    (v.iter().sum::<f64>() / v.len() as f64).max(MIN_SLEEP_NEED_HOURS)
                }
            }),
        ];
        for (name, f) in nulls {
            assert!(
                cases.iter().zip(want).any(|(c, w)| (f(c) - w).abs() > 1e-9),
                "the '{name}' scorer reproduces every row, so this gate is blind to it"
            );
        }
    }

    /// The need reaches the composite only through the duration term, so a one-hour change in need moves
    /// Rest by `W_DURATION` times the duration score it shifts, and nothing else.
    #[test]
    fn the_personal_need_enters_rest_at_the_duration_weight() {
        let night = |need: f64| {
            rest(8.0 * 3600.0, 0.95, 0.30 * 8.0 * 3600.0, 0.25 * 8.0 * 3600.0, Some(need), Some(1.0)).unwrap()
        };
        let at_need = personal_sleep_need_hours(&[9.0, 0.0, 9.0]); // 9.0 h
        let floored = personal_sleep_need_hours(&[6.0, 0.0, 6.5]); // 7.5 h, so duration saturates at 100
        assert!((at_need - 9.0).abs() < 1e-9 && (floored - MIN_SLEEP_NEED_HOURS).abs() < 1e-9);
        // 8 h against a 9 h need scores 88.888…; against 7.5 h it saturates at 100.
        let duration_delta = 100.0 - (8.0 / 9.0 * 100.0);
        assert!((night(floored) - night(at_need) - W_DURATION * duration_delta).abs() < 0.01);
        assert!(night(floored) > night(at_need), "a lower need cannot score a night worse");
    }

    /// Rest is exactly the four weighted terms. The perfect night is 99.0 = 0.5*100 + 0.2*95 + 0.2*100 +
    /// 0.1*100, and moving one input moves the total by that weight times the term it changed.
    #[test]
    fn each_rest_weight_moves_the_composite_by_its_own_share() {
        let perfect = rest(8.0 * 3600.0, 0.95, 0.30 * 8.0 * 3600.0, 0.25 * 8.0 * 3600.0, Some(8.0), Some(1.0)).unwrap();
        assert_eq!(99.0, perfect);
        // Duration 100 -> 75: six hours against the same eight-hour need.
        let short = rest(6.0 * 3600.0, 0.95, 0.30 * 6.0 * 3600.0, 0.25 * 6.0 * 3600.0, Some(8.0), Some(1.0)).unwrap();
        assert!((perfect - short - W_DURATION * 25.0).abs() < 1e-9, "duration weight, got {short}");
        // Efficiency 95 -> 85.
        let leaky = rest(8.0 * 3600.0, 0.85, 0.30 * 8.0 * 3600.0, 0.25 * 8.0 * 3600.0, Some(8.0), Some(1.0)).unwrap();
        assert!((perfect - leaky - W_EFFICIENCY * 10.0).abs() < 1e-9, "efficiency weight, got {leaky}");
        // Restorative 100 -> 50: a 0.25 deep+REM share against the 0.50 target, deep still at its own.
        let thin = rest(8.0 * 3600.0, 0.95, 0.13 * 8.0 * 3600.0, 0.12 * 8.0 * 3600.0, Some(8.0), Some(1.0)).unwrap();
        assert!((perfect - thin - W_RESTORATIVE * 50.0).abs() < 1e-9, "restorative weight, got {thin}");
        // Consistency 100 -> 0.
        let erratic = rest(8.0 * 3600.0, 0.95, 0.30 * 8.0 * 3600.0, 0.25 * 8.0 * 3600.0, Some(8.0), Some(0.0)).unwrap();
        assert!((perfect - erratic - W_CONSISTENCY * 100.0).abs() < 1e-9, "consistency weight, got {erratic}");

        assert!((1.0 - (W_DURATION + W_EFFICIENCY + W_RESTORATIVE + W_CONSISTENCY)).abs() < 1e-12);
        // No constant scorer reproduces the five nights. The thin and erratic ones both land on 89.0, so
        // that pair alone cannot separate the two weights; each is pinned by its own delta above.
        let all = [perfect, short, leaky, thin, erratic];
        assert_eq!(4, {
            let mut d: Vec<f64> = all.to_vec();
            d.sort_by(|a, b| a.partial_cmp(b).expect("rest returns finite scores"));
            d.dedup_by(|a, b| (*a - *b).abs() < 1e-9);
            d.len()
        });
        for c in all {
            assert!(all.iter().any(|v| (v - c).abs() > 1e-9), "the constant {c} would satisfy every night");
        }
    }

    #[test]
    fn short_night_scores_low() {
        // 4h asleep, 80% efficiency, same stage proportions
        let r = rest(4.0 * 3600.0, 0.80, 0.30 * 4.0 * 3600.0, 0.25 * 4.0 * 3600.0, Some(8.0), Some(0.5));
        let v = r.unwrap();
        assert!(v < 70.0 && v > 30.0, "got {v}");
    }

    #[test]
    fn zero_deep_halves_restorative() {
        // No deep sleep → deepFactor = 0.5 → restorative term halved
        let normal = rest(8.0 * 3600.0, 0.90, 0.25 * 8.0 * 3600.0, 0.25 * 8.0 * 3600.0, Some(8.0), Some(1.0));
        let no_deep = rest(8.0 * 3600.0, 0.90, 0.0, 0.25 * 8.0 * 3600.0, Some(8.0), Some(1.0));
        assert!(normal.unwrap() > no_deep.unwrap());
    }

    #[test]
    fn no_asleep_returns_null() {
        assert!(rest(0.0, 0.90, 3600.0, 3600.0, None, None).is_none());
        assert!(rest(-1.0, 0.90, 3600.0, 3600.0, None, None).is_none());
    }

    #[test]
    fn consistency_defaults_neutral() {
        let with_neutral = rest(8.0 * 3600.0, 0.90, 7200.0, 7200.0, Some(8.0), Some(0.5));
        let with_default = rest(8.0 * 3600.0, 0.90, 7200.0, 7200.0, Some(8.0), None);
        assert!((with_neutral.unwrap() - with_default.unwrap()).abs() < 0.01);
    }
}
