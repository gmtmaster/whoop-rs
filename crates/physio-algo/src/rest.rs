//! Rest (sleep performance) composite, 0–100. Weighted sum of duration vs personal need, efficiency,
//! restorative (deep+REM) share, and sleep/wake consistency. Pure; absent consistency defaults neutral.

const W_DURATION: f64 = 0.50;
const W_EFFICIENCY: f64 = 0.20;
const W_RESTORATIVE: f64 = 0.20;
const W_CONSISTENCY: f64 = 0.10;
const DEFAULT_SLEEP_NEED_HOURS: f64 = 8.0;
const RESTORATIVE_TARGET_SHARE: f64 = 0.50;
const DEEP_SHARE_TARGET: f64 = 0.13;
const DEEP_FLOOR_FACTOR: f64 = 0.5;
const NEUTRAL_CONSISTENCY: f64 = 0.5;

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perfect_night_scores_high() {
        // 8h asleep, 95% efficiency, 30% deep, 25% REM, 8h need, perfect consistency
        let r = rest(8.0 * 3600.0, 0.95, 0.30 * 8.0 * 3600.0, 0.25 * 8.0 * 3600.0, Some(8.0), Some(1.0));
        assert!(r.unwrap() > 85.0, "got {}", r.unwrap());
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
