//! Daily autonomic stress: today's resting HR and HRV z-scored against a personal baseline of
//! prior days, squashed onto the shared 0–3 scale. RHR up and HRV down both push it high.

use super::{mean_opt, population_std, squash, SD_FLOOR};

/// Prior days required before a baseline is trusted.
const BASELINE_DAYS: usize = 14;

/// One day's input for the daily-stress baseline.
#[derive(Clone, Copy, Debug)]
pub struct StressDay {
    pub rhr: Option<f64>,
    pub hrv: Option<f64>,
}

/// Daily autonomic stress (0–3) from today's RHR+HRV against a prior-days baseline.
/// Returns `None` when there are too few baseline days or today has no signal.
pub fn daily_stress(today: StressDay, baseline: &[StressDay]) -> Option<f64> {
    if baseline.len() < BASELINE_DAYS {
        return None;
    }
    let rhr_base: Vec<f64> = baseline.iter().filter_map(|d| d.rhr).collect();
    let hrv_base: Vec<f64> = baseline.iter().filter_map(|d| d.hrv).collect();

    let mean_rhr = mean_opt(&rhr_base);
    let sd_rhr = population_std(&rhr_base, mean_rhr);
    let mean_hrv = mean_opt(&hrv_base);
    let sd_hrv = population_std(&hrv_base, mean_hrv);

    let has_signal = (today.rhr.is_some() && mean_rhr.is_some())
        || (today.hrv.is_some() && mean_hrv.is_some());
    if !has_signal {
        return None;
    }
    let mut raw = 0.0;
    if let (Some(r), Some(m)) = (today.rhr, mean_rhr) {
        if sd_rhr > SD_FLOOR {
            raw += (r - m) / sd_rhr;
        }
    }
    if let (Some(h), Some(m)) = (today.hrv, mean_hrv) {
        if sd_hrv > SD_FLOOR {
            raw += (m - h) / sd_hrv;
        }
    }
    Some(squash(raw))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn day(rhr: Option<f64>, hrv: Option<f64>) -> StressDay {
        StressDay { rhr, hrv }
    }

    #[test]
    fn cold_start_rejects() {
        let today = day(Some(55.0), Some(60.0));
        assert!(daily_stress(today, &[]).is_none());
        assert!(daily_stress(today, &[day(Some(55.0), Some(60.0)); 5]).is_none());
    }

    #[test]
    fn stable_day_scores_neutral() {
        let baseline: Vec<StressDay> = (0..20).map(|_| day(Some(55.0), Some(60.0))).collect();
        let today = day(Some(55.0), Some(60.0));
        let s = daily_stress(today, &baseline).unwrap();
        assert!((s - 1.5).abs() < 0.1, "stable day should be ~1.5, got {s}");
    }

    #[test]
    fn elevated_rhr_low_hrv_scores_high() {
        // Baseline with realistic spread (SD ~5 bpm, ~5 ms)
        let baseline: Vec<StressDay> = (0..20).map(|i| day(Some(50.0 + (i % 10) as f64), Some(55.0 + (i % 10) as f64))).collect();
        let today = day(Some(65.0), Some(45.0)); // RHR well above, HRV well below
        let s = daily_stress(today, &baseline).unwrap();
        assert!(s > 2.0, "stressed day should be high, got {s}");
    }

    #[test]
    fn only_rhr_still_scores() {
        let baseline: Vec<StressDay> = (0..20).map(|i| day(Some(50.0 + (i % 10) as f64), None)).collect();
        let today = day(Some(65.0), None);
        let s = daily_stress(today, &baseline).unwrap();
        assert!(s > 1.5, "RHR-only stress should score, got {s}");
    }

    #[test]
    fn zero_spread_returns_neutral() {
        let baseline: Vec<StressDay> = (0..20).map(|_| day(Some(55.0), Some(60.0))).collect();
        let today = day(Some(65.0), Some(60.0));
        let s = daily_stress(today, &baseline).unwrap();
        assert!((s - 1.5).abs() < 0.1, "zero-spread baseline → always neutral, got {s}");
    }
}
