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
    use crate::stress::{band_of, StressBand};

    /// A z-sum of +2 (one standard deviation of RHR up and one of HRV down) on the shared scale.
    const STRESSED_SCORE: f64 = 2.642391233933647;
    /// A z-sum of +1: RHR up one SD with no HRV to read.
    const RHR_ONLY_SCORE: f64 = 2.193175735890015;
    /// A z-sum of -2: RHR down one SD and HRV up one.
    const CALM_SCORE: f64 = 0.35760876606635335;

    fn day(rhr: Option<f64>, hrv: Option<f64>) -> StressDay {
        StressDay { rhr, hrv }
    }

    /// `n` days alternating 50/60 bpm and 65/55 ms. At an even `n` the RHR mean is exactly 55 with a
    /// population SD of exactly 5, and the HRV mean exactly 60 with the same SD, so a z of one is a
    /// round step the assertions can name.
    fn spread_baseline(n: usize) -> Vec<StressDay> {
        (0..n)
            .map(|i| if i % 2 == 0 { day(Some(50.0), Some(65.0)) } else { day(Some(60.0), Some(55.0)) })
            .collect()
    }

    /// RHR-only history over the same 50/60 alternation.
    fn rhr_only_baseline(n: usize) -> Vec<StressDay> {
        (0..n).map(|i| if i % 2 == 0 { day(Some(50.0), None) } else { day(Some(60.0), None) }).collect()
    }

    /// The rows the module must reproduce: today, the history behind it, and the score it must yield.
    fn table() -> Vec<(&'static str, StressDay, Vec<StressDay>, Option<f64>)> {
        vec![
            ("no history at all", day(Some(60.0), Some(55.0)), vec![], None),
            ("one day short", day(Some(60.0), Some(55.0)), spread_baseline(BASELINE_DAYS - 1), None),
            ("at the minimum", day(Some(60.0), Some(55.0)), spread_baseline(BASELINE_DAYS), Some(STRESSED_SCORE)),
            ("today has nothing", day(None, None), spread_baseline(20), None),
            ("on the baseline mean", day(Some(55.0), Some(60.0)), spread_baseline(20), Some(1.5)),
            ("a calm day", day(Some(50.0), Some(65.0)), spread_baseline(20), Some(CALM_SCORE)),
            ("RHR only", day(Some(60.0), None), rhr_only_baseline(20), Some(RHR_ONLY_SCORE)),
        ]
    }

    fn reproduces(scorer: impl Fn(StressDay, &[StressDay]) -> Option<f64>) -> bool {
        table().into_iter().all(|(_, today, base, want)| match (scorer(today, &base), want) {
            (Some(got), Some(w)) => (got - w).abs() < 1e-12,
            (None, None) => true,
            _ => false,
        })
    }

    #[test]
    fn the_shipped_score_reproduces_the_table_and_three_do_nothing_scorers_do_not() {
        for (name, today, base, want) in table() {
            match (daily_stress(today, &base), want) {
                (Some(got), Some(w)) => assert!((got - w).abs() < 1e-12, "{name}: got {got}, want {w}"),
                (None, None) => {}
                (got, w) => panic!("{name}: got {got:?}, want {w:?}"),
            }
        }
        assert!(reproduces(daily_stress));
        // Stand-ins that read no baseline: one refuses everything, one parks on the neutral point,
        // one always calls the day stressed. Each must miss at least one row.
        type Null = fn(StressDay, &[StressDay]) -> Option<f64>;
        let nulls: [(&str, Null); 3] = [
            ("always none", |_, _| None),
            ("always neutral", |_, _| Some(1.5)),
            ("always stressed", |_, _| Some(STRESSED_SCORE)),
        ];
        for (name, null) in nulls {
            assert!(!reproduces(null), "{name} reproduced every row; the table cannot tell it apart");
        }
    }

    #[test]
    fn the_baseline_minimum_is_the_edge_between_none_and_a_score() {
        assert_eq!(BASELINE_DAYS, 14);
        let today = day(Some(60.0), Some(55.0));
        assert!(daily_stress(today, &[]).is_none(), "no history at all must refuse");
        assert!(
            daily_stress(today, &spread_baseline(BASELINE_DAYS - 1)).is_none(),
            "one day short must refuse"
        );
        let s = daily_stress(today, &spread_baseline(BASELINE_DAYS)).expect("the minimum itself scores");
        assert!((s - squash(2.0)).abs() < 1e-12, "one SD on each channel is a z-sum of 2, got {s}");
        assert!((s - STRESSED_SCORE).abs() < 1e-12, "got {s}");
    }

    #[test]
    fn a_day_with_no_signal_of_its_own_is_none_however_long_the_history() {
        assert!(daily_stress(day(None, None), &spread_baseline(30)).is_none());
        // A history that never reported either channel cannot anchor today either.
        let blank: Vec<StressDay> = (0..30).map(|_| day(None, None)).collect();
        assert!(daily_stress(day(Some(60.0), Some(55.0)), &blank).is_none());
    }

    #[test]
    fn a_day_on_the_baseline_mean_lands_exactly_on_the_neutral_point() {
        let s = daily_stress(day(Some(55.0), Some(60.0)), &spread_baseline(20)).unwrap();
        assert!((s - 1.5).abs() < 1e-12, "a zero z-sum is squash(0) = 1.5, got {s}");
        assert_eq!(band_of(s), StressBand::Medium);
    }

    #[test]
    fn rhr_up_and_hrv_down_both_push_the_score_the_same_way() {
        let base = spread_baseline(20);
        let calm = daily_stress(day(Some(50.0), Some(65.0)), &base).unwrap();
        let neutral = daily_stress(day(Some(55.0), Some(60.0)), &base).unwrap();
        let stressed = daily_stress(day(Some(60.0), Some(55.0)), &base).unwrap();
        assert!(calm < neutral && neutral < stressed, "{calm} < {neutral} < {stressed}");
        assert!((calm - CALM_SCORE).abs() < 1e-12, "got {calm}");
        assert!((stressed - STRESSED_SCORE).abs() < 1e-12, "got {stressed}");
        assert_eq!(band_of(calm), StressBand::Low);
        assert_eq!(band_of(stressed), StressBand::High);
    }

    #[test]
    fn one_channel_alone_still_scores_on_its_own_z() {
        let s = daily_stress(day(Some(60.0), None), &rhr_only_baseline(20)).unwrap();
        assert!((s - squash(1.0)).abs() < 1e-12, "RHR alone is a z-sum of 1, got {s}");
        assert!((s - RHR_ONLY_SCORE).abs() < 1e-12, "got {s}");
    }

    #[test]
    fn a_flat_history_has_no_scale_so_every_day_reads_neutral() {
        let flat: Vec<StressDay> = (0..20).map(|_| day(Some(55.0), Some(60.0))).collect();
        assert!(population_std(&[55.0; 20], Some(55.0)) <= SD_FLOOR);
        for today in [day(Some(65.0), Some(60.0)), day(Some(45.0), Some(60.0)), day(Some(55.0), Some(60.0))] {
            let s = daily_stress(today, &flat).unwrap();
            assert!((s - 1.5).abs() < 1e-12, "an SD under the floor contributes nothing, got {s}");
        }
    }
}
