//! HRV-readiness: a log-domain rolling-baseline vs personal-normal-band reading over a nightly RMSSD
//! series. Input = the nightly RMSSD (ms) derived from decoded R-R; output = a tier + the band in ms.
//! Returns `None` (calibrating) below `MIN_NIGHTS` valid nights. Wellness only, never medical.

use crate::stats::{least_squares_slope, mean, sample_sd};
use whoop_protocol::HistoryRecord;

/// Seconds per calendar day, the bucket a nightly metric is keyed on.
pub const SECS_PER_DAY: u32 = 86_400;

const HRV_MIN_MS: f64 = 5.0;
const HRV_MAX_MS: f64 = 250.0;
const ROLL_WINDOW: usize = 7;
const LONG_WINDOW: usize = 60;
const LONG_WINDOW_FALLBACK: usize = 30;
const SWC_K: f64 = 0.5;
// The readiness tier mirrors WHOOP's recovery score, which unlocks after 3 recoveries; the 7-night
// baseline and long-window band keep refining past that.
const MIN_NIGHTS: usize = crate::calibration::RECOVERY_SCORE.unlock as usize;
const CV_TREND_WINDOW: usize = 28;
const LONG_SD_FLOOR: f64 = 1e-9;
/// R-R records more than this many seconds apart start a new run (an offload gap breaks the beat chain).
const MAX_GAP_S: u32 = 5;
/// A physiologically plausible R-R interval (ms); values outside break the chain rather than inflate RMSSD.
const RR_MIN_MS: u16 = 300;
const RR_MAX_MS: u16 = 2000;

/// Where the short baseline sits vs the personal normal band.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReadinessTier {
    Primed,
    Normal,
    Suppressed,
}

/// A readiness reading. The `*_ms` fields are back in milliseconds (exp of the log-domain the engine
/// works in). `overreaching_watch` is informational and never changes the tier.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HrvReadinessResult {
    pub tier: ReadinessTier,
    pub baseline7_ms: f64,
    pub normal_low_ms: f64,
    pub normal_high_ms: f64,
    pub overreaching_watch: bool,
}

pub struct HrvReadiness;

impl HrvReadiness {
    /// Gap-aware RMSSD (ms): pools squared successive differences **within** each run of consecutive beats
    /// and never across the break between runs, so a jump across an offload gap or an artifact interval
    /// can't inflate it. `None` if no run has two beats. The caller splits R-R into runs on time gaps.
    pub fn rmssd_runs<'a>(runs: impl IntoIterator<Item = &'a [u16]>) -> Option<f64> {
        let (mut sumsq, mut pairs) = (0.0f64, 0usize);
        for run in runs {
            for w in run.windows(2) {
                let d = w[1] as f64 - w[0] as f64;
                sumsq += d * d;
                pairs += 1;
            }
        }
        (pairs > 0).then(|| (sumsq / pairs as f64).sqrt())
    }

    /// RMSSD (ms) of one run of consecutive R-R beats. `None` for < 2 beats.
    pub fn rmssd(rr_ms: &[u16]) -> Option<f64> {
        Self::rmssd_runs(std::iter::once(rr_ms))
    }

    /// Gap-aware nightly RMSSD (ms) from one night's per-record `(unix, R-R)` beats. Splits them into
    /// time-contiguous, physiologically-plausible runs (breaking on a gap over `MAX_GAP_S` or an interval
    /// outside the plausible band) and pools only within-run successive differences, so an offload gap or
    /// an artifact can't inflate it. Input need not be sorted. `None` if no run has two beats.
    pub fn rmssd_gap_aware(beats: &[(u32, Vec<u16>)]) -> Option<f64> {
        let mut order: Vec<&(u32, Vec<u16>)> = beats.iter().collect();
        order.sort_by_key(|(t, _)| *t);
        let mut runs: Vec<Vec<u16>> = Vec::new();
        let mut cur: Vec<u16> = Vec::new();
        let mut last: Option<u32> = None;
        for (unix, rr) in order {
            if last.is_some_and(|p| unix.saturating_sub(p) > MAX_GAP_S) {
                runs.push(std::mem::take(&mut cur));
            }
            for &ms in rr {
                if (RR_MIN_MS..=RR_MAX_MS).contains(&ms) {
                    cur.push(ms);
                } else {
                    runs.push(std::mem::take(&mut cur));
                }
            }
            last = Some(*unix);
        }
        runs.push(cur);
        Self::rmssd_runs(runs.iter().map(Vec::as_slice))
    }

    /// Per-calendar-day gap-aware RMSSD series (ms) from history records, oldest → newest, for `evaluate`.
    /// Groups records into UTC days, then applies `rmssd_gap_aware` per day. A sleep that straddles UTC
    /// midnight is split across the two days.
    pub fn nightly_rmssd(history: &[HistoryRecord]) -> Vec<Option<f64>> {
        let mut by_day: std::collections::BTreeMap<u32, Vec<(u32, Vec<u16>)>> = std::collections::BTreeMap::new();
        for h in history {
            if !h.rr_intervals.is_empty() {
                by_day.entry(h.unix / SECS_PER_DAY).or_default().push((h.unix, h.rr_intervals.clone()));
            }
        }
        by_day.values().map(|beats| Self::rmssd_gap_aware(beats)).collect()
    }

    /// Readiness over a nightly RMSSD series (ms), oldest → newest; `None` slots = missing nights.
    /// Implausible nights (outside 5..250 ms) are dropped. `None` result = calibrating.
    pub fn evaluate(nightly_rmssd_ms: &[Option<f64>]) -> Option<HrvReadinessResult> {
        let valid: Vec<f64> = nightly_rmssd_ms
            .iter()
            .filter_map(|&v| v)
            .filter(|&v| (HRV_MIN_MS..=HRV_MAX_MS).contains(&v))
            .collect();
        if valid.len() < MIN_NIGHTS {
            return None;
        }

        let ell: Vec<f64> = valid.iter().map(|&v| v.max(1.0).ln()).collect();
        let baseline7 = mean(tail(&ell, ROLL_WINDOW));

        let long_win = if valid.len() >= LONG_WINDOW { LONG_WINDOW } else { LONG_WINDOW_FALLBACK };
        let long_ell = tail(&ell, long_win);
        let long_mean = mean(long_ell);
        let long_sd_raw = if long_ell.len() >= 2 { sample_sd(long_ell) } else { sample_sd(tail(&ell, ROLL_WINDOW)) };
        let long_sd = long_sd_raw.max(LONG_SD_FLOOR);

        let swc_half = SWC_K * long_sd;
        let normal_low = long_mean - swc_half;
        let normal_high = long_mean + swc_half;

        let tier = if baseline7 > normal_high {
            ReadinessTier::Primed
        } else if baseline7 >= normal_low {
            ReadinessTier::Normal
        } else {
            ReadinessTier::Suppressed
        };

        let overreaching_watch = cv_slope(&ell) < 0.0 && baseline7 < long_mean;

        Some(HrvReadinessResult {
            tier,
            baseline7_ms: baseline7.exp(),
            normal_low_ms: normal_low.exp(),
            normal_high_ms: normal_high.exp(),
            overreaching_watch,
        })
    }
}

/// The last `n` elements, or all if fewer.
fn tail(xs: &[f64], n: usize) -> &[f64] {
    &xs[xs.len().saturating_sub(n)..]
}

/// OLS slope of the rolling 7-night coefficient-of-variation series over the trailing `CV_TREND_WINDOW`.
fn cv_slope(ell: &[f64]) -> f64 {
    let start = (ROLL_WINDOW - 1).max(ell.len().saturating_sub(CV_TREND_WINDOW));
    let mut cv = Vec::new();
    for i in start..ell.len() {
        let w = &ell[i + 1 - ROLL_WINDOW..=i];
        let m = mean(w);
        cv.push(if m != 0.0 { 100.0 * sample_sd(w) / m } else { 0.0 });
    }
    least_squares_slope(&cv)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rmssd_of_known_intervals() {
        // successive diffs 10, 10 → mean square 100 → sqrt 10.
        assert_eq!(HrvReadiness::rmssd(&[800, 810, 820]), Some(10.0));
        assert_eq!(HrvReadiness::rmssd(&[800]), None);
    }

    #[test]
    fn rmssd_runs_pools_within_runs_only() {
        // Two runs; the break between them (810 → 100) must never be differenced.
        let a: &[u16] = &[800, 810];
        let b: &[u16] = &[100, 110];
        assert!((HrvReadiness::rmssd_runs([a, b]).unwrap() - 10.0).abs() < 1e-9);
        // No run with two beats → None.
        assert_eq!(HrvReadiness::rmssd_runs([[500u16].as_slice(), [600].as_slice()]), None);
        assert_eq!(HrvReadiness::rmssd_runs(std::iter::empty::<&[u16]>()), None);
    }

    #[test]
    fn nightly_rmssd_produces_one_gap_aware_value_per_day() {
        let rec = |unix: u32, rr: Vec<u16>| HistoryRecord { version: 18, unix, rr_intervals: rr, ..Default::default() };
        // day 0: two contiguous records; day 1: one record with two beats.
        let hist = vec![rec(0, vec![600, 610]), rec(1, vec![605, 615]), rec(SECS_PER_DAY, vec![700, 720])];
        let series = HrvReadiness::nightly_rmssd(&hist);
        assert_eq!(series.len(), 2);
        assert!(series.iter().all(|v| v.is_some()));
    }

    #[test]
    fn rmssd_gap_aware_breaks_on_time_gaps_and_artifacts() {
        // Steady contiguous beats, then a far-away single beat (unsorted input) and an artifact interval.
        let beats = vec![
            (10_000u32, vec![1400u16]), // far gap isolates it into a lone 1-beat run (1400 ms is plausible)
            (0, vec![600, 610]),
            (1, vec![605, 615]),
            (2, vec![5, 620]), // the 5 ms artifact breaks the chain, the 620 survives
        ];
        let g = HrvReadiness::rmssd_gap_aware(&beats).unwrap();
        assert!(g < 20.0, "gap-aware RMSSD should stay small, got {g}");
    }

    #[test]
    fn evaluate_calibrating_below_min_nights() {
        let nights = vec![Some(50.0); MIN_NIGHTS - 1];
        assert!(HrvReadiness::evaluate(&nights).is_none());
    }

    #[test]
    fn flat_history_reads_normal() {
        let nights = vec![Some(50.0); 20];
        assert_eq!(HrvReadiness::evaluate(&nights).unwrap().tier, ReadinessTier::Normal);
    }

    #[test]
    fn rising_last_week_is_primed_falling_is_suppressed() {
        let mut rising: Vec<Option<f64>> = vec![Some(40.0); 13];
        rising.extend(vec![Some(80.0); 7]);
        assert_eq!(HrvReadiness::evaluate(&rising).unwrap().tier, ReadinessTier::Primed);

        let mut falling: Vec<Option<f64>> = vec![Some(80.0); 13];
        falling.extend(vec![Some(40.0); 7]);
        assert_eq!(HrvReadiness::evaluate(&falling).unwrap().tier, ReadinessTier::Suppressed);
    }

    #[test]
    fn implausible_and_missing_nights_are_dropped() {
        // 2 valid + a null + an out-of-range 400 → still only 2 valid → below MIN_NIGHTS → calibrating.
        let mut nights = vec![Some(50.0); 2];
        nights.push(None);
        nights.push(Some(400.0));
        assert!(HrvReadiness::evaluate(&nights).is_none());
    }
}
