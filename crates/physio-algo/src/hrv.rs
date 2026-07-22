//! HRV-readiness: a log-domain rolling-baseline vs personal-normal-band reading over a nightly RMSSD
//! series. Input = the nightly RMSSD (ms) derived from decoded R-R; output = a tier + the band in ms.
//! Returns `None` (calibrating) below `MIN_NIGHTS` valid nights. Wellness only, never medical.

use crate::stats::{least_squares_slope, mean, median, sample_sd};
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
/// A physiologically plausible R-R interval (ms); values outside are dropped before cleaning.
const RR_MIN_MS: u16 = 300;
const RR_MAX_MS: u16 = 2000;
/// Width (s) of the tumbling window the app pools per-bucket RMSSD over for its stored session avgHrv.
const HRV_WINDOW_SECS: u64 = 300;
/// A beat-to-beat R-R change beyond this (ms) is an artifact (ectopic/missed beat), not real variability,
/// so its squared difference is dropped from RMSSD — the standard HRV artifact-correction step.
const MAX_BEAT_DELTA_MS: f64 = 200.0;
/// Malik ectopic rejection: a beat deviating over this fraction from its local median is dropped.
const ECTOPIC_THRESHOLD: f64 = 0.20;
/// Half-width (beats) of the centred median window; a 5-beat window at radius 2.
const ECTOPIC_WINDOW_RADIUS: usize = 2;
/// Minimum clean NN intervals before a full [`HrvReadiness::analyze_raw`] result is trustworthy.
pub const MIN_BEATS: usize = 20;
/// A successive |ΔNN| above this (ms) counts toward pNN50.
const PNN50_THRESHOLD_MS: f64 = 50.0;

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

/// Full HRV analysis over one raw R-R capture: cleaned RMSSD/SDNN/pNN50/meanNN plus the input and clean
/// beat counts. Fields are `None` (and `n_clean` 0) on an insufficient/refused reading.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HrvAnalysis {
    pub rmssd: Option<f64>,
    pub sdnn: Option<f64>,
    pub mean_nn: Option<f64>,
    pub pnn50: Option<f64>,
    pub n_input: u32,
    pub n_clean: u32,
}

pub struct HrvReadiness;

impl HrvReadiness {
    /// Full clean-and-analyze over a raw R-R series (ms): range filter -> Malik ectopic -> gap-aware RMSSD +
    /// pNN50, with SDNN + meanNN over the clean beats. Empty (all `None`, `n_clean` 0) when fewer than
    /// [`MIN_BEATS`] clean survive, or -- when `max_rejected_fraction` is set (the spot honesty gate) -- when
    /// the dropped fraction exceeds it. The nightly path passes `None` (no gate), so it is unchanged.
    pub fn analyze_raw(rr_ms: &[u16], max_rejected_fraction: Option<f64>) -> HrvAnalysis {
        let n_input = rr_ms.len();
        let empty = HrvAnalysis {
            rmssd: None,
            sdnn: None,
            mean_nn: None,
            pnn50: None,
            n_input: n_input as u32,
            n_clean: 0,
        };
        let (nn, contiguous) = clean_rr_gap_aware(rr_ms);
        if nn.len() < MIN_BEATS {
            return empty;
        }
        if let Some(max_rej) = max_rejected_fraction {
            if n_input > 0 && 1.0 - nn.len() as f64 / n_input as f64 > max_rej {
                return empty;
            }
        }
        // meanNN over the clean beats; SDNN reuses [`Self::sdnn`] over the same beats (n >= MIN_BEATS here).
        let mean_nn = nn.iter().map(|&v| v as f64).sum::<f64>() / nn.len() as f64;
        HrvAnalysis {
            rmssd: rmssd_from_clean(&nn, &contiguous),
            sdnn: Self::sdnn(&nn),
            mean_nn: Some(mean_nn),
            pnn50: pnn50_from_clean(&nn, &contiguous),
            n_input: n_input as u32,
            n_clean: nn.len() as u32,
        }
    }

    /// Gap-aware, artifact-corrected RMSSD (ms): pools squared successive differences **within** each run
    /// of consecutive beats, never across the break between runs, and drops any single beat-to-beat change
    /// over `MAX_BEAT_DELTA_MS` (an ectopic/missed beat), so neither an offload gap nor an artifact can
    /// inflate it. `None` if no run has two beats. The caller splits R-R into runs on time gaps.
    pub fn rmssd_runs<'a>(runs: impl IntoIterator<Item = &'a [u16]>) -> Option<f64> {
        let (mut sumsq, mut pairs) = (0.0f64, 0usize);
        for run in runs {
            for w in run.windows(2) {
                let d = w[1] as f64 - w[0] as f64;
                if d.abs() > MAX_BEAT_DELTA_MS {
                    continue; // ectopic / missed beat, not physiological variability
                }
                sumsq += d * d;
                pairs += 1;
            }
        }
        (pairs > 0).then(|| (sumsq / pairs as f64).sqrt())
    }

    /// RMSSD (ms) of one run of consecutive R-R beats. `None` for < 2 beats. Uses the run-based
    /// path which drops individual beat deltas > `MAX_BEAT_DELTA_MS`.
    pub fn rmssd(rr_ms: &[u16]) -> Option<f64> {
        Self::rmssd_runs(std::iter::once(rr_ms))
    }

    /// Plain RMSSD without artifact filtering. Accepts u16 values.
    pub fn rmssd_plain(rr_ms: &[u16]) -> Option<f64> {
        if rr_ms.len() < 2 { return None; }
        let mut sum_sq = 0.0;
        for i in 1..rr_ms.len() {
            let d = rr_ms[i] as f64 - rr_ms[i - 1] as f64;
            sum_sq += d * d;
        }
        Some((sum_sq / (rr_ms.len() - 1) as f64).sqrt())
    }

    /// Range-filter R-R intervals (keep 300–2000 ms).
    pub fn range_filter(rr_ms: &[u16]) -> Vec<u16> {
        rr_ms.iter().copied().filter(|&v| (RR_MIN_MS..=RR_MAX_MS).contains(&v)).collect()
    }

    /// Sample standard deviation (ddof=1) of NN intervals (ms). `None` for < 2 values. Sums in f64 so a
    /// long clean series can't overflow a u16 accumulator.
    pub fn sdnn(rr_ms: &[u16]) -> Option<f64> {
        if rr_ms.len() < 2 {
            return None;
        }
        let mean = rr_ms.iter().map(|&v| v as f64).sum::<f64>() / rr_ms.len() as f64;
        let var = rr_ms.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / (rr_ms.len() - 1) as f64;
        Some(var.sqrt())
    }

    /// Range + Malik-ectopic cleaned NN series (ms), in input order. The clean beats only (no contiguity
    /// mask); a successive-difference metric wants [`clean_rr_gap_aware`] instead.
    pub fn clean_rr(rr_ms: &[u16]) -> Vec<u16> {
        clean_rr_gap_aware(rr_ms).0
    }

    /// Median of a slice of f64 values. Empty → 0.
    pub fn median_f64(values: &[f64]) -> f64 {
        if values.is_empty() { return 0.0; }
        let mut s = values.to_vec();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = s.len();
        if n % 2 == 1 { s[n / 2] } else { 0.5 * (s[n / 2 - 1] + s[n / 2]) }
    }

    /// Gap-aware nightly RMSSD (ms) from one night's per-record `(unix, R-R)` beats, matching the app's
    /// cleaned RMSSD. Flattens the beats in time order, range-filters and Malik-ectopic-cleans them, then
    /// pools only successive differences whose two beats were adjacent in the source: a dropped range or
    /// ectopic beat splices its neighbours apart and that difference is skipped. Divides by the
    /// contiguous-pair count. Input need not be sorted. `None` if no contiguous pair survives.
    pub fn rmssd_gap_aware(beats: &[(u32, Vec<u16>)]) -> Option<f64> {
        let mut order: Vec<&(u32, Vec<u16>)> = beats.iter().collect();
        order.sort_by_key(|(t, _)| *t);
        let flat: Vec<u16> = order.iter().flat_map(|(_, rr)| rr.iter().copied()).collect();
        let (nn, contiguous) = clean_rr_gap_aware(&flat);
        rmssd_from_clean(&nn, &contiguous)
    }

    /// Windowed session RMSSD (ms): the mean of per-`HRV_WINDOW_SECS`-bucket gap-aware RMSSD over the
    /// `[start, end]` span, matching the app's stored session avgHrv. `beats` are `(unix, rr)` in the
    /// caller's chronological order; buckets tumble from `start`, each range+ectopic-cleaned then
    /// gap-aware-RMSSD'd (kept only with >= 2 clean beats and a surviving contiguous pair). No whole-night
    /// beat-count gate. `None` when no bucket yields a value.
    pub fn windowed_avg_hrv(start: u32, end: u32, beats: &[(u32, u16)]) -> Option<f64> {
        Self::windowed_avg_hrv_inner(start, end, beats, |_| true)
    }

    /// Deep-sleep-windowed session avgHrv (ms): per-bucket RMSSD like [windowed_avg_hrv], keeping only
    /// buckets whose center falls inside a `deep_spans` span. `None` when no deep bucket yields a value.
    pub fn windowed_avg_hrv_deep(start: u32, end: u32, beats: &[(u32, u16)], deep_spans: &[(u32, u32)]) -> Option<f64> {
        // Boxing so the closure outlives this call while the inner function borrows the spans.
        let spans: Vec<(u64, u64)> = deep_spans.iter().map(|&(s, e)| (s as u64, e as u64)).collect();
        Self::windowed_avg_hrv_inner(start, end, beats, move |t| {
            let center = t + HRV_WINDOW_SECS / 2;
            spans.iter().any(|&(ds, de)| center >= ds && center < de)
        })
    }

    /// Common bucket-loop body shared by [windowed_avg_hrv] and [windowed_avg_hrv_deep].
    /// The `bucket_filter` predicate gates each bucket's centre; it is called once per bucket.
    fn windowed_avg_hrv_inner<F: Fn(u64) -> bool>(
        start: u32, end: u32, beats: &[(u32, u16)], bucket_filter: F,
    ) -> Option<f64> {
        let seg: Vec<(u32, u16)> = beats.iter().copied().filter(|&(t, _)| t >= start && t <= end).collect();
        if seg.is_empty() {
            return None;
        }
        let (start, end) = (start as u64, end as u64);
        let (mut sum, mut n) = (0.0f64, 0usize);
        let mut t = start;
        while t < end {
            let hi = t + HRV_WINDOW_SECS;
            if bucket_filter(t) {
                let bucket: Vec<u16> =
                    seg.iter().filter(|&&(ts, _)| ts as u64 >= t && (ts as u64) < hi).map(|&(_, rr)| rr).collect();
                let (nn, contiguous) = clean_rr_gap_aware(&bucket);
                let val = if nn.len() >= 2 { rmssd_from_clean(&nn, &contiguous) } else { None };
                if let Some(v) = val {
                    sum += v;
                    n += 1;
                }
            }
            t = hi;
        }
        (n > 0).then(|| sum / n as f64)
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

/// Task-Force RMSSD (ms) over a cleaned series, pooling only successive differences whose two beats were
/// adjacent in the source (`contiguous[i]`). `None` when no contiguous pair survives.
fn rmssd_from_clean(nn: &[u16], contiguous: &[bool]) -> Option<f64> {
    let (mut sumsq, mut count) = (0.0f64, 0usize);
    for i in 1..nn.len() {
        if !contiguous[i] {
            continue;
        }
        let d = nn[i] as f64 - nn[i - 1] as f64;
        sumsq += d * d;
        count += 1;
    }
    (count > 0).then(|| (sumsq / count as f64).sqrt())
}

/// Gap-aware pNN50 (% of contiguous successive |ΔNN| > 50 ms). `None` when no contiguous pair survives.
fn pnn50_from_clean(nn: &[u16], contiguous: &[bool]) -> Option<f64> {
    let (mut nn50, mut pairs) = (0usize, 0usize);
    for i in 1..nn.len() {
        if !contiguous[i] {
            continue;
        }
        if (nn[i] as f64 - nn[i - 1] as f64).abs() > PNN50_THRESHOLD_MS {
            nn50 += 1;
        }
        pairs += 1;
    }
    (pairs > 0).then(|| nn50 as f64 / pairs as f64 * 100.0)
}

/// The last `n` elements, or all if fewer.
fn tail(xs: &[f64], n: usize) -> &[f64] {
    &xs[xs.len().saturating_sub(n)..]
}

/// Range-filter then Malik ectopic-reject an ordered R-R series (ms), returning the clean beats and a
/// contiguity mask: `contiguous[i]` is true only when beats `i` and `i-1` were adjacent in the input
/// (no beat dropped between them). A splice from a dropped beat is where a successive-difference metric
/// must skip. Index 0 is always false.
fn clean_rr_gap_aware(rr: &[u16]) -> (Vec<u16>, Vec<bool>) {
    let mut ranged_idx: Vec<usize> = Vec::new();
    let mut ranged_val: Vec<u16> = Vec::new();
    for (i, &v) in rr.iter().enumerate() {
        if (RR_MIN_MS..=RR_MAX_MS).contains(&v) {
            ranged_idx.push(i);
            ranged_val.push(v);
        }
    }
    let (mut kept_orig, mut kept_val): (Vec<usize>, Vec<u16>) = (Vec::new(), Vec::new());
    if ranged_val.len() <= ECTOPIC_WINDOW_RADIUS {
        (kept_orig, kept_val) = (ranged_idx, ranged_val);
    } else {
        for i in 0..ranged_val.len() {
            let lo = i.saturating_sub(ECTOPIC_WINDOW_RADIUS);
            let hi = (i + ECTOPIC_WINDOW_RADIUS).min(ranged_val.len() - 1);
            let mut neighbours: Vec<f64> = Vec::with_capacity(hi - lo);
            for (j, &v) in ranged_val.iter().enumerate().take(hi + 1).skip(lo) {
                if j != i {
                    neighbours.push(v as f64);
                }
            }
            let keep = if neighbours.len() < 2 {
                true
            } else {
                let med = median(&neighbours);
                med <= 0.0 || (ranged_val[i] as f64 - med).abs() / med <= ECTOPIC_THRESHOLD
            };
            if keep {
                kept_orig.push(ranged_idx[i]);
                kept_val.push(ranged_val[i]);
            }
        }
    }
    let contiguous: Vec<bool> = (0..kept_val.len())
        .map(|i| i > 0 && kept_orig[i] == kept_orig[i - 1] + 1)
        .collect();
    (kept_val, contiguous)
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
    fn rmssd_runs_excludes_artifact_beat_jumps() {
        // A 900 ms jump in the middle (an ectopic/missed beat) must not inflate RMSSD; only the clean
        // 10 ms diffs survive → ~10 ms, not the ~600 ms a raw pooling would give.
        let run: &[u16] = &[600, 610, 1500, 610, 600];
        let v = HrvReadiness::rmssd_runs([run]).unwrap();
        assert!(v < 20.0, "artifact beat-to-beat jumps should be excluded, got {v}");
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
    fn clean_rr_gap_aware_drops_ectopic_and_splices() {
        // A Malik-ectopic spike (1300) is dropped; its removal splices 810→806 apart so that difference is
        // not contiguous. Values hand-verified against the app's cleanRRGapAware.
        let (nn, contig) = clean_rr_gap_aware(&[800, 805, 810, 1300, 806, 802, 808]);
        assert_eq!(nn, vec![800, 805, 810, 806, 802, 808]);
        assert_eq!(contig, vec![false, true, true, false, true, true]);
    }

    fn alternating(n: usize, lo: u16, hi: u16) -> Vec<u16> {
        (0..n).map(|i| if i % 2 == 0 { lo } else { hi }).collect()
    }

    #[test]
    fn analyze_raw_full_clean_series() {
        let a = HrvReadiness::analyze_raw(&alternating(24, 800, 810), None);
        assert_eq!((a.n_input, a.n_clean), (24, 24));
        assert!((a.rmssd.unwrap() - 10.0).abs() < 1e-9); // every successive Δ = 10
        assert_eq!(a.mean_nn, Some(805.0));
        assert_eq!(a.pnn50, Some(0.0)); // all |Δ| = 10 < 50
        assert!((a.sdnn.unwrap() - (600.0f64 / 23.0).sqrt()).abs() < 1e-9);
    }

    #[test]
    fn analyze_raw_too_few_clean_is_empty() {
        let a = HrvReadiness::analyze_raw(&[800, 810, 820], None);
        assert_eq!(
            a,
            HrvAnalysis { rmssd: None, sdnn: None, mean_nn: None, pnn50: None, n_input: 3, n_clean: 0 }
        );
    }

    #[test]
    fn analyze_raw_spot_rejected_fraction_gate() {
        // 25 clean beats + 15 out-of-range (100 ms) = 40 input; cleaning keeps 25 (0.375 rejected).
        let mut s = alternating(25, 800, 810);
        s.extend(std::iter::repeat_n(100u16, 15));
        let open = HrvReadiness::analyze_raw(&s, None);
        assert_eq!((open.n_input, open.n_clean), (40, 25));
        assert!(open.rmssd.is_some());
        // Spot gate at 0.35: 0.375 > 0.35 → refused (empty, n_clean 0).
        let gated = HrvReadiness::analyze_raw(&s, Some(0.35));
        assert_eq!(gated.n_clean, 0);
        assert!(gated.rmssd.is_none());
        // A looser gate passes.
        assert!(HrvReadiness::analyze_raw(&s, Some(0.40)).rmssd.is_some());
    }

    #[test]
    fn analyze_raw_pnn50_counts_big_jumps() {
        // 20 beats alternating 700/800 → every contiguous |Δ| = 100 > 50 → pNN50 = 100 %.
        let a = HrvReadiness::analyze_raw(&alternating(20, 700, 800), None);
        assert_eq!(a.pnn50, Some(100.0));
    }

    #[test]
    fn sdnn_does_not_overflow_on_a_long_series() {
        // 300 × ~800 ms sums to ~240k, far past u16::MAX; a u16 accumulator would have wrapped/panicked.
        // Alternating 800/810 → sample SD = sqrt(7500 / 299) ≈ 5.0084.
        let s = HrvReadiness::analyze_raw(&alternating(300, 800, 810), None).sdnn.unwrap();
        assert!((s - (7500.0f64 / 299.0).sqrt()).abs() < 1e-9, "got {s}");
        assert_eq!(HrvReadiness::sdnn(&vec![900u16; 400]), Some(0.0)); // flat long series → zero spread
    }

    #[test]
    fn rmssd_gap_aware_matches_cleaned_kotlin() {
        // Malik-ectopic splice: the dropped 1300 removes the 810→806 difference. sqrt(102/4).
        let malik = vec![(0u32, vec![800u16, 805, 810, 1300, 806, 802, 808])];
        assert!((HrvReadiness::rmssd_gap_aware(&malik).unwrap() - 5.049752469181039).abs() < 1e-12);
        // Out-of-range 5 ms beat is dropped and splices 605→620 apart. sqrt(100/2).
        let range = vec![(0u32, vec![600u16, 610, 605, 5, 620, 615])];
        assert!((HrvReadiness::rmssd_gap_aware(&range).unwrap() - 7.0710678118654755).abs() < 1e-12);
        // A clean series has no splices, so it equals the plain Task-Force RMSSD (÷ n-1). sqrt(325/4).
        let clean = vec![(0u32, vec![800u16, 810, 820, 815, 805])];
        assert!((HrvReadiness::rmssd_gap_aware(&clean).unwrap() - 9.013878188659973).abs() < 1e-12);
    }

    #[test]
    fn windowed_avg_hrv_means_per_bucket_rmssd() {
        // Bucket A [100,400): clean [800,810,820,815,805] → sqrt(325/4). Bucket B [400,700): [700,720,710]
        // → sqrt(500/2). avgHrv = mean of the two bucket RMSSDs.
        let a = 9.013878188659973f64;
        let b = 15.811388300841896f64;
        let beats: Vec<(u32, u16)> = vec![
            (100, 800), (100, 810), (100, 820), (100, 815), (100, 805),
            (400, 700), (400, 720), (400, 710),
        ];
        let got = HrvReadiness::windowed_avg_hrv(100, 700, &beats).unwrap();
        assert!((got - (a + b) / 2.0).abs() < 1e-12, "got {got}");
    }

    #[test]
    fn windowed_avg_hrv_drops_buckets_without_a_contiguous_pair() {
        // A single in-range beat per bucket yields no successive difference → no bucket contributes → None.
        let beats: Vec<(u32, u16)> = vec![(100, 800), (400, 810)];
        assert!(HrvReadiness::windowed_avg_hrv(100, 700, &beats).is_none());
        // Beats outside [start, end] are filtered out before bucketing.
        let outside: Vec<(u32, u16)> = vec![(50, 800), (50, 810)];
        assert!(HrvReadiness::windowed_avg_hrv(100, 700, &outside).is_none());
    }

    #[test]
    fn windowed_avg_hrv_single_bucket_equals_bucket_rmssd() {
        // One bucket, one clean series → avgHrv equals that bucket's gap-aware RMSSD.
        let beats: Vec<(u32, u16)> = vec![(100, 800), (100, 810), (100, 820), (100, 815), (100, 805)];
        let got = HrvReadiness::windowed_avg_hrv(100, 400, &beats).unwrap();
        assert!((got - 9.013878188659973).abs() < 1e-12, "got {got}");
    }

    #[test]
    fn deep_windowed_avg_hrv_keeps_buckets_in_deep_spans() {
        // Two buckets: [100,400) and [400,700). Only [400,700) is deep → result = bucket B RMSSD.
        let beats: Vec<(u32, u16)> = vec![
            (100, 800), (100, 810), (100, 820), (100, 815), (100, 805),
            (400, 700), (400, 720), (400, 710),
        ];
        let deep = vec![(400u32, 700u32)];
        let got = HrvReadiness::windowed_avg_hrv_deep(100, 700, &beats, &deep).unwrap();
        let b = 15.811388300841896f64; // sqrt(500/2) from [700,720,710]
        assert!((got - b).abs() < 1e-12, "got {got}");
    }

    #[test]
    fn deep_windowed_avg_hrv_returns_none_without_deep_spans() {
        let beats: Vec<(u32, u16)> = vec![(100, 800), (100, 810)];
        let got = HrvReadiness::windowed_avg_hrv_deep(100, 400, &beats, &[]);
        assert!(got.is_none());
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
