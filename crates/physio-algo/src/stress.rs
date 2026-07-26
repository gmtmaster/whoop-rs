//! Stress metrics: daily autonomic stress (0–3 from RHR+HRV vs prior-days baseline), daytime
//! intraday activation, Baevsky Stress Index.
//!
//! Baevsky Stress Index (SI) — a histogram-based autonomic-balance proxy over an R-R series.
//! `SI = AMo / (2 * Mo * MxDMn)`: Mo is the modal R-R (s), AMo the modal bin's share (%), MxDMn the
//! R-R range (s). A tall, narrow, low-range histogram (rigid, sympathetic) reads high; a broad, flat
//! one reads low. R-R is cleaned first (range band + Malik ectopic). Wellness only, never medical.

use crate::stats::{median, percentile};

/// Histogram bin width in seconds (Baevsky's 50 ms cardiointervalography grid).
const BIN_WIDTH_SEC: f64 = 0.05;
/// Minimum clean intervals before an SI is computed.
pub const MIN_BEATS: usize = 20;

/// R-R keep-band (ms); intervals outside are dropouts/ectopics.
const RR_MIN_MS: f64 = 300.0;
const RR_MAX_MS: f64 = 2000.0;
/// Malik ectopic rejection: beat dropped if it deviates over 20% from its local median.
const ECTOPIC_THRESHOLD: f64 = 0.20;
/// Half-width (beats) of the centred median window; a 5-beat window at radius 2.
const ECTOPIC_WINDOW_RADIUS: usize = 2;

/// Intermediate histogram terms behind an SI, exposed so a caller can show the "why".
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StressComponents {
    pub mo_sec: f64,
    pub amo_percent: f64,
    pub mxdmn_sec: f64,
    pub si: f64,
}

/// Baevsky Stress Index from a raw R-R series (ms). `None` when too few clean beats survive or the
/// range is degenerate (all-equal beats → MxDMn 0 → an honest `None`, not infinity).
pub fn stress_index_raw(rr_ms: &[f64]) -> Option<f64> {
    components_raw(rr_ms).map(|c| c.si)
}

/// Full SI components from a raw R-R series (ms). Pure and deterministic.
pub fn components_raw(rr_ms: &[f64]) -> Option<StressComponents> {
    let clean = clean_rr(rr_ms);
    if clean.len() < MIN_BEATS {
        return None;
    }
    let sec: Vec<f64> = clean.iter().map(|v| v / 1000.0).collect();
    let min_v = sec.iter().copied().fold(f64::INFINITY, f64::min);
    let max_v = sec.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mxdmn = max_v - min_v;
    if mxdmn <= 0.0 {
        return None;
    }

    let bin_count = ((mxdmn / BIN_WIDTH_SEC).floor() as usize + 1).max(1);
    let mut counts = vec![0usize; bin_count];
    for &v in &sec {
        let mut idx = ((v - min_v) / BIN_WIDTH_SEC).floor() as isize;
        if idx < 0 {
            idx = 0;
        }
        let mut idx = idx as usize;
        if idx >= bin_count {
            idx = bin_count - 1;
        }
        counts[idx] += 1;
    }
    // Modal bin: highest count; ties resolve to the lowest index (deterministic across platforms).
    let mut mode_idx = 0usize;
    let mut mode_count = counts[0];
    for (i, &c) in counts.iter().enumerate().skip(1) {
        if c > mode_count {
            mode_count = c;
            mode_idx = i;
        }
    }
    let mo = min_v + (mode_idx as f64 + 0.5) * BIN_WIDTH_SEC;
    let amo = mode_count as f64 / sec.len() as f64 * 100.0;
    if mo <= 0.0 {
        return None;
    }
    let si = amo / (2.0 * mo * mxdmn);
    Some(StressComponents { mo_sec: mo, amo_percent: amo, mxdmn_sec: mxdmn, si })
}

/// Full clean: range band then Malik ectopic rejection, order preserved.
fn clean_rr(rr: &[f64]) -> Vec<f64> {
    let ranged: Vec<f64> = rr.iter().copied().filter(|&v| (RR_MIN_MS..=RR_MAX_MS).contains(&v)).collect();
    reject_ectopic(&ranged)
}

/// Drop any beat deviating over `ECTOPIC_THRESHOLD` from its local median; short series pass through.
fn reject_ectopic(nn: &[f64]) -> Vec<f64> {
    if nn.len() <= ECTOPIC_WINDOW_RADIUS {
        return nn.to_vec();
    }
    let mut kept = Vec::with_capacity(nn.len());
    for i in 0..nn.len() {
        let lo = i.saturating_sub(ECTOPIC_WINDOW_RADIUS);
        let hi = (i + ECTOPIC_WINDOW_RADIUS).min(nn.len() - 1);
        let mut neighbours: Vec<f64> = Vec::with_capacity(hi - lo);
        for (j, &v) in nn.iter().enumerate().take(hi + 1).skip(lo) {
            if j != i {
                neighbours.push(v);
            }
        }
        if neighbours.len() < 2 {
            kept.push(nn[i]);
            continue;
        }
        let med = median(&neighbours);
        if med <= 0.0 {
            kept.push(nn[i]);
            continue;
        }
        if (nn[i] - med).abs() / med <= ECTOPIC_THRESHOLD {
            kept.push(nn[i]);
        }
    }
    kept
}

#[cfg(test)]
mod tests {
    use super::*;

    const GOLDEN: [f64; 22] = [
        700.0, 720.0, 740.0, 760.0, 780.0, 800.0, 820.0, 840.0, 860.0, 800.0, 800.0, 800.0, 800.0,
        820.0, 780.0, 800.0, 810.0, 790.0, 800.0, 800.0, 805.0, 795.0,
    ];

    #[test]
    fn golden_stress_index_hand_computed() {
        let comp = components_raw(&GOLDEN).expect("scorable");
        assert!((comp.mxdmn_sec - 0.16).abs() < 1e-9);
        assert!((comp.mo_sec - 0.825).abs() < 1e-9);
        assert!((comp.amo_percent - 59.09090909090909).abs() < 1e-9);
        assert!((comp.si - 223.82920110192836).abs() < 1e-9);
        assert!((stress_index_raw(&GOLDEN).unwrap() - 223.82920110192836).abs() < 1e-9);
    }

    #[test]
    fn tighter_histogram_raises_si() {
        let broad: Vec<f64> = (0..30).map(|it| 700.0 + (it % 11) as f64 * 18.0).collect();
        let rigid: Vec<f64> = (0..30).map(|it| if it % 6 == 0 { 810.0 } else { 800.0 }).collect();
        let si_broad = stress_index_raw(&broad).expect("broad scorable");
        let si_rigid = stress_index_raw(&rigid).expect("rigid scorable");
        assert!(si_rigid > si_broad, "a rigid, tightly-clustered rhythm has a higher SI");
    }

    #[test]
    fn too_few_beats_returns_none() {
        let rr = vec![800.0; MIN_BEATS - 1];
        assert!(stress_index_raw(&rr).is_none());
    }

    #[test]
    fn degenerate_range_returns_none() {
        let rr = vec![800.0; 30];
        assert!(stress_index_raw(&rr).is_none());
    }
}

// ── Daily Stress (autonomic baseline deviation) ────────────────────────────────

const DAILY_STRESS_BASELINE_DAYS: usize = 14;
const STRESS_SD_FLOOR: f64 = 0.0001;
const STRESS_MAX: f64 = 3.0;

/// One day's input for the daily-stress baseline.
#[derive(Clone, Copy, Debug)]
pub struct StressDay {
    pub rhr: Option<f64>,
    pub hrv: Option<f64>,
}

/// Daily autonomic stress (0–3) from today's RHR+HRV against a prior-days baseline.
/// Returns `None` when there are too few baseline days or today has no signal.
pub fn daily_stress(today: StressDay, baseline: &[StressDay]) -> Option<f64> {
    if baseline.len() < DAILY_STRESS_BASELINE_DAYS {
        return None;
    }
    let rhr_base: Vec<f64> = baseline.iter().filter_map(|d| d.rhr).collect();
    let hrv_base: Vec<f64> = baseline.iter().filter_map(|d| d.hrv).collect();

    let mean_rhr = mean(&rhr_base);
    let sd_rhr = population_std(&rhr_base, mean_rhr);
    let mean_hrv = mean(&hrv_base);
    let sd_hrv = population_std(&hrv_base, mean_hrv);

    let has_signal = (today.rhr.is_some() && mean_rhr.is_some())
        || (today.hrv.is_some() && mean_hrv.is_some());
    if !has_signal {
        return None;
    }
    let mut raw = 0.0;
    if let (Some(r), Some(m)) = (today.rhr, mean_rhr) {
        if sd_rhr > STRESS_SD_FLOOR {
            raw += (r - m) / sd_rhr;
        }
    }
    if let (Some(h), Some(m)) = (today.hrv, mean_hrv) {
        if sd_hrv > STRESS_SD_FLOOR {
            raw += (m - h) / sd_hrv;
        }
    }
    Some((STRESS_MAX / (1.0 + (-raw).exp())).clamp(0.0, STRESS_MAX))
}

// ── Daytime Stress (intraday autonomic activation) ─────────────────────────────

const WAKING_START_HOUR: i64 = 6;
const WAKING_END_HOUR: i64 = 22;
const HIGH_BAND_FLOOR: f64 = 2.0;
const SUSTAINED_HOURS: usize = 3;
const CALM_QUARTILE_MIN_COUNT: usize = 4;
const DAYTIME_SD_FLOOR: f64 = 0.0001;

/// One hour's data for daytime stress scoring.
#[derive(Clone, Copy, Debug)]
pub struct HourPoint {
    /// Local hour (0-23).
    pub hour: i32,
    /// Mean HR for the hour, or None if below the sample gate.
    pub mean_hr: Option<f64>,
    /// RMSSD for the hour, or None if insufficient clean R-R.
    pub rmssd: Option<f64>,
}

/// Daytime stress result.
#[derive(Clone, Debug)]
pub struct DaytimeStressResult {
    /// Per-hour scores (only waking hours with HR above gate).
    pub hours: Vec<ScoredHour>,
    /// Mean across all scored hours.
    pub day_mean: Option<f64>,
    /// Hour with the highest score.
    pub peak_hour: Option<i32>,
    /// True when the last 3+ scored hours are all >= 2.
    pub sustained_high: bool,
    /// Length of the trailing high run.
    pub sustained_run: usize,
}

/// One scored waking hour.
#[derive(Clone, Copy, Debug)]
pub struct ScoredHour {
    pub hour: i32,
    pub mean_hr: f64,
    pub rmssd: Option<f64>,
    pub stress: f64,
}

/// Score daytime hours for autonomic activation. Each hour needs ≥300 HR samples to score;
/// RMSSD enriches the score when present. Reference values are the day's own calm-hour
/// quartiles (Q25 for HR, Q75 for RMSSD) — no cross-day history needed.
pub fn daytime_stress(hours: &[HourPoint]) -> DaytimeStressResult {
    let waking: Vec<&HourPoint> = hours.iter().filter(|h| is_waking(h.hour)).collect();
    if waking.is_empty() {
        return DaytimeStressResult {
            hours: Vec::new(), day_mean: None, peak_hour: None,
            sustained_high: false, sustained_run: 0,
        };
    }
    let hr_vals: Vec<f64> = waking.iter().filter_map(|h| h.mean_hr).collect();
    let rmssd_vals: Vec<f64> = waking.iter().filter_map(|h| h.rmssd).collect();

    let calm_hr = calm_reference(&hr_vals, true);
    let calm_rmssd = calm_reference(&rmssd_vals, false);
    let hr_mean = mean(&hr_vals);
    let sd_hr = population_std(&hr_vals, hr_mean);
    let rmssd_mean = mean(&rmssd_vals);
    let sd_rmssd = population_std(&rmssd_vals, rmssd_mean);

    let mut scored: Vec<(ScoredHour, f64)> = Vec::new();
    for h in &waking {
        let Some(mean_hr) = h.mean_hr else { continue };
        let mut raw = 0.0;
        if let Some(ref_hr) = calm_hr {
            if sd_hr > DAYTIME_SD_FLOOR {
                raw += (mean_hr - ref_hr) / sd_hr;
            }
        }
        if let (Some(r), Some(ref_r)) = (h.rmssd, calm_rmssd) {
            if sd_rmssd > DAYTIME_SD_FLOOR {
                raw += (ref_r - r) / sd_rmssd;
            }
        }
        let stress = (3.0 / (1.0 + (-raw).exp())).clamp(0.0, 3.0);
        scored.push((ScoredHour { hour: h.hour, mean_hr, rmssd: h.rmssd, stress }, stress));
    }
    let mut run = 0;
    for (_, s) in scored.iter().rev() {
        if *s >= HIGH_BAND_FLOOR { run += 1 } else { break; }
    }
    let sustained = run >= SUSTAINED_HOURS;
    let day_mean = if scored.is_empty() { None } else { Some(scored.iter().map(|(_, s)| s).sum::<f64>() / scored.len() as f64) };
    let peak_hour = scored.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).map(|(s, _)| s.hour);

    DaytimeStressResult {
        hours: scored.into_iter().map(|(s, _)| s).collect(),
        day_mean, peak_hour, sustained_high: sustained, sustained_run: run,
    }
}

fn is_waking(hour: i32) -> bool {
    (WAKING_START_HOUR..WAKING_END_HOUR).contains(&(hour as i64))
}

fn calm_reference(xs: &[f64], calm_is_low: bool) -> Option<f64> {
    if xs.is_empty() { return None; }
    if xs.len() < CALM_QUARTILE_MIN_COUNT { return mean(xs); }
    let mut s = xs.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Some(percentile(&s, if calm_is_low { 0.25 } else { 0.75 }))
}

fn mean(xs: &[f64]) -> Option<f64> {
    if xs.is_empty() { None } else { Some(xs.iter().sum::<f64>() / xs.len() as f64) }
}

fn population_std(xs: &[f64], m: Option<f64>) -> f64 {
    let Some(m) = m else { return 0.0 };
    if xs.len() <= 1 { return 0.0; }
    let var = xs.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / xs.len() as f64;
    var.sqrt()
}

#[cfg(test)]
mod stress_tests {
    use super::*;

    fn day(rhr: Option<f64>, hrv: Option<f64>) -> StressDay { StressDay { rhr, hrv } }

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

    // ── daytime stress ──

    #[test]
    fn flat_day_scores_neutral() {
        // All hours identical → zero spread → all z-scores zero → neutral 1.5
        let hours: Vec<HourPoint> = (6..22).map(|h| HourPoint { hour: h, mean_hr: Some(70.0), rmssd: Some(40.0) }).collect();
        let r = daytime_stress(&hours);
        assert!((r.day_mean.unwrap() - 1.5).abs() < 0.1, "flat day → neutral, got {}", r.day_mean.unwrap());
        assert!(!r.sustained_high);
    }

    #[test]
    fn spiky_afternoon_scores_high_and_sustained() {
        let mut hours: Vec<HourPoint> = (6..14).map(|h| HourPoint { hour: h, mean_hr: Some(70.0), rmssd: Some(40.0) }).collect();
        // Afternoon spike: HR up, RMSSD down for 3+ hours
        for h in 14..18 {
            hours.push(HourPoint { hour: h, mean_hr: Some(90.0), rmssd: Some(25.0) });
        }
        let r = daytime_stress(&hours);
        assert!(r.sustained_high, "3+ spiky hours should trigger sustained high");
        assert!(r.peak_hour.is_some());
    }

    #[test]
    fn below_hr_gate_excluded() {
        let hours = vec![HourPoint { hour: 10, mean_hr: None, rmssd: Some(40.0) }];
        let r = daytime_stress(&hours);
        assert!(r.hours.is_empty());
        assert!(r.day_mean.is_none());
    }

    #[test]
    fn sleep_hours_excluded() {
        let hours = vec![HourPoint { hour: 2, mean_hr: Some(60.0), rmssd: Some(50.0) }];
        let r = daytime_stress(&hours);
        assert!(r.hours.is_empty(), "2am should be excluded from waking window");
    }
}
