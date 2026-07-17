//! Recovery scoring ("Charge") — a transparent z-score + logistic composite over nightly drivers
//! (HRV dominant, resting HR, respiration, sleep performance, skin-temp deviation, plus two optional
//! Oura-style terms: overnight HR-decline slope and previous-day effort). APPROXIMATE, not
//! WHOOP-identical. Plain-value inputs; each driver z is standardised against a personal baseline
//! (mean + EWMA-abs-dev spread). Missing terms drop and the weights renormalise. Wellness only.

/// One heart-rate sample: wall-clock unix seconds and beats per minute.
#[derive(Clone, Copy, Debug)]
pub struct HrSample {
    pub ts: i64,
    pub bpm: i32,
}

/// A baseline driver: personal mean and spread (internal EWMA-abs-dev units).
#[derive(Clone, Copy, Debug)]
pub struct DriverBaseline {
    pub mean: f64,
    pub spread: f64,
}

// Driver weights (higher HRV / lower RHR / lower resp / higher sleep / near-baseline skin temp = better).
pub const W_HRV: f64 = 0.55;
pub const W_RHR: f64 = 0.20;
pub const W_RESP: f64 = 0.05;
pub const W_SLEEP: f64 = 0.15;
pub const W_SKIN_TEMP: f64 = 0.05;
/// Skin-temp deviation scale (°C per z-unit): the term is −|dev| / scale.
pub const SKIN_TEMP_DEV_SCALE: f64 = 1.0;
/// Recovery-Index (overnight HR-decline slope) weight and its bpm/hour scale.
pub const W_RECOVERY_INDEX: f64 = 0.05;
pub const RECOVERY_INDEX_SCALE_BPM_PER_HR: f64 = 2.0;
/// Activity-Balance (previous-day effort vs baseline) weight.
pub const W_ACTIVITY_BALANCE: f64 = 0.05;

/// Logistic spread and offset so Z = 0 maps to ~58% (population-average recovery).
pub const LOGISTIC_K: f64 = 1.6;
pub const LOGISTIC_Z0: f64 = -0.20;
/// Cold-start fallback recovery (%).
pub const POPULATION_MEAN: f64 = 58.0;

/// Recovery-band thresholds (WHOOP colour scheme).
pub const BAND_RED_MAX: f64 = 34.0;
pub const BAND_YELLOW_MAX: f64 = 67.0;

/// Sleep-performance centre ("good night") and scale.
pub const SLEEP_PERF_CENTER: f64 = 0.85;
pub const SLEEP_PERF_SCALE: f64 = 0.12;

/// Non-overlapping bin width (seconds) for the overnight HR series.
pub const RESTING_HR_WINDOW_S: i64 = 5 * 60;
/// Minimum populated bins before an overnight slope is trusted (30 minutes of coverage).
pub const RECOVERY_INDEX_MIN_BINS: usize = 6;

/// HRV validity bounds (ms) — the seed signal for every baseline; used by [`banked_nights`].
pub const HRV_MIN_MS: f64 = 5.0;
pub const HRV_MAX_MS: f64 = 250.0;

/// Nights carrying a usable (in-bounds) nightly HRV — the calibration-progress count.
pub fn banked_nights(nightly_hrv: &[Option<f64>], min_val: f64, max_val: f64) -> usize {
    nightly_hrv
        .iter()
        .filter(|v| v.is_some_and(|x| x >= min_val && x <= max_val))
        .count()
}

/// WHOOP-style colour band for a recovery score in [0, 100].
pub fn band(score: f64) -> &'static str {
    if score < BAND_RED_MAX {
        "red"
    } else if score < BAND_YELLOW_MAX {
        "yellow"
    } else {
        "green"
    }
}

/// Robust z-score using EWMA spread: (value − mean) / (1.253 × spread).
pub fn z_score(value: f64, mean: f64, spread: f64) -> f64 {
    let sigma = (1.253 * spread).max(1e-9);
    (value - mean) / sigma
}

/// Overnight resting-HR DECLINE slope (bpm/hour) across the in-bed window. Negative = declining (the
/// expected, good pattern); positive = rising. `None` when fewer than [`RECOVERY_INDEX_MIN_BINS`] bins
/// hold data, or there are no samples — it never fabricates a slope from a sliver of the night.
pub fn recovery_index_slope(hr: &[HrSample], start: i64, end: i64) -> Option<f64> {
    let seg: Vec<&HrSample> = hr.iter().filter(|s| s.ts >= start && s.ts <= end).collect();
    if seg.is_empty() {
        return None;
    }
    let mut points: Vec<(f64, f64)> = Vec::new(); // (t_hours, mean_bpm)
    let mut t = start;
    while t < end {
        let bin_end = t + RESTING_HR_WINDOW_S;
        let win: Vec<&&HrSample> = seg.iter().filter(|s| s.ts >= t && s.ts < bin_end).collect();
        if !win.is_empty() {
            let mean = win.iter().map(|s| s.bpm as f64).sum::<f64>() / win.len() as f64;
            let midpoint_s = (t - start) as f64 + RESTING_HR_WINDOW_S as f64 / 2.0;
            points.push((midpoint_s / 3600.0, mean));
        }
        t += RESTING_HR_WINDOW_S;
    }
    if points.len() < RECOVERY_INDEX_MIN_BINS {
        return None;
    }
    let n = points.len() as f64;
    let t_bar = points.iter().map(|p| p.0).sum::<f64>() / n;
    let y_bar = points.iter().map(|p| p.1).sum::<f64>() / n;
    let (mut num, mut den) = (0.0, 0.0);
    for (t_hours, mean_bpm) in &points {
        let dt = t_hours - t_bar;
        num += dt * (mean_bpm - y_bar);
        den += dt * dt;
    }
    if den <= 1e-9 {
        return Some(0.0);
    }
    Some(num / den)
}

/// Nightly drivers for one recovery score. Defaults leave every optional term dropped and the HRV
/// baseline usable, so a bare `RecoveryInput { hrv, rhr, hrv_baseline, ..Default::default() }` scores.
#[derive(Clone, Copy, Debug)]
pub struct RecoveryInput {
    pub hrv: f64,
    pub rhr: f64,
    pub resp: Option<f64>,
    pub hrv_baseline: Option<DriverBaseline>,
    pub rhr_baseline: Option<DriverBaseline>,
    pub resp_baseline: Option<DriverBaseline>,
    pub sleep_perf: Option<f64>,
    pub skin_temp_dev: Option<f64>,
    pub hrv_baseline_usable: bool,
    pub recovery_index_slope: Option<f64>,
    pub effort_baseline: Option<DriverBaseline>,
    pub prior_day_effort: Option<f64>,
}

impl Default for RecoveryInput {
    fn default() -> Self {
        Self {
            hrv: 0.0,
            rhr: 0.0,
            resp: None,
            hrv_baseline: None,
            rhr_baseline: None,
            resp_baseline: None,
            sleep_perf: None,
            skin_temp_dev: None,
            hrv_baseline_usable: true,
            recovery_index_slope: None,
            effort_baseline: None,
            prior_day_effort: None,
        }
    }
}

/// Z-score + logistic recovery score in [0, 100]. APPROXIMATE. `None` when the HRV baseline is not
/// yet usable (cold-start) or no valid driver is available. Missing terms drop and the weights
/// renormalise, so a caller omitting the optional signals scores identically to before they existed.
pub fn recovery(input: &RecoveryInput) -> Option<f64> {
    if !input.hrv_baseline_usable {
        return None;
    }
    let mut terms: Vec<(f64, f64)> = Vec::new(); // (z, weight)

    if let Some(b) = input.hrv_baseline {
        terms.push((z_score(input.hrv, b.mean, b.spread), W_HRV));
    }
    if let Some(b) = input.rhr_baseline {
        terms.push((z_score(b.mean, input.rhr, b.spread), W_RHR));
    }
    if let (Some(resp), Some(b)) = (input.resp, input.resp_baseline) {
        terms.push((z_score(b.mean, resp, b.spread), W_RESP));
    }
    if let Some(sp) = input.sleep_perf {
        terms.push(((sp - SLEEP_PERF_CENTER) / SLEEP_PERF_SCALE, W_SLEEP));
    }
    if let Some(dev) = input.skin_temp_dev {
        terms.push((-dev.abs() / SKIN_TEMP_DEV_SCALE, W_SKIN_TEMP));
    }
    if let Some(slope) = input.recovery_index_slope {
        terms.push((-slope / RECOVERY_INDEX_SCALE_BPM_PER_HR, W_RECOVERY_INDEX));
    }
    if let (Some(effort), Some(b)) = (input.prior_day_effort, input.effort_baseline) {
        terms.push((z_score(b.mean, effort, b.spread), W_ACTIVITY_BALANCE));
    }

    if terms.is_empty() {
        return None;
    }
    let total_weight: f64 = terms.iter().map(|t| t.1).sum();
    if total_weight <= 0.0 {
        return None;
    }
    let z = terms.iter().map(|t| t.0 * t.1).sum::<f64>() / total_weight;
    let score = 100.0 / (1.0 + (-LOGISTIC_K * (z - LOGISTIC_Z0)).exp());
    Some(score.clamp(0.0, 100.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    // DriverBaseline from a Gaussian sigma (spread is internal abs-dev units) — mirrors the Kotlin
    // `baseline(mean, sigma)` helper: spread = sigma / 1.253.
    fn baseline(mean: f64, sigma: f64) -> DriverBaseline {
        DriverBaseline { mean, spread: sigma / 1.253 }
    }

    fn hr(ts: i64, bpm: i32) -> HrSample {
        HrSample { ts, bpm }
    }

    // Synthetic full-night HR series with a KNOWN injected slope: a sample every 30 s, rounded to the
    // integer bpm the wire carries. Mirrors the Kotlin `slopeSeries` helper.
    fn slope_series(start_bpm: f64, slope_per_hour: f64, hours: i64) -> (Vec<HrSample>, i64, i64) {
        let origin: i64 = 100_000;
        let total = hours * 3600;
        let mut samples = Vec::new();
        let mut t = 0;
        while t < total {
            let bpm = start_bpm + slope_per_hour * (t as f64 / 3600.0);
            samples.push(hr(origin + t, bpm.round() as i32));
            t += 30;
        }
        (samples, origin, origin + total)
    }

    #[test]
    fn recovery_index_slope_null_when_no_samples() {
        assert_eq!(recovery_index_slope(&[], 0, 1000), None);
    }

    #[test]
    fn recovery_index_slope_null_when_too_few_bins() {
        let start = 5000;
        let mut samples = Vec::new();
        for i in 0..300 {
            samples.push(hr(start + i, 60));
        }
        for i in 0..300 {
            samples.push(hr(start + 300 + i, 55));
        }
        assert_eq!(recovery_index_slope(&samples, start, start + 600), None);
    }

    #[test]
    fn recovery_index_slope_recovers_multiple_distinct_injected_slopes() {
        let flat = slope_series(62.0, 0.0, 6);
        let mild = slope_series(62.0, -1.0, 6);
        let steep = slope_series(68.0, -4.0, 6);
        let rising = slope_series(55.0, 2.0, 6);

        let fs = recovery_index_slope(&flat.0, flat.1, flat.2).unwrap();
        let ms = recovery_index_slope(&mild.0, mild.1, mild.2).unwrap();
        let ss = recovery_index_slope(&steep.0, steep.1, steep.2).unwrap();
        let rs = recovery_index_slope(&rising.0, rising.1, rising.2).unwrap();

        assert!((fs - 0.0).abs() < 0.3);
        assert!((ms - -1.0).abs() < 0.3);
        assert!((ss - -4.0).abs() < 0.3);
        assert!((rs - 2.0).abs() < 0.3);

        assert!(ss < ms && ms < fs && fs < rs);
    }

    #[test]
    fn default_null_terms_are_byte_identical_to_omitting_them() {
        let omitted = recovery(&RecoveryInput {
            hrv: 55.0,
            rhr: 52.0,
            resp: Some(14.0),
            hrv_baseline: Some(baseline(50.0, 6.0)),
            rhr_baseline: Some(baseline(55.0, 3.0)),
            resp_baseline: Some(baseline(14.5, 1.0)),
            sleep_perf: Some(0.9),
            skin_temp_dev: Some(0.4),
            ..Default::default()
        })
        .unwrap();
        let explicit = recovery(&RecoveryInput {
            hrv: 55.0,
            rhr: 52.0,
            resp: Some(14.0),
            hrv_baseline: Some(baseline(50.0, 6.0)),
            rhr_baseline: Some(baseline(55.0, 3.0)),
            resp_baseline: Some(baseline(14.5, 1.0)),
            sleep_perf: Some(0.9),
            skin_temp_dev: Some(0.4),
            hrv_baseline_usable: true,
            recovery_index_slope: None,
            effort_baseline: None,
            prior_day_effort: None,
        })
        .unwrap();
        assert!((omitted - explicit).abs() < 1e-9);
    }

    #[test]
    fn steeper_decline_raises_charge_more_than_flat_or_rising() {
        let score = |slope: Option<f64>| {
            recovery(&RecoveryInput {
                hrv: 50.0,
                rhr: 55.0,
                hrv_baseline: Some(baseline(50.0, 6.0)),
                rhr_baseline: Some(baseline(55.0, 3.0)),
                sleep_perf: Some(SLEEP_PERF_CENTER),
                recovery_index_slope: slope,
                ..Default::default()
            })
            .unwrap()
        };
        let flat = score(Some(0.0));
        let mild = score(Some(-1.0));
        let steep = score(Some(-4.0));
        let rising = score(Some(2.0));
        assert!(steep > mild && mild > flat && flat > rising);
    }

    #[test]
    fn activity_balance_high_effort_yesterday_lowers_charge() {
        let score = |effort: Option<f64>| {
            recovery(&RecoveryInput {
                hrv: 58.0,
                rhr: 50.0,
                hrv_baseline: Some(baseline(50.0, 6.0)),
                rhr_baseline: Some(baseline(55.0, 3.0)),
                sleep_perf: Some(0.92),
                effort_baseline: Some(baseline(40.0, 15.0)),
                prior_day_effort: effort,
                ..Default::default()
            })
            .unwrap()
        };
        let neutral = score(None);
        let at_baseline = score(Some(40.0));
        let rest_day = score(Some(10.0));
        let hard_day = score(Some(65.0));
        let very_hard = score(Some(90.0));

        assert!(at_baseline < neutral);
        assert!(rest_day > at_baseline);
        assert!(at_baseline > hard_day);
        assert!(rest_day > hard_day && hard_day > very_hard);
    }

    #[test]
    fn activity_balance_drops_term_unless_both_value_and_baseline_present() {
        let score = |effort: Option<f64>, eb: Option<DriverBaseline>| {
            recovery(&RecoveryInput {
                hrv: 50.0,
                rhr: 55.0,
                hrv_baseline: Some(baseline(50.0, 6.0)),
                rhr_baseline: Some(baseline(55.0, 3.0)),
                sleep_perf: Some(SLEEP_PERF_CENTER),
                effort_baseline: eb,
                prior_day_effort: effort,
                ..Default::default()
            })
            .unwrap()
        };
        let neither = score(None, None);
        let value_only = score(Some(80.0), None);
        let baseline_only = score(None, Some(baseline(40.0, 15.0)));
        let both = score(Some(80.0), Some(baseline(40.0, 15.0)));

        assert!((neither - value_only).abs() < 1e-9);
        assert!((neither - baseline_only).abs() < 1e-9);
        assert!((neither - both).abs() > 1e-9);
    }

    #[test]
    fn cold_start_gate_returns_none() {
        let out = recovery(&RecoveryInput {
            hrv: 60.0,
            rhr: 50.0,
            hrv_baseline: Some(baseline(50.0, 6.0)),
            hrv_baseline_usable: false,
            ..Default::default()
        });
        assert_eq!(out, None);
        // No driver at all → None even when usable.
        assert_eq!(recovery(&RecoveryInput { hrv: 60.0, rhr: 50.0, ..Default::default() }), None);
    }

    #[test]
    fn neutral_drivers_map_z_zero_to_population_center() {
        // Every driver exactly at baseline / centre → composite z = 0 → logistic anchor at ~57.93%.
        let out = recovery(&RecoveryInput {
            hrv: 50.0,
            rhr: 55.0,
            hrv_baseline: Some(baseline(50.0, 6.0)),
            rhr_baseline: Some(baseline(55.0, 3.0)),
            sleep_perf: Some(SLEEP_PERF_CENTER),
            ..Default::default()
        })
        .unwrap();
        let expected = 100.0 / (1.0 + (-LOGISTIC_K * (0.0 - LOGISTIC_Z0)).exp());
        assert!((out - expected).abs() < 1e-9);
        assert!((out - 57.93).abs() < 0.05);
    }

    #[test]
    fn band_thresholds_match_whoop_scheme() {
        assert_eq!(band(20.0), "red");
        assert_eq!(band(33.9), "red");
        assert_eq!(band(34.0), "yellow");
        assert_eq!(band(66.9), "yellow");
        assert_eq!(band(67.0), "green");
        assert_eq!(band(95.0), "green");
    }

    #[test]
    fn banked_nights_counts_only_in_bounds_hrv() {
        let nights = [Some(55.0), None, Some(3.0), Some(300.0), Some(80.0), Some(HRV_MIN_MS)];
        // 55, 80, and the 5.0 boundary count; None / 3 (< min) / 300 (> max) do not.
        assert_eq!(banked_nights(&nights, HRV_MIN_MS, HRV_MAX_MS), 3);
    }
}
