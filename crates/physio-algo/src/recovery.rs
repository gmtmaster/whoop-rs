//! Recovery scoring ("Charge") — a transparent z-score + logistic composite over nightly drivers
//! (HRV dominant, resting HR, respiration, sleep performance, skin-temp deviation, plus two optional
//! Oura-style terms: overnight HR-decline slope and previous-day effort). APPROXIMATE, not
//! WHOOP-identical. Plain-value inputs; each driver z is standardised against a personal baseline
//! (mean + EWMA-abs-dev spread). Missing terms drop and the weights renormalise. Wellness only.

pub use crate::hr_sample::HrSample;

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
/// Skin-temp deviation scale (°C per z-unit): the term is −|dev| / scale. Must stay 1.0 — halving it
/// doubles this term's effective penalty weight.
pub const SKIN_TEMP_DEV_SCALE: f64 = 1.0;
/// Recovery-Index (overnight HR-decline slope) weight and its bpm/hour scale.
pub const W_RECOVERY_INDEX: f64 = 0.05;
pub const RECOVERY_INDEX_SCALE_BPM_PER_HR: f64 = 2.0;
/// Activity-Balance (previous-day effort vs baseline) weight.
pub const W_ACTIVITY_BALANCE: f64 = 0.05;

/// Logistic spread and offset so Z = 0 maps to ~58% (population-average recovery).
pub const LOGISTIC_K: f64 = 1.6;
pub const LOGISTIC_Z0: f64 = -0.20;

/// Recovery-band thresholds (WHOOP colour scheme).
pub const BAND_RED_MAX: f64 = 34.0;
pub const BAND_YELLOW_MAX: f64 = 67.0;

/// Charge-state floors (0-100), each the bottom of the band it names. Finer than the three colour
/// bands above: these cut the one-word state a score reads as, not its colour.
pub const STATE_LOW_FLOOR: f64 = 25.0;
pub const STATE_MODERATE_FLOOR: f64 = 50.0;
pub const STATE_PRIMED_FLOOR: f64 = 70.0;
pub const STATE_PEAK_FLOOR: f64 = 88.0;

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

/// The one-word state a recovery score reads as, cut by the `STATE_*_FLOOR` constants.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RecoveryState {
    Depleted,
    Low,
    Moderate,
    Primed,
    Peak,
}

/// The state band a recovery score in [0, 100] falls in. The word and colour are the caller's.
pub fn state(score: f64) -> RecoveryState {
    if score < STATE_LOW_FLOOR {
        RecoveryState::Depleted
    } else if score < STATE_MODERATE_FLOOR {
        RecoveryState::Low
    } else if score < STATE_PRIMED_FLOOR {
        RecoveryState::Moderate
    } else if score < STATE_PEAK_FLOOR {
        RecoveryState::Primed
    } else {
        RecoveryState::Peak
    }
}

/// The Charge logistic: a composite z to a 0-100 score. The single squash both the score and its
/// per-driver breakdown read.
pub fn score_of(z: f64) -> f64 {
    let score = 100.0 / (1.0 + (-LOGISTIC_K * (z - LOGISTIC_Z0)).exp());
    score.clamp(0.0, 100.0)
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
    Some(score_of(z))
}

#[cfg(test)]
mod tests {
    use super::*;

    // DriverBaseline from a Gaussian sigma (spread is internal abs-dev units): spread = sigma / 1.253.
    fn baseline(mean: f64, sigma: f64) -> DriverBaseline {
        DriverBaseline {
            mean,
            spread: sigma / 1.253,
        }
    }

    fn hr(ts: i64, bpm: i32) -> HrSample {
        HrSample { ts, bpm }
    }

    // Synthetic full-night HR series with a KNOWN injected slope: a sample every 30 s, rounded to the
    // integer bpm the wire carries.
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

    // One sample dead-centre of each of `bins` consecutive windows, descending 1 bpm per bin.
    fn one_sample_per_bin(bins: usize) -> (Vec<HrSample>, i64, i64) {
        let origin: i64 = 100_000;
        let samples = (0..bins)
            .map(|b| hr(origin + b as i64 * RESTING_HR_WINDOW_S + 150, 60 - b as i32))
            .collect();
        (samples, origin, origin + bins as i64 * RESTING_HR_WINDOW_S)
    }

    /// Injected rates (bpm/hour) and the widest error any of them may show. The tolerance is the
    /// binning + integer-bpm error measured across the set, not a round number chosen for comfort.
    const SLOPE_TARGETS: [f64; 9] = [0.0, -0.5, -1.0, -2.0, -4.0, -8.0, 1.0, 2.0, 5.0];
    const SLOPE_TOL: f64 = 0.05;

    #[test]
    fn recovery_index_slope_null_when_the_window_holds_no_samples() {
        assert_eq!(recovery_index_slope(&[], 0, 1000), None);
        // Samples exist but all sit outside the in-bed window, which is the same emptiness.
        let outside: Vec<HrSample> = (0..600).map(|i| hr(20_000 + i, 60)).collect();
        assert_eq!(recovery_index_slope(&outside, 0, 1000), None);
    }

    #[test]
    fn recovery_index_needs_six_populated_bins_and_counts_bins_not_samples() {
        for bins in 1..=8 {
            let (s, a, b) = one_sample_per_bin(bins);
            let got = recovery_index_slope(&s, a, b);
            assert_eq!(
                got.is_some(),
                bins >= RECOVERY_INDEX_MIN_BINS,
                "{bins} populated bins returned {got:?}"
            );
        }
        // Density is not coverage: a solid five bins sampled every second is still refused.
        let origin: i64 = 100_000;
        let span = 5 * RESTING_HR_WINDOW_S;
        let dense: Vec<HrSample> = (0..span).map(|i| hr(origin + i, 60)).collect();
        assert_eq!(dense.len(), 1500);
        assert_eq!(recovery_index_slope(&dense, origin, origin + span), None);
        // Empty bins inside the window do not count against it: six populated of thirteen passes.
        let gapped: Vec<HrSample> = [0i64, 1, 2, 10, 11, 12]
            .iter()
            .map(|&b| hr(origin + b * RESTING_HR_WINDOW_S + 10, 70 - b as i32))
            .collect();
        assert!(recovery_index_slope(&gapped, origin, origin + 13 * RESTING_HR_WINDOW_S).is_some());
        // An estimator that always answers cannot survive the refusals above.
        for bins in 1..RECOVERY_INDEX_MIN_BINS {
            let (s, a, b) = one_sample_per_bin(bins);
            assert_ne!(Some(0.0), recovery_index_slope(&s, a, b), "{bins} bins");
        }
    }

    #[test]
    fn recovery_index_slope_recovers_the_injected_rate_in_bpm_per_hour() {
        let got: Vec<f64> = SLOPE_TARGETS
            .iter()
            .map(|&t| {
                let (s, a, b) = slope_series(65.0, t, 6);
                recovery_index_slope(&s, a, b).unwrap()
            })
            .collect();
        for (&target, &out) in SLOPE_TARGETS.iter().zip(&got) {
            assert!(
                (out - target).abs() < SLOPE_TOL,
                "injected {target} read as {out}"
            );
        }
        let mut pairs: Vec<(f64, f64)> = SLOPE_TARGETS
            .iter()
            .copied()
            .zip(got.iter().copied())
            .collect();
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        assert!(
            pairs.windows(2).all(|w| w[0].1 < w[1].1),
            "reading is not monotone in the rate"
        );

        // The unit is per HOUR, not per night: the same rate over 2 to 9 hours reads the same, and
        // the reading converges on it as the night lengthens.
        let errs: Vec<f64> = [2i64, 3, 6, 9]
            .iter()
            .map(|&h| {
                let (s, a, b) = slope_series(65.0, -2.0, h);
                (recovery_index_slope(&s, a, b).unwrap() + 2.0).abs()
            })
            .collect();
        assert!(
            errs.iter().all(|e| *e < 0.10),
            "the rate drifted with night length: {errs:?}"
        );
        assert!(
            errs.windows(2).all(|w| w[1] < w[0]),
            "longer nights did not converge: {errs:?}"
        );

        // Estimators that do no work: any single constant misses at least one injected rate, and a
        // night flattened to one bpm cannot reproduce the steep row.
        for null in [0.0f64, -2.0, 5.0] {
            let worst = SLOPE_TARGETS
                .iter()
                .map(|t| (null - t).abs())
                .fold(0.0, f64::max);
            assert!(
                worst > SLOPE_TOL,
                "constant estimator {null} fits every injected rate"
            );
        }
        let (flat, a, b) = slope_series(65.0, 0.0, 6);
        assert!((recovery_index_slope(&flat, a, b).unwrap() - -4.0).abs() > SLOPE_TOL);
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
        assert_eq!(
            recovery(&RecoveryInput {
                hrv: 60.0,
                rhr: 50.0,
                ..Default::default()
            }),
            None
        );
    }

    /// Nine three-driver nights and the score each must return, spanning 0 to 100. The expectations
    /// are literals, so a weight or logistic constant moving fails here instead of moving with itself.
    const SPREAD_NIGHTS: [(f64, f64, f64, f64); 9] = [
        (26.0, 70.0, 0.40, 0.171112969842),
        (32.0, 66.0, 0.55, 1.011365438792),
        (38.0, 62.0, 0.65, 5.167958386150),
        (44.0, 58.0, 0.75, 22.521055297547),
        (50.0, 55.0, 0.85, 57.932425214875),
        (56.0, 52.0, 0.90, 85.376542210808),
        (62.0, 49.0, 0.95, 96.116741498880),
        (70.0, 46.0, 1.00, 99.316784313320),
        (80.0, 42.0, 1.05, 99.924954045721),
    ];
    const SPREAD_TOL: f64 = 1e-9;

    // HRV / resting HR / sleep performance against a fixed personal baseline pair.
    fn three_driver(hrv: f64, rhr: f64, sleep_perf: f64) -> f64 {
        recovery(&RecoveryInput {
            hrv,
            rhr,
            hrv_baseline: Some(baseline(50.0, 6.0)),
            rhr_baseline: Some(baseline(55.0, 3.0)),
            sleep_perf: Some(sleep_perf),
            ..Default::default()
        })
        .unwrap()
    }

    #[test]
    fn charge_score_reproduces_the_composite_across_the_whole_0_100_range() {
        let got: Vec<f64> = SPREAD_NIGHTS
            .iter()
            .map(|&(h, r, s, _)| three_driver(h, r, s))
            .collect();
        for (&(h, r, s, want), &out) in SPREAD_NIGHTS.iter().zip(&got) {
            assert!(
                (out - want).abs() < SPREAD_TOL,
                "hrv {h} rhr {r} sleep {s}: {out} != {want}"
            );
        }
        // Each score is the weight-renormalised mean of the driver z's, squashed once. Resting HR is
        // inverted, so a sign flip on it moves every row.
        for (&(h, r, s, _), &out) in SPREAD_NIGHTS.iter().zip(&got) {
            let z = (W_HRV * z_score(h, 50.0, 6.0 / 1.253)
                + W_RHR * z_score(55.0, r, 3.0 / 1.253)
                + W_SLEEP * (s - SLEEP_PERF_CENTER) / SLEEP_PERF_SCALE)
                / (W_HRV + W_RHR + W_SLEEP);
            assert!(
                (out - score_of(z)).abs() < SPREAD_TOL,
                "hrv {h}: {out} != score_of({z})"
            );
        }
        // The fixture spans the range and rises with the composite, so no flat scorer can fit it.
        assert!(got.windows(2).all(|w| w[1] > w[0]), "not monotone: {got:?}");
        assert!(got.last().unwrap() - got.first().unwrap() > 99.0);
        for null in [
            57.932425214875f64,
            got.iter().sum::<f64>() / got.len() as f64,
            50.0,
        ] {
            let worst = SPREAD_NIGHTS
                .iter()
                .map(|&(.., w)| (null - w).abs())
                .fold(0.0, f64::max);
            assert!(
                worst > 10.0,
                "constant scorer {null} is within 10 points of every night"
            );
        }
        // Every driver exactly at baseline / centre is the population anchor the bands are cut around.
        let anchor = three_driver(50.0, 55.0, SLEEP_PERF_CENTER);
        assert!(
            (anchor - 100.0 / (1.0 + (-LOGISTIC_K * (0.0 - LOGISTIC_Z0)).exp())).abs() < SPREAD_TOL
        );
        assert!((anchor - 57.93).abs() < 0.05);
    }

    #[test]
    fn all_seven_drivers_renormalise_into_the_same_single_squash() {
        let night = |hrv, rhr, resp, sp, dev, slope, effort| {
            recovery(&RecoveryInput {
                hrv,
                rhr,
                resp: Some(resp),
                hrv_baseline: Some(baseline(50.0, 6.0)),
                rhr_baseline: Some(baseline(55.0, 3.0)),
                resp_baseline: Some(baseline(16.0, 2.0)),
                sleep_perf: Some(sp),
                skin_temp_dev: Some(dev),
                hrv_baseline_usable: true,
                recovery_index_slope: Some(slope),
                effort_baseline: Some(baseline(40.0, 15.0)),
                prior_day_effort: Some(effort),
            })
            .unwrap()
        };
        let supporting = night(62.0, 51.0, 15.0, 0.90, 0.4, -3.0, 75.0);
        let limiting = night(38.0, 62.0, 18.0, 0.62, 0.9, 1.5, 88.0);
        assert!((supporting - 91.257225183808).abs() < SPREAD_TOL);
        assert!((limiting - 5.719331490657).abs() < SPREAD_TOL);
        // Weights sum to 1.0 already, so seven terms and three must share one anchor, not two.
        let all_seven = night(50.0, 55.0, 16.0, SLEEP_PERF_CENTER, 0.0, 0.0, 40.0);
        assert!((all_seven - three_driver(50.0, 55.0, SLEEP_PERF_CENTER)).abs() < SPREAD_TOL);
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
    fn state_floors_partition_the_range_and_real_charge_scores_reach_all_five() {
        // Inclusive at every floor, exclusive one tenth of a point below it.
        assert_eq!(state(0.0), RecoveryState::Depleted);
        assert_eq!(state(24.9), RecoveryState::Depleted);
        assert_eq!(state(STATE_LOW_FLOOR), RecoveryState::Low);
        assert_eq!(state(49.9), RecoveryState::Low);
        assert_eq!(state(STATE_MODERATE_FLOOR), RecoveryState::Moderate);
        assert_eq!(state(69.9), RecoveryState::Moderate);
        assert_eq!(state(STATE_PRIMED_FLOOR), RecoveryState::Primed);
        assert_eq!(state(87.9), RecoveryState::Primed);
        assert_eq!(state(STATE_PEAK_FLOOR), RecoveryState::Peak);
        assert_eq!(state(100.0), RecoveryState::Peak);

        // One contiguous run per state over the displayed range, in order, no gap and no overlap.
        // The run starts are literals, so raising a floor fails here as loudly as lowering one.
        let mut runs: Vec<(RecoveryState, f64)> = Vec::new();
        for i in 0..=1000 {
            let s = i as f64 / 10.0;
            if runs.last().map(|r| r.0) != Some(state(s)) {
                runs.push((state(s), s));
            }
        }
        assert_eq!(
            runs,
            vec![
                (RecoveryState::Depleted, 0.0),
                (RecoveryState::Low, 25.0),
                (RecoveryState::Moderate, 50.0),
                (RecoveryState::Primed, 70.0),
                (RecoveryState::Peak, 88.0),
            ]
        );

        // Every state is reachable from a real night, not only from a hand-fed number.
        let reached: Vec<RecoveryState> = [
            (26.0, 70.0, 0.40),
            (46.0, 58.0, 0.75),
            (50.0, 55.0, 0.85),
            (56.0, 52.0, 0.90),
            (62.0, 49.0, 0.95),
        ]
        .iter()
        .map(|&(h, r, s)| state(three_driver(h, r, s)))
        .collect();
        assert_eq!(
            reached,
            vec![
                RecoveryState::Depleted,
                RecoveryState::Low,
                RecoveryState::Moderate,
                RecoveryState::Primed,
                RecoveryState::Peak,
            ]
        );

        // Classifiers that do no work: one word for every score, and a single split at the midpoint.
        let probes = [0.0, 24.9, 25.0, 49.9, 50.0, 69.9, 70.0, 87.9, 88.0, 100.0];
        let always: fn(f64) -> RecoveryState = |_| RecoveryState::Moderate;
        let split: fn(f64) -> RecoveryState = |s| {
            if s < 50.0 {
                RecoveryState::Low
            } else {
                RecoveryState::Primed
            }
        };
        for (name, null) in [("constant Moderate", always), ("one split at 50", split)] {
            let wrong = probes.iter().filter(|&&s| null(s) != state(s)).count();
            assert!(
                wrong >= 5,
                "null classifier '{name}' agreed on all but {wrong} probes"
            );
        }
    }

    #[test]
    fn banked_nights_counts_only_hrv_inside_the_validity_bounds() {
        // Both bounds are inclusive; missing, out-of-range and non-finite readings all bank nothing.
        // The 5.0 / 250.0 literals sit beside the constants so moving either bound fails.
        let probes: [(Option<f64>, usize, &str); 12] = [
            (None, 0, "no reading"),
            (Some(4.9999), 0, "just below the minimum"),
            (Some(HRV_MIN_MS), 1, "exactly the minimum"),
            (Some(5.0), 1, "the minimum as a literal"),
            (Some(55.0), 1, "mid-range"),
            (Some(HRV_MAX_MS), 1, "exactly the maximum"),
            (Some(250.0), 1, "the maximum as a literal"),
            (Some(250.0001), 0, "just above the maximum"),
            (Some(f64::NAN), 0, "not a number"),
            (Some(f64::INFINITY), 0, "infinite"),
            (Some(-1.0), 0, "negative"),
            (Some(0.0), 0, "zero"),
        ];
        for (v, want, what) in probes {
            assert_eq!(banked_nights(&[v], HRV_MIN_MS, HRV_MAX_MS), want, "{what}");
        }

        let nights = [
            Some(55.0),
            None,
            Some(3.0),
            Some(300.0),
            Some(80.0),
            Some(HRV_MIN_MS),
        ];
        assert_eq!(banked_nights(&nights, HRV_MIN_MS, HRV_MAX_MS), 3);
        // The bounds are arguments, not a hidden rule: widened, the same series banks every reading.
        assert_eq!(banked_nights(&nights, 0.0, f64::INFINITY), 5);
        // Counters that do no work: the slot count, the present-value count and always-nothing.
        for (name, n) in [
            ("every slot counted", nights.len()),
            (
                "every present value counted",
                nights.iter().filter(|v| v.is_some()).count(),
            ),
            ("nothing counted", 0),
        ] {
            assert_ne!(n, 3, "null counter '{name}' reproduces the shipped count");
        }
    }
}
