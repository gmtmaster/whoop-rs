//! Strain / cardiovascular Effort (0–100) from an HR series: Karvonen %HRR per sample, TRIMP
//! accumulation (Edwards 5-zone or Banister exponential), then a logarithmic map onto [0, 100].
//! Pure; a light or absent series scores an honest 0 and too-little data returns `None`.

pub const MIN_READINGS: usize = 600;
pub const MIN_SPARSE_READINGS: usize = 20;
pub const MIN_SPAN_SECONDS: i64 = 600;
pub const MAX_STRAIN: f64 = 100.0;
pub const STRAIN_DENOMINATOR: f64 = 7201.0;
pub const FALLBACK_SAMPLE_MIN: f64 = 1.0 / 60.0;
pub const DEFAULT_AGE: i32 = 30;
pub const DEFAULT_RESTING_HR: f64 = 60.0;
pub const HRMAX_MIN_SAMPLES: usize = 600;
pub const HRMAX_PERCENTILE: f64 = 99.5;
pub const BANISTER_SCALE: f64 = 0.64;
pub const BANISTER_B_MEN: f64 = 1.92;
pub const BANISTER_B_WOMEN: f64 = 1.67;
pub const DROPOUT_CAP_SECONDS: i64 = 1200;

/// Edwards cut-offs as (%HRR threshold, weight), highest-first.
const EDWARDS_ZONES: [(f64, i64); 5] = [(90.0, 5), (80.0, 4), (70.0, 3), (60.0, 2), (50.0, 1)];

pub use crate::hr_sample::HrSample;

/// TRIMP accumulation method.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Method {
    Edwards,
    Banister,
}

/// Denominator-fit failure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StrainError {
    TooFewPairs,
    Degenerate,
}

/// Tanaka HRmax = 208 − 0.7 × age (gender-independent).
pub fn tanaka_hrmax(age: f64) -> f64 {
    208.0 - 0.7 * age
}

/// Classic 220 − age; last-resort fallback.
pub fn default_max_hr(age: i32) -> i32 {
    220 - age
}

/// [`stats::percentile`] on the 0..100 scale, for the constants here that are written as percents.
/// The unit is in the name because the two take the SAME argument on different scales, and a
/// bare `percentile` imported from either would compile against both.
///
/// [`stats::percentile`]: crate::stats::percentile
pub fn percentile_pct(sorted_values: &[f64], pct: f64) -> f64 {
    crate::stats::percentile(sorted_values, pct / 100.0)
}

/// Personalized HRmax from a trailing HR series → (bpm, source ∈ observed/tanaka/unknown).
pub fn estimate_hrmax(hr_history: &[f64], age: Option<f64>) -> (f64, &'static str) {
    let tanaka = age.map(tanaka_hrmax);
    if hr_history.len() >= HRMAX_MIN_SAMPLES {
        let mut sorted = hr_history.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let observed = percentile_pct(&sorted, HRMAX_PERCENTILE);
        return match tanaka {
            None => (observed, "observed"),
            Some(t) if observed >= t => (observed, "observed"),
            Some(t) => (t, "tanaka"),
        };
    }
    match tanaka {
        Some(t) => (t, "tanaka"),
        None => (0.0, "unknown"),
    }
}

/// Karvonen %HRR, clamped [0, 100].
pub fn pct_hrr(bpm: f64, resting_hr: f64, hr_reserve: f64) -> f64 {
    ((bpm - resting_hr) / hr_reserve * 100.0).clamp(0.0, 100.0)
}

/// Edwards 5-zone weight (0–5) from %HRR (unclamped; extremes agree with the clamped path).
pub fn zone_weight(bpm: f64, resting_hr: f64, hr_reserve: f64) -> i64 {
    let pct = (bpm - resting_hr) / hr_reserve * 100.0;
    for (threshold, weight) in EDWARDS_ZONES {
        if pct >= threshold {
            return weight;
        }
    }
    0
}

/// Per-sample duration (minutes) from the first two timestamps; 1 s fallback.
pub fn sample_duration_minutes(hr: &[HrSample]) -> f64 {
    if hr.len() < 2 {
        return FALLBACK_SAMPLE_MIN;
    }
    let delta_s = (hr[1].ts - hr[0].ts).abs() as f64;
    if delta_s > 0.0 {
        delta_s / 60.0
    } else {
        FALLBACK_SAMPLE_MIN
    }
}

pub fn banister_trimp(hr: &[HrSample], resting_hr: f64, hr_reserve: f64, sample_dur_min: f64, b: f64) -> f64 {
    let mut acc = 0.0;
    for s in hr {
        let x = pct_hrr(s.bpm as f64, resting_hr, hr_reserve) / 100.0;
        if x > 0.0 {
            acc += sample_dur_min * x * BANISTER_SCALE * (b * x).exp();
        }
    }
    acc
}

/// Per-interval Edwards TRIMP: each sample's zone weight times its own interval-derived duration
/// (minutes). Uses the gap to the nearest neighbour(s), capped at `dropout_cap_s` so a data
/// dropout cannot inflate one sample's contribution.
pub fn edwards_trimp_interval(hr: &[HrSample], resting_hr: f64, hr_reserve: f64, dropout_cap_s: i64) -> f64 {
    let n = hr.len();
    if n == 0 {
        return 0.0;
    }
    let mut total = 0.0;
    for i in 0..n {
        let w = zone_weight(hr[i].bpm as f64, resting_hr, hr_reserve) as f64;
        let gap_s = if n == 1 {
            60.0
        } else if i == 0 {
            ((hr[1].ts - hr[0].ts).unsigned_abs() as i64).min(dropout_cap_s) as f64
        } else if i == n - 1 {
            ((hr[i].ts - hr[i - 1].ts).unsigned_abs() as i64).min(dropout_cap_s) as f64
        } else {
            let fwd = ((hr[i + 1].ts - hr[i].ts).unsigned_abs() as i64).min(dropout_cap_s) as f64;
            let bwd = ((hr[i].ts - hr[i - 1].ts).unsigned_abs() as i64).min(dropout_cap_s) as f64;
            (fwd + bwd) / 2.0
        };
        total += w * gap_s / 60.0;
    }
    total
}

/// Map accumulated TRIMP onto [0, 100] via 100 × ln(TRIMP+1) / ln(D), 2 dp. TRIMP ≤ 0 → 0.
pub fn trimp_to_strain(trimp: f64, denominator: f64) -> f64 {
    if trimp <= 0.0 {
        return 0.0;
    }
    let value = MAX_STRAIN * (trimp + 1.0).ln() / denominator.ln();
    (value * 100.0).round() / 100.0
}

/// Calibrate D from (TRIMP, reference_strain) pairs via the through-origin least-squares line:
/// ln(D) = maxStrain × Σ(x²) / Σ(xy), x = ln(TRIMP+1). Reference strains are on the 0–100 scale.
pub fn fit_strain_denominator(pairs: &[(f64, f64)]) -> Result<f64, StrainError> {
    let usable: Vec<(f64, f64)> = pairs.iter().copied().filter(|(t, s)| *t > 0.0 && *s > 0.0).collect();
    if usable.len() < 2 {
        return Err(StrainError::TooFewPairs);
    }
    let mut sum_xx = 0.0;
    let mut sum_xy = 0.0;
    for (trimp, strain) in usable {
        let x = (trimp + 1.0).ln();
        sum_xx += x * x;
        sum_xy += x * strain;
    }
    if !(sum_xy > 0.0 && sum_xx > 0.0) {
        return Err(StrainError::Degenerate);
    }
    Ok((MAX_STRAIN * sum_xx / sum_xy).exp())
}

/// Cardiovascular Effort (0–100) from an HR series, or `None` when there isn't enough data
/// (fewer than [`MIN_READINGS`] samples AND under [`MIN_SPAN_SECONDS`] of coverage) or HRR ≤ 0.
pub fn strain(
    hr: &[HrSample],
    max_hr: Option<f64>,
    resting_hr: f64,
    method: Method,
    sex: &str,
    denominator: f64,
) -> Option<f64> {
    let eff_max = max_hr.unwrap_or_else(|| default_max_hr(DEFAULT_AGE) as f64);
    let enough_data = if hr.len() >= MIN_READINGS {
        true
    } else if hr.len() >= MIN_SPARSE_READINGS {
        let max = hr.iter().map(|s| s.ts).max().unwrap_or(0);
        let min = hr.iter().map(|s| s.ts).min().unwrap_or(0);
        max - min >= MIN_SPAN_SECONDS
    } else {
        false
    };
    if !enough_data || eff_max <= resting_hr {
        return None;
    }

    let hr_reserve = eff_max - resting_hr;
    let trimp = match method {
        Method::Banister => {
            let sample_dur = sample_duration_minutes(hr);
            let b = if sex.to_lowercase().starts_with('f') { BANISTER_B_WOMEN } else { BANISTER_B_MEN };
            banister_trimp(hr, resting_hr, hr_reserve, sample_dur, b)
        }
        Method::Edwards => edwards_trimp_interval(hr, resting_hr, hr_reserve, DROPOUT_CAP_SECONDS),
    };
    Some(trimp_to_strain(trimp, denominator))
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 0.01;

    fn hr_constant(bpm: i32, n: usize) -> Vec<HrSample> {
        (0..n).map(|i| HrSample { ts: i as i64, bpm }).collect()
    }

    fn hr_every(bpm: i32, n: usize, step_s: i64) -> Vec<HrSample> {
        (0..n).map(|i| HrSample { ts: i as i64 * step_s, bpm }).collect()
    }

    fn eff(hr: &[HrSample], max_hr: f64, resting_hr: f64) -> Option<f64> {
        strain(hr, Some(max_hr), resting_hr, Method::Edwards, "male", STRAIN_DENOMINATOR)
    }

    #[test]
    fn trimp_to_strain_goldens() {
        assert_eq!(trimp_to_strain(0.0, STRAIN_DENOMINATOR), 0.0);
        assert_eq!(trimp_to_strain(-5.0, STRAIN_DENOMINATOR), 0.0);
        assert!((trimp_to_strain(100.0, STRAIN_DENOMINATOR) - 51.96).abs() < EPS);
        assert!((trimp_to_strain(500.0, STRAIN_DENOMINATOR) - 69.99).abs() < EPS);
        assert!((trimp_to_strain(1000.0, STRAIN_DENOMINATOR) - 77.78).abs() < EPS);
        assert!((trimp_to_strain(3600.0, STRAIN_DENOMINATOR) - 92.20).abs() < EPS);
        assert!((trimp_to_strain(7200.0, STRAIN_DENOMINATOR) - 100.0).abs() < EPS);
    }

    #[test]
    fn edwards_zone_goldens() {
        // rest 60, max 160 → HRR 100 → %HRR = bpm − 60; 600 samples at 1 Hz, per-interval gaps=1s each.
        // Each sample: gap=1s/60=1/60 min. total = 600 × weight × 1/60 = 10 × weight.
        let v115 = eff(&hr_constant(115, 600), 160.0, 60.0).unwrap(); // zone 1 (50-59%): weight=1
        assert!((v115 - 27.0).abs() < EPS, "115bpm got {v115}");
        let v135 = eff(&hr_constant(135, 600), 160.0, 60.0).unwrap(); // zone 3 (70-79%): weight=3
        assert!((v135 - 38.66).abs() < EPS, "135bpm got {v135}");
        let v155 = eff(&hr_constant(155, 600), 160.0, 60.0).unwrap(); // zone 5 (90-100%): weight=5
        assert!((v155 - 44.27).abs() < EPS, "155bpm got {v155}");
    }

    #[test]
    fn interval_method_handles_cadence_transition() {
        // First gap 30s, remaining 599 samples at 1s — old method inflated, new method correct.
        let mut hr = Vec::new();
        hr.push(HrSample { ts: 0, bpm: 135 });
        for i in 1..600 {
            hr.push(HrSample { ts: 30 + i as i64 - 1, bpm: 135 });
        }
        let v = eff(&hr, 160.0, 60.0).unwrap();
        // Each sample gets its own interval (1s for most, 30s for first). Sum of gaps ≈ 30+599*1=629s.
        // weight=3, total = 3 × 629/60 = 31.45 min. TRIMP = 31.45, strain ≈ ...
        assert!(v < 40.0, "cadence transition should not inflate, got {v}");
        assert!(v > 20.0, "cadence transition should score, got {v}");
    }

    #[test]
    fn interval_method_caps_dropout() {
        // Two clusters with a 1-hour gap — gap capped at 20 min dropout.
        let mut hr = Vec::new();
        for i in 0..600 { hr.push(HrSample { ts: i as i64, bpm: 135 }); }
        let gap_start = 3600i64;
        for i in 0..600 { hr.push(HrSample { ts: gap_start + i as i64, bpm: 135 }); }
        let v = eff(&hr, 160.0, 60.0).unwrap();
        assert!(v > 0.0, "gapped day still scores");
    }

    #[test]
    fn single_interval_sample_does_not_crash() {
        let hr = vec![HrSample { ts: 0, bpm: 120 }];
        assert!(eff(&hr, 160.0, 60.0).is_none()); // <600 samples
        // but the function itself shouldn't crash on 1 sample
        let trimp = edwards_trimp_interval(&hr, 60.0, 100.0, 1200);
        assert!(trimp > 0.0);
    }

    #[test]
    fn uniform_stream_credits_one_second_per_sample() {
        // 600 samples 1 s apart at zone-3 intensity: every gap is 1 s, so TRIMP = 600 × 3 × 1/60 = 30 min.
        let uniform = hr_constant(135, 600);
        let trimp = edwards_trimp_interval(&uniform, 60.0, 100.0, 1200);
        assert!((trimp - 30.0).abs() < 0.1, "got {trimp}");
    }

    #[test]
    fn null_when_too_few_or_invalid_hrr() {
        assert!(eff(&hr_constant(135, 599), 160.0, 60.0).is_none());
        assert!(eff(&hr_constant(135, 600), 60.0, 60.0).is_none());
    }

    #[test]
    fn sparse_stream_scores_once_it_spans_enough_time() {
        let sparse = hr_every(155, 30, 30);
        assert!(sparse.last().unwrap().ts - sparse.first().unwrap().ts >= MIN_SPAN_SECONDS);
        assert!(eff(&sparse, 160.0, 60.0).is_some());
    }

    #[test]
    fn sparse_stream_null_under_sample_floor() {
        let too_few = hr_every(155, 5, 200);
        assert!(eff(&too_few, 160.0, 60.0).is_none());
    }

    #[test]
    fn light_day_honestly_scores_zero() {
        assert_eq!(eff(&hr_constant(105, 1200), 184.0, 60.0).unwrap(), 0.0);
        assert_eq!(eff(&hr_every(105, 40, 30), 184.0, 60.0).unwrap(), 0.0);
    }

    #[test]
    fn sparse_stream_scores_real_workout() {
        let s = eff(&hr_every(175, 40, 30), 184.0, 60.0);
        assert!(s.is_some() && s.unwrap() > 0.0);
    }

    #[test]
    fn banister_tracks_rising_load() {
        // Banister is monotonic in intensity: a higher %HRR stream must out-score a lower one.
        let low = strain(&hr_constant(120, 600), Some(184.0), 60.0, Method::Banister, "male", STRAIN_DENOMINATOR);
        let high = strain(&hr_constant(175, 600), Some(184.0), 60.0, Method::Banister, "male", STRAIN_DENOMINATOR);
        assert!(high.unwrap() > low.unwrap());
    }

    #[test]
    fn hrmax_and_percentile() {
        assert!((tanaka_hrmax(30.0) - 187.0).abs() < 1e-9);
        let sorted: Vec<f64> = (0..=100).map(|v| v as f64).collect();
        assert!((percentile_pct(&sorted, 50.0) - 50.0).abs() < 1e-9);
        assert_eq!(estimate_hrmax(&[], Some(40.0)).1, "tanaka");
        assert_eq!(estimate_hrmax(&[], None).1, "unknown");
    }

    #[test]
    fn fit_denominator_recovers_seed() {
        // Round-trip: strains generated from D=7201 refit back to ~7201 (through-origin fit).
        let pairs: Vec<(f64, f64)> = [100.0, 500.0, 1000.0, 3600.0]
            .iter()
            .map(|&t| (t, trimp_to_strain(t, STRAIN_DENOMINATOR)))
            .collect();
        let d = fit_strain_denominator(&pairs).unwrap();
        assert!((d - STRAIN_DENOMINATOR).abs() / STRAIN_DENOMINATOR < 0.01, "got {d}");
        assert_eq!(fit_strain_denominator(&[(100.0, 50.0)]), Err(StrainError::TooFewPairs));
    }
}
