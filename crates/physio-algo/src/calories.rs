//! HR-based calorie estimation: Keytel 2005 active energy + revised Harris-Benedict BMR.
//! Per-bout and whole-day paths share the same per-second model but differ in how they weight
//! inter-sample gaps (elapsed-time vs flat one-second). APPROXIMATE, not laboratory calorimetry.

/// Sex-specific BMR + active-EE coefficients. Mirrors the Kotlin `Calories.Coeffs`.
#[derive(Clone, Copy, Debug)]
pub struct Coeffs {
    pub resting_alpha: f64,
    pub resting_weight: f64,
    /// Applied to height in METRES.
    pub resting_height: f64,
    pub resting_age: f64,
    pub workout_hr: f64,
    pub workout_weight: f64,
    pub workout_age: f64,
    pub workout_alpha: f64,
}

pub const MALE: Coeffs = Coeffs {
    resting_alpha: 88.362, resting_weight: 13.397, resting_height: 479.9,
    resting_age: 5.677, workout_hr: 0.6309, workout_weight: 0.1988,
    workout_age: 0.2017, workout_alpha: -55.0969,
};

pub const FEMALE: Coeffs = Coeffs {
    resting_alpha: 447.593, resting_weight: 9.247, resting_height: 309.8,
    resting_age: 4.33, workout_hr: 0.4472, workout_weight: -0.1263,
    workout_age: 0.0740, workout_alpha: -20.4022,
};

pub const NONBINARY: Coeffs = Coeffs {
    resting_alpha: 267.9775, resting_weight: 11.322, resting_height: 394.85,
    resting_age: 5.0035, workout_hr: 0.53905, workout_weight: 0.03625,
    workout_age: 0.13785, workout_alpha: -37.74955,
};

/// HR-fraction gate for the bout path: active rate applies above resting + this fraction × HRR.
pub const ACTIVE_HRR_FRACTION: f64 = 0.30;
/// HR-fraction gate for the day path: higher (50 %) so ordinary low-intensity HR stays on resting BMR.
pub const DAY_ACTIVE_HRR_FRACTION: f64 = 0.50;
/// 60 s/min × 4.184 kJ/kcal.
pub const WORKOUT_DIVISOR: f64 = 251.04;
/// Max inter-sample gap credited in the bout path (mirrors WorkoutDetector.mergeGapS).
pub const MERGE_GAP_S: f64 = 150.0;

/// Resolve the coefficient set for a sex string.
pub fn resolve_coeffs(sex: &str) -> Coeffs {
    match sex.to_lowercase().as_str() {
        "male" => MALE,
        "female" => FEMALE,
        _ => NONBINARY,
    }
}

/// Resting metabolic rate (kcal/s) from the revised Harris-Benedict equation.
pub fn resting_kcal_per_s(c: &Coeffs, weight_kg: f64, height_cm: f64, age: f64) -> f64 {
    let height_m = height_cm / 100.0;
    let bmr = c.resting_alpha + c.resting_weight * weight_kg
        + c.resting_height * height_m - c.resting_age * age;
    (bmr.max(0.0)) / 86_400.0
}

/// Active energy rate (kcal/s) from the Keytel 2005 equation.
pub fn active_kcal_per_s(c: &Coeffs, hr: f64, hrmax: f64, weight_kg: f64, age: f64) -> f64 {
    let ee_kj_min = c.workout_hr * hr.min(hrmax) + c.workout_weight * weight_kg
        + c.workout_age * age + c.workout_alpha;
    (ee_kj_min.max(0.0)) / WORKOUT_DIVISOR
}

/// One HR sample for calorie estimation.
#[derive(Clone, Copy, Debug)]
pub struct HrSample {
    pub ts: i64,
    pub bpm: i32,
}

/// Estimate (kcal, kJ) for a workout bout. Each sample is weighted by the elapsed time to the next
/// sample (capped at [MERGE_GAP_S]), so sparse streams are counted over real seconds.
/// Mirrors Kotlin `Calories.estimateBoutCalories`.
pub fn estimate_bout_calories(
    hr_samples: &[HrSample],
    weight_kg: f64,
    height_cm: f64,
    age: f64,
    sex: &str,
    hrmax: f64,
    resting_hr: f64,
) -> (f64, f64) {
    if hr_samples.is_empty() {
        return (0.0, 0.0);
    }

    let coeffs = resolve_coeffs(sex);
    let hr_reserve = (hrmax - resting_hr).max(1.0);
    let active_threshold = resting_hr + ACTIVE_HRR_FRACTION * hr_reserve;
    let resting_rate = resting_kcal_per_s(&coeffs, weight_kg, height_cm, age);

    let mut ordered: Vec<&HrSample> = hr_samples.iter().collect();
    ordered.sort_by_key(|s| s.ts);

    let mut total_kcal = 0.0;
    for i in 0..ordered.len() {
        let bpm = ordered[i].bpm as f64;
        let dur: f64 = if i < ordered.len() - 1 {
            let gap = (ordered[i + 1].ts - ordered[i].ts) as f64;
            if gap > 0.0 { gap.min(MERGE_GAP_S) } else { 1.0 }
        } else {
            1.0
        };
        total_kcal += if bpm < active_threshold {
            resting_rate * dur
        } else {
            active_kcal_per_s(&coeffs, bpm, hrmax, weight_kg, age) * dur
        };
    }
    (total_kcal, total_kcal * 4.184)
}

/// Whole-day energy estimate (kcal) from HR samples. Each sample counts as exactly one second
/// (flat per-sample, no elapsed-time weighting). Uses [DAY_ACTIVE_HRR_FRACTION] gate.
/// Mirrors Kotlin `Calories.estimateDayCalories`.
pub fn estimate_day_calories(
    hr_samples: &[HrSample],
    weight_kg: f64,
    height_cm: f64,
    age: f64,
    sex: &str,
    hrmax: f64,
    resting_hr: f64,
) -> f64 {
    if hr_samples.is_empty() {
        return 0.0;
    }

    let coeffs = resolve_coeffs(sex);
    let hr_reserve = (hrmax - resting_hr).max(1.0);
    let active_threshold = resting_hr + DAY_ACTIVE_HRR_FRACTION * hr_reserve;
    let resting_rate = resting_kcal_per_s(&coeffs, weight_kg, height_cm, age);

    let mut total_kcal = 0.0;
    for s in hr_samples {
        let bpm = s.bpm as f64;
        total_kcal += if bpm < active_threshold {
            resting_rate
        } else {
            resting_rate.max(active_kcal_per_s(&coeffs, bpm, hrmax, weight_kg, age))
        };
    }
    total_kcal
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hr_day(bpm: i32, n: usize) -> Vec<HrSample> {
        (0..n).map(|i| HrSample { ts: i as i64, bpm }).collect()
    }

    fn default_profile() -> (f64, f64, f64, &'static str) {
        (70.0, 170.0, 30.0, "nonbinary")
    }

    #[test]
    fn day_calories_empty_is_zero() {
        assert_eq!(estimate_day_calories(&[], 70.0, 170.0, 30.0, "nonbinary", 190.0, 55.0), 0.0);
    }

    #[test]
    fn day_calories_matches_bout_at_one_hz() {
        let profile = (80.0, 180.0, 35.0, "male");
        let hr = hr_day(130, 600);
        let day = estimate_day_calories(&hr, profile.0, profile.1, profile.2, profile.3, 185.0, 55.0);
        let (bout, _) = estimate_bout_calories(&hr, profile.0, profile.1, profile.2, profile.3, 185.0, 55.0);
        assert!((bout - day).abs() < 1e-9);
    }

    #[test]
    fn sparse_hr_tracks_elapsed_time_not_sample_count() {
        let profile = (80.0, 180.0, 35.0, "male");
        let dense: Vec<HrSample> = (0..600).map(|i| HrSample { ts: i, bpm: 130 }).collect();
        let sparse: Vec<HrSample> = (0..600).step_by(10).map(|i| HrSample { ts: i, bpm: 130 }).collect();
        let (dense_kcal, _) = estimate_bout_calories(&dense, profile.0, profile.1, profile.2, profile.3, 185.0, 55.0);
        let (sparse_kcal, _) = estimate_bout_calories(&sparse, profile.0, profile.1, profile.2, profile.3, 185.0, 55.0);
        assert!((sparse_kcal - dense_kcal).abs() < dense_kcal * 0.05);
        assert!(sparse_kcal > dense_kcal * 0.5);
    }

    #[test]
    fn wear_gap_is_capped_not_credited_in_full() {
        let profile = (80.0, 180.0, 35.0, "male");
        let gapped = vec![
            HrSample { ts: 0, bpm: 130 },
            HrSample { ts: 3600, bpm: 130 },
        ];
        let capped_equiv: Vec<HrSample> = (0..=150).map(|i| HrSample { ts: i, bpm: 130 }).collect();
        let (gap_kcal, _) = estimate_bout_calories(&gapped, profile.0, profile.1, profile.2, profile.3, 185.0, 55.0);
        let (equiv_kcal, _) = estimate_bout_calories(&capped_equiv, profile.0, profile.1, profile.2, profile.3, 185.0, 55.0);
        assert!((gap_kcal - equiv_kcal).abs() < equiv_kcal * 0.001);
    }

    #[test]
    fn day_path_does_not_overcount_gappy_days() {
        let profile = (80.0, 180.0, 35.0, "male");
        let gapped = vec![
            HrSample { ts: 0, bpm: 130 },
            HrSample { ts: 3600, bpm: 130 },
        ];
        let two_adjacent = vec![
            HrSample { ts: 0, bpm: 130 },
            HrSample { ts: 1, bpm: 130 },
        ];
        let gap_day = estimate_day_calories(&gapped, profile.0, profile.1, profile.2, profile.3, 185.0, 55.0);
        let adj_day = estimate_day_calories(&two_adjacent, profile.0, profile.1, profile.2, profile.3, 185.0, 55.0);
        assert!((gap_day - adj_day).abs() < 1e-9);
        let (bout_gapped, _) = estimate_bout_calories(&gapped, profile.0, profile.1, profile.2, profile.3, 185.0, 55.0);
        assert!(bout_gapped > gap_day * 10.0);
    }

    #[test]
    fn resting_day_is_lower_than_active_day() {
        let profile = default_profile();
        let resting = estimate_day_calories(&hr_day(60, 3600), profile.0, profile.1, profile.2, profile.3, 185.0, 55.0);
        let active = estimate_day_calories(&hr_day(150, 3600), profile.0, profile.1, profile.2, profile.3, 185.0, 55.0);
        assert!(resting > 0.0);
        assert!(active > resting);
    }

    #[test]
    fn sedentary_full_day_approximates_bmr() {
        let profile = (80.0, 180.0, 35.0, "male");
        let sedentary = hr_day(55, 86_400);
        let total = estimate_day_calories(&sedentary, profile.0, profile.1, profile.2, profile.3, 185.0, 55.0);
        assert!((total - 1825.25).abs() < 1.0);
    }

    #[test]
    fn light_activity_day_is_far_below_old_inflated_total() {
        let profile = (80.0, 180.0, 35.0, "male");
        let mut light_day = Vec::new();
        light_day.extend(hr_day(55, 8 * 3600));
        light_day.extend(hr_day(70, 8 * 3600));
        light_day.extend(hr_day(100, 8 * 3600));
        let total = estimate_day_calories(&light_day, profile.0, profile.1, profile.2, profile.3, 185.0, 55.0);
        assert!((total - 1825.25).abs() < 1.0);
        assert!(total < 4768.0 - 2000.0);
    }

    #[test]
    fn coeff_resolves_sex() {
        let m = resolve_coeffs("male");
        assert!((m.resting_alpha - 88.362).abs() < 1e-9);
        let f = resolve_coeffs("female");
        assert!((f.resting_alpha - 447.593).abs() < 1e-9);
        // unknown falls to nonbinary
        let n = resolve_coeffs("unknown");
        assert!((n.resting_alpha - 267.9775).abs() < 1e-9);
        assert!((n.resting_alpha - NONBINARY.resting_alpha).abs() < 1e-9);
    }

    #[test]
    fn resting_kcal_per_s_produces_positive_rate() {
        let c = MALE;
        let rate = resting_kcal_per_s(&c, 80.0, 180.0, 35.0);
        assert!(rate > 0.0);
        // For an 80kg/180cm/35y male: BMR ≈ 1825 kcal/day → ~0.0211 kcal/s
        let day_kcal = rate * 86_400.0;
        assert!((day_kcal - 1825.0).abs() < 100.0);
    }

    #[test]
    fn active_kcal_per_s_tracks_intensity() {
        let c = MALE;
        let low = active_kcal_per_s(&c, 100.0, 185.0, 80.0, 35.0);
        let high = active_kcal_per_s(&c, 160.0, 185.0, 80.0, 35.0);
        assert!(high > low);
    }
}
