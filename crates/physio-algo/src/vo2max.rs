//! VO2max + Fitness Age from resting HR, physical activity and profile. Non-exercise estimate
//! (Nes 2011 HUNT waist-circumference model); the Fitness Age inverts the same equation so the body
//! term cancels. Protocol-free: plain profile scalars in, wellness estimate out, never medical.

// Nes waist coefficients: VO2 = intercept - ageC*age + paiC*PA - wcC*waist - rhrC*RHR.
const MEN: Coeffs = Coeffs { intercept: 100.27, age: 0.296, wc: 0.369, rhr: 0.155, pai: 0.226 };
const WOMEN: Coeffs = Coeffs { intercept: 74.74, age: 0.247, wc: 0.259, rhr: 0.114, pai: 0.198 };

pub const SEE_MEN: f64 = 5.70;
pub const SEE_WOMEN: f64 = 5.14;

// Normative "average peer" the Fitness Age compares against.
pub const RESTING_HR_REFERENCE: f64 = 65.0;
pub const PAI_REFERENCE: f64 = 5.0;

pub const DISPLAY_BAND_YEARS: f64 = 5.0;
pub const MIN_AGE: f64 = 20.0;
pub const MAX_AGE: f64 = 80.0;

struct Coeffs {
    intercept: f64,
    age: f64,
    wc: f64,
    rhr: f64,
    pai: f64,
}

fn is_female(sex: &str) -> bool {
    sex.eq_ignore_ascii_case("female")
}

fn coeffs(sex: &str) -> &'static Coeffs {
    if is_female(sex) {
        &WOMEN
    } else {
        &MEN
    }
}

/// Nes waist-variant VO2max (ml/kg/min). Needs a waist measurement.
pub fn estimate_vo2max(age: f64, sex: &str, waist_cm: f64, resting_hr: f64, pa_index: f64) -> f64 {
    let c = coeffs(sex);
    c.intercept - c.age * age + c.pai * pa_index - c.wc * waist_cm - c.rhr * resting_hr
}

/// Self-consistent Fitness Age (years, clamped `[MIN_AGE, MAX_AGE]`). The waist term cancels.
pub fn fitness_age(age: f64, sex: &str, resting_hr: f64, pa_index: f64) -> f64 {
    let c = coeffs(sex);
    let fa = age
        + (c.rhr * (resting_hr - RESTING_HR_REFERENCE) - c.pai * (pa_index - PAI_REFERENCE)) / c.age;
    fa.clamp(MIN_AGE, MAX_AGE)
}

fn frequency_factor(active_days_per_week: i32) -> f64 {
    match active_days_per_week {
        d if d < 1 => 0.0,
        1 => 0.5,
        2 => 1.0,
        d if d <= 4 => 2.5,
        _ => 5.0,
    }
}

/// HUNT PA-index (0-15 = frequency x intensity x duration) from measured weekly aggregates.
pub fn physical_activity_index(
    active_days_per_week: i32,
    avg_active_minutes_per_day: f64,
    high_intensity_fraction: f64,
) -> f64 {
    let frequency = frequency_factor(active_days_per_week);
    let intensity = if high_intensity_fraction < 0.15 {
        1.0
    } else if high_intensity_fraction < 0.5 {
        2.0
    } else {
        3.0
    };
    let duration = if avg_active_minutes_per_day < 15.0 {
        0.10
    } else if avg_active_minutes_per_day < 30.0 {
        0.38
    } else if avg_active_minutes_per_day < 60.0 {
        0.75
    } else {
        1.0
    };
    if frequency == 0.0 {
        return 0.0;
    }
    frequency * intensity * duration
}

/// PA-index (0-15) from mean active-day strain — strain already folds intensity x duration.
pub fn physical_activity_index_from_strain(active_days_per_week: i32, mean_active_strain: f64) -> f64 {
    let frequency = frequency_factor(active_days_per_week);
    if frequency == 0.0 {
        return 0.0;
    }
    let intensity_duration = (mean_active_strain / 30.0).clamp(0.0, 3.0);
    frequency * intensity_duration
}

/// A computed Fitness Age with the inputs to present it. `vo2max` is filled only with a waist.
/// `advance_years` follows the same convention as the biological-age model: POSITIVE = older than
/// chronological (worse), negative = younger (fitter).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FitnessAgeResult {
    pub vo2max: Option<f64>,
    pub fitness_age: f64,
    pub chrono_age: f64,
    pub advance_years: f64,
    pub band_years: f64,
    pub lower_confidence: bool,
}

/// Full Fitness Age. `None` only if RHR or age is missing.
pub fn compute(
    age: f64,
    sex: &str,
    resting_hr: f64,
    pa_index: f64,
    waist_cm: Option<f64>,
    lower_confidence: bool,
) -> Option<FitnessAgeResult> {
    if age <= 0.0 || resting_hr <= 0.0 {
        return None;
    }
    let fa = fitness_age(age, sex, resting_hr, pa_index);
    let vo2 = match waist_cm {
        Some(w) if w > 0.0 => Some(estimate_vo2max(age, sex, w, resting_hr, pa_index)),
        _ => None,
    };
    let nb = !sex.eq_ignore_ascii_case("male") && !sex.eq_ignore_ascii_case("female");
    Some(FitnessAgeResult {
        vo2max: vo2,
        fitness_age: fa,
        chrono_age: age,
        advance_years: fa - age,
        band_years: DISPLAY_BAND_YEARS,
        lower_confidence: lower_confidence || nb,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, eps: f64) {
        assert!((a - b).abs() < eps, "{a} != {b} (eps {eps})");
    }

    #[test]
    fn vo2max_men() {
        approx(estimate_vo2max(40.0, "male", 90.0, 65.0, 5.0), 46.275, 1e-3);
    }

    #[test]
    fn vo2max_women() {
        approx(estimate_vo2max(40.0, "female", 80.0, 65.0, 5.0), 37.72, 1e-3);
    }

    #[test]
    fn bmi_helper() {
    }

    #[test]
    fn reference_fit_person_equals_chrono_age() {
        approx(fitness_age(40.0, "male", 65.0, 5.0), 40.0, 1e-9);
        approx(fitness_age(55.0, "female", 65.0, 5.0), 55.0, 1e-9);
    }

    #[test]
    fn fitter_is_younger() {
        approx(fitness_age(40.0, "male", 50.0, 10.0), 28.33, 0.05);
    }

    #[test]
    fn unfitter_is_older() {
        approx(fitness_age(40.0, "male", 80.0, 2.0), 50.15, 0.05);
    }

    #[test]
    fn clamp_high() {
        approx(fitness_age(75.0, "male", 120.0, 0.0), 80.0, 1e-9);
    }

    #[test]
    fn clamp_low() {
        approx(fitness_age(25.0, "male", 35.0, 15.0), 20.0, 1e-9);
    }

    #[test]
    fn pai_sedentary() {
        approx(physical_activity_index(0, 0.0, 0.0), 0.0, 1e-9);
    }

    #[test]
    fn pai_high() {
        approx(physical_activity_index(7, 75.0, 0.8), 15.0, 1e-9);
    }

    #[test]
    fn pai_moderate() {
        approx(physical_activity_index(3, 40.0, 0.3), 3.75, 1e-9);
    }

    #[test]
    fn pai_from_strain() {
        approx(physical_activity_index_from_strain(0, 0.0), 0.0, 1e-9);
        approx(physical_activity_index_from_strain(7, 90.0), 15.0, 1e-9);
        approx(physical_activity_index_from_strain(3, 45.0), 3.75, 1e-9);
        approx(physical_activity_index_from_strain(4, 60.0), 5.0, 1e-9);
    }

    #[test]
    fn compute_reference_person() {
        let r = compute(40.0, "male", 65.0, 5.0, None, false).unwrap();
        approx(r.fitness_age, 40.0, 1e-9);
        approx(r.advance_years, 0.0, 1e-9);
        assert!(r.vo2max.is_none());
        assert!(!r.lower_confidence);
    }

    #[test]
    fn compute_with_waist_fills_vo2max() {
        let r = compute(40.0, "male", 65.0, 5.0, Some(90.0), false).unwrap();
        approx(r.vo2max.unwrap(), 46.275, 1e-3);
    }

    #[test]
    fn compute_non_binary_lower_confidence() {
        let r = compute(40.0, "nonbinary", 60.0, 6.0, None, false).unwrap();
        assert!(r.lower_confidence);
    }

    #[test]
    fn compute_nil_no_rhr() {
        assert!(compute(40.0, "male", 0.0, 7.5, None, false).is_none());
    }

    // Synthetic recovered-value: VO2max must track a moving RHR monotonically down.
    #[test]
    fn vo2max_tracks_declining_fitness() {
        let base = estimate_vo2max(40.0, "male", 90.0, 55.0, 5.0);
        let mut prev = base;
        for rhr in [60.0, 65.0, 70.0, 75.0] {
            let v = estimate_vo2max(40.0, "male", 90.0, rhr, 5.0);
            assert!(v < prev, "rhr {rhr}: {v} !< {prev}");
            prev = v;
        }
    }
}
