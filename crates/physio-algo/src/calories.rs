//! HR-based calorie estimation: Keytel 2005 active energy + revised Harris-Benedict BMR.
//! One per-second model behind both paths. A second whose motion the workout detector confirmed faces
//! the ACTIVITY gate; one with no motion evidence faces the TRUST gate. APPROXIMATE, not calorimetry.

/// Sex-specific BMR + active-EE coefficients.
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
    resting_alpha: 88.362,
    resting_weight: 13.397,
    resting_height: 479.9,
    resting_age: 5.677,
    workout_hr: 0.6309,
    workout_weight: 0.1988,
    workout_age: 0.2017,
    workout_alpha: -55.0969,
};

pub const FEMALE: Coeffs = Coeffs {
    resting_alpha: 447.593,
    resting_weight: 9.247,
    resting_height: 309.8,
    resting_age: 4.33,
    workout_hr: 0.4472,
    workout_weight: -0.1263,
    workout_age: 0.0740,
    workout_alpha: -20.4022,
};

pub const NONBINARY: Coeffs = Coeffs {
    resting_alpha: 267.9775,
    resting_weight: 11.322,
    resting_height: 394.85,
    resting_age: 5.0035,
    workout_hr: 0.53905,
    workout_weight: 0.03625,
    workout_age: 0.13785,
    workout_alpha: -37.74955,
};

/// The ACTIVITY question — "is this second work?" — asked only where motion is already confirmed, so
/// Keytel is applied to the steady-state dynamic exercise it was regressed on. 0.30 HRR is the
/// light/moderate boundary.
pub const ACTIVITY_HRR_FRACTION: f64 = 0.30;
/// The TRUST question — "how high must an HR run before an unconfirmed second is believed to be work?"
/// A different question from [ACTIVITY_HRR_FRACTION], never a second copy of it: its input carries no
/// motion evidence, so caffeine, stress, posture and PPG artifact all reach it. UNSOURCED value.
pub const TRUST_HRR_FRACTION: f64 = 0.50;
/// 60 s/min × 4.184 kJ/kcal.
pub const WORKOUT_DIVISOR: f64 = 251.04;
/// Max inter-sample gap credited inside a bout; the workout detector merges runs on the same gap.
pub const MERGE_GAP_S: f64 = 150.0;
/// Max inter-sample gap credited outside one. Wider than [MERGE_GAP_S] because an ordinary day has
/// legitimate sparse stretches, but bounded so a wear gap is not billed as resting metabolism.
pub const DAY_GAP_CAP_S: f64 = 300.0;

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
    let bmr = c.resting_alpha + c.resting_weight * weight_kg + c.resting_height * height_m
        - c.resting_age * age;
    (bmr.max(0.0)) / 86_400.0
}

/// Active energy rate (kcal/s) from the Keytel 2005 equation.
pub fn active_kcal_per_s(c: &Coeffs, hr: f64, hrmax: f64, weight_kg: f64, age: f64) -> f64 {
    let ee_kj_min = c.workout_hr * hr.min(hrmax)
        + c.workout_weight * weight_kg
        + c.workout_age * age
        + c.workout_alpha;
    (ee_kj_min.max(0.0)) / WORKOUT_DIVISOR
}

pub use crate::hr_sample::HrSample;

/// A day's energy split. `total_kcal` is what the model bills, `resting_kcal` the Harris-Benedict floor
/// over the same credited seconds, and `active_kcal` the excess over lying still — the only one of the
/// three comparable to a phone's "active energy", which excludes BMR.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DayCalories {
    pub total_kcal: f64,
    pub resting_kcal: f64,
    pub active_kcal: f64,
}

/// Bill one time-ordered HR series. Each sample is weighted by the elapsed seconds to the next, capped
/// by its own rule, and billed at Keytel above its own HRR gate, floored at the resting rate. `rule`
/// maps a sample's timestamp to its `(hrr_fraction, gap_cap_s)`. Returns `(kcal, credited seconds)`.
#[allow(clippy::too_many_arguments)]
fn bill<F: Fn(i64) -> (f64, f64)>(
    ordered: &[&HrSample],
    coeffs: &Coeffs,
    weight_kg: f64,
    age: f64,
    hrmax: f64,
    resting_hr: f64,
    resting_rate: f64,
    rule: F,
) -> (f64, f64) {
    let hr_reserve = (hrmax - resting_hr).max(1.0);
    let (mut kcal, mut secs) = (0.0, 0.0);
    for (i, s) in ordered.iter().enumerate() {
        let (fraction, gap_cap) = rule(s.ts);
        let dur = match ordered.get(i + 1) {
            Some(next) => {
                let gap = (next.ts - s.ts) as f64;
                if gap > 0.0 { gap.min(gap_cap) } else { 1.0 }
            }
            None => 1.0,
        };
        let bpm = s.bpm as f64;
        let rate = if bpm < resting_hr + fraction * hr_reserve {
            resting_rate
        } else {
            resting_rate.max(active_kcal_per_s(coeffs, bpm, hrmax, weight_kg, age))
        };
        kcal += rate * dur;
        secs += dur;
    }
    (kcal, secs)
}

/// Time-ordered borrows of `hr_samples`, the order [bill] needs.
fn ordered(hr_samples: &[HrSample]) -> Vec<&HrSample> {
    let mut v: Vec<&HrSample> = hr_samples.iter().collect();
    v.sort_by_key(|s| s.ts);
    v
}

/// Estimate (kcal, kJ) for a workout bout — a span the detector already confirmed motion over, so every
/// second faces the [ACTIVITY_HRR_FRACTION] gate. Samples are weighted by elapsed time (capped at
/// [MERGE_GAP_S]) and floored at the resting rate, so a bout never burns less than lying still.
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
    let resting_rate = resting_kcal_per_s(&coeffs, weight_kg, height_cm, age);
    let (kcal, _) = bill(
        &ordered(hr_samples),
        &coeffs,
        weight_kg,
        age,
        hrmax,
        resting_hr,
        resting_rate,
        |_| (ACTIVITY_HRR_FRACTION, MERGE_GAP_S),
    );
    (kcal, kcal * 4.184)
}

/// Whole-day energy from HR samples, split into its resting and active halves.
///
/// A second inside one of `confirmed_bouts` (the `[start, end]` spans the workout detector qualified)
/// is billed exactly as [estimate_bout_calories] bills it — same gate, same gap cap — because motion is
/// confirmed there. Every other second faces [TRUST_HRR_FRACTION] and [DAY_GAP_CAP_S].
#[allow(clippy::too_many_arguments)]
pub fn estimate_day_calories(
    hr_samples: &[HrSample],
    confirmed_bouts: &[(i64, i64)],
    weight_kg: f64,
    height_cm: f64,
    age: f64,
    sex: &str,
    hrmax: f64,
    resting_hr: f64,
) -> DayCalories {
    if hr_samples.is_empty() {
        return DayCalories {
            total_kcal: 0.0,
            resting_kcal: 0.0,
            active_kcal: 0.0,
        };
    }
    let coeffs = resolve_coeffs(sex);
    let resting_rate = resting_kcal_per_s(&coeffs, weight_kg, height_cm, age);
    let (total_kcal, secs) = bill(
        &ordered(hr_samples),
        &coeffs,
        weight_kg,
        age,
        hrmax,
        resting_hr,
        resting_rate,
        |ts| {
            if confirmed_bouts.iter().any(|&(s, e)| ts >= s && ts <= e) {
                (ACTIVITY_HRR_FRACTION, MERGE_GAP_S)
            } else {
                (TRUST_HRR_FRACTION, DAY_GAP_CAP_S)
            }
        },
    );
    let resting_kcal = resting_rate * secs;
    DayCalories {
        total_kcal,
        resting_kcal,
        active_kcal: total_kcal - resting_kcal,
    }
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

    /// The whole series as one confirmed bout, for the fixtures that stand in for a detected workout.
    fn all_confirmed(hr: &[HrSample]) -> Vec<(i64, i64)> {
        vec![(hr[0].ts, hr[hr.len() - 1].ts)]
    }

    #[test]
    fn day_calories_empty_is_zero() {
        let d = estimate_day_calories(&[], &[], 70.0, 170.0, 30.0, "nonbinary", 190.0, 55.0);
        assert_eq!(
            (d.total_kcal, d.resting_kcal, d.active_kcal),
            (0.0, 0.0, 0.0)
        );
    }

    /// Both gates, for the (185 hrmax, 55 resting) profile every calorie test uses.
    fn gates() -> (f64, f64) {
        let reserve = 185.0 - 55.0;
        (
            55.0 + ACTIVITY_HRR_FRACTION * reserve,
            55.0 + TRUST_HRR_FRACTION * reserve,
        )
    }

    /// Above BOTH gates the two paths agree whether or not a bout was confirmed, because the trust gate
    /// is already cleared. The weaker of the two agreement claims; the band one below is the real gate.
    #[test]
    fn day_calories_matches_bout_at_one_hz_above_both_gates() {
        let profile = (80.0, 180.0, 35.0, "male");
        let bpm = 130;
        let (activity_gate, trust_gate) = gates();
        assert!(
            (bpm as f64) >= activity_gate && (bpm as f64) >= trust_gate,
            "fixture {bpm} bpm must clear BOTH gates (activity {activity_gate}, trust {trust_gate}) or \
             this test proves nothing about agreement"
        );
        let hr = hr_day(bpm, 600);
        let day = estimate_day_calories(
            &hr,
            &[],
            profile.0,
            profile.1,
            profile.2,
            profile.3,
            185.0,
            55.0,
        );
        let (bout, _) =
            estimate_bout_calories(&hr, profile.0, profile.1, profile.2, profile.3, 185.0, 55.0);
        assert!((bout - day.total_kcal).abs() < 1e-9);
    }

    /// THE gate this module lacked. 94..=119 bpm is the band the two gates open between them, and the
    /// workout detector qualifies a bout at 50 % of samples in zone 2+, so up to half a real bout's
    /// seconds sit in it. Inside a confirmed bout the day path must bill exactly what the bout path does.
    #[test]
    fn bout_and_day_agree_across_the_light_band_inside_a_confirmed_bout() {
        let profile = (80.0, 180.0, 35.0, "male");
        let (activity_gate, trust_gate) = gates();
        assert!(
            (activity_gate, trust_gate) == (94.0, 120.0),
            "band edges {activity_gate}..{trust_gate}"
        );

        let mut checked = 0;
        for bpm in activity_gate.ceil() as i32..trust_gate.ceil() as i32 {
            let hr = hr_day(bpm, 3600);
            let day = estimate_day_calories(
                &hr,
                &all_confirmed(&hr),
                profile.0,
                profile.1,
                profile.2,
                profile.3,
                185.0,
                55.0,
            );
            let (bout, _) = estimate_bout_calories(
                &hr, profile.0, profile.1, profile.2, profile.3, 185.0, 55.0,
            );
            assert!(
                (bout - day.total_kcal).abs() < 1e-9,
                "{bpm} bpm inside a confirmed bout: bout {bout} vs day {}",
                day.total_kcal
            );
            // The whole band is Keytel, so none of it may read as a resting-only hour.
            assert!(day.active_kcal > 0.0, "{bpm} bpm bills no active energy");
            checked += 1;
        }
        assert_eq!(checked, 26, "the band is 94..=119 bpm for this profile");
    }

    /// The band survives ONLY for seconds no bout confirmed, which is the leg no measurement settles.
    /// Ratios recorded, not endorsed: a fixed 1.5x assertion passes identically at 1.5x and at 8.1x.
    #[test]
    fn an_unconfirmed_light_band_second_still_bills_bmr_only() {
        let profile = (80.0, 180.0, 35.0, "male");
        let (activity_gate, trust_gate) = gates();

        let mut ratios = Vec::new();
        for bpm in activity_gate.ceil() as i32..trust_gate.ceil() as i32 {
            let hr = hr_day(bpm, 3600);
            let day = estimate_day_calories(
                &hr,
                &[],
                profile.0,
                profile.1,
                profile.2,
                profile.3,
                185.0,
                55.0,
            );
            let (bout, _) = estimate_bout_calories(
                &hr, profile.0, profile.1, profile.2, profile.3, 185.0, 55.0,
            );
            assert!(
                (day.active_kcal).abs() < 1e-6,
                "{bpm} bpm unconfirmed must bill BMR only"
            );
            ratios.push(bout / day.total_kcal);
        }
        assert_eq!(ratios.len(), 26);

        let lo = ratios.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = ratios.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mean = ratios.iter().sum::<f64>() / ratios.len() as f64;
        assert!((lo - 5.123397).abs() < 1e-5, "band-floor ratio {lo}");
        assert!((hi - 8.097457).abs() < 1e-5, "band-ceiling ratio {hi}");
        assert!((mean - 6.610427).abs() < 1e-5, "band-mean ratio {mean}");
    }

    /// Outside the band the two paths are identical with no bout confirmed; a regression that widened
    /// the band shows up here rather than in a fixture that happens to sit clear of it.
    #[test]
    fn bout_and_day_agree_everywhere_outside_the_band() {
        let profile = (80.0, 180.0, 35.0, "male");
        let (activity_gate, trust_gate) = gates();
        for bpm in (40..=200).filter(|b| (*b as f64) < activity_gate || (*b as f64) >= trust_gate) {
            let hr = hr_day(bpm, 600);
            let day = estimate_day_calories(
                &hr,
                &[],
                profile.0,
                profile.1,
                profile.2,
                profile.3,
                185.0,
                55.0,
            );
            let (bout, _) = estimate_bout_calories(
                &hr, profile.0, profile.1, profile.2, profile.3, 185.0, 55.0,
            );
            assert!(
                (bout - day.total_kcal).abs() < 1e-9,
                "{bpm} bpm outside the band: bout {bout} day {}",
                day.total_kcal
            );
        }
    }

    /// A confirmed span bills only the seconds INSIDE it at the activity gate. The seconds outside keep
    /// the trust gate, so a bout cannot re-rate the rest of the day around it.
    #[test]
    fn a_confirmed_span_moves_only_the_seconds_inside_it() {
        let profile = (80.0, 180.0, 35.0, "male");
        // Two hours at 110 bpm (inside the band); only the second hour is a confirmed bout.
        let hr = hr_day(110, 7200);
        let none = estimate_day_calories(
            &hr,
            &[],
            profile.0,
            profile.1,
            profile.2,
            profile.3,
            185.0,
            55.0,
        );
        let half = estimate_day_calories(
            &hr,
            &[(3600, 7199)],
            profile.0,
            profile.1,
            profile.2,
            profile.3,
            185.0,
            55.0,
        );
        let all = estimate_day_calories(
            &hr,
            &all_confirmed(&hr),
            profile.0,
            profile.1,
            profile.2,
            profile.3,
            185.0,
            55.0,
        );

        assert!(
            (none.active_kcal).abs() < 1e-6,
            "nothing confirmed, nothing active"
        );
        assert!(
            (half.active_kcal - all.active_kcal / 2.0).abs() < 1e-6,
            "half {} vs all {}",
            half.active_kcal,
            all.active_kcal
        );
        // The resting floor is over the same credited seconds either way, so only the active half moved.
        assert!((none.resting_kcal - all.resting_kcal).abs() < 1e-6);
    }

    /// The split the "Active Energy" label needs: a day that never clears a gate is ALL resting, so a
    /// caller merging `total_kcal` with a phone's active-only series is comparing two different things.
    #[test]
    fn a_sedentary_day_is_all_resting_and_no_active_energy() {
        let profile = (80.0, 180.0, 35.0, "male");
        let d = estimate_day_calories(
            &hr_day(55, 86_400),
            &[],
            profile.0,
            profile.1,
            profile.2,
            profile.3,
            185.0,
            55.0,
        );
        assert!(
            (d.total_kcal - 1825.25).abs() < 1.0,
            "total {}",
            d.total_kcal
        );
        assert!(
            (d.resting_kcal - d.total_kcal).abs() < 1e-6,
            "a still day is BMR end to end"
        );
        assert!(d.active_kcal.abs() < 1e-6, "active {}", d.active_kcal);
    }

    #[test]
    fn sparse_hr_tracks_elapsed_time_not_sample_count() {
        let profile = (80.0, 180.0, 35.0, "male");
        let dense: Vec<HrSample> = (0..600).map(|i| HrSample { ts: i, bpm: 130 }).collect();
        let sparse: Vec<HrSample> = (0..600)
            .step_by(10)
            .map(|i| HrSample { ts: i, bpm: 130 })
            .collect();
        let (dense_kcal, _) = estimate_bout_calories(
            &dense, profile.0, profile.1, profile.2, profile.3, 185.0, 55.0,
        );
        let (sparse_kcal, _) = estimate_bout_calories(
            &sparse, profile.0, profile.1, profile.2, profile.3, 185.0, 55.0,
        );
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
        let (gap_kcal, _) = estimate_bout_calories(
            &gapped, profile.0, profile.1, profile.2, profile.3, 185.0, 55.0,
        );
        let (equiv_kcal, _) = estimate_bout_calories(
            &capped_equiv,
            profile.0,
            profile.1,
            profile.2,
            profile.3,
            185.0,
            55.0,
        );
        assert!((gap_kcal - equiv_kcal).abs() < equiv_kcal * 0.001);
    }

    #[test]
    fn day_path_caps_a_wear_gap_instead_of_billing_it() {
        let profile = (80.0, 180.0, 35.0, "male");
        // One hour between two samples: credited at DAY_GAP_CAP_S, not the full 3600 s.
        let gapped = vec![
            HrSample { ts: 0, bpm: 130 },
            HrSample { ts: 3600, bpm: 130 },
        ];
        let gap_day = estimate_day_calories(
            &gapped,
            &[],
            profile.0,
            profile.1,
            profile.2,
            profile.3,
            185.0,
            55.0,
        )
        .total_kcal;
        // The identical rate held for exactly the cap + the tail second.
        let capped: Vec<HrSample> = (0..=DAY_GAP_CAP_S as i64)
            .map(|i| HrSample { ts: i, bpm: 130 })
            .collect();
        let capped_day = estimate_day_calories(
            &capped,
            &[],
            profile.0,
            profile.1,
            profile.2,
            profile.3,
            185.0,
            55.0,
        )
        .total_kcal;
        assert!(
            (gap_day - capped_day).abs() < capped_day * 0.01,
            "gap {gap_day} vs capped {capped_day}"
        );
        // Far below billing the whole hour.
        let full_hour = estimate_day_calories(
            &hr_day(130, 3601),
            &[],
            profile.0,
            profile.1,
            profile.2,
            profile.3,
            185.0,
            55.0,
        )
        .total_kcal;
        assert!(
            gap_day < full_hour * 0.2,
            "a wear gap must not read as an hour of effort"
        );
    }

    #[test]
    fn day_path_credits_elapsed_time_on_a_sparse_stream() {
        // A day sampled every 60 s must total the same as the dense 1 Hz day it stands in for.
        let profile = (80.0, 180.0, 35.0, "male");
        let dense: Vec<HrSample> = (0..3600).map(|i| HrSample { ts: i, bpm: 70 }).collect();
        let sparse: Vec<HrSample> = (0..3600)
            .step_by(60)
            .map(|i| HrSample { ts: i, bpm: 70 })
            .collect();
        let d = estimate_day_calories(
            &dense,
            &[],
            profile.0,
            profile.1,
            profile.2,
            profile.3,
            185.0,
            55.0,
        )
        .total_kcal;
        let s = estimate_day_calories(
            &sparse,
            &[],
            profile.0,
            profile.1,
            profile.2,
            profile.3,
            185.0,
            55.0,
        )
        .total_kcal;
        assert!(
            (s - d).abs() < d * 0.05,
            "sparse {s} should track dense {d}"
        );
    }

    #[test]
    fn resting_day_is_lower_than_active_day() {
        let profile = default_profile();
        let resting = estimate_day_calories(
            &hr_day(60, 3600),
            &[],
            profile.0,
            profile.1,
            profile.2,
            profile.3,
            185.0,
            55.0,
        )
        .total_kcal;
        let active = estimate_day_calories(
            &hr_day(150, 3600),
            &[],
            profile.0,
            profile.1,
            profile.2,
            profile.3,
            185.0,
            55.0,
        )
        .total_kcal;
        assert!(resting > 0.0);
        assert!(active > resting);
    }

    #[test]
    fn sedentary_full_day_approximates_bmr() {
        let profile = (80.0, 180.0, 35.0, "male");
        let sedentary = hr_day(55, 86_400);
        let total = estimate_day_calories(
            &sedentary,
            &[],
            profile.0,
            profile.1,
            profile.2,
            profile.3,
            185.0,
            55.0,
        )
        .total_kcal;
        assert!((total - 1825.25).abs() < 1.0);
    }

    /// One block on a shared 24 h timeline; [hr_day] restarts at ts 0 on every call, so concatenating
    /// its output stacks blocks on the same seconds instead of laying them end to end.
    fn hr_block(start: i64, bpm: i32, n: usize) -> Vec<HrSample> {
        (0..n)
            .map(|i| HrSample {
                ts: start + i as i64,
                bpm,
            })
            .collect()
    }

    /// One sample per second of a whole day, no two on the same timestamp. A day fixture that fails
    /// this is billed one flat second per sample and silently covers less than 24 h.
    fn assert_is_one_full_day(day: &[HrSample]) {
        assert_eq!(day.len(), 86_400, "expected one sample per second of a day");
        let mut ts: Vec<i64> = day.iter().map(|s| s.ts).collect();
        ts.sort_unstable();
        ts.dedup();
        assert_eq!(
            ts.len(),
            86_400,
            "samples share timestamps: the fixture is not a 24 h timeline"
        );
        assert_eq!(
            day[day.len() - 1].ts - day[0].ts,
            86_399,
            "fixture does not span 24 h"
        );
    }

    /// Every sample sits UNDER the day gate, so Keytel is never reached and the total is the BMR the
    /// sedentary day already pins. This bounds the day path's floor, never its active energy.
    #[test]
    fn light_activity_day_stays_under_the_day_gate_and_totals_bmr() {
        let profile = (80.0, 180.0, 35.0, "male");
        let mut light_day = hr_block(0, 55, 8 * 3600);
        light_day.extend(hr_block(8 * 3600, 70, 8 * 3600));
        light_day.extend(hr_block(16 * 3600, 100, 8 * 3600));
        assert_is_one_full_day(&light_day);
        let (_, day_gate) = gates();
        assert!(
            light_day.iter().all(|s| (s.bpm as f64) < day_gate),
            "every sample must stay under the {day_gate} bpm day gate or this is not a BMR-only day"
        );
        let total = estimate_day_calories(
            &light_day,
            &[],
            profile.0,
            profile.1,
            profile.2,
            profile.3,
            185.0,
            55.0,
        )
        .total_kcal;
        assert!((total - 1825.25).abs() < 1.0);
        assert!(total < 4768.0 - 2000.0);
    }

    /// 8 h @55, 8 h @70, 7 h @100, then 30 min each at 125 and 140 bpm on one 24 h timeline.
    fn active_day_fixture() -> Vec<HrSample> {
        let mut day = hr_block(0, 55, 8 * 3600);
        day.extend(hr_block(8 * 3600, 70, 8 * 3600));
        day.extend(hr_block(16 * 3600, 100, 7 * 3600));
        day.extend(hr_block(23 * 3600, 125, 1800));
        day.extend(hr_block(23 * 3600 + 1800, 140, 1800));
        day
    }

    /// Pins the MAGNITUDE of active energy on the day path: one hour clears the day gate, so 3600 s
    /// are billed by Keytel and the other 82 800 s at the resting rate.
    #[test]
    fn active_day_bills_keytel_above_the_day_gate() {
        let profile = (80.0, 180.0, 35.0, "male");
        let active_day = active_day_fixture();
        assert_is_one_full_day(&active_day);
        let (_, day_gate) = gates();
        let over = active_day
            .iter()
            .filter(|s| (s.bpm as f64) >= day_gate)
            .count();
        assert_eq!(
            over, 3600,
            "exactly one hour must clear the {day_gate} bpm day gate"
        );

        let day = estimate_day_calories(
            &active_day,
            &[],
            profile.0,
            profile.1,
            profile.2,
            profile.3,
            185.0,
            55.0,
        );
        let total = day.total_kcal;
        assert!((total - 2487.16).abs() < 1.0, "active day total {total}");
        assert!(total < 4768.0 - 2000.0);

        // Everything above a BMR-only day is Keytel's contribution, and it is what the wearer reads
        // as active energy.
        let bmr_day = resting_kcal_per_s(&MALE, profile.0, profile.1, profile.2) * 86_400.0;
        assert!(
            (total - bmr_day - 661.91).abs() < 1.0,
            "active energy {}",
            total - bmr_day
        );
        // The same number the split reports, so the label "active energy" has one source.
        assert!(
            (day.active_kcal - 661.91).abs() < 1.0,
            "split active {}",
            day.active_kcal
        );
        assert!(
            (day.resting_kcal - bmr_day).abs() < 1e-6,
            "split resting {}",
            day.resting_kcal
        );

        // Rebuilt from the two public rates: fails if the over-gate hour stops routing to Keytel.
        let rebuilt = resting_kcal_per_s(&MALE, profile.0, profile.1, profile.2) * 82_800.0
            + active_kcal_per_s(&MALE, 125.0, 185.0, profile.0, profile.2) * 1_800.0
            + active_kcal_per_s(&MALE, 140.0, 185.0, profile.0, profile.2) * 1_800.0;
        assert!(
            (total - rebuilt).abs() < 1e-6,
            "total {total} vs rebuilt {rebuilt}"
        );
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
