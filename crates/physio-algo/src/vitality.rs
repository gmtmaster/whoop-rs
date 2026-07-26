//! Vitality (0–100) and Body Age (years) from wearable drivers. Each driver maps to its published
//! all-cause-mortality log-hazard versus a population reference; the sum is overlap-corrected for
//! correlated inputs and divided by the Gompertz slope to become an age offset. Body Age =
//! chronological age + that offset. A wellness comparison, never a clinical biological age.

/// Human mortality-rate doubling time (years) — the Gompertz slope this model converts hazard into age
/// with. Sourced: Richmond & Roehner, "Predictive implications of Gompertz's law" (arXiv:1509.07271),
/// which fits a doubling time of about 10 years for humans above age 35 across national and historical
/// series. The literature spans roughly 7–10 and the Strehler-Mildvan correlation says the Gompertz
/// intercept and slope are not independent, so treat this as a SENSITIVITY parameter, not a constant of
/// nature: it scales the size of a deviation and nothing else. A person at the population reference on
/// every driver sums to zero hazard and reads their own age whatever value this takes.
pub const MORTALITY_DOUBLING_YEARS: f64 = 10.0;

/// Log-hazard per year of ageing = ln(2) / the doubling time.
const LN_HAZARD_PER_YEAR: f64 = std::f64::consts::LN_2 / MORTALITY_DOUBLING_YEARS;

/// The drivers are correlated (fitness, sleep and HRV move together), so the naive sum over-counts.
/// A flat shrink is the conservative correction; it is a modelling choice, not a published figure.
const OVERLAP_SHRINK: f64 = 0.75;

/// Body Age is clamped to this band — the model is fitted on adults and says nothing outside it.
pub const MIN_BODY_AGE: f64 = 20.0;
pub const MAX_BODY_AGE: f64 = 90.0;

/// Vitality points per year younger than chronological, centred on 50.
const VITALITY_PER_YEAR: f64 = 2.5;

/// Drivers needed before a score is offered; below this the read is too thin to mean anything.
pub const MIN_FACTORS: usize = 3;

/// The ± band the readout is presented with (years).
pub const BAND_YEARS: f64 = 5.0;

// ── Per-driver log-hazards, each versus its population reference ──────────────────────────────────
//
// Each coefficient is carried as ln(HR) against a population reference. Provenance restored here; the
// original citations lived in a Swift twin that no longer exists.
//
// VERIFIED against Europe PMC, 2026-07-26. Every coefficient below names a real study and a real number.
//
// THE SHAPE OF THE EVIDENCE, stated once so it is not restated per driver:
//   - Only VO2max is published as a per-unit slope. Steps, HRV and sleep regularity are published as
//     QUARTILE or PERCENTILE CONTRASTS, so their coefficients here are our linearisations of a
//     categorical result. Sleep duration is categorical too and its per-hour figure is unsourced.
//   - Every coefficient sits AT OR BELOW the published effect size, except resting HR, which falls
//     between two meta-analyses that disagree with each other. The model therefore UNDER-states rather
//     than over-states, which is the correct direction for a wellness readout.
//   - This is a wellness comparison on consumer hardware, not a clinical instrument. The numbers are
//     honest about their own precision; the readout should be read the same way.

/// Resting heart rate, per 10 bpm above 65. VERIFIED against two meta-analyses that DISAGREE: Zhang et
/// al. (CMAJ 2016, PMID 26598376) give RR 1.09 (95% CI 1.07-1.12) per 10 bpm, Aune et al. (NMCD 2017,
/// PMID 28552551) give 1.17 (1.14-1.19). This coefficient is ln(1.105) ≈ 0.100 — inside Zhang's interval
/// and below Aune's, i.e. the conservative end of the published range, not a single quoted figure.
const HR_RESTING_PER_10BPM: f64 = 0.100;
const RESTING_HR_REFERENCE: f64 = 65.0;

/// Cardiorespiratory fitness, per 1 MET (3.5 mL/kg/min) BELOW the age/sex expectation. VERIFIED: Kodama
/// et al. (JAMA 2009, PMID 19454641, doi 10.1001/jama.2009.681) pool RR 0.87 (95% CI 0.84-0.90) per
/// 1-MET higher maximal aerobic capacity. That inverts to ln(1/0.87) ≈ 0.139; this coefficient is 0.130,
/// marginally conservative.
const HR_VO2MAX_PER_MET: f64 = 0.130;
const MET_ML_KG_MIN: f64 = 3.5;
const VO2MAX_MET_CLAMP: f64 = 4.0;

/// Sleep duration, per hour of deviation outside a 7.0-8.0 h band. The U-shape is well established, but
/// the modern cohorts report CATEGORIES, not a per-hour slope: Gu et al. (Nat Sci Sleep 2024, PMID
/// 39011490), Liang et al. (Eur J Prev Cardiol 2023, PMID 36990109) and the accelerometer-measured Zhou
/// et al. (Sleep 2024, PMID 38995667) all contrast bands such as <7 h against 7-8 h. This per-hour
/// coefficient is therefore OURS, chosen to sit inside those contrasts rather than read off any of them.
const HR_SLEEP_PER_HOUR: f64 = 0.110;
const SLEEP_OPTIMUM_HOURS: f64 = 7.5;
const SLEEP_DEADBAND_HOURS: f64 = 0.5;
const SLEEP_DEVIATION_CLAMP: f64 = 3.0;

/// Sleep regularity, per point of Sleep Regularity Index below the population median. CALIBRATED, not
/// chosen: Cribb et al. (eLife 2023, PMID 37995126, UK Biobank n=88 975) report HR 1.53 (95% CI
/// 1.41-1.66) at SRI 41 and 0.90 (0.81-1.00) at SRI 75, both relative to the median. Fitting a line
/// through those two points in log-hazard gives a slope of 0.0156 per SRI point and puts the median
/// (HR = 1, ln = 0) at SRI 68.3. Corroborated by Kalkanis et al. (Sleep Med Rev 2025, PMID 41259946):
/// 20-88 % higher all-cause mortality for the least regular sleepers.
const HR_SLEEP_REGULARITY_PER_SRI_POINT: f64 = 0.015609;
const SRI_MEDIAN_REFERENCE: f64 = 68.25;
/// Cribb's evidence spans roughly the 5th to 95th percentile; do not extrapolate the line past it.
const SRI_LN_HAZARD_CLAMP: f64 = 0.50;

/// The DURATION-regularity fallback, used only when the real index cannot be computed. `1 - CV` of
/// nightly hours is blind to a shifting bedtime at constant duration, which is exactly the pattern
/// Cribb's hazard sits in, so it is the weaker read and is scaled conservatively.
const HR_SLEEP_DURATION_REGULARITY: f64 = 0.450;
const SLEEP_REGULARITY_REFERENCE: f64 = 0.75;

/// HRV, per whole fractional shortfall against the age norm. Jarczok et al. (Neurosci Biobehav Rev 2022,
/// PMID 36243195) pool the lowest quartile of 5-min RMSSD against the other quartiles at HR 1.56 (95% CI
/// 1.32-1.85) across diverse populations. That contrast is ln(1.56) ≈ 0.445; this coefficient is 0.160 at
/// a whole fractional shortfall, well under it. A quartile contrast again, not a per-unit slope, and the
/// age-norm anchors below are ours rather than the paper's.
const HR_HRV_PER_FRACTION: f64 = 0.160;

/// Daily steps, per 1000 below 7000. CAUTION — this is a LINEARISATION, not a quoted rate. Paluch et al.
/// (Lancet Public Health 2022, PMID 35247352) report quartiles, not a per-1000 slope: HR 0.47 (0.39-0.57)
/// for quartile 4 (median 10 901 steps/day) versus quartile 1 (median 3 553). Linearised across that
/// 7 348-step gap the implied coefficient is ln(1/0.47)/7.348 ≈ 0.103 per 1000. This coefficient is 0.064,
/// roughly 60 % of that, deliberately conservative because the same paper finds the benefit PLATEAUS at
/// 6000-8000 steps/day — a straight line across the whole range would overstate the slope at the top end.
/// The clamp below keeps the credited range near the plateau.
const HR_STEPS_PER_1000: f64 = 0.064;
const STEPS_REFERENCE: f64 = 7000.0;
const STEPS_CLAMP_HI: f64 = 11_000.0;
const STEPS_CLAMP: f64 = 4.0;

/// Nocturnal RMSSD age norms (age, ms) — the ~50th percentile by decade. A person at their age norm
/// contributes zero hazard. Interpolated linearly between anchors, flat outside them.
const RMSSD_NORM_ANCHORS: [(f64, f64); 7] = [
    (20.0, 47.0), (30.0, 40.0), (40.0, 33.0), (50.0, 29.0), (60.0, 25.0), (70.0, 22.0), (80.0, 20.0),
];

/// The wearable drivers for one reading. Every field is optional; an absent driver simply drops.
#[derive(Clone, Copy, Debug, Default)]
pub struct VitalityInput {
    pub chrono_age: f64,
    pub resting_hr: Option<f64>,
    pub vo2max: Option<f64>,
    pub expected_vo2max: Option<f64>,
    pub sleep_hours: Option<f64>,
    /// The real Sleep Regularity Index (-100..100) from [`crate::sleep_regularity`]. Preferred.
    pub sleep_regularity_index: Option<f64>,
    /// Duration-regularity fallback (1 - CV of nightly hours, 0..1). Used only when the index is absent.
    pub sleep_consistency: Option<f64>,
    pub rmssd: Option<f64>,
    pub rmssd_norm: Option<f64>,
    pub steps: Option<f64>,
}

/// One driver's signed contribution: positive ages you, negative is protective.
#[derive(Clone, Debug, PartialEq)]
pub struct Contribution {
    pub key: String,
    pub label: String,
    pub ln_hazard: f64,
}

/// A Vitality reading. `advance_years` is Body Age minus chronological: POSITIVE = older than your
/// years, matching the rhythm-age and fitness-age conventions.
#[derive(Clone, Debug, PartialEq)]
pub struct Vitality {
    pub vitality: f64,
    pub body_age: f64,
    pub chrono_age: f64,
    pub advance_years: f64,
    pub band_years: f64,
    pub contributions: Vec<Contribution>,
    pub factors_used: u32,
}

/// The nocturnal RMSSD norm (ms) for an age, interpolated between the decade anchors.
pub fn rmssd_norm(for_age: f64) -> f64 {
    let first = RMSSD_NORM_ANCHORS[0];
    let last = RMSSD_NORM_ANCHORS[RMSSD_NORM_ANCHORS.len() - 1];
    if for_age <= first.0 {
        return first.1;
    }
    if for_age >= last.0 {
        return last.1;
    }
    for w in RMSSD_NORM_ANCHORS.windows(2) {
        let ((a0, v0), (a1, v1)) = (w[0], w[1]);
        if for_age <= a1 {
            return v0 + (v1 - v0) * (for_age - a0) / (a1 - a0);
        }
    }
    last.1
}

/// Sleep regularity in [0, 1] from nightly durations (hours): 1 − coefficient of variation.
/// `None` below three nights, which is too few to read a rhythm from.
pub fn sleep_consistency(nightly_hours: &[f64]) -> Option<f64> {
    let xs: Vec<f64> = nightly_hours.iter().copied().filter(|&h| h > 0.0).collect();
    if xs.len() < 3 {
        return None;
    }
    let mean = xs.iter().sum::<f64>() / xs.len() as f64;
    if mean <= 0.0 {
        return None;
    }
    let var = xs.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / xs.len() as f64;
    Some((1.0 - var.sqrt() / mean).clamp(0.0, 1.0))
}

fn push(out: &mut Vec<Contribution>, key: &str, label: &str, ln_hazard: f64) {
    out.push(Contribution { key: key.to_string(), label: label.to_string(), ln_hazard });
}

/// Each present driver's signed log-hazard against its population reference.
pub fn contributions(input: &VitalityInput) -> Vec<Contribution> {
    let mut out = Vec::new();
    if let Some(rhr) = input.resting_hr {
        push(&mut out, "rhr", "Resting heart rate",
            ((rhr - RESTING_HR_REFERENCE) / 10.0) * HR_RESTING_PER_10BPM);
    }
    if let (Some(vo2), Some(expected)) = (input.vo2max, input.expected_vo2max) {
        if expected > 0.0 {
            let mets = ((expected - vo2) / MET_ML_KG_MIN).clamp(-VO2MAX_MET_CLAMP, VO2MAX_MET_CLAMP);
            push(&mut out, "vo2max", "Cardio fitness", mets * HR_VO2MAX_PER_MET);
        }
    }
    if let Some(hours) = input.sleep_hours {
        let deviation = ((hours - SLEEP_OPTIMUM_HOURS).abs() - SLEEP_DEADBAND_HOURS).max(0.0);
        push(&mut out, "sleep", "Sleep duration",
            deviation.clamp(0.0, SLEEP_DEVIATION_CLAMP) * HR_SLEEP_PER_HOUR);
    }
    // The real index wins when it is available; the duration proxy is the fallback, never both.
    if let Some(sri) = input.sleep_regularity_index {
        let ln_hazard = ((SRI_MEDIAN_REFERENCE - sri.clamp(-100.0, 100.0))
            * HR_SLEEP_REGULARITY_PER_SRI_POINT)
            .clamp(-SRI_LN_HAZARD_CLAMP, SRI_LN_HAZARD_CLAMP);
        push(&mut out, "consistency", "Sleep regularity", ln_hazard);
    } else if let Some(consistency) = input.sleep_consistency {
        push(&mut out, "consistency", "Sleep regularity",
            (SLEEP_REGULARITY_REFERENCE - consistency.clamp(0.0, 1.0)) * HR_SLEEP_DURATION_REGULARITY);
    }
    if let (Some(rmssd), Some(norm)) = (input.rmssd, input.rmssd_norm) {
        if norm > 0.0 {
            let shortfall = ((norm - rmssd) / norm).clamp(-1.0, 1.0);
            push(&mut out, "hrv", "Heart-rate variability", shortfall * HR_HRV_PER_FRACTION);
        }
    }
    if let Some(steps) = input.steps {
        let deficit = (STEPS_REFERENCE - steps.clamp(0.0, STEPS_CLAMP_HI)) / 1000.0;
        push(&mut out, "steps", "Daily steps", deficit.clamp(-STEPS_CLAMP, STEPS_CLAMP) * HR_STEPS_PER_1000);
    }
    out
}

/// Full Vitality + Body Age. `None` below [`MIN_FACTORS`] present drivers or without a chronological age.
pub fn compute(input: &VitalityInput) -> Option<Vitality> {
    if input.chrono_age <= 0.0 {
        return None;
    }
    let contribs = contributions(input);
    if contribs.len() < MIN_FACTORS {
        return None;
    }
    let sum_ln = contribs.iter().map(|c| c.ln_hazard).sum::<f64>() * OVERLAP_SHRINK;
    let delta_age = sum_ln / LN_HAZARD_PER_YEAR;
    let body_age = (input.chrono_age + delta_age).clamp(MIN_BODY_AGE, MAX_BODY_AGE);
    // Vitality reads the "years younger" direction; the exposed advance is the opposite sign so it
    // matches the rhythm-age and fitness-age conventions.
    let years_younger = input.chrono_age - body_age;
    let vitality = (50.0 + years_younger * VITALITY_PER_YEAR).clamp(0.0, 100.0);
    let factors_used = contribs.len() as u32;
    Some(Vitality {
        vitality,
        body_age,
        chrono_age: input.chrono_age,
        advance_years: -years_younger,
        band_years: BAND_YEARS,
        contributions: contribs,
        factors_used,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference() -> VitalityInput {
        VitalityInput {
            chrono_age: 40.0,
            resting_hr: Some(65.0),
            vo2max: Some(45.0),
            expected_vo2max: Some(45.0),
            sleep_hours: Some(7.5),
            sleep_regularity_index: Some(SRI_MEDIAN_REFERENCE),
            sleep_consistency: None,
            rmssd: Some(45.0),
            rmssd_norm: Some(45.0),
            steps: Some(7000.0),
        }
    }

    #[test]
    fn a_person_at_every_reference_reads_their_own_age() {
        // The zero point is invariant to the doubling time: all drivers at reference sum to zero
        // hazard, so Body Age is chronological age whatever MORTALITY_DOUBLING_YEARS is set to.
        let r = compute(&reference()).unwrap();
        assert!((r.body_age - 40.0).abs() < 1e-9, "body age {}", r.body_age);
        assert!((r.vitality - 50.0).abs() < 1e-9);
        assert!(r.advance_years.abs() < 1e-9);
        assert_eq!(r.factors_used, 6);
    }

    #[test]
    fn healthier_drivers_read_younger_and_worse_read_older() {
        let mut healthy = reference();
        healthy.resting_hr = Some(52.0);
        healthy.vo2max = Some(55.5);
        healthy.sleep_consistency = Some(0.9);
        healthy.rmssd = Some(54.0);
        healthy.steps = Some(11_000.0);
        let h = compute(&healthy).unwrap();
        assert!(h.body_age < 40.0 && h.advance_years < 0.0, "healthy {h:?}");

        let mut worn = reference();
        worn.resting_hr = Some(80.0);
        worn.vo2max = Some(34.5);
        worn.sleep_hours = Some(5.5);
        worn.sleep_consistency = Some(0.5);
        worn.rmssd = Some(31.5);
        worn.steps = Some(3000.0);
        let w = compute(&worn).unwrap();
        assert!(w.body_age > 40.0 && w.advance_years > 0.0, "worn {w:?}");
    }

    #[test]
    fn the_doubling_time_scales_the_deviation_not_the_zero_point() {
        // Δage = Σln(HR) × MRDT / ln2, so the offset is linear in the doubling time. Pinning the
        // relationship keeps the sensitivity of the readout explicit.
        let mut worn = reference();
        worn.resting_hr = Some(80.0);
        let r = compute(&worn).unwrap();
        let sum_ln = contributions(&worn).iter().map(|c| c.ln_hazard).sum::<f64>() * OVERLAP_SHRINK;
        let expected = sum_ln * MORTALITY_DOUBLING_YEARS / std::f64::consts::LN_2;
        assert!((r.advance_years - expected).abs() < 1e-9, "{} vs {expected}", r.advance_years);
    }

    #[test]
    fn too_few_drivers_is_none() {
        let thin = VitalityInput { chrono_age: 40.0, resting_hr: Some(60.0), ..Default::default() };
        assert!(compute(&thin).is_none());
        assert!(compute(&VitalityInput { chrono_age: 0.0, ..reference() }).is_none());
    }

    #[test]
    fn the_sri_driver_reproduces_cribbs_two_published_points() {
        // Cribb: HR 1.53 at SRI 41 and 0.90 at SRI 75, relative to the median. The fitted line must land
        // on both, which is what makes this coefficient calibrated rather than chosen.
        let at = |sri: f64| {
            let i = VitalityInput { sleep_regularity_index: Some(sri), ..Default::default() };
            contributions(&i).first().unwrap().ln_hazard
        };
        assert!((at(41.0) - 1.53f64.ln()).abs() < 0.01, "SRI 41 -> {}", at(41.0));
        assert!((at(75.0) - 0.90f64.ln()).abs() < 0.01, "SRI 75 -> {}", at(75.0));
        assert!(at(SRI_MEDIAN_REFERENCE).abs() < 1e-9, "the median must be zero hazard");
    }

    #[test]
    fn the_real_index_supersedes_the_duration_proxy() {
        let both = VitalityInput {
            sleep_regularity_index: Some(41.0), sleep_consistency: Some(1.0), ..Default::default()
        };
        let c = contributions(&both);
        assert_eq!(c.len(), 1, "one regularity driver, never two");
        assert!(c[0].ln_hazard > 0.0, "the index says irregular, so the proxy must not override it");
    }

    #[test]
    fn rmssd_norm_interpolates_and_flattens() {
        assert_eq!(rmssd_norm(20.0), 47.0);
        assert_eq!(rmssd_norm(10.0), 47.0); // flat below the first anchor
        assert_eq!(rmssd_norm(90.0), 20.0); // flat above the last
        assert!((rmssd_norm(35.0) - 36.5).abs() < 1e-9); // midway 40 -> 33
    }

    #[test]
    fn sleep_consistency_is_one_minus_cv() {
        assert_eq!(sleep_consistency(&[7.0, 7.0]), None); // under three nights
        assert_eq!(sleep_consistency(&[8.0, 8.0, 8.0]), Some(1.0)); // no spread at all
        let mixed = sleep_consistency(&[6.0, 8.0, 10.0]).unwrap();
        assert!(mixed > 0.0 && mixed < 1.0, "got {mixed}");
    }

    #[test]
    fn body_age_is_clamped_to_the_modelled_band() {
        let mut extreme = reference();
        extreme.chrono_age = 25.0;
        extreme.resting_hr = Some(40.0);
        extreme.vo2max = Some(70.0);
        extreme.sleep_consistency = Some(1.0);
        extreme.rmssd = Some(90.0);
        extreme.steps = Some(11_000.0);
        assert!(compute(&extreme).unwrap().body_age >= MIN_BODY_AGE);
    }
}
