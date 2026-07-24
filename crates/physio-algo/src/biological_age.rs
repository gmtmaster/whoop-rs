//! CosinorAge: a circadian-rhythm biological age (years) from a rest-activity cosinor fit via a Gompertz
//! proportional-hazards transform with published sex-specific coefficients. v1 is RELATIVE — the activity
//! scale is not yet calibrated to the model's mg-ENMO training units. A wellness estimate, never medical.

/// A sex coefficient set for the linear index. `shape` from the published model is intentionally NOT
/// carried: it is unused in the transform, and `M_D` below hardcodes the generic shape for every sex.
#[derive(Clone, Copy)]
struct Coeffs {
    rate: f64,
    mesor: f64,
    amp1: f64,
    phi1: f64,
    age: f64,
}

const GENERIC: Coeffs = Coeffs {
    rate: -13.36715309,
    mesor: -0.03204933,
    amp1: -0.01971357,
    phi1: -0.01664718,
    age: 0.10033692,
};
const FEMALE: Coeffs = Coeffs {
    rate: -13.28530410,
    mesor: -0.02569062,
    amp1: -0.02170987,
    phi1: -0.13191562,
    age: 0.08840283,
};
const MALE: Coeffs = Coeffs {
    rate: -13.016951633,
    mesor: -0.023988922,
    amp1: -0.030620390,
    phi1: 0.008960155,
    age: 0.101726103,
};

/// Scale from the on-chip activity magnitude (g) to the model's mg-ENMO training units: 1000 is the exact
/// g→mg unit conversion; the AFE/placement transfer factor (WHOOP wrist vs the training accelerometer) is
/// folded in as an UNVALIDATED 1.0, so the age stays a relative index until calibrated to a moving reference.
pub const ACTIVITY_TO_MG_ENMO_SCALE: f64 = 1000.0;

/// Global transform constants. `M_D` is the generic shape, hardcoded for every sex (matching the model).
const M_N: f64 = -1.405276;
const M_D: f64 = 0.01462774;
const BA_N: f64 = -0.01447851;
const BA_D: f64 = 0.112165;
const BA_I: f64 = 133.5989;

/// Sex selector for the coefficient set; unknown → the generic (pooled) set.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sex {
    Female,
    Male,
    Unknown,
}

/// A CosinorAge result: the biological age (years) and its advance vs chronological age (+ = older).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BiologicalAge {
    pub cosinor_age_years: f64,
    pub advance_years: f64,
}

fn coeffs(sex: Sex) -> Coeffs {
    match sex {
        Sex::Female => FEMALE,
        Sex::Male => MALE,
        Sex::Unknown => GENERIC,
    }
}

/// CosinorAge from a cosinor fit's MESOR + amplitude + acrophase (radians, CosinorPy convention) and the
/// chronological age (years). `None` on a non-finite / degenerate transform. RELATIVE index until the
/// activity scale is calibrated to mg-ENMO.
pub fn cosinor_age(
    mesor: f64,
    amplitude: f64,
    acrophase_radians: f64,
    chronological_age: f64,
    sex: Sex,
) -> Option<BiologicalAge> {
    let c = coeffs(sex);
    let xb = mesor * c.mesor
        + amplitude * c.amp1
        + acrophase_radians * c.phi1
        + chronological_age * c.age
        + c.rate;
    // survival = 1 − m_val, where m_val = 1 − exp(M_N·exp(xb)/M_D) is the 5-yr mortality risk.
    let survival = (M_N * xb.exp() / M_D).exp();
    let outer = BA_N * survival.ln();
    if outer <= 0.0 || !outer.is_finite() {
        return None;
    }
    let cosinor_age_years = outer.ln() / BA_D + BA_I;
    if !cosinor_age_years.is_finite() {
        return None;
    }
    Some(BiologicalAge {
        cosinor_age_years,
        advance_years: cosinor_age_years - chronological_age,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pinned against the reference implementation for (M, amp, φ, age) = (30, 25, −1.5, 40).
    #[test]
    fn matches_reference_outputs() {
        let g = cosinor_age(30.0, 25.0, -1.5, 40.0, Sex::Unknown).unwrap();
        assert!((g.cosinor_age_years - 40.4054333204).abs() < 1e-4, "generic {}", g.cosinor_age_years);
        let f = cosinor_age(30.0, 25.0, -1.5, 40.0, Sex::Female).unwrap();
        assert!((f.cosinor_age_years - 39.6765236784).abs() < 1e-4, "female {}", f.cosinor_age_years);
        let m = cosinor_age(30.0, 25.0, -1.5, 40.0, Sex::Male).unwrap();
        assert!((m.cosinor_age_years - 43.4054735692).abs() < 1e-4, "male {}", m.cosinor_age_years);
    }

    #[test]
    fn advance_is_relative_to_chronological() {
        let r = cosinor_age(30.0, 25.0, -1.5, 40.0, Sex::Unknown).unwrap();
        assert!((r.advance_years - (r.cosinor_age_years - 40.0)).abs() < 1e-9);
    }
}
