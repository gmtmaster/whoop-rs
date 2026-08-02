//! The per-driver breakdown behind one Charge score: each present term's marginal point swing and how
//! it reads against the personal baseline. Built from the same terms, weights and logistic as
//! [`crate::recovery::recovery`], so a row can never describe a term the score did not use.

use crate::recovery::{
    score_of, z_score, RecoveryInput, RECOVERY_INDEX_SCALE_BPM_PER_HR, SKIN_TEMP_DEV_SCALE,
    SLEEP_PERF_CENTER, SLEEP_PERF_SCALE, W_ACTIVITY_BALANCE, W_HRV, W_RECOVERY_INDEX, W_RESP, W_RHR,
    W_SKIN_TEMP, W_SLEEP,
};

/// Half-width (°C) of the skin-temp band that reads as typical, inclusive at both edges. Outside it
/// the symmetric penalty is worth naming to the wearer; inside it the drift is ordinary night noise.
pub const SKIN_TEMP_TYPICAL_BAND_C: f64 = 0.3;

/// Which signal a row describes. The caller owns its label, unit and value text.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DriverKind {
    Hrv,
    RestingHr,
    Sleep,
    Respiratory,
    SkinTemp,
    RecoveryIndex,
    ActivityBalance,
}

/// How a driver reads against its baseline. `LimitingHigh` / `LimitingLow` carry the side for the
/// symmetric skin-temp term; every single-sided driver yields only the first three.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DriverVerdict {
    Supporting,
    Neutral,
    Limiting,
    LimitingHigh,
    LimitingLow,
}

/// One driver row: the signal, its marginal swing in whole Charge points, and its direction.
/// `delta_points` is NaN when the driver's own value is, so a caller decides rather than displays it.
#[derive(Clone, Copy, Debug)]
pub struct DriverRow {
    pub kind: DriverKind,
    pub delta_points: f64,
    pub verdict: DriverVerdict,
}

/// Emission order before the biggest-mover sort; a stable sort keeps it as the tie-break.
const ROW_ORDER: [DriverKind; 7] = [
    DriverKind::Hrv,
    DriverKind::RestingHr,
    DriverKind::Sleep,
    DriverKind::Respiratory,
    DriverKind::SkinTemp,
    DriverKind::RecoveryIndex,
    DriverKind::ActivityBalance,
];

/// One (z, weight) term with the row it feeds.
struct Term {
    kind: DriverKind,
    z: f64,
    w: f64,
}

/// Nearest whole value, ties toward +∞ — the whole points a row shows. NaN in, NaN out.
fn round_half_up(x: f64) -> f64 {
    let floor = x.floor();
    if x - floor >= 0.5 {
        floor + 1.0
    } else {
        floor
    }
}

/// A z already oriented so higher is better: positive supports recovery, negative limits it.
fn direction(z: f64) -> DriverVerdict {
    if z > 0.0 {
        DriverVerdict::Supporting
    } else if z < 0.0 {
        DriverVerdict::Limiting
    } else {
        DriverVerdict::Neutral
    }
}

/// Skin temp is a symmetric penalty: inside the typical band it reads neutral, beyond it limits
/// recovery either way, and the side is kept so the caller can say warmer or cooler.
fn skin_temp_verdict(dev: f64) -> DriverVerdict {
    if dev.abs() <= SKIN_TEMP_TYPICAL_BAND_C {
        DriverVerdict::Neutral
    } else if dev > 0.0 {
        DriverVerdict::LimitingHigh
    } else {
        DriverVerdict::LimitingLow
    }
}

/// The ordered driver rows behind [`crate::recovery::recovery`] for the same input: biggest mover
/// first, ties in emission order. Empty exactly where the score itself returns `None`, and a term
/// whose input is missing yields no row rather than a fabricated zero-contribution one.
pub fn driver_rows(input: &RecoveryInput) -> Vec<DriverRow> {
    if !input.hrv_baseline_usable {
        return Vec::new();
    }
    let mut terms: Vec<Term> = Vec::new();
    if let Some(b) = input.hrv_baseline {
        terms.push(Term { kind: DriverKind::Hrv, z: z_score(input.hrv, b.mean, b.spread), w: W_HRV });
    }
    if let Some(b) = input.rhr_baseline {
        terms.push(Term { kind: DriverKind::RestingHr, z: z_score(b.mean, input.rhr, b.spread), w: W_RHR });
    }
    if let (Some(resp), Some(b)) = (input.resp, input.resp_baseline) {
        terms.push(Term { kind: DriverKind::Respiratory, z: z_score(b.mean, resp, b.spread), w: W_RESP });
    }
    if let Some(sp) = input.sleep_perf {
        terms.push(Term {
            kind: DriverKind::Sleep,
            z: (sp - SLEEP_PERF_CENTER) / SLEEP_PERF_SCALE,
            w: W_SLEEP,
        });
    }
    if let Some(dev) = input.skin_temp_dev {
        terms.push(Term {
            kind: DriverKind::SkinTemp,
            z: -dev.abs() / SKIN_TEMP_DEV_SCALE,
            w: W_SKIN_TEMP,
        });
    }
    if let Some(slope) = input.recovery_index_slope {
        terms.push(Term {
            kind: DriverKind::RecoveryIndex,
            z: -slope / RECOVERY_INDEX_SCALE_BPM_PER_HR,
            w: W_RECOVERY_INDEX,
        });
    }
    if let (Some(effort), Some(b)) = (input.prior_day_effort, input.effort_baseline) {
        terms.push(Term {
            kind: DriverKind::ActivityBalance,
            z: z_score(b.mean, effort, b.spread),
            w: W_ACTIVITY_BALANCE,
        });
    }

    let total_weight: f64 = terms.iter().map(|t| t.w).sum();
    if terms.is_empty() || total_weight <= 0.0 {
        return Vec::new();
    }
    let actual = score_of(terms.iter().map(|t| t.z * t.w).sum::<f64>() / total_weight);

    // Marginal swing of one term: the score minus the score with that ONE term neutralised to z = 0
    // (the signal sitting at its own baseline), its weight still occupying the denominator.
    let delta = |idx: usize| -> f64 {
        let neutral = terms
            .iter()
            .enumerate()
            .map(|(i, t)| if i == idx { 0.0 } else { t.z * t.w })
            .sum::<f64>()
            / total_weight;
        round_half_up(actual - score_of(neutral))
    };

    let mut rows: Vec<DriverRow> = Vec::new();
    for kind in ROW_ORDER {
        let Some(idx) = terms.iter().position(|t| t.kind == kind) else { continue };
        let verdict = match kind {
            DriverKind::SkinTemp => skin_temp_verdict(input.skin_temp_dev.unwrap_or(0.0)),
            _ => direction(terms[idx].z),
        };
        rows.push(DriverRow { kind, delta_points: delta(idx), verdict });
    }
    rows.sort_by(|a, b| {
        b.delta_points
            .abs()
            .partial_cmp(&a.delta_points.abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    rows
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recovery::DriverBaseline;

    // DriverBaseline from a Gaussian sigma (spread is internal abs-dev units): spread = sigma / 1.253.
    fn baseline(mean: f64, sigma: f64) -> DriverBaseline {
        DriverBaseline { mean, spread: sigma / 1.253 }
    }

    fn full_night() -> RecoveryInput {
        RecoveryInput {
            hrv: 62.0,
            rhr: 51.0,
            resp: Some(15.0),
            hrv_baseline: Some(baseline(50.0, 6.0)),
            rhr_baseline: Some(baseline(55.0, 3.0)),
            resp_baseline: Some(baseline(16.0, 2.0)),
            sleep_perf: Some(0.9),
            skin_temp_dev: Some(0.4),
            hrv_baseline_usable: true,
            recovery_index_slope: Some(-3.0),
            effort_baseline: Some(baseline(40.0, 15.0)),
            prior_day_effort: Some(75.0),
        }
    }

    fn two_term(mutate: impl FnOnce(&mut RecoveryInput)) -> Vec<DriverRow> {
        let mut input = RecoveryInput {
            hrv: 50.0,
            rhr: 55.0,
            hrv_baseline: Some(baseline(50.0, 6.0)),
            rhr_baseline: Some(baseline(55.0, 3.0)),
            ..Default::default()
        };
        mutate(&mut input);
        driver_rows(&input)
    }

    #[test]
    fn every_term_present_sorts_biggest_mover_first() {
        let rows = driver_rows(&full_night());
        let got: Vec<(DriverKind, f64)> = rows.iter().map(|r| (r.kind, r.delta_points)).collect();
        assert_eq!(
            got,
            vec![
                (DriverKind::Hrv, 23.0),
                (DriverKind::RestingHr, 4.0),
                (DriverKind::Sleep, 1.0),
                (DriverKind::RecoveryIndex, 1.0),
                (DriverKind::ActivityBalance, -1.0),
                (DriverKind::Respiratory, 0.0),
                (DriverKind::SkinTemp, 0.0),
            ]
        );
    }

    #[test]
    fn a_missing_input_drops_its_row_and_reweights_the_rest() {
        let mut input = full_night();
        input.rhr_baseline = None;
        let rows = driver_rows(&input);
        assert!(!rows.iter().any(|r| r.kind == DriverKind::RestingHr));
        // The HRV term now carries the weight the resting-HR term released.
        let hrv = rows.iter().find(|r| r.kind == DriverKind::Hrv).unwrap();
        assert_eq!(hrv.delta_points, 32.0);
    }

    #[test]
    fn direction_flips_either_side_of_the_baseline_and_is_neutral_on_it() {
        let verdict = |rows: Vec<DriverRow>| {
            rows.iter().find(|r| r.kind == DriverKind::Hrv).unwrap().verdict
        };
        assert_eq!(verdict(two_term(|i| i.hrv = 49.0)), DriverVerdict::Limiting);
        assert_eq!(verdict(two_term(|i| i.hrv = 50.0)), DriverVerdict::Neutral);
        assert_eq!(verdict(two_term(|i| i.hrv = 51.0)), DriverVerdict::Supporting);
    }

    // The skin-temp row for one nightly deviation, against a two-driver night.
    fn skin_row(dev: f64) -> DriverRow {
        *two_term(|i| i.skin_temp_dev = Some(dev))
            .iter()
            .find(|r| r.kind == DriverKind::SkinTemp)
            .unwrap()
    }

    /// The band as the wearer meets it. The 0.3 literals sit beside the constant so widening or
    /// narrowing the band fails here rather than moving the expectation with itself.
    const SKIN_CASES: [(f64, DriverVerdict); 13] = [
        (-1.0, DriverVerdict::LimitingLow),
        (-0.31, DriverVerdict::LimitingLow),
        (-0.30000001, DriverVerdict::LimitingLow),
        (-0.3, DriverVerdict::Neutral),
        (-SKIN_TEMP_TYPICAL_BAND_C, DriverVerdict::Neutral),
        (-0.1, DriverVerdict::Neutral),
        (0.0, DriverVerdict::Neutral),
        (0.1, DriverVerdict::Neutral),
        (SKIN_TEMP_TYPICAL_BAND_C, DriverVerdict::Neutral),
        (0.3, DriverVerdict::Neutral),
        (0.30000001, DriverVerdict::LimitingHigh),
        (0.31, DriverVerdict::LimitingHigh),
        (1.0, DriverVerdict::LimitingHigh),
    ];

    #[test]
    fn skin_temp_is_the_only_driver_whose_verdict_carries_a_side() {
        for (dev, want) in SKIN_CASES {
            assert_eq!(skin_row(dev).verdict, want, "dev {dev}");
        }
        // Both sides are reached and the band is not the whole range, so a one-word verdict fails.
        for (name, null) in [
            ("always Neutral", DriverVerdict::Neutral),
            ("always LimitingHigh", DriverVerdict::LimitingHigh),
            ("always Supporting", DriverVerdict::Supporting),
        ] {
            let wrong = SKIN_CASES.iter().filter(|(_, want)| null != *want).count();
            assert!(wrong >= 4, "null verdict '{name}' disagreed on only {wrong} cases");
        }
        // Every single-sided driver yields only the first three verdicts; the side is skin temp's.
        for r in driver_rows(&full_night()).iter().filter(|r| r.kind != DriverKind::SkinTemp) {
            assert!(
                matches!(
                    r.verdict,
                    DriverVerdict::Supporting | DriverVerdict::Neutral | DriverVerdict::Limiting
                ),
                "{:?} carried a side: {:?}",
                r.kind,
                r.verdict
            );
        }
    }

    /// The typical band and the points the row shows do not share an edge: the penalty is continuous
    /// and has no deadband, so a row reads "typical" at +0.22 °C while already costing a whole point.
    #[test]
    fn the_typical_band_edge_and_the_first_lost_point_do_not_coincide() {
        let seen = |dev: f64| {
            let r = skin_row(dev);
            (r.verdict, r.delta_points)
        };
        assert_eq!(seen(0.20), (DriverVerdict::Neutral, 0.0));
        assert_eq!(seen(0.22), (DriverVerdict::Neutral, -1.0));
        assert_eq!(seen(SKIN_TEMP_TYPICAL_BAND_C), (DriverVerdict::Neutral, -1.0));
        assert_eq!(seen(-0.22), (DriverVerdict::Neutral, -1.0));
    }

    #[test]
    fn the_logistic_saturates_the_swing_at_the_range_ends() {
        let hrv_delta = |hrv: f64| {
            two_term(|i| i.hrv = hrv).iter().find(|r| r.kind == DriverKind::Hrv).unwrap().delta_points
        };
        assert_eq!(hrv_delta(5000.0), 42.0);
        assert_eq!(hrv_delta(-5000.0), -58.0);
        // A zero-spread baseline saturates through the z-score floor rather than diverging.
        let zero_spread = driver_rows(&RecoveryInput {
            hrv: 60.0,
            rhr: 55.0,
            hrv_baseline: Some(DriverBaseline { mean: 50.0, spread: 0.0 }),
            rhr_baseline: Some(baseline(55.0, 3.0)),
            ..Default::default()
        });
        assert_eq!(zero_spread[0].delta_points, 42.0);
    }

    #[test]
    fn an_unusable_hrv_baseline_yields_no_rows() {
        let mut input = full_night();
        input.hrv_baseline_usable = false;
        assert!(driver_rows(&input).is_empty());
        // No driver at all is the other empty: nothing to break down.
        assert!(driver_rows(&RecoveryInput { hrv: 60.0, rhr: 50.0, ..Default::default() }).is_empty());
    }

    #[test]
    fn a_non_finite_driver_value_yields_non_finite_points_rather_than_a_zero() {
        let rows = two_term(|i| i.skin_temp_dev = Some(f64::NAN));
        assert_eq!(rows.len(), 3);
        assert!(rows.iter().all(|r| r.delta_points.is_nan()));
    }

    #[test]
    fn round_half_up_matches_nearest_with_ties_toward_positive_infinity() {
        assert_eq!(round_half_up(22.5), 23.0);
        assert_eq!(round_half_up(-0.5), 0.0);
        assert_eq!(round_half_up(-1.5), -1.0);
        assert_eq!(round_half_up(0.49999999999999994), 0.0);
        assert!(round_half_up(f64::NAN).is_nan());
    }
}
