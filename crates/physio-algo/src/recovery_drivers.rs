//! The per-driver breakdown behind one Charge score: each present term's point swing and how it reads
//! against the personal baseline. Built from the same terms, weights and logistic as
//! [`crate::recovery::recovery`], so a row can never describe a term the score did not use.
//!
//! The swing is an exact Shapley share, not a leave-one-out marginal: the score is a logistic, so once
//! one driver alone saturates it every other leave-one-out swing collapses to 0 however far off
//! baseline that driver sits.

use crate::recovery::{
    RECOVERY_INDEX_SCALE_BPM_PER_HR, RecoveryInput, SKIN_TEMP_DEV_SCALE, SLEEP_PERF_CENTER,
    SLEEP_PERF_SCALE, W_ACTIVITY_BALANCE, W_HRV, W_RECOVERY_INDEX, W_RESP, W_RHR, W_SKIN_TEMP,
    W_SLEEP, score_of, z_score,
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

/// One driver row: the signal, its share of the Charge swing in whole points, and its direction.
/// `delta_points` is NaN when ANY driver's value is, so a caller decides rather than displays it.
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
    if x - floor >= 0.5 { floor + 1.0 } else { floor }
}

/// Exact Shapley shares of the Charge swing, one per term, in the order `terms` holds them: each
/// driver's marginal swing averaged over every order the drivers could be added in. The shares sum to
/// `score(all terms) − score(no terms)`. [`ROW_ORDER`] caps n at 7, so all 2^n coalitions are walked.
fn shapley_points(terms: &[Term], total_weight: f64) -> Vec<f64> {
    let n = terms.len();
    // v(S) for every coalition, indexed by a bitmask over `terms`. A driver outside S sits at its own
    // baseline (z = 0) with its weight still in the denominator.
    let value: Vec<f64> = (0..(1usize << n))
        .map(|mask| {
            let z: f64 = terms
                .iter()
                .enumerate()
                .filter(|(i, _)| mask >> i & 1 == 1)
                .map(|(_, t)| t.z * t.w)
                .sum();
            score_of(z / total_weight)
        })
        .collect();
    // |S|! (n − |S| − 1)! / n! — the fraction of orderings that add this driver straight after S.
    let mut factorial = [1.0f64; ROW_ORDER.len() + 1];
    for i in 1..=n {
        factorial[i] = factorial[i - 1] * i as f64;
    }
    (0..n)
        .map(|i| {
            let bit = 1usize << i;
            (0..(1usize << n))
                .filter(|mask| mask & bit == 0)
                .map(|mask| {
                    let size = mask.count_ones() as usize;
                    factorial[size] * factorial[n - size - 1] / factorial[n]
                        * (value[mask | bit] - value[mask])
                })
                .sum()
        })
        .collect()
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
/// whose input is missing yields no row rather than a fabricated zero-contribution one. A term that
/// IS present and sits exactly on its baseline reads 0 points, which is the only honest zero here.
pub fn driver_rows(input: &RecoveryInput) -> Vec<DriverRow> {
    if !input.hrv_baseline_usable {
        return Vec::new();
    }
    let mut terms: Vec<Term> = Vec::new();
    if let Some(b) = input.hrv_baseline {
        terms.push(Term {
            kind: DriverKind::Hrv,
            z: z_score(input.hrv, b.mean, b.spread),
            w: W_HRV,
        });
    }
    if let Some(b) = input.rhr_baseline {
        terms.push(Term {
            kind: DriverKind::RestingHr,
            z: z_score(b.mean, input.rhr, b.spread),
            w: W_RHR,
        });
    }
    if let (Some(resp), Some(b)) = (input.resp, input.resp_baseline) {
        terms.push(Term {
            kind: DriverKind::Respiratory,
            z: z_score(b.mean, resp, b.spread),
            w: W_RESP,
        });
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
    let points = shapley_points(&terms, total_weight);

    let mut rows: Vec<DriverRow> = Vec::new();
    for kind in ROW_ORDER {
        let Some(idx) = terms.iter().position(|t| t.kind == kind) else {
            continue;
        };
        let verdict = match kind {
            DriverKind::SkinTemp => skin_temp_verdict(input.skin_temp_dev.unwrap_or(0.0)),
            _ => direction(terms[idx].z),
        };
        rows.push(DriverRow {
            kind,
            delta_points: round_half_up(points[idx]),
            verdict,
        });
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
        DriverBaseline {
            mean,
            spread: sigma / 1.253,
        }
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
                (DriverKind::Hrv, 26.0),
                (DriverKind::RestingHr, 6.0),
                (DriverKind::ActivityBalance, -3.0),
                (DriverKind::RecoveryIndex, 2.0),
                (DriverKind::Sleep, 1.0),
                (DriverKind::Respiratory, 1.0),
                (DriverKind::SkinTemp, 0.0),
            ]
        );
    }

    /// The night a leave-one-out marginal got wrong: HRV alone saturates the logistic, so it scored
    /// +20 and the other four drivers each read exactly 0 — including a resting heart rate 16 bpm
    /// below its own baseline. Every present driver here is off baseline, so none may read 0.
    #[test]
    fn a_saturated_score_still_attributes_every_off_baseline_driver() {
        let input = RecoveryInput {
            hrv: 150.0,
            rhr: 42.0,
            resp: Some(19.0),
            hrv_baseline: Some(baseline(70.0, 8.0)),
            rhr_baseline: Some(baseline(58.0, 4.0)),
            resp_baseline: Some(baseline(14.6, 1.2)),
            sleep_perf: Some(0.95),
            skin_temp_dev: Some(1.4),
            hrv_baseline_usable: true,
            ..Default::default()
        };
        let rows = driver_rows(&input);
        let got: Vec<(DriverKind, f64)> = rows.iter().map(|r| (r.kind, r.delta_points)).collect();
        assert_eq!(
            got,
            vec![
                (DriverKind::Hrv, 31.0),
                (DriverKind::RestingHr, 13.0),
                (DriverKind::Respiratory, -3.0),
                (DriverKind::Sleep, 2.0),
                (DriverKind::SkinTemp, -1.0),
            ]
        );
        // The score itself is pinned, which is what broke the old attribution.
        let score = crate::recovery::recovery(&input).expect("a usable baseline scores");
        assert!(
            score > 99.9,
            "score {score} is not saturated, so this is the wrong fixture"
        );
        // Not one present driver is off its baseline and silent.
        for r in &rows {
            assert_ne!(
                r.verdict,
                DriverVerdict::Neutral,
                "{:?} read neutral",
                r.kind
            );
            assert_ne!(
                r.delta_points, 0.0,
                "{:?} moved a saturated score by nothing",
                r.kind
            );
        }
    }

    /// The shares decompose the whole swing, so they sum to the score's distance from an all-baseline
    /// night, to within the half point per row that rounding costs.
    #[test]
    fn the_rows_sum_to_the_distance_from_an_all_baseline_night() {
        for input in [
            full_night(),
            RecoveryInput {
                sleep_perf: None,
                ..full_night()
            },
        ] {
            let rows = driver_rows(&input);
            let swing = crate::recovery::recovery(&input).expect("scores") - score_of(0.0);
            let summed: f64 = rows.iter().map(|r| r.delta_points).sum();
            assert!(
                (summed - swing).abs() <= rows.len() as f64 / 2.0,
                "rows summed to {summed}, swing is {swing}"
            );
        }
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
            rows.iter()
                .find(|r| r.kind == DriverKind::Hrv)
                .unwrap()
                .verdict
        };
        assert_eq!(verdict(two_term(|i| i.hrv = 49.0)), DriverVerdict::Limiting);
        assert_eq!(verdict(two_term(|i| i.hrv = 50.0)), DriverVerdict::Neutral);
        assert_eq!(
            verdict(two_term(|i| i.hrv = 51.0)),
            DriverVerdict::Supporting
        );
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
            assert!(
                wrong >= 4,
                "null verdict '{name}' disagreed on only {wrong} cases"
            );
        }
        // Every single-sided driver yields only the first three verdicts; the side is skin temp's.
        for r in driver_rows(&full_night())
            .iter()
            .filter(|r| r.kind != DriverKind::SkinTemp)
        {
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
        assert_eq!(
            seen(SKIN_TEMP_TYPICAL_BAND_C),
            (DriverVerdict::Neutral, -1.0)
        );
        assert_eq!(seen(-0.22), (DriverVerdict::Neutral, -1.0));
    }

    #[test]
    fn the_logistic_saturates_the_swing_at_the_range_ends() {
        let hrv_delta = |hrv: f64| {
            two_term(|i| i.hrv = hrv)
                .iter()
                .find(|r| r.kind == DriverKind::Hrv)
                .unwrap()
                .delta_points
        };
        assert_eq!(hrv_delta(5000.0), 42.0);
        assert_eq!(hrv_delta(-5000.0), -58.0);
        // A zero-spread baseline saturates through the z-score floor rather than diverging.
        let zero_spread = driver_rows(&RecoveryInput {
            hrv: 60.0,
            rhr: 55.0,
            hrv_baseline: Some(DriverBaseline {
                mean: 50.0,
                spread: 0.0,
            }),
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
        assert!(
            driver_rows(&RecoveryInput {
                hrv: 60.0,
                rhr: 50.0,
                ..Default::default()
            })
            .is_empty()
        );
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
