//! Whole percentages of a split that add up to exactly 100 — largest-remainder apportionment.
//! Rounding each share on its own sums to 99 or 101 on ordinary splits, so a split is turned into
//! integers once, here, instead of at each place it is drawn.

use std::cmp::Ordering;

/// What a displayed split adds up to.
const WHOLE: i64 = 100;

/// `parts` as whole percentages summing to exactly [`WHOLE`]. A part that is zero, negative or
/// non-finite reads zero and never takes a point. `None` when nothing was measured, so an empty
/// split stays empty rather than reading as a full one.
pub fn whole_percentages(parts: &[f64]) -> Option<Vec<u32>> {
    let measured: Vec<f64> = parts
        .iter()
        .map(|p| if p.is_finite() && *p > 0.0 { *p } else { 0.0 })
        .collect();
    let total: f64 = measured.iter().sum();
    if total <= 0.0 {
        return None;
    }
    let exact: Vec<f64> = measured.iter().map(|p| p / total * WHOLE as f64).collect();
    let mut out: Vec<i64> = exact.iter().map(|e| e.floor() as i64).collect();

    // What the floors left over goes to the largest fractional remainders, lowest index first on a
    // tie, so one split always reads the same way. Only a part with something in it is a candidate.
    let mut order: Vec<usize> = (0..measured.len()).filter(|&i| measured[i] > 0.0).collect();
    order.sort_by(|&a, &b| {
        exact[b]
            .fract()
            .partial_cmp(&exact[a].fract())
            .unwrap_or(Ordering::Equal)
            .then(a.cmp(&b))
    });
    let leftover = (WHOLE - out.iter().sum::<i64>()).max(0) as usize;
    for k in 0..leftover {
        out[order[k % order.len()]] += 1;
    }
    Some(out.into_iter().map(|v| v.max(0) as u32).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each share rounded on its own — the defect this module exists to remove.
    fn naive(parts: &[f64]) -> Vec<i64> {
        let total: f64 = parts.iter().sum();
        parts
            .iter()
            .map(|p| (p / total * 100.0).round() as i64)
            .collect()
    }

    #[test]
    fn nothing_measured_has_no_split() {
        assert_eq!(whole_percentages(&[]), None);
        assert_eq!(whole_percentages(&[0.0, 0.0, 0.0, 0.0]), None);
        assert_eq!(whole_percentages(&[f64::NAN, -3.0, f64::INFINITY]), None);
    }

    #[test]
    fn a_lone_measured_part_owns_the_split() {
        assert_eq!(whole_percentages(&[42.0]), Some(vec![100]));
        assert_eq!(
            whole_percentages(&[0.0, 90.0, 0.0, 0.0]),
            Some(vec![0, 100, 0, 0])
        );
    }

    #[test]
    fn a_three_way_even_split_lands_on_a_hundred_where_rounding_lands_on_ninety_nine() {
        assert_eq!(naive(&[1.0, 1.0, 1.0]).iter().sum::<i64>(), 99);
        assert_eq!(whole_percentages(&[1.0, 1.0, 1.0]), Some(vec![34, 33, 33]));
    }

    #[test]
    fn halves_that_all_round_up_land_on_a_hundred_where_rounding_lands_on_a_hundred_and_two() {
        let parts = [50.0, 12.5, 12.5, 12.5, 12.5];
        assert_eq!(naive(&parts).iter().sum::<i64>(), 102);
        assert_eq!(whole_percentages(&parts), Some(vec![50, 13, 13, 12, 12]));
    }

    #[test]
    fn a_part_with_nothing_in_it_never_takes_a_point() {
        let split = whole_percentages(&[0.0, 1.0, 1.0, 1.0]).unwrap();
        assert_eq!(split[0], 0);
        assert_eq!(split.iter().sum::<u32>(), 100);
    }

    #[test]
    fn no_share_moves_a_whole_point_off_its_exact_value() {
        let parts = [63.0, 17.0, 11.0, 9.5];
        let total: f64 = parts.iter().sum();
        let split = whole_percentages(&parts).unwrap();
        for (i, p) in parts.iter().enumerate() {
            let exact = p / total * 100.0;
            assert!(
                (split[i] as f64 - exact).abs() < 1.0,
                "{i}: {} vs {exact}",
                split[i]
            );
        }
    }

    #[test]
    fn every_four_way_split_of_a_sweep_sums_to_exactly_a_hundred() {
        let mut naive_wrong = 0usize;
        let mut splits = 0usize;
        for a in 0..=20u32 {
            for b in 0..=20u32 {
                for c in 0..=20u32 {
                    for d in 0..=20u32 {
                        let parts = [f64::from(a), f64::from(b), f64::from(c), f64::from(d)];
                        let Some(split) = whole_percentages(&parts) else {
                            assert_eq!(a + b + c + d, 0);
                            continue;
                        };
                        splits += 1;
                        assert_eq!(split.iter().sum::<u32>(), 100, "{parts:?} -> {split:?}");
                        for (i, p) in parts.iter().enumerate() {
                            if *p == 0.0 {
                                assert_eq!(split[i], 0, "{parts:?} -> {split:?}");
                            }
                        }
                        if naive(&parts).iter().sum::<i64>() != 100 {
                            naive_wrong += 1;
                        }
                    }
                }
            }
        }
        // Plain rounding is wrong on a large share of ordinary four-way splits, not on a rare one.
        assert!(naive_wrong * 4 > splits, "{naive_wrong} of {splits}");
    }

    #[test]
    fn a_split_with_a_half_boundary_in_it_still_sums_to_a_hundred() {
        for n in 1..=12usize {
            for k in 0..n {
                let mut parts = vec![1.0; n];
                parts[k] = 0.5;
                let split = whole_percentages(&parts).unwrap();
                assert_eq!(split.iter().sum::<u32>(), 100, "n={n} k={k} -> {split:?}");
            }
        }
    }
}
