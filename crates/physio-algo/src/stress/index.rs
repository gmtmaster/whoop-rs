//! Baevsky Stress Index (SI) — a histogram-based autonomic-balance proxy over an R-R series.
//! `SI = AMo / (2 * Mo * MxDMn)`: Mo is the modal R-R (s), AMo the modal bin's share (%), MxDMn the
//! R-R range (s). A tall, narrow, low-range histogram (rigid, sympathetic) reads high; a broad, flat
//! one reads low. R-R is cleaned first (range band + Malik ectopic). Unbounded, so it carries no
//! band and never reaches [`super::squash`].

use crate::stats::median;

/// Histogram bin width in seconds (Baevsky's 50 ms cardiointervalography grid).
const BIN_WIDTH_SEC: f64 = 0.05;
/// Minimum clean intervals before an SI is computed.
pub const MIN_BEATS: usize = 20;

/// R-R keep-band (ms); intervals outside are dropouts/ectopics.
const RR_MIN_MS: f64 = 300.0;
const RR_MAX_MS: f64 = 2000.0;
/// Malik ectopic rejection: beat dropped if it deviates over 20% from its local median.
const ECTOPIC_THRESHOLD: f64 = 0.20;
/// Half-width (beats) of the centred median window; a 5-beat window at radius 2.
const ECTOPIC_WINDOW_RADIUS: usize = 2;

/// Intermediate histogram terms behind an SI, exposed so a caller can show the "why".
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StressComponents {
    pub mo_sec: f64,
    pub amo_percent: f64,
    pub mxdmn_sec: f64,
    pub si: f64,
}

/// Baevsky Stress Index from a raw R-R series (ms). `None` when too few clean beats survive or the
/// range is degenerate (all-equal beats → MxDMn 0 → an honest `None`, not infinity).
pub fn stress_index_raw(rr_ms: &[f64]) -> Option<f64> {
    components_raw(rr_ms).map(|c| c.si)
}

/// Full SI components from a raw R-R series (ms). Pure and deterministic.
pub fn components_raw(rr_ms: &[f64]) -> Option<StressComponents> {
    let clean = clean_rr(rr_ms);
    if clean.len() < MIN_BEATS {
        return None;
    }
    let sec: Vec<f64> = clean.iter().map(|v| v / 1000.0).collect();
    let min_v = sec.iter().copied().fold(f64::INFINITY, f64::min);
    let max_v = sec.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mxdmn = max_v - min_v;
    if mxdmn <= 0.0 {
        return None;
    }

    let bin_count = ((mxdmn / BIN_WIDTH_SEC).floor() as usize + 1).max(1);
    let mut counts = vec![0usize; bin_count];
    for &v in &sec {
        let mut idx = ((v - min_v) / BIN_WIDTH_SEC).floor() as isize;
        if idx < 0 {
            idx = 0;
        }
        let mut idx = idx as usize;
        if idx >= bin_count {
            idx = bin_count - 1;
        }
        counts[idx] += 1;
    }
    // Modal bin: highest count; ties resolve to the lowest index (deterministic across platforms).
    let mut mode_idx = 0usize;
    let mut mode_count = counts[0];
    for (i, &c) in counts.iter().enumerate().skip(1) {
        if c > mode_count {
            mode_count = c;
            mode_idx = i;
        }
    }
    let mo = min_v + (mode_idx as f64 + 0.5) * BIN_WIDTH_SEC;
    let amo = mode_count as f64 / sec.len() as f64 * 100.0;
    if mo <= 0.0 {
        return None;
    }
    let si = amo / (2.0 * mo * mxdmn);
    Some(StressComponents { mo_sec: mo, amo_percent: amo, mxdmn_sec: mxdmn, si })
}

/// Full clean: range band then Malik ectopic rejection, order preserved.
fn clean_rr(rr: &[f64]) -> Vec<f64> {
    let ranged: Vec<f64> = rr.iter().copied().filter(|&v| (RR_MIN_MS..=RR_MAX_MS).contains(&v)).collect();
    reject_ectopic(&ranged)
}

/// Drop any beat deviating over `ECTOPIC_THRESHOLD` from its local median; short series pass through.
fn reject_ectopic(nn: &[f64]) -> Vec<f64> {
    if nn.len() <= ECTOPIC_WINDOW_RADIUS {
        return nn.to_vec();
    }
    let mut kept = Vec::with_capacity(nn.len());
    for i in 0..nn.len() {
        let lo = i.saturating_sub(ECTOPIC_WINDOW_RADIUS);
        let hi = (i + ECTOPIC_WINDOW_RADIUS).min(nn.len() - 1);
        let mut neighbours: Vec<f64> = Vec::with_capacity(hi - lo);
        for (j, &v) in nn.iter().enumerate().take(hi + 1).skip(lo) {
            if j != i {
                neighbours.push(v);
            }
        }
        if neighbours.len() < 2 {
            kept.push(nn[i]);
            continue;
        }
        let med = median(&neighbours);
        if med <= 0.0 {
            kept.push(nn[i]);
            continue;
        }
        if (nn[i] - med).abs() / med <= ECTOPIC_THRESHOLD {
            kept.push(nn[i]);
        }
    }
    kept
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 22 beats all inside the keep-band and none of them a Malik outlier, so the histogram terms
    /// below are computed over the whole series. Cleaning is exercised by its own fixtures.
    const GOLDEN: [f64; 22] = [
        700.0, 720.0, 740.0, 760.0, 780.0, 800.0, 820.0, 840.0, 860.0, 800.0, 800.0, 800.0, 800.0,
        820.0, 780.0, 800.0, 810.0, 790.0, 800.0, 800.0, 805.0, 795.0,
    ];

    const GOLDEN_SI: f64 = 223.82920110192836;
    /// SI of [`clean_series`] at any length: one 50 ms bin holds every beat.
    const CLEAN_SERIES_SI: f64 = 1552.7950310559033;

    /// `n` beats spanning 780-820 ms: wide enough that MxDMn is non-zero, tight enough that every
    /// beat survives both cleaning rules, so only [`MIN_BEATS`] can refuse it.
    fn clean_series(n: usize) -> Vec<f64> {
        (0..n).map(|i| 780.0 + (i % 5) as f64 * 10.0).collect()
    }

    /// The rows the module must reproduce, each an input and the SI it must yield.
    fn table() -> Vec<(&'static str, Vec<f64>, Option<f64>)> {
        vec![
            ("the golden histogram", GOLDEN.to_vec(), Some(GOLDEN_SI)),
            ("at the beat minimum", clean_series(MIN_BEATS), Some(CLEAN_SERIES_SI)),
            ("one clean beat short", clean_series(MIN_BEATS - 1), None),
            ("every beat equal", vec![800.0; MIN_BEATS + 10], None),
        ]
    }

    fn reproduces(scorer: impl Fn(&[f64]) -> Option<f64>) -> bool {
        table().into_iter().all(|(_, rr, want)| match (scorer(&rr), want) {
            (Some(got), Some(w)) => (got - w).abs() < 1e-9,
            (None, None) => true,
            _ => false,
        })
    }

    #[test]
    fn the_shipped_index_reproduces_the_table_and_three_do_nothing_scorers_do_not() {
        for (name, rr, want) in table() {
            match (stress_index_raw(&rr), want) {
                (Some(got), Some(w)) => assert!((got - w).abs() < 1e-9, "{name}: got {got}, want {w}"),
                (None, None) => {}
                (got, w) => panic!("{name}: got {got:?}, want {w:?}"),
            }
        }
        assert!(reproduces(stress_index_raw));
        // Stand-ins that do no histogram work: one refuses everything, one always answers with the
        // golden SI, one reads the mean interval. Each must miss at least one row.
        type Null = fn(&[f64]) -> Option<f64>;
        let nulls: [(&str, Null); 3] = [
            ("always none", |_| None),
            ("constant golden SI", |_| Some(GOLDEN_SI)),
            ("mean interval", |rr| (!rr.is_empty()).then(|| rr.iter().sum::<f64>() / rr.len() as f64)),
        ];
        for (name, null) in nulls {
            assert!(!reproduces(null), "{name} reproduced every row; the table cannot tell it apart");
        }
    }

    #[test]
    fn the_golden_vector_passes_cleaning_untouched_so_it_measures_the_histogram_alone() {
        assert_eq!(clean_rr(&GOLDEN).len(), GOLDEN.len(), "no beat may be dropped before the histogram");
        let comp = components_raw(&GOLDEN).expect("scorable");
        assert!((comp.mxdmn_sec - 0.16).abs() < 1e-9);
        assert!((comp.mo_sec - 0.825).abs() < 1e-9);
        assert!((comp.amo_percent - 59.09090909090909).abs() < 1e-9);
        assert!((comp.si - GOLDEN_SI).abs() < 1e-9);
        assert!((stress_index_raw(&GOLDEN).unwrap() - GOLDEN_SI).abs() < 1e-9);
    }

    #[test]
    fn tighter_histogram_raises_si() {
        let broad: Vec<f64> = (0..30).map(|it| 700.0 + (it % 11) as f64 * 18.0).collect();
        let rigid: Vec<f64> = (0..30).map(|it| if it % 6 == 0 { 810.0 } else { 800.0 }).collect();
        let si_broad = stress_index_raw(&broad).expect("broad scorable");
        let si_rigid = stress_index_raw(&rigid).expect("rigid scorable");
        assert!(si_rigid > si_broad, "a rigid, tightly-clustered rhythm has a higher SI");
    }

    #[test]
    fn min_beats_is_the_edge_between_none_and_a_score_and_it_counts_clean_beats() {
        assert_eq!(MIN_BEATS, 20);
        assert!(components_raw(&clean_series(MIN_BEATS - 1)).is_none(), "one beat short must refuse");
        let comp = components_raw(&clean_series(MIN_BEATS)).expect("the minimum itself must score");
        assert!((comp.si - CLEAN_SERIES_SI).abs() < 1e-9, "got {}", comp.si);

        // The count is taken after cleaning, so a dropout does not buy a beat toward the minimum.
        let mut short = clean_series(MIN_BEATS - 1);
        short.insert(7, 2500.0);
        assert_eq!(short.len(), MIN_BEATS);
        assert_eq!(clean_rr(&short).len(), MIN_BEATS - 1);
        assert!(stress_index_raw(&short).is_none(), "a dropout must not count toward the minimum");
        let mut exact = clean_series(MIN_BEATS);
        exact.insert(7, 2500.0);
        assert_eq!(clean_rr(&exact).len(), MIN_BEATS);
        assert!(stress_index_raw(&exact).is_some());
    }

    #[test]
    fn an_all_equal_series_refuses_on_range_not_on_count() {
        let rr = vec![800.0; MIN_BEATS + 10];
        assert!(clean_rr(&rr).len() >= MIN_BEATS, "the count gate must already be satisfied");
        assert!(stress_index_raw(&rr).is_none(), "MxDMn 0 is an honest None, never infinity");
    }

    #[test]
    fn the_keep_band_is_inclusive_at_300_and_2000_ms() {
        assert_eq!(RR_MIN_MS, 300.0);
        assert_eq!(RR_MAX_MS, 2000.0);
        let low = [300.0, 305.0, 310.0, 305.0, 300.0, 299.9, 305.0, 310.0];
        let kept = clean_rr(&low);
        assert_eq!(kept.len(), 7, "only the 299.9 ms beat leaves");
        assert!(kept.iter().all(|&v| v != 299.9));
        assert_eq!(kept.iter().filter(|&&v| v == 300.0).count(), 2, "the floor itself is kept");

        let high = [2000.0, 1980.0, 1960.0, 1980.0, 2000.0, 2000.1, 1980.0, 1960.0];
        let kept = clean_rr(&high);
        assert_eq!(kept.len(), 7, "only the 2000.1 ms beat leaves");
        assert!(kept.iter().all(|&v| v != 2000.1));
        assert_eq!(kept.iter().filter(|&&v| v == 2000.0).count(), 2, "the ceiling itself is kept");
    }

    #[test]
    fn malik_drops_a_beat_past_twenty_percent_of_its_local_median_and_keeps_one_at_it() {
        assert_eq!(ECTOPIC_THRESHOLD, 0.20);
        let mut at = [800.0; 10];
        at[5] = 960.0; // exactly +20 % of the 800 ms local median
        assert_eq!(reject_ectopic(&at).len(), 10, "the threshold is inclusive");
        let mut over = [800.0; 10];
        over[5] = 961.0; // +20.125 %
        let kept = reject_ectopic(&over);
        assert_eq!(kept.len(), 9);
        assert!(kept.iter().all(|&v| v == 800.0));
    }

    #[test]
    fn the_ectopic_window_reaches_two_beats_either_side() {
        assert_eq!(ECTOPIC_WINDOW_RADIUS, 2);
        // Two adjacent outliers: at radius 1 each is the other's reference and both survive, at
        // radius 2 the window is still mostly baseline and both go.
        let pair = [800.0, 800.0, 1000.0, 1000.0, 800.0, 800.0, 800.0, 800.0];
        assert_eq!(reject_ectopic(&pair).len(), 6, "radius 1 would keep all eight");
        // Three adjacent outliers own a radius-2 window's median but not a radius-3 one.
        let triple = [800.0, 800.0, 800.0, 1000.0, 1000.0, 1000.0, 800.0, 800.0, 800.0];
        assert_eq!(reject_ectopic(&triple).len(), 9, "radius 3 would drop the middle three");
    }

    #[test]
    fn a_series_no_longer_than_the_window_passes_cleaning_through() {
        assert_eq!(reject_ectopic(&[800.0, 2000.0]).len(), ECTOPIC_WINDOW_RADIUS);
        assert!(reject_ectopic(&[]).is_empty());
    }
}
