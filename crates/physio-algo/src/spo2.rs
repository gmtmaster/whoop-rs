//! SpO2 (%) from dual-wavelength PPG via ratio-of-ratios, from the 4.0 v24 paired red/IR samples. 5.0/MG
//! has no SpO2 path: its v26 optical buffer is a single AC-coupled waveform (one wavelength), not a red/IR
//! pair, so `from_paired`/`from_history` only produce a value on 4.0 — and there a pulsatility gate
//! withholds it, because the 1 Hz pair aliases the cardiac band away. Wellness estimate, never clinical.

use whoop_protocol::HistoryRecord;

use crate::stats::{amplitude, mean, median};

const WINDOW_SECONDS: usize = 30;
const MIN_SAMPLES_PER_WINDOW: usize = 10;
/// Windows that must be pulsatile before a night is scored. Ratio-of-ratios reads the AC/DC ratio of a
/// cardiac waveform (0.83-3.0 Hz at 50-180 bpm), but the 4.0 pair arrives at 1 Hz, so that band sits
/// entirely above the 0.5 Hz Nyquist limit and is aliased away. Below this fraction the night is `None`.
const MIN_PULSATILE_FRACTION: f64 = 0.5;
const CURVE_A: f64 = 110.0;
const CURVE_B: f64 = 25.0;
const CLAMP_LOW: f64 = 70.0;
const CLAMP_HIGH: f64 = 100.0;

// Rolling multi-night readout.
const ROLL_WINDOW_NIGHTS: usize = 30;
const RECENT_NIGHTS: usize = 7;
const ANCHOR: f64 = 96.5;
const ROLLING_CLAMP_LOW: f64 = 88.0;
const ROLLING_CLAMP_HIGH: f64 = 100.0;
// WHOOP shows blood oxygen after one recovery, so we report from the first night.
const MIN_NIGHTS: usize = crate::calibration::BLOOD_OXYGEN.unlock as usize;

/// A smoothed multi-night readout: `pct` once calibrated, else `calibrating_nights` carries the count.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RollingReading {
    pub pct: Option<f64>,
    pub calibrating_nights: Option<usize>,
}

pub struct Spo2;

impl Spo2 {
    /// SpO2 for a night from parallel per-sample red/IR ADC (the 4.0 v24 pair per second). `None` if no
    /// window survives.
    pub fn from_paired(red: &[f64], ir: &[f64]) -> Option<f64> {
        let n = red.len().min(ir.len());
        let mut per_window = Vec::new();
        let mut eligible = 0usize;
        let mut start = 0;
        while start < n {
            let end = (start + WINDOW_SECONDS).min(n);
            if end - start >= MIN_SAMPLES_PER_WINDOW {
                eligible += 1;
                if let Some(s) = window_spo2(&red[start..end], &ir[start..end]) {
                    per_window.push(s);
                }
            }
            start = end;
        }
        if eligible == 0 || (per_window.len() as f64) < MIN_PULSATILE_FRACTION * eligible as f64 {
            return None;
        }
        finish(per_window)
    }

    /// SpO2 from 4.0 history records — extracts the v24 red/IR pair each carries, then `from_paired`.
    pub fn from_history(records: &[HistoryRecord]) -> Option<f64> {
        let (mut red, mut ir) = (Vec::new(), Vec::new());
        for r in records {
            if let Some((rd, i)) = r.spo2 {
                red.push(rd as f64);
                ir.push(i as f64);
            }
        }
        Self::from_paired(&red, &ir)
    }

    /// Integer-truncated nightly means of the 4.0 raw red/IR PPG ADC over the detected in-bed `spans`
    /// (`[start, end]` inclusive, unix seconds). A sample counts when its `ts` lies inside any span; the
    /// means truncate toward zero (sum/kept). `None` when either input is empty or no sample landed in-span.
    /// Raw ADC only, never a calibrated percent.
    pub fn nightly_raw_means(spans: &[(i64, i64)], samples: &[(i64, i32, i32)]) -> Option<(i32, i32)> {
        if spans.is_empty() || samples.is_empty() {
            return None;
        }
        let (mut red_sum, mut ir_sum): (i64, i64) = (0, 0);
        let mut kept: i64 = 0;
        for &(ts, red, ir) in samples {
            if !spans.iter().any(|&(start, end)| ts >= start && ts <= end) {
                continue;
            }
            red_sum += red as i64;
            ir_sum += ir as i64;
            kept += 1;
        }
        if kept == 0 {
            return None;
        }
        Some(((red_sum / kept) as i32, (ir_sum / kept) as i32))
    }

    /// Smoothed multi-night readout: soft-anchor the 30-night median to a plausible baseline (removing an
    /// uncalibrated DC offset while preserving spread), then report the 7-night median at that offset.
    /// `pct` is `None` while calibrating (< `MIN_NIGHTS`). `recent_nightly` is oldest → newest.
    pub fn rolling_reading(recent_nightly: &[f64]) -> RollingReading {
        let window = if recent_nightly.len() > ROLL_WINDOW_NIGHTS {
            &recent_nightly[recent_nightly.len() - ROLL_WINDOW_NIGHTS..]
        } else {
            recent_nightly
        };
        if window.len() < MIN_NIGHTS {
            return RollingReading { pct: None, calibrating_nights: Some(window.len()) };
        }
        let offset = ANCHOR - median(window);
        let recent_count = RECENT_NIGHTS.min(window.len());
        let recent = median(&window[window.len() - recent_count..]);
        let clamped = (recent + offset).clamp(ROLLING_CLAMP_LOW, ROLLING_CLAMP_HIGH);
        RollingReading { pct: Some((clamped + 0.5).floor()), calibrating_nights: None }
    }
}

/// One window's SpO2 via ratio-of-ratios; `None` if any DC/AC is non-positive.
fn window_spo2(red: &[f64], ir: &[f64]) -> Option<f64> {
    let (dc_red, dc_ir) = (mean(red), mean(ir));
    if dc_red <= 0.0 || dc_ir <= 0.0 {
        return None;
    }
    let (ac_red, ac_ir) = (amplitude(red), amplitude(ir));
    if ac_red <= 0.0 || ac_ir <= 0.0 {
        return None;
    }
    let ratio_ir = ac_ir / dc_ir;
    if ratio_ir <= 0.0 {
        return None;
    }
    let r = (ac_red / dc_red) / ratio_ir;
    Some((CURVE_A - CURVE_B * r).clamp(CLAMP_LOW, CLAMP_HIGH))
}

/// The night value is the median of the surviving per-window SpO2, or `None` if none survived.
fn finish(per_window: Vec<f64>) -> Option<f64> {
    if per_window.is_empty() {
        None
    } else {
        Some(median(&per_window))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A non-pulsatile channel must score nothing. The 4.0 red/IR pair carries a per-second baseline,
    /// not a waveform: on a real 3.8-day offload only 17 of 995 windows varied at all, and the value
    /// they produced (81.8%) would read as severe hypoxaemia.
    #[test]
    fn a_flat_channel_is_not_scored() {
        // Flat but for one varying window in sixty, the shape the real strap produced.
        let mut red = vec![526.0; 1800];
        let mut ir = vec![594.0; 1800];
        for i in 0..30 {
            red[i] += (i % 3) as f64;
            ir[i] += (i % 2) as f64;
        }
        assert_eq!(Spo2::from_paired(&red, &ir), None, "a baseline channel must not yield a percent");
    }

    /// Fraction of the eligible 30 s windows that carry enough AC to score - what
    /// `MIN_PULSATILE_FRACTION` is compared against inside `from_paired`.
    fn pulsatile_fraction(red: &[f64], ir: &[f64]) -> (usize, usize) {
        let n = red.len().min(ir.len());
        let (mut eligible, mut survived) = (0usize, 0usize);
        let mut start = 0;
        while start < n {
            let end = (start + WINDOW_SECONDS).min(n);
            if end - start >= MIN_SAMPLES_PER_WINDOW {
                eligible += 1;
                survived += usize::from(window_spo2(&red[start..end], &ir[start..end]).is_some());
            }
            start = end;
        }
        (survived, eligible)
    }

    /// A pair carrying real AC still scores at every cardiac rate, sampled once per second the way
    /// `from_paired` reads the 4.0 pair: every window survives, so the gate silences nothing that
    /// pulses. Ratio-of-ratios reads only AC/DC, so the value does not depend on the rate.
    #[test]
    fn a_pulsatile_pair_scores_at_every_cardiac_rate() {
        // R = (6/520) / (12/590). 60 and 180 bpm are whole multiples of the 1 Hz sampling, so they
        // land on one phase and leave only float noise as AC; they are not part of the claim.
        let expected = CURVE_A - CURVE_B * (0.5 * 590.0 / 520.0);
        for bpm in [50.0f64, 66.0, 72.0, 90.0, 100.0, 150.0] {
            let (mut red, mut ir) = (Vec::new(), Vec::new());
            for i in 0..1800 {
                let beat = (std::f64::consts::TAU * (bpm / 60.0) * i as f64).sin();
                red.push(520.0 + 6.0 * beat);
                ir.push(590.0 + 12.0 * beat);
            }
            assert_eq!(pulsatile_fraction(&red, &ir), (60, 60), "{bpm} bpm lost windows");
            let v = Spo2::from_paired(&red, &ir).unwrap_or_else(|| panic!("{bpm} bpm must score"));
            assert!((v - expected).abs() < 1e-9, "{bpm} bpm got {v}, want {expected}");
        }
    }

    /// The measured 4.0 night, where the banked red/IR pair is a slow quantised level rather than a
    /// waveform: only ~7% of the 1,420 windows carried any AC, an order of magnitude under the gate,
    /// so the night is withheld instead of printing the ~81% ratio-of-ratios would have claimed.
    #[test]
    fn a_real_4_0_night_is_withheld() {
        let (mut red, mut ir) = (Vec::new(), Vec::new());
        for w in 0..1420usize {
            for s in 0..WINDOW_SECONDS {
                let step = f64::from(w % 14 == 0 && s >= WINDOW_SECONDS / 2);
                red.push(526.0 + step);
                ir.push(594.0 + step);
            }
        }
        let (survived, eligible) = pulsatile_fraction(&red, &ir);
        assert_eq!((survived, eligible), (102, 1420));
        let frac = survived as f64 / eligible as f64;
        assert!(frac < MIN_PULSATILE_FRACTION, "{frac} is not under the gate");
        assert_eq!(Spo2::from_paired(&red, &ir), None, "a slow level channel must not yield a percent");
    }

    /// A 20-sample window with DC = `dc` and p95−p5 amplitude ≈ `ac` (half low, half high).
    fn win(dc: f64, ac: f64) -> Vec<f64> {
        std::iter::repeat_n(dc - ac / 2.0, 10).chain(std::iter::repeat_n(dc + ac / 2.0, 10)).collect()
    }

    /// A window at ratio-of-ratios `r`, with both DC at 100 and the IR AC/DC pinned at 0.04.
    fn at_ratio(r: f64) -> Option<f64> {
        Spo2::from_paired(&win(100.0, 4.0 * r), &win(100.0, 4.0))
    }

    /// `(R, percent)` walking the whole unclamped span of the curve. Two points already fix a line, so
    /// these are literals, not `CURVE_A - CURVE_B * r` recomputed, so a moved constant must fail here.
    const CURVE_POINTS: [(f64, f64); 6] =
        [(0.6, 95.0), (0.8, 90.0), (1.0, 85.0), (1.2, 80.0), (1.4, 75.0), (1.6, 70.0)];

    /// Points of the walk a scorer of R misses. Empty = it reproduces the whole curve.
    fn walk_misses(scorer: &dyn Fn(f64) -> Option<f64>) -> Vec<(f64, Option<f64>)> {
        CURVE_POINTS
            .iter()
            .filter_map(|&(r, want)| {
                let got = scorer(r);
                (!got.is_some_and(|v| (v - want).abs() < 1e-9)).then_some((r, got))
            })
            .collect()
    }

    /// The curve is a line of intercept 110 and slope -25 in R, walked at six ratios rather than one.
    #[test]
    fn the_ratio_of_ratios_curve_is_walked_end_to_end() {
        assert!(walk_misses(&at_ratio).is_empty(), "{:?}", walk_misses(&at_ratio));
        // The two constants, recovered from the walk rather than read from the module.
        let ((r0, v0), (r1, v1)) = (CURVE_POINTS[0], CURVE_POINTS[5]);
        let slope = (v1 - v0) / (r1 - r0);
        assert!((v0 - slope * r0 - CURVE_A).abs() < 1e-9 && (slope + CURVE_B).abs() < 1e-9);
    }

    /// The null arm. Two ratios fix any two-parameter curve, so a scorer bent through both of the
    /// previously gated anchors (R=1 -> 85, R=0.5 -> 97.5) passed while reading wrong everywhere else.
    #[test]
    fn a_curve_through_both_old_anchors_still_fails_the_walk() {
        let bent = |k: f64| move |r: f64| Some(CURVE_A - CURVE_B * r + k * (r - 0.5) * (r - 1.0));
        for k in [0.5f64, 2.0, 20.0] {
            let f = bent(k);
            assert!((f(1.0).unwrap() - 85.0).abs() < 1e-9, "bend {k} moved the R=1 anchor");
            assert!((f(0.5).unwrap() - 97.5).abs() < 1e-9, "bend {k} moved the R=0.5 anchor");
            assert!(!walk_misses(&f).is_empty(), "bend {k} walked the curve");
        }
        for c in [70.0f64, 85.0, 95.0, 97.0, 97.5, 100.0] {
            assert!(!walk_misses(&|_| Some(c)).is_empty(), "constant {c} walked the curve");
        }
        assert_eq!(walk_misses(&|_| None).len(), CURVE_POINTS.len());
    }

    /// Outside the walk the percent is clamped, not extrapolated: R below 0.4 would read over 100 and
    /// R above 1.6 below 70.
    #[test]
    fn the_curve_clamps_instead_of_extrapolating() {
        assert_eq!(at_ratio(0.2), Some(CLAMP_HIGH));
        assert_eq!(at_ratio(2.0), Some(CLAMP_LOW));
        const { assert!(CURVE_A - CURVE_B * 0.2 > CLAMP_HIGH && CURVE_A - CURVE_B * 2.0 < CLAMP_LOW) };
    }

    #[test]
    fn flat_or_absent_signal_is_none() {
        assert_eq!(Spo2::from_paired(&[], &[]), None);
        let flat = vec![100.0; 20];
        assert_eq!(Spo2::from_paired(&flat, &flat), None); // zero AC → no window survives
    }

    #[test]
    fn nightly_raw_means_truncates_in_span() {
        let spans = [(1000i64, 2000i64)];
        // red mean 30000, ir mean 20000; one out-of-span sample is dropped.
        let mut samples: Vec<(i64, i32, i32)> =
            (0..20).map(|i| (1000 + i, if i % 2 == 0 { 29000 } else { 31000 }, if i % 2 == 0 { 19000 } else { 21000 })).collect();
        samples.push((5000, 99999, 99999));
        assert_eq!(Spo2::nightly_raw_means(&spans, &samples), Some((30000, 20000)));
    }

    #[test]
    fn nightly_raw_means_empty_or_no_in_span_is_none() {
        assert_eq!(Spo2::nightly_raw_means(&[], &[(1000, 1, 1)]), None);
        assert_eq!(Spo2::nightly_raw_means(&[(1000, 2000)], &[]), None);
        assert_eq!(Spo2::nightly_raw_means(&[(1000, 2000)], &[(5000, 1, 1)]), None);
    }

    /// Thirty nights whose recent week sits below the month, so `median(30) != median(7)`.
    const SPREAD_NIGHTS: [f64; 30] = [
        93.4, 92.8, 94.1, 93.0, 92.5, 94.6, 93.3, 92.9, 93.8, 94.2, 92.6, 93.5, 93.1, 94.0, 92.7,
        93.9, 93.2, 92.4, 94.3, 93.6, 92.3, 93.7, 94.4, 91.2, 90.5, 91.8, 90.9, 91.5, 90.2, 91.0,
    ];

    #[test]
    fn rolling_reading_calibrates_then_reports() {
        // WHOOP unlocks blood oxygen after one recovery: 0 nights = calibrating, 1 = reported.
        assert_eq!(Spo2::rolling_reading(&[]).calibrating_nights, Some(0));
        // A constant window cancels `offset` against `recent`, so these pin the ANCHOR, not the value.
        for night in [70.0, 96.0, 100.0] {
            assert_eq!(Spo2::rolling_reading(&[night]).pct, Some(97.0), "one night at {night}");
            assert_eq!(Spo2::rolling_reading(&[night; 30]).pct, Some(97.0), "30 nights at {night}");
        }
    }

    #[test]
    fn rolling_reading_carries_the_recent_week_against_the_month() {
        // month median 93.05 → offset 3.45; recent-week median 91.0 → 94.45 → round 94.
        assert_eq!(Spo2::rolling_reading(&SPREAD_NIGHTS).pct, Some(94.0));
        // Flattening the spread removes the differential and collapses the readout onto the ANCHOR.
        assert_eq!(Spo2::rolling_reading(&[93.05; 30]).pct, Some(97.0));
    }
}
