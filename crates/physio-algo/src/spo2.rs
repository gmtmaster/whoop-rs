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

    /// A genuinely pulsatile pair still scores, so the gate does not silence real data.
    #[test]
    fn a_pulsatile_channel_still_scores() {
        let mut red = Vec::new();
        let mut ir = Vec::new();
        for i in 0..1800 {
            let beat = ((i as f64) * 0.7).sin();
            red.push(520.0 + 6.0 * beat);
            ir.push(590.0 + 12.0 * beat);
        }
        let v = Spo2::from_paired(&red, &ir).expect("a pulsatile pair must score");
        assert!((70.0..=100.0).contains(&v), "got {v}");
    }

    /// A 20-sample window with DC = `dc` and p95−p5 amplitude ≈ `ac` (half low, half high).
    fn win(dc: f64, ac: f64) -> Vec<f64> {
        std::iter::repeat_n(dc - ac / 2.0, 10).chain(std::iter::repeat_n(dc + ac / 2.0, 10)).collect()
    }

    #[test]
    fn ratio_one_gives_the_curve_midpoint() {
        // identical red/IR → R = 1 → 110 − 25 = 85.
        let w = win(100.0, 4.0);
        assert_eq!(Spo2::from_paired(&w, &w), Some(85.0));
    }

    #[test]
    fn half_ratio_reads_higher() {
        // red AC/DC 0.02, IR AC/DC 0.04 → R = 0.5 → 110 − 12.5 = 97.5.
        let red = win(100.0, 2.0);
        let ir = win(100.0, 4.0);
        assert_eq!(Spo2::from_paired(&red, &ir), Some(97.5));
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

    #[test]
    fn rolling_reading_calibrates_then_reports() {
        // WHOOP unlocks blood oxygen after one recovery: 0 nights = calibrating, 1 = reported.
        assert_eq!(Spo2::rolling_reading(&[]).calibrating_nights, Some(0));
        // median 96 → offset 0.5 → recent median 96 → 96.5 → round 97.
        assert_eq!(Spo2::rolling_reading(&[96.0]).pct, Some(97.0));
    }
}
