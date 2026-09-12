//! Approximate sleeping respiratory rate (breaths/min) from the R-R interval stream via respiratory
//! sinus arrhythmia: reconstruct a beat-interval tachogram, resample to 4 Hz, detrend, then per 5-min
//! window peak-pick the breathing modulation. Each window's rate is screened against the plausible band
//! before it counts, and the nightly value is the median of the surviving windows — the robust nightly
//! aggregate over per-window estimates, never a single selected window. Pure; `None` = no usable estimate.
//! The only respiratory source for both generations: the 4.0 v24 register decoded as `resp_raw` carries
//! a status byte, not a breathing signal, so nothing here reads it (measured; see `docs/algorithms.md`).

use crate::signal::{find_peaks, moving_average_centred};
use crate::stats::{median, population_sd};

const RR_MIN_MS: f64 = 300.0;
const RR_MAX_MS: f64 = 2000.0;

const RSA_RESAMPLE_HZ: f64 = 4.0;
const RSA_DETREND_WINDOW_S: f64 = 8.0;
const RSA_MIN_PEAK_DISTANCE_S: f64 = 2.5;
const RSA_WINDOW_S: f64 = 300.0;
const RSA_MIN_BREATH_INTERVAL_S: f64 = 2.5;
const RSA_MAX_BREATH_INTERVAL_S: f64 = 10.0;
// Report anchors are integer seconds, but whether a WHOOP historical (v18) record's timestamp
// leads or trails the RR payload it carries is NOT established anywhere in the decode path
// (gen5.rs / cloud-map / HistoricalStreams.swift all copy one record timestamp onto its whole
// interval array without documenting which end of the array it names). This threshold is
// deliberately conservative given that ambiguity: it exists to catch only large, unambiguous
// missing-time blocks, not to certify that anything below it is physiologically harmless.
//
// On the Sep 2-12 cohort (236,050 adjacent report comparisons), ordinary report geometry is
// compact under EITHER possible timestamp orientation (p99 < 1s), and the two orientations
// identify the exact same 577 gaps above 10 seconds. Below 10s the two orientations disagree on
// a handful of borderline events whose missing-time semantics are unproven either way, so the
// threshold is set above that borderline band rather than inside it.
const REPORT_GAP_EXCESS_TOLERANCE_S: f64 = 10.0;

/// Version of the RR-derived respiratory estimator persisted by its production caller.
pub const RESPIRATORY_ALGORITHM_VERSION: &str = "rr-rsa-4hz-p1-gapaware-v3";

/// Plausible sleeping-respiratory-rate band (bpm); an estimate outside it is `None`.
pub const RESP_PLAUSIBLE_MIN_BPM: f64 = 8.0;
pub const RESP_PLAUSIBLE_MAX_BPM: f64 = 25.0;

/// One accepted per-window nightly RR estimate: the window's `[start, end]` (unix seconds, anchored to
/// the first accepted beat in the span) and the breaths/min it resolved to. Only windows that already
/// cleared the peak-count and plausible-band gates appear here — the pool [`nightly_resp_rate`] takes
/// its median over.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RespRateWindow {
    pub start: i64,
    pub end: i64,
    pub bpm: f64,
}

/// The full nightly respiratory-rate diagnostic: every valid per-window estimate plus their median, the
/// nightly value. `median_bpm` is `None` (with `windows` empty) when nothing valid survives — no
/// fabricated number, matching every other physio-algo estimator's empty contract.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RespRateNightly {
    pub windows: Vec<RespRateWindow>,
    pub median_bpm: Option<f64>,
    pub wall_clock_span_s: f64,
    pub represented_rr_time_s: f64,
    pub rr_time_coverage_ratio: f64,
    pub detected_gap_count: u32,
    pub max_gap_excess_s: f64,
    pub respiratory_windows_total: u32,
    pub respiratory_windows_rejected_gap: u32,
    pub respiratory_windows_accepted: u32,
}

/// Sleeping respiratory rate over the in-bed `[start, end]` window (unix seconds), from `(ts, rr_ms)`
/// beats. An on-device wellness estimate, never a clinical measurement; `None` on too-little data.
/// The nightly median of [`nightly_resp_rate`]'s valid per-window estimates.
pub fn resp_rate_from_rr(rr: &[(i64, u16)], start: i64, end: i64) -> Option<f64> {
    nightly_resp_rate(rr, start, end).median_bpm
}

/// Per-5-minute-window respiratory-rate estimates over the in-bed `[start, end]` span (unix seconds),
/// from `(ts, rr_ms)` beats already scoped to ONE session (a caller must never pool beats across a
/// primary sleep session and a nap: call once per session). Each window is screened against
/// [`RESP_PLAUSIBLE_MIN_BPM`]/[`RESP_PLAUSIBLE_MAX_BPM`] before it is kept, so the nightly median can
/// never be pulled off-band by a single noisy window. `windows` is empty and `median_bpm` is `None`
/// when too little data survives the RR-interval range gate, the RSA detrend has no signal, or no
/// window's peak-picked breathing rate clears the plausible band.
pub fn nightly_resp_rate(rr: &[(i64, u16)], start: i64, end: i64) -> RespRateNightly {
    nightly_resp_rate_with_refinement(rr, start, end, true)
}

fn nightly_resp_rate_with_refinement(
    rr: &[(i64, u16)],
    start: i64,
    end: i64,
    refine_peaks: bool,
) -> RespRateNightly {
    if end <= start {
        return RespRateNightly::default();
    }

    let mut in_bed: Vec<(i64, f64)> = rr
        .iter()
        .filter(|(ts, _)| *ts >= start && *ts <= end)
        .map(|(ts, ms)| (*ts, *ms as f64))
        .collect();
    in_bed.sort_by_key(|(ts, _)| *ts);
    let wall_clock_span_s = (end - start) as f64;
    let represented_rr_time_s = in_bed
        .iter()
        .filter(|(_, ms)| (RR_MIN_MS..=RR_MAX_MS).contains(ms))
        .map(|(_, ms)| ms / 1000.0)
        .sum::<f64>();
    let rr_time_coverage_ratio = represented_rr_time_s / wall_clock_span_s;
    let (segments, detected_gap_count, max_gap_excess_s) = contiguous_segments(&in_bed);
    let mut windows = Vec::new();
    let mut examined = 0u32;
    let mut rejected_gap = 0u32;
    // Every independent segment - whether or not a gap created it - is scored under the SAME
    // pre-existing estimator minimum-length semantics that gap-free input always used
    // (`score_segment`'s `RSA_WINDOW_S / 2.0`, i.e. 150s). There is no separate, stricter minimum
    // for a boundary-created fragment: the only behavioral change from gap-free scoring is that no
    // window may straddle a detected gap.
    for (i, segment) in segments.into_iter().enumerate() {
        let filtered: Vec<(i64, f64)> = segment
            .into_iter()
            .filter(|(_, ms)| (RR_MIN_MS..=RR_MAX_MS).contains(ms))
            .collect();
        let (mut segment_windows, segment_examined) = score_segment(&filtered, refine_peaks);
        if i > 0 && segment_windows.is_empty() {
            // A segment created by splitting at a detected gap that could not, on its own, clear
            // the estimator's ordinary length/signal gates - the same gates a gap-free session is
            // held to. Not a separate gap-specific rule; just an honest count of it happening.
            rejected_gap += 1;
        }
        windows.append(&mut segment_windows);
        examined += segment_examined;
    }
    let bpms: Vec<f64> = windows.iter().map(|w| w.bpm).collect();
    let respiratory_windows_accepted = windows.len() as u32;
    RespRateNightly {
        windows,
        median_bpm: (!bpms.is_empty()).then(|| median(&bpms)),
        wall_clock_span_s,
        represented_rr_time_s,
        rr_time_coverage_ratio,
        detected_gap_count,
        max_gap_excess_s,
        respiratory_windows_total: examined + rejected_gap,
        respiratory_windows_rejected_gap: rejected_gap,
        respiratory_windows_accepted,
    }
}

fn contiguous_segments(rows: &[(i64, f64)]) -> (Vec<Vec<(i64, f64)>>, u32, f64) {
    if rows.is_empty() {
        return (Vec::new(), 0, 0.0);
    }
    let mut segments = vec![Vec::new()];
    let mut gaps = 0u32;
    let mut max_excess = 0.0_f64;
    let mut group_start = 0usize;
    while group_start < rows.len() {
        let ts = rows[group_start].0;
        let mut group_end = group_start;
        let mut payload_s = 0.0;
        while group_end < rows.len() && rows[group_end].0 == ts {
            payload_s += rows[group_end].1 / 1000.0;
            group_end += 1;
        }
        segments
            .last_mut()
            .unwrap()
            .extend_from_slice(&rows[group_start..group_end]);
        if group_end < rows.len() {
            let wall_step_s = (rows[group_end].0 - ts).max(0) as f64;
            let excess_s = wall_step_s - payload_s;
            max_excess = max_excess.max(excess_s);
            if excess_s > REPORT_GAP_EXCESS_TOLERANCE_S {
                gaps += 1;
                segments.push(Vec::new());
            }
        }
        group_start = group_end;
    }
    (segments, gaps, max_excess.max(0.0))
}

fn score_segment(filtered_pairs: &[(i64, f64)], refine_peaks: bool) -> (Vec<RespRateWindow>, u32) {
    if filtered_pairs.len() < 30 {
        return (Vec::new(), 0);
    }
    let anchor_ts = filtered_pairs[0].0;
    let filtered: Vec<f64> = filtered_pairs.iter().map(|&(_, ms)| ms).collect();
    let mut beat_times = vec![0.0; filtered.len()];
    let mut acc = 0.0;
    for (i, &ms) in filtered.iter().enumerate() {
        acc += ms / 1000.0;
        beat_times[i] = acc;
    }
    let total_span_s = *beat_times.last().unwrap();
    if total_span_s < RSA_WINDOW_S / 2.0 {
        return (Vec::new(), 0);
    }
    let dt = 1.0 / RSA_RESAMPLE_HZ;
    let n_grid = (total_span_s / dt) as usize + 1;
    if n_grid < 8 {
        return (Vec::new(), 0);
    }
    let mut grid = vec![0.0; n_grid];
    let mut seg = 0usize;
    for (g, cell) in grid.iter_mut().enumerate() {
        let t = g as f64 * dt;
        while seg < beat_times.len() - 2 && beat_times[seg + 1] < t {
            seg += 1;
        }
        let (t0, t1) = (beat_times[seg], beat_times[seg + 1]);
        let (v0, v1) = (filtered[seg], filtered[seg + 1]);
        *cell = if t1 <= t0 {
            v0
        } else {
            v0 + ((t - t0) / (t1 - t0)).clamp(0.0, 1.0) * (v1 - v0)
        };
    }
    let half_w = ((RSA_DETREND_WINDOW_S * RSA_RESAMPLE_HZ / 2.0).round() as usize).max(1);
    let baseline = moving_average_centred(&grid, 2 * half_w + 1);
    let detrended: Vec<f64> = (0..n_grid).map(|i| grid[i] - baseline[i]).collect();
    if population_sd(&detrended) <= 1e-9 {
        return (Vec::new(), 0);
    }
    let min_dist = ((RSA_MIN_PEAK_DISTANCE_S * RSA_RESAMPLE_HZ).round() as usize).max(2);
    let window_samples = ((RSA_WINDOW_S * RSA_RESAMPLE_HZ).round() as usize).max(min_dist * 3);
    let mut windows = Vec::new();
    let mut examined = 0u32;
    let mut w = 0usize;
    while w < n_grid {
        let w_end = (w + window_samples).min(n_grid);
        if w_end - w >= min_dist * 3 {
            examined += 1;
            let signal = &detrended[w..w_end];
            let peaks = find_peaks(signal, min_dist, 0.0);
            if peaks.len() >= 3 {
                let refined: Vec<f64> = peaks
                    .iter()
                    .map(|&p| {
                        if refine_peaks {
                            refine_peak_parabolic(signal, p)
                        } else {
                            p as f64
                        }
                    })
                    .collect();
                let intervals: Vec<f64> = refined
                    .windows(2)
                    .map(|p| (p[1] - p[0]) * dt)
                    .filter(|iv| {
                        (RSA_MIN_BREATH_INTERVAL_S..=RSA_MAX_BREATH_INTERVAL_S).contains(iv)
                    })
                    .collect();
                if intervals.len() >= 2 {
                    let med = median(&intervals);
                    if med > 0.0 {
                        let bpm = 60.0 / med;
                        if (RESP_PLAUSIBLE_MIN_BPM..=RESP_PLAUSIBLE_MAX_BPM).contains(&bpm) {
                            windows.push(RespRateWindow {
                                start: anchor_ts + (w as f64 * dt) as i64,
                                end: anchor_ts + (w_end as f64 * dt) as i64,
                                bpm,
                            });
                        }
                    }
                }
            }
        }
        w += window_samples;
    }
    (windows, examined)
}

fn refine_peak_parabolic(signal: &[f64], peak: usize) -> f64 {
    let integer = peak as f64;
    if peak == 0 || peak + 1 >= signal.len() {
        return integer;
    }
    let (left, center, right) = (signal[peak - 1], signal[peak], signal[peak + 1]);
    if !left.is_finite()
        || !center.is_finite()
        || !right.is_finite()
        || center < left
        || center < right
    {
        return integer;
    }
    let denominator = left - 2.0 * center + right;
    let scale = left.abs().max(center.abs()).max(right.abs()).max(1.0);
    if !denominator.is_finite()
        || denominator >= 0.0
        || denominator.abs() <= 16.0 * f64::EPSILON * scale
    {
        return integer;
    }
    parabolic_offset(left, right, denominator).map_or(integer, |delta| integer + delta)
}

fn parabolic_offset(left: f64, right: f64, denominator: f64) -> Option<f64> {
    let delta = 0.5 * (left - right) / denominator;
    delta.is_finite().then(|| delta.clamp(-0.5, 0.5))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parabolic_peak_refinement_handles_valid_and_invalid_shapes() {
        assert_eq!(refine_peak_parabolic(&[0.0, 2.0, 0.0], 1), 1.0);
        assert!(
            (refine_peak_parabolic(&[0.0, 4.0, 2.0], 1) - 1.166_666_666_666_666_7).abs() < 1e-12
        );
        assert!(
            (refine_peak_parabolic(&[2.0, 4.0, 0.0], 1) - 0.833_333_333_333_333_4).abs() < 1e-12
        );
        assert_eq!(refine_peak_parabolic(&[1.0, 1.0, 1.0], 1), 1.0);
        assert_eq!(refine_peak_parabolic(&[2.0, 1.0, 2.0], 1), 1.0);
        assert_eq!(refine_peak_parabolic(&[2.0, 1.0, 0.0], 0), 0.0);
        assert_eq!(refine_peak_parabolic(&[0.0, 1.0, 2.0], 2), 2.0);
        assert_eq!(refine_peak_parabolic(&[f64::NAN, 2.0, 0.0], 1), 1.0);
        assert_eq!(refine_peak_parabolic(&[0.0, f64::INFINITY, 0.0], 1), 1.0);
    }

    #[test]
    fn parabolic_peak_refinement_clamps_and_preserves_order() {
        assert_eq!(parabolic_offset(4.0, 0.0, -2.0), Some(-0.5));
        assert_eq!(parabolic_offset(0.0, 4.0, -2.0), Some(0.5));
        let signal = [0.0, 5.0, 0.0, 0.0, 4.0, 0.0];
        let refined = [
            refine_peak_parabolic(&signal, 1),
            refine_peak_parabolic(&signal, 4),
        ];
        assert!(refined[0] < refined[1]);
    }

    #[test]
    fn p1_removes_the_integer_grid_value_without_a_large_nightly_shift() {
        let (rows, start, end) = synth(12.5 / 60.0, 1_000.0, 45.0, 900.0);
        let p0 = nightly_resp_rate_with_refinement(&rows, start, end, false);
        let p1 = nightly_resp_rate(&rows, start, end);
        let p0_rate = p0.median_bpm.expect("P0 estimate");
        let p1_rate = p1.median_bpm.expect("P1 estimate");

        assert_eq!(p0_rate, 12.0);
        assert_ne!(p1_rate, p0_rate);
        assert!((p1_rate - p0_rate).abs() < 0.5);
        assert_eq!(p1.windows.len(), p0.windows.len());
    }

    fn shift_after(rows: &mut [(i64, u16)], boundary: i64, seconds: i64) {
        for (ts, _) in rows {
            if *ts >= boundary {
                *ts += seconds;
            }
        }
    }

    #[test]
    fn contiguous_input_retains_result_windows_and_p1() {
        let (rows, start, end) = synth(14.0 / 60.0, 1_000.0, 40.0, 900.0);
        let result = nightly_resp_rate(&rows, start, end);
        let direct = score_segment(
            &rows
                .iter()
                .map(|&(t, rr)| (t, f64::from(rr)))
                .collect::<Vec<_>>(),
            true,
        );
        assert_eq!(result.detected_gap_count, 0);
        assert_eq!(result.windows, direct.0);
        assert_eq!(
            result.respiratory_windows_accepted,
            result.windows.len() as u32
        );
    }

    #[test]
    fn obvious_gap_splits_and_never_emits_a_cross_gap_window() {
        let (mut rows, start, end) = synth(14.0 / 60.0, 1_000.0, 40.0, 720.0);
        let boundary = start + 360;
        shift_after(&mut rows, boundary, 45);
        let result = nightly_resp_rate(&rows, start, end + 45);
        assert_eq!(result.detected_gap_count, 1);
        assert!(result.max_gap_excess_s > 40.0);
        // Both 360s segments clear the historical 150s minimum on their own, so neither was
        // actually rejected for being too short.
        assert_eq!(result.respiratory_windows_rejected_gap, 0);
        assert!(result
            .windows
            .iter()
            .all(|w| w.end <= boundary || w.start >= boundary + 45));
        assert!(result.windows.iter().any(|w| w.end <= boundary));
        assert!(result.windows.iter().any(|w| w.start >= boundary + 45));
    }

    #[test]
    fn boundary_segments_between_150_and_300_seconds_are_independently_eligible() {
        // Each side of the gap is ~200s of RR time: above the historical 150s minimum
        // (`RSA_WINDOW_S / 2.0`) but below the old, removed 300s post-gap special case. Both
        // must be scored - there is no separate, stricter minimum for a boundary-created
        // segment - and neither may combine with the other across the gap.
        let (mut rows, start, end) = synth(14.0 / 60.0, 1_000.0, 40.0, 400.0);
        let boundary = start + 200;
        shift_after(&mut rows, boundary, 40);
        let result = nightly_resp_rate(&rows, start, end + 40);
        assert_eq!(result.detected_gap_count, 1);
        assert_eq!(result.respiratory_windows_rejected_gap, 0);
        assert!(result.median_bpm.is_some(), "{:?}", result);
        // Both sides produce a window (neither was too short to score), and the two windows are
        // separated by a gap-sized jump rather than being adjacent, gap-free 5-minute windows -
        // proof the two segments were never pooled into shared RSA windows.
        assert_eq!(result.windows.len(), 2, "{:?}", result.windows);
        assert!(result.windows[0].end < result.windows[1].start);
        assert!(
            result.windows[1].start - result.windows[0].end >= 35,
            "windows must be separated by the ~40s gap, not adjacent: {:?}",
            result.windows
        );
    }

    #[test]
    fn a_post_gap_segment_under_the_historical_minimum_is_honestly_counted_as_rejected() {
        // First side is comfortably long; the far side of the gap is only ~50s, well under the
        // 150s historical minimum, so it produces no windows and must be honestly reflected in
        // `respiratory_windows_rejected_gap` (not silently absorbed).
        let base = 1_700_000_000_i64;
        let (mut first, first_end) = synth_from(base, 14.0 / 60.0, 1_000.0, 40.0, 360.0);
        let short_start = first_end + 40;
        let (short, end) = synth_from(short_start, 14.0 / 60.0, 1_000.0, 40.0, 50.0);
        first.extend(short);
        let result = nightly_resp_rate(&first, base, end);
        assert_eq!(result.detected_gap_count, 1);
        assert_eq!(result.respiratory_windows_rejected_gap, 1);
        assert!(!result.windows.is_empty());
        // Every surviving window belongs to the first (long) segment; none reaches anywhere near
        // the short segment's own span, which starts well after the gap.
        assert!(result.windows.iter().all(|w| w.end < short_start));
    }

    #[test]
    fn exactly_ten_seconds_unexplained_does_not_break_a_segment() {
        // Comparison against the threshold is strict `>`, so an excess of exactly 10.0s must
        // stay inside one segment, not split it.
        let base = 1_700_000_000;
        let rows = vec![(base, 1_000.0), (base + 11, 1_000.0)];
        let (segments, gaps, _) = contiguous_segments(&rows);
        assert_eq!(gaps, 0, "exactly 10.0s excess must not cross the strict > threshold");
        assert_eq!(segments.len(), 1);
    }

    #[test]
    fn borderline_three_to_ten_second_excess_does_not_split_a_segment() {
        // Representative geometry in the 3-5s and 5-10s bands (the range the conservative
        // threshold deliberately leaves alone) must not create a segment boundary.
        for excess in [3, 4, 5, 9, 10] {
            let base = 1_700_000_000;
            let wall_step = 1 + excess;
            let rows = vec![(base, 1_000.0), (base + wall_step, 1_000.0)];
            let (segments, gaps, _) = contiguous_segments(&rows);
            assert_eq!(gaps, 0, "excess {excess}s must not create a boundary");
            assert_eq!(segments.len(), 1);
        }
    }

    #[test]
    fn nightly_median_uses_only_windows_surviving_gap_segmentation() {
        let base = 1_700_000_000_i64;
        let (mut rows, first_end) = synth_from(base, 12.0 / 60.0, 1_000.0, 40.0, 360.0);
        let middle_start = first_end + 40;
        let (middle, middle_end) = synth_from(middle_start, 18.0 / 60.0, 1_000.0, 40.0, 120.0);
        rows.extend(middle);
        let last_start = middle_end + 40;
        let (last, end) = synth_from(last_start, 20.0 / 60.0, 1_000.0, 40.0, 360.0);
        rows.extend(last);
        let result = nightly_resp_rate(&rows, base, end);
        assert_eq!(result.detected_gap_count, 2);
        assert_eq!(result.windows.len(), 4);
        let expected = median(&result.windows.iter().map(|w| w.bpm).collect::<Vec<_>>());
        assert_eq!(result.median_bpm, Some(expected));
        assert!(result
            .windows
            .iter()
            .all(|w| w.end <= middle_start || w.start >= last_start));
    }

    #[test]
    fn benign_quantization_batching_and_shared_timestamps_do_not_break() {
        let base = 1_700_000_000;
        let rows = vec![
            (base, 1_000),
            (base, 1_000),
            (base + 4, 1_000),
            (base + 4, 1_000),
        ];
        let (_, gaps, max_excess) = contiguous_segments(
            &rows
                .iter()
                .map(|&(t, rr)| (t, f64::from(rr)))
                .collect::<Vec<_>>(),
        );
        assert_eq!(
            gaps, 0,
            "three-second excess is tolerated, got {max_excess}"
        );
    }

    #[test]
    fn range_rejection_does_not_create_a_wall_clock_gap() {
        let base = 1_700_000_000;
        let rows = vec![(base, 1_000.0), (base + 1, 2_500.0), (base + 2, 1_000.0)];
        let (segments, gaps, _) = contiguous_segments(&rows);
        assert_eq!(gaps, 0);
        assert_eq!(segments.len(), 1);
    }

    /// Synthetic tachogram: mean HR with a known-Hz RSA modulation, so the recovered rate can be
    /// cross-checked against the planted breathing frequency.
    fn synth(
        breath_hz: f64,
        base_rr_ms: f64,
        amp_ms: f64,
        span_s: f64,
    ) -> (Vec<(i64, u16)>, i64, i64) {
        let start = 1_700_000_000_i64;
        let mut rows = Vec::new();
        let mut t_sec = 0.0_f64;
        while t_sec < span_s {
            let rr_ms =
                base_rr_ms + amp_ms * (2.0 * std::f64::consts::PI * breath_hz * t_sec).sin();
            t_sec += rr_ms / 1000.0;
            rows.push((start + t_sec as i64, rr_ms as u16));
        }
        let end = start + t_sec as i64;
        (rows, start, end)
    }

    /// Planted breathing rates the sweep walks; the 10 bpm span is what no constant can cover.
    const SWEPT_BPM: [f64; 5] = [10.0, 12.0, 15.0, 18.0, 20.0];
    /// `(mean HR, RSA amplitude ms, span s)`: three tachogram densities per planted rate.
    const SWEPT_PROFILES: [(f64, f64, f64); 3] = [
        (60.0, 40.0, 600.0),
        (55.0, 45.0, 600.0),
        (70.0, 30.0, 900.0),
    ];
    /// Worst measured error over the sweep is 2.0 bpm, at 18/min on the 60 bpm profile.
    const SWEPT_TOL_BPM: f64 = 2.5;

    /// Anything that turns an in-bed R-R window into a breathing rate: the shipped function, or a null.
    type RespScorer<'a> = &'a dyn Fn(&[(i64, u16)], i64, i64) -> Option<f64>;

    /// Arms of the sweep a scorer misses, as `(planted, profile HR, returned)`. Empty = it tracks.
    fn sweep_misses(scorer: RespScorer) -> Vec<(f64, f64, Option<f64>)> {
        let mut bad = Vec::new();
        for bpm in SWEPT_BPM {
            for (hr, amp, span) in SWEPT_PROFILES {
                let (rows, start, end) = synth(bpm / 60.0, 60_000.0 / hr, amp, span);
                let got = scorer(&rows, start, end);
                if !got.is_some_and(|v| (v - bpm).abs() <= SWEPT_TOL_BPM) {
                    bad.push((bpm, hr, got));
                }
            }
        }
        bad
    }

    /// The estimate tracks a planted rate across 10-20 breaths/min on three tachogram densities, so
    /// what is measured is the recovery of a VARYING rate, not one lucky tone.
    #[test]
    fn tracks_a_swept_breathing_rate_across_the_band() {
        assert!(
            sweep_misses(&resp_rate_from_rr).is_empty(),
            "{:?}",
            sweep_misses(&resp_rate_from_rr)
        );
    }

    /// The null arm: no constant, and no refusal, survives the sweep. A single 15/min tone alone was
    /// satisfied by the constant 13.0, which also satisfied the slow-breather gate below.
    #[test]
    fn no_constant_breathing_rate_survives_the_sweep() {
        let mut c = RESP_PLAUSIBLE_MIN_BPM;
        while c <= RESP_PLAUSIBLE_MAX_BPM {
            assert!(
                !sweep_misses(&|_, _, _| Some(c)).is_empty(),
                "constant {c} passed the sweep"
            );
            c += 0.5;
        }
        assert!(
            !sweep_misses(&|_, _, _| None).is_empty(),
            "a refusing scorer passed the sweep"
        );
        // The old single-tone pair was blind to exactly this value.
        assert!(!sweep_misses(&|_, _, _| Some(13.0)).is_empty());
    }

    /// Sensitivity, not just accuracy: a 10 bpm move in the truth must move the estimate at least
    /// 6 bpm on every profile. Smallest measured spread is 8.24 bpm.
    #[test]
    fn a_ten_bpm_move_in_the_truth_moves_the_estimate() {
        for (hr, amp, span) in SWEPT_PROFILES {
            let at = |bpm: f64| {
                let (rows, s, e) = synth(bpm / 60.0, 60_000.0 / hr, amp, span);
                resp_rate_from_rr(&rows, s, e).expect("finite estimate")
            };
            let spread = at(20.0) - at(10.0);
            assert!(
                spread >= 6.0,
                "HR {hr}: 10 -> 20 bpm moved the estimate only {spread}"
            );
        }
    }

    /// Recorded limit, not a desired one: the 2.5 s peak spacing caps the estimator at 24 breaths/min,
    /// so above ~20 on a slow tachogram adjacent breaths merge and the rate HALVES into a normal-looking
    /// value. `RESP_PLAUSIBLE_MAX_BPM` sits above that cap and does not catch it.
    #[test]
    fn above_the_peak_spacing_cap_the_rate_halves_inside_the_plausible_band() {
        let at = |bpm: f64, hr: f64, span: f64| {
            let (rows, s, e) = synth(bpm / 60.0, 60_000.0 / hr, 45.0, span);
            resp_rate_from_rr(&rows, s, e).expect("finite estimate")
        };
        let halved = at(24.0, 60.0, 600.0);
        assert!((halved - 12.0).abs() < 0.5, "24/min at HR 60 read {halved}");
        assert!(
            (RESP_PLAUSIBLE_MIN_BPM..=RESP_PLAUSIBLE_MAX_BPM).contains(&halved),
            "and it is in band"
        );
        // The band ceiling sits above the peak-spacing cap, which is why the halved value stays in band.
        const { assert!(60.0 / RSA_MIN_PEAK_DISTANCE_S < RESP_PLAUSIBLE_MAX_BPM) };
        // A faster tachogram carries the same rate, so it is the beat density, not the rate.
        assert!((at(24.0, 70.0, 900.0) - 24.0).abs() <= 1.0);
    }

    #[test]
    fn slow_breather_is_not_doubled() {
        // 11 breaths/min must read ~11, not the doubled ~20-21 harmonic.
        let (rows, start, end) = synth(11.0 / 60.0, 60000.0 / 55.0, 45.0, 480.0);
        let est = resp_rate_from_rr(&rows, start, end).expect("finite estimate");
        assert!((est - 11.0).abs() <= 2.0, "expected ~11, got {est}");
        assert!(est < 16.0, "must not double toward ~22, got {est}");
    }

    #[test]
    fn too_few_beats_is_none() {
        let start = 1_700_000_000_i64;
        let rows = vec![(start + 1, 1000u16), (start + 2, 1000), (start + 3, 1000)];
        assert!(resp_rate_from_rr(&rows, start, start + 10).is_none());
        assert!(resp_rate_from_rr(&[], start, start + 10).is_none());
    }

    #[test]
    fn empty_or_inverted_window_is_none() {
        let (rows, start, end) = synth(0.25, 1000.0, 40.0, 420.0);
        assert!(resp_rate_from_rr(&rows, end, start).is_none());
    }

    /// As [`synth`], but the caller picks `start` so several planted-rate segments can be concatenated
    /// back to back into one continuous tachogram, each landing in its own 5-minute RSA window.
    fn synth_from(
        start: i64,
        breath_hz: f64,
        base_rr_ms: f64,
        amp_ms: f64,
        span_s: f64,
    ) -> (Vec<(i64, u16)>, i64) {
        let mut rows = Vec::new();
        let mut t_sec = 0.0_f64;
        while t_sec < span_s {
            let rr_ms =
                base_rr_ms + amp_ms * (2.0 * std::f64::consts::PI * breath_hz * t_sec).sin();
            t_sec += rr_ms / 1000.0;
            rows.push((start + t_sec as i64, rr_ms as u16));
        }
        (rows, start + t_sec as i64)
    }

    /// Three planted rates, one per 5-minute window: the nightly value is the MEDIAN across windows
    /// (the middle one), not the mean, the min, or a single selected window.
    #[test]
    fn nightly_median_over_an_odd_window_count() {
        let base = 1_700_000_000_i64;
        let (mut rows, t1) = synth_from(base, 10.0 / 60.0, 1000.0, 40.0, 300.0);
        let (rows2, t2) = synth_from(t1, 14.0 / 60.0, 1000.0, 40.0, 300.0);
        rows.extend(rows2);
        let (rows3, t3) = synth_from(t2, 20.0 / 60.0, 1000.0, 40.0, 300.0);
        rows.extend(rows3);

        let result = nightly_resp_rate(&rows, base, t3);
        assert_eq!(result.windows.len(), 3, "{:?}", result.windows);
        let m = result.median_bpm.expect("finite nightly estimate");
        // median(10, 14, 20) = 14, the middle planted rate, not mean(14.67) or min(10).
        assert!((m - 14.0).abs() < 3.0, "expected ~14, got {m}");
    }

    /// Two planted rates -> the nightly value is the AVERAGE of the two middle (here, the only two)
    /// window estimates, per the standard even-count median convention.
    #[test]
    fn nightly_median_over_an_even_window_count() {
        let base = 1_700_000_000_i64;
        let (mut rows, t1) = synth_from(base, 10.0 / 60.0, 1000.0, 40.0, 300.0);
        let (rows2, t2) = synth_from(t1, 18.0 / 60.0, 1000.0, 40.0, 300.0);
        rows.extend(rows2);

        let result = nightly_resp_rate(&rows, base, t2);
        assert_eq!(result.windows.len(), 2, "{:?}", result.windows);
        let m = result.median_bpm.expect("finite nightly estimate");
        // median(10, 18) = 14.0 = (10 + 18) / 2.
        assert!((m - 14.0).abs() < 3.0, "expected ~14, got {m}");
    }

    /// A window whose peak-picked rate falls outside the plausible band must be dropped from the pool
    /// BEFORE the median is taken, not just checked on the final aggregate: two windows at 7 and 15 bpm
    /// averaging to ~11 would (wrongly) pass an aggregate-only gate, silently letting the implausible
    /// window corrupt the nightly value. Only the 15 bpm window may survive.
    #[test]
    fn implausible_window_is_excluded_before_the_nightly_median() {
        assert!(
            7.0 < RESP_PLAUSIBLE_MIN_BPM,
            "fixture must sit below the band"
        );
        let base = 1_700_000_000_i64;
        let (mut rows, t1) = synth_from(base, 7.0 / 60.0, 1000.0, 40.0, 300.0);
        let (rows2, t2) = synth_from(t1, 15.0 / 60.0, 1000.0, 40.0, 300.0);
        rows.extend(rows2);

        let result = nightly_resp_rate(&rows, base, t2);
        assert_eq!(result.windows.len(), 1, "{:?}", result.windows);
        let m = result.median_bpm.expect("the surviving 15 bpm window");
        assert!((m - 15.0).abs() < 3.0, "expected ~15, got {m}");
    }

    /// A nap sitting outside `[start, end]` must never enter the primary session's nightly pool: the
    /// caller scopes by passing the primary session's own span, and the range filter enforces it.
    #[test]
    fn beats_outside_the_session_span_are_never_pooled() {
        let base = 1_700_000_000_i64;
        let (primary_rows, primary_end) = synth_from(base, 14.0 / 60.0, 1000.0, 40.0, 600.0);
        // A "nap" recorded hours later, at a very different planted rate.
        let nap_start = primary_end + 4 * 3600;
        let (nap_rows, _) = synth_from(nap_start, 22.0 / 60.0, 1000.0, 40.0, 600.0);

        let mut all_beats = primary_rows.clone();
        all_beats.extend(nap_rows);

        let primary_only = nightly_resp_rate(&primary_rows, base, primary_end);
        let scoped_from_pooled = nightly_resp_rate(&all_beats, base, primary_end);
        assert_eq!(
            primary_only.median_bpm, scoped_from_pooled.median_bpm,
            "the nap's beats (outside [start, end]) must not change the primary session's result"
        );
    }
}
