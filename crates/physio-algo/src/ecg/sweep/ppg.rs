//! Beat-by-beat agreement between a candidate ECG decode and the optical channel's own beat times, and
//! the sample rate that falls out of it.
//!
//! The optical beats arrive as absolute milliseconds; the ECG peaks are sample indices under a candidate
//! rate. That is the whole point: converting one to the other IS the rate hypothesis, so a wrong rate
//! misaligns the two sequences and shows up as a collapsed match fraction rather than as a plausible
//! average heart rate. Matching per beat is the claim; matching on rate alone is not evidence.
//!
//! Pulse transit time to the wrist is real and varies beat to beat, so the alignment is searched over a
//! range of offsets rather than assumed, and the middle of the plateau of offsets that all match is
//! reported as the fitted transit time.
//!
//! **A wrong rate that is an integer multiple of the true one still matches, at exactly 1/k.** Reading a
//! stream at half its rate puts every ECG peak on every OTHER optical beat, precisely and with a perfect
//! linear fit; at a third of the rate it lands on every third. Any per-beat gate therefore has to sit
//! above a half, or a doubled rate reads as a partial success rather than as the wrong answer it is.

use crate::ecg::agreement::matched_pairs;
use crate::ecg::{MAX_FS_HZ, MIN_FS_HZ, beat_agreement};
use crate::stats::linear_fit;

/// Offsets searched, in ms. Wider than the plausible band on purpose: an offset found outside it is
/// reported with `offset_plausible` clear rather than hidden, because "the beats line up at 20 ms" is
/// information about a wrong rate and silently refusing to look there would throw it away.
pub const PTT_SEARCH_MIN_MS: f64 = 0.0;
pub const PTT_SEARCH_MAX_MS: f64 = 500.0;
/// Physiological wrist pulse transit time.
pub const PTT_PLAUSIBLE_MIN_MS: f64 = 150.0;
pub const PTT_PLAUSIBLE_MAX_MS: f64 = 300.0;
/// Offset grid step. Finer than the beat-to-beat transit-time variation it is fitting a mean through.
pub const PTT_STEP_MS: f64 = 5.0;
/// Per-beat tolerance around the fitted offset. Transit time varies beat to beat by tens of ms, so a
/// tighter window would reject the truth; a wider one starts matching the neighbouring beat.
pub const PPG_JITTER_MS: f64 = 40.0;
/// Fewest beats on either side for an alignment to mean anything.
pub const PPG_MIN_BEATS: usize = 4;

/// The 1/1024-second conversion the BLE heart-rate characteristic invites. An R-R series that has been
/// through it once is off by this ratio, and so is any rate solved from it.
pub const BLE_RR_SCALE: f64 = 1000.0 / 1024.0;

/// How well a candidate's R peaks line up with the optical beats, at the offset that lines them up best.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PpgAgreement {
    /// Fitted mean pulse transit time (ms): ECG peak plus this lands on the optical beat.
    pub offset_ms: f64,
    /// Whether that offset is inside [`PTT_PLAUSIBLE_MIN_MS`]..=[`PTT_PLAUSIBLE_MAX_MS`].
    pub offset_plausible: bool,
    pub matched: usize,
    pub ecg_beats: usize,
    pub ppg_beats: usize,
    /// `matched / min(ecg_beats, ppg_beats)` — the per-beat claim.
    pub match_fraction: f64,
    /// F1 of the one-to-one match, minus the F1 two independent beat sets reach at these densities.
    /// The raw figure rises with detection density alone, so this is the one to read.
    pub excess: f64,
    pub f1: f64,
    /// `fs = N / (T / 1000)` over every matched pair at once: the least-squares slope of R-peak sample
    /// index against optical beat time, times 1000. `None` on fewer than [`PPG_MIN_BEATS`] matched pairs.
    pub fs_solved_hz: Option<f64>,
    /// Correlation of that fit. A rate solved off a scattered set of pairs is not a rate.
    pub fs_fit_r: Option<f64>,
}

/// Best alignment of `ecg_peaks` (sample indices at `fs_hz`, over a `span_samples` record) against
/// `ppg_beats_ms` (absolute ms from the same record's first sample). `None` when either side is too
/// short or the rate is unusable.
pub fn ppg_agreement(
    ecg_peaks: &[usize],
    span_samples: usize,
    fs_hz: f64,
    ppg_beats_ms: &[f64],
) -> Option<PpgAgreement> {
    if !(MIN_FS_HZ..=MAX_FS_HZ).contains(&fs_hz) || span_samples == 0 {
        return None;
    }
    if ecg_peaks.len() < PPG_MIN_BEATS || ppg_beats_ms.len() < PPG_MIN_BEATS {
        return None;
    }
    let mut scored: Vec<PpgAgreement> = Vec::new();
    let steps = ((PTT_SEARCH_MAX_MS - PTT_SEARCH_MIN_MS) / PTT_STEP_MS).round() as usize;
    for k in 0..=steps {
        let offset = PTT_SEARCH_MIN_MS + PTT_STEP_MS * k as f64;
        // Shift the optical beats back by the transit time, then compare in the ECG's own sample grid.
        let (mut shifted, mut times) = (Vec::new(), Vec::new());
        for &t in ppg_beats_ms {
            if !t.is_finite() || t < offset {
                continue;
            }
            let i = ((t - offset) / 1000.0 * fs_hz).round() as usize;
            if i < span_samples {
                shifted.push(i);
                times.push(t);
            }
        }
        if shifted.len() < PPG_MIN_BEATS {
            continue;
        }
        let g = beat_agreement(ecg_peaks, &shifted, span_samples, fs_hz, PPG_JITTER_MS);
        let denom = ecg_peaks.len().min(shifted.len()) as f64;
        let (solved, fit_r) = solve_rate(ecg_peaks, &shifted, &times, fs_hz);
        let cand = PpgAgreement {
            offset_ms: offset,
            offset_plausible: (PTT_PLAUSIBLE_MIN_MS..=PTT_PLAUSIBLE_MAX_MS).contains(&offset),
            matched: g.matched,
            ecg_beats: ecg_peaks.len(),
            ppg_beats: shifted.len(),
            match_fraction: if denom > 0.0 {
                g.matched as f64 / denom
            } else {
                0.0
            },
            excess: g.excess,
            f1: g.f1,
            fs_solved_hz: solved,
            fs_fit_r: fit_r,
        };
        scored.push(cand);
    }
    // Beat-to-beat transit-time variation makes a PLATEAU of offsets that all match every beat, not one
    // sharp maximum. Taking the first of them would report the earliest offset that happens to work,
    // which is systematically early by half the jitter window; the middle of the plateau is the fitted
    // mean transit time the alignment is actually claiming.
    let best = scored
        .iter()
        .map(|a| a.excess)
        .fold(f64::NEG_INFINITY, f64::max);
    let tied: Vec<&PpgAgreement> = scored.iter().filter(|a| a.excess >= best - 1e-12).collect();
    tied.get(tied.len() / 2).map(|a| **a)
}

/// `fs = N / (T / 1000)` fitted over every matched pair at once: the least-squares slope of R-peak
/// sample index against optical beat time, in samples per ms, times 1000.
///
/// The slope form, not a ratio of two medians. Two beat series detected by different means do not
/// contain exactly the same beats, so their median intervals can differ by a couple of percent on a
/// varying rhythm — which lands squarely inside the 2.4 % window the 1/1024-second check watches, and
/// measured on this corpus it did exactly that. A regression across the whole window is insensitive to a
/// missed or extra beat and tightens with beat count instead.
///
/// This is a REFINEMENT of the candidate rate, not an independent measurement of it: the pairing was
/// made at the candidate rate, so no pairing means no estimate. What makes it evidence is that a wrong
/// candidate fails to pair at all rather than producing a wrong slope.
fn solve_rate(
    ecg_peaks: &[usize],
    shifted: &[usize],
    times_ms: &[f64],
    fs_hz: f64,
) -> (Option<f64>, Option<f64>) {
    let tolerance = (PPG_JITTER_MS / 1000.0 * fs_hz).round() as usize;
    let pairs = matched_pairs(ecg_peaks, shifted, tolerance);
    if pairs.len() < PPG_MIN_BEATS {
        return (None, None);
    }
    let xs: Vec<f64> = pairs.iter().map(|&(_, j)| times_ms[j]).collect();
    let ys: Vec<f64> = pairs.iter().map(|&(i, _)| ecg_peaks[i] as f64).collect();
    match linear_fit(&xs, &ys) {
        Some(fit) if fit.scale > 0.0 => (Some(fit.scale * 1000.0), Some(fit.r)),
        _ => (None, None),
    }
}

/// A rate that `fs_solved` would be if the optical R-R had been through the 1/1024-second conversion
/// once, in either direction, plus the direction it was applied.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UnitErrorSuspicion {
    /// The rate the stream would really be running at.
    pub true_fs_hz: f64,
    /// `true` when the R-R came out LONG (`×1024/1000`), which makes the solved rate low.
    pub rr_reported_long: bool,
}

/// Flag a solved rate that sits one 1/1024-second conversion away from a rate in `rates`.
///
/// The check is two-sided, and the direction matters more than it looks. An R-R reported SHORT by
/// `1000/1024` — the failure this project has actually had — divides into the solve and pushes the rate
/// HIGH: a true 512 Hz solves to 524.3. An R-R reported LONG by `1024/1000` pushes it LOW, and that is
/// the direction in which a true 512 Hz solves to exactly 500.0. Only one of those two is the coincidence
/// that would read as confirmation of an assumed 500 Hz, so testing one side would miss it half the time.
pub fn suspected_unit_error(fs_solved: f64, rates: &[f64], tol: f64) -> Option<UnitErrorSuspicion> {
    if !fs_solved.is_finite() || fs_solved <= 0.0 {
        return None;
    }
    for &r in rates {
        if r <= 0.0 {
            continue;
        }
        if (fs_solved - r * BLE_RR_SCALE).abs() / r <= tol {
            return Some(UnitErrorSuspicion {
                true_fs_hz: r,
                rr_reported_long: true,
            });
        }
        if (fs_solved - r / BLE_RR_SCALE).abs() / r <= tol {
            return Some(UnitErrorSuspicion {
                true_fs_hz: r,
                rr_reported_long: false,
            });
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Beats every `rr_ms` for `n` beats, in ms.
    fn beat_times(n: usize, rr_ms: f64, first_ms: f64) -> Vec<f64> {
        (0..n).map(|k| first_ms + k as f64 * rr_ms).collect()
    }

    #[test]
    fn a_true_rate_aligns_at_the_transit_time_and_a_wrong_one_does_not() {
        let fs = 512.0;
        let rr_ms = 900.0;
        let peaks: Vec<usize> = (0..30)
            .map(|k| (k as f64 * rr_ms / 1000.0 * fs) as usize + 100)
            .collect();
        let span = 30 * (rr_ms / 1000.0 * fs) as usize + 500;
        let ppg = beat_times(30, rr_ms, 100.0 / fs * 1000.0 + 220.0);

        let right = ppg_agreement(&peaks, span, fs, &ppg).unwrap();
        assert!(
            (right.offset_ms - 220.0).abs() <= 2.0 * PTT_STEP_MS,
            "fitted PTT {}",
            right.offset_ms
        );
        assert!(
            right.offset_plausible && right.match_fraction > 0.95,
            "{right:?}"
        );

        // Half the rate maps every ECG peak onto every OTHER optical beat, exactly and with a perfect
        // linear fit, so the match fraction lands on exactly 1/2 rather than collapsing. A factor-of-k
        // rate error scores 1/k, which is why the gate has to sit above a half - see the module note.
        let wrong = ppg_agreement(&peaks, span, fs / 2.0, &ppg).unwrap();
        assert!(
            (wrong.match_fraction - 0.5).abs() < 0.05,
            "half rate matches every other beat: {wrong:?}"
        );
        assert!(wrong.excess < right.excess * 0.75);
    }

    #[test]
    fn the_rate_solves_out_of_the_matched_pairs() {
        let fs = 256.0;
        let rr_ms = 750.0;
        let peaks: Vec<usize> = (0..25)
            .map(|k| (k as f64 * rr_ms / 1000.0 * fs) as usize)
            .collect();
        let ppg = beat_times(25, rr_ms, 200.0);
        let a = ppg_agreement(&peaks, 6000, fs, &ppg).unwrap();
        let solved = a.fs_solved_hz.unwrap();
        assert!((solved - fs).abs() < 1.0, "solved {solved} for a true {fs}");
        assert!(a.fs_fit_r.unwrap() > 0.999);
        // A rhythm that varies beat to beat still solves, because the fit spans the whole window rather
        // than comparing two medians over two different beat sets.
        let varied: Vec<f64> = (0..25)
            .map(|k| {
                200.0
                    + (0..k)
                        .map(|j| rr_ms + 60.0 * ((j % 3) as f64 - 1.0))
                        .sum::<f64>()
            })
            .collect();
        let vpeaks: Vec<usize> = varied
            .iter()
            .map(|t| ((t - 200.0) / 1000.0 * fs).round() as usize)
            .collect();
        let b = ppg_agreement(&vpeaks, 6000, fs, &varied).unwrap();
        assert!(
            (b.fs_solved_hz.unwrap() - fs).abs() < 1.0,
            "{:?}",
            b.fs_solved_hz
        );
    }

    #[test]
    fn the_ten_twenty_four_coincidence_is_caught_from_both_directions() {
        let rates = [128.0, 256.0, 500.0, 512.0, 1024.0];
        // R-R long: a true 512 solves to exactly 500.0 - the value already assumed, so it would read as
        // confirmation rather than as an error.
        let low = 512.0 * BLE_RR_SCALE;
        assert!((low - 500.0).abs() < 1e-9);
        let s = suspected_unit_error(low, &rates, 0.01).unwrap();
        assert_eq!((s.true_fs_hz, s.rr_reported_long), (512.0, true));
        // R-R short: the same true 512 solves high instead.
        let high = 512.0 / BLE_RR_SCALE;
        assert!((high - 524.288).abs() < 1e-3);
        let s = suspected_unit_error(high, &rates, 0.01).unwrap();
        assert_eq!((s.true_fs_hz, s.rr_reported_long), (512.0, false));
        // A rate that is simply itself is not suspicious.
        assert!(suspected_unit_error(256.0, &[256.0], 0.005).is_none());
        assert!(suspected_unit_error(f64::NAN, &rates, 0.01).is_none());
    }

    #[test]
    fn degenerate_inputs_are_none_not_a_panic() {
        let peaks: Vec<usize> = (0..20).map(|k| k * 100).collect();
        let ppg = beat_times(20, 500.0, 200.0);
        assert!(ppg_agreement(&peaks, 4000, 5.0, &ppg).is_none());
        assert!(ppg_agreement(&peaks, 0, 200.0, &ppg).is_none());
        assert!(ppg_agreement(&peaks[..2], 4000, 200.0, &ppg).is_none());
        assert!(ppg_agreement(&peaks, 4000, 200.0, &[]).is_none());
        // Non-finite beat times are dropped, not propagated.
        let mut dirty = ppg.clone();
        dirty[3] = f64::NAN;
        assert!(ppg_agreement(&peaks, 4000, 200.0, &dirty).is_some());
    }
}
