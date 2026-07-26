//! Approximate sleeping respiratory rate (breaths/min) from the R-R interval stream via respiratory
//! sinus arrhythmia: reconstruct a beat-interval tachogram, resample to 4 Hz, detrend, then per 5-min
//! window peak-pick the breathing modulation and take the median rate. Pure; `None` = no usable estimate.

use crate::stats::{median, population_sd};

const RR_MIN_MS: f64 = 300.0;
const RR_MAX_MS: f64 = 2000.0;

const RSA_RESAMPLE_HZ: f64 = 4.0;
const RSA_DETREND_WINDOW_S: f64 = 8.0;
const RSA_MIN_PEAK_DISTANCE_S: f64 = 2.5;
const RSA_WINDOW_S: f64 = 300.0;
const RSA_MIN_BREATH_INTERVAL_S: f64 = 2.5;
const RSA_MAX_BREATH_INTERVAL_S: f64 = 10.0;

/// Plausible sleeping-respiratory-rate band (bpm); an estimate outside it is `None`.
pub const RESP_PLAUSIBLE_MIN_BPM: f64 = 8.0;
pub const RESP_PLAUSIBLE_MAX_BPM: f64 = 25.0;

/// Sleeping respiratory rate over the in-bed `[start, end]` window (unix seconds), from `(ts, rr_ms)`
/// beats. An on-device wellness estimate, never a clinical measurement; `None` on too-little data.
pub fn resp_rate_from_rr(rr: &[(i64, u16)], start: i64, end: i64) -> Option<f64> {
    if end <= start {
        return None;
    }

    let mut in_bed: Vec<(i64, f64)> = rr
        .iter()
        .filter(|(ts, _)| *ts >= start && *ts <= end)
        .map(|(ts, ms)| (*ts, *ms as f64))
        .collect();
    in_bed.sort_by_key(|(ts, _)| *ts);
    let filtered: Vec<f64> = in_bed
        .into_iter()
        .map(|(_, ms)| ms)
        .filter(|ms| *ms >= RR_MIN_MS && *ms <= RR_MAX_MS)
        .collect();
    if filtered.len() < 30 {
        return None;
    }

    let mut beat_times = vec![0.0; filtered.len()];
    let mut acc = 0.0;
    for (i, &ms) in filtered.iter().enumerate() {
        acc += ms / 1000.0;
        beat_times[i] = acc;
    }
    let total_span_s = beat_times[beat_times.len() - 1];
    if total_span_s < RSA_WINDOW_S / 2.0 {
        return None;
    }

    let dt = 1.0 / RSA_RESAMPLE_HZ;
    let n_grid = (total_span_s / dt) as usize + 1;
    if n_grid < 8 {
        return None;
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
            let frac = ((t - t0) / (t1 - t0)).clamp(0.0, 1.0);
            v0 + frac * (v1 - v0)
        };
    }

    let half_w = ((RSA_DETREND_WINDOW_S * RSA_RESAMPLE_HZ / 2.0).round() as usize).max(1);
    let mut detrended = vec![0.0; n_grid];
    for i in 0..n_grid {
        let lo = i.saturating_sub(half_w);
        let hi = (i + half_w).min(n_grid - 1);
        let sum: f64 = grid[lo..=hi].iter().sum();
        detrended[i] = grid[i] - sum / (hi - lo + 1) as f64;
    }
    if population_sd(&detrended) <= 1e-9 {
        return None;
    }

    let min_dist = ((RSA_MIN_PEAK_DISTANCE_S * RSA_RESAMPLE_HZ).round() as usize).max(2);
    let window_samples = ((RSA_WINDOW_S * RSA_RESAMPLE_HZ).round() as usize).max(min_dist * 3);
    let mut per_window = Vec::new();
    let mut w = 0usize;
    while w < n_grid {
        let w_end = (w + window_samples).min(n_grid);
        if w_end - w >= min_dist * 3 {
            let peaks = find_peaks(&detrended[w..w_end], min_dist, 0.0);
            if peaks.len() >= 3 {
                let mut intervals = Vec::with_capacity(peaks.len() - 1);
                for k in 1..peaks.len() {
                    let iv_s = (peaks[k] - peaks[k - 1]) as f64 * dt;
                    if (RSA_MIN_BREATH_INTERVAL_S..=RSA_MAX_BREATH_INTERVAL_S).contains(&iv_s) {
                        intervals.push(iv_s);
                    }
                }
                if intervals.len() >= 2 {
                    let med = median(&intervals);
                    if med > 0.0 {
                        per_window.push(60.0 / med);
                    }
                }
            }
        }
        w += window_samples;
    }
    if per_window.is_empty() {
        return None;
    }
    let m = median(&per_window);
    (RESP_PLAUSIBLE_MIN_BPM..=RESP_PLAUSIBLE_MAX_BPM).contains(&m).then_some(m)
}

/// Local-maxima peak finder: a plateau-aware maximum at or above `height`, with peaks closer than
/// `distance` samples resolved by keeping the taller.
fn find_peaks(x: &[f64], distance: usize, height: f64) -> Vec<usize> {
    let n = x.len();
    if n < 3 {
        return Vec::new();
    }
    let mut candidates = Vec::new();
    let mut i = 1;
    while i < n - 1 {
        if x[i] > x[i - 1] && x[i] >= height {
            let mut j = i;
            while j + 1 < n && x[j + 1] == x[i] {
                j += 1;
            }
            if j + 1 < n && x[j + 1] < x[i] {
                candidates.push((i + j) / 2);
            }
            i = j + 1;
        } else {
            i += 1;
        }
    }
    if distance <= 1 || candidates.is_empty() {
        return candidates;
    }
    let mut by_height: Vec<usize> = (0..candidates.len()).collect();
    by_height.sort_by(|&a, &b| x[candidates[b]].partial_cmp(&x[candidates[a]]).unwrap());
    let mut keep = vec![true; candidates.len()];
    for &pi in &by_height {
        if !keep[pi] {
            continue;
        }
        let p = candidates[pi] as isize;
        for qi in 0..candidates.len() {
            if qi != pi && keep[qi] && (candidates[qi] as isize - p).unsigned_abs() < distance {
                keep[qi] = false;
            }
        }
    }
    candidates
        .iter()
        .enumerate()
        .filter(|(off, _)| keep[*off])
        .map(|(_, &c)| c)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Synthetic tachogram: mean HR with a known-Hz RSA modulation, so the recovered rate can be
    /// cross-checked against the planted breathing frequency.
    fn synth(breath_hz: f64, base_rr_ms: f64, amp_ms: f64, span_s: f64) -> (Vec<(i64, u16)>, i64, i64) {
        let start = 1_700_000_000_i64;
        let mut rows = Vec::new();
        let mut t_sec = 0.0_f64;
        while t_sec < span_s {
            let rr_ms = base_rr_ms + amp_ms * (2.0 * std::f64::consts::PI * breath_hz * t_sec).sin();
            t_sec += rr_ms / 1000.0;
            rows.push((start + t_sec as i64, rr_ms as u16));
        }
        let end = start + t_sec as i64;
        (rows, start, end)
    }

    #[test]
    fn recovers_known_breathing_frequency() {
        // 15 breaths/min (0.25 Hz), mean HR 60, +/-40 ms RSA, ~7 min.
        let (rows, start, end) = synth(0.25, 1000.0, 40.0, 420.0);
        let est = resp_rate_from_rr(&rows, start, end).expect("finite estimate");
        assert!((est - 15.0).abs() <= 3.0, "expected ~15, got {est}");
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
}
