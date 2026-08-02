//! Sample-domain signal primitives shared by the waveform metrics: a plateau-aware peak finder, a
//! zero-phase centred moving average (edge-truncated or reflect-padded), a running median, and a
//! MAD-based robust sigma. Consumed by `respiratory_rate`, `ppg` and by both `ecg` QRS detectors, so a
//! threshold means the same thing in all of them.

use crate::stats::median;

/// Local-maxima peak finder: a plateau-aware maximum at or above `height`, with peaks closer than
/// `distance` samples resolved by keeping the taller. `distance <= 1` suppresses nothing, which is what
/// a caller with its own refractory rule wants.
pub fn find_peaks(x: &[f64], distance: usize, height: f64) -> Vec<usize> {
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

/// Zero-phase moving average over a centred window of `len` samples (odd or even; the half-width is
/// `len / 2` each side). Edge windows are truncated and divided by their own count, so no padding value
/// is invented. `len <= 1` or an empty input returns the input unchanged.
pub fn moving_average_centred(x: &[f64], len: usize) -> Vec<f64> {
    let n = x.len();
    if n == 0 || len <= 1 {
        return x.to_vec();
    }
    let half = (len / 2) as isize;
    let mut prefix = vec![0.0f64; n + 1];
    for i in 0..n {
        prefix[i + 1] = prefix[i] + x[i];
    }
    (0..n)
        .map(|i| {
            let lo = (i as isize - half).max(0) as usize;
            let hi = ((i as isize + half) as usize).min(n - 1);
            (prefix[hi + 1] - prefix[lo]) / (hi - lo + 1) as f64
        })
        .collect()
}

/// Mirror an out-of-range index back inside `0..n` without repeating the edge sample (`-1` → `1`).
/// The padding rule behind [`moving_average_reflect`] and [`median_filter_reflect`].
fn reflect_index(i: isize, n: usize) -> usize {
    if n <= 1 {
        return 0;
    }
    let period = 2 * (n as isize - 1);
    let k = i.rem_euclid(period);
    (if k >= n as isize { period - k } else { k }) as usize
}

/// `x` extended by `half` reflected samples each side, so a centred window is always full width.
fn reflect_pad(x: &[f64], half: usize) -> Vec<f64> {
    (0..x.len() + 2 * half)
        .map(|i| x[reflect_index(i as isize - half as isize, x.len())])
        .collect()
}

/// Zero-phase moving average like [`moving_average_centred`], but edge windows are filled by the
/// mirror of the signal instead of being truncated. A detrend needs this: a truncated edge mean is
/// pulled toward the sample itself and a zero-pad toward zero, both of which bend the trend at the ends.
pub fn moving_average_reflect(x: &[f64], len: usize) -> Vec<f64> {
    let n = x.len();
    if n == 0 || len <= 1 {
        return x.to_vec();
    }
    let half = len / 2;
    let padded = reflect_pad(x, half);
    let mut prefix = vec![0.0f64; padded.len() + 1];
    for i in 0..padded.len() {
        prefix[i + 1] = prefix[i] + padded[i];
    }
    let w = 2 * half + 1;
    (0..n).map(|i| (prefix[i + w] - prefix[i]) / w as f64).collect()
}

/// Running median over a centred, reflect-padded window of `len` samples. A despiker: it deletes a
/// one-sample excursion outright where a mean would smear a fraction of it into every neighbour.
pub fn median_filter_reflect(x: &[f64], len: usize) -> Vec<f64> {
    let n = x.len();
    if n == 0 || len <= 1 {
        return x.to_vec();
    }
    let half = len / 2;
    let w = 2 * half + 1;
    let padded = reflect_pad(x, half);
    (0..n).map(|i| median(&padded[i..i + w])).collect()
}

/// Robust scale estimate: median absolute deviation from the median, scaled by 1.4826 so it matches the
/// standard deviation on gaussian data. Unlike an SD it is not inflated by the QRS complexes themselves,
/// which is why a wavelet threshold is set from it. `0.0` when empty or when over half the samples are equal.
pub fn robust_sigma(x: &[f64]) -> f64 {
    if x.is_empty() {
        return 0.0;
    }
    let m = median(x);
    let dev: Vec<f64> = x.iter().map(|v| (v - m).abs()).collect();
    1.4826 * median(&dev)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn find_peaks_is_plateau_aware_and_honours_distance() {
        // Two peaks 2 apart: distance 1 keeps both, distance 3 keeps only the taller.
        let x = [0.0, 5.0, 0.0, 9.0, 0.0];
        assert_eq!(find_peaks(&x, 1, f64::NEG_INFINITY), vec![1, 3]);
        assert_eq!(find_peaks(&x, 3, f64::NEG_INFINITY), vec![3]);
        // A flat top reports its centre once, not twice.
        assert_eq!(find_peaks(&[0.0, 4.0, 4.0, 0.0], 1, f64::NEG_INFINITY), vec![1]);
        // Height gate rejects.
        assert!(find_peaks(&x, 1, 6.0).contains(&3) && !find_peaks(&x, 1, 6.0).contains(&1));
        // Degenerate inputs are empty, never a panic.
        assert!(find_peaks(&[], 1, 0.0).is_empty());
        assert!(find_peaks(&[1.0, 1.0, 1.0], 1, 0.0).is_empty());
    }

    #[test]
    fn moving_average_is_centred_and_edge_truncated() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = moving_average_centred(&x, 3);
        assert!((y[0] - 1.5).abs() < 1e-12); // truncated: (1+2)/2
        assert!((y[2] - 3.0).abs() < 1e-12); // centred: (2+3+4)/3
        assert!((y[4] - 4.5).abs() < 1e-12);
        assert_eq!(moving_average_centred(&x, 1), x.to_vec());
        assert!(moving_average_centred(&[], 5).is_empty());
    }

    #[test]
    fn reflect_padding_mirrors_without_repeating_the_edge() {
        // Index -1 mirrors to 1 and -2 to 2; past the right edge mirrors the same way.
        assert_eq!(reflect_index(-1, 5), 1);
        assert_eq!(reflect_index(-2, 5), 2);
        assert_eq!(reflect_index(5, 5), 3);
        assert_eq!(reflect_index(0, 1), 0); // degenerate length never panics
        assert_eq!(reflect_pad(&[1.0, 2.0, 3.0], 2), vec![3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0]);
    }

    #[test]
    fn reflect_moving_average_holds_a_ramp_where_truncation_bends_it() {
        // On a straight ramp the reflected mean is NOT the ramp — mirroring folds the line back — but it
        // is symmetric, so subtracting it leaves a symmetric residual instead of the truncated version's
        // one-sided edge droop. Interior samples agree with the truncated form exactly.
        let x: Vec<f64> = (0..9).map(|i| i as f64).collect();
        let r = moving_average_reflect(&x, 3);
        let t = moving_average_centred(&x, 3);
        assert!((r[4] - 4.0).abs() < 1e-12 && (t[4] - 4.0).abs() < 1e-12);
        assert!((r[0] - 2.0 / 3.0).abs() < 1e-12); // (1 + 0 + 1) / 3
        assert!((t[0] - 0.5).abs() < 1e-12); // truncated: (0 + 1) / 2
        assert!((r[0] - x[0] + (r[8] - x[8])).abs() < 1e-12); // ends err by equal and opposite amounts
        assert_eq!(moving_average_reflect(&x, 1), x);
        assert!(moving_average_reflect(&[], 5).is_empty());
        assert_eq!(moving_average_reflect(&[7.0], 5), vec![7.0]);
    }

    #[test]
    fn median_filter_deletes_a_lone_spike_a_mean_would_smear() {
        let mut x = vec![1.0; 9];
        x[4] = 100.0;
        let med = median_filter_reflect(&x, 3);
        assert!(med.iter().all(|v| (*v - 1.0).abs() < 1e-12), "spike survived: {med:?}");
        // The mean leaves a third of the spike in all three samples it touches.
        let avg = moving_average_reflect(&x, 3);
        assert!((avg[4] - 34.0).abs() < 1e-12 && (avg[3] - 34.0).abs() < 1e-12);
        assert_eq!(median_filter_reflect(&x, 1), x);
        assert!(median_filter_reflect(&[], 3).is_empty());
    }

    #[test]
    fn robust_sigma_ignores_a_few_huge_outliers() {
        let mut x = vec![0.0; 100];
        for (i, v) in x.iter_mut().enumerate() {
            *v = if i % 2 == 0 { -1.0 } else { 1.0 };
        }
        let clean = robust_sigma(&x);
        x[0] = 10_000.0;
        x[1] = -10_000.0;
        // Two extreme samples in 100 move the MAD not at all; they move an SD by orders of magnitude.
        assert!((robust_sigma(&x) - clean).abs() < 1e-9);
        assert_eq!(robust_sigma(&[]), 0.0);
        assert_eq!(robust_sigma(&[7.0, 7.0, 7.0]), 0.0);
    }
}
