//! Shared numeric helpers for the sleep stagers. Kept private to the `sleep` module; each mirrors the
//! exact primitive the platform twins use (population/sample std, numpy-style percentile, a Gaussian
//! kernel, reflect-padded convolution, a scipy-style peak finder, and the R-R range filter).

use super::input::RrRun;
use super::SleepStage;

/// R-R keep-band (ms): intervals outside are dropouts/ectopics.
pub(super) const RR_MIN_MS: f64 = 300.0;
pub(super) const RR_MAX_MS: f64 = 2000.0;

/// Population standard deviation (divide by n). Empty → 0.
pub(super) fn population_std(vals: &[f64]) -> f64 {
    if vals.is_empty() {
        return 0.0;
    }
    let mean = vals.iter().sum::<f64>() / vals.len() as f64;
    let var = vals.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / vals.len() as f64;
    if var < 0.0 {
        0.0
    } else {
        var.sqrt()
    }
}

/// Task-Force RMSSD over already-clean intervals (sample denominator n-1). `None` for < 2 values.
pub(super) fn rmssd_raw(nn: &[f64]) -> Option<f64> {
    if nn.len() < 2 {
        return None;
    }
    let mut sum_sq = 0.0;
    for i in 1..nn.len() {
        let d = nn[i] - nn[i - 1];
        sum_sq += d * d;
    }
    Some((sum_sq / (nn.len() - 1) as f64).sqrt())
}

/// Keep only intervals inside [RR_MIN_MS, RR_MAX_MS], order preserved.
pub(super) fn range_filter(rr: &[f64]) -> Vec<f64> {
    rr.iter().copied().filter(|&v| (RR_MIN_MS..=RR_MAX_MS).contains(&v)).collect()
}

/// Median of a slice (sorts a copy). Empty → 0.
pub(super) fn median(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut s = values.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = s.len();
    if n % 2 == 1 {
        s[n / 2]
    } else {
        0.5 * (s[n / 2 - 1] + s[n / 2])
    }
}

/// numpy-style linear-interpolated percentile over the finite values (`pct` in 0..100). `None` if none.
pub(super) fn percentile(values: &[f64], pct: f64) -> Option<f64> {
    let mut vals: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    if vals.is_empty() {
        return None;
    }
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = vals.len();
    if n == 1 {
        return Some(vals[0]);
    }
    let position = (pct / 100.0) * (n - 1) as f64;
    let lower = position.floor() as usize;
    let upper = (lower + 1).min(n - 1);
    let frac = position - lower as f64;
    Some(vals[lower] + frac * (vals[upper] - vals[lower]))
}

/// Normalised Gaussian kernel with σ given in seconds, sampled on the 30 s epoch grid.
pub(super) fn gaussian_kernel(sigma_s: f64, dt_s: f64) -> Vec<f64> {
    let sigma = (sigma_s / dt_s).max(1e-6);
    let radius = ((3.0 * sigma).ceil() as i64).max(1);
    let mut k: Vec<f64> = Vec::with_capacity((2 * radius + 1) as usize);
    for x in -radius..=radius {
        k.push((-0.5 * (x as f64 / sigma).powi(2)).exp());
    }
    let sum: f64 = k.iter().sum();
    k.iter().map(|v| v / sum).collect()
}

/// Same-length convolution with numpy 'reflect' padding (mirror without repeating the edge sample).
pub(super) fn convolve_reflect(x: &[f64], kernel: &[f64]) -> Vec<f64> {
    let r = kernel.len() / 2;
    if r == 0 || x.len() <= r {
        return x.to_vec();
    }
    let n = x.len();
    let mut padded: Vec<f64> = Vec::with_capacity(n + 2 * r);
    for i in 0..r {
        padded.push(x[r - i]);
    }
    padded.extend_from_slice(x);
    for i in 0..r {
        padded.push(x[n - 2 - i]);
    }
    let m = kernel.len();
    let mut out: Vec<f64> = Vec::with_capacity(n);
    for i in 0..=(padded.len() - m) {
        let mut acc = 0.0;
        for j in 0..m {
            acc += padded[i + j] * kernel[m - 1 - j];
        }
        out.push(acc);
        if out.len() == n {
            break;
        }
    }
    out
}

/// Local-maxima peak finder mirroring `scipy.find_peaks(distance, height)`: a strict local max ≥ height,
/// plateaus resolved to their midpoint, then a greedy tallest-first minimum-distance prune.
pub(super) fn find_peaks(x: &[f64], distance: i64, height: f64) -> Vec<usize> {
    let n = x.len();
    if n < 3 {
        return Vec::new();
    }
    let mut candidates: Vec<usize> = Vec::new();
    let mut i = 1usize;
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
        let p = candidates[pi] as i64;
        for qi in 0..candidates.len() {
            if qi != pi && keep[qi] && (candidates[qi] as i64 - p).abs() < distance {
                keep[qi] = false;
            }
        }
    }
    candidates.iter().enumerate().filter(|(off, _)| keep[*off]).map(|(_, &c)| c).collect()
}

/// A per-night z-scorer over present values: population std, with a flat channel (0 std → 1) neutral,
/// and a missing value scoring the neutral centre 0.
pub(super) struct ZScore {
    mean: f64,
    sd: f64,
    empty: bool,
}

impl ZScore {
    pub(super) fn build(vals: &[Option<f64>]) -> Self {
        let present: Vec<f64> = vals.iter().filter_map(|v| *v).collect();
        if present.is_empty() {
            return ZScore { mean: 0.0, sd: 1.0, empty: true };
        }
        let mean = present.iter().sum::<f64>() / present.len() as f64;
        let sd0 = population_std(&present);
        let sd = if sd0 == 0.0 { 1.0 } else { sd0 };
        ZScore { mean, sd, empty: false }
    }

    pub(super) fn apply(&self, value: Option<f64>) -> f64 {
        match value {
            _ if self.empty => 0.0,
            None => 0.0,
            Some(v) => (v - self.mean) / self.sd,
        }
    }
}

/// Flatten grouped R-R runs into `(ts, rr_ms)` pairs in emission order — the shape both stagers bucket by
/// second. A run reports several beats under one whole-second anchor.
pub(super) fn flatten_rr(runs: &[RrRun]) -> Vec<(i64, f64)> {
    let mut out = Vec::new();
    for run in runs {
        for &ms in &run.intervals {
            out.push((run.ts, ms as f64));
        }
    }
    out
}

/// Sleep-depth rank (lighter → deeper) for the fragment-merge lighter-bias tie-break.
pub(super) fn stage_depth_rank(stage: SleepStage) -> i32 {
    match stage {
        SleepStage::Light => 1,
        SleepStage::Rem => 2,
        SleepStage::Deep => 3,
        SleepStage::Wake => 0,
    }
}
