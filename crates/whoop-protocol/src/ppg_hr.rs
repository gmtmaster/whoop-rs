//! Derive heart rate from the v26 optical PPG waveform (24 Hz), which carries no per-second HR: group
//! samples into consecutive-second runs, detrend a centred window, remove the record-rate comb, then
//! pick the fundamental autocorrelation period. Pure.

use std::collections::{HashMap, HashSet};

pub const SAMPLE_RATE_HZ: usize = 24;
pub const WINDOW_SECONDS: usize = 8;
pub const MIN_BPM: f64 = 30.0;
pub const MAX_BPM: f64 = 220.0;
pub const MIN_CONFIDENCE: f64 = 0.3;

/// One concatenated PPG sample: its wall-clock second and raw ADC value.
#[derive(Clone, Copy, Debug)]
pub struct Sample {
    pub ts: i64,
    pub value: i32,
}

/// A derived HR estimate at the window-centre second.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Estimate {
    pub ts: i64,
    pub bpm: i32,
    pub conf: f64,
}

/// Per-second PPG-HR over the concatenated samples (one estimate per confident second, ascending).
pub fn estimate(samples: &[Sample]) -> Vec<Estimate> {
    if samples.is_empty() {
        return Vec::new();
    }
    let mut secs: HashMap<i64, Vec<f64>> = HashMap::new();
    for s in samples {
        secs.entry(s.ts).or_default().push(s.value as f64);
    }
    let mut order: Vec<i64> = secs.keys().copied().collect();
    order.sort_unstable();

    // Consecutive-second runs (PPG phase is only continuous within a run).
    let mut runs: Vec<Vec<i64>> = Vec::new();
    let mut cur = vec![order[0]];
    for &u in &order[1..] {
        if u - cur[cur.len() - 1] == 1 {
            cur.push(u);
        } else {
            runs.push(std::mem::take(&mut cur));
            cur = vec![u];
        }
    }
    runs.push(cur);

    let half = (WINDOW_SECONDS / 2) as i64;
    let mut out = Vec::new();
    for run in &runs {
        if run.len() < 3 {
            continue;
        }
        let run_set: HashSet<i64> = run.iter().copied().collect();
        for &t in run {
            let mut win = Vec::new();
            let mut u = t - half;
            while u <= t + half {
                if run_set.contains(&u) {
                    win.push(u);
                }
                u += 1;
            }
            if win.len() < 3 {
                continue;
            }
            let mut sig = Vec::new();
            for w in &win {
                if let Some(v) = secs.get(w) {
                    sig.extend_from_slice(v);
                }
            }
            if let Some(e) = estimate_window(&sig, t) {
                out.push(e);
            }
        }
    }
    out.sort_by_key(|e| e.ts);
    out
}

fn detrend(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    if n <= 1 {
        return vec![0.0; n];
    }
    let nd = n as f64;
    let sum_i = nd * (nd - 1.0) / 2.0;
    let sum_i2 = (nd - 1.0) * nd * (2.0 * nd - 1.0) / 6.0;
    let mut sum_y = 0.0;
    let mut sum_iy = 0.0;
    for (i, &v) in x.iter().enumerate() {
        sum_y += v;
        sum_iy += i as f64 * v;
    }
    let denom = nd * sum_i2 - sum_i * sum_i;
    if denom == 0.0 {
        let mean = sum_y / nd;
        return x.iter().map(|v| v - mean).collect();
    }
    let slope = (nd * sum_iy - sum_i * sum_y) / denom;
    let intercept = (sum_y - slope * sum_i) / nd;
    (0..n).map(|i| x[i] - (slope * i as f64 + intercept)).collect()
}

fn acf(x: &[f64], lag: usize) -> f64 {
    if x.len() <= lag {
        return 0.0;
    }
    let n = x.len() - lag;
    let mean = x.iter().sum::<f64>() / x.len() as f64;
    let den: f64 = x.iter().map(|v| (v - mean) * (v - mean)).sum();
    if den == 0.0 {
        return 0.0;
    }
    let mut num = 0.0;
    for i in 0..n {
        num += (x[i] - mean) * (x[i + lag] - mean);
    }
    num / den
}

fn remove_record_rate_component(x: &[f64], fs: usize) -> Vec<f64> {
    let n = x.len();
    if fs <= 1 || n < fs * 4 {
        return x.to_vec();
    }
    let (mut within_sum, mut within_count, mut boundary_sum, mut boundary_count) = (0.0, 0usize, 0.0, 0usize);
    for i in 1..n {
        let d = (x[i] - x[i - 1]).abs();
        if i % fs == 0 {
            boundary_sum += d;
            boundary_count += 1;
        } else {
            within_sum += d;
            within_count += 1;
        }
    }
    if within_count == 0 || boundary_count == 0 {
        return x.to_vec();
    }
    let within = within_sum / within_count as f64;
    let boundary = boundary_sum / boundary_count as f64;
    if within <= 0.0 || boundary <= within * 3.0 {
        return x.to_vec(); // smooth boundary → real pulse → leave it
    }
    let mut col_sum = vec![0.0; fs];
    let mut col_count = vec![0usize; fs];
    for (i, &v) in x.iter().enumerate() {
        col_sum[i % fs] += v;
        col_count[i % fs] += 1;
    }
    let col_mean: Vec<f64> = (0..fs)
        .map(|p| if col_count[p] > 0 { col_sum[p] / col_count[p] as f64 } else { 0.0 })
        .collect();
    (0..n).map(|i| x[i] - col_mean[i % fs]).collect()
}

fn estimate_window(values: &[f64], ts: i64) -> Option<Estimate> {
    if values.len() < SAMPLE_RATE_HZ * 3 {
        return None;
    }
    let x = detrend(&remove_record_rate_component(values, SAMPLE_RATE_HZ));
    let fs = SAMPLE_RATE_HZ as f64;
    let lo_lag = ((fs * 60.0 / MAX_BPM).round() as i64).max(2);
    let hi_lag = ((fs * 60.0 / MIN_BPM).round() as i64).min(x.len() as i64 - 2);
    if hi_lag <= lo_lag {
        return None;
    }

    let mut vals: HashMap<i64, f64> = HashMap::new();
    let mut peak = f64::NEG_INFINITY;
    for lag in lo_lag..=hi_lag {
        let v = acf(&x, lag as usize);
        vals.insert(lag, v);
        if v > peak {
            peak = v;
        }
    }
    if peak < MIN_CONFIDENCE {
        return None;
    }

    // Fundamental-period preference: the smallest-lag local max that is nearly as strong as the peak.
    let mut best_lag: i64 = -1;
    if lo_lag < hi_lag - 1 {
        for lag in (lo_lag + 1)..=(hi_lag - 1) {
            let v = vals[&lag];
            if v >= 0.85 * peak && v >= vals[&(lag - 1)] && v >= vals[&(lag + 1)] {
                best_lag = lag;
                break;
            }
        }
    }
    if best_lag < 0 {
        let mut argmax = lo_lag;
        let mut best = vals[&lo_lag];
        for lag in (lo_lag + 1)..=hi_lag {
            let v = vals[&lag];
            if v > best {
                best = v;
                argmax = lag;
            }
        }
        best_lag = argmax;
    }

    let bpm = (fs * 60.0 / best_lag as f64).round() as i32;
    let conf = (vals[&best_lag] * 1000.0).round() / 1000.0;
    Some(Estimate { ts, bpm, conf })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recovers_bpm_from_a_clean_sine() {
        // 90 bpm = 1.5 Hz → period 16 samples at 24 Hz → lag 16 → exactly 90 bpm.
        let freq = 90.0 / 60.0;
        let mut samples = Vec::new();
        for sec in 0..9i64 {
            for i in 0..SAMPLE_RATE_HZ {
                let n = (sec as usize * SAMPLE_RATE_HZ + i) as f64;
                let v = (1000.0 * (2.0 * std::f64::consts::PI * freq * n / SAMPLE_RATE_HZ as f64).sin()) as i32;
                samples.push(Sample { ts: sec, value: v });
            }
        }
        let est = estimate(&samples);
        assert!(!est.is_empty());
        assert!(est.iter().any(|e| (86..=94).contains(&e.bpm)), "got {est:?}");
    }
}
