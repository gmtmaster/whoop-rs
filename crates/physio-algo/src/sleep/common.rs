//! Shared numeric helpers for the V2 sleep stager: population std, a median, a per-night z-scorer, and the
//! R-R run flattener. Kept private to the `sleep` module.

use super::input::RrRun;

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

/// Flatten grouped R-R runs into `(ts, rr_ms)` pairs in emission order — the shape the V2 stager buckets by
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
