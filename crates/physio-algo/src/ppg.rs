//! Derive heart rate from the v26 optical PPG waveform (24 Hz), which carries no per-second HR: group
//! samples into consecutive-second runs, remove the record-rate comb and detrend each run once, then per
//! centred window pick the fundamental autocorrelation period and sub-lag parabolic-refine it. Also
//! reduces the per-second result to coarser buckets and to a span-level trust verdict. Pure.

use crate::signal::{median_filter_reflect, moving_average_reflect};
use crate::stats::mean;
use std::collections::{BTreeMap, HashMap, HashSet};

pub const SAMPLE_RATE_HZ: usize = 24;
pub const WINDOW_SECONDS: usize = 8;
pub const MIN_BPM: f64 = 30.0;
pub const MAX_BPM: f64 = 220.0;
pub const MIN_CONFIDENCE: f64 = 0.3;

/// Trend cascade windows (s): a 2 s then 3 s box mean is a linear-phase lowpass whose stopband starts
/// below the 0.5 Hz slowest pulse [`MIN_BPM`] admits, so subtracting it removes breathing and wrist-shift
/// wander while leaving every representable heart rate intact.
const TREND_SHORT_SECONDS: f64 = 2.0;
const TREND_LONG_SECONDS: f64 = 3.0;
/// Despike window (samples): the shortest median that can delete a lone optical excursion.
const DESPIKE_SAMPLES: usize = 3;

/// Confidence a second must clear to count as clean in [`signal_check`]. Above the [`MIN_CONFIDENCE`]
/// emission gate on purpose: that gate only asks for a periodicity, this asks for a clear one.
pub const GOOD_CONFIDENCE: f64 = 0.5;
/// Clean-second fractions separating the three [`SignalCheck`] levels. Cleared by well over half a span
/// of clean wrist PPG; a span with under a fifth is a motion artefact, not a heart rate.
pub const SIGNAL_CHECK_FAIR_FRACTION: f64 = 0.20;
pub const SIGNAL_CHECK_GOOD_FRACTION: f64 = 0.50;

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
        // Comb-remove and detrend the whole run ONCE, then slice windows out of the result. Filtering
        // each 9 s window on its own would fit a trend to 9 s of data and bend it at both ends of every
        // window, and every second sits near the edge of some window.
        let mut sig = Vec::new();
        let mut span: HashMap<i64, (usize, usize)> = HashMap::new();
        for &u in run {
            let v = &secs[&u];
            span.insert(u, (sig.len(), v.len()));
            sig.extend_from_slice(v);
        }
        let filtered = detrend(&remove_record_rate_component(&sig, SAMPLE_RATE_HZ));

        let (first, last) = (run[0], run[run.len() - 1]);
        for &t in run {
            // The run is contiguous, so the window's seconds are a contiguous slice of `filtered`.
            let (lo, hi) = ((t - half).max(first), (t + half).min(last));
            if hi - lo + 1 < 3 {
                continue;
            }
            let (start, _) = span[&lo];
            let (last_start, last_len) = span[&hi];
            if let Some(e) = estimate_window(&filtered[start..last_start + last_len], t) {
                out.push(e);
            }
        }
    }
    out.sort_by_key(|e| e.ts);
    out
}

/// Confidence-weighted downsample of [`estimate`] to `bucket_secs` buckets: each bucket's bpm is
/// `Σ(bpm·conf) / Σ(conf)`, stamped at the bucket start and carrying its mean conf. A plain mean lets one
/// motion-corrupted second drag a bucket as hard as a clean one. Buckets tumble from the unix epoch.
pub fn aggregate(est: &[Estimate], bucket_secs: i64) -> Vec<Estimate> {
    if bucket_secs <= 1 {
        return est.to_vec();
    }
    // Per bucket: Σ(bpm·conf), Σconf, Σbpm, count. The last two only serve the zero-weight fallback.
    let mut buckets: BTreeMap<i64, (f64, f64, f64, usize)> = BTreeMap::new();
    for e in est {
        let key = e.ts.div_euclid(bucket_secs) * bucket_secs;
        let w = e.conf.max(0.0);
        let slot = buckets.entry(key).or_insert((0.0, 0.0, 0.0, 0));
        slot.0 += e.bpm as f64 * w;
        slot.1 += w;
        slot.2 += e.bpm as f64;
        slot.3 += 1;
    }
    buckets
        .into_iter()
        .map(|(ts, (weighted, conf_sum, bpm_sum, n))| {
            // Every conf zero or negative → no usable weight, fall back to a plain mean.
            let bpm = if conf_sum > 0.0 { weighted / conf_sum } else { bpm_sum / n as f64 };
            let conf = conf_sum / n as f64;
            Estimate { ts, bpm: bpm.round() as i32, conf: (conf * 1000.0).round() / 1000.0 }
        })
        .collect()
}

/// Re-weight per-second estimates by the strap's own optical quality flag: every second in `poor_secs`
/// (a v18 record whose `optical_signal_poor` is set) drops to zero confidence, so it stops voting in
/// [`aggregate`] and stops counting clean in [`signal_check`]. Every other second is returned unchanged.
pub fn derate_poor_seconds(est: &[Estimate], poor_secs: &HashSet<i64>) -> Vec<Estimate> {
    est.iter()
        .map(|e| if poor_secs.contains(&e.ts) { Estimate { conf: 0.0, ..*e } } else { *e })
        .collect()
}

/// How far a whole span's PPG-derived HR can be trusted, coarsest first.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SignalCheck {
    Poor,
    Fair,
    Good,
}

/// Span-level verdict from the fraction of its seconds carrying a clean estimate: `[start, end]` inclusive
/// is the denominator, so a second the estimator refused counts against the span exactly like a
/// low-confidence one. An empty or inverted span is [`SignalCheck::Poor`].
pub fn signal_check(est: &[Estimate], start: i64, end: i64) -> SignalCheck {
    if end < start {
        return SignalCheck::Poor;
    }
    let span = (end - start + 1) as f64;
    let clean: HashSet<i64> = est
        .iter()
        .filter(|e| e.ts >= start && e.ts <= end && e.conf > GOOD_CONFIDENCE)
        .map(|e| e.ts)
        .collect();
    let frac = clean.len() as f64 / span;
    if frac > SIGNAL_CHECK_GOOD_FRACTION {
        SignalCheck::Good
    } else if frac > SIGNAL_CHECK_FAIR_FRACTION {
        SignalCheck::Fair
    } else {
        SignalCheck::Poor
    }
}

/// Highpass by subtracting a cascaded 2 s + 3 s box mean, then median-3 despike. A straight line cannot
/// follow baseline wander within one window and leaves it in the autocorrelation; the cascade can, and
/// the median drops a single-sample spike the ACF would otherwise read as signal.
fn detrend(x: &[f64]) -> Vec<f64> {
    let win = |secs: f64| (secs * SAMPLE_RATE_HZ as f64).round().max(1.0) as usize;
    let short = moving_average_reflect(x, win(TREND_SHORT_SECONDS));
    let trend = moving_average_reflect(&short, win(TREND_LONG_SECONDS));
    let high: Vec<f64> = x.iter().zip(&trend).map(|(v, t)| v - t).collect();
    median_filter_reflect(&high, DESPIKE_SAMPLES)
}

fn acf(x: &[f64], lag: usize) -> f64 {
    if x.len() <= lag {
        return 0.0;
    }
    let n = x.len() - lag;
    let m = mean(x);
    let den: f64 = x.iter().map(|v| (v - m) * (v - m)).sum();
    if den == 0.0 {
        return 0.0;
    }
    let mut num = 0.0;
    for i in 0..n {
        num += (x[i] - m) * (x[i + lag] - m);
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

/// One window of the ALREADY comb-removed and detrended run: pick the fundamental ACF period and refine it.
fn estimate_window(x: &[f64], ts: i64) -> Option<Estimate> {
    if x.len() < SAMPLE_RATE_HZ * 3 {
        return None;
    }
    let fs = SAMPLE_RATE_HZ as f64;
    let lo_lag = ((fs * 60.0 / MAX_BPM).round() as i64).max(2);
    let hi_lag = ((fs * 60.0 / MIN_BPM).round() as i64).min(x.len() as i64 - 2);
    if hi_lag <= lo_lag {
        return None;
    }

    let mut vals: HashMap<i64, f64> = HashMap::new();
    let mut peak = f64::NEG_INFINITY;
    for lag in lo_lag..=hi_lag {
        let v = acf(x, lag as usize);
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

    // Sub-lag parabolic refine: fit a parabola to the ACF peak and its two neighbours so a true HR
    // between two integer lags is not quantized (integer steps reach ~16 bpm near 150). Interior lags
    // only; a non-concave fit falls back to the integer lag. conf stays the integer peak.
    let mut refined = best_lag as f64;
    if best_lag > lo_lag && best_lag < hi_lag {
        let (y0, y1, y2) = (vals[&(best_lag - 1)], vals[&best_lag], vals[&(best_lag + 1)]);
        let denom = y0 - 2.0 * y1 + y2;
        if denom < 0.0 {
            let delta = (0.5 * (y0 - y2) / denom).clamp(-1.0, 1.0);
            refined = (best_lag as f64 + delta).clamp(lo_lag as f64, hi_lag as f64);
        }
    }
    let bpm = (fs * 60.0 / refined).round() as i32;
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

    /// 90 bpm pulse under baseline wander 10x its amplitude at 0.06 and 0.13 Hz — the shape a real v26
    /// window carries. A straight line cannot follow a curve, so the wander survives into the ACF, which
    /// then falls monotonically and pins the shortest lag: the OLS detrend returns 206 bpm for 33 of these
    /// 40 seconds. The box cascade follows the wander and every second comes back at the true rate.
    #[test]
    fn baseline_wander_does_not_move_the_recovered_rate() {
        let freq = 90.0 / 60.0;
        let mut samples = Vec::new();
        for sec in 0..40i64 {
            for i in 0..SAMPLE_RATE_HZ {
                let t = sec as f64 + i as f64 / SAMPLE_RATE_HZ as f64;
                let tau = 2.0 * std::f64::consts::PI * t;
                let wander = 6000.0 * (0.06 * tau).sin() + 4000.0 * (0.13 * tau + 1.0).sin();
                samples.push(Sample { ts: sec, value: (1000.0 * (freq * tau).sin() + wander) as i32 });
            }
        }
        let est = estimate(&samples);
        assert_eq!(est.len(), 40, "every second should still yield an estimate");
        assert!(est.iter().all(|e| (80..=100).contains(&e.bpm)), "got {est:?}");
    }

    #[test]
    fn aggregate_lets_the_confident_second_decide_the_bucket() {
        // One clean 60 bpm second against three noisy 150 bpm ones: the plain mean says 128, the
        // weighted mean stays near the second we trust.
        let est = vec![
            Estimate { ts: 100, bpm: 60, conf: 0.9 },
            Estimate { ts: 101, bpm: 150, conf: 0.05 },
            Estimate { ts: 102, bpm: 150, conf: 0.05 },
            Estimate { ts: 103, bpm: 150, conf: 0.05 },
        ];
        let out = aggregate(&est, 60);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].ts, 60); // buckets tumble from the epoch, not from the first sample
        assert_eq!(out[0].bpm, 73); // (60·0.9 + 150·0.15) / 1.05
        assert!((out[0].conf - 0.2625).abs() < 1e-3);
        assert_eq!((60 + 150 * 3) / 4, 127); // what the unweighted mean would have said
    }

    #[test]
    fn aggregate_splits_buckets_and_survives_degenerate_input() {
        let est = vec![
            Estimate { ts: 0, bpm: 60, conf: 1.0 },
            Estimate { ts: 59, bpm: 80, conf: 1.0 },
            Estimate { ts: 60, bpm: 100, conf: 1.0 },
        ];
        let out = aggregate(&est, 60);
        assert_eq!(out.iter().map(|e| (e.ts, e.bpm)).collect::<Vec<_>>(), vec![(0, 70), (60, 100)]);
        // A zero-weight bucket falls back to the plain mean rather than dividing by zero.
        let zero = vec![Estimate { ts: 5, bpm: 70, conf: 0.0 }, Estimate { ts: 6, bpm: 90, conf: 0.0 }];
        assert_eq!(aggregate(&zero, 60)[0].bpm, 80);
        // Bucket sizes at or below one second are a no-op, and negative timestamps floor downward.
        assert_eq!(aggregate(&est, 1), est);
        assert_eq!(aggregate(&[Estimate { ts: -5, bpm: 70, conf: 1.0 }], 60)[0].ts, -60);
        assert!(aggregate(&[], 60).is_empty());
    }

    #[test]
    fn derate_poor_seconds_silences_the_flagged_seconds_only() {
        let est = vec![
            Estimate { ts: 10, bpm: 60, conf: 0.9 },
            Estimate { ts: 11, bpm: 150, conf: 0.9 },
            Estimate { ts: 12, bpm: 61, conf: 0.9 },
        ];
        let poor: HashSet<i64> = [11].into_iter().collect();
        let out = derate_poor_seconds(&est, &poor);
        assert_eq!(out.iter().map(|e| e.conf).collect::<Vec<_>>(), vec![0.9, 0.0, 0.9]);
        assert_eq!(out.iter().map(|e| (e.ts, e.bpm)).collect::<Vec<_>>(), vec![(10, 60), (11, 150), (12, 61)]);
        // An empty set, and seconds that carry no estimate, both leave the series untouched.
        assert_eq!(derate_poor_seconds(&est, &HashSet::new()), est);
        assert_eq!(derate_poor_seconds(&est, &[99i64].into_iter().collect()), est);
        assert!(derate_poor_seconds(&[], &poor).is_empty());
    }

    #[test]
    fn a_derated_second_stops_voting_and_stops_counting_clean() {
        // One flagged 150 bpm second against two clean 60 bpm ones, all equally confident: with the flag
        // applied the bucket is the clean pair, without it the strap's unreadable second drags it to 90.
        let est = vec![
            Estimate { ts: 0, bpm: 60, conf: 0.9 },
            Estimate { ts: 1, bpm: 150, conf: 0.9 },
            Estimate { ts: 2, bpm: 60, conf: 0.9 },
        ];
        let poor: HashSet<i64> = [1].into_iter().collect();
        assert_eq!(aggregate(&est, 60)[0].bpm, 90);
        assert_eq!(aggregate(&derate_poor_seconds(&est, &poor), 60)[0].bpm, 60);
        // A whole span the strap could not read falls to Poor, where the raw confidences said Good.
        let span: Vec<Estimate> = (0..80i64).map(|t| Estimate { ts: t, bpm: 60, conf: 0.9 }).collect();
        let all: HashSet<i64> = (0..80i64).collect();
        assert_eq!(signal_check(&span, 0, 99), SignalCheck::Good);
        assert_eq!(signal_check(&derate_poor_seconds(&span, &all), 0, 99), SignalCheck::Poor);
    }

    #[test]
    fn signal_check_grades_a_span_by_its_clean_second_fraction() {
        let span = |clean: usize| -> Vec<Estimate> {
            (0..clean as i64).map(|t| Estimate { ts: t, bpm: 60, conf: 0.9 }).collect()
        };
        assert_eq!(signal_check(&span(80), 0, 99), SignalCheck::Good);
        assert_eq!(signal_check(&span(30), 0, 99), SignalCheck::Fair);
        assert_eq!(signal_check(&span(10), 0, 99), SignalCheck::Poor);
        // Seconds the estimator never emitted count against the span exactly like low-confidence ones.
        let low: Vec<Estimate> = (0..80).map(|t| Estimate { ts: t, bpm: 60, conf: 0.4 }).collect();
        assert_eq!(signal_check(&low, 0, 99), SignalCheck::Poor);
        // Estimates outside the span do not rescue it, and a degenerate span is Poor.
        assert_eq!(signal_check(&span(80), 200, 299), SignalCheck::Poor);
        assert_eq!(signal_check(&[], 0, 99), SignalCheck::Poor);
        assert_eq!(signal_check(&span(80), 10, 0), SignalCheck::Poor);
    }
}
