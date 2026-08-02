//! The cheap, rate-free half of the sweep: reject a reading rule from the decoded samples alone, before
//! anything that costs a spectrum or a detector.
//!
//! Two statistics carry it, and both are invariant to the amplitude scale, so neither needs to know what
//! a count is worth. [`roughness`] is `1 − r₁`, the fraction of the signal's power that is not explained
//! by the previous sample: a misaligned field mixes one sample's high bits into the next and comes out
//! near-white at 1.0, while any real waveform sampled at hundreds of hertz is heavily oversampled and
//! sits far below that. Kurtosis carries the other half — a ramp, a counter field and a sawtooth are all
//! perfectly smooth and all sit near 1.8, so smoothness alone would let them through.
//!
//! What survives is then grouped by waveform, because a reading rule is not uniquely identified by its
//! output: the top 16 bits of a 24-bit field are the same waveform divided by 256, and with no
//! counts-per-mV in the search space those two are indistinguishable **by construction**. Ranking them
//! as separate answers would make the runner-up margin zero forever.

use crate::ecg::sqi::k_sqi;
use crate::stats::{mean, pearson};

/// Rate-free description of one decoded candidate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LayoutStats {
    pub samples: usize,
    /// Population standard deviation, in the caller's raw counts. Zero is a dead channel.
    pub sd: f64,
    /// `1 − r₁`: 1.0 for white noise, far below it for anything oversampled.
    pub roughness: f64,
    /// Pearson's b2, as [`crate::ecg::sqi::k_sqi`] defines it.
    pub kurtosis: f64,
}

/// `1 − r₁`, computed as `mean(Δ²) / 2σ²`. `None` on fewer than two samples or a constant series.
pub fn roughness(x: &[f64]) -> Option<f64> {
    if x.len() < 2 {
        return None;
    }
    let m = mean(x);
    let var = x.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / x.len() as f64;
    if var.is_nan() || var <= 0.0 {
        return None;
    }
    let d = x.windows(2).map(|w| (w[1] - w[0]) * (w[1] - w[0])).sum::<f64>() / (x.len() - 1) as f64;
    Some(d / (2.0 * var))
}

/// Every rate-free statistic for one decoded candidate; `None` when the series is too short, not finite
/// throughout, or constant.
pub fn layout_stats(x: &[f64]) -> Option<LayoutStats> {
    if x.len() < 4 || !x.iter().all(|v| v.is_finite()) {
        return None;
    }
    let m = mean(x);
    let var = x.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / x.len() as f64;
    if var.is_nan() || var <= 0.0 {
        return None;
    }
    Some(LayoutStats { samples: x.len(), sd: var.sqrt(), roughness: roughness(x)?, kurtosis: k_sqi(x)? })
}

/// Waveform equivalence: `|r| ≥ min_r` over the common prefix, and lengths within
/// [`MAX_LENGTH_SLACK`] of each other so two rules with different time bases are never fused.
///
/// The sign is dropped because an inverted read of the same field is the same evidence about the stream;
/// the QRS detectors are polarity-tolerant and the indices are scale-invariant, so a lead inversion
/// cannot be resolved here and pretending otherwise would inflate the candidate count.
pub const MAX_LENGTH_SLACK: usize = 2;
/// Longest prefix correlated when grouping. Enough to separate two waveforms; short enough that grouping
/// stays cheap when many rules survive.
pub const CLUSTER_PREFIX: usize = 4096;

/// Class index per waveform, assigned greedily in input order so the result is deterministic and the
/// lowest-indexed member of a class is its representative.
pub fn cluster(waveforms: &[&[f64]], min_r: f64) -> Vec<usize> {
    let mut reps: Vec<usize> = Vec::new();
    let mut class = vec![0usize; waveforms.len()];
    for (i, w) in waveforms.iter().enumerate() {
        let found = reps.iter().position(|&r| equivalent(waveforms[r], w, min_r));
        class[i] = match found {
            Some(c) => c,
            None => {
                reps.push(i);
                reps.len() - 1
            }
        };
    }
    class
}

fn equivalent(a: &[f64], b: &[f64], min_r: f64) -> bool {
    a.len().abs_diff(b.len()) <= MAX_LENGTH_SLACK + MAX_LAG && correlates(a, b, min_r)
}

/// Whole-sample shifts allowed when asking whether two readings are the same waveform.
///
/// Where a rule's first whole sample lands depends on its own header and stride, so two readings of one
/// stream routinely differ by a sample or two, and at these rates a lag-1 autocorrelation of about 0.99
/// sits right on the equivalence threshold — which made a pure phase shift look like a rival answer and
/// left the leader with no margin. A genuinely different answer is not a four-sample shift of the leader.
pub const MAX_LAG: usize = 4;

/// `|r| ≥ min_r` at the best whole-sample lag within [`MAX_LAG`], over the common prefix.
fn correlates(a: &[f64], b: &[f64], min_r: f64) -> bool {
    let at = |x: &[f64], y: &[f64]| {
        let n = x.len().min(y.len()).min(CLUSTER_PREFIX);
        pearson(&x[..n], &y[..n]).is_some_and(|r| r.abs() >= min_r)
    };
    (0..=MAX_LAG).any(|lag| {
        (a.len() > lag && at(&a[lag..], b)) || (b.len() > lag && at(a, &b[lag..]))
    })
}

/// Deepest decimation considered when asking whether two candidates are the same signal.
pub const MAX_DECIMATION: usize = 8;

/// `true` when `a` read at `fs_a` and `b` read at `fs_b` are the same signal in wall-clock time, one
/// being every k-th sample of the other.
///
/// This is the ambiguity that would otherwise stop the sweep converging on anything. If a stream is
/// 16-bit words at 400 Hz, then "every other 16-bit word, at 200 Hz" is a perfectly good ECG covering
/// the same seconds, scores identically on every index, and lines up with the same optical beats — so
/// treating it as a rival answer leaves the leader with no margin forever. It is not a rival; it is the
/// same answer with samples thrown away, and the reading that throws none away is the one to report.
///
/// The rate ratio has to be an exact integer and the decimated series has to match at some phase, which
/// is what keeps this from fusing two genuinely different rate hypotheses: the SAME waveform read at two
/// rates fails the length test, because a decimation has proportionally fewer samples and a reinterpreted
/// rate has exactly as many.
pub fn same_time_base(a: &[f64], fs_a: f64, b: &[f64], fs_b: f64, min_r: f64) -> bool {
    if !(fs_a > 0.0 && fs_b > 0.0) {
        return false;
    }
    if fs_a < fs_b {
        return same_time_base(b, fs_b, a, fs_a, min_r);
    }
    let ratio = fs_a / fs_b;
    let k = ratio.round();
    if (ratio - k).abs() > 1e-6 || k < 1.0 || k > MAX_DECIMATION as f64 {
        return false;
    }
    let k = k as usize;
    if k == 1 {
        return equivalent(a, b, min_r);
    }
    (0..k).any(|phase| {
        let d: Vec<f64> = a.iter().skip(phase).step_by(k).copied().collect();
        let (la, lb) = (d.len(), b.len());
        let slack = 0.05 * la.max(lb) as f64 + 2.0;
        la.abs_diff(lb) as f64 <= slack && correlates(&d, b, min_r)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn smooth(n: usize) -> Vec<f64> {
        (0..n).map(|i| (2.0 * PI * i as f64 / 40.0).sin() + 0.3 * (2.0 * PI * i as f64 / 7.0).sin()).collect()
    }

    fn white(n: usize, seed: u64) -> Vec<f64> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5
            })
            .collect()
    }

    #[test]
    fn roughness_separates_oversampled_from_white() {
        let r_smooth = roughness(&smooth(4000)).unwrap();
        let r_white = roughness(&white(4000, 3)).unwrap();
        assert!(r_smooth < 0.1, "an oversampled waveform must be smooth, got {r_smooth}");
        assert!((r_white - 1.0).abs() < 0.1, "white noise must sit at 1, got {r_white}");
        // Alternating sign is the roughest a series can be: r1 = -1, so 1 - r1 = 2.
        let alt: Vec<f64> = (0..1000).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
        assert!((roughness(&alt).unwrap() - 2.0).abs() < 0.01);
        assert!(roughness(&[1.0; 100]).is_none());
        assert!(roughness(&[1.0]).is_none());
    }

    #[test]
    fn layout_stats_refuses_a_dead_or_non_finite_channel() {
        assert!(layout_stats(&[7.0; 500]).is_none());
        assert!(layout_stats(&[1.0, 2.0]).is_none());
        let mut bad = smooth(500);
        bad[10] = f64::NAN;
        assert!(layout_stats(&bad).is_none());
        let s = layout_stats(&smooth(4000)).unwrap();
        assert_eq!(s.samples, 4000);
        assert!(s.sd > 0.0 && s.roughness < 0.1);
    }

    #[test]
    fn a_scaled_truncated_or_inverted_read_is_the_same_class() {
        let base = smooth(3000);
        let scaled: Vec<f64> = base.iter().map(|v| v * 256.0).collect();
        let inverted: Vec<f64> = base.iter().map(|v| -v).collect();
        let offset: Vec<f64> = base.iter().map(|v| v + 1000.0).collect();
        let other = white(3000, 11);
        let set: Vec<&[f64]> = vec![&base, &scaled, &inverted, &offset, &other];
        let c = cluster(&set, 0.99);
        assert_eq!(c[0], c[1]);
        assert_eq!(c[0], c[2], "an inverted lead is not a second answer");
        assert_eq!(c[0], c[3]);
        assert_ne!(c[0], c[4], "an unrelated waveform must be its own class");
        assert_eq!(c.iter().copied().max().unwrap(), 1);
    }

    #[test]
    fn a_decimated_read_is_the_same_answer_but_a_reinterpreted_rate_is_not() {
        let base = smooth(4000);
        let half: Vec<f64> = base.iter().step_by(2).copied().collect();
        // Every other sample at half the rate covers the same seconds: the same answer.
        assert!(same_time_base(&base, 400.0, &half, 200.0, 0.99));
        assert!(same_time_base(&half, 200.0, &base, 400.0, 0.99));
        // Phase matters not at all - the odd samples are the same answer as the even ones.
        let odd: Vec<f64> = base.iter().skip(1).step_by(2).copied().collect();
        assert!(same_time_base(&base, 400.0, &odd, 200.0, 0.99));
        // The SAME samples called 200 Hz instead of 400 Hz is a different answer, and has to stay one.
        assert!(!same_time_base(&base, 400.0, &base, 200.0, 0.99));
        // A non-integer ratio is not a decimation.
        assert!(!same_time_base(&base, 400.0, &half, 250.0, 0.99));
        // Nor is an unrelated waveform at an integer ratio.
        assert!(!same_time_base(&base, 400.0, &white(2000, 5), 200.0, 0.99));
    }

    #[test]
    fn a_different_time_base_is_never_fused_however_similar() {
        // Half as many samples of the same thing is a different rate hypothesis, not the same answer.
        let base = smooth(3000);
        let decimated: Vec<f64> = base.iter().step_by(2).copied().collect();
        let c = cluster(&[&base, &decimated], 0.5);
        assert_ne!(c[0], c[1]);
    }
}
