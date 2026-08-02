//! Multiscale wavelet QRS detection: a dyadic derivative-of-smoothing transform, modulus-maxima pairs of
//! opposite sign straddling a zero crossing, confirmed across an octave of scales. The SHAPE half of the
//! detector pair (see the module header).
//!
//! Nothing here is squared or rectified before the decision, so the criterion is a signed slope reversal
//! inside one QRS width — a question Pan-Tompkins cannot ask and does not answer.

use super::{samples_for_ms, sanitized, usable_rate, MAX_QRS_MS, REFRACTORY_MS};
use crate::signal::{find_peaks, robust_sigma};

/// Modulus-maximum threshold in robust sigmas of the scale's own detail. 3 sigma is the ordinary outlier
/// line; a QRS sits far above it because the MAD is set by the baseline, not by the complexes.
const SIGMA_MULTIPLE: f64 = 3.0;
/// Fallback threshold as a fraction of the scale's peak modulus, used only when the robust sigma is 0
/// (over half the samples identical — a synthetic or a stuck channel has no noise floor to measure).
const NO_NOISE_FLOOR_FRACTION: f64 = 0.25;
/// Two scales' zero crossings this far apart (ms) are the same beat. Coarser scales smooth more, so the
/// crossing drifts a little; wider than this and two neighbouring beats would merge.
const COHERENCE_MS: f64 = 50.0;
/// A candidate must appear at this many distinct scales. Below 2 there is no coherence test at all.
const MIN_SCALES: usize = 2;
/// Index of the finest scale in the ladder — the only one that contributes strength (see `Cluster`).
const FINEST_RANK: usize = 0;
/// Multiplicative gap in fine-scale strength that marks the T-wave population off from the R-wave one.
/// Measured on real overnight ECG the two sit 14-17x apart, so 3x cuts well inside the empty band while
/// leaving an ordinary beat-to-beat amplitude variation (well under 2x) untouched.
const T_WAVE_STRENGTH_RATIO: f64 = 3.0;
/// Beats needed before the split means anything; below it every candidate is kept.
const MIN_BEATS_FOR_STRENGTH_GATE: usize = 6;
/// The QRS band centre the scale ladder is aimed at (Hz).
const QRS_BAND_CENTRE_HZ: f64 = 60.0;

/// R-peak sample indices from a single-lead ECG, ascending and strictly increasing.
///
/// Polarity-agnostic: an upright R gives a positive-then-negative modulus pair and an inverted R the
/// reverse, and both are accepted, because the strap's lead orientation is not known. Returns empty
/// (never panics) on empty, constant, non-finite or too-short input, or an unsupported `fs_hz`.
pub fn detect_wavelet(samples: &[f64], fs_hz: f64) -> Vec<usize> {
    if !usable_rate(fs_hz) {
        return Vec::new();
    }
    let x = sanitized(samples);
    let scales = scale_ladder(fs_hz);
    let max_dilation = 1usize << (scales[scales.len() - 1] - 1);
    if x.len() < max_dilation * 8 {
        return Vec::new();
    }

    let max_pair = samples_for_ms(MAX_QRS_MS, fs_hz, 2);
    let coherence = samples_for_ms(COHERENCE_MS, fs_hz, 1);
    let refractory = samples_for_ms(REFRACTORY_MS, fs_hz, 1);

    // (fiducial, scale, strength) from every scale, merged into one time-ordered list.
    let mut hits: Vec<(usize, usize, f64)> = Vec::new();
    let mut smooth = x;
    for (rank, &j) in scales.iter().enumerate() {
        let dilation = 1usize << (j - 1);
        let detail = dyadic_detail(&smooth, dilation);
        smooth = dyadic_smooth(&smooth, dilation);
        for (idx, strength) in modulus_maxima_pairs(&detail, max_pair) {
            hits.push((idx, rank, strength));
        }
    }
    if hits.is_empty() {
        return Vec::new();
    }
    hits.sort_by_key(|h| h.0);

    coherent_clusters(&hits, coherence, refractory)
}

/// Scales whose detail band brackets the QRS. The dilated derivative at dilation `d` peaks near
/// `fs / (4d)`, so the centre scale tracks the rate and the neighbours give it an octave either side.
fn scale_ladder(fs_hz: f64) -> Vec<usize> {
    let centre = ((fs_hz / QRS_BAND_CENTRE_HZ).log2().round() as i64 + 1).clamp(2, 8) as usize;
    vec![centre - 1, centre, centre + 1]
}

/// Centred dilated first difference — a zero-phase derivative of the current smoothing. Edges are clamped
/// so the transform stays the same length as the input.
fn dyadic_detail(x: &[f64], dilation: usize) -> Vec<f64> {
    let n = x.len();
    (0..n)
        .map(|i| x[(i + dilation).min(n - 1)] - x[i.saturating_sub(dilation)])
        .collect()
}

/// Centred dilated [1, 3, 3, 1]/8 smoothing — one rung up the a-trous ladder. Symmetric, so zero phase.
fn dyadic_smooth(x: &[f64], dilation: usize) -> Vec<f64> {
    let n = x.len();
    let at = |i: isize| -> f64 { x[i.clamp(0, n as isize - 1) as usize] };
    (0..n)
        .map(|i| {
            let i = i as isize;
            let d = dilation as isize;
            (at(i - d) + 3.0 * at(i) + 3.0 * at(i + d) + at(i + 2 * d)) / 8.0
        })
        .collect()
}

/// Fiducials from opposite-sign modulus-maximum pairs: the zero crossing between them, which is by
/// construction the extremum of the underlying signal. Strength is the pair's modulus in threshold units,
/// so it is comparable across scales and free of the amplitude scale.
fn modulus_maxima_pairs(detail: &[f64], max_pair: usize) -> Vec<(usize, f64)> {
    let sigma = robust_sigma(detail);
    let modulus: Vec<f64> = detail.iter().map(|v| v.abs()).collect();
    let threshold = if sigma > 0.0 {
        SIGMA_MULTIPLE * sigma
    } else {
        NO_NOISE_FLOOR_FRACTION * modulus.iter().copied().fold(0.0f64, f64::max)
    };
    if threshold <= 0.0 {
        return Vec::new();
    }
    let maxima = find_peaks(&modulus, 1, threshold);

    let mut out = Vec::new();
    for w in maxima.windows(2) {
        let (a, b) = (w[0], w[1]);
        if b - a > max_pair || detail[a] * detail[b] >= 0.0 {
            continue;
        }
        if let Some(cross) = zero_crossing(detail, a, b) {
            out.push((cross, (modulus[a] + modulus[b]) / threshold));
        }
    }
    out
}

/// First index in `(a, b]` where the detail changes sign relative to `a`.
fn zero_crossing(detail: &[f64], a: usize, b: usize) -> Option<usize> {
    let sign = detail[a].signum();
    (a + 1..=b).find(|&k| detail[k].signum() != sign)
}

/// Group hits by proximity, keep groups seen at [`MIN_SCALES`] distinct scales, apply the refractory
/// strongest-first, then drop what is left of the T waves. The kept fiducial is the cluster's STRONGEST
/// hit, which is the R itself — taking the earliest instead drags the mark onto the Q and can push the
/// following T wave back outside the refractory.
fn coherent_clusters(hits: &[(usize, usize, f64)], coherence: usize, refractory: usize) -> Vec<usize> {
    let mut clusters: Vec<Cluster> = Vec::new();
    for &(idx, rank, strength) in hits {
        match clusters.last_mut() {
            Some(c) if idx - c.fiducial <= coherence => c.absorb(idx, rank, strength),
            _ => clusters.push(Cluster::new(idx, rank, strength)),
        }
    }

    let mut ranked: Vec<(usize, f64)> = clusters
        .into_iter()
        .filter(|c| c.ranks.len() >= MIN_SCALES)
        .map(|c| (c.fiducial, c.strength))
        .collect();
    // Strongest first, index as a tie-break so the result cannot depend on sort stability.
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));

    let mut kept: Vec<(usize, f64)> = Vec::new();
    for (idx, strength) in ranked {
        if kept.iter().all(|&(k, _)| idx.abs_diff(k) >= refractory) {
            kept.push((idx, strength));
        }
    }

    let floor = t_wave_floor(&kept);
    let mut out: Vec<usize> = kept.into_iter().filter(|&(_, s)| s >= floor).map(|(i, _)| i).collect();
    out.sort_unstable();
    out
}

/// Strength below which a surviving cluster is a T wave, or 0.0 to keep everything.
///
/// A T wave outside the refractory survives the greedy pass, and it arrives roughly one per beat — so the
/// two populations are near equal in COUNT and any quantile lands on the boundary and tips either way with
/// a single detection. They are far apart in VALUE, so the split is the widest multiplicative gap in the
/// sorted strengths, taken only when it exceeds [`T_WAVE_STRENGTH_RATIO`].
fn t_wave_floor(kept: &[(usize, f64)]) -> f64 {
    if kept.len() < MIN_BEATS_FOR_STRENGTH_GATE {
        return 0.0;
    }
    let mut sorted: Vec<f64> = kept.iter().map(|&(_, s)| s).collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let (mut best_ratio, mut floor) = (1.0f64, 0.0f64);
    for i in 1..sorted.len() {
        if sorted[i - 1] <= 0.0 {
            continue;
        }
        let ratio = sorted[i] / sorted[i - 1];
        if ratio > best_ratio {
            best_ratio = ratio;
            floor = sorted[i];
        }
    }
    if best_ratio >= T_WAVE_STRENGTH_RATIO {
        floor
    } else {
        0.0
    }
}

/// One time-local group of modulus-maxima pairs. `strength` is the group total (how much evidence there
/// is), `best` the largest single hit (which one is the R).
struct Cluster {
    fiducial: usize,
    best: f64,
    ranks: Vec<usize>,
    strength: f64,
}

impl Cluster {
    fn new(idx: usize, rank: usize, strength: f64) -> Self {
        let mut c = Cluster { fiducial: idx, best: 0.0, ranks: Vec::new(), strength: 0.0 };
        c.absorb(idx, rank, strength);
        c
    }

    fn absorb(&mut self, idx: usize, rank: usize, strength: f64) {
        if !self.ranks.contains(&rank) {
            self.ranks.push(rank);
        }
        // Only the finest scale contributes strength. A slow T wave raises a large modulus on the coarse
        // scales and a small one on the fine scale, so summing all three lets it rival a real QRS; scoring
        // it on the fine scale alone is what separates them.
        if rank == FINEST_RANK {
            self.strength += strength;
            if strength > self.best {
                self.best = strength;
                self.fiducial = idx;
            }
        }
    }
}


#[cfg(test)]
mod tests {
    use super::super::test_signals::{constant, synthetic_ecg};
    use super::*;

    #[test]
    fn recovers_a_known_beat_count_at_every_supported_rate() {
        for fs in [100.0, 200.0, 250.0, 500.0, 1000.0] {
            let (x, truth) = synthetic_ecg(fs, 20.0, 60.0, 1.0, 0.01, 11);
            let peaks = detect_wavelet(&x, fs);
            assert_eq!(peaks.len(), truth.len(), "fs {fs}: got {} want {}", peaks.len(), truth.len());
            for (p, t) in peaks.iter().zip(&truth) {
                let err_ms = (*p as f64 - *t as f64).abs() / fs * 1000.0;
                assert!(err_ms <= 30.0, "fs {fs}: fiducial off by {err_ms:.1} ms");
            }
        }
    }

    #[test]
    fn an_inverted_lead_reads_the_same_beats() {
        let (x, truth) = synthetic_ecg(200.0, 20.0, 66.0, 1.0, 0.01, 12);
        let flipped: Vec<f64> = x.iter().map(|v| -v).collect();
        assert_eq!(detect_wavelet(&flipped, 200.0).len(), truth.len());
    }

    #[test]
    fn degenerate_input_is_empty_never_a_panic() {
        assert!(detect_wavelet(&[], 200.0).is_empty());
        assert!(detect_wavelet(&constant(4000, -7.5), 200.0).is_empty());
        assert!(detect_wavelet(&[f64::NAN; 4000], 200.0).is_empty());
        assert!(detect_wavelet(&[f64::NEG_INFINITY; 4000], 200.0).is_empty());
        let (x, _) = synthetic_ecg(200.0, 10.0, 60.0, 1.0, 0.01, 13);
        assert!(detect_wavelet(&x, 99.0).is_empty());
        assert!(detect_wavelet(&x, 1025.0).is_empty());
        assert!(detect_wavelet(&x, f64::INFINITY).is_empty());
        assert!(detect_wavelet(&x[..20], 200.0).is_empty());
    }

    #[test]
    fn output_is_deterministic_and_strictly_increasing() {
        let (x, _) = synthetic_ecg(250.0, 20.0, 72.0, 1.0, 0.05, 14);
        let a = detect_wavelet(&x, 250.0);
        assert_eq!(a, detect_wavelet(&x, 250.0));
        assert!(a.windows(2).all(|w| w[1] > w[0]));
    }

    #[test]
    fn amplitude_scale_does_not_change_the_answer() {
        let (x, _) = synthetic_ecg(200.0, 20.0, 66.0, 1.0, 0.02, 15);
        let base = detect_wavelet(&x, 200.0);
        for gain in [1e-3, 37.0, 5000.0] {
            let scaled: Vec<f64> = x.iter().map(|v| v * gain).collect();
            assert_eq!(detect_wavelet(&scaled, 200.0), base, "gain {gain}");
        }
    }

    #[test]
    fn the_scale_ladder_tracks_the_sample_rate() {
        // The centre scale's dilated derivative must peak inside the QRS band at every supported rate.
        for fs in [100.0, 200.0, 250.0, 500.0, 1000.0] {
            let centre = scale_ladder(fs)[1];
            let peak_hz = fs / (4.0 * (1usize << (centre - 1)) as f64);
            assert!((8.0..=25.0).contains(&peak_hz), "fs {fs}: centre scale peaks at {peak_hz} Hz");
        }
    }
}
