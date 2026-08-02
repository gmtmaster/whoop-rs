//! Test-only scaffolding for the decode sweep: re-encode a known waveform at a chosen layout and rate,
//! and derive the optical beat train the sweep is given.
//!
//! Everything here builds ground truth. `encode_stream` produces a byte buffer whose true reading rule
//! and true sample rate are known by construction, so recovery is measurable rather than plausible. The
//! beat extractor is deliberately in the harness and not in the library: the strap reports its own R-R
//! over BLE, so a real caller never detects optical beats, and a detector living in the library would be
//! a second source of truth for something the hardware already provides.

use physio_algo::ecg::sweep::layout::{encode, Layout};
use physio_algo::ecg::{beat_agreement, detect_pan_tompkins, detect_wavelet, DEFAULT_MATCH_WINDOW_MS};
use physio_algo::signal::{find_peaks, moving_average_centred};

use crate::ecg_corpus::{max_of, mean, min_of, resample, Rng};

/// Scale a waveform into `bits`-wide two's complement counts, using the widest span that cannot clip.
/// The gain is arbitrary and deliberately so — nothing downstream may read it as a counts-per-mV.
pub fn quantise(x: &[f64], bits: u8) -> Vec<i64> {
    let m = mean(x);
    let span = (max_of(x) - m).abs().max((min_of(x) - m).abs()).max(1e-12);
    let full = ((1i64 << (bits - 1)) - 1) as f64;
    x.iter().map(|v| (((v - m) / span) * full * 0.9).round() as i64).collect()
}

/// A byte buffer carrying `x` under `layout`, resampled from `fs_in` to `fs_out` first. Slots the layout
/// does not claim are filled with `filler` counts, so a sweep that picks the wrong channel finds
/// something there rather than zeros.
pub fn encode_stream(
    x: &[f64],
    fs_in: f64,
    fs_out: f64,
    layout: &Layout,
    filler: Option<&[f64]>,
) -> Vec<u8> {
    encode_at(&resample(x, fs_in, fs_out), layout, filler.map(|f| resample(f, fs_in, fs_out)).as_deref())
}

/// The same, for a waveform already at the target rate — the path a test needs when it has to add
/// something (power-line hum, say) at a frequency that only means what it means at that rate.
pub fn encode_at(wave: &[f64], layout: &Layout, filler: Option<&[f64]>) -> Vec<u8> {
    let counts = quantise(wave, layout.bits);
    let bits_needed = layout.start_bit + counts.len() * layout.stride_bits;
    let len_bytes = bits_needed.div_ceil(8) + 8;
    let mut bytes = encode(layout, &counts, len_bytes);
    if let Some(f) = filler {
        // Only a frame with room for a second field has unclaimed slots; writing one into a densely
        // packed stream would overlap the data and quietly destroy the ground truth.
        let other = Layout { start_bit: layout.start_bit + layout.bits as usize, ..*layout };
        if layout.stride_bits >= 2 * layout.bits as usize {
            let fc = quantise(f, layout.bits);
            let filled = encode(&other, &fc, len_bytes);
            for (b, o) in bytes.iter_mut().zip(filled.iter()) {
                *b |= o;
            }
        }
    }
    bytes
}

/// R peaks both detectors agree on, which is the closest thing to a beat annotation this corpus has —
/// AAUWSS ships none. Used only to build the optical beat train, never to score anything.
pub fn detector_consensus(x: &[f64], fs_hz: f64) -> Vec<usize> {
    let a = detect_pan_tompkins(x, fs_hz);
    let b = detect_wavelet(x, fs_hz);
    let tol = (DEFAULT_MATCH_WINDOW_MS / 1000.0 * fs_hz).round() as usize;
    assert!(beat_agreement(&a, &b, x.len(), fs_hz, DEFAULT_MATCH_WINDOW_MS).matched > 0);
    a.iter().copied().filter(|p| b.iter().any(|q| p.abs_diff(*q) <= tol)).collect()
}

/// A wrist pulse train from known R positions: each beat delayed by `ptt_ms` plus a per-beat draw of up
/// to `jitter_ms`, because pulse transit time varies beat to beat and a train that did not would test a
/// matcher no real strap will ever feed.
pub fn pulse_train(r_samples: &[usize], fs_hz: f64, ptt_ms: f64, jitter_ms: f64, seed: u64) -> Vec<f64> {
    let mut rng = Rng(seed);
    r_samples
        .iter()
        .map(|&p| p as f64 / fs_hz * 1000.0 + ptt_ms + (rng.unit() - 0.5) * 2.0 * jitter_ms)
        .collect()
}

/// Beat times (ms) from a real photoplethysmogram: band-limit, find the dominant pulse period by
/// autocorrelation, then take local maxima no closer than 70 % of it.
///
/// Crude on purpose and only ever a test reference. A real caller never does this — the strap reports its
/// own R-R over BLE — and its accuracy is the limit of what the cross-modal check can claim.
pub fn ppg_beats_ms(x: &[f64], fs_hz: f64) -> Vec<f64> {
    let smooth = moving_average_centred(x, (fs_hz / 8.0).round().max(3.0) as usize);
    let base = moving_average_centred(&smooth, (fs_hz * 1.2).round().max(5.0) as usize);
    let ac: Vec<f64> = smooth.iter().zip(base.iter()).map(|(a, b)| a - b).collect();
    let (lo, hi) = ((0.35 * fs_hz) as usize, (2.0 * fs_hz) as usize);
    let period = (lo..hi.min(ac.len() / 2))
        .max_by(|&a, &b| correlation_at(&ac, a).total_cmp(&correlation_at(&ac, b)))
        .unwrap_or(lo);
    find_peaks(&ac, (0.7 * period as f64) as usize, 0.0).iter().map(|&i| i as f64 / fs_hz * 1000.0).collect()
}

fn correlation_at(x: &[f64], lag: usize) -> f64 {
    x.iter().zip(x[lag..].iter()).map(|(a, b)| a * b).sum::<f64>() / (x.len() - lag) as f64
}
