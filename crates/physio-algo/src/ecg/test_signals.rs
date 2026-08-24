//! Deterministic waveform generators for the detector unit tests: a gaussian-mixture ECG with known R
//! positions, and the degenerate inputs. Test-only; the integration gate scores real recordings.

use std::f64::consts::PI;

/// Splitmix64 — a deterministic seeded stream, so a test failure is always reproducible.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Box-Muller, one value per call (the second is discarded to keep the call site simple).
    fn gaussian(&mut self) -> f64 {
        let u1 = self.unit().max(1e-12);
        let u2 = self.unit();
        (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
    }
}

/// (centre offset in ms, amplitude in units of the R wave, sigma in ms) for P, Q, R, S, T. The gaussian
/// mixture puts the QRS at ~100 ms wide, inside the 60-140 ms physiological band, and the T wave at a
/// realistic QT of ~300 ms — well OUTSIDE the 200 ms refractory, so a T wave has to be rejected on its
/// own merits rather than hidden by the refractory.
const WAVES: [(f64, f64, f64); 5] = [
    (-180.0, 0.15, 25.0),
    (-25.0, -0.10, 8.0),
    (0.0, 1.0, 10.0),
    (25.0, -0.25, 12.0),
    (300.0, 0.30, 50.0),
];

/// Synthetic single-lead ECG plus the true R-peak sample indices.
///
/// R-R carries a 0.25 Hz +/-4% respiratory modulation, so a detector that merely locks onto a fixed period
/// cannot pass. `noise_sd` is in units of the R amplitude. Deterministic in `seed`.
pub fn synthetic_ecg(
    fs_hz: f64,
    seconds: f64,
    bpm: f64,
    amplitude: f64,
    noise_sd: f64,
    seed: u64,
) -> (Vec<f64>, Vec<usize>) {
    let n = (seconds * fs_hz) as usize;
    let mut out = vec![0.0f64; n];
    let mut truth = Vec::new();

    let mean_rr_s = 60.0 / bpm;
    let mut t = mean_rr_s; // first beat one R-R in, so the leading P wave fits
    while t < seconds - mean_rr_s * 0.5 {
        let centre = (t * fs_hz).round() as usize;
        if centre >= n {
            break;
        }
        truth.push(centre);
        for &(offset_ms, gain, sigma_ms) in &WAVES {
            let mu = centre as f64 + offset_ms / 1000.0 * fs_hz;
            let sigma = (sigma_ms / 1000.0 * fs_hz).max(0.5);
            let span = (4.0 * sigma).ceil() as isize;
            let lo = (mu as isize - span).max(0) as usize;
            let hi = ((mu as isize + span) as usize).min(n - 1);
            for (i, cell) in out.iter_mut().enumerate().take(hi + 1).skip(lo) {
                let d = (i as f64 - mu) / sigma;
                *cell += amplitude * gain * (-0.5 * d * d).exp();
            }
        }
        t += mean_rr_s * (1.0 + 0.04 * (2.0 * PI * 0.25 * t).sin());
    }

    if noise_sd > 0.0 {
        let mut rng = Rng(seed);
        for cell in out.iter_mut() {
            *cell += amplitude * noise_sd * rng.gaussian();
        }
    }
    (out, truth)
}

/// A flat line — the dead-channel case.
pub fn constant(n: usize, value: f64) -> Vec<f64> {
    vec![value; n]
}
