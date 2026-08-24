//! Uniform-grid spectral primitive for the ECG indices: a linear-detrended, Hann-windowed periodogram
//! evaluated at any normalised frequency by Goertzel recurrence (one multiply-add per sample per bin,
//! no trigonometry in the inner loop). Feeds `ecg::sqi` (band-power ratios) and `ecg::mains` (narrow
//! peak search). `hrv_freq` holds the *uneven*-tachogram estimator (Lomb-Scargle) and spends a sin and
//! a cos per sample per frequency; this is its uniform-grid counterpart, which the mains scan needs
//! because that scan evaluates thousands of frequencies over the same buffer.

use crate::stats::least_squares_line;
use std::f64::consts::PI;

/// A prepared sample buffer: linearly detrended, then Hann-windowed. Powers are raw `|X(f)|²` in
/// arbitrary units — only ratios and peak-to-floor contrasts are meaningful.
#[derive(Clone, Debug)]
pub struct Periodogram {
    windowed: Vec<f64>,
}

impl Periodogram {
    /// Detrend (OLS line) then apply a Hann window. Hann trades main-lobe width (4/N instead of 2/N)
    /// for sidelobes that fall away fast, which is what keeps the ECG's own low-frequency power from
    /// leaking across the whole band and burying a narrow interference peak.
    pub fn new(x: &[f64]) -> Self {
        let n = x.len();
        if n < 2 {
            return Periodogram {
                windowed: Vec::new(),
            };
        }
        let (slope, intercept) = least_squares_line(x);
        let denom = (n - 1) as f64;
        let windowed = x
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                let w = 0.5 - 0.5 * (2.0 * PI * i as f64 / denom).cos();
                (v - (slope * i as f64 + intercept)) * w
            })
            .collect();
        Periodogram { windowed }
    }

    pub fn len(&self) -> usize {
        self.windowed.len()
    }

    pub fn is_empty(&self) -> bool {
        self.windowed.is_empty()
    }

    /// Natural DFT bin spacing, `1/N` in cycles per sample — the resolution floor of this buffer.
    pub fn bin_width(&self) -> f64 {
        if self.windowed.is_empty() {
            0.0
        } else {
            1.0 / self.windowed.len() as f64
        }
    }

    /// `|X(f)|²` at normalised frequency `f` in cycles per sample, by Goertzel. `0.0` outside
    /// `[0, 0.5]`, which is where a real-sampled spectrum stops being distinct.
    pub fn power_at(&self, f: f64) -> f64 {
        if self.windowed.is_empty() || !(0.0..=0.5).contains(&f) {
            return 0.0;
        }
        let coeff = 2.0 * (2.0 * PI * f).cos();
        let (mut s1, mut s2) = (0.0f64, 0.0f64);
        for &v in &self.windowed {
            let s0 = v + coeff * s1 - s2;
            s2 = s1;
            s1 = s0;
        }
        (s1 * s1 + s2 * s2 - coeff * s1 * s2).max(0.0)
    }

    /// Trapezoidal integral of the power across `[f_lo, f_hi]` on a grid one bin wide (>= 8 points).
    /// `0.0` for an empty or inverted band.
    pub fn band_power(&self, f_lo: f64, f_hi: f64) -> f64 {
        if self.windowed.is_empty() || f_hi <= f_lo {
            return 0.0;
        }
        let steps = (((f_hi - f_lo) / self.bin_width()).ceil() as usize).max(8);
        let step = (f_hi - f_lo) / steps as f64;
        let mut acc = 0.0;
        let mut prev = self.power_at(f_lo);
        for k in 1..=steps {
            let p = self.power_at(f_lo + step * k as f64);
            acc += 0.5 * (p + prev) * step;
            prev = p;
        }
        acc
    }

    /// The power across `[f_lo, f_hi]` on a fixed grid, as `(frequency, power)` ascending. The mains
    /// scan oversamples the bin width so a narrow peak lands on more than one grid point.
    pub fn scan(&self, f_lo: f64, f_hi: f64, step: f64) -> Vec<(f64, f64)> {
        if self.windowed.is_empty() || f_hi <= f_lo || step <= 0.0 {
            return Vec::new();
        }
        let count = ((f_hi - f_lo) / step).floor() as usize;
        (0..=count)
            .map(|k| {
                let f = f_lo + step * k as f64;
                (f, self.power_at(f))
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine(n: usize, f: f64, amp: f64) -> Vec<f64> {
        (0..n)
            .map(|i| amp * (2.0 * PI * f * i as f64).sin())
            .collect()
    }

    #[test]
    fn a_pure_tone_peaks_at_its_own_frequency() {
        let p = Periodogram::new(&sine(2048, 0.1, 1.0));
        let on = p.power_at(0.1);
        assert!(
            on > 1000.0 * p.power_at(0.2),
            "tone must dominate an empty part of the band"
        );
        assert!(on > p.power_at(0.1 + 4.0 * p.bin_width()));
    }

    #[test]
    fn band_power_follows_where_the_tone_sits() {
        let p = Periodogram::new(&sine(2048, 0.30, 1.0));
        let inside = p.band_power(0.28, 0.32);
        let outside = p.band_power(0.05, 0.09);
        assert!(
            inside > 100.0 * outside,
            "inside {inside:e} outside {outside:e}"
        );
    }

    #[test]
    fn a_linear_ramp_carries_no_power_after_detrending() {
        // The detrend is the whole reason baseline drift does not swamp basSQI.
        let ramp: Vec<f64> = (0..1024).map(|i| i as f64).collect();
        let p = Periodogram::new(&ramp);
        assert!(
            p.band_power(0.001, 0.02) < 1e-12,
            "a pure ramp must be removed, got {}",
            p.band_power(0.001, 0.02)
        );
    }

    #[test]
    fn degenerate_inputs_are_zero_not_a_panic() {
        let p = Periodogram::new(&[1.0]);
        assert!(p.is_empty() && p.power_at(0.25) == 0.0 && p.bin_width() == 0.0);
        let q = Periodogram::new(&sine(256, 0.1, 1.0));
        assert_eq!(q.power_at(-0.1), 0.0);
        assert_eq!(q.power_at(0.7), 0.0);
        assert_eq!(q.band_power(0.3, 0.3), 0.0);
        assert!(q.scan(0.3, 0.1, 0.01).is_empty());
    }
}
