//! Activity features from decoded 5/MG raw 6-axis IMU: turn a window of 100 Hz accel+gyro samples into a
//! compact vector (energy, jerk, gait cadence) for coarse activity classification. Cadence is an
//! autocorrelation-peak FEATURE reported with its own strength, never fed to a physiological gate. Pure.
//!
//! The 1 Hz on-chip ENMO channel is a DIFFERENT motion source: good for magnitude, activity level and
//! circadian rhythm, never for cadence or steps — see [`CADENCE_BAND_LO_HZ`].

/// One raw IMU sample: 3-axis accelerometer (g) + 3-axis gyroscope (deg/s).
#[derive(Clone, Copy, Debug)]
pub struct ImuSample {
    pub ax: f64,
    pub ay: f64,
    pub az: f64,
    pub gx: f64,
    pub gy: f64,
    pub gz: f64,
}

/// Compact activity features over a window of raw IMU samples.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ImuActivityFeatures {
    /// RMS of the accel-magnitude AC (gravity removed), in g — overall movement intensity.
    pub accel_energy_g: f64,
    /// Mean gyroscope magnitude over the window, in deg/s — rotational intensity.
    pub gyro_energy_dps: f64,
    /// RMS of the accel first-difference (jerk), in g/sample — impact / explosiveness.
    pub jerk_rms: f64,
    /// Dominant gait-band cadence, Hz — `None` when no peak clears [`MIN_CADENCE_STRENGTH`].
    pub cadence_hz: Option<f64>,
    /// Normalized strength (0..1) of that cadence peak — high = rhythmic, low = bursty or still.
    pub cadence_strength: f64,
    pub sample_count: usize,
}

/// Cadence search band, Hz — human gait/pedal foot rate (~72-210 steps/min), below the IMU Nyquist.
/// A 1 Hz source (the on-chip ENMO channel) has a 0.5 Hz Nyquist UNDER this whole band, so any peak it
/// showed would be an alias; at `sample_rate_hz = 1` the lag window collapses and cadence stays `None`.
pub const CADENCE_BAND_LO_HZ: f64 = 1.2;
pub const CADENCE_BAND_HI_HZ: f64 = 3.5;

/// A cadence peak below this normalized autocorrelation strength reads as "no rhythm" (→ `None` Hz).
pub const MIN_CADENCE_STRENGTH: f64 = 0.20;

const MIN_IMU_SAMPLES: usize = 8;

/// Extract features from `samples` (from one or more IMU frames, in order) at `sample_rate_hz`.
pub fn extract(samples: &[ImuSample], sample_rate_hz: i32) -> ImuActivityFeatures {
    let n = samples.len();
    if n < MIN_IMU_SAMPLES || sample_rate_hz <= 0 {
        return ImuActivityFeatures {
            accel_energy_g: 0.0,
            gyro_energy_dps: 0.0,
            jerk_rms: 0.0,
            cadence_hz: None,
            cadence_strength: 0.0,
            sample_count: n,
        };
    }
    let nf = n as f64;
    let amag: Vec<f64> = samples
        .iter()
        .map(|s| (s.ax * s.ax + s.ay * s.ay + s.az * s.az).sqrt())
        .collect();
    let gyro_energy: f64 = samples
        .iter()
        .map(|s| (s.gx * s.gx + s.gy * s.gy + s.gz * s.gz).sqrt())
        .sum::<f64>()
        / nf;

    // Accel AC: remove the DC (~gravity) then RMS.
    let mean: f64 = amag.iter().sum::<f64>() / nf;
    let ac: Vec<f64> = amag.iter().map(|v| v - mean).collect();
    let accel_energy = (ac.iter().map(|v| v * v).sum::<f64>() / nf).sqrt();

    // Jerk: RMS of |accel| first difference.
    let mut jerk_sq = 0.0;
    for i in 1..n {
        let d = amag[i] - amag[i - 1];
        jerk_sq += d * d;
    }
    let jerk = (jerk_sq / (n as f64 - 1.0)).sqrt();

    // Cadence: normalized autocorrelation over the gait-band lags; strongest peak's freq + strength.
    // Strength = peak ACF / zero-lag ACF (0..1), so it is amplitude-scale free.
    let ac0: f64 = ac.iter().map(|v| v * v).sum();
    let mut best_freq: Option<f64> = None;
    let mut best_strength = 0.0;
    if ac0 > 0.0 {
        let rate = sample_rate_hz as f64;
        let lo_lag = ((rate / CADENCE_BAND_HI_HZ).round() as i64).max(1);
        let hi_lag = ((rate / CADENCE_BAND_LO_HZ).round() as i64).min(n as i64 - 1);
        if lo_lag < hi_lag {
            for lag in lo_lag..=hi_lag {
                let mut s = 0.0;
                let l = lag as usize;
                for i in 0..(n - l) {
                    s += ac[i] * ac[i + l];
                }
                let strength = s / ac0;
                if strength > best_strength {
                    best_strength = strength;
                    best_freq = Some(rate / lag as f64);
                }
            }
        }
    }
    let cadence = if best_strength >= MIN_CADENCE_STRENGTH {
        best_freq
    } else {
        None
    };
    ImuActivityFeatures {
        accel_energy_g: accel_energy,
        gyro_energy_dps: gyro_energy,
        jerk_rms: jerk,
        cadence_hz: cadence,
        cadence_strength: best_strength.max(0.0),
        sample_count: n,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    // Parity gate: a synthetic gait generator whose injected cadence the extractor must recover —
    // MULTIPLE distinct rates, so a lucky single match cannot pass.
    fn gait_samples(cadence: f64, amp: f64, seconds: f64, gyro_dps: f64) -> Vec<ImuSample> {
        let rate = 100.0;
        let n = (seconds * rate) as usize;
        (0..n)
            .map(|i| {
                let az = 1.0 + amp * (2.0 * PI * cadence * i as f64 / rate).sin();
                ImuSample {
                    ax: 0.0,
                    ay: 0.0,
                    az,
                    gx: gyro_dps,
                    gy: 0.0,
                    gz: 0.0,
                }
            })
            .collect()
    }

    #[test]
    fn recovers_multiple_distinct_cadences() {
        for target in [1.4, 1.8, 2.4, 3.0] {
            let f = extract(&gait_samples(target, 0.2, 6.0, 0.0), 100);
            let c = f
                .cadence_hz
                .unwrap_or_else(|| panic!("cadence {target} Hz should be detected"));
            assert!(
                (c - target).abs() <= 0.15,
                "recovered {c} vs injected {target}"
            );
            assert!(
                f.cadence_strength > 0.4,
                "clean sinusoid should read rhythmic"
            );
        }
    }

    #[test]
    fn still_has_no_cadence_and_low_energy() {
        let still: Vec<ImuSample> = (0..600)
            .map(|_| ImuSample {
                ax: 0.0,
                ay: 0.0,
                az: 1.0,
                gx: 0.0,
                gy: 0.0,
                gz: 0.0,
            })
            .collect();
        let f = extract(&still, 100);
        assert!(f.cadence_hz.is_none(), "still wrist has no gait cadence");
        assert!(f.accel_energy_g < 0.01);
        assert!(f.gyro_energy_dps < 0.01);
    }

    #[test]
    fn energy_and_jerk_rise_with_motion() {
        let still = extract(&gait_samples(2.0, 0.0, 6.0, 0.0), 100);
        let moving = extract(&gait_samples(2.0, 0.3, 6.0, 0.0), 100);
        assert!(moving.accel_energy_g > still.accel_energy_g + 0.05);
        assert!(moving.jerk_rms > still.jerk_rms);
    }

    #[test]
    fn gyro_energy_reflects_rotation() {
        let f = extract(&gait_samples(2.0, 0.2, 6.0, 45.0), 100);
        assert!(
            (f.gyro_energy_dps - 45.0).abs() <= 1.0,
            "constant 45 dps should read back"
        );
    }

    /// Guards the on-chip ENMO channel against being fed here for cadence: at 1 Hz the gait-band lag window
    /// collapses, so a genuinely varying signal still reports no rhythm rather than an aliased one.
    #[test]
    fn a_one_hz_source_reports_no_cadence() {
        let enmo: Vec<ImuSample> = (0..600)
            .map(|i| ImuSample {
                ax: 0.0,
                ay: 0.0,
                az: 1.0 + 0.2 * (i as f64 * 0.7).sin(),
                gx: 0.0,
                gy: 0.0,
                gz: 0.0,
            })
            .collect();
        let f = extract(&enmo, 1);
        assert!(
            f.accel_energy_g > 0.05,
            "the input must vary, or this passes for the wrong reason"
        );
        assert!(
            f.cadence_hz.is_none(),
            "0.5 Hz Nyquist cannot carry a 1.2-3.5 Hz cadence"
        );
        assert_eq!(f.cadence_strength, 0.0);
    }

    #[test]
    fn too_few_samples_return_empty() {
        let f = extract(&gait_samples(2.0, 0.2, 0.05, 0.0), 100);
        assert!(f.cadence_hz.is_none());
        assert_eq!(f.cadence_strength, 0.0);
    }
}
