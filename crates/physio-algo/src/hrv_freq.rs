//! Frequency-domain HRV (LF / HF / LF-HF / total power) via the Lomb-Scargle periodogram over the R-R
//! tachogram. Lomb-Scargle estimates the spectrum directly from the irregular tachogram (no resampling),
//! the recommended estimator for HRV on uneven samples. Task Force (1996) bands. Approximate, non-clinical.

use std::f64::consts::PI;

use crate::hrv::HrvReadiness;

// Task Force (1996) band edges (Hz).
const VLF_LOW_HZ: f64 = 0.0033;
const LF_LOW_HZ: f64 = 0.04;
const LF_HIGH_HZ: f64 = 0.15;
const HF_LOW_HZ: f64 = 0.15;
const HF_HIGH_HZ: f64 = 0.40;
/// HF needs >= 60 s of R-R span; LF (and LF/HF, total power) need >= 250 s.
const MIN_SPAN_FOR_HF_SEC: f64 = 60.0;
const MIN_SPAN_FOR_LF_SEC: f64 = 250.0;
const MIN_BEATS: usize = 20;
/// Frequency grid step (Hz) for the trapezoidal band integral.
const FREQ_STEP_HZ: f64 = 0.005;

/// Frequency-domain HRV bands, powers in ms². `lf` / `lfhf` are `None` when the span is < 250 s (a
/// 60..250 s window yields HF only); `hf` and `total_power` are present whenever the result is `Some`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HrvBands {
    pub lf: Option<f64>,
    pub hf: f64,
    pub lfhf: Option<f64>,
    pub total_power: f64,
}

/// Frequency-domain HRV from a time-ordered R-R series (ms). Range + Malik-ectopic cleaned before the
/// tachogram is built. `None` when fewer than [`MIN_BEATS`] clean beats or the span is under
/// [`MIN_SPAN_FOR_HF_SEC`].
pub fn freq_domain(rr_ms: &[u16]) -> Option<HrvBands> {
    let clean = HrvReadiness::clean_rr(rr_ms);
    if clean.len() < MIN_BEATS {
        return None;
    }
    // Tachogram: time of beat k = cumulative sum of the first k clean intervals (s); sample = R-R.
    let mut times = vec![0.0f64; clean.len()];
    let mut acc = 0.0;
    for (i, &rr) in clean.iter().enumerate() {
        times[i] = acc / 1000.0;
        acc += rr as f64;
    }
    let span = times[times.len() - 1] - times[0];
    if span < MIN_SPAN_FOR_HF_SEC {
        return None;
    }
    let mean = clean.iter().map(|&v| v as f64).sum::<f64>() / clean.len() as f64;
    let y: Vec<f64> = clean.iter().map(|&v| v as f64 - mean).collect();

    let hf = band_power(&times, &y, HF_LOW_HZ, HF_HIGH_HZ);
    let lf_trusted = span >= MIN_SPAN_FOR_LF_SEC;
    let lf = lf_trusted.then(|| band_power(&times, &y, LF_LOW_HZ, LF_HIGH_HZ));
    let lfhf = match lf {
        Some(l) if hf > 0.0 => Some(l / hf),
        _ => None,
    };
    // Sum the sub-band integrals (VLF + LF + HF) so total_power >= hf and stays grid-consistent with the
    // reported bands; a single wide integral samples on an offset grid and can undercount a narrow peak.
    let total_power = match lf {
        Some(l) => band_power(&times, &y, VLF_LOW_HZ, LF_LOW_HZ) + l + hf,
        None => hf,
    };
    Some(HrvBands {
        lf,
        hf,
        lfhf,
        total_power,
    })
}

/// Trapezoidal integral of the Lomb-Scargle power across `[f_low, f_high]`, stepped by [`FREQ_STEP_HZ`].
fn band_power(times: &[f64], y: &[f64], f_low: f64, f_high: f64) -> f64 {
    if f_high <= f_low {
        return 0.0;
    }
    let variance = y.iter().map(|v| v * v).sum::<f64>() / y.len() as f64;
    if variance <= 0.0 {
        return 0.0;
    }
    let (mut power, mut prev_p, mut prev_f, mut first, mut f) = (0.0, 0.0, f_low, true, f_low);
    while f <= f_high + 1e-12 {
        let p = lomb_scargle_power(times, y, f, variance);
        if !first {
            power += 0.5 * (p + prev_p) * (f - prev_f);
        }
        prev_p = p;
        prev_f = f;
        first = false;
        f += FREQ_STEP_HZ;
    }
    power
}

/// Lomb-Scargle normalised power at one frequency (Press et al. form). `variance` is the sample variance of
/// the mean-removed series. The time offset tau makes the estimate invariant to time translation, which is
/// what correctly handles the uneven tachogram spacing.
fn lomb_scargle_power(times: &[f64], y: &[f64], freq_hz: f64, variance: f64) -> f64 {
    let omega = 2.0 * PI * freq_hz;
    let (mut sin2, mut cos2) = (0.0, 0.0);
    for &t in times {
        let a = 2.0 * omega * t;
        sin2 += a.sin();
        cos2 += a.cos();
    }
    let tau = sin2.atan2(cos2) / (2.0 * omega);
    let (mut c_term, mut c_den, mut s_term, mut s_den) = (0.0, 0.0, 0.0, 0.0);
    for i in 0..times.len() {
        let arg = omega * (times[i] - tau);
        let c = arg.cos();
        let s = arg.sin();
        c_term += y[i] * c;
        c_den += c * c;
        s_term += y[i] * s;
        s_den += s * s;
    }
    let cos_part = if c_den > 0.0 {
        c_term * c_term / c_den
    } else {
        0.0
    };
    let sin_part = if s_den > 0.0 {
        s_term * s_term / s_den
    } else {
        0.0
    };
    (cos_part + sin_part) / (2.0 * variance)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tachogram with a strong ~0.25 Hz (HF/respiratory) modulation over `secs` seconds of beats.
    fn resp_night(secs: f64, resp_hz: f64) -> Vec<u16> {
        let mut rr = Vec::new();
        let mut t = 0.0;
        while t < secs {
            let ms = 900.0 + 40.0 * (2.0 * PI * resp_hz * t).sin();
            let v = ms.round() as u16;
            rr.push(v);
            t += v as f64 / 1000.0;
        }
        rr
    }

    #[test]
    fn hf_only_below_the_lf_span_gate() {
        // ~90 s window: past the 60 s HF gate, under the 250 s LF gate.
        let b = freq_domain(&resp_night(90.0, 0.25)).unwrap();
        assert!(b.lf.is_none(), "LF must be gated under 250 s");
        assert!(b.lfhf.is_none());
        assert!(b.hf > 0.0);
        assert_eq!(b.total_power, b.hf); // HF-only -> total is HF
    }

    /// How many of the four presence/ordering claims a band set satisfies, past the LF span gate.
    /// These are shape claims only: none of them compares a band power to a target.
    fn presence_claims(b: Option<HrvBands>) -> usize {
        let Some(b) = b else { return 0 };
        usize::from(b.lf.is_some())
            + usize::from(b.lfhf.is_some())
            + usize::from(b.total_power >= b.hf)
            + usize::from(b.lf.is_some_and(|lf| b.hf > lf))
    }

    #[test]
    fn lf_and_total_present_past_the_lf_gate() {
        // The respiratory power sits in HF, so HF dominates LF for a 0.25 Hz drive.
        assert_eq!(presence_claims(freq_domain(&resp_night(300.0, 0.25))), 4);
    }

    /// A tachogram carrying an LF tone at 0.10 Hz and an HF tone at 0.25 Hz. A sinusoid of amplitude
    /// `a` contributes `a^2/2` to the variance, so the planted LF/HF power ratio is `(lf_amp/hf_amp)^2`.
    fn two_tone(secs: f64, lf_amp: f64, hf_amp: f64) -> Vec<u16> {
        let mut rr = Vec::new();
        let mut t = 0.0;
        while t < secs {
            let ms =
                900.0 + lf_amp * (2.0 * PI * 0.10 * t).sin() + hf_amp * (2.0 * PI * 0.25 * t).sin();
            let v = ms.round() as u16;
            rr.push(v);
            t += v as f64 / 1000.0;
        }
        rr
    }

    /// `(lf_amp, hf_amp)` pairs planting LF/HF ratios of 1, 4, 1/4, 16 and 1/16 over a 600 s tachogram.
    const TONE_PAIRS: [(f64, f64); 5] = [
        (40.0, 40.0),
        (60.0, 30.0),
        (30.0, 60.0),
        (80.0, 20.0),
        (20.0, 80.0),
    ];
    /// Worst measured relative error over `TONE_PAIRS` is 5.50%, at the planted ratio of 16.
    const LFHF_REL_TOL: f64 = 0.08;

    /// Planted ratios a scorer fails to recover, as `(planted, returned)`. Empty = it tracks.
    fn ratio_misses(scorer: &dyn Fn(&[u16]) -> Option<HrvBands>) -> Vec<(f64, Option<f64>)> {
        let mut bad = Vec::new();
        for (lf_amp, hf_amp) in TONE_PAIRS {
            let planted = (lf_amp / hf_amp).powi(2);
            let got = scorer(&two_tone(600.0, lf_amp, hf_amp)).and_then(|b| b.lfhf);
            if !got.is_some_and(|v| (v / planted - 1.0).abs() <= LFHF_REL_TOL) {
                bad.push((planted, got));
            }
        }
        bad
    }

    /// LF/HF recovers a planted power ratio over a 256-fold range. The ratio is the one band number the
    /// Lomb-Scargle normalisation cancels out of, so it is comparable to a planted truth.
    #[test]
    fn lfhf_recovers_a_planted_power_ratio() {
        assert!(
            ratio_misses(&freq_domain).is_empty(),
            "{:?}",
            ratio_misses(&freq_domain)
        );
    }

    /// The null arm. A scorer that keeps the shipped span/beat gating but returns fixed magnitudes
    /// satisfies every presence/ordering claim in this file and recovers no planted ratio at all.
    #[test]
    fn a_correctly_gated_constant_passes_presence_and_fails_the_ratios() {
        let gated_constant = |rr: &[u16]| {
            freq_domain(rr).map(|b| HrvBands {
                lf: b.lf.map(|_| 0.5),
                hf: 1.0,
                lfhf: b.lfhf.map(|_| 0.5),
                total_power: if b.lf.is_some() { 1.5 } else { 1.0 },
            })
        };
        // It clears both shape gates in this file.
        assert_eq!(presence_claims(gated_constant(&resp_night(300.0, 0.25))), 4);
        let short = gated_constant(&resp_night(90.0, 0.25)).unwrap();
        assert!(short.lf.is_none() && short.lfhf.is_none() && short.total_power == short.hf);
        assert!(gated_constant(&resp_night(30.0, 0.25)).is_none());
        // And it recovers nothing.
        assert_eq!(ratio_misses(&gated_constant).len(), TONE_PAIRS.len());
        for lfhf in [0.0625f64, 0.25, 1.0, 4.0, 16.0] {
            let c = HrvBands {
                lf: Some(lfhf),
                hf: 1.0,
                lfhf: Some(lfhf),
                total_power: 2.0 + lfhf,
            };
            assert!(
                !ratio_misses(&|_| Some(c)).is_empty(),
                "constant lfhf {lfhf} passed"
            );
        }
        assert!(
            !ratio_misses(&|_| None).is_empty(),
            "a refusing scorer passed the ratio sweep"
        );
    }

    /// A tone in one band stays in that band: a lone LF tone leaves HF three orders down, and the
    /// reverse. This is what separates band assignment from a scorer that splits power evenly.
    #[test]
    fn a_lone_tone_lands_in_its_own_band() {
        let lf_only = freq_domain(&two_tone(600.0, 40.0, 0.0)).unwrap();
        assert!(
            lf_only.lf.unwrap() / lf_only.hf > 1000.0,
            "lf-only leaked: {lf_only:?}"
        );
        let hf_only = freq_domain(&two_tone(600.0, 0.0, 40.0)).unwrap();
        assert!(
            hf_only.hf / hf_only.lf.unwrap() > 1000.0,
            "hf-only leaked: {hf_only:?}"
        );
    }

    /// Recorded defect, not a desired property: the band integrals are normalised Lomb-Scargle power,
    /// not ms^2, so doubling the record doubles them. Only `lfhf` is comparable to a planted truth.
    #[test]
    fn band_powers_scale_with_record_length_and_are_not_ms_squared() {
        let short = freq_domain(&two_tone(600.0, 40.0, 40.0)).unwrap();
        let long = freq_domain(&two_tone(1200.0, 40.0, 40.0)).unwrap();
        assert!(
            (long.total_power / short.total_power - 2.0).abs() < 1e-3,
            "{short:?} vs {long:?}"
        );
        // A 40 ms tone carries 800 ms^2; the reported HF is ~1000x below that.
        assert!(
            short.hf < 1.0 && 800.0 / short.hf > 900.0,
            "hf {} is not ~1000x under 800",
            short.hf
        );
        // The ratio is unaffected, which is why the gate above can use it.
        assert!((long.lfhf.unwrap() / short.lfhf.unwrap() - 1.0).abs() < 0.01);
    }

    #[test]
    fn too_short_or_too_few_is_none() {
        assert!(freq_domain(&resp_night(30.0, 0.25)).is_none()); // span < 60 s
        assert!(freq_domain(&[800, 810, 820]).is_none()); // < MIN_BEATS
    }
}
