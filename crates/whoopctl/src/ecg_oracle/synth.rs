//! Synthetic ECG at KNOWN dimensions — the input half of the renderer oracle.
//!
//! Samples are millivolts, never counts. Counts-per-mV is not known for this device, so a generator
//! that emitted counts would have to invent it; [`Signal::to_counts`] exists but demands the scale.
//!
//! What each signal measures on screen at 10 mm/mV and 25 mm/s:
//!
//! | signal | peak-to-peak | one cycle |
//! |---|---|---|
//! | `calibration_pulse` (1 mV, 1 Hz square) | 1.00 mV = 10 mm | 1.000 s = 25 mm |
//! | `square_wave(a, f)` | `a` mV = 10*a mm | 1/f s = 25/f mm |
//! | `pqrst(bpm, r)` | 1.25*`r` mV | 60/bpm s |
//!
//! `pqrst` is gaussian bumps, each as a fraction of `r` and an offset from the R peak:
//! P +0.15 @ -200 ms (sigma 40 ms), Q -0.10 @ -40 ms (10 ms), R +1.00 @ 0 (10 ms),
//! S -0.25 @ +40 ms (10 ms), T +0.30 @ +300 ms (80 ms). Cross-talk at the R peak is under 0.1%, so the
//! 1.25 peak-to-peak ratio holds to well under a percent; [`Signal::peak_to_peak_mv`] is the exact
//! figure and is what a test should assert against.

/// A sample series in millivolts, with the rate that produced it.
#[derive(Clone, Debug, PartialEq)]
pub struct Signal {
    pub sample_rate_hz: f64,
    pub samples_mv: Vec<f64>,
}

impl Signal {
    pub fn len(&self) -> usize {
        self.samples_mv.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples_mv.is_empty()
    }

    pub fn duration_s(&self) -> f64 {
        self.samples_mv.len() as f64 / self.sample_rate_hz
    }

    /// Largest minus smallest sample — the height the picture must depict.
    pub fn peak_to_peak_mv(&self) -> f64 {
        let (lo, hi) = self
            .samples_mv
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(l, h), v| (l.min(*v), h.max(*v)));
        if lo.is_finite() && hi.is_finite() {
            hi - lo
        } else {
            0.0
        }
    }

    /// Quantise to ADC counts at a SUPPLIED counts-per-mV. There is no default: the strap's conversion
    /// has not been read, and a guessed one draws a plausible, wrongly-scaled trace.
    pub fn to_counts(&self, counts_per_mv: f64) -> Vec<i32> {
        self.samples_mv.iter().map(|mv| (mv * counts_per_mv).round() as i32).collect()
    }
}

fn sample_count(sample_rate_hz: f64, duration_s: f64) -> usize {
    (sample_rate_hz * duration_s).round().max(0.0) as usize
}

/// A square wave starting LOW, 50% duty, baseline 0 mV, high `amplitude_mv` — so peak-to-peak IS
/// `amplitude_mv` and the first rising edge falls inside the frame at t = 0.5/`freq_hz`.
pub fn square_wave(sample_rate_hz: f64, duration_s: f64, amplitude_mv: f64, freq_hz: f64) -> Signal {
    let n = sample_count(sample_rate_hz, duration_s);
    let samples_mv = (0..n)
        .map(|i| {
            let phase = (i as f64 / sample_rate_hz * freq_hz).rem_euclid(1.0);
            if phase >= 0.5 {
                amplitude_mv
            } else {
                0.0
            }
        })
        .collect();
    Signal { sample_rate_hz, samples_mv }
}

/// The 1 mV / 1 Hz calibration pulse: 10 mm tall and 25 mm per cycle at 10 mm/mV and 25 mm/s.
pub fn calibration_pulse(sample_rate_hz: f64, duration_s: f64) -> Signal {
    square_wave(sample_rate_hz, duration_s, 1.0, 1.0)
}

/// A flat line at `level_mv` — the baseline condition.
pub fn flat(sample_rate_hz: f64, duration_s: f64, level_mv: f64) -> Signal {
    Signal { sample_rate_hz, samples_mv: vec![level_mv; sample_count(sample_rate_hz, duration_s)] }
}

/// Each wave of the synthetic beat: amplitude as a fraction of R, offset from the R peak, sigma.
const WAVES: [(f64, f64, f64); 5] =
    [(0.15, -0.200, 0.040), (-0.10, -0.040, 0.010), (1.00, 0.000, 0.010), (-0.25, 0.040, 0.010), (0.30, 0.300, 0.080)];

/// The first R peak, far enough in that the P wave of beat 0 fits inside the frame.
const FIRST_R_S: f64 = 0.4;

/// A synthetic PQRST at `bpm` with an R peak of `r_amplitude_mv`. Not physiological — it exists to
/// have KNOWN dimensions; see the module header for each wave's amplitude.
pub fn pqrst(sample_rate_hz: f64, duration_s: f64, bpm: f64, r_amplitude_mv: f64) -> Signal {
    let n = sample_count(sample_rate_hz, duration_s);
    let beat_s = 60.0 / bpm;
    let samples_mv = (0..n)
        .map(|i| {
            let t = i as f64 / sample_rate_hz;
            let k = ((t - FIRST_R_S) / beat_s).round();
            // Only the nearest beats can contribute; the bumps die inside half a beat.
            [k - 1.0, k, k + 1.0]
                .iter()
                .map(|kk| {
                    let r_at = FIRST_R_S + kk * beat_s;
                    WAVES
                        .iter()
                        .map(|(amp, off, sigma)| {
                            let z = (t - (r_at + off)) / sigma;
                            amp * r_amplitude_mv * (-0.5 * z * z).exp()
                        })
                        .sum::<f64>()
                })
                .sum()
        })
        .collect();
    Signal { sample_rate_hz, samples_mv }
}

/// Gaussian noise of a stated RMS on a flat baseline — the no-contact condition, where there is no
/// signal to define a ratio against.
pub fn noise_baseline(sample_rate_hz: f64, duration_s: f64, rms_mv: f64, seed: u64) -> Signal {
    let mut rng = Rng::new(seed);
    let n = sample_count(sample_rate_hz, duration_s);
    Signal { sample_rate_hz, samples_mv: (0..n).map(|_| rng.normal() * rms_mv).collect() }
}

/// Add gaussian noise at `snr_db` relative to the base's own RMS.
///
/// `None` when the base carries no power: an SNR against a flat line is undefined, and a silent
/// fallback would invent the noise level it was asked to derive.
pub fn add_noise_snr(base: &Signal, snr_db: f64, seed: u64) -> Option<Signal> {
    let signal_rms = rms(&base.samples_mv);
    if signal_rms <= 0.0 {
        return None;
    }
    let noise_rms = signal_rms / 10f64.powf(snr_db / 20.0);
    let mut rng = Rng::new(seed);
    Some(Signal {
        sample_rate_hz: base.sample_rate_hz,
        samples_mv: base.samples_mv.iter().map(|v| v + rng.normal() * noise_rms).collect(),
    })
}

fn rms(v: &[f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    (v.iter().map(|x| x * x).sum::<f64>() / v.len() as f64).sqrt()
}

/// One span of an intermittent-contact signal, in samples (end exclusive) and seconds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ContactSpan {
    pub contact: bool,
    pub start_sample: usize,
    pub end_sample: usize,
    pub start_s: f64,
    pub end_s: f64,
}

/// An intermittent-contact signal and the switch points that produced it.
#[derive(Clone, Debug, PartialEq)]
pub struct Intermittent {
    pub signal: Signal,
    pub spans: Vec<ContactSpan>,
}

/// Alternate `contact_s` of `trace` with `noise_s` of noise at `noise_rms_mv`, starting in contact.
/// The spans come back alongside so a test can assert the renderer marked the right columns.
pub fn intermittent_contact(
    trace: &Signal,
    contact_s: f64,
    noise_s: f64,
    noise_rms_mv: f64,
    seed: u64,
) -> Intermittent {
    let rate = trace.sample_rate_hz;
    let contact_n = sample_count(rate, contact_s).max(1);
    let noise_n = sample_count(rate, noise_s).max(1);
    let mut rng = Rng::new(seed);
    let (mut samples_mv, mut spans) = (Vec::with_capacity(trace.len()), Vec::new());
    let (mut i, mut contact) = (0usize, true);
    while i < trace.len() {
        let run = (if contact { contact_n } else { noise_n }).min(trace.len() - i);
        for j in i..i + run {
            samples_mv.push(if contact { trace.samples_mv[j] } else { rng.normal() * noise_rms_mv });
        }
        spans.push(ContactSpan {
            contact,
            start_sample: i,
            end_sample: i + run,
            start_s: i as f64 / rate,
            end_s: (i + run) as f64 / rate,
        });
        i += run;
        contact = !contact;
    }
    Intermittent { signal: Signal { sample_rate_hz: rate, samples_mv }, spans }
}

/// Deterministic SplitMix64 — reproducible noise with no dependency, so a seed names a waveform.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1).
    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Standard normal, by Box-Muller.
    pub fn normal(&mut self) -> f64 {
        let u1 = self.unit().max(f64::MIN_POSITIVE);
        let u2 = self.unit();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}
