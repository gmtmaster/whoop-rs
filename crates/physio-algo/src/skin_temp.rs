//! Skin temperature: family-aware raw → Celsius conversion, and the nightly worn-filter pipeline
//! that reduces a night of raw samples to a single mean. Ported from the agreed Swift skin-temperature
//! funnel semantics. Pure and deterministic: absent or implausible signal returns `None`, never 0.

/// Which device family a raw skin-temperature sample came from. The raw domain and its worn band
/// are device-specific; the funnel converts each family into Celsius on its own terms.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceFamily {
    /// WHOOP 4.0: raw units on a provisional, Swift-compatible linear calibration.
    Whoop4,
    /// WHOOP 5.0 / MG: raw units are centi-degrees Celsius (`raw / 100.0 == degC`).
    Whoop5Mg,
}

/// WHOOP4's provisional raw → Celsius calibration, kept bit-for-bit compatible with the Swift
/// implementation it was ported from. This is a provisional linear fit, not new calibration science:
/// anchor raw 826 -> 33.0 C, slope 0.05 C per raw unit, worn only inside the 550..=2040 raw band.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Whoop4Calibration {
    pub anchor_raw: f64,
    pub anchor_c: f64,
    pub slope_c_per_raw: f64,
    pub worn_raw_lo: f64,
    pub worn_raw_hi: f64,
}

impl Whoop4Calibration {
    /// The provisional Swift-compatible calibration: anchor raw 826 -> 33.0 C, slope 0.05 C/raw,
    /// worn raw band 550..=2040.
    pub fn provisional() -> Self {
        Self { anchor_raw: 826.0, anchor_c: 33.0, slope_c_per_raw: 0.05, worn_raw_lo: 550.0, worn_raw_hi: 2040.0 }
    }
}

impl Default for Whoop4Calibration {
    fn default() -> Self { Self::provisional() }
}

/// A converted skin temperature outside this band cannot be trusted, worn or not.
pub const CONVERTED_TEMP_MIN_C: f64 = 28.0;
pub const CONVERTED_TEMP_MAX_C: f64 = 42.0;

/// Concurrent heart rate must fall in this band for a sample to count as worn.
pub const MIN_HR_BPM: f64 = 30.0;
pub const MAX_HR_BPM: f64 = 220.0;

/// A night needs at least this many valid samples to report a mean at all.
pub const MIN_NIGHTLY_SAMPLES: usize = 300;

/// Convert one raw skin-temperature sample to Celsius, family aware.
///
/// WHOOP 5.0/MG: `raw / 100.0`, unconditionally — the raw domain has no separate worn band; the
/// converted-domain plausibility check downstream is what gates it.
///
/// WHOOP4: the raw-domain worn band is checked first (`None` outside `worn_raw_lo..=worn_raw_hi`),
/// then the provisional linear fit is applied.
///
/// Returns `None` for non-finite input or a WHOOP4 raw sample outside its worn band — never a
/// fabricated Celsius value.
pub fn skin_temp_celsius(raw: f64, family: DeviceFamily, whoop4_cal: &Whoop4Calibration) -> Option<f64> {
    if !raw.is_finite() {
        return None;
    }
    match family {
        DeviceFamily::Whoop5Mg => Some(raw / 100.0),
        DeviceFamily::Whoop4 => {
            if raw < whoop4_cal.worn_raw_lo || raw > whoop4_cal.worn_raw_hi {
                return None;
            }
            Some(whoop4_cal.anchor_c + (raw - whoop4_cal.anchor_raw) * whoop4_cal.slope_c_per_raw)
        }
    }
}

/// One raw skin-temperature sample plus the context the nightly funnel needs: when it was taken,
/// and the concurrent heart rate used to decide "worn".
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SkinTempSample {
    /// Sample timestamp, seconds since epoch (or any consistent origin shared with the sleep window).
    pub timestamp_secs: f64,
    /// Raw, unconverted sensor reading, in the units of `family`.
    pub raw: f64,
    /// Concurrent heart rate in bpm. `None` when no HR sample overlaps this reading.
    pub hr_bpm: Option<f64>,
}

/// Reduce one night's raw samples to a single nightly mean skin temperature in Celsius, or `None`
/// when the night cannot support one.
///
/// Pipeline, applied per sample in order:
/// 1. concurrent HR must be present and within `MIN_HR_BPM..=MAX_HR_BPM` ("worn")
/// 2. the sample's timestamp must fall inside `[sleep_start_secs, sleep_end_secs]`
/// 3. (WHOOP4 only) the raw-domain worn band is checked before conversion
/// 4. family-aware raw → Celsius conversion
/// 5. the converted temperature must fall inside `CONVERTED_TEMP_MIN_C..=CONVERTED_TEMP_MAX_C`
///
/// A night needs at least `MIN_NIGHTLY_SAMPLES` samples surviving every step; short of that, or with
/// no samples at all, this returns `None` rather than an unreliable or fabricated average. The
/// nightly result is the arithmetic mean of the surviving, converted samples.
pub fn nightly_mean(
    samples: &[SkinTempSample],
    sleep_start_secs: f64,
    sleep_end_secs: f64,
    family: DeviceFamily,
    whoop4_cal: &Whoop4Calibration,
) -> Option<f64> {
    let mut sum = 0.0_f64;
    let mut n = 0usize;

    for s in samples {
        let Some(hr) = s.hr_bpm else { continue };
        if !(hr.is_finite() && hr >= MIN_HR_BPM && hr <= MAX_HR_BPM) {
            continue;
        }
        if !(s.timestamp_secs >= sleep_start_secs && s.timestamp_secs <= sleep_end_secs) {
            continue;
        }
        let Some(c) = skin_temp_celsius(s.raw, family, whoop4_cal) else { continue };
        if !(c.is_finite() && c >= CONVERTED_TEMP_MIN_C && c <= CONVERTED_TEMP_MAX_C) {
            continue;
        }
        sum += c;
        n += 1;
    }

    if n < MIN_NIGHTLY_SAMPLES {
        return None;
    }
    Some(sum / n as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn whoop5_conversion_divides_by_one_hundred() {
        let cal = Whoop4Calibration::provisional();
        assert_eq!(skin_temp_celsius(3300.0, DeviceFamily::Whoop5Mg, &cal), Some(33.0));
        assert_eq!(skin_temp_celsius(2800.0, DeviceFamily::Whoop5Mg, &cal), Some(28.0));
        // WHOOP5/MG has no raw-domain worn band: an out-of-WHOOP4-band raw value still converts.
        assert_eq!(skin_temp_celsius(100.0, DeviceFamily::Whoop5Mg, &cal), Some(1.0));
    }

    #[test]
    fn whoop4_provisional_conversion_matches_the_agreed_calibration() {
        let cal = Whoop4Calibration::provisional();
        // Anchor: raw 826 -> 33.0 C exactly.
        assert_eq!(skin_temp_celsius(826.0, DeviceFamily::Whoop4, &cal), Some(33.0));
        // Slope: +100 raw -> +5.0 C.
        let got = skin_temp_celsius(926.0, DeviceFamily::Whoop4, &cal).unwrap();
        assert!((got - 38.0).abs() < 1e-9, "got {got}");
        // Slope the other direction: -100 raw -> -5.0 C.
        let got = skin_temp_celsius(726.0, DeviceFamily::Whoop4, &cal).unwrap();
        assert!((got - 28.0).abs() < 1e-9, "got {got}");
        // Raw-domain worn band: inclusive at both ends, rejected just outside.
        assert!(skin_temp_celsius(550.0, DeviceFamily::Whoop4, &cal).is_some());
        assert!(skin_temp_celsius(2040.0, DeviceFamily::Whoop4, &cal).is_some());
        assert_eq!(skin_temp_celsius(549.9, DeviceFamily::Whoop4, &cal), None);
        assert_eq!(skin_temp_celsius(2040.1, DeviceFamily::Whoop4, &cal), None);
    }

    fn worn_sample(t: f64) -> SkinTempSample {
        SkinTempSample { timestamp_secs: t, raw: 3300.0, hr_bpm: Some(60.0) }
    }

    #[test]
    fn valid_full_worn_night_produces_a_mean() {
        let samples: Vec<SkinTempSample> = (0..400).map(|i| worn_sample(i as f64)).collect();
        let mean = nightly_mean(&samples, 0.0, 1000.0, DeviceFamily::Whoop5Mg, &Whoop4Calibration::provisional());
        assert_eq!(mean, Some(33.0));
    }

    #[test]
    fn samples_outside_the_sleep_window_are_excluded() {
        let mut samples: Vec<SkinTempSample> = (0..350).map(|i| worn_sample(100.0 + i as f64)).collect();
        // 60 samples fall before the window and must not count.
        for i in 0..60 {
            samples.push(worn_sample(i as f64));
        }
        let n = samples.len();
        let mean = nightly_mean(&samples, 100.0, 449.0, DeviceFamily::Whoop5Mg, &Whoop4Calibration::provisional());
        assert_eq!(mean, Some(33.0), "only the 350 in-window samples should be averaged");
        assert_eq!(n, 410);
    }

    #[test]
    fn missing_or_invalid_hr_is_excluded() {
        let mut samples: Vec<SkinTempSample> = (0..300).map(|i| worn_sample(i as f64)).collect();
        // No HR at all.
        samples.push(SkinTempSample { timestamp_secs: 301.0, raw: 3300.0, hr_bpm: None });
        // HR outside 30..=220.
        samples.push(SkinTempSample { timestamp_secs: 302.0, raw: 3300.0, hr_bpm: Some(29.0) });
        samples.push(SkinTempSample { timestamp_secs: 303.0, raw: 3300.0, hr_bpm: Some(221.0) });
        let mean = nightly_mean(&samples, 0.0, 1000.0, DeviceFamily::Whoop5Mg, &Whoop4Calibration::provisional());
        // Only the original 300 worn samples survive.
        assert_eq!(mean, Some(33.0));
    }

    #[test]
    fn converted_out_of_range_temperature_is_excluded() {
        let mut samples: Vec<SkinTempSample> = (0..300).map(|i| worn_sample(i as f64)).collect();
        // Converts to 27.9 C, just under the floor.
        samples.push(SkinTempSample { timestamp_secs: 301.0, raw: 2790.0, hr_bpm: Some(60.0) });
        // Converts to 42.1 C, just over the ceiling.
        samples.push(SkinTempSample { timestamp_secs: 302.0, raw: 4210.0, hr_bpm: Some(60.0) });
        let mean = nightly_mean(&samples, 0.0, 1000.0, DeviceFamily::Whoop5Mg, &Whoop4Calibration::provisional());
        assert_eq!(mean, Some(33.0));
    }

    #[test]
    fn below_three_hundred_valid_samples_returns_none() {
        let samples: Vec<SkinTempSample> = (0..299).map(|i| worn_sample(i as f64)).collect();
        let mean = nightly_mean(&samples, 0.0, 1000.0, DeviceFamily::Whoop5Mg, &Whoop4Calibration::provisional());
        assert_eq!(mean, None);
    }

    #[test]
    fn exactly_three_hundred_valid_samples_returns_a_value() {
        let samples: Vec<SkinTempSample> = (0..300).map(|i| worn_sample(i as f64)).collect();
        let mean = nightly_mean(&samples, 0.0, 1000.0, DeviceFamily::Whoop5Mg, &Whoop4Calibration::provisional());
        assert_eq!(mean, Some(33.0));
    }

    #[test]
    fn empty_input_returns_none() {
        let mean = nightly_mean(&[], 0.0, 1000.0, DeviceFamily::Whoop5Mg, &Whoop4Calibration::provisional());
        assert_eq!(mean, None);
    }

    #[test]
    fn whoop4_raw_domain_worn_filter_applies_before_conversion() {
        let mut samples: Vec<SkinTempSample> = (0..300)
            .map(|i| SkinTempSample { timestamp_secs: i as f64, raw: 826.0, hr_bpm: Some(60.0) })
            .collect();
        // Off-wrist raw values, well outside the WHOOP4 worn band, must be dropped before conversion.
        for i in 0..50 {
            samples.push(SkinTempSample { timestamp_secs: 300.0 + i as f64, raw: 10.0, hr_bpm: Some(60.0) });
        }
        let mean = nightly_mean(&samples, 0.0, 1000.0, DeviceFamily::Whoop4, &Whoop4Calibration::provisional());
        assert_eq!(mean, Some(33.0));
    }
}
