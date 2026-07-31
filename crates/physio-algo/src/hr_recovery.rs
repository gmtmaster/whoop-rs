//! Heart-rate recovery after a sufficiently intense workout: the bpm drop 1/2/5 minutes past the end of
//! a bout that sustained a high-intensity effort. Eligibility needs a sustained >=70% max-HR effort; each
//! reading is `end_hr` minus the median bpm in a +/-15s window. Absent coverage stays `None` (never
//! interpolated across a gap, never turned into zero). Wellness estimate, never medical.

use crate::hr_sample::HrSample;

/// Recovery deltas after a bout: `end_hr` minus the median bpm at +1/+2/+5 minutes. A reading with too
/// little post-workout coverage is `None`; a HR *rise* stays signed (negative), never clamped to zero.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HrRecovery {
    pub end_hr: i32,
    pub after_1min: Option<i32>,
    pub after_2min: Option<i32>,
    pub after_5min: Option<i32>,
}

impl HrRecovery {
    /// True when at least one of the 1/2/5-minute readings landed.
    pub fn has_measurement(&self) -> bool {
        self.after_1min.is_some() || self.after_2min.is_some() || self.after_5min.is_some()
    }
}

const ELIGIBILITY_FRACTION_OF_MAX_HR: f64 = 0.70;
const MINIMUM_HIGH_INTENSITY_SECONDS: i64 = 120;
const ELIGIBILITY_LOOKBACK_SECONDS: i64 = 300;
const CESSATION_WINDOW_SECONDS: i64 = 30;
const MEASUREMENT_TOLERANCE_SECONDS: i64 = 15;
const MINIMUM_SAMPLES_PER_READING: usize = 3;
const MAXIMUM_CONTINUOUS_GAP_SECONDS: i64 = 10;

/// HR recovery for a bout `[workout_start, workout_end]` against `max_hr`. `None` when the bout is
/// invalid, was not a sustained high-intensity effort, or has too few samples around the readings.
pub fn calculate(samples: &[HrSample], workout_start: i64, workout_end: i64, max_hr: f64) -> Option<HrRecovery> {
    if workout_start <= 0 || workout_end <= workout_start || max_hr <= 0.0 {
        return None;
    }
    let lower_bound = workout_start.max(workout_end - ELIGIBILITY_LOOKBACK_SECONDS);
    let upper_bound = workout_end + 5 * 60 + MEASUREMENT_TOLERANCE_SECONDS;
    let mut sorted: Vec<HrSample> = samples
        .iter()
        .copied()
        .filter(|s| s.ts >= lower_bound && s.ts <= upper_bound && (30..=250).contains(&s.bpm))
        .collect();
    // Ties broken by bpm so the sustained scan and the cessation max are deterministic.
    sorted.sort_by(|a, b| a.ts.cmp(&b.ts).then(a.bpm.cmp(&b.bpm)));
    if sorted.len() < MINIMUM_SAMPLES_PER_READING {
        return None;
    }

    let before_end: Vec<HrSample> = sorted.iter().copied().filter(|s| s.ts <= workout_end).collect();
    let threshold = max_hr * ELIGIBILITY_FRACTION_OF_MAX_HR;
    if sustained_seconds(threshold, &before_end) < MINIMUM_HIGH_INTENSITY_SECONDS {
        return None;
    }

    let cessation: Vec<i32> = before_end
        .iter()
        .filter(|s| s.ts >= workout_end - CESSATION_WINDOW_SECONDS)
        .map(|s| s.bpm)
        .collect();
    if cessation.len() < MINIMUM_SAMPLES_PER_READING {
        return None;
    }
    let end_hr = *cessation.iter().max()?;

    let recovery = |minutes: i64| -> Option<i32> {
        let target = workout_end + minutes * 60;
        let values: Vec<i32> = sorted
            .iter()
            .filter(|s| (s.ts - target).abs() <= MEASUREMENT_TOLERANCE_SECONDS)
            .map(|s| s.bpm)
            .collect();
        if values.len() < MINIMUM_SAMPLES_PER_READING {
            return None;
        }
        Some(end_hr - median(&values)?)
    };

    let result = HrRecovery {
        end_hr,
        after_1min: recovery(1),
        after_2min: recovery(2),
        after_5min: recovery(5),
    };
    result.has_measurement().then_some(result)
}

/// Seconds of contiguous samples (gap 1..=10s) at or above `threshold` bpm. A gap wider than the cap
/// breaks the streak, so a disconnected high-HR fragment never counts toward eligibility.
fn sustained_seconds(threshold: f64, samples: &[HrSample]) -> i64 {
    if samples.len() < 2 {
        return 0;
    }
    let mut seconds = 0i64;
    for i in 0..samples.len() - 1 {
        let gap = samples[i + 1].ts - samples[i].ts;
        if (1..=MAXIMUM_CONTINUOUS_GAP_SECONDS).contains(&gap) && samples[i].bpm as f64 >= threshold {
            seconds += gap;
        }
    }
    seconds
}

/// Integer median; even counts round the mean half-up.
fn median(values: &[i32]) -> Option<i32> {
    if values.is_empty() {
        return None;
    }
    let mut s = values.to_vec();
    s.sort_unstable();
    let mid = s.len() / 2;
    Some(if s.len().is_multiple_of(2) {
        ((s[mid - 1] + s[mid]) as f64 / 2.0 + 0.5).floor() as i32
    } else {
        s[mid]
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const END: i64 = 10_000;

    fn s(ts: i64, bpm: i32) -> HrSample {
        HrSample { ts, bpm }
    }

    /// Dense 1 Hz coverage over the 5-min lookback: 145 bpm (above the 140 threshold at max 200), then
    /// the last 30s at `end_hr` (the cessation reading).
    fn dense_eligible(end_hr: i32) -> Vec<HrSample> {
        (END - 300..=END).map(|ts| s(ts, if ts >= END - 30 { end_hr } else { 145 })).collect()
    }

    /// A cluster of `values` centred on `END + minutes*60`, one per second.
    fn window(minutes: i64, values: &[i32]) -> Vec<HrSample> {
        let target = END + minutes * 60;
        values
            .iter()
            .enumerate()
            .map(|(i, &bpm)| s(target - values.len() as i64 / 2 + i as i64, bpm))
            .collect()
    }

    fn concat(parts: &[Vec<HrSample>]) -> Vec<HrSample> {
        parts.iter().flatten().copied().collect()
    }

    #[test]
    fn calculates_one_two_and_five_minute_drops_from_robust_readings() {
        let samples = concat(&[
            dense_eligible(170),
            window(1, &[146, 146, 220, 146, 146]),
            window(2, &[132, 132, 132]),
            window(5, &[112, 112, 112]),
        ]);
        assert_eq!(
            Some(HrRecovery { end_hr: 170, after_1min: Some(24), after_2min: Some(38), after_5min: Some(58) }),
            calculate(&samples, END - 300, END, 200.0),
        );
    }

    #[test]
    fn requires_sustained_high_intensity_rather_than_one_peak() {
        let mut samples: Vec<HrSample> = (END - 300..=END).map(|t| s(t, 120)).collect();
        samples.push(s(END, 190));
        samples.extend(window(1, &[140, 140, 140]));
        assert_eq!(None, calculate(&samples, END - 300, END, 200.0));
    }

    #[test]
    fn rejects_disconnected_high_intensity_fragments() {
        let mut sparse: Vec<HrSample> = (END - 300..=END).step_by(15).map(|t| s(t, 170)).collect();
        sparse.extend(window(1, &[140, 140, 140]));
        assert_eq!(None, calculate(&sparse, END - 300, END, 200.0));
    }

    #[test]
    fn does_not_credit_pre_workout_heart_rate_toward_eligibility() {
        let samples = concat(&[dense_eligible(170), window(1, &[140, 140, 140])]);
        assert_eq!(None, calculate(&samples, END - 60, END, 200.0));
    }

    #[test]
    fn returns_only_measurements_with_real_coverage() {
        let samples = concat(&[dense_eligible(170), window(1, &[150, 150, 150]), window(5, &[110, 110])]);
        assert_eq!(
            Some(HrRecovery { end_hr: 170, after_1min: Some(20), after_2min: None, after_5min: None }),
            calculate(&samples, END - 300, END, 200.0),
        );
    }

    #[test]
    fn no_post_workout_coverage_returns_null() {
        assert_eq!(None, calculate(&dense_eligible(170), END - 300, END, 200.0));
    }

    #[test]
    fn a_heart_rate_rise_remains_signed_instead_of_being_clamped() {
        let samples = concat(&[dense_eligible(160), window(1, &[165, 165, 165])]);
        let r = calculate(&samples, END - 300, END, 200.0).unwrap();
        assert_eq!(Some(-5), r.after_1min);
    }
}
