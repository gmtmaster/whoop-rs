//! Longer-horizon wellness: daily and daytime stress, circadian phase, vitality and the
//! series helpers they share.

use crate::*;

/// One day's RHR + HRV for daily-stress scoring (either may be absent).
#[derive(uniffi::Record)]
pub struct StressDayInfo {
    pub rhr: Option<f64>,
    pub hrv: Option<f64>,
}

/// Daily autonomic stress (0–3) from today's RHR + HRV against the prior-days baseline. `None` on too few
/// baseline days or no signal today.
#[uniffi::export]
pub fn daily_stress(today: StressDayInfo, baseline: Vec<StressDayInfo>) -> Option<f64> {
    let b: Vec<stress::StressDay> = baseline.into_iter().map(|d| stress::StressDay { rhr: d.rhr, hrv: d.hrv }).collect();
    stress::daily_stress(stress::StressDay { rhr: today.rhr, hrv: today.hrv }, &b)
}

// ── Daytime stress (intraday autonomic activation) ─────────────────────────

/// One hour's aggregates: local `hour` (0–23), mean HR (None below the sample gate) and RMSSD (None on
/// insufficient clean R-R).
#[derive(uniffi::Record)]
pub struct HourPointInfo {
    pub hour: i32,
    pub mean_hr: Option<f64>,
    pub rmssd: Option<f64>,
}

/// One scored waking hour.
#[derive(uniffi::Record)]
pub struct ScoredHourInfo {
    pub hour: i32,
    pub mean_hr: f64,
    pub rmssd: Option<f64>,
    pub stress: f64,
}

/// Daytime-stress result: the per-hour scores plus the day mean, peak hour, and the trailing high run.
/// The core's band minutes are not on this record yet; adding them is a uniffi record change.
#[derive(uniffi::Record)]
pub struct DaytimeStressInfo {
    pub hours: Vec<ScoredHourInfo>,
    pub day_mean: Option<f64>,
    pub peak_hour: Option<i32>,
    pub sustained_high: bool,
    pub sustained_run: u32,
}

/// Score waking hours for autonomic activation against the day's own calm-hour quartiles (Q25 HR, Q75
/// RMSSD). Each hour needs its own HR gate applied by the caller (a `None` mean_hr hour is skipped).
#[uniffi::export]
pub fn daytime_stress(hours: Vec<HourPointInfo>) -> DaytimeStressInfo {
    let h: Vec<stress::HourPoint> = hours
        .into_iter()
        .map(|p| stress::HourPoint { hour: p.hour, mean_hr: p.mean_hr, rmssd: p.rmssd })
        .collect();
    let r = stress::daytime_stress(&h);
    DaytimeStressInfo {
        hours: r
            .buckets
            .into_iter()
            .map(|s| ScoredHourInfo { hour: s.hour, mean_hr: s.mean_hr, rmssd: s.rmssd, stress: s.stress })
            .collect(),
        day_mean: r.mean,
        peak_hour: r.peak_hour,
        sustained_high: r.sustained_high,
        sustained_run: r.sustained_run as u32,
    }
}

// ── Circadian rhythm + CosinorAge (Rhythm Age) ─────────────────────────────

/// Sex selector for the biological-age coefficient set.
#[derive(uniffi::Enum, Clone, Copy)]
pub enum SexInput {
    Female,
    Male,
    Unknown,
}

impl From<SexInput> for biological_age::Sex {
    fn from(s: SexInput) -> Self {
        match s {
            SexInput::Female => biological_age::Sex::Female,
            SexInput::Male => biological_age::Sex::Male,
            SexInput::Unknown => biological_age::Sex::Unknown,
        }
    }
}

/// A body-clock phase estimate. `confidence` is "unreadable"/"wide"/"solid"; `lean` is
/// "earlier"/"aligned"/"later" (the app renders the sentence from these).
#[derive(uniffi::Record, Clone)]
pub struct PhaseEstimateInfo {
    pub temp_min_hour: f64,
    pub acrophase_hours: f64,
    pub offset_vs_schedule_minutes: f64,
    pub confidence: String,
    pub lean: String,
}

/// Body-clock phase from raw (unix, activity) samples + tz offset, days observed, habitual wake hour, and an
/// optional observed skin-temp minimum hour. `None` when the cosinor is degenerate.
#[uniffi::export]
pub fn circadian_phase_from_samples(
    samples: Vec<ActivitySample>,
    tz_offset_seconds: i64,
    days_observed: u32,
    habitual_wake_hour: f64,
    observed_temp_min_hour: Option<f64>,
) -> Option<PhaseEstimateInfo> {
    let bins = circadian::hourly_bins(&to_pairs(samples), tz_offset_seconds);
    let e = circadian::estimate_phase(&bins, days_observed, habitual_wake_hour, observed_temp_min_hour)?;
    let confidence = match e.confidence {
        circadian::PhaseConfidence::Unreadable => "unreadable",
        circadian::PhaseConfidence::Wide => "wide",
        circadian::PhaseConfidence::Solid => "solid",
    };
    let lean = match e.lean {
        circadian::PhaseLean::Earlier => "earlier",
        circadian::PhaseLean::Aligned => "aligned",
        circadian::PhaseLean::Later => "later",
    };
    Some(PhaseEstimateInfo {
        temp_min_hour: e.temp_min_hour,
        acrophase_hours: e.acrophase_hours,
        offset_vs_schedule_minutes: e.offset_vs_schedule_minutes,
        confidence: confidence.to_string(),
        lean: lean.to_string(),
    })
}

// ── Vitality / Body Age ──────────────────────────────────────────────────────

/// A `[start, end)` unix-second span.
#[derive(uniffi::Record)]
pub struct TimeSpan {
    pub start: i64,
    pub end: i64,
}

/// One driver's signed log-hazard against its population reference.
#[derive(uniffi::Record)]
pub struct VitalityContribution {
    pub key: String,
    pub label: String,
    pub ln_hazard: f64,
}

/// A Vitality reading. `advance_years` is POSITIVE when Body Age is above chronological.
#[derive(uniffi::Record)]
pub struct VitalityInfo {
    pub vitality: f64,
    pub body_age: f64,
    pub chrono_age: f64,
    pub advance_years: f64,
    pub band_years: f64,
    pub contributions: Vec<VitalityContribution>,
    pub factors_used: u32,
}

/// Vitality (0-100) + Body Age (years) from the wearable drivers. `None` below three present drivers.
#[allow(clippy::too_many_arguments)]
#[uniffi::export]
pub fn vitality_compute(
    chrono_age: f64,
    resting_hr: Option<f64>,
    vo2max: Option<f64>,
    expected_vo2max: Option<f64>,
    sleep_hours: Option<f64>,
    sleep_regularity_index: Option<f64>,
    sleep_consistency: Option<f64>,
    rmssd: Option<f64>,
    rmssd_norm: Option<f64>,
    steps: Option<f64>,
) -> Option<VitalityInfo> {
    let input = vitality::VitalityInput {
        chrono_age, resting_hr, vo2max, expected_vo2max, sleep_hours,
        sleep_regularity_index, sleep_consistency, rmssd, rmssd_norm, steps,
    };
    vitality::compute(&input).map(|r| VitalityInfo {
        vitality: r.vitality,
        body_age: r.body_age,
        chrono_age: r.chrono_age,
        advance_years: r.advance_years,
        band_years: r.band_years,
        contributions: r.contributions.into_iter()
            .map(|c| VitalityContribution { key: c.key, label: c.label, ln_hazard: c.ln_hazard })
            .collect(),
        factors_used: r.factors_used,
    })
}

/// Each present driver's signed log-hazard, without the three-driver gate — the "why" behind a
/// reading, and the door a caller uses to inspect one driver at a time.
#[allow(clippy::too_many_arguments)]
#[uniffi::export]
pub fn vitality_contributions(
    chrono_age: f64,
    resting_hr: Option<f64>,
    vo2max: Option<f64>,
    expected_vo2max: Option<f64>,
    sleep_hours: Option<f64>,
    sleep_regularity_index: Option<f64>,
    sleep_consistency: Option<f64>,
    rmssd: Option<f64>,
    rmssd_norm: Option<f64>,
    steps: Option<f64>,
) -> Vec<VitalityContribution> {
    let input = vitality::VitalityInput {
        chrono_age, resting_hr, vo2max, expected_vo2max, sleep_hours,
        sleep_regularity_index, sleep_consistency, rmssd, rmssd_norm, steps,
    };
    vitality::contributions(&input).into_iter()
        .map(|c| VitalityContribution { key: c.key, label: c.label, ln_hazard: c.ln_hazard })
        .collect()
}

/// Coverage windows from a sample series: consecutive timestamps within `max_gap_s` form one span.
/// Feed the HR timestamps here rather than a day's min..max, or every mid-day gap counts as worn.
#[uniffi::export]
pub fn coverage_spans(timestamps: Vec<i64>, max_gap_s: i64) -> Vec<TimeSpan> {
    sleep_regularity::coverage_spans(&timestamps, max_gap_s)
        .into_iter()
        .map(|(start, end)| TimeSpan { start, end })
        .collect()
}

/// Sleep regularity in [0, 1] from nightly durations (hours). `None` below three nights.
#[uniffi::export]
pub fn vitality_sleep_consistency(nightly_hours: Vec<f64>) -> Option<f64> {
    vitality::sleep_consistency(&nightly_hours)
}

/// Median of a series. `0.0` when empty — the one median the app and the algorithms share.
#[uniffi::export]
pub fn series_median(values: Vec<f64>) -> f64 {
    physio_algo::stats::median(&values)
}

/// OLS slope of a series over x = 0, 1, 2, … — the trend direction behind a week-over-week read.
/// `0.0` for fewer than two points or a degenerate spread.
#[uniffi::export]
pub fn series_slope(values: Vec<f64>) -> f64 {
    physio_algo::stats::least_squares_slope(&values)
}
