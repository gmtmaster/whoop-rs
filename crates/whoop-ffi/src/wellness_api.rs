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

// ── Windowed stress (intraday + overnight autonomic activation) ────────────

/// One bucket's aggregates: epoch-ms start, local `hour` (0–23), mean HR (None below the sample gate),
/// RMSSD (None on insufficient clean R-R) and the bucket's mean dynamic accel in g (None when absent).
#[derive(uniffi::Record)]
pub struct HourPointInfo {
    pub hour: i32,
    pub mean_hr: Option<f64>,
    pub rmssd: Option<f64>,
    pub start_ms: i64,
    pub motion_g: Option<f64>,
}

/// One scored bucket.
#[derive(uniffi::Record)]
pub struct ScoredHourInfo {
    pub hour: i32,
    pub mean_hr: f64,
    pub rmssd: Option<f64>,
    pub stress: f64,
    pub start_ms: i64,
}

/// A `[start, end)` sleep or nap span in epoch ms.
#[derive(uniffi::Record)]
pub struct SleepSpanMsInfo {
    pub start_ms: i64,
    pub end_ms: i64,
}

/// Why a bucket was held out of the score — a known state the caller can word, not a gap.
#[derive(uniffi::Enum, Clone, Copy, PartialEq, Eq)]
pub enum SuppressionInfo {
    Active,
    Asleep,
}

/// One bucket excluded from both the score and its calm reference.
#[derive(uniffi::Record)]
pub struct SuppressedBucketInfo {
    pub start_ms: i64,
    pub hour: i32,
    pub suppression: SuppressionInfo,
}

/// One scored window set — a day or a night, since both derive from one formula: the per-bucket
/// scores, the set mean, its peak hour, the trailing high run, and the minutes in each band.
/// `high_share_pct` is the high band's share of the scored minutes, `None` when nothing scored.
#[derive(uniffi::Record)]
pub struct WindowedStressInfo {
    pub hours: Vec<ScoredHourInfo>,
    pub day_mean: Option<f64>,
    pub peak_hour: Option<i32>,
    pub sustained_high: bool,
    pub sustained_run: u32,
    pub low_minutes: i64,
    pub medium_minutes: i64,
    pub high_minutes: i64,
    pub high_share_pct: Option<f64>,
    pub peak_start_ms: Option<i64>,
    pub suppressed: Vec<SuppressedBucketInfo>,
}

impl From<stress::Suppression> for SuppressionInfo {
    fn from(s: stress::Suppression) -> Self {
        match s {
            stress::Suppression::Active => SuppressionInfo::Active,
            stress::Suppression::Asleep => SuppressionInfo::Asleep,
        }
    }
}

impl From<stress::StressWindows> for WindowedStressInfo {
    fn from(r: stress::StressWindows) -> Self {
        WindowedStressInfo {
            day_mean: r.mean,
            peak_hour: r.peak_hour,
            peak_start_ms: r.peak_start_ms,
            suppressed: r
                .suppressed
                .iter()
                .map(|s| SuppressedBucketInfo {
                    start_ms: s.start_ms, hour: s.hour, suppression: s.suppression.into(),
                })
                .collect(),
            sustained_high: r.sustained_high,
            sustained_run: r.sustained_run as u32,
            low_minutes: r.low_minutes,
            medium_minutes: r.medium_minutes,
            high_minutes: r.high_minutes,
            high_share_pct: r.high_share_pct(),
            hours: r
                .buckets
                .into_iter()
                .map(|s| ScoredHourInfo {
                    hour: s.hour, mean_hr: s.mean_hr, rmssd: s.rmssd, stress: s.stress,
                    start_ms: s.start_ms,
                })
                .collect(),
        }
    }
}

fn to_points(hours: Vec<HourPointInfo>) -> Vec<stress::HourPoint> {
    hours
        .into_iter()
        .map(|p| stress::HourPoint {
            start_ms: p.start_ms, hour: p.hour, mean_hr: p.mean_hr, rmssd: p.rmssd,
            motion_g: p.motion_g,
        })
        .collect()
}

/// Score waking buckets for autonomic activation against the day's own calm quartiles (Q25 HR, Q75
/// RMSSD). Buckets overlapping `sleep_spans`, and buckets over the motion gate, are dropped BEFORE
/// that reference is built and returned in `suppressed` instead. Each bucket needs its own HR gate
/// applied by the caller (a `None` mean_hr bucket is skipped).
#[uniffi::export]
pub fn daytime_stress(hours: Vec<HourPointInfo>, sleep_spans: Vec<SleepSpanMsInfo>) -> WindowedStressInfo {
    let spans: Vec<stress::SpanMs> = sleep_spans
        .into_iter()
        .map(|s| stress::SpanMs { start_ms: s.start_ms, end_ms: s.end_ms })
        .collect();
    stress::daytime_stress(&to_points(hours), &spans).into()
}

/// Score one sleep window's buckets on the same formula and the same 0–3 bands. No hour-of-day filter
/// is applied, so the caller passes ONLY the buckets inside the span — a night crosses midnight and
/// one hour range cannot say "22:00 to 06:00".
#[uniffi::export]
pub fn sleep_stress(hours: Vec<HourPointInfo>) -> WindowedStressInfo {
    stress::sleep_stress(&to_points(hours)).into()
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

/// Arithmetic mean of a series. `0.0` when empty, so a caller that must distinguish "no data" checks
/// the input rather than the result.
#[uniffi::export]
pub fn series_mean(values: Vec<f64>) -> f64 {
    physio_algo::stats::mean(&values)
}

/// Sample standard deviation (n − 1) of a series. `0.0` under two points.
#[uniffi::export]
pub fn series_sample_sd(values: Vec<f64>) -> f64 {
    physio_algo::stats::sample_sd(&values)
}

/// Population standard deviation (÷ n) of a series — the per-window spread the z-scorers use, as
/// distinct from the n − 1 [`series_sample_sd`] the baselines use. `0.0` when empty.
#[uniffi::export]
pub fn series_population_sd(values: Vec<f64>) -> f64 {
    physio_algo::stats::population_sd(&values)
}

/// Pearson correlation of two equal-length series. `None` under two pairs or on a zero-variance series.
#[uniffi::export]
pub fn series_pearson(xs: Vec<f64>, ys: Vec<f64>) -> Option<f64> {
    physio_algo::stats::pearson(&xs, &ys)
}

/// Robust z-score against a baseline mean + EWMA-abs-dev spread: (value − mean) / (1.253 × spread).
/// The one z the Charge drivers and their trace share.
#[uniffi::export]
pub fn z_score(value: f64, mean: f64, spread: f64) -> f64 {
    recovery::z_score(value, mean, spread)
}

/// Which way a series moved once its interval is accounted for. `Flat` means the interval straddles
/// zero, so the direction is not separable from noise — it is not "no movement".
#[derive(uniffi::Enum)]
pub enum TrendDirectionInfo {
    Rising,
    Falling,
    Flat,
}

/// A weighted linear trend over day offsets carrying its own uncertainty, so no caller picks a slope
/// threshold. `slope` is per day; `startValue`/`endValue` are FITTED, not observed.
#[derive(uniffi::Record)]
pub struct TrendlineInfo {
    pub slope: f64,
    pub intercept: f64,
    pub slope_se: f64,
    pub slope_ci_lo: f64,
    pub slope_ci_hi: f64,
    pub start_day: f64,
    pub end_day: f64,
    pub start_value: f64,
    pub end_value: f64,
    pub total_change: f64,
    pub total_change_ci_lo: f64,
    pub total_change_ci_hi: f64,
    pub slope_z: f64,
    pub significance: f64,
    pub direction: TrendDirectionInfo,
    pub n: u32,
}

/// Weighted trendline of `values` over `days` (day offsets, not sample index) across a `window_days`-wide
/// request, with a residual-based 80 % interval. `weights` may be empty for unit weights.
/// `None` under three finite points, under the window's minimum span, or with no weighted x-spread.
#[uniffi::export]
pub fn series_trendline(
    days: Vec<f64>,
    values: Vec<f64>,
    weights: Vec<f64>,
    window_days: f64,
) -> Option<TrendlineInfo> {
    let min_span = physio_algo::stats::trend_min_span_days(window_days);
    let t = physio_algo::stats::weighted_trendline(&days, &values, &weights, min_span)?;
    Some(TrendlineInfo {
        slope: t.slope,
        intercept: t.intercept,
        slope_se: t.slope_se,
        slope_ci_lo: t.slope_ci_lo,
        slope_ci_hi: t.slope_ci_hi,
        start_day: t.start_day,
        end_day: t.end_day,
        start_value: t.start_value,
        end_value: t.end_value,
        total_change: t.total_change,
        total_change_ci_lo: t.total_change_ci_lo,
        total_change_ci_hi: t.total_change_ci_hi,
        slope_z: t.slope_z,
        significance: t.significance,
        direction: match t.direction {
            physio_algo::stats::TrendDirection::Rising => TrendDirectionInfo::Rising,
            physio_algo::stats::TrendDirection::Falling => TrendDirectionInfo::Falling,
            physio_algo::stats::TrendDirection::Flat => TrendDirectionInfo::Flat,
        },
        n: t.n as u32,
    })
}

/// Second-half mean minus first-half mean of a series, the odd point going to the recent half. `None`
/// under four points — the window-change number a trend chip shows.
#[uniffi::export]
pub fn series_half_change(values: Vec<f64>) -> Option<f64> {
    physio_algo::stats::half_change(&values)
}
