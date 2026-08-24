//! Versioned nightly physiology: dynamic sleep RHR and final-reliable-SWS HRV.

use crate::hrv::{MIN_BEATS, clean_rr_gap_aware_breaking, report_seam_breaks};
use crate::sleep::{AccelSample, HrSample, RrRun, SleepStage, StageSegment};

pub const ALGORITHM_VERSION: &str = "physiology-dynamic-rhr-final-sws-hrv-v3";

const EPOCH_SECONDS: i64 = 30;
const MIN_EPISODE_COVERAGE: f64 = 0.80;
const MIN_HRV_REPORT_SECONDS: u32 = 240;
const MAX_WRIST_OFF_FRACTION: f64 = 0.01;
const MAX_RR_ARTIFACT_REJECTION: f64 = 0.35;
const MIN_RR_BEAT_TIME_RATIO: f64 = 0.70;
const MAX_RR_BEAT_TIME_RATIO: f64 = 1.10;
const MAX_RR_PAIR_CHANGE_MS: i32 = 200;
const PRIMARY_MIN_DEEP_SECONDS: u32 = 600;
const FALLBACK_MIN_DEEP_SECONDS: u32 = 360;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QualityHrSample {
    pub unix: i64,
    pub bpm: i32,
    pub quality_valid: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DynamicRhrEpoch {
    pub start: i64,
    pub stage: SleepStage,
    pub bpm: f64,
    pub valid_seconds: u32,
    pub coverage: f64,
    pub progress: f64,
    pub stage_weight: f64,
    pub progress_weight: f64,
    pub combined_weight: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DynamicRhrResult {
    pub algorithm_version: &'static str,
    pub rhr_bpm: f64,
    pub rounded_bpm: i32,
    pub usable_hr_seconds: u32,
    pub total_sleep_seconds: u32,
    pub hr_coverage: f64,
    pub total_weight: f64,
    pub deep_weight_fraction: f64,
    pub epochs: Vec<DynamicRhrEpoch>,
}

/// Quality-gated 30-second sleep epochs, weighted 3x in Deep and exponentially 1x to 4x by sleep progress.
pub fn dynamic_rhr(
    sleep_start: i64,
    sleep_end: i64,
    hr: &[QualityHrSample],
    stages: &[StageSegment],
) -> Option<DynamicRhrResult> {
    if sleep_end <= sleep_start {
        return None;
    }
    let total_sleep_seconds: i64 = stages
        .iter()
        .filter(|s| s.stage != SleepStage::Wake)
        .map(|s| (s.end.min(sleep_end) - s.start.max(sleep_start)).max(0))
        .sum();
    if total_sleep_seconds == 0 {
        return None;
    }

    let stage_at = |unix: i64| {
        stages
            .iter()
            .find(|segment| unix >= segment.start && unix < segment.end)
            .map(|segment| segment.stage)
    };
    let mut epochs = Vec::new();
    let mut epoch_start = sleep_start.div_euclid(EPOCH_SECONDS) * EPOCH_SECONDS;
    while epoch_start < sleep_end {
        let epoch_end = (epoch_start + EPOCH_SECONDS).min(sleep_end);
        let mut stage_seconds = [0u32; 4];
        for unix in epoch_start.max(sleep_start)..epoch_end {
            if let Some(stage) = stage_at(unix) {
                let index = match stage {
                    SleepStage::Wake => 0,
                    SleepStage::Light => 1,
                    SleepStage::Deep => 2,
                    SleepStage::Rem => 3,
                };
                stage_seconds[index] += 1;
            }
        }
        let asleep_seconds = stage_seconds[1..].iter().sum::<u32>();
        let max_stage_seconds = *stage_seconds[1..].iter().max().unwrap_or(&0);
        let stage = if max_stage_seconds == 0 {
            None
        } else if stage_seconds[2] == max_stage_seconds {
            Some(SleepStage::Deep)
        } else if stage_seconds[1] == max_stage_seconds {
            Some(SleepStage::Light)
        } else {
            Some(SleepStage::Rem)
        };
        let mut values: Vec<i32> = hr
            .iter()
            .filter(|sample| {
                sample.unix >= epoch_start
                    && sample.unix < epoch_end
                    && stage_at(sample.unix).is_some_and(|stage| stage != SleepStage::Wake)
                    && (25..=220).contains(&sample.bpm)
            })
            .map(|sample| sample.bpm)
            .collect();
        let valid_seconds = hr
            .iter()
            .filter(|sample| {
                sample.unix >= epoch_start
                    && sample.unix < epoch_end
                    && sample.quality_valid
                    && stage_at(sample.unix).is_some_and(|stage| stage != SleepStage::Wake)
            })
            .map(|sample| sample.unix)
            .collect::<std::collections::HashSet<_>>()
            .len() as u32;
        if let Some(stage) =
            stage.filter(|_| !values.is_empty() && asleep_seconds > 0 && valid_seconds > 0)
        {
            values.sort_unstable();
            let bpm = median_i32(&values);
            let coverage = valid_seconds as f64 / asleep_seconds as f64;
            let progress = ((epoch_start - sleep_start) as f64 / (sleep_end - sleep_start) as f64)
                .clamp(0.0, 1.0);
            let stage_weight = if stage == SleepStage::Deep { 3.0 } else { 1.0 };
            let progress_weight = 4.0f64.powf(progress);
            epochs.push(DynamicRhrEpoch {
                start: epoch_start,
                stage,
                bpm,
                valid_seconds,
                coverage,
                progress,
                stage_weight,
                progress_weight,
                combined_weight: coverage * stage_weight * progress_weight,
            });
        }
        epoch_start += EPOCH_SECONDS;
    }
    if epochs.is_empty() {
        return None;
    }

    let values: Vec<f64> = epochs.iter().map(|e| e.bpm).collect();
    let weights: Vec<f64> = epochs.iter().map(|e| e.combined_weight).collect();
    let rhr_bpm = weighted_huber_location(&values, &weights);
    let total_weight: f64 = weights.iter().sum();
    let deep_weight: f64 = epochs
        .iter()
        .filter(|e| e.stage == SleepStage::Deep)
        .map(|e| e.combined_weight)
        .sum();
    let usable_hr_seconds = epochs.iter().map(|e| e.valid_seconds).sum();
    Some(DynamicRhrResult {
        algorithm_version: ALGORITHM_VERSION,
        rhr_bpm,
        rounded_bpm: (rhr_bpm + 0.5).floor() as i32,
        usable_hr_seconds,
        total_sleep_seconds: total_sleep_seconds as u32,
        hr_coverage: usable_hr_seconds as f64 / total_sleep_seconds as f64,
        total_weight,
        deep_weight_fraction: deep_weight / total_weight,
        epochs,
    })
}

#[derive(Clone, Debug, PartialEq)]
pub struct NightlyPhysiologyResult {
    pub algorithm_version: &'static str,
    pub rhr: Option<DynamicRhrResult>,
    pub hrv: FinalSwsHrvResult,
    /// Audit-only quality facts for every Deep episode considered by final-SWS HRV.
    pub deep_episodes: Vec<DeepEpisodeQuality>,
}

/// Computes both immutable v3 nightly metrics from one staged sleep span and its raw streams.
pub fn nightly_physiology(
    sleep_start: i64,
    sleep_end: i64,
    hr: &[HrSample],
    accel: &[AccelSample],
    rr: &[RrRun],
    wrist_off: &[(i64, i64)],
    stages: &[StageSegment],
) -> NightlyPhysiologyResult {
    let quality_hr = quality_gated_hr(sleep_start, sleep_end, hr, accel, wrist_off);
    let valid_hr_seconds: std::collections::HashSet<i64> = quality_hr
        .iter()
        .filter(|sample| sample.quality_valid)
        .map(|sample| sample.unix)
        .collect();
    let reports: Vec<QualityRrReport> = rr
        .iter()
        .filter(|run| run.ts >= sleep_start && run.ts < sleep_end)
        .filter_map(|run| {
            u32::try_from(run.ts).ok().map(|unix| QualityRrReport {
                unix,
                rr: run.intervals.clone(),
                optical_signal_poor: None,
                quality_valid: valid_hr_seconds.contains(&run.ts),
            })
        })
        .collect();
    let episodes: Vec<DeepEpisodeQuality> = stages
        .iter()
        .filter(|segment| segment.stage == SleepStage::Deep)
        .filter_map(|segment| {
            let start = segment.start.max(sleep_start);
            let end = segment.end.min(sleep_end);
            (end > start)
                .then(|| episode_quality(start, end, &quality_hr, accel, &reports, wrist_off))
        })
        .collect();
    NightlyPhysiologyResult {
        algorithm_version: ALGORITHM_VERSION,
        rhr: dynamic_rhr(sleep_start, sleep_end, &quality_hr, stages),
        hrv: final_sws_last_five_hrv_with_streams(
            &episodes,
            &reports,
            &quality_hr,
            accel,
            wrist_off,
        ),
        deep_episodes: episodes,
    }
}

/// Exact immutable episode-level gates, exposed for provenance/reporting only.
pub fn episode_gate_failures(e: &DeepEpisodeQuality) -> Vec<&'static str> {
    episode_gate_failures_with_minimum(e, PRIMARY_MIN_DEEP_SECONDS)
}

pub fn fallback_episode_gate_failures(e: &DeepEpisodeQuality) -> Vec<&'static str> {
    episode_gate_failures_with_minimum(e, FALLBACK_MIN_DEEP_SECONDS)
}

fn episode_gate_failures_with_minimum(
    e: &DeepEpisodeQuality,
    minimum_seconds: u32,
) -> Vec<&'static str> {
    let mut failures = Vec::new();
    if e.end.saturating_sub(e.start) < minimum_seconds {
        failures.push(if minimum_seconds == PRIMARY_MIN_DEEP_SECONDS {
            "duration_lt_600_seconds"
        } else {
            "duration_lt_360_seconds"
        });
    }
    if e.hr_coverage < MIN_EPISODE_COVERAGE { failures.push("hr_coverage"); }
    if e.accelerometer_coverage < MIN_EPISODE_COVERAGE { failures.push("accelerometer_coverage"); }
    if e.clean_hr_coverage < MIN_EPISODE_COVERAGE { failures.push("clean_hr_coverage"); }
    if e.rr_report_coverage < MIN_EPISODE_COVERAGE { failures.push("rr_report_coverage"); }
    if e.wrist_off_fraction > MAX_WRIST_OFF_FRACTION { failures.push("wrist_off_fraction"); }
    if e.rr_artifact_rejection > MAX_RR_ARTIFACT_REJECTION { failures.push("rr_artifact_rejection"); }
    if e.rr_hr_agreement.is_some_and(|v| v < MIN_EPISODE_COVERAGE) { failures.push("rr_hr_agreement"); }
    if !(MIN_RR_BEAT_TIME_RATIO..=MAX_RR_BEAT_TIME_RATIO).contains(&e.rr_beat_time_ratio) { failures.push("rr_beat_time_ratio"); }
    failures
}

fn quality_gated_hr(
    start: i64,
    end: i64,
    hr: &[HrSample],
    accel: &[AccelSample],
    wrist_off: &[(i64, i64)],
) -> Vec<QualityHrSample> {
    use std::collections::BTreeMap;
    let mut hr_groups: BTreeMap<i64, Vec<i32>> = BTreeMap::new();
    for sample in hr.iter().filter(|s| s.ts >= start && s.ts < end) {
        hr_groups
            .entry(sample.ts)
            .or_default()
            .push(sample.bpm as i32);
    }
    let mut accel_groups: BTreeMap<i64, Vec<(f64, f64, f64)>> = BTreeMap::new();
    for sample in accel
        .iter()
        .filter(|s| s.ts >= start.saturating_sub(1) && s.ts < end)
    {
        accel_groups
            .entry(sample.ts)
            .or_default()
            .push((sample.x, sample.y, sample.z));
    }
    let hr_by_second: BTreeMap<i64, f64> = hr_groups
        .into_iter()
        .map(|(ts, values)| {
            (
                ts,
                values.iter().map(|&v| v as f64).sum::<f64>() / values.len() as f64,
            )
        })
        .collect();
    let accel_by_second: Vec<(i64, (f64, f64, f64))> = accel_groups
        .into_iter()
        .map(|(ts, values)| {
            let n = values.len() as f64;
            let sum = values
                .iter()
                .fold((0.0, 0.0, 0.0), |a, v| (a.0 + v.0, a.1 + v.1, a.2 + v.2));
            (ts, (sum.0 / n, sum.1 / n, sum.2 / n))
        })
        .collect();
    let mut jerk = BTreeMap::new();
    let mut jerk_values = Vec::new();
    for pair in accel_by_second.windows(2) {
        let (ts, value) = pair[1];
        let prior = pair[0].1;
        let j = ((value.0 - prior.0).powi(2)
            + (value.1 - prior.1).powi(2)
            + (value.2 - prior.2).powi(2))
        .sqrt();
        jerk.insert(ts, j);
        if j.is_finite() {
            jerk_values.push(j);
        }
    }
    jerk_values.sort_by(f64::total_cmp);
    let movement_threshold = if jerk_values.is_empty() {
        1e-6
    } else {
        median_f64(&jerk_values).max(1e-6) * 75.0
    };
    let seconds: Vec<i64> = (start..end).collect();
    let bpms: Vec<Option<f64>> = seconds
        .iter()
        .map(|ts| hr_by_second.get(ts).copied())
        .collect();
    let local_medians: Vec<Option<f64>> = (0..seconds.len())
        .map(|i| rolling_median(&bpms, i, 30, 15))
        .collect();
    let deviations: Vec<Option<f64>> = bpms
        .iter()
        .zip(&local_medians)
        .map(|(value, median)| Some((value.as_ref()? - median.as_ref()?).abs()))
        .collect();
    let local_mads: Vec<Option<f64>> = (0..seconds.len())
        .map(|i| rolling_median(&deviations, i, 30, 15))
        .collect();
    let base_contamination: Vec<bool> = (0..seconds.len())
        .map(|i| {
            let movement = jerk
                .get(&seconds[i])
                .is_some_and(|j| *j > movement_threshold);
            let spike = match (deviations[i], local_mads[i]) {
                (Some(d), Some(mad)) => d > 8.0f64.max(6.0 * 1.4826 * mad),
                _ => false,
            };
            let jump = bpms[i].is_some_and(|bpm| {
                i.checked_sub(1)
                    .and_then(|j| bpms[j])
                    .is_some_and(|v| (bpm - v).abs() > 10.0)
                    || bpms
                        .get(i + 1)
                        .and_then(|v| *v)
                        .is_some_and(|v| (bpm - v).abs() > 10.0)
            });
            movement || spike || jump
        })
        .collect();
    seconds
        .into_iter()
        .enumerate()
        .filter_map(|(i, unix)| {
            bpms[i].map(|bpm| {
                let lo = i.saturating_sub(2);
                let hi = (i + 2).min(base_contamination.len().saturating_sub(1));
                let contaminated = base_contamination[lo..=hi].iter().any(|&v| v);
                let off_wrist = wrist_off.iter().any(|&(a, b)| unix >= a && unix < b);
                QualityHrSample {
                    unix,
                    bpm: bpm.round() as i32,
                    quality_valid: (25.0..=220.0).contains(&bpm) && !contaminated && !off_wrist,
                }
            })
        })
        .collect()
}

fn rolling_median(
    values: &[Option<f64>],
    center: usize,
    radius: usize,
    minimum: usize,
) -> Option<f64> {
    let lo = center.saturating_sub(radius);
    let hi = (center + radius + 1).min(values.len());
    let mut present: Vec<f64> = values[lo..hi].iter().filter_map(|v| *v).collect();
    if present.len() < minimum {
        return None;
    }
    present.sort_by(f64::total_cmp);
    Some(median_f64(&present))
}

fn episode_quality(
    start: i64,
    end: i64,
    hr: &[QualityHrSample],
    accel: &[AccelSample],
    reports: &[QualityRrReport],
    wrist_off: &[(i64, i64)],
) -> DeepEpisodeQuality {
    let duration = (end - start).max(1) as f64;
    let hr_seconds = hr
        .iter()
        .filter(|s| s.unix >= start && s.unix < end)
        .count();
    let clean_hr_seconds = hr
        .iter()
        .filter(|s| s.unix >= start && s.unix < end && s.quality_valid)
        .count();
    let first_accel = accel.iter().map(|s| s.ts).min();
    let accel_seconds = accel
        .iter()
        .filter(|s| s.ts >= start && s.ts < end && Some(s.ts) != first_accel)
        .map(|s| s.ts)
        .collect::<std::collections::HashSet<_>>()
        .len();
    let start_u32 = u32::try_from(start).unwrap_or(0);
    let end_u32 = u32::try_from(end).unwrap_or(start_u32);
    let rr_quality = rr_window_quality(start_u32, end_u32, reports);
    let hr_by_second: std::collections::HashMap<u32, f64> = hr
        .iter()
        .filter_map(|s| u32::try_from(s.unix).ok().map(|ts| (ts, s.bpm as f64)))
        .collect();
    let agreement: Vec<bool> = reports
        .iter()
        .filter(|r| r.unix >= start_u32 && r.unix < end_u32)
        .filter_map(|r| {
            let bpm = *hr_by_second.get(&r.unix)?;
            let mut valid: Vec<f64> =
                r.rr.iter()
                    .filter(|&&v| (300..=2000).contains(&v))
                    .map(|&v| v as f64)
                    .collect();
            valid.sort_by(f64::total_cmp);
            (!valid.is_empty()).then(|| (60_000.0 / median_f64(&valid) - bpm).abs() <= 15.0)
        })
        .collect();
    DeepEpisodeQuality {
        start: start_u32,
        end: end_u32,
        hr_coverage: hr_seconds as f64 / duration,
        accelerometer_coverage: accel_seconds as f64 / duration,
        clean_hr_coverage: clean_hr_seconds as f64 / duration,
        wrist_off_fraction: overlap_fraction(start, end, wrist_off),
        rr_report_coverage: rr_quality.report_seconds as f64 / duration,
        rr_artifact_rejection: rr_quality.artifact_rejection,
        rr_hr_agreement: (!agreement.is_empty())
            .then(|| agreement.iter().filter(|&&v| v).count() as f64 / agreement.len() as f64),
        rr_beat_time_ratio: rr_quality.beat_time_ms as f64 / 1000.0 / duration,
    }
}

fn overlap_fraction(start: i64, end: i64, intervals: &[(i64, i64)]) -> f64 {
    let mut clipped: Vec<(i64, i64)> = intervals
        .iter()
        .map(|&(a, b)| (a.max(start), b.min(end)))
        .filter(|&(a, b)| b > a)
        .collect();
    clipped.sort_unstable();
    let mut covered = 0i64;
    let mut current: Option<(i64, i64)> = None;
    for (a, b) in clipped {
        current = match current {
            None => Some((a, b)),
            Some((x, y)) if a <= y => Some((x, y.max(b))),
            Some((x, y)) => {
                covered += y - x;
                Some((a, b))
            }
        };
    }
    if let Some((a, b)) = current {
        covered += b - a;
    }
    covered as f64 / (end - start).max(1) as f64
}

#[derive(Clone, Copy, Debug, Default)]
struct RrWindowQuality {
    report_seconds: u32,
    input_intervals: u32,
    clean_intervals: u32,
    contiguous_pairs: u32,
    sudden_change_pairs_rejected: u32,
    beat_time_ms: u64,
    artifact_rejection: f64,
    rmssd_ms: Option<f64>,
}

fn rr_window_quality(start: u32, end: u32, reports: &[QualityRrReport]) -> RrWindowQuality {
    let mut selected: Vec<&QualityRrReport> = reports
        .iter()
        .filter(|r| {
            r.unix >= start
                && r.unix < end
                && r.quality_valid
                && crate::hrv::rr_trusted(r.optical_signal_poor)
        })
        .collect();
    selected.sort_by_key(|r| r.unix);
    let report_seconds = selected
        .iter()
        .map(|r| r.unix)
        .collect::<std::collections::HashSet<_>>()
        .len() as u32;
    let input_intervals = selected.iter().map(|r| r.rr.len() as u32).sum();
    let beat_time_ms = selected
        .iter()
        .flat_map(|r| r.rr.iter().copied())
        .filter(|v| (300..=2000).contains(v))
        .map(u64::from)
        .sum();
    let slices: Vec<(u32, &[u16])> = selected.iter().map(|r| (r.unix, r.rr.as_slice())).collect();
    let flat: Vec<u16> = selected.iter().flat_map(|r| r.rr.iter().copied()).collect();
    let mut breaks = report_seam_breaks(&slices);
    let mut offset = 0usize;
    for pair in selected.windows(2) {
        offset += pair[0].rr.len();
        if pair[1].unix > pair[0].unix.saturating_add(1) && offset < breaks.len() {
            breaks[offset] = true;
        }
    }
    let (clean, contiguous) = clean_rr_gap_aware_breaking(&flat, &breaks);
    let mut sum_sq = 0.0;
    let mut contiguous_pairs = 0u32;
    let mut sudden_change_pairs_rejected = 0u32;
    for i in 1..clean.len() {
        if !contiguous[i] {
            continue;
        }
        let delta = i32::from(clean[i]) - i32::from(clean[i - 1]);
        if delta.abs() > MAX_RR_PAIR_CHANGE_MS {
            sudden_change_pairs_rejected += 1;
        }
        sum_sq += f64::from(delta * delta);
        contiguous_pairs += 1;
    }
    RrWindowQuality {
        report_seconds,
        input_intervals,
        clean_intervals: clean.len() as u32,
        contiguous_pairs,
        sudden_change_pairs_rejected,
        beat_time_ms,
        artifact_rejection: 1.0 - clean.len() as f64 / flat.len().max(1) as f64,
        rmssd_ms: (contiguous_pairs > 0).then(|| (sum_sq / contiguous_pairs as f64).sqrt()),
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DeepEpisodeQuality {
    pub start: u32,
    pub end: u32,
    pub hr_coverage: f64,
    pub accelerometer_coverage: f64,
    pub clean_hr_coverage: f64,
    pub wrist_off_fraction: f64,
    pub rr_report_coverage: f64,
    pub rr_artifact_rejection: f64,
    pub rr_hr_agreement: Option<f64>,
    pub rr_beat_time_ratio: f64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QualityRrReport {
    pub unix: u32,
    pub rr: Vec<u16>,
    pub optical_signal_poor: Option<bool>,
    pub quality_valid: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HrvUnavailableReason {
    NoDeepEpisode,
    NoQualifyingTenMinuteDeepEpisode,
    NoQualifyingSixMinuteDeepEpisode,
    NoReliableDeepEpisode,
    InsufficientReportCoverage,
    InsufficientCleanIntervals,
    NoContiguousPairs,
    NoQualityValidFiveMinuteWindow,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HrvMeasurementMode {
    PrimaryFinalSws,
    FallbackShortSws,
}

impl HrvMeasurementMode {
    pub fn baseline_eligible(self) -> bool {
        self == Self::PrimaryFinalSws
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct HrvSelectionAttempt {
    pub rmssd_ms: Option<f64>,
    pub selected_episode: Option<(u32, u32)>,
    pub selected_episode_quality: Option<DeepEpisodeQuality>,
    pub selected_window: Option<(u32, u32)>,
    pub usable_report_seconds: u32,
    pub input_rr_intervals: u32,
    pub clean_rr_intervals: u32,
    pub contiguous_pairs: u32,
    pub sudden_change_pairs_rejected: u32,
    pub rejection_reason: Option<HrvUnavailableReason>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FinalSwsHrvResult {
    pub algorithm_version: &'static str,
    pub measurement_mode: Option<HrvMeasurementMode>,
    pub primary_attempt: HrvSelectionAttempt,
    pub fallback_attempt: Option<HrvSelectionAttempt>,
    pub rmssd_ms: Option<f64>,
    pub selected_episode: Option<(u32, u32)>,
    pub selected_episode_quality: Option<DeepEpisodeQuality>,
    pub selected_window: Option<(u32, u32)>,
    pub usable_report_seconds: u32,
    pub input_rr_intervals: u32,
    pub clean_rr_intervals: u32,
    pub contiguous_pairs: u32,
    pub sudden_change_pairs_rejected: u32,
    pub rejection_reason: Option<HrvUnavailableReason>,
}

/// Primary final-SWS RMSSD with the six-minute short-SWS fallback whenever primary is unavailable.
pub fn final_sws_last_five_hrv(
    episodes: &[DeepEpisodeQuality],
    reports: &[QualityRrReport],
) -> FinalSwsHrvResult {
    final_sws_last_five_hrv_inner(episodes, reports, |episode, start, end| {
        let rr = rr_window_quality(start, end, reports);
        let mut quality = *episode;
        quality.start = start;
        quality.end = end;
        quality.rr_report_coverage = rr.report_seconds as f64 / f64::from(end - start);
        quality.rr_artifact_rejection = rr.artifact_rejection;
        quality.rr_beat_time_ratio = rr.beat_time_ms as f64 / 1000.0 / f64::from(end - start);
        quality
    })
}

fn final_sws_last_five_hrv_with_streams(
    episodes: &[DeepEpisodeQuality],
    reports: &[QualityRrReport],
    hr: &[QualityHrSample],
    accel: &[AccelSample],
    wrist_off: &[(i64, i64)],
) -> FinalSwsHrvResult {
    final_sws_last_five_hrv_inner(episodes, reports, |_, start, end| {
        let mut quality = episode_quality(
            i64::from(start),
            i64::from(end),
            hr,
            accel,
            reports,
            wrist_off,
        );
        quality.wrist_off_fraction = 0.0;
        quality
    })
}

fn final_sws_last_five_hrv_inner<F>(
    episodes: &[DeepEpisodeQuality],
    reports: &[QualityRrReport],
    quality_for_window: F,
) -> FinalSwsHrvResult
where
    F: Fn(&DeepEpisodeQuality, u32, u32) -> DeepEpisodeQuality + Copy,
{
    let primary = select_primary(episodes, reports, quality_for_window);
    if primary.rmssd_ms.is_some() {
        return from_attempt(HrvMeasurementMode::PrimaryFinalSws, primary, None);
    }
    let fallback = select_fallback(episodes, reports, quality_for_window, &primary);
    if fallback.rmssd_ms.is_some() {
        return from_attempt(
            HrvMeasurementMode::FallbackShortSws,
            primary,
            Some(fallback),
        );
    }
    from_attempt_without_value(primary, Some(fallback))
}

fn select_primary<F>(
    episodes: &[DeepEpisodeQuality],
    reports: &[QualityRrReport],
    quality_for_window: F,
) -> HrvSelectionAttempt
where
    F: Fn(&DeepEpisodeQuality, u32, u32) -> DeepEpisodeQuality,
{
    if episodes.is_empty() {
        return unavailable_attempt(HrvUnavailableReason::NoDeepEpisode);
    }
    if !episodes
        .iter()
        .any(|e| e.end.saturating_sub(e.start) >= PRIMARY_MIN_DEEP_SECONDS)
    {
        return unavailable_attempt(HrvUnavailableReason::NoQualifyingTenMinuteDeepEpisode);
    }
    let Some(episode) = episodes
        .iter()
        .filter(|e| reliable_episode(e, PRIMARY_MIN_DEEP_SECONDS))
        .max_by_key(|e| e.end)
    else {
        return unavailable_attempt(HrvUnavailableReason::NoReliableDeepEpisode);
    };
    select_window(episode, reports, quality_for_window)
}

fn select_fallback<F>(
    episodes: &[DeepEpisodeQuality],
    reports: &[QualityRrReport],
    quality_for_window: F,
    primary_attempt: &HrvSelectionAttempt,
) -> HrvSelectionAttempt
where
    F: Fn(&DeepEpisodeQuality, u32, u32) -> DeepEpisodeQuality + Copy,
{
    let mut candidates: Vec<&DeepEpisodeQuality> = episodes
        .iter()
        .filter(|e| e.end.saturating_sub(e.start) >= FALLBACK_MIN_DEEP_SECONDS)
        .collect();
    if candidates.is_empty() {
        return unavailable_attempt(if episodes.is_empty() {
            HrvUnavailableReason::NoDeepEpisode
        } else {
            HrvUnavailableReason::NoQualifyingSixMinuteDeepEpisode
        });
    }
    candidates.retain(|episode| {
        primary_attempt.selected_episode != Some((episode.start, episode.end))
    });
    if candidates.is_empty() {
        return primary_attempt.clone();
    }
    candidates.sort_by_key(|e| std::cmp::Reverse(e.end));
    let mut latest_refusal = None;
    for episode in candidates {
        if !reliable_episode(episode, FALLBACK_MIN_DEEP_SECONDS) {
            continue;
        }
        let attempt = select_window(episode, reports, quality_for_window);
        if attempt.rmssd_ms.is_some() {
            return attempt;
        }
        if latest_refusal.is_none() {
            latest_refusal = Some(attempt);
        }
    }
    latest_refusal.unwrap_or_else(|| {
        if primary_attempt.selected_episode.is_some() {
            primary_attempt.clone()
        } else {
            unavailable_attempt(HrvUnavailableReason::NoReliableDeepEpisode)
        }
    })
}

fn select_window<F>(
    episode: &DeepEpisodeQuality,
    reports: &[QualityRrReport],
    quality_for_window: F,
) -> HrvSelectionAttempt
where
    F: Fn(&DeepEpisodeQuality, u32, u32) -> DeepEpisodeQuality,
{
    let mut window_end = episode.end;
    while window_end >= episode.start.saturating_add(300) {
        let window_start = window_end - 300;
        let quality = quality_for_window(episode, window_start, window_end);
        let rr = rr_window_quality(window_start, window_end, reports);
        if reliable_window(&quality, &rr) {
            return HrvSelectionAttempt {
                rmssd_ms: rr.rmssd_ms,
                selected_episode: Some((episode.start, episode.end)),
                selected_episode_quality: Some(*episode),
                selected_window: Some((window_start, window_end)),
                usable_report_seconds: rr.report_seconds,
                input_rr_intervals: rr.input_intervals,
                clean_rr_intervals: rr.clean_intervals,
                contiguous_pairs: rr.contiguous_pairs,
                sudden_change_pairs_rejected: rr.sudden_change_pairs_rejected,
                rejection_reason: None,
            };
        }
        if window_end < episode.start.saturating_add(330) {
            break;
        }
        window_end -= 30;
    }
    let window = (episode.end.saturating_sub(300), episode.end);
    let rr = rr_window_quality(window.0, window.1, reports);
    let reason = if rr.report_seconds < MIN_HRV_REPORT_SECONDS {
        HrvUnavailableReason::InsufficientReportCoverage
    } else if rr.clean_intervals < MIN_BEATS as u32 {
        HrvUnavailableReason::InsufficientCleanIntervals
    } else if rr.contiguous_pairs == 0 {
        HrvUnavailableReason::NoContiguousPairs
    } else {
        HrvUnavailableReason::NoQualityValidFiveMinuteWindow
    };
    hrv_refusal(episode, window, rr, reason)
}

fn reliable_episode(e: &DeepEpisodeQuality, minimum_seconds: u32) -> bool {
    e.end.saturating_sub(e.start) >= minimum_seconds
        && e.hr_coverage >= MIN_EPISODE_COVERAGE
        && e.accelerometer_coverage >= MIN_EPISODE_COVERAGE
        && e.clean_hr_coverage >= MIN_EPISODE_COVERAGE
        && e.rr_report_coverage >= MIN_EPISODE_COVERAGE
        && e.wrist_off_fraction <= MAX_WRIST_OFF_FRACTION
        && e.rr_artifact_rejection <= MAX_RR_ARTIFACT_REJECTION
        && e.rr_hr_agreement
            .is_none_or(|agreement| agreement >= MIN_EPISODE_COVERAGE)
        && (MIN_RR_BEAT_TIME_RATIO..=MAX_RR_BEAT_TIME_RATIO).contains(&e.rr_beat_time_ratio)
}

fn reliable_window(quality: &DeepEpisodeQuality, rr: &RrWindowQuality) -> bool {
    quality.hr_coverage >= MIN_EPISODE_COVERAGE
        && quality.accelerometer_coverage >= MIN_EPISODE_COVERAGE
        && quality.clean_hr_coverage >= MIN_EPISODE_COVERAGE
        && quality.rr_report_coverage >= MIN_EPISODE_COVERAGE
        && quality.wrist_off_fraction <= MAX_WRIST_OFF_FRACTION
        && quality.rr_artifact_rejection <= MAX_RR_ARTIFACT_REJECTION
        && quality
            .rr_hr_agreement
            .is_none_or(|agreement| agreement >= MIN_EPISODE_COVERAGE)
        && (MIN_RR_BEAT_TIME_RATIO..=MAX_RR_BEAT_TIME_RATIO).contains(&quality.rr_beat_time_ratio)
        && rr.clean_intervals >= MIN_BEATS as u32
        && rr.contiguous_pairs > 0
        && rr.rmssd_ms.is_some()
}

fn unavailable_attempt(reason: HrvUnavailableReason) -> HrvSelectionAttempt {
    HrvSelectionAttempt {
        rmssd_ms: None,
        selected_episode: None,
        selected_episode_quality: None,
        selected_window: None,
        usable_report_seconds: 0,
        input_rr_intervals: 0,
        clean_rr_intervals: 0,
        contiguous_pairs: 0,
        sudden_change_pairs_rejected: 0,
        rejection_reason: Some(reason),
    }
}

fn hrv_refusal(
    e: &DeepEpisodeQuality,
    window: (u32, u32),
    rr: RrWindowQuality,
    reason: HrvUnavailableReason,
) -> HrvSelectionAttempt {
    HrvSelectionAttempt {
        rmssd_ms: None,
        selected_episode: Some((e.start, e.end)),
        selected_episode_quality: Some(*e),
        selected_window: Some(window),
        usable_report_seconds: rr.report_seconds,
        input_rr_intervals: rr.input_intervals,
        clean_rr_intervals: rr.clean_intervals,
        contiguous_pairs: rr.contiguous_pairs,
        sudden_change_pairs_rejected: rr.sudden_change_pairs_rejected,
        rejection_reason: Some(reason),
    }
}

fn from_attempt(
    mode: HrvMeasurementMode,
    primary_attempt: HrvSelectionAttempt,
    fallback_attempt: Option<HrvSelectionAttempt>,
) -> FinalSwsHrvResult {
    let selected = fallback_attempt
        .as_ref()
        .unwrap_or(&primary_attempt)
        .clone();
    FinalSwsHrvResult {
        algorithm_version: ALGORITHM_VERSION,
        measurement_mode: Some(mode),
        primary_attempt,
        fallback_attempt,
        rmssd_ms: selected.rmssd_ms,
        selected_episode: selected.selected_episode,
        selected_episode_quality: selected.selected_episode_quality,
        selected_window: selected.selected_window,
        usable_report_seconds: selected.usable_report_seconds,
        input_rr_intervals: selected.input_rr_intervals,
        clean_rr_intervals: selected.clean_rr_intervals,
        contiguous_pairs: selected.contiguous_pairs,
        sudden_change_pairs_rejected: selected.sudden_change_pairs_rejected,
        rejection_reason: None,
    }
}

fn from_attempt_without_value(
    primary_attempt: HrvSelectionAttempt,
    fallback_attempt: Option<HrvSelectionAttempt>,
) -> FinalSwsHrvResult {
    let selected = fallback_attempt
        .as_ref()
        .unwrap_or(&primary_attempt)
        .clone();
    FinalSwsHrvResult {
        algorithm_version: ALGORITHM_VERSION,
        measurement_mode: None,
        primary_attempt,
        fallback_attempt,
        rmssd_ms: None,
        selected_episode: selected.selected_episode,
        selected_episode_quality: selected.selected_episode_quality,
        selected_window: selected.selected_window,
        usable_report_seconds: selected.usable_report_seconds,
        input_rr_intervals: selected.input_rr_intervals,
        clean_rr_intervals: selected.clean_rr_intervals,
        contiguous_pairs: selected.contiguous_pairs,
        sudden_change_pairs_rejected: selected.sudden_change_pairs_rejected,
        rejection_reason: selected.rejection_reason,
    }
}

fn median_i32(values: &[i32]) -> f64 {
    let n = values.len();
    if n.is_multiple_of(2) {
        (values[n / 2 - 1] + values[n / 2]) as f64 / 2.0
    } else {
        values[n / 2] as f64
    }
}

fn weighted_huber_location(values: &[f64], weights: &[f64]) -> f64 {
    let mut ordered = values.to_vec();
    ordered.sort_by(f64::total_cmp);
    let mut location = median_f64(&ordered);
    let mut deviations: Vec<f64> = values.iter().map(|v| (v - location).abs()).collect();
    deviations.sort_by(f64::total_cmp);
    let scale = (1.4826 * median_f64(&deviations)).max(1.0);
    for _ in 0..50 {
        let (sum, weight_sum) =
            values
                .iter()
                .zip(weights)
                .fold((0.0, 0.0), |(sum, weight_sum), (&value, &base)| {
                    let residual = (value - location).abs();
                    let robust = if residual == 0.0 {
                        1.0
                    } else {
                        (1.345 * scale / residual).min(1.0)
                    };
                    (sum + value * base * robust, weight_sum + base * robust)
                });
        let next = sum / weight_sum;
        if (next - location).abs() < 1e-9 {
            return next;
        }
        location = next;
    }
    location
}

fn median_f64(values: &[f64]) -> f64 {
    let n = values.len();
    if n.is_multiple_of(2) {
        (values[n / 2 - 1] + values[n / 2]) / 2.0
    } else {
        values[n / 2]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quality(start: u32, end: u32) -> DeepEpisodeQuality {
        DeepEpisodeQuality {
            start,
            end,
            hr_coverage: 1.0,
            accelerometer_coverage: 1.0,
            clean_hr_coverage: 1.0,
            wrist_off_fraction: 0.0,
            rr_report_coverage: 1.0,
            rr_artifact_rejection: 0.0,
            rr_hr_agreement: Some(1.0),
            rr_beat_time_ratio: 1.0,
        }
    }

    #[test]
    fn aug_16_dynamic_rhr_uses_deep_and_late_weights_without_a_floor() {
        let stages = [
            StageSegment {
                start: 0,
                end: 300,
                stage: SleepStage::Light,
            },
            StageSegment {
                start: 300,
                end: 600,
                stage: SleepStage::Deep,
            },
        ];
        let hr: Vec<QualityHrSample> = (0..600)
            .map(|unix| QualityHrSample {
                unix,
                bpm: if unix < 300 { 90 } else { 75 },
                quality_valid: unix != 450,
            })
            .collect();
        let result = dynamic_rhr(0, 600, &hr, &stages).unwrap();
        assert!(result.rhr_bpm > 75.0 && result.rhr_bpm < 90.0);
        assert!(result.deep_weight_fraction > 0.75);
        assert_eq!(result.usable_hr_seconds, 599);
    }

    #[test]
    fn short_sws_fallback_uses_a_valid_six_minute_deep_episode() {
        let episodes = [
            DeepEpisodeQuality {
                start: 0,
                end: 420,
                hr_coverage: 1.0,
                accelerometer_coverage: 1.0,
                clean_hr_coverage: 1.0,
                wrist_off_fraction: 0.0,
                rr_report_coverage: 1.0,
                rr_artifact_rejection: 0.0,
                rr_hr_agreement: Some(1.0),
                rr_beat_time_ratio: 1.0,
            },
            DeepEpisodeQuality {
                start: 500,
                end: 590,
                hr_coverage: 1.0,
                accelerometer_coverage: 1.0,
                clean_hr_coverage: 1.0,
                wrist_off_fraction: 0.0,
                rr_report_coverage: 1.0,
                rr_artifact_rejection: 0.0,
                rr_hr_agreement: Some(1.0),
                rr_beat_time_ratio: 1.0,
            },
        ];
        let reports = reports(120, 420, 10);
        let result = final_sws_last_five_hrv(&episodes, &reports);
        assert!(result.rmssd_ms.is_some());
        assert_eq!(
            result.measurement_mode,
            Some(HrvMeasurementMode::FallbackShortSws)
        );
        assert_eq!(result.selected_episode, Some((0, 420)));
        assert_eq!(result.selected_window, Some((120, 420)));
    }

    #[test]
    fn final_sws_hrv_uses_only_the_last_reliable_episode_and_last_five_minutes() {
        let episodes = [quality(0, 600), quality(1_000, 1_900)];
        let reports: Vec<QualityRrReport> = (1_600..1_900)
            .map(|unix| QualityRrReport {
                unix,
                rr: vec![800 + (unix % 2) as u16 * 10],
                optical_signal_poor: None,
                quality_valid: true,
            })
            .collect();
        let result = final_sws_last_five_hrv(&episodes, &reports);
        assert_eq!(
            result.measurement_mode,
            Some(HrvMeasurementMode::PrimaryFinalSws)
        );
        assert!(result.fallback_attempt.is_none());
        assert_eq!(result.selected_episode, Some((1_000, 1_900)));
        assert_eq!(result.selected_window, Some((1_600, 1_900)));
        assert_eq!(result.usable_report_seconds, 300);
        assert!(result.rmssd_ms.is_some());
    }

    #[test]
    fn short_terminal_deep_fragment_does_not_replace_the_last_reliable_episode() {
        let episodes = [quality(0, 900), quality(1_000, 1_300)];
        let reports: Vec<QualityRrReport> = (600..900)
            .map(|unix| QualityRrReport {
                unix,
                rr: vec![800 + (unix % 2) as u16 * 10],
                optical_signal_poor: None,
                quality_valid: true,
            })
            .collect();
        let result = final_sws_last_five_hrv(&episodes, &reports);
        assert_eq!(result.selected_episode, Some((0, 900)));
        assert_eq!(result.selected_window, Some((600, 900)));
    }

    #[test]
    fn final_sws_hrv_rejects_sub_80_percent_coverage_and_wrist_off() {
        let reports: Vec<QualityRrReport> = (300..540)
            .map(|unix| QualityRrReport {
                unix,
                rr: vec![800],
                optical_signal_poor: None,
                quality_valid: true,
            })
            .collect();
        let mut low_coverage = quality(0, 600);
        low_coverage.rr_report_coverage = 0.79;
        assert_eq!(
            final_sws_last_five_hrv(&[low_coverage], &reports).rejection_reason,
            Some(HrvUnavailableReason::NoReliableDeepEpisode)
        );
        let mut off_wrist = quality(0, 600);
        off_wrist.wrist_off_fraction = 0.02;
        assert_eq!(
            final_sws_last_five_hrv(&[off_wrist], &reports).rejection_reason,
            Some(HrvUnavailableReason::NoReliableDeepEpisode)
        );
    }

    #[test]
    fn dynamic_rhr_is_robust_to_low_and_high_single_epoch_outliers() {
        let stages = [StageSegment {
            start: 0,
            end: 900,
            stage: SleepStage::Light,
        }];
        let base: Vec<QualityHrSample> = (0..900)
            .map(|unix| QualityHrSample {
                unix,
                bpm: 60,
                quality_valid: true,
            })
            .collect();
        for outlier in [25, 220] {
            let mut samples = base.clone();
            for sample in samples.iter_mut().filter(|s| (450..480).contains(&s.unix)) {
                sample.bpm = outlier;
            }
            let result = dynamic_rhr(0, 900, &samples, &stages).unwrap();
            assert!((result.rhr_bpm - 60.0).abs() < 1.0);
            assert_ne!(result.rounded_bpm, outlier);
        }
    }

    #[test]
    fn late_weighting_can_raise_or_lower_rhr_without_floor_behavior() {
        let stages = [StageSegment {
            start: 0,
            end: 600,
            stage: SleepStage::Deep,
        }];
        let make = |early, late| {
            (0..600)
                .map(|unix| QualityHrSample {
                    unix,
                    bpm: if unix < 300 { early } else { late },
                    quality_valid: true,
                })
                .collect::<Vec<_>>()
        };
        let lower = dynamic_rhr(0, 600, &make(80, 50), &stages).unwrap();
        let raise = dynamic_rhr(0, 600, &make(50, 80), &stages).unwrap();
        assert!(lower.rhr_bpm > 50.0 && lower.rhr_bpm < 65.0);
        assert!(raise.rhr_bpm > 65.0 && raise.rhr_bpm < 80.0);
        assert!(raise.rhr_bpm > lower.rhr_bpm);
    }

    #[test]
    fn immutable_version_is_emitted_by_both_metrics() {
        let stages = [StageSegment {
            start: 0,
            end: 600,
            stage: SleepStage::Deep,
        }];
        let hr: Vec<QualityHrSample> = (0..600)
            .map(|unix| QualityHrSample {
                unix,
                bpm: 60,
                quality_valid: true,
            })
            .collect();
        let rhr = dynamic_rhr(0, 600, &hr, &stages).unwrap();
        let hrv = final_sws_last_five_hrv(&[], &[]);
        assert_eq!(rhr.algorithm_version, ALGORITHM_VERSION);
        assert_eq!(hrv.algorithm_version, ALGORITHM_VERSION);
    }

    fn reports(start: u32, end: u32, delta: u16) -> Vec<QualityRrReport> {
        (start..end)
            .map(|unix| QualityRrReport {
                unix,
                rr: vec![800 + (unix % 2) as u16 * delta],
                optical_signal_poor: None,
                quality_valid: true,
            })
            .collect()
    }

    #[test]
    fn fallback_selects_latest_eligible_episode_not_lowest_hrv() {
        let episodes = [quality(0, 390), quality(1_000, 1_390)];
        let mut rr = reports(90, 390, 2);
        rr.extend(reports(1_090, 1_390, 30));
        let result = final_sws_last_five_hrv(&episodes, &rr);
        assert_eq!(
            result.measurement_mode,
            Some(HrvMeasurementMode::FallbackShortSws)
        );
        assert_eq!(result.selected_episode, Some((1_000, 1_390)));
        assert!(result.rmssd_ms.unwrap() > 20.0);
    }

    #[test]
    fn fallback_runs_when_ten_minute_episodes_fail_primary_quality() {
        let mut unreliable_primary = quality(0, 600);
        unreliable_primary.wrist_off_fraction = 0.02;
        let episodes = [unreliable_primary, quality(1_000, 1_360)];
        let result = final_sws_last_five_hrv(&episodes, &reports(1_060, 1_360, 10));
        assert_eq!(
            result.measurement_mode,
            Some(HrvMeasurementMode::FallbackShortSws)
        );
        assert_eq!(result.selected_episode, Some((1_000, 1_360)));
    }

    #[test]
    fn fallback_runs_after_primary_has_no_quality_valid_window() {
        let episodes = [quality(0, 360), quality(1_000, 1_600)];
        let result = final_sws_last_five_hrv(&episodes, &reports(60, 360, 10));
        assert_eq!(
            result.primary_attempt.rejection_reason,
            Some(HrvUnavailableReason::InsufficientReportCoverage)
        );
        assert_eq!(
            result.measurement_mode,
            Some(HrvMeasurementMode::FallbackShortSws)
        );
        assert_eq!(result.selected_episode, Some((0, 360)));
        assert_eq!(result.selected_window, Some((60, 360)));
    }

    #[test]
    fn fallback_refuses_six_minutes_without_a_quality_valid_window() {
        let result = final_sws_last_five_hrv(&[quality(0, 360)], &[]);
        assert_eq!(result.measurement_mode, None);
        assert_eq!(
            result.primary_attempt.rejection_reason,
            Some(HrvUnavailableReason::NoQualifyingTenMinuteDeepEpisode)
        );
        assert_eq!(
            result.rejection_reason,
            Some(HrvUnavailableReason::InsufficientReportCoverage)
        );
    }

    #[test]
    fn fallback_refuses_deep_fragments_shorter_than_six_minutes() {
        let result = final_sws_last_five_hrv(&[quality(0, 359)], &reports(59, 359, 10));
        assert_eq!(result.measurement_mode, None);
        assert_eq!(
            result.rejection_reason,
            Some(HrvUnavailableReason::NoQualifyingSixMinuteDeepEpisode)
        );
    }

    #[test]
    fn fallback_does_not_bypass_wrist_off_or_rr_quality_gates() {
        let mut off_wrist = quality(0, 360);
        off_wrist.wrist_off_fraction = 0.02;
        let wrist_result = final_sws_last_five_hrv(&[off_wrist], &reports(60, 360, 10));
        assert_eq!(wrist_result.measurement_mode, None);
        assert_eq!(
            wrist_result.rejection_reason,
            Some(HrvUnavailableReason::NoReliableDeepEpisode)
        );

        let poor_rr: Vec<QualityRrReport> = reports(60, 360, 10)
            .into_iter()
            .map(|mut report| {
                report.optical_signal_poor = Some(true);
                report
            })
            .collect();
        let rr_result = final_sws_last_five_hrv(&[quality(0, 360)], &poor_rr);
        assert_eq!(rr_result.measurement_mode, None);
        assert_eq!(
            rr_result.rejection_reason,
            Some(HrvUnavailableReason::InsufficientReportCoverage)
        );
    }

    #[test]
    fn only_primary_measurements_are_baseline_eligible() {
        assert!(HrvMeasurementMode::PrimaryFinalSws.baseline_eligible());
        assert!(!HrvMeasurementMode::FallbackShortSws.baseline_eligible());
    }

    #[test]
    fn fallback_measurement_remains_recovery_eligible() {
        let result = final_sws_last_five_hrv(&[quality(0, 360)], &reports(60, 360, 20));
        assert_eq!(
            result.measurement_mode,
            Some(HrvMeasurementMode::FallbackShortSws)
        );
        let score = crate::recovery::recovery(&crate::recovery::RecoveryInput {
            hrv: result.rmssd_ms.unwrap(),
            rhr: 50.0,
            hrv_baseline: Some(crate::recovery::DriverBaseline {
                mean: 50.0,
                spread: 10.0,
            }),
            rhr_baseline: Some(crate::recovery::DriverBaseline {
                mean: 55.0,
                spread: 5.0,
            }),
            ..Default::default()
        });
        assert!(score.is_some());
    }
}
