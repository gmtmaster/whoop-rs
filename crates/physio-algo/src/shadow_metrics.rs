//! Versioned nightly physiology: dynamic sleep RHR and final-reliable-SWS HRV.

use crate::hrv::{MIN_BEATS, clean_rr_gap_aware_breaking, report_seam_breaks};
use crate::sleep::{AccelSample, HrSample, RrRun, SleepStage, StageSegment};

pub const ALGORITHM_VERSION: &str = "physiology-dynamic-rhr-final-sws-hrv-v3";

const EPOCH_SECONDS: i64 = 30;
const MIN_EPISODE_COVERAGE: f64 = 0.80;
const MAX_WRIST_OFF_FRACTION: f64 = 0.01;
const MAX_RR_ARTIFACT_REJECTION: f64 = 0.35;
const MIN_RR_BEAT_TIME_RATIO: f64 = 0.70;
const MAX_RR_BEAT_TIME_RATIO: f64 = 1.10;
const MAX_RR_PAIR_CHANGE_MS: i32 = 200;
const PRIMARY_MIN_DEEP_SECONDS: u32 = 600;
const FALLBACK_MIN_DEEP_SECONDS: u32 = 360;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RrDeviceGeneration {
    Whoop4,
    #[default]
    Whoop5Mg,
}

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
    /// v3 dynamic-RHR output. Kept for research/testing (see the v3-vs-v4 parity benchmark) and no
    /// longer read by noop-engine for the authoritative nightly RHR - see `rhr_v4` for that.
    pub rhr: Option<DynamicRhrResult>,
    /// v4 quality-weighted RHR output - the authoritative nightly RHR as of
    /// `ALGORITHM_VERSION_V4`. `rhr_v4.quality_v4` is the production value; `rhr_v4.fixed_v4` is
    /// the fixed-60/40 variant, kept for research/debug only and never used as authoritative.
    pub rhr_v4: RhrV4Result,
    pub hrv: FinalSwsHrvResult,
    /// Audit-only quality facts for every Deep episode considered by final-SWS HRV.
    pub deep_episodes: Vec<DeepEpisodeQuality>,
}

/// Computes the immutable nightly physiology metrics from one staged sleep span and its raw
/// streams: authoritative v4 RHR (`rhr_v4`, see `dynamic_rhr_v4`), the v3 RHR kept for research/
/// parity (`rhr`, see `dynamic_rhr`, unchanged), and final-SWS HRV (`hrv`, unchanged).
pub fn nightly_physiology(
    sleep_start: i64,
    sleep_end: i64,
    hr: &[HrSample],
    accel: &[AccelSample],
    rr: &[RrRun],
    wrist_off: &[(i64, i64)],
    stages: &[StageSegment],
) -> NightlyPhysiologyResult {
    nightly_physiology_for_generation(
        sleep_start, sleep_end, hr, accel, rr, wrist_off, stages,
        RrDeviceGeneration::Whoop5Mg,
    )
}

/// Generation-aware nightly entry point. WHOOP 5/MG deliberately takes the historical path
/// byte-for-byte; only explicitly identified WHOOP 4 rolling reports are normalized.
pub fn nightly_physiology_for_generation(
    sleep_start: i64,
    sleep_end: i64,
    hr: &[HrSample],
    accel: &[AccelSample],
    rr: &[RrRun],
    wrist_off: &[(i64, i64)],
    stages: &[StageSegment],
    rr_generation: RrDeviceGeneration,
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
    let reports = match rr_generation {
        RrDeviceGeneration::Whoop4 => normalize_whoop4_rr_reports(&reports),
        RrDeviceGeneration::Whoop5Mg => reports,
    };
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
        // The bundle-level stamp: what noop-engine emits as `algorithm_version` (and what
        // ultimately reaches daily_metrics/daily_canonical/sleep_sessions) now that v4 RHR is
        // authoritative. The nested `hrv` result below keeps its own, unrelated v3 stamp - see
        // `ALGORITHM_VERSION` and `final_sws_last_five_hrv_with_streams`, both unchanged.
        algorithm_version: ALGORITHM_VERSION_V4,
        rhr: dynamic_rhr(sleep_start, sleep_end, &quality_hr, stages),
        rhr_v4: dynamic_rhr_v4(sleep_start, sleep_end, hr, accel, wrist_off, stages),
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

/// WHOOP 4 emits a rolling interval window each second. Anchor each report to its wall-clock
/// second, walk its intervals backwards from the end of that second, and retain only intervals
/// whose end falls in that report's own second. Thus a beat is retained because it advances time,
/// never because its numeric R-R value happens to differ from a previous value.
fn normalize_whoop4_rr_reports(reports: &[QualityRrReport]) -> Vec<QualityRrReport> {
    let mut ordered: Vec<&QualityRrReport> = reports.iter().collect();
    ordered.sort_by_key(|report| report.unix);
    let mut out = Vec::with_capacity(ordered.len());
    let mut previous_report_second: Option<u32> = None;
    let mut previous_values: Vec<u16> = Vec::new();
    let mut run_reports = 0u64;
    let mut retained_beat_time_ms = 0u64;
    for report in ordered {
        if previous_report_second.is_none_or(|previous| report.unix > previous.saturating_add(1)) {
            // A missing report is a real seam. Do not use the new rolling prefix to backfill it.
            run_reports = 0;
            retained_beat_time_ms = 0;
        }
        run_reports += 1;
        let target_ms = run_reports * 1_000;
        let valid: Vec<u16> = report.rr.iter().copied().filter(|v| (300..=2_000).contains(v)).collect();
        let consecutive = previous_report_second.is_some_and(|p| report.unix == p.saturating_add(1));
        let max_overlap = previous_values.len().min(valid.len().saturating_sub(1));
        let overlap = if consecutive {
            (1..=max_overlap)
                .rev()
                .find(|&n| previous_values[previous_values.len() - n..] == valid[..n])
                .unwrap_or(0)
        } else {
            0
        };
        let advancing = &valid[overlap..];
        // The newest beats are the suffix of a rolling report. Retain the suffix length that makes
        // cumulative beat time track cumulative wall time most closely; zero is allowed, so a slow
        // pulse does not force one fabricated beat into every wall-clock second.
        let mut suffix_ms = 0u64;
        let mut best_len = 0usize;
        let mut best_error = retained_beat_time_ms.abs_diff(target_ms);
        for len in 1..=advancing.len() {
            suffix_ms += u64::from(advancing[advancing.len() - len]);
            let error = (retained_beat_time_ms + suffix_ms).abs_diff(target_ms);
            if error < best_error {
                best_error = error;
                best_len = len;
            }
        }
        let retained = advancing[advancing.len().saturating_sub(best_len)..].to_vec();
        retained_beat_time_ms += retained.iter().map(|&v| u64::from(v)).sum::<u64>();
        previous_report_second = Some(report.unix);
        previous_values = valid;
        out.push(QualityRrReport {
            unix: report.unix,
            rr: retained,
            optical_signal_poor: report.optical_signal_poor,
            quality_valid: report.quality_valid,
        });
    }
    out
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
    let reason = if rr.clean_intervals < MIN_BEATS as u32 {
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

// ============================================================================================
// RHR v4 — AUTHORITATIVE. Called from `nightly_physiology` (via `dynamic_rhr_v4`, below) and wired
// into `noop-engine` as the nightly RHR that is persisted to `daily_metrics`/`daily_canonical`
// (and, per-session, to `sleep_sessions`). `dynamic_rhr` (v3, above) is kept unchanged and directly
// callable/testable for research and parity comparison, but is no longer read by noop-engine.
//
// Design (episode-level, not epoch-level, unlike v3): for every qualifying Deep/SWS episode,
// compute one robust HR value H_i (10% trimmed mean of quality-valid samples) and one reliability
// score Q_i derived ONLY from coverage/duration/dispersion — never from the HR value itself. Then
// compare the FINAL qualifying episode (F) against the LOWEST-valued qualifying episode (L) and
// either use one (if they coincide, or only one/no second candidate exists) or blend both.
//
// The AUTHORITATIVE nightly RHR is `quality_v4` — the reliability-weighted blend
// `(Q_F*H_F + Q_L*H_L) / (Q_F+Q_L)`. `fixed_v4` (the fixed-60/40 blend) is kept only for
// research/debug comparison and is never read as the production value.
// ============================================================================================

/// Algorithm version string for the v4 nightly-RHR pipeline. Distinct from `ALGORITHM_VERSION`
/// (the v3 identifier, kept unchanged above) so the two can never be confused. This is the
/// identifier `nightly_physiology`/`noop-engine` now stamp as the immutable physiology algorithm
/// version, and the one persisted to `daily_metrics.physiology_algorithm_version` /
/// `daily_canonical.algorithm_version` / `sleep_sessions.algorithm_version`.
pub const ALGORITHM_VERSION_V4: &str = "physiology-dynamic-rhr-sws-v4";

/// Provisional: minimum Deep-episode duration to be considered "reliable" for v4. Matches the
/// 300-second floor used in the offline benchmark (analysis/rhr_v4_benchmark/rhr_v4_benchmark.py).
const V4_MIN_EPISODE_DURATION_SECONDS: i64 = 300;

/// Provisional: minimum quality-valid HR coverage fraction within an episode to be "reliable".
/// Reuses `MIN_EPISODE_COVERAGE` (line 9) — the same constant HRV episode gating already uses in
/// this file — rather than inventing a second, RHR-specific coverage threshold.
const V4_MIN_EPISODE_COVERAGE: f64 = MIN_EPISODE_COVERAGE;

/// Provisional: the 10% trim applied to an episode's quality-valid HR samples to form H_i.
const V4_TRIM_FRACTION: f64 = 0.10;

/// Provisional: duration (seconds) at which `duration_confidence` saturates to 1.0. A 30-minute
/// episode gets the same duration_confidence as a 10-minute one once both clear this floor — this
/// is the mechanism that stops a long early Deep episode from automatically outweighing a shorter,
/// equally-clean later one just by being longer. Value chosen to match the same order of magnitude
/// as `V4_MIN_EPISODE_DURATION_SECONDS` (2x it); not derived from data.
const V4_DURATION_SATURATION_SECONDS: f64 = 600.0;

/// Provisional: the disagreement collapse is intentionally absent here — v4 always blends F and L
/// when they differ (Tier 1) rather than testing a threshold to decide whether to blend at all.
/// A disagreement-triggered reconsideration policy was discussed as a design option but is NOT
/// implemented in this shadow; it remains an open question pending WHOOP data (see report).

/// One Deep/SWS episode's contribution to v4: a robust value paired with a reliability score that
/// is provably independent of that value (nothing in `quality_confidence` below reads `value`).
#[derive(Clone, Debug, PartialEq)]
pub struct RhrV4Candidate {
    /// H_i: 10%-trimmed mean of quality-valid HR samples in this episode. Never rounded.
    pub value: f64,
    /// Q_i in (0, 1]: `coverage * duration_confidence * stability_confidence`. See
    /// `episode_quality_score` for the exact formula. HR value has zero influence on this.
    pub quality: f64,
    pub start: i64,
    pub end: i64,
    pub duration_seconds: i64,
    /// Fraction of this episode's seconds that were quality-valid (in-range, not wrist-off, not
    /// contamination-flagged) — the same `quality_valid` semantics `dynamic_rhr` uses.
    pub coverage: f64,
    /// MAD (median absolute deviation, bpm) of the episode's quality-valid samples — the
    /// dispersion/stability signal. Lower = more stable. Independent of the episode's HR level.
    pub stability_mad: f64,
    pub sample_count: usize,
}

/// Which tier of the fallback hierarchy produced a given `RhrV4Result`. See the module doc
/// on `dynamic_rhr_v4` for the full trigger/estimator/rationale table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RhrV4Tier {
    /// >=2 reliable Deep episodes, final != lowest -> blend F and L.
    FinalAndLowestDistinct,
    /// >=2 reliable Deep episodes, final == lowest -> that episode alone, not double-counted.
    FinalEqualsLowest,
    /// Exactly 1 reliable Deep episode -> its H_i alone.
    SingleReliableEpisode,
    /// No individually reliable episode, but >=300s of pooled quality-valid Deep HR exists.
    PooledDeep,
    /// No usable Deep at all -> quality-filtered whole-sleep (non-Wake) robust estimate.
    WholeSleep,
    /// Not enough data anywhere to produce a value.
    Unavailable,
}

/// Full v4 result: both blend variants (fixed 60/40 and reliability-weighted) plus full provenance
/// for debugging/replay. `quality_v4` is the authoritative nightly RHR persisted to canonical
/// storage; `fixed_v4` is kept for research/debug comparison only and is never used as the
/// authoritative value. `fixed_v4`/`quality_v4` collapse to the SAME single value outside Tier 1
/// (no second candidate to blend against), which is intentional — they only diverge from each
/// other when F and L are both present and distinct.
#[derive(Clone, Debug, PartialEq)]
pub struct RhrV4Result {
    pub algorithm_version: &'static str,
    pub tier: RhrV4Tier,
    /// Unrounded. `0.60*H_F + 0.40*H_L` in Tier 1; the tier's single value otherwise.
    pub fixed_v4_raw: Option<f64>,
    /// Unrounded. `(Q_F*H_F + Q_L*H_L) / (Q_F+Q_L)` in Tier 1; the tier's single value otherwise.
    pub quality_v4_raw: Option<f64>,
    /// Half-up rounded presentation values — the ONLY rounding boundary in this pipeline.
    pub fixed_v4: Option<i32>,
    pub quality_v4: Option<i32>,
    pub final_candidate: Option<RhrV4Candidate>,
    pub lowest_candidate: Option<RhrV4Candidate>,
    pub final_equals_lowest: bool,
    pub qualifying_deep_episodes: usize,
}

fn v4_round_half_up(x: f64) -> i32 {
    (x + 0.5).floor() as i32
}

fn v4_trimmed_mean(mut values: Vec<i32>) -> f64 {
    values.sort_unstable();
    let n = values.len();
    let trim = ((n as f64 * V4_TRIM_FRACTION).floor() as usize).min(n / 2);
    let kept = &values[trim..n - trim];
    kept.iter().map(|&v| v as f64).sum::<f64>() / kept.len() as f64
}

/// MAD of a bpm sample set, computed the same way `weighted_huber_location` computes its residual
/// scale elsewhere in this file (sort, median, absolute deviations, median of those) — reused here
/// as a stability signal rather than inventing a second dispersion measure for this module.
fn v4_mad(values: &[i32]) -> f64 {
    let floats: Vec<f64> = values.iter().map(|&v| v as f64).collect();
    let mut sorted = floats.clone();
    sorted.sort_by(f64::total_cmp);
    let med = median_f64(&sorted);
    let mut deviations: Vec<f64> = floats.iter().map(|&v| (v - med).abs()).collect();
    deviations.sort_by(f64::total_cmp);
    median_f64(&deviations)
}

/// `Q_i = coverage * duration_confidence * stability_confidence`. Every factor is derived only
/// from reliability information (coverage, duration, dispersion) — NEVER from the episode's HR
/// value — so a low-HR episode can never score higher quality purely for being low.
/// `duration_confidence` saturates at `V4_DURATION_SATURATION_SECONDS` so a long episode cannot
/// automatically out-rank a shorter, equally clean one. `stability_confidence = 1/(1+MAD)` is a
/// simple, bounded, monotonically-decreasing-in-dispersion mapping — deliberately not a more
/// elaborate model, per the "keep it small and explainable" brief.
fn v4_quality_score(duration_seconds: i64, coverage: f64, stability_mad: f64) -> f64 {
    let duration_confidence = (duration_seconds as f64 / V4_DURATION_SATURATION_SECONDS).min(1.0);
    let stability_confidence = 1.0 / (1.0 + stability_mad);
    coverage * duration_confidence * stability_confidence
}

/// Builds one `RhrV4Candidate` for a Deep span if it qualifies (duration + coverage gates), else
/// `None`. `quality_hr` is the same `quality_gated_hr` output `dynamic_rhr`/`nightly_physiology`
/// already compute — no separate quality pass is run for v4.
fn v4_episode_candidate(
    span_start: i64,
    span_end: i64,
    sleep_start: i64,
    sleep_end: i64,
    quality_hr: &[QualityHrSample],
) -> Option<RhrV4Candidate> {
    let start = span_start.max(sleep_start);
    let end = span_end.min(sleep_end);
    let duration_seconds = end - start;
    if duration_seconds < V4_MIN_EPISODE_DURATION_SECONDS {
        return None;
    }
    let valid: Vec<i32> = quality_hr
        .iter()
        .filter(|s| s.unix >= start && s.unix < end && s.quality_valid)
        .map(|s| s.bpm)
        .collect();
    if valid.is_empty() {
        return None;
    }
    let coverage = valid.len() as f64 / duration_seconds as f64;
    if coverage < V4_MIN_EPISODE_COVERAGE {
        return None;
    }
    let stability_mad = v4_mad(&valid);
    let sample_count = valid.len();
    let value = v4_trimmed_mean(valid);
    let quality = v4_quality_score(duration_seconds, coverage, stability_mad);
    Some(RhrV4Candidate {
        value,
        quality,
        start,
        end,
        duration_seconds,
        coverage,
        stability_mad,
        sample_count,
    })
}

/// The v4 entry point — the authoritative nightly RHR. Called directly from `nightly_physiology`
/// (see the `rhr_v4` field it populates), and also directly callable on its own for research/
/// replay with the same raw-stream inputs `nightly_physiology` takes. `dynamic_rhr` (v3, above)
/// remains fully independent of this function and is unaffected by it. See the module banner
/// above for the full design rationale.
///
/// Fallback hierarchy (see `RhrV4Tier` for the enum, this is the trigger/estimator/rationale):
///
/// | Tier | Trigger                                                          | Estimator                          |
/// |------|-------------------------------------------------------------------|-------------------------------------|
/// | 1    | >=2 reliable Deep episodes, final != lowest                       | blend of H_F and H_L (both variants)|
/// | 2    | >=2 reliable Deep episodes, final == lowest                       | H of that one episode               |
/// | 3    | exactly 1 reliable Deep episode                                   | its H_i                             |
/// | 4    | no reliable episode, but >=300s pooled quality-valid Deep HR       | trimmed mean of pooled Deep samples |
/// | 5    | no usable Deep at all                                             | trimmed mean of whole-sleep (non-Wake) quality-valid samples |
/// | 6    | insufficient data anywhere                                        | `None`                              |
///
/// Never falls back to a raw minimum at any tier.
pub fn dynamic_rhr_v4(
    sleep_start: i64,
    sleep_end: i64,
    hr: &[HrSample],
    accel: &[AccelSample],
    wrist_off: &[(i64, i64)],
    stages: &[StageSegment],
) -> RhrV4Result {
    let quality_hr = quality_gated_hr(sleep_start, sleep_end, hr, accel, wrist_off);

    let mut candidates: Vec<RhrV4Candidate> = stages
        .iter()
        .filter(|s| s.stage == SleepStage::Deep)
        .filter_map(|s| v4_episode_candidate(s.start, s.end, sleep_start, sleep_end, &quality_hr))
        .collect();
    candidates.sort_by_key(|c| c.start);

    let empty_result = |tier: RhrV4Tier, value: Option<f64>| RhrV4Result {
        algorithm_version: ALGORITHM_VERSION_V4,
        tier,
        fixed_v4_raw: value,
        quality_v4_raw: value,
        fixed_v4: value.map(v4_round_half_up),
        quality_v4: value.map(v4_round_half_up),
        final_candidate: None,
        lowest_candidate: None,
        final_equals_lowest: false,
        qualifying_deep_episodes: 0,
    };

    if candidates.is_empty() {
        // Tier 4: pooled Deep, regardless of any single episode's own reliability.
        let pooled: Vec<i32> = stages
            .iter()
            .filter(|s| s.stage == SleepStage::Deep)
            .flat_map(|s| {
                let start = s.start.max(sleep_start);
                let end = s.end.min(sleep_end);
                quality_hr
                    .iter()
                    .filter(move |q| q.unix >= start && q.unix < end && q.quality_valid)
                    .map(|q| q.bpm)
            })
            .collect();
        if pooled.len() as i64 >= V4_MIN_EPISODE_DURATION_SECONDS {
            return empty_result(RhrV4Tier::PooledDeep, Some(v4_trimmed_mean(pooled)));
        }
        // Tier 5: whole-sleep (non-Wake) quality-valid samples.
        let stage_at = |unix: i64| {
            stages
                .iter()
                .find(|segment| unix >= segment.start && unix < segment.end)
                .map(|segment| segment.stage)
        };
        let whole: Vec<i32> = quality_hr
            .iter()
            .filter(|q| {
                q.quality_valid && stage_at(q.unix).is_some_and(|st| st != SleepStage::Wake)
            })
            .map(|q| q.bpm)
            .collect();
        if whole.is_empty() {
            return empty_result(RhrV4Tier::Unavailable, None);
        }
        return empty_result(RhrV4Tier::WholeSleep, Some(v4_trimmed_mean(whole)));
    }

    if candidates.len() == 1 {
        let only = candidates.into_iter().next().unwrap();
        let value = only.value;
        return RhrV4Result {
            algorithm_version: ALGORITHM_VERSION_V4,
            tier: RhrV4Tier::SingleReliableEpisode,
            fixed_v4_raw: Some(value),
            quality_v4_raw: Some(value),
            fixed_v4: Some(v4_round_half_up(value)),
            quality_v4: Some(v4_round_half_up(value)),
            final_candidate: Some(only.clone()),
            lowest_candidate: Some(only),
            final_equals_lowest: true,
            qualifying_deep_episodes: 1,
        };
    }

    let qualifying_deep_episodes = candidates.len();
    let final_candidate = candidates.last().cloned().unwrap();
    let lowest_candidate = candidates
        .iter()
        .cloned()
        .min_by(|a, b| a.value.total_cmp(&b.value))
        .unwrap();
    let final_equals_lowest = final_candidate.start == lowest_candidate.start;

    if final_equals_lowest {
        let value = final_candidate.value;
        return RhrV4Result {
            algorithm_version: ALGORITHM_VERSION_V4,
            tier: RhrV4Tier::FinalEqualsLowest,
            fixed_v4_raw: Some(value),
            quality_v4_raw: Some(value),
            fixed_v4: Some(v4_round_half_up(value)),
            quality_v4: Some(v4_round_half_up(value)),
            final_candidate: Some(final_candidate),
            lowest_candidate: Some(lowest_candidate),
            final_equals_lowest: true,
            qualifying_deep_episodes,
        };
    }

    let (h_f, h_l) = (final_candidate.value, lowest_candidate.value);
    let (q_f, q_l) = (final_candidate.quality, lowest_candidate.quality);
    let fixed_raw = 0.60 * h_f + 0.40 * h_l;
    let quality_raw = (q_f * h_f + q_l * h_l) / (q_f + q_l);

    RhrV4Result {
        algorithm_version: ALGORITHM_VERSION_V4,
        tier: RhrV4Tier::FinalAndLowestDistinct,
        fixed_v4_raw: Some(fixed_raw),
        quality_v4_raw: Some(quality_raw),
        fixed_v4: Some(v4_round_half_up(fixed_raw)),
        quality_v4: Some(v4_round_half_up(quality_raw)),
        final_candidate: Some(final_candidate),
        lowest_candidate: Some(lowest_candidate),
        final_equals_lowest: false,
        qualifying_deep_episodes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rr_report(unix: u32, rr: &[u16]) -> QualityRrReport {
        QualityRrReport {
            unix,
            rr: rr.to_vec(),
            optical_signal_poor: None,
            quality_valid: true,
        }
    }

    #[test]
    fn whoop4_rolling_reports_are_normalized_by_time_not_value() {
        let raw = vec![
            rr_report(100, &[800, 810, 820]),
            rr_report(101, &[810, 820, 830]),
            rr_report(102, &[820, 830, 840]),
            rr_report(103, &[830, 840, 850]),
        ];
        let normalized = normalize_whoop4_rr_reports(&raw);
        let retained: Vec<u16> = normalized.iter().flat_map(|r| r.rr.iter().copied()).collect();
        let beat_time: u64 = retained.iter().map(|&v| u64::from(v)).sum();
        assert!((beat_time as f64 / 4_000.0 - 1.0).abs() < 0.20);
        assert!(retained.len() < raw.iter().map(|r| r.rr.len()).sum());
        assert_eq!(retained, vec![820, 830, 840, 850]);
    }

    #[test]
    fn whoop4_equal_real_beats_are_not_value_deduplicated() {
        let raw = vec![rr_report(100, &[800, 800, 800]), rr_report(101, &[800, 800, 800])];
        let normalized = normalize_whoop4_rr_reports(&raw);
        assert_eq!(normalized.iter().flat_map(|r| &r.rr).copied().collect::<Vec<_>>(), vec![800, 800]);
    }

    #[test]
    fn whoop4_gap_is_not_backfilled_from_the_next_rolling_report() {
        let raw = vec![rr_report(100, &[800, 810, 820]), rr_report(103, &[810, 820, 830])];
        let normalized = normalize_whoop4_rr_reports(&raw);
        assert_eq!(normalized[0].rr, vec![820]);
        assert_eq!(normalized[1].rr, vec![830]);
        let quality = rr_window_quality(100, 104, &normalized);
        assert_eq!(quality.contiguous_pairs, 0, "the missing report seconds must remain an RMSSD seam");
    }

    #[test]
    fn whoop5_path_is_value_for_value_unchanged() {
        let raw = vec![rr_report(100, &[800, 810, 820]), rr_report(101, &[810, 820, 830])];
        let unchanged = match RrDeviceGeneration::Whoop5Mg {
            RrDeviceGeneration::Whoop4 => normalize_whoop4_rr_reports(&raw),
            RrDeviceGeneration::Whoop5Mg => raw.clone(),
        };
        assert_eq!(unchanged, raw);
    }

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
    fn final_sws_hrv_rejects_wrist_off() {
        let reports: Vec<QualityRrReport> = (300..540)
            .map(|unix| QualityRrReport {
                unix,
                rr: vec![800],
                optical_signal_poor: None,
                quality_valid: true,
            })
            .collect();
        let mut off_wrist = quality(0, 600);
        off_wrist.wrist_off_fraction = 0.02;
        assert_eq!(
            final_sws_last_five_hrv(&[off_wrist], &reports).rejection_reason,
            Some(HrvUnavailableReason::NoReliableDeepEpisode)
        );
    }

    /// `rr_report_coverage` counts distinct report-second density, not RR temporal coverage: at
    /// low HR a report arrives roughly once per beat, so this alone is naturally low even at 100%
    /// beat-time coverage. It must not, by itself, make an otherwise-clean episode/window unreliable.
    #[test]
    fn sub_80_percent_report_coverage_alone_does_not_reject_a_deep_episode() {
        let reports: Vec<QualityRrReport> = (300..600)
            .map(|unix| QualityRrReport {
                unix,
                rr: vec![800],
                optical_signal_poor: None,
                quality_valid: true,
            })
            .collect();
        let mut low_coverage = quality(0, 600);
        low_coverage.rr_report_coverage = 0.79;
        let result = final_sws_last_five_hrv(&[low_coverage], &reports);
        assert_eq!(result.measurement_mode, Some(HrvMeasurementMode::PrimaryFinalSws));
        assert!(result.rmssd_ms.is_some());
    }

    /// Production regression: ~45bpm cadence yields ~75% distinct report-second density even
    /// though RR beat-time coverage is ~100% (interval ~1333ms matches average HR of 45bpm).
    /// Must be accepted at both episode and window reliability, not refused for report density.
    #[test]
    fn low_hr_rr_stream_with_full_beat_time_coverage_is_not_rejected_for_report_density() {
        let reports: Vec<QualityRrReport> = (0u32..1080)
            .filter(|unix| unix % 4 != 3)
            .map(|unix| QualityRrReport {
                unix,
                rr: vec![1333],
                optical_signal_poor: None,
                quality_valid: true,
            })
            .collect();
        let episode = quality(0, 1080);
        let result = final_sws_last_five_hrv(&[episode], &reports);
        assert_eq!(result.measurement_mode, Some(HrvMeasurementMode::PrimaryFinalSws));
        assert!(result.rmssd_ms.is_some());
        // `selected_episode_quality` is the passed-through INPUT episode struct (see
        // `select_window`'s success branch), not the recomputed per-window quality - it is not the
        // right field to check real report density on. The real recomputed count is
        // `usable_report_seconds` (`rr.report_seconds` for the actual 300s window selected).
        let window_seconds = result.selected_window.map(|(s, e)| e - s).unwrap();
        let actual_coverage = result.usable_report_seconds as f64 / window_seconds as f64;
        assert!(
            actual_coverage < 0.80,
            "must actually exercise the low report-density regime: got {actual_coverage} ({}/{window_seconds})",
            result.usable_report_seconds
        );
    }

    /// Production regression for the absolute-count guard: a 300s HRV window at ~45bpm carries
    /// only ~225 distinct RR report timestamps (well under the old 240 floor), but beat-time
    /// coverage is ~100%, artifact rejection is low, and RR/HR agreement and HR/accel/clean-HR/
    /// wrist-off quality are all good. `MIN_HRV_REPORT_SECONDS` must no longer refuse this window.
    #[test]
    fn low_report_second_count_under_the_old_240_floor_still_produces_hrv() {
        let reports: Vec<QualityRrReport> = (0u32..600)
            .filter(|unix| unix % 4 != 3) // 75% density -> 225 report-seconds in any 300s span
            .map(|unix| QualityRrReport {
                unix,
                rr: vec![1333], // ~45.01bpm, matching the ~100% beat-time-coverage production case
                optical_signal_poor: None,
                quality_valid: true,
            })
            .collect();
        let episode = quality(0, 600);
        let result = final_sws_last_five_hrv(&[episode], &reports);
        assert_eq!(result.measurement_mode, Some(HrvMeasurementMode::PrimaryFinalSws));
        assert!(result.rmssd_ms.is_some());
        assert_eq!(
            result.usable_report_seconds, 225,
            "must land under the old 240-report-second floor"
        );
        // `selected_episode_quality` is the passed-through INPUT episode struct, not the recomputed
        // per-window quality (see the sibling low-report-density test above) - recompute the real
        // window quality directly to check beat-time ratio and artifact rejection for real.
        let (window_start, window_end) = result.selected_window.unwrap();
        let rr = rr_window_quality(window_start, window_end, &reports);
        let beat_time_ratio = rr.beat_time_ms as f64 / 1000.0 / f64::from(window_end - window_start);
        assert!(beat_time_ratio > 0.95, "beat-time coverage must be near-complete: got {beat_time_ratio}");
        assert!(rr.artifact_rejection < 0.05, "artifact rejection must be low: got {}", rr.artifact_rejection);
    }

    /// Negative counterpart: genuinely sparse/interrupted RR data (bad beat-time ratio) must still
    /// be refused. Removing the report-second-count guard must not make sparse RR data eligible -
    /// temporal RR coverage stays protected by `rr_beat_time_ratio` (see also
    /// `final_sws_hrv_rejects_bad_rr_beat_time_ratio`, `reliable_window_rejects_bad_rr_beat_time_ratio`).
    #[test]
    fn sparse_rr_coverage_is_still_rejected_via_beat_time_ratio_not_report_count() {
        // Same 75%-density report pattern as the positive case above, but each report's interval
        // is tiny (100ms), so beat-time coverage collapses far below MIN_RR_BEAT_TIME_RATIO even
        // though the report-second count (225) is identical to the accepted production case.
        let reports: Vec<QualityRrReport> = (0u32..600)
            .filter(|unix| unix % 4 != 3)
            .map(|unix| QualityRrReport {
                unix,
                rr: vec![100],
                optical_signal_poor: None,
                quality_valid: true,
            })
            .collect();
        let episode = quality(0, 600);
        let result = final_sws_last_five_hrv(&[episode], &reports);
        assert_eq!(result.measurement_mode, None);
        assert!(result.rmssd_ms.is_none());
    }

    #[test]
    fn reliable_episode_rejects_bad_rr_beat_time_ratio() {
        let mut e = quality(0, 600);
        e.rr_beat_time_ratio = 0.5;
        assert!(!reliable_episode(&e, 600));
        e.rr_beat_time_ratio = 1.0;
        assert!(reliable_episode(&e, 600));
    }

    #[test]
    fn reliable_window_rejects_bad_rr_beat_time_ratio() {
        let mut q = quality(0, 600);
        let rr = RrWindowQuality {
            report_seconds: 300,
            input_intervals: 300,
            clean_intervals: 300,
            contiguous_pairs: 299,
            sudden_change_pairs_rejected: 0,
            beat_time_ms: 90_000,
            artifact_rejection: 0.0,
            rmssd_ms: Some(5.0),
        };
        q.rr_beat_time_ratio = 0.3;
        assert!(!reliable_window(&q, &rr));
        q.rr_beat_time_ratio = 1.5;
        assert!(!reliable_window(&q, &rr));
        q.rr_beat_time_ratio = 1.0;
        assert!(reliable_window(&q, &rr));
    }

    #[test]
    fn reliable_episode_ignores_rr_report_coverage() {
        let mut e = quality(0, 600);
        e.rr_report_coverage = 0.0;
        assert!(reliable_episode(&e, 600), "rr_report_coverage must no longer gate episode reliability");
    }

    #[test]
    fn reliable_window_ignores_rr_report_coverage() {
        let mut q = quality(0, 600);
        q.rr_report_coverage = 0.0;
        let rr = RrWindowQuality {
            report_seconds: 300,
            input_intervals: 300,
            clean_intervals: 300,
            contiguous_pairs: 299,
            sudden_change_pairs_rejected: 0,
            beat_time_ms: 270_000,
            artifact_rejection: 0.0,
            rmssd_ms: Some(5.0),
        };
        assert!(reliable_window(&q, &rr), "rr_report_coverage must no longer gate window reliability");
    }

    #[test]
    fn final_sws_hrv_rejects_bad_rr_beat_time_ratio() {
        let reports: Vec<QualityRrReport> = (300..600)
            .map(|unix| QualityRrReport {
                unix,
                rr: vec![300],
                optical_signal_poor: None,
                quality_valid: true,
            })
            .collect();
        let episode = quality(0, 600);
        let result = final_sws_last_five_hrv(&[episode], &reports);
        assert_eq!(result.measurement_mode, None);
        assert!(result.rmssd_ms.is_none());
    }

    #[test]
    fn episode_gate_failures_no_longer_flags_report_coverage_but_still_flags_beat_time_ratio() {
        let mut e = quality(0, 600);
        e.rr_report_coverage = 0.5;
        assert!(!episode_gate_failures(&e).contains(&"rr_report_coverage"));
        e.rr_report_coverage = 1.0;
        e.rr_beat_time_ratio = 0.5;
        assert!(episode_gate_failures(&e).contains(&"rr_beat_time_ratio"));
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
            Some(HrvUnavailableReason::InsufficientCleanIntervals),
            "the primary episode's window has zero overlapping RR reports"
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
            Some(HrvUnavailableReason::InsufficientCleanIntervals),
            "no reports were supplied at all"
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
            Some(HrvUnavailableReason::InsufficientCleanIntervals),
            "optical_signal_poor reports are untrusted, leaving zero clean intervals"
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

#[cfg(test)]
mod rhr_v4_tests {
    use super::*;

    /// One second of HR per tick across `[start, end)`, all at `bpm` — the plain building block
    /// every fixture below composes from.
    fn flat_hr(start: i64, end: i64, bpm: u16) -> Vec<HrSample> {
        (start..end).map(|ts| HrSample { ts, bpm }).collect()
    }

    fn deep(start: i64, end: i64) -> StageSegment {
        StageSegment { start, end, stage: SleepStage::Deep }
    }

    fn light(start: i64, end: i64) -> StageSegment {
        StageSegment { start, end, stage: SleepStage::Light }
    }

    const NO_ACCEL: &[AccelSample] = &[];
    const NO_WRIST_OFF: &[(i64, i64)] = &[];

    #[test]
    fn final_and_lowest_distinct_blends_both_variants() {
        // Episode 1 (early, warm, 600s): H ~60. Episode 2 (final, cooler, 600s): H ~50.
        // Final (ep2) != lowest (ep2 is also the lowest here would collapse to Tier 2, so make a
        // THIRD, middle episode the lowest instead, keeping final (ep3, warmer again) distinct.
        let hr = [flat_hr(0, 600, 60), flat_hr(1200, 1800, 48), flat_hr(2400, 3000, 55)].concat();
        let stages = [deep(0, 600), deep(1200, 1800), deep(2400, 3000)];
        let r = dynamic_rhr_v4(0, 3000, &hr, NO_ACCEL, NO_WRIST_OFF, &stages);
        assert_eq!(r.tier, RhrV4Tier::FinalAndLowestDistinct);
        assert_eq!(r.qualifying_deep_episodes, 3);
        let f = r.final_candidate.as_ref().unwrap();
        let l = r.lowest_candidate.as_ref().unwrap();
        assert_eq!(f.start, 2400, "final must be the chronologically last qualifying episode");
        assert_eq!(l.start, 1200, "lowest must be the lowest-valued qualifying episode, not ep1");
        assert!(!r.final_equals_lowest);
        let expected_fixed = 0.60 * f.value + 0.40 * l.value;
        assert!((r.fixed_v4_raw.unwrap() - expected_fixed).abs() < 1e-9);
        assert_eq!(r.fixed_v4, Some(v4_round_half_up(expected_fixed)));
        // quality-weighted variant differs from the fixed one whenever Q_F != Q_L (same duration
        // here, so this mainly documents that both variants are computed and both are present).
        assert!(r.quality_v4_raw.is_some());
    }

    #[test]
    fn final_equals_lowest_uses_the_single_value_without_double_counting() {
        // Three qualifying episodes; the LAST one is also the lowest-valued one.
        let hr = [flat_hr(0, 600, 60), flat_hr(1200, 1800, 55), flat_hr(2400, 3000, 47)].concat();
        let stages = [deep(0, 600), deep(1200, 1800), deep(2400, 3000)];
        let r = dynamic_rhr_v4(0, 3000, &hr, NO_ACCEL, NO_WRIST_OFF, &stages);
        assert_eq!(r.tier, RhrV4Tier::FinalEqualsLowest);
        assert_eq!(r.qualifying_deep_episodes, 3, "all 3 episodes were seen, not collapsed away");
        let f = r.final_candidate.as_ref().unwrap();
        let l = r.lowest_candidate.as_ref().unwrap();
        assert_eq!(f.start, l.start, "final and lowest must be the SAME episode");
        assert!(r.final_equals_lowest);
        assert_eq!(r.fixed_v4_raw, r.quality_v4_raw);
        assert!((r.fixed_v4_raw.unwrap() - 47.0).abs() < 1e-9);
        assert_eq!(r.fixed_v4, Some(47));
        assert_eq!(r.quality_v4, Some(47));
    }

    #[test]
    fn exactly_one_reliable_deep_episode_uses_its_own_value() {
        let hr = flat_hr(0, 600, 51);
        let stages = [deep(0, 600)];
        let r = dynamic_rhr_v4(0, 600, &hr, NO_ACCEL, NO_WRIST_OFF, &stages);
        assert_eq!(r.tier, RhrV4Tier::SingleReliableEpisode);
        assert_eq!(r.qualifying_deep_episodes, 1);
        assert_eq!(r.fixed_v4, Some(51));
        assert_eq!(r.quality_v4, Some(51));
        assert!(r.final_equals_lowest);
    }

    #[test]
    fn short_final_deep_fragment_is_never_selected_merely_for_being_low() {
        // A long, reliable early episode at 55, then a 120-second (< 300s floor) FINAL fragment
        // at a much lower 40 bpm. The fragment must be rejected outright, not chosen as "final".
        let hr = [flat_hr(0, 600, 55), flat_hr(900, 1020, 40)].concat();
        let stages = [deep(0, 600), deep(900, 1020)];
        let r = dynamic_rhr_v4(0, 1020, &hr, NO_ACCEL, NO_WRIST_OFF, &stages);
        assert_eq!(r.tier, RhrV4Tier::SingleReliableEpisode, "the 120s fragment must not qualify");
        assert_eq!(r.qualifying_deep_episodes, 1);
        assert_eq!(r.fixed_v4, Some(55), "must reflect the long episode, not the rejected 40bpm fragment");
    }

    #[test]
    fn fragmented_deep_with_no_reliable_episode_falls_back_to_pooled_deep() {
        // Five Deep fragments, each 120s (< 300s floor) so none is individually reliable, but
        // their pooled quality-valid seconds (600s) clears the 300s pooled-Deep floor.
        let mut hr = Vec::new();
        let mut stages = Vec::new();
        for i in 0..5 {
            let start = i * 300;
            hr.extend(flat_hr(start, start + 120, 48));
            stages.push(deep(start, start + 120));
        }
        let r = dynamic_rhr_v4(0, 1500, &hr, NO_ACCEL, NO_WRIST_OFF, &stages);
        assert_eq!(r.tier, RhrV4Tier::PooledDeep);
        assert_eq!(r.qualifying_deep_episodes, 0);
        assert_eq!(r.fixed_v4, Some(48));
    }

    #[test]
    fn no_deep_at_all_falls_back_to_whole_sleep_not_raw_minimum() {
        let hr = [flat_hr(0, 600, 58), flat_hr(600, 1200, 62)].concat();
        let stages = [light(0, 1200)];
        let r = dynamic_rhr_v4(0, 1200, &hr, NO_ACCEL, NO_WRIST_OFF, &stages);
        assert_eq!(r.tier, RhrV4Tier::WholeSleep);
        assert_eq!(r.qualifying_deep_episodes, 0);
        // trimmed mean of {58*600, 62*600} must sit near 60, nowhere near the raw min (58).
        assert!(r.fixed_v4.unwrap() >= 59);
    }

    #[test]
    fn wrist_off_contamination_can_drop_an_episode_below_the_coverage_gate() {
        // A 600s Deep episode where 200s (33%) is wrist-off: coverage drops to ~0.67, below the
        // 0.80 floor, so it must NOT qualify, even though 300s of raw duration is enough on paper.
        let hr = flat_hr(0, 600, 50);
        let stages = [deep(0, 600)];
        let wrist_off = [(0i64, 200i64)];
        let r = dynamic_rhr_v4(0, 600, &hr, NO_ACCEL, &wrist_off, &stages);
        assert_ne!(r.tier, RhrV4Tier::SingleReliableEpisode, "contaminated episode must not qualify");
        assert_eq!(r.qualifying_deep_episodes, 0);
    }

    #[test]
    fn poor_sample_coverage_is_rejected_even_with_enough_duration() {
        // 600s span, but only every 3rd second has a sample -> ~33% coverage, well under 0.80.
        let hr: Vec<HrSample> = (0..600).step_by(3).map(|ts| HrSample { ts, bpm: 49 }).collect();
        let stages = [deep(0, 600)];
        let r = dynamic_rhr_v4(0, 600, &hr, NO_ACCEL, NO_WRIST_OFF, &stages);
        assert_eq!(r.qualifying_deep_episodes, 0, "sparse coverage must fail the reliability gate");
    }

    #[test]
    fn duration_confidence_saturates_instead_of_growing_without_bound() {
        // Same coverage (1.0) and stability (0.0 MAD, flat bpm) at three durations either side of
        // the saturation point: a 10-min episode must NOT score 3x a 30-min one.
        let q_10min = v4_quality_score(600, 1.0, 0.0);
        let q_20min = v4_quality_score(1200, 1.0, 0.0);
        let q_30min = v4_quality_score(1800, 1.0, 0.0);
        assert!((q_10min - 1.0).abs() < 1e-9, "600s == the saturation point -> confidence 1.0");
        assert!((q_20min - 1.0).abs() < 1e-9);
        assert!((q_30min - 1.0).abs() < 1e-9, "no unbounded growth past saturation");
        assert_eq!(q_10min, q_30min, "a 3x-longer episode must not get 3x the confidence");
        // Below saturation, confidence scales linearly, not to zero and not to 1.0 early.
        let q_5min = v4_quality_score(300, 1.0, 0.0);
        assert!((q_5min - 0.5).abs() < 1e-9);
    }

    #[test]
    fn low_hr_does_not_increase_quality() {
        // Two episodes, identical duration/coverage/dispersion, differing ONLY in absolute HR
        // level (45 vs 65). Their quality scores must be exactly equal.
        let hr = [flat_hr(0, 600, 45), flat_hr(1200, 1800, 65)].concat();
        let stages = [deep(0, 600), deep(1200, 1800)];
        let r = dynamic_rhr_v4(0, 1800, &hr, NO_ACCEL, NO_WRIST_OFF, &stages);
        let f = r.final_candidate.as_ref().unwrap();
        let l = r.lowest_candidate.as_ref().unwrap();
        assert!((f.value - 65.0).abs() < 1e-9);
        assert!((l.value - 45.0).abs() < 1e-9);
        assert!(
            (f.quality - l.quality).abs() < 1e-9,
            "the 45bpm episode must not score higher quality merely for being lower: {} vs {}",
            l.quality, f.quality
        );
    }

    #[test]
    fn intermediate_candidate_values_are_never_pre_rounded() {
        // 300 samples: 240 at 47bpm, 60 at 50bpm -> 10%-trimmed mean is a genuine non-integer
        // (47.375), not something that happens to land on a whole number.
        let mut hr: Vec<HrSample> = (0..240).map(|ts| HrSample { ts, bpm: 47 }).collect();
        hr.extend((240..300).map(|ts| HrSample { ts, bpm: 50 }));
        let stages = [deep(0, 300)];
        let r = dynamic_rhr_v4(0, 300, &hr, NO_ACCEL, NO_WRIST_OFF, &stages);
        let value = r.final_candidate.as_ref().unwrap().value;
        assert!((value - 47.375).abs() < 1e-9, "got {value}");
        assert!(r.fixed_v4_raw.unwrap().fract().abs() > 1e-9, "raw value must retain its fraction");
        assert_eq!(r.fixed_v4, Some(47), "only the FINAL presentation value is rounded");
    }

    #[test]
    fn final_rounding_is_deterministic_half_up() {
        assert_eq!(v4_round_half_up(49.5), 50, "ties round toward positive infinity");
        assert_eq!(v4_round_half_up(49.4999), 49);
        assert_eq!(v4_round_half_up(49.5001), 50);
        assert_eq!(v4_round_half_up(-0.5), 0, "the ties-up convention applies uniformly");
        // Same input rounded twice must be bit-for-bit identical (no hidden nondeterminism from
        // iteration order, hashing, or float-summation order across repeated calls).
        assert_eq!(v4_round_half_up(52.5), v4_round_half_up(52.5));
    }

    /// Tier 6: no Deep at all AND no whole-sleep quality-valid samples either -> `None` outright,
    /// never a fabricated number.
    #[test]
    fn insufficient_data_anywhere_is_unavailable_not_a_fabricated_value() {
        let hr: Vec<HrSample> = Vec::new();
        let stages = [light(0, 600)];
        let r = dynamic_rhr_v4(0, 600, &hr, NO_ACCEL, NO_WRIST_OFF, &stages);
        assert_eq!(r.tier, RhrV4Tier::Unavailable);
        assert_eq!(r.qualifying_deep_episodes, 0);
        assert_eq!(r.fixed_v4, None);
        assert_eq!(r.quality_v4, None);
        assert_eq!(r.fixed_v4_raw, None);
        assert_eq!(r.quality_v4_raw, None);
    }

    /// The emitted `algorithm_version` on every `RhrV4Result` - regardless of tier - is the v4
    /// identifier, and it is textually distinct from the v3 identifier `dynamic_rhr` still emits.
    #[test]
    fn emitted_algorithm_version_is_the_v4_identifier() {
        assert_ne!(ALGORITHM_VERSION_V4, ALGORITHM_VERSION);
        assert_eq!(ALGORITHM_VERSION_V4, "physiology-dynamic-rhr-sws-v4");

        let hr = flat_hr(0, 600, 51);
        let stages = [deep(0, 600)];
        let single = dynamic_rhr_v4(0, 600, &hr, NO_ACCEL, NO_WRIST_OFF, &stages);
        assert_eq!(single.algorithm_version, ALGORITHM_VERSION_V4);

        let unavailable = dynamic_rhr_v4(0, 600, &[], NO_ACCEL, NO_WRIST_OFF, &[light(0, 600)]);
        assert_eq!(unavailable.algorithm_version, ALGORITHM_VERSION_V4);
    }
}

/// Proves the v4 pipeline is actually wired into `nightly_physiology` - the entry point
/// `noop-engine` calls for both the per-session and the grouped/authoritative nightly RHR - and
/// that doing so leaves the v3 helper and the unrelated HRV pipeline untouched.
#[cfg(test)]
mod nightly_physiology_v4_wiring_tests {
    use super::*;

    fn flat_hr(start: i64, end: i64, bpm: u16) -> Vec<HrSample> {
        (start..end).map(|ts| HrSample { ts, bpm }).collect()
    }

    /// `nightly_physiology`'s `rhr_v4` field must be bit-for-bit identical to calling
    /// `dynamic_rhr_v4` directly on the same inputs - `nightly_physiology` must not be running a
    /// second, divergent copy of the v4 pipeline.
    #[test]
    fn nightly_physiology_rhr_v4_matches_a_direct_call() {
        let stages = [StageSegment { start: 0, end: 600, stage: SleepStage::Deep }];
        let hr = flat_hr(0, 600, 52);
        let night = nightly_physiology(0, 600, &hr, &[], &[], &[], &stages);

        let direct = dynamic_rhr_v4(0, 600, &hr, &[], &[], &stages);
        assert_eq!(night.rhr_v4, direct);
        assert_eq!(night.rhr_v4.algorithm_version, ALGORITHM_VERSION_V4);
        assert_eq!(night.rhr_v4.quality_v4, Some(52));
    }

    /// Promoting v4 must not perturb the v3 `dynamic_rhr` helper still riding along on `rhr`: same
    /// inputs, same v3-only formula, same v3 version stamp, still directly testable.
    #[test]
    fn nightly_physiology_rhr_v3_field_is_unaffected_by_the_v4_addition() {
        let stages = [StageSegment { start: 0, end: 600, stage: SleepStage::Deep }];
        let hr = flat_hr(0, 600, 52);
        let night = nightly_physiology(0, 600, &hr, &[], &[], &[], &stages);

        let quality_hr: Vec<QualityHrSample> = (0..600)
            .map(|unix| QualityHrSample { unix, bpm: 52, quality_valid: true })
            .collect();
        let direct_v3 = dynamic_rhr(0, 600, &quality_hr, &stages);
        assert_eq!(night.rhr, direct_v3);
        assert_eq!(night.rhr.as_ref().unwrap().algorithm_version, ALGORITHM_VERSION);
    }

    /// The bundled nightly-physiology version stamp (what noop-engine actually emits as
    /// `algorithm_version`, and what feeds `daily_metrics`/`daily_canonical`/`sleep_sessions`) is
    /// the v4 identifier now that v4 is authoritative.
    #[test]
    fn nightly_physiology_algorithm_version_is_v4() {
        let stages = [StageSegment { start: 0, end: 600, stage: SleepStage::Deep }];
        let hr = flat_hr(0, 600, 52);
        let night = nightly_physiology(0, 600, &hr, &[], &[], &[], &stages);
        assert_eq!(night.algorithm_version, ALGORITHM_VERSION_V4);
    }

    /// HRV is explicitly "unrelated physiology" for this change: same formula, same reliability
    /// gates, same v3 version stamp it always emitted - completely untouched by the RHR promotion.
    #[test]
    fn nightly_physiology_hrv_is_unaffected_by_the_v4_addition() {
        let stages = [StageSegment { start: 0, end: 600, stage: SleepStage::Deep }];
        let hr = flat_hr(0, 600, 52);
        let night = nightly_physiology(0, 600, &hr, &[], &[], &[], &stages);
        // No RR data supplied, so HRV must refuse the same way it always has - unrelated to RHR.
        assert!(night.hrv.rmssd_ms.is_none());
        assert_eq!(night.hrv.algorithm_version, ALGORITHM_VERSION);
    }
}
