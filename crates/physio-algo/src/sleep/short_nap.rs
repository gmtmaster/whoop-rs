//! Non-authoritative evidence for post-merge sleep candidates rejected only by duration policy.
//! The evaluator observes the active detector's boundaries and never emits a sleep session.

use std::collections::HashSet;

use super::common::median;
use super::detect::{run_is_deeply_quiescent, Period};
use super::input::{AccelSample, HrSample, RrRun, StepSample};

// LEGACY_PRECEDENT: the disconnected review-only nap proposal used this inclusive range.
const MIN_SHORT_NAP_MIN: i64 = 20;
const MAX_SHORT_NAP_MIN: i64 = 90;

// EXPERIMENTAL_SHADOW: these values collect evidence only and cannot create a session.
const SURROUNDING_HR_WINDOW_MIN: i64 = 30;
const HR_DROP_MIN_FRACTION: f64 = 0.10;
const RR_COVERAGE_MIN_FRACTION: f64 = 0.80;
const NON_LOCOMOTION_MIN_FRACTION: f64 = 0.95;
const MIN_CORROBORATORS: usize = 2;

// EXISTING: the active morning re-onset guard already uses this band-asleep fraction.
const BAND_ASLEEP_MIN_FRACTION: f64 = 0.60;
const BAND_STATE_ASLEEP: i32 = 2;
const SECONDS_PER_DAY: i64 = 86_400;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShortNapOriginalRejection {
    BaseMinimumDuration,
    DaytimeMinimumDuration,
}

impl ShortNapOriginalRejection {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::BaseMinimumDuration => "base_minimum_duration",
            Self::DaytimeMinimumDuration => "daytime_minimum_duration",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShortNapShadowVerdict {
    Eligible,
    InsufficientEvidence,
    Rejected,
}

impl ShortNapShadowVerdict {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Eligible => "ELIGIBLE",
            Self::InsufficientEvidence => "INSUFFICIENT_EVIDENCE",
            Self::Rejected => "REJECTED",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShortNapCorroborators {
    pub hr_drop: bool,
    pub rr_coverage: bool,
    pub non_locomotion: bool,
    pub band_asleep: bool,
    pub deeply_quiescent: bool,
}

impl ShortNapCorroborators {
    fn count(self) -> usize {
        [
            self.hr_drop,
            self.rr_coverage,
            self.non_locomotion,
            self.band_asleep,
            self.deeply_quiescent,
        ]
        .into_iter()
        .filter(|value| *value)
        .count()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ShortNapShadowDiagnostic {
    pub start: i64,
    pub end: i64,
    pub duration_seconds: i64,
    pub original_rejection_reason: ShortNapOriginalRejection,
    pub is_daytime: bool,
    pub local_center_hour: i64,
    pub candidate_median_hr: Option<f64>,
    pub surrounding_hr_baseline: Option<f64>,
    pub hr_drop_percentage: Option<f64>,
    pub resting_floor: Option<i32>,
    pub hr_gate_passed: bool,
    pub daytime_floor_passed: bool,
    pub morning_reonset_applies: bool,
    pub morning_reonset_passed: bool,
    pub rr_coverage_fraction: Option<f64>,
    pub non_locomotion_fraction: Option<f64>,
    pub band_asleep_fraction: Option<f64>,
    pub off_wrist_fraction: f64,
    pub corroborators: ShortNapCorroborators,
    pub verdict: ShortNapShadowVerdict,
    pub reasons: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct DurationRejectedCandidate {
    pub period: Period,
    pub original_rejection_reason: ShortNapOriginalRejection,
    pub tz_offset_s: i64,
    pub is_daytime: bool,
    pub candidate_median_hr: Option<f64>,
    pub resting_floor: Option<i32>,
    pub hr_gate_passed: bool,
    pub daytime_floor_passed: bool,
    pub morning_reonset_applies: bool,
    pub morning_reonset_passed: bool,
    pub off_wrist_fraction: f64,
}

pub(super) fn evaluate(
    candidate: DurationRejectedCandidate,
    hr: &[HrSample],
    rr: &[RrRun],
    accel: &[AccelSample],
    steps: &[StepSample],
    band_sleep_state: &[(i64, i32)],
) -> ShortNapShadowDiagnostic {
    let period = candidate.period;
    let duration_seconds = period.end - period.start;
    let duration_minutes = duration_seconds as f64 / 60.0;
    let local_center_hour = (period.start + duration_seconds / 2 + candidate.tz_offset_s)
        .rem_euclid(SECONDS_PER_DAY)
        / 3_600;
    let surrounding_hr_baseline = surrounding_hr_baseline(period, hr);
    let hr_drop_percentage = candidate.candidate_median_hr.and_then(|inside| {
        surrounding_hr_baseline
            .filter(|outside| *outside > 0.0)
            .map(|outside| (outside - inside) / outside * 100.0)
    });
    let rr_coverage_fraction = rr_coverage(period, rr);
    let non_locomotion_fraction = non_locomotion_fraction(period, steps);
    let band_asleep_fraction = band_asleep_fraction(period, band_sleep_state);
    let corroborators = ShortNapCorroborators {
        hr_drop: hr_drop_percentage.is_some_and(|drop| drop >= HR_DROP_MIN_FRACTION * 100.0),
        rr_coverage: rr_coverage_fraction
            .is_some_and(|coverage| coverage >= RR_COVERAGE_MIN_FRACTION),
        non_locomotion: non_locomotion_fraction
            .is_some_and(|fraction| fraction >= NON_LOCOMOTION_MIN_FRACTION),
        band_asleep: band_asleep_fraction
            .is_some_and(|fraction| fraction >= BAND_ASLEEP_MIN_FRACTION),
        deeply_quiescent: run_is_deeply_quiescent(period, accel),
    };

    let hard_rejected = !candidate.is_daytime
        || duration_minutes < MIN_SHORT_NAP_MIN as f64
        || duration_minutes > MAX_SHORT_NAP_MIN as f64;
    let mut reasons = Vec::new();
    if !candidate.is_daytime {
        reasons.push("outside_daytime_band".to_string());
    }
    if duration_minutes < MIN_SHORT_NAP_MIN as f64 {
        reasons.push("below_legacy_short_nap_range".to_string());
    }
    if duration_minutes > MAX_SHORT_NAP_MIN as f64 {
        reasons.push("above_legacy_short_nap_range".to_string());
    }

    for (name, passed) in [
        ("hr_drop", corroborators.hr_drop),
        ("rr_coverage", corroborators.rr_coverage),
        ("non_locomotion", corroborators.non_locomotion),
        ("band_asleep", corroborators.band_asleep),
        ("deeply_quiescent", corroborators.deeply_quiescent),
    ] {
        if passed {
            reasons.push(format!("corroborator:{name}"));
        }
    }

    let verdict = if hard_rejected {
        ShortNapShadowVerdict::Rejected
    } else if corroborators.count() >= MIN_CORROBORATORS {
        reasons.push(format!(
            "corroborator_count:{}_required:{MIN_CORROBORATORS}",
            corroborators.count()
        ));
        ShortNapShadowVerdict::Eligible
    } else {
        reasons.push(format!(
            "insufficient_corroborators:{}_required:{MIN_CORROBORATORS}",
            corroborators.count()
        ));
        ShortNapShadowVerdict::InsufficientEvidence
    };

    ShortNapShadowDiagnostic {
        start: period.start,
        end: period.end,
        duration_seconds,
        original_rejection_reason: candidate.original_rejection_reason,
        is_daytime: candidate.is_daytime,
        local_center_hour,
        candidate_median_hr: candidate.candidate_median_hr,
        surrounding_hr_baseline,
        hr_drop_percentage,
        resting_floor: candidate.resting_floor,
        hr_gate_passed: candidate.hr_gate_passed,
        daytime_floor_passed: candidate.daytime_floor_passed,
        morning_reonset_applies: candidate.morning_reonset_applies,
        morning_reonset_passed: candidate.morning_reonset_passed,
        rr_coverage_fraction,
        non_locomotion_fraction,
        band_asleep_fraction,
        off_wrist_fraction: candidate.off_wrist_fraction,
        corroborators,
        verdict,
        reasons,
    }
}

fn values_in_period(period: Period, hr: &[HrSample]) -> Vec<f64> {
    hr.iter()
        .filter(|sample| sample.ts >= period.start && sample.ts <= period.end)
        .map(|sample| sample.bpm as f64)
        .collect()
}

pub(super) fn candidate_median_hr(period: Period, hr: &[HrSample]) -> Option<f64> {
    let values = values_in_period(period, hr);
    (!values.is_empty()).then(|| median(&values))
}

fn surrounding_hr_baseline(period: Period, hr: &[HrSample]) -> Option<f64> {
    let pad = SURROUNDING_HR_WINDOW_MIN * 60;
    let values: Vec<f64> = hr
        .iter()
        .filter(|sample| {
            (sample.ts >= period.start - pad && sample.ts < period.start)
                || (sample.ts > period.end && sample.ts <= period.end + pad)
        })
        .map(|sample| sample.bpm as f64)
        .collect();
    (!values.is_empty()).then(|| median(&values))
}

fn rr_coverage(period: Period, rr: &[RrRun]) -> Option<f64> {
    if period.end <= period.start {
        return None;
    }
    let seconds: HashSet<i64> = rr
        .iter()
        .filter(|run| run.ts >= period.start && run.ts < period.end && !run.intervals.is_empty())
        .map(|run| run.ts)
        .collect();
    (!seconds.is_empty())
        .then(|| (seconds.len() as f64 / (period.end - period.start) as f64).min(1.0))
}

fn non_locomotion_fraction(period: Period, steps: &[StepSample]) -> Option<f64> {
    let rows: Vec<&StepSample> = steps
        .iter()
        .filter(|sample| sample.ts >= period.start && sample.ts < period.end)
        .collect();
    (!rows.is_empty()).then(|| {
        rows.iter()
            .filter(|sample| !matches!(sample.activity_class, Some(1 | 2)))
            .count() as f64
            / rows.len() as f64
    })
}

fn band_asleep_fraction(period: Period, states: &[(i64, i32)]) -> Option<f64> {
    let rows: Vec<i32> = states
        .iter()
        .filter(|(ts, _)| *ts >= period.start && *ts < period.end)
        .map(|(_, state)| *state)
        .collect();
    (!rows.is_empty()).then(|| {
        rows.iter()
            .filter(|state| **state == BAND_STATE_ASLEEP)
            .count() as f64
            / rows.len() as f64
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const NAP_A_START: i64 = 1_788_621_707; // 2026-09-05 15:21:47 UTC
    const NAP_A_END: i64 = NAP_A_START + 53 * 60 + 29;

    fn nap_a_candidate() -> DurationRejectedCandidate {
        DurationRejectedCandidate {
            period: Period {
                is_sleep: true,
                start: NAP_A_START,
                end: NAP_A_END,
            },
            original_rejection_reason: ShortNapOriginalRejection::BaseMinimumDuration,
            tz_offset_s: 2 * 3600,
            is_daytime: true,
            candidate_median_hr: Some(70.0),
            resting_floor: Some(69),
            hr_gate_passed: true,
            daytime_floor_passed: true,
            morning_reonset_applies: false,
            morning_reonset_passed: true,
            off_wrist_fraction: 0.0,
        }
    }

    #[test]
    fn nap_a_reference_boundaries_reach_shadow_without_becoming_a_session() {
        let mut hr: Vec<HrSample> = (1..=1_800)
            .map(|offset| HrSample {
                ts: NAP_A_START - offset,
                bpm: 84,
            })
            .collect();
        hr.extend((NAP_A_START..NAP_A_END).map(|ts| HrSample { ts, bpm: 70 }));
        let rr: Vec<RrRun> = (NAP_A_START..NAP_A_END)
            .map(|ts| RrRun {
                ts,
                intervals: vec![850],
            })
            .collect();
        let accel: Vec<AccelSample> = (NAP_A_START..NAP_A_END)
            .map(|ts| AccelSample {
                ts,
                x: 0.0,
                y: 0.0,
                z: 1.0,
            })
            .collect();
        let steps: Vec<StepSample> = (NAP_A_START..NAP_A_END)
            .map(|ts| StepSample {
                ts,
                counter: 0,
                activity_class: Some(0),
            })
            .collect();

        let diagnostic = evaluate(nap_a_candidate(), &hr, &rr, &accel, &steps, &[]);

        assert_eq!((diagnostic.start, diagnostic.end), (NAP_A_START, NAP_A_END));
        assert_eq!(diagnostic.duration_seconds, 53 * 60 + 29);
        assert_eq!(diagnostic.verdict, ShortNapShadowVerdict::Eligible);
    }

    #[test]
    fn sparse_uncorroborated_candidate_is_insufficient_not_accepted() {
        let diagnostic = evaluate(nap_a_candidate(), &[], &[], &[], &[], &[]);
        assert_eq!(
            diagnostic.verdict,
            ShortNapShadowVerdict::InsufficientEvidence
        );
    }

    #[test]
    fn non_daytime_duration_rejection_is_diagnostic_only_and_rejected() {
        let mut candidate = nap_a_candidate();
        candidate.is_daytime = false;
        candidate.tz_offset_s = -8 * 3600;
        let diagnostic = evaluate(candidate, &[], &[], &[], &[], &[]);
        assert_eq!(diagnostic.verdict, ShortNapShadowVerdict::Rejected);
        assert!(diagnostic
            .reasons
            .iter()
            .any(|reason| reason == "outside_daytime_band"));
    }
}
