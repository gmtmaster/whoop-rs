//! Healthspan's intensity classification: light / moderate / vigorous time, computed DIRECTLY from
//! continuous personalized %HRR - not derived from, and not required to agree with, whatever
//! workout Z0-5 boundaries `hrr_zones.rs` uses for display. This is the deliberate architecture
//! decision from the final implementation note: Healthspan bypasses workout-zone semantics entirely.
//!
//! Thresholds: CONFIDENCE HIGH. `<40% HRR` = light, `40-59% HRR` = moderate, `>=60% HRR` = vigorous.
//! Per `hr-intensity-model-spec.md` Part 3/Freeze section: 40% and 60% HRR are the two cut points
//! with the strongest cross-source agreement between ACSM's classification and WHO's relative-
//! intensity equivalents for moderate/vigorous activity - the one part of that whole spec rated
//! stronger than "ACSM-consistent convention." These are population-health-guideline boundaries,
//! not a workout-UX choice, and should not be revised for UI/display reasons - only if the
//! underlying WHO/ACSM guidance itself changes (in which case bump `MODEL_VERSION`, do not mutate
//! the constants in place, for the same historical-reproducibility reason as `hrr_zones.rs`).

use crate::hr_gap::{GapPosition, GapVerdict, classify, creditable_seconds};
use crate::hr_sample::HrSample;
use crate::personal_cardio::{heart_rate_reserve, relative_intensity_pct};

pub const MODEL_VERSION: &str = "healthspan-intensity-v1";

/// CONFIDENCE HIGH (see module doc).
pub const LIGHT_UPPER_HRR_PCT: f64 = 40.0;
/// CONFIDENCE HIGH (see module doc).
pub const VIGOROUS_LOWER_HRR_PCT: f64 = 60.0;

/// One of the three Healthspan-facing intensity bands for a single sample.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HealthspanBand {
    Light,
    Moderate,
    Vigorous,
}

/// Classify a single %HRR value. Exposed on its own (not just inside `time_in_bands`) so callers
/// that already have a %HRR series (e.g. a live session) can classify a single point without
/// building a full sample-and-gap accounting pass.
pub fn band_for_pct_hrr(pct_hrr: f64) -> HealthspanBand {
    if pct_hrr < LIGHT_UPPER_HRR_PCT {
        HealthspanBand::Light
    } else if pct_hrr < VIGOROUS_LOWER_HRR_PCT {
        HealthspanBand::Moderate
    } else {
        HealthspanBand::Vigorous
    }
}

/// Seconds in each Healthspan band over a day/session, plus gap provenance - same shape and same
/// gap-accounting contract as `hr_zones::TimeInZone`, so canonical persistence and downstream
/// consumers can treat it identically (measured vs bridged vs refused).
#[derive(Clone, Debug, PartialEq, Default)]
pub struct TimeInBands {
    pub light_seconds: f64,
    pub moderate_seconds: f64,
    pub vigorous_seconds: f64,
    pub bridged_seconds: f64,
    pub refused_seconds: f64,
}

impl TimeInBands {
    pub fn total(&self) -> f64 {
        self.light_seconds + self.moderate_seconds + self.vigorous_seconds
    }
}

/// Direct %HRR time-in-band over a raw HR stream. `None` when the reserve is non-physiological
/// (`max_hr <= rhr_baseline`) - callers should treat this the same way `hrr_zones::zones_from_hrr`'s
/// `None` is treated: fall back, do not fabricate a classification from a negative reserve.
///
/// Gap handling deliberately mirrors `hr_zones::time_in_zone` (same `hr_gap` ceilings, same
/// bridged/refused bookkeeping) precisely so a day's `light+moderate+vigorous` total and its
/// `zone1..5` total account for gaps identically - the two are independent classifications of the
/// same underlying credited seconds, not two different notions of "how much of the day counts."
pub fn time_in_bands(hr: &[HrSample], max_hr: f64, rhr_baseline: f64) -> Option<TimeInBands> {
    let hrr = heart_rate_reserve(max_hr, rhr_baseline)?;
    let mut sorted: Vec<HrSample> = hr.to_vec();
    sorted.sort_by_key(|s| s.ts);
    if sorted.is_empty() {
        return Some(TimeInBands::default());
    }

    let mut out = TimeInBands::default();
    let tail_duration = crate::hr_zones::median_interval(&sorted);
    for i in 0..sorted.len() {
        let (gap, position) = if i < sorted.len() - 1 {
            let g = (sorted[i + 1].ts - sorted[i].ts) as f64;
            if g > 0.0 {
                (g, GapPosition::Interior)
            } else {
                (tail_duration, GapPosition::Interior)
            }
        } else {
            (tail_duration, GapPosition::Trailing)
        };
        if classify(gap, position) == GapVerdict::Refuse {
            out.refused_seconds += gap;
            continue;
        }
        if classify(gap, position) == GapVerdict::Bridge {
            out.bridged_seconds += gap;
        }
        let dur = creditable_seconds(gap, position);
        let pct = relative_intensity_pct(sorted[i].bpm as f64, rhr_baseline, hrr);
        match band_for_pct_hrr(pct) {
            HealthspanBand::Light => out.light_seconds += dur,
            HealthspanBand::Moderate => out.moderate_seconds += dur,
            HealthspanBand::Vigorous => out.vigorous_seconds += dur,
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn none_for_nonpositive_reserve() {
        assert_eq!(time_in_bands(&[], 180.0, 180.0), None);
    }

    #[test]
    fn empty_stream_is_zero() {
        let bands = time_in_bands(&[], 194.0, 52.0).unwrap();
        assert_eq!(bands.total(), 0.0);
    }

    #[test]
    fn boundary_cases_match_the_frozen_thresholds() {
        // RHR 52, MaxHR 194, HRR 142. 40% -> 108.8bpm, 60% -> 137.2bpm.
        assert_eq!(band_for_pct_hrr(39.9), HealthspanBand::Light);
        assert_eq!(band_for_pct_hrr(40.0), HealthspanBand::Moderate);
        assert_eq!(band_for_pct_hrr(59.9), HealthspanBand::Moderate);
        assert_eq!(band_for_pct_hrr(60.0), HealthspanBand::Vigorous);
    }

    #[test]
    fn the_worked_example_from_the_model_spec_lands_where_the_spec_says() {
        // RHR=52, MaxHr=194 (HRR=142); HR=140 -> 62.0% HRR -> vigorous, matching Part 10's table.
        let hr = vec![HrSample { ts: 0, bpm: 140 }, HrSample { ts: 1, bpm: 140 }];
        let bands = time_in_bands(&hr, 194.0, 52.0).unwrap();
        assert!(bands.vigorous_seconds > 0.0);
        assert_eq!(bands.moderate_seconds, 0.0);
    }

    #[test]
    fn is_independent_of_workout_zone_edges() {
        // Same HR classified as Healthspan-moderate here (120bpm, 47.9% HRR) may sit in a
        // different-numbered workout zone under hrr_zones::CANDIDATE_A_EDGES (Z2, 40-50%) - this
        // test documents that the two are intentionally decoupled, not that they must agree.
        let hr = vec![HrSample { ts: 0, bpm: 120 }, HrSample { ts: 1, bpm: 120 }];
        let bands = time_in_bands(&hr, 194.0, 52.0).unwrap();
        assert!(bands.moderate_seconds > 0.0);
    }
}
