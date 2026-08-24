//! Personalized, HRR-based workout Z0-5 zones - the UI/live-session/strain-continuity consumer of
//! the continuous %HRR primitive in `personal_cardio.rs`. Deliberately a SEPARATE model from
//! `healthspan_intensity.rs`: per the final architecture decision, workout zone boundaries and
//! Healthspan's moderate/vigorous classification are two independent consumers of the same %HRR,
//! not one derived from the other. Do not add a dependency from `healthspan_intensity` on this
//! module's edges, or vice versa.
//!
//! Reuses `hr_zones::{HrZone, HrZoneSet, TimeInZone, time_in_zone}` unchanged for the actual
//! time-accounting (gap/bridge/refuse handling is identical regardless of what produced the bpm
//! edges) - only the edge CONSTRUCTION differs (from %HRR instead of %HRmax). This is the
//! consolidation the implementation request asked for: one gap-accounting algorithm, one HRR
//! formula (`strain::pct_hrr`, via `personal_cardio::heart_rate_reserve`), edges built two ways.

use crate::hr_zones::{HrZone, HrZoneSet};
use crate::personal_cardio::heart_rate_reserve;

/// CONFIDENCE: MEDIUM. This is "Candidate A" from `hr-intensity-model-spec.md` Part 2.2 - the
/// directly ACSM-aligned six-way split (moderate 40-59% HRR evenly split into Z1/Z2, vigorous
/// 60-84% evenly split into Z3/Z4, near-maximal-to-maximal combined into Z5). It is evidence-
/// consistent (ACSM-consistent cut points at 40/60% HRR, which the spec rates moderate-strong), but
/// the exact even split within moderate and within vigorous, and where near-maximal/maximal get
/// combined, are ACSM-edition-dependent / not independently outcome-validated for THIS six-way
/// split as workout-display zones. Versioned and swappable for exactly that reason - see
/// `WorkoutZoneModel` below. NOT the same split as Healthspan's moderate/vigorous boundary (that is
/// intentional; see module doc).
pub const CANDIDATE_A_EDGES: [f64; 6] = [0.40, 0.50, 0.60, 0.70, 0.85, 1.00];

/// A named, versioned set of %HRR edges for the six workout zones. New edge sets get a new
/// `version` string and are additive - never mutate `CANDIDATE_A_EDGES` in place once a version has
/// shipped, since `zone_model_version` is persisted per-day for reproducibility (see the daily
/// canonical provenance columns in the architecture doc, Part C.4/D.5).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WorkoutZoneModel {
    pub version: &'static str,
    pub edges: [f64; 6],
}

impl WorkoutZoneModel {
    /// CONFIDENCE: MEDIUM (see `CANDIDATE_A_EDGES` doc). This is the v1 default, but is passed
    /// explicitly rather than implied, so a caller/config can pin an older or alternate model
    /// without a code change - do not hardcode this as the only option in call sites.
    pub const HRR_V1: WorkoutZoneModel = WorkoutZoneModel {
        version: "hrr-v1",
        edges: CANDIDATE_A_EDGES,
    };
}

/// Build a personalized `HrZoneSet` (bpm-space, ready for `hr_zones::time_in_zone`) from %HRR edges,
/// a Max HR estimate and a chronic RHR baseline. `None` when the reserve is non-physiological
/// (`max_hr <= rhr_baseline`) - callers should fall back to the age/%MaxHR path
/// (`hr_zones::zones_for_age`) in that case, not fabricate zones from a negative reserve.
///
/// `source` on the returned `HrZoneSet` is the model's version string (e.g. `"hrr-v1"`), NOT
/// `"tanaka"`/`"manual"` - this is what lets a daily canonical row record which zone model produced
/// its `zone1Seconds..zone5Seconds`, per the architecture doc's provenance columns.
pub fn zones_from_hrr(
    max_hr: f64,
    rhr_baseline: f64,
    model: WorkoutZoneModel,
) -> Option<HrZoneSet> {
    let hrr = heart_rate_reserve(max_hr, rhr_baseline)?;
    let mut zones = Vec::with_capacity(5);
    for i in 0..5 {
        let lo_pct = model.edges[i];
        let hi_pct = model.edges[i + 1];
        zones.push(HrZone {
            number: (i + 1) as u8,
            lower: rhr_baseline + lo_pct * hrr,
            upper: rhr_baseline + hi_pct * hrr,
            lower_pct: lo_pct,
            upper_pct: hi_pct,
        });
    }
    Some(HrZoneSet {
        zones,
        max_hr,
        source: model.version.to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hr_sample::HrSample;
    use crate::hr_zones::time_in_zone;

    #[test]
    fn none_for_nonpositive_reserve() {
        assert_eq!(zones_from_hrr(180.0, 180.0, WorkoutZoneModel::HRR_V1), None);
    }

    #[test]
    fn edges_convert_pct_hrr_into_bpm_against_the_reserve() {
        // RHR 52, MaxHR 194 -> HRR 142. Z1 lower = 52 + 0.40*142 = 108.8.
        let zs = zones_from_hrr(194.0, 52.0, WorkoutZoneModel::HRR_V1).unwrap();
        assert!((zs.zones[0].lower - 108.8).abs() < 1e-9);
        assert_eq!(zs.source, "hrr-v1");
    }

    #[test]
    fn reuses_hr_zones_time_in_zone_unchanged() {
        let zs = zones_from_hrr(194.0, 52.0, WorkoutZoneModel::HRR_V1).unwrap();
        let hr: Vec<HrSample> = (0..10).map(|t| HrSample { ts: t, bpm: 140 }).collect();
        let tiz = time_in_zone(&hr, &zs);
        // 140bpm at RHR 52/HRR 142 -> 62.0% HRR -> falls in Z3 (0.60-0.70) under Candidate A.
        assert!((tiz.seconds_in_zone(3) - 10.0).abs() < 1e-6, "{:?}", tiz);
    }
}
