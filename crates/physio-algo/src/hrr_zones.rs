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

/// HISTORICAL, FROZEN: "Candidate A" from `hr-intensity-model-spec.md` Part 2.2 - the six-way split
/// (moderate 40-59% HRR evenly split into Z1/Z2, vigorous 60-84% evenly split into Z3/Z4,
/// near-maximal-to-maximal combined into Z5). SUPERSEDED as the canonical workout model by
/// `CANONICAL_WORKOUT_EDGES`/`HRR_V2` below (product decision: align with the publicly documented
/// WHOOP Karvonen/HRR zone convention - 40/60/70/80/90 - not this evenly-split variant). Kept, and
/// its `hrr-v1` version tag kept, ONLY so historical `zone_model_version = "hrr-v1"` rows (persisted
/// before this change) can still be replayed/interpreted correctly - never mutate this constant in
/// place, and never point new computations at `HRR_V1`.
pub const CANDIDATE_A_EDGES: [f64; 6] = [0.40, 0.50, 0.60, 0.70, 0.85, 1.00];

/// CANONICAL, CURRENT: NOOP's workout Z1-5 %HRR edges - a deliberate product decision (not
/// literature-derived like `CANDIDATE_A_EDGES` was), chosen to align with the publicly documented
/// WHOOP Karvonen/HRR workout-zone convention: Z1 40-60%, Z2 60-70%, Z3 70-80%, Z4 80-90%, Z5
/// 90-100% HRR. This is standard Karvonen/HRR physiology, not a reproduction of any WHOOP
/// proprietary implementation. See `WorkoutZoneModel::HRR_V2`.
pub const CANONICAL_WORKOUT_EDGES: [f64; 6] = [0.40, 0.60, 0.70, 0.80, 0.90, 1.00];

/// A named, versioned set of %HRR edges for the six workout zones. New edge sets get a new
/// `version` string and are additive - never mutate a shipped edge set in place, since
/// `zone_model_version` is persisted per-day (and, going forward, per-workout - see
/// `docs/architecture.md`'s workout-physiology-snapshot note) for reproducibility.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WorkoutZoneModel {
    pub version: &'static str,
    pub edges: [f64; 6],
}

impl WorkoutZoneModel {
    /// HISTORICAL, FROZEN: the original ACSM-evenly-split model. Do not use for new computations -
    /// kept only so `zone_model_version = "hrr-v1"` history remains replayable. See
    /// `CANDIDATE_A_EDGES`'s doc.
    pub const HRR_V1: WorkoutZoneModel = WorkoutZoneModel {
        version: "hrr-v1",
        edges: CANDIDATE_A_EDGES,
    };

    /// CANONICAL, CURRENT default for all new workout-zone computations (live, persisted, and
    /// daily/Healthspan aggregation alike - see `CANONICAL_WORKOUT_EDGES`'s doc for why these
    /// specific edges). A semantic change from `HRR_V1` (different boundaries for the same bpm), so
    /// it gets its own version string rather than silently reusing `"hrr-v1"` for different numbers.
    pub const HRR_V2: WorkoutZoneModel = WorkoutZoneModel {
        version: "hrr-v2",
        edges: CANONICAL_WORKOUT_EDGES,
    };

    /// The model new computations should use. A single named spot so "which version is current"
    /// is never duplicated/hardcoded at each call site.
    pub const CURRENT: WorkoutZoneModel = Self::HRR_V2;
}

/// Build a personalized `HrZoneSet` (bpm-space, ready for `hr_zones::time_in_zone`) from %HRR edges,
/// a Max HR estimate and a chronic RHR baseline. `None` when the reserve is non-physiological
/// (`max_hr <= rhr_baseline`) - callers should fall back to the age/%MaxHR path
/// (`hr_zones::zones_for_age`) in that case, not fabricate zones from a negative reserve.
///
/// `source` on the returned `HrZoneSet` is the model's version string (e.g. `"hrr-v2"`), NOT
/// `"tanaka"`/`"manual"` - this is what lets a daily/workout row record which zone model produced
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
        assert_eq!(zones_from_hrr(180.0, 180.0, WorkoutZoneModel::HRR_V2), None);
    }

    #[test]
    fn edges_convert_pct_hrr_into_bpm_against_the_reserve() {
        // RHR 52, MaxHR 194 -> HRR 142. Z1 lower = 52 + 0.40*142 = 108.8 (unchanged: 40% floor is
        // the same in both v1 and v2).
        let zs = zones_from_hrr(194.0, 52.0, WorkoutZoneModel::HRR_V2).unwrap();
        assert!((zs.zones[0].lower - 108.8).abs() < 1e-9);
        assert_eq!(zs.source, "hrr-v2");
    }

    #[test]
    fn reuses_hr_zones_time_in_zone_unchanged() {
        let zs = zones_from_hrr(194.0, 52.0, WorkoutZoneModel::HRR_V2).unwrap();
        let hr: Vec<HrSample> = (0..10).map(|t| HrSample { ts: t, bpm: 140 }).collect();
        let tiz = time_in_zone(&hr, &zs);
        // 140bpm at RHR 52/HRR 142 -> 62.0% HRR -> falls in Z2 (0.60-0.70) under the canonical v2
        // edges (was Z3 under v1's 50/60/70 split - the exact semantic shift this version bump
        // exists to make visible/traceable).
        assert!((tiz.seconds_in_zone(2) - 10.0).abs() < 1e-6, "{:?}", tiz);
    }

    #[test]
    fn v1_and_v2_are_both_still_directly_usable_and_disagree_where_expected() {
        // Historical replay contract: HRR_V1 must still produce the OLD boundaries unchanged, so a
        // persisted `zone_model_version = "hrr-v1"` row stays reproducible even after this change.
        let v1 = zones_from_hrr(194.0, 52.0, WorkoutZoneModel::HRR_V1).unwrap();
        let v2 = zones_from_hrr(194.0, 52.0, WorkoutZoneModel::HRR_V2).unwrap();
        assert_eq!(v1.source, "hrr-v1");
        assert_eq!(v2.source, "hrr-v2");
        // Z2 lower edge: v1 = 52 + 0.50*142 = 123.0, v2 = 52 + 0.60*142 = 137.2 - genuinely different.
        assert!((v1.zones[1].lower - 123.0).abs() < 1e-9);
        assert!((v2.zones[1].lower - 137.2).abs() < 1e-9);
    }

    #[test]
    fn current_points_at_v2() {
        assert_eq!(WorkoutZoneModel::CURRENT.version, "hrr-v2");
        assert_eq!(WorkoutZoneModel::CURRENT.edges, CANONICAL_WORKOUT_EDGES);
    }

    #[test]
    fn canonical_edges_match_the_documented_whoop_hrr_convention() {
        // 40/60/70/80/90 - see module doc: standard Karvonen/HRR physiology, a deliberate NOOP
        // product decision aligned with (not copied from) WHOOP's publicly documented convention.
        assert_eq!(CANONICAL_WORKOUT_EDGES, [0.40, 0.60, 0.70, 0.80, 0.90, 1.00]);
    }
}
