//! HR-max and five %HRmax heart-rate zones with time-in-zone from an HR stream. Max HR comes from the
//! Tanaka age formula (208 − 0.7·age) or a manual override; zones are the conventional 50/60/70/80/90/100%
//! bands and the top zone is inclusive at HRmax. Time-in-zone credits each sample until the next reading,
//! gated by the position-dependent ceilings in [`crate::hr_gap`] so a wear gap is not counted as time in
//! a zone; the tail sample gets the median inter-sample gap. Pure.

/// %HRmax band edges for zones 1..5.
pub const ZONE_EDGES: [f64; 6] = [0.50, 0.60, 0.70, 0.80, 0.90, 1.00];

pub use crate::hr_gap::{GapAccounting, GapPosition, creditable_seconds};
pub use crate::hr_sample::HrSample;

/// A single zone as a bpm interval `[lower, upper)` plus its %HRmax band.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HrZone {
    pub number: u8,
    pub lower: f64,
    pub upper: f64,
    pub lower_pct: f64,
    pub upper_pct: f64,
}

/// Five zones (ascending), the max HR they were built from, and its source ("tanaka"/"manual").
#[derive(Clone, Debug, PartialEq)]
pub struct HrZoneSet {
    pub zones: Vec<HrZone>,
    pub max_hr: f64,
    pub source: String,
}

impl HrZoneSet {
    /// Zone number (1..5) for a bpm value, or 0 when below Zone 1. Top zone is inclusive at its edge.
    pub fn zone_number(&self, bpm: f64) -> u8 {
        for z in &self.zones {
            if z.number == 5 {
                if bpm >= z.lower {
                    return 5;
                }
            } else if bpm >= z.lower && bpm < z.upper {
                return z.number;
            }
        }
        0
    }
}

/// Seconds in each of the five zones (index 0 == Zone 1) plus time below Zone 1, and the provenance
/// of those seconds: how many were bridged across a real gap rather than measured, and how much
/// elapsed time was refused outright. A bridged second is held-forward, not observed.
#[derive(Clone, Debug, PartialEq)]
pub struct TimeInZone {
    pub seconds: [f64; 5],
    pub below_zone1: f64,
    pub bridged_seconds: f64,
    pub refused_seconds: f64,
}

impl TimeInZone {
    /// Total counted seconds (Zone 1..5 plus below-Zone-1).
    pub fn total(&self) -> f64 {
        self.seconds.iter().sum::<f64>() + self.below_zone1
    }

    /// Seconds in a specific zone (1..5); 0 for out-of-range zone numbers.
    pub fn seconds_in_zone(&self, zone: u8) -> f64 {
        if !(1..=5).contains(&zone) {
            return 0.0;
        }
        self.seconds[(zone - 1) as usize]
    }
}

/// Tanaka age-predicted max HR: 208 − 0.7 × age (gender-independent).
pub fn tanaka_max_hr(age: f64) -> f64 {
    crate::strain::tanaka_hrmax(age)
}

/// Build the 5-zone set from age (Tanaka) or a manual override.
pub fn zones_for_age(age: f64, max_hr_override: Option<f64>) -> HrZoneSet {
    match max_hr_override {
        Some(m) => zones_from_max(m, "manual"),
        None => zones_from_max(tanaka_max_hr(age), "tanaka"),
    }
}

/// Build the 5-zone set directly from a known max HR.
pub fn zones_from_max(max_hr: f64, source: &str) -> HrZoneSet {
    let mut built = Vec::with_capacity(5);
    for i in 0..5 {
        let lo_pct = ZONE_EDGES[i];
        let hi_pct = ZONE_EDGES[i + 1];
        built.push(HrZone {
            number: (i + 1) as u8,
            lower: lo_pct * max_hr,
            upper: hi_pct * max_hr,
            lower_pct: lo_pct,
            upper_pct: hi_pct,
        });
    }
    HrZoneSet {
        zones: built,
        max_hr,
        source: source.to_string(),
    }
}

/// Time-in-zone (seconds) from a time-ordered (or unordered) HR stream. Each sample is credited with the
/// duration until the next sample when that gap is within its position ceiling and with nothing when it
/// is past it; the tail sample gets the median gap, judged as a trailing gap.
pub fn time_in_zone(hr: &[HrSample], zone_set: &HrZoneSet) -> TimeInZone {
    let mut sorted: Vec<HrSample> = hr.to_vec();
    sorted.sort_by_key(|s| s.ts);
    let mut zone_seconds = [0.0f64; 5];
    let mut below = 0.0;
    let mut acct = GapAccounting::default();
    if sorted.is_empty() {
        return TimeInZone {
            seconds: zone_seconds,
            below_zone1: 0.0,
            bridged_seconds: 0.0,
            refused_seconds: 0.0,
        };
    }

    let tail_duration = median_interval(&sorted);
    for i in 0..sorted.len() {
        // Non-positive spacing means same-second duplicates, not a gap: they get the median cadence.
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
        acct.add(gap, position);
        let dur = creditable_seconds(gap, position);
        let z = zone_set.zone_number(sorted[i].bpm as f64);
        if z >= 1 {
            zone_seconds[(z - 1) as usize] += dur;
        } else {
            below += dur;
        }
    }
    TimeInZone {
        seconds: zone_seconds,
        below_zone1: below,
        bridged_seconds: acct.bridged_seconds,
        refused_seconds: acct.refused_seconds,
    }
}

/// Median spacing between consecutive timestamps, restricted to plausible (0, 300 s) gaps. Falls back to
/// 1.0 s when no plausible gap exists.
pub fn median_interval(sorted: &[HrSample]) -> f64 {
    crate::stats::median_gap_s(&sorted.iter().map(|s| s.ts).collect::<Vec<_>>(), 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tanaka_matches_formula() {
        assert!((tanaka_max_hr(30.0) - 187.0).abs() < 1e-9);
        let zs = zones_for_age(30.0, None);
        assert_eq!(zs.source, "tanaka");
        assert!((zs.max_hr - 187.0).abs() < 1e-9);
    }

    #[test]
    fn manual_override_sets_source_and_edges() {
        let zs = zones_for_age(30.0, Some(200.0));
        assert_eq!(zs.source, "manual");
        assert_eq!(zs.zones.len(), 5);
        assert!((zs.zones[0].lower - 100.0).abs() < 1e-9); // 50% of 200
        assert!((zs.zones[0].upper - 120.0).abs() < 1e-9); // 60% of 200
        assert!((zs.zones[4].upper - 200.0).abs() < 1e-9); // 100% of 200
    }

    #[test]
    fn zone_number_boundaries() {
        let zs = zones_from_max(200.0, "manual");
        assert_eq!(zs.zone_number(90.0), 0); // below 50%
        assert_eq!(zs.zone_number(100.0), 1); // Zone 1 lower edge, inclusive
        assert_eq!(zs.zone_number(119.9), 1);
        assert_eq!(zs.zone_number(120.0), 2); // Zone 2 lower edge
        assert_eq!(zs.zone_number(200.0), 5); // HRmax lands in Zone 5 (inclusive top)
        assert_eq!(zs.zone_number(300.0), 5);
    }

    #[test]
    fn constant_stream_fully_accounted() {
        // Ten 1 Hz zone-3 samples: median gap 1 s, tail gets 1 s → 10 s total, all in one zone.
        let zs = zones_from_max(200.0, "manual");
        let hr: Vec<HrSample> = (0..10).map(|t| HrSample { ts: t, bpm: 150 }).collect();
        let tiz = time_in_zone(&hr, &zs);
        assert!((tiz.total() - 10.0).abs() < 1e-9);
        assert!((tiz.seconds_in_zone(3) - 10.0).abs() < 1e-9); // 150/200 = 75% → Zone 3
        assert_eq!(tiz.bridged_seconds, 0.0); // every second measured at cadence
        assert_eq!(tiz.refused_seconds, 0.0);
    }

    #[test]
    fn refuses_an_interior_gap_past_its_ceiling() {
        // Three 1 Hz zone-1 samples then one an hour later. The 3600 s gap is past the interior
        // ceiling, so it is billed to no zone at all and shows up as refused time.
        let zs = zones_from_max(200.0, "manual");
        let hr = [
            HrSample { ts: 0, bpm: 110 },
            HrSample { ts: 1, bpm: 110 },
            HrSample { ts: 2, bpm: 110 },
            HrSample { ts: 3602, bpm: 110 },
        ];
        let tiz = time_in_zone(&hr, &zs);
        assert!((tiz.total() - 3.0).abs() < 1e-9, "got {}", tiz.total()); // 1 + 1 + 0 + tail(1)
        assert!(
            (tiz.refused_seconds - 3600.0).abs() < 1e-9,
            "got {}",
            tiz.refused_seconds
        );
        assert_eq!(tiz.bridged_seconds, 0.0);
        assert!((tiz.total() - tiz.seconds_in_zone(1)).abs() < 1e-9); // all zone 1
    }

    #[test]
    fn bridges_an_interior_gap_within_its_ceiling_and_marks_it() {
        // A 900 s gap is bracketed by measured HR and inside the 1800 s interior ceiling, so it is
        // billed in full — but every one of those seconds is flagged as bridged, not measured.
        let zs = zones_from_max(200.0, "manual");
        let hr = [
            HrSample { ts: 0, bpm: 110 },
            HrSample { ts: 1, bpm: 110 },
            HrSample { ts: 2, bpm: 110 },
            HrSample { ts: 902, bpm: 110 },
        ];
        let tiz = time_in_zone(&hr, &zs);
        assert!((tiz.total() - 903.0).abs() < 1e-9, "got {}", tiz.total()); // 1 + 1 + 900 + tail(1)
        assert!(
            (tiz.bridged_seconds - 900.0).abs() < 1e-9,
            "got {}",
            tiz.bridged_seconds
        );
        assert_eq!(tiz.refused_seconds, 0.0);
        assert!(
            tiz.bridged_seconds < tiz.total(),
            "bridged time is a subset of counted time"
        );
    }

    #[test]
    fn interior_ceiling_is_inclusive_at_its_edge() {
        let zs = zones_from_max(200.0, "manual");
        let at = [
            HrSample { ts: 0, bpm: 110 },
            HrSample { ts: 1800, bpm: 110 },
        ];
        let past = [
            HrSample { ts: 0, bpm: 110 },
            HrSample { ts: 1810, bpm: 110 },
        ];
        // Two samples 1800 s apart: no plausible (0, 300 s) gap exists, so the tail falls back to 1 s.
        assert!((time_in_zone(&at, &zs).total() - 1801.0).abs() < 1e-9);
        assert!((time_in_zone(&past, &zs).total() - 1.0).abs() < 1e-9);
        assert!((time_in_zone(&past, &zs).refused_seconds - 1810.0).abs() < 1e-9);
    }

    #[test]
    fn unsorted_input_is_sorted_defensively() {
        let zs = zones_from_max(200.0, "manual");
        let hr = [
            HrSample { ts: 2, bpm: 110 },
            HrSample { ts: 0, bpm: 110 },
            HrSample { ts: 1, bpm: 110 },
        ];
        let tiz = time_in_zone(&hr, &zs);
        assert!((tiz.total() - 3.0).abs() < 1e-9);
    }

    #[test]
    fn empty_stream_is_zero() {
        let zs = zones_from_max(200.0, "manual");
        let tiz = time_in_zone(&[], &zs);
        assert!((tiz.total()).abs() < 1e-9);
    }
}
