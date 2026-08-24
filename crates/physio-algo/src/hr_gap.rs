//! Position-dependent policy for a run of missing HR. A gap that OPENS a series, one bracketed by
//! measured HR on both sides, and one that CLOSES it get different ceilings, because only the
//! interior gap has measured context on both sides. Classification and accounting only — nothing
//! here fills, holds or interpolates a value; `hr_zones` and `strain` ask it how much of a gap they
//! may bill and how much of what they billed was inferred.

/// Shortest run of missing HR that counts as a gap; below this it is ordinary sampling cadence and
/// the elapsed time is measured, not inferred.
pub const MIN_GAP_SECONDS: f64 = 60.0;

/// Ceiling for a gap opening the series: measured HR on its right only, so it is trusted for a
/// third as long as an interior one.
pub const MAX_LEAD_GAP_SECONDS: f64 = 600.0;

/// Ceiling for a gap with measured HR on both sides — the only position that can be reasoned about
/// from both directions, so the most permissive.
pub const MAX_INTERIOR_GAP_SECONDS: f64 = 1800.0;

/// Ceiling for a gap closing the series: measured HR on its left only and nothing after it, so the
/// tightest of the three.
pub const MAX_TRAIL_GAP_SECONDS: f64 = 300.0;

/// Where a gap sits in the series, which is what decides its ceiling.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GapPosition {
    /// Before the first measured sample: right-hand context only.
    Leading,
    /// Between two measured samples.
    Interior,
    /// After the last measured sample: left-hand context only.
    Trailing,
}

/// What the policy allows for one gap.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GapVerdict {
    /// Shorter than [`MIN_GAP_SECONDS`]: normal cadence, the elapsed time is measured.
    Cadence,
    /// A real gap within its position ceiling: the elapsed time may be billed, but it is inferred.
    Bridge,
    /// Past its position ceiling: billed to nothing, and it stays a hole.
    Refuse,
}

/// The longest gap this position may be billed for.
pub fn ceiling_seconds(position: GapPosition) -> f64 {
    match position {
        GapPosition::Leading => MAX_LEAD_GAP_SECONDS,
        GapPosition::Interior => MAX_INTERIOR_GAP_SECONDS,
        GapPosition::Trailing => MAX_TRAIL_GAP_SECONDS,
    }
}

/// Verdict for a gap of `gap_seconds` at `position`. A non-positive gap is `Cadence` (no elapsed
/// time to account for).
pub fn classify(gap_seconds: f64, position: GapPosition) -> GapVerdict {
    if gap_seconds < MIN_GAP_SECONDS {
        return GapVerdict::Cadence;
    }
    if gap_seconds <= ceiling_seconds(position) {
        GapVerdict::Bridge
    } else {
        GapVerdict::Refuse
    }
}

/// Seconds a gap may contribute to a duration-weighted metric: the whole gap when it is within its
/// position ceiling, zero past it. A refused gap gets no placeholder and no partial credit.
pub fn creditable_seconds(gap_seconds: f64, position: GapPosition) -> f64 {
    if gap_seconds <= 0.0 {
        return 0.0;
    }
    match classify(gap_seconds, position) {
        GapVerdict::Refuse => 0.0,
        _ => gap_seconds,
    }
}

/// Provenance for a duration-weighted total: which of its seconds were measured at cadence, which
/// were bridged across a real gap (inferred, not measured) and which were refused outright. Keeps a
/// bridged second distinguishable from a measured one after the two have been summed.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GapAccounting {
    pub measured_seconds: f64,
    pub bridged_seconds: f64,
    pub refused_seconds: f64,
}

impl GapAccounting {
    /// Seconds actually billed to a metric: measured plus bridged. Refused seconds are excluded.
    pub fn credited_seconds(&self) -> f64 {
        self.measured_seconds + self.bridged_seconds
    }

    /// Elapsed seconds seen, billed or not.
    pub fn elapsed_seconds(&self) -> f64 {
        self.credited_seconds() + self.refused_seconds
    }

    /// File one gap under its verdict. Non-positive gaps are ignored.
    pub fn add(&mut self, gap_seconds: f64, position: GapPosition) {
        if gap_seconds <= 0.0 {
            return;
        }
        match classify(gap_seconds, position) {
            GapVerdict::Cadence => self.measured_seconds += gap_seconds,
            GapVerdict::Bridge => self.bridged_seconds += gap_seconds,
            GapVerdict::Refuse => self.refused_seconds += gap_seconds,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ceilings_are_position_dependent() {
        assert!(ceiling_seconds(GapPosition::Trailing) < ceiling_seconds(GapPosition::Leading));
        assert!(ceiling_seconds(GapPosition::Leading) < ceiling_seconds(GapPosition::Interior));
    }

    #[test]
    fn the_same_gap_is_bridged_or_refused_by_position() {
        // 450 s: bracketed on both sides it is reconstructible, opening the series it still is, but
        // closing it there is nothing after it to reason from. One flat ceiling cannot say this.
        let g = 450.0;
        assert_eq!(classify(g, GapPosition::Interior), GapVerdict::Bridge);
        assert_eq!(classify(g, GapPosition::Leading), GapVerdict::Bridge);
        assert_eq!(classify(g, GapPosition::Trailing), GapVerdict::Refuse);
        assert_eq!(creditable_seconds(g, GapPosition::Interior), 450.0);
        assert_eq!(creditable_seconds(g, GapPosition::Trailing), 0.0);

        // 900 s: only an interior gap survives.
        assert_eq!(classify(900.0, GapPosition::Interior), GapVerdict::Bridge);
        assert_eq!(classify(900.0, GapPosition::Leading), GapVerdict::Refuse);
        assert_eq!(classify(900.0, GapPosition::Trailing), GapVerdict::Refuse);
    }

    #[test]
    fn position_ceilings_are_inclusive_at_the_edge() {
        for p in [
            GapPosition::Leading,
            GapPosition::Interior,
            GapPosition::Trailing,
        ] {
            let c = ceiling_seconds(p);
            assert_eq!(classify(c, p), GapVerdict::Bridge, "{p:?} at ceiling");
            assert_eq!(
                classify(c + 1.0, p),
                GapVerdict::Refuse,
                "{p:?} past ceiling"
            );
            assert_eq!(creditable_seconds(c, p), c);
            assert_eq!(creditable_seconds(c + 1.0, p), 0.0);
        }
    }

    #[test]
    fn a_short_run_is_cadence_not_a_gap() {
        assert_eq!(classify(59.0, GapPosition::Trailing), GapVerdict::Cadence);
        assert_eq!(creditable_seconds(59.0, GapPosition::Trailing), 59.0);
        assert_eq!(classify(60.0, GapPosition::Trailing), GapVerdict::Bridge);
    }

    #[test]
    fn a_non_positive_gap_credits_and_accounts_nothing() {
        assert_eq!(creditable_seconds(0.0, GapPosition::Interior), 0.0);
        assert_eq!(creditable_seconds(-5.0, GapPosition::Interior), 0.0);
        let mut a = GapAccounting::default();
        a.add(0.0, GapPosition::Interior);
        a.add(-5.0, GapPosition::Interior);
        assert_eq!(a, GapAccounting::default());
    }

    #[test]
    fn accounting_keeps_measured_bridged_and_refused_apart() {
        let mut a = GapAccounting::default();
        a.add(1.0, GapPosition::Interior); // cadence
        a.add(1.0, GapPosition::Interior); // cadence
        a.add(900.0, GapPosition::Interior); // bridged
        a.add(3600.0, GapPosition::Interior); // refused
        assert_eq!(a.measured_seconds, 2.0);
        assert_eq!(a.bridged_seconds, 900.0);
        assert_eq!(a.refused_seconds, 3600.0);
        assert_eq!(a.credited_seconds(), 902.0);
        assert_eq!(a.elapsed_seconds(), 4502.0);
    }
}
