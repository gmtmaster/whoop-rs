//! WHOOP's feature calibration schedule: how many recoveries (nights) a metric needs to unlock and to be
//! fully calibrated. Our readouts gate on the same periods the app uses, so a value appears on the same
//! schedule. `unlock == full` = shown and trusted at once; `full > unlock` = shown early, refines to `full`.

/// Recoveries (nights) a metric needs to unlock and to be fully calibrated.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Calibration {
    pub unlock: u32,
    pub full: u32,
}

impl Calibration {
    const fn at(unlock: u32, full: u32) -> Self {
        Calibration { unlock, full }
    }
    /// Enough recoveries to show the metric.
    pub fn unlocked(&self, nights: usize) -> bool {
        nights as u32 >= self.unlock
    }
    /// Enough recoveries for the metric to be fully calibrated.
    pub fn calibrated(&self, nights: usize) -> bool {
        nights as u32 >= self.full
    }
}

pub const BLOOD_OXYGEN: Calibration = Calibration::at(1, 1);
pub const HRV: Calibration = Calibration::at(1, 1);
pub const RHR: Calibration = Calibration::at(1, 1);
pub const RESPIRATORY_RATE: Calibration = Calibration::at(1, 1);
pub const RECOVERY_SCORE: Calibration = Calibration::at(3, 3);
pub const SLEEP_CONSISTENCY: Calibration = Calibration::at(5, 5);
pub const SKIN_TEMP: Calibration = Calibration::at(7, 7);
pub const CALORIES: Calibration = Calibration::at(1, 14);
pub const VO2_MAX: Calibration = Calibration::at(14, 14);
pub const HEALTH_MONITOR: Calibration = Calibration::at(7, 7);
pub const DAY_STRAIN: Calibration = Calibration::at(0, 0);
pub const SLEEP_EFFICIENCY: Calibration = Calibration::at(1, 1);
pub const SLEEP_PERFORMANCE: Calibration = Calibration::at(1, 1);
pub const HOURS_VS_NEEDED: Calibration = Calibration::at(1, 1);
pub const HEALTH_REPORT: Calibration = Calibration::at(14, 14);

/// Steps is the one metric whose schedule depends on the strap. 5.0/MG counts steps on-chip and shows
/// them straight away; 4.0 has no step counter, so the figure is estimated from motion and needs a
/// recovery first. Use [`steps`] rather than picking a constant by hand.
pub const STEPS_GEN5: Calibration = Calibration::at(0, 0);
pub const STEPS_GEN4: Calibration = Calibration::at(1, 1);

/// The steps schedule for a strap generation. `true` = a WHOOP 4.0.
pub const fn steps(is_gen4: bool) -> Calibration {
    if is_gen4 { STEPS_GEN4 } else { STEPS_GEN5 }
}

// Two of the published rules are WINDOWED and this night-count model cannot express them; both are
// counted here as plain totals, so a sparse history unlocks earlier than it would on WHOOP:
//   • VO2 max needs 14 sleeps within a 21-day period.
//   • Sleep consistency needs 5 CONSECUTIVE nights.
// HEALTH_MONITOR is also ambiguous upstream (listed as unlocking at 7, described as accessible from 0
// and fully calibrated at 7); it is kept at 7/7 until that is settled.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unlock_and_full_gate_on_night_count() {
        assert!(!BLOOD_OXYGEN.unlocked(0));
        assert!(BLOOD_OXYGEN.unlocked(1));
        assert!(!SKIN_TEMP.unlocked(6) && SKIN_TEMP.unlocked(7));
        assert!(CALORIES.unlocked(1) && !CALORIES.calibrated(1) && CALORIES.calibrated(14));
    }

    #[test]
    fn a_zero_night_metric_is_available_immediately() {
        assert!(DAY_STRAIN.unlocked(0) && DAY_STRAIN.calibrated(0));
    }

    #[test]
    fn steps_waits_a_recovery_on_gen4_and_not_on_gen5() {
        // The only generation-dependent schedule: 5.0/MG counts steps on-chip, 4.0 estimates them.
        assert!(
            steps(false).unlocked(0),
            "5.0/MG shows steps from the first day"
        );
        assert!(
            !steps(true).unlocked(0),
            "4.0 must not show an estimate before a recovery"
        );
        assert!(steps(true).unlocked(1));
    }
}
