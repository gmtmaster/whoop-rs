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
}
