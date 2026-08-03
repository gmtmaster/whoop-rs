//! Was the strap on the wrist for this record? Three v18 channels answer it and no single one is
//! sufficient: a real captured off-wrist record has the off-wrist BIT CLEAR and is caught only by its
//! dead optical baselines. Three-way, because "no reading" is not "off wrist".

use whoop_protocol::HistoryRecord;

/// `signal_flags` bit that marks the strap off-wrist.
const OFF_WRIST_BIT: u8 = 0x10;

/// Wear verdict for one record. `Unknown` = the record carries no wear channel at all (every 4.0
/// version, and any v18 too short for the optical block); a consumer admits it rather than dropping it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WornState {
    Worn,
    NotWorn,
    Unknown,
}

impl WornState {
    /// True only when the record is PROVEN off-wrist — `Unknown` is not off-wrist.
    pub fn is_off(self) -> bool {
        self == WornState::NotWorn
    }
}

/// Classify one record. A conjunction with a fixed precedence: any channel saying off-wrist wins,
/// `Worn` needs at least one channel saying worn and none dissenting, and no channel at all is
/// `Unknown`.
pub fn worn_state(h: &HistoryRecord) -> WornState {
    match (flag_worn(h), optical_worn(h)) {
        (Some(false), _) | (_, Some(false)) => WornState::NotWorn,
        (None, None) => WornState::Unknown,
        _ => WornState::Worn,
    }
}

/// The band's own off-wrist bit; `None` when the flags byte is not carried.
fn flag_worn(h: &HistoryRecord) -> Option<bool> {
    h.signal_flags.map(|f| f & OFF_WRIST_BIT == 0)
}

/// The optical front end: both baselines read 0 (decoded to `None`) off the wrist. `optical_signal_poor`
/// is the presence witness — it is `Some` exactly when the baseline bytes were readable, so a poor
/// signal never votes on wear alone, it only makes the baseline evidence usable.
fn optical_worn(h: &HistoryRecord) -> Option<bool> {
    h.optical_signal_poor.map(|_| h.optical_baseline_a.is_some() || h.optical_baseline_b.is_some())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The real captured off-wrist v18 record: flags byte 0 (bit 4 CLEAR), both baselines 0, both
    /// amplitudes at the 128 sentinel.
    fn real_off_wrist() -> HistoryRecord {
        HistoryRecord {
            version: 18,
            unix: 1_784_000_000,
            signal_flags: Some(0),
            signal_quality: Some(0),
            optical_baseline_a: None,
            optical_baseline_b: None,
            optical_signal_poor: Some(true),
            ..Default::default()
        }
    }

    /// The real captured worn v18 record.
    fn real_worn() -> HistoryRecord {
        HistoryRecord {
            version: 18,
            unix: 1_784_000_000,
            heart_rate: Some(60),
            signal_flags: Some(0),
            signal_quality: Some(255),
            optical_baseline_a: Some(101),
            optical_baseline_b: Some(111),
            optical_signal_poor: Some(false),
            ..Default::default()
        }
    }

    #[test]
    fn the_off_wrist_bit_alone_misses_a_real_off_wrist_record() {
        let h = real_off_wrist();
        assert_eq!(flag_worn(&h), Some(true), "the bit says worn on a record that is not");
        assert_eq!(worn_state(&h), WornState::NotWorn);
    }

    #[test]
    fn a_real_worn_record_is_worn() {
        assert_eq!(worn_state(&real_worn()), WornState::Worn);
        assert!(!worn_state(&real_worn()).is_off());
    }

    #[test]
    fn the_off_wrist_bit_overrides_live_baselines() {
        let mut h = real_worn();
        h.signal_flags = Some(OFF_WRIST_BIT);
        assert_eq!(worn_state(&h), WornState::NotWorn);
    }

    #[test]
    fn a_record_carrying_no_wear_channel_is_unknown_not_off_wrist() {
        let h = HistoryRecord { version: 24, unix: 1_784_000_000, heart_rate: Some(60), ..Default::default() };
        assert_eq!(worn_state(&h), WornState::Unknown);
        assert!(!worn_state(&h).is_off());
    }

    #[test]
    fn a_poor_signal_on_a_live_baseline_is_still_worn() {
        let mut h = real_worn();
        h.optical_signal_poor = Some(true);
        h.optical_baseline_a = None; // one dead channel is not both
        assert_eq!(worn_state(&h), WornState::Worn);
    }

    #[test]
    fn flags_alone_decide_when_the_optical_block_is_absent() {
        let mut h = real_worn();
        h.optical_signal_poor = None;
        h.optical_baseline_a = None;
        h.optical_baseline_b = None;
        assert_eq!(worn_state(&h), WornState::Worn);
        h.signal_flags = Some(OFF_WRIST_BIT);
        assert_eq!(worn_state(&h), WornState::NotWorn);
    }
}
