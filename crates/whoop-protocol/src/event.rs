//! Small inner-byte-2 vocabularies: METADATA sub-type, EVENT number, COMMAND_RESPONSE result.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MetadataType {
    HistoryStart,
    HistoryEnd,
    HistoryComplete,
    Unknown(u8),
}

impl MetadataType {
    pub fn from_u8(v: u8) -> Self {
        match v {
            1 => MetadataType::HistoryStart,
            2 => MetadataType::HistoryEnd,
            3 => MetadataType::HistoryComplete,
            other => MetadataType::Unknown(other),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResultCode {
    Failure,
    Success,
    Pending,
    Unsupported,
    Unknown(u8),
}

impl ResultCode {
    pub fn from_u8(v: u8) -> Self {
        match v {
            0 => ResultCode::Failure,
            1 => ResultCode::Success,
            2 => ResultCode::Pending,
            3 => ResultCode::Unsupported,
            other => ResultCode::Unknown(other),
        }
    }
}

// EVENT numbers (EVENT inner byte 2).
pub const BATTERY_LEVEL: u8 = 3;
pub const CHARGING_ON: u8 = 7;
pub const CHARGING_OFF: u8 = 8;
pub const WRIST_ON: u8 = 9;
pub const WRIST_OFF: u8 = 10;
pub const DOUBLE_TAP: u8 = 14;
pub const TEMPERATURE_LEVEL: u8 = 17;
/// The strap's own attach/detach signals for a battery pack — the presence facts, ahead of any poll.
pub const BATTERY_PACK_CONNECTED: u8 = 21;
pub const BATTERY_PACK_REMOVED: u8 = 22;
pub const BLE_BONDED: u8 = 23;
pub const BLE_REALTIME_HR_ON: u8 = 33;
pub const BLE_REALTIME_HR_OFF: u8 = 34;
pub const STRAP_DRIVEN_ALARM_EXECUTED: u8 = 57;
pub const APP_DRIVEN_ALARM_EXECUTED: u8 = 58;
pub const HAPTICS_FIRED: u8 = 60;
/// The pack block, unprompted — the same fields the pack command answers with.
pub const BATTERY_PACK_INFO: u8 = 109;
