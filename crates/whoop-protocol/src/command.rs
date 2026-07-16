//! Command opcodes (inner byte 2 of an outbound COMMAND). The destructive / blind-explore set is listed
//! so the client can refuse them.

pub const TOGGLE_REALTIME_HR: u8 = 3;
pub const REPORT_VERSION_INFO: u8 = 7;
pub const SET_CLOCK: u8 = 10;
pub const GET_CLOCK_GEN4: u8 = 11;
pub const RUN_HAPTIC_PATTERN_MAVERICK: u8 = 19;
pub const ABORT_HISTORICAL_TRANSMITS: u8 = 20;
pub const SEND_HISTORICAL_DATA: u8 = 22;
pub const HISTORICAL_DATA_RESULT: u8 = 23;
pub const FORCE_TRIM: u8 = 25;
pub const GET_BATTERY_LEVEL: u8 = 26;
pub const REBOOT_STRAP: u8 = 29;
pub const POWER_CYCLE_STRAP: u8 = 32;
pub const GET_DATA_RANGE: u8 = 34;
pub const GET_HELLO_HARVARD: u8 = 35;
pub const ENTER_BLE_DFU: u8 = 45;
pub const GET_AFE_PARAMETERS: u8 = 62;
pub const SEND_R10_R11_REALTIME: u8 = 63;
pub const SET_ALARM_TIME: u8 = 66;
pub const GET_ALARM_TIME: u8 = 67;
pub const RUN_ALARM: u8 = 68;
pub const DISABLE_ALARM: u8 = 69;
pub const GET_ADVERTISING_NAME: u8 = 141;
pub const SET_ADVERTISING_NAME: u8 = 77;
pub const RUN_HAPTICS_PATTERN: u8 = 79;
pub const GET_ALL_HAPTICS_PATTERN: u8 = 80;
pub const START_RAW_DATA: u8 = 81;
pub const STOP_RAW_DATA: u8 = 82;
pub const SELECT_WRIST: u8 = 123;
pub const GET_EXTENDED_BATTERY_INFO: u8 = 98;
pub const RESET_FUEL_GAUGE: u8 = 99;
pub const SET_IMU_DATA_STREAM: u8 = 106;
pub const GET_BATTERY_PACK_INFO: u8 = 151;
pub const SET_DEVICE_CONFIG: u8 = 119;
pub const SET_CONFIG: u8 = 120;
pub const STOP_HAPTICS: u8 = 122;
pub const GET_HELLO: u8 = 145;
pub const SET_CLOCK_MAVERICK: u8 = 146;
pub const GET_CLOCK_GEN5: u8 = 147;

/// Firmware-load / trim / dfu / reboot / config-write / set-clock / adv-name / wrist-select. Never send
/// via a blind explore path — the legitimate ones have dedicated, gated methods.
pub const FORBIDDEN: &[u8] = &[
    SET_CLOCK,
    SET_CLOCK_MAVERICK,
    FORCE_TRIM,
    REBOOT_STRAP,
    POWER_CYCLE_STRAP,
    ENTER_BLE_DFU,
    SET_ADVERTISING_NAME, // 77 — persistently renames the strap's BLE advertising name
    SET_DEVICE_CONFIG,
    SET_CONFIG,
    RESET_FUEL_GAUGE, // 99 — clears the pack/strap fuel-gauge state
    SELECT_WRIST, // 123 — persistent device-config write
    142,
    143,
    144,
];

/// Genuinely destructive — never send at all, even via a gated action.
pub const DESTRUCTIVE: &[u8] = &[FORCE_TRIM, ENTER_BLE_DFU, 142, 143, 144];

pub fn is_forbidden(op: u8) -> bool {
    FORBIDDEN.contains(&op)
}

pub fn is_destructive(op: u8) -> bool {
    DESTRUCTIVE.contains(&op)
}
