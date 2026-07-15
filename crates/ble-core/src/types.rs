use uuid::Uuid;

/// Standard SIG identity characteristics (GAP device name + Device Information), readable without a bond.
pub const DEVICE_NAME: Uuid = Uuid::from_u128(0x00002a00_0000_1000_8000_00805f9b34fb);
pub const MODEL_NUMBER: Uuid = Uuid::from_u128(0x00002a24_0000_1000_8000_00805f9b34fb);
pub const SERIAL_NUMBER: Uuid = Uuid::from_u128(0x00002a25_0000_1000_8000_00805f9b34fb);
pub const FIRMWARE_REVISION: Uuid = Uuid::from_u128(0x00002a26_0000_1000_8000_00805f9b34fb);
pub const HARDWARE_REVISION: Uuid = Uuid::from_u128(0x00002a27_0000_1000_8000_00805f9b34fb);

/// A value notification from a subscribed characteristic.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Notification {
    pub uuid: Uuid,
    pub value: Vec<u8>,
}

/// Decode a GATT string value: lossy UTF-8, then strip the NUL padding + surrounding whitespace that
/// devices pad these fixed-width chars with.
pub fn gatt_string(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).trim_matches(|c: char| c == '\0' || c.is_whitespace()).to_string()
}

#[derive(thiserror::Error, Debug)]
pub enum BleError {
    #[error("no BLE adapter found")]
    NoAdapter,
    #[error("device not found")]
    NotFound,
    #[error("not connected")]
    NotConnected,
    #[error("pairing/bond failed: {0}")]
    Pairing(String),
    #[error("characteristic {0} not found")]
    NoCharacteristic(Uuid),
    #[error("backend error: {0}")]
    Backend(String),
}
