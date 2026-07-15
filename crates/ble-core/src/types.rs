use uuid::Uuid;

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
