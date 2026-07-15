//! Generic BLE central transport. Knows nothing about WHOOP — it deals in raw characteristic UUIDs and
//! byte payloads, so the WHOOP stack (whoop-client) sits entirely above it and the concrete backend
//! (btleplug, or a mock) is swappable. `ensure_paired` is the OS-bond seam: a no-op where the OS bonds
//! transparently (macOS), a real step on Linux/Windows (see ble-btleplug).

mod mock;
mod transport;
mod types;

pub use mock::MockTransport;
pub use transport::BleTransport;
pub use types::{gatt_string, BleError, Notification};
