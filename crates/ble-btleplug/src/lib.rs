//! btleplug 0.12 implementation of `ble_core::BleTransport`. The only crate that links btleplug, so
//! the rest of the stack stays backend-agnostic. Desktop (Windows/WinRT, macOS/CoreBluetooth,
//! Linux/BlueZ) — the mobile apps do BLE natively and feed bytes to the pure Rust codec instead.

mod transport;

pub use transport::{scan_whoops, BtleplugTransport, Scanned};
