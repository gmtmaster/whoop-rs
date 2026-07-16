//! The WHOOP layer over a generic BLE transport. `WhoopClient<T>` maps logical channels to the
//! per-family vendor UUIDs, runs the bond handshake, and drives the pure `Offload` state machine off
//! the notification stream. Being generic over `T: BleTransport`, it is exercised end-to-end with the
//! mock (no radio) and monomorphizes onto the real btleplug transport in `whoopctl`.

mod capture;
mod client;
mod error;
mod policy;
mod uuids;

pub use capture::{archive_line, capture_line, decode_capture};
pub use client::WhoopClient;
pub use error::Error;
pub use policy::{reconnect_delay_s, should_run, BackfillTrigger};
pub use uuids::{all_services, characteristic, channel_of, service};
