//! The WHOOP layer over a generic BLE transport. `WhoopClient<T>` maps logical channels to the
//! per-family vendor UUIDs, runs the bond handshake, and drives the pure `Offload` state machine off
//! the notification stream. Being generic over `T: BleTransport`, it is exercised end-to-end with the
//! mock (no radio) and monomorphizes onto the real btleplug transport in `whoopctl`.

mod capture;
mod client;
mod error;
mod flash;
mod policy;
mod uuids;

pub use capture::{capture_line, decode_capture};
pub use client::WhoopClient;
pub use error::Error;
pub use flash::{
    FlashArm, FlashFault, FlashOptions, FlashProgress, FlashReport, FlashStep, Quiesce, BATTERY_FLOOR_PCT,
};
pub use policy::{should_run, BackfillTrigger};
pub use uuids::{all_services, characteristic, channel_of, service};
