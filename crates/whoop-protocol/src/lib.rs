//! Pure WHOOP BLE wire codec. No BLE, no async — every item here is unit-testable with plain bytes.
//!
//! Flow: `deframe::Deframer` turns notification fragments into complete `Frame`s → `packet::Frame`
//! carries the CRC-checked inner record → `live`/`records` decode it → `offload::Offload` drives the
//! historical drain. `framing::encode` builds outbound command frames. `family::Family` dispatches the
//! per-generation header/CRC shape in exactly one place.

pub mod advertising;
pub mod alarm;
pub mod bytes;
pub mod clock;
pub mod command;
pub mod config;
pub mod console;
pub mod crc;
pub mod deframe;
pub mod error;
pub mod event;
pub mod family;
pub mod framing;
pub mod haptic;
pub mod hello;
pub mod live;
pub mod offload;
pub mod pack;
pub mod packet;
pub mod records;
pub mod response;
pub mod trim;
pub mod variant;

pub use error::ProtocolError;
pub use family::{Channel, Family};
pub use variant::Variant;
pub use offload::{Offload, OffloadStep};
pub use packet::{Frame, PacketType};
pub use records::{HistoryRecord, ImuRecord, PpgRecord, Record, IMU_ACCEL_SCALE_G, IMU_GYRO_SCALE_DPS};
pub use response::CommandResponse;
