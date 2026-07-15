//! The WHOOP vendor-characteristic UUID table, per generation. This is where the pure protocol's
//! logical `Channel` meets the transport's raw `Uuid` — kept out of `whoop-protocol` so that crate stays
//! BLE-free. GEN4 base `…-8d6d-82b8-614a-1c8cb0f8dcc6`, GEN5 base `…-cce1-4033-93ce-002d5875f58a`.

use uuid::Uuid;
use whoop_protocol::{Channel, Family};

/// Standard SIG chars, both generations, readable/notify unbonded. Subscribed BEFORE the CLIENT_HELLO
/// bond (they work without it), matching the proven connect sequence.
pub const HEART_RATE_MEASUREMENT: Uuid = Uuid::from_u128(0x00002a37_0000_1000_8000_00805f9b34fb);
pub const BATTERY_LEVEL: Uuid = Uuid::from_u128(0x00002a19_0000_1000_8000_00805f9b34fb);

/// Identity chars (GAP + Device Information), readable without a bond.
pub const DEVICE_NAME: Uuid = Uuid::from_u128(0x00002a00_0000_1000_8000_00805f9b34fb);
pub const MODEL_NUMBER: Uuid = Uuid::from_u128(0x00002a24_0000_1000_8000_00805f9b34fb);
pub const SERIAL_NUMBER: Uuid = Uuid::from_u128(0x00002a25_0000_1000_8000_00805f9b34fb);
pub const FIRMWARE_REVISION: Uuid = Uuid::from_u128(0x00002a26_0000_1000_8000_00805f9b34fb);
pub const HARDWARE_REVISION: Uuid = Uuid::from_u128(0x00002a27_0000_1000_8000_00805f9b34fb);

const GEN5_SERVICE: Uuid = Uuid::from_u128(0xfd4b0001_cce1_4033_93ce_002d5875f58a);
const GEN5_CMD_WRITE: Uuid = Uuid::from_u128(0xfd4b0002_cce1_4033_93ce_002d5875f58a);
const GEN5_CMD_NOTIFY: Uuid = Uuid::from_u128(0xfd4b0003_cce1_4033_93ce_002d5875f58a);
const GEN5_EVENTS: Uuid = Uuid::from_u128(0xfd4b0004_cce1_4033_93ce_002d5875f58a);
const GEN5_DATA: Uuid = Uuid::from_u128(0xfd4b0005_cce1_4033_93ce_002d5875f58a);
const GEN5_CONSOLE: Uuid = Uuid::from_u128(0xfd4b0007_cce1_4033_93ce_002d5875f58a);

const GEN4_SERVICE: Uuid = Uuid::from_u128(0x61080001_8d6d_82b8_614a_1c8cb0f8dcc6);
const GEN4_CMD_WRITE: Uuid = Uuid::from_u128(0x61080002_8d6d_82b8_614a_1c8cb0f8dcc6);
const GEN4_CMD_NOTIFY: Uuid = Uuid::from_u128(0x61080003_8d6d_82b8_614a_1c8cb0f8dcc6);
const GEN4_EVENTS: Uuid = Uuid::from_u128(0x61080004_8d6d_82b8_614a_1c8cb0f8dcc6);
const GEN4_DATA: Uuid = Uuid::from_u128(0x61080005_8d6d_82b8_614a_1c8cb0f8dcc6);
const GEN4_CONSOLE: Uuid = Uuid::from_u128(0x61080007_8d6d_82b8_614a_1c8cb0f8dcc6);

pub fn service(family: Family) -> Uuid {
    match family {
        Family::Gen4 => GEN4_SERVICE,
        Family::Gen5 => GEN5_SERVICE,
    }
}

pub fn characteristic(family: Family, channel: Channel) -> Uuid {
    match (family, channel) {
        (Family::Gen5, Channel::CmdWrite) => GEN5_CMD_WRITE,
        (Family::Gen5, Channel::CmdNotify) => GEN5_CMD_NOTIFY,
        (Family::Gen5, Channel::Events) => GEN5_EVENTS,
        (Family::Gen5, Channel::Data) => GEN5_DATA,
        (Family::Gen5, Channel::Console) => GEN5_CONSOLE,
        (Family::Gen4, Channel::CmdWrite) => GEN4_CMD_WRITE,
        (Family::Gen4, Channel::CmdNotify) => GEN4_CMD_NOTIFY,
        (Family::Gen4, Channel::Events) => GEN4_EVENTS,
        (Family::Gen4, Channel::Data) => GEN4_DATA,
        (Family::Gen4, Channel::Console) => GEN4_CONSOLE,
    }
}

/// Reverse lookup — which logical channel a notification's characteristic belongs to.
pub fn channel_of(family: Family, characteristic_uuid: Uuid) -> Option<Channel> {
    [
        Channel::CmdWrite,
        Channel::CmdNotify,
        Channel::Events,
        Channel::Data,
        Channel::Console,
    ]
    .into_iter()
    .find(|&ch| characteristic(family, ch) == characteristic_uuid)
}
