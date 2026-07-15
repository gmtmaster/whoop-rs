//! The decoded frame + its packet-type vocabulary. A `Frame` owns the CRC-checked inner record
//! `[type][seq][cmd][payload]`; decoders read it INNER-relative, which collapses the GEN4/GEN5 +4
//! offset delta so shared types share one decoder.

use crate::family::Family;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PacketType {
    Command,
    CommandResponse,
    PuffinCommand,
    PuffinCommandResponse,
    RealtimeData,
    RealtimeRawData,
    HistoricalData,
    Event,
    Metadata,
    ConsoleLogs,
    RealtimeImuStream,
    HistoricalImuStream,
    RelativePuffinEvents,
    PuffinEventsFromStrap,
    PuffinMetadata,
    Unknown(u8),
}

impl PacketType {
    pub fn from_u8(v: u8) -> Self {
        use PacketType::*;
        match v {
            35 => Command,
            36 => CommandResponse,
            37 => PuffinCommand,
            38 => PuffinCommandResponse,
            40 => RealtimeData,
            43 => RealtimeRawData,
            47 => HistoricalData,
            48 => Event,
            49 => Metadata,
            50 => ConsoleLogs,
            51 => RealtimeImuStream,
            52 => HistoricalImuStream,
            53 => RelativePuffinEvents,
            54 => PuffinEventsFromStrap,
            56 => PuffinMetadata,
            other => Unknown(other),
        }
    }

    /// The wire name (used in the capture JSONL).
    pub fn name(self) -> &'static str {
        use PacketType::*;
        match self {
            Command => "COMMAND",
            CommandResponse => "COMMAND_RESPONSE",
            PuffinCommand => "PUFFIN_COMMAND",
            PuffinCommandResponse => "PUFFIN_COMMAND_RESPONSE",
            RealtimeData => "REALTIME_DATA",
            RealtimeRawData => "REALTIME_RAW_DATA",
            HistoricalData => "HISTORICAL_DATA",
            Event => "EVENT",
            Metadata => "METADATA",
            ConsoleLogs => "CONSOLE_LOGS",
            RealtimeImuStream => "REALTIME_IMU_DATA_STREAM",
            HistoricalImuStream => "HISTORICAL_IMU_DATA_STREAM",
            RelativePuffinEvents => "RELATIVE_PUFFIN_EVENTS",
            PuffinEventsFromStrap => "PUFFIN_EVENTS_FROM_STRAP",
            PuffinMetadata => "PUFFIN_METADATA",
            Unknown(_) => "UNKNOWN",
        }
    }

    pub fn to_u8(self) -> u8 {
        use PacketType::*;
        match self {
            Command => 35,
            CommandResponse => 36,
            PuffinCommand => 37,
            PuffinCommandResponse => 38,
            RealtimeData => 40,
            RealtimeRawData => 43,
            HistoricalData => 47,
            Event => 48,
            Metadata => 49,
            ConsoleLogs => 50,
            RealtimeImuStream => 51,
            HistoricalImuStream => 52,
            RelativePuffinEvents => 53,
            PuffinEventsFromStrap => 54,
            PuffinMetadata => 56,
            Unknown(v) => v,
        }
    }

    /// Fold GEN5 puffin aliases onto their base so routing is generation-agnostic.
    /// 51/52 (IMU streams) are deliberately NOT folded — they are their own raw types.
    pub fn canonical(self) -> Self {
        use PacketType::*;
        match self {
            PuffinCommand => Command,
            PuffinCommandResponse => CommandResponse,
            RelativePuffinEvents | PuffinEventsFromStrap => Event,
            PuffinMetadata => Metadata,
            other => other,
        }
    }
}

/// A single CRC-checked frame. Owns the full wire bytes (SOF..CRC32) as the single source of truth;
/// `inner()` = `[type][seq][cmd][payload]`, decoded INNER-relative. Keeping `raw` enables the lossless
/// capture tap and a byte-exact re-decode.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Frame {
    pub family: Family,
    pub crc_ok: bool,
    raw: Vec<u8>,
    inner_start: usize,
    inner_end: usize,
}

impl Frame {
    pub(crate) fn new(family: Family, raw: Vec<u8>, inner_start: usize, inner_end: usize, crc_ok: bool) -> Self {
        Frame { family, crc_ok, raw, inner_start, inner_end }
    }

    /// The complete wire frame (SOF..CRC32).
    pub fn raw(&self) -> &[u8] {
        &self.raw
    }

    /// The inner record `[type][seq][cmd][payload]`.
    pub fn inner(&self) -> &[u8] {
        &self.raw[self.inner_start..self.inner_end]
    }

    pub fn packet_type(&self) -> u8 {
        self.inner().first().copied().unwrap_or(0)
    }

    pub fn packet(&self) -> PacketType {
        PacketType::from_u8(self.packet_type())
    }

    pub fn seq(&self) -> u8 {
        self.inner().get(1).copied().unwrap_or(0)
    }

    /// Alias for `seq` — the type-47 HISTORICAL_DATA layout version lives in the seq byte.
    pub fn version(&self) -> u8 {
        self.seq()
    }

    pub fn cmd(&self) -> u8 {
        self.inner().get(2).copied().unwrap_or(0)
    }

    pub fn payload(&self) -> &[u8] {
        self.inner().get(3..).unwrap_or(&[])
    }
}
