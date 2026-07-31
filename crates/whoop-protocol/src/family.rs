//! Generation as a closed runtime enum. `HeaderSpec` carries the frame-HEADER shape only, consumed in
//! one place (`framing`); opcodes, the GEN5 event residual and the GEN5-only record versions each branch
//! at their own site. Adding a third generation makes every `match` a compiler-enforced porting
//! checklist, but an `if family ==` would compile and be silently wrong — `architecture.md` counts them.
//! Session/policy differences (bond, clock, epoch) belong to `whoop-client`; this stays pure framing.

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Family {
    /// WHOOP 4.0 — "Harvard". crc8 header, inner record at byte 4. Connect/bond path UNVERIFIED (N=1).
    Gen4,
    /// WHOOP 5.0 + MG — "Maverick" / "puffin". crc16-modbus header, inner record at byte 8. Verified.
    Gen5,
}

/// Which header checksum guards the frame header.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeaderCrc {
    /// crc8 over the 2 declared-length bytes (`frame[1..3]`), stored at `frame[3]`.
    Crc8LenBytes,
    /// crc16-modbus over the 6-byte header (`frame[0..6]`), stored LE at `frame[6..8]`.
    Crc16Header,
}

/// The per-generation envelope geometry. `declLen` = inner_len + 4 (payload + CRC32), excludes header.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HeaderSpec {
    pub sof: u8,
    /// Byte offset where the inner record `[type][seq][cmd][payload]` starts (4 = GEN4, 8 = GEN5).
    pub inner_start: usize,
    /// Byte offset of the declared-length u16 LE (1 = GEN4, 2 = GEN5).
    pub len_offset: usize,
    pub header_crc: HeaderCrc,
    /// GEN5 command inners are zero-padded up to a 4-byte multiple before CRC (the strap rejects otherwise).
    pub pad_inner_to_4: bool,
}

/// Logical vendor characteristic (mapped to a concrete BLE UUID by `whoop-client`, per family).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Channel {
    CmdWrite,
    CmdNotify,
    Events,
    Data,
    Console,
}

impl Family {
    pub const fn header(self) -> HeaderSpec {
        match self {
            Family::Gen4 => HeaderSpec {
                sof: 0xAA,
                inner_start: 4,
                len_offset: 1,
                header_crc: HeaderCrc::Crc8LenBytes,
                pad_inner_to_4: false,
            },
            Family::Gen5 => HeaderSpec {
                sof: 0xAA,
                inner_start: 8,
                len_offset: 2,
                header_crc: HeaderCrc::Crc16Header,
                pad_inner_to_4: true,
            },
        }
    }

    /// `0xAA 0x01 …` ⇒ GEN5 (the 0x01 format byte), a lone `0xAA` ⇒ GEN4 (a length byte follows).
    pub fn detect(bytes: &[u8]) -> Option<Family> {
        match bytes.first().copied()? {
            0xAA => Some(if bytes.get(1) == Some(&0x01) { Family::Gen5 } else { Family::Gen4 }),
            _ => None,
        }
    }
}
