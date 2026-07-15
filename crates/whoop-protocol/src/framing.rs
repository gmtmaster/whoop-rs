//! The one place the per-generation header/CRC shape is matched. `encode` builds an outbound COMMAND
//! frame; `decode` parses one complete frame and computes `crc_ok`. Both dispatch on `Family::header`.

use crate::bytes::{u16_at, u32_at};
use crate::crc::{crc16_modbus, crc32_zlib, crc8};
use crate::error::ProtocolError;
use crate::family::{Family, HeaderCrc};
use crate::packet::Frame;

/// Build a complete outbound frame for `[type][seq][cmd] + payload`. GEN5 inners are padded to a
/// 4-byte boundary before length/CRC (the strap rejects unpadded 5/MG command inners).
pub fn encode(family: Family, packet_type: u8, seq: u8, cmd: u8, payload: &[u8]) -> Vec<u8> {
    let h = family.header();

    let mut inner = Vec::with_capacity(3 + payload.len() + 3);
    inner.push(packet_type);
    inner.push(seq);
    inner.push(cmd);
    inner.extend_from_slice(payload);
    if h.pad_inner_to_4 {
        let pad = (4 - inner.len() % 4) % 4;
        inner.resize(inner.len() + pad, 0);
    }

    let decl_len = (inner.len() + 4) as u16; // inner + the 4-byte CRC32 trailer
    let mut frame = Vec::with_capacity(h.inner_start + inner.len() + 4);
    match h.header_crc {
        HeaderCrc::Crc8LenBytes => {
            frame.push(h.sof);
            frame.extend_from_slice(&decl_len.to_le_bytes());
            frame.push(crc8(&frame[1..3]));
        }
        HeaderCrc::Crc16Header => {
            frame.push(h.sof);
            frame.push(0x01);
            frame.extend_from_slice(&decl_len.to_le_bytes());
            frame.extend_from_slice(&[0x00, 0x01]);
            frame.extend_from_slice(&crc16_modbus(&frame[0..6]).to_le_bytes());
        }
    }
    frame.extend_from_slice(&inner);
    frame.extend_from_slice(&crc32_zlib(&inner).to_le_bytes());
    frame
}

/// Parse one complete frame (caller supplies the connection's family). Errors on a structurally broken
/// frame; a checksum mismatch is reported via `Frame::crc_ok`, not an error.
pub fn decode(family: Family, bytes: &[u8]) -> Result<Frame, ProtocolError> {
    let h = family.header();
    if bytes.first().copied() != Some(h.sof) {
        return Err(ProtocolError::BadSof);
    }
    let need_len = h.len_offset + 2;
    if bytes.len() < need_len {
        return Err(ProtocolError::TooShort { have: bytes.len(), need: need_len });
    }
    let decl_len = u16_at(bytes, h.len_offset).unwrap() as usize;
    let inner_len = decl_len.checked_sub(4).ok_or(ProtocolError::BadLength)?;
    if inner_len == 0 {
        return Err(ProtocolError::EmptyInner);
    }
    let total = decl_len + h.inner_start;
    if bytes.len() < total {
        return Err(ProtocolError::TooShort { have: bytes.len(), need: total });
    }

    let inner_end = h.inner_start + inner_len;
    let inner = &bytes[h.inner_start..inner_end];

    let header_ok = match h.header_crc {
        HeaderCrc::Crc8LenBytes => crc8(&bytes[1..3]) == bytes[3],
        HeaderCrc::Crc16Header => u16_at(bytes, 6) == Some(crc16_modbus(&bytes[0..6])),
    };
    let payload_ok = u32_at(bytes, inner_end) == Some(crc32_zlib(inner));

    Ok(Frame::new(family, bytes[..total].to_vec(), h.inner_start, inner_end, header_ok && payload_ok))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::command::GET_HELLO;
    use crate::hello::GEN5_CLIENT_HELLO;

    #[test]
    fn gen5_client_hello_encodes_byte_identical() {
        let f = encode(Family::Gen5, 35, 1, GET_HELLO, &[0x01]);
        assert_eq!(f, GEN5_CLIENT_HELLO);
    }

    #[test]
    fn gen5_roundtrip_crc_ok() {
        let f = decode(Family::Gen5, &GEN5_CLIENT_HELLO).unwrap();
        assert!(f.crc_ok);
        assert_eq!(f.packet_type(), 35);
        assert_eq!(f.cmd(), GET_HELLO);
        assert_eq!(f.payload(), &[0x01]);
    }

    #[test]
    fn gen4_roundtrip_crc_ok() {
        let f = encode(Family::Gen4, 35, 0, GET_HELLO, &[0xAB, 0xCD]);
        let d = decode(Family::Gen4, &f).unwrap();
        assert!(d.crc_ok);
        assert_eq!(d.payload(), &[0xAB, 0xCD]);
    }

    #[test]
    fn corrupt_payload_flags_crc_bad_not_err() {
        let mut f = GEN5_CLIENT_HELLO.to_vec();
        let n = f.len();
        f[n - 1] ^= 0xFF; // trash the CRC32 tail
        let d = decode(Family::Gen5, &f).unwrap();
        assert!(!d.crc_ok);
    }
}
