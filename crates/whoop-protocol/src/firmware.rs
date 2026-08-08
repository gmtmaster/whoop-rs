//! Firmware-image transfer frames (GEN5 / AMBIQ). Builds the four DFU commands and splits an image
//! into offset-tagged chunks. Sans-IO: nothing here writes to a strap, and the opcodes stay in
//! `command::DESTRUCTIVE` so the blind path still refuses them.

use crate::bytes::prefixed;
use crate::command;
use crate::family::Family;
use crate::framing;

/// Payload bytes per `LOAD_FIRMWARE_DATA_NEW` frame on AMBIQ straps.
pub const CHUNK_AMBIQ: usize = 220;

/// Plaintext header ahead of the compressed image in a `.zbin`.
pub const IMAGE_HEADER_LEN: usize = 512;

/// Offset of the u32 LE payload length inside the header.
const LEN_OFFSET: usize = 4;

/// Revision marker carried as the first payload byte of every firmware command.
const REVISION_1: u8 = 0x01;

/// Transfer length of a `.zbin`: its header plus the payload length the header declares. `None` when
/// `image` is shorter than the header or the declared length overruns it.
pub fn transfer_len(image: &[u8]) -> Option<usize> {
    if image.len() < IMAGE_HEADER_LEN {
        return None;
    }
    let declared = u32::from_le_bytes(image[LEN_OFFSET..LEN_OFFSET + 4].try_into().ok()?) as usize;
    let total = IMAGE_HEADER_LEN.checked_add(declared)?;
    (total <= image.len()).then_some(total)
}

/// A firmware command whose whole payload is the revision marker.
fn bare_frame(seq: u8, cmd: u8) -> Vec<u8> {
    framing::command(Family::Gen5, seq, cmd, &[REVISION_1])
}

/// Open a firmware load.
pub fn start_frame(seq: u8) -> Vec<u8> {
    bare_frame(seq, command::START_FIRMWARE_LOAD_NEW)
}

/// Ask the strap to check what it has received.
pub fn verify_frame(seq: u8) -> Vec<u8> {
    bare_frame(seq, command::VERIFY_FIRMWARE_IMAGE)
}

/// Commit the received image.
pub fn process_frame(seq: u8) -> Vec<u8> {
    bare_frame(seq, command::PROCESS_FIRMWARE_IMAGE_NEW)
}

/// One image chunk: `[rev][offset u32 LE][len u8][data]`. `chunk` must be 1..=[`CHUNK_AMBIQ`] bytes,
/// since the length field is a single byte.
pub fn data_frame(seq: u8, offset: u32, chunk: &[u8]) -> Option<Vec<u8>> {
    if chunk.is_empty() || chunk.len() > CHUNK_AMBIQ {
        return None;
    }
    let mut body = Vec::with_capacity(5 + chunk.len());
    body.extend_from_slice(&offset.to_le_bytes());
    body.push(chunk.len() as u8);
    body.extend_from_slice(chunk);
    Some(framing::command(
        Family::Gen5,
        seq,
        command::LOAD_FIRMWARE_DATA_NEW,
        &prefixed(REVISION_1, &body),
    ))
}

/// One planned write: the frame and the image offset it carries.
pub struct Chunk {
    pub offset: u32,
    pub frame: Vec<u8>,
}

/// Split `image` into [`CHUNK_AMBIQ`]-sized data frames, seq-numbered from `start_seq` (wrapping).
/// Only the bytes [`transfer_len`] reports are sent; `None` when the header does not parse.
pub fn data_frames(image: &[u8], start_seq: u8) -> Option<Vec<Chunk>> {
    let total = transfer_len(image)?;
    let mut out = Vec::with_capacity(total.div_ceil(CHUNK_AMBIQ));
    for (i, chunk) in image[..total].chunks(CHUNK_AMBIQ).enumerate() {
        let offset = (i * CHUNK_AMBIQ) as u32;
        out.push(Chunk { offset, frame: data_frame(start_seq.wrapping_add(i as u8), offset, chunk)? });
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::packet::PacketType;

    /// A `.zbin`-shaped buffer: header declaring `payload` bytes, then that payload.
    fn image(payload: usize) -> Vec<u8> {
        let mut v = vec![0u8; IMAGE_HEADER_LEN + payload];
        v[LEN_OFFSET..LEN_OFFSET + 4].copy_from_slice(&(payload as u32).to_le_bytes());
        v
    }

    /// Inner bytes of an encoded GEN5 command: `[type][seq][cmd] + payload`, header stripped.
    fn inner(frame: &[u8]) -> &[u8] {
        &frame[Family::Gen5.header().inner_start..frame.len() - 4]
    }

    #[test]
    fn transfer_len_is_the_header_plus_the_declared_payload() {
        assert_eq!(transfer_len(&image(1000)), Some(IMAGE_HEADER_LEN + 1000));
    }

    #[test]
    fn transfer_len_rejects_a_short_buffer_and_an_overrunning_length() {
        assert_eq!(transfer_len(&[0u8; 16]), None);
        let mut v = image(1000);
        v[LEN_OFFSET..LEN_OFFSET + 4].copy_from_slice(&u32::MAX.to_le_bytes());
        assert_eq!(transfer_len(&v), None);
    }

    #[test]
    fn the_bare_commands_carry_only_the_revision_marker() {
        for (frame, cmd) in [
            (start_frame(0), command::START_FIRMWARE_LOAD_NEW),
            (verify_frame(0), command::VERIFY_FIRMWARE_IMAGE),
            (process_frame(0), command::PROCESS_FIRMWARE_IMAGE_NEW),
        ] {
            let i = inner(&frame);
            assert_eq!(i[0], PacketType::Command.to_u8());
            assert_eq!(i[2], cmd);
            assert_eq!(i[3], REVISION_1);
        }
    }

    #[test]
    fn a_data_frame_is_revision_offset_len_then_the_chunk() {
        let frame = data_frame(7, 0x0001_2340, &[0xAA; 8]).unwrap();
        let i = inner(&frame);
        assert_eq!(i[2], command::LOAD_FIRMWARE_DATA_NEW);
        assert_eq!(i[3], REVISION_1);
        assert_eq!(&i[4..8], &0x0001_2340u32.to_le_bytes());
        assert_eq!(i[8], 8);
        assert_eq!(&i[9..17], &[0xAA; 8]);
    }

    #[test]
    fn a_chunk_that_cannot_fit_the_one_byte_length_is_refused() {
        assert!(data_frame(0, 0, &[]).is_none());
        assert!(data_frame(0, 0, &vec![0u8; CHUNK_AMBIQ + 1]).is_none());
        assert!(data_frame(0, 0, &vec![0u8; CHUNK_AMBIQ]).is_some());
    }

    #[test]
    fn chunk_offsets_are_contiguous_and_cover_exactly_the_transfer_length() {
        let img = image(CHUNK_AMBIQ * 3 + 7);
        let total = transfer_len(&img).unwrap();
        let chunks = data_frames(&img, 0).unwrap();
        assert_eq!(chunks.len(), total.div_ceil(CHUNK_AMBIQ));
        for (i, c) in chunks.iter().enumerate() {
            assert_eq!(c.offset as usize, i * CHUNK_AMBIQ);
        }
        let last = chunks.last().unwrap();
        assert_eq!(last.offset as usize + total % CHUNK_AMBIQ, total);
    }

    #[test]
    fn trailing_bytes_past_the_declared_length_are_not_sent() {
        let mut img = image(100);
        img.extend_from_slice(&[0xFF; 4096]);
        let chunks = data_frames(&img, 0).unwrap();
        assert_eq!(chunks.len(), (IMAGE_HEADER_LEN + 100).div_ceil(CHUNK_AMBIQ));
    }

    #[test]
    fn the_firmware_opcodes_stay_refused_on_the_blind_path() {
        for op in [
            command::START_FIRMWARE_LOAD_NEW,
            command::LOAD_FIRMWARE_DATA_NEW,
            command::PROCESS_FIRMWARE_IMAGE_NEW,
        ] {
            assert!(command::is_destructive(op));
            assert!(command::is_forbidden(op));
        }
    }
}
