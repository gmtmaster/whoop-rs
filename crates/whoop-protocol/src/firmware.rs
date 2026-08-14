//! Firmware-image transfer frames (GEN5 / AMBIQ). Builds the four DFU commands and splits an image
//! into offset-tagged chunks; `firmware_image` checks the header those bytes carry. Sans-IO: nothing
//! here writes to a strap, and the opcodes stay in `command::DESTRUCTIVE` so the blind path refuses them.

use crate::bytes::prefixed;
use crate::command;
use crate::family::Family;
use crate::firmware_image::transfer_len;
use crate::framing;

/// Payload bytes per `LOAD_FIRMWARE_DATA_NEW` frame on AMBIQ straps.
pub const CHUNK_AMBIQ: usize = 220;

/// Revision marker carried as the first payload byte of every firmware command.
const REVISION_1: u8 = 0x01;

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

/// How the final data frame is filled.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Tail {
    /// Real remaining bytes only.
    Exact,
    /// Zero-padded to [`CHUNK_AMBIQ`], length byte [`CHUNK_AMBIQ`].
    #[default]
    Pad,
}

/// Split `image` into [`CHUNK_AMBIQ`]-sized data frames, seq-numbered from `start_seq` (wrapping).
/// Only the declared transfer length is sent; `None` when the header does not parse.
pub fn data_frames(image: &[u8], start_seq: u8) -> Option<Vec<Chunk>> {
    data_frames_with(image, start_seq, Tail::Exact)
}

/// [`data_frames`] with an explicit tail policy. Every chunk but the last is full width either way, so
/// the policy only ever changes the final frame.
pub fn data_frames_with(image: &[u8], start_seq: u8, tail: Tail) -> Option<Vec<Chunk>> {
    let total = transfer_len(image)?;
    let mut out = Vec::with_capacity(total.div_ceil(CHUNK_AMBIQ));
    for (i, chunk) in image[..total].chunks(CHUNK_AMBIQ).enumerate() {
        let offset = (i * CHUNK_AMBIQ) as u32;
        let seq = start_seq.wrapping_add(i as u8);
        let frame = if tail == Tail::Pad && chunk.len() < CHUNK_AMBIQ {
            let mut padded = vec![0u8; CHUNK_AMBIQ];
            padded[..chunk.len()].copy_from_slice(chunk);
            data_frame(seq, offset, &padded)?
        } else {
            data_frame(seq, offset, chunk)?
        };
        out.push(Chunk { offset, frame });
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::firmware_image::{fixture::sized, IMAGE_HEADER_LEN};
    use crate::packet::PacketType;

    /// Inner bytes of an encoded GEN5 command: `[type][seq][cmd] + payload`, header stripped.
    fn inner(frame: &[u8]) -> &[u8] {
        &frame[Family::Gen5.header().inner_start..frame.len() - 4]
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
        let img = sized(CHUNK_AMBIQ * 3 + 7);
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
        let mut img = sized(100);
        img.extend_from_slice(&[0xFF; 4096]);
        let chunks = data_frames(&img, 0).unwrap();
        assert_eq!(chunks.len(), (IMAGE_HEADER_LEN + 100).div_ceil(CHUNK_AMBIQ));
    }

    #[test]
    fn a_padded_tail_declares_the_full_width_and_zero_fills_it() {
        let img = sized(CHUNK_AMBIQ * 3 + 7);
        let exact = data_frames_with(&img, 0, Tail::Exact).unwrap();
        let padded = data_frames_with(&img, 0, Tail::Pad).unwrap();
        assert_eq!(padded.len(), exact.len());

        let last = inner(&padded.last().unwrap().frame);
        assert_eq!(last[8], CHUNK_AMBIQ as u8);
        let real = transfer_len(&img).unwrap() % CHUNK_AMBIQ;
        assert!(last[9 + real..9 + CHUNK_AMBIQ].iter().all(|&b| b == 0));
        assert_eq!(&last[9..9 + real], &inner(&exact.last().unwrap().frame)[9..9 + real]);
    }

    /// The padded tail is unconditional: a final frame whose full width would run past the image, or
    /// past a flash block edge, still declares and carries [`CHUNK_AMBIQ`].
    #[test]
    fn the_padded_tail_does_not_shrink_near_a_block_edge() {
        let img = sized(65_440 - IMAGE_HEADER_LEN);
        let chunks = data_frames_with(&img, 0, Tail::Pad).unwrap();
        let last = chunks.last().unwrap();
        assert_eq!(last.offset, 65_340);
        assert_eq!(inner(&last.frame)[8], CHUNK_AMBIQ as u8);
    }

    #[test]
    fn the_old_entry_point_still_emits_the_short_tail() {
        let img = sized(CHUNK_AMBIQ * 3 + 7);
        let old = data_frames(&img, 4).unwrap();
        let exact = data_frames_with(&img, 4, Tail::Exact).unwrap();
        assert_eq!(old.len(), exact.len());
        for (a, b) in old.iter().zip(&exact) {
            assert_eq!((a.offset, &a.frame), (b.offset, &b.frame));
        }
        let tail_len = (IMAGE_HEADER_LEN + CHUNK_AMBIQ * 3 + 7) % CHUNK_AMBIQ;
        assert_eq!(inner(&old.last().unwrap().frame)[8], tail_len as u8);
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
