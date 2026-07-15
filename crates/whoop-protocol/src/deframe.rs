//! Length-based BLE reassembly: buffers notification fragments, reads the declared length, emits complete
//! frames, resyncs on SOF (not delimiter-reset — payloads contain 0xAA). One `Deframer` per notify
//! characteristic, so interleaved channel floods stay aligned.

use std::collections::HashMap;

use crate::bytes::u16_at;
use crate::family::{Channel, Family};
use crate::framing;
use crate::packet::Frame;

const MAX_FRAME: usize = 8192; // ~4x the largest real ~1920-byte raw/historical frame

pub struct Deframer {
    family: Family,
    buf: Vec<u8>,
    head: usize,
}

impl Deframer {
    pub fn new(family: Family) -> Self {
        Deframer { family, buf: Vec::new(), head: 0 }
    }

    /// Drop any buffered partial frame — call on (re)connect so a stalled frame can't wedge the session.
    pub fn reset(&mut self) {
        self.buf.clear();
        self.head = 0;
    }

    pub fn push(&mut self, data: &[u8]) -> Vec<Frame> {
        self.buf.extend_from_slice(data);
        let h = self.family.header();
        let mut out = Vec::new();

        loop {
            while self.head < self.buf.len() && self.buf[self.head] != h.sof {
                self.head += 1;
            }
            let avail = self.buf.len() - self.head;
            if avail < h.len_offset + 2 {
                break; // not enough to read the declared length yet
            }
            let decl = u16_at(&self.buf, self.head + h.len_offset).unwrap() as usize;
            let total = decl + h.inner_start;
            if decl < 5 || total > MAX_FRAME {
                self.head += 1; // implausible length → this 0xAA wasn't a real SOF
                continue;
            }
            if avail < total {
                break; // whole frame not here yet
            }
            let end = self.head + total;
            if let Ok(frame) = framing::decode(self.family, &self.buf[self.head..end]) {
                out.push(frame);
            }
            self.head = end;
        }

        if self.head > 0 {
            self.buf.drain(0..self.head);
            self.head = 0;
        }
        out
    }
}

/// One `Deframer` per logical notify channel, so interleaved streams stay frame-aligned.
pub struct DeframerMap {
    family: Family,
    map: HashMap<Channel, Deframer>,
}

impl DeframerMap {
    pub fn new(family: Family) -> Self {
        DeframerMap { family, map: HashMap::new() }
    }

    pub fn push(&mut self, channel: Channel, data: &[u8]) -> Vec<Frame> {
        let family = self.family;
        self.map.entry(channel).or_insert_with(|| Deframer::new(family)).push(data)
    }

    pub fn reset(&mut self) {
        for d in self.map.values_mut() {
            d.reset();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hello::GEN5_CLIENT_HELLO;

    #[test]
    fn reassembles_across_fragments() {
        let mut d = Deframer::new(Family::Gen5);
        assert!(d.push(&GEN5_CLIENT_HELLO[..7]).is_empty()); // partial → nothing yet
        let out = d.push(&GEN5_CLIENT_HELLO[7..]);
        assert_eq!(out.len(), 1);
        assert!(out[0].crc_ok);
    }

    #[test]
    fn splits_two_packed_frames() {
        let mut d = Deframer::new(Family::Gen5);
        let mut wire = GEN5_CLIENT_HELLO.to_vec();
        wire.extend_from_slice(&GEN5_CLIENT_HELLO);
        let out = d.push(&wire);
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn resyncs_past_leading_garbage() {
        let mut d = Deframer::new(Family::Gen5);
        let mut wire = vec![0x00, 0x11, 0x22];
        wire.extend_from_slice(&GEN5_CLIENT_HELLO);
        let out = d.push(&wire);
        assert_eq!(out.len(), 1);
        assert!(out[0].crc_ok);
    }
}
