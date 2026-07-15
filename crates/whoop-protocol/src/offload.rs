//! Sans-IO historical-offload driver. Pure state machine: feed it CRC-checked frames, get back steps
//! (emit a record / write an ACK / done). A non-CRC-valid frame yields nothing — a forged HISTORY_END
//! must not advance the strap's trim cursor over unstored data.

use crate::command;
use crate::event::MetadataType;
use crate::family::Family;
use crate::framing;
use crate::live::history_end_data;
use crate::packet::{Frame, PacketType};
use crate::records::{self, Record};

#[derive(Clone, Debug, PartialEq)]
pub enum OffloadStep {
    /// A decoded historical record.
    Record(Record),
    /// A HISTORICAL_DATA_RESULT command frame to write (confirmed) — echoes the 8-byte end_data so the
    /// strap advances instead of re-serving the same chunk forever.
    Ack(Vec<u8>),
    /// The drain finished (HISTORY_COMPLETE).
    Complete,
}

pub struct Offload {
    family: Family,
    seq: u8,
}

impl Offload {
    pub fn new(family: Family) -> Self {
        Offload { family, seq: 0 }
    }

    /// The SEND_HISTORICAL_DATA(22) command frame that starts the drain (write confirmed to cmd-write).
    /// b3 (first payload byte) = 0x00, the value the strap expects for this opcode.
    pub fn start_frame(&mut self) -> Vec<u8> {
        self.command(command::SEND_HISTORICAL_DATA, &[0x00])
    }

    /// The ABORT_HISTORICAL_TRANSMITS(20) command frame to cleanly stop an in-flight drain.
    pub fn abort_frame(&mut self) -> Vec<u8> {
        self.command(command::ABORT_HISTORICAL_TRANSMITS, &[])
    }

    /// Feed one decoded frame; returns the steps to perform.
    pub fn on_frame(&mut self, f: &Frame) -> Vec<OffloadStep> {
        if !f.crc_ok {
            return Vec::new();
        }
        match f.packet().canonical() {
            PacketType::HistoricalData => match records::decode(f) {
                Some(r) => vec![OffloadStep::Record(r)],
                None => Vec::new(),
            },
            PacketType::Metadata => match MetadataType::from_u8(f.cmd()) {
                MetadataType::HistoryEnd => match history_end_data(f) {
                    Some(ed) => {
                        let mut payload = Vec::with_capacity(9);
                        payload.push(0x01);
                        payload.extend_from_slice(&ed);
                        vec![OffloadStep::Ack(self.command(command::HISTORICAL_DATA_RESULT, &payload))]
                    }
                    None => Vec::new(),
                },
                MetadataType::HistoryComplete => vec![OffloadStep::Complete],
                _ => Vec::new(),
            },
            _ => Vec::new(),
        }
    }

    fn command(&mut self, cmd: u8, payload: &[u8]) -> Vec<u8> {
        let seq = self.seq;
        self.seq = self.seq.wrapping_add(1);
        framing::encode(self.family, PacketType::Command.to_u8(), seq, cmd, payload)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn metadata_frame(meta: u8, mut body: Vec<u8>) -> Frame {
        // METADATA: inner = [49, seq, meta] + payload. end_data lives at inner 13..21 = payload 10..18.
        body.resize(18, 0);
        let wire = framing::encode(Family::Gen5, PacketType::Metadata.to_u8(), 0, meta, &body);
        framing::decode(Family::Gen5, &wire).unwrap()
    }

    #[test]
    fn history_end_produces_ack_echoing_end_data() {
        let end_data = [0xDE, 0xAD, 0xBE, 0xEF, 0x01, 0x02, 0x03, 0x04];
        let mut body = vec![0u8; 18];
        body[10..18].copy_from_slice(&end_data);
        let frame = metadata_frame(2, body); // 2 = HISTORY_END

        let mut off = Offload::new(Family::Gen5);
        let steps = off.on_frame(&frame);
        assert_eq!(steps.len(), 1);
        match &steps[0] {
            OffloadStep::Ack(wire) => {
                let ack = framing::decode(Family::Gen5, wire).unwrap();
                assert_eq!(ack.cmd(), command::HISTORICAL_DATA_RESULT);
                assert_eq!(&ack.payload()[..9], &[&[0x01][..], &end_data[..]].concat()[..]);
            }
            other => panic!("expected Ack, got {other:?}"),
        }
    }

    #[test]
    fn history_complete_signals_done() {
        let frame = metadata_frame(3, vec![]); // 3 = HISTORY_COMPLETE
        let mut off = Offload::new(Family::Gen5);
        assert_eq!(off.on_frame(&frame), vec![OffloadStep::Complete]);
    }

    #[test]
    fn crc_bad_frame_is_ignored() {
        let mut frame = metadata_frame(3, vec![]);
        frame.crc_ok = false;
        let mut off = Offload::new(Family::Gen5);
        assert!(off.on_frame(&frame).is_empty());
    }
}
