//! Lossless capture-tap + reject-archive JSONL encoder. One snake_case line per reassembled frame; the
//! file IO (rotation / evict-oldest) belongs to the platform layer, this is the pure line.

use whoop_protocol::bytes::to_hex;
use whoop_protocol::{live, records, Frame, PacketType, Record};

/// Encode one captured frame as a JSONL line. `offload` marks frames seen during a history drain.
pub fn capture_line(captured_at_ms: i64, session_id: &str, characteristic: &str, frame: &Frame, offload: bool) -> String {
    format!(
        "{{\"captured_at_ms\":{},\"session_id\":{},\"characteristic\":{},\"type_name\":{},\"crc_ok\":{},\"offload\":{},\"size\":{},\"parsed\":{},\"hex\":\"{}\"}}",
        captured_at_ms,
        json_str(session_id),
        json_str(characteristic),
        json_str(frame.packet().name()),
        frame.crc_ok,
        offload,
        frame.raw().len(),
        parsed_fields(frame),
        to_hex(frame.raw()),
    )
}

/// The reject archive (undecodable frames captured before the trim ACK) uses the same line shape.
pub fn archive_line(captured_at_ms: i64, session_id: &str, frame: &Frame) -> String {
    capture_line(captured_at_ms, session_id, "rejected", frame, true)
}

fn parsed_fields(f: &Frame) -> String {
    let hr = match f.packet().canonical() {
        PacketType::RealtimeData => live::realtime(f).map(|r| r.heart_rate as i64),
        PacketType::HistoricalData => match records::decode(f) {
            Some(Record::History(h)) => h.heart_rate.map(|v| v as i64),
            _ => None,
        },
        _ => None,
    };
    match hr {
        Some(v) => format!("{{\"heart_rate\":{v}}}"),
        None => "{}".to_string(),
    }
}

fn json_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            _ => out.push(c),
        }
    }
    out.push('"');
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use whoop_protocol::{framing, Family};

    #[test]
    fn capture_line_matches_schema() {
        let mut v18 = vec![0u8; 40];
        v18[4..8].copy_from_slice(&1_784_000_000u32.to_le_bytes()); // unix
        v18[11] = 96; // heart_rate @ payload 11 (inner 14)
        let wire = framing::encode(Family::Gen5, 47, 18, 0, &v18);
        let frame = framing::decode(Family::Gen5, &wire).unwrap();

        let line = capture_line(1_700_000_000_000, "sess1", "fd4b0005", &frame, true);
        assert!(line.contains("\"type_name\":\"HISTORICAL_DATA\""));
        assert!(line.contains("\"crc_ok\":true"));
        assert!(line.contains("\"offload\":true"));
        assert!(line.contains("\"heart_rate\":96"));
        assert!(line.contains(&format!("\"size\":{}", frame.raw().len())));
    }
}
