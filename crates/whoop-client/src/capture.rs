//! Lossless capture-tap + reject-archive JSONL encoder. One snake_case line per reassembled frame; the
//! file IO (rotation / evict-oldest) belongs to the platform layer, this is the pure line.

use whoop_protocol::bytes::{from_hex, to_hex};
use whoop_protocol::{framing, live, records, Family, Frame, PacketType, Record};

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

/// Decode a raw-capture JSONL blob back into history records (plus the strap serial from `session_id`),
/// the file-level inverse of `capture_line`. Lines without a decodable historical frame are skipped.
pub fn decode_capture(text: &str, family: Family) -> (Option<String>, Vec<Record>) {
    let mut records = Vec::new();
    let mut strap = None;
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if strap.is_none() {
            strap = string_field(line, "session_id");
        }
        let Some(bytes) = string_field(line, "hex").and_then(|h| from_hex(&h)) else { continue };
        let Ok(frame) = framing::decode(family, &bytes) else { continue };
        if frame.packet().canonical() == PacketType::HistoricalData {
            if let Some(rec) = records::decode(&frame) {
                records.push(rec);
            }
        }
    }
    (strap, records)
}

/// Extract a quoted string field `"key":"value"` from one capture line (these values carry no escaped quotes).
fn string_field(line: &str, key: &str) -> Option<String> {
    let pat = format!("\"{key}\":\"");
    let start = line.find(&pat)? + pat.len();
    let rest = &line[start..];
    let end = rest.find('"')?;
    Some(rest[..end].to_string())
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

    #[test]
    fn decode_capture_inverts_capture_line() {
        let mut v18 = vec![0u8; 40];
        v18[4..8].copy_from_slice(&1_784_000_000u32.to_le_bytes());
        v18[11] = 96; // heart_rate
        let wire = framing::encode(Family::Gen5, 47, 18, 0, &v18);
        let frame = framing::decode(Family::Gen5, &wire).unwrap();
        let line = capture_line(1_700_000_000_000, "5AG0507838", "fd4b0005", &frame, true);

        let (strap, records) = decode_capture(&line, Family::Gen5);
        assert_eq!(strap.as_deref(), Some("5AG0507838"));
        assert_eq!(records.len(), 1);
        match &records[0] {
            Record::History(h) => assert_eq!(h.heart_rate, Some(96)),
            other => panic!("expected history, got {other:?}"),
        }
    }
}
