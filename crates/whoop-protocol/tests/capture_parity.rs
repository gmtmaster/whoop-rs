//! Offline parity over a real 5.0 capture (BackfillCaptureJsonl). Set WHOOP_CAPTURE to the .jsonl path;
//! self-skips when unset. Proves the Rust decoders reproduce the fields the Kotlin frame layer wrote at
//! capture time — the #1 target being HISTORY_END trim_cursor/unix (a wrong trim = deleted history).

use serde_json::Value;
use std::io::{BufRead, BufReader};

use whoop_protocol::{console, framing, live, records, response, Family, Record};

fn paren_raw(s: &str) -> Option<u8> {
    let l = s.rfind('(')?;
    let r = s.rfind(')')?;
    s.get(l + 1..r)?.parse().ok()
}

#[test]
fn capture_parity() {
    let Ok(path) = std::env::var("WHOOP_CAPTURE") else {
        eprintln!("WHOOP_CAPTURE unset — skipping capture parity");
        return;
    };
    let file = std::fs::File::open(&path).expect("open capture");
    let rdr = BufReader::new(file);

    let (mut lines, mut decode_err) = (0u64, 0u64);
    let (mut meta_all, mut meta_type_ok, mut end_ok, mut end_total) = (0u64, 0u64, 0u64, 0u64);
    let (mut hist_total, mut hist_ok) = (0u64, 0u64);
    let (mut ev_total, mut ev_ok) = (0u64, 0u64);
    let (mut con_total, mut con_ok) = (0u64, 0u64);
    let (mut cr_total, mut cr_cmd_ok) = (0u64, 0u64);
    let mut mismatches: Vec<String> = Vec::new();

    for line in rdr.lines() {
        let line = line.unwrap();
        if line.trim().is_empty() {
            continue;
        }
        lines += 1;
        let o: Value = serde_json::from_str(&line).expect("json");
        let hex = o["hex"].as_str().unwrap();
        let raw = whoop_protocol::bytes::from_hex(hex).expect("hex");
        let tn = o["type_name"].as_str().unwrap_or("");
        let p = &o["parsed"];

        let frame = match framing::decode(Family::Gen5, &raw) {
            Ok(f) => f,
            Err(_) => {
                decode_err += 1;
                continue;
            }
        };

        match tn {
            "METADATA" => {
                meta_all += 1;
                let m = live::metadata(&frame).expect("metadata decode");
                let want_mt = p["meta_type"].as_str().and_then(paren_raw);
                if want_mt == Some(m.meta_type) {
                    meta_type_ok += 1;
                }
                if p["meta_type"].as_str().unwrap_or("").starts_with("HISTORY_END") {
                    end_total += 1;
                    // The fixture stored u32 as a possibly-negative signed int; widen back through i64.
                    let want_unix = p["unix"].as_i64().unwrap() as u32;
                    let want_trim = p["trim_cursor"].as_i64().unwrap() as u32;
                    if m.unix == want_unix && m.trim_cursor == want_trim {
                        end_ok += 1;
                    } else if mismatches.len() < 20 {
                        mismatches.push(format!(
                            "END unix rust={} want={} trim rust={} want={}",
                            m.unix, want_unix, m.trim_cursor, want_trim
                        ));
                    }
                }
            }
            "HISTORICAL_DATA" => {
                hist_total += 1;
                // v26 PPG carries no History summary; count a decode as OK if it yields any record.
                let ok = match records::decode(&frame) {
                    Some(Record::History(_)) | Some(Record::Ppg(_)) | Some(Record::Imu(_))
                    | Some(Record::Optical(_)) => true,
                    None => false,
                };
                if ok {
                    hist_ok += 1;
                } else if mismatches.len() < 20 {
                    mismatches.push(format!("HIST decode None ver={}", frame.inner().first().copied().unwrap_or(0)));
                }
            }
            "EVENT" => {
                ev_total += 1;
                let e = live::event(&frame);
                let want_num = p["event"].as_str().and_then(paren_raw);
                let want_ts = p["event_timestamp"].as_u64().map(|v| v as u32);
                let num_ok = e.as_ref().map(|e| Some(e.number) == want_num).unwrap_or(false);
                let ts_ok = match (&e, want_ts) {
                    (Some(e), Some(w)) => e.timestamp == w,
                    (_, None) => true,
                    _ => false,
                };
                // payload hex parity when present
                let hex_ok = match p["event_payload_hex"].as_str() {
                    Some(w) => live::event_payload_hex(&frame).as_deref() == Some(w),
                    None => true,
                };
                if num_ok && ts_ok && hex_ok {
                    ev_ok += 1;
                } else if mismatches.len() < 20 {
                    mismatches.push(format!("EVENT num_ok={num_ok} ts_ok={ts_ok} hex_ok={hex_ok}"));
                }
            }
            "CONSOLE_LOGS" => {
                con_total += 1;
                match p["console"].as_str() {
                    Some(w) => {
                        if console::text(&frame).as_deref() == Some(w) {
                            con_ok += 1;
                        } else if mismatches.len() < 20 {
                            mismatches.push(format!(
                                "CONSOLE rust={:?} want={:?}",
                                console::text(&frame),
                                w
                            ));
                        }
                    }
                    None => con_ok += 1,
                }
            }
            "COMMAND_RESPONSE" => {
                cr_total += 1;
                // resp_cmd raw == frame cmd byte (label built app-side from this raw).
                let want_cmd = p["resp_cmd"].as_str().and_then(paren_raw);
                if want_cmd == Some(frame.cmd()) {
                    cr_cmd_ok += 1;
                }
                let _ = response::decode(&frame);
            }
            _ => {}
        }
    }

    eprintln!("== capture parity ({lines} lines, {decode_err} frame-decode errors) ==");
    eprintln!("METADATA:  {meta_type_ok}/{meta_all} meta_type,  HISTORY_END trim+unix {end_ok}/{end_total}");
    eprintln!("HISTORICAL:{hist_ok}/{hist_total} decode a record");
    eprintln!("EVENT:     {ev_ok}/{ev_total} num+ts+payload");
    eprintln!("CONSOLE:   {con_ok}/{con_total} text");
    eprintln!("CMD_RESP:  {cr_cmd_ok}/{cr_total} resp_cmd raw == cmd byte");
    for m in &mismatches {
        eprintln!("  mismatch: {m}");
    }

    // The safety-critical gate: every HISTORY_END trim+unix must be byte-identical, and every metadata
    // meta_type must match, or routing the offload trim through Rust is unsafe.
    assert_eq!(end_ok, end_total, "HISTORY_END trim/unix parity");
    assert_eq!(meta_type_ok, meta_all, "metadata meta_type parity");
    assert_eq!(ev_ok, ev_total, "EVENT parity");
    assert_eq!(con_ok, con_total, "CONSOLE parity");
    assert_eq!(cr_cmd_ok, cr_total, "COMMAND_RESPONSE resp_cmd parity");
}

/// Real WHOOP 4.0 GET_DATA_RANGE frames — pins the scan
/// on hardware frames before the app routes the sync gate through the FFI.
#[test]
fn data_range_scan_matches_real_whoop4_frames() {
    use whoop_protocol::bytes::from_hex;
    use whoop_protocol::response::{data_range_scan_newest, data_range_scan_oldest};
    let wall_now = 1_783_786_000u64;
    let skew = 48 * 3600u64;
    let cases: [(&str, u32); 3] = [
        ("aa100057305d22009968526a083900001d2e2263", 1_783_785_625),
        ("aa10005730612200a268526ab0290000e87d155d", 1_783_785_634),
        ("aa100057307c2200e768526a78760000c997138d", 1_783_785_703),
    ];
    for (h, want) in cases {
        let f = from_hex(h).unwrap();
        assert_eq!(data_range_scan_newest(&f, wall_now, skew), Some(want), "newest {h}");
    }
    // oldest aligned-from-7 skips the spurious offset-6 straddle → None.
    assert_eq!(
        data_range_scan_oldest(&from_hex("aa100057305d22009968526a083900001d2e2263").unwrap()),
        None,
    );
}
