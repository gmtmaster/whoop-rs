//! Decode a NOOP raw-capture JSONL through the real codec and compare the strap's own record time
//! against the phone's arrival time.
//!
//! Run: `cargo run -p whoop-protocol --example capture_clock -- <capture.jsonl>`

use std::collections::BTreeMap;
use std::env;
use std::fs;

use whoop_protocol::deframe::Deframer;
use whoop_protocol::records::{decode, Record};
use whoop_protocol::Family;

fn hex_bytes(s: &str) -> Option<Vec<u8>> {
    if !s.len().is_multiple_of(2) {
        return None;
    }
    (0..s.len()).step_by(2).map(|i| u8::from_str_radix(&s[i..i + 2], 16).ok()).collect()
}

/// One decoded per-second record with the wall-clock instant the phone received it.
struct Row {
    arrival_s: i64,
    unix: i64,
    hr: Option<u8>,
    rr: Vec<u16>,
}

fn main() {
    let path = env::args().nth(1).expect("usage: capture_clock <capture.jsonl>");
    let text = fs::read_to_string(&path).expect("read capture");

    let mut rows: Vec<Row> = Vec::new();
    let (mut lines, mut framed, mut decoded, mut crc_bad) = (0u64, 0u64, 0u64, 0u64);

    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        lines += 1;
        let v: serde_json::Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(_) => continue,
        };
        if v.get("type_name").and_then(|t| t.as_str()) != Some("HISTORICAL_DATA") {
            continue;
        }
        let (Some(hex), Some(ms)) =
            (v.get("hex").and_then(|h| h.as_str()), v.get("captured_at_ms").and_then(|m| m.as_i64()))
        else {
            continue;
        };
        let Some(bytes) = hex_bytes(hex) else { continue };

        // A capture line already holds one complete frame, so the deframer runs per line.
        let mut d = Deframer::new(Family::Gen5);
        for frame in d.push(&bytes) {
            framed += 1;
            if !frame.crc_ok {
                crc_bad += 1;
                continue;
            }
            if let Some(Record::History(h)) = decode(&frame) {
                decoded += 1;
                rows.push(Row {
                    arrival_s: ms / 1000,
                    unix: h.unix as i64,
                    hr: h.heart_rate,
                    rr: h.rr_intervals,
                });
            }
        }
    }

    println!("lines={lines}  frames={framed}  crc_bad={crc_bad}  history decoded={decoded}");
    if rows.is_empty() {
        return;
    }

    // Clock: the strap stamps the record, the phone stamps arrival. Their difference is the offset
    // plus the offload's own latency, so the FLOOR of the spread is the clock term.
    let mut deltas: Vec<i64> = rows.iter().map(|r| r.arrival_s - r.unix).collect();
    deltas.sort_unstable();
    let pct = |p: f64| deltas[((deltas.len() - 1) as f64 * p) as usize];
    println!(
        "\narrival - record, seconds:  min={}  p01={}  p25={}  median={}  p75={}  p99={}  max={}",
        deltas[0],
        pct(0.01),
        pct(0.25),
        pct(0.50),
        pct(0.75),
        pct(0.99),
        deltas[deltas.len() - 1]
    );

    // R-R per record, and per distinct strap second.
    let mut per_record: BTreeMap<usize, usize> = BTreeMap::new();
    let mut by_second: BTreeMap<i64, usize> = BTreeMap::new();
    for r in &rows {
        *per_record.entry(r.rr.len()).or_default() += 1;
        *by_second.entry(r.unix).or_default() += r.rr.len();
    }
    println!("\nR-R per record: {per_record:?}");

    let beats: usize = by_second.values().sum();
    let seconds = by_second.len();
    let span = rows.iter().map(|r| r.unix).max().unwrap() - rows.iter().map(|r| r.unix).min().unwrap();
    let beat_ms: u64 = rows.iter().flat_map(|r| r.rr.iter()).map(|&v| u64::from(v)).sum();
    println!(
        "wire: records={}  distinct seconds={}  beats={}  beats/sec={:.2}  span={}s ({:.2} h)",
        rows.len(),
        seconds,
        beats,
        beats as f64 / seconds as f64,
        span,
        span as f64 / 3600.0
    );
    println!(
        "wire beat-time / elapsed = {:.2}x   (1.0 = every beat accounted for, no gaps)",
        beat_ms as f64 / (span as f64 * 1000.0)
    );

    let mut over: Vec<(i64, usize)> = by_second.iter().filter(|(_, &c)| c > 4).map(|(&t, &c)| (t, c)).collect();
    over.sort_by_key(|&(_, c)| std::cmp::Reverse(c));
    println!("\nstrap seconds carrying more than the 4-slot record cap: {}", over.len());
    for (t, c) in over.iter().take(5) {
        println!("  unix={t}  beats={c}");
    }

    // A repeated strap second is the only way the offload alone could double a second.
    let repeats = rows.len() - seconds;
    println!("\nrecords={} distinct seconds={} -> repeated seconds={repeats}", rows.len(), seconds);

    // `--dump <path>`: one `unix,rr_ms` line per beat, for joining against what the app stored.
    let mut a = env::args().skip(2);
    if let (Some(flag), Some(out)) = (a.next(), a.next()) {
        if flag == "--dump" {
            let mut s = String::from("unix,hr,rr\n");
            for r in &rows {
                for v in &r.rr {
                    s.push_str(&format!("{},{},{}\n", r.unix, r.hr.map_or(0, u16::from), v));
                }
            }
            fs::write(&out, s).expect("write dump");
            println!("dumped {beats} beats to {out}");
        }
    }
}
