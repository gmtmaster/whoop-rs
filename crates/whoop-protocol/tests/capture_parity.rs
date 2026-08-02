//! Parity between the Rust decoders and the fields the app's frame layer recorded at capture time.
//! One tally over two inputs: a capture built in this file, which runs from a clean checkout, and a
//! whole capture off disk (`#[ignore]`d — real captures hold personal biometric frames and are
//! git-ignored by policy, so none can be tracked here).
//!
//! The safety-critical claim is HISTORY_END trim_cursor/unix — a wrong trim deletes strap history.
//! Every ratio is paired with a non-zero denominator check, because `ok == total` at zero frames is
//! the shape this gate used to pass in: a capture carrying no HISTORY_END satisfied it outright.

use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Cursor};

use whoop_protocol::bytes::to_hex;
use whoop_protocol::{console, framing, live, records, response, Family, Record};

fn paren_raw(s: &str) -> Option<u8> {
    let l = s.rfind('(')?;
    let r = s.rfind(')')?;
    s.get(l + 1..r)?.parse().ok()
}

/// Which record variant a version byte has to route to. An unlisted version is a fault, not a pass.
fn variant_ok(version: u8, r: &Record) -> bool {
    matches!(
        (version, r),
        (18, Record::History(_))
            | (20, Record::Optical(_))
            | (21, Record::Imu(_))
            | (26, Record::Ppg(_))
    )
}

#[derive(Default)]
struct Tally {
    lines: u64,
    decode_err: u64,
    meta_all: u64,
    meta_type_ok: u64,
    end_total: u64,
    end_ok: u64,
    hist_total: u64,
    hist_ok: u64,
    /// version byte → (frames, frames that routed to that version's record and carried a unix)
    by_version: BTreeMap<u8, (u64, u64)>,
    ev_total: u64,
    ev_ok: u64,
    con_total: u64,
    con_ok: u64,
    cr_total: u64,
    cr_cmd_ok: u64,
    /// COMMAND_RESPONSE frames the capture recorded no `resp_cmd` for: no expected value, so they
    /// are reported rather than counted as either a pass or a failure.
    cr_no_expected: u64,
    mismatches: Vec<String>,
}

impl Tally {
    fn report(&self) -> String {
        let vers: Vec<String> =
            self.by_version.iter().map(|(v, (t, o))| format!("v{v} {o}/{t}")).collect();
        let mut s = format!(
            "== capture parity ({} frames, {} frame-decode errors) ==\n\
             METADATA:  {}/{} meta_type,  HISTORY_END trim+unix {}/{}\n\
             HISTORICAL:{}/{} route to their version's record  [{}]\n\
             EVENT:     {}/{} num+ts+payload\n\
             CONSOLE:   {}/{} text\n\
             CMD_RESP:  {}/{} resp_cmd raw == cmd byte (no expected value on {} frames)",
            self.lines,
            self.decode_err,
            self.meta_type_ok,
            self.meta_all,
            self.end_ok,
            self.end_total,
            self.hist_ok,
            self.hist_total,
            vers.join(", "),
            self.ev_ok,
            self.ev_total,
            self.con_ok,
            self.con_total,
            self.cr_cmd_ok,
            self.cr_total,
            self.cr_no_expected,
        );
        for m in &self.mismatches {
            s.push_str(&format!("\n  mismatch: {m}"));
        }
        s
    }

    /// Every violated claim, empty when the capture is fully reproduced. A missing denominator is a
    /// violation in its own right: no frames of a class means that class carries no evidence.
    fn faults(&self) -> Vec<String> {
        let mut f = Vec::new();
        let mut need = |n: u64, what: &str| {
            if n == 0 {
                f.push(format!("no {what}: that claim has no evidence in this capture"));
            }
        };
        need(self.lines, "frames at all");
        need(self.meta_all, "METADATA frames");
        need(self.end_total, "HISTORY_END frames");
        need(self.hist_total, "HISTORICAL_DATA frames");
        need(self.ev_total, "EVENT frames");
        need(self.con_total, "CONSOLE_LOGS frames");
        need(self.cr_total, "COMMAND_RESPONSE frames carrying an expected resp_cmd");

        let mut eq = |ok: u64, total: u64, what: &str| {
            if ok != total {
                f.push(format!("{what}: {ok}/{total}"));
            }
        };
        eq(self.end_ok, self.end_total, "HISTORY_END trim_cursor/unix parity");
        eq(self.meta_type_ok, self.meta_all, "metadata meta_type parity");
        eq(self.hist_ok, self.hist_total, "HISTORICAL_DATA version-to-record routing");
        eq(self.ev_ok, self.ev_total, "EVENT number/timestamp/payload parity");
        eq(self.con_ok, self.con_total, "CONSOLE_LOGS text parity");
        eq(self.cr_cmd_ok, self.cr_total, "COMMAND_RESPONSE resp_cmd parity");

        if self.decode_err != 0 {
            f.push(format!("{} frames failed to decode at all", self.decode_err));
        }
        if self.by_version.is_empty() {
            f.push("no record versions seen".to_string());
        }
        for (v, (total, ok)) in &self.by_version {
            if ok != total {
                f.push(format!("v{v} routing: {ok}/{total}"));
            }
        }
        f
    }
}

fn assert_parity(t: &Tally) {
    let faults = t.faults();
    assert!(faults.is_empty(), "{}\n{}", faults.join("\n"), t.report());
}

fn tally(rdr: impl BufRead) -> Tally {
    let mut t = Tally::default();

    for line in rdr.lines() {
        let line = line.unwrap();
        // The capture writer prepends a `#` provenance header (app build, device, a biometric-content
        // warning). Without this skip the whole gate panics on line 1 of any current capture.
        if line.trim().is_empty() || line.trim_start().starts_with('#') {
            continue;
        }
        t.lines += 1;
        let o: Value = serde_json::from_str(&line).expect("json");
        let hex = o["hex"].as_str().unwrap();
        let raw = whoop_protocol::bytes::from_hex(hex).expect("hex");
        let tn = o["type_name"].as_str().unwrap_or("");
        let p = &o["parsed"];

        let frame = match framing::decode(Family::Gen5, &raw) {
            Ok(f) => f,
            Err(_) => {
                t.decode_err += 1;
                continue;
            }
        };

        match tn {
            "METADATA" => {
                t.meta_all += 1;
                let m = live::metadata(&frame).expect("metadata decode");
                let want_mt = p["meta_type"].as_str().and_then(paren_raw);
                if want_mt == Some(m.meta_type) {
                    t.meta_type_ok += 1;
                } else if t.mismatches.len() < 20 {
                    t.mismatches
                        .push(format!("META meta_type rust={} want={want_mt:?}", m.meta_type));
                }
                if p["meta_type"].as_str().unwrap_or("").starts_with("HISTORY_END") {
                    t.end_total += 1;
                    // The capture stores u32 as a possibly-negative signed int; widen back through i64.
                    let want_unix = p["unix"].as_i64().unwrap() as u32;
                    let want_trim = p["trim_cursor"].as_i64().unwrap() as u32;
                    if m.unix == want_unix && m.trim_cursor == want_trim {
                        t.end_ok += 1;
                    } else if t.mismatches.len() < 20 {
                        t.mismatches.push(format!(
                            "END unix rust={} want={want_unix} trim rust={} want={want_trim}",
                            m.unix, m.trim_cursor
                        ));
                    }
                }
            }
            "HISTORICAL_DATA" => {
                t.hist_total += 1;
                let version = frame.version();
                let slot = t.by_version.entry(version).or_default();
                slot.0 += 1;
                // The capture format records no per-record expected fields, so what is checkable
                // here is that the version byte routes to its own record type and yields a stamp.
                let ok = match records::decode(&frame) {
                    Some(r) => {
                        variant_ok(version, &r)
                            && match &r {
                                Record::History(x) => x.unix > 0,
                                Record::Ppg(x) => x.unix > 0,
                                Record::Imu(x) => x.unix > 0,
                                Record::Optical(x) => x.unix > 0,
                            }
                    }
                    None => false,
                };
                if ok {
                    t.hist_ok += 1;
                    slot.1 += 1;
                } else if t.mismatches.len() < 20 {
                    t.mismatches.push(format!("HIST v{version} did not route to its record"));
                }
            }
            "EVENT" => {
                t.ev_total += 1;
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
                    t.ev_ok += 1;
                } else if t.mismatches.len() < 20 {
                    t.mismatches.push(format!("EVENT num_ok={num_ok} ts_ok={ts_ok} hex_ok={hex_ok}"));
                }
            }
            "CONSOLE_LOGS" => {
                t.con_total += 1;
                match p["console"].as_str() {
                    Some(w) => {
                        if console::text(&frame).as_deref() == Some(w) {
                            t.con_ok += 1;
                        } else if t.mismatches.len() < 20 {
                            t.mismatches
                                .push(format!("CONSOLE rust={:?} want={w:?}", console::text(&frame)));
                        }
                    }
                    None => t.con_ok += 1,
                }
            }
            "COMMAND_RESPONSE" => {
                // resp_cmd raw == frame cmd byte (the label is built app-side from this raw). Frames
                // the capture left unparsed carry no expected value and cannot be compared.
                match p["resp_cmd"].as_str().and_then(paren_raw) {
                    Some(want_cmd) => {
                        t.cr_total += 1;
                        if want_cmd == frame.cmd() {
                            t.cr_cmd_ok += 1;
                        } else if t.mismatches.len() < 20 {
                            t.mismatches
                                .push(format!("CMD_RESP rust={} want={want_cmd}", frame.cmd()));
                        }
                    }
                    None => t.cr_no_expected += 1,
                }
                let _ = response::decode(&frame);
            }
            _ => {}
        }
    }

    t
}

// ---------------------------------------------------------------------------------------------
// The sample capture: the shape a real one has (header, blank line, every frame class, all four
// record versions), built here because real captures may not be committed.
// ---------------------------------------------------------------------------------------------

const SAMPLE_UNIX: u32 = 1_784_000_000;

fn line(type_name: &str, frame: &[u8], parsed: Value) -> String {
    json!({ "type_name": type_name, "parsed": parsed, "hex": to_hex(frame) }).to_string()
}

fn metadata_line(label: &str, cmd: u8, unix: u32, trim: Option<u32>) -> String {
    let mut payload = vec![0u8; 21];
    payload[0..4].copy_from_slice(&unix.to_le_bytes()); // unix @ inner 3
    payload[10..14].copy_from_slice(&trim.unwrap_or(0).to_le_bytes()); // trim @ inner 13
    let frame = framing::encode(Family::Gen5, 49, 1, cmd, &payload);
    let parsed = match trim {
        // The capture writer stores u32 as a signed int, so a high trim arrives negative.
        Some(t) => json!({ "meta_type": label, "unix": unix, "trim_cursor": t as i32 }),
        None => json!({ "meta_type": label }),
    };
    line("METADATA", &frame, parsed)
}

fn v18_line(unix: u32, hr: u8, rr: u16, steps: u16) -> String {
    let mut payload = vec![0u8; 110];
    payload[0..4].copy_from_slice(&(unix - SAMPLE_UNIX).to_le_bytes()); // record_index @ inner 3
    payload[4..8].copy_from_slice(&unix.to_le_bytes()); // unix @ inner 7
    payload[11] = hr; // hr @ inner 14
    payload[12] = 1; // rr_count @ inner 15
    payload[13..15].copy_from_slice(&rr.to_le_bytes()); // rr[0] @ inner 16
    payload[34..38].copy_from_slice(&0.0f32.to_le_bytes()); // gravity @ inner 37
    payload[38..42].copy_from_slice(&0.0f32.to_le_bytes());
    payload[42..46].copy_from_slice(&1.0f32.to_le_bytes());
    payload[46..48].copy_from_slice(&steps.to_le_bytes()); // steps @ inner 49
    payload[62..64].copy_from_slice(&3400u16.to_le_bytes()); // skin temp @ inner 65
    line("HISTORICAL_DATA", &framing::encode(Family::Gen5, 47, 18, 0x80, &payload), json!({}))
}

fn v26_line(unix: u32) -> String {
    let mut payload = vec![0u8; 70];
    payload[0..2].copy_from_slice(&4242u16.to_le_bytes()); // record_id @ inner 3
    payload[4..8].copy_from_slice(&unix.to_le_bytes()); // unix @ inner 7
    for i in 0..24 {
        let v = -700i16 + 30 * i as i16; // samples @ inner 19 + 2i
        payload[16 + i * 2..18 + i * 2].copy_from_slice(&v.to_le_bytes());
    }
    line("HISTORICAL_DATA", &framing::encode(Family::Gen5, 47, 26, 0x80, &payload), json!({}))
}

fn v20_line(unix: u32) -> String {
    let mut payload = vec![0u8; 2125];
    payload[4..8].copy_from_slice(&unix.to_le_bytes()); // unix @ inner 7
    payload[17..19].copy_from_slice(&1400u16.to_le_bytes()); // green LED @ inner 20
    payload[20..22].copy_from_slice(&2800u16.to_le_bytes()); // 2×green echo @ inner 23
    for (ch, base) in [36usize, 236, 1302, 1502, 1724, 1924].into_iter().enumerate() {
        for s in 0..25 {
            let v = ((ch as i32 + 1) * 10_000 + s as i32) as u32 & 0x000F_FFFF;
            payload[base + s * 4..base + s * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }
    }
    line("HISTORICAL_DATA", &framing::encode(Family::Gen5, 47, 20, 0x80, &payload), json!({}))
}

fn v21_line(unix: u32) -> String {
    let mut payload = vec![0u8; 1229];
    payload[4..8].copy_from_slice(&unix.to_le_bytes()); // unix @ inner 7
    payload[13..15].copy_from_slice(&100u16.to_le_bytes()); // count_a @ inner 16
    payload[619..621].copy_from_slice(&100u16.to_le_bytes()); // count_b @ inner 622
    for (axis, base) in [17usize, 217, 417, 629, 829, 1029].into_iter().enumerate() {
        for i in 0..100 {
            let v = (axis as i16 + 1) * 500 + i as i16;
            payload[base + i * 2..base + i * 2 + 2].copy_from_slice(&v.to_le_bytes());
        }
    }
    line("HISTORICAL_DATA", &framing::encode(Family::Gen5, 47, 21, 0x80, &payload), json!({}))
}

fn event_line(number: u8, unix: u32, tail: [u8; 8]) -> String {
    let mut payload = vec![0u8; 13]; // inner 16 = a 4-byte multiple, so encode adds no padding
    payload[1..5].copy_from_slice(&unix.to_le_bytes()); // timestamp @ inner 4
    payload[5..13].copy_from_slice(&tail); // the payload hex the app records is inner[8..]
    let frame = framing::encode(Family::Gen5, 48, 1, number, &payload);
    let parsed = json!({
        "event": format!("0x{number:02X}({number})"),
        "event_timestamp": unix,
        "event_payload_hex": to_hex(&tail),
    });
    line("EVENT", &frame, parsed)
}

fn console_line(text: &str) -> String {
    let mut payload = vec![0u8; 10]; // the record header the text decoder skips
    payload.extend_from_slice(text.as_bytes());
    let frame = framing::encode(Family::Gen5, 50, 1, 0, &payload);
    line("CONSOLE_LOGS", &frame, json!({ "console": text }))
}

fn cmd_resp_line(label: Option<&str>, cmd: u8) -> String {
    let frame = framing::encode(Family::Gen5, 36, 1, cmd, &[0u8; 8]);
    let parsed = match label {
        Some(l) => json!({ "resp_cmd": format!("{l}({cmd})"), "result": "SUCCESS(1)" }),
        None => json!({}),
    };
    line("COMMAND_RESPONSE", &frame, parsed)
}

/// A capture with the real writer's shape: a three-line `#` header, a blank line mid-file, every
/// frame class, both HISTORY_END forms (a plain trim and the one that arrives signed-negative), and
/// one COMMAND_RESPONSE the writer left unparsed.
fn sample_capture() -> String {
    let mut out = String::new();
    out.push_str("# NOOP 5/MG raw backfill capture (JSONL; one frame per line)\n");
    out.push_str("# App: build id, android version, phone model\n");
    out.push_str("# NOTE: contains raw biometric frames and the strap's console text.\n");

    let rows = [
        cmd_resp_line(None, 34),
        cmd_resp_line(Some("GET_DATA_RANGE"), 34),
        cmd_resp_line(Some("SEND_HISTORICAL_DATA"), 22),
        metadata_line("HISTORY_START(1)", 1, SAMPLE_UNIX, Some(82)),
        console_line("hist transfer start\n"),
        console_line("start response ack, start burst\n"),
        v18_line(SAMPLE_UNIX + 1, 70, 873, 57_932),
        v18_line(SAMPLE_UNIX + 2, 71, 834, 57_933),
        v18_line(SAMPLE_UNIX + 3, 72, 799, 57_934),
        v18_line(SAMPLE_UNIX + 4, 73, 759, 57_935),
        metadata_line("HISTORY_END(2)", 2, SAMPLE_UNIX, Some(54_587)),
        metadata_line("HISTORY_COMPLETE(3)", 3, SAMPLE_UNIX, None),
        // A trim above i32::MAX: the writer stores it as -1 and the harness has to widen it back.
        metadata_line("HISTORY_END(2)", 2, SAMPLE_UNIX + 1, Some(u32::MAX)),
        v26_line(SAMPLE_UNIX + 5),
        v20_line(SAMPLE_UNIX + 6),
        v21_line(SAMPLE_UNIX + 7),
        event_line(0x1D, SAMPLE_UNIX + 8, [1, 2, 3, 4, 5, 6, 7, 8]),
        event_line(0x6E, SAMPLE_UNIX + 9, [9, 10, 11, 12, 13, 14, 15, 16]),
    ];
    for (i, r) in rows.iter().enumerate() {
        if i == 6 {
            out.push('\n'); // the writer emits a blank line on a session break
        }
        out.push_str(r);
        out.push('\n');
    }
    out
}

fn sample() -> Tally {
    tally(Cursor::new(sample_capture()))
}

/// The sample reproduces with every denominator named. This is the whole-capture gate's success
/// test, and unlike that gate it runs from a clean checkout.
#[test]
fn sample_capture_reproduces_every_recorded_field() {
    let t = sample();
    eprintln!("{}", t.report());
    assert_parity(&t);

    assert_eq!(t.lines, 18, "sample frame count");
    assert_eq!((t.end_ok, t.end_total), (2, 2), "HISTORY_END frames");
    assert_eq!((t.meta_type_ok, t.meta_all), (4, 4), "METADATA frames");
    assert_eq!((t.hist_ok, t.hist_total), (7, 7), "HISTORICAL_DATA frames");
    assert_eq!((t.ev_ok, t.ev_total), (2, 2), "EVENT frames");
    assert_eq!((t.con_ok, t.con_total), (2, 2), "CONSOLE_LOGS frames");
    assert_eq!((t.cr_cmd_ok, t.cr_total), (2, 2), "COMMAND_RESPONSE frames with an expected value");
    assert_eq!(t.cr_no_expected, 1, "the unparsed COMMAND_RESPONSE frame");
    assert_eq!(t.decode_err, 0);
    let vers: Vec<(u8, u64)> = t.by_version.iter().map(|(v, (n, _))| (*v, *n)).collect();
    assert_eq!(vers, vec![(18, 4), (20, 1), (21, 1), (26, 1)], "all four 5.0 record versions");
}

/// The `#` header skip is load-bearing: line 1 of a capture is not JSON, so without the skip this
/// harness panics before it reads a single frame.
#[test]
fn the_hash_provenance_header_is_skipped_and_would_otherwise_panic() {
    let text = sample_capture();
    let header: Vec<&str> = text.lines().take_while(|l| l.starts_with('#')).collect();
    assert_eq!(header.len(), 3, "the sample carries a three-line header");
    for h in &header {
        assert!(serde_json::from_str::<Value>(h).is_err(), "header line parses as JSON: {h}");
    }
    assert!(text.lines().any(|l| l.trim().is_empty()), "the sample carries a blank line too");
    assert_eq!(sample().lines, 18, "only the data lines are counted");
}

/// Each version byte routes to its own record and the decoded values come back. Field parity against
/// real hardware lives in the v18 real-frame test in `records::gen5` and in `whole_capture_parity`.
#[test]
fn sample_capture_routes_every_version_to_its_own_record() {
    let mut seen = 0;
    for l in sample_capture().lines().filter(|l| !l.trim().is_empty() && !l.starts_with('#')) {
        let o: Value = serde_json::from_str(l).unwrap();
        if o["type_name"].as_str() != Some("HISTORICAL_DATA") {
            continue;
        }
        let raw = whoop_protocol::bytes::from_hex(o["hex"].as_str().unwrap()).unwrap();
        let frame = framing::decode(Family::Gen5, &raw).unwrap();
        match records::decode(&frame).expect("every sample record decodes") {
            Record::History(h) if h.unix == SAMPLE_UNIX + 1 => {
                assert_eq!(h.version, 18);
                assert_eq!(h.heart_rate, Some(70));
                assert_eq!(h.rr_intervals, vec![873]);
                assert_eq!(h.steps, Some(57_932));
                assert_eq!(h.record_index, Some(1));
                seen += 1;
            }
            Record::Ppg(p) => {
                assert_eq!(p.unix, SAMPLE_UNIX + 5);
                assert_eq!(p.record_id, Some(4242));
                assert_eq!(p.samples.len(), 24);
                assert_eq!(&p.samples[..3], &[-700, -670, -640]);
                seen += 1;
            }
            Record::Optical(x) => {
                assert_eq!(x.unix, SAMPLE_UNIX + 6);
                assert_eq!(x.sample_rate_hz, 25);
                let firsts: Vec<i32> = x.channels.iter().map(|c| c[0]).collect();
                assert_eq!(firsts, vec![10_000, 20_000, 30_000, 40_000, 50_000, 60_000]);
                assert_eq!(x.channels[5][24], 60_024);
                seen += 1;
            }
            Record::Imu(i) => {
                assert_eq!(i.unix, SAMPLE_UNIX + 7);
                assert_eq!(i.sample_rate_hz, 100);
                assert_eq!(i.accel[0], [500, 1000, 1500]);
                assert_eq!(i.gyro[99], [2099, 2599, 3099]);
                seen += 1;
            }
            Record::History(_) => {} // the three later v18 seconds
        }
    }
    assert_eq!(seen, 4, "one pinned frame per record version");
}

/// Drop a frame class and the tally must report the missing denominator instead of passing on
/// `0 == 0`. The HISTORY_END arm is the data-loss null: a capture with no end frame used to satisfy
/// the trim-cursor claim outright, which is how this gate stayed green on a capture that has none.
#[test]
fn a_capture_missing_a_frame_class_fails_instead_of_passing_vacuously() {
    let text = sample_capture();
    let drop_if = |pred: &dyn Fn(&str) -> bool| -> Vec<String> {
        let kept: String = text.lines().filter(|l| !pred(l)).map(|l| format!("{l}\n")).collect();
        tally(Cursor::new(kept)).faults()
    };

    let header_only = drop_if(&|l: &str| !l.starts_with('#'));
    assert!(header_only.iter().any(|f| f.contains("no frames at all")), "{header_only:?}");
    assert_eq!(header_only.len(), 8, "an empty capture violates every denominator: {header_only:?}");

    let no_end = drop_if(&|l: &str| l.contains("HISTORY_END"));
    assert!(
        no_end.iter().any(|f| f.contains("no HISTORY_END frames")),
        "a capture with no HISTORY_END must not satisfy the trim-cursor claim: {no_end:?}"
    );

    let no_hist = drop_if(&|l: &str| l.contains("HISTORICAL_DATA"));
    assert!(no_hist.iter().any(|f| f.contains("no HISTORICAL_DATA frames")), "{no_hist:?}");

    let no_cr = drop_if(&|l: &str| l.contains("resp_cmd"));
    assert!(no_cr.iter().any(|f| f.contains("expected resp_cmd")), "{no_cr:?}");
}

/// Flip one byte of the HISTORY_END trim cursor on the wire and the gate has to catch it. This is
/// the decoder-side null: the recorded expectations are untouched, only what Rust reads changes.
#[test]
fn a_one_byte_shift_in_the_wire_trim_cursor_fails_the_gate() {
    // trim_cursor is inner 13, and the GEN5 inner starts at frame byte 8.
    const TRIM_BYTE: usize = 21;
    let mutated: String = sample_capture()
        .lines()
        .map(|l| {
            if !l.contains("HISTORY_END") {
                return format!("{l}\n");
            }
            let key = "\"hex\":\"";
            let start = l.find(key).unwrap() + key.len();
            let end = start + l[start..].find('"').unwrap();
            let mut raw = whoop_protocol::bytes::from_hex(&l[start..end]).unwrap();
            raw[TRIM_BYTE] ^= 0x01;
            format!("{}{}{}\n", &l[..start], to_hex(&raw), &l[end..])
        })
        .collect();

    let t = tally(Cursor::new(mutated));
    assert_eq!(t.end_total, 2, "both HISTORY_END frames still present");
    assert_eq!(t.end_ok, 0, "neither may still match");
    assert!(
        t.faults().iter().any(|f| f.contains("HISTORY_END trim_cursor/unix parity")),
        "{:?}",
        t.faults()
    );
}

/// Each recorded expectation is compared, not merely read: corrupt one and its own claim fails.
#[test]
fn corrupting_any_single_recorded_expectation_fails_its_own_claim() {
    let text = sample_capture();
    let cases: [(&str, &str, &str); 4] = [
        ("\"trim_cursor\":54587", "\"trim_cursor\":54588", "HISTORY_END trim_cursor/unix parity"),
        ("\"unix\":1784000001", "\"unix\":1784000099", "HISTORY_END trim_cursor/unix parity"),
        ("HISTORY_END(2)", "HISTORY_END(9)", "metadata meta_type parity"),
        ("GET_DATA_RANGE(34)", "GET_DATA_RANGE(35)", "COMMAND_RESPONSE resp_cmd parity"),
    ];
    for (from, to, want) in cases {
        assert!(text.contains(from), "the sample no longer contains {from}");
        let faults = tally(Cursor::new(text.replace(from, to))).faults();
        assert!(faults.iter().any(|f| f.contains(want)), "{from} -> {to} not caught: {faults:?}");
    }
}

/// A capture whose frames cannot be framed at all must fail, not report a clean zero.
#[test]
fn undecodable_frames_fail_the_gate() {
    let mutated = sample_capture().replace("\"hex\":\"aa", "\"hex\":\"bb");
    let faults = tally(Cursor::new(mutated)).faults();
    assert!(faults.iter().any(|f| f.contains("failed to decode at all")), "{faults:?}");
}

/// The same tally over a whole capture off disk. Ignored because captures hold personal biometric
/// frames and are git-ignored by policy: point `WHOOP_CAPTURE` at one .jsonl and run `--ignored`.
#[test]
#[ignore = "needs an out-of-repo capture: WHOOP_CAPTURE=<path>.jsonl, run with --ignored"]
fn whole_capture_parity() {
    let path = std::env::var("WHOOP_CAPTURE").expect("WHOOP_CAPTURE must name a capture .jsonl");
    let file = std::fs::File::open(&path).expect("open capture");
    let t = tally(BufReader::new(file));
    eprintln!("{}", t.report());
    assert_parity(&t);
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
