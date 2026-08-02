//! Data-driven regression over real captured type-47 frames: decode each fixture frame through the public
//! API and assert only the fields its `expect` pins. Frames live in `fixtures/real_frames.json`.
//!
//! Each gate here carries its own null arm, because a checksum, a burst and an axis triplet are all
//! reproducible by a decoder that does no work: a checksum hardwired true, three frames sold as forty,
//! a constant unit vector. The null arm is what separates this from a reproduction check.

use serde_json::Value;
use whoop_protocol::bytes::from_hex;
use whoop_protocol::crc::crc32_zlib;
use whoop_protocol::family::Family;
use whoop_protocol::records::{decode, HistoryRecord, Record};
use whoop_protocol::{framing, live};

fn oracle() -> Value {
    serde_json::from_str(include_str!("fixtures/real_frames.json")).unwrap()
}

fn family_of(f: &Value) -> Family {
    match f["family"].as_str() {
        Some("gen4") => Family::Gen4,
        Some("gen5") | None => Family::Gen5, // ppg_frames / event_frames are 5.0/MG only
        Some(other) => panic!("unknown family {other}"),
    }
}

fn wire_of(f: &Value) -> Vec<u8> {
    from_hex(f["hex"].as_str().unwrap()).unwrap()
}

fn gen5_frame(f: &Value) -> whoop_protocol::packet::Frame {
    framing::decode(Family::Gen5, &wire_of(f)).unwrap()
}

/// Every fixture frame as `(cohort, name, family, wire)`. `frames` name themselves; the ppg burst is
/// numbered by position.
fn all_frames(o: &Value) -> Vec<(&'static str, String, Family, Vec<u8>)> {
    let mut out = Vec::new();
    for (cohort, key) in [("history", "frames"), ("ppg", "ppg_frames"), ("event", "event_frames")] {
        for (i, f) in o[key].as_array().unwrap().iter().enumerate() {
            let name = f["name"].as_str().map_or_else(|| format!("{cohort}[{i}]"), str::to_owned);
            out.push((cohort, name, family_of(f), wire_of(f)));
        }
    }
    out
}

#[test]
fn real_frames_decode_to_pinned_values() {
    let oracle = oracle();
    let frames = oracle["frames"].as_array().unwrap();
    assert!(frames.len() >= 9, "fixture shrank unexpectedly");

    // Every `expect` key below is asserted only when present, so a deleted key deletes its assert
    // WITHOUT changing the test count. Fields whose coverage is load-bearing are therefore also
    // counted across the cohort and checked after the loop.
    let mut activity_codes: Vec<u8> = Vec::new();
    let mut gravity_pinned: Vec<&str> = Vec::new();

    for f in frames {
        let name = f["name"].as_str().unwrap();
        let family = family_of(f);
        let wire = wire_of(f);
        // A bad checksum is reported on `Frame::crc_ok`, never as an Err, so this only catches a
        // structurally broken frame. The checksums have their own gate below.
        let frame = framing::decode(family, &wire)
            .unwrap_or_else(|e| panic!("{name}: frame structurally invalid (not a checksum failure): {e:?}"));
        let r = match decode(&frame) {
            Some(Record::History(h)) => h,
            other => panic!("{name}: expected a History record, got {other:?}"),
        };

        let e = &f["expect"];
        let u8_of = |k: &str| e[k].as_u64().unwrap() as u8;
        let u16_of = |k: &str| e[k].as_u64().unwrap() as u16;

        assert_eq!(r.version, u8_of("version"), "{name}: version");
        assert_eq!(u64::from(r.unix), e["unix"].as_u64().unwrap(), "{name}: unix");
        if e.get("heart_rate").is_some() {
            assert_eq!(r.heart_rate, Some(u8_of("heart_rate")), "{name}: heart_rate");
        }
        if let Some(rr) = e.get("rr").and_then(Value::as_array) {
            let want: Vec<u16> = rr.iter().map(|x| x.as_u64().unwrap() as u16).collect();
            assert_eq!(r.rr_intervals, want, "{name}: rr");
        }
        if e.get("gravity_unit").and_then(Value::as_bool) == Some(true) {
            let g = r.gravity.unwrap_or_else(|| panic!("{name}: gravity absent"));
            let mag = (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]).sqrt();
            assert!((0.9..1.1).contains(&mag), "{name}: |g| = {mag}");
        }
        // Per-axis pin. The magnitude band above is satisfied by any unit vector, so only this
        // rejects a decoder that reads the wrong offset or applies the wrong scale.
        if let Some(gv) = e.get("gravity").and_then(Value::as_array) {
            let g = r.gravity.unwrap_or_else(|| panic!("{name}: gravity absent"));
            for (i, want) in gv.iter().enumerate() {
                assert!((f64::from(g[i]) - want.as_f64().unwrap()).abs() < 1e-6, "{name}: gravity[{i}]");
            }
            gravity_pinned.push(name);
        }
        if e.get("skin_temp_raw").is_some() {
            assert_eq!(r.skin_temp_raw, Some(u16_of("skin_temp_raw")), "{name}: skin_temp_raw");
        }
        // The coarse activity label. Code 0 alone does not pin the offset (52 of the worn frame's 124
        // bytes are also 0); all three codes together leave exactly one offset, inner 55.
        if e.get("activity_class").is_some() {
            let want = u8_of("activity_class");
            assert_eq!(r.activity_class, Some(want), "{name}: activity_class");
            activity_codes.push(want);
        }
        if e.get("steps").is_some() {
            assert_eq!(r.steps, Some(u16_of("steps")), "{name}: steps");
        }
        if e.get("sleep_state").is_some() {
            assert_eq!(r.sleep_state, Some(u8_of("sleep_state")), "{name}: sleep_state");
        }
        if let Some(sp) = e.get("spo2").and_then(Value::as_array) {
            let (red, ir) = (sp[0].as_u64().unwrap() as u16, sp[1].as_u64().unwrap() as u16);
            assert_eq!(r.spo2, Some((red, ir)), "{name}: spo2");
        }
    }

    // Cohort, not per-frame: dropping an `activity_class` key would otherwise delete its assert
    // silently. All three codes must stay pinned or the offset is no longer pinned at all.
    activity_codes.sort_unstable();
    activity_codes.dedup();
    assert_eq!(activity_codes, vec![0, 1, 2], "activity_class codes pinned by the fixture cohort");

    // Same rule for the accelerometer: 4.0 gravity is the generation's ONLY motion channel, and both
    // its wire scales (v24 f32, v25 i16/16384) must stay pinned per axis.
    gravity_pinned.sort_unstable();
    assert_eq!(
        gravity_pinned,
        vec![
            "v18_real_whoop5_worn",
            "v24_real_whoop4_worn",
            "v25_real_whoop4_a",
            "v25_real_whoop4_b",
            "v25_real_whoop4_c",
        ],
        "frames carrying a per-axis gravity pin"
    );
}

/// Every real frame in the fixture is checksum-valid, and a single flipped bit in either the header
/// checksum field or the inner body flips `crc_ok` to false. Covers all three functions: crc8 (4.0
/// header), crc16-modbus (5.0/MG header) and crc32-zlib (the inner body of both).
///
/// The null this replaces: `framing::decode` returns `Ok` on a checksum MISMATCH and reports it on
/// `Frame::crc_ok`, so unwrapping the decode checks no checksum at all.
#[test]
fn every_real_frame_is_checksum_valid_and_one_flipped_bit_breaks_header_or_body() {
    let oracle = oracle();
    let all = all_frames(&oracle);
    let (mut gen4, mut gen5) = (0usize, 0usize);

    for (cohort, name, family, wire) in &all {
        let f = framing::decode(*family, wire).unwrap_or_else(|e| panic!("{cohort}/{name}: {e:?}"));
        assert!(f.crc_ok, "{cohort}/{name}: real frame failed its checksums");
        match family {
            Family::Gen4 => gen4 += 1,
            Family::Gen5 => gen5 += 1,
        }

        let h = family.header();
        let inner_len = u16::from_le_bytes([wire[h.len_offset], wire[h.len_offset + 1]]) as usize - 4;

        // Header checksum only: flip a bit inside the stored header CRC field, body untouched.
        let hdr_crc_at = match family {
            Family::Gen4 => 3, // crc8 over the two length bytes
            Family::Gen5 => 6, // crc16-modbus LE over the 6-byte header
        };
        let mut bad = wire.clone();
        bad[hdr_crc_at] ^= 0x01;
        let d = framing::decode(*family, &bad).unwrap_or_else(|e| panic!("{cohort}/{name}: header arm {e:?}"));
        assert!(!d.crc_ok, "{cohort}/{name}: a flipped header-checksum bit was not detected");

        // Body checksum only: flip a bit in the middle of the inner record, header untouched.
        let mut bad = wire.clone();
        bad[h.inner_start + inner_len / 2] ^= 0x01;
        let d = framing::decode(*family, &bad).unwrap_or_else(|e| panic!("{cohort}/{name}: body arm {e:?}"));
        assert!(!d.crc_ok, "{cohort}/{name}: a flipped inner-body bit was not detected by crc32");

        // And the body checksum is a real function of the body, not a stored-equals-stored tautology.
        assert_ne!(
            crc32_zlib(&wire[h.inner_start..h.inner_start + inner_len]),
            crc32_zlib(&bad[h.inner_start..h.inner_start + inner_len]),
            "{cohort}/{name}: crc32 did not move on a flipped body bit"
        );
    }

    // Both header checksums must stay covered: dropping the 4.0 frames would leave crc8 unexercised.
    assert_eq!(gen4, 4, "4.0 frames exercising crc8 + crc32");
    assert_eq!(gen5, 47, "5.0/MG frames exercising crc16-modbus + crc32");
    assert_eq!(all.len(), 51, "real frames under checksum coverage");
}

/// The v26 optical burst is forty CONSECUTIVE strap-seconds: forty frames, forty distinct unix
/// seconds spanning 39 s, forty consecutive record ids, 24 samples each = 24 Hz.
///
/// The null this replaces: `frames.len() >= 3` called a 40-second burst proven by three frames, and
/// no assert rejected forty copies of one second.
#[test]
fn real_v26_ppg_burst_is_40_consecutive_strap_seconds_at_24_hz() {
    let oracle = oracle();
    let frames = oracle["ppg_frames"].as_array().unwrap();
    assert_eq!(frames.len(), 40, "the v26 burst is 40 strap-seconds; a shorter cohort is not this burst");

    let mut last_rid: Option<u16> = None;
    let mut unix_seen: Vec<u32> = Vec::new();
    let mut distinct_samples: std::collections::BTreeSet<Vec<i16>> = std::collections::BTreeSet::new();
    let mut total_samples = 0usize;

    for f in frames {
        let frame = gen5_frame(f);
        let p = match decode(&frame) {
            Some(Record::Ppg(p)) => p,
            other => panic!("v26: expected a Ppg record, got {other:?}"),
        };
        assert_eq!(p.version, 26, "v26: version");
        assert_eq!(u64::from(p.unix), f["unix"].as_u64().unwrap(), "v26: unix");
        let rid = f["record_id"].as_u64().unwrap() as u16;
        assert_eq!(p.record_id, Some(rid), "v26: record_id");
        assert_eq!(p.samples.len(), 24, "v26: sample count");
        // record_id is a monotonic strap counter across the consecutive burst.
        if let Some(prev) = last_rid {
            assert_eq!(rid, prev.wrapping_add(1), "v26: record_id not consecutive");
        }
        last_rid = Some(rid);
        unix_seen.push(p.unix);
        total_samples += p.samples.len();
        distinct_samples.insert(p.samples.clone());
        // Exact optical samples pinned on the first frame only (24 i16 LE from the wire).
        if let Some(s) = f.get("samples").and_then(Value::as_array) {
            let want: Vec<i16> = s.iter().map(|x| x.as_i64().unwrap() as i16).collect();
            assert_eq!(p.samples, want, "v26: samples");
        }
    }

    // Consecutive seconds, no gap and no repeat: the burst covers a real 40-second window.
    for (i, ts) in unix_seen.iter().enumerate() {
        assert_eq!(*ts, unix_seen[0] + i as u32, "v26: second {i} is not consecutive");
    }
    assert_eq!(unix_seen.last().unwrap() - unix_seen[0], 39, "v26: burst span in seconds");
    assert_eq!(total_samples, 960, "v26: 40 s x 24 Hz");
    // Null arm: a decoder returning a constant (or one frame's) waveform yields one distinct vector.
    assert_eq!(distinct_samples.len(), 40, "v26: every second must carry its own waveform");
}

/// Both real EVENT kinds decode: the numbered event plus its timestamp on each, and the battery body
/// (soc, millivolts, charging) on the one frame that carries one.
///
/// The null this replaces: no cohort assert, so deleting the battery frame left the gate green while
/// `battery_event` went untested; and nothing checked that a non-battery event returns `None`.
#[test]
fn real_event_frames_pin_both_event_kinds_and_only_one_carries_a_battery_body() {
    let oracle = oracle();
    let frames = oracle["event_frames"].as_array().unwrap();
    assert_eq!(frames.len(), 2, "event cohort: battery_level + wrist_on");

    let (mut numbers, mut with_battery) = (Vec::new(), 0usize);
    for f in frames {
        let name = f["name"].as_str().unwrap();
        let frame = gen5_frame(f);
        let ev = live::event(&frame).unwrap_or_else(|| panic!("{name}: event absent"));
        assert_eq!(ev.number, f["number"].as_u64().unwrap() as u8, "{name}: number");
        assert_eq!(u64::from(ev.timestamp), f["timestamp"].as_u64().unwrap(), "{name}: timestamp");
        numbers.push(ev.number);
        match f.get("battery") {
            Some(b) => {
                with_battery += 1;
                let be = live::battery_event(&frame).unwrap_or_else(|| panic!("{name}: battery absent"));
                assert!((be.soc_deci as f64 / 10.0 - b["soc_percent"].as_f64().unwrap()).abs() < 1e-3, "{name}: soc");
                assert_eq!(u64::from(be.millivolts), b["millivolts"].as_u64().unwrap(), "{name}: mv");
                assert_eq!(be.charging, b["charging"].as_bool().unwrap(), "{name}: charging");
            }
            // Null arm: a parser that reports a battery on every event frame fails here.
            None => assert!(
                live::battery_event(&frame).is_none(),
                "{name}: a non-battery event must not yield a battery body"
            ),
        }
    }

    numbers.sort_unstable();
    assert_eq!(numbers, vec![3, 9], "event numbers pinned by the cohort (battery_level, wrist_on)");
    assert_eq!(with_battery, 1, "exactly one fixture event carries a battery body");
}

/// The strict gate that lets an UNMAPPED 4.0 version through the v24 layout: HR in 25..=230 and
/// |g| in 0.8..1.2, both required. Nothing else in the tree reaches it.
///
/// Each arm is the real captured v24 frame with one field rewritten and the CRC32 resealed, so the
/// only thing that varies is the value the gate reads.
#[test]
fn unmapped_gen4_version_passes_the_fallback_only_on_a_plausible_hr_and_gravity() {
    let base = v24_wire();
    let real_g = [-0.403_115_24_f32, 0.450_590_82, 0.872_478];

    // Success: an unmapped version decodes through the v24 layout with the real frame's values.
    for version in [13u8, 99] {
        let h = decode_gen4(&reseal(with_version(&base, version)))
            .unwrap_or_else(|| panic!("v{version}: the fallback rejected a plausible real frame"));
        assert_eq!(h.version, version, "v{version}: version passthrough");
        assert_eq!(h.heart_rate, Some(109), "v{version}: heart_rate");
        assert_eq!(h.gravity, Some(real_g), "v{version}: gravity");
    }
    // Control: a MAPPED version that carries no gravity is not the fallback path.
    assert_eq!(decode_gen4(&reseal(with_version(&base, 5))).unwrap().gravity, None, "v5 carries no gravity");

    // Failure, heart rate: the band edges pass, one step outside either edge is rejected.
    for (hr, want) in [(0u8, false), (24, false), (25, true), (230, true), (231, false), (240, false)] {
        let mut w = with_version(&base, 13);
        w[4 + 17] = hr;
        assert_eq!(decode_gen4(&reseal(w)).is_some(), want, "fallback with hr={hr}");
    }

    // Failure, gravity: |g| = 0.6 is inside `accept_gravity` (0.5..1.5) so the MAPPED v24 still
    // decodes it, and only the fallback's tighter 0.8..1.2 rejects it. That separates the two gates.
    let shrunk = scaled_gravity(&base, 0.6 / 1.061_486);
    assert!(decode_gen4(&reseal(with_version(&shrunk, 24))).unwrap().gravity.is_some(), "mapped v24 keeps |g|=0.6");
    assert!(decode_gen4(&reseal(with_version(&shrunk, 13))).is_none(), "fallback must reject |g|=0.6");
    assert!(decode_gen4(&reseal(with_version(&scaled_gravity(&base, 0.9), 13))).is_some(), "|g|=0.955 passes");

    // Null arm for the per-axis pins: a decoder emitting a constant unit vector satisfies every
    // magnitude band in the tree, and is rejected only by the exact triplet.
    let constant = decode_gen4(&reseal(constant_gravity(&base, [0.0, 0.0, 1.0]))).unwrap().gravity.unwrap();
    let mag = (constant[0] * constant[0] + constant[1] * constant[1] + constant[2] * constant[2]).sqrt();
    assert!((0.9..1.1).contains(&mag), "the constant null passes the magnitude band");
    assert_ne!(constant, real_g, "the per-axis pin rejects the constant null");
}

fn v24_wire() -> Vec<u8> {
    let o = oracle();
    let f = o["frames"]
        .as_array()
        .unwrap()
        .iter()
        .find(|f| f["name"] == "v24_real_whoop4_worn")
        .expect("fixture v24_real_whoop4_worn missing")
        .clone();
    wire_of(&f)
}

/// 4.0 inner starts at byte 4 and its length is the declared length minus the 4-byte CRC32 trailer.
fn gen4_inner(w: &[u8]) -> std::ops::Range<usize> {
    let decl = u16::from_le_bytes([w[1], w[2]]) as usize;
    4..4 + decl - 4
}

/// Recompute the inner CRC32 so a rewritten frame is checksum-valid, not merely well-shaped.
fn reseal(mut w: Vec<u8>) -> Vec<u8> {
    let r = gen4_inner(&w);
    let end = r.end;
    let c = crc32_zlib(&w[r]);
    w[end..end + 4].copy_from_slice(&c.to_le_bytes());
    w
}

fn with_version(w: &[u8], version: u8) -> Vec<u8> {
    let mut w = w.to_vec();
    w[5] = version; // inner[1], the version byte
    w
}

fn scaled_gravity(w: &[u8], k: f32) -> Vec<u8> {
    let mut w = w.to_vec();
    for i in 0..3 {
        let at = 4 + 36 + i * 4;
        let v = f32::from_le_bytes(w[at..at + 4].try_into().unwrap()) * k;
        w[at..at + 4].copy_from_slice(&v.to_le_bytes());
    }
    w
}

fn constant_gravity(w: &[u8], g: [f32; 3]) -> Vec<u8> {
    let mut w = w.to_vec();
    for (i, v) in g.iter().enumerate() {
        let at = 4 + 36 + i * 4;
        w[at..at + 4].copy_from_slice(&v.to_le_bytes());
    }
    w
}

fn decode_gen4(w: &[u8]) -> Option<HistoryRecord> {
    let f = framing::decode(Family::Gen4, w).unwrap();
    assert!(f.crc_ok, "the rewritten frame must stay checksum-valid");
    match decode(&f) {
        Some(Record::History(h)) => Some(h),
        None => None,
        other => panic!("expected a History record, got {other:?}"),
    }
}

/// The v18 optical channels are two u8 baselines and two u8 amplitudes, not two u16s, and the 128 on
/// the amplitude pair is a record-level quality sentinel rather than a magnitude.
#[test]
fn v18_optical_channels_are_paired_u8s_with_a_record_level_sentinel() {
    let oracle = oracle();
    let by_name = |n: &str| {
        let f = oracle["frames"]
            .as_array()
            .unwrap()
            .iter()
            .find(|f| f["name"] == n)
            .unwrap_or_else(|| panic!("fixture {n} missing"));
        match decode(&gen5_frame(f)) {
            Some(Record::History(h)) => h,
            other => panic!("{n}: expected a History record, got {other:?}"),
        }
    };

    // Worn: both baselines present, both amplitudes real, quality good.
    let worn = by_name("v18_real_whoop5_worn");
    assert_eq!(worn.optical_baseline_a, Some(101));
    assert_eq!(worn.optical_baseline_b, Some(111));
    assert_eq!(worn.optical_amp_a, Some(30));
    assert_eq!(worn.optical_amp_b, Some(30));
    assert_eq!(worn.optical_signal_poor, Some(false));

    // Off the wrist the baselines read 0 TOGETHER — 0 is the off-wrist mark here, not 128.
    let off = by_name("v18_real_whoop5_offwrist");
    assert_eq!(off.optical_baseline_a, None);
    assert_eq!(off.optical_baseline_b, None);
    assert_eq!(off.optical_signal_poor, Some(true));

    // 128 on a BASELINE is an ordinary worn value, so it must not be mistaken for the sentinel: this
    // frame carries baseline_b = 128 with a real heart rate.
    let second = by_name("v18_real_whoop5_second_device");
    assert_eq!(second.optical_baseline_b, Some(128));
    assert_eq!(second.heart_rate, Some(57));
    // Its amplitudes DO carry the sentinel, so they are withheld rather than reported as 128.
    assert_eq!(second.optical_amp_a, None);
    assert_eq!(second.optical_amp_b, None);
    assert_eq!(second.optical_signal_poor, Some(true));
}
