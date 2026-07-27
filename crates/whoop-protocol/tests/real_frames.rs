//! Data-driven regression over real captured type-47 frames: decode each fixture frame through the public
//! API and assert only the fields its `expect` pins. Frames live in `fixtures/real_frames.json`.

use serde_json::Value;
use whoop_protocol::bytes::from_hex;
use whoop_protocol::family::Family;
use whoop_protocol::records::{decode, Record};
use whoop_protocol::{framing, live};

fn oracle() -> Value {
    serde_json::from_str(include_str!("fixtures/real_frames.json")).unwrap()
}

fn gen5_frame(f: &Value) -> whoop_protocol::packet::Frame {
    let wire = from_hex(f["hex"].as_str().unwrap()).unwrap();
    framing::decode(Family::Gen5, &wire).unwrap()
}

#[test]
fn real_frames_decode_to_pinned_values() {
    let oracle = oracle();
    let frames = oracle["frames"].as_array().unwrap();
    assert!(frames.len() >= 7, "fixture shrank unexpectedly");

    for f in frames {
        let name = f["name"].as_str().unwrap();
        let family = match f["family"].as_str().unwrap() {
            "gen4" => Family::Gen4,
            "gen5" => Family::Gen5,
            other => panic!("{name}: unknown family {other}"),
        };
        let wire = from_hex(f["hex"].as_str().unwrap()).unwrap();
        let frame = framing::decode(family, &wire).unwrap_or_else(|e| panic!("{name}: frame/CRC invalid: {e:?}"));
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
        // Exact f32->f64 widening: the decoded triplet must equal the wire f32s (not an i16/16384 scale).
        if let Some(gv) = e.get("gravity").and_then(Value::as_array) {
            let g = r.gravity.unwrap_or_else(|| panic!("{name}: gravity absent"));
            for (i, want) in gv.iter().enumerate() {
                assert!((f64::from(g[i]) - want.as_f64().unwrap()).abs() < 1e-6, "{name}: gravity[{i}]");
            }
        }
        if e.get("skin_temp_raw").is_some() {
            assert_eq!(r.skin_temp_raw, Some(u16_of("skin_temp_raw")), "{name}: skin_temp_raw");
        }
        if e.get("activity_class").is_some() {
            assert_eq!(r.activity_class, Some(u8_of("activity_class")), "{name}: activity_class");
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
}

#[test]
fn real_v26_ppg_frames_decode_to_pinned_values() {
    let oracle = oracle();
    let frames = oracle["ppg_frames"].as_array().unwrap();
    assert!(frames.len() >= 3, "ppg fixture shrank unexpectedly");

    let mut last_rid: Option<u16> = None;
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
        // Exact optical samples pinned on the first frame only (24 i16 LE from the wire).
        if let Some(s) = f.get("samples").and_then(Value::as_array) {
            let want: Vec<i16> = s.iter().map(|x| x.as_i64().unwrap() as i16).collect();
            assert_eq!(p.samples, want, "v26: samples");
        }
    }
}

#[test]
fn real_event_frames_decode_to_pinned_values() {
    let oracle = oracle();
    for f in oracle["event_frames"].as_array().unwrap() {
        let name = f["name"].as_str().unwrap();
        let frame = gen5_frame(f);
        let ev = live::event(&frame).unwrap_or_else(|| panic!("{name}: event absent"));
        assert_eq!(ev.number, f["number"].as_u64().unwrap() as u8, "{name}: number");
        assert_eq!(u64::from(ev.timestamp), f["timestamp"].as_u64().unwrap(), "{name}: timestamp");
        if let Some(b) = f.get("battery") {
            let be = live::battery_event(&frame).unwrap_or_else(|| panic!("{name}: battery absent"));
            assert!((be.soc_deci as f64 / 10.0 - b["soc_percent"].as_f64().unwrap()).abs() < 1e-3, "{name}: soc");
            assert_eq!(u64::from(be.millivolts), b["millivolts"].as_u64().unwrap(), "{name}: mv");
            assert_eq!(be.charging, b["charging"].as_bool().unwrap(), "{name}: charging");
        }
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
