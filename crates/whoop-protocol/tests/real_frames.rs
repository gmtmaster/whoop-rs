//! Data-driven regression over real captured type-47 frames: decode each fixture frame through the public
//! API and assert only the fields its `expect` pins. Frames live in `fixtures/real_frames.json`.

use serde_json::Value;
use whoop_protocol::bytes::from_hex;
use whoop_protocol::family::Family;
use whoop_protocol::framing;
use whoop_protocol::records::{decode, Record};

#[test]
fn real_frames_decode_to_pinned_values() {
    let oracle: Value = serde_json::from_str(include_str!("fixtures/real_frames.json")).unwrap();
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
