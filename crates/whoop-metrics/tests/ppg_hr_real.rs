//! Golden PPG-HR over the real v26 optical bursts in the shared protocol fixture: decode the frames,
//! concatenate their samples, and pin the derived HR against the recorded estimate.

use serde_json::Value;
use whoop_metrics::ppg_hr::{estimate, Sample};
use whoop_protocol::bytes::from_hex;
use whoop_protocol::family::Family;
use whoop_protocol::framing;
use whoop_protocol::records::{decode, Record};

#[test]
fn real_v26_ppg_hr_matches_golden() {
    let oracle: Value =
        serde_json::from_str(include_str!("../../whoop-protocol/tests/fixtures/real_frames.json")).unwrap();
    let frames = oracle["ppg_frames"].as_array().unwrap();

    let mut samples = Vec::new();
    for f in frames {
        let wire = from_hex(f["hex"].as_str().unwrap()).unwrap();
        let frame = framing::decode(Family::Gen5, &wire).unwrap();
        let p = match decode(&frame) {
            Some(Record::Ppg(p)) => p,
            other => panic!("expected a Ppg record, got {other:?}"),
        };
        for v in p.samples {
            samples.push(Sample { ts: i64::from(p.unix), value: i32::from(v) });
        }
    }

    let est = estimate(&samples);
    let g = &oracle["ppg_hr"];
    assert_eq!(est.len(), g["estimate_count"].as_u64().unwrap() as usize, "estimate count");

    // The first estimate is the deterministic anchor (pure autocorrelation over the leading window).
    let first = est.first().expect("at least one estimate");
    assert_eq!(first.ts, g["first"]["ts"].as_i64().unwrap(), "first.ts");
    assert_eq!(first.bpm, g["first"]["bpm"].as_i64().unwrap() as i32, "first.bpm");
    assert!((first.conf - g["first"]["conf"].as_f64().unwrap()).abs() < 1e-9, "first.conf");

    // Every high-confidence estimate lands in a physiological resting-HR band.
    let range = g["confident_bpm_range"].as_array().unwrap();
    let (lo, hi) = (range[0].as_i64().unwrap() as i32, range[1].as_i64().unwrap() as i32);
    for e in est.iter().filter(|e| e.conf >= 0.7) {
        assert!((lo..=hi).contains(&e.bpm), "confident bpm {} out of [{lo},{hi}] at {}", e.bpm, e.ts);
    }
}
