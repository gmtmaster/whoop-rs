//! Frozen-golden + contract tests for the sleep stagers. The golden hypnogram pins the whole V2 recipe
//! (features → emissions → Viterbi → tiling) stage-for-stage on a crafted integer-only night, so any drift
//! from the shipped recipe fails immediately. Integer-literal input keeps the two languages bit-identical.

use super::input::{AccelSample, HrSample, RrRun, SleepInput};
use super::params::Params;
use super::{analyze, stage_v2, SleepStage, SleepStreams, DEEP_GATE_THRESH};

const REF_MIDNIGHT: i64 = 1_749_513_600;

fn rsa_wave(ph: usize, i: i64) -> i64 {
    let amp = [12i64, 60, 30, 20][ph];
    [0, amp, 0, -amp][(i % 4) as usize]
}

/// The crafted 4-phase night (deep-favorable → high-RSA → mild → restless) used by the frozen golden.
fn golden_input() -> SleepInput {
    let start = REF_MIDNIGHT + 3_600;
    let phase: i64 = 90 * 60;
    let dur = phase * 4;
    let mut accel = Vec::new();
    let mut hr = Vec::new();
    let mut rr = Vec::new();
    for i in 0..dur {
        let ts = start + i;
        let ph = (i / phase) as usize;
        let restless = ph == 3 && (i % 20) < 6;
        if restless {
            accel.push(AccelSample { ts, x: 0.2, y: 0.15, z: 0.96 });
        } else {
            accel.push(AccelSample { ts, x: 0.0, y: 0.0, z: 1.0 });
        }
        let bpm: i64 = match ph {
            0 => 50,
            1 => 54 + [0, 1, 2, 3, 2, 1][((i / 20) % 6) as usize],
            2 => 56 + (i / 60) % 4,
            _ => 66 + (i / 30) % 6,
        };
        hr.push(HrSample { ts, bpm: bpm as u16 });
        let rr_ms = 60_000 / bpm + rsa_wave(ph, i);
        rr.push(RrRun { ts, intervals: vec![rr_ms as u16] });
    }
    SleepInput { start, end: start + dur, hr, rr, accel, resp: Vec::new() }
}

#[test]
fn frozen_golden_hypnogram_v2() {
    let input = golden_input();
    let start = input.start;
    let segs = stage_v2(&input);
    let golden = [
        (0i64, 5070i64, SleepStage::Deep),
        (5070, 5280, SleepStage::Light),
        (5280, 5550, SleepStage::Rem),
        (5550, 10740, SleepStage::Light),
        (10740, 16290, SleepStage::Rem),
        (16290, 21600, SleepStage::Wake),
    ];
    assert_eq!(golden.len(), segs.len(), "segment count");
    for (k, g) in golden.iter().enumerate() {
        assert_eq!(start + g.0, segs[k].start, "seg {k} start");
        assert_eq!(start + g.1, segs[k].end, "seg {k} end");
        assert_eq!(g.2, segs[k].stage, "seg {k} stage");
    }
}

#[test]
fn tuned_deep_boundary_constants_are_pinned() {
    assert_eq!(0.40, DEEP_GATE_THRESH);
    assert_eq!([0.76, 0.012, 0.216, 0.012], Params::SHIPPED.transition[0]);
    for row in Params::SHIPPED.transition.iter() {
        let sum: f64 = row.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9, "transition row must sum to 1.0");
    }
}

#[test]
fn v2_segments_tile_the_span_contiguously() {
    let input = golden_input();
    let segs = stage_v2(&input);
    assert!(!segs.is_empty());
    assert_eq!(input.start, segs.first().unwrap().start);
    assert_eq!(input.end, segs.last().unwrap().end);
    for w in segs.windows(2) {
        assert_eq!(w[0].end, w[1].start, "no gap/overlap");
        assert!(w[1].end > w[1].start, "non-empty");
    }
}

#[test]
fn v2_degenerate_input_falls_back_to_single_light_block() {
    let start = REF_MIDNIGHT;
    let end = start + 3_600;
    let input = SleepInput {
        start,
        end,
        hr: Vec::new(),
        rr: Vec::new(),
        accel: vec![AccelSample { ts: start, x: 0.0, y: 0.0, z: 1.0 }],
        resp: Vec::new(),
    };
    let segs = stage_v2(&input);
    assert_eq!(1, segs.len());
    assert_eq!(SleepStage::Light, segs[0].stage);
    assert_eq!(start, segs[0].start);
    assert_eq!(end, segs[0].end);
}

#[test]
fn analyze_still_night_stages_and_tiles_the_span() {
    let input = golden_input();
    let streams =
        SleepStreams { hr: input.hr.clone(), rr: input.rr.clone(), accel: input.accel.clone(), tz_offset_s: 0, ..Default::default() };
    let sessions = analyze(&streams);
    assert_eq!(1, sessions.len());
    let segs = &sessions[0].segments;
    assert_eq!(sessions[0].start, segs.first().unwrap().start);
    assert_eq!(sessions[0].end, segs.last().unwrap().end);
    for w in segs.windows(2) {
        assert_eq!(w[0].end, w[1].start);
    }
}

#[test]
fn analyze_detects_a_simple_still_night() {
    let start = REF_MIDNIGHT;
    let dur = 2 * 3600;
    let mut hr = Vec::new();
    let mut accel = Vec::new();
    for i in 0..dur {
        let ts = start + i;
        hr.push(HrSample { ts, bpm: 50 });
        accel.push(AccelSample { ts, x: 0.0, y: 0.0, z: 1.0 });
    }
    let streams = SleepStreams { hr, accel, tz_offset_s: 0, ..Default::default() };
    let sessions = analyze(&streams);
    assert_eq!(1, sessions.len());
    let s = &sessions[0];
    assert_eq!(start, s.start);
    assert!(s.end - s.start >= 3600);
    assert!((0.0..=1.0).contains(&s.efficiency));
    assert_eq!(s.start, s.segments.first().unwrap().start);
    assert_eq!(s.end, s.segments.last().unwrap().end);
}
