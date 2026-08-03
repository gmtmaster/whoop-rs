//! Negative controls for the `decode` metric family. Falsifies the claim that the shipped decode gates
//! are REGRESSION checks rather than mere REPRODUCTION checks.
//!
//! Every gate's exact target and tolerance is copied in below as a `const`, tagged with the `file:line`
//! it came from, so the arms are scored against the real claim and not a paraphrase. A parameterised
//! REPLICA of each decoder then runs against that same gate under three mutation families:
//!   NULL       — the decoder does no work (constant / zeroed output). The gate MUST fail.
//!   STRUCTURAL — magnitude kept, shape broken (axes reversed, fields swapped, series shifted).
//!   PARAMETER  — one tunable moved by +-10%, plus a +-0.5%-equivalent floor probe.
//! An arm the gate still PASSes is a blind spot in the gate, and that measurement is the deliverable.
//!
//! Reports only; nothing here changes shipped behaviour. Two things are asserted per metric: the
//! baseline reproduces the shipped figure, and at least one NULL arm FAILS (proof the harness reaches
//! the algorithm). A PARAMETER arm that passes is never asserted — measuring that it passes is the
//! finding. A single NULL arm that PASSes is printed as CRITICAL and left for a later phase.
//!
//! `#[ignore]`d because it is a report, NOT because its data is restricted: the fixture is tracked
//! beside it and `include_str!`ed, so it builds and runs from a clean checkout.
//! Run: cargo test --release -p whoop-protocol --test sensitivity_decode -- --ignored --nocapture

use serde_json::Value;

use whoop_protocol::advertising::{advertising_name_payload, advertising_name_payload_gen5};
use whoop_protocol::bytes::{f32_at, from_hex, i16_at, u16_at, u32_at, u8_at};
use whoop_protocol::command::GET_HELLO;
use whoop_protocol::crc::{crc16_modbus, crc32_zlib, crc8};
use whoop_protocol::family::Family;
use whoop_protocol::hello::GEN5_CLIENT_HELLO;
use whoop_protocol::packet::Frame;
use whoop_protocol::records::{decode as shipped_decode, Record};
use whoop_protocol::response::{data_range_scan_newest, data_range_scan_oldest};
use whoop_protocol::trim::{inert_probe, reset_to_oldest, IGNORE, RESET_TO_OLDEST, TRIM_ALL};
use whoop_protocol::variant::Variant;
use whoop_protocol::{framing, live};

// =================================================================================================
// The shipped gates, copied verbatim. Each const names the file:line it was taken from.
// =================================================================================================

/// crates/whoop-protocol/src/crc.rs:46
const GATE_CRC16_CLIENT_HELLO: u16 = 0x71E6;
/// crates/whoop-protocol/src/crc.rs:51
const GATE_CRC32_CLIENT_HELLO: u32 = 0x8D5C_3E36;
/// crates/whoop-protocol/src/crc.rs:57 — `crc8(&[0x00, 0x00]) == 0x00`
const GATE_CRC8_LEN_BYTES: u8 = 0x00;

/// crates/whoop-protocol/tests/real_frames.rs:89 — the only tolerance in the record gates.
const GATE_GRAVITY_UNIT_LO: f32 = 0.9;
const GATE_GRAVITY_UNIT_HI: f32 = 1.1;
/// crates/whoop-protocol/tests/real_frames.rs:96 — the per-axis gravity pin, now on all 4 gen4 frames too.
const GATE_GRAVITY_EXACT_TOL: f64 = 1e-6;
/// crates/whoop-protocol/tests/real_frames.rs:223 — `assert_eq!(p.samples.len(), 24)`
const GATE_V26_SAMPLE_COUNT: usize = 24;

/// crates/whoop-ffi/src/lib.rs:215-218 (and crates/whoop-protocol/src/live.rs:158-160).
const GATE_META_TYPE: u8 = 2;
const GATE_META_UNIX: u32 = 1_784_236_473;
const GATE_META_TRIM_CURSOR: u32 = 113_405;
/// The METADATA HISTORY_END frame both goldens read.
const HISTORY_END_HEX: &str = "aa011c00010023d1319102b949596a705d3b000000fdba010010000000000000f269faec";

/// crates/whoop-protocol/tests/capture_parity.rs:168-174 — wall clock + skew the scan gate pins.
const GATE_RANGE_WALL_NOW: u64 = 1_783_786_000;
const GATE_RANGE_SKEW: u64 = 48 * 3600;
const GATE_RANGE_CASES: [(&str, u32); 3] = [
    ("aa100057305d22009968526a083900001d2e2263", 1_783_785_625),
    ("aa10005730612200a268526ab0290000e87d155d", 1_783_785_634),
    ("aa100057307c2200e768526a78760000c997138d", 1_783_785_703),
];
/// crates/whoop-ffi/src/lib.rs:274-275 — the same scan, a fourth real frame, a 1 h skew.
const GATE_RANGE_FFI_HEX: &str = "aa014c00010032d124f22204010140bb0100f9ba010001bb0100f9ba010010000000000002006a00000088ff1d001432b869cc4c00004549596ab83e00004549596ab83e0000ae49596aeb1100000000d0da9256";
const GATE_RANGE_FFI_NEWEST: u32 = 1_784_236_462;
const GATE_RANGE_FFI_OLDEST: u32 = 1_778_385_408;
const GATE_RANGE_FFI_WALL_NOW: u64 = 1_784_236_480;
const GATE_RANGE_FFI_SKEW: u64 = 3600;

/// crates/whoop-protocol/src/trim.rs:42/47-49/55-57 — the destructive-command guard.
const GATE_TRIM_RESET_PAYLOAD: [u8; 8] = [0xFD; 8];
const GATE_TRIM_ERASE_PAYLOAD: [u8; 8] = [0xFE; 8];

/// crates/whoop-protocol/src/advertising.rs:59 — `[0,0] + 24 bytes + [0]`.
const GATE_ADV_CLAMPED_LEN: usize = 27;
/// crates/whoop-protocol/src/variant.rs:151-156 — every strap actually read.
const GATE_OBSERVED: [(&str, &str, Variant); 4] = [
    ("WS50_r00", "MG", Variant::WhoopMg),
    ("WG50_r52", "5.0", Variant::Whoop5),
    ("WG50_r45", "5.0", Variant::Whoop5),
    ("WS50_r03", "MG", Variant::WhoopMg),
];

/// crates/whoop-protocol/src/records/gen5.rs:242-244 — one real offloaded v18 second, and the register
/// goldens crates/whoop-protocol/src/records/gen5.rs:250-277 pins on it.
const V18_GOLDEN_HEX: &str = "aa01740001003fb12f128066b7760180fc546aeb710056011d0200000000000000002c0481555700\
ffb063f13d852b853db87ef43d298e803f7a01a100000000000000000051015901db0d6006010c020c000000000000000000\
00000000000000000000000000000100adb18080000000c8d69bc00000000d27d7e3";
const GATE_V18_GOLDEN_UNIX: u32 = 1_783_954_560;
const GATE_V18_GOLDEN_RECORD_INDEX: u32 = 24_557_414;
const GATE_V18_GOLDEN_HR: u8 = 86;
const GATE_V18_GOLDEN_RR: u16 = 541;
const GATE_V18_GOLDEN_AUX1: u16 = 337;
const GATE_V18_GOLDEN_AUX2: u16 = 345;
const GATE_V18_GOLDEN_SKIN_RAW: u16 = 3547;
const GATE_V18_GOLDEN_SKIN_C: f32 = 35.47;
const GATE_V18_GOLDEN_F105: f32 = -4.869_968_4;

/// crates/whoop-protocol/tests/real_frames.rs:397-424 — the v18 optical sentinel semantics.
const GATE_OPT_WORN: (u8, u8, u8, u8, bool) = (101, 111, 30, 30, false);
const GATE_OPT_SECOND_BASE_B: u8 = 128;
const GATE_OPT_SECOND_HR: u8 = 57;

// =================================================================================================
// Harness
// =================================================================================================

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Baseline,
    Null,
    Structural,
    Param,
}

impl Kind {
    fn tag(self) -> &'static str {
        match self {
            Kind::Baseline => "baseline",
            Kind::Null => "null",
            Kind::Structural => "structural",
            Kind::Param => "param",
        }
    }
}

struct Arm {
    name: String,
    kind: Kind,
    value: f64,
    pass: bool,
}

struct Table {
    metric: &'static str,
    gate: &'static str,
    arms: Vec<Arm>,
    notes: Vec<String>,
}

/// Running tally of one gate's pinned assertions. `value` is the fraction reproduced, so a gate that
/// mixes exact equalities with banded tolerances still yields one comparable scalar per arm.
#[derive(Default)]
struct Check {
    ok: u32,
    total: u32,
}

impl Check {
    fn t(&mut self, cond: bool) {
        self.total += 1;
        self.ok += u32::from(cond);
    }

    /// All `n` sub-claims of a frame the decoder refused: they cannot hold, but they must still count.
    fn miss(&mut self, n: u32) {
        self.total += n;
    }

    fn rate(&self) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        f64::from(self.ok) / f64::from(self.total)
    }

    /// The gate is a conjunction of `assert!`s: it passes only when every pinned claim reproduces.
    fn passes(&self) -> bool {
        self.total > 0 && self.ok == self.total
    }
}

impl Table {
    fn new(metric: &'static str, gate: &'static str) -> Self {
        Table { metric, gate, arms: Vec::new(), notes: Vec::new() }
    }

    fn arm(&mut self, kind: Kind, name: &str, c: Check) {
        self.arms.push(Arm { name: name.to_string(), kind, value: c.rate(), pass: c.passes() });
    }

    fn note(&mut self, s: String) {
        self.notes.push(s);
    }

    fn baseline(&self) -> f64 {
        self.arms.first().map_or(0.0, |a| a.value)
    }

    fn report(&self) -> Summary {
        println!("\n== {} ==", self.metric);
        println!("   gate: {}", self.gate);
        println!("   {:<11} {:<58} {:>8} {:>9}  shipped gate", "kind", "arm", "value", "delta");
        let base = self.baseline();
        let (mut caught, mut missed) = (0usize, 0usize);
        let mut floor: Option<f64> = None;
        let mut critical: Vec<&str> = Vec::new();
        for (i, a) in self.arms.iter().enumerate() {
            let delta = a.value - base;
            let verdict = if a.pass { "PASS" } else { "FAIL" };
            let marker = if i == 0 {
                "(expected)"
            } else if a.pass {
                "<-- MISSED"
            } else {
                "<-- caught"
            };
            println!(
                "   {:<11} {:<58} {:>8.4} {:>+9.4}  {}  {}",
                a.kind.tag(),
                a.name,
                a.value,
                delta,
                verdict,
                marker
            );
            if i == 0 {
                continue;
            }
            if a.pass {
                missed += 1;
                if a.kind == Kind::Null {
                    critical.push(&a.name);
                }
            } else {
                caught += 1;
                let d = delta.abs();
                floor = Some(floor.map_or(d, |f: f64| f.min(d)));
            }
        }
        println!("   caught {caught}, missed {missed}");
        match floor {
            Some(f) => println!("   sensitivity floor: smallest caught delta = {f:.4}"),
            None => println!("   sensitivity floor: NONE — this gate caught nothing"),
        }
        for c in &critical {
            println!("   CRITICAL: a NULL arm PASSED the shipped gate — '{c}' does no work and the gate agrees");
        }
        for n in &self.notes {
            println!("   note: {n}");
        }
        let probes: Vec<(&str, f64)> = self
            .arms
            .iter()
            .filter(|a| matches!(a.kind, Kind::Null | Kind::Structural))
            .map(|a| (a.name.as_str(), a.value))
            .collect();
        enforce_floors(self.metric, base, &probes);
        Summary {
            metric: self.metric,
            caught,
            missed,
            floor,
            baseline_passes: self.arms.first().is_some_and(|a| a.pass && a.kind == Kind::Baseline),
            some_null_failed: self.arms.iter().any(|a| a.kind == Kind::Null && !a.pass),
            critical_nulls: critical.len(),
        }
    }
}

struct Summary {
    metric: &'static str,
    caught: usize,
    missed: usize,
    floor: Option<f64>,
    baseline_passes: bool,
    some_null_failed: bool,
    critical_nulls: usize,
}

// =================================================================================================
// Fixtures
// =================================================================================================

/// `include_str!` — a missing fixture is a COMPILE error here, never a silent skip.
fn oracle() -> Value {
    serde_json::from_str(include_str!("fixtures/real_frames.json")).unwrap()
}

/// The cohort every arm below is scored over. A shrinking fixture silently weakens each mutation,
/// so the sizes are pinned here rather than inferred: 9 history frames, the 40-second v26 burst, 2 events.
fn assert_fixture_cohort(o: &Value) {
    let n = |k: &str| o[k].as_array().map_or(0, Vec::len);
    assert!(n("frames") >= 9, "history cohort shrank to {} frames", n("frames"));
    assert_eq!(n("ppg_frames"), 40, "the v26 burst is 40 strap-seconds");
    assert_eq!(n("event_frames"), 2, "event cohort: battery_level + wrist_on");
    let gen4 = o["frames"].as_array().unwrap().iter().filter(|f| f["family"] == "gen4").count();
    assert_eq!(gen4, 4, "4.0 frames: the only cohort exercising crc8 and the gen4 record layouts");
}

fn fixture_family(f: &Value) -> Family {
    match f["family"].as_str().unwrap_or("gen5") {
        "gen4" => Family::Gen4,
        _ => Family::Gen5,
    }
}

fn fixture_frame(f: &Value) -> Frame {
    let wire = from_hex(f["hex"].as_str().unwrap()).unwrap();
    framing::decode(fixture_family(f), &wire).unwrap()
}

fn frame_from_hex(family: Family, hex: &str) -> Frame {
    framing::decode(family, &from_hex(hex).unwrap()).unwrap()
}

// =================================================================================================
// Replica: the decoded history fields the gates pin
// =================================================================================================

#[derive(Clone, Debug, Default, PartialEq)]
struct Hist {
    version: u8,
    unix: u32,
    hr: Option<u8>,
    rr: Vec<u16>,
    gravity: Option<[f32; 3]>,
    skin_raw: Option<u16>,
    skin_c: Option<f32>,
    spo2: Option<(u16, u16)>,
    spo2_pct: Option<u8>,
    steps: Option<u16>,
    act: Option<u8>,
    sleep: Option<u8>,
    base_a: Option<u8>,
    base_b: Option<u8>,
    amp_a: Option<u8>,
    amp_b: Option<u8>,
    poor: Option<bool>,
    aux1: Option<u16>,
    aux2: Option<u16>,
    rec_index: Option<u32>,
    f105: Option<f32>,
}

fn hist_of_shipped(h: &whoop_protocol::HistoryRecord) -> Hist {
    Hist {
        version: h.version,
        unix: h.unix,
        hr: h.heart_rate,
        rr: h.rr_intervals.clone(),
        gravity: h.gravity,
        skin_raw: h.skin_temp_raw,
        skin_c: h.skin_temp_c,
        spo2: h.spo2,
        spo2_pct: h.spo2_pct,
        steps: h.steps,
        act: h.activity_class,
        sleep: h.sleep_state,
        base_a: h.optical_baseline_a,
        base_b: h.optical_baseline_b,
        amp_a: h.optical_amp_a,
        amp_b: h.optical_amp_b,
        poor: h.optical_signal_poor,
        aux1: h.temp_aux_1_raw,
        aux2: h.temp_aux_2_raw,
        rec_index: h.record_index,
        f105: h.raw_f32_105,
    }
}

/// `bytes::rr_intervals` — count byte, `max` slots, zero = empty slot.
fn rr_read(b: &[u8], count_off: usize, first: usize, max: usize) -> Vec<u16> {
    let count = (u8_at(b, count_off).unwrap_or(0) as usize).min(max);
    let mut rr = Vec::new();
    for i in 0..count {
        if let Some(v) = u16_at(b, first + i * 2) {
            if v != 0 {
                rr.push(v);
            }
        }
    }
    rr
}

fn magnitude(g: [f32; 3]) -> f32 {
    (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]).sqrt()
}

/// `records::gravity3` + `accept_gravity`, with the acceptance band exposed as a tunable.
fn gravity3(b: &[u8], off: usize, lo: f32, hi: f32, reverse: bool) -> Option<[f32; 3]> {
    let mut g = [f32_at(b, off)?, f32_at(b, off + 4)?, f32_at(b, off + 8)?];
    if reverse {
        g.reverse();
    }
    if !g.iter().all(|v| v.is_finite()) {
        return None;
    }
    (lo..hi).contains(&magnitude(g)).then_some(g)
}

// ------------------------------------------------------------------ gen5 v18

#[derive(Clone)]
struct V18P {
    unix_off: usize,
    hr_off: usize,
    rr_count_off: usize,
    rr_first_off: usize,
    rr_max: usize,
    rr_reverse: bool,
    grav_off: usize,
    grav_lo: f32,
    grav_hi: f32,
    grav_reverse: bool,
    skin_off: usize,
    skin_div: f32,
    skin_band_lo: f32,
    skin_band_hi: f32,
    spo2_off: usize,
    spo2_lo: u8,
    spo2_hi: u8,
    steps_off: usize,
    act_off: usize,
    act_max: u8,
    sleep_off: usize,
    sleep_shift: u8,
    sleep_mask: u8,
    base_a_off: usize,
    base_b_off: usize,
    amp_a_off: usize,
    amp_b_off: usize,
    amp_sentinel: u8,
    aux1_off: usize,
    aux2_off: usize,
    rec_index_off: usize,
    f105_off: usize,
    constant: bool,
}

impl Default for V18P {
    /// The shipped values, crates/whoop-protocol/src/records/gen5.rs:16-61.
    fn default() -> Self {
        V18P {
            unix_off: 7,
            hr_off: 14,
            rr_count_off: 15,
            rr_first_off: 16,
            rr_max: 4,
            rr_reverse: false,
            grav_off: 37,
            grav_lo: 0.5,
            grav_hi: 1.5,
            grav_reverse: false,
            skin_off: 65,
            skin_div: 100.0,
            skin_band_lo: 5.0,
            skin_band_hi: 45.0,
            spo2_off: 74,
            spo2_lo: 70,
            spo2_hi: 100,
            steps_off: 49,
            act_off: 55,
            act_max: 2,
            sleep_off: 73,
            sleep_shift: 4,
            sleep_mask: 3,
            base_a_off: 98,
            base_b_off: 99,
            amp_a_off: 100,
            amp_b_off: 101,
            amp_sentinel: 128,
            aux1_off: 61,
            aux2_off: 63,
            rec_index_off: 3,
            f105_off: 105,
            constant: false,
        }
    }
}

fn v18_replica(f: &Frame, p: &V18P) -> Option<Hist> {
    if p.constant {
        return Some(Hist { version: 18, ..Default::default() });
    }
    let b = f.inner();
    let unix = u32_at(b, p.unix_off)?;
    let skin_raw = u16_at(b, p.skin_off)
        .filter(|&r| (p.skin_band_lo..p.skin_band_hi).contains(&(f32::from(r) / p.skin_div)));
    let mut rr = rr_read(b, p.rr_count_off, p.rr_first_off, p.rr_max);
    if p.rr_reverse {
        rr.reverse();
    }
    let sentinel = match (u8_at(b, p.amp_a_off), u8_at(b, p.amp_b_off)) {
        (Some(a), Some(bb)) => a == p.amp_sentinel && bb == p.amp_sentinel,
        _ => false,
    };
    Some(Hist {
        version: 18,
        unix,
        hr: u8_at(b, p.hr_off).filter(|&v| v > 0),
        rr,
        gravity: gravity3(b, p.grav_off, p.grav_lo, p.grav_hi, p.grav_reverse),
        skin_raw,
        skin_c: skin_raw.map(|r| f32::from(r) / p.skin_div),
        spo2: None,
        spo2_pct: u8_at(b, p.spo2_off).filter(|&v| (p.spo2_lo..=p.spo2_hi).contains(&v)),
        steps: u16_at(b, p.steps_off),
        act: u8_at(b, p.act_off).filter(|&v| v <= p.act_max),
        sleep: u8_at(b, p.sleep_off).map(|v| (v >> p.sleep_shift) & p.sleep_mask),
        base_a: u8_at(b, p.base_a_off).filter(|&v| v != 0),
        base_b: u8_at(b, p.base_b_off).filter(|&v| v != 0),
        amp_a: u8_at(b, p.amp_a_off).filter(|_| !sentinel),
        amp_b: u8_at(b, p.amp_b_off).filter(|_| !sentinel),
        poor: u8_at(b, p.amp_a_off).zip(u8_at(b, p.amp_b_off)).map(|_| sentinel),
        aux1: u16_at(b, p.aux1_off),
        aux2: u16_at(b, p.aux2_off),
        rec_index: u32_at(b, p.rec_index_off),
        f105: f32_at(b, p.f105_off).filter(|v| v.is_finite()),
    })
}

// ------------------------------------------------------------------ gen5 v26

#[derive(Clone, Copy)]
struct V26P {
    unix_off: usize,
    rid_off: usize,
    first_off: usize,
    stride: usize,
    count: usize,
    reverse: bool,
    constant: bool,
}

impl Default for V26P {
    /// crates/whoop-protocol/src/records/gen5.rs:69-78.
    fn default() -> Self {
        V26P { unix_off: 7, rid_off: 3, first_off: 19, stride: 2, count: 24, reverse: false, constant: false }
    }
}

struct Ppg {
    unix: u32,
    rid: Option<u16>,
    samples: Vec<i16>,
}

fn v26_replica(f: &Frame, p: &V26P) -> Option<Ppg> {
    if p.constant {
        return Some(Ppg { unix: 0, rid: Some(0), samples: vec![0; p.count] });
    }
    let b = f.inner();
    let unix = u32_at(b, p.unix_off)?;
    let mut samples = Vec::with_capacity(p.count);
    for i in 0..p.count {
        samples.push(i16_at(b, p.first_off + i * p.stride)?);
    }
    if p.reverse {
        samples.reverse();
    }
    Some(Ppg { unix, rid: u16_at(b, p.rid_off), samples })
}

// ------------------------------------------------------------------ gen5 v21 IMU / v20 optical

#[derive(Clone, Copy)]
struct V21P {
    unix_off: usize,
    count_a_off: usize,
    count_b_off: usize,
    ax_off: usize,
    ay_off: usize,
    az_off: usize,
    gx_off: usize,
    gy_off: usize,
    gz_off: usize,
    samples: usize,
}

impl Default for V21P {
    /// crates/whoop-protocol/src/records/gen5.rs:83-97 (inner-relative).
    fn default() -> Self {
        V21P {
            unix_off: 7,
            count_a_off: 16,
            count_b_off: 622,
            ax_off: 20,
            ay_off: 220,
            az_off: 420,
            gx_off: 632,
            gy_off: 832,
            gz_off: 1032,
            samples: 100,
        }
    }
}

impl V21P {
    /// Shift every offset together — the mutation a synthetic round-trip cannot see.
    fn shifted(mut self, d: usize) -> Self {
        for o in [
            &mut self.count_a_off,
            &mut self.count_b_off,
            &mut self.ax_off,
            &mut self.ay_off,
            &mut self.az_off,
            &mut self.gx_off,
            &mut self.gy_off,
            &mut self.gz_off,
        ] {
            *o += d;
        }
        self
    }
}

/// The v21 buffer the shipped unit test builds (crates/whoop-protocol/src/records/gen5.rs:304-326).
fn build_v21(p: &V21P) -> Frame {
    let mut payload = vec![0u8; 1229 + 8];
    let put16 = |pl: &mut Vec<u8>, inner: usize, v: i16| {
        pl[inner - 3..inner - 1].copy_from_slice(&v.to_le_bytes());
    };
    payload[p.unix_off - 3..p.unix_off + 1].copy_from_slice(&1_784_000_000u32.to_le_bytes());
    put16(&mut payload, p.count_a_off, 100);
    put16(&mut payload, p.count_b_off, 100);
    put16(&mut payload, p.ax_off, 4096);
    put16(&mut payload, p.gx_off, 250);
    frame_of_payload(21, &payload)
}

struct Imu {
    unix: u32,
    rate: u16,
    accel0: [i16; 3],
    gyro0: [i16; 3],
    accel_len: usize,
    gyro_len: usize,
}

fn v21_replica(f: &Frame, p: &V21P) -> Option<Imu> {
    let b = f.inner();
    if u16_at(b, p.count_a_off)? != p.samples as u16 || u16_at(b, p.count_b_off)? != p.samples as u16 {
        return None;
    }
    let unix = u32_at(b, p.unix_off)?;
    let (mut accel, mut gyro) = (Vec::new(), Vec::new());
    for i in 0..p.samples {
        let o = i * 2;
        accel.push([i16_at(b, p.ax_off + o)?, i16_at(b, p.ay_off + o)?, i16_at(b, p.az_off + o)?]);
        gyro.push([i16_at(b, p.gx_off + o)?, i16_at(b, p.gy_off + o)?, i16_at(b, p.gz_off + o)?]);
    }
    Some(Imu {
        unix,
        rate: p.samples as u16,
        accel0: accel[0],
        gyro0: gyro[0],
        accel_len: accel.len(),
        gyro_len: gyro.len(),
    })
}

#[derive(Clone, Copy)]
struct V20P {
    unix_off: usize,
    green_off: usize,
    echo_off: usize,
    echo_mult: u16,
    ch0_off: usize,
    samples: usize,
}

impl Default for V20P {
    /// crates/whoop-protocol/src/records/gen5.rs:99-118 (first channel only; the other five follow).
    fn default() -> Self {
        V20P { unix_off: 7, green_off: 20, echo_off: 23, echo_mult: 2, ch0_off: 39, samples: 25 }
    }
}

/// The v20 buffer the shipped unit test builds (crates/whoop-protocol/src/records/gen5.rs:350-369).
fn build_v20(p: &V20P) -> Frame {
    let mut payload = vec![0u8; 2125 + 16];
    payload[p.unix_off - 3..p.unix_off + 1].copy_from_slice(&1_784_000_000u32.to_le_bytes());
    payload[p.green_off - 3..p.green_off - 1].copy_from_slice(&1400u16.to_le_bytes());
    payload[p.echo_off - 3..p.echo_off - 1].copy_from_slice(&2800u16.to_le_bytes());
    payload[p.ch0_off - 3..p.ch0_off + 1].copy_from_slice(&12345u32.to_le_bytes());
    payload[p.ch0_off + 1..p.ch0_off + 5].copy_from_slice(&0x000F_FFFBu32.to_le_bytes());
    frame_of_payload(20, &payload)
}

fn sign_extend_20(v: u32) -> i32 {
    ((v << 12) as i32) >> 12
}

fn v20_replica(f: &Frame, p: &V20P) -> Option<(u32, u16, usize, Vec<i32>)> {
    let b = f.inner();
    let unix = u32_at(b, p.unix_off)?;
    let green = u16_at(b, p.green_off)?;
    if green == 0 || u16_at(b, p.echo_off)? != green.wrapping_mul(p.echo_mult) {
        return None;
    }
    let mut ch0 = Vec::with_capacity(p.samples);
    for s in 0..p.samples {
        ch0.push(sign_extend_20(u32_at(b, p.ch0_off + s * 4)?));
    }
    Some((unix, p.samples as u16, ch0.len(), ch0))
}

fn frame_of_payload(version: u8, payload: &[u8]) -> Frame {
    let wire = framing::encode(Family::Gen5, 47, version, 0, payload);
    framing::decode(Family::Gen5, &wire).unwrap()
}

// ------------------------------------------------------------------ gen4 v24 / v25 / v5

#[derive(Clone, Copy)]
struct Gen4P {
    unix_off: usize,
    hr_off: usize,
    rr_count_off: usize,
    rr_first_off: usize,
    rr_max: usize,
    grav_off: usize,
    grav_lo: f32,
    grav_hi: f32,
    grav_reverse: bool,
    skin_off: usize,
    skin_scale: f32,
    skin_band_lo: f32,
    skin_band_hi: f32,
    spo2_red_off: usize,
    spo2_ir_off: usize,
    v25_grav_off: usize,
    v25_grav_stride: usize,
    v25_grav_div: f32,
    v25_reverse: bool,
    constant: bool,
}

impl Default for Gen4P {
    /// The shipped values, crates/whoop-protocol/src/records/gen4.rs:10-59.
    fn default() -> Self {
        Gen4P {
            unix_off: 7,
            hr_off: 17,
            rr_count_off: 18,
            rr_first_off: 19,
            rr_max: 4,
            grav_off: 36,
            grav_lo: 0.5,
            grav_hi: 1.5,
            grav_reverse: false,
            skin_off: 68,
            skin_scale: 0.04,
            skin_band_lo: 20.0,
            skin_band_hi: 45.0,
            spo2_red_off: 64,
            spo2_ir_off: 66,
            v25_grav_off: 69,
            v25_grav_stride: 2,
            v25_grav_div: 16384.0,
            v25_reverse: false,
            constant: false,
        }
    }
}

fn gen4_replica(f: &Frame, p: &Gen4P) -> Option<Hist> {
    if p.constant {
        return Some(Hist { version: f.version(), ..Default::default() });
    }
    let b = f.inner();
    let unix = u32_at(b, p.unix_off)?;
    match f.version() {
        25 => {
            let mut g = [
                f32::from(i16_at(b, p.v25_grav_off)?) / p.v25_grav_div,
                f32::from(i16_at(b, p.v25_grav_off + p.v25_grav_stride)?) / p.v25_grav_div,
                f32::from(i16_at(b, p.v25_grav_off + 2 * p.v25_grav_stride)?) / p.v25_grav_div,
            ];
            if p.v25_reverse {
                g.reverse();
            }
            let ok = g.iter().all(|v| v.is_finite()) && (p.grav_lo..p.grav_hi).contains(&magnitude(g));
            Some(Hist { version: 25, unix, gravity: ok.then_some(g), ..Default::default() })
        }
        12 | 24 => Some(Hist {
            version: f.version(),
            unix,
            hr: u8_at(b, p.hr_off).filter(|&v| v > 0),
            rr: rr_read(b, p.rr_count_off, p.rr_first_off, p.rr_max),
            gravity: gravity3(b, p.grav_off, p.grav_lo, p.grav_hi, p.grav_reverse),
            skin_c: i16_at(b, p.skin_off)
                .map(|r| f32::from(r) * p.skin_scale)
                .filter(|c| (p.skin_band_lo..p.skin_band_hi).contains(c)),
            skin_raw: u16_at(b, p.skin_off),
            spo2: match (u16_at(b, p.spo2_red_off), u16_at(b, p.spo2_ir_off)) {
                (Some(red), Some(ir)) => Some((red, ir)),
                _ => None,
            },
            ..Default::default()
        }),
        _ => Some(Hist {
            version: f.version(),
            unix,
            hr: u8_at(b, p.hr_off).filter(|&v| v > 0),
            rr: rr_read(b, p.rr_count_off, p.rr_first_off, p.rr_max),
            ..Default::default()
        }),
    }
}

// =================================================================================================
// Metric 1 — gen5 record decode (v18 / v20 / v21 / v26)
// =================================================================================================

/// Every claim real_frames.rs:49-141 / :203-247 / :397-424 and the gen5.rs register goldens pin, scored
/// through the replica so a mutation is measured against the real gate.
fn gen5_gate(v18p: &V18P, v26p: &V26P, v21p: &V21P, v20p: &V20P) -> Check {
    let o = oracle();
    let mut c = Check::default();

    // --- real_frames.rs:43-79, the three gen5 history frames.
    for f in o["frames"].as_array().unwrap().iter().filter(|f| fixture_family(f) == Family::Gen5) {
        let e = &f["expect"];
        let pinned = 1 + 1
            + u32::from(e.get("heart_rate").is_some())
            + u32::from(e.get("rr").is_some())
            + u32::from(e.get("gravity_unit").is_some())
            + u32::from(e.get("gravity").is_some())
            + u32::from(e.get("skin_temp_raw").is_some())
            + u32::from(e.get("activity_class").is_some())
            + u32::from(e.get("steps").is_some())
            + u32::from(e.get("sleep_state").is_some());
        let Some(r) = v18_replica(&fixture_frame(f), v18p) else {
            c.miss(pinned);
            continue;
        };
        c.t(u64::from(r.version) == e["version"].as_u64().unwrap());
        c.t(u64::from(r.unix) == e["unix"].as_u64().unwrap());
        if let Some(v) = e.get("heart_rate") {
            c.t(r.hr == Some(v.as_u64().unwrap() as u8));
        }
        if let Some(v) = e.get("rr").and_then(Value::as_array) {
            let want: Vec<u16> = v.iter().map(|x| x.as_u64().unwrap() as u16).collect();
            c.t(r.rr == want);
        }
        if e.get("gravity_unit").is_some() {
            c.t(r.gravity.is_some_and(|g| (GATE_GRAVITY_UNIT_LO..GATE_GRAVITY_UNIT_HI).contains(&magnitude(g))));
        }
        if let Some(v) = e.get("gravity").and_then(Value::as_array) {
            c.t(r.gravity.is_some_and(|g| {
                v.iter().enumerate().all(|(i, w)| (f64::from(g[i]) - w.as_f64().unwrap()).abs() < GATE_GRAVITY_EXACT_TOL)
            }));
        }
        if let Some(v) = e.get("skin_temp_raw") {
            c.t(r.skin_raw == Some(v.as_u64().unwrap() as u16));
        }
        if let Some(v) = e.get("activity_class") {
            c.t(r.act == Some(v.as_u64().unwrap() as u8));
        }
        if let Some(v) = e.get("steps") {
            c.t(r.steps == Some(v.as_u64().unwrap() as u16));
        }
        if let Some(v) = e.get("sleep_state") {
            c.t(r.sleep == Some(v.as_u64().unwrap() as u8));
        }
    }

    // --- real_frames.rs:152-172, the v18 optical sentinel semantics.
    let by_name = |n: &str| {
        let f = o["frames"].as_array().unwrap().iter().find(|f| f["name"] == n).unwrap().clone();
        v18_replica(&fixture_frame(&f), v18p)
    };
    match by_name("v18_real_whoop5_worn") {
        Some(w) => {
            c.t(w.base_a == Some(GATE_OPT_WORN.0));
            c.t(w.base_b == Some(GATE_OPT_WORN.1));
            c.t(w.amp_a == Some(GATE_OPT_WORN.2));
            c.t(w.amp_b == Some(GATE_OPT_WORN.3));
            c.t(w.poor == Some(GATE_OPT_WORN.4));
        }
        None => c.miss(5),
    }
    match by_name("v18_real_whoop5_offwrist") {
        Some(off) => {
            c.t(off.base_a.is_none());
            c.t(off.base_b.is_none());
            c.t(off.poor == Some(true));
        }
        None => c.miss(3),
    }
    match by_name("v18_real_whoop5_second_device") {
        Some(s) => {
            c.t(s.base_b == Some(GATE_OPT_SECOND_BASE_B));
            c.t(s.hr == Some(GATE_OPT_SECOND_HR));
            c.t(s.amp_a.is_none());
            c.t(s.amp_b.is_none());
            c.t(s.poor == Some(true));
        }
        None => c.miss(5),
    }

    // --- gen5.rs:250-277, the register-by-register golden on one real offloaded second.
    let golden = frame_from_hex(Family::Gen5, &V18_GOLDEN_HEX.replace(['\n', '\r'], ""));
    match v18_replica(&golden, v18p) {
        Some(g) => {
            c.t(g.unix == GATE_V18_GOLDEN_UNIX);
            c.t(g.rec_index == Some(GATE_V18_GOLDEN_RECORD_INDEX));
            c.t(g.hr == Some(GATE_V18_GOLDEN_HR));
            c.t(g.rr == vec![GATE_V18_GOLDEN_RR]);
            c.t(g.aux1 == Some(GATE_V18_GOLDEN_AUX1));
            c.t(g.aux2 == Some(GATE_V18_GOLDEN_AUX2));
            c.t(g.skin_raw == Some(GATE_V18_GOLDEN_SKIN_RAW));
            c.t(g.skin_c == Some(GATE_V18_GOLDEN_SKIN_C));
            c.t(g.sleep == Some(0));
            c.t(g.f105 == Some(GATE_V18_GOLDEN_F105));
        }
        None => c.miss(10),
    }

    // --- real_frames.rs:96-110, the forty v26 PPG frames.
    let ppg = o["ppg_frames"].as_array().unwrap();
    let mut last_rid: Option<u16> = None;
    for (i, f) in ppg.iter().enumerate() {
        let pinned = 3 + u32::from(i > 0) + u32::from(f.get("samples").is_some());
        let Some(p) = v26_replica(&fixture_frame(f), v26p) else {
            c.miss(pinned);
            continue;
        };
        c.t(u64::from(p.unix) == f["unix"].as_u64().unwrap());
        let rid = f["record_id"].as_u64().unwrap() as u16;
        c.t(p.rid == Some(rid));
        c.t(p.samples.len() == GATE_V26_SAMPLE_COUNT);
        if let Some(prev) = last_rid {
            c.t(rid == prev.wrapping_add(1));
        }
        last_rid = Some(rid);
        if let Some(s) = f.get("samples").and_then(Value::as_array) {
            let want: Vec<i16> = s.iter().map(|x| x.as_i64().unwrap() as i16).collect();
            c.t(p.samples == want);
        }
    }

    // --- gen5.rs:304-326 (v21) and :350-369 (v20). NO real-hardware fixture exists for either: the
    // buffer is built by the same offsets the decoder reads, so a coordinated shift is invisible.
    match v21_replica(&build_v21(&V21P::default()), v21p) {
        Some(r) => {
            c.t(r.unix == 1_784_000_000);
            c.t(r.rate == 100);
            c.t(r.accel_len == 100);
            c.t(r.gyro_len == 100);
            c.t(r.accel0 == [4096, 0, 0]);
            c.t(r.gyro0 == [250, 0, 0]);
        }
        None => c.miss(6),
    }
    match v20_replica(&build_v20(&V20P::default()), v20p) {
        Some((unix, rate, len, ch0)) => {
            c.t(unix == 1_784_000_000);
            c.t(rate == 25);
            c.t(len == 25);
            c.t(ch0[0] == 12345);
            c.t(ch0[1] == -5);
        }
        None => c.miss(5),
    }
    c
}

/// A gen5 gate run whose synthetic v21/v20 buffers move WITH the decoder — the honest model of "the
/// offsets were always these", which a self-consistent round-trip cannot falsify.
fn gen5_gate_coordinated(v21p: &V21P) -> Check {
    let mut c = Check::default();
    match v21_replica(&build_v21(v21p), v21p) {
        Some(r) => {
            c.t(r.unix == 1_784_000_000);
            c.t(r.rate == 100);
            c.t(r.accel_len == 100);
            c.t(r.gyro_len == 100);
            c.t(r.accel0 == [4096, 0, 0]);
            c.t(r.gyro0 == [250, 0, 0]);
        }
        None => c.miss(6),
    }
    c
}

fn metric_gen5() -> Table {
    let mut t = Table::new(
        "gen5 record decode (v18 / v20 / v21 / v26)",
        "real_frames.rs:43-79 + :96-110 + :152-172, gen5.rs:250-277 — exact field equality, no tolerance",
    );
    let (d18, d26, d21, d20) = (V18P::default(), V26P::default(), V21P::default(), V20P::default());
    let base = |t: &mut Table, k: Kind, n: &str, p: &V18P| t.arm(k, n, gen5_gate(p, &d26, &d21, &d20));

    t.arm(Kind::Baseline, "baseline (unmutated replica)", gen5_gate(&d18, &d26, &d21, &d20));

    // NULL
    base(&mut t, Kind::Null, "output: every field absent (constant record)", &V18P { constant: true, ..d18.clone() });
    t.arm(
        Kind::Null,
        "output: v26 waveform = 24 zeros, unix 0",
        gen5_gate(&d18, &V26P { constant: true, ..d26 }, &d21, &d20),
    );

    // STRUCTURAL
    base(&mut t, Kind::Structural, "v18: every offset slipped +1 byte", &V18P {
        hr_off: 15,
        rr_count_off: 16,
        rr_first_off: 17,
        grav_off: 38,
        skin_off: 66,
        steps_off: 50,
        act_off: 56,
        sleep_off: 74,
        base_a_off: 99,
        base_b_off: 100,
        amp_a_off: 101,
        amp_b_off: 102,
        aux1_off: 62,
        aux2_off: 64,
        ..d18.clone()
    });
    base(&mut t, Kind::Structural, "v18: optical baselines A<->B swapped", &V18P {
        base_a_off: 99,
        base_b_off: 98,
        ..d18.clone()
    });
    base(&mut t, Kind::Structural, "v18: thermal aux 1 <-> aux 2 swapped", &V18P {
        aux1_off: 63,
        aux2_off: 61,
        ..d18.clone()
    });
    base(&mut t, Kind::Structural, "v18: gravity axes reversed (x,y,z -> z,y,x)", &V18P {
        grav_reverse: true,
        ..d18.clone()
    });
    base(&mut t, Kind::Structural, "v18: R-R slot order reversed", &V18P { rr_reverse: true, ..d18.clone() });
    base(&mut t, Kind::Structural, "v18: activity_class offset 55 -> 56 (that field alone)", &V18P {
        act_off: 56,
        ..d18.clone()
    });
    base(&mut t, Kind::Structural, "v18: R-R slots 4 -> 3 (drop the last slot)", &V18P { rr_max: 3, ..d18.clone() });
    t.arm(
        Kind::Structural,
        "v26: waveform reversed",
        gen5_gate(&d18, &V26P { reverse: true, ..d26 }, &d21, &d20),
    );
    t.arm(
        Kind::Structural,
        "v26: waveform shifted +1 sample",
        gen5_gate(&d18, &V26P { first_off: 21, ..d26 }, &d21, &d20),
    );
    t.arm(
        Kind::Structural,
        "v21: decoder offsets +2, synthetic buffer unchanged",
        gen5_gate(&d18, &d26, &d21.shifted(2), &d20),
    );
    t.arm(
        Kind::Structural,
        "v21: buffer AND decoder offsets +2 (coordinated)",
        gen5_gate_coordinated(&d21.shifted(2)),
    );

    // PARAMETER — one tunable each
    base(&mut t, Kind::Param, "v18: skin_temp divisor 100 -> 110 (+10%)", &V18P { skin_div: 110.0, ..d18.clone() });
    base(&mut t, Kind::Param, "v18: skin_temp divisor 100 -> 100.5 (+0.5%)", &V18P { skin_div: 100.5, ..d18.clone() });
    base(&mut t, Kind::Param, "v18: skin band 5..45 -> 5.5..49.5 (+10%)", &V18P {
        skin_band_lo: 5.5,
        skin_band_hi: 49.5,
        ..d18.clone()
    });
    base(&mut t, Kind::Param, "v18: skin band floor 5 -> 33 degC", &V18P { skin_band_lo: 33.0, ..d18.clone() });
    base(&mut t, Kind::Param, "v18: activity_class ceiling 2 -> 1", &V18P { act_max: 1, ..d18.clone() });
    base(&mut t, Kind::Param, "v18: SpO2 window 70..100 -> 63..110 (+-10%)", &V18P {
        spo2_lo: 63,
        spo2_hi: 110,
        ..d18.clone()
    });
    base(&mut t, Kind::Param, "v18: gravity accept band 0.5..1.5 -> 0.55..1.65 (+10%)", &V18P {
        grav_lo: 0.55,
        grav_hi: 1.65,
        ..d18.clone()
    });
    base(&mut t, Kind::Param, "v18: gravity accept band -> 0.99..1.01 (near-exact)", &V18P {
        grav_lo: 0.99,
        grav_hi: 1.01,
        ..d18.clone()
    });
    base(&mut t, Kind::Param, "v18: sleep_state shift 4 -> 3", &V18P { sleep_shift: 3, ..d18.clone() });
    base(&mut t, Kind::Param, "v18: optical amp sentinel 128 -> 141 (+10%)", &V18P { amp_sentinel: 141, ..d18.clone() });
    t.arm(
        Kind::Param,
        "v26: sample count 24 -> 26 (+10%)",
        gen5_gate(&d18, &V26P { count: 26, ..d26 }, &d21, &d20),
    );
    t.arm(
        Kind::Param,
        "v21: sample-count gate 100 -> 110 (+10%)",
        gen5_gate(&d18, &d26, &V21P { samples: 110, ..d21 }, &d20),
    );
    t.arm(
        Kind::Param,
        "v20: LED echo multiplier 2 -> 3",
        gen5_gate(&d18, &d26, &d21, &V20P { echo_mult: 3, ..d20 }),
    );

    t.note("real_frames.json carries NO v20 and NO v21 frame: both are covered only by synthetic \
            round-trips whose builder shares the decoder's offsets."
        .to_string());
    t.note("gen5 fixture frames pin activity_class across all three codes (0/1/2), which leaves exactly \
            one wire offset satisfying them; they still never pin spo2_pct, dynamic_acceleration_g, \
            signal_flags or signal_quality."
        .to_string());
    t.note("no real frame carries an activity_class outside 0..=2 (80,951 captured v18 frames: 80,217 \
            still, 733 walk, 1 run), so the decoder's <= 2 sentinel filter is exercised only by the \
            synthetic round-trip in gen5.rs, never by real data."
        .to_string());
    t
}

// =================================================================================================
// Metric 2 — gen4 record decode (v5 / v24 / v25)
// =================================================================================================

fn gen4_gate(p: &Gen4P) -> Check {
    let o = oracle();
    let mut c = Check::default();
    for f in o["frames"].as_array().unwrap().iter().filter(|f| fixture_family(f) == Family::Gen4) {
        let e = &f["expect"];
        let pinned = 1 + 1
            + u32::from(e.get("heart_rate").is_some())
            + u32::from(e.get("rr").is_some())
            + u32::from(e.get("gravity_unit").is_some())
            + u32::from(e.get("skin_temp_raw").is_some())
            + u32::from(e.get("spo2").is_some());
        let Some(r) = gen4_replica(&fixture_frame(f), p) else {
            c.miss(pinned);
            continue;
        };
        c.t(u64::from(r.version) == e["version"].as_u64().unwrap());
        c.t(u64::from(r.unix) == e["unix"].as_u64().unwrap());
        if let Some(v) = e.get("heart_rate") {
            c.t(r.hr == Some(v.as_u64().unwrap() as u8));
        }
        if let Some(v) = e.get("rr").and_then(Value::as_array) {
            let want: Vec<u16> = v.iter().map(|x| x.as_u64().unwrap() as u16).collect();
            c.t(r.rr == want);
        }
        if e.get("gravity_unit").is_some() {
            c.t(r.gravity.is_some_and(|g| (GATE_GRAVITY_UNIT_LO..GATE_GRAVITY_UNIT_HI).contains(&magnitude(g))));
        }
        if let Some(v) = e.get("skin_temp_raw") {
            c.t(r.skin_raw == Some(v.as_u64().unwrap() as u16));
        }
        if let Some(v) = e.get("spo2").and_then(Value::as_array) {
            let (red, ir) = (v[0].as_u64().unwrap() as u16, v[1].as_u64().unwrap() as u16);
            c.t(r.spo2 == Some((red, ir)));
        }
    }
    c
}

fn metric_gen4() -> Table {
    let mut t = Table::new(
        "gen4 record decode (v5 / v24 / v25)",
        "real_frames.rs:43-79 over 4 frames — v24 pins 7 fields, the three v25 frames pin only \
         version, unix and |g| in 0.9..1.1",
    );
    let d = Gen4P::default();
    let arm = |t: &mut Table, k: Kind, n: &str, p: Gen4P| t.arm(k, n, gen4_gate(&p));

    t.arm(Kind::Baseline, "baseline (unmutated replica)", gen4_gate(&d));

    arm(&mut t, Kind::Null, "output: every field absent (constant record)", Gen4P { constant: true, ..d });

    arm(&mut t, Kind::Structural, "v25: gravity axes reversed (x,y,z -> z,y,x)", Gen4P { v25_reverse: true, ..d });
    arm(&mut t, Kind::Structural, "v24: gravity axes reversed", Gen4P { grav_reverse: true, ..d });
    arm(&mut t, Kind::Structural, "v24: SpO2 red <-> IR swapped", Gen4P {
        spo2_red_off: 66,
        spo2_ir_off: 64,
        ..d
    });
    arm(&mut t, Kind::Structural, "v24: R-R slots 4 -> 3 (drop the last slot)", Gen4P { rr_max: 3, ..d });
    arm(&mut t, Kind::Structural, "v25: gravity read big-endian (stride kept)", Gen4P {
        v25_grav_off: 70,
        ..d
    });

    arm(&mut t, Kind::Param, "v25: gravity divisor 16384 -> 18022 (+10%)", Gen4P { v25_grav_div: 18022.0, ..d });
    arm(&mut t, Kind::Param, "v25: gravity divisor 16384 -> 14746 (-10%)", Gen4P { v25_grav_div: 14746.0, ..d });
    arm(&mut t, Kind::Param, "v25: gravity divisor 16384 -> 16466 (+0.5%)", Gen4P { v25_grav_div: 16466.0, ..d });
    arm(&mut t, Kind::Param, "v24: skin scale 0.04 -> 0.044 (+10%)", Gen4P { skin_scale: 0.044, ..d });
    arm(&mut t, Kind::Param, "v24: skin band 20..45 -> 22..49.5 (+10%)", Gen4P {
        skin_band_lo: 22.0,
        skin_band_hi: 49.5,
        ..d
    });
    arm(&mut t, Kind::Param, "v24: heart-rate offset 17 -> 18", Gen4P { hr_off: 18, ..d });
    arm(&mut t, Kind::Param, "gravity accept band 0.5..1.5 -> 0.55..1.65 (+10%)", Gen4P {
        grav_lo: 0.55,
        grav_hi: 1.65,
        ..d
    });

    t.note("v25's only physical claim is |g| in 0.9..1.1, which is invariant under any rotation or \
            permutation of the three axes."
        .to_string());
    t.note("resp_raw (v24 inner 76) is decoded and pinned by NO fixture; gen4.rs carries zero unit tests."
        .to_string());
    t
}

// =================================================================================================
// Metric 3 — CRC (modbus-16, zlib-32, crc8)
// =================================================================================================

#[derive(Clone, Copy)]
struct CrcP {
    c8_poly: u8,
    c8_init: u8,
    c8_zero: bool,
    c16_poly: u16,
    c16_init: u16,
    c16_zero: bool,
    c32_poly: u32,
    c32_init: u32,
    c32_xor: u32,
    c32_zero: bool,
    reverse_input: bool,
}

impl Default for CrcP {
    /// The shipped params, crates/whoop-protocol/src/crc.rs:6-38.
    fn default() -> Self {
        CrcP {
            c8_poly: 0x07,
            c8_init: 0x00,
            c8_zero: false,
            c16_poly: 0xA001,
            c16_init: 0xFFFF,
            c16_zero: false,
            c32_poly: 0xEDB8_8320,
            c32_init: 0xFFFF_FFFF,
            c32_xor: 0xFFFF_FFFF,
            c32_zero: false,
            reverse_input: false,
        }
    }
}

fn crc8_replica(data: &[u8], p: &CrcP) -> u8 {
    if p.c8_zero {
        return 0;
    }
    let mut crc = p.c8_init;
    for &b in data {
        crc ^= b;
        for _ in 0..8 {
            crc = if crc & 0x80 != 0 { (crc << 1) ^ p.c8_poly } else { crc << 1 };
        }
    }
    crc
}

fn crc16_replica(data: &[u8], p: &CrcP) -> u16 {
    if p.c16_zero {
        return 0;
    }
    let mut crc = p.c16_init;
    for &b in data {
        crc ^= u16::from(b);
        for _ in 0..8 {
            crc = if crc & 1 != 0 { (crc >> 1) ^ p.c16_poly } else { crc >> 1 };
        }
    }
    crc
}

fn crc32_replica(data: &[u8], p: &CrcP) -> u32 {
    if p.c32_zero {
        return 0;
    }
    let mut crc = p.c32_init;
    for &b in data {
        crc ^= u32::from(b);
        for _ in 0..8 {
            crc = if crc & 1 != 0 { (crc >> 1) ^ p.c32_poly } else { crc >> 1 };
        }
    }
    crc ^ p.c32_xor
}

fn crc_gate(p: &CrcP) -> Check {
    let mut c = Check::default();
    let feed = |v: &[u8]| -> Vec<u8> {
        let mut x = v.to_vec();
        if p.reverse_input {
            x.reverse();
        }
        x
    };
    c.t(crc16_replica(&feed(&[0xAA, 0x01, 0x08, 0x00, 0x00, 0x01]), p) == GATE_CRC16_CLIENT_HELLO);
    c.t(crc32_replica(&feed(&[0x23, 0x01, 0x91, 0x01]), p) == GATE_CRC32_CLIENT_HELLO);
    c.t(crc8_replica(&feed(&[0x00, 0x00]), p) == GATE_CRC8_LEN_BYTES);
    // The real-frame half of the gate. `crc8(&[0,0]) == 0` is reproduced by a crc8 that does no work,
    // so the 51 captured frames are scored too: their stored header and body checksums are not zero.
    for (family, wire) in real_wires() {
        let h = family.header();
        let inner = h.inner_start..h.inner_start + u16_at(&wire, h.len_offset).unwrap() as usize - 4;
        let end = inner.end;
        match family {
            Family::Gen4 => c.t(crc8_replica(&feed(&wire[1..3]), p) == wire[3]),
            Family::Gen5 => c.t(crc16_replica(&feed(&wire[0..6]), p) == u16_at(&wire, 6).unwrap()),
        }
        c.t(crc32_replica(&feed(&wire[inner]), p) == u32_at(&wire, end).unwrap());
    }
    c
}

/// Every captured frame as `(family, wire)` — the history cohort, the v26 burst and the two events.
fn real_wires() -> Vec<(Family, Vec<u8>)> {
    let o = oracle();
    assert_fixture_cohort(&o);
    ["frames", "ppg_frames", "event_frames"]
        .iter()
        .flat_map(|k| o[*k].as_array().unwrap().clone())
        .map(|f| (fixture_family(&f), from_hex(f["hex"].as_str().unwrap()).unwrap()))
        .collect()
}

fn metric_crc() -> Table {
    let mut t = Table::new(
        "CRC (modbus-16, zlib-32, crc8)",
        "crc.rs:46/:51/:57 three known answers, plus real_frames.rs:151 every stored checksum on 51 real frames",
    );
    let d = CrcP::default();
    let arm = |t: &mut Table, k: Kind, n: &str, p: CrcP| t.arm(k, n, crc_gate(&p));

    t.arm(Kind::Baseline, "baseline (unmutated replica)", crc_gate(&d));

    arm(&mut t, Kind::Null, "output: all three checksums return 0", CrcP {
        c8_zero: true,
        c16_zero: true,
        c32_zero: true,
        ..d
    });
    arm(&mut t, Kind::Null, "output: crc8 alone returns 0 (does no work)", CrcP { c8_zero: true, ..d });

    arm(&mut t, Kind::Structural, "input byte order reversed for all three", CrcP { reverse_input: true, ..d });

    arm(&mut t, Kind::Param, "crc8 poly 0x07 -> 0x08 (one bit)", CrcP { c8_poly: 0x08, ..d });
    arm(&mut t, Kind::Param, "crc8 poly 0x07 -> 0x31 (a different standard)", CrcP { c8_poly: 0x31, ..d });
    arm(&mut t, Kind::Param, "crc8 init 0x00 -> 0x01", CrcP { c8_init: 0x01, ..d });
    arm(&mut t, Kind::Param, "crc16 poly 0xA001 -> 0xA002 (one bit)", CrcP { c16_poly: 0xA002, ..d });
    arm(&mut t, Kind::Param, "crc16 init 0xFFFF -> 0xFFFE (one bit)", CrcP { c16_init: 0xFFFE, ..d });
    arm(&mut t, Kind::Param, "crc32 poly 0xEDB88320 -> 0xEDB88321 (one bit)", CrcP { c32_poly: 0xEDB8_8321, ..d });
    arm(&mut t, Kind::Param, "crc32 xorout 0xFFFFFFFF -> 0 (drop the final xor)", CrcP { c32_xor: 0, ..d });

    // The shipped functions must agree with the unmutated replica, or the arms score the wrong thing.
    assert_eq!(crc8(&[0x00, 0x00]), crc8_replica(&[0x00, 0x00], &d), "crc8 replica drifted from shipped");
    assert_eq!(
        crc16_modbus(&[0xAA, 0x01, 0x08, 0x00, 0x00, 0x01]),
        crc16_replica(&[0xAA, 0x01, 0x08, 0x00, 0x00, 0x01], &d),
        "crc16 replica drifted from shipped"
    );
    assert_eq!(
        crc32_zlib(&[0x23, 0x01, 0x91, 0x01]),
        crc32_replica(&[0x23, 0x01, 0x91, 0x01], &d),
        "crc32 replica drifted from shipped"
    );

    let gen4_len = real_wires().into_iter().find(|(f, _)| *f == Family::Gen4).map(|(_, w)| [w[1], w[2]]).unwrap();
    t.note(format!(
        "the crc.rs vector is an ALL-ZERO input with an all-zero init, so it is 0x00 for every polynomial \
         (poly 0x07 -> {:#04x}, poly 0x31 -> {:#04x}); only the real 4.0 header bytes separate them \
         ({:#04x} vs {:#04x})",
        crc8_replica(&[0, 0], &d),
        crc8_replica(&[0, 0], &CrcP { c8_poly: 0x31, ..d }),
        crc8_replica(&gen4_len, &d),
        crc8_replica(&gen4_len, &CrcP { c8_poly: 0x31, ..d })
    ));
    t
}

// =================================================================================================
// Metric 4 — Framing / deframing
// =================================================================================================

#[derive(Clone, Copy)]
struct FrameP {
    sof: u8,
    inner_start: usize,
    len_offset: usize,
    crc16_header: bool,
    pad_to_4: bool,
    len_big_endian: bool,
    crc_ok_always: bool,
    min_decl: usize,
    max_frame: usize,
}

impl FrameP {
    /// The shipped `HeaderSpec`, crates/whoop-protocol/src/family.rs:52-68, plus the deframe.rs bounds.
    fn gen5() -> Self {
        FrameP {
            sof: 0xAA,
            inner_start: 8,
            len_offset: 2,
            crc16_header: true,
            pad_to_4: true,
            len_big_endian: false,
            crc_ok_always: false,
            min_decl: 5,
            max_frame: 8192,
        }
    }

    fn gen4() -> Self {
        FrameP { inner_start: 4, len_offset: 1, crc16_header: false, pad_to_4: false, ..FrameP::gen5() }
    }
}

fn encode_replica(p: &FrameP, packet_type: u8, seq: u8, cmd: u8, payload: &[u8]) -> Vec<u8> {
    let mut inner = vec![packet_type, seq, cmd];
    inner.extend_from_slice(payload);
    if p.pad_to_4 {
        let pad = (4 - inner.len() % 4) % 4;
        inner.resize(inner.len() + pad, 0);
    }
    let decl = (inner.len() + 4) as u16;
    let decl_bytes = if p.len_big_endian { decl.to_be_bytes() } else { decl.to_le_bytes() };
    let mut frame = Vec::new();
    if p.crc16_header {
        frame.push(p.sof);
        frame.push(0x01);
        frame.extend_from_slice(&decl_bytes);
        frame.extend_from_slice(&[0x00, 0x01]);
        frame.extend_from_slice(&crc16_modbus(&frame[0..6]).to_le_bytes());
    } else {
        frame.push(p.sof);
        frame.extend_from_slice(&decl_bytes);
        frame.push(crc8(&frame[1..3]));
    }
    frame.extend_from_slice(&inner);
    frame.extend_from_slice(&crc32_zlib(&inner).to_le_bytes());
    frame
}

struct Parsed {
    inner: Vec<u8>,
    crc_ok: bool,
}

fn decode_replica(p: &FrameP, bytes: &[u8]) -> Option<Parsed> {
    if bytes.first().copied() != Some(p.sof) || bytes.len() < p.len_offset + 2 {
        return None;
    }
    let s = &bytes[p.len_offset..p.len_offset + 2];
    let decl = if p.len_big_endian {
        u16::from_be_bytes([s[0], s[1]]) as usize
    } else {
        u16::from_le_bytes([s[0], s[1]]) as usize
    };
    let inner_len = decl.checked_sub(4)?;
    if inner_len == 0 {
        return None;
    }
    let total = decl + p.inner_start;
    if bytes.len() < total {
        return None;
    }
    let inner_end = p.inner_start + inner_len;
    let inner = &bytes[p.inner_start..inner_end];
    let header_ok = if p.crc16_header {
        u16_at(bytes, 6) == Some(crc16_modbus(&bytes[0..6]))
    } else {
        crc8(&bytes[1..3]) == bytes[3]
    };
    let payload_ok = u32_at(bytes, inner_end) == Some(crc32_zlib(inner));
    Some(Parsed { inner: inner.to_vec(), crc_ok: p.crc_ok_always || (header_ok && payload_ok) })
}

/// `deframe::Deframer::push` in one shot: how many complete frames come out of `data`.
fn deframe_replica(p: &FrameP, data: &[u8]) -> usize {
    let (mut head, mut out) = (0usize, 0usize);
    loop {
        while head < data.len() && data[head] != p.sof {
            head += 1;
        }
        if data.len() - head < p.len_offset + 2 {
            return out;
        }
        let s = &data[head + p.len_offset..head + p.len_offset + 2];
        let decl = if p.len_big_endian {
            u16::from_be_bytes([s[0], s[1]]) as usize
        } else {
            u16::from_le_bytes([s[0], s[1]]) as usize
        };
        let total = decl + p.inner_start;
        if decl < p.min_decl || total > p.max_frame {
            head += 1;
            continue;
        }
        if data.len() - head < total {
            return out;
        }
        if decode_replica(p, &data[head..head + total]).is_some() {
            out += 1;
        }
        head += total;
    }
}

/// A real worn v18 frame — crates/whoop-ffi/src/lib.rs:247-250 flips its last byte and requires the drop.
const FFI_V18_HEX: &str = "aa01740001003fb12f1280733d8401b69f266a66460066025a0265020000000000007b0a8d656463ff0012163cf6a439bf2924fd3ed763fe3e3200aa000000000000000000f7000901f10b0007010c020c00000000000000000000000000000000000000000000000100656f1e1e0000009d61a7c00000003e862817";

fn framing_gate(p5: &FrameP, p4: &FrameP) -> Check {
    let mut c = Check::default();

    // framing.rs:88 — byte-identical encode of the bond frame.
    c.t(encode_replica(p5, 35, 1, GET_HELLO, &[0x01]) == GEN5_CLIENT_HELLO.to_vec());

    // framing.rs:94-99 — round-trip fields + crc_ok.
    match decode_replica(p5, &GEN5_CLIENT_HELLO) {
        Some(d) => {
            c.t(d.crc_ok);
            c.t(d.inner.first() == Some(&35));
            c.t(d.inner.get(2) == Some(&GET_HELLO));
            c.t(d.inner.get(3..) == Some(&[0x01][..]));
        }
        None => c.miss(4),
    }

    // framing.rs:110 — a trashed CRC32 tail must report crc_ok false, not an error.
    let mut bad = GEN5_CLIENT_HELLO.to_vec();
    *bad.last_mut().unwrap() ^= 0xFF;
    c.t(decode_replica(p5, &bad).is_some_and(|d| !d.crc_ok));

    // framing.rs:104-107 — the gen4 round-trip.
    let g4 = encode_replica(p4, 35, 0, GET_HELLO, &[0xAB, 0xCD]);
    match decode_replica(p4, &g4) {
        Some(d) => {
            c.t(d.crc_ok);
            c.t(d.inner.get(3..) == Some(&[0xAB, 0xCD][..]));
        }
        None => c.miss(2),
    }

    // whoop-ffi/src/lib.rs:119-125 — a whole frame fed after a reset yields exactly one frame.
    c.t(deframe_replica(p5, &from_hex(FFI_V18_HEX).unwrap()) == 1);

    // whoop-ffi/src/lib.rs:247-250 — good CRC decodes, one flipped byte drops the record.
    let mut raw = from_hex(FFI_V18_HEX).unwrap();
    c.t(decode_replica(p5, &raw).is_some_and(|d| d.crc_ok));
    *raw.last_mut().unwrap() ^= 0xFF;
    c.t(decode_replica(p5, &raw).is_some_and(|d| !d.crc_ok));
    c
}

fn metric_framing() -> Table {
    let mut t = Table::new(
        "Framing / deframing (header, CRC placement, reassembly)",
        "framing.rs:88/94-99/104-107/110 + whoop-ffi lib.rs:119-125 and :247-250 — byte-identical \
         encode, round-trip, and a flipped CRC byte must drop the frame",
    );
    let (d5, d4) = (FrameP::gen5(), FrameP::gen4());
    let arm = |t: &mut Table, k: Kind, n: &str, p: FrameP| t.arm(k, n, framing_gate(&p, &d4));

    t.arm(Kind::Baseline, "baseline (unmutated replica)", framing_gate(&d5, &d4));

    arm(&mut t, Kind::Null, "crc_ok always true (checksum never consulted)", FrameP { crc_ok_always: true, ..d5 });

    arm(&mut t, Kind::Structural, "declared length written/read big-endian", FrameP { len_big_endian: true, ..d5 });
    arm(&mut t, Kind::Structural, "gen5 inner_start 8 -> 9", FrameP { inner_start: 9, ..d5 });
    arm(&mut t, Kind::Structural, "gen5 len_offset 2 -> 3", FrameP { len_offset: 3, ..d5 });
    arm(&mut t, Kind::Structural, "SOF 0xAA -> 0xAB", FrameP { sof: 0xAB, ..d5 });

    arm(&mut t, Kind::Param, "pad_inner_to_4 disabled", FrameP { pad_to_4: false, ..d5 });
    arm(&mut t, Kind::Param, "deframer MAX_FRAME 8192 -> 9011 (+10%)", FrameP { max_frame: 9011, ..d5 });
    arm(&mut t, Kind::Param, "deframer MAX_FRAME 8192 -> 7373 (-10%)", FrameP { max_frame: 7373, ..d5 });
    arm(&mut t, Kind::Param, "deframer min decl 5 -> 6 (+20%, integer floor of +10%)", FrameP { min_decl: 6, ..d5 });
    arm(&mut t, Kind::Param, "deframer min decl 5 -> 9 (above the bond frame's decl 8)", FrameP { min_decl: 9, ..d5 });

    // The shipped encoder must agree with the unmutated replica.
    assert_eq!(
        framing::encode(Family::Gen5, 35, 1, GET_HELLO, &[0x01]),
        encode_replica(&d5, 35, 1, GET_HELLO, &[0x01]),
        "framing replica drifted from shipped"
    );

    t.note(
        "the bond frame's inner is exactly 4 bytes, so pad_inner_to_4 is a no-op on the only \
         byte-identical encode golden the codec has."
            .to_string(),
    );
    t
}

// =================================================================================================
// Metric 5 — HISTORY_END trim cursor + metadata parity (the safety-critical decode)
// =================================================================================================

#[derive(Clone, Copy)]
struct MetaP {
    type_off: usize,
    unix_off: usize,
    trim_off: usize,
    big_endian: bool,
    zero: bool,
}

impl Default for MetaP {
    /// crates/whoop-protocol/src/live.rs:96-104.
    fn default() -> Self {
        MetaP { type_off: 2, unix_off: 3, trim_off: 13, big_endian: false, zero: false }
    }
}

fn read_u32(b: &[u8], i: usize, be: bool) -> Option<u32> {
    let s = b.get(i..i + 4)?;
    Some(if be {
        u32::from_be_bytes([s[0], s[1], s[2], s[3]])
    } else {
        u32::from_le_bytes([s[0], s[1], s[2], s[3]])
    })
}

fn metadata_replica(f: &Frame, p: &MetaP) -> Option<(u8, u32, u32)> {
    if p.zero {
        return Some((0, 0, 0));
    }
    let b = f.inner();
    Some((
        *b.get(p.type_off)?,
        read_u32(b, p.unix_off, p.big_endian).unwrap_or(0),
        read_u32(b, p.trim_off, p.big_endian).unwrap_or(0),
    ))
}

fn metadata_gate(p: &MetaP) -> Check {
    let mut c = Check::default();
    let f = frame_from_hex(Family::Gen5, HISTORY_END_HEX);
    match metadata_replica(&f, p) {
        Some((mt, unix, trim)) => {
            c.t(mt == GATE_META_TYPE);
            c.t(unix == GATE_META_UNIX);
            c.t(trim == GATE_META_TRIM_CURSOR);
        }
        None => c.miss(3),
    }
    c.t(f.crc_ok);
    // live.rs:172-179 — HISTORY_COMPLETE carries meta_type only.
    let complete = framing::decode(Family::Gen5, &framing::encode(Family::Gen5, 49, 0, 3, &[])).unwrap();
    match metadata_replica(&complete, p) {
        Some((mt, unix, trim)) => {
            c.t(mt == 3);
            c.t(unix == 0);
            c.t(trim == 0);
        }
        None => c.miss(3),
    }
    c
}

fn metric_metadata() -> Table {
    let mut t = Table::new(
        "HISTORY_END trim cursor + metadata parity",
        "whoop-ffi lib.rs:215-218 meta_type 2 / unix 1784236473 / trim_cursor 113405 / crc_ok, \
         live.rs:158-160 the same frame, live.rs:172-179 the COMPLETE case",
    );
    let d = MetaP::default();
    let arm = |t: &mut Table, k: Kind, n: &str, p: MetaP| t.arm(k, n, metadata_gate(&p));

    t.arm(Kind::Baseline, "baseline (unmutated replica)", metadata_gate(&d));

    arm(&mut t, Kind::Null, "output: meta_type/unix/trim all zero", MetaP { zero: true, ..d });

    arm(&mut t, Kind::Structural, "unix <-> trim_cursor offsets swapped", MetaP {
        unix_off: 13,
        trim_off: 3,
        ..d
    });
    arm(&mut t, Kind::Structural, "both u32 read big-endian", MetaP { big_endian: true, ..d });
    arm(&mut t, Kind::Param, "trim_cursor offset 13 -> 14 (one byte)", MetaP { trim_off: 14, ..d });
    arm(&mut t, Kind::Param, "trim_cursor offset 13 -> 17 (+4, one word)", MetaP { trim_off: 17, ..d });
    arm(&mut t, Kind::Param, "meta_type offset 2 -> 3", MetaP { type_off: 3, ..d });

    // The shipped decoder must agree with the unmutated replica.
    let f = frame_from_hex(Family::Gen5, HISTORY_END_HEX);
    let m = live::metadata(&f).unwrap();
    assert_eq!(
        (m.meta_type, m.unix, m.trim_cursor),
        metadata_replica(&f, &d).unwrap(),
        "metadata replica drifted from shipped"
    );

    match std::env::var("WHOOP_CAPTURE") {
        Ok(path) => t.note(format!("WHOOP_CAPTURE is set ({path}); capture_parity.rs:155-159 is runnable")),
        Err(_) => t.note(
            "FIXTURE ABSENT — WHOOP_CAPTURE is unset and capture_parity.rs has NO default path, so \
             the whole-capture HISTORY_END/metadata/EVENT/CONSOLE/CMD_RESP parity claim \
             (capture_parity.rs:155-159) is UNRUN. The arms above score the single always-on \
             golden frame only: exactly ONE HISTORY_END is under test in this repo."
                .to_string(),
        ),
    }
    t
}

// =================================================================================================
// Metric 6 — Data-range scan (sync window)
// =================================================================================================

#[derive(Clone, Copy)]
struct RangeP {
    lo: u32,
    hi: u32,
    skew: u64,
    newest_stride: usize,
    prefer_not_future: bool,
    newest_takes_min: bool,
    grid_start: usize,
    grid_stride: usize,
    newest_const: bool,
}

impl Default for RangeP {
    /// crates/whoop-protocol/src/response.rs:28-29/40-72 plus the skew the gate passes in.
    fn default() -> Self {
        RangeP {
            lo: 1_700_000_000,
            hi: 1_900_000_000,
            skew: GATE_RANGE_SKEW,
            newest_stride: 1,
            prefer_not_future: true,
            newest_takes_min: false,
            grid_start: 7,
            grid_stride: 4,
            newest_const: false,
        }
    }
}

fn word_le(f: &[u8], i: usize) -> u32 {
    u32::from_le_bytes([f[i], f[i + 1], f[i + 2], f[i + 3]])
}

fn newest_replica(frame: &[u8], wall_now: u64, p: &RangeP) -> Option<u32> {
    if p.newest_const {
        return Some(p.lo);
    }
    let cutoff = wall_now.saturating_add(p.skew);
    let (mut not_future, mut any) = (None::<u32>, None::<u32>);
    let pick = |cur: Option<u32>, w: u32| {
        Some(cur.map_or(w, |m: u32| if p.newest_takes_min { m.min(w) } else { m.max(w) }))
    };
    let mut i = 0;
    while i + 4 <= frame.len() {
        let w = word_le(frame, i);
        if (p.lo..=p.hi).contains(&w) {
            any = pick(any, w);
            if u64::from(w) <= cutoff {
                not_future = pick(not_future, w);
            }
        }
        i += p.newest_stride;
    }
    if p.prefer_not_future {
        not_future.or(any)
    } else {
        any
    }
}

fn oldest_replica(frame: &[u8], p: &RangeP) -> Option<u32> {
    let mut oldest: Option<u32> = None;
    let mut i = p.grid_start;
    while i + 4 <= frame.len() {
        let w = word_le(frame, i);
        if (p.lo..=p.hi).contains(&w) {
            oldest = Some(oldest.map_or(w, |m| m.min(w)));
        }
        i += p.grid_stride;
    }
    oldest
}

fn range_gate(p: &RangeP) -> Check {
    let mut c = Check::default();
    for (h, want) in GATE_RANGE_CASES {
        let f = from_hex(h).unwrap();
        c.t(newest_replica(&f, GATE_RANGE_WALL_NOW, p) == Some(want));
    }
    c.t(oldest_replica(&from_hex(GATE_RANGE_CASES[0].0).unwrap(), p).is_none());
    let ffi = from_hex(GATE_RANGE_FFI_HEX).unwrap();
    let ffi_p = RangeP { skew: GATE_RANGE_FFI_SKEW, ..*p };
    c.t(newest_replica(&ffi, GATE_RANGE_FFI_WALL_NOW, &ffi_p) == Some(GATE_RANGE_FFI_NEWEST));
    c.t(oldest_replica(&ffi, &ffi_p) == Some(GATE_RANGE_FFI_OLDEST));
    c
}

fn metric_range() -> Table {
    let mut t = Table::new(
        "Data-range scan (oldest / newest offloadable record)",
        "capture_parity.rs:177 over 3 real 4.0 frames + :180 oldest == None, whoop-ffi lib.rs:274-275 \
         newest 1784236462 / oldest 1778385408",
    );
    let d = RangeP::default();
    let arm = |t: &mut Table, k: Kind, n: &str, p: RangeP| t.arm(k, n, range_gate(&p));

    t.arm(Kind::Baseline, "baseline (unmutated replica)", range_gate(&d));

    arm(&mut t, Kind::Null, "newest: always return PLAUSIBLE_LO (ignores the frame)", RangeP {
        newest_const: true,
        ..d
    });

    arm(&mut t, Kind::Structural, "newest takes the MINIMUM plausible word", RangeP {
        newest_takes_min: true,
        ..d
    });
    arm(&mut t, Kind::Structural, "oldest scans every offset (drops the aligned-from-7 rule)", RangeP {
        grid_stride: 1,
        ..d
    });
    arm(&mut t, Kind::Structural, "oldest grid start 7 -> 8", RangeP { grid_start: 8, ..d });
    arm(&mut t, Kind::Structural, "newest scans the 4-byte grid instead of every offset", RangeP {
        newest_stride: 4,
        ..d
    });

    arm(&mut t, Kind::Param, "future skew 48 h -> 52.8 h (+10%)", RangeP { skew: 190_080, ..d });
    arm(&mut t, Kind::Param, "future skew 48 h -> 0 (branch removed)", RangeP { skew: 0, ..d });
    arm(&mut t, Kind::Param, "prefer-not-future disabled (take newest-any)", RangeP {
        prefer_not_future: false,
        ..d
    });
    arm(&mut t, Kind::Param, "PLAUSIBLE_LO 1.70e9 -> 1.87e9 (+10%)", RangeP { lo: 1_870_000_000, ..d });
    arm(&mut t, Kind::Param, "PLAUSIBLE_LO 1.70e9 -> 1.7085e9 (+0.5%)", RangeP { lo: 1_708_500_000, ..d });
    arm(&mut t, Kind::Param, "PLAUSIBLE_LO 1.70e9 -> 1.779e9 (just under the target)", RangeP {
        lo: 1_779_000_000,
        ..d
    });
    arm(&mut t, Kind::Param, "PLAUSIBLE_HI 1.90e9 -> 2.09e9 (+10%)", RangeP { hi: 2_090_000_000, ..d });
    arm(&mut t, Kind::Param, "PLAUSIBLE_HI 1.90e9 -> 1.71e9 (-10%)", RangeP { hi: 1_710_000_000, ..d });

    // The shipped scans must agree with the unmutated replica.
    let f0 = from_hex(GATE_RANGE_CASES[0].0).unwrap();
    assert_eq!(
        data_range_scan_newest(&f0, GATE_RANGE_WALL_NOW, GATE_RANGE_SKEW),
        newest_replica(&f0, GATE_RANGE_WALL_NOW, &d),
        "newest replica drifted from shipped"
    );
    assert_eq!(data_range_scan_oldest(&f0), oldest_replica(&f0, &d), "oldest replica drifted from shipped");

    t.note(
        "every pinned target word is in the PAST of the wall clock the gate passes, so the \
         future-skew branch is never exercised by this gate."
            .to_string(),
    );
    t
}

// =================================================================================================
// Metric 7 — Flash trim magic words (destructive-command guard)
// =================================================================================================

#[derive(Clone, Copy)]
struct TrimP {
    reset: u32,
    erase: u32,
    ignore: u32,
    swap_words: bool,
    zero: bool,
}

impl Default for TrimP {
    /// crates/whoop-protocol/src/trim.rs:6-11.
    fn default() -> Self {
        TrimP { reset: RESET_TO_OLDEST, erase: TRIM_ALL, ignore: IGNORE, swap_words: false, zero: false }
    }
}

fn trim_words(p: &TrimP, page: u32, wrap: u32) -> [u8; 8] {
    let (a, b) = if p.swap_words { (wrap, page) } else { (page, wrap) };
    let mut out = [0u8; 8];
    out[..4].copy_from_slice(&a.to_le_bytes());
    out[4..].copy_from_slice(&b.to_le_bytes());
    out
}

fn trim_reset(p: &TrimP) -> [u8; 8] {
    if p.zero {
        return [0u8; 8];
    }
    trim_words(p, p.reset, p.reset)
}

fn trim_inert(p: &TrimP) -> [u8; 8] {
    trim_words(p, p.ignore, p.ignore)
}

fn trim_gate(p: &TrimP) -> Check {
    let mut c = Check::default();
    c.t(trim_reset(p) == GATE_TRIM_RESET_PAYLOAD); // trim.rs:42
    c.t(p.reset != p.erase); // trim.rs:47 and the compile-time const at :14
    c.t(trim_reset(p) != trim_words(p, p.erase, p.erase)); // trim.rs:48
    c.t(trim_words(p, p.erase, p.erase) == GATE_TRIM_ERASE_PAYLOAD); // trim.rs:49
    c.t(trim_inert(p)[..4] == p.ignore.to_le_bytes()); // trim.rs:55
    c.t(trim_inert(p) != GATE_TRIM_RESET_PAYLOAD); // trim.rs:56
    c.t(trim_inert(p) != GATE_TRIM_ERASE_PAYLOAD); // trim.rs:57
    c.t(p.reset.to_le_bytes()[0] == 0xFD && p.erase.to_le_bytes()[0] == 0xFE); // trim.rs:15
    c
}

fn metric_trim() -> Table {
    let mut t = Table::new(
        "Flash trim magic words (destructive-command guard)",
        "trim.rs:42 reset == [0xFD; 8], :47-49 reset != erase and erase == [0xFE; 8], :55-57 the \
         inert probe, :14-15 the compile-time separation",
    );
    let d = TrimP::default();
    let arm = |t: &mut Table, k: Kind, n: &str, p: TrimP| t.arm(k, n, trim_gate(&p));

    t.arm(Kind::Baseline, "baseline (unmutated replica)", trim_gate(&d));

    arm(&mut t, Kind::Null, "reset builder emits eight zero bytes", TrimP { zero: true, ..d });

    arm(&mut t, Kind::Structural, "RESET_TO_OLDEST <-> TRIM_ALL transposed", TrimP {
        reset: TRIM_ALL,
        erase: RESET_TO_OLDEST,
        ..d
    });
    arm(&mut t, Kind::Structural, "page/wrap word order swapped", TrimP { swap_words: true, ..d });

    arm(&mut t, Kind::Param, "RESET 0xFDFDFDFD -> 0xFDFDFDFE (one byte drifts to the erase byte)", TrimP {
        reset: 0xFDFD_FDFE,
        ..d
    });
    arm(&mut t, Kind::Param, "RESET 0xFDFDFDFD -> 0xFDFDFDFC (-1)", TrimP { reset: 0xFDFD_FDFC, ..d });
    arm(&mut t, Kind::Param, "TRIM_ALL 0xFEFEFEFE -> 0xFEFEFEFF (+1)", TrimP { erase: 0xFEFE_FEFF, ..d });
    arm(&mut t, Kind::Param, "IGNORE 0xFFFFFFFF -> 0xFFFFFFFE (one bit)", TrimP { ignore: 0xFFFF_FFFE, ..d });

    // The shipped builders must agree with the unmutated replica.
    assert_eq!(reset_to_oldest(), trim_reset(&d), "trim replica drifted from shipped");
    assert_eq!(inert_probe(), trim_inert(&d), "inert replica drifted from shipped");

    let bits: u32 = GATE_TRIM_RESET_PAYLOAD
        .iter()
        .zip(GATE_TRIM_ERASE_PAYLOAD.iter())
        .map(|(a, b)| (a ^ b).count_ones())
        .sum();
    t.note(format!(
        "measured separation reset vs erase payload: {bits} bits over 8 bytes ({} per byte), \
         all 8 bytes differ",
        bits / 8
    ));
    t.note(
        "`words` is private, so the page/wrap ORDER is only observable through builders whose two \
         words are identical: the word-order arm cannot be falsified from outside the module."
            .to_string(),
    );
    t
}

// =================================================================================================
// Metric 8 — Advertising parse + strap variant identification
// =================================================================================================

#[derive(Clone, Copy)]
struct IdentP {
    max_name_bytes: usize,
    lead_gen4: u8,
    lead_gen5: u8,
    mg_prefix: &'static str,
    w5_prefix: &'static str,
    contains: bool,
    prefix_len: usize,
    always_whoop5: bool,
}

impl Default for IdentP {
    /// crates/whoop-protocol/src/advertising.rs:6/33-44 and variant.rs:47-49/68-81.
    fn default() -> Self {
        IdentP {
            max_name_bytes: 24,
            lead_gen4: 0x00,
            lead_gen5: 0x01,
            mg_prefix: "WS50",
            w5_prefix: "WG50",
            contains: false,
            prefix_len: 4,
            always_whoop5: false,
        }
    }
}

fn clamp_name(name: &str, max: usize) -> &str {
    let mut end = name.len();
    while end > max {
        end -= 1;
        while !name.is_char_boundary(end) {
            end -= 1;
        }
    }
    &name[..end]
}

fn adv_body(lead: u8, name: &str, max: usize) -> Vec<u8> {
    let clamped = clamp_name(name, max);
    let mut out = vec![lead, 0];
    out.extend_from_slice(clamped.as_bytes());
    out.push(0);
    out
}

fn classify_replica(hw: &str, family: Family, p: &IdentP) -> Variant {
    if p.always_whoop5 {
        return Variant::Whoop5;
    }
    let h = hw.trim();
    let hit = |pref: &str| {
        let pref = &pref[..p.prefix_len.min(pref.len())];
        if p.contains {
            h.contains(pref)
        } else {
            h.starts_with(pref)
        }
    };
    if hit(p.mg_prefix) {
        return Variant::WhoopMg;
    }
    if hit(p.w5_prefix) {
        return Variant::Whoop5;
    }
    match family {
        Family::Gen4 => Variant::Whoop4,
        Family::Gen5 => Variant::Unknown,
    }
}

fn ident_gate(p: &IdentP) -> Check {
    let mut c = Check::default();
    // advertising.rs:52-79
    c.t(adv_body(p.lead_gen4, "noop", p.max_name_bytes) == b"\x00\x00noop\x00".to_vec());
    let long4 = adv_body(p.lead_gen4, &"a".repeat(40), p.max_name_bytes);
    c.t(long4.len() == GATE_ADV_CLAMPED_LEN);
    c.t(long4.get(2..26) == Some(&"a".repeat(24).into_bytes()[..]));
    c.t(long4.last() == Some(&0));
    let multi = adv_body(p.lead_gen4, &"\u{e9}".repeat(13), p.max_name_bytes);
    c.t(multi.len() == 2 + 24 + 1);
    c.t(std::str::from_utf8(&multi[2..multi.len() - 1]).is_ok());
    c.t(adv_body(p.lead_gen5, "noop", p.max_name_bytes) == b"\x01\x00noop\x00".to_vec());
    let long5 = adv_body(p.lead_gen5, &"a".repeat(40), p.max_name_bytes);
    c.t(long5.len() == GATE_ADV_CLAMPED_LEN && long5[0] == 0x01);

    // variant.rs:158-215
    for (hw, model, want) in GATE_OBSERVED {
        let v = classify_replica(hw, Family::Gen5, p);
        c.t(v == want);
        c.t(v.model_agrees(model) == Some(true));
    }
    c.t(classify_replica("WS50_r00", Family::Gen5, p).model_agrees("5.0") == Some(false));
    c.t(classify_replica("WS50_r03", Family::Gen5, p).has_ecg());
    c.t(!classify_replica("WG50_r45", Family::Gen5, p).has_ecg());
    c.t(!classify_replica("", Family::Gen4, p).has_ecg());
    c.t(!classify_replica("", Family::Gen5, p).has_ecg());
    c.t(classify_replica("WX99_r01", Family::Gen5, p) == Variant::Unknown);
    c.t(classify_replica("", Family::Gen4, p) == Variant::Whoop4);
    for rev in ["WS50_r01", "WS50_r03", "WS50_r99", " WS50_r03 "] {
        c.t(classify_replica(rev, Family::Gen5, p) == Variant::WhoopMg);
    }
    c
}

fn metric_identity() -> Table {
    let mut t = Table::new(
        "Advertising parse + strap variant identification",
        "advertising.rs:52-79 (5 tests) + variant.rs:158-215 (6 tests) — the name clamp and the \
         WS50 = MG / WG50 = 5.0 prefix table that gates ECG",
    );
    let d = IdentP::default();
    let arm = |t: &mut Table, k: Kind, n: &str, p: IdentP| t.arm(k, n, ident_gate(&p));

    t.arm(Kind::Baseline, "baseline (unmutated replica)", ident_gate(&d));

    arm(&mut t, Kind::Null, "classify: every strap is a 5.0 (identity never read)", IdentP {
        always_whoop5: true,
        ..d
    });

    arm(&mut t, Kind::Structural, "WS50 <-> WG50 prefixes transposed (ECG follows the wrong board)", IdentP {
        mg_prefix: "WG50",
        w5_prefix: "WS50",
        ..d
    });
    arm(&mut t, Kind::Structural, "prefix match starts_with -> contains", IdentP { contains: true, ..d });

    arm(&mut t, Kind::Param, "MAX_NAME_BYTES 24 -> 26 (+10%)", IdentP { max_name_bytes: 26, ..d });
    arm(&mut t, Kind::Param, "MAX_NAME_BYTES 24 -> 25 (+4%, the integer floor)", IdentP {
        max_name_bytes: 25,
        ..d
    });
    arm(&mut t, Kind::Param, "prefix length 4 -> 2 (compare 'WS'/'WG' only)", IdentP { prefix_len: 2, ..d });
    arm(&mut t, Kind::Param, "gen5 lead selector 0x01 -> 0x02", IdentP { lead_gen5: 0x02, ..d });

    // The shipped builders/classifier must agree with the unmutated replica.
    assert_eq!(advertising_name_payload("noop"), adv_body(d.lead_gen4, "noop", d.max_name_bytes));
    assert_eq!(advertising_name_payload_gen5("noop"), adv_body(d.lead_gen5, "noop", d.max_name_bytes));
    for (hw, _, want) in GATE_OBSERVED {
        assert_eq!(Variant::classify(hw, Family::Gen5), want, "variant replica reference {hw}");
        assert_eq!(classify_replica(hw, Family::Gen5, &d), want, "variant replica drifted from shipped");
    }
    t
}

// =================================================================================================
// The run
// =================================================================================================

/// The replica must BE the algorithm, or every arm above scores a different function. Compares the
/// unmutated replica against the shipped decoder on every fixture frame, field for field.
fn assert_replicas_reproduce_the_shipped_decoders() {
    let o = oracle();
    let (v18p, gen4p, v26p) = (V18P::default(), Gen4P::default(), V26P::default());
    for f in o["frames"].as_array().unwrap() {
        let name = f["name"].as_str().unwrap();
        let frame = fixture_frame(f);
        let shipped = match shipped_decode(&frame) {
            Some(Record::History(h)) => hist_of_shipped(&h),
            other => panic!("{name}: expected a History record, got {other:?}"),
        };
        let mine = match fixture_family(f) {
            Family::Gen5 => v18_replica(&frame, &v18p),
            Family::Gen4 => gen4_replica(&frame, &gen4p),
        }
        .unwrap_or_else(|| panic!("{name}: replica refused a frame the shipped decoder accepted"));
        assert_eq!(mine, shipped, "{name}: replica != shipped decoder");
    }
    for f in o["ppg_frames"].as_array().unwrap() {
        let frame = fixture_frame(f);
        let shipped = match shipped_decode(&frame) {
            Some(Record::Ppg(p)) => p,
            other => panic!("v26: expected a Ppg record, got {other:?}"),
        };
        let mine = v26_replica(&frame, &v26p).expect("v26 replica refused a real frame");
        assert_eq!((mine.unix, mine.rid, mine.samples), (shipped.unix, shipped.record_id, shipped.samples));
    }
}

// ── Sensitivity floors ─────────────────────────────────────────────────────────────────────────

/// `(metric, arm, minimum |delta| from the baseline)`. A floor asserts the arm still MOVES the number,
/// which is what catches an algorithm that stopped being reached; each is 0.45x the delta measured
/// 2026-08-02, so it sits well below the observed move and well above zero.
const FLOORS: &[(&str, &str, f64)] = &[
    ("gen5 record decode (v18 / v20 / v21 / v26)", "output: every field absent (constant record)", 0.0685),
    ("gen5 record decode (v18 / v20 / v21 / v26)", "output: v26 waveform = 24 zeros, unix 0", 0.173),
    ("gen5 record decode (v18 / v20 / v21 / v26)", "v18: every offset slipped +1 byte", 0.0492),
    ("gen5 record decode (v18 / v20 / v21 / v26)", "v18: optical baselines A<->B swapped", 0.00643),
    ("gen5 record decode (v18 / v20 / v21 / v26)", "v18: thermal aux 1 <-> aux 2 swapped", 0.00427),
    ("gen5 record decode (v18 / v20 / v21 / v26)", "v18: gravity axes reversed (x,y,z -> z,y,x)", 0.00215),
    ("gen5 record decode (v18 / v20 / v21 / v26)", "v18: R-R slot order reversed", 0.00215),
    (
        "gen5 record decode (v18 / v20 / v21 / v26)",
        "v18: activity_class offset 55 -> 56 (that field alone)",
        0.00409,
    ),
    ("gen5 record decode (v18 / v20 / v21 / v26)", "v26: waveform reversed", 0.00215),
    ("gen5 record decode (v18 / v20 / v21 / v26)", "v26: waveform shifted +1 sample", 0.00215),
    ("gen5 record decode (v18 / v20 / v21 / v26)", "v21: decoder offsets +2, synthetic buffer unchanged", 0.0128),
    ("gen4 record decode (v5 / v24 / v25)", "output: every field absent (constant record)", 0.337),
    ("gen4 record decode (v5 / v24 / v25)", "v24: SpO2 red <-> IR swapped", 0.0281),
    ("gen4 record decode (v5 / v24 / v25)", "v25: gravity read big-endian (stride kept)", 0.0843),
    ("CRC (modbus-16, zlib-32, crc8)", "output: all three checksums return 0", 0.3),
    ("CRC (modbus-16, zlib-32, crc8)", "input byte order reversed for all three", 0.3),
    ("CRC (modbus-16, zlib-32, crc8)", "output: crc8 alone returns 0 (does no work)", 0.03),
    ("Framing / deframing (header, CRC placement, reassembly)", "crc_ok always true (checksum never consulted)", 0.0818),
    ("Framing / deframing (header, CRC placement, reassembly)", "declared length written/read big-endian", 0.368),
    ("Framing / deframing (header, CRC placement, reassembly)", "gen5 inner_start 8 -> 9", 0.327),
    ("Framing / deframing (header, CRC placement, reassembly)", "gen5 len_offset 2 -> 3", 0.327),
    ("Framing / deframing (header, CRC placement, reassembly)", "SOF 0xAA -> 0xAB", 0.368),
    ("HISTORY_END trim cursor + metadata parity", "output: meta_type/unix/trim all zero", 0.257),
    ("HISTORY_END trim cursor + metadata parity", "unix <-> trim_cursor offsets swapped", 0.128),
    ("HISTORY_END trim cursor + metadata parity", "both u32 read big-endian", 0.128),
    ("Data-range scan (oldest / newest offloadable record)", "newest: always return PLAUSIBLE_LO (ignores the frame)", 0.3),
    ("Data-range scan (oldest / newest offloadable record)", "newest takes the MINIMUM plausible word", 0.3),
    ("Data-range scan (oldest / newest offloadable record)", "oldest scans every offset (drops the aligned-from-7 rule)", 0.149),
    ("Data-range scan (oldest / newest offloadable record)", "oldest grid start 7 -> 8", 0.149),
    ("Data-range scan (oldest / newest offloadable record)", "newest scans the 4-byte grid instead of every offset", 0.075),
    ("Flash trim magic words (destructive-command guard)", "reset builder emits eight zero bytes", 0.0562),
    ("Flash trim magic words (destructive-command guard)", "RESET_TO_OLDEST <-> TRIM_ALL transposed", 0.168),
    ("Advertising parse + strap variant identification", "classify: every strap is a 5.0 (identity never read)", 0.199),
    ("Advertising parse + strap variant identification", "WS50 <-> WG50 prefixes transposed (ECG follows the wrong board)", 0.25),
];

/// `(metric, arm, why)`. Probe arms that cannot carry a floor, because the mutation does not move the
/// number at all. Their blindness is the finding, not a defect to assert away.
const NO_FLOOR: &[(&str, &str, &str)] = &[
    ("gen5 record decode (v18 / v20 / v21 / v26)", "v18: R-R slots 4 -> 3 (drop the last slot)", "measured delta is exactly zero: this mutation does not move the number"),
    ("gen5 record decode (v18 / v20 / v21 / v26)", "v21: buffer AND decoder offsets +2 (coordinated)", "measured delta is exactly zero: this mutation does not move the number"),
    ("gen4 record decode (v5 / v24 / v25)", "v25: gravity axes reversed (x,y,z -> z,y,x)", "measured delta is exactly zero: this mutation does not move the number"),
    ("gen4 record decode (v5 / v24 / v25)", "v24: gravity axes reversed", "measured delta is exactly zero: this mutation does not move the number"),
    ("gen4 record decode (v5 / v24 / v25)", "v24: R-R slots 4 -> 3 (drop the last slot)", "measured delta is exactly zero: this mutation does not move the number"),
    ("Flash trim magic words (destructive-command guard)", "page/wrap word order swapped", "measured delta is exactly zero: this mutation does not move the number"),
    ("Advertising parse + strap variant identification", "prefix match starts_with -> contains", "measured delta is exactly zero: this mutation does not move the number"),
];

/// Assert one metric's floors, and require every NULL/STRUCTURAL arm to be classified.
fn enforce_floors(metric: &str, base: f64, probes: &[(&str, f64)]) {
    let (mut asserted, mut waived) = (0usize, 0usize);
    let mut breached: Vec<String> = Vec::new();
    let mut unclassified: Vec<&str> = Vec::new();
    for &(arm, value) in probes {
        let floor = FLOORS.iter().find(|(m, a, _)| *m == metric && *a == arm).map(|t| t.2);
        let waiver = NO_FLOOR.iter().find(|(m, a, _)| *m == metric && *a == arm).map(|t| t.2);
        match (floor, waiver) {
            (Some(_), Some(_)) => breached.push(format!("'{arm}' carries both a floor and a waiver")),
            (Some(d), None) => {
                asserted += 1;
                let moved = (value - base).abs();
                if moved.is_nan() || moved < d {
                    breached.push(format!("'{arm}' moved {moved} against a floor of {d}"));
                }
            }
            (None, Some(w)) => {
                waived += 1;
                println!("   no floor: {arm} — {w}");
            }
            (None, None) => unclassified.push(arm),
        }
    }
    let orphans: Vec<&str> = FLOORS
        .iter()
        .filter(|(m, _, _)| *m == metric)
        .map(|t| t.1)
        .chain(NO_FLOOR.iter().filter(|(m, _, _)| *m == metric).map(|t| t.1))
        .filter(|a| !probes.iter().any(|(p, _)| *p == *a))
        .collect();
    println!("   floors: {asserted} asserted, {waived} un-floorable");
    assert!(
        unclassified.is_empty(),
        "{metric}: probe arms carry neither a floor nor a waiver — classify them: {unclassified:?}"
    );
    assert!(orphans.is_empty(), "{metric}: floor rows match no arm — stale or misspelt: {orphans:?}");
    assert!(breached.is_empty(), "{metric}: SENSITIVITY FLOOR BREACHED — {}", breached.join(" | "));
}

/// Reachable from a clean checkout: the fixture it reads is tracked beside it and pulled in with
/// `include_str!`, so nothing here is data-gated. It is `#[ignore]`d only because it is a report, not
/// a regression check — the numbers it prints are the deliverable and CI has no use for them.
#[test]
#[ignore = "negative control, not data-gated: the fixture is tracked and include_str!'d. Ignored because \
            this prints a mutation report rather than gating a build. Run: cargo test --release -p \
            whoop-protocol --test sensitivity_decode -- --ignored --nocapture"]
fn sensitivity_decode() {
    assert_fixture_cohort(&oracle());
    assert_replicas_reproduce_the_shipped_decoders();

    let tables = [
        metric_gen5(),
        metric_gen4(),
        metric_crc(),
        metric_framing(),
        metric_metadata(),
        metric_range(),
        metric_trim(),
        metric_identity(),
    ];

    println!("\n######## negative controls — decode family ########");
    let summaries: Vec<Summary> = tables.iter().map(Table::report).collect();

    println!("\n== family headline ==");
    println!("   {:<46} {:>7} {:>7} {:>12}", "metric", "caught", "missed", "floor");
    let (mut caught, mut missed, mut critical) = (0usize, 0usize, 0usize);
    for s in &summaries {
        let floor = s.floor.map_or("none".to_string(), |f| format!("{f:.4}"));
        println!("   {:<46} {:>7} {:>7} {:>12}", s.metric, s.caught, s.missed, floor);
        caught += s.caught;
        missed += s.missed;
        critical += s.critical_nulls;
    }
    println!("\n   DECODE FAMILY: caught {caught}, missed {missed}");
    if critical > 0 {
        println!("   CRITICAL: {critical} NULL arm(s) passed a shipped gate — see the per-metric lines above");
    }

    // Only the two trustworthiness claims are asserted. A parameter arm that passes is a measurement.
    for s in &summaries {
        assert!(s.baseline_passes, "{}: the baseline arm does not reproduce the shipped gate", s.metric);
        assert!(
            s.some_null_failed,
            "{}: no NULL arm moved the gate — the harness does not reach the algorithm",
            s.metric
        );
    }
}
