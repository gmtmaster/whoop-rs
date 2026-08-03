//! Negative control for the STRAIN metric family: strain map, denominator fit, HR-max, zone edges,
//! time-in-zone, gap ceilings, calories, steps, workout, HR recovery, VO2max/fitness age, IMU features.
//!
//! The claim this file falsifies: "the shipped gates for this family would catch a regression."
//! Each shipped gate's exact target and tolerance is copied here with the `file:line` it came from,
//! then re-evaluated against mutated arms — a NULL arm that does no work, STRUCTURAL arms that get the
//! shape wrong, and a PARAMETER arm per tunable the algorithm reads. Only two things are asserted:
//! the baseline reproduces, and the do-nothing null fails. A parameter arm PASSING is the
//! measurement, not an error, so it is printed and counted, never asserted.
//!
//! Diagnostic only, never a CI gate: `#[ignore]`d, prints `caught N, missed M` per metric plus the
//! smallest delta each gate catches. Every number here is a wellness estimate, never medical.

use physio_algo::calories;
use physio_algo::hr_gap::{self, GapPosition};
use physio_algo::hr_recovery::{self, HrRecovery};
use physio_algo::hr_sample::HrSample;
use physio_algo::hr_zones::{self, HrZone, HrZoneSet, TimeInZone};
use physio_algo::imu_features::{self, ImuActivityFeatures, ImuSample};
use physio_algo::sleep::StepSample;
use physio_algo::steps;
use physio_algo::strain::{self, Method};
use physio_algo::vo2max;
use physio_algo::workout::{self, ActivityPoint, GravitySample};

// ── Harness ────────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Baseline,
    Null,
    Structural,
    Param,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Verdict {
    Pass,
    Fail,
    NoGate,
}

struct Arm {
    name: String,
    kind: Kind,
    value: f64,
    verdict: Verdict,
}

struct Table {
    metric: &'static str,
    gate: &'static str,
    arms: Vec<Arm>,
}

#[derive(Default, Clone, Copy)]
struct Tally {
    caught: usize,
    missed: usize,
    ungated: usize,
    floor: Option<f64>,
    caught_without_delta: usize,
}

impl Table {
    fn new(metric: &'static str, gate: &'static str) -> Self {
        Table { metric, gate, arms: Vec::new() }
    }

    fn arm(&mut self, name: impl Into<String>, kind: Kind, value: f64, pass: bool) {
        let verdict = if pass { Verdict::Pass } else { Verdict::Fail };
        self.arms.push(Arm { name: name.into(), kind, value, verdict });
    }

}

fn num(v: f64) -> String {
    if v.is_nan() {
        "none".to_string()
    } else {
        format!("{v:.4}")
    }
}

fn signed(v: f64) -> String {
    if v.is_nan() {
        "--".to_string()
    } else {
        format!("{v:+.4}")
    }
}

/// Prints one metric's arm table and returns its tally. The delta column is against the baseline arm.
fn print_table(t: &Table) -> Tally {
    let base = t.arms.iter().find(|a| a.kind == Kind::Baseline).map(|a| a.value).unwrap_or(f64::NAN);
    println!("\nmetric: {}", t.metric);
    println!("gate:   {}", t.gate);
    println!("  {:<62} {:>11} {:>11}  shipped gate", "arm", "value", "delta");
    let mut tally = Tally::default();
    for a in &t.arms {
        let delta = a.value - base;
        let mark = match (a.verdict, a.kind) {
            (Verdict::Pass, Kind::Baseline) => "PASS (expected)",
            (Verdict::Fail, Kind::Baseline) => "FAIL  <-- BASELINE BROKEN",
            (Verdict::Pass, _) => "PASS  <-- MISSED",
            (Verdict::Fail, _) => "FAIL  <-- caught",
            (Verdict::NoGate, _) => "n/a   <-- NO GATE",
        };
        println!("  {:<62} {:>11} {:>11}  {}", a.name, num(a.value), signed(delta), mark);
        if a.kind == Kind::Baseline {
            continue;
        }
        match a.verdict {
            Verdict::Pass => tally.missed += 1,
            Verdict::NoGate => tally.ungated += 1,
            Verdict::Fail => {
                tally.caught += 1;
                if delta.is_finite() && delta != 0.0 {
                    let d = delta.abs();
                    tally.floor = Some(tally.floor.map_or(d, |f: f64| f.min(d)));
                } else {
                    tally.caught_without_delta += 1;
                }
            }
        }
    }
    let floor = tally.floor.map_or("n/a".to_string(), |f| format!("{f:.4}"));
    println!(
        "  caught {}, missed {}, ungated {} | smallest caught |delta| {} ({} caught with no move in the printed scalar)",
        tally.caught, tally.missed, tally.ungated, floor, tally.caught_without_delta
    );
    for a in &t.arms {
        if a.kind == Kind::Null && a.verdict == Verdict::Pass {
            println!("  CRITICAL: null arm PASSED the shipped gate -> {}", a.name);
        }
    }
    let probes: Vec<(&str, f64)> = t
        .arms
        .iter()
        .filter(|a| matches!(a.kind, Kind::Null | Kind::Structural))
        .map(|a| (a.name.as_str(), a.value))
        .collect();
    enforce_floors(t.metric, base, &probes);
    tally
}

// ── Shared fixtures ────────────────────────────────────────────────────────────

fn hr_constant(bpm: i32, n: usize) -> Vec<HrSample> {
    (0..n).map(|i| HrSample { ts: i as i64, bpm }).collect()
}

fn hr_every(bpm: i32, n: usize, step_s: i64) -> Vec<HrSample> {
    (0..n).map(|i| HrSample { ts: i as i64 * step_s, bpm }).collect()
}

fn round2(v: f64) -> f64 {
    (v * 100.0).round() / 100.0
}

// ── 1. Strain: TRIMP -> 0-100 map ──────────────────────────────────────────────

/// strain.rs:244 EPS; strain.rs:262-266 goldens.
const STRAIN_EPS: f64 = 0.01;
const TRIMP_GOLDENS: [(f64, f64); 5] =
    [(100.0, 51.96), (500.0, 69.99), (1000.0, 77.78), (3600.0, 92.20), (7200.0, 100.0)];

fn map_arm(t: &mut Table, name: &str, kind: Kind, f: impl Fn(f64) -> f64) {
    let pass = TRIMP_GOLDENS.iter().all(|(x, want)| (f(*x) - want).abs() < STRAIN_EPS);
    t.arm(name, kind, f(1000.0), pass);
}

fn trimp_map_table() -> Table {
    let d = strain::STRAIN_DENOMINATOR;
    let mut t = Table::new(
        "Strain: TRIMP -> 0-100 map (trimp_to_strain)",
        "strain.rs:262-266 — 100->51.96, 500->69.99, 1000->77.78, 3600->92.20, 7200->100.0, EPS 0.01",
    );
    map_arm(&mut t, "baseline (unmutated)", Kind::Baseline, |x| strain::trimp_to_strain(x, d));
    map_arm(&mut t, "null[hard]: output pinned to 0", Kind::Null, |_| 0.0);
    map_arm(&mut t, "null: constant scorer (always the 100-TRIMP answer)", Kind::Null, |_| 51.96);
    map_arm(&mut t, "struct: linear map instead of logarithmic", Kind::Structural, |x| {
        round2((strain::MAX_STRAIN * x / 7200.0).min(strain::MAX_STRAIN))
    });
    map_arm(&mut t, "struct: ln(TRIMP) instead of ln(TRIMP+1)", Kind::Structural, |x| {
        if x <= 0.0 {
            0.0
        } else {
            round2(strain::MAX_STRAIN * x.ln() / d.ln())
        }
    });
    map_arm(&mut t, "struct: rounded to whole points instead of 2 dp", Kind::Structural, |x| {
        strain::trimp_to_strain(x, d).round()
    });
    map_arm(&mut t, "param: STRAIN_DENOMINATOR 7201 -> 7921.1 (+10%)", Kind::Param, |x| {
        strain::trimp_to_strain(x, d * 1.1)
    });
    map_arm(&mut t, "param: STRAIN_DENOMINATOR 7201 -> 6480.9 (-10%)", Kind::Param, |x| {
        strain::trimp_to_strain(x, d * 0.9)
    });
    map_arm(&mut t, "param: STRAIN_DENOMINATOR +0.5%", Kind::Param, |x| {
        strain::trimp_to_strain(x, d * 1.005)
    });
    map_arm(&mut t, "param: STRAIN_DENOMINATOR +0.1% (floor probe)", Kind::Param, |x| {
        strain::trimp_to_strain(x, d * 1.001)
    });
    // MAX_STRAIN is baked in, but D^(100/110) is an exact reparameterisation of a 110-point ceiling.
    map_arm(&mut t, "param: MAX_STRAIN 100 -> 110 (exact D reparameterisation)", Kind::Param, |x| {
        strain::trimp_to_strain(x, d.powf(strain::MAX_STRAIN / (strain::MAX_STRAIN * 1.1)))
    });
    t
}

// ── 2. Strain: Edwards zone goldens ────────────────────────────────────────────

/// strain.rs:274-278 — 600 samples at 1 Hz, max 160, resting 60.
const EDWARDS_GOLDENS: [(i32, f64); 3] = [(115, 27.0), (135, 38.66), (155, 44.27)];

fn eff(hr: &[HrSample], max_hr: f64, resting: f64) -> Option<f64> {
    strain::strain(hr, Some(max_hr), resting, Method::Edwards, "male", strain::STRAIN_DENOMINATOR)
}

fn edwards_arm(t: &mut Table, name: &str, kind: Kind, f: impl Fn(i32) -> Option<f64>) {
    let pass = EDWARDS_GOLDENS
        .iter()
        .all(|(bpm, want)| matches!(f(*bpm), Some(v) if (v - want).abs() < STRAIN_EPS));
    t.arm(name, kind, f(135).unwrap_or(f64::NAN), pass);
}

fn edwards_table() -> Table {
    let mut t = Table::new(
        "Strain: Edwards zone goldens (strain, Method::Edwards)",
        "strain.rs:274-278 — 115bpm->27.0, 135bpm->38.66, 155bpm->44.27, EPS 0.01",
    );
    edwards_arm(&mut t, "baseline (unmutated)", Kind::Baseline, |bpm| {
        eff(&hr_constant(bpm, 600), 160.0, 60.0)
    });
    edwards_arm(&mut t, "null[hard]: output pinned to 0", Kind::Null, |_| Some(0.0));
    edwards_arm(&mut t, "null: HR ignored (every sample at the resting value)", Kind::Null, |_| {
        eff(&hr_constant(60, 600), 160.0, 60.0)
    });
    edwards_arm(&mut t, "null: constant scorer (always the 115-bpm answer)", Kind::Null, |_| {
        eff(&hr_constant(115, 600), 160.0, 60.0)
    });
    edwards_arm(&mut t, "struct: every bpm +1 (inside its zone)", Kind::Structural, |bpm| {
        eff(&hr_constant(bpm + 1, 600), 160.0, 60.0)
    });
    edwards_arm(&mut t, "struct: timestamps reversed (series runs backwards)", Kind::Structural, |bpm| {
        let hr: Vec<HrSample> = (0..600).map(|i| HrSample { ts: 599 - i as i64, bpm }).collect();
        eff(&hr, 160.0, 60.0)
    });
    edwards_arm(&mut t, "struct: cadence halved (301 samples, 2 s, same 600 s span)", Kind::Structural, |bpm| {
        eff(&hr_every(bpm, 301, 2), 160.0, 60.0)
    });
    edwards_arm(&mut t, "struct: last 10% of the series dropped", Kind::Structural, |bpm| {
        eff(&hr_constant(bpm, 540), 160.0, 60.0)
    });
    // Scaling HR reserve is exactly equivalent to scaling every EDWARDS_ZONES threshold.
    edwards_arm(&mut t, "param: EDWARDS_ZONES all +10% (exact, via HR reserve)", Kind::Param, |bpm| {
        eff(&hr_constant(bpm, 600), 170.0, 60.0)
    });
    edwards_arm(&mut t, "param: EDWARDS_ZONES all -10% (exact, via HR reserve)", Kind::Param, |bpm| {
        eff(&hr_constant(bpm, 600), 150.0, 60.0)
    });
    edwards_arm(&mut t, "param: EDWARDS_ZONES all +0.5% (exact, via HR reserve)", Kind::Param, |bpm| {
        eff(&hr_constant(bpm, 600), 160.5, 60.0)
    });
    edwards_arm(&mut t, "param: STRAIN_DENOMINATOR +10%", Kind::Param, |bpm| {
        strain::strain(
            &hr_constant(bpm, 600),
            Some(160.0),
            60.0,
            Method::Edwards,
            "male",
            strain::STRAIN_DENOMINATOR * 1.1,
        )
    });
    edwards_arm(&mut t, "param: MIN_READINGS 600 -> 660 (proxy: fixture cut to 545)", Kind::Param, |bpm| {
        eff(&hr_constant(bpm, 545), 160.0, 60.0)
    });
    edwards_arm(&mut t, "param: Banister instead of Edwards accumulation", Kind::Param, |bpm| {
        strain::strain(
            &hr_constant(bpm, 600),
            Some(160.0),
            60.0,
            Method::Banister,
            "male",
            strain::STRAIN_DENOMINATOR,
        )
    });
    t
}

// ── 3. Strain: denominator fit ─────────────────────────────────────────────────

/// strain.rs:418 — round-trip: `|d - STRAIN_DENOMINATOR| / STRAIN_DENOMINATOR < 0.01`.
const FIT_TOL_REL: f64 = 0.01;

fn fit_arm(t: &mut Table, name: &str, kind: Kind, pairs: &[(f64, f64)]) {
    let d = strain::STRAIN_DENOMINATOR;
    let got = strain::fit_strain_denominator(pairs);
    let value = got.unwrap_or(f64::NAN);
    let pass = value.is_finite() && (value - d).abs() / d < FIT_TOL_REL;
    t.arm(name, kind, value, pass);
}

fn seeded_pairs(seed_d: f64, strain_scale: f64) -> Vec<(f64, f64)> {
    [100.0, 500.0, 1000.0, 3600.0]
        .iter()
        .map(|&x| (x, strain::trimp_to_strain(x, seed_d) * strain_scale))
        .collect()
}

fn fit_table() -> Table {
    let d = strain::STRAIN_DENOMINATOR;
    let mut t = Table::new(
        "Strain: personal denominator fit (fit_strain_denominator)",
        "strain.rs:418 — |d - 7201| / 7201 < 0.01 on strains generated from 7201 (a round trip)",
    );
    fit_arm(&mut t, "baseline (unmutated)", Kind::Baseline, &seeded_pairs(d, 1.0));
    fit_arm(
        &mut t,
        "null[hard]: reference strains carry no information (all 50.0)",
        Kind::Null,
        &[(100.0, 50.0), (500.0, 50.0), (1000.0, 50.0), (3600.0, 50.0)],
    );
    {
        let mut shuffled = seeded_pairs(d, 1.0);
        let strains: Vec<f64> = shuffled.iter().rev().map(|(_, s)| *s).collect();
        for (i, p) in shuffled.iter_mut().enumerate() {
            p.1 = strains[i];
        }
        fit_arm(&mut t, "null: TRIMP/strain pairing shuffled (reversed)", Kind::Null, &shuffled);
    }
    fit_arm(&mut t, "struct: only the two smallest pairs kept", Kind::Structural, &seeded_pairs(d, 1.0)[..2]);
    fit_arm(&mut t, "struct: reference strains offset +2%", Kind::Structural, &seeded_pairs(d, 1.02));
    fit_arm(&mut t, "struct: reference strains offset +0.2%", Kind::Structural, &seeded_pairs(d, 1.002));
    fit_arm(&mut t, "struct: reference strains offset +0.1% (floor probe)", Kind::Structural, &seeded_pairs(d, 1.001));
    fit_arm(&mut t, "param: seed denominator +10%", Kind::Param, &seeded_pairs(d * 1.1, 1.0));
    fit_arm(&mut t, "param: seed denominator +1%", Kind::Param, &seeded_pairs(d * 1.01, 1.0));
    fit_arm(&mut t, "param: seed denominator +0.5%", Kind::Param, &seeded_pairs(d * 1.005, 1.0));
    t
}

// ── 4. HR-max (Tanaka) ─────────────────────────────────────────────────────────

/// hr_zones.rs:163 and strain.rs:403 — `tanaka(30) == 187.0` within 1e-9.
const TANAKA_AT_30: f64 = 187.0;
const TANAKA_TOL: f64 = 1e-9;

fn tanaka_arm(t: &mut Table, name: &str, kind: Kind, f: impl Fn(f64) -> f64) {
    let v = f(30.0);
    t.arm(name, kind, v, (v - TANAKA_AT_30).abs() < TANAKA_TOL);
}

fn tanaka_table() -> Table {
    let mut t = Table::new(
        "HR-max: Tanaka 208 - 0.7*age",
        "hr_zones.rs:163 + strain.rs:403 — tanaka(30) == 187.0 within 1e-9",
    );
    tanaka_arm(&mut t, "baseline (unmutated)", Kind::Baseline, hr_zones::tanaka_max_hr);
    tanaka_arm(&mut t, "null[hard]: constant scorer (age ignored, 190)", Kind::Null, |_| 190.0);
    tanaka_arm(&mut t, "struct: 220 - age instead of Tanaka", Kind::Structural, |a| {
        strain::default_max_hr(a as i32) as f64
    });
    tanaka_arm(&mut t, "struct: age read in months", Kind::Structural, |a| hr_zones::tanaka_max_hr(a * 12.0));
    // The slope is baked in; scaling age is its exact equivalent.
    tanaka_arm(&mut t, "param: slope 0.7 -> 0.77 (+10%, exact via age)", Kind::Param, |a| {
        hr_zones::tanaka_max_hr(a * 1.1)
    });
    tanaka_arm(&mut t, "param: slope +0.5% (exact via age)", Kind::Param, |a| hr_zones::tanaka_max_hr(a * 1.005));
    tanaka_arm(&mut t, "param: slope +0.01% (floor probe)", Kind::Param, |a| hr_zones::tanaka_max_hr(a * 1.0001));
    tanaka_arm(&mut t, "param: intercept 208 -> 228.8 (+10%, output offset)", Kind::Param, |a| {
        hr_zones::tanaka_max_hr(a) + 20.8
    });
    t
}

// ── 5. HR zones: edges + zone_number ───────────────────────────────────────────

/// hr_zones.rs:174-176 edges at max 200; hr_zones.rs:182-187 the six zone_number edges.
const ZONE1_LOWER: f64 = 100.0;
const ZONE1_UPPER: f64 = 120.0;
const ZONE5_UPPER: f64 = 200.0;
const ZONE_EDGE_TOL: f64 = 1e-9;
const ZONE_NUMBER_EDGES: [(f64, u8); 6] =
    [(90.0, 0), (100.0, 1), (119.9, 1), (120.0, 2), (200.0, 5), (300.0, 5)];

fn zones_with_edges(max_hr: f64, edges: [f64; 6]) -> HrZoneSet {
    let zones = (0..5)
        .map(|i| HrZone {
            number: (i + 1) as u8,
            lower: edges[i] * max_hr,
            upper: edges[i + 1] * max_hr,
            lower_pct: edges[i],
            upper_pct: edges[i + 1],
        })
        .collect();
    HrZoneSet { zones, max_hr, source: "manual".to_string() }
}

fn scaled_edges(k: f64) -> [f64; 6] {
    let mut out = hr_zones::ZONE_EDGES;
    for e in out.iter_mut() {
        *e *= k;
    }
    out
}

fn zone_arm(t: &mut Table, name: &str, kind: Kind, zs: &HrZoneSet, zn: impl Fn(f64) -> u8) {
    let pass = (zs.zones[0].lower - ZONE1_LOWER).abs() < ZONE_EDGE_TOL
        && (zs.zones[0].upper - ZONE1_UPPER).abs() < ZONE_EDGE_TOL
        && (zs.zones[4].upper - ZONE5_UPPER).abs() < ZONE_EDGE_TOL
        && ZONE_NUMBER_EDGES.iter().all(|(bpm, want)| zn(*bpm) == *want);
    t.arm(name, kind, zs.zones[0].lower, pass);
}

fn zone_edges_table() -> Table {
    let mut t = Table::new(
        "HR zones: %HRmax edges + zone_number",
        "hr_zones.rs:174-176 (100/120/200 at max 200, 1e-9) + :182-187 (90->0, 100->1, 119.9->1, 120->2, 200->5, 300->5)",
    );
    let base = hr_zones::zones_from_max(200.0, "manual");
    zone_arm(&mut t, "baseline (unmutated)", Kind::Baseline, &base, |b| base.zone_number(b));
    zone_arm(&mut t, "null[hard]: constant classifier (everything is Zone 3)", Kind::Null, &base, |_| 3);
    zone_arm(&mut t, "struct: zone numbers reversed (1<->5)", Kind::Structural, &base, |b| {
        let z = base.zone_number(b);
        if z == 0 {
            0
        } else {
            6 - z
        }
    });
    {
        let shifted = zones_with_edges(200.0, [0.60, 0.70, 0.80, 0.90, 1.00, 1.10]);
        zone_arm(&mut t, "struct: edges shifted one band up", Kind::Structural, &shifted, |b| {
            shifted.zone_number(b)
        });
    }
    {
        let zs = zones_with_edges(200.0, scaled_edges(1.1));
        zone_arm(&mut t, "param: ZONE_EDGES all +10% (injected set)", Kind::Param, &zs, |b| zs.zone_number(b));
    }
    {
        let zs = zones_with_edges(200.0, scaled_edges(1.005));
        zone_arm(&mut t, "param: ZONE_EDGES all +0.5% (injected set)", Kind::Param, &zs, |b| zs.zone_number(b));
    }
    {
        let zs = zones_with_edges(200.0, scaled_edges(1.000_000_001));
        zone_arm(&mut t, "param: ZONE_EDGES all +1e-7% (floor probe)", Kind::Param, &zs, |b| zs.zone_number(b));
    }
    {
        let zs = hr_zones::zones_from_max(220.0, "manual");
        zone_arm(&mut t, "param: max_hr 200 -> 220 (+10%)", Kind::Param, &zs, |b| zs.zone_number(b));
    }
    t
}

// ── 6. Time in zone ────────────────────────────────────────────────────────────

/// hr_zones.rs:196-199, :214-216, :232-234, :244-246 — totals and gap provenance, all within 1e-9.
const TIZ_A_TOTAL: f64 = 10.0;
const TIZ_A_ZONE3: f64 = 10.0;
const TIZ_B_TOTAL: f64 = 3.0;
const TIZ_B_REFUSED: f64 = 3600.0;
const TIZ_C_TOTAL: f64 = 903.0;
const TIZ_C_BRIDGED: f64 = 900.0;
const TIZ_D_AT_TOTAL: f64 = 1801.0;
const TIZ_D_PAST_TOTAL: f64 = 1.0;
const TIZ_D_PAST_REFUSED: f64 = 1810.0;
const TIZ_TOL: f64 = 1e-9;

fn tiz_fixtures() -> [Vec<HrSample>; 5] {
    [
        (0..10).map(|t| HrSample { ts: t, bpm: 150 }).collect(),
        vec![
            HrSample { ts: 0, bpm: 110 },
            HrSample { ts: 1, bpm: 110 },
            HrSample { ts: 2, bpm: 110 },
            HrSample { ts: 3602, bpm: 110 },
        ],
        vec![
            HrSample { ts: 0, bpm: 110 },
            HrSample { ts: 1, bpm: 110 },
            HrSample { ts: 2, bpm: 110 },
            HrSample { ts: 902, bpm: 110 },
        ],
        vec![HrSample { ts: 0, bpm: 110 }, HrSample { ts: 1800, bpm: 110 }],
        vec![HrSample { ts: 0, bpm: 110 }, HrSample { ts: 1810, bpm: 110 }],
    ]
}

/// Mutant engine: one second per sample, ignoring elapsed time entirely.
fn tiz_count_only(hr: &[HrSample], zs: &HrZoneSet) -> TimeInZone {
    let mut seconds = [0.0f64; 5];
    let mut below = 0.0;
    for s in hr {
        let z = zs.zone_number(s.bpm as f64);
        if z >= 1 {
            seconds[(z - 1) as usize] += 1.0;
        } else {
            below += 1.0;
        }
    }
    TimeInZone { seconds, below_zone1: below, bridged_seconds: 0.0, refused_seconds: 0.0 }
}

/// Mutant engine: every gap credited in full, no position ceiling.
fn tiz_no_ceiling(hr: &[HrSample], zs: &HrZoneSet) -> TimeInZone {
    let mut sorted = hr.to_vec();
    sorted.sort_by_key(|s| s.ts);
    let mut seconds = [0.0f64; 5];
    let mut below = 0.0;
    for i in 0..sorted.len() {
        let dur = if i + 1 < sorted.len() {
            ((sorted[i + 1].ts - sorted[i].ts) as f64).max(1.0)
        } else {
            1.0
        };
        let z = zs.zone_number(sorted[i].bpm as f64);
        if z >= 1 {
            seconds[(z - 1) as usize] += dur;
        } else {
            below += dur;
        }
    }
    TimeInZone { seconds, below_zone1: below, bridged_seconds: 0.0, refused_seconds: 0.0 }
}

fn tiz_arm(
    t: &mut Table,
    name: &str,
    kind: Kind,
    zs: &HrZoneSet,
    engine: impl Fn(&[HrSample], &HrZoneSet) -> TimeInZone,
    prep: impl Fn(&[HrSample]) -> Vec<HrSample>,
) {
    let f = tiz_fixtures();
    let out: Vec<TimeInZone> = f.iter().map(|fx| engine(&prep(fx), zs)).collect();
    let near = |a: f64, b: f64| (a - b).abs() < TIZ_TOL;
    let pass = near(out[0].total(), TIZ_A_TOTAL)
        && near(out[0].seconds_in_zone(3), TIZ_A_ZONE3)
        && out[0].bridged_seconds == 0.0
        && out[0].refused_seconds == 0.0
        && near(out[1].total(), TIZ_B_TOTAL)
        && near(out[1].refused_seconds, TIZ_B_REFUSED)
        && out[1].bridged_seconds == 0.0
        && near(out[2].total(), TIZ_C_TOTAL)
        && near(out[2].bridged_seconds, TIZ_C_BRIDGED)
        && out[2].refused_seconds == 0.0
        && near(out[3].total(), TIZ_D_AT_TOTAL)
        && near(out[4].total(), TIZ_D_PAST_TOTAL)
        && near(out[4].refused_seconds, TIZ_D_PAST_REFUSED);
    t.arm(name, kind, out[2].total(), pass);
}

fn time_in_zone_table() -> Table {
    let mut t = Table::new(
        "Time in zone (seconds per zone + gap provenance)",
        "hr_zones.rs:196-199/:214-216/:232-234/:244-246 — totals 10.0/3.0/903.0/1801.0/1.0, refused 3600/1810, bridged 900, 1e-9",
    );
    let zs = hr_zones::zones_from_max(200.0, "manual");
    tiz_arm(&mut t, "baseline (unmutated)", Kind::Baseline, &zs, hr_zones::time_in_zone, |h| h.to_vec());
    tiz_arm(&mut t, "null[hard]: one second per sample (elapsed time ignored)", Kind::Null, &zs, tiz_count_only, |h| {
        h.to_vec()
    });
    tiz_arm(&mut t, "null: every gap credited in full (hr_gap policy ignored)", Kind::Null, &zs, tiz_no_ceiling, |h| {
        h.to_vec()
    });
    tiz_arm(&mut t, "struct: sample order reversed", Kind::Structural, &zs, hr_zones::time_in_zone, |h| {
        h.iter().rev().copied().collect()
    });
    tiz_arm(&mut t, "struct: whole series shifted +3600 s", Kind::Structural, &zs, hr_zones::time_in_zone, |h| {
        h.iter().map(|s| HrSample { ts: s.ts + 3600, bpm: s.bpm }).collect()
    });
    tiz_arm(&mut t, "struct: last sample dropped", Kind::Structural, &zs, hr_zones::time_in_zone, |h| {
        h[..h.len() - 1].to_vec()
    });
    {
        let z = hr_zones::zones_from_max(220.0, "manual");
        tiz_arm(&mut t, "param: max_hr 200 -> 220 (+10%)", Kind::Param, &z, hr_zones::time_in_zone, |h| h.to_vec());
    }
    {
        let z = zones_with_edges(200.0, scaled_edges(1.005));
        tiz_arm(&mut t, "param: ZONE_EDGES all +0.5% (injected set)", Kind::Param, &z, hr_zones::time_in_zone, |h| {
            h.to_vec()
        });
    }
    {
        let z = zones_with_edges(200.0, scaled_edges(1.1));
        tiz_arm(&mut t, "param: ZONE_EDGES all +10% (injected set)", Kind::Param, &z, hr_zones::time_in_zone, |h| {
            h.to_vec()
        });
    }
    tiz_arm(&mut t, "param: median_gap_s tail fallback (proxy: every gap >= 300 s)", Kind::Param, &zs, hr_zones::time_in_zone, |h| {
        h.iter().map(|s| HrSample { ts: s.ts * 300, bpm: s.bpm }).collect()
    });
    t
}

// ── 7. hr_gap position ceilings ────────────────────────────────────────────────

/// hr_gap.rs:141-149 — a gap AT its position ceiling is credited in full, one past it is refused.
/// The same edge is pinned through totals at hr_zones.rs:244-246 and through TRIMP at strain.rs:329.
const SHIPPED_CEILINGS: [(GapPosition, f64); 3] = [
    (GapPosition::Leading, 600.0),
    (GapPosition::Interior, 1800.0),
    (GapPosition::Trailing, 300.0),
];

/// Evaluates the shipped edge claim with the ceilings a mutant would use.
fn ceiling_claim(ceilings: [f64; 3], credit: impl Fn(f64, GapPosition) -> f64) -> bool {
    SHIPPED_CEILINGS.iter().enumerate().all(|(i, (pos, _))| {
        let c = ceilings[i];
        (credit(c, *pos) - c).abs() < 1e-9 && credit(c + 1.0, *pos) == 0.0
    })
}

fn ceiling_arm(t: &mut Table, name: &str, kind: Kind, ceilings: [f64; 3], credit: impl Fn(f64, GapPosition) -> f64) {
    t.arm(name, kind, ceilings[1], ceiling_claim(ceilings, credit));
}

fn hr_gap_table() -> Table {
    let shipped = [600.0, 1800.0, 300.0];
    let mut t = Table::new(
        "hr_gap: position ceilings (lead 600 / interior 1800 / trail 300 s)",
        "hr_gap.rs:141-149 — creditable_seconds(C) == C and creditable_seconds(C+1) == 0 per position",
    );
    ceiling_arm(&mut t, "baseline (unmutated)", Kind::Baseline, shipped, hr_gap::creditable_seconds);
    ceiling_arm(&mut t, "null[hard]: no ceiling at all (every gap credited)", Kind::Null, shipped, |g, _| g.max(0.0));
    ceiling_arm(&mut t, "null: nothing ever credited", Kind::Null, shipped, |_, _| 0.0);
    ceiling_arm(&mut t, "struct: lead and trail ceilings swapped", Kind::Structural, [300.0, 1800.0, 600.0], hr_gap::creditable_seconds);
    ceiling_arm(&mut t, "struct: one flat ceiling for every position", Kind::Structural, [1800.0, 1800.0, 1800.0], hr_gap::creditable_seconds);
    ceiling_arm(&mut t, "param: all ceilings +10%", Kind::Param, [660.0, 1980.0, 330.0], hr_gap::creditable_seconds);
    ceiling_arm(&mut t, "param: all ceilings -10%", Kind::Param, [540.0, 1620.0, 270.0], hr_gap::creditable_seconds);
    ceiling_arm(&mut t, "param: all ceilings +0.5%", Kind::Param, [603.0, 1809.0, 301.5], hr_gap::creditable_seconds);
    ceiling_arm(&mut t, "param: interior ceiling +1 s (floor probe)", Kind::Param, [600.0, 1801.0, 300.0], hr_gap::creditable_seconds);
    ceiling_arm(&mut t, "param: MIN_GAP_SECONDS 60 -> 66 (proxy: unobservable at the ceiling)", Kind::Param, shipped, |g, p| {
        if g < hr_gap::MIN_GAP_SECONDS * 1.1 {
            g.max(0.0)
        } else {
            hr_gap::creditable_seconds(g, p)
        }
    });
    t
}

// ── 8. Calories: day totals ────────────────────────────────────────────────────

/// The two BMR-only days — sedentary and light — both `|total - 1825.25| < 1.0`. Neither reaches the
/// day gate, so neither says anything about active energy; [CAL_ACTIVE_GOLDEN] is the arm that does.
const CAL_DAY_GOLDEN: f64 = 1825.25;
const CAL_DAY_TOL: f64 = 1.0;
/// The active day: 82 800 s at the resting rate plus 3600 s billed by Keytel over the day gate.
const CAL_ACTIVE_GOLDEN: f64 = 2487.16;
/// The day path equals the bout path on a 1 Hz stream ONLY above both activity gates (>= 120 bpm for
/// this profile). Inside 94..=119 bpm they diverge by construction; that band is its own table.
const CAL_BOUT_TOL: f64 = 1e-9;

const CAL_W: f64 = 80.0;
const CAL_H: f64 = 180.0;
const CAL_AGE: f64 = 35.0;
const CAL_SEX: &str = "male";
const CAL_HRMAX: f64 = 185.0;
const CAL_REST: f64 = 55.0;

fn cal_day(hr: &[HrSample]) -> f64 {
    calories::estimate_day_calories(hr, CAL_W, CAL_H, CAL_AGE, CAL_SEX, CAL_HRMAX, CAL_REST)
}

fn cal_sedentary() -> Vec<HrSample> {
    hr_constant(55, 86_400)
}

/// One block on a shared 24 h timeline; [hr_constant] restarts at ts 0, so concatenating it stacks
/// blocks on the same seconds instead of laying them end to end.
fn hr_block(start: i64, bpm: i32, n: usize) -> Vec<HrSample> {
    (0..n).map(|i| HrSample { ts: start + i as i64, bpm }).collect()
}

/// Every sample under the 120 bpm day gate, so this day is BMR only.
fn cal_light() -> Vec<HrSample> {
    let mut v = hr_block(0, 55, 8 * 3600);
    v.extend(hr_block(8 * 3600, 70, 8 * 3600));
    v.extend(hr_block(16 * 3600, 100, 8 * 3600));
    v
}

/// The same day with 30 min each at 125 and 140 bpm, the only blocks that clear the day gate.
fn cal_active() -> Vec<HrSample> {
    let mut v = hr_block(0, 55, 8 * 3600);
    v.extend(hr_block(8 * 3600, 70, 8 * 3600));
    v.extend(hr_block(16 * 3600, 100, 7 * 3600));
    v.extend(hr_block(23 * 3600, 125, 1800));
    v.extend(hr_block(23 * 3600 + 1800, 140, 1800));
    v
}

/// The active day rebuilt from the two public rates: resting for 82 800 s, Keytel for the other hour.
fn cal_active_rebuilt(c: &calories::Coeffs) -> f64 {
    calories::resting_kcal_per_s(c, CAL_W, CAL_H, CAL_AGE) * 82_800.0
        + calories::active_kcal_per_s(c, 125.0, CAL_HRMAX, CAL_W, CAL_AGE) * 1_800.0
        + calories::active_kcal_per_s(c, 140.0, CAL_HRMAX, CAL_W, CAL_AGE) * 1_800.0
}

fn male_with(f: impl FnOnce(&mut calories::Coeffs)) -> calories::Coeffs {
    let mut c = calories::MALE;
    f(&mut c);
    c
}

fn cal_arm(
    t: &mut Table,
    name: &str,
    kind: Kind,
    sedentary: &[HrSample],
    light: &[HrSample],
    engine: impl Fn(&[HrSample]) -> f64,
) {
    let a = engine(sedentary);
    let b = engine(light);
    let pass = (a - CAL_DAY_GOLDEN).abs() < CAL_DAY_TOL && (b - CAL_DAY_GOLDEN).abs() < CAL_DAY_TOL;
    t.arm(name, kind, a, pass);
}

fn calories_day_table(sedentary: &[HrSample], light: &[HrSample]) -> Table {
    let mut t = Table::new(
        "Calories: whole-day total under the day gate (BMR only)",
        "calories.rs — sedentary day and light-activity day both |total - 1825.25| < 1.0; every sample \
         is under the 120 bpm day gate, so Keytel is never called and only the BMR half is pinned",
    );
    cal_arm(&mut t, "baseline (unmutated)", Kind::Baseline, sedentary, light, cal_day);
    cal_arm(&mut t, "null[hard]: output pinned to 0", Kind::Null, sedentary, light, |_| 0.0);
    cal_arm(&mut t, "null: constant population day (2000 kcal)", Kind::Null, sedentary, light, |_| 2000.0);
    cal_arm(&mut t, "null: Keytel disabled (every sample billed the resting rate)", Kind::Null, sedentary, light, |h| {
        let flat: Vec<HrSample> = h.iter().map(|s| HrSample { ts: s.ts, bpm: 0 }).collect();
        cal_day(&flat)
    });
    // Validates the reconstruction the coefficient arms below run through.
    cal_arm(&mut t, "path check: resting_kcal_per_s x 86400 reproduces the day", Kind::Structural, sedentary, light, |_| {
        calories::resting_kcal_per_s(&calories::MALE, CAL_W, CAL_H, CAL_AGE) * 86_400.0
    });
    cal_arm(&mut t, "struct: sample order reversed", Kind::Structural, sedentary, light, |h| {
        let r: Vec<HrSample> = h.iter().rev().copied().collect();
        cal_day(&r)
    });
    cal_arm(&mut t, "struct: last 10% of the day dropped", Kind::Structural, sedentary, light, |h| {
        cal_day(&h[..h.len() * 9 / 10])
    });
    cal_arm(&mut t, "struct: every sample duplicated at the same second", Kind::Structural, sedentary, light, |h| {
        let d: Vec<HrSample> = h.iter().flat_map(|s| [*s, *s]).collect();
        cal_day(&d)
    });
    cal_arm(&mut t, "param: sex male -> female (selects the whole coeff set)", Kind::Param, sedentary, light, |h| {
        calories::estimate_day_calories(h, CAL_W, CAL_H, CAL_AGE, "female", CAL_HRMAX, CAL_REST)
    });
    cal_arm(&mut t, "param: MALE.resting_alpha +10% (via reconstruction)", Kind::Param, sedentary, light, |_| {
        calories::resting_kcal_per_s(&male_with(|c| c.resting_alpha *= 1.1), CAL_W, CAL_H, CAL_AGE) * 86_400.0
    });
    cal_arm(&mut t, "param: MALE.resting_alpha +0.5% (floor probe)", Kind::Param, sedentary, light, |_| {
        calories::resting_kcal_per_s(&male_with(|c| c.resting_alpha *= 1.005), CAL_W, CAL_H, CAL_AGE) * 86_400.0
    });
    cal_arm(&mut t, "param: MALE.resting_weight +10% (via reconstruction)", Kind::Param, sedentary, light, |_| {
        calories::resting_kcal_per_s(&male_with(|c| c.resting_weight *= 1.1), CAL_W, CAL_H, CAL_AGE) * 86_400.0
    });
    cal_arm(&mut t, "param: MALE.resting_height +10% (via reconstruction)", Kind::Param, sedentary, light, |_| {
        calories::resting_kcal_per_s(&male_with(|c| c.resting_height *= 1.1), CAL_W, CAL_H, CAL_AGE) * 86_400.0
    });
    cal_arm(&mut t, "param: MALE.resting_age +10% (via reconstruction)", Kind::Param, sedentary, light, |_| {
        calories::resting_kcal_per_s(&male_with(|c| c.resting_age *= 1.1), CAL_W, CAL_H, CAL_AGE) * 86_400.0
    });
    cal_arm(&mut t, "param: MALE.workout_hr +10% (unreachable: no sample clears the day gate)", Kind::Param, sedentary, light, |_| {
        calories::resting_kcal_per_s(&male_with(|c| c.workout_hr *= 1.1), CAL_W, CAL_H, CAL_AGE) * 86_400.0
    });
    cal_arm(&mut t, "param: DAY_ACTIVE_HRR_FRACTION +10% (unreachable: the gate moves further away)", Kind::Param, sedentary, light, |h| {
        calories::estimate_day_calories(h, CAL_W, CAL_H, CAL_AGE, CAL_SEX, CAL_HRMAX * 1.1, CAL_REST)
    });
    cal_arm(&mut t, "param: DAY_GAP_CAP_S 300 -> 330 (unreachable: fixture has no gap > 1 s)", Kind::Param, sedentary, light, cal_day);
    t
}

// ── 8b. Calories: active energy above the day gate ─────────────────────────────

fn cal_active_arm(t: &mut Table, name: &str, kind: Kind, active: &[HrSample], engine: impl Fn(&[HrSample]) -> f64) {
    let v = engine(active);
    t.arm(name, kind, v, (v - CAL_ACTIVE_GOLDEN).abs() < CAL_DAY_TOL);
}

fn calories_active_day_table(active: &[HrSample]) -> Table {
    let mut t = Table::new(
        "Calories: active energy above the day gate (Keytel, day path)",
        "calories.rs — active day |total - 2487.16| < 1.0, of which 661.91 kcal is Keytel over the \
         3600 s that clear the 120 bpm day gate",
    );
    cal_active_arm(&mut t, "baseline (unmutated)", Kind::Baseline, active, cal_day);
    cal_active_arm(&mut t, "null[hard]: output pinned to 0", Kind::Null, active, |_| 0.0);
    cal_active_arm(&mut t, "null: Keytel disabled (every sample billed the resting rate)", Kind::Null, active, |h| {
        let flat: Vec<HrSample> = h.iter().map(|s| HrSample { ts: s.ts, bpm: 0 }).collect();
        cal_day(&flat)
    });
    cal_active_arm(&mut t, "null: constant population day (2000 kcal)", Kind::Null, active, |_| 2000.0);
    // Validates the reconstruction the coefficient arms below run through.
    cal_active_arm(&mut t, "path check: resting x 82800 + Keytel x 3600 reproduces the day", Kind::Structural, active, |_| {
        cal_active_rebuilt(&calories::MALE)
    });
    cal_active_arm(&mut t, "struct: sample order reversed", Kind::Structural, active, |h| {
        let r: Vec<HrSample> = h.iter().rev().copied().collect();
        cal_day(&r)
    });
    cal_active_arm(&mut t, "struct: last 10% of the day dropped (loses both over-gate blocks)", Kind::Structural, active, |h| {
        cal_day(&h[..h.len() * 9 / 10])
    });
    cal_active_arm(&mut t, "struct: every sample duplicated at the same second", Kind::Structural, active, |h| {
        let d: Vec<HrSample> = h.iter().flat_map(|s| [*s, *s]).collect();
        cal_day(&d)
    });
    cal_active_arm(&mut t, "struct: the 125 bpm block dropped to 119 (just under the gate)", Kind::Structural, active, |h| {
        let m: Vec<HrSample> = h.iter().map(|s| HrSample { ts: s.ts, bpm: if s.bpm == 125 { 119 } else { s.bpm } }).collect();
        cal_day(&m)
    });
    cal_active_arm(&mut t, "param: sex male -> female (selects the whole coeff set)", Kind::Param, active, |h| {
        calories::estimate_day_calories(h, CAL_W, CAL_H, CAL_AGE, "female", CAL_HRMAX, CAL_REST)
    });
    cal_active_arm(&mut t, "param: DAY_ACTIVE_HRR_FRACTION 0.50 -> 0.55 (gate 120 -> 126.5)", Kind::Param, active, |h| {
        calories::estimate_day_calories(h, CAL_W, CAL_H, CAL_AGE, CAL_SEX, CAL_HRMAX, 68.0)
    });
    cal_active_arm(&mut t, "param: MALE.workout_hr +10% (via reconstruction)", Kind::Param, active, |_| {
        cal_active_rebuilt(&male_with(|c| c.workout_hr *= 1.1))
    });
    cal_active_arm(&mut t, "param: MALE.workout_hr +0.5% (floor probe)", Kind::Param, active, |_| {
        cal_active_rebuilt(&male_with(|c| c.workout_hr *= 1.005))
    });
    cal_active_arm(&mut t, "param: MALE.workout_alpha +10% (via reconstruction)", Kind::Param, active, |_| {
        cal_active_rebuilt(&male_with(|c| c.workout_alpha *= 1.1))
    });
    cal_active_arm(&mut t, "param: MALE.workout_weight +10% (via reconstruction)", Kind::Param, active, |_| {
        cal_active_rebuilt(&male_with(|c| c.workout_weight *= 1.1))
    });
    cal_active_arm(&mut t, "param: MALE.workout_age +10% (via reconstruction)", Kind::Param, active, |_| {
        cal_active_rebuilt(&male_with(|c| c.workout_age *= 1.1))
    });
    cal_active_arm(&mut t, "param: WORKOUT_DIVISOR +10% (via reconstruction)", Kind::Param, active, |_| {
        calories::resting_kcal_per_s(&calories::MALE, CAL_W, CAL_H, CAL_AGE) * 82_800.0
            + calories::active_kcal_per_s(&calories::MALE, 125.0, CAL_HRMAX, CAL_W, CAL_AGE) / 1.1 * 1_800.0
            + calories::active_kcal_per_s(&calories::MALE, 140.0, CAL_HRMAX, CAL_W, CAL_AGE) / 1.1 * 1_800.0
    });
    t
}

// ── 9. Calories: day/bout agreement ────────────────────────────────────────────

fn cal_bout_fixture() -> Vec<HrSample> {
    hr_constant(130, 600)
}

fn cal_bout_arm(t: &mut Table, name: &str, kind: Kind, engine: impl Fn(&[HrSample]) -> f64) {
    let fx = cal_bout_fixture();
    let day = engine(&fx);
    let (bout, _) = calories::estimate_bout_calories(&fx, CAL_W, CAL_H, CAL_AGE, CAL_SEX, CAL_HRMAX, CAL_REST);
    t.arm(name, kind, day, (day - bout).abs() < CAL_BOUT_TOL);
}

fn calories_bout_table() -> Table {
    let mut t = Table::new(
        "Calories: day path equals bout path at 1 Hz ABOVE BOTH ACTIVITY GATES",
        "|bout - day| < 1e-9 on 600 samples at 130 bpm — 130 clears the bout gate (94) and the day \
         gate (120), so this measures gap weighting only, NOT the 94..=119 band where they diverge",
    );
    cal_bout_arm(&mut t, "baseline (unmutated)", Kind::Baseline, cal_day);
    cal_bout_arm(&mut t, "null[hard]: output pinned to 0", Kind::Null, |_| 0.0);
    cal_bout_arm(&mut t, "null: Keytel disabled (resting rate for every sample)", Kind::Null, |h| {
        let flat: Vec<HrSample> = h.iter().map(|s| HrSample { ts: s.ts, bpm: 0 }).collect();
        cal_day(&flat)
    });
    cal_bout_arm(&mut t, "struct: every sample duplicated at the same second", Kind::Structural, |h| {
        let d: Vec<HrSample> = h.iter().flat_map(|s| [*s, *s]).collect();
        cal_day(&d)
    });
    cal_bout_arm(&mut t, "param: sex male -> female (moves BOTH sides)", Kind::Param, |h| {
        calories::estimate_day_calories(h, CAL_W, CAL_H, CAL_AGE, "female", CAL_HRMAX, CAL_REST)
    });
    cal_bout_arm(&mut t, "param: hrmax +10% (moves BOTH sides)", Kind::Param, |h| {
        calories::estimate_day_calories(h, CAL_W, CAL_H, CAL_AGE, CAL_SEX, CAL_HRMAX * 1.1, CAL_REST)
    });
    t
}

// ── 10. Calories: coefficient identities ───────────────────────────────────────

/// calories.rs:269-275 — resolve_coeffs identities, 1e-9.
const CAL_ALPHA_MALE: f64 = 88.362;
const CAL_ALPHA_FEMALE: f64 = 447.593;
const CAL_ALPHA_NONBINARY: f64 = 267.9775;
const CAL_ALPHA_TOL: f64 = 1e-9;

fn coeff_arm(t: &mut Table, name: &str, kind: Kind, f: impl Fn(&str) -> f64) {
    let pass = (f("male") - CAL_ALPHA_MALE).abs() < CAL_ALPHA_TOL
        && (f("female") - CAL_ALPHA_FEMALE).abs() < CAL_ALPHA_TOL
        && (f("unknown") - CAL_ALPHA_NONBINARY).abs() < CAL_ALPHA_TOL;
    t.arm(name, kind, f("male"), pass);
}

fn calories_coeff_table() -> Table {
    let mut t = Table::new(
        "Calories: resolve_coeffs identities",
        "calories.rs:269-275 — male 88.362 / female 447.593 / unknown 267.9775 resting_alpha, 1e-9",
    );
    coeff_arm(&mut t, "baseline (unmutated)", Kind::Baseline, |s| calories::resolve_coeffs(s).resting_alpha);
    coeff_arm(&mut t, "null[hard]: one coeff set for every sex (male)", Kind::Null, |_| calories::MALE.resting_alpha);
    coeff_arm(&mut t, "struct: male and female sets swapped", Kind::Structural, |s| match s {
        "male" => calories::FEMALE.resting_alpha,
        "female" => calories::MALE.resting_alpha,
        _ => calories::NONBINARY.resting_alpha,
    });
    coeff_arm(&mut t, "struct: unknown sex falls back to male, not nonbinary", Kind::Structural, |s| match s {
        "female" => calories::FEMALE.resting_alpha,
        _ => calories::MALE.resting_alpha,
    });
    coeff_arm(&mut t, "param: every resting_alpha +10%", Kind::Param, |s| {
        calories::resolve_coeffs(s).resting_alpha * 1.1
    });
    coeff_arm(&mut t, "param: every resting_alpha +1e-9 relative (floor probe)", Kind::Param, |s| {
        calories::resolve_coeffs(s).resting_alpha * (1.0 + 1e-9)
    });
    coeff_arm(&mut t, "param: every resting_alpha +1e-12 relative (floor probe)", Kind::Param, |s| {
        calories::resolve_coeffs(s).resting_alpha * (1.0 + 1e-12)
    });
    t
}

// ── 11. Steps: window totals ───────────────────────────────────────────────────

/// steps.rs:48, :59, :89-90 — window goldens over the wrap-aware counter delta.
const STEPS_PLAIN: u32 = 120;
const STEPS_WRAP: u32 = 116;
const STEPS_UNDER_CEILING: u32 = 511;

fn step(ts: i64, counter: u16) -> StepSample {
    StepSample { ts, counter, activity_class: None }
}

fn steps_arm(t: &mut Table, name: &str, kind: Kind, f: impl Fn(&[StepSample]) -> Option<u32>) {
    let plain = [step(0, 100), step(60, 150), step(120, 220)];
    let wrap = [step(0, 65500), step(60, 20), step(120, 80)];
    let at_ceiling = [step(0, 0), step(60, 512)];
    let under_ceiling = [step(0, 0), step(60, 511)];
    let pass = f(&plain) == Some(STEPS_PLAIN)
        && f(&wrap) == Some(STEPS_WRAP)
        && f(&at_ceiling).is_none()
        && f(&under_ceiling) == Some(STEPS_UNDER_CEILING);
    t.arm(name, kind, f(&plain).map_or(f64::NAN, |v| v as f64), pass);
}

fn steps_table() -> Table {
    let mut t = Table::new(
        "Steps: wrap-aware motion-tick total",
        "steps.rs:48/:59/:89-90 — plain 120, wrap 116, delta 512 -> None, delta 511 -> 511",
    );
    steps_arm(&mut t, "baseline (unmutated)", Kind::Baseline, steps::steps_in_window);
    steps_arm(&mut t, "null[hard]: output pinned to None", Kind::Null, |_| None);
    steps_arm(&mut t, "null: sample count instead of counter deltas", Kind::Null, |s| Some(s.len() as u32));
    steps_arm(&mut t, "null: last minus first, no wrap and no ceiling", Kind::Null, |s| {
        let d = s.last()?.counter as i64 - s.first()?.counter as i64;
        (d > 0).then_some(d as u32)
    });
    steps_arm(&mut t, "struct: counters reversed (series runs backwards)", Kind::Structural, |s| {
        let r: Vec<StepSample> = s
            .iter()
            .rev()
            .zip(s.iter())
            .map(|(c, ts)| step(ts.ts, c.counter))
            .collect();
        steps::steps_in_window(&r)
    });
    steps_arm(&mut t, "struct: middle sample dropped", Kind::Structural, |s| {
        let kept: Vec<StepSample> = s.iter().enumerate().filter(|(i, _)| *i != 1).map(|(_, v)| *v).collect();
        steps::steps_in_window(&kept)
    });
    steps_arm(&mut t, "struct: only the first two samples kept", Kind::Structural, |s| {
        steps::steps_in_window(&s[..2])
    });
    steps_arm(&mut t, "param: MAX_STEP_DELTA 512 -> 563 (+10%)", Kind::Param, |s| {
        let mut total = 0u32;
        for w in s.windows(2) {
            let d = w[1].counter.wrapping_sub(w[0].counter);
            if d > 0 && d < 563 {
                total += d as u32;
            }
        }
        (total > 0).then_some(total)
    });
    steps_arm(&mut t, "param: MAX_STEP_DELTA 512 -> 513 (floor probe)", Kind::Param, |s| {
        let mut total = 0u32;
        for w in s.windows(2) {
            let d = w[1].counter.wrapping_sub(w[0].counter);
            if d > 0 && d < 513 {
                total += d as u32;
            }
        }
        (total > 0).then_some(total)
    });
    t
}

// ── 12. Steps: MAX_STEP_DELTA ceiling ──────────────────────────────────────────

/// steps.rs:80-84 — `tick_delta` accepts a delta below the ceiling and refuses one at it.
fn ceiling_holds_at(c: u16) -> bool {
    steps::tick_delta(&step(0, 0), &step(1, c)).is_none()
        && steps::tick_delta(&step(0, 0), &step(1, c - 1)) == Some(c - 1)
}

fn steps_ceiling_table() -> Table {
    let mut t = Table::new(
        "Steps: MAX_STEP_DELTA ceiling is exclusive",
        "steps.rs:80-84 — tick_delta at 512 is None, at 511 is Some(511); wrap 65500->20 is Some(56)",
    );
    let wrap_ok = steps::tick_delta(&step(0, 65500), &step(1, 20)) == Some(56);
    // A ceiling-free mutant accepts every forward delta, so the exclusivity clause cannot hold.
    let no_ceiling = |a: &StepSample, b: &StepSample| {
        let d = b.counter.wrapping_sub(a.counter);
        (d > 0).then_some(d)
    };
    let null_holds = no_ceiling(&step(0, 0), &step(1, 512)).is_none()
        && no_ceiling(&step(0, 0), &step(1, 511)) == Some(511)
        && no_ceiling(&step(0, 65500), &step(1, 20)) == Some(56);
    t.arm("baseline (unmutated)", Kind::Baseline, 512.0, ceiling_holds_at(512) && wrap_ok);
    t.arm("null[hard]: no ceiling (every forward delta accepted)", Kind::Null, 512.0, null_holds);
    t.arm("param: MAX_STEP_DELTA 512 -> 563 (+10%)", Kind::Param, 563.0, ceiling_holds_at(563) && wrap_ok);
    t.arm("param: MAX_STEP_DELTA 512 -> 461 (-10%)", Kind::Param, 461.0, ceiling_holds_at(461) && wrap_ok);
    t.arm("param: MAX_STEP_DELTA 512 -> 515 (+0.5%)", Kind::Param, 515.0, ceiling_holds_at(515) && wrap_ok);
    t.arm("param: MAX_STEP_DELTA 512 -> 513 (+1 tick, floor probe)", Kind::Param, 513.0, ceiling_holds_at(513) && wrap_ok);
    t
}

// ── 13. Workout: the gated helpers ─────────────────────────────────────────────

/// workout.rs:434-435 (series shape), :452-455 (nearest), :469-470 (smoothing), :482 (zone %),
/// :491 and :503 (bridge counts).
const WO_ZONE3_PCT: f64 = 100.0;
const WO_ZONE_TOL: f64 = 0.1;

#[derive(Clone, Copy)]
struct WoKnobs {
    rest: f64,
    max: f64,
    smooth_window: f64,
    bridge_floor: f64,
    near_tol: f64,
    reverse_hr: bool,
    drop_tail: bool,
    zone_pin: Option<i32>,
}

fn wo_base() -> WoKnobs {
    WoKnobs {
        rest: 60.0,
        max: 160.0,
        smooth_window: 5.0,
        bridge_floor: 75.0,
        near_tol: 2.0,
        reverse_hr: false,
        drop_tail: false,
        zone_pin: None,
    }
}

fn wo_eval(k: WoKnobs) -> (f64, bool) {
    let g = [
        GravitySample { ts: 10, x: 0.0, y: 0.0, z: 0.0 },
        GravitySample { ts: 0, x: 1.0, y: 0.0, z: 0.0 },
    ];
    let s = workout::activity_series(&g);
    let series_ok = s.len() == 2 && s[0].ts == 0 && s[1].ts == 10;

    let ts = [10i64, 20, 30];
    let vals = [1.0, 2.0, 3.0];
    let near_ok = workout::nearest(&ts, &vals, 20, k.near_tol) == Some(2.0)
        && workout::nearest(&ts, &vals, 29, k.near_tol) == Some(3.0)
        && workout::nearest(&ts, &vals, 11, k.near_tol) == Some(1.0)
        && workout::nearest(&ts, &vals, 25, k.near_tol).is_none();

    let m = [
        ActivityPoint { ts: 0, intensity: 0.0 },
        ActivityPoint { ts: 1, intensity: 1.0 },
        ActivityPoint { ts: 2, intensity: 0.0 },
        ActivityPoint { ts: 3, intensity: 1.0 },
        ActivityPoint { ts: 11, intensity: 0.0 },
    ];
    let sm = workout::smoothed_intensity(&m, k.smooth_window);
    let smooth_ok = sm[0].abs() < 1e-9 && (sm[1] - 0.5).abs() < 1e-9;

    let mut hr: Vec<HrSample> = (0..10).map(|i| HrSample { ts: i, bpm: 130 }).collect();
    if k.reverse_hr {
        hr.reverse();
    }
    if k.drop_tail {
        hr.truncate(9);
    }
    let (zp, _) = workout::bout_intensity(&hr, k.rest, k.max);
    let z3 = match k.zone_pin {
        Some(z) => {
            if z == 3 {
                100.0
            } else {
                0.0
            }
        }
        None => zp.iter().find(|(z, _)| *z == 3).map(|(_, p)| *p).unwrap_or(0.0),
    };
    let zone_ok = (z3 - WO_ZONE3_PCT).abs() < WO_ZONE_TOL;

    let runs = [(0i64, 100i64), (200, 300)];
    let elevated: Vec<HrSample> = (0..=300).map(|i| HrSample { ts: i, bpm: 100 }).collect();
    let merged_ok = workout::bridge_runs(&runs, &elevated, k.bridge_floor).len() == 1;
    let mut rest_gap: Vec<HrSample> = (0..100).map(|i| HrSample { ts: i, bpm: 100 }).collect();
    rest_gap.extend((101..200).map(|i| HrSample { ts: i, bpm: 60 }));
    rest_gap.extend((201..301).map(|i| HrSample { ts: i, bpm: 100 }));
    let split_ok = workout::bridge_runs(&runs, &rest_gap, k.bridge_floor).len() == 2;

    (z3, series_ok && near_ok && smooth_ok && zone_ok && merged_ok && split_ok)
}

fn wo_arm(t: &mut Table, name: &str, kind: Kind, k: WoKnobs) {
    let (value, pass) = wo_eval(k);
    t.arm(name, kind, value, pass);
}

fn workout_helper_table() -> Table {
    let mut t = Table::new(
        "Workout: gated helpers (series, nearest, smoothing, zone %, bridge)",
        "workout.rs:434-435/:452-455/:469-470/:482 (z3 == 100.0 +/- 0.1)/:491/:503",
    );
    wo_arm(&mut t, "baseline (unmutated)", Kind::Baseline, wo_base());
    wo_arm(&mut t, "null[hard]: zone classifier pinned below Zone 1", Kind::Null, WoKnobs { zone_pin: Some(0), ..wo_base() });
    wo_arm(&mut t, "null: zone classifier pinned to Zone 1", Kind::Null, WoKnobs { zone_pin: Some(1), ..wo_base() });
    wo_arm(&mut t, "struct: bout HR series reversed", Kind::Structural, WoKnobs { reverse_hr: true, ..wo_base() });
    wo_arm(&mut t, "struct: last 10% of the bout dropped", Kind::Structural, WoKnobs { drop_tail: true, ..wo_base() });
    wo_arm(&mut t, "param: resting_hr 60 -> 66 (+10%)", Kind::Param, WoKnobs { rest: 66.0, ..wo_base() });
    wo_arm(&mut t, "param: resting_hr 60 -> 54 (-10%)", Kind::Param, WoKnobs { rest: 54.0, ..wo_base() });
    wo_arm(&mut t, "param: max_hr 160 -> 176 (+10%)", Kind::Param, WoKnobs { max: 176.0, ..wo_base() });
    wo_arm(&mut t, "param: max_hr 160 -> 144 (-10%)", Kind::Param, WoKnobs { max: 144.0, ..wo_base() });
    wo_arm(&mut t, "param: max_hr +0.5% (floor probe)", Kind::Param, WoKnobs { max: 160.8, ..wo_base() });
    wo_arm(&mut t, "param: MOTION_SMOOTH_S window +10%", Kind::Param, WoKnobs { smooth_window: 5.5, ..wo_base() });
    wo_arm(&mut t, "param: MOTION_SMOOTH_S window -90%", Kind::Param, WoKnobs { smooth_window: 0.5, ..wo_base() });
    wo_arm(&mut t, "param: ALIGN_TOLERANCE_S +10% (proxy: nearest tol)", Kind::Param, WoKnobs { near_tol: 2.2, ..wo_base() });
    wo_arm(&mut t, "param: ALIGN_TOLERANCE_S 2 -> 5 (proxy: nearest tol)", Kind::Param, WoKnobs { near_tol: 5.0, ..wo_base() });
    wo_arm(&mut t, "param: HR_MARGIN_BPM +10% (proxy: bridge hr_floor 82.5)", Kind::Param, WoKnobs { bridge_floor: 82.5, ..wo_base() });
    wo_arm(&mut t, "param: HR_MARGIN_BPM 15 -> 50 (proxy: bridge hr_floor 110)", Kind::Param, WoKnobs { bridge_floor: 110.0, ..wo_base() });
    t
}

// ── 14. Workout: detect (no shipped golden) ────────────────────────────────────

fn workout_fixture(bout_len: i64, bout_bpm: i32, motion_amp: f64) -> (Vec<HrSample>, Vec<GravitySample>) {
    let mut hr = Vec::new();
    let mut g = Vec::new();
    let start = 900i64;
    let end = start + bout_len;
    for t in 0..3600i64 {
        let inside = t >= start && t < end;
        hr.push(HrSample { ts: t, bpm: if inside { bout_bpm } else { 60 } });
        let x = if inside && t % 2 == 0 { motion_amp } else { 0.0 };
        g.push(GravitySample { ts: t, x, y: 0.0, z: 1.0 });
    }
    (hr, g)
}

/// workout.rs `detect_pins_every_field_of_one_bout_and_refuses_a_still_or_motionless_day`.
const WO_DETECT_STRAIN: f64 = 46.24;
const WO_DETECT_KCAL: f64 = 446.404432759732;

/// The four dials one `detect` arm varies: bout length, plateau HR, motion amplitude, and the
/// resting / max HR pair the zones are cut from.
struct DetectArm {
    bout_len: i64,
    bpm: i32,
    amp: f64,
    rest: f64,
    max: f64,
}

fn detect_arm(t: &mut Table, name: &str, kind: Kind, a: DetectArm) {
    let DetectArm { bout_len, bpm, amp, rest, max } = a;
    let (hr, g) = workout_fixture(bout_len, bpm, amp);
    let s = workout::detect(&hr, &g, Some(rest), Some(max), Some(35.0), 80.0, 180.0, "male");
    let value = s.first().and_then(|b| b.strain).unwrap_or(f64::NAN);
    let pass = s.len() == 1
        && s[0].strain == Some(WO_DETECT_STRAIN)
        && s[0].calories_kcal.is_some_and(|k| (k - WO_DETECT_KCAL).abs() < 1e-9)
        && (s[0].avg_hr - 150.0).abs() < 1e-9
        && s[0].peak_hr == 150
        && (s[0].duration_s - 1792.0).abs() < 1e-9
        && s[0].avg_hrr_pct == Some(69.2)
        && s[0].hrmax == Some(190.0)
        && s[0].zone_time_pct == vec![(0, 0.0), (1, 0.0), (2, 100.0), (3, 0.0), (4, 0.0), (5, 0.0)];
    t.arm(format!("{name} [bouts {}]", s.len()), kind, value, pass);
}

fn workout_detect_table() -> Table {
    let mut t = Table::new(
        "Workout: detect (bout strain / calories / avg HR)",
        "workout.rs detect_pins_every_field_of_one_bout... (1 bout, strain 46.24, 446.4044 kcal, avg/peak 150, 1792 s)",
    );
    detect_arm(&mut t, "baseline (unmutated)", Kind::Baseline, DetectArm { bout_len: 1800, bpm: 150, amp: 0.3, rest: 60.0, max: 190.0 });
    detect_arm(&mut t, "null: motion flat (MOTION_THRESHOLD never cleared)", Kind::Null, DetectArm { bout_len: 1800, bpm: 150, amp: 0.0, rest: 60.0, max: 190.0 });
    detect_arm(&mut t, "null: HR flat at resting (HR_MARGIN_BPM never cleared)", Kind::Null, DetectArm { bout_len: 1800, bpm: 60, amp: 0.3, rest: 60.0, max: 190.0 });
    detect_arm(&mut t, "struct: bout shortened to 300 s (MIN_EXERCISE_MIN edge)", Kind::Structural, DetectArm { bout_len: 300, bpm: 150, amp: 0.3, rest: 60.0, max: 190.0 });
    detect_arm(&mut t, "struct: bout shortened to 260 s (under the edge)", Kind::Structural, DetectArm { bout_len: 260, bpm: 150, amp: 0.3, rest: 60.0, max: 190.0 });
    detect_arm(&mut t, "param: MOTION_THRESHOLD 0.20 -> 0.22 (proxy: amp 0.22)", Kind::Param, DetectArm { bout_len: 1800, bpm: 150, amp: 0.22, rest: 60.0, max: 190.0 });
    detect_arm(&mut t, "param: MOTION_THRESHOLD 0.20 -> 0.18 (proxy: amp 0.19)", Kind::Param, DetectArm { bout_len: 1800, bpm: 150, amp: 0.19, rest: 60.0, max: 190.0 });
    detect_arm(&mut t, "param: resting_hr 60 -> 66 (+10%)", Kind::Param, DetectArm { bout_len: 1800, bpm: 150, amp: 0.3, rest: 66.0, max: 190.0 });
    detect_arm(&mut t, "param: max_hr 190 -> 209 (+10%)", Kind::Param, DetectArm { bout_len: 1800, bpm: 150, amp: 0.3, rest: 60.0, max: 209.0 });
    t
}

// ── 15. HR recovery ────────────────────────────────────────────────────────────

/// hr_recovery.rs:159 (full), :189 (partial coverage), :204 (a signed rise).
const HRR_END: i64 = 10_000;
const HRR_FULL: HrRecovery =
    HrRecovery { end_hr: 170, after_1min: Some(24), after_2min: Some(38), after_5min: Some(58) };
const HRR_PARTIAL: HrRecovery =
    HrRecovery { end_hr: 170, after_1min: Some(20), after_2min: None, after_5min: None };
const HRR_RISE: i32 = -5;

fn hrr_dense(end_hr: i32) -> Vec<HrSample> {
    (HRR_END - 300..=HRR_END)
        .map(|ts| HrSample { ts, bpm: if ts >= HRR_END - 30 { end_hr } else { 145 } })
        .collect()
}

fn hrr_window(minutes: i64, values: &[i32]) -> Vec<HrSample> {
    let target = HRR_END + minutes * 60;
    values
        .iter()
        .enumerate()
        .map(|(i, &bpm)| HrSample { ts: target - values.len() as i64 / 2 + i as i64, bpm })
        .collect()
}

fn hrr_concat(parts: &[Vec<HrSample>]) -> Vec<HrSample> {
    parts.iter().flatten().copied().collect()
}

/// Mean-instead-of-median mutant for one reading, computed over the same +/-15 s window.
fn hrr_mean_reading(samples: &[HrSample], workout_end: i64, end_hr: i32, minutes: i64) -> Option<i32> {
    let target = workout_end + minutes * 60;
    let v: Vec<i32> = samples
        .iter()
        .filter(|s| (s.ts - target).abs() <= 15)
        .map(|s| s.bpm)
        .collect();
    if v.len() < 3 {
        return None;
    }
    let mean = v.iter().sum::<i32>() as f64 / v.len() as f64;
    Some(end_hr - (mean + 0.5).floor() as i32)
}

fn hrr_arm(
    t: &mut Table,
    name: &str,
    kind: Kind,
    max_hr: f64,
    prep: impl Fn(&[HrSample]) -> Vec<HrSample>,
    engine: impl Fn(&[HrSample], i64, i64, f64) -> Option<HrRecovery>,
) {
    let full = hrr_concat(&[
        hrr_dense(170),
        hrr_window(1, &[146, 146, 220, 146, 146]),
        hrr_window(2, &[132, 132, 132]),
        hrr_window(5, &[112, 112, 112]),
    ]);
    let partial = hrr_concat(&[hrr_dense(170), hrr_window(1, &[150, 150, 150]), hrr_window(5, &[110, 110])]);
    let rise = hrr_concat(&[hrr_dense(160), hrr_window(1, &[165, 165, 165])]);
    let a = engine(&prep(&full), HRR_END - 300, HRR_END, max_hr);
    let b = engine(&prep(&partial), HRR_END - 300, HRR_END, max_hr);
    let c = engine(&prep(&rise), HRR_END - 300, HRR_END, max_hr);
    let pass = a == Some(HRR_FULL) && b == Some(HRR_PARTIAL) && c.and_then(|r| r.after_1min) == Some(HRR_RISE);
    let value = a.and_then(|r| r.after_1min).map_or(f64::NAN, f64::from);
    t.arm(name, kind, value, pass);
}

fn hr_recovery_table() -> Table {
    let keep = |h: &[HrSample]| h.to_vec();
    let shift = |by: i64| move |h: &[HrSample]| {
        h.iter()
            .map(|s| if s.ts > HRR_END { HrSample { ts: s.ts + by, bpm: s.bpm } } else { *s })
            .collect::<Vec<HrSample>>()
    };
    let mut t = Table::new(
        "HR recovery: 1/2/5-minute post-bout drop",
        "hr_recovery.rs:159 {170, 24, 38, 58} + :189 {170, 20, None, None} + :204 after_1min == -5 (exact equality)",
    );
    hrr_arm(&mut t, "baseline (unmutated)", Kind::Baseline, 200.0, keep, hr_recovery::calculate);
    hrr_arm(&mut t, "null[hard]: every reading pinned to 0", Kind::Null, 200.0, keep, |s, a, b, m| {
        hr_recovery::calculate(s, a, b, m)
            .map(|r| HrRecovery { after_1min: Some(0), after_2min: Some(0), after_5min: Some(0), ..r })
    });
    hrr_arm(&mut t, "null: output pinned to None", Kind::Null, 200.0, keep, |_, _, _, _| None);
    hrr_arm(&mut t, "struct: mean instead of median in the +1 min window", Kind::Structural, 200.0, keep, |s, a, b, m| {
        hr_recovery::calculate(s, a, b, m)
            .map(|r| HrRecovery { after_1min: hrr_mean_reading(s, b, r.end_hr, 1), ..r })
    });
    hrr_arm(&mut t, "struct: post-bout readings shifted +30 s", Kind::Structural, 200.0, shift(30), hr_recovery::calculate);
    hrr_arm(&mut t, "struct: post-bout readings shifted +5 s", Kind::Structural, 200.0, shift(5), hr_recovery::calculate);
    hrr_arm(&mut t, "param: max_hr 200 -> 220 (== ELIGIBILITY_FRACTION +10%)", Kind::Param, 220.0, keep, hr_recovery::calculate);
    hrr_arm(&mut t, "param: max_hr 200 -> 180 (== ELIGIBILITY_FRACTION -10%)", Kind::Param, 180.0, keep, hr_recovery::calculate);
    hrr_arm(&mut t, "param: max_hr +3% (floor probe)", Kind::Param, 206.0, keep, hr_recovery::calculate);
    hrr_arm(&mut t, "param: max_hr +4% (floor probe)", Kind::Param, 208.0, keep, hr_recovery::calculate);
    hrr_arm(&mut t, "param: MEASUREMENT_TOLERANCE_SECONDS 15 -> 16.5 (proxy: +16 s)", Kind::Param, 200.0, shift(16), hr_recovery::calculate);
    t
}

// ── 16. VO2max, Fitness Age, PAI ───────────────────────────────────────────────

/// vo2max.rs:151/:156 (VO2max, 1e-3), :171/:176 (fitness age, 0.05), :196/:201 (PAI, 1e-9).
const VO2_MALE: f64 = 46.275;
const VO2_FEMALE: f64 = 37.72;
const VO2_TOL: f64 = 1e-3;
const FA_FIT: f64 = 28.33;
const FA_UNFIT: f64 = 50.15;
const FA_TOL: f64 = 0.05;
const PAI_HIGH: f64 = 15.0;
const PAI_MODERATE: f64 = 3.75;
const PAI_TOL: f64 = 1e-9;

fn vo2_arm(
    t: &mut Table,
    name: &str,
    kind: Kind,
    v: impl Fn(f64, &str, f64, f64, f64) -> f64,
    fa: impl Fn(f64, &str, f64, f64) -> f64,
    pai: impl Fn(i32, f64, f64) -> f64,
) {
    let male = v(40.0, "male", 90.0, 65.0, 5.0);
    let pass = (male - VO2_MALE).abs() < VO2_TOL
        && (v(40.0, "female", 80.0, 65.0, 5.0) - VO2_FEMALE).abs() < VO2_TOL
        && (fa(40.0, "male", 50.0, 10.0) - FA_FIT).abs() < FA_TOL
        && (fa(40.0, "male", 80.0, 2.0) - FA_UNFIT).abs() < FA_TOL
        && (pai(7, 75.0, 0.8) - PAI_HIGH).abs() < PAI_TOL
        && (pai(3, 40.0, 0.3) - PAI_MODERATE).abs() < PAI_TOL;
    t.arm(name, kind, male, pass);
}

fn vo2max_table() -> Table {
    let v0 = vo2max::estimate_vo2max;
    let f0 = vo2max::fitness_age;
    let p0 = vo2max::physical_activity_index;
    let mut t = Table::new(
        "VO2max / Fitness Age / PAI",
        "vo2max.rs:151 46.275 + :156 37.72 (1e-3), :171 28.33 + :176 50.15 (0.05), :196 15.0 + :201 3.75 (1e-9)",
    );
    vo2_arm(&mut t, "baseline (unmutated)", Kind::Baseline, v0, f0, p0);
    vo2_arm(&mut t, "null[hard]: constant VO2max (population mean 40)", Kind::Null, |_, _, _, _, _| 40.0, f0, p0);
    vo2_arm(&mut t, "null: fitness age = chronological age", Kind::Null, v0, |a, _, _, _| a, p0);
    vo2_arm(&mut t, "null: PAI pinned to its reference (5.0)", Kind::Null, v0, f0, |_, _, _| 5.0);
    vo2_arm(&mut t, "struct: sex ignored (male coeffs for everyone)", Kind::Structural, |a, _, w, r, p| v0(a, "male", w, r, p), f0, p0);
    vo2_arm(&mut t, "struct: resting-HR term sign flipped", Kind::Structural, |a, s, w, r, p| v0(a, s, w, -r, p), f0, p0);
    vo2_arm(&mut t, "param: age coefficient +10% (exact via age)", Kind::Param, |a, s, w, r, p| v0(a * 1.1, s, w, r, p), |a, s, r, p| f0(a * 1.1, s, r, p), p0);
    vo2_arm(&mut t, "param: waist coefficient +10% (exact via waist)", Kind::Param, |a, s, w, r, p| v0(a, s, w * 1.1, r, p), f0, p0);
    vo2_arm(&mut t, "param: resting-HR coefficient +10% (exact via RHR)", Kind::Param, |a, s, w, r, p| v0(a, s, w, r * 1.1, p), |a, s, r, p| f0(a, s, r * 1.1, p), p0);
    vo2_arm(&mut t, "param: PAI coefficient +10% (exact via PA index)", Kind::Param, |a, s, w, r, p| v0(a, s, w, r, p * 1.1), |a, s, r, p| f0(a, s, r, p * 1.1), p0);
    vo2_arm(&mut t, "param: age coefficient +0.01% (floor probe)", Kind::Param, |a, s, w, r, p| v0(a * 1.0001, s, w, r, p), f0, p0);
    vo2_arm(&mut t, "param: age coefficient +0.0002% (floor probe)", Kind::Param, |a, s, w, r, p| v0(a * 1.000_002, s, w, r, p), f0, p0);
    vo2_arm(&mut t, "param: RESTING_HR_REFERENCE 65 -> 71.5 (exact via RHR shift)", Kind::Param, v0, |a, s, r, p| f0(a, s, r - 6.5, p), p0);
    vo2_arm(&mut t, "param: PAI_REFERENCE 5.0 -> 5.5 (exact via PA shift)", Kind::Param, v0, |a, s, r, p| f0(a, s, r, p - 0.5), p0);
    vo2_arm(&mut t, "param: PAI duration + intensity inputs +10%", Kind::Param, v0, f0, |d, m, f| p0(d, m * 1.1, f * 1.1));
    vo2_arm(&mut t, "param: PAI active days +1", Kind::Param, v0, f0, |d, m, f| p0(d + 1, m, f));
    vo2_arm(&mut t, "param: PAI derived from strain instead of minutes", Kind::Param, v0, f0, |d, m, _| {
        vo2max::physical_activity_index_from_strain(d, m * 0.75)
    });
    t
}

// ── 17. IMU activity features ──────────────────────────────────────────────────

/// imu_features.rs:147-148 (cadence +/-0.15, strength > 0.4), :174 (gyro 45 +/- 1), :193 (1 Hz -> None).
const IMU_TARGETS: [f64; 4] = [1.4, 1.8, 2.4, 3.0];
const IMU_CADENCE_TOL: f64 = 0.15;
const IMU_STRENGTH_FLOOR: f64 = 0.4;
const IMU_GYRO_DPS: f64 = 45.0;
const IMU_GYRO_TOL: f64 = 1.0;

fn imu_gait(cadence: f64, amp: f64, seconds: f64, gyro_dps: f64) -> Vec<ImuSample> {
    let rate = 100.0;
    let n = (seconds * rate) as usize;
    (0..n)
        .map(|i| {
            let az = 1.0 + amp * (2.0 * std::f64::consts::PI * cadence * i as f64 / rate).sin();
            ImuSample { ax: 0.0, ay: 0.0, az, gx: gyro_dps, gy: 0.0, gz: 0.0 }
        })
        .collect()
}

fn imu_enmo() -> Vec<ImuSample> {
    (0..600)
        .map(|i| ImuSample { ax: 0.0, ay: 0.0, az: 1.0 + 0.2 * (i as f64 * 0.7).sin(), gx: 0.0, gy: 0.0, gz: 0.0 })
        .collect()
}

fn imu_arm(
    t: &mut Table,
    name: &str,
    kind: Kind,
    rate: i32,
    prep: impl Fn(&[ImuSample]) -> Vec<ImuSample>,
    engine: impl Fn(&[ImuSample], i32) -> ImuActivityFeatures,
) {
    let mut pass = true;
    let mut value = f64::NAN;
    for target in IMU_TARGETS {
        let f = engine(&prep(&imu_gait(target, 0.2, 6.0, 0.0)), rate);
        let ok = matches!(f.cadence_hz, Some(c) if (c - target).abs() <= IMU_CADENCE_TOL)
            && f.cadence_strength > IMU_STRENGTH_FLOOR;
        pass = pass && ok;
        if target == 2.4 {
            value = f.cadence_hz.unwrap_or(f64::NAN);
        }
    }
    let gyro = engine(&prep(&imu_gait(2.0, 0.2, 6.0, IMU_GYRO_DPS)), rate);
    pass = pass && (gyro.gyro_energy_dps - IMU_GYRO_DPS).abs() <= IMU_GYRO_TOL;
    let enmo = engine(&prep(&imu_enmo()), 1);
    pass = pass && enmo.cadence_hz.is_none() && enmo.cadence_strength == 0.0;
    t.arm(name, kind, value, pass);
}

fn imu_table() -> Table {
    let keep = |s: &[ImuSample]| s.to_vec();
    let mut t = Table::new(
        "IMU activity features (cadence, gyro energy, 1 Hz refusal)",
        "imu_features.rs:147-148 cadence +/-0.15 and strength > 0.4, :174 gyro 45 +/- 1, :193 a 1 Hz source has no cadence",
    );
    imu_arm(&mut t, "baseline (unmutated)", Kind::Baseline, 100, keep, imu_features::extract);
    imu_arm(&mut t, "null[hard]: constant cadence 2.0 Hz for every input", Kind::Null, 100, keep, |s, r| {
        ImuActivityFeatures { cadence_hz: Some(2.0), cadence_strength: 1.0, ..imu_features::extract(s, r) }
    });
    imu_arm(&mut t, "null: cadence never reported", Kind::Null, 100, keep, |s, r| {
        ImuActivityFeatures { cadence_hz: None, ..imu_features::extract(s, r) }
    });
    imu_arm(&mut t, "struct: window reversed in time", Kind::Structural, 100, |s| s.iter().rev().copied().collect(), imu_features::extract);
    imu_arm(&mut t, "struct: accel amplitude x0.1", Kind::Structural, 100, |s| {
        s.iter().map(|x| ImuSample { az: 1.0 + (x.az - 1.0) * 0.1, ..*x }).collect()
    }, imu_features::extract);
    imu_arm(&mut t, "struct: window halved (3 s)", Kind::Structural, 100, |s| s[..s.len() / 2].to_vec(), imu_features::extract);
    imu_arm(&mut t, "param: sample_rate_hz 100 -> 110 (+10%)", Kind::Param, 110, keep, imu_features::extract);
    imu_arm(&mut t, "param: sample_rate_hz 100 -> 105 (+5%)", Kind::Param, 105, keep, imu_features::extract);
    imu_arm(&mut t, "param: sample_rate_hz 100 -> 101 (+1%, floor probe)", Kind::Param, 101, keep, imu_features::extract);
    // Raising the strength floor only ever turns a reported cadence into None, so this is exact.
    imu_arm(&mut t, "param: MIN_CADENCE_STRENGTH 0.20 -> 0.22 (+10%, exact)", Kind::Param, 100, keep, |s, r| {
        let f = imu_features::extract(s, r);
        if f.cadence_strength < imu_features::MIN_CADENCE_STRENGTH * 1.1 {
            ImuActivityFeatures { cadence_hz: None, ..f }
        } else {
            f
        }
    });
    imu_arm(&mut t, "param: MIN_CADENCE_STRENGTH 0.20 -> 0.60 (exact)", Kind::Param, 100, keep, |s, r| {
        let f = imu_features::extract(s, r);
        if f.cadence_strength < 0.60 {
            ImuActivityFeatures { cadence_hz: None, ..f }
        } else {
            f
        }
    });
    t
}

// ── The control run ────────────────────────────────────────────────────────────

// ── Sensitivity floors ─────────────────────────────────────────────────────────────────────────

/// `(metric, arm, minimum |delta| from the baseline)`. A floor asserts the arm still MOVES the number,
/// which is what catches an algorithm that stopped being reached; each is 0.45x the delta measured
/// 2026-08-02, so it sits well below the observed move and well above zero.
const FLOORS: &[(&str, &str, f64)] = &[
    ("Strain: TRIMP -> 0-100 map (trimp_to_strain)", "null[hard]: output pinned to 0", 35.0),
    ("Strain: TRIMP -> 0-100 map (trimp_to_strain)", "null: constant scorer (always the 100-TRIMP answer)", 11.6),
    ("Strain: TRIMP -> 0-100 map (trimp_to_strain)", "struct: linear map instead of logarithmic", 28.7),
    ("Strain: TRIMP -> 0-100 map (trimp_to_strain)", "struct: ln(TRIMP) instead of ln(TRIMP+1)", 0.0045),
    ("Strain: TRIMP -> 0-100 map (trimp_to_strain)", "struct: rounded to whole points instead of 2 dp", 0.099),
    ("Strain: Edwards zone goldens (strain, Method::Edwards)", "null[hard]: output pinned to 0", 17.3),
    ("Strain: Edwards zone goldens (strain, Method::Edwards)", "null: HR ignored (every sample at the resting value)", 17.3),
    ("Strain: Edwards zone goldens (strain, Method::Edwards)", "null: constant scorer (always the 115-bpm answer)", 5.24),
    ("Strain: Edwards zone goldens (strain, Method::Edwards)", "struct: cadence halved (301 samples, 2 s, same 600 s span)", 0.018),
    ("Strain: personal denominator fit (fit_strain_denominator)", "null[hard]: reference strains carry no information (all 50.0)", 317000.0),
    ("Strain: personal denominator fit (fit_strain_denominator)", "null: TRIMP/strain pairing shuffled (reversed)", 3470.0),
    ("Strain: personal denominator fit (fit_strain_denominator)", "struct: only the two smallest pairs kept", 0.171),
    ("Strain: personal denominator fit (fit_strain_denominator)", "struct: reference strains offset +2%", 517.0),
    ("Strain: personal denominator fit (fit_strain_denominator)", "struct: reference strains offset +0.2%", 56.9),
    ("Strain: personal denominator fit (fit_strain_denominator)", "struct: reference strains offset +0.1% (floor probe)", 28.6),
    ("HR-max: Tanaka 208 - 0.7*age", "null[hard]: constant scorer (age ignored, 190)", 1.35),
    ("HR-max: Tanaka 208 - 0.7*age", "struct: 220 - age instead of Tanaka", 1.35),
    ("HR-max: Tanaka 208 - 0.7*age", "struct: age read in months", 103.0),
    ("HR zones: %HRmax edges + zone_number", "struct: edges shifted one band up", 9.0),
    ("Time in zone (seconds per zone + gap provenance)", "null[hard]: one second per sample (elapsed time ignored)", 404.0),
    ("Time in zone (seconds per zone + gap provenance)", "struct: last sample dropped", 405.0),
    ("Calories: whole-day total under the day gate (BMR only)", "null[hard]: output pinned to 0", 821.0),
    ("Calories: whole-day total under the day gate (BMR only)", "null: constant population day (2000 kcal)", 78.6),
    ("Calories: whole-day total under the day gate (BMR only)", "struct: last 10% of the day dropped", 82.1),
    ("Calories: whole-day total under the day gate (BMR only)", "struct: every sample duplicated at the same second", 821.0),
    ("Calories: active energy above the day gate (Keytel, day path)", "null[hard]: output pinned to 0", 1119.0),
    ("Calories: active energy above the day gate (Keytel, day path)", "null: Keytel disabled (every sample billed the resting rate)", 297.0),
    ("Calories: active energy above the day gate (Keytel, day path)", "null: constant population day (2000 kcal)", 219.0),
    ("Calories: active energy above the day gate (Keytel, day path)", "struct: last 10% of the day dropped (loses both over-gate blocks)", 380.0),
    ("Calories: active energy above the day gate (Keytel, day path)", "struct: every sample duplicated at the same second", 1119.0),
    ("Calories: active energy above the day gate (Keytel, day path)", "struct: the 125 bpm block dropped to 119 (just under the gate)", 133.0),
    ("Calories: day path equals bout path at 1 Hz ABOVE BOTH ACTIVITY GATES", "null[hard]: output pinned to 0", 53.6),
    ("Calories: day path equals bout path at 1 Hz ABOVE BOTH ACTIVITY GATES", "null: Keytel disabled (resting rate for every sample)", 47.9),
    ("Calories: day path equals bout path at 1 Hz ABOVE BOTH ACTIVITY GATES", "struct: every sample duplicated at the same second", 53.6),
    ("Calories: resolve_coeffs identities", "struct: male and female sets swapped", 161.0),
    ("Steps: wrap-aware motion-tick total", "null: sample count instead of counter deltas", 52.6),
    ("Steps: wrap-aware motion-tick total", "struct: only the first two samples kept", 31.5),
    ("Workout: gated helpers (series, nearest, smoothing, zone %, bridge)", "null[hard]: zone classifier pinned below Zone 1", 45.0),
    ("Workout: gated helpers (series, nearest, smoothing, zone %, bridge)", "null: zone classifier pinned to Zone 1", 45.0),
    ("HR recovery: 1/2/5-minute post-bout drop", "null[hard]: every reading pinned to 0", 10.8),
    ("HR recovery: 1/2/5-minute post-bout drop", "struct: mean instead of median in the +1 min window", 6.75),
    ("VO2max / Fitness Age / PAI", "null[hard]: constant VO2max (population mean 40)", 2.82),
    ("VO2max / Fitness Age / PAI", "struct: resting-HR term sign flipped", 9.06),
    ("IMU activity features (cadence, gyro energy, 1 Hz refusal)", "null[hard]: constant cadence 2.0 Hz for every input", 0.171),
];

/// `(metric, arm, why)`. Probe arms that cannot carry a floor, because the mutation does not move the
/// number at all. Their blindness is the finding, not a defect to assert away.
const NO_FLOOR: &[(&str, &str, &str)] = &[
    ("Strain: Edwards zone goldens (strain, Method::Edwards)", "struct: every bpm +1 (inside its zone)", "measured delta is exactly zero: this mutation does not move the number"),
    ("Strain: Edwards zone goldens (strain, Method::Edwards)", "struct: timestamps reversed (series runs backwards)", "measured delta is exactly zero: this mutation does not move the number"),
    ("Strain: Edwards zone goldens (strain, Method::Edwards)", "struct: last 10% of the series dropped", "the arm yields no number, so it has no distance from the baseline"),
    ("HR zones: %HRmax edges + zone_number", "null[hard]: constant classifier (everything is Zone 3)", "measured delta is exactly zero: this mutation does not move the number"),
    ("HR zones: %HRmax edges + zone_number", "struct: zone numbers reversed (1<->5)", "measured delta is exactly zero: this mutation does not move the number"),
    ("Time in zone (seconds per zone + gap provenance)", "null: every gap credited in full (hr_gap policy ignored)", "measured delta is exactly zero: this mutation does not move the number"),
    ("Time in zone (seconds per zone + gap provenance)", "struct: sample order reversed", "measured delta is exactly zero: this mutation does not move the number"),
    ("Time in zone (seconds per zone + gap provenance)", "struct: whole series shifted +3600 s", "measured delta is exactly zero: this mutation does not move the number"),
    ("hr_gap: position ceilings (lead 600 / interior 1800 / trail 300 s)", "null[hard]: no ceiling at all (every gap credited)", "measured delta is exactly zero: this mutation does not move the number"),
    ("hr_gap: position ceilings (lead 600 / interior 1800 / trail 300 s)", "null: nothing ever credited", "measured delta is exactly zero: this mutation does not move the number"),
    ("hr_gap: position ceilings (lead 600 / interior 1800 / trail 300 s)", "struct: lead and trail ceilings swapped", "measured delta is exactly zero: this mutation does not move the number"),
    ("hr_gap: position ceilings (lead 600 / interior 1800 / trail 300 s)", "struct: one flat ceiling for every position", "measured delta is exactly zero: this mutation does not move the number"),
    ("Calories: whole-day total under the day gate (BMR only)", "null: Keytel disabled (every sample billed the resting rate)", "every sample is already under the day gate, so disabling Keytel changes nothing here — the active-day table is what sees it"),
    ("Calories: whole-day total under the day gate (BMR only)", "path check: resting_kcal_per_s x 86400 reproduces the day", "measured delta is exactly zero: this mutation does not move the number"),
    ("Calories: whole-day total under the day gate (BMR only)", "struct: sample order reversed", "measured delta is exactly zero: this mutation does not move the number"),
    ("Calories: active energy above the day gate (Keytel, day path)", "path check: resting x 82800 + Keytel x 3600 reproduces the day", "measured delta is exactly zero: this mutation does not move the number"),
    ("Calories: active energy above the day gate (Keytel, day path)", "struct: sample order reversed", "measured delta is exactly zero: this mutation does not move the number"),
    ("Calories: resolve_coeffs identities", "null[hard]: one coeff set for every sex (male)", "measured delta is exactly zero: this mutation does not move the number"),
    ("Calories: resolve_coeffs identities", "struct: unknown sex falls back to male, not nonbinary", "measured delta is exactly zero: this mutation does not move the number"),
    ("Steps: wrap-aware motion-tick total", "null[hard]: output pinned to None", "the arm yields no number, so it has no distance from the baseline"),
    ("Steps: wrap-aware motion-tick total", "null: last minus first, no wrap and no ceiling", "measured delta is exactly zero: this mutation does not move the number"),
    ("Steps: wrap-aware motion-tick total", "struct: counters reversed (series runs backwards)", "the arm yields no number, so it has no distance from the baseline"),
    ("Steps: wrap-aware motion-tick total", "struct: middle sample dropped", "measured delta is exactly zero: this mutation does not move the number"),
    ("Steps: MAX_STEP_DELTA ceiling is exclusive", "null[hard]: no ceiling (every forward delta accepted)", "measured delta is exactly zero: this mutation does not move the number"),
    ("Workout: gated helpers (series, nearest, smoothing, zone %, bridge)", "struct: bout HR series reversed", "measured delta is exactly zero: this mutation does not move the number"),
    ("Workout: gated helpers (series, nearest, smoothing, zone %, bridge)", "struct: last 10% of the bout dropped", "measured delta is exactly zero: this mutation does not move the number"),
    ("Workout: detect (bout strain / calories / avg HR)", "null: motion flat (MOTION_THRESHOLD never cleared) [bouts 0]", "the arm yields no number, so it has no distance from the baseline"),
    ("Workout: detect (bout strain / calories / avg HR)", "null: HR flat at resting (HR_MARGIN_BPM never cleared) [bouts 0]", "the arm yields no number, so it has no distance from the baseline"),
    ("Workout: detect (bout strain / calories / avg HR)", "struct: bout shortened to 300 s (MIN_EXERCISE_MIN edge) [bouts 1]", "the arm yields no number, so it has no distance from the baseline"),
    ("Workout: detect (bout strain / calories / avg HR)", "struct: bout shortened to 260 s (under the edge) [bouts 0]", "the arm yields no number, so it has no distance from the baseline"),
    ("HR recovery: 1/2/5-minute post-bout drop", "null: output pinned to None", "the arm yields no number, so it has no distance from the baseline"),
    ("HR recovery: 1/2/5-minute post-bout drop", "struct: post-bout readings shifted +30 s", "the arm yields no number, so it has no distance from the baseline"),
    ("HR recovery: 1/2/5-minute post-bout drop", "struct: post-bout readings shifted +5 s", "measured delta is exactly zero: this mutation does not move the number"),
    ("VO2max / Fitness Age / PAI", "null: fitness age = chronological age", "measured delta is exactly zero: this mutation does not move the number"),
    ("VO2max / Fitness Age / PAI", "null: PAI pinned to its reference (5.0)", "measured delta is exactly zero: this mutation does not move the number"),
    ("VO2max / Fitness Age / PAI", "struct: sex ignored (male coeffs for everyone)", "measured delta is exactly zero: this mutation does not move the number"),
    ("IMU activity features (cadence, gyro energy, 1 Hz refusal)", "null: cadence never reported", "the arm yields no number, so it has no distance from the baseline"),
    ("IMU activity features (cadence, gyro energy, 1 Hz refusal)", "struct: window reversed in time", "measured delta is exactly zero: this mutation does not move the number"),
    ("IMU activity features (cadence, gyro energy, 1 Hz refusal)", "struct: accel amplitude x0.1", "measured delta is exactly zero: this mutation does not move the number"),
    ("IMU activity features (cadence, gyro energy, 1 Hz refusal)", "struct: window halved (3 s)", "measured delta is exactly zero: this mutation does not move the number"),
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

/// Cohort guard. A hand-kept run list silently shrinks: drop a table and the family's caught/missed
/// improves. Reads this file's own source, so a builder that stops being run is named here.
fn assert_the_run_list_covers_every_declared_table(run: &[Table]) {
    let declared: Vec<&str> = include_str!("sensitivity_strain.rs")
        .lines()
        .filter(|l| l.ends_with("-> Table {"))
        .filter_map(|l| l.strip_prefix("fn "))
        .filter_map(|l| l.split('(').next())
        .collect();
    assert_eq!(
        declared.len(),
        run.len(),
        "{} table builders are declared but {} are run: {declared:?}",
        declared.len(),
        run.len()
    );
    for t in run {
        assert!(
            t.arms.iter().any(|a| a.kind == Kind::Baseline),
            "{}: no baseline arm, so nothing here reproduces a shipped figure",
            t.metric
        );
        assert!(
            t.arms.iter().any(|a| a.kind == Kind::Null),
            "{}: no null arm, so a do-nothing scorer is untested on this metric",
            t.metric
        );
    }
}

#[test]
#[ignore = "negative control: prints a sensitivity table for the strain family, not a CI gate"]
fn strain_family_sensitivity() {
    let sedentary = cal_sedentary();
    let light = cal_light();
    let active = cal_active();
    let tables = vec![
        trimp_map_table(),
        edwards_table(),
        fit_table(),
        tanaka_table(),
        zone_edges_table(),
        time_in_zone_table(),
        hr_gap_table(),
        calories_day_table(&sedentary, &light),
        calories_active_day_table(&active),
        calories_bout_table(),
        calories_coeff_table(),
        steps_table(),
        steps_ceiling_table(),
        workout_helper_table(),
        workout_detect_table(),
        hr_recovery_table(),
        vo2max_table(),
        imu_table(),
    ];

    assert_the_run_list_covers_every_declared_table(&tables);

    println!("\n=== negative control: STRAIN family ===");
    let mut totals = Tally::default();
    let mut summary: Vec<(&'static str, Tally)> = Vec::new();
    for t in &tables {
        let tally = print_table(t);
        totals.caught += tally.caught;
        totals.missed += tally.missed;
        totals.ungated += tally.ungated;
        if let Some(f) = tally.floor {
            totals.floor = Some(totals.floor.map_or(f, |g: f64| g.min(f)));
        }
        summary.push((t.metric, tally));
    }

    println!("\n=== summary ===");
    println!("  {:<62} {:>7} {:>7} {:>8} {:>14}", "metric", "caught", "missed", "ungated", "floor |delta|");
    for (metric, tally) in &summary {
        let floor = tally.floor.map_or("n/a".to_string(), |f| format!("{f:.4}"));
        println!(
            "  {:<62} {:>7} {:>7} {:>8} {:>14}",
            metric, tally.caught, tally.missed, tally.ungated, floor
        );
    }
    println!("\nSTRAIN FAMILY: caught {}, missed {}, ungated {}", totals.caught, totals.missed, totals.ungated);

    // Only two things are asserted: the baseline reproduces, and the do-nothing null is caught.
    for t in &tables {
        let base = t.arms.iter().find(|a| a.kind == Kind::Baseline).unwrap_or_else(|| panic!("{}: no baseline arm", t.metric));
        if base.verdict != Verdict::NoGate {
            assert!(base.verdict == Verdict::Pass, "{}: baseline does not reproduce the shipped figure", t.metric);
        }
        if let Some(hard) = t.arms.iter().find(|a| a.kind == Kind::Null && a.name.starts_with("null[hard]")) {
            assert!(
                hard.verdict != Verdict::Pass,
                "CRITICAL — {}: the do-nothing null PASSED the shipped gate ({})",
                t.metric,
                hard.name
            );
        }
    }
}
