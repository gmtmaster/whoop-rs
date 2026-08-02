//! Negative controls for the **stress** family. This file does not test the algorithms; it tests
//! the algorithms' own gates.
//!
//! The claim it falsifies: *"the shipped stress gates would notice if the stress code changed."*
//! A gate that only reproduces the number its own code produced today is a REPRODUCTION check, not
//! a REGRESSION check, and the difference is invisible until something drifts. So each metric gets
//! three families of arm:
//!
//! * **NULL** — a scorer that does no work (a constant, the mean, the input never read). The gate
//!   MUST fail. If it passes, the gate is fake, and this file asserts that it does not.
//! * **STRUCTURAL** — a wrong SHAPE with the right magnitude (reversed, shuffled, shifted, sign
//!   flipped, channel-swapped, truncated, offset). These teach what the metric is made of.
//! * **PARAMETER** — every tunable constant moved +-10%, plus one +0.5% floor probe. Whether these
//!   are caught is the measurement; this file deliberately does NOT assert that they fail.
//!
//! Every parameter arm runs against a LOCAL TWIN of the shipped function, because the shipped
//! constants are private `const`s an integration test cannot reach and widening their visibility to
//! measure them would be the tail wagging the dog. Arm 1 of every table is the twin at the SHIPPED
//! constants, and this file asserts it passes the shipped gate — a twin that does not reproduce the
//! shipped result makes every parameter arm below it meaningless. That row is a fidelity check, not
//! a perturbation, so the caught/missed tally excludes it along with the baseline. Where a constant
//! is reachable for
//! real (`StressWindowCfg::bucket_seconds`, `::sustained_buckets`, `::activity_gate_g`) the arm
//! drives the SHIPPED function and says so.
//!
//! Each gate's exact target and tolerance is copied into a `const` below, tagged with the
//! `file:line` it came from, so the arms are scored against the real claim rather than a paraphrase.
//!
//! No external fixtures: every stress gate is an in-source literal, so nothing here can silently
//! skip. The tests are `#[ignore]`d so they stay out of CI — they are instruments, not gates.
//!
//!   cargo test --release -p physio-algo --test sensitivity_stress -- --ignored --nocapture
//!
//! Everything measured here is a wellness estimate, never medical and never diagnostic.

use physio_algo::hrv::HrvReadiness;
use physio_algo::stats::{median, percentile};
use physio_algo::stress::{
    band_of, components_raw, daily_stress, daytime_stress, squash, stress_index_raw,
    windowed_stress, HourPoint, SpanMs, StressBand, StressDay, StressWindowCfg, StressWindows,
    HIGH_BAND_FLOOR, MEDIUM_BAND_FLOOR, STRESS_MAX,
};
use physio_algo::stress_onset::{evaluate, OnsetReason, OnsetState};

// ─────────────────────────────────────────────────────────────────────────────────────────────
// reporting
// ─────────────────────────────────────────────────────────────────────────────────────────────

/// Arm 1 of every table: the local twin at the SHIPPED constants, a fidelity check the file
/// REQUIRES to pass. `print` excludes it from the caught/missed tally and checks the name matches.
const FIDELITY_ARM: usize = 1;
const FIDELITY_ARM_NAME: &str = "twin at shipped constants";

/// First arm that is an actual perturbation. Arms below it are the baseline and the fidelity twin.
const FIRST_PERTURBATION_ARM: usize = FIDELITY_ARM + 1;

/// One arm: what was perturbed, what the metric read, and whether the SHIPPED gate still passed.
struct Arm {
    name: String,
    value: Option<f64>,
    pass: bool,
}

/// One metric's arms, with the gate they are all scored against.
struct Table {
    metric: &'static str,
    gate_ref: &'static str,
    arms: Vec<Arm>,
}

impl Table {
    fn new(metric: &'static str, gate_ref: &'static str) -> Self {
        Self { metric, gate_ref, arms: Vec::new() }
    }

    fn push(&mut self, name: impl Into<String>, value: Option<f64>, pass: bool) {
        self.arms.push(Arm { name: name.into(), value, pass });
    }

    fn baseline(&self) -> Option<f64> {
        self.arms.first().and_then(|a| a.value)
    }

    /// Print the table. Returns `(caught, missed, smallest caught non-zero |delta|)`.
    ///
    /// Only arms `FIRST_PERTURBATION_ARM..` are scored. Arm 0 is the shipped baseline and arm 1 is
    /// the twin at the SHIPPED constants; both are REQUIRED to pass by `assert_trustworthy`, so
    /// counting either as "missed" would report a fidelity check as gate blindness.
    fn print(&self) -> (usize, usize, Option<f64>) {
        assert!(
            self.arms.len() > FIRST_PERTURBATION_ARM,
            "{}: fewer arms than the two fixed header rows",
            self.metric
        );
        assert!(
            self.arms[FIDELITY_ARM].name.starts_with(FIDELITY_ARM_NAME),
            "{}: arm {FIDELITY_ARM} is '{}', not the twin-fidelity row — the tally excludes that \
             index, so a reordered table would silently exempt a real perturbation arm",
            self.metric,
            self.arms[FIDELITY_ARM].name
        );
        let base = self.baseline();
        println!("\n== {} ==", self.metric);
        println!("   shipped gate: {}", self.gate_ref);
        println!("   {:<62} {:>18} {:>16}  shipped gate", "arm", "value", "delta");
        let mut caught = 0usize;
        let mut missed = 0usize;
        let mut floor: Option<f64> = None;
        for (i, a) in self.arms.iter().enumerate() {
            let value_s = match a.value {
                Some(v) => format!("{v:>18.9}"),
                None => format!("{:>18}", "none"),
            };
            let delta = match (a.value, base) {
                (Some(v), Some(b)) => Some(v - b),
                _ => None,
            };
            let delta_s = match delta {
                Some(d) => fmt_delta(d),
                None => format!("{:>16}", "n/a"),
            };
            let verdict = if i == 0 {
                if a.pass { "PASS (baseline, expected)" } else { "FAIL  <-- BASELINE BROKEN" }
            } else if i == FIDELITY_ARM {
                if a.pass {
                    "PASS (twin fidelity, expected; not a perturbation)"
                } else {
                    "FAIL  <-- TWIN NOT FAITHFUL"
                }
            } else if a.pass {
                missed += 1;
                "PASS  <-- MISSED"
            } else {
                caught += 1;
                if let Some(d) = delta {
                    let d = d.abs();
                    // A waived arm is one whose true delta is known to be zero, so whatever the
                    // bit-exact gate saw is float residue and must not be read as a floor.
                    if d > 0.0 && waiver_for(self.metric, &a.name).is_none() {
                        floor = Some(match floor {
                            Some(f) if f <= d => f,
                            _ => d,
                        });
                    }
                }
                "FAIL  <-- caught"
            };
            println!("   {:<62} {value_s} {delta_s}  {verdict}", a.name);
        }
        println!(
            "   caught {caught}, missed {missed} of {} perturbation arms (arms 0-{FIDELITY_ARM} \
             are the baseline and twin-fidelity rows, both required to pass, so neither is scored)",
            caught + missed
        );
        match floor {
            Some(f) => println!("   smallest REAL delta this gate catches: {f:.3e}"),
            None => println!(
                "   smallest REAL delta this gate catches: n/a (no caught arm moved the reported \
                 number)"
            ),
        }
        let probes: Vec<(&str, f64)> = self
            .arms
            .iter()
            .filter(|a| a.name.starts_with("null") || a.name.starts_with("structural"))
            .map(|a| (a.name.as_str(), a.value.unwrap_or(f64::NAN)))
            .collect();
        enforce_floors(self.metric, base.unwrap_or(f64::NAN), &probes);
        (caught, missed, floor)
    }
}

fn fmt_delta(d: f64) -> String {
    if d == 0.0 {
        format!("{:>16}", "+0.000000000")
    } else if d.abs() < 1e-6 {
        format!("{d:>16.3e}")
    } else {
        format!("{d:>+16.9}")
    }
}

/// The only things this file asserts: the baseline reproduces, the twin is faithful, and at least
/// one NULL arm is caught. Everything else is a measurement, not a requirement.
fn assert_trustworthy(t: &Table) {
    assert!(
        t.arms[0].pass,
        "{}: the SHIPPED baseline no longer reproduces its own gate — the harness is broken, or the \
         algorithm moved. Nothing below this line means anything.",
        t.metric
    );
    assert!(
        t.arms[FIDELITY_ARM].pass,
        "{}: the local twin at the SHIPPED constants does NOT reproduce the shipped result, so every \
         parameter arm in this table is measuring the twin's own drift, not the gate's sensitivity.",
        t.metric
    );
    let nulls: Vec<&Arm> = t.arms.iter().filter(|a| a.name.starts_with("null:")).collect();
    assert!(!nulls.is_empty(), "{}: no NULL arm — this control proves nothing", t.metric);
    assert!(
        nulls.iter().any(|a| !a.pass),
        "{}: CRITICAL — EVERY null arm PASSES the shipped gate. The gate does not notice a scorer \
         that does no work, so it is not reaching the algorithm at all.",
        t.metric
    );
}

/// Deterministic shuffle (LCG-driven Fisher-Yates), so a "shuffled" arm is reproducible.
fn shuffled<T: Clone>(xs: &[T], seed: u64) -> Vec<T> {
    let mut v = xs.to_vec();
    let mut s = seed;
    for i in (1..v.len()).rev() {
        s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        let j = ((s >> 33) as usize) % (i + 1);
        v.swap(i, j);
    }
    v
}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// shared twin helpers — byte-for-byte mirrors of the private helpers in stress/mod.rs
// ─────────────────────────────────────────────────────────────────────────────────────────────

/// Mirror of the private `stress::mean_opt` (stress/mod.rs:52-55).
fn mean_opt_twin(xs: &[f64]) -> Option<f64> {
    if xs.is_empty() { None } else { Some(xs.iter().sum::<f64>() / xs.len() as f64) }
}

/// Mirror of the private `stress::population_std` (stress/mod.rs:59-67).
fn pop_std_twin(xs: &[f64], m: Option<f64>) -> f64 {
    let Some(m) = m else { return 0.0 };
    if xs.len() <= 1 {
        return 0.0;
    }
    let var = xs.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / xs.len() as f64;
    var.sqrt()
}

/// Mirror of `stress::squash` (stress/mod.rs:48-50) with `STRESS_MAX` opened as a parameter.
fn squash_twin(raw: f64, stress_max: f64) -> f64 {
    (stress_max / (1.0 + (-raw).exp())).clamp(0.0, stress_max)
}

/// The shipped `SD_FLOOR`, a private const in stress/mod.rs:25. Copied, not imported.
const SD_FLOOR: f64 = 0.0001;

// ─────────────────────────────────────────────────────────────────────────────────────────────
// METRIC 1 — Baevsky Stress Index (SI) and its components
// gate: crates/physio-algo/src/stress/index.rs:131-135, golden vector, |delta| < 1e-9
// ─────────────────────────────────────────────────────────────────────────────────────────────

/// The golden R-R series from stress/index.rs:120-123.
const SI_GOLDEN_RR: [f64; 22] = [
    700.0, 720.0, 740.0, 760.0, 780.0, 800.0, 820.0, 840.0, 860.0, 800.0, 800.0, 800.0, 800.0,
    820.0, 780.0, 800.0, 810.0, 790.0, 800.0, 800.0, 805.0, 795.0,
];

const SI_GATE_MXDMN: f64 = 0.16; // index.rs:131
const SI_GATE_MO: f64 = 0.825; // index.rs:132
const SI_GATE_AMO: f64 = 59.090_909_090_909_09; // index.rs:133
const SI_GATE_SI: f64 = 223.829_201_101_928_36; // index.rs:134 and :135
const SI_GATE_TOL: f64 = 1e-9; // index.rs:131-135

/// Mo, AMo, MxDMn, SI — the four numbers index.rs:131-135 pins.
type SiOut = Option<(f64, f64, f64, f64)>;

fn si_gate(out: SiOut) -> bool {
    match out {
        None => false,
        Some((mo, amo, mxdmn, si)) => {
            (mxdmn - SI_GATE_MXDMN).abs() < SI_GATE_TOL
                && (mo - SI_GATE_MO).abs() < SI_GATE_TOL
                && (amo - SI_GATE_AMO).abs() < SI_GATE_TOL
                && (si - SI_GATE_SI).abs() < SI_GATE_TOL
        }
    }
}

fn si_value(out: SiOut) -> Option<f64> {
    out.map(|(_, _, _, si)| si)
}

/// The shipped path, as the gate calls it. Includes the `stress_index_raw` cross-check at :135.
fn si_shipped(rr: &[f64]) -> SiOut {
    let c = components_raw(rr)?;
    let si_via_raw = stress_index_raw(rr)?;
    if si_via_raw.to_bits() != c.si.to_bits() {
        return None; // index.rs:135 would fail; report it as a gate failure, not a panic
    }
    Some((c.mo_sec, c.amo_percent, c.mxdmn_sec, c.si))
}

/// Every tunable `stress/index.rs:9-21` reads. All private consts, so the twin is the only handle.
#[derive(Clone, Copy)]
struct SiParams {
    bin_width_sec: f64,
    min_beats: usize,
    rr_min_ms: f64,
    rr_max_ms: f64,
    ectopic_threshold: f64,
    ectopic_window_radius: usize,
}

const SI_SHIPPED_PARAMS: SiParams = SiParams {
    bin_width_sec: 0.05, // index.rs:9
    min_beats: 20,       // index.rs:11
    rr_min_ms: 300.0,    // index.rs:14
    rr_max_ms: 2000.0,   // index.rs:15
    ectopic_threshold: 0.20, // index.rs:18
    ectopic_window_radius: 2, // index.rs:20
};

/// Twin of `components_raw` (index.rs:37-77) plus `clean_rr`/`reject_ectopic` (index.rs:80-113).
fn si_twin(rr_ms: &[f64], p: SiParams) -> SiOut {
    let ranged: Vec<f64> =
        rr_ms.iter().copied().filter(|&v| (p.rr_min_ms..=p.rr_max_ms).contains(&v)).collect();
    let clean = si_reject_ectopic(&ranged, p);
    if clean.len() < p.min_beats {
        return None;
    }
    let sec: Vec<f64> = clean.iter().map(|v| v / 1000.0).collect();
    let min_v = sec.iter().copied().fold(f64::INFINITY, f64::min);
    let max_v = sec.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mxdmn = max_v - min_v;
    if mxdmn <= 0.0 {
        return None;
    }
    let bin_count = ((mxdmn / p.bin_width_sec).floor() as usize + 1).max(1);
    let mut counts = vec![0usize; bin_count];
    for &v in &sec {
        let mut idx = ((v - min_v) / p.bin_width_sec).floor() as isize;
        if idx < 0 {
            idx = 0;
        }
        let mut idx = idx as usize;
        if idx >= bin_count {
            idx = bin_count - 1;
        }
        counts[idx] += 1;
    }
    let mut mode_idx = 0usize;
    let mut mode_count = counts[0];
    for (i, &c) in counts.iter().enumerate().skip(1) {
        if c > mode_count {
            mode_count = c;
            mode_idx = i;
        }
    }
    let mo = min_v + (mode_idx as f64 + 0.5) * p.bin_width_sec;
    let amo = mode_count as f64 / sec.len() as f64 * 100.0;
    if mo <= 0.0 {
        return None;
    }
    Some((mo, amo, mxdmn, amo / (2.0 * mo * mxdmn)))
}

/// Twin of `reject_ectopic` (index.rs:88-113).
fn si_reject_ectopic(nn: &[f64], p: SiParams) -> Vec<f64> {
    if nn.len() <= p.ectopic_window_radius {
        return nn.to_vec();
    }
    let mut kept = Vec::with_capacity(nn.len());
    for i in 0..nn.len() {
        let lo = i.saturating_sub(p.ectopic_window_radius);
        let hi = (i + p.ectopic_window_radius).min(nn.len() - 1);
        let mut neighbours: Vec<f64> = Vec::with_capacity(hi - lo);
        for (j, &v) in nn.iter().enumerate().take(hi + 1).skip(lo) {
            if j != i {
                neighbours.push(v);
            }
        }
        if neighbours.len() < 2 {
            kept.push(nn[i]);
            continue;
        }
        let med = median(&neighbours);
        if med <= 0.0 {
            kept.push(nn[i]);
            continue;
        }
        if (nn[i] - med).abs() / med <= p.ectopic_threshold {
            kept.push(nn[i]);
        }
    }
    kept
}

fn si_table() -> Table {
    let mut t = Table::new(
        "Baevsky Stress Index (SI) + components (Mo, AMo, MxDMn)",
        "physio-algo/src/stress/index.rs:131-135 — golden vector, all four components, |delta| < 1e-9",
    );

    let base = si_shipped(&SI_GOLDEN_RR);
    t.push("baseline (shipped components_raw + stress_index_raw)", si_value(base), si_gate(base));

    let twin = si_twin(&SI_GOLDEN_RR, SI_SHIPPED_PARAMS);
    t.push("twin at shipped constants (fidelity check)", si_value(twin), si_gate(twin));

    // ── NULL ──
    let m = SI_GOLDEN_RR.iter().sum::<f64>() / SI_GOLDEN_RR.len() as f64;
    let flat: Vec<f64> = vec![m; SI_GOLDEN_RR.len()];
    let o = si_shipped(&flat);
    t.push("null: every beat replaced by the series mean", si_value(o), si_gate(o));

    let o = si_shipped(&vec![800.0; SI_GOLDEN_RR.len()]);
    t.push("null: every beat a constant 800 ms", si_value(o), si_gate(o));

    // ── STRUCTURAL ──
    let mut rev = SI_GOLDEN_RR.to_vec();
    rev.reverse();
    let o = si_shipped(&rev);
    t.push("structural: series reversed", si_value(o), si_gate(o));

    let o = si_shipped(&shuffled(&SI_GOLDEN_RR, 7));
    t.push("structural: series shuffled (order destroyed, histogram kept)", si_value(o), si_gate(o));

    let o = si_shipped(&SI_GOLDEN_RR[..20]);
    t.push("structural: last 10% of beats dropped (22 -> 20)", si_value(o), si_gate(o));

    let off: Vec<f64> = SI_GOLDEN_RR.iter().map(|v| v + 10.0).collect();
    let o = si_shipped(&off);
    t.push("structural: every beat offset +10 ms", si_value(o), si_gate(o));

    let sc: Vec<f64> = SI_GOLDEN_RR.iter().map(|v| v * 1.10).collect();
    let o = si_shipped(&sc);
    t.push("structural: every beat scaled x1.10", si_value(o), si_gate(o));

    // ── PARAMETER (twin) ──
    let arms: [(&str, SiParams); 11] = [
        (
            "param: BIN_WIDTH_SEC 0.05 -> 0.055 (+10%)",
            SiParams { bin_width_sec: 0.055, ..SI_SHIPPED_PARAMS },
        ),
        (
            "param: BIN_WIDTH_SEC 0.05 -> 0.045 (-10%)",
            SiParams { bin_width_sec: 0.045, ..SI_SHIPPED_PARAMS },
        ),
        (
            "param: BIN_WIDTH_SEC 0.05 -> 0.05025 (+0.5% floor probe)",
            SiParams { bin_width_sec: 0.050_25, ..SI_SHIPPED_PARAMS },
        ),
        (
            "param: RR_MIN_MS 300 -> 330 (+10%)",
            SiParams { rr_min_ms: 330.0, ..SI_SHIPPED_PARAMS },
        ),
        (
            "param: RR_MIN_MS 300 -> 270 (-10%)",
            SiParams { rr_min_ms: 270.0, ..SI_SHIPPED_PARAMS },
        ),
        (
            "param: RR_MAX_MS 2000 -> 1800 (-10%)",
            SiParams { rr_max_ms: 1800.0, ..SI_SHIPPED_PARAMS },
        ),
        (
            "param: ECTOPIC_THRESHOLD 0.20 -> 0.22 (+10%)",
            SiParams { ectopic_threshold: 0.22, ..SI_SHIPPED_PARAMS },
        ),
        (
            "param: ECTOPIC_THRESHOLD 0.20 -> 0.18 (-10%)",
            SiParams { ectopic_threshold: 0.18, ..SI_SHIPPED_PARAMS },
        ),
        (
            "param: ECTOPIC_WINDOW_RADIUS 2 -> 3 (nearest integer step up)",
            SiParams { ectopic_window_radius: 3, ..SI_SHIPPED_PARAMS },
        ),
        (
            "param: MIN_BEATS 20 -> 22 (+10%)",
            SiParams { min_beats: 22, ..SI_SHIPPED_PARAMS },
        ),
        (
            "param: MIN_BEATS 20 -> 18 (-10%)",
            SiParams { min_beats: 18, ..SI_SHIPPED_PARAMS },
        ),
    ];
    for (name, p) in arms {
        let o = si_twin(&SI_GOLDEN_RR, p);
        t.push(name, si_value(o), si_gate(o));
    }
    t
}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// METRIC 2 — Daily autonomic stress (0-3, RHR + HRV vs a 14-day baseline)
// gate: crates/physio-algo/src/stress/daily.rs:60,61,69,78,86,94 — a property suite, no golden day
// ─────────────────────────────────────────────────────────────────────────────────────────────

const DAILY_STABLE_TARGET: f64 = 1.5; // daily.rs:69 and :94
const DAILY_STABLE_TOL: f64 = 0.1; // daily.rs:69 and :94
const DAILY_STRESSED_FLOOR: f64 = 2.0; // daily.rs:78
const DAILY_RHR_ONLY_FLOOR: f64 = 1.5; // daily.rs:86
const DAILY_BASELINE_DAYS: usize = 14; // daily.rs:8

fn day(rhr: Option<f64>, hrv: Option<f64>) -> StressDay {
    StressDay { rhr, hrv }
}

/// The five scenarios daily.rs:55-95 builds, as data.
struct DailyCases {
    stable_base: Vec<StressDay>,
    stable_today: StressDay,
    stressed_base: Vec<StressDay>,
    stressed_today: StressDay,
    rhr_only_base: Vec<StressDay>,
    rhr_only_today: StressDay,
    zero_spread_base: Vec<StressDay>,
    zero_spread_today: StressDay,
    cold_today: StressDay,
    cold_short_base: Vec<StressDay>,
}

fn daily_cases() -> DailyCases {
    DailyCases {
        stable_base: (0..20).map(|_| day(Some(55.0), Some(60.0))).collect(),
        stable_today: day(Some(55.0), Some(60.0)),
        stressed_base: (0..20)
            .map(|i| day(Some(50.0 + (i % 10) as f64), Some(55.0 + (i % 10) as f64)))
            .collect(),
        stressed_today: day(Some(65.0), Some(45.0)),
        rhr_only_base: (0..20).map(|i| day(Some(50.0 + (i % 10) as f64), None)).collect(),
        rhr_only_today: day(Some(65.0), None),
        zero_spread_base: (0..20).map(|_| day(Some(55.0), Some(60.0))).collect(),
        zero_spread_today: day(Some(65.0), Some(60.0)),
        cold_today: day(Some(55.0), Some(60.0)),
        cold_short_base: vec![day(Some(55.0), Some(60.0)); 5],
    }
}

/// Every tunable behind `daily_stress`. `rhr_sign`/`hrv_sign` are not shipped constants — they are
/// the handle the sign-flip STRUCTURAL arm needs, and are +1.0 in the shipped twin.
#[derive(Clone, Copy)]
struct DailyParams {
    baseline_days: usize,
    sd_floor: f64,
    stress_max: f64,
    rhr_sign: f64,
    hrv_sign: f64,
}

const DAILY_SHIPPED_PARAMS: DailyParams = DailyParams {
    baseline_days: DAILY_BASELINE_DAYS,
    sd_floor: SD_FLOOR,
    stress_max: STRESS_MAX,
    rhr_sign: 1.0,
    hrv_sign: 1.0,
};

/// Twin of `daily_stress` (daily.rs:22-46).
fn daily_twin(today: StressDay, baseline: &[StressDay], p: DailyParams) -> Option<f64> {
    if baseline.len() < p.baseline_days {
        return None;
    }
    let rhr_base: Vec<f64> = baseline.iter().filter_map(|d| d.rhr).collect();
    let hrv_base: Vec<f64> = baseline.iter().filter_map(|d| d.hrv).collect();
    let mean_rhr = mean_opt_twin(&rhr_base);
    let sd_rhr = pop_std_twin(&rhr_base, mean_rhr);
    let mean_hrv = mean_opt_twin(&hrv_base);
    let sd_hrv = pop_std_twin(&hrv_base, mean_hrv);

    let has_signal = (today.rhr.is_some() && mean_rhr.is_some())
        || (today.hrv.is_some() && mean_hrv.is_some());
    if !has_signal {
        return None;
    }
    let mut raw = 0.0;
    if let (Some(r), Some(m)) = (today.rhr, mean_rhr) {
        if sd_rhr > p.sd_floor {
            raw += p.rhr_sign * (r - m) / sd_rhr;
        }
    }
    if let (Some(h), Some(m)) = (today.hrv, mean_hrv) {
        if sd_hrv > p.sd_floor {
            raw += p.hrv_sign * (m - h) / sd_hrv;
        }
    }
    Some(squash_twin(raw, p.stress_max))
}

/// Score one scorer against the whole shipped property suite. Returns `(gate, stressed-day value)`.
fn daily_gate(c: &DailyCases, f: &dyn Fn(StressDay, &[StressDay]) -> Option<f64>) -> (bool, Option<f64>) {
    let stressed = f(c.stressed_today, &c.stressed_base);
    let stable = f(c.stable_today, &c.stable_base);
    let rhr_only = f(c.rhr_only_today, &c.rhr_only_base);
    let zero_spread = f(c.zero_spread_today, &c.zero_spread_base);
    let pass = f(c.cold_today, &[]).is_none() // daily.rs:60
        && f(c.cold_today, &c.cold_short_base).is_none() // daily.rs:61
        && stable.is_some_and(|s| (s - DAILY_STABLE_TARGET).abs() < DAILY_STABLE_TOL) // :69
        && stressed.is_some_and(|s| s > DAILY_STRESSED_FLOOR) // :78
        && rhr_only.is_some_and(|s| s > DAILY_RHR_ONLY_FLOOR) // :86
        && zero_spread.is_some_and(|s| (s - DAILY_STABLE_TARGET).abs() < DAILY_STABLE_TOL); // :94
    (pass, stressed)
}

fn daily_table() -> Table {
    let mut t = Table::new(
        "Daily autonomic stress (0-3, RHR + HRV vs 14-day baseline)",
        "physio-algo/src/stress/daily.rs:60,61,69,78,86,94 — property suite; value column is the \
         stressed day (:78 s > 2.0)",
    );
    let c = daily_cases();

    let (pass, v) = daily_gate(&c, &daily_stress);
    t.push("baseline (shipped daily_stress)", v, pass);

    let (pass, v) = daily_gate(&c, &|d: StressDay, b: &[StressDay]| daily_twin(d, b, DAILY_SHIPPED_PARAMS));
    t.push("twin at shipped constants (fidelity check)", v, pass);

    // ── NULL ──
    // A scorer that never reads today: today is replaced by the baseline's own mean day.
    let (pass, v) = daily_gate(&c, &|_today: StressDay, b: &[StressDay]| {
        let r = mean_opt_twin(&b.iter().filter_map(|d| d.rhr).collect::<Vec<f64>>());
        let h = mean_opt_twin(&b.iter().filter_map(|d| d.hrv).collect::<Vec<f64>>());
        daily_twin(day(r, h), b, DAILY_SHIPPED_PARAMS)
    });
    t.push("null: today replaced by the baseline mean day (today never read)", v, pass);

    // A scorer that returns the neutral midpoint whatever it is handed.
    let (pass, v) = daily_gate(&c, &|_t: StressDay, b: &[StressDay]| {
        if b.len() < DAILY_BASELINE_DAYS { None } else { Some(squash(0.0)) }
    });
    t.push("null: constant neutral 1.5 for every day", v, pass);

    // ── STRUCTURAL ──
    let (pass, v) = daily_gate(&c, &|d: StressDay, b: &[StressDay]| {
        daily_twin(d, b, DailyParams { rhr_sign: -1.0, hrv_sign: -1.0, ..DAILY_SHIPPED_PARAMS })
    });
    t.push("structural: both z-terms sign-flipped (calm reads as stressed)", v, pass);

    let (pass, v) = daily_gate(&c, &|d: StressDay, b: &[StressDay]| {
        daily_twin(d, b, DailyParams { hrv_sign: -1.0, ..DAILY_SHIPPED_PARAMS })
    });
    t.push("structural: HRV term sign-flipped only (high HRV reads as stress)", v, pass);

    let (pass, v) = daily_gate(&c, &|d: StressDay, b: &[StressDay]| daily_twin(d, &shuffled(b, 13), DAILY_SHIPPED_PARAMS));
    t.push("structural: baseline days shuffled", v, pass);

    let (pass, v) = daily_gate(&c, &|d: StressDay, b: &[StressDay]| {
        let keep = b.len() - b.len() / 10;
        daily_twin(d, &b[..keep], DAILY_SHIPPED_PARAMS)
    });
    t.push("structural: last 10% of baseline days dropped", v, pass);

    let (pass, v) = daily_gate(&c, &|d: StressDay, b: &[StressDay]| {
        daily_twin(day(d.hrv, d.rhr), b, DAILY_SHIPPED_PARAMS)
    });
    t.push("structural: today's RHR and HRV channels swapped", v, pass);

    // ── PARAMETER (twin) ──
    let arms: [(&str, DailyParams); 10] = [
        (
            "param: BASELINE_DAYS 14 -> 15 (+10%)",
            DailyParams { baseline_days: 15, ..DAILY_SHIPPED_PARAMS },
        ),
        (
            "param: BASELINE_DAYS 14 -> 13 (-10%)",
            DailyParams { baseline_days: 13, ..DAILY_SHIPPED_PARAMS },
        ),
        (
            "param: SD_FLOOR 1e-4 -> 1.1e-4 (+10%)",
            DailyParams { sd_floor: 0.000_11, ..DAILY_SHIPPED_PARAMS },
        ),
        (
            "param: SD_FLOOR 1e-4 -> 9e-5 (-10%)",
            DailyParams { sd_floor: 0.000_09, ..DAILY_SHIPPED_PARAMS },
        ),
        (
            "param: SD_FLOOR 1e-4 -> 3.0 (extreme: above the data's own SD)",
            DailyParams { sd_floor: 3.0, ..DAILY_SHIPPED_PARAMS },
        ),
        (
            "param: STRESS_MAX 3.0 -> 3.3 (+10%)",
            DailyParams { stress_max: 3.3, ..DAILY_SHIPPED_PARAMS },
        ),
        (
            "param: STRESS_MAX 3.0 -> 2.7 (-10%)",
            DailyParams { stress_max: 2.7, ..DAILY_SHIPPED_PARAMS },
        ),
        (
            "param: STRESS_MAX 3.0 -> 3.21 (+7%)",
            DailyParams { stress_max: 3.21, ..DAILY_SHIPPED_PARAMS },
        ),
        (
            "param: STRESS_MAX 3.0 -> 3.18 (+6%)",
            DailyParams { stress_max: 3.18, ..DAILY_SHIPPED_PARAMS },
        ),
        (
            "param: STRESS_MAX 3.0 -> 3.015 (+0.5% floor probe)",
            DailyParams { stress_max: 3.015, ..DAILY_SHIPPED_PARAMS },
        ),
    ];
    for (name, p) in arms {
        let (pass, v) = daily_gate(&c, &|d: StressDay, b: &[StressDay]| daily_twin(d, b, p));
        t.push(name, v, pass);
    }

    // BASELINE_DAYS is a length gate, invisible unless the baseline sits on it. This arm trims the
    // scenarios to exactly 14 days so the constant becomes load-bearing — clearly a changed input,
    // not a pure parameter move.
    let trimmed = DailyCases {
        stable_base: c.stable_base[..14].to_vec(),
        stressed_base: c.stressed_base[..14].to_vec(),
        rhr_only_base: c.rhr_only_base[..14].to_vec(),
        zero_spread_base: c.zero_spread_base[..14].to_vec(),
        ..daily_cases()
    };
    let (pass, v) = daily_gate(&trimmed, &|d: StressDay, b: &[StressDay]| {
        daily_twin(d, b, DailyParams { baseline_days: 15, ..DAILY_SHIPPED_PARAMS })
    });
    t.push(
        "param+input: BASELINE_DAYS 14 -> 15 with baselines trimmed to exactly 14 days",
        v,
        pass,
    );
    t
}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// METRIC 3 — Windowed stress (daytime + sleep), per-bucket score, bands, sustained run
// gate: crates/physio-algo/src/stress/window.rs:339-380 — bit-exact golden day
// ─────────────────────────────────────────────────────────────────────────────────────────────

const DAY_START_MS: i64 = 1_753_920_000_000; // window.rs:239
const HOUR_MS: i64 = 3_600_000; // window.rs:240

/// The banked real worn day from window.rs:244-269.
const REAL_DAY: [(i32, f64, f64); 24] = [
    (0, 69.660_555_555_555_56, 31.853_136_716_696_262),
    (1, 74.612_222_222_222_23, 28.580_312_554_873_604),
    (2, 72.715, 32.341_108_532_331_19),
    (3, 77.725, 23.600_810_436_828_418),
    (4, 71.618_055_555_555_56, 31.828_555_205_346_81),
    (5, 74.883_333_333_333_34, 31.901_111_810_600_415),
    (6, 76.936_388_888_888_89, 33.418_354_487_431_16),
    (7, 79.580_555_555_555_55, 32.929_887_835_467_57),
    (8, 81.86, 39.194_740_739_980_375),
    (9, 88.1025, 33.650_741_652_942_86),
    (10, 80.358_639_642_734_46, 46.149_393_178_731_735),
    (11, 81.621_067_031_463_75, 47.297_223_049_482_07),
    (12, 82.695_118_469_882_96, 49.173_545_252_936_73),
    (13, 97.033_333_333_333_33, 38.656_262_968_200_465),
    (14, 90.943_055_555_555_56, 43.176_522_393_501_24),
    (15, 86.318_333_333_333_33, 48.907_914_455_490_15),
    (16, 82.340_555_555_555_55, 64.585_556_992_743_39),
    (17, 87.784_722_222_222_23, 47.961_006_823_870_79),
    (18, 87.666_666_666_666_67, 38.866_656_271_114_57),
    (19, 85.719_166_666_666_67, 31.259_841_796_659_202),
    (20, 84.636_010_002_778_55, 31.958_253_315_144_663),
    (21, 88.182_272_853_570_44, 37.038_316_554_669_21),
    (22, 94.056_666_666_666_67, 39.540_994_308_513_05),
    (23, 85.924_166_666_666_66, 48.716_288_452_255_68),
];

/// The 16 scored hours window.rs:328-345 pins bit-for-bit.
const WIN_EXPECTED: [(i32, f64); 16] = [
    (6, 1.948_703_870_556_964_8),
    (7, 2.319_564_923_423_633_7),
    (8, 2.177_791_250_731_700_3),
    (9, 2.846_928_133_190_912_6),
    (10, 1.388_891_794_063_300_5),
    (11, 1.486_394_005_767_722_5),
    (12, 1.491_039_930_317_088_7),
    (13, 2.955_978_700_967_047),
    (14, 2.752_477_971_332_585_3),
    (15, 2.054_423_468_681_809_7),
    (16, 0.397_720_072_769_574_05),
    (17, 2.301_539_910_923_305),
    (18, 2.707_531_044_539_024_4),
    (19, 2.811_642_863_863_905),
    (20, 2.749_485_844_900_177),
    (21, 2.781_870_795_005_547),
];

const WIN_GATE_MEAN: f64 = 2.198_249_036_314_643_4; // window.rs:350
const WIN_GATE_PEAK_HOUR: i32 = 13; // window.rs:351
const WIN_GATE_SUSTAINED_RUN: usize = 5; // window.rs:354
const WIN_GATE_SUPPRESSED_ASLEEP: usize = 8; // window.rs:355-356
const WIN_GATE_BANDS: (i64, i64, i64) = (60, 240, 660); // window.rs:363

/// Everything the shipped golden-day gate pins.
#[derive(Clone, Debug, PartialEq)]
struct WinOut {
    stresses: Vec<(i32, f64)>,
    mean: Option<f64>,
    peak_hour: Option<i32>,
    sustained_high: bool,
    sustained_run: usize,
    suppressed_total: usize,
    suppressed_asleep: usize,
    bands: (i64, i64, i64),
}

fn win_gate(o: &WinOut) -> bool {
    o.stresses.len() == WIN_EXPECTED.len()
        && o.stresses
            .iter()
            .zip(WIN_EXPECTED.iter())
            .all(|((h, s), (eh, es))| h == eh && s.to_bits() == es.to_bits())
        && o.mean.map(|m| m.to_bits()) == Some(WIN_GATE_MEAN.to_bits())
        && o.peak_hour == Some(WIN_GATE_PEAK_HOUR)
        && o.sustained_high
        && o.sustained_run == WIN_GATE_SUSTAINED_RUN
        && o.suppressed_total == WIN_GATE_SUPPRESSED_ASLEEP
        && o.suppressed_asleep == WIN_GATE_SUPPRESSED_ASLEEP
        && o.bands == WIN_GATE_BANDS
}

fn win_value(o: &WinOut) -> Option<f64> {
    o.mean
}

/// The banked motion channel from window.rs — mean dynamic accel (g) per hourly bucket. Every arm
/// carries it, so `activity_gate_g` is a knob these tables can actually move; with `motion_g: None`
/// the gate was never evaluated and both its arms reported a delta of exactly zero.
const WIN_REAL_DAY_MOTION_G: [f64; 24] = [
    0.021_853, 0.006_737, 0.010_248, 0.009_181, 0.008_856, 0.008_557,
    0.015_612, 0.011_207, 0.016_108, 0.015_935, 0.093_877, 0.068_195,
    0.111_046, 0.085_727, 0.142_547, 0.039_923, 0.129_114, 0.119_880,
    0.110_810, 0.097_985, 0.107_272, 0.118_880, 0.086_236, 0.079_337,
];

fn win_point(hour: i32, mean_hr: Option<f64>, rmssd: Option<f64>) -> HourPoint {
    HourPoint {
        start_ms: DAY_START_MS + hour as i64 * HOUR_MS,
        hour,
        mean_hr,
        rmssd,
        motion_g: WIN_REAL_DAY_MOTION_G.get(hour as usize).copied(),
    }
}

fn real_day() -> Vec<HourPoint> {
    REAL_DAY.iter().map(|&(h, hr, rmssd)| win_point(h, Some(hr), Some(rmssd))).collect()
}

fn win_span(from_hour: i64, to_hour: i64) -> SpanMs {
    SpanMs {
        start_ms: DAY_START_MS + from_hour * HOUR_MS,
        end_ms: DAY_START_MS + to_hour * HOUR_MS,
    }
}

/// The retired clock window `WAKING_HOURS = (6, 22)` as spans, matching
/// `window.rs waking_window_as_spans`. NOT the day's sleep — the fixture carries no sleep record —
/// so no arm below may be read as evidence about span selection. See the note under the table.
fn waking_window_as_spans() -> Vec<SpanMs> {
    vec![win_span(0, 6), win_span(22, 24)]
}

fn win_from_shipped(w: &StressWindows) -> WinOut {
    WinOut {
        stresses: w.buckets.iter().map(|b| (b.hour, b.stress)).collect(),
        mean: w.mean,
        peak_hour: w.peak_hour,
        sustained_high: w.sustained_high,
        sustained_run: w.sustained_run,
        suppressed_total: w.suppressed.len(),
        suppressed_asleep: w.suppressed_count(physio_algo::stress::Suppression::Asleep),
        bands: (w.low_minutes, w.medium_minutes, w.high_minutes),
    }
}

/// Every tunable `windowed_stress` reads. `bucket_seconds`, `sustained_buckets` and
/// `activity_gate_g` are REAL knobs on `StressWindowCfg`; the rest are private consts.
#[derive(Clone, Copy)]
struct WinParams {
    calm_quartile_min_count: usize,
    hr_calm_quantile: f64,
    rmssd_calm_quantile: f64,
    bucket_seconds: i64,
    sustained_buckets: usize,
    activity_gate_g: Option<f64>,
    sd_floor: f64,
    stress_max: f64,
    medium_band_floor: f64,
    high_band_floor: f64,
}

const WIN_SHIPPED_PARAMS: WinParams = WinParams {
    calm_quartile_min_count: 4, // window.rs:10
    hr_calm_quantile: 0.25,     // window.rs:355 (calm_reference, calm_is_low)
    rmssd_calm_quantile: 0.75,  // window.rs:355 (calm_reference, !calm_is_low)
    bucket_seconds: 3_600,      // HOUR_SECONDS, window.rs:15
    sustained_buckets: 3,       // SUSTAINED_BUCKETS, window.rs:16
    activity_gate_g: Some(0.15), // ACTIVITY_GATE_G, window.rs:19
    sd_floor: SD_FLOOR,
    stress_max: STRESS_MAX,
    medium_band_floor: MEDIUM_BAND_FLOOR,
    high_band_floor: HIGH_BAND_FLOOR,
};

/// Twin of `calm_reference` (window.rs:349-356) with the quantile opened as a parameter.
fn calm_ref_twin(xs: &[f64], quantile: f64, min_count: usize) -> Option<f64> {
    if xs.is_empty() {
        return None;
    }
    if xs.len() < min_count {
        return mean_opt_twin(xs);
    }
    let mut s = xs.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Some(percentile(&s, quantile))
}

/// Twin of `windowed_stress` (window.rs:157-217) plus `suppression_of` (window.rs:143-153).
fn win_twin(points: &[HourPoint], spans: &[SpanMs], p: WinParams) -> WinOut {
    let mut selected: Vec<&HourPoint> = Vec::with_capacity(points.len());
    let mut suppressed_total = 0usize;
    let mut suppressed_asleep = 0usize;
    for h in points {
        let active = p.activity_gate_g.is_some_and(|g| h.motion_g.is_some_and(|m| m >= g));
        let bucket_end = h.start_ms + p.bucket_seconds.max(0) * 1_000;
        let asleep = !active
            && spans.iter().any(|s| h.start_ms < s.end_ms && s.start_ms < bucket_end);
        if active {
            suppressed_total += 1;
        } else if asleep {
            suppressed_total += 1;
            suppressed_asleep += 1;
        } else {
            selected.push(h);
        }
    }
    let empty = WinOut {
        stresses: Vec::new(),
        mean: None,
        peak_hour: None,
        sustained_high: false,
        sustained_run: 0,
        suppressed_total,
        suppressed_asleep,
        bands: (0, 0, 0),
    };
    if selected.is_empty() {
        return empty.clone();
    }
    let hr_vals: Vec<f64> = selected.iter().filter_map(|h| h.mean_hr).collect();
    let rmssd_vals: Vec<f64> = selected.iter().filter_map(|h| h.rmssd).collect();
    let calm_hr = calm_ref_twin(&hr_vals, p.hr_calm_quantile, p.calm_quartile_min_count);
    let calm_rmssd = calm_ref_twin(&rmssd_vals, p.rmssd_calm_quantile, p.calm_quartile_min_count);
    let sd_hr = pop_std_twin(&hr_vals, mean_opt_twin(&hr_vals));
    let sd_rmssd = pop_std_twin(&rmssd_vals, mean_opt_twin(&rmssd_vals));

    let mut scored: Vec<(i32, f64)> = Vec::new();
    for h in &selected {
        let Some(mean_hr) = h.mean_hr else { continue };
        let mut raw = 0.0;
        if let Some(ref_hr) = calm_hr {
            if sd_hr > p.sd_floor {
                raw += (mean_hr - ref_hr) / sd_hr;
            }
        }
        if let (Some(r), Some(ref_r)) = (h.rmssd, calm_rmssd) {
            if sd_rmssd > p.sd_floor {
                raw += (ref_r - r) / sd_rmssd;
            }
        }
        scored.push((h.hour, squash_twin(raw, p.stress_max)));
    }
    if scored.is_empty() {
        return empty;
    }
    let mut run = 0usize;
    for (_, s) in scored.iter().rev() {
        if *s >= p.high_band_floor {
            run += 1
        } else {
            break;
        }
    }
    let mean = mean_opt_twin(&scored.iter().map(|(_, s)| *s).collect::<Vec<f64>>());
    // `max_by` keeps the LAST of equal maxima, so `>=` is the faithful fold.
    let mut peak_hour = scored[0].0;
    let mut peak_val = scored[0].1;
    for &(h, s) in scored.iter().skip(1) {
        if s >= peak_val {
            peak_val = s;
            peak_hour = h;
        }
    }
    let per_bucket = p.bucket_seconds / 60;
    let count = |f: &dyn Fn(f64) -> bool| {
        scored.iter().filter(|(_, s)| f(*s)).count() as i64 * per_bucket
    };
    let bands = (
        count(&|s| s < p.medium_band_floor),
        count(&|s| s >= p.medium_band_floor && s < p.high_band_floor),
        count(&|s| s >= p.high_band_floor),
    );
    WinOut {
        stresses: scored,
        mean,
        peak_hour: Some(peak_hour),
        sustained_high: run >= p.sustained_buckets,
        sustained_run: run,
        suppressed_total,
        suppressed_asleep,
        bands,
    }
}

/// Rebuild the fixture day from perturbed (hr, rmssd) pairs, keeping every bucket's clock position.
fn day_from_channels(pairs: &[(f64, f64)]) -> Vec<HourPoint> {
    pairs
        .iter()
        .enumerate()
        .map(|(i, &(hr, rmssd))| win_point(i as i32, Some(hr), Some(rmssd)))
        .collect()
}

fn channels() -> Vec<(f64, f64)> {
    REAL_DAY.iter().map(|&(_, hr, rmssd)| (hr, rmssd)).collect()
}

fn win_table() -> Table {
    let mut t = Table::new(
        "Windowed stress (daytime + sleep): per-bucket score, bands, sustained run",
        "physio-algo/src/stress/window.rs:339-380 — bit-exact golden day (16 hour bits, mean bits, \
         peak hour, sustained_run 5, suppressed 8, bands 60/240/660)",
    );
    // Every arm below holds the spans FIXED, so this table measures the formula's sensitivity, not
    // the window selection's. `sleep_spans` is a real input and no arm perturbs it; moving the night
    // off the clock is caught (mean 2.1982 -> 2.2337, run 5 -> 0) but that arm does not exist yet.
    let spans = waking_window_as_spans();

    let base = win_from_shipped(&daytime_stress(&real_day(), &spans));
    t.push("baseline (shipped daytime_stress)", win_value(&base), win_gate(&base));

    let twin = win_twin(&real_day(), &spans, WIN_SHIPPED_PARAMS);
    t.push("twin at shipped constants (fidelity check)", win_value(&twin), win_gate(&twin));

    // ── NULL ──
    let ch = channels();
    let hr_mean = ch.iter().map(|c| c.0).sum::<f64>() / ch.len() as f64;
    let rm_mean = ch.iter().map(|c| c.1).sum::<f64>() / ch.len() as f64;
    let flat: Vec<(f64, f64)> = ch.iter().map(|_| (hr_mean, rm_mean)).collect();
    let o = win_from_shipped(&daytime_stress(&day_from_channels(&flat), &spans));
    t.push("null: every bucket replaced by the day mean (no spread left)", win_value(&o), win_gate(&o));

    // A scorer that ignores its inputs and reports the neutral midpoint everywhere.
    let neutral = WinOut {
        stresses: WIN_EXPECTED.iter().map(|&(h, _)| (h, squash(0.0))).collect(),
        mean: Some(squash(0.0)),
        peak_hour: Some(WIN_EXPECTED[WIN_EXPECTED.len() - 1].0),
        sustained_high: false,
        sustained_run: 0,
        suppressed_total: WIN_GATE_SUPPRESSED_ASLEEP,
        suppressed_asleep: WIN_GATE_SUPPRESSED_ASLEEP,
        bands: (0, 960, 0),
    };
    t.push("null: constant neutral 1.5 in every bucket", win_value(&neutral), win_gate(&neutral));

    // ── STRUCTURAL ──
    let mut rolled = ch.clone();
    rolled.rotate_right(1);
    let o = win_from_shipped(&daytime_stress(&day_from_channels(&rolled), &spans));
    t.push("structural: channels shifted +1 bucket (60 min)", win_value(&o), win_gate(&o));

    let mut rev = ch.clone();
    rev.reverse();
    let o = win_from_shipped(&daytime_stress(&day_from_channels(&rev), &spans));
    t.push("structural: day reversed", win_value(&o), win_gate(&o));

    let o = win_from_shipped(&daytime_stress(&day_from_channels(&shuffled(&ch, 21)), &spans));
    t.push("structural: buckets shuffled (same values, wrong hours)", win_value(&o), win_gate(&o));

    let swapped: Vec<(f64, f64)> = ch.iter().map(|&(hr, rm)| (rm, hr)).collect();
    let o = win_from_shipped(&daytime_stress(&day_from_channels(&swapped), &spans));
    t.push("structural: HR and RMSSD channels swapped", win_value(&o), win_gate(&o));

    let dropped: Vec<HourPoint> = real_day().into_iter().take(22).collect();
    let o = win_from_shipped(&daytime_stress(&dropped, &spans));
    t.push("structural: last 10% of buckets dropped (24 -> 22)", win_value(&o), win_gate(&o));

    let offset: Vec<(f64, f64)> = ch.iter().map(|&(hr, rm)| (hr + 10.0, rm)).collect();
    let o = win_from_shipped(&daytime_stress(&day_from_channels(&offset), &spans));
    t.push("structural: HR offset +10 bpm on every bucket", win_value(&o), win_gate(&o));

    // Score is `(hr - calm_hr)/sd_hr`, so any affine `hr -> c*hr + a` (c > 0) cancels and moves the
    // true value by exactly zero. x1.10 leaves ULP residue the bit-exact gate reads as a catch; the
    // x2.0 arm below is exact in binary and shows what this mutation really does.
    let scaled: Vec<(f64, f64)> = ch.iter().map(|&(hr, rm)| (hr * 1.10, rm)).collect();
    let o = win_from_shipped(&daytime_stress(&day_from_channels(&scaled), &spans));
    t.push("structural: HR scaled x1.10 on every bucket", win_value(&o), win_gate(&o));

    let doubled: Vec<(f64, f64)> = ch.iter().map(|&(hr, rm)| (hr * 2.0, rm)).collect();
    let o = win_from_shipped(&daytime_stress(&day_from_channels(&doubled), &spans));
    t.push("structural: HR scaled x2.00 on every bucket (exact in binary)", win_value(&o), win_gate(&o));

    // ── PARAMETER: the three knobs that are REAL, driven through the shipped function ──
    let real_knobs: [(&str, StressWindowCfg); 6] = [
        (
            "param(real cfg): bucket_seconds 3600 -> 3960 (+10%)",
            StressWindowCfg { bucket_seconds: 3_960, sleep_spans: &spans, activity_gate_g: Some(0.15), sustained_buckets: 3 },
        ),
        (
            "param(real cfg): bucket_seconds 3600 -> 3240 (-10%)",
            StressWindowCfg { bucket_seconds: 3_240, sleep_spans: &spans, activity_gate_g: Some(0.15), sustained_buckets: 3 },
        ),
        (
            "param(real cfg): activity_gate_g 0.15 -> 0.165 (+10%)",
            StressWindowCfg { bucket_seconds: 3_600, sleep_spans: &spans, activity_gate_g: Some(0.165), sustained_buckets: 3 },
        ),
        (
            "param(real cfg): activity_gate_g 0.15 -> 0.135 (-10%)",
            StressWindowCfg { bucket_seconds: 3_600, sleep_spans: &spans, activity_gate_g: Some(0.135), sustained_buckets: 3 },
        ),
        (
            "param(real cfg): sustained_buckets 3 -> 4 (integer step up)",
            StressWindowCfg { bucket_seconds: 3_600, sleep_spans: &spans, activity_gate_g: Some(0.15), sustained_buckets: 4 },
        ),
        (
            "param(real cfg): sustained_buckets 3 -> 2 (integer step down)",
            StressWindowCfg { bucket_seconds: 3_600, sleep_spans: &spans, activity_gate_g: Some(0.15), sustained_buckets: 2 },
        ),
    ];
    for (name, cfg) in real_knobs {
        let o = win_from_shipped(&windowed_stress(&real_day(), cfg));
        t.push(name, win_value(&o), win_gate(&o));
    }

    // ── PARAMETER: the private consts, via the twin ──
    let arms: [(&str, WinParams); 11] = [
        (
            "param: HR calm quantile 0.25 -> 0.275 (+10%)",
            WinParams { hr_calm_quantile: 0.275, ..WIN_SHIPPED_PARAMS },
        ),
        (
            "param: HR calm quantile 0.25 -> 0.225 (-10%)",
            WinParams { hr_calm_quantile: 0.225, ..WIN_SHIPPED_PARAMS },
        ),
        (
            "param: HR calm quantile 0.25 -> 0.25125 (+0.5% floor probe)",
            WinParams { hr_calm_quantile: 0.251_25, ..WIN_SHIPPED_PARAMS },
        ),
        (
            "param: RMSSD calm quantile 0.75 -> 0.825 (+10%)",
            WinParams { rmssd_calm_quantile: 0.825, ..WIN_SHIPPED_PARAMS },
        ),
        (
            "param: RMSSD calm quantile 0.75 -> 0.675 (-10%)",
            WinParams { rmssd_calm_quantile: 0.675, ..WIN_SHIPPED_PARAMS },
        ),
        (
            "param: CALM_QUARTILE_MIN_COUNT 4 -> 5 (integer step up)",
            WinParams { calm_quartile_min_count: 5, ..WIN_SHIPPED_PARAMS },
        ),
        (
            "param: CALM_QUARTILE_MIN_COUNT 4 -> 3 (integer step down)",
            WinParams { calm_quartile_min_count: 3, ..WIN_SHIPPED_PARAMS },
        ),
        (
            "param: SD_FLOOR 1e-4 -> 1.1e-4 (+10%)",
            WinParams { sd_floor: 0.000_11, ..WIN_SHIPPED_PARAMS },
        ),
        (
            "param: STRESS_MAX 3.0 -> 3.015 (+0.5% floor probe)",
            WinParams { stress_max: 3.015, ..WIN_SHIPPED_PARAMS },
        ),
        (
            "param: MEDIUM_BAND_FLOOR 1.0 -> 1.1 (+10%)",
            WinParams { medium_band_floor: 1.1, ..WIN_SHIPPED_PARAMS },
        ),
        (
            "param: HIGH_BAND_FLOOR 2.0 -> 2.2 (+10%)",
            WinParams { high_band_floor: 2.2, ..WIN_SHIPPED_PARAMS },
        ),
    ];
    for (name, p) in arms {
        let o = win_twin(&real_day(), &spans, p);
        t.push(name, win_value(&o), win_gate(&o));
    }
    t
}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// METRIC 4 — Stress band + squash (the shared 0-3 scale)
// gate: crates/physio-algo/src/stress/mod.rs:73-74 and :79-84
// ─────────────────────────────────────────────────────────────────────────────────────────────

const SQUASH_GATE_AT_ZERO: f64 = 1.5; // mod.rs:73
const SQUASH_GATE_TOL: f64 = 1e-12; // mod.rs:73

/// Band codes, so the twin does not need to construct the shipped enum.
const LOW: u8 = 0;
const MEDIUM: u8 = 1;
const HIGH: u8 = 2;

/// The six band claims at mod.rs:79-84, as (score, expected band).
const BAND_GATE: [(f64, u8); 6] = [
    (0.0, LOW),        // mod.rs:79
    (0.999, LOW),      // mod.rs:80
    (1.0, MEDIUM),     // mod.rs:81
    (1.999, MEDIUM),   // mod.rs:82
    (2.0, HIGH),       // mod.rs:83
    (STRESS_MAX, HIGH), // mod.rs:84
];

fn scale_gate(sq: &dyn Fn(f64) -> f64, bd: &dyn Fn(f64) -> u8) -> bool {
    (sq(0.0) - SQUASH_GATE_AT_ZERO).abs() < SQUASH_GATE_TOL // mod.rs:73
        && sq(-40.0) >= 0.0 // mod.rs:74
        && sq(40.0) <= STRESS_MAX // mod.rs:74
        && BAND_GATE.iter().all(|&(s, want)| bd(s) == want) // mod.rs:79-84
}

fn band_twin(score: f64, medium_floor: f64, high_floor: f64) -> u8 {
    if score < medium_floor {
        LOW
    } else if score < high_floor {
        MEDIUM
    } else {
        HIGH
    }
}

fn band_shipped(score: f64) -> u8 {
    match band_of(score) {
        StressBand::Low => LOW,
        StressBand::Medium => MEDIUM,
        StressBand::High => HIGH,
    }
}

fn scale_table() -> Table {
    let mut t = Table::new(
        "Stress band + squash (shared 0-3 scale)",
        "physio-algo/src/stress/mod.rs:73-74 (squash(0) = 1.5 +- 1e-12) and :79-84 (six band \
         boundaries); value column is squash(0.0)",
    );

    let pass = scale_gate(&squash, &band_shipped);
    t.push("baseline (shipped squash + band_of)", Some(squash(0.0)), pass);

    let sq_t = |m: f64| move |x: f64| squash_twin(x, m);
    let bd_t = |med: f64, hi: f64| move |x: f64| band_twin(x, med, hi);

    let sq = sq_t(STRESS_MAX);
    let bd = bd_t(MEDIUM_BAND_FLOOR, HIGH_BAND_FLOOR);
    t.push("twin at shipped constants (fidelity check)", Some(sq(0.0)), scale_gate(&sq, &bd));

    // ── NULL ──
    let zero = |_x: f64| 0.0f64;
    t.push("null: squash returns 0.0 for every input", Some(zero(0.0)), scale_gate(&zero, &bd));
    let all_low = |_x: f64| LOW;
    t.push("null: band_of returns Low for every score", Some(sq(0.0)), scale_gate(&sq, &all_low));

    // ── STRUCTURAL ──
    let flipped = |x: f64| squash_twin(-x, STRESS_MAX);
    t.push(
        "structural: squash sign-flipped (stress reads as calm)",
        Some(flipped(0.0)),
        scale_gate(&flipped, &bd),
    );
    let reversed = |x: f64| match band_twin(x, MEDIUM_BAND_FLOOR, HIGH_BAND_FLOOR) {
        LOW => HIGH,
        HIGH => LOW,
        other => other,
    };
    t.push(
        "structural: bands reversed (Low <-> High)",
        Some(sq(0.0)),
        scale_gate(&sq, &reversed),
    );
    let inclusive = |x: f64| {
        if x <= MEDIUM_BAND_FLOOR {
            LOW
        } else if x <= HIGH_BAND_FLOOR {
            MEDIUM
        } else {
            HIGH
        }
    };
    t.push(
        "structural: band comparisons '<' -> '<=' (boundary off-by-one)",
        Some(sq(0.0)),
        scale_gate(&sq, &inclusive),
    );

    // ── PARAMETER ──
    for (name, m) in [
        ("param: STRESS_MAX 3.0 -> 3.3 (+10%)", 3.3f64),
        ("param: STRESS_MAX 3.0 -> 2.7 (-10%)", 2.7),
        ("param: STRESS_MAX 3.0 -> 3.015 (+0.5% floor probe)", 3.015),
    ] {
        let f = sq_t(m);
        t.push(name, Some(f(0.0)), scale_gate(&f, &bd));
    }
    for (name, med, hi) in [
        ("param: MEDIUM_BAND_FLOOR 1.0 -> 1.1 (+10%)", 1.1f64, HIGH_BAND_FLOOR),
        ("param: MEDIUM_BAND_FLOOR 1.0 -> 0.9 (-10%)", 0.9, HIGH_BAND_FLOOR),
        ("param: MEDIUM_BAND_FLOOR 1.0 -> 1.005 (+0.5% floor probe)", 1.005, HIGH_BAND_FLOOR),
        ("param: HIGH_BAND_FLOOR 2.0 -> 2.2 (+10%)", MEDIUM_BAND_FLOOR, 2.2),
        ("param: HIGH_BAND_FLOOR 2.0 -> 1.8 (-10%)", MEDIUM_BAND_FLOOR, 1.8),
        ("param: HIGH_BAND_FLOOR 2.0 -> 2.01 (+0.5% floor probe)", MEDIUM_BAND_FLOOR, 2.01),
    ] {
        let f = bd_t(med, hi);
        t.push(name, Some(sq(0.0)), scale_gate(&sq, &f));
    }
    t
}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// METRIC 5 — Live stress-onset detector (HRV dip, exercise-gated)
// gate: crates/physio-algo/src/stress_onset.rs — the decision table: every row's reason AND
// should_nudge, plus the firing row's fast RMSSD and folded baseline to the bit
// ─────────────────────────────────────────────────────────────────────────────────────────────

/// The shipped cohort this control replicates: every `#[test]` in `stress_onset.rs`, counted from
/// the source so a test added or removed there fails loudly here instead of silently shrinking what
/// the control measures.
const ONSET_SHIPPED_TESTS: usize = 11;

fn onset_shipped_test_count() -> usize {
    include_str!("../src/stress_onset.rs").matches("#[test]").count()
}

/// 80 beats alternating `800` and `800 + spread`, so the fast window's RMSSD is exactly `spread`.
fn onset_buffer(spread: u16) -> Vec<u16> {
    (0..80).map(|i| if i % 2 == 0 { 800u16 } else { 800 + spread }).collect()
}

const ONSET_NOW: i64 = 100_000;
/// 03:46 local at tz 0 — the minute the quiet-hours rows straddle.
const ONSET_NOW_LOCAL_MIN: i32 = 226;
const ONSET_SEED_BASELINE: f64 = 40.0;
/// 23 ms is under the crossing point the shipped constants set, 24 ms is over it.
const ONSET_DIP_SPREAD: u16 = 23;
const ONSET_CALM_SPREAD: u16 = 24;
/// The row the value column reports: the plain dip that fires.
const ONSET_FIRING_CASE: usize = 5;
const ONSET_GATE_FAST: f64 = 23.0;
const ONSET_GATE_BASELINE: f64 = 39.660_000_000_000_004;

/// One row of the shipped decision table: the inputs, and the decision `evaluate` must return.
struct OnsetCase {
    rr: Vec<u16>,
    hr: Option<f64>,
    motion: Option<f64>,
    session_active: bool,
    state: OnsetState,
    enabled: bool,
    auto_nudge: bool,
    quiet: Option<(i32, i32)>,
    tz_offset_sec: i64,
    reason: OnsetReason,
    nudge: bool,
}

fn onset_case(reason: OnsetReason, nudge: bool) -> OnsetCase {
    OnsetCase {
        rr: onset_buffer(ONSET_DIP_SPREAD),
        hr: Some(70.0),
        motion: None,
        session_active: false,
        state: OnsetState {
            baseline_rmssd: ONSET_SEED_BASELINE,
            was_below: false,
            last_fire_at: 0,
        },
        enabled: true,
        auto_nudge: true,
        quiet: None,
        tz_offset_sec: 0,
        reason,
        nudge,
    }
}

/// The twenty rows of the shipped decision table, in source order.
fn onset_cases() -> Vec<OnsetCase> {
    use OnsetReason::*;
    let seeded = onset_case(Onset, true).state;
    vec![
        OnsetCase { enabled: false, ..onset_case(Disabled, false) },
        OnsetCase { auto_nudge: false, ..onset_case(Disabled, false) },
        OnsetCase { rr: vec![800u16; 19], ..onset_case(InsufficientData, false) },
        OnsetCase { state: OnsetState::default(), ..onset_case(NoDip, false) },
        OnsetCase { rr: onset_buffer(ONSET_CALM_SPREAD), ..onset_case(NoDip, false) },
        onset_case(Onset, true),
        OnsetCase { state: OnsetState { was_below: true, ..seeded }, ..onset_case(NotAnEdge, false) },
        OnsetCase { hr: None, ..onset_case(ExerciseGated, false) },
        OnsetCase { hr: Some(100.1), ..onset_case(ExerciseGated, false) },
        OnsetCase { hr: Some(54.9), ..onset_case(ExerciseGated, false) },
        OnsetCase { motion: Some(0.15), ..onset_case(ExerciseGated, false) },
        OnsetCase { motion: Some(0.15 - 1e-4), ..onset_case(Onset, true) },
        OnsetCase { session_active: true, ..onset_case(Suppressed, false) },
        OnsetCase {
            state: OnsetState { last_fire_at: ONSET_NOW - 899, ..seeded },
            ..onset_case(Suppressed, false)
        },
        OnsetCase {
            state: OnsetState { last_fire_at: ONSET_NOW - 900, ..seeded },
            ..onset_case(Onset, true)
        },
        OnsetCase {
            quiet: Some((ONSET_NOW_LOCAL_MIN, ONSET_NOW_LOCAL_MIN + 1)),
            ..onset_case(Suppressed, false)
        },
        OnsetCase {
            quiet: Some((ONSET_NOW_LOCAL_MIN + 1, ONSET_NOW_LOCAL_MIN + 2)),
            ..onset_case(Onset, true)
        },
        OnsetCase { quiet: Some((23 * 60, 7 * 60)), ..onset_case(Suppressed, false) },
        OnsetCase { quiet: Some((23 * 60, 3 * 60)), ..onset_case(Onset, true) },
        OnsetCase {
            quiet: Some((ONSET_NOW_LOCAL_MIN, ONSET_NOW_LOCAL_MIN + 1)),
            tz_offset_sec: 3_600,
            ..onset_case(Onset, true)
        },
    ]
}

struct OnsetOut {
    decisions: Vec<(OnsetReason, bool)>,
    firing_fast: Option<f64>,
    firing_baseline: Option<f64>,
}

/// The shipped gate: every row's reason and nudge, plus the firing row's two numbers to the bit.
fn onset_gate(o: &OnsetOut) -> bool {
    let cases = onset_cases();
    o.decisions.len() == cases.len()
        && o.decisions
            .iter()
            .zip(cases.iter())
            .all(|(&(r, n), c)| r == c.reason && n == c.nudge)
        && o.firing_fast.map(|v| v.to_bits()) == Some(ONSET_GATE_FAST.to_bits())
        && o.firing_baseline.map(|v| v.to_bits()) == Some(ONSET_GATE_BASELINE.to_bits())
}

/// Every tunable `stress_onset.rs` reads, plus `stress::ACTIVITY_GATE_G`. All private.
/// `fast_override` and `force` are NOT shipped constants — they are the NULL arms' stand-ins for
/// "the R-R buffer is never read" and "the verdict is a constant".
#[derive(Clone, Copy)]
struct OnsetParams {
    baseline_ema_alpha: f64,
    drop_ratio: f64,
    fast_window_beats: usize,
    min_beats: usize,
    resting_hr_low: f64,
    resting_hr_high: f64,
    min_seconds_between_fires: i64,
    activity_gate_g: f64,
    fast_override: Option<f64>,
    force: Option<(OnsetReason, bool)>,
}

const ONSET_SHIPPED_PARAMS: OnsetParams = OnsetParams {
    baseline_ema_alpha: 0.98,       // stress_onset.rs:8
    drop_ratio: 0.6,                // :9
    fast_window_beats: 60,          // :10
    min_beats: 20,                  // :11
    resting_hr_low: 55.0,           // :12
    resting_hr_high: 100.0,         // :13
    min_seconds_between_fires: 900, // :14
    activity_gate_g: 0.15,          // stress::ACTIVITY_GATE_G, window.rs:18
    fast_override: None,
    force: None,
};

/// Twin of `evaluate`, quiet hours and time zone included.
fn onset_twin(
    c: &OnsetCase,
    p: OnsetParams,
    rr: &[u16],
) -> (OnsetReason, Option<f64>, Option<f64>, bool) {
    let base_or_none =
        if c.state.baseline_rmssd > 0.0 { Some(c.state.baseline_rmssd) } else { None };
    let forced = |r: OnsetReason, n: bool| p.force.unwrap_or((r, n));
    if !c.enabled || !c.auto_nudge {
        let (r, n) = forced(OnsetReason::Disabled, false);
        return (r, None, base_or_none, n);
    }
    let clean_all = HrvReadiness::clean_rr(rr);
    let fast_window = if clean_all.len() > p.fast_window_beats {
        &clean_all[clean_all.len() - p.fast_window_beats..]
    } else {
        &clean_all[..]
    };
    let fast = if fast_window.len() >= p.min_beats {
        match p.fast_override {
            Some(v) => Some(v),
            None => HrvReadiness::rmssd_plain(fast_window),
        }
    } else {
        None
    };
    let Some(fast) = fast else {
        let (r, n) = forced(OnsetReason::InsufficientData, false);
        return (r, None, base_or_none, n);
    };
    let new_baseline = if c.state.baseline_rmssd == 0.0 {
        fast
    } else {
        c.state.baseline_rmssd * p.baseline_ema_alpha + fast * (1.0 - p.baseline_ema_alpha)
    };
    let is_below = fast < new_baseline * p.drop_ratio;
    let is_edge = is_below && !c.state.was_below;
    let out = |r: OnsetReason, nudge: bool| {
        let (r, nudge) = forced(r, nudge);
        (r, Some(fast), Some(new_baseline), nudge)
    };

    if !is_below {
        return out(OnsetReason::NoDip, false);
    }
    if !is_edge {
        return out(OnsetReason::NotAnEdge, false);
    }
    let hr_in_band = c.hr.is_some_and(|h| (p.resting_hr_low..=p.resting_hr_high).contains(&h));
    let moving = c.motion.is_some_and(|m| m >= p.activity_gate_g);
    if !hr_in_band || moving {
        return out(OnsetReason::ExerciseGated, false);
    }
    if c.session_active {
        return out(OnsetReason::Suppressed, false);
    }
    if c.state.last_fire_at != 0
        && (ONSET_NOW - c.state.last_fire_at) < p.min_seconds_between_fires
    {
        return out(OnsetReason::Suppressed, false);
    }
    if let Some((start, end)) = c.quiet {
        let local_min = ((ONSET_NOW + c.tz_offset_sec).rem_euclid(86_400) / 60) as i32;
        let in_window = if start <= end {
            (start..end).contains(&local_min)
        } else {
            local_min >= start || local_min < end
        };
        if in_window {
            return out(OnsetReason::Suppressed, false);
        }
    }
    out(OnsetReason::Onset, true)
}

fn onset_run_shipped() -> OnsetOut {
    let cases = onset_cases();
    let mut decisions = Vec::new();
    let (mut fast, mut base) = (None, None);
    for (i, c) in cases.iter().enumerate() {
        let (qs, qe) = c.quiet.unwrap_or((0, 0));
        let d = evaluate(
            &c.rr, c.hr, c.motion, c.session_active, c.state, c.enabled, c.auto_nudge,
            c.quiet.is_some(), qs, qe, ONSET_NOW, c.tz_offset_sec,
        );
        decisions.push((d.reason, d.should_nudge));
        if i == ONSET_FIRING_CASE {
            fast = d.fast_rmssd;
            base = d.baseline_rmssd;
        }
    }
    OnsetOut { decisions, firing_fast: fast, firing_baseline: base }
}

fn onset_run_twin(p: OnsetParams, rr_map: &dyn Fn(&[u16]) -> Vec<u16>) -> OnsetOut {
    let cases = onset_cases();
    let mut decisions = Vec::new();
    let (mut fast, mut base) = (None, None);
    for (i, c) in cases.iter().enumerate() {
        let rr = rr_map(&c.rr);
        let (reason, f, b, nudge) = onset_twin(c, p, &rr);
        decisions.push((reason, nudge));
        if i == ONSET_FIRING_CASE {
            fast = f;
            base = b;
        }
    }
    OnsetOut { decisions, firing_fast: fast, firing_baseline: base }
}

fn onset_table() -> Table {
    assert_eq!(
        onset_shipped_test_count(),
        ONSET_SHIPPED_TESTS,
        "stress_onset.rs changed its test count: this control replicates a fixed decision table, so \
         it now measures the wrong gate — re-derive it before quoting a caught/missed tally"
    );
    let mut t = Table::new(
        "Live stress-onset detector (HRV dip, exercise-gated)",
        "physio-algo/src/stress_onset.rs — 20-row decision table (reason AND should_nudge on every \
         row, 6 of them firing) plus the firing row's fast RMSSD and folded baseline, bit-exact",
    );
    let identity = |rr: &[u16]| rr.to_vec();

    let o = onset_run_shipped();
    t.push("baseline (shipped evaluate over all 20 rows)", o.firing_fast, onset_gate(&o));

    let o = onset_run_twin(ONSET_SHIPPED_PARAMS, &identity);
    t.push("twin at shipped constants (fidelity check)", o.firing_fast, onset_gate(&o));

    // ── NULL ──
    let o = onset_run_twin(
        OnsetParams { force: Some((OnsetReason::NoDip, false)), ..ONSET_SHIPPED_PARAMS },
        &identity,
    );
    t.push("null: the detector never nudges (every row refuses)", o.firing_fast, onset_gate(&o));

    let o = onset_run_twin(
        OnsetParams { force: Some((OnsetReason::Onset, true)), ..ONSET_SHIPPED_PARAMS },
        &identity,
    );
    t.push("null: the detector always nudges (every row fires)", o.firing_fast, onset_gate(&o));

    let o = onset_run_twin(
        OnsetParams { fast_override: Some(ONSET_SEED_BASELINE), ..ONSET_SHIPPED_PARAMS },
        &identity,
    );
    t.push(
        "null: fast RMSSD is the baseline itself (R-R buffer never read)",
        o.firing_fast,
        onset_gate(&o),
    );

    let o =
        onset_run_twin(OnsetParams { fast_override: Some(0.0), ..ONSET_SHIPPED_PARAMS }, &identity);
    t.push("null: fast RMSSD is a constant 0 ms (R-R buffer never read)", o.firing_fast, onset_gate(&o));

    // ── STRUCTURAL ──
    let o = onset_run_twin(ONSET_SHIPPED_PARAMS, &|rr: &[u16]| {
        let mut v = rr.to_vec();
        v.reverse();
        v
    });
    t.push("structural: R-R buffer reversed", o.firing_fast, onset_gate(&o));

    let o = onset_run_twin(ONSET_SHIPPED_PARAMS, &|rr: &[u16]| shuffled(rr, 5));
    t.push("structural: R-R buffer shuffled (beat order destroyed)", o.firing_fast, onset_gate(&o));

    let o = onset_run_twin(ONSET_SHIPPED_PARAMS, &|rr: &[u16]| rr.iter().map(|v| v + 50).collect());
    t.push("structural: every beat offset +50 ms (differences unchanged)", o.firing_fast, onset_gate(&o));

    let o = onset_run_twin(ONSET_SHIPPED_PARAMS, &|rr: &[u16]| {
        let keep = rr.len() - rr.len() / 10;
        rr[..keep].to_vec()
    });
    t.push("structural: last 10% of the buffer dropped", o.firing_fast, onset_gate(&o));

    // ── PARAMETER (twin) ──
    let arms: [(&str, OnsetParams); 18] = [
        (
            "param: BASELINE_EMA_ALPHA 0.98 -> 1.078 (+10%, outside [0,1])",
            OnsetParams { baseline_ema_alpha: 1.078, ..ONSET_SHIPPED_PARAMS },
        ),
        (
            "param: BASELINE_EMA_ALPHA 0.98 -> 0.882 (-10%)",
            OnsetParams { baseline_ema_alpha: 0.882, ..ONSET_SHIPPED_PARAMS },
        ),
        (
            "param: DROP_RATIO 0.6 -> 0.66 (+10%)",
            OnsetParams { drop_ratio: 0.66, ..ONSET_SHIPPED_PARAMS },
        ),
        (
            "param: DROP_RATIO 0.6 -> 0.54 (-10%)",
            OnsetParams { drop_ratio: 0.54, ..ONSET_SHIPPED_PARAMS },
        ),
        (
            "param: DROP_RATIO 0.6 -> 0.603 (+0.5% floor probe)",
            OnsetParams { drop_ratio: 0.603, ..ONSET_SHIPPED_PARAMS },
        ),
        (
            "param: DROP_RATIO 0.6 -> 999.0 (extreme: everything is a dip)",
            OnsetParams { drop_ratio: 999.0, ..ONSET_SHIPPED_PARAMS },
        ),
        (
            "param: FAST_WINDOW_BEATS 60 -> 66 (+10%)",
            OnsetParams { fast_window_beats: 66, ..ONSET_SHIPPED_PARAMS },
        ),
        (
            "param: FAST_WINDOW_BEATS 60 -> 54 (-10%)",
            OnsetParams { fast_window_beats: 54, ..ONSET_SHIPPED_PARAMS },
        ),
        (
            "param: MIN_BEATS 20 -> 22 (+10%)",
            OnsetParams { min_beats: 22, ..ONSET_SHIPPED_PARAMS },
        ),
        (
            "param: MIN_BEATS 20 -> 18 (-10%)",
            OnsetParams { min_beats: 18, ..ONSET_SHIPPED_PARAMS },
        ),
        (
            "param: MIN_BEATS 20 -> 9 (extreme: below the short row)",
            OnsetParams { min_beats: 9, ..ONSET_SHIPPED_PARAMS },
        ),
        (
            "param: RESTING_HR_LOW 55 -> 60.5 (+10%)",
            OnsetParams { resting_hr_low: 60.5, ..ONSET_SHIPPED_PARAMS },
        ),
        (
            "param: RESTING_HR_LOW 55 -> 49.5 (-10%)",
            OnsetParams { resting_hr_low: 49.5, ..ONSET_SHIPPED_PARAMS },
        ),
        (
            "param: RESTING_HR_HIGH 100 -> 110 (+10%)",
            OnsetParams { resting_hr_high: 110.0, ..ONSET_SHIPPED_PARAMS },
        ),
        (
            "param: RESTING_HR_HIGH 100 -> 90 (-10%)",
            OnsetParams { resting_hr_high: 90.0, ..ONSET_SHIPPED_PARAMS },
        ),
        (
            "param: MIN_SECONDS_BETWEEN_FIRES 900 -> 990 (+10%)",
            OnsetParams { min_seconds_between_fires: 990, ..ONSET_SHIPPED_PARAMS },
        ),
        (
            "param: ACTIVITY_GATE_G 0.15 -> 0.165 (+10%)",
            OnsetParams { activity_gate_g: 0.165, ..ONSET_SHIPPED_PARAMS },
        ),
        (
            "param: ACTIVITY_GATE_G 0.15 -> 0.135 (-10%)",
            OnsetParams { activity_gate_g: 0.135, ..ONSET_SHIPPED_PARAMS },
        ),
    ];
    for (name, p) in arms {
        let o = onset_run_twin(p, &identity);
        t.push(name, o.firing_fast, onset_gate(&o));
    }
    t
}


// ─────────────────────────────────────────────────────────────────────────────────────────────
// the run
// ─────────────────────────────────────────────────────────────────────────────────────────────

// ── Sensitivity floors ─────────────────────────────────────────────────────────────────────────

/// `(metric, arm, minimum |delta| from the baseline)`. A floor asserts the arm still MOVES the number,
/// which is what catches an algorithm that stopped being reached; each is 0.45x the delta measured
/// 2026-08-02, so it sits well below the observed move and well above zero.
const FLOORS: &[(&str, &str, f64)] = &[
    ("Baevsky Stress Index (SI) + components (Mo, AMo, MxDMn)", "structural: last 10% of beats dropped (22 -> 20)", 1.54),
    ("Baevsky Stress Index (SI) + components (Mo, AMo, MxDMn)", "structural: every beat offset +10 ms", 1.2),
    ("Baevsky Stress Index (SI) + components (Mo, AMo, MxDMn)", "structural: every beat scaled x1.10", 16.3),
    ("Daily autonomic stress (0-3, RHR + HRV vs 14-day baseline)", "null: today replaced by the baseline mean day (today never read)", 0.674),
    ("Daily autonomic stress (0-3, RHR + HRV vs 14-day baseline)", "null: constant neutral 1.5 for every day", 0.674),
    ("Daily autonomic stress (0-3, RHR + HRV vs 14-day baseline)", "structural: both z-terms sign-flipped (calm reads as stressed)", 1.34),
    ("Daily autonomic stress (0-3, RHR + HRV vs 14-day baseline)", "structural: HRV term sign-flipped only (high HRV reads as stress)", 1.08),
    ("Daily autonomic stress (0-3, RHR + HRV vs 14-day baseline)", "structural: last 10% of baseline days dropped", 0.000105),
    ("Daily autonomic stress (0-3, RHR + HRV vs 14-day baseline)", "structural: today's RHR and HRV channels swapped", 1.34),
    ("Windowed stress (daytime + sleep): per-bucket score, bands, sustained run", "null: every bucket replaced by the day mean (no spread left)", 0.314),
    ("Windowed stress (daytime + sleep): per-bucket score, bands, sustained run", "null: constant neutral 1.5 in every bucket", 0.314),
    ("Windowed stress (daytime + sleep): per-bucket score, bands, sustained run", "structural: channels shifted +1 bucket (60 min)", 0.0144),
    ("Windowed stress (daytime + sleep): per-bucket score, bands, sustained run", "structural: day reversed", 0.0373),
    ("Windowed stress (daytime + sleep): per-bucket score, bands, sustained run", "structural: buckets shuffled (same values, wrong hours)", 0.0569),
    ("Windowed stress (daytime + sleep): per-bucket score, bands, sustained run", "structural: HR and RMSSD channels swapped", 0.0156),
    ("Stress band + squash (shared 0-3 scale)", "null: squash returns 0.0 for every input", 0.675),
    ("Live stress-onset detector (HRV dip, exercise-gated)", "null: fast RMSSD is the baseline itself (R-R buffer never read)", 7.6),
    ("Live stress-onset detector (HRV dip, exercise-gated)", "null: fast RMSSD is a constant 0 ms (R-R buffer never read)", 10.3),
    ("Live stress-onset detector (HRV dip, exercise-gated)", "structural: R-R buffer shuffled (beat order destroyed)", 3.21),
];

/// `(metric, arm, why)`. Probe arms that cannot carry a floor, because the mutation does not move the
/// number at all. Their blindness is the finding, not a defect to assert away.
const NO_FLOOR: &[(&str, &str, &str)] = &[
    ("Baevsky Stress Index (SI) + components (Mo, AMo, MxDMn)", "null: every beat replaced by the series mean", "the arm yields no number, so it has no distance from the baseline"),
    ("Baevsky Stress Index (SI) + components (Mo, AMo, MxDMn)", "null: every beat a constant 800 ms", "the arm yields no number, so it has no distance from the baseline"),
    ("Baevsky Stress Index (SI) + components (Mo, AMo, MxDMn)", "structural: series reversed", "measured delta is exactly zero: this mutation does not move the number"),
    ("Baevsky Stress Index (SI) + components (Mo, AMo, MxDMn)", "structural: series shuffled (order destroyed, histogram kept)", "measured delta is exactly zero: this mutation does not move the number"),
    ("Daily autonomic stress (0-3, RHR + HRV vs 14-day baseline)", "structural: baseline days shuffled", "measured delta is exactly zero: this mutation does not move the number"),
    ("Windowed stress (daytime + sleep): per-bucket score, bands, sustained run", "structural: last 10% of buckets dropped (24 -> 22)", "measured delta is exactly zero: this mutation does not move the number"),
    ("Windowed stress (daytime + sleep): per-bucket score, bands, sustained run", "structural: HR offset +10 bpm on every bucket", "measured delta is exactly zero: this mutation does not move the number"),
    ("Windowed stress (daytime + sleep): per-bucket score, bands, sustained run", "structural: HR scaled x2.00 on every bucket (exact in binary)", "measured delta is exactly zero: this mutation does not move the number"),
    ("Windowed stress (daytime + sleep): per-bucket score, bands, sustained run", "structural: HR scaled x1.10 on every bucket", "the true delta is exactly zero — the score is affine-invariant in HR — and the ULP residue x1.10 leaves is float rounding, not sensitivity: x2.00 moves it 0.0, x1.000000001 moves it MORE than x1.10 does"),
    ("Stress band + squash (shared 0-3 scale)", "null: band_of returns Low for every score", "measured delta is exactly zero: this mutation does not move the number"),
    ("Stress band + squash (shared 0-3 scale)", "structural: squash sign-flipped (stress reads as calm)", "measured delta is exactly zero: this mutation does not move the number"),
    ("Stress band + squash (shared 0-3 scale)", "structural: bands reversed (Low <-> High)", "measured delta is exactly zero: this mutation does not move the number"),
    ("Stress band + squash (shared 0-3 scale)", "structural: band comparisons '<' -> '<=' (boundary off-by-one)", "measured delta is exactly zero: this mutation does not move the number"),
    ("Live stress-onset detector (HRV dip, exercise-gated)", "null: the detector never nudges (every row refuses)", "the value column is the firing row's fast RMSSD, which a forced verdict does not move: measured delta is exactly zero"),
    ("Live stress-onset detector (HRV dip, exercise-gated)", "null: the detector always nudges (every row fires)", "the value column is the firing row's fast RMSSD, which a forced verdict does not move: measured delta is exactly zero"),
    ("Live stress-onset detector (HRV dip, exercise-gated)", "structural: R-R buffer reversed", "measured delta is exactly zero: this mutation does not move the number"),
    ("Live stress-onset detector (HRV dip, exercise-gated)", "structural: every beat offset +50 ms (differences unchanged)", "measured delta is exactly zero: this mutation does not move the number"),
    ("Live stress-onset detector (HRV dip, exercise-gated)", "structural: last 10% of the buffer dropped", "measured delta is exactly zero: this mutation does not move the number"),
];

/// Assert one metric's floors, and require every NULL/STRUCTURAL arm to be classified.
/// The `NO_FLOOR` reason for one arm, if it has one. An arm with a waiver is declared NOT to move the
/// number, so its measured delta is not a sensitivity reading and cannot serve as a floor.
fn waiver_for(metric: &str, arm: &str) -> Option<&'static str> {
    NO_FLOOR.iter().find(|(m, a, _)| *m == metric && *a == arm).map(|t| t.2)
}

fn enforce_floors(metric: &str, base: f64, probes: &[(&str, f64)]) {
    let (mut asserted, mut waived) = (0usize, 0usize);
    let mut breached: Vec<String> = Vec::new();
    let mut unclassified: Vec<&str> = Vec::new();
    for &(arm, value) in probes {
        let floor = FLOORS.iter().find(|(m, a, _)| *m == metric && *a == arm).map(|t| t.2);
        let waiver = waiver_for(metric, arm);
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

#[test]
#[ignore = "negative control, not a gate: run explicitly with --ignored"]
fn stress_family_sensitivity_controls() {
    let tables = [si_table(), daily_table(), win_table(), scale_table(), onset_table()];

    println!(
        "\nNEGATIVE CONTROLS — stress family. Each arm perturbs the metric and re-runs the SHIPPED \
         gate.\nA caught arm is a gate doing its job; a missed arm is drift the gate would not \
         notice.\nThe tally counts perturbation arms only: every table's first two rows are the \
         shipped baseline\nand the twin at the shipped constants, both required to pass.\nWellness \
         estimates only, never medical."
    );

    let mut caught = 0usize;
    let mut missed = 0usize;
    let mut floors: Vec<(&str, Option<f64>)> = Vec::new();
    for t in &tables {
        let (c, m, f) = t.print();
        caught += c;
        missed += m;
        floors.push((t.metric, f));
    }

    // Every table contributes exactly two unscored rows, so the printed rows must reconcile with
    // the tally. Printing the split keeps the denominator auditable against the raw arm count.
    let fixed_rows = tables.len() * FIRST_PERTURBATION_ARM;
    let printed_rows: usize = tables.iter().map(|t| t.arms.len()).sum();
    assert_eq!(
        printed_rows,
        caught + missed + fixed_rows,
        "tally does not reconcile with the rows printed — an arm was scored twice or not at all"
    );
    println!(
        "\n=== stress family: caught {caught}, missed {missed} of {} perturbation arms ===",
        caught + missed
    );
    println!(
        "    {printed_rows} rows printed = {} perturbation arms + {fixed_rows} fixed rows \
         ({n} baselines + {n} twin-fidelity), neither kind being a perturbation",
        caught + missed,
        n = tables.len()
    );
    println!(
        "sensitivity floor per metric (smallest delta the shipped gate catches, ignoring arms \
         whose true delta is zero and whose caught residue is only float rounding):"
    );
    for (metric, f) in &floors {
        match f {
            Some(v) => println!("   {v:.3e}  {metric}"),
            None => println!("   {:>9}  {metric}", "n/a"),
        }
    }

    for t in &tables {
        assert_trustworthy(t);
    }
}
