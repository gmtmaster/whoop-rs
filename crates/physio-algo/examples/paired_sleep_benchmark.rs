//! Canonical replay adapter for the paired WHOOP-vs-NOOP sleep benchmark (`dev-docs/paired-sleep-benchmark/`).
//!
//! One thin input adapter (parse this night's raw CSV export) feeding STOCK `physio-algo` — this file
//! must never grow a second implementation of staging/HRV/RR. Any future night replays through this same
//! binary; only the env vars below change.
//!
//!   WHOOP_NIGHT_DIR=/path/to/raw/night-YYYY-MM-DD \
//!   WHOOP_NIGHT_ID=YYYY-MM-DD \
//!   WHOOP_SESSION_START="2026-08-31 22:07:47+00" \
//!   WHOOP_SESSION_END="2026-09-01 06:47:16+00" \
//!   cargo run --release -p physio-algo --example paired_sleep_benchmark
//!
//! Reads `heart_rate.csv`, `accelerometer.csv`, `rr_runs.csv` (all required) and `steps.csv` (optional --
//! required for `refine_wake` to actually run; see `dev-docs/paired-sleep-benchmark/raw-data-manifest.md`)
//! from `WHOOP_NIGHT_DIR`. Read-only: every call below is `Params::SHIPPED` / the shipped constants --
//! nothing here may run under a swept or tuned recipe. Writes
//! `dev-docs/paired-sleep-benchmark/nights/<WHOOP_NIGHT_ID>/epoch_diagnostics_rust.csv`.
//!
//! Night 1 (2026-09-01) shipped before a Rust toolchain was available in the session that built this, so
//! its numbers came from a hand-validated Python port kept at
//! `dev-docs/paired-sleep-benchmark/nights/2026-09-01/forensic/python_replay_port.py` — a **forensic
//! fallback**, not the canonical implementation. This file is the canonical path for Night 2 onward, and
//! should be re-run against Night 1's raw export too once `cargo` is available, to confirm byte-identical
//! output against the Python port's already-published numbers.

use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::PathBuf;

use physio_algo::nightly_physiology::{nightly_physiology_for_generation, RrDeviceGeneration};
use physio_algo::respiratory_rate::resp_rate_from_rr;
use physio_algo::sleep::params::Params;
use physio_algo::sleep::{
    diagnostics_v2, emissions_v2, prepare_v2, refine_wake, segments_v2, stage_v2, AccelSample, HrSample,
    RrRun, SleepInput, StepSample,
};

fn env_var(name: &str) -> String {
    env::var(name).unwrap_or_else(|_| panic!("set {name} (see this file's module doc for the full list)"))
}

fn night_dir() -> PathBuf {
    PathBuf::from(env_var("WHOOP_NIGHT_DIR"))
}

/// "2026-08-31 22:07:47+00" -> unix seconds. No chrono dependency: the format is fixed-width.
fn parse_ts(s: &str) -> i64 {
    let y: i32 = s[0..4].parse().unwrap();
    let mo: u32 = s[5..7].parse().unwrap();
    let d: u32 = s[8..10].parse().unwrap();
    let h: i64 = s[11..13].parse().unwrap();
    let mi: i64 = s[14..16].parse().unwrap();
    let se: i64 = s[17..19].parse().unwrap();
    // Days since epoch via a plain civil-to-days conversion (Howard Hinnant's algorithm), UTC only.
    let (y, mo) = if mo <= 2 { (y - 1, mo + 12) } else { (y, mo) };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = (y - era * 400) as i64;
    let doy = (153 * (mo as i64 - 3) + 2) / 5 + d as i64 - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era as i64 * 146_097 + doe - 719_468;
    days * 86_400 + h * 3600 + mi * 60 + se
}

fn read_hr(path: &PathBuf) -> Vec<HrSample> {
    fs::read_to_string(path)
        .unwrap()
        .lines()
        .skip(1)
        .map(|l| {
            let mut c = l.split(',');
            let ts = parse_ts(c.next().unwrap());
            let bpm: u16 = c.next().unwrap().parse().unwrap();
            HrSample { ts, bpm }
        })
        .collect()
}

fn read_accel(path: &PathBuf) -> Vec<AccelSample> {
    fs::read_to_string(path)
        .unwrap()
        .lines()
        .skip(1)
        .map(|l| {
            let mut c = l.split(',');
            let ts = parse_ts(c.next().unwrap());
            let x: f64 = c.next().unwrap().parse().unwrap();
            let y: f64 = c.next().unwrap().parse().unwrap();
            let z: f64 = c.next().unwrap().parse().unwrap();
            AccelSample { ts, x, y, z }
        })
        .collect()
}

fn read_rr(path: &PathBuf) -> Vec<RrRun> {
    fs::read_to_string(path)
        .unwrap()
        .lines()
        .skip(1)
        .map(|l| {
            let mut c = l.splitn(4, ',');
            let ts = parse_ts(c.next().unwrap());
            let ivs_raw = c.next().unwrap(); // "[1088]" or "[1088,1011]"
            let intervals: Vec<u16> = ivs_raw
                .trim_matches(|ch| ch == '[' || ch == ']')
                .split(',')
                .filter(|s| !s.is_empty())
                .map(|s| s.parse().unwrap())
                .collect();
            RrRun { ts, intervals }
        })
        .collect()
}

/// Optional: absent on nights collected before `steps.csv` was required (Night 1). `activity_class`
/// column 3 follows the forensic dataset's convention: -1/absent means "not decoded", not "still".
fn read_steps(path: &PathBuf) -> Vec<StepSample> {
    let Ok(text) = fs::read_to_string(path) else { return Vec::new() };
    text.lines()
        .skip(1)
        .map(|l| {
            let mut c = l.split(',');
            let ts = parse_ts(c.next().unwrap());
            let counter: u16 = c.next().unwrap().parse().unwrap();
            let cls: i32 = c.next().unwrap_or("-1").trim().parse().unwrap_or(-1);
            StepSample { ts, counter, activity_class: (cls >= 0).then(|| cls as u8) }
        })
        .collect()
}

fn main() {
    let night_id = env_var("WHOOP_NIGHT_ID");
    let d = night_dir();
    let hr = read_hr(&d.join("heart_rate.csv"));
    let accel = read_accel(&d.join("accelerometer.csv"));
    let rr = read_rr(&d.join("rr_runs.csv"));
    let steps = read_steps(&d.join("steps.csv"));
    if steps.is_empty() {
        eprintln!(
            "WARNING: no steps.csv for {night_id} -- refine_wake's density gate will decline; \
             Awake/Light boundary will not reflect the app's actual refined staging (see raw-data-manifest.md)"
        );
    }

    let start = parse_ts(&env_var("WHOOP_SESSION_START"));
    let end = parse_ts(&env_var("WHOOP_SESSION_END"));

    let input = SleepInput { start, end, hr: hr.clone(), rr: rr.clone(), accel: accel.clone() };
    let p = Params::SHIPPED;

    let prep = prepare_v2(&input, &p);
    let diag = diagnostics_v2(&prep, &p);
    let em = emissions_v2(&prep, &p);
    assert_eq!(em.len(), diag.len(), "diagnostics_v2 must mirror emissions_v2 row-for-row");

    let segments = stage_v2(&input); // SHIPPED, unrefined
    let refined = refine_wake(&segments, &accel, &steps);
    println!("staged {} segments, refine changed anything: {}", segments.len(), refined != segments);

    let rr_flat: Vec<(i64, u16)> =
        rr.iter().flat_map(|run| run.intervals.iter().map(move |&ms| (run.ts, ms))).collect();
    let resp = resp_rate_from_rr(&rr_flat, start, end);
    println!("resp_rate_from_rr = {resp:?}");

    let physiology = nightly_physiology_for_generation(
        start, end, &hr, &accel, &rr, &[], &refined, RrDeviceGeneration::Whoop5Mg,
    );
    println!("hrv.rmssd_ms = {:?}  mode = {:?}", physiology.hrv.rmssd_ms, physiology.hrv.measurement_mode);
    println!("hrv.selected_window = {:?}", physiology.hrv.selected_window);
    println!("rhr_v4.quality_v4 = {:?}", physiology.rhr_v4.quality_v4);

    let mut out = String::from(
        "start_utc,hr,hr_var,hr_flat11,hr_flat11_pct,move_frac,jerk_max,jerk_scale,resp_reg,clock,\
         z_hr,z_hr_var,z_move,z_resp_reg,deep_gate,cycle_prior_deep,cycle_prior_rem,rem_guard,\
         motion_gate_boost_applied,late_deep_bonus,em_deep,em_rem,em_light,em_awake\n",
    );
    for d in &diag {
        out.push_str(&format!(
            "{},{:?},{:?},{:?},{:.4},{:?},{:.6},{:.6},{:?},{:.5},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{},{:.4},{:.4},{:.4},{:.4},{:.4}\n",
            d.start, d.hr, d.hr_var, d.hr_flat11, d.hr_flat11_pct, d.move_frac, d.jerk_max, d.jerk_scale,
            d.resp_reg, d.clock, d.z_hr, d.z_hr_var, d.z_move, d.z_resp_reg, d.deep_gate,
            d.cycle_prior_deep, d.cycle_prior_rem, d.rem_guard, d.motion_gate_boost_applied,
            d.late_deep_bonus, d.em_deep, d.em_rem, d.em_light, d.em_awake,
        ));
    }
    let out_dir = format!("dev-docs/paired-sleep-benchmark/nights/{night_id}");
    fs::create_dir_all(&out_dir).ok();
    let out_path = format!("{out_dir}/epoch_diagnostics_rust.csv");
    fs::write(&out_path, out).unwrap();
    println!("wrote {out_path}");

    let _ = HashMap::<(), ()>::new(); // silence unused-import lints on some toolchains
    let _ = segments_v2; // exported for a caller that wants to decode its own path; unused in this replay
}
