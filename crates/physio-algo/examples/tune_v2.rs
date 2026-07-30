//! Coordinate-descent tuner for the V2 sleep recipe. `--fit` names the fitting set (`pooled` = every set
//! except `--holdout`); the held-out set is never scored during the search, so the report separates a real
//! generalising gain from a fit to one population.
//!
//!   cargo run --release -p physio-algo --example tune_v2 -- --fit pooled --holdout killa5 --rounds 5
//!
//! Prints the winning `Params` as Rust source. Nothing is written; adopting a result is a manual edit.
//!
//! `stage_v2` ONLY. The fitting sets are PSG cohorts with no step stream, so `refine_wake` would
//! decline; a recipe fitted here is fitted on the unrefined path.

mod common;

use std::collections::BTreeMap;
use std::io::Write;
use std::fs;
use std::path::{Path, PathBuf};

use physio_algo::sleep::{
    params::Params, prepare_v2, stage_v2_prepared, AccelSample, HrSample, Prepared, RrRun, SleepInput,
    SleepStage,
};

const DATASETS: [&str; 4] = ["dreamt", "aauwss", "killa5", "sleep-accel"];

fn fixtures_root() -> PathBuf {
    common::fixtures_root()
}

struct Night {
    input: SleepInput,
    w0: i64,
    n_epochs: usize,
    truth: BTreeMap<usize, i32>,
}

fn read_csv(path: &Path) -> Vec<Vec<f64>> {
    fs::read_to_string(path)
        .map(|t| {
            t.lines()
                .filter(|l| !l.trim().is_empty())
                .map(|l| l.split(',').map(|c| c.trim().parse::<f64>().unwrap()).collect())
                .collect()
        })
        .unwrap_or_default()
}

fn load_night(dir: &Path) -> Option<Night> {
    let meta = fs::read_to_string(dir.join("meta.txt")).ok()?;
    let m: Vec<i64> = meta.split_whitespace().map(|x| x.parse().unwrap()).collect();
    let (w0, w1, n_epochs) = (m[1], m[2], m[3] as usize);

    let accel = read_csv(&dir.join("gravity.csv"))
        .iter()
        .map(|r| AccelSample { ts: r[0] as i64, x: r[1], y: r[2], z: r[3] })
        .collect();
    let hr = read_csv(&dir.join("hr.csv"))
        .iter()
        .map(|r| HrSample { ts: r[0] as i64, bpm: r[1] as u16 })
        .collect();
    let mut rr: Vec<RrRun> = Vec::new();
    for row in read_csv(&dir.join("rr.csv")) {
        let (ts, ms) = (row[0] as i64, row[1] as u16);
        match rr.last_mut() {
            Some(last) if last.ts == ts => last.intervals.push(ms),
            _ => rr.push(RrRun { ts, intervals: vec![ms] }),
        }
    }
    let mut truth = BTreeMap::new();
    for row in read_csv(&dir.join("truth.csv")) {
        truth.insert(row[0] as usize, row[1] as i32);
    }
    if truth.is_empty() {
        return None; // an empty ground truth measures nothing
    }
    Some(Night { input: SleepInput { start: w0, end: w1, hr, rr, accel }, w0, n_epochs, truth })
}

fn load_dataset(ds: &str) -> Vec<Night> {
    let root = fixtures_root().join(ds);
    let mut dirs: Vec<PathBuf> = fs::read_dir(&root)
        .map(|rd| rd.filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.is_dir()).collect())
        .unwrap_or_default();
    dirs.sort();
    dirs.iter().filter_map(|d| load_night(d)).collect()
}

fn stage_to_int(s: SleepStage) -> i32 {
    match s {
        SleepStage::Wake => 0,
        SleepStage::Light => 1,
        SleepStage::Deep => 2,
        SleepStage::Rem => 3,
    }
}

fn cohen_kappa(cm: &[[i64; 4]; 4]) -> f64 {
    let tot: i64 = cm.iter().flatten().sum();
    if tot == 0 {
        return 0.0;
    }
    let tot = tot as f64;
    let agree: i64 = (0..4).map(|i| cm[i][i]).sum();
    let po = agree as f64 / tot;
    let mut pe = 0.0;
    for (j, row_j) in cm.iter().enumerate() {
        let col: i64 = cm.iter().map(|r| r[j]).sum();
        let row: i64 = row_j.iter().sum();
        pe += col as f64 * row as f64;
    }
    pe /= tot * tot;
    if pe >= 1.0 {
        0.0
    } else {
        (po - pe) / (1.0 - pe)
    }
}

/// Feature extraction is the expensive half and only `jerk_move_mult` changes it, so a sweep extracts
/// once and re-labels. Re-prepare whenever that axis moves.
fn prepare_all(nights: &[Night], p: &Params) -> Vec<Prepared> {
    nights.iter().map(|n| prepare_v2(&n.input, p)).collect()
}

fn kappa(nights: &[Night], prep: &[Prepared], p: &Params) -> f64 {
    let mut cm = [[0i64; 4]; 4];
    accumulate(nights, prep, p, &mut cm);
    cohen_kappa(&cm)
}

/// Pooled kappa over several datasets: one confusion matrix across all of them, so the objective is
/// overall agreement rather than a mean that a small set can swing.
fn pooled_kappa(sets: &[(&[Night], &[Prepared])], p: &Params) -> f64 {
    let mut cm = [[0i64; 4]; 4];
    for (nights, prep) in sets {
        accumulate(nights, prep, p, &mut cm);
    }
    cohen_kappa(&cm)
}

fn accumulate(nights: &[Night], prep: &[Prepared], p: &Params, cm: &mut [[i64; 4]; 4]) {
    for (n, pr) in nights.iter().zip(prep) {
        let segs = stage_v2_prepared(pr, p);
        for (k, &t) in &n.truth {
            if *k >= n.n_epochs || !(0..4).contains(&t) {
                continue;
            }
            let mid = n.w0 + *k as i64 * 30 + 15;
            let stage = segs
                .iter()
                .find(|s| s.start <= mid && mid < s.end)
                .map(|s| s.stage)
                .unwrap_or_else(|| segs.last().unwrap().stage);
            cm[t as usize][stage_to_int(stage) as usize] += 1;
        }
    }
}

/// The swept axes: a label, a read, a write, and the step ladder to try around the current value.
struct Axis {
    name: &'static str,
    get: fn(&Params) -> f64,
    set: fn(&mut Params, f64),
    steps: &'static [f64],
}

const WEIGHT_STEPS: &[f64] = &[-0.4, -0.2, -0.1, -0.05, 0.05, 0.1, 0.2, 0.4];
const RATE_STEPS: &[f64] = &[-0.08, -0.04, -0.02, 0.02, 0.04, 0.08];
const MULT_STEPS: &[f64] = &[-20.0, -10.0, -5.0, 5.0, 10.0, 20.0];

fn axes() -> Vec<Axis> {
    macro_rules! ax {
        ($n:literal, $f:ident, $s:expr) => {
            Axis { name: $n, get: |p| p.$f, set: |p, v| p.$f = v, steps: $s }
        };
    }
    vec![
        ax!("deep_hrv", deep_hrv, WEIGHT_STEPS),
        ax!("deep_hr", deep_hr, WEIGHT_STEPS),
        ax!("deep_motion", deep_motion, WEIGHT_STEPS),
        ax!("rem_hrv", rem_hrv, WEIGHT_STEPS),
        ax!("rem_motion", rem_motion, WEIGHT_STEPS),
        ax!("rem_hr", rem_hr, WEIGHT_STEPS),
        ax!("awake_motion", awake_motion, WEIGHT_STEPS),
        ax!("awake_hrv", awake_hrv, WEIGHT_STEPS),
        ax!("awake_hr", awake_hr, WEIGHT_STEPS),
        ax!("awake_deadzone", awake_deadzone, RATE_STEPS),
        ax!("deep_gate_thresh", deep_gate_thresh, RATE_STEPS),
        ax!("deep_gate_slope", deep_gate_slope, WEIGHT_STEPS),
        ax!("motion_gate_boost", motion_gate_boost, WEIGHT_STEPS),
        ax!("resp_weight", resp_weight, WEIGHT_STEPS),
        ax!("cycle_deep_scale", cycle_deep_scale, WEIGHT_STEPS),
        ax!("cycle_deep_decay", cycle_deep_decay, RATE_STEPS),
        ax!("cycle_rem_scale", cycle_rem_scale, WEIGHT_STEPS),
        ax!("cycle_rem_early_frac", cycle_rem_early_frac, RATE_STEPS),
        ax!("cycle_rem_early_penalty", cycle_rem_early_penalty, WEIGHT_STEPS),
        ax!("jerk_move_mult", jerk_move_mult, MULT_STEPS),
        ax!("jerk_gate_mult", jerk_gate_mult, MULT_STEPS),
        Axis { name: "base_deep", get: |p| p.base_rate[0], set: |p, v| p.base_rate[0] = v, steps: RATE_STEPS },
        Axis { name: "base_rem", get: |p| p.base_rate[1], set: |p, v| p.base_rate[1] = v, steps: RATE_STEPS },
        Axis { name: "base_light", get: |p| p.base_rate[2], set: |p, v| p.base_rate[2] = v, steps: RATE_STEPS },
        Axis { name: "base_awake", get: |p| p.base_rate[3], set: |p, v| p.base_rate[3] = v, steps: RATE_STEPS },
    ]
}

/// A candidate is rejected outright when it is not a usable recipe (non-positive base rate, a negative
/// gate, an inverted decay), so the search never spends a night's staging on nonsense.
fn valid(p: &Params) -> bool {
    p.base_rate.iter().all(|&r| r > 0.001 && r < 1.0)
        && p.awake_deadzone >= 0.0
        && p.deep_gate_thresh > 0.0
        && p.deep_gate_thresh < 1.0
        && p.deep_gate_slope >= 0.0
        && p.cycle_deep_decay > 0.05
        && p.cycle_rem_early_frac >= 0.0
        && p.jerk_move_mult > 1.0
        && p.jerk_gate_mult > 1.0
        && p.resp_weight >= 0.0
}

fn emit(p: &Params) {
    println!("\n// ---- winning recipe ----");
    println!("    deep_hrv: {:?},\n    deep_hr: {:?},\n    deep_motion: {:?},", p.deep_hrv, p.deep_hr, p.deep_motion);
    println!("    rem_hrv: {:?},\n    rem_motion: {:?},\n    rem_hr: {:?},", p.rem_hrv, p.rem_motion, p.rem_hr);
    println!("    awake_motion: {:?},\n    awake_hrv: {:?},\n    awake_hr: {:?},", p.awake_motion, p.awake_hrv, p.awake_hr);
    println!("    awake_deadzone: {:?},", p.awake_deadzone);
    println!("    deep_gate_thresh: {:?},\n    deep_gate_slope: {:?},", p.deep_gate_thresh, p.deep_gate_slope);
    println!("    jerk_move_mult: {:?},\n    jerk_gate_mult: {:?},", p.jerk_move_mult, p.jerk_gate_mult);
    println!("    motion_gate_boost: {:?},\n    resp_weight: {:?},", p.motion_gate_boost, p.resp_weight);
    println!("    base_rate: {:?},", p.base_rate);
    println!("    cycle_deep_scale: {:?},\n    cycle_deep_decay: {:?},", p.cycle_deep_scale, p.cycle_deep_decay);
    println!("    cycle_rem_scale: {:?},\n    cycle_rem_early_frac: {:?},\n    cycle_rem_early_penalty: {:?},",
        p.cycle_rem_scale, p.cycle_rem_early_frac, p.cycle_rem_early_penalty);
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let arg = |k: &str, d: &str| -> String {
        args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned().unwrap_or_else(|| d.to_string())
    };
    let fit_on = arg("--fit", "pooled");
    let holdout = arg("--holdout", "killa5");
    let rounds: usize = arg("--rounds", "5").parse().unwrap_or(5);

    let sets: Vec<(String, Vec<Night>)> =
        DATASETS.iter().map(|d| (d.to_string(), load_dataset(d))).filter(|(_, n)| !n.is_empty()).collect();
    for (name, nights) in &sets {
        println!("loaded {name:<12} {} nights", nights.len());
    }
    let in_fit = |name: &str| -> bool {
        if fit_on == "pooled" { name != holdout } else { name == fit_on }
    };
    let names = |keep: bool| -> String {
        sets.iter().filter(|(n, _)| in_fit(n) == keep).map(|(n, _)| n.as_str()).collect::<Vec<_>>().join(" + ")
    };
    println!("\nfitting on: {}", names(true));
    println!("held out  : {}\n", names(false));

    let mut best = Params::SHIPPED;
    let mut preps: Vec<Vec<Prepared>> = sets.iter().map(|(_, n)| prepare_all(n, &best)).collect();
    let score = |preps: &[Vec<Prepared>], p: &Params| -> f64 {
        let view: Vec<(&[Night], &[Prepared])> = sets
            .iter()
            .zip(preps)
            .filter(|((name, _), _)| in_fit(name))
            .map(|((_, n), pr)| (n.as_slice(), pr.as_slice()))
            .collect();
        pooled_kappa(&view, p)
    };

    let mut best_k = score(&preps, &best);
    println!("baseline pooled kappa on the fitting set: {best_k:.4}\n");

    for round in 1..=rounds {
        let mut improved_any = false;
        for ax in axes() {
            let base = (ax.get)(&best);
            let mut local_best = (best_k, base);
            for d in ax.steps {
                let mut cand = best;
                (ax.set)(&mut cand, base + d);
                if !valid(&cand) {
                    continue;
                }
                let k = if ax.name == "jerk_move_mult" {
                    let cand_preps: Vec<Vec<Prepared>> =
                        sets.iter().map(|(_, n)| prepare_all(n, &cand)).collect();
                    score(&cand_preps, &cand)
                } else {
                    score(&preps, &cand)
                };
                if k > local_best.0 + 1e-6 {
                    local_best = (k, base + d);
                }
            }
            if local_best.1 != base {
                (ax.set)(&mut best, local_best.1);
                println!("  round {round}  {:<24} {base:>8.4} -> {:>8.4}   kappa {best_k:.4} -> {:.4}",
                    ax.name, local_best.1, local_best.0);
                let _ = std::io::stdout().flush();
                best_k = local_best.0;
                if ax.name == "jerk_move_mult" {
                    preps = sets.iter().map(|(_, n)| prepare_all(n, &best)).collect();
                }
                improved_any = true;
            }
        }
        if !improved_any {
            println!("  round {round}: no axis improved; converged");
            break;
        }
    }

    println!("\n{:<14} {:>10} {:>10} {:>9}   role", "dataset", "shipped", "tuned", "delta");
    for (name, nights) in &sets {
        let before = kappa(nights, &prepare_all(nights, &Params::SHIPPED), &Params::SHIPPED);
        let after = kappa(nights, &prepare_all(nights, &best), &best);
        let role = if in_fit(name) { "fit" } else { "HELD OUT" };
        println!("{name:<14} {before:>10.4} {after:>10.4} {:>+9.4}   {role}", after - before);
    }
    emit(&best);
}
