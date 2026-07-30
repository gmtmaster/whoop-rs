//! Sweep the REM cycle prior's shape against PSG truth: the early guard (magnitude and the minutes it
//! grades over) and the ceiling on the rising ramp. Reports agreement, REM-fraction calibration and
//! first-REM placement side by side, because a term can win one and lose another.
//!
//!   cargo run --release -p physio-algo --example sweep_rem
//!
//! Features are extracted once per dataset and re-labelled, so only the prior moves between rows.
//!
//! `stage_v2` ONLY: no PSG cohort carries a step stream dense enough for `refine_wake`, so the gate
//! would decline on every row. Stated because the gate is silent. The prior it sweeps is a deep/REM
//! term, and the refinement can move neither.

mod common;

use common::kappa4 as kappa;

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use physio_algo::sleep::{
    params::Params, prepare_v2, stage_v2_prepared, AccelSample, HrSample, Prepared, RrRun, SleepInput,
    SleepStage,
};

const SETS: [&str; 3] = ["sleep-accel", "dreamt", "aauwss"];
const REM: usize = 3;
const WAKE: usize = 0;

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
        return None;
    }
    Some(Night { input: SleepInput { start: w0, end: w1, hr, rr, accel }, w0, n_epochs, truth })
}

fn load(ds: &str) -> Vec<Night> {
    let mut dirs: Vec<PathBuf> = fs::read_dir(fixtures_root().join(ds))
        .map(|rd| rd.filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.is_dir()).collect())
        .unwrap_or_default();
    dirs.sort();
    dirs.iter().filter_map(|d| load_night(d)).collect()
}

fn stage_idx(s: SleepStage) -> usize {
    match s {
        SleepStage::Wake => 0,
        SleepStage::Light => 1,
        SleepStage::Deep => 2,
        SleepStage::Rem => 3,
    }
}

fn labels(n: &Night, prep: &Prepared, p: &Params) -> Vec<usize> {
    let segs = stage_v2_prepared(prep, p);
    (0..n.n_epochs)
        .map(|k| {
            let mid = n.w0 + k as i64 * 30 + 15;
            stage_idx(
                segs.iter()
                    .find(|s| s.start <= mid && mid < s.end)
                    .map(|s| s.stage)
                    .unwrap_or_else(|| segs.last().unwrap().stage),
            )
        })
        .collect()
}

fn onset_of(seq: &[usize]) -> Option<usize> {
    let mut run = 0;
    for (i, &l) in seq.iter().enumerate() {
        run = if l == WAKE { 0 } else { run + 1 };
        if run == 10 {
            return Some(i - 9);
        }
    }
    None
}

fn median(v: &mut [f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 1 { v[n / 2] } else { (v[n / 2 - 1] + v[n / 2]) / 2.0 }
}

struct Set {
    nights: Vec<Night>,
    prep: Vec<Prepared>,
    truth_rem: f64,
    truth_lat: f64,
}

/// Pooled agreement plus the two calibration errors the prior actually controls.
fn score(sets: &[Set], p: &Params) -> (f64, f64, f64, f64) {
    let mut cm = [[0i64; 4]; 4];
    let (mut rem, mut all) = (0i64, 0i64);
    let mut lat_err = Vec::new();
    let mut rem_err = Vec::new();
    for s in sets {
        let (mut srem, mut sall) = (0i64, 0i64);
        let mut lats = Vec::new();
        for (n, pr) in s.nights.iter().zip(&s.prep) {
            let lab = labels(n, pr, p);
            for (k, &t) in &n.truth {
                if *k < n.n_epochs && (0..4).contains(&t) {
                    cm[t as usize][lab[*k]] += 1;
                    sall += 1;
                    srem += (lab[*k] == REM) as i64;
                }
            }
            if let Some(on) = onset_of(&lab) {
                if let Some(f) = (on..n.n_epochs).find(|k| lab[*k] == REM) {
                    lats.push((f - on) as f64 * 0.5);
                }
            }
        }
        rem += srem;
        all += sall;
        rem_err.push((100.0 * srem as f64 / sall as f64 - s.truth_rem).abs());
        lat_err.push((median(&mut lats) - s.truth_lat).abs());
    }
    (
        kappa(&cm),
        100.0 * rem as f64 / all as f64,
        rem_err.iter().sum::<f64>() / rem_err.len() as f64,
        lat_err.iter().sum::<f64>() / lat_err.len() as f64,
    )
}

fn main() {
    let shipped = Params::SHIPPED;
    let sets: Vec<Set> = SETS
        .iter()
        .filter_map(|ds| {
            let nights = load(ds);
            if nights.is_empty() {
                return None;
            }
            let prep = nights.iter().map(|n| prepare_v2(&n.input, &shipped)).collect();
            let tot: f64 = nights.iter().map(|n| n.truth.len() as f64).sum();
            let rem: f64 = nights.iter().map(|n| n.truth.values().filter(|&&v| v == REM as i32).count() as f64).sum();
            let mut lats = Vec::new();
            for n in &nights {
                let seq: Vec<usize> = (0..n.n_epochs).map(|k| *n.truth.get(&k).unwrap_or(&0) as usize).collect();
                if let Some(on) = onset_of(&seq) {
                    if let Some(f) = (on..n.n_epochs).find(|k| seq[*k] == REM) {
                        lats.push((f - on) as f64 * 0.5);
                    }
                }
            }
            Some(Set { nights, prep, truth_rem: 100.0 * rem / tot, truth_lat: median(&mut lats) })
        })
        .collect();
    for (ds, s) in SETS.iter().zip(&sets) {
        println!("{ds}: truth REM {:.1}%, truth first-REM {:.1} min", s.truth_rem, s.truth_lat);
    }

    let show = |label: &str, p: &Params| {
        let (k, rem, rerr, lerr) = score(&sets, p);
        println!("{label:<30} {k:>6.3} {rem:>8.1} {rerr:>10.2} {lerr:>11.1}");
    };
    println!("\n{:<30} {:>6} {:>8} {:>10} {:>11}", "variant", "kappa", "REM %", "|REM err|", "|lat err|");
    show("shipped (onset guard)", &shipped);
    // The window-anchored step the shipped guard replaced, so this sweep still shows what it bought.
    show(
        "pre-F1 window step at 12%",
        &Params { cycle_rem_onset_minutes: 0.0, cycle_rem_early_penalty: 3.0, ..shipped },
    );

    println!("\n-- onset-anchored guard: K x minutes --");
    for k in [1.0f64, 2.0, 3.0, 4.0, 6.0] {
        for m0 in [30.0f64, 45.0, 60.0, 90.0, 120.0] {
            let mut p = shipped;
            p.cycle_rem_early_penalty = k;
            p.cycle_rem_onset_minutes = m0;
            show(&format!("K={k:.0} M0={m0:.0}min"), &p);
        }
    }

    println!("\n-- ceiling on the rising ramp (guard held at K=3, M0=60) --");
    for cap in [0.4f64, 0.5, 0.6, 0.7, 0.8, 1.0] {
        let mut p = shipped;
        p.cycle_rem_early_penalty = 3.0;
        p.cycle_rem_onset_minutes = 60.0;
        p.cycle_rem_ramp_cap = cap;
        show(&format!("cap={cap:.1}"), &p);
    }

    println!("\n-- candidates per cohort: kappa / REM% / |lat err| min --");
    let mut cands: Vec<(String, Params)> = vec![("shipped".into(), shipped)];
    for (k, m0, cap) in [(3.0, 60.0, 1.0), (4.0, 60.0, 1.0), (4.0, 60.0, 0.7), (3.0, 60.0, 0.7)] {
        let mut p = shipped;
        p.cycle_rem_early_penalty = k;
        p.cycle_rem_onset_minutes = m0;
        p.cycle_rem_ramp_cap = cap;
        cands.push((format!("K={k:.0} M0={m0:.0} cap={cap:.1}"), p));
    }
    print!("{:<24}", "variant");
    for ds in SETS {
        print!("{ds:>26}");
    }
    println!();
    for (label, p) in &cands {
        print!("{label:<24}");
        for s in &sets {
            let (k, rem, _, lerr) = score(std::slice::from_ref(s), p);
            print!("{k:>9.3}{rem:>8.1}{lerr:>9.1}");
        }
        println!();
    }
    print!("{:<24}", "TRUTH");
    for s in &sets {
        print!("{:>9}{:>8.1}{:>9.1}", "-", s.truth_rem, 0.0);
    }
    println!();

    println!("\n-- ramp scale at the best ceiling --");
    for scale in [0.6f64, 0.8, 1.0, 1.3, 1.6] {
        for cap in [0.5f64, 0.7, 1.0] {
            let mut p = shipped;
            p.cycle_rem_early_penalty = 3.0;
            p.cycle_rem_onset_minutes = 60.0;
            p.cycle_rem_ramp_cap = cap;
            p.cycle_rem_scale = scale;
            show(&format!("scale={scale:.1} cap={cap:.1}"), &p);
        }
    }
}
