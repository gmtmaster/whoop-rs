//! Parity gate: run the ported stagers on the on-disk DREAMT + AAUWSS fixtures and reproduce the shipped
//! Cohen's-kappa within a tight tolerance. Ignored by default (the datasets live outside the repo); point
//! `WHOOP_SLEEP_FIXTURES` at a `fixtures_multi` directory, or rely on the default sibling path, then run:
//!   cargo test -p physio-algo --test dataset_parity -- --ignored --nocapture

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use physio_algo::sleep::{
    stage_v2, AccelSample, HrSample, RrRun, SleepInput, SleepStage, StageSegment,
};

fn fixtures_root() -> PathBuf {
    if let Ok(p) = std::env::var("WHOOP_SLEEP_FIXTURES") {
        return PathBuf::from(p);
    }
    PathBuf::from("C:/Users/DavidGillot/Projects/whoop/sleep-benchmark/fixtures_multi")
}

fn read_csv(path: &Path) -> Vec<Vec<f64>> {
    let text = match fs::read_to_string(path) {
        Ok(t) => t,
        Err(_) => return Vec::new(),
    };
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.split(',').map(|c| c.trim().parse::<f64>().unwrap()).collect())
        .collect()
}

struct Fixture {
    input: SleepInput,
    w0: i64,
    n_epochs: usize,
    truth: BTreeMap<usize, i32>,
}

fn load_fixture(dir: &Path) -> Fixture {
    let meta = fs::read_to_string(dir.join("meta.txt")).unwrap();
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

    // Group consecutive same-timestamp R-R rows into one run (col 0 = ts, col 1 = rr_ms).
    let mut rr: Vec<RrRun> = Vec::new();
    for row in read_csv(&dir.join("rr.csv")) {
        let ts = row[0] as i64;
        let ms = row[1] as u16;
        match rr.last_mut() {
            Some(last) if last.ts == ts => last.intervals.push(ms),
            _ => rr.push(RrRun { ts, intervals: vec![ms] }),
        }
    }

    let mut truth = BTreeMap::new();
    for row in read_csv(&dir.join("truth.csv")) {
        truth.insert(row[0] as usize, row[1] as i32);
    }

    Fixture {
        input: SleepInput { start: w0, end: w1, hr, rr, accel, resp: Vec::new() },
        w0,
        n_epochs,
        truth,
    }
}

fn stage_to_int(s: SleepStage) -> i32 {
    match s {
        SleepStage::Wake => 0,
        SleepStage::Light => 1,
        SleepStage::Deep => 2,
        SleepStage::Rem => 3,
    }
}

/// Probe each labelled epoch's midpoint against the tiled segments (the harness rule).
fn predict_epochs(segs: &[StageSegment], w0: i64, n_epochs: usize) -> Vec<i32> {
    (0..n_epochs)
        .map(|k| {
            let mid = w0 + k as i64 * 30 + 15;
            let stage = segs
                .iter()
                .find(|s| s.start <= mid && mid < s.end)
                .map(|s| s.stage)
                .unwrap_or(segs.last().unwrap().stage);
            stage_to_int(stage)
        })
        .collect()
}

fn cohen_kappa(cm: &[[i64; 4]; 4]) -> f64 {
    let tot: i64 = cm.iter().flatten().sum();
    if tot == 0 {
        return 0.0;
    }
    let tot = tot as f64;
    let trace: i64 = (0..4).map(|i| cm[i][i]).sum();
    let a4 = trace as f64 / tot;
    let mut pe = 0.0;
    for (j, row_j) in cm.iter().enumerate() {
        let col: i64 = cm.iter().map(|r| r[j]).sum();
        let row: i64 = row_j.iter().sum();
        pe += (col as f64) * (row as f64);
    }
    pe /= tot * tot;
    if pe >= 1.0 {
        0.0
    } else {
        (a4 - pe) / (1.0 - pe)
    }
}

fn run_dataset(ds: &str) -> (f64, usize) {
    let root = fixtures_root().join(ds);
    let mut dirs: Vec<PathBuf> = fs::read_dir(&root)
        .unwrap_or_else(|_| panic!("dataset dir missing: {}", root.display()))
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_dir())
        .collect();
    dirs.sort();

    let mut cm = [[0i64; 4]; 4];
    let mut subjects = 0;
    for dir in &dirs {
        if !dir.join("meta.txt").exists() {
            continue;
        }
        let fx = load_fixture(dir);
        let segs = stage_v2(&fx.input);
        let pred = predict_epochs(&segs, fx.w0, fx.n_epochs);
        for (k, &t) in &fx.truth {
            if *k < pred.len() && (0..4).contains(&t) {
                cm[t as usize][pred[*k] as usize] += 1;
            }
        }
        subjects += 1;
    }
    (cohen_kappa(&cm), subjects)
}

#[test]
#[ignore = "reads external fixtures; run with --ignored for the parity gate"]
fn v2_dreamt_kappa_matches_shipped() {
    let (kappa, n) = run_dataset("dreamt");
    println!("V2 DREAMT: kappa={kappa:.3} over {n} subjects (target ~0.311)");
    assert!((kappa - 0.311).abs() < 0.008, "V2 DREAMT kappa {kappa:.3} off target 0.311");
}

#[test]
#[ignore = "reads external fixtures; run with --ignored for the parity gate"]
fn v2_aauwss_kappa_matches_shipped() {
    let (kappa, n) = run_dataset("aauwss");
    println!("V2 AAUWSS: kappa={kappa:.3} over {n} subjects (target ~0.412)");
    assert!((kappa - 0.412).abs() < 0.008, "V2 AAUWSS kappa {kappa:.3} off target 0.412");
}
