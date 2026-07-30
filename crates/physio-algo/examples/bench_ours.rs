//! Score the V2 stager against the strap's own on-board sleep_state over every night in our backups.
//!
//!   cargo run --release -p physio-algo --example bench_ours
//!
//! The band label is two-class (wake vs asleep), so it judges the wake fraction and nothing finer. That
//! is the quantity the field regression reports turn on, and it is the one label here that noop did not
//! produce itself. Nights without band coverage still contribute stage fractions, which need no label.
//!
//! `stage_v2` ONLY. `ours` gravity is held forward across dropouts, so the refinement's posture check
//! reads artificially stable on this set and its wake fraction cannot be trusted here; the refined
//! figures come from `continuous` in `emit_wake`.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use physio_algo::sleep::{
    params::Params, prepare_v2, stage_v2_prepared, AccelSample, HrSample, Prepared, RrRun, SleepInput,
    SleepStage,
};

fn root() -> PathBuf {
    std::env::var("WHOOP_SLEEP_FIXTURES")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("C:/Users/DavidGillot/Projects/whoop/sleep-benchmark/fixtures_multi"))
        .join("ours")
}

struct Night {
    owner: String,
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

fn load(dir: &Path) -> Option<Night> {
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
    let owner = dir.file_name()?.to_string_lossy().split('_').next()?.to_string();
    Some(Night { owner, input: SleepInput { start: w0, end: w1, hr, rr, accel }, w0, n_epochs, truth })
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

fn median(v: &mut [f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 1 { v[n / 2] } else { (v[n / 2 - 1] + v[n / 2]) / 2.0 }
}

fn main() {
    let mut dirs: Vec<PathBuf> = fs::read_dir(root())
        .map(|rd| rd.filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.is_dir()).collect())
        .unwrap_or_default();
    dirs.sort();
    let nights: Vec<Night> = dirs.iter().filter_map(|d| load(d)).collect();
    if nights.is_empty() {
        println!("no nights under {}", root().display());
        return;
    }

    let shipped = Params::SHIPPED;
    let mut no_guard = shipped;
    no_guard.cycle_rem_early_penalty = 0.0;
    let mut no_ramp = shipped;
    no_ramp.cycle_rem_scale = 0.0;
    let pre = Params {
        deep_hrv: -1.1, deep_hr: 0.0, deep_motion: -0.5,
        rem_hrv: 0.6, rem_motion: -0.6, rem_hr: 0.4,
        awake_motion: 1.0, awake_hrv: 0.8, awake_hr: 0.4, awake_deadzone: 0.0,
        deep_gate_thresh: 0.25, jerk_move_mult: 38.0, jerk_gate_mult: 55.0, motion_gate_boost: 2.0,
        base_rate: [0.18, 0.22, 0.50, 0.10],
        transition: [
            [0.86, 0.007, 0.126, 0.007],
            [0.005, 0.88, 0.10, 0.015],
            [0.06, 0.06, 0.85, 0.03],
            [0.01, 0.02, 0.27, 0.70],
        ],
        ..shipped
    };

    let labelled: usize = nights.iter().filter(|n| !n.truth.is_empty()).count();
    let hours: f64 = nights.iter().map(|n| n.n_epochs as f64 / 120.0).sum();
    println!("{} nights, {:.1} h, {} band-labelled\n", nights.len(), hours, labelled);

    let mut owners: Vec<String> = nights.iter().map(|n| n.owner.clone()).collect();
    owners.sort();
    owners.dedup();

    println!(
        "{:<14} {:>26} {:>10} {:>26} {:>9}",
        "variant", "predicted % w/l/d/r", "1st REM", "on labelled: pred/band wake%", "rec/prec"
    );
    for (name, p) in [("shipped", &shipped), ("no guard", &no_guard), ("no ramp", &no_ramp), ("pre-retune", &pre)] {
        let prep: Vec<Prepared> = nights.iter().map(|n| prepare_v2(&n.input, p)).collect();
        let mut frac = [0i64; 4];
        // A second count over only the band-labelled epochs, so predicted and band wake share a denominator.
        let mut frac_lab = [0i64; 4];
        let mut lat = Vec::new();
        // Two-class agreement: the band calls asleep or awake, so every sleep stage folds to asleep.
        let (mut tot, mut wake_hit, mut wake_true, mut wake_pred) = (0i64, 0i64, 0i64, 0i64);
        for (n, pr) in nights.iter().zip(&prep) {
            let lab = labels(n, pr, p);
            for &l in &lab {
                frac[l] += 1;
            }
            if let Some(k) = lab.iter().position(|&s| s == 3) {
                lat.push(k as f64 * 0.5);
            }
            for (k, &t) in &n.truth {
                if *k >= n.n_epochs {
                    continue;
                }
                frac_lab[lab[*k]] += 1;
                let pred_wake = lab[*k] == 0;
                let true_wake = t == 0;
                tot += 1;
                wake_true += true_wake as i64;
                wake_pred += pred_wake as i64;
                wake_hit += (pred_wake && true_wake) as i64;
            }
        }
        let all: i64 = frac.iter().sum();
        let all_lab: i64 = frac_lab.iter().sum();
        let pct = |v: i64| 100.0 * v as f64 / all as f64;
        println!(
            "{:<14} {:>6.1}{:>6.1}{:>6.1}{:>7.1} {:>7.1} min {:>10.1}{:>9.1}{:>9.1} {:>9.1}",
            name,
            pct(frac[0]), pct(frac[1]), pct(frac[2]), pct(frac[3]),
            median(&mut lat),
            100.0 * frac_lab[0] as f64 / all_lab as f64,
            100.0 * wake_true as f64 / tot as f64,
            100.0 * wake_hit as f64 / wake_true.max(1) as f64,
            100.0 * wake_hit as f64 / wake_pred.max(1) as f64,
        );
    }

    // WHOOP's own labels shift with night length, so the same trend is a check the priors can fail.
    println!("\nstage % by time in bed, as shipped / pre-retune / with ramp cap 0.7");
    println!("{:<10} {:>4}   {:^24}   {:^24}   {:^24}", "band", "n", "light", "rem", "awake");
    let mut capped = shipped;
    capped.cycle_rem_ramp_cap = 0.7;
    for (lo, hi, lab) in [(0.0, 6.0, "<6 h"), (6.0, 7.0, "6-7 h"), (7.0, 8.0, "7-8 h"), (8.0, 10.0, "8-10 h"), (10.0, 99.0, "10+ h")] {
        let sub: Vec<&Night> = nights
            .iter()
            .filter(|n| {
                let h = n.n_epochs as f64 / 120.0;
                h >= lo && h < hi
            })
            .collect();
        if sub.len() < 4 {
            continue;
        }
        let mut cells = [[0.0f64; 3]; 3];
        for (vi, p) in [&shipped, &pre, &capped].iter().enumerate() {
            let mut frac = [0i64; 4];
            for n in &sub {
                let pr = prepare_v2(&n.input, p);
                for &l in &labels(n, &pr, p) {
                    frac[l] += 1;
                }
            }
            let all: f64 = frac.iter().sum::<i64>() as f64;
            cells[vi] = [100.0 * frac[1] as f64 / all, 100.0 * frac[3] as f64 / all, 100.0 * frac[0] as f64 / all];
        }
        print!("{lab:<10} {:>4}", sub.len());
        for (i, _) in ["light", "rem", "awake"].iter().enumerate() {
            print!("   {:>7.1}{:>8.1}{:>8.1}", cells[0][i], cells[1][i], cells[2][i]);
        }
        println!();
    }

    println!("\nwake % per person (predicted, all nights; band = the strap's own, labelled nights only)");
    println!("{:<16} {:>7} {:>9} {:>9} {:>11} {:>9}", "owner", "nights", "shipped", "pre-retune", "band", "labelled");
    for o in &owners {
        let sub: Vec<&Night> = nights.iter().filter(|n| &n.owner == o).collect();
        let mut cells = Vec::new();
        for p in [&shipped, &pre] {
            let (mut wake, mut all) = (0i64, 0i64);
            for n in &sub {
                let pr = prepare_v2(&n.input, p);
                for &l in &labels(n, &pr, p) {
                    all += 1;
                    wake += (l == 0) as i64;
                }
            }
            cells.push(100.0 * wake as f64 / all as f64);
        }
        let (bw, bt): (i64, i64) = sub.iter().fold((0, 0), |(w, t), n| {
            (w + n.truth.values().filter(|&&v| v == 0).count() as i64, t + n.truth.len() as i64)
        });
        let lab_nights = sub.iter().filter(|n| !n.truth.is_empty()).count();
        let band = if bt > 0 { format!("{:.1}", 100.0 * bw as f64 / bt as f64) } else { "-".into() };
        println!("{:<16} {:>7} {:>9.1} {:>9.1} {:>11} {:>9}", o, sub.len(), cells[0], cells[1], band, lab_nights);
    }
}
