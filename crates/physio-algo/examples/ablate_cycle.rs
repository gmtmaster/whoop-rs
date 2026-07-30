//! Ablation of the V2 sleep-cycle prior against labelled ground truth. Drops the REM ramp, the early-REM
//! step, or both, and reports kappa, per-stage recall, stage fractions and first-REM latency next to the
//! PSG truth for the same nights, so a term that moves placement without moving agreement still shows.
//!
//!   cargo run --release -p physio-algo --example ablate_cycle
//!
//! Reads the same fixture tree as `tune_v2`. Nothing is written; this only measures.
//!
//! `stage_v2` ONLY, and deliberately: the PSG cohorts carry no step stream at one sample a minute, so
//! `refine_wake`'s density gate declines on every one of them. Stated because the gate is silent.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use physio_algo::sleep::{
    params::Params, prepare_v2, stage_v2_prepared, AccelSample, HrSample, Prepared, RrRun, SleepInput,
    SleepStage,
};

const DATASETS: [&str; 5] = ["dreamt", "aauwss", "sleep-accel", "killa5", "strap"];

fn fixtures_root() -> PathBuf {
    std::env::var("WHOOP_SLEEP_FIXTURES")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("C:/Users/DavidGillot/Projects/whoop/sleep-benchmark/fixtures_multi"))
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

fn load_dataset(ds: &str) -> Vec<Night> {
    let root = fixtures_root().join(ds);
    let mut dirs: Vec<PathBuf> = fs::read_dir(&root)
        .map(|rd| rd.filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.is_dir()).collect())
        .unwrap_or_default();
    dirs.sort();
    dirs.iter().filter_map(|d| load_night(d)).collect()
}

fn stage_to_int(s: SleepStage) -> usize {
    match s {
        SleepStage::Wake => 0,
        SleepStage::Light => 1,
        SleepStage::Deep => 2,
        SleepStage::Rem => 3,
    }
}

const NAMES: [&str; 4] = ["wake", "light", "deep", "rem"];
const REM: usize = 3;

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

/// Per-epoch predicted label for every epoch of the night, from the segment tiling.
fn labels(night: &Night, prep: &Prepared, p: &Params) -> Vec<usize> {
    let segs = stage_v2_prepared(prep, p);
    (0..night.n_epochs)
        .map(|k| {
            let mid = night.w0 + k as i64 * 30 + 15;
            let s = segs
                .iter()
                .find(|s| s.start <= mid && mid < s.end)
                .map(|s| s.stage)
                .unwrap_or_else(|| segs.last().unwrap().stage);
            stage_to_int(s)
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

struct Score {
    kappa: f64,
    /// Predicted share of scored epochs per stage, and the same for the truth column.
    pred_frac: [f64; 4],
    truth_frac: [f64; 4],
    recall: [f64; 4],
    /// Median minutes from window start to the first epoch of REM, predicted and truth.
    lat_pred: f64,
    lat_truth: f64,
    /// Nights where any REM was called at all.
    rem_nights: usize,
    n_nights: usize,
}

fn score(nights: &[Night], prep: &[Prepared], p: &Params) -> Score {
    let mut cm = [[0i64; 4]; 4];
    let (mut lp, mut lt) = (Vec::new(), Vec::new());
    let mut rem_nights = 0;
    for (n, pr) in nights.iter().zip(prep) {
        let lab = labels(n, pr, p);
        for (k, &t) in &n.truth {
            if *k < n.n_epochs && (0..4).contains(&t) {
                cm[t as usize][lab[*k]] += 1;
            }
        }
        if let Some(k) = lab.iter().position(|&s| s == REM) {
            lp.push(k as f64 * 0.5);
            rem_nights += 1;
        }
        if let Some(k) = n.truth.iter().find(|(_, &t)| t == REM as i32).map(|(k, _)| *k) {
            lt.push(k as f64 * 0.5);
        }
    }
    let tot: i64 = cm.iter().flatten().sum();
    let mut pred_frac = [0.0; 4];
    let mut truth_frac = [0.0; 4];
    let mut recall = [0.0; 4];
    for i in 0..4 {
        let col: i64 = cm.iter().map(|r| r[i]).sum();
        let row: i64 = cm[i].iter().sum();
        pred_frac[i] = 100.0 * col as f64 / tot as f64;
        truth_frac[i] = 100.0 * row as f64 / tot as f64;
        recall[i] = if row == 0 { f64::NAN } else { 100.0 * cm[i][i] as f64 / row as f64 };
    }
    Score {
        kappa: cohen_kappa(&cm),
        pred_frac,
        truth_frac,
        recall,
        lat_pred: median(&mut lp),
        lat_truth: median(&mut lt),
        rem_nights,
        n_nights: nights.len(),
    }
}

fn main() {
    let shipped = Params::SHIPPED;
    let mut no_step = shipped;
    no_step.cycle_rem_early_penalty = 0.0;
    let mut no_ramp = shipped;
    no_ramp.cycle_rem_scale = 0.0;
    let mut no_prior = shipped;
    no_prior.cycle_rem_scale = 0.0;
    no_prior.cycle_rem_early_penalty = 0.0;
    // The recipe as it stood before the DREAMT re-tune, reconstructed from that commit's diff. Its
    // movement floor differs, so its features are extracted separately rather than re-labelled.
    let pre = Params {
        deep_hrv: -1.1,
        deep_hr: 0.0,
        deep_motion: -0.5,
        rem_hrv: 0.6,
        rem_motion: -0.6,
        rem_hr: 0.4,
        awake_motion: 1.0,
        awake_hrv: 0.8,
        awake_hr: 0.4,
        awake_deadzone: 0.0,
        deep_gate_thresh: 0.25,
        jerk_move_mult: 38.0,
        jerk_gate_mult: 55.0,
        motion_gate_boost: 2.0,
        base_rate: [0.18, 0.22, 0.50, 0.10],
        transition: [
            [0.86, 0.007, 0.126, 0.007],
            [0.005, 0.88, 0.10, 0.015],
            [0.06, 0.06, 0.85, 0.03],
            [0.01, 0.02, 0.27, 0.70],
        ],
        ..shipped
    };
    let mut pre_no_step = pre;
    pre_no_step.cycle_rem_early_penalty = 0.0;
    let variants: [(&str, Params); 4] = [
        ("shipped", shipped),
        ("no step (-3.0 off)", no_step),
        ("no ramp (1.0*c off)", no_ramp),
        ("no REM prior", no_prior),
    ];

    for ds in DATASETS {
        let nights = load_dataset(ds);
        if nights.is_empty() {
            println!("{ds}: no nights\n");
            continue;
        }
        let prep: Vec<Prepared> = nights.iter().map(|n| prepare_v2(&n.input, &shipped)).collect();

        // Session spans decide the guard width, since the step fires below a fraction of the span.
        let mut spans: Vec<f64> = nights.iter().map(|n| (n.input.end - n.input.start) as f64 / 3600.0).collect();
        spans.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let guard = |h: f64| h * 60.0 * shipped.cycle_rem_early_frac;
        println!("=== {ds}  n={} nights ===", nights.len());
        println!(
            "span h  min {:.2}  med {:.2}  max {:.2}   guard min  {:.1} .. {:.1}  (median {:.1}, spread {:.1}x)",
            spans[0],
            spans[spans.len() / 2],
            spans[spans.len() - 1],
            guard(spans[0]),
            guard(spans[spans.len() - 1]),
            guard(spans[spans.len() / 2]),
            spans[spans.len() - 1] / spans[0],
        );

        println!(
            "{:<21} {:>6} {:>26} {:>26} {:>10}",
            "variant", "kappa", "predicted % w/l/d/r", "recall % w/l/d/r", "1st REM"
        );
        for (name, p) in &variants {
            let s = score(&nights, &prep, p);
            println!(
                "{:<21} {:>6.3} {:>7.1}{:>6.1}{:>6.1}{:>7.1} {:>7.1}{:>6.1}{:>6.1}{:>7.1} {:>7.1} min  ({}/{} nights w/ REM)",
                name,
                s.kappa,
                s.pred_frac[0], s.pred_frac[1], s.pred_frac[2], s.pred_frac[3],
                s.recall[0], s.recall[1], s.recall[2], s.recall[3],
                s.lat_pred,
                s.rem_nights, s.n_nights,
            );
        }
        let pre_prep: Vec<Prepared> = nights.iter().map(|n| prepare_v2(&n.input, &pre)).collect();
        for (name, p) in [("pre-retune", &pre), ("pre-retune, no step", &pre_no_step)] {
            let s = score(&nights, &pre_prep, p);
            println!(
                "{:<21} {:>6.3} {:>7.1}{:>6.1}{:>6.1}{:>7.1} {:>7.1}{:>6.1}{:>6.1}{:>7.1} {:>7.1} min  ({}/{} nights w/ REM)",
                name,
                s.kappa,
                s.pred_frac[0], s.pred_frac[1], s.pred_frac[2], s.pred_frac[3],
                s.recall[0], s.recall[1], s.recall[2], s.recall[3],
                s.lat_pred,
                s.rem_nights, s.n_nights,
            );
        }
        let s = score(&nights, &prep, &shipped);
        println!(
            "{:<21} {:>6} {:>7.1}{:>6.1}{:>6.1}{:>7.1} {:>34.1} min   <- PSG",
            "TRUTH", "-", s.truth_frac[0], s.truth_frac[1], s.truth_frac[2], s.truth_frac[3], s.lat_truth,
        );

        // The step is claimed to touch only the first cycle: count where the two labellings differ.
        let (mut diff_early, mut diff_late, mut total) = (0usize, 0usize, 0usize);
        for (n, pr) in nights.iter().zip(&prep) {
            let (a, b) = (labels(n, pr, &shipped), labels(n, pr, &no_step));
            let span = (n.input.end - n.input.start) as f64;
            for k in 0..n.n_epochs {
                total += 1;
                if a[k] != b[k] {
                    let c = (k as f64 * 30.0 + 15.0) / span;
                    if c < shipped.cycle_rem_early_frac { diff_early += 1 } else { diff_late += 1 }
                }
            }
        }
        println!(
            "step on/off differs on {}/{} epochs: {} inside the guard, {} after it ({:.1}% of the change is late)\n",
            diff_early + diff_late,
            total,
            diff_early,
            diff_late,
            100.0 * diff_late as f64 / (diff_early + diff_late).max(1) as f64,
        );

        // REM share by hour into the session, shipped against truth.
        for (name, p) in [("shipped", &shipped), ("no step", &no_step)] {
            let mut hits = [0i64; 12];
            let mut seen = [0i64; 12];
            for (n, pr) in nights.iter().zip(&prep) {
                let lab = labels(n, pr, p);
                for (k, &l) in lab.iter().enumerate() {
                    let h = (k * 30 / 3600).min(11);
                    seen[h] += 1;
                    if l == REM {
                        hits[h] += 1;
                    }
                }
            }
            print!("REM %/h {:<8}", name);
            for h in 0..12 {
                if seen[h] > 0 { print!("{:>6.1}", 100.0 * hits[h] as f64 / seen[h] as f64) } else { print!("{:>6}", "-") }
            }
            println!();
        }
        let mut hits = [0i64; 12];
        let mut seen = [0i64; 12];
        for n in &nights {
            for (k, &t) in &n.truth {
                let h = (k * 30 / 3600).min(11);
                seen[h] += 1;
                if t == REM as i32 {
                    hits[h] += 1;
                }
            }
        }
        print!("REM %/h {:<8}", "TRUTH");
        for h in 0..12 {
            if seen[h] > 0 { print!("{:>6.1}", 100.0 * hits[h] as f64 / seen[h] as f64) } else { print!("{:>6}", "-") }
        }
        println!("\n");
    }
    println!("stage order: {NAMES:?}");
}
