//! Check the onset-anchored REM guard and the statistics offered for it, against PSG truth and against
//! our own straps. Reports the units claim (is true latency correlated with session length, is it less
//! variable in minutes than as a fraction), the cost of a guard at each width, and the guard's effect on
//! kappa, stage fractions and first-REM latency under both the shipped and the pre-retune recipe.
//!
//!   cargo run --release -p physio-algo --example verify_934
//!
//! `stage_v2` ONLY. The guard is a REM-placement term and the refinement rewrites wake to light, so it
//! can move no first-REM latency; the PSG cohorts have no step stream for it either way.

mod common;

use common::{median_avg, read_csv, stage_idx};

use common::kappa4 as kappa;

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use physio_algo::sleep::{
    params::Params, prepare_v2, stage_v2_prepared, AccelSample, HrSample, Prepared, RrRun, SleepInput,
};

const PSG: [&str; 3] = ["sleep-accel", "dreamt", "aauwss"];
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

fn load_night(dir: &Path, need_truth: bool) -> Option<Night> {
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
    if need_truth && truth.is_empty() {
        return None;
    }
    Some(Night { input: SleepInput { start: w0, end: w1, hr, rr, accel }, w0, n_epochs, truth })
}

fn load_dataset(ds: &str, need_truth: bool) -> Vec<Night> {
    let mut dirs: Vec<PathBuf> = fs::read_dir(fixtures_root().join(ds))
        .map(|rd| rd.filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.is_dir()).collect())
        .unwrap_or_default();
    dirs.sort();
    dirs.iter().filter_map(|d| load_night(d, need_truth)).collect()
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

/// The first epoch of the earliest ten-epoch run without a wake label, mirroring the staging rule.
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

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}

fn sd(v: &[f64]) -> f64 {
    let m = mean(v);
    (v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (v.len() as f64 - 1.0)).sqrt()
}

fn pearson(a: &[f64], b: &[f64]) -> f64 {
    let (ma, mb) = (mean(a), mean(b));
    let num: f64 = a.iter().zip(b).map(|(x, y)| (x - ma) * (y - mb)).sum();
    let da: f64 = a.iter().map(|x| (x - ma) * (x - ma)).sum::<f64>().sqrt();
    let db: f64 = b.iter().map(|y| (y - mb) * (y - mb)).sum::<f64>().sqrt();
    num / (da * db)
}

fn rank(v: &[f64]) -> Vec<f64> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&i, &j| v[i].partial_cmp(&v[j]).unwrap());
    let mut r = vec![0.0; v.len()];
    let mut i = 0;
    while i < idx.len() {
        let mut j = i;
        while j + 1 < idx.len() && v[idx[j + 1]] == v[idx[i]] {
            j += 1;
        }
        let avg = (i + j) as f64 / 2.0 + 1.0;
        for &k in &idx[i..=j] {
            r[k] = avg;
        }
        i = j + 1;
    }
    r
}

/// The units argument, from the PSG column alone: does true first-REM latency track session length,
/// and is it steadier measured in minutes or as a share of the night.
fn units_claim(ds: &str, nights: &[Night]) {
    let (mut lat_min, mut lat_frac, mut spans) = (Vec::new(), Vec::new(), Vec::new());
    for n in nights {
        let span_min = n.n_epochs as f64 * 0.5;
        // Latency from the truth column's own onset, so detection error cannot enter it.
        let seq: Vec<usize> = (0..n.n_epochs).map(|k| *n.truth.get(&k).unwrap_or(&0) as usize).collect();
        let Some(on) = onset_of(&seq) else { continue };
        let Some(first) = (on..n.n_epochs).find(|k| seq[*k] == REM) else { continue };
        lat_min.push((first - on) as f64 * 0.5);
        lat_frac.push((first - on) as f64 * 0.5 / span_min);
        spans.push(span_min);
    }
    if lat_min.len() < 3 {
        println!("{ds}: too few scorable first-REM periods");
        return;
    }
    let rl = rank(&lat_min);
    let rs = rank(&spans);
    println!(
        "{:<12} n={:<4} r(latency,span) pearson {:+.3} spearman {:+.3}   CV minutes {:.3} vs fraction {:.3}   \
         median latency {:.1} min, span {:.0} min",
        ds, lat_min.len(),
        pearson(&lat_min, &spans), pearson(&rl, &rs),
        sd(&lat_min) / mean(&lat_min), sd(&lat_frac) / mean(&lat_frac),
        median_avg(&mut lat_min.clone()), median_avg(&mut spans.clone()),
    );
}

/// What a guard of each width would cover: the share of all real REM inside it, and how many nights have
/// any REM there at all.
fn guard_cost(ds: &str, nights: &[Night]) {
    print!("{ds:<12} REM inside a guard of ");
    for width in [45.0f64, 60.0, 90.0] {
        let (mut inside, mut total, mut hit) = (0i64, 0i64, 0usize);
        for n in nights {
            let seq: Vec<usize> = (0..n.n_epochs).map(|k| *n.truth.get(&k).unwrap_or(&0) as usize).collect();
            let Some(on) = onset_of(&seq) else { continue };
            let mut any = false;
            for (k, &l) in seq.iter().enumerate().skip(on) {
                if n.truth.contains_key(&k) && l == REM {
                    total += 1;
                    if (k - on) as f64 * 0.5 < width {
                        inside += 1;
                        any = true;
                    }
                }
            }
            hit += any as usize;
        }
        print!("{:.0}m: {:.2}% ({}/{} nights)   ", width, 100.0 * inside as f64 / total.max(1) as f64, hit, nights.len());
    }
    println!();
}

struct Row {
    kappa: f64,
    frac: [f64; 4],
    lat_from_onset: f64,
    lat_min_of_nights: f64,
    changed: usize,
}

fn score(nights: &[Night], p: &Params, base: Option<&Vec<Vec<usize>>>) -> (Row, Vec<Vec<usize>>) {
    let mut cm = [[0i64; 4]; 4];
    let mut frac = [0i64; 4];
    let mut lat = Vec::new();
    let mut all = Vec::new();
    let mut changed = 0;
    for (i, n) in nights.iter().enumerate() {
        let prep = prepare_v2(&n.input, p);
        let lab = labels(n, &prep, p);
        for (k, &t) in &n.truth {
            if *k < n.n_epochs && (0..4).contains(&t) {
                cm[t as usize][lab[*k]] += 1;
            }
        }
        for &l in &lab {
            frac[l] += 1;
        }
        if let Some(on) = onset_of(&lab) {
            if let Some(first) = (on..n.n_epochs).find(|k| lab[*k] == REM) {
                lat.push((first - on) as f64 * 0.5);
            }
        }
        if let Some(b) = base {
            if b[i] != lab {
                changed += 1;
            }
        }
        all.push(lab);
    }
    let tot: i64 = frac.iter().sum();
    let mut f = [0.0; 4];
    for i in 0..4 {
        f[i] = 100.0 * frac[i] as f64 / tot as f64;
    }
    let lo = lat.iter().cloned().fold(f64::INFINITY, f64::min);
    (
        Row { kappa: kappa(&cm), frac: f, lat_from_onset: median_avg(&mut lat), lat_min_of_nights: lo, changed },
        all,
    )
}

fn main() {
    let shipped = Params::SHIPPED;
    let pre = common::pre_retune(&shipped);
    let mut shipped_no_step = shipped;
    shipped_no_step.cycle_rem_early_penalty = 0.0;
    let mut pre_no_step = pre;
    pre_no_step.cycle_rem_early_penalty = 0.0;
    // The window-anchored step the shipped guard replaced. Kept as an explicit config, not as the
    // default, so the evidence for the change outlives the change.
    let shipped_pre_guard = Params { cycle_rem_onset_minutes: 0.0, cycle_rem_early_penalty: 3.0, ..shipped };
    let pre_pre_guard = Params { cycle_rem_onset_minutes: 0.0, cycle_rem_early_penalty: 3.0, ..pre };

    println!("--- the units claim, from PSG truth only ---");
    for ds in PSG {
        units_claim(ds, &load_dataset(ds, true));
    }
    println!("\n--- what a guard of each width would suppress, from PSG truth only ---");
    for ds in PSG {
        guard_cost(ds, &load_dataset(ds, true));
    }

    for ds in PSG.iter().chain(["ours"].iter()) {
        let nights = load_dataset(ds, *ds != "ours");
        if nights.is_empty() {
            continue;
        }
        let mut spans: Vec<f64> = nights.iter().map(|n| n.n_epochs as f64 / 120.0).collect();
        spans.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!(
            "\n=== {ds}  n={}  span {:.2}-{:.2} h ({:.1}x) ===",
            nights.len(), spans[0], spans[spans.len() - 1], spans[spans.len() - 1] / spans[0]
        );
        println!("{:<22} {:>6} {:>26} {:>12} {:>10} {:>8}", "recipe", "kappa", "% w/l/d/r", "1stREM med", "min", "changed");
        for (label, base_p, alts) in [
            ("shipped", &shipped, [("shipped, no guard", &shipped_no_step), ("shipped, pre-F1 window step", &shipped_pre_guard)]),
            ("pre-retune", &pre, [("pre-retune, no guard", &pre_no_step), ("pre-retune, window step", &pre_pre_guard)]),
        ] {
            let (r, base) = score(&nights, base_p, None);
            println!(
                "{:<22} {:>6.3} {:>6.1}{:>6.1}{:>6.1}{:>7.1} {:>9.1} min {:>7.1} {:>8}",
                label, r.kappa, r.frac[0], r.frac[1], r.frac[2], r.frac[3], r.lat_from_onset, r.lat_min_of_nights, "-"
            );
            for (n2, p2) in alts {
                let (r2, _) = score(&nights, p2, Some(&base));
                println!(
                    "{:<22} {:>6.3} {:>6.1}{:>6.1}{:>6.1}{:>7.1} {:>9.1} min {:>7.1} {:>7}/{}",
                    n2, r2.kappa, r2.frac[0], r2.frac[1], r2.frac[2], r2.frac[3], r2.lat_from_onset,
                    r2.lat_min_of_nights, r2.changed, nights.len()
                );
            }
        }
        if !nights[0].truth.is_empty() {
            let (t, mut lat) = (nights.iter().flat_map(|n| n.truth.values().copied()).collect::<Vec<_>>(), {
                let mut v = Vec::new();
                for n in &nights {
                    let seq: Vec<usize> = (0..n.n_epochs).map(|k| *n.truth.get(&k).unwrap_or(&0) as usize).collect();
                    if let Some(on) = onset_of(&seq) {
                        if let Some(f) = (on..n.n_epochs).find(|k| seq[*k] == REM) {
                            v.push((f - on) as f64 * 0.5);
                        }
                    }
                }
                v
            });
            let tot = t.len() as f64;
            let pc = |s: i32| 100.0 * t.iter().filter(|&&x| x == s).count() as f64 / tot;
            println!(
                "{:<22} {:>6} {:>6.1}{:>6.1}{:>6.1}{:>7.1} {:>9.1} min   <- truth",
                "TRUTH", "-", pc(0), pc(1), pc(2), pc(3), median_avg(&mut lat)
            );
        }
    }
}
