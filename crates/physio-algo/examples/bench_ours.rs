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

mod common;

use common::{dirs_of, median_avg, read_accel, read_hr, read_meta, read_rr, read_truth, root, stage_idx};

use std::collections::BTreeMap;
use std::path::Path;

use physio_algo::sleep::{params::Params, prepare_v2, stage_v2_prepared, Prepared, SleepInput};

struct Night {
    owner: String,
    input: SleepInput,
    w0: i64,
    n_epochs: usize,
    truth: BTreeMap<usize, i32>,
}

fn load(dir: &Path) -> Option<Night> {
    let (w0, w1, n_epochs) = read_meta(dir)?;
    let accel = read_accel(dir);
    let hr = read_hr(dir);
    let rr = read_rr(dir);
    let truth = read_truth(dir);
    let owner = dir.file_name()?.to_string_lossy().split('_').next()?.to_string();
    Some(Night { owner, input: SleepInput { start: w0, end: w1, hr, rr, accel }, w0, n_epochs, truth })
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

fn main() {
    let dirs = dirs_of("ours");
    let nights: Vec<Night> = dirs.iter().filter_map(|d| load(d)).collect();
    if nights.is_empty() {
        println!("no nights under {}", root("ours").display());
        return;
    }

    let shipped = Params::SHIPPED;
    let mut no_guard = shipped;
    no_guard.cycle_rem_early_penalty = 0.0;
    // The window-anchored step the shipped onset guard replaced. Kept as an explicit config so the
    // first-REM reading the change was measured against still has a row here.
    let pre_guard = Params { cycle_rem_onset_minutes: 0.0, cycle_rem_early_penalty: 3.0, ..shipped };
    let mut no_ramp = shipped;
    no_ramp.cycle_rem_scale = 0.0;
    let pre = common::pre_retune(&shipped);

    let labelled: usize = nights.iter().filter(|n| !n.truth.is_empty()).count();
    let hours: f64 = nights.iter().map(|n| n.n_epochs as f64 / 120.0).sum();
    println!("{} nights, {:.1} h, {} band-labelled\n", nights.len(), hours, labelled);

    let mut owners: Vec<String> = nights.iter().map(|n| n.owner.clone()).collect();
    owners.sort();
    owners.dedup();

    println!(
        "{:<19} {:>26} {:>10} {:>26} {:>9}",
        "variant", "predicted % w/l/d/r", "1st REM", "on labelled: pred/band wake%", "rec/prec"
    );
    for (name, p) in [
        ("shipped", &shipped),
        ("pre-F1 window step", &pre_guard),
        ("no guard", &no_guard),
        ("no ramp", &no_ramp),
        ("pre-retune", &pre),
    ] {
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
            "{:<19} {:>6.1}{:>6.1}{:>6.1}{:>7.1} {:>7.1} min {:>10.1}{:>9.1}{:>9.1} {:>9.1}",
            name,
            pct(frac[0]), pct(frac[1]), pct(frac[2]), pct(frac[3]),
            median_avg(&mut lat),
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
