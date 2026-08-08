//! Where does the REM excess come from, and does it come from WAKE?
//!
//!   cargo run --release -p physio-algo --example rem_wake_confusion
//!
//! The aggregate says REM is over-called by up to +10.3 pp against PSG truth. That is compatible with
//! stealing from light, from deep, or from wake, and only the last is the "stood up to use the bathroom
//! and it scored REM" fault. This splits it: the row-normalised confusion matrix, then the same rows
//! restricted to the epochs that actually MOVED, which is where a bathroom trip lives.
//!
//! Motion per epoch is the same quantity the app stores, sum of |delta gravity| over the epoch's 1 Hz
//! samples, and it is ranked WITHIN each night so a strap's own gravity scale cannot bias the split.

mod common;

use common::{dirs_of, read_accel, read_hr, read_meta, read_rr, read_truth, stage_at, stage_idx};

use std::collections::BTreeMap;
use std::path::Path;

use physio_algo::sleep::{params::Params, prepare_v2, stage_v2_prepared, SleepInput};

const EPOCH: i64 = 30;
const NAMES: [&str; 4] = ["wake", "light", "deep", "rem"];
const COHORTS: [&str; 3] = ["aauwss", "sleep-accel", "dreamt"];

struct Night {
    input: SleepInput,
    w0: i64,
    n: usize,
    truth: BTreeMap<usize, i32>,
}

fn load(dir: &Path) -> Option<Night> {
    let (w0, w1, n) = read_meta(dir)?;
    let truth = read_truth(dir);
    if truth.is_empty() {
        return None;
    }
    Some(Night {
        input: SleepInput { start: w0, end: w1, hr: read_hr(dir), rr: read_rr(dir), accel: read_accel(dir) },
        w0,
        n,
        truth,
    })
}

/// Sum of |delta gravity| per 30 s epoch — the stored `motionJSON` quantity, recomputed from raw.
fn epoch_motion(night: &Night) -> Vec<f64> {
    let mut m = vec![0.0; night.n];
    let a = &night.input.accel;
    for w in a.windows(2) {
        let d = (w[1].x - w[0].x).abs() + (w[1].y - w[0].y).abs() + (w[1].z - w[0].z).abs();
        let k = ((w[1].ts - night.w0) / EPOCH) as usize;
        if k < m.len() {
            m[k] += d;
        }
    }
    m
}

/// Rank of each epoch's motion within its own night, as a 0..1 fraction. Within-night so one strap's
/// gravity scale cannot decide which epochs count as "moved".
fn motion_rank(m: &[f64]) -> Vec<f64> {
    let mut idx: Vec<usize> = (0..m.len()).collect();
    idx.sort_by(|a, b| m[*a].partial_cmp(&m[*b]).unwrap_or(std::cmp::Ordering::Equal));
    let mut r = vec![0.0; m.len()];
    for (rank, &i) in idx.iter().enumerate() {
        r[i] = rank as f64 / (m.len().max(2) - 1) as f64;
    }
    r
}

fn pct_row(cm: &[[i64; 4]; 4], t: usize) -> [f64; 4] {
    let tot = cm[t].iter().sum::<i64>().max(1) as f64;
    std::array::from_fn(|c| 100.0 * cm[t][c] as f64 / tot)
}

fn main() {
    let p = Params::SHIPPED;
    println!("Row-normalised: of the epochs PSG calls X, what do WE call them? SHIPPED params.\n");

    for ds in COHORTS {
        let nights: Vec<Night> = dirs_of(ds).iter().filter_map(|d| load(d)).collect();
        if nights.is_empty() {
            continue;
        }
        // A bathroom trip is a handful of epochs, not a tenth of a night, so the strata run all the way
        // up to the top 0.5% within each night.
        const CUTS: [f64; 4] = [0.0, 0.90, 0.98, 0.995];
        const CUT_NAMES: [&str; 4] = ["all", "top 10% motion", "top 2% motion", "top 0.5% motion"];
        let mut cms = [[[0i64; 4]; 4]; 4];
        let mut ns = [0i64; 4];
        for night in &nights {
            let prep = prepare_v2(&night.input, &p);
            let segs = stage_v2_prepared(&prep, &p);
            let rank = motion_rank(&epoch_motion(night));
            for (k, &t) in &night.truth {
                if *k >= night.n || !(0..4).contains(&t) {
                    continue;
                }
                let mid = night.w0 + *k as i64 * EPOCH + EPOCH / 2;
                let ours = stage_idx(stage_at(&segs, mid).unwrap_or_else(|| segs.last().unwrap().stage));
                let r = rank.get(*k).copied().unwrap_or(0.0);
                for (c, cut) in CUTS.iter().enumerate() {
                    if r >= *cut {
                        cms[c][t as usize][ours] += 1;
                        if t == 0 {
                            ns[c] += 1;
                        }
                    }
                }
            }
        }

        println!("== {ds} (n={})", nights.len());
        println!("   {:<24}{:>9}{:>9}{:>9}{:>9}", "PSG says \\ we say", "wake", "light", "deep", "rem");
        for (t, truth_name) in NAMES.iter().enumerate() {
            for (c, name) in CUT_NAMES.iter().enumerate() {
                let r = pct_row(&cms[c], t);
                let lbl =
                    if c == 0 { format!("{truth_name} ({name})") } else { format!("   {name}") };
                println!("   {:<24}{:>8.1}%{:>8.1}%{:>8.1}%{:>8.1}%", lbl, r[0], r[1], r[2], r[3]);
            }
        }
        println!("   -> of PSG WAKE, share we call REM by motion stratum:");
        for (c, name) in CUT_NAMES.iter().enumerate() {
            println!(
                "      {name:<18} REM {:>5.1}%   correct wake {:>5.1}%   (n={})",
                pct_row(&cms[c], 0)[3],
                pct_row(&cms[c], 0)[0],
                ns[c]
            );
        }

        // Which truth class the REM excess is actually made of: each row's REM rate times that row's
        // share of the night. A rate alone cannot say, because light is half the night and wake is not.
        let tot = cms[0].iter().flatten().sum::<i64>().max(1) as f64;
        println!("   -> the REM excess, decomposed by where it comes FROM:");
        let mut inflow = 0.0;
        for (t, truth_name) in NAMES.iter().enumerate().take(3) {
            let share = cms[0][t].iter().sum::<i64>() as f64 / tot;
            let rate = pct_row(&cms[0], t)[3];
            inflow += rate * share;
            println!(
                "      {truth_name:<12} -> REM  {rate:>5.1}% of a {:>4.1}% class  =  {:+5.1} pp of the night",
                100.0 * share,
                rate * share
            );
        }
        let rem_share = cms[0][3].iter().sum::<i64>() as f64 / tot;
        let outflow = (100.0 - pct_row(&cms[0], 3)[3]) * rem_share;
        println!("      REM -> elsewhere                                     {:+5.1} pp", -outflow);
        println!("      net                                                  {:+5.1} pp\n", inflow - outflow);
    }
}
