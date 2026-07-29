//! Does the detection window change staging beyond its own edges?
//!
//!   cargo run --release -p physio-algo --example window_effect
//!
//! Stages each band-labelled night twice: over the detected window, and over the strap's own asleep run.
//! Compares only the epochs BOTH runs contain, so any disagreement there is the window moving the
//! z-score baselines and the deep percentile gate, not the trimmed edges being counted differently.

use std::fs;
use std::path::{Path, PathBuf};

use physio_algo::sleep::{
    params::Params, prepare_v2, stage_v2_prepared, AccelSample, HrSample, RrRun, SleepInput, SleepStage,
};

fn root() -> PathBuf {
    std::env::var("WHOOP_SLEEP_FIXTURES")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("C:/Users/DavidGillot/Projects/whoop/sleep-benchmark/fixtures_multi"))
        .join("ours")
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

fn idx(s: SleepStage) -> usize {
    match s {
        SleepStage::Wake => 0,
        SleepStage::Light => 1,
        SleepStage::Deep => 2,
        SleepStage::Rem => 3,
    }
}

/// Label every epoch of `[w0, w0 + n*30)` under `p`, from a staging over `input`.
fn label(input: &SleepInput, p: &Params, w0: i64, n: usize) -> Vec<usize> {
    let segs = stage_v2_prepared(&prepare_v2(input, p), p);
    (0..n)
        .map(|k| {
            let mid = w0 + k as i64 * 30 + 15;
            idx(segs
                .iter()
                .find(|s| s.start <= mid && mid < s.end)
                .map(|s| s.stage)
                .unwrap_or_else(|| segs.last().unwrap().stage))
        })
        .collect()
}

fn main() {
    let p = Params::SHIPPED;
    let mut dirs: Vec<PathBuf> = fs::read_dir(root())
        .map(|rd| rd.filter_map(|e| e.ok().map(|e| e.path())).filter(|q| q.is_dir()).collect())
        .unwrap_or_default();
    dirs.sort();

    let (mut nights, mut shared, mut moved) = (0usize, 0usize, 0usize);
    let (mut wake_full, mut wake_trim, mut any) = (0i64, 0i64, 0i64);
    let mut per_night: Vec<f64> = Vec::new();
    println!("{:<40} {:>7} {:>9} {:>10} {:>10}", "night", "shared", "changed", "wake% full", "wake% trim");
    for d in dirs {
        let Ok(meta) = fs::read_to_string(d.join("meta.txt")) else { continue };
        let m: Vec<i64> = meta.split_whitespace().map(|x| x.parse().unwrap()).collect();
        let (w0, w1, n) = (m[1], m[2], m[3] as usize);
        let truth = read_csv(&d.join("truth.csv"));
        let asleep: Vec<usize> = truth.iter().filter(|r| r[1] == 1.0).map(|r| r[0] as usize).collect();
        if asleep.len() < 240 {
            continue;
        }
        let (first, last) = (asleep[0], asleep[asleep.len() - 1]);
        if first == 0 && last + 1 >= n {
            continue; // nothing to trim
        }

        let accel: Vec<AccelSample> = read_csv(&d.join("gravity.csv"))
            .iter()
            .map(|r| AccelSample { ts: r[0] as i64, x: r[1], y: r[2], z: r[3] })
            .collect();
        let hr: Vec<HrSample> = read_csv(&d.join("hr.csv"))
            .iter()
            .map(|r| HrSample { ts: r[0] as i64, bpm: r[1] as u16 })
            .collect();
        let mut rr: Vec<RrRun> = Vec::new();
        for row in read_csv(&d.join("rr.csv")) {
            let (ts, ms) = (row[0] as i64, row[1] as u16);
            match rr.last_mut() {
                Some(l) if l.ts == ts => l.intervals.push(ms),
                _ => rr.push(RrRun { ts, intervals: vec![ms] }),
            }
        }
        let full = SleepInput { start: w0, end: w1, hr: hr.clone(), rr: rr.clone(), accel: accel.clone() };
        // The strap's own asleep run, as a window.
        let (t0, t1) = (w0 + first as i64 * 30, w0 + (last + 1) as i64 * 30);
        let trim = SleepInput { start: t0, end: t1, hr, rr, accel };

        let inner = last + 1 - first;
        let a = label(&full, &p, w0, n);
        let b = label(&trim, &p, t0, inner);
        let changed = (0..inner).filter(|k| a[first + k] != b[*k]).count();

        let wf = (0..inner).filter(|k| a[first + k] == 0).count() as i64;
        let wt = (0..inner).filter(|k| b[*k] == 0).count() as i64;
        wake_full += wf;
        wake_trim += wt;
        any += inner as i64;
        nights += 1;
        shared += inner;
        moved += changed;
        per_night.push(100.0 * changed as f64 / inner as f64);
        if changed > 0 {
            println!(
                "{:<40} {:>7} {:>8.1}% {:>9.1} {:>10.1}",
                d.file_name().unwrap().to_string_lossy().chars().take(38).collect::<String>(),
                inner,
                100.0 * changed as f64 / inner as f64,
                100.0 * wf as f64 / inner as f64,
                100.0 * wt as f64 / inner as f64,
            );
        }
    }
    per_night.sort_by(|x, y| x.partial_cmp(y).unwrap());
    println!(
        "\n{nights} nights, {shared} shared epochs: {moved} relabelled ({:.1}%), median per night {:.1}%",
        100.0 * moved as f64 / shared.max(1) as f64,
        if per_night.is_empty() { 0.0 } else { per_night[per_night.len() / 2] },
    );
    println!(
        "wake over the SHARED interior only: detected window {:.1}%, strap's asleep run {:.1}%",
        100.0 * wake_full as f64 / any.max(1) as f64,
        100.0 * wake_trim as f64 / any.max(1) as f64,
    );
}
