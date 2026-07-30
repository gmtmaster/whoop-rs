//! Write per-epoch staged labels for the real-strap set, so a comparison against an external label can
//! align windows itself instead of taking our detection window as given.
//!
//!   cargo run --release -p physio-algo --example dump_labels > labels.csv
//!
//! One row per night: fixture id, recipe, then one character per epoch (w/l/d/r).
//!
//! `stage_v2` ONLY, and it is the right path for a dump off THIS set: `ours` gravity is held forward
//! across dropouts, so the refinement's posture check reads artificially stable here and its wake would
//! be optimistic. An aligner wanting the app's wake labels needs `continuous` and the density gate.

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

fn main() {
    let shipped = Params::SHIPPED;
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
    let mut capped = shipped;
    capped.cycle_rem_ramp_cap = 0.7;

    let mut dirs: Vec<PathBuf> = fs::read_dir(root())
        .map(|rd| rd.filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.is_dir()).collect())
        .unwrap_or_default();
    dirs.sort();
    println!("sid,recipe,epochs,labels");
    for d in dirs {
        let Ok(meta) = fs::read_to_string(d.join("meta.txt")) else { continue };
        let m: Vec<i64> = meta.split_whitespace().map(|x| x.parse().unwrap()).collect();
        let (w0, w1, n) = (m[1], m[2], m[3] as usize);
        let accel = read_csv(&d.join("gravity.csv"))
            .iter()
            .map(|r| AccelSample { ts: r[0] as i64, x: r[1], y: r[2], z: r[3] })
            .collect();
        let hr = read_csv(&d.join("hr.csv"))
            .iter()
            .map(|r| HrSample { ts: r[0] as i64, bpm: r[1] as u16 })
            .collect();
        let mut rr: Vec<RrRun> = Vec::new();
        for row in read_csv(&d.join("rr.csv")) {
            let (ts, ms) = (row[0] as i64, row[1] as u16);
            match rr.last_mut() {
                Some(last) if last.ts == ts => last.intervals.push(ms),
                _ => rr.push(RrRun { ts, intervals: vec![ms] }),
            }
        }
        let input = SleepInput { start: w0, end: w1, hr, rr, accel };
        let sid = d.file_name().unwrap().to_string_lossy().to_string();
        for (name, p) in [("shipped", &shipped), ("pre", &pre), ("cap07", &capped)] {
            let prep = prepare_v2(&input, p);
            let segs = stage_v2_prepared(&prep, p);
            let s: String = (0..n)
                .map(|k| {
                    let mid = w0 + k as i64 * 30 + 15;
                    match segs.iter().find(|s| s.start <= mid && mid < s.end).map(|s| s.stage) {
                        Some(SleepStage::Wake) | None => 'w',
                        Some(SleepStage::Light) => 'l',
                        Some(SleepStage::Deep) => 'd',
                        Some(SleepStage::Rem) => 'r',
                    }
                })
                .collect();
            println!("{sid},{name},{n},{s}");
        }
    }
}
