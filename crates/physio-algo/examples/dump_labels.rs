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

mod common;

use common::{read_accel, read_hr, read_meta, read_rr, root};

use std::fs;
use std::path::PathBuf;

use physio_algo::sleep::{params::Params, prepare_v2, stage_v2_prepared, SleepInput, SleepStage};

fn main() {
    let shipped = Params::SHIPPED;
    let pre = common::pre_retune(&shipped);
    let mut capped = shipped;
    capped.cycle_rem_ramp_cap = 0.7;

    let mut dirs: Vec<PathBuf> = fs::read_dir(root("ours"))
        .map(|rd| rd.filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.is_dir()).collect())
        .unwrap_or_default();
    dirs.sort();
    println!("sid,recipe,epochs,labels");
    for d in dirs {
        let Some((w0, w1, n)) = read_meta(&d) else { continue };
        let accel = read_accel(&d);
        let hr = read_hr(&d);
        let rr = read_rr(&d);
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
