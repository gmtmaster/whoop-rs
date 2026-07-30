//! End-to-end scoring of the sleep flow over continuous wear blocks, with the strap's own sleep_state
//! as the reference at every stage. Runs the whole shipped chain: detect -> window -> stage -> refine.
//!
//!   cargo run --release -p physio-algo --example flow_az
//!
//! Stage A/B  detection: does a span exist for each run the strap called asleep, and where are its edges
//! Stage C    selection: is one strap run split across spans, or one span covering several runs
//! Stage E-I  staging: the wake fraction inside the spans, against the strap's wake/asleep call
//!
//! Blocks are cut on wear, not on labels, so nothing here tells the detector where to look.
//!
//! The wake row is printed on BOTH paths: `stage_v2` alone, which is what every A-Z number before the
//! refinement was exported was measured on, and `stage_v2` + `refine_wake`, which is what `analyze`
//! runs. The refined figure carries its own census, so a span the density gate declined cannot be
//! pooled into it.

mod common;

use common::{asleep_runs, median, pct, read_csv, root, RefineCensus};

use std::fs;
use std::path::{Path, PathBuf};

use physio_algo::sleep::{
    detect_sessions, params::Params, prepare_v2, stage_v2_prepared, AccelSample, HrSample, RrRun,
    SleepInput, SleepStage, StepSample,
};

/// One row of `sessions.csv`: the window the app kept, and whether the user set it by hand. A
/// hand-corrected window is frozen against re-detection by design, so scoring it against the strap
/// measures the USER, not persistence.
struct Stored {
    start: i64,
    end: i64,
    edited: bool,
}

struct Block {
    stored: Vec<Stored>,
    hr: Vec<HrSample>,
    rr: Vec<RrRun>,
    accel: Vec<AccelSample>,
    steps: Vec<StepSample>,
    band: Vec<(i64, i32)>,
}

fn load(dir: &Path) -> Option<Block> {
    let stored = read_csv(&dir.join("sessions.csv"))
        .iter()
        .map(|r| Stored { start: r[0] as i64, end: r[1] as i64, edited: r.get(2).copied().unwrap_or(0.0) != 0.0 })
        .collect();
    let hr = read_csv(&dir.join("hr.csv"))
        .iter()
        .map(|r| HrSample { ts: r[0] as i64, bpm: r[1] as u16 })
        .collect();
    let accel = read_csv(&dir.join("gravity.csv"))
        .iter()
        .map(|r| AccelSample { ts: r[0] as i64, x: r[1], y: r[2], z: r[3] })
        .collect();
    let mut rr: Vec<RrRun> = Vec::new();
    for row in read_csv(&dir.join("rr.csv")) {
        let (ts, ms) = (row[0] as i64, row[1] as u16);
        match rr.last_mut() {
            Some(l) if l.ts == ts => l.intervals.push(ms),
            _ => rr.push(RrRun { ts, intervals: vec![ms] }),
        }
    }
    let band = read_csv(&dir.join("band.csv")).iter().map(|r| (r[0] as i64, r[1] as i32)).collect();
    // `-1` in column 3 means the activity class was not decoded, which is not "still".
    let steps = read_csv(&dir.join("steps.csv"))
        .iter()
        .map(|r| StepSample { ts: r[0] as i64, counter: r[1] as u16, activity_class: (r[2] >= 0.0).then(|| r[2] as u8) })
        .collect();
    Some(Block { stored, hr, rr, accel, steps, band })
}

fn main() {
    let params = Params::SHIPPED;
    let mut dirs: Vec<PathBuf> = fs::read_dir(root("continuous"))
        .map(|rd| rd.filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.is_dir()).collect())
        .unwrap_or_default();
    dirs.sort();

    let (mut runs_total, mut runs_hit, mut split, mut merged, mut spurious) = (0, 0, 0, 0, 0);
    let (mut naps, mut head_bridged, mut unjudged) = (0, 0, 0);
    let (mut nap_min, mut no_sleep_min): (Vec<f64>, Vec<f64>) = (Vec::new(), Vec::new());
    let (mut head, mut tail) = (Vec::new(), Vec::new());
    let (mut stored_tail, mut fresh_tail, mut edited_tail) = (Vec::new(), Vec::new(), Vec::new());
    let (mut pred_wake, mut pred_all) = (0i64, 0i64);
    let (mut ref_wake, mut ref_all) = (0i64, 0i64);
    let (mut band_wake, mut band_all) = (0i64, 0i64);
    let mut census = RefineCensus::default();
    let mut blocks = 0;

    for d in &dirs {
        let Some(b) = load(d) else { continue };
        if b.band.is_empty() || b.accel.len() < 120 {
            continue;
        }
        blocks += 1;
        let spans = detect_sessions(&b.hr, &b.accel, 0, &[], &b.band, None);
        let runs = asleep_runs(&b.band, 90, 300);
        runs_total += runs.len();

        for &(a, z) in &runs {
            let cov: Vec<_> = spans.iter().filter(|s| s.start < z && s.end > a).collect();
            if cov.is_empty() {
                continue;
            }
            runs_hit += 1;
            if cov.len() > 1 {
                split += 1;
            }
            let (s0, s1) = (cov.iter().map(|s| s.start).min().unwrap(), cov.iter().map(|s| s.end).max().unwrap());
            tail.push((z - s1) as f64 / 60.0);
            // Head only where the span covers this run alone: a span bridging two runs legitimately
            // starts long before the second, which would otherwise read as a huge early open.
            let solo = cov.len() == 1 && runs.iter().filter(|(x, y)| s0 < *y && s1 > *x).count() == 1;
            if solo {
                head.push((a - s0) as f64 / 60.0);
            } else {
                head_bridged += 1;
            }
        }
        // A span matching no >=90 min run is only a false positive if it matches no asleep stretch at
        // all; the strap's short runs are naps, which a detector is right to find.
        let any_run = asleep_runs(&b.band, 10, 300);
        for s in &spans {
            let n = runs.iter().filter(|(a, z)| s.start < *z && s.end > *a).count();
            if n == 0 {
                // A span the band does not cover cannot be judged either way; counting it as a false
                // positive would blame the detector for a gap in the reference.
                let seen = b.band.iter().filter(|(t, _)| *t >= s.start && *t < s.end).count() as f64;
                if seen / ((s.end - s.start) as f64).max(1.0) < 0.5 {
                    unjudged += 1;
                    continue;
                }
                let short: Vec<_> = any_run.iter().filter(|(a, z)| s.start < *z && s.end > *a).collect();
                if short.is_empty() {
                    spurious += 1;
                    no_sleep_min.push((s.end - s.start) as f64 / 60.0);
                } else {
                    naps += 1;
                    nap_min.push(short.iter().map(|(a, z)| (z - a) as f64 / 60.0).sum::<f64>());
                }
            } else if n > 1 {
                merged += 1;
            }
        }

        for &(a, z) in &runs {
            let fresh: Vec<_> = spans.iter().filter(|s| s.start < z && s.end > a).collect();
            let kept: Vec<_> = b.stored.iter().filter(|s| s.start < z && s.end > a).collect();
            if fresh.is_empty() || kept.is_empty() {
                continue;
            }
            let tail = (z - kept.iter().map(|s| s.end).max().unwrap()) as f64 / 60.0;
            if kept.iter().any(|s| s.edited) {
                edited_tail.push(tail);
            } else {
                stored_tail.push(tail);
                fresh_tail.push((z - fresh.iter().map(|s| s.end).max().unwrap()) as f64 / 60.0);
            }
        }

        // Staging inside each detected span, against the band's own call on the same epochs.
        for s in &spans {
            let input = SleepInput {
                start: s.start,
                end: s.end,
                hr: b.hr.iter().filter(|h| h.ts >= s.start && h.ts < s.end).cloned().collect(),
                rr: b.rr.iter().filter(|r| r.ts >= s.start && r.ts < s.end).cloned().collect(),
                accel: b.accel.iter().filter(|g| g.ts >= s.start && g.ts < s.end).cloned().collect(),
            };
            if input.hr.len() < 120 || input.accel.len() < 120 {
                continue;
            }
            let segs = stage_v2_prepared(&prepare_v2(&input, &params), &params);
            // The same staging refined, counting which side of the density gate this span fell on, so
            // the refined row below is the app's path or is reported as a mixture.
            let mut probe = RefineCensus::default();
            let refined = probe.refine(&segs, &input.accel, &steps_in(&b.steps, s.start, s.end));
            census.absorb(&probe);
            for &(ts, st) in b.band.iter().filter(|(t, _)| *t >= s.start && *t < s.end) {
                let stage = segs.iter().find(|g| g.start <= ts && ts < g.end).map(|g| g.stage);
                if let Some(stage) = stage {
                    pred_all += 1;
                    pred_wake += i64::from(stage == SleepStage::Wake);
                    band_all += 1;
                    band_wake += i64::from(st != 2);
                }
                if let Some(stage) = refined.iter().find(|g| g.start <= ts && ts < g.end).map(|g| g.stage) {
                    ref_all += 1;
                    ref_wake += i64::from(stage == SleepStage::Wake);
                }
            }
        }
    }

    println!("A/B  detection, over {blocks} band-carrying wear blocks");
    println!("     strap asleep runs >= 90 min : {runs_total}");
    println!("     runs with a detected span   : {runs_hit}  (missed {})", runs_total - runs_hit);
    println!(
        "     spans over a nap (10-90 min): {naps}  (median nap {:.0} min)",
        median(&mut nap_min.clone())
    );
    println!(
        "     spans over NO strap sleep   : {spurious}  (median span {:.0} min)",
        median(&mut no_sleep_min.clone())
    );
    println!("     spans the band cannot judge : {unjudged}");
    println!(
        "     open before strap-asleep    : median {:+.1} min, p90 {:+.1}, max {:+.1}  (n={}, {} bridged runs excluded)",
        median(&mut head.clone()), pct(&mut head.clone(), 0.9), pct(&mut head.clone(), 1.0),
        head.len(), head_bridged
    );
    println!(
        "     strap asleep after we close : median {:+.1} min, p90 {:+.1}, max {:+.1}",
        median(&mut tail.clone()), pct(&mut tail.clone(), 0.9), pct(&mut tail.clone(), 1.0)
    );
    let lost: f64 = tail.iter().filter(|x| **x > 15.0).sum();
    println!(
        "     runs closed >15 min early   : {} of {runs_hit}, {:.1} h of strap-sleep lost",
        tail.iter().filter(|x| **x > 15.0).count(), lost / 60.0
    );
    // The same runs scored twice: what the app kept, and what the detector produces now. Hand-edited
    // rows are excluded and reported separately — their window is the user's, frozen against
    // re-detection, so counting them here reads a correction as a persistence defect.
    println!(
        "\n     STORED sessions, same runs  : median {:+.1} min, p90 {:+.1}, >15 min early {} of {}",
        median(&mut stored_tail.clone()), pct(&mut stored_tail.clone(), 0.9),
        stored_tail.iter().filter(|x| **x > 15.0).count(), stored_tail.len()
    );
    println!(
        "     FRESH detection, same runs  : median {:+.1} min, p90 {:+.1}, >15 min early {} of {}",
        median(&mut fresh_tail.clone()), pct(&mut fresh_tail.clone(), 0.9),
        fresh_tail.iter().filter(|x| **x > 15.0).count(), fresh_tail.len()
    );
    println!(
        "     hand-edited, NOT persistence: median {:+.1} min, p90 {:+.1}, >15 min early {} of {}",
        median(&mut edited_tail.clone()), pct(&mut edited_tail.clone(), 0.9),
        edited_tail.iter().filter(|x| **x > 15.0).count(), edited_tail.len()
    );
    println!("\nC    selection");
    println!("     one strap run over >1 span  : {split}");
    println!("     one span over >1 strap run  : {merged}");
    println!("\nE-I  staging inside the detected spans");
    println!(
        "     wake fraction  ours {:.1}%  vs strap {:.1}%  over {} epochs",
        100.0 * pred_wake as f64 / pred_all.max(1) as f64,
        100.0 * band_wake as f64 / band_all.max(1) as f64,
        pred_all
    );
    println!(
        "     the same + refine_wake (the app): ours {:.1}%  vs strap {:.1}%  over {} epochs",
        100.0 * ref_wake as f64 / ref_all.max(1) as f64,
        100.0 * band_wake as f64 / band_all.max(1) as f64,
        ref_all
    );
    println!("{}", census.line("the refined wake fraction"));
}

/// One span's slice of the strap's step stream, which is what the refinement's density gate reads.
fn steps_in(steps: &[StepSample], start: i64, end: i64) -> Vec<StepSample> {
    steps.iter().filter(|s| s.ts >= start && s.ts < end).cloned().collect()
}
