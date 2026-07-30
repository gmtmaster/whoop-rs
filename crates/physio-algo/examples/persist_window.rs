//! Step D — the persisted window: what a stored session holds, and whether re-running detection over
//! the same stream reproduces it.
//!
//!   cargo run --release -p physio-algo --example persist_window
//!
//! 1  what is stored, split by provenance: detected rows against hand-edited ones
//! 2  the close edge, stored against fresh, reported blind and then split by provenance
//! 3  every run whose stored session ends >15 min before the strap's, named with its provenance
//! 4  the open edge, same split (the fixture carries the IMMUTABLE detected start, see below)
//! 5  detected rows only: does fresh detection reproduce what was stored
//!
//! `sessions.csv` column 3 is `sleepSession.userEdited`. A hand-corrected window is frozen against
//! re-detection by design, so scoring it against the strap measures the USER, not persistence — the
//! blind comparison in section 2 is kept so that mistake stays visible next to its correction.
//!
//! The fixture carries `startTs` (the immutable primary key), not `startTsAdjusted`, so an edited
//! row's open edge here is its DETECTED onset, not the one the app displays.
//!
//! This harness never stages, so the wake refinement is out of scope by construction rather than by
//! choice: every figure is a window bound, and `refine_wake` rewrites labels inside a fixed span.

mod common;

use common::{median, read_csv, root};

use std::fs;
use std::path::{Path, PathBuf};

use physio_algo::sleep::{detect_sessions_with, AccelSample, DetectParams, DetectedSpan, HrSample};

const BAND_ASLEEP: i32 = 2;
/// Interruption a band asleep run tolerates before it counts as two runs (seconds).
const RUN_TOLERANCE_S: i64 = 300;
/// A close this far before the strap's own asleep end is the truncation Step D exists to find.
const EARLY_CLOSE_MIN: f64 = 15.0;

/// One row of `sessions.csv`: the window the app kept, and whether the user set it by hand.
#[derive(Clone, Copy)]
struct Stored {
    start: i64,
    end: i64,
    edited: bool,
}

struct Block {
    name: String,
    stored: Vec<Stored>,
    hr: Vec<HrSample>,
    accel: Vec<AccelSample>,
    band: Vec<(i64, i32)>,
}

fn load(dir: &Path) -> Option<Block> {
    let band: Vec<(i64, i32)> = read_csv(&dir.join("band.csv")).iter().map(|r| (r[0] as i64, r[1] as i32)).collect();
    if band.is_empty() {
        return None;
    }
    let accel: Vec<AccelSample> = read_csv(&dir.join("gravity.csv"))
        .iter()
        .map(|r| AccelSample { ts: r[0] as i64, x: r[1], y: r[2], z: r[3] })
        .collect();
    if accel.len() < 120 {
        return None;
    }
    let hr = read_csv(&dir.join("hr.csv")).iter().map(|r| HrSample { ts: r[0] as i64, bpm: r[1] as u16 }).collect();
    // Column 3 is absent on a fixture built before the flag was carried; absent reads as un-edited,
    // which is the same direction the blind comparison already assumed.
    let stored = read_csv(&dir.join("sessions.csv"))
        .iter()
        .map(|r| Stored { start: r[0] as i64, end: r[1] as i64, edited: r.get(2).copied().unwrap_or(0.0) != 0.0 })
        .collect();
    let name = dir.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    Some(Block { name, stored, hr, accel, band })
}

/// Contiguous stretches the strap itself called asleep, at least `min_min` long, tolerating a `tol_s`
/// interruption. This is the reference the whole flow is scored against.
fn asleep_runs(band: &[(i64, i32)], min_min: i64, tol_s: i64) -> Vec<(i64, i64)> {
    let (mut out, mut start, mut last) = (Vec::new(), None::<i64>, 0i64);
    for &(ts, st) in band {
        if st != BAND_ASLEEP {
            continue;
        }
        match start {
            None => start = Some(ts),
            Some(s) if ts - last > tol_s => {
                if last - s >= min_min * 60 {
                    out.push((s, last));
                }
                start = Some(ts);
            }
            _ => {}
        }
        last = ts;
    }
    if let Some(s) = start {
        if last - s >= min_min * 60 {
            out.push((s, last));
        }
    }
    out
}

fn pct(v: &mut [f64], p: f64) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[((v.len() - 1) as f64 * p) as usize]
}

fn blocks() -> Vec<Block> {
    let mut dirs: Vec<PathBuf> = fs::read_dir(root("continuous"))
        .map(|rd| rd.filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.is_dir()).collect())
        .unwrap_or_default();
    dirs.sort();
    dirs.iter().filter_map(|d| load(d)).collect()
}

fn spans_of(b: &Block, p: &DetectParams) -> Vec<DetectedSpan> {
    detect_sessions_with(&b.hr, &b.accel, 0, &[], &b.band, None, p)
}

/// One strap run paired with what storage kept for it and what detection gives now.
struct Pair {
    block: String,
    run_end: i64,
    stored_end: i64,
    fresh_end: i64,
    stored_start: i64,
    fresh_start: i64,
    run_start: i64,
    edited: bool,
}

/// Every strap run covered by BOTH a stored session and a fresh span, so the two are scored on the
/// same nights. A run only one side found would otherwise change the denominator, not the answer.
fn pairs(bs: &[Block], p: &DetectParams) -> Vec<Pair> {
    let mut out = Vec::new();
    for b in bs {
        let spans = spans_of(b, p);
        for (a, z) in asleep_runs(&b.band, 90, RUN_TOLERANCE_S) {
            let fresh: Vec<&DetectedSpan> = spans.iter().filter(|s| s.start < z && s.end > a).collect();
            let kept: Vec<&Stored> = b.stored.iter().filter(|s| s.start < z && s.end > a).collect();
            if fresh.is_empty() || kept.is_empty() {
                continue;
            }
            out.push(Pair {
                block: b.name.clone(),
                run_start: a,
                run_end: z,
                stored_end: kept.iter().map(|s| s.end).max().unwrap(),
                fresh_end: fresh.iter().map(|s| s.end).max().unwrap(),
                stored_start: kept.iter().map(|s| s.start).min().unwrap(),
                fresh_start: fresh.iter().map(|s| s.start).min().unwrap(),
                edited: kept.iter().any(|s| s.edited),
            });
        }
    }
    out
}

fn tail_row(label: &str, v: &[f64]) {
    println!(
        "   {label:<28} n={:>2}  median {:+6.1} min  p90 {:+7.1}  max {:+7.1}  >{:.0} min early {} ",
        v.len(),
        median(&mut v.to_vec()),
        pct(&mut v.to_vec(), 0.9),
        pct(&mut v.to_vec(), 1.0),
        EARLY_CLOSE_MIN,
        v.iter().filter(|x| **x > EARLY_CLOSE_MIN).count(),
    );
}

// ── 1  what is stored ──────────────────────────────────────────────────────────────────────────────

fn section_inventory(bs: &[Block]) {
    println!("1  what storage holds, by provenance");
    let all: Vec<&Stored> = bs.iter().flat_map(|b| b.stored.iter()).collect();
    let edited: Vec<&&Stored> = all.iter().filter(|s| s.edited).collect();
    // The wake picker zeroes the seconds field and the bedtime picker passes the existing end straight
    // through, so a whole-minute end is the fingerprint of a hand-set wake time.
    let round = |rows: &[&Stored]| rows.iter().filter(|s| s.end % 60 == 0).count();
    println!("   rows                        : {}", all.len());
    println!("   userEdited = 1              : {}", edited.len());
    println!(
        "   end on a whole minute       : {} of {} edited, {} of {} detected",
        edited.iter().filter(|s| s.end % 60 == 0).count(),
        edited.len(),
        round(&all.iter().filter(|s| !s.edited).copied().collect::<Vec<_>>()),
        all.len() - edited.len(),
    );
    println!(
        "   A whole-minute end on a DETECTED row would be a 1-in-60 coincidence; on an edited row it is\n   \
         the wake picker, which sets SECOND = 0. The bedtime picker passes the stored end through\n   \
         unchanged, so a bed-only edit keeps a detected end and freezes it."
    );
}

// ── 2  the close edge ──────────────────────────────────────────────────────────────────────────────

fn section_close_edge(bs: &[Block], p: &DetectParams, label: &str) {
    let ps = pairs(bs, p);
    println!("\n2  the close edge under {label} — strap-asleep left after the window ends");
    let stored: Vec<f64> = ps.iter().map(|x| (x.run_end - x.stored_end) as f64 / 60.0).collect();
    let fresh: Vec<f64> = ps.iter().map(|x| (x.run_end - x.fresh_end) as f64 / 60.0).collect();
    tail_row("STORED, blind", &stored);
    tail_row("FRESH", &fresh);
    let det: Vec<f64> =
        ps.iter().filter(|x| !x.edited).map(|x| (x.run_end - x.stored_end) as f64 / 60.0).collect();
    let ed: Vec<f64> = ps.iter().filter(|x| x.edited).map(|x| (x.run_end - x.stored_end) as f64 / 60.0).collect();
    tail_row("STORED, detected rows", &det);
    tail_row("STORED, hand-edited rows", &ed);
    let det_fresh: Vec<f64> =
        ps.iter().filter(|x| !x.edited).map(|x| (x.run_end - x.fresh_end) as f64 / 60.0).collect();
    tail_row("FRESH, same detected nights", &det_fresh);
}

// ── 3  every early close, named ────────────────────────────────────────────────────────────────────

fn section_named(bs: &[Block], p: &DetectParams) {
    println!("\n3  every run whose STORED session closes >{EARLY_CLOSE_MIN:.0} min before the strap's asleep run");
    println!("   early(min)  provenance    end on a whole minute  fresh close(min)  block");
    let mut n = 0;
    for x in pairs(bs, p) {
        let early = (x.run_end - x.stored_end) as f64 / 60.0;
        if early <= EARLY_CLOSE_MIN {
            continue;
        }
        n += 1;
        println!(
            "   {:>10.1}  {:<12}  {:>21}  {:>16.1}  {}",
            early,
            if x.edited { "hand-edited" } else { "detected" },
            if x.stored_end % 60 == 0 { "yes" } else { "no" },
            (x.run_end - x.fresh_end) as f64 / 60.0,
            x.block,
        );
    }
    println!("   {n} in total");
}

// ── 4  the open edge ───────────────────────────────────────────────────────────────────────────────

fn section_open_edge(bs: &[Block], p: &DetectParams) {
    println!("\n4  the open edge — how long before the strap's asleep run the window starts");
    println!("   The fixture carries the IMMUTABLE detected startTs, so an edited row's onset here is the");
    println!("   detected one, not the corrected one the app shows.");
    let ps = pairs(bs, p);
    let head = |v: &Pair, stored: bool| {
        ((if stored { v.run_start - v.stored_start } else { v.run_start - v.fresh_start }) as f64) / 60.0
    };
    for (label, edited_only) in [("detected rows", false), ("hand-edited rows", true)] {
        let sel: Vec<&Pair> = ps.iter().filter(|x| x.edited == edited_only).collect();
        let s: Vec<f64> = sel.iter().map(|x| head(x, true)).collect();
        let f: Vec<f64> = sel.iter().map(|x| head(x, false)).collect();
        println!(
            "   {label:<18} n={:>2}  stored median {:+.1} / p90 {:+.1}   fresh median {:+.1} / p90 {:+.1}",
            sel.len(),
            median(&mut s.clone()),
            pct(&mut s.clone(), 0.9),
            median(&mut f.clone()),
            pct(&mut f.clone(), 0.9),
        );
    }
}

// ── 5  does re-detection reproduce a detected row ──────────────────────────────────────────────────

fn section_reproduce(bs: &[Block], p: &DetectParams, label: &str) {
    println!("\n5  detected rows only: fresh detection against what was stored, under {label}");
    let ps: Vec<Pair> = pairs(bs, p).into_iter().filter(|x| !x.edited).collect();
    let mut d_end: Vec<f64> = ps.iter().map(|x| (x.fresh_end - x.stored_end) as f64 / 60.0).collect();
    let mut d_start: Vec<f64> = ps.iter().map(|x| (x.fresh_start - x.stored_start) as f64 / 60.0).collect();
    let mut abs_end: Vec<f64> = d_end.iter().map(|x| x.abs()).collect();
    println!(
        "   end   n={:>2}  median {:+.1} min  p90 {:+.1}  max |Δ| {:.1}  within 15 min {} of {}",
        d_end.len(),
        median(&mut d_end),
        pct(&mut d_end.clone(), 0.9),
        pct(&mut abs_end, 1.0),
        ps.iter().filter(|x| ((x.fresh_end - x.stored_end) as f64 / 60.0).abs() <= EARLY_CLOSE_MIN).count(),
        ps.len(),
    );
    println!(
        "   start n={:>2}  median {:+.1} min  p90 {:+.1}",
        d_start.len(),
        median(&mut d_start),
        pct(&mut d_start.clone(), 0.9),
    );
}

fn main() {
    let bs = blocks();
    let runs: usize = bs.iter().map(|b| asleep_runs(&b.band, 90, RUN_TOLERANCE_S).len()).sum();
    println!("{} band-carrying wear blocks, {runs} strap asleep runs >= 90 min\n", bs.len());

    section_inventory(&bs);
    // PRE_HYSTERESIS is the spine that wrote every stored row; SHIPPED is what an install runs now.
    section_close_edge(&bs, &DetectParams::PRE_HYSTERESIS, "PRE_HYSTERESIS (the detector that wrote the rows)");
    section_close_edge(&bs, &DetectParams::SHIPPED, "SHIPPED");
    section_named(&bs, &DetectParams::PRE_HYSTERESIS);
    section_open_edge(&bs, &DetectParams::PRE_HYSTERESIS);
    section_reproduce(&bs, &DetectParams::PRE_HYSTERESIS, "PRE_HYSTERESIS");
    section_reproduce(&bs, &DetectParams::SHIPPED, "SHIPPED");
}
