//! The motion channel's INPUT: the strap's own on-chip ENMO against the |delta gravity| proxy we derive.
//!
//!   PYTHONIOENCODING=utf-8 python sleep-benchmark/graft_dynaccel.py   # once, to add the column
//!   cargo run --release -p physio-algo --example motion_channel
//!
//! 1  what the two series are, side by side, on the blocks that carry both
//! 2  the staging each produces, scored against the band
//! 3  the move threshold swept on both, so one setting cannot decide it
//!
//! Motion is the largest emission channel by ablation, and every sweep this project has run moved its
//! WEIGHTS. `gravitySample.dynAccelG` is an ENMO the strap computes itself, and the fixture builder was
//! dropping it, so the channel's own EVIDENCE had never been swapped.
//!
//! The substitution needs no library change and takes none. `prepare_v2` reads gravity for exactly one
//! purpose — the magnitude of the second-to-second difference — so feeding it a stream whose consecutive
//! differences ARE the on-chip ENMO puts that reading in the channel exactly. Detection and the wake
//! refinement keep the REAL gravity, since detection must be held constant and the refinement's posture
//! check needs true posture.
//!
//! **Bound, and it is not a small one.** Three blocks, one wearer, one device, and no PSG cohort records
//! the column at all. This can move band kappa2 and nothing else; it cannot speak to F's two-reference
//! ceiling, which rests on cohorts that have no ENMO to give.

mod common;

use common::{
    dirs_of, median, read_accel, read_band, read_dyn_accel, read_hr, read_rr, read_steps, stage_at,
    RefineCensus, TwoClass,
};

use physio_algo::sleep::{
    detect_sessions, params::Params, prepare_v2, stage_v2_prepared, AccelSample, HrSample, RrRun,
    SleepInput, SleepStage, StepSample,
};

const BAND_ASLEEP: i32 = 2;
/// The shipped move multiplier, and the sweep either side of it.
const MOVE_MULTS: [f64; 7] = [8.0, 16.0, 24.0, 32.0, 38.0, 48.0, 64.0];

/// A block carrying both motion series over one wear window.
struct Block {
    name: String,
    hr: Vec<HrSample>,
    rr: Vec<RrRun>,
    accel: Vec<AccelSample>,
    steps: Vec<StepSample>,
    band: Vec<(i64, i32)>,
    dyn_g: Vec<(i64, f64)>,
}

/// A gravity stream whose consecutive-second difference magnitudes ARE `dyn_g`, so the stager's jerk
/// reads the strap's ENMO instead of our own derivative. The walk alternates sign, which keeps the
/// coordinate bounded without changing any difference.
fn as_gravity(dyn_g: &[(i64, f64)]) -> Vec<AccelSample> {
    let mut out = Vec::with_capacity(dyn_g.len());
    let (mut x, mut sign) = (0.0f64, 1.0f64);
    for (i, &(ts, d)) in dyn_g.iter().enumerate() {
        if i > 0 {
            x += sign * d;
            sign = -sign;
        }
        out.push(AccelSample { ts, x, y: 0.0, z: 1.0 });
    }
    out
}

fn load() -> Vec<Block> {
    let mut out = Vec::new();
    for d in dirs_of("continuous") {
        let dyn_g = read_dyn_accel(&d);
        let band = read_band(&d);
        if dyn_g.is_empty() || band.is_empty() {
            continue;
        }
        let accel = read_accel(&d);
        if accel.len() < 120 {
            continue;
        }
        out.push(Block {
            name: d.file_name().unwrap_or_default().to_string_lossy().to_string(),
            hr: read_hr(&d),
            rr: read_rr(&d),
            steps: read_steps(&d),
            accel,
            band,
            dyn_g,
        });
    }
    out
}

/// The derived series in the same `(ts, magnitude)` shape the on-chip column arrives in, so it can be
/// pushed back through [`as_gravity`] as a positive control. The first second carries no difference.
fn derived_series(accel: &[AccelSample]) -> Vec<(i64, f64)> {
    let mut out = Vec::with_capacity(accel.len());
    for (i, g) in accel.iter().enumerate() {
        let d = if i == 0 {
            0.0
        } else {
            let (a, b) = (accel[i - 1], *g);
            let (dx, dy, dz) = (b.x - a.x, b.y - a.y, b.z - a.z);
            (dx * dx + dy * dy + dz * dz).sqrt()
        };
        out.push((g.ts, d));
    }
    out
}

/// Per-second |delta gravity| over one block, which is the series the stager derives today.
fn derived_jerks(accel: &[AccelSample]) -> Vec<f64> {
    accel
        .windows(2)
        .filter(|w| w[1].ts == w[0].ts + 1)
        .map(|w| {
            let (dx, dy, dz) = (w[1].x - w[0].x, w[1].y - w[0].y, w[1].z - w[0].z);
            (dx * dx + dy * dy + dz * dz).sqrt()
        })
        .collect()
}

/// Which motion series the staging arm reads. `Control` is `Derived` pushed through the same synthetic
/// construction `OnChip` uses, so a difference between those two would be the construction and not the data.
#[derive(Clone, Copy, PartialEq)]
enum Motion {
    Derived,
    Control,
    OnChip,
}

/// Stage every detected span of every block under one motion source, scored per band second. Detection
/// runs on the real gravity in every arm, so only the staging channel differs.
fn score(blocks: &[Block], p: &Params, which: Motion, census: &mut RefineCensus) -> TwoClass {
    let mut tc = TwoClass::default();
    for b in blocks {
        let motion: Vec<AccelSample> = match which {
            Motion::Derived => b.accel.clone(),
            Motion::Control => as_gravity(&derived_series(&b.accel)),
            Motion::OnChip => as_gravity(&b.dyn_g),
        };
        for s in detect_sessions(&b.hr, &b.accel, 0, &[], &b.band, None) {
            let cut = |v: &[AccelSample]| -> Vec<AccelSample> {
                v.iter().filter(|g| g.ts >= s.start && g.ts < s.end).cloned().collect()
            };
            let input = SleepInput {
                start: s.start,
                end: s.end,
                hr: b.hr.iter().filter(|h| h.ts >= s.start && h.ts < s.end).cloned().collect(),
                rr: b.rr.iter().filter(|r| r.ts >= s.start && r.ts < s.end).cloned().collect(),
                accel: cut(&motion),
            };
            if input.hr.len() < 120 || input.accel.len() < 120 {
                continue;
            }
            let real = cut(&b.accel);
            let steps: Vec<StepSample> =
                b.steps.iter().filter(|t| t.ts >= s.start && t.ts < s.end).cloned().collect();
            let segs = stage_v2_prepared(&prepare_v2(&input, p), p);
            // The refinement reads posture, so it gets the real stream whichever arm this is.
            let out = census.refine(&segs, &real, &steps);
            for &(ts, code) in b.band.iter().filter(|(t, _)| *t >= s.start && *t < s.end) {
                let Some(g) = stage_at(&out, ts) else { continue };
                tc.add(g == SleepStage::Wake, code != BAND_ASLEEP);
            }
        }
    }
    tc
}

fn row(label: &str, tc: &TwoClass) {
    println!(
        "   {label:<40} {:>10} {:>9.1}% {:>9.1}% {:>9.3} {:>9.1} {:>11.1}",
        tc.n,
        tc.pred_pct(),
        tc.true_pct(),
        tc.kappa(),
        tc.recall(),
        tc.precision()
    );
}

fn header() {
    println!(
        "   {:<40} {:>10} {:>10} {:>10} {:>9} {:>9} {:>11}",
        "motion source", "band sec", "ours w%", "band w%", "kappa2", "recall", "precision"
    );
}

fn main() {
    let shipped = Params::SHIPPED;
    let blocks = load();
    if blocks.is_empty() {
        println!("no `continuous` block carries dynaccel.csv under {}", common::fixtures_root().display());
        println!("run: PYTHONIOENCODING=utf-8 python sleep-benchmark/graft_dynaccel.py");
        return;
    }

    println!("1  the two series, on the blocks that carry both");
    println!(
        "   {:<52} {:>10} {:>10} {:>12} {:>12}",
        "block", "grav sec", "ENMO sec", "median jerk", "median ENMO"
    );
    let (mut all_j, mut all_d) = (Vec::new(), Vec::new());
    for b in &blocks {
        let mut j = derived_jerks(&b.accel);
        let mut d: Vec<f64> = b.dyn_g.iter().map(|x| x.1).collect();
        println!(
            "   {:<52} {:>10} {:>10} {:>12.5} {:>12.5}",
            b.name,
            b.accel.len(),
            b.dyn_g.len(),
            median(&mut j),
            median(&mut d)
        );
        all_j.extend_from_slice(&j);
        all_d.extend_from_slice(&d);
    }
    println!(
        "\n   pooled: {} derived jerks, median {:.5} g · {} on-chip ENMO, median {:.5} g",
        all_j.len(),
        median(&mut all_j.clone()),
        all_d.len(),
        median(&mut all_d.clone())
    );
    println!("   Both feed the same night-relative threshold (median x jerk_move_mult), so the levels do");
    println!("   not have to match — what matters is whether the SHAPE separates movement better.");

    println!("\n2  the staging each produces, against the band");
    let mut c1 = RefineCensus::default();
    let derived = score(&blocks, &shipped, Motion::Derived, &mut c1);
    let mut c0 = RefineCensus::default();
    let control = score(&blocks, &shipped, Motion::Control, &mut c0);
    let mut c2 = RefineCensus::default();
    let on_chip = score(&blocks, &shipped, Motion::OnChip, &mut c2);
    println!();
    header();
    row("|delta gravity|, derived (SHIPPED)", &derived);
    row("the same, through the synthetic stream", &control);
    row("dynAccelG, the strap's own ENMO", &on_chip);
    println!("{}", c1.line("derived"));
    println!("{}", c2.line("on-chip"));
    println!(
        "   POSITIVE CONTROL: pushing the DERIVED series through the same construction reads kappa2 {:.3}",
        control.kappa()
    );
    println!(
        "   against the real stream's {:.3}, a difference of {:+.4}. The construction is inert, so what",
        derived.kappa(),
        control.kappa() - derived.kappa()
    );
    println!("   the on-chip row measures is the DATA.");
    println!("   kappa2 {:+.3} for swapping the channel's input.", on_chip.kappa() - derived.kappa());
    println!("   Read the SHIPPED row against itself, not against H's headline: these are 4 spans on one");
    println!("   wearer's three newest blocks, where the derived channel scores far below the {:.3} it reaches", 0.193);
    println!("   over the 28 spans of the whole corpus. A low baseline makes a delta cheaper to earn.");

    println!("\n3  the move threshold swept on both, so one setting cannot decide it");
    println!("   {:>8} {:>12} {:>10} {:>12} {:>10} {:>12}", "mult", "derived k2", "w%", "on-chip k2", "w%", "difference");
    for m in MOVE_MULTS {
        let p = Params { jerk_move_mult: m, ..shipped };
        let (mut a, mut b) = (RefineCensus::default(), RefineCensus::default());
        let (x, y) = (score(&blocks, &p, Motion::Derived, &mut a), score(&blocks, &p, Motion::OnChip, &mut b));
        let mark = if (m - shipped.jerk_move_mult).abs() < f64::EPSILON { " <- shipped" } else { "" };
        println!(
            "   {m:>8.0} {:>12.3} {:>9.1}% {:>12.3} {:>9.1}% {:>+12.3}{mark}",
            x.kappa(),
            x.pred_pct(),
            y.kappa(),
            y.pred_pct(),
            y.kappa() - x.kappa()
        );
    }
    println!("\n   One wearer, one device, three blocks, band only. No PSG cohort records this column, so");
    println!("   nothing here can move a two-reference argument — it can only say whether the channel is");
    println!("   at a ceiling on its EVIDENCE as well as on its weights.");
}
