//! Does the detection window change staging beyond its own edges?
//!
//!   cargo run --release -p physio-algo --example window_effect
//!
//! Stages each band-labelled night twice: over the detected window, and over the strap's own asleep run.
//! Compares only the epochs BOTH runs contain, so any disagreement there is the window moving the
//! z-score baselines and the deep percentile gate, not the trimmed edges being counted differently.
//!
//! Reported twice: `stage_v2` alone, and the app's path `stage_v2` + `refine_wake`. The refined figure is
//! restricted to nights whose step stream passes the refinement's own density gate, and the count is
//! stated — a night the gate declines is the unrefined path and pooling it would report neither.
//!
//! CAVEAT on the refined figure: `ours` gravity is HELD FORWARD across dropouts, so the refinement's
//! posture check reads artificially stable here. It is the only set that can measure a window
//! *sensitivity* (the window is baked in, so both stagings see the same night), which is why it is used.

mod common;

use common::{
    dirs_of, labels_at, median, read_accel, read_csv, read_hr, read_meta, read_rr, read_steps, RefineCensus,
};

use physio_algo::sleep::{params::Params, prepare_v2, stage_v2_prepared, SleepInput, StageSegment, StepSample};

const EPOCH_SEC: i64 = 30;

/// Stage one window, then run the app's last stage over it through the census. `refine` false gives the
/// `stage_v2`-only path every A-Z number before this was measured on.
fn stage_window(
    input: &SleepInput,
    p: &Params,
    steps: &[StepSample],
    refine: bool,
    census: &mut RefineCensus,
) -> Vec<StageSegment> {
    let segs = stage_v2_prepared(&prepare_v2(input, p), p);
    if !refine {
        return segs;
    }
    census.refine(&segs, &input.accel, steps)
}

/// One night's window sensitivity, on both paths, plus whether the gate accepted both stagings.
struct NightEffect {
    name: String,
    inner: usize,
    moved_raw: usize,
    moved_ref: usize,
    wake_full: usize,
    wake_trim: usize,
    wake_full_ref: usize,
    wake_trim_ref: usize,
    dense: bool,
}

/// Total, median-per-night and the wake split over a set of nights, on one of the two paths.
fn summarise(ns: &[&NightEffect], refined: bool) -> (usize, usize, usize, f64, f64, f64) {
    let shared: usize = ns.iter().map(|n| n.inner).sum();
    let moved: usize = ns.iter().map(|n| if refined { n.moved_ref } else { n.moved_raw }).sum();
    let mut per: Vec<f64> = ns
        .iter()
        .map(|n| 100.0 * (if refined { n.moved_ref } else { n.moved_raw }) as f64 / n.inner.max(1) as f64)
        .collect();
    let med = median(&mut per);
    let wf: usize = ns.iter().map(|n| if refined { n.wake_full_ref } else { n.wake_full }).sum();
    let wt: usize = ns.iter().map(|n| if refined { n.wake_trim_ref } else { n.wake_trim }).sum();
    (
        ns.len(),
        shared,
        moved,
        med,
        100.0 * wf as f64 / shared.max(1) as f64,
        100.0 * wt as f64 / shared.max(1) as f64,
    )
}

fn report(label: &str, ns: &[&NightEffect], refined: bool) {
    let (nights, shared, moved, med, wf, wt) = summarise(ns, refined);
    println!(
        "   {label:<34} {nights:>3}  {shared:>7}  {moved:>6}  {:>6.1}%  {med:>10.1}%   {wf:>5.1}% / {wt:.1}%",
        100.0 * moved as f64 / shared.max(1) as f64,
    );
}

fn main() {
    let p = Params::SHIPPED;
    let mut all: Vec<NightEffect> = Vec::new();
    // One census per path per scope, so a declined span cannot be counted into a refined figure.
    let mut census = RefineCensus::default();

    for d in dirs_of("ours") {
        let Some((w0, w1, n)) = read_meta(&d) else { continue };
        let truth = read_csv(&d.join("truth.csv"));
        let asleep: Vec<usize> = truth.iter().filter(|r| r[1] == 1.0).map(|r| r[0] as usize).collect();
        if asleep.len() < 240 {
            continue;
        }
        let (first, last) = (asleep[0], asleep[asleep.len() - 1]);
        if first == 0 && last + 1 >= n {
            continue; // nothing to trim
        }

        let steps = read_steps(&d);
        let full = SleepInput { start: w0, end: w1, hr: read_hr(&d), rr: read_rr(&d), accel: read_accel(&d) };
        let (t0, t1) = (w0 + first as i64 * EPOCH_SEC, w0 + (last + 1) as i64 * EPOCH_SEC);
        let trim = SleepInput { start: t0, end: t1, ..full.clone() };
        let inner = last + 1 - first;

        // Both paths over both windows. A night counts as dense only if the gate accepted BOTH stagings,
        // because the sensitivity is a comparison and half of it being unrefined measures nothing.
        let mut per_night = RefineCensus::default();
        let a = labels_at(&stage_window(&full, &p, &steps, false, &mut per_night), w0, n, EPOCH_SEC);
        let b = labels_at(&stage_window(&trim, &p, &steps, false, &mut per_night), t0, inner, EPOCH_SEC);
        let ar = labels_at(&stage_window(&full, &p, &steps, true, &mut per_night), w0, n, EPOCH_SEC);
        let br = labels_at(&stage_window(&trim, &p, &steps, true, &mut per_night), t0, inner, EPOCH_SEC);
        census.absorb(&per_night);

        all.push(NightEffect {
            name: d.file_name().unwrap().to_string_lossy().chars().take(38).collect(),
            inner,
            moved_raw: (0..inner).filter(|&k| a[first + k] != b[k]).count(),
            moved_ref: (0..inner).filter(|&k| ar[first + k] != br[k]).count(),
            wake_full: (0..inner).filter(|&k| a[first + k] == 0).count(),
            wake_trim: (0..inner).filter(|&k| b[k] == 0).count(),
            wake_full_ref: (0..inner).filter(|&k| ar[first + k] == 0).count(),
            wake_trim_ref: (0..inner).filter(|&k| br[k] == 0).count(),
            dense: per_night.all_refined(),
        });
    }

    println!("{:<40} {:>7} {:>9} {:>11} {:>7}", "night", "shared", "unrefined", "refined", "dense");
    for n in &all {
        if n.moved_raw > 0 || n.moved_ref > 0 {
            println!(
                "{:<40} {:>7} {:>8.1}% {:>10.1}% {:>7}",
                n.name,
                n.inner,
                100.0 * n.moved_raw as f64 / n.inner as f64,
                100.0 * n.moved_ref as f64 / n.inner as f64,
                if n.dense { "yes" } else { "NO" },
            );
        }
    }

    let every: Vec<&NightEffect> = all.iter().collect();
    let dense: Vec<&NightEffect> = all.iter().filter(|n| n.dense).collect();
    println!("\n   path / scope                       nights   shared   moved          median   wake% full / trim");
    report("stage_v2, every night", &every, false);
    report("stage_v2, dense-stream nights", &dense, false);
    report("+ refine_wake (the app), dense", &dense, true);
    println!("\n{}", census.line("window sensitivity, both windows of every night"));
    println!(
        "   {} of {} nights carry a step stream dense enough for the refinement on BOTH windows.",
        dense.len(),
        all.len()
    );
    println!("   Read the two dense-scope rows against each other; the every-night row is the recorded figure.");
    println!("   `ours` gravity is held forward across gaps, so the refined row's posture check is optimistic.");
}
