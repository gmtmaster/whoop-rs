//! Step C — selection and bridging, scored against the strap's own sleep_state over continuous wear
//! blocks. Runs the real path: detect -> bridge -> pick the main night.
//!
//!   cargo run --release -p physio-algo --example select_bridge
//!
//! 1  the band's own asleep-run length distribution, which is what decides MIN_SLEEP_MIN
//! 2  the MIN_SLEEP_MIN sweep, crossed with the wake absorb, since that changes what a span IS
//! 3  run-to-span cardinality, adjudicated case by case with the band's call across every seam
//! 4  bridging both directions: gaps joined and gaps left, with the interval neither can falsify
//! 5  the open edge with no run dropped, under both named detector configs
//! 6  main-night selection per local day: which group wins and what the pick leaves behind
//! 7  what a split costs: one night staged as fragments against the same night staged whole
//! 8  the wake-absorb sweep, two-sided
//! 9  the whole change on one balanced number: epoch agreement with the strap's own call, on both the
//!    `stage_v2` path every earlier C number was measured on and the `+ refine_wake` path the app runs
//!
//! Blocks are cut on wear, not on labels, so nothing here tells the detector where to look. The
//! fixture carries no timezone, so local time is UTC; section 6 diffs that against +2 h.
//!
//! Sections 1-6 and 8 read detection and selection, which `refine_wake` cannot reach — it rewrites stage
//! labels inside a span the detector already fixed. Only 7 and 9 stage, so only those two carry a census.

mod common;

use common::{dirs_of, read_accel, read_band, read_hr, read_rr, read_steps, RefineCensus};

use std::path::Path;

use physio_algo::sleep::{
    bridged_night_groups, detect_sessions_with, main_night_group_indices, main_night_selection, params::Params,
    prepare_v2, stage_v2_prepared, AccelSample, DetectParams, DetectedSpan, HrSample, MainNightReason, NightBlock,
    RrRun, SleepInput, SleepStage, StageSegment, StepSample,
};

const BAND_ASLEEP: i32 = 2;
/// Interruption a band asleep run tolerates before it counts as two runs (seconds).
const RUN_TOLERANCE_S: i64 = 300;

struct Block {
    name: String,
    hr: Vec<HrSample>,
    rr: Vec<RrRun>,
    accel: Vec<AccelSample>,
    /// The strap's own step stream — what the refinement's density gate reads.
    steps: Vec<StepSample>,
    band: Vec<(i64, i32)>,
}

fn load(dir: &Path) -> Option<Block> {
    let (band, accel) = (read_band(dir), read_accel(dir));
    if band.is_empty() || accel.len() < 120 {
        return None;
    }
    let name = dir.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    Some(Block { name, hr: read_hr(dir), rr: read_rr(dir), accel, steps: read_steps(dir), band })
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

/// Fraction of banded seconds in `(a, b)` the strap called asleep; `None` when the band does not cover it.
fn band_asleep_frac(band: &[(i64, i32)], a: i64, b: i64) -> Option<f64> {
    let seg: Vec<i32> = band.iter().filter(|(t, _)| *t > a && *t < b).map(|(_, s)| *s).collect();
    if seg.is_empty() {
        return None;
    }
    Some(seg.iter().filter(|&&s| s == BAND_ASLEEP).count() as f64 / seg.len() as f64)
}

/// Seconds the strap called asleep inside `[a, b]`, on a 1 Hz band.
fn band_asleep_secs(band: &[(i64, i32)], a: i64, b: i64) -> i64 {
    band.iter().filter(|(t, s)| *t >= a && *t <= b && *s == BAND_ASLEEP).count() as i64
}

fn median(v: &mut [f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn local_day(ts: i64, offset_s: i64) -> i64 {
    (ts + offset_s).div_euclid(86_400)
}

fn blocks() -> Vec<Block> {
    dirs_of("continuous").iter().filter_map(|d| load(d)).collect()
}

fn spans_of(b: &Block, p: &DetectParams) -> Vec<DetectedSpan> {
    detect_sessions_with(&b.hr, &b.accel, 0, &[], &b.band, None, p)
}

// ── 1  the band's own asleep-run lengths ───────────────────────────────────────────────────────────

fn section_run_lengths(bs: &[Block]) {
    println!("1  the band's own asleep runs — the distribution MIN_SLEEP_MIN has to sit in");
    println!("   tol   runs   <30  30-45  45-60  60-70  70-90  90-120  120-240  240+   median short (min)");
    for tol in [60i64, 300, 900] {
        let mut lens: Vec<f64> = Vec::new();
        for b in bs {
            for (a, z) in asleep_runs(&b.band, 0, tol) {
                lens.push((z - a) as f64 / 60.0);
            }
        }
        let c = |lo: f64, hi: f64| lens.iter().filter(|&&x| x >= lo && x < hi).count();
        let mut short: Vec<f64> = lens.iter().copied().filter(|&x| x < 90.0).collect();
        println!(
            "   {:>4}  {:>4}   {:>3}  {:>5}  {:>5}  {:>5}  {:>5}  {:>6}  {:>7}  {:>4}   {:.1}",
            tol,
            lens.len(),
            c(0.0, 30.0),
            c(30.0, 45.0),
            c(45.0, 60.0),
            c(60.0, 70.0),
            c(70.0, 90.0),
            c(90.0, 120.0),
            c(120.0, 240.0),
            c(240.0, f64::INFINITY),
            median(&mut short),
        );
    }
    let mut short: Vec<f64> =
        bs.iter().flat_map(|b| asleep_runs(&b.band, 0, RUN_TOLERANCE_S)).map(|(a, z)| (z - a) as f64 / 60.0).filter(|&x| x < 180.0).collect();
    short.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!("   every run under 180 min at tol {RUN_TOLERANCE_S} s, sorted (min):");
    println!("   {}", short.iter().map(|x| format!("{x:.0}")).collect::<Vec<_>>().join(" "));
}

// ── 2  MIN_SLEEP_MIN ───────────────────────────────────────────────────────────────────────────────

struct SweepRow {
    min_sleep: i64,
    spans: usize,
    runs_hit: usize,
    naps: usize,
    spurious: usize,
    spurious_min: Vec<f64>,
}

fn sweep_min_sleep(bs: &[Block], absorb: i64) -> Vec<SweepRow> {
    let mut out = Vec::new();
    for min_sleep in [30i64, 40, 45, 50, 55, 60, 65, 70, 75, 80, 90] {
        let p = DetectParams { min_sleep_min: min_sleep, wake_absorb_max_min: absorb, ..DetectParams::SHIPPED };
        let (mut spans_n, mut runs_hit, mut naps, mut spurious) = (0, 0, 0, 0);
        let mut spurious_min = Vec::new();
        for b in bs {
            let spans = spans_of(b, &p);
            spans_n += spans.len();
            let runs = asleep_runs(&b.band, 90, RUN_TOLERANCE_S);
            let any_run = asleep_runs(&b.band, 10, RUN_TOLERANCE_S);
            for &(a, z) in &runs {
                if spans.iter().any(|s| s.start < z && s.end > a) {
                    runs_hit += 1;
                }
            }
            for s in &spans {
                if runs.iter().any(|(a, z)| s.start < *z && s.end > *a) {
                    continue;
                }
                let seen = b.band.iter().filter(|(t, _)| *t >= s.start && *t < s.end).count() as f64;
                if seen / ((s.end - s.start) as f64).max(1.0) < 0.5 {
                    continue; // the band cannot judge this span either way
                }
                if any_run.iter().any(|(a, z)| s.start < *z && s.end > *a) {
                    naps += 1;
                } else {
                    spurious += 1;
                    spurious_min.push((s.end - s.start) as f64 / 60.0);
                }
            }
        }
        out.push(SweepRow { min_sleep, spans: spans_n, runs_hit, naps, spurious, spurious_min });
    }
    out
}

fn section_min_sleep(bs: &[Block], runs_total: usize) {
    println!("\n2  MIN_SLEEP_MIN, swept — the gate is `duration <= min` so the value itself is rejected");
    println!("   Swept at both wake-absorb settings, because absorbing a stir changes what a span IS.");
    for absorb in [0i64, DetectParams::SHIPPED.wake_absorb_max_min] {
        println!("   wake-absorb {absorb} min");
        println!("   min  spans  runs found  spans over a nap  spans over NO strap sleep  (their lengths, min)");
        for r in sweep_min_sleep(bs, absorb) {
            let mark = if r.min_sleep == 60 { "*" } else { " " };
            println!(
                "  {}{:>3}  {:>5}  {:>5} of {:>2}  {:>14}  {:>23}  {}",
                mark,
                r.min_sleep,
                r.spans,
                r.runs_hit,
                runs_total,
                r.naps,
                r.spurious,
                r.spurious_min.iter().map(|x| format!("{x:.0}")).collect::<Vec<_>>().join(" "),
            );
        }
    }
    // The gate can only choose between spans, so the span lengths are what fixes it. Listed with the
    // strap's own call over each, from the loosest gate, so nothing the tighter ones drop is invisible.
    let loose = DetectParams { min_sleep_min: 5, ..DetectParams::SHIPPED };
    let mut near: Vec<(f64, String)> = Vec::new();
    for b in bs {
        for s in spans_of(b, &loose) {
            let dur = (s.end - s.start) as f64 / 60.0;
            if dur >= 180.0 {
                continue;
            }
            let frac = band_asleep_frac(&b.band, s.start - 1, s.end + 1);
            near.push((dur, frac.map(|f| format!("{:.0}%", 100.0 * f)).unwrap_or_else(|| "unbanded".into())));
        }
    }
    near.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    println!("   every candidate span under 180 min at a 5-min gate, with the strap's asleep fraction over it:");
    for (d, f) in &near {
        println!("     {d:>5.0} min  {f}");
    }
}

// ── 3  run-to-span cardinality, adjudicated ────────────────────────────────────────────────────────

fn section_cardinality(bs: &[Block]) {
    println!("\n3  run-to-span cardinality, every case named (shipped detector)");
    println!("   Refinement-independent by construction: `refine_wake` rewrites stage labels inside a span");
    println!("   the detector already fixed, so it can move no span edge and no run-to-span count here.");
    let (mut split, mut merged) = (0, 0);
    for b in bs {
        let spans = spans_of(b, &DetectParams::SHIPPED);
        let runs = asleep_runs(&b.band, 90, RUN_TOLERANCE_S);
        for &(a, z) in &runs {
            let cov: Vec<&DetectedSpan> = spans.iter().filter(|s| s.start < z && s.end > a).collect();
            if cov.len() > 1 {
                split += 1;
                println!(
                    "   SPLIT  {}  strap run {:.1} h, cut into {} spans",
                    b.name,
                    (z - a) as f64 / 3600.0,
                    cov.len()
                );
                for w in cov.windows(2) {
                    let (g0, g1) = (w[0].end, w[1].start);
                    let frac = band_asleep_frac(&b.band, g0, g1);
                    println!(
                        "          seam {:.0} min wide, strap asleep across it {}",
                        (g1 - g0) as f64 / 60.0,
                        frac.map(|f| format!("{:.0}%", 100.0 * f)).unwrap_or_else(|| "unbanded".into())
                    );
                }
            }
        }
        for s in &spans {
            let cov: Vec<&(i64, i64)> = runs.iter().filter(|(a, z)| s.start < *z && s.end > *a).collect();
            if cov.len() > 1 {
                merged += 1;
                println!(
                    "   MERGE  {}  one {:.1} h span over {} strap runs",
                    b.name,
                    (s.end - s.start) as f64 / 3600.0,
                    cov.len()
                );
                for w in cov.windows(2) {
                    let (g0, g1) = (w[0].1, w[1].0);
                    let frac = band_asleep_frac(&b.band, g0, g1);
                    println!(
                        "          strap-awake gap {:.0} min, strap asleep across it {}",
                        (g1 - g0) as f64 / 60.0,
                        frac.map(|f| format!("{:.0}%", 100.0 * f)).unwrap_or_else(|| "unbanded".into())
                    );
                }
            }
        }
    }
    println!("   one strap run over >1 span : {split}");
    println!("   one span over >1 strap run : {merged}");
}

// ── 4  bridging, both directions ───────────────────────────────────────────────────────────────────

/// Every inter-span gap in a block as `(minutes, bridged, strap-asleep fraction across it)`.
fn gaps_of(b: &Block) -> Vec<(f64, bool, Option<f64>)> {
    let spans = spans_of(b, &DetectParams::SHIPPED);
    let nb: Vec<NightBlock> = spans.iter().map(|s| NightBlock { start: s.start, end: s.end }).collect();
    let groups = bridged_night_groups(&nb, 0);
    let mut out = Vec::new();
    for g in &groups {
        for w in g.indices.windows(2) {
            let (g0, g1) = (nb[w[0]].end, nb[w[1]].start);
            out.push(((g1 - g0) as f64 / 60.0, true, band_asleep_frac(&b.band, g0, g1)));
        }
    }
    for w in groups.windows(2) {
        let last = w[0].indices.iter().map(|&i| nb[i].end).max().unwrap_or(0);
        let next = w[1].indices.iter().map(|&i| nb[i].start).min().unwrap_or(0);
        out.push(((next - last) as f64 / 60.0, false, band_asleep_frac(&b.band, last, next)));
    }
    out
}

fn section_bridging(bs: &[Block]) {
    println!("\n4  bridging — every inter-span gap, joined or not, with the strap's call across it");
    println!("   Joining a strap-AWAKE gap is the intent (a mid-night waking). The two failures are");
    println!("   joining a gap that separates two real sleeps, and leaving one the strap slept through.");
    println!("   gap(min)  bridged  strap asleep across  block");
    let (mut left_asleep, mut unbanded) = (0, 0);
    let (mut joined, mut left) = (Vec::new(), Vec::new());
    for b in bs {
        for (g, bridged, frac) in gaps_of(b) {
            println!(
                "   {:>8.0}  {:<7}  {:>19}  {}",
                g,
                if bridged { "yes" } else { "no" },
                frac.map(|f| format!("{:.0}%", 100.0 * f)).unwrap_or_else(|| "unbanded".into()),
                b.name
            );
            match frac {
                None => unbanded += 1,
                Some(f) if f >= 0.5 && !bridged => left_asleep += 1,
                Some(_) => {}
            }
            if bridged {
                joined.push(g);
            } else {
                left.push(g);
            }
        }
    }
    let widest_joined = joined.iter().cloned().fold(0.0f64, f64::max);
    let narrowest_left = left.iter().cloned().fold(f64::INFINITY, f64::min);
    println!("   gaps joined {} · gaps left {} · the band cannot judge {unbanded}", joined.len(), left.len());
    println!("   left a gap the strap slept through : {left_asleep}");
    println!("   widest joined {widest_joined:.0} min · narrowest left {narrowest_left:.0} min");
    println!(
        "   NO gap falls in ({widest_joined:.0}, {narrowest_left:.0}) min, so every selection-layer bridge threshold in\n   \
         that interval partitions this corpus identically — the shipped 60 / 90 sit inside it and are\n   \
         unfalsifiable here."
    );
    if joined.is_empty() {
        println!("   The selection-layer bridge fires on NOTHING now: detection absorbs the stirs first.");
    }
}

// ── 5  the open edge, with the bridged runs scored instead of dropped (B4) ─────────────────────────

fn section_open_edge(bs: &[Block], runs_total: usize, p: &DetectParams, label: &str) {
    let (mut first, mut interior, mut unhit) = (Vec::new(), Vec::new(), 0usize);
    for b in bs {
        let spans = spans_of(b, p);
        let nb: Vec<NightBlock> = spans.iter().map(|s| NightBlock { start: s.start, end: s.end }).collect();
        let groups = bridged_night_groups(&nb, 0);
        let runs = asleep_runs(&b.band, 90, RUN_TOLERANCE_S);
        for g in &groups {
            let gs = g.indices.iter().map(|&i| nb[i].start).min().unwrap_or(0);
            let ge = g.indices.iter().map(|&i| nb[i].end).max().unwrap_or(0);
            let mut covered: Vec<&(i64, i64)> = runs.iter().filter(|(a, z)| gs < *z && ge > *a).collect();
            covered.sort_by_key(|r| r.0);
            for (k, (a, _)) in covered.iter().enumerate() {
                if k == 0 {
                    first.push((a - gs) as f64 / 60.0);
                } else {
                    interior.push((a - gs) as f64 / 60.0);
                }
            }
        }
        for &(a, z) in &runs {
            if !groups.iter().any(|g| {
                let gs = g.indices.iter().map(|&i| nb[i].start).min().unwrap_or(0);
                let ge = g.indices.iter().map(|&i| nb[i].end).max().unwrap_or(0);
                gs < z && ge > a
            }) {
                unhit += 1;
            }
        }
    }
    println!(
        "   {label:<14} opening run of a group n={:>2}  median {:+.1} min, p90 {:+.1}, max {:+.1}  ·  \
         re-onsets inside a group {}  ·  unhit {unhit}  ·  accounted {} of {runs_total}",
        first.len(),
        median(&mut first.clone()),
        pct(&mut first.clone(), 0.9),
        pct(&mut first.clone(), 1.0),
        interior.len(),
        first.len() + interior.len() + unhit,
    );
}

fn pct(v: &mut [f64], p: f64) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[((v.len() - 1) as f64 * p) as usize]
}

// ── 6  main-night selection ────────────────────────────────────────────────────────────────────────

fn section_selection(bs: &[Block], offset_s: i64, verbose: bool) -> (i64, i64, usize) {
    if verbose {
        println!("\n6  main-night selection, per local day (offset {offset_s} s)");
        println!("   A day's headline sleep is the winning GROUP only; a second sleep far from it stays its");
        println!("   own row by design, so each shortfall is named with the gap that separates it.");
    }
    let (mut kept, mut available, mut days) = (0i64, 0i64, 0usize);
    let (mut dropped_days, mut reasons) = (0usize, [0usize; 4]);
    for b in bs {
        let spans = spans_of(b, &DetectParams::SHIPPED);
        let mut day_keys: Vec<i64> = spans.iter().map(|s| local_day(s.end, offset_s)).collect();
        day_keys.sort_unstable();
        day_keys.dedup();
        for d in day_keys {
            let matched: Vec<&DetectedSpan> = spans.iter().filter(|s| local_day(s.end, offset_s) == d).collect();
            let nb: Vec<NightBlock> = matched.iter().map(|s| NightBlock { start: s.start, end: s.end }).collect();
            let Some(group) = main_night_group_indices(&nb, offset_s, None) else { continue };
            if let Some(sel) = main_night_selection(&nb, offset_s, None) {
                reasons[match sel.reason {
                    MainNightReason::OnlyBlock => 0,
                    MainNightReason::Longest => 1,
                    MainNightReason::LongestNearUsual => 2,
                    MainNightReason::AlignedToUsual => 3,
                }] += 1;
            }
            days += 1;
            let day_avail: i64 = nb.iter().map(|s| band_asleep_secs(&b.band, s.start, s.end)).sum();
            let day_kept: i64 = group.iter().map(|&i| band_asleep_secs(&b.band, nb[i].start, nb[i].end)).sum();
            available += day_avail;
            kept += day_kept;
            if day_kept < day_avail {
                dropped_days += 1;
                if verbose {
                    let (gs, ge) = (
                        group.iter().map(|&i| nb[i].start).min().unwrap_or(0),
                        group.iter().map(|&i| nb[i].end).max().unwrap_or(0),
                    );
                    let gap = nb
                        .iter()
                        .enumerate()
                        .filter(|(i, _)| !group.contains(i))
                        .map(|(_, s)| if s.start >= ge { s.start - ge } else { gs - s.end })
                        .min()
                        .unwrap_or(0);
                    println!(
                        "   {}  day {d}: {} spans, group of {}, strap-asleep kept {:.2} h of {:.2} h — nearest\n      \
                         excluded span sits {:.1} h away, strap asleep across that gap {}",
                        b.name,
                        nb.len(),
                        group.len(),
                        day_kept as f64 / 3600.0,
                        day_avail as f64 / 3600.0,
                        gap as f64 / 3600.0,
                        band_asleep_frac(&b.band, ge, ge + gap)
                            .map(|f| format!("{:.0}%", 100.0 * f))
                            .unwrap_or_else(|| "unbanded".into())
                    );
                }
            }
        }
    }
    if verbose {
        println!(
            "   {days} local days, {dropped_days} where the pick left strap-sleep out of the day's figure"
        );
        println!(
            "   strap-asleep inside the selected group: {:.1} h of {:.1} h detected ({:.1}%)",
            kept as f64 / 3600.0,
            available as f64 / 3600.0,
            100.0 * kept as f64 / available.max(1) as f64
        );
        println!(
            "   pick reason: onlyBlock {} · longest {} · longestNearUsual {} · alignedToUsual {}",
            reasons[0], reasons[1], reasons[2], reasons[3]
        );
    }
    (kept, available, days)
}

// ── 7  what a split costs downstream ───────────────────────────────────────────────────────────────

/// Stage `[start, end]`, returning the wall-clock segments so two stagings that begin at different
/// seconds can still be compared on one shared grid.
fn stage_span(b: &Block, start: i64, end: i64, params: &Params) -> Vec<StageSegment> {
    let input = SleepInput {
        start,
        end,
        hr: b.hr.iter().filter(|h| h.ts >= start && h.ts < end).cloned().collect(),
        rr: b.rr.iter().filter(|r| r.ts >= start && r.ts < end).cloned().collect(),
        accel: b.accel.iter().filter(|g| g.ts >= start && g.ts < end).cloned().collect(),
    };
    if input.hr.len() < 120 || input.accel.len() < 120 {
        return Vec::new();
    }
    stage_v2_prepared(&prepare_v2(&input, params), params)
}

/// The span's own gravity and step slices, which is what `analyze` hands the refinement.
fn span_streams(b: &Block, start: i64, end: i64) -> (Vec<AccelSample>, Vec<StepSample>) {
    (
        b.accel.iter().filter(|g| g.ts >= start && g.ts < end).cloned().collect(),
        b.steps.iter().filter(|t| t.ts >= start && t.ts < end).cloned().collect(),
    )
}

fn label_at(segs: &[StageSegment], t: i64) -> Option<SleepStage> {
    segs.iter().find(|g| g.start <= t && t < g.end).map(|g| g.stage)
}

fn section_split_cost(bs: &[Block]) {
    println!("\n7  what a split costs: the same night staged as fragments and as one window.");
    println!("   Measured under the detector WITHOUT the wake absorb, because with it there is no split");
    println!("   left to measure — the evidence for a fix has to outlive the fix.");
    println!("   stage_v2 ONLY, stated so it is not read as the app's path: refine_wake shrinks wake on both");
    println!("   sides of this comparison, which would mask the relabelling the split causes, not measure it.");
    let params = Params::SHIPPED;
    let detect = DetectParams { wake_absorb_max_min: 0, ..DetectParams::SHIPPED };
    let (mut cases, mut moved, mut total) = (0usize, 0i64, 0i64);
    for b in bs {
        let spans = spans_of(b, &detect);
        let nb: Vec<NightBlock> = spans.iter().map(|s| NightBlock { start: s.start, end: s.end }).collect();
        for g in bridged_night_groups(&nb, 0) {
            if g.indices.len() < 2 {
                continue;
            }
            let (gs, ge) = (
                g.indices.iter().map(|&i| nb[i].start).min().unwrap_or(0),
                g.indices.iter().map(|&i| nb[i].end).max().unwrap_or(0),
            );
            let joined = stage_span(b, gs, ge, &params);
            let mut split: Vec<StageSegment> = Vec::new();
            for &i in &g.indices {
                split.extend(stage_span(b, nb[i].start, nb[i].end, &params));
            }
            if joined.is_empty() || split.is_empty() {
                continue;
            }
            // One shared wall-clock grid, so a fragment starting off the joined span's phase is still seen.
            // Each label is also scored against the strap's own two-class call, so "they differ" becomes
            // "and this one is closer".
            let (mut same, mut diff) = (0i64, 0i64);
            let (mut j_ok, mut s_ok, mut banded) = (0i64, 0i64, 0i64);
            let mut t = gs;
            while t < ge {
                if let (Some(j), Some(s)) = (label_at(&joined, t), label_at(&split, t)) {
                    if j == s {
                        same += 1;
                    } else {
                        diff += 1;
                    }
                    if let Some(f) = band_asleep_frac(&b.band, t - 1, t + 30) {
                        let band_asleep = f >= 0.5;
                        banded += 1;
                        j_ok += i64::from((j != SleepStage::Wake) == band_asleep);
                        s_ok += i64::from((s != SleepStage::Wake) == band_asleep);
                    }
                }
                t += 30;
            }
            cases += 1;
            moved += diff;
            total += same + diff;
            println!(
                "   {}  {:.1} h group of {} spans: {} of {} shared epochs relabelled ({:.1}%)",
                b.name,
                (ge - gs) as f64 / 3600.0,
                g.indices.len(),
                diff,
                same + diff,
                100.0 * diff as f64 / (same + diff).max(1) as f64
            );
            println!(
                "      agrees with the strap's asleep/wake call: staged as one {:.1}%, staged as fragments {:.1}%  (n={banded})",
                100.0 * j_ok as f64 / banded.max(1) as f64,
                100.0 * s_ok as f64 / banded.max(1) as f64
            );
        }
    }
    println!(
        "   {cases} multi-span groups, {moved} of {total} shared epochs staged differently ({:.1}%)",
        100.0 * moved as f64 / total.max(1) as f64
    );
}

// ── 8  the dense-path bridge, swept two-sided ──────────────────────────────────────────────────────

fn section_dense_bridge(bs: &[Block], runs_total: usize) {
    println!("\n8  absorbing a short mid-sleep wake run, swept — splits closed against sleep wrongly swallowed");
    println!("   gap  spans  runs  splits  merges  no-sleep  asleep caught  awake inside spans  open edge med/p90/max");
    for gap in [0i64, 10, 15, 18, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 120] {
        let p = DetectParams { wake_absorb_max_min: gap, ..DetectParams::SHIPPED };
        let (mut spans_n, mut hit, mut split, mut merged, mut spurious) = (0, 0, 0, 0, 0);
        let (mut caught, mut asleep_all, mut awake_in, mut span_s) = (0i64, 0i64, 0i64, 0i64);
        let mut open: Vec<f64> = Vec::new();
        for b in bs {
            let spans = spans_of(b, &p);
            spans_n += spans.len();
            let runs = asleep_runs(&b.band, 90, RUN_TOLERANCE_S);
            let any_run = asleep_runs(&b.band, 10, RUN_TOLERANCE_S);
            let nb: Vec<NightBlock> = spans.iter().map(|s| NightBlock { start: s.start, end: s.end }).collect();
            for g in bridged_night_groups(&nb, 0) {
                let gs = g.indices.iter().map(|&i| nb[i].start).min().unwrap_or(0);
                let ge = g.indices.iter().map(|&i| nb[i].end).max().unwrap_or(0);
                if let Some((a, _)) = runs.iter().filter(|(a, z)| gs < *z && ge > *a).min_by_key(|r| r.0) {
                    open.push((a - gs) as f64 / 60.0);
                }
            }
            asleep_all += b.band.iter().filter(|(_, s)| *s == BAND_ASLEEP).count() as i64;
            for &(a, z) in &runs {
                let cov = spans.iter().filter(|s| s.start < z && s.end > a).count();
                if cov >= 1 {
                    hit += 1;
                }
                if cov > 1 {
                    split += 1;
                }
            }
            for s in &spans {
                span_s += s.end - s.start;
                caught += band_asleep_secs(&b.band, s.start, s.end);
                awake_in += b.band.iter().filter(|(t, st)| *t >= s.start && *t <= s.end && *st != BAND_ASLEEP).count() as i64;
                let n = runs.iter().filter(|(a, z)| s.start < *z && s.end > *a).count();
                if n > 1 {
                    merged += 1;
                } else if n == 0 {
                    let seen = b.band.iter().filter(|(t, _)| *t >= s.start && *t < s.end).count() as f64;
                    if seen / ((s.end - s.start) as f64).max(1.0) >= 0.5
                        && !any_run.iter().any(|(a, z)| s.start < *z && s.end > *a)
                    {
                        spurious += 1;
                    }
                }
            }
        }
        let mark = if gap == 0 { "*" } else { " " };
        println!(
            "  {mark}{gap:>3}  {spans_n:>5}  {hit:>2}/{runs_total}  {split:>6}  {merged:>6}  {spurious:>8}  \
             {:>12.1}%  {:>7.1}% ({:>4.0} min)  {:>+6.1} {:>+6.1} {:>+6.1}",
            100.0 * caught as f64 / asleep_all.max(1) as f64,
            100.0 * awake_in as f64 / span_s.max(1) as f64,
            awake_in as f64 / 60.0,
            median(&mut open.clone()),
            pct(&mut open.clone(), 0.9),
            pct(&mut open.clone(), 1.0),
        );
    }
}

// ── 9  the whole change, scored on one balanced number ─────────────────────────────────────────────

/// One agreement scoreboard: banded seconds, hits, and each side's wake count. Raw accuracy rewards a
/// config that rarely says wake on a reference calling ~90% of these seconds asleep, so kappa is the
/// number to read and both wake rates are printed beside it.
#[derive(Default, Clone, Copy)]
struct Score {
    n: i64,
    ok: i64,
    our_wake: i64,
    band_wake: i64,
    hit: i64,
}

impl Score {
    fn add(&mut self, ours_wake: bool, strap_wake: bool) {
        self.n += 1;
        self.ok += i64::from(ours_wake == strap_wake);
        self.our_wake += i64::from(ours_wake);
        self.band_wake += i64::from(strap_wake);
        self.hit += i64::from(ours_wake && strap_wake);
    }
    fn accuracy(&self) -> f64 {
        100.0 * self.ok as f64 / self.n.max(1) as f64
    }
    /// Cohen's kappa over the 2x2 table: accuracy with the base rate divided out.
    fn kappa(&self) -> f64 {
        let n = self.n.max(1) as f64;
        let (pw, tw) = (self.our_wake as f64, self.band_wake as f64);
        let po = self.ok as f64 / n;
        let pe = (pw / n) * (tw / n) + (1.0 - pw / n) * (1.0 - tw / n);
        if pe >= 1.0 { 0.0 } else { (po - pe) / (1.0 - pe) }
    }
    fn row(&self, label: &str) {
        println!(
            "   {label:<44} {:>7}  {:>8.2}%  {:>7.3}  {:>8.1}%  {:>10.1}%",
            self.n,
            self.accuracy(),
            self.kappa(),
            100.0 * self.our_wake as f64 / self.n.max(1) as f64,
            100.0 * self.band_wake as f64 / self.n.max(1) as f64,
        );
    }
}

/// One absorb setting scored three ways over one block set, plus which spans the density gate accepted.
#[derive(Default)]
struct AbsorbRun {
    all: Score,
    dense_raw: Score,
    dense_ref: Score,
    /// The same three, restricted to band seconds every absorb setting covers, so the absorb changing
    /// what a span IS cannot move the denominator between the rows being compared.
    paired_raw: Score,
    paired_ref: Score,
    census: RefineCensus,
    spans: usize,
    dense_spans: usize,
}

/// Band seconds covered by a staged span under `p`, as a sorted list — the denominator one setting offers.
fn covered_seconds(b: &Block, p: &DetectParams, params: &Params) -> Vec<i64> {
    let mut out: Vec<i64> = Vec::new();
    for s in spans_of(b, p) {
        if stage_span(b, s.start, s.end, params).is_empty() {
            continue;
        }
        out.extend(b.band.iter().filter(|(t, _)| *t >= s.start && *t < s.end).map(|(t, _)| *t));
    }
    out.sort_unstable();
    out
}

fn absorb_run(bs: &[Block], absorb: i64, params: &Params, common: &[Vec<i64>]) -> AbsorbRun {
    let p = DetectParams { wake_absorb_max_min: absorb, ..DetectParams::SHIPPED };
    let mut r = AbsorbRun::default();
    for (bi, b) in bs.iter().enumerate() {
        for s in spans_of(b, &p) {
            let segs = stage_span(b, s.start, s.end, params);
            if segs.is_empty() {
                continue;
            }
            r.spans += 1;
            let (grav, steps) = span_streams(b, s.start, s.end);
            // The shipped gate decides, asked before the refinement rather than inferred after it.
            let mut probe = RefineCensus::default();
            let fine = probe.refine(&segs, &grav, &steps);
            let dense = probe.all_refined();
            r.census.absorb(&probe);
            if dense {
                r.dense_spans += 1;
            }
            for &(ts, st) in b.band.iter().filter(|(t, _)| *t >= s.start && *t < s.end) {
                let Some(stage) = label_at(&segs, ts) else { continue };
                let (ours, strap) = (stage == SleepStage::Wake, st != BAND_ASLEEP);
                r.all.add(ours, strap);
                if !dense {
                    continue;
                }
                r.dense_raw.add(ours, strap);
                let Some(refined) = label_at(&fine, ts) else { continue };
                let ours_ref = refined == SleepStage::Wake;
                r.dense_ref.add(ours_ref, strap);
                if common[bi].binary_search(&ts).is_ok() {
                    r.paired_raw.add(ours, strap);
                    r.paired_ref.add(ours_ref, strap);
                }
            }
        }
    }
    r
}

fn section_agreement(bs: &[Block]) {
    println!("\n9  every detected span staged and scored against the strap's own asleep/wake call");
    println!("   Three paths per setting. `stage_v2, all spans` is what the absorb was shipped on. The dense");
    println!("   rows restrict to spans whose step stream passes the refinement's density gate, so the");
    println!("   stage_v2 one is the control for `+ refine_wake`, which is the path `analyze` runs. A span the");
    println!("   gate declines is NOT the app's path and is never pooled into a refined row.");
    println!("   The PAIRED rows drop every second only one setting covers, since the absorb changes what a");
    println!("   span is; those two rows are the only strict within-comparison the corpus allows.");
    let params = Params::SHIPPED;
    let settings = [0i64, DetectParams::SHIPPED.wake_absorb_max_min];
    // Seconds every setting covers, per block.
    let common: Vec<Vec<i64>> = bs
        .iter()
        .map(|b| {
            let mut acc = covered_seconds(b, &DetectParams { wake_absorb_max_min: settings[0], ..DetectParams::SHIPPED }, &params);
            for &a in &settings[1..] {
                let next = covered_seconds(b, &DetectParams { wake_absorb_max_min: a, ..DetectParams::SHIPPED }, &params);
                acc.retain(|t| next.binary_search(t).is_ok());
            }
            acc
        })
        .collect();

    println!("\n   setting / path                                 epochs  accuracy   kappa2  our wake%  strap wake%");
    let runs: Vec<(i64, AbsorbRun)> =
        settings.iter().map(|&a| (a, absorb_run(bs, a, &params, &common))).collect();
    for (absorb, r) in &runs {
        let mark = if *absorb == 0 { "absorb 0 (before)" } else { "absorb 45 (shipped)" };
        r.all.row(&format!("{mark}, stage_v2, all {} spans", r.spans));
        r.dense_raw.row(&format!("{mark}, stage_v2, {} dense spans", r.dense_spans));
        r.dense_ref.row(&format!("{mark}, + refine_wake (the app)"));
        r.paired_raw.row(&format!("{mark}, PAIRED seconds, stage_v2"));
        r.paired_ref.row(&format!("{mark}, PAIRED seconds, + refine_wake"));
        println!("{}", r.census.line(mark));
        // The refined rows are built only from spans the gate accepted, and this is what says so.
        assert_eq!(r.census.refined, r.dense_spans, "{mark}: a declined span reached the refined row");
    }
    let (a, z) = (&runs[0].1, &runs[1].1);
    println!("\n   what the absorb is worth, as a delta from 0 to 45 min:");
    for (label, before, after) in [
        ("stage_v2, all spans (how it shipped)", &a.all, &z.all),
        ("stage_v2, dense spans", &a.dense_raw, &z.dense_raw),
        ("+ refine_wake, dense spans", &a.dense_ref, &z.dense_ref),
        ("stage_v2, PAIRED seconds", &a.paired_raw, &z.paired_raw),
        ("+ refine_wake, PAIRED seconds", &a.paired_ref, &z.paired_ref),
    ] {
        println!(
            "   {label:<40} accuracy {:+.2} pp   kappa2 {:+.3}",
            after.accuracy() - before.accuracy(),
            after.kappa() - before.kappa(),
        );
    }
}

fn main() {
    let bs = blocks();
    let runs_total: usize = bs.iter().map(|b| asleep_runs(&b.band, 90, RUN_TOLERANCE_S).len()).sum();
    println!("{} band-carrying wear blocks, {runs_total} strap asleep runs >= 90 min\n", bs.len());

    section_run_lengths(&bs);
    section_min_sleep(&bs, runs_total);
    section_cardinality(&bs);
    section_bridging(&bs);
    println!("\n5  the open edge with NO run dropped — each run scored against its own bridged group.");
    println!("   The solo-only view excluded every bridged run, which biased the error LOW; this scores all.");
    section_open_edge(&bs, runs_total, &DetectParams::PRE_HYSTERESIS, "pre-hysteresis");
    section_open_edge(&bs, runs_total, &DetectParams::SHIPPED, "shipped");
    let (k0, a0, d0) = section_selection(&bs, 0, true);
    let (k2, a2, d2) = section_selection(&bs, 2 * 3600, false);
    println!(
        "   offset sensitivity: at +2 h {d2} days (vs {d0}), {:.1} h kept of {:.1} h (vs {:.1} of {:.1})",
        k2 as f64 / 3600.0,
        a2 as f64 / 3600.0,
        k0 as f64 / 3600.0,
        a0 as f64 / 3600.0
    );
    section_split_cost(&bs);
    section_dense_bridge(&bs, runs_total);
    section_agreement(&bs);
}
