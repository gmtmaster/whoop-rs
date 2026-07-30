//! Step E — the staging window: where the epoch grid's edges fall, and everything that depends on it.
//!
//!   cargo run --release -p physio-algo --example stage_window
//!
//! 1  epoch-grid origins: the stager's grid against every reference grid this project scores against
//! 2  grid PHASE sensitivity: the same night shifted 0..29 s, so content is identical and only the
//!    epoch boundaries move — which separates a grid artefact from a window effect
//! 3  window sensitivity reproduced, then decomposed by ablating each channel the window feeds
//! 4  the two main-night scorers, adjudicated against the strap's own asleep runs
//!
//! Sections 1-3 read `ours` (window baked in, so they measure sensitivity, not our error); 1 and 4
//! read `continuous` (real unix seconds, whole wear blocks, the band as reference).

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use physio_algo::sleep::{
    detect_sessions_with, main_night_group_indices, main_night_group_indices_scored, params::Params, prepare_v2,
    stage_v2_prepared, AccelSample, DetectParams, DetectedSpan, HrSample, RrRun, ScoredNightBlock, SleepInput,
    SleepStage, StageSegment,
};

/// The stager's epoch width, and the band code for asleep.
const EPOCH_SEC: i64 = 30;
const BAND_ASLEEP: i32 = 2;
/// Shortest band asleep run the flow is scored against (minutes).
const MIN_RUN_MIN: i64 = 90;

fn root(set: &str) -> PathBuf {
    std::env::var("WHOOP_SLEEP_FIXTURES")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("C:/Users/DavidGillot/Projects/whoop/sleep-benchmark/fixtures_multi"))
        .join(set)
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

fn dirs_of(set: &str) -> Vec<PathBuf> {
    let mut d: Vec<PathBuf> = fs::read_dir(root(set))
        .map(|rd| rd.filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.is_dir()).collect())
        .unwrap_or_default();
    d.sort();
    d
}

fn median(v: &mut [f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// The stager's first epoch start for a window opening at `start`: the first multiple of `EPOCH_SEC`
/// at or after it. Mirrors `v2::features`, which anchors the grid to ABSOLUTE unix time, not to `start`.
fn first_epoch(start: i64) -> i64 {
    ((start + EPOCH_SEC - 1) / EPOCH_SEC) * EPOCH_SEC
}

/// Seconds a reference epoch shares with the stager epoch a MIDPOINT lookup lands in, given the stager
/// grid sitting `offset` seconds into each reference epoch. The midpoint picks the larger of the two
/// overlapping epochs, so this is the best a time-keyed lookup can do at that offset.
fn midpoint_overlap(offset: i64) -> i64 {
    if offset > EPOCH_SEC / 2 {
        offset
    } else {
        EPOCH_SEC - offset
    }
}

// ── the `ours` rig: one night's streams plus its band truth per epoch ──────────────────────────────

struct Night {
    name: String,
    w0: i64,
    w1: i64,
    n: usize,
    hr: Vec<HrSample>,
    rr: Vec<RrRun>,
    accel: Vec<AccelSample>,
    /// Band label per epoch index: 0 wake, 1 asleep. Absent epochs are unlabelled.
    truth: BTreeMap<usize, i32>,
}

fn load_night(d: &Path) -> Option<Night> {
    let meta = fs::read_to_string(d.join("meta.txt")).ok()?;
    let m: Vec<i64> = meta.split_whitespace().map(|x| x.parse().unwrap()).collect();
    let accel: Vec<AccelSample> = read_csv(&d.join("gravity.csv"))
        .iter()
        .map(|r| AccelSample { ts: r[0] as i64, x: r[1], y: r[2], z: r[3] })
        .collect();
    let hr: Vec<HrSample> =
        read_csv(&d.join("hr.csv")).iter().map(|r| HrSample { ts: r[0] as i64, bpm: r[1] as u16 }).collect();
    let mut rr: Vec<RrRun> = Vec::new();
    for row in read_csv(&d.join("rr.csv")) {
        let (ts, ms) = (row[0] as i64, row[1] as u16);
        match rr.last_mut() {
            Some(l) if l.ts == ts => l.intervals.push(ms),
            _ => rr.push(RrRun { ts, intervals: vec![ms] }),
        }
    }
    let truth = read_csv(&d.join("truth.csv")).iter().map(|r| (r[0] as usize, r[1] as i32)).collect();
    Some(Night {
        name: d.file_name().unwrap().to_string_lossy().chars().take(38).collect(),
        w0: m[1],
        w1: m[2],
        n: m[3] as usize,
        hr,
        rr,
        accel,
        truth,
    })
}

fn nights() -> Vec<Night> {
    dirs_of("ours").iter().filter_map(|d| load_night(d)).collect()
}

fn stage_idx(s: SleepStage) -> usize {
    match s {
        SleepStage::Wake => 0,
        SleepStage::Light => 1,
        SleepStage::Deep => 2,
        SleepStage::Rem => 3,
    }
}

/// One label per epoch of `[w0, w0 + n*EPOCH_SEC)`, read at each epoch's MIDPOINT out of a staging's
/// wall-clock segments — so a caller's grid and the stager's grid are matched by time, not by index.
fn labels_at(segs: &[StageSegment], w0: i64, n: usize) -> Vec<usize> {
    (0..n)
        .map(|k| {
            let mid = w0 + k as i64 * EPOCH_SEC + EPOCH_SEC / 2;
            stage_idx(
                segs.iter()
                    .find(|s| s.start <= mid && mid < s.end)
                    .map(|s| s.stage)
                    .unwrap_or_else(|| segs.last().expect("staging is never empty here").stage),
            )
        })
        .collect()
}

/// Shift every timestamp of a night by `d` seconds. Content, span and epoch count are untouched; only
/// which seconds fall into which absolute epoch moves.
fn shifted(n: &Night, d: i64) -> (SleepInput, i64) {
    let input = SleepInput {
        start: n.w0 + d,
        end: n.w1 + d,
        hr: n.hr.iter().map(|h| HrSample { ts: h.ts + d, bpm: h.bpm }).collect(),
        rr: n.rr.iter().map(|r| RrRun { ts: r.ts + d, intervals: r.intervals.clone() }).collect(),
        accel: n.accel.iter().map(|g| AccelSample { ts: g.ts + d, x: g.x, y: g.y, z: g.z }).collect(),
    };
    (input, n.w0 + d)
}

fn stage(input: &SleepInput, p: &Params) -> Vec<StageSegment> {
    stage_v2_prepared(&prepare_v2(input, p), p)
}

/// The band's own asleep runs, long enough to score, as `(first, last)` epoch indices.
fn truth_run(n: &Night) -> Option<(usize, usize)> {
    let asleep: Vec<usize> = n.truth.iter().filter(|(_, &v)| v == 1).map(|(&k, _)| k).collect();
    if asleep.len() < 240 {
        return None;
    }
    Some((asleep[0], asleep[asleep.len() - 1]))
}

// ── 1  epoch-grid origins ─────────────────────────────────────────────────────────────────────────

fn section_grid_origins() {
    println!("1  epoch-grid origins — does the stager's grid share one with what scores it");
    println!("   The stager anchors epochs to ABSOLUTE unix time (first epoch = ceil(start/30)*30), so a");
    println!("   window edge moving does NOT move the grid. Every reference grid here is anchored at the");
    println!("   SPAN START instead, so the two share an origin only when start % 30 == 0.");

    let ns = nights();
    let mut phases: BTreeMap<i64, usize> = BTreeMap::new();
    for n in &ns {
        *phases.entry(first_epoch(n.w0) - n.w0).or_default() += 1;
    }
    println!(
        "\n   ours ({} nights): window start % 30 -> stager grid offset {:?}",
        ns.len(),
        phases.iter().map(|(o, c)| format!("{o} s x{c}")).collect::<Vec<_>>()
    );
    println!(
        "   truth.csv epoch k covers [w0+30k, w0+30k+30); the stager epoch a midpoint lookup lands in"
    );
    println!(
        "   overlaps it by {} s of 30, every night — so a third of every scored epoch is labelled from",
        ns.first().map(|n| midpoint_overlap(first_epoch(n.w0) - n.w0)).unwrap_or(0)
    );
    println!("   data outside it. The lookup is already the better of the two overlapping epochs.");

    // The continuous blocks keep real unix seconds, so their detected spans sample the phase honestly.
    let mut cont_phase: BTreeMap<i64, usize> = BTreeMap::new();
    let mut spans_seen = 0usize;
    for d in dirs_of("continuous") {
        let band: Vec<(i64, i32)> = read_csv(&d.join("band.csv")).iter().map(|r| (r[0] as i64, r[1] as i32)).collect();
        if band.is_empty() {
            continue;
        }
        let accel: Vec<AccelSample> = read_csv(&d.join("gravity.csv"))
            .iter()
            .map(|r| AccelSample { ts: r[0] as i64, x: r[1], y: r[2], z: r[3] })
            .collect();
        if accel.len() < 120 {
            continue;
        }
        let hr: Vec<HrSample> =
            read_csv(&d.join("hr.csv")).iter().map(|r| HrSample { ts: r[0] as i64, bpm: r[1] as u16 }).collect();
        for s in detect_sessions_with(&hr, &accel, 0, &[], &band, None, &DetectParams::SHIPPED) {
            *cont_phase.entry(first_epoch(s.start) - s.start).or_default() += 1;
            spans_seen += 1;
        }
    }
    let aligned = cont_phase.get(&0).copied().unwrap_or(0);
    let mut ov: Vec<f64> = cont_phase.iter().flat_map(|(o, c)| vec![midpoint_overlap(*o) as f64; *c]).collect();
    println!(
        "\n   continuous ({spans_seen} detected spans, real unix seconds): {} distinct grid offsets, {aligned} \
         span(s) land on the grid exactly",
        cont_phase.len()
    );
    println!(
        "   offsets present: {}",
        cont_phase.keys().map(|o| format!("{o}")).collect::<Vec<_>>().join(" ")
    );
    println!(
        "   median epoch overlap with a span-anchored reference grid: {:.0} s of 30",
        median(&mut ov)
    );
    println!("   So in the app the offset is arbitrary per span, not the fixture's fixed 20 s.");
}

// ── 2  grid PHASE sensitivity ─────────────────────────────────────────────────────────────────────

fn section_grid_phase() {
    println!("\n2  grid PHASE sensitivity — the same night, shifted 0..29 s");
    println!("   Every timestamp moves together, so the data, the span and the epoch count are identical");
    println!("   and only the epoch BOUNDARIES move against the samples. Anything that changes here is");
    println!("   the grid, not the window. Labels are read at each truth epoch's midpoint in shifted time.");
    let p = Params::SHIPPED;
    // The same 21 band-labelled nights section 3 scores, so the two numbers share a denominator.
    let ns: Vec<Night> = nights().into_iter().filter(|n| truth_run(n).is_some()).collect();
    let mut per_shift: Vec<(i64, f64, f64)> = Vec::new();
    // Per-night worst movement over the sweep, so one bad night cannot hide in a mean.
    let mut worst_per_night: Vec<f64> = Vec::new();
    let mut base_labels: Vec<Vec<usize>> = Vec::new();
    let mut agreement: Vec<(i64, f64)> = Vec::new();

    for n in &ns {
        let (input, w0) = shifted(n, 0);
        base_labels.push(labels_at(&stage(&input, &p), w0, n.n));
    }
    for d in 0..EPOCH_SEC {
        let (mut moved, mut total) = (0usize, 0usize);
        let (mut agree, mut scored) = (0usize, 0usize);
        let mut per_night: Vec<f64> = Vec::new();
        for (i, n) in ns.iter().enumerate() {
            let (input, w0) = shifted(n, d);
            let lab = labels_at(&stage(&input, &p), w0, n.n);
            let ch = (0..n.n).filter(|&k| lab[k] != base_labels[i][k]).count();
            moved += ch;
            total += n.n;
            per_night.push(100.0 * ch as f64 / n.n.max(1) as f64);
            for (&k, &t) in &n.truth {
                if k < n.n {
                    scored += 1;
                    agree += usize::from((lab[k] == 0) == (t == 0));
                }
            }
        }
        per_shift.push((d, 100.0 * moved as f64 / total.max(1) as f64, median(&mut per_night.clone())));
        agreement.push((d, 100.0 * agree as f64 / scored.max(1) as f64));
        if d == 0 {
            worst_per_night = vec![0.0; ns.len()];
        }
        for (i, v) in per_night.iter().enumerate() {
            if *v > worst_per_night[i] {
                worst_per_night[i] = *v;
            }
        }
    }
    println!("\n   shift  labels moved vs shift 0   median/night   two-class agreement with the band");
    for (d, all, med) in &per_shift {
        let ag = agreement.iter().find(|(x, _)| x == d).map(|(_, a)| *a).unwrap_or(f64::NAN);
        let mark = if (ns[0].w0 + d).rem_euclid(EPOCH_SEC) == 0 { "  <- grid ON the reference epochs" } else { "" };
        println!("   {d:>4} s  {all:>18.2}%  {med:>13.2}%  {ag:>32.2}%{mark}");
    }
    let all_moved: Vec<f64> = per_shift.iter().skip(1).map(|(_, a, _)| *a).collect();
    println!(
        "\n   over the 29 non-zero shifts: median {:.2}% of labels move, worst {:.2}%",
        median(&mut all_moved.clone()),
        all_moved.iter().copied().fold(0.0, f64::max)
    );
    let mut w = worst_per_night.clone();
    let worst_i = (0..worst_per_night.len()).max_by(|&a, &b| worst_per_night[a].partial_cmp(&worst_per_night[b]).unwrap());
    println!(
        "   per night, worst shift: median {:.2}%, worst night {:.2}% ({})",
        median(&mut w),
        worst_i.map(|i| worst_per_night[i]).unwrap_or(0.0),
        worst_i.map(|i| ns[i].name.clone()).unwrap_or_default()
    );
    let (lo, hi) = (
        agreement.iter().map(|(_, a)| *a).fold(f64::INFINITY, f64::min),
        agreement.iter().map(|(_, a)| *a).fold(f64::NEG_INFINITY, f64::max),
    );
    println!("   band agreement across all 30 phases: {lo:.2}% to {hi:.2}% (spread {:.2} pp)", hi - lo);
}

// ── 3  window sensitivity, and which channel carries it ───────────────────────────────────────────

/// Stage each night twice — over the fixture window and over the strap's own asleep run — and count the
/// SHARED interior epochs that disagree. The two stagings share an absolute grid, so a disagreement is
/// the window moving a normalisation, never an index slipping.
struct WindowEffect {
    nights: usize,
    shared: usize,
    moved: usize,
    median_night: f64,
    /// Two-class agreement with the band over the shared interior, staged over the full window and over
    /// the strap's own run — what the relabelling actually costs.
    agree_full: f64,
    agree_trim: f64,
}

fn window_effect(ns: &[Night], p: &Params) -> WindowEffect {
    let (mut nights, mut shared, mut moved) = (0usize, 0usize, 0usize);
    let (mut ok_full, mut ok_trim, mut scored) = (0usize, 0usize, 0usize);
    let mut per_night: Vec<f64> = Vec::new();
    for n in ns {
        let Some((first, last)) = truth_run(n) else { continue };
        if first == 0 && last + 1 >= n.n {
            continue;
        }
        let full = SleepInput {
            start: n.w0,
            end: n.w1,
            hr: n.hr.clone(),
            rr: n.rr.clone(),
            accel: n.accel.clone(),
        };
        let (t0, t1) = (n.w0 + first as i64 * EPOCH_SEC, n.w0 + (last + 1) as i64 * EPOCH_SEC);
        let trim = SleepInput { start: t0, end: t1, ..full.clone() };
        let inner = last + 1 - first;
        let a = labels_at(&stage(&full, p), n.w0, n.n);
        let b = labels_at(&stage(&trim, p), t0, inner);
        let changed = (0..inner).filter(|&k| a[first + k] != b[k]).count();
        for k in 0..inner {
            let Some(&t) = n.truth.get(&(first + k)) else { continue };
            scored += 1;
            ok_full += usize::from((a[first + k] == 0) == (t == 0));
            ok_trim += usize::from((b[k] == 0) == (t == 0));
        }
        nights += 1;
        shared += inner;
        moved += changed;
        per_night.push(100.0 * changed as f64 / inner as f64);
    }
    WindowEffect {
        nights,
        shared,
        moved,
        median_night: median(&mut per_night),
        agree_full: 100.0 * ok_full as f64 / scored.max(1) as f64,
        agree_trim: 100.0 * ok_trim as f64 / scored.max(1) as f64,
    }
}

fn section_window_channels() {
    println!("\n3  window sensitivity, reproduced then decomposed");
    println!("   Each night staged over the fixture window and over the strap's own asleep run, comparing");
    println!("   only the epochs both contain. The window reaches staging through four channels and this");
    println!("   ablates three of them by zeroing the coefficient that reads them.");
    let ns = nights();
    let shipped = Params::SHIPPED;

    // The clock channel: `clock` is (epoch - window start) / span, read ONLY by the cycle prior.
    let no_clock = Params { cycle_deep_scale: 0.0, cycle_rem_scale: 0.0, cycle_rem_early_penalty: 0.0, ..shipped };
    // The deep-eligibility percentile, ranked over the window's own epochs.
    let no_gate = Params { deep_gate_slope: 0.0, ..shipped };
    // The motion channel: the per-window median jerk sets the movement threshold, so trimming moves
    // move_frac itself as well as its z-score.
    let no_motion = Params {
        deep_motion: 0.0,
        rem_motion: 0.0,
        awake_motion: 0.0,
        motion_gate_boost: 0.0,
        jerk_gate_mult: 0.0,
        ..shipped
    };
    let cardiac_only = Params {
        cycle_deep_scale: 0.0,
        cycle_rem_scale: 0.0,
        cycle_rem_early_penalty: 0.0,
        deep_gate_slope: 0.0,
        deep_motion: 0.0,
        rem_motion: 0.0,
        awake_motion: 0.0,
        motion_gate_boost: 0.0,
        jerk_gate_mult: 0.0,
        ..shipped
    };

    // F3: the same prior read from detected sleep onset instead of from the window start, which is the
    // one change that can move the clock channel without switching it off.
    let from_onset = Params { cycle_clock_from_onset: true, ..shipped };
    let onset_guard = Params {
        cycle_clock_from_onset: true,
        cycle_rem_early_penalty: 4.0,
        cycle_rem_onset_minutes: 60.0,
        ..shipped
    };

    println!("\n   config                                 nights  shared   moved  median/night   band agreement");
    println!("                                                                                  full / trimmed");
    for (label, p) in [
        ("SHIPPED", &shipped),
        ("- clock prior (cycle deep+REM off)", &no_clock),
        ("- deep percentile gate", &no_gate),
        ("- motion (median jerk + its z)", &no_motion),
        ("cardiac z-scores ONLY", &cardiac_only),
        ("clock from ONSET (F3)", &from_onset),
        ("clock + guard from ONSET (F3+F1)", &onset_guard),
    ] {
        let w = window_effect(&ns, p);
        println!(
            "   {label:<38} {:>4}  {:>6}  {:>5.1}%  {:>11.1}%   {:>5.2}% / {:.2}%",
            w.nights,
            w.shared,
            100.0 * w.moved as f64 / w.shared.max(1) as f64,
            w.median_night,
            w.agree_full,
            w.agree_trim,
        );
    }
    println!("\n   A per-night z-score is a function of the epochs in the window by construction, so the");
    println!("   residual under `cardiac z-scores ONLY` is the floor no window change can go below. The");
    println!("   agreement columns are what the relabelling COSTS: labels move, the score barely does.");
}

// ── 4  the two main-night scorers, against the strap ──────────────────────────────────────────────

struct ContBlock {
    name: String,
    hr: Vec<HrSample>,
    rr: Vec<RrRun>,
    accel: Vec<AccelSample>,
    band: Vec<(i64, i32)>,
}

fn load_cont(d: &Path) -> Option<ContBlock> {
    let band: Vec<(i64, i32)> = read_csv(&d.join("band.csv")).iter().map(|r| (r[0] as i64, r[1] as i32)).collect();
    if band.is_empty() {
        return None;
    }
    let accel: Vec<AccelSample> = read_csv(&d.join("gravity.csv"))
        .iter()
        .map(|r| AccelSample { ts: r[0] as i64, x: r[1], y: r[2], z: r[3] })
        .collect();
    if accel.len() < 120 {
        return None;
    }
    let hr: Vec<HrSample> =
        read_csv(&d.join("hr.csv")).iter().map(|r| HrSample { ts: r[0] as i64, bpm: r[1] as u16 }).collect();
    let mut rr: Vec<RrRun> = Vec::new();
    for row in read_csv(&d.join("rr.csv")) {
        let (ts, ms) = (row[0] as i64, row[1] as u16);
        match rr.last_mut() {
            Some(l) if l.ts == ts => l.intervals.push(ms),
            _ => rr.push(RrRun { ts, intervals: vec![ms] }),
        }
    }
    Some(ContBlock { name: d.file_name().unwrap().to_string_lossy().into_owned(), hr, rr, accel, band })
}

fn band_asleep_secs(band: &[(i64, i32)], a: i64, b: i64) -> i64 {
    band.iter().filter(|(t, s)| *t >= a && *t <= b && *s == BAND_ASLEEP).count() as i64
}

fn local_day(ts: i64, offset_s: i64) -> i64 {
    (ts + offset_s).div_euclid(86_400)
}

/// Asleep and in-bed seconds a staging decodes: the numbers the stages-scored pick reads, from the
/// stage segments rather than from the clock.
fn decoded(segs: &[StageSegment]) -> (f64, f64) {
    let mut asleep = 0.0;
    let mut in_bed = 0.0;
    for s in segs {
        let d = (s.end - s.start) as f64;
        in_bed += d;
        if s.stage != SleepStage::Wake {
            asleep += d;
        }
    }
    (asleep, in_bed)
}

fn section_main_night_scorers() {
    println!("\n4  the two main-night scorers, adjudicated against the strap");
    println!("   `main_night_index` scores a candidate's CLOCK span; the stages path scores its DECODED");
    println!("   asleep time. Both share one bridge, so this isolates the score. Every day with more than");
    println!("   one candidate span is scored by how much of its strap-asleep time the pick keeps.");
    let p = Params::SHIPPED;
    let offset_s = 0i64;
    let (mut days, mut multi) = (0usize, 0usize);
    let (mut clock_kept, mut stages_kept, mut avail) = (0i64, 0i64, 0i64);
    let mut differ: Vec<String> = Vec::new();
    let mut zero_sleep = 0usize;
    let cont: Vec<ContBlock> = dirs_of("continuous").iter().filter_map(|d| load_cont(d)).collect();

    for b in &cont {
        let spans = detect_sessions_with(&b.hr, &b.accel, 0, &[], &b.band, None, &DetectParams::SHIPPED);
        // A candidate the strap never called asleep at all: the clock score cannot see that it is empty.
        zero_sleep += spans
            .iter()
            .filter(|s| band_asleep_secs(&b.band, s.start, s.end) == 0 && s.end - s.start >= MIN_RUN_MIN * 60 / 2)
            .count();
        let mut keys: Vec<i64> = spans.iter().map(|s| local_day(s.end, offset_s)).collect();
        keys.sort_unstable();
        keys.dedup();
        for d in keys {
            let matched: Vec<&DetectedSpan> = spans.iter().filter(|s| local_day(s.end, offset_s) == d).collect();
            let clock: Vec<physio_algo::sleep::NightBlock> =
                matched.iter().map(|s| physio_algo::sleep::NightBlock { start: s.start, end: s.end }).collect();
            let scored: Vec<ScoredNightBlock> = matched
                .iter()
                .map(|s| {
                    let input = SleepInput {
                        start: s.start,
                        end: s.end,
                        hr: b.hr.iter().filter(|h| h.ts >= s.start && h.ts < s.end).cloned().collect(),
                        rr: b.rr.iter().filter(|r| r.ts >= s.start && r.ts < s.end).cloned().collect(),
                        accel: b.accel.iter().filter(|g| g.ts >= s.start && g.ts < s.end).cloned().collect(),
                    };
                    let (asleep_s, in_bed_s) = if input.hr.len() < 120 || input.accel.len() < 120 {
                        (0.0, (s.end - s.start) as f64)
                    } else {
                        decoded(&stage(&input, &p))
                    };
                    ScoredNightBlock { onset: s.start, asleep_s, in_bed_s }
                })
                .collect();
            let Some(by_clock) = main_night_group_indices(&clock, offset_s, None) else { continue };
            let Some(by_stages) = main_night_group_indices_scored(&scored, offset_s, None) else { continue };
            days += 1;
            if clock.len() < 2 {
                continue;
            }
            multi += 1;
            let day_avail: i64 = clock.iter().map(|c| band_asleep_secs(&b.band, c.start, c.end)).sum();
            let ck: i64 = by_clock.iter().map(|&i| band_asleep_secs(&b.band, clock[i].start, clock[i].end)).sum();
            let sk: i64 = by_stages.iter().map(|&i| band_asleep_secs(&b.band, clock[i].start, clock[i].end)).sum();
            avail += day_avail;
            clock_kept += ck;
            stages_kept += sk;
            if by_clock != by_stages {
                differ.push(format!(
                    "   {} day {d}: clock={by_clock:?} keeps {:.2} h · stages={by_stages:?} keeps {:.2} h of {:.2} h",
                    b.name,
                    ck as f64 / 3600.0,
                    sk as f64 / 3600.0,
                    day_avail as f64 / 3600.0
                ));
            }
        }
    }
    println!("\n   {days} local days, {multi} with more than one candidate span");
    println!("   days the two scorers pick differently: {}", differ.len());
    for line in &differ {
        println!("{line}");
    }
    println!(
        "   strap-asleep kept over the multi-candidate days: clock {:.1} h · stages {:.1} h of {:.1} h",
        clock_kept as f64 / 3600.0,
        stages_kept as f64 / 3600.0,
        avail as f64 / 3600.0
    );
    println!("   detected spans >= {} min the strap called asleep for 0 s: {zero_sleep}", MIN_RUN_MIN / 2);
}

fn main() {
    section_grid_origins();
    section_grid_phase();
    section_window_channels();
    section_main_night_scorers();
}
