//! Does splitting the stillness threshold into separate enter/exit fractions remove the detector's early
//! open? Sweeps the pair and scores every cell against two independent references, then re-labels the
//! shared interior under the old and new windows.
//!
//!   cargo run --release -p physio-algo --example detect_hysteresis
//!
//! `continuous` + the strap's own asleep runs answer "do we agree with the band", on both edges.
//! `dreamt` + its PSG column answer "are we right", on the open edge only — those recordings end at wake,
//! so they cannot judge a close. Neither set is cut to a session window, so both can see a window error.
//!
//! The interior-churn rows stage with `stage_v2` ONLY. They compare two DETECTION windows against each
//! other, and `refine_wake` shrinks wake on both sides of that comparison, so it would mask the
//! relabelling rather than measure it. Not the app's absolute wake figure — see `emit_wake` for that.

mod common;

use common::{read_accel, read_csv, read_hr, read_rr};

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

use physio_algo::sleep::{
    detect_sessions_with, params::Params, prepare_v2, stage_v2_prepared, AccelSample, DetectParams,
    DetectedSpan, HrSample, RrRun, SleepInput, SleepStage,
};

const BAND_ASLEEP: i32 = 2;
const PSG_WAKE: i32 = 0;
const RUN_MIN_MINUTES: i64 = 90;
/// A scored non-wake stretch this long fixes PSG onset, matching the staging rule.
const ONSET_RUN_EPOCHS: usize = 10;

fn subdirs(ds: &str) -> Vec<PathBuf> {
    let mut dirs: Vec<PathBuf> = fs::read_dir(common::fixtures_root().join(ds))
        .map(|rd| rd.filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.is_dir()).collect())
        .unwrap_or_default();
    dirs.sort();
    dirs
}

struct Block {
    name: String,
    hr: Vec<HrSample>,
    rr: Vec<RrRun>,
    accel: Vec<AccelSample>,
    band: Vec<(i64, i32)>,
}

struct PsgNight {
    hr: Vec<HrSample>,
    accel: Vec<AccelSample>,
    /// PSG onset, first scored epoch and last scored non-wake epoch, as wall-clock seconds.
    onset: i64,
    first_scored: i64,
    last_sleep: i64,
}

fn load_blocks() -> Vec<Block> {
    subdirs("continuous")
        .iter()
        .map(|d| Block {
            name: d.file_name().unwrap().to_string_lossy().into_owned(),
            hr: read_hr(d),
            rr: read_rr(d),
            accel: read_accel(d),
            band: read_csv(&d.join("band.csv")).iter().map(|r| (r[0] as i64, r[1] as i32)).collect(),
        })
        .filter(|b| !b.band.is_empty() && b.accel.len() >= 120)
        .collect()
}

fn load_psg_nights() -> Vec<PsgNight> {
    let mut out = Vec::new();
    for d in subdirs("dreamt") {
        let Ok(meta) = fs::read_to_string(d.join("meta.txt")) else { continue };
        let m: Vec<i64> = meta.split_whitespace().map(|x| x.parse().unwrap()).collect();
        let w0 = m[1];
        let truth: BTreeMap<i64, i32> =
            read_csv(&d.join("truth.csv")).iter().map(|r| (r[0] as i64, r[1] as i32)).collect();
        if truth.is_empty() {
            continue;
        }
        let mut run = 0usize;
        let mut onset = None;
        for (k, &v) in &truth {
            run = if v == PSG_WAKE { 0 } else { run + 1 };
            if run == ONSET_RUN_EPOCHS {
                onset = Some(k - ONSET_RUN_EPOCHS as i64 + 1);
                break;
            }
        }
        let (Some(onset), Some(first), Some(last)) = (
            onset,
            truth.keys().next().copied(),
            truth.iter().filter(|(_, v)| **v != PSG_WAKE).map(|(k, _)| *k).next_back(),
        ) else {
            continue;
        };
        out.push(PsgNight {
            hr: read_hr(&d),
            accel: read_accel(&d),
            onset: w0 + onset * 30,
            first_scored: w0 + first * 30,
            last_sleep: w0 + (last + 1) * 30,
        });
    }
    out
}

/// Contiguous stretches the strap itself called asleep, at least `min_min` long, tolerating a 5-minute
/// interruption. This is the band-side reference.
fn asleep_runs(band: &[(i64, i32)], min_min: i64) -> Vec<(i64, i64)> {
    let (mut out, mut start, mut last) = (Vec::new(), None::<i64>, 0i64);
    for &(ts, st) in band {
        if st != BAND_ASLEEP {
            continue;
        }
        match start {
            None => start = Some(ts),
            Some(s) if ts - last > 300 => {
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

fn median(v: &mut [f64]) -> f64 {
    pct(v, 0.5)
}

fn pct(v: &mut [f64], p: f64) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[((v.len() - 1) as f64 * p) as usize]
}

#[derive(Default)]
struct BandScore {
    /// Signed minutes; `+` = we opened before the strap called asleep, `-` = we opened after it.
    head: Vec<f64>,
    /// Signed minutes; `+` = the strap was still asleep after we closed.
    tail: Vec<f64>,
    runs: usize,
    hit: usize,
    /// Block, span minutes and band coverage of each span the band judged and found no sleep in.
    spurious: Vec<(String, f64, f64)>,
    /// Runs excluded from `head` because one span covers them together with another run.
    bridged: usize,
}

fn score_band(blocks: &[Block], p: &DetectParams) -> BandScore {
    let mut s = BandScore::default();
    for b in blocks {
        let spans = detect_sessions_with(&b.hr, &b.accel, 0, &[], &b.band, None, p);
        let runs = asleep_runs(&b.band, RUN_MIN_MINUTES);
        let any_run = asleep_runs(&b.band, 10);
        s.runs += runs.len();
        for &(a, z) in &runs {
            let cov: Vec<&DetectedSpan> = spans.iter().filter(|x| x.start < z && x.end > a).collect();
            if cov.is_empty() {
                continue;
            }
            s.hit += 1;
            let (s0, s1) = (cov.iter().map(|x| x.start).min().unwrap(), cov.iter().map(|x| x.end).max().unwrap());
            s.tail.push((z - s1) as f64 / 60.0);
            // Head only where the span covers this run alone: a span bridging two runs legitimately starts
            // long before the second, which would otherwise read as a huge early open.
            if cov.len() == 1 && runs.iter().filter(|(x, y)| s0 < *y && s1 > *x).count() == 1 {
                s.head.push((a - s0) as f64 / 60.0);
            } else {
                s.bridged += 1;
            }
        }
        for x in &spans {
            let n = runs.iter().filter(|(a, z)| x.start < *z && x.end > *a).count();
            if n == 0 {
                // A span the band barely covers cannot be judged either way, and a short run is a nap a
                // detector is right to find; neither is a false positive.
                let seen = b.band.iter().filter(|(t, _)| *t >= x.start && *t < x.end).count() as f64;
                let cover = seen / ((x.end - x.start) as f64).max(1.0);
                let nap = any_run.iter().any(|(a, z)| x.start < *z && x.end > *a);
                if cover >= 0.5 && !nap {
                    s.spurious.push((b.name.clone(), (x.end - x.start) as f64 / 60.0, cover));
                }
            }
        }
    }
    s
}

#[derive(Default)]
struct PsgScore {
    /// Signed minutes; `+` = we opened before PSG onset.
    head: Vec<f64>,
    missed: usize,
    /// Opens that land before the first PSG-scored epoch, i.e. in the unscored pre-recording lead-in.
    before_scoring: usize,
}

fn score_psg(nights: &[PsgNight], p: &DetectParams) -> PsgScore {
    let mut s = PsgScore::default();
    for n in nights {
        let spans = detect_sessions_with(&n.hr, &n.accel, 0, &[], &[], None, p);
        match spans.iter().filter(|x| x.start < n.last_sleep && x.end > n.onset).map(|x| x.start).min() {
            Some(start) => {
                s.head.push((n.onset - start) as f64 / 60.0);
                s.before_scoring += usize::from(start < n.first_scored);
            }
            None => s.missed += 1,
        }
    }
    s
}

/// Signed open error per strap run, keyed by the run, for the runs one span covers alone. Keying it lets
/// two configs be compared on the SAME runs — the solo/bridged split itself moves as the thresholds move.
fn head_by_run(blocks: &[Block], p: &DetectParams) -> BTreeMap<(usize, i64), f64> {
    let mut out = BTreeMap::new();
    for (bi, b) in blocks.iter().enumerate() {
        let spans = detect_sessions_with(&b.hr, &b.accel, 0, &[], &b.band, None, p);
        let runs = asleep_runs(&b.band, RUN_MIN_MINUTES);
        for &(a, z) in &runs {
            let cov: Vec<&DetectedSpan> = spans.iter().filter(|x| x.start < z && x.end > a).collect();
            if cov.len() != 1 {
                continue;
            }
            let (s0, s1) = (cov[0].start, cov[0].end);
            if runs.iter().filter(|(x, y)| s0 < *y && s1 > *x).count() == 1 {
                out.insert((bi, a), (a - s0) as f64 / 60.0);
            }
        }
    }
    out
}

/// Share of shared-interior epochs that change stage when the same strap run is staged over our detected
/// window instead of over the run itself — the staging churn our window error causes.
fn window_churn(blocks: &[Block], p: &DetectParams) -> (f64, usize) {
    let params = Params::SHIPPED;
    let (mut moved, mut shared) = (0usize, 0usize);
    for b in blocks {
        let spans = detect_sessions_with(&b.hr, &b.accel, 0, &[], &b.band, None, p);
        for (a, z) in asleep_runs(&b.band, RUN_MIN_MINUTES) {
            let cov: Vec<&DetectedSpan> = spans.iter().filter(|x| x.start < z && x.end > a).collect();
            if cov.len() != 1 {
                continue;
            }
            let (s0, s1) = (cov[0].start, cov[0].end);
            let cut = |start: i64, end: i64| SleepInput {
                start,
                end,
                hr: b.hr.iter().filter(|h| h.ts >= start && h.ts < end).cloned().collect(),
                rr: b.rr.iter().filter(|r| r.ts >= start && r.ts < end).cloned().collect(),
                accel: b.accel.iter().filter(|g| g.ts >= start && g.ts < end).cloned().collect(),
            };
            let (ours, truth) = (cut(s0, s1), cut(a, z));
            if ours.hr.len() < 120 || ours.accel.len() < 120 || truth.hr.len() < 120 || truth.accel.len() < 120 {
                continue;
            }
            let seg_ours = stage_v2_prepared(&prepare_v2(&ours, &params), &params);
            let seg_truth = stage_v2_prepared(&prepare_v2(&truth, &params), &params);
            let at = |segs: &[physio_algo::sleep::StageSegment], t: i64| -> Option<SleepStage> {
                segs.iter().find(|g| g.start <= t && t < g.end).map(|g| g.stage)
            };
            let (lo, hi) = (a.max(s0), z.min(s1));
            let mut t = lo + 15;
            while t < hi {
                if let (Some(x), Some(y)) = (at(&seg_ours, t), at(&seg_truth, t)) {
                    shared += 1;
                    moved += usize::from(x != y);
                }
                t += 30;
            }
        }
    }
    (100.0 * moved as f64 / shared.max(1) as f64, shared)
}

/// Signed-error summary: `n`, median, p90, max, min, then the two-sided figures a median alone hides —
/// nights opening late at all and by more than 15 min, mean absolute error, and nights inside +/-15 min.
fn open_stats(head: &[f64]) -> String {
    let abs: Vec<f64> = head.iter().map(|x| x.abs()).collect();
    format!(
        "{:>5} {:>7.1} {:>7.1} {:>7.1} {:>7.1} {:>5} {:>5} {:>6.1} {:>5}",
        head.len(),
        median(&mut head.to_vec()),
        pct(&mut head.to_vec(), 0.9),
        pct(&mut head.to_vec(), 1.0),
        pct(&mut head.to_vec(), 0.0),
        head.iter().filter(|x| **x < 0.0).count(),
        head.iter().filter(|x| **x < -15.0).count(),
        abs.iter().sum::<f64>() / abs.len().max(1) as f64,
        head.iter().filter(|x| x.abs() <= 15.0).count(),
    )
}

fn print_band(label: &str, s: &BandScore) {
    println!(
        "{:<16}{} {:>6.1} {:>7.1} {:>6} {:>6} {:>5} {:>5}",
        label,
        open_stats(&s.head),
        median(&mut s.tail.clone()),
        pct(&mut s.tail.clone(), 1.0),
        s.tail.iter().filter(|x| **x > 15.0).count(),
        s.runs - s.hit,
        s.spurious.len(),
        s.bridged,
    );
}

fn print_psg(label: &str, s: &PsgScore) {
    println!("{:<16}{} {:>10} {:>7}", label, open_stats(&s.head), s.before_scoring, s.missed);
}

const OPEN_HEAD: &str = "    n  med-op    p90    max    min  late  l15   mae  w15";
const BAND_HEAD: &str = "  med-cl  max-cl  early  missed  spur  brdg";
const PSG_HEAD: &str = "  pre-scored  missed";

fn main() {
    let blocks = load_blocks();
    let nights = load_psg_nights();
    let runs: usize = blocks.iter().map(|b| asleep_runs(&b.band, RUN_MIN_MINUTES).len()).sum();
    println!(
        "band: {} wear blocks, {runs} strap asleep runs >= {RUN_MIN_MINUTES} min   PSG: {} nights",
        blocks.len(),
        nights.len()
    );
    println!("open error is signed minutes: + = we open before the reference calls sleep\n");

    let grid: Vec<DetectParams> = [0.70f64, 0.75, 0.80, 0.85, 0.90]
        .iter()
        .flat_map(|&enter| {
            [0.50f64, 0.55, 0.60, 0.65, 0.70]
                .iter()
                .filter(move |&&exit| exit <= enter)
                .map(move |&exit| DetectParams { still_enter: enter, still_exit: exit, ..DetectParams::SHIPPED })
        })
        .collect();

    println!("--- band, the strap's own asleep runs (do we agree with the band) ---");
    println!("enter/exit    {OPEN_HEAD}{BAND_HEAD}");
    for p in &grid {
        print_band(&format!("{:.2}/{:.2}", p.still_enter, p.still_exit), &score_band(&blocks, p));
    }

    println!("\n--- PSG onset, open edge only (are we right) ---");
    println!("enter/exit    {OPEN_HEAD}{PSG_HEAD}");
    for p in &grid {
        print_psg(&format!("{:.2}/{:.2}", p.still_enter, p.still_exit), &score_psg(&nights, p));
    }

    println!("\n--- the same runs under both named configs, so the solo/bridged split cannot move ---");
    let (before, after) = (
        head_by_run(&blocks, &DetectParams::PRE_HYSTERESIS),
        head_by_run(&blocks, &DetectParams::SHIPPED),
    );
    let common: Vec<(f64, f64)> =
        before.iter().filter_map(|(k, v)| after.get(k).map(|w| (*v, *w))).collect();
    println!("paired         {OPEN_HEAD}");
    println!("{:<16}{}", "pre-hysteresis", open_stats(&common.iter().map(|x| x.0).collect::<Vec<_>>()));
    println!("{:<16}{}", "shipped", open_stats(&common.iter().map(|x| x.1).collect::<Vec<_>>()));
    let mut delta: Vec<f64> = common.iter().map(|(b, a)| a - b).collect();
    println!(
        "{} runs solo under both: {} open later, {} unchanged, {} earlier; median shift {:+.1} min",
        common.len(),
        delta.iter().filter(|d| **d < -0.001).count(),
        delta.iter().filter(|d| d.abs() <= 0.001).count(),
        delta.iter().filter(|d| **d > 0.001).count(),
        median(&mut delta),
    );

    println!("\n--- the two named configs, and the staging churn each window causes ---");
    println!("config        {OPEN_HEAD}{BAND_HEAD}");
    for (label, p) in [("pre-hysteresis", DetectParams::PRE_HYSTERESIS), ("shipped", DetectParams::SHIPPED)] {
        print_band(label, &score_band(&blocks, &p));
    }
    println!("config        {OPEN_HEAD}{PSG_HEAD}");
    for (label, p) in [("pre-hysteresis", DetectParams::PRE_HYSTERESIS), ("shipped", DetectParams::SHIPPED)] {
        print_psg(label, &score_psg(&nights, &p));
    }
    for (label, p) in [("pre-hysteresis", DetectParams::PRE_HYSTERESIS), ("shipped", DetectParams::SHIPPED)] {
        for (name, min, cover) in score_band(&blocks, &p).spurious {
            println!("{label:<16} span over no strap sleep: {name}  {min:.0} min, band covers {:.0}%", 100.0 * cover);
        }
    }
    println!();
    for (label, p) in [("pre-hysteresis", DetectParams::PRE_HYSTERESIS), ("shipped", DetectParams::SHIPPED)] {
        let (churn, shared) = window_churn(&blocks, &p);
        println!("{label:<16} interior epochs relabelled by staging on our window: {churn:.1}% of {shared}");
    }
}
