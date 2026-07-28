//! Which sleep window the nightly HRV should be measured over, on real nights.
//!
//!   cargo run --release -p physio-algo --example hrv_window
//!
//! Stages each fixture night with the shipped recipe, then scores the same beats three ways: every 5-min
//! bucket of the night, only buckets centred in deep sleep, and only buckets in the LAST deep run. Also
//! reports how often a night yields no deep bucket at all, which is the coverage cost of the deep window.

use std::fs;
use std::path::{Path, PathBuf};

use physio_algo::hrv::HrvReadiness;
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
                .map(|l| l.split(',').filter_map(|c| c.trim().parse::<f64>().ok()).collect())
                .collect()
        })
        .unwrap_or_default()
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if v.is_empty() { f64::NAN } else { v[v.len() / 2] }
}

/// A half-open span of unix seconds.
type Span = (u32, u32);

/// One wearer's tally: nights staged, nights with no deep span, and the whole-night and deep-only values.
#[derive(Default)]
struct Owner {
    nights: usize,
    no_deep: usize,
    whole: Vec<f64>,
    deep: Vec<f64>,
}

/// The last contiguous run of deep spans, the comparator a strap-style "last slow-wave sleep" uses.
fn last_deep_run(deep: &[Span]) -> Vec<Span> {
    let mut last: Vec<Span> = Vec::new();
    let mut cur: Vec<Span> = Vec::new();
    for &(s, e) in deep {
        match cur.last() {
            Some(&(_, pe)) if pe == s => cur.push((s, e)),
            Some(_) => {
                last = std::mem::take(&mut cur);
                cur.push((s, e));
            }
            None => cur.push((s, e)),
        }
    }
    if !cur.is_empty() { cur } else { last }
}

fn main() {
    let mut dirs: Vec<PathBuf> = fs::read_dir(root())
        .map(|rd| rd.filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.is_dir()).collect())
        .unwrap_or_default();
    dirs.sort();
    let params = Params::default();

    let (mut whole, mut deep_only, mut last_sws) = (Vec::new(), Vec::new(), Vec::new());
    let (mut nights, mut no_deep, mut deep_under_5min) = (0usize, 0usize, 0usize);
    let mut shifts: Vec<f64> = Vec::new();
    let mut per_owner: std::collections::BTreeMap<String, Owner> = std::collections::BTreeMap::new();

    for d in &dirs {
        let hr: Vec<HrSample> = read_csv(&d.join("hr.csv"))
            .iter()
            .map(|r| HrSample { ts: r[0] as i64, bpm: r[1] as u16 })
            .collect();
        let accel: Vec<AccelSample> = read_csv(&d.join("gravity.csv"))
            .iter()
            .map(|r| AccelSample { ts: r[0] as i64, x: r[1], y: r[2], z: r[3] })
            .collect();
        let mut rr: Vec<RrRun> = Vec::new();
        for row in read_csv(&d.join("rr.csv")) {
            let (ts, ms) = (row[0] as i64, row[1] as u16);
            match rr.last_mut() {
                Some(l) if l.ts == ts => l.intervals.push(ms),
                _ => rr.push(RrRun { ts, intervals: vec![ms] }),
            }
        }
        if hr.len() < 120 || accel.len() < 120 || rr.is_empty() {
            continue;
        }
        let (start, end) = (hr[0].ts, hr[hr.len() - 1].ts);
        let input = SleepInput { start, end, hr, rr: rr.clone(), accel };
        let segs = stage_v2_prepared(&prepare_v2(&input, &params), &params);
        let deep: Vec<Span> = segs
            .iter()
            .filter(|s| s.stage == SleepStage::Deep)
            .map(|s| (s.start as u32, s.end as u32))
            .collect();

        let beats: Vec<(u32, u16)> =
            rr.iter().flat_map(|r| r.intervals.iter().map(|&v| (r.ts as u32, v))).collect();
        let (s, e) = (start as u32, end as u32);
        nights += 1;
        let owner = d.file_name().unwrap().to_string_lossy().split('_').next().unwrap_or("?").to_string();
        let row = per_owner.entry(owner).or_default();
        row.nights += 1;
        let deep_secs: u32 = deep.iter().map(|(a, b)| b - a).sum();
        if deep.is_empty() {
            no_deep += 1;
            row.no_deep += 1;
        }
        if deep_secs < 300 {
            deep_under_5min += 1;
        }
        let w = HrvReadiness::windowed_avg_hrv(s, e, &beats);
        let dp = HrvReadiness::windowed_avg_hrv_deep(s, e, &beats, &deep);
        if let Some(v) = w {
            whole.push(v);
            row.whole.push(v);
        }
        if let Some(v) = dp {
            deep_only.push(v);
            row.deep.push(v);
        }
        if let (Some(a), Some(b)) = (w, dp) {
            shifts.push(100.0 * (b - a) / a);
        }
        if let Some(v) = HrvReadiness::windowed_avg_hrv_deep(s, e, &beats, &last_deep_run(&deep)) {
            last_sws.push(v);
        }
    }

    println!("{nights} nights staged");
    println!(
        "median avgHrv   whole-night {:.1} ms (n={})   deep-only {:.1} ms (n={})   last-deep {:.1} ms (n={})",
        median(&mut whole), whole.len(),
        median(&mut deep_only), deep_only.len(),
        median(&mut last_sws), last_sws.len(),
    );
    println!(
        "deep-window coverage: {} of {} nights have no deep span ({:.1}%); {} have under 5 min ({:.1}%)",
        no_deep, nights, 100.0 * no_deep as f64 / nights as f64,
        deep_under_5min, 100.0 * deep_under_5min as f64 / nights as f64,
    );
    let scored = deep_only.len();
    println!(
        "nights yielding a deep-window value: {} of {} ({:.1}% show no HRV)",
        scored, nights, 100.0 * (nights - scored) as f64 / nights as f64,
    );
    println!("per-night whole->deep shift: median {:+.1}%", median(&mut shifts));
    println!();
    println!("{:<14} {:>7} {:>8} {:>12} {:>11}", "wearer", "nights", "no-deep", "whole-night", "deep-only");
    for (o, r) in &mut per_owner {
        println!(
            "{:<14} {:>7} {:>8} {:>11.1} {:>11.1}",
            o, r.nights, r.no_deep, median(&mut r.whole), median(&mut r.deep),
        );
    }
}
