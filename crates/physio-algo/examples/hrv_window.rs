//! Which sleep window the nightly HRV should be measured over, on real nights.
//!
//!   cargo run --release -p physio-algo --example hrv_window
//!
//! Stages each fixture night with the shipped recipe, then scores the same beats three ways: every 5-min
//! bucket of the night, only buckets centred in deep sleep, and only buckets in the LAST deep run. Also
//! reports how often a night yields no deep bucket at all, which is the coverage cost of the deep window,
//! and pairs the wearer with an export against WHOOP's own published nightly HRV for the same nights.

use std::fs;
use std::path::{Path, PathBuf};

use physio_algo::hrv::HrvReadiness;
use physio_algo::sleep::{
    params::Params, prepare_v2, stage_v2_prepared, AccelSample, HrSample, RrRun, SleepInput, SleepStage,
};

/// How far a staged night's start may sit from a published sleep onset and still be the same night.
const MATCH_SLACK_S: i64 = 4 * 3600;

fn fixtures() -> PathBuf {
    std::env::var("WHOOP_SLEEP_FIXTURES")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("C:/Users/DavidGillot/Projects/whoop/sleep-benchmark/fixtures_multi"))
}

fn root() -> PathBuf {
    fixtures().join("ours")
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

/// One staged night's identity and the two windowed values, for the pairing against the export.
/// `start` is the night's REAL unix onset, read from the fixture name — the streams are rebased.
struct Night {
    owner: String,
    start: i64,
    whole: Option<f64>,
    deep: Option<f64>,
}

/// Owner and real unix onset out of an `owner_device_day_onset` fixture directory name.
fn night_id(dir: &Path) -> (String, i64) {
    let name = dir.file_name().unwrap_or_default().to_string_lossy().to_string();
    let owner = name.split('_').next().unwrap_or("?").to_string();
    let onset = name.rsplit('_').next().and_then(|s| s.parse::<i64>().ok()).unwrap_or(0);
    (owner, onset)
}

/// WHOOP's own published nightly HRV per wearer, keyed by the unix onset of the night it belongs to.
fn published_hrv() -> Vec<(String, i64, f64)> {
    let text = match fs::read_to_string(fixtures().join("whoop-hrv.json")) {
        Ok(t) => t,
        Err(_) => return Vec::new(),
    };
    let root: serde_json::Value = match serde_json::from_str(&text) {
        Ok(v) => v,
        Err(_) => return Vec::new(),
    };
    let mut out = Vec::new();
    for (owner, nights) in root.as_object().into_iter().flatten() {
        for n in nights.as_array().into_iter().flatten() {
            if let (Some(t), Some(h)) = (n["onset_unix"].as_i64(), n["hrv_ms"].as_f64()) {
                out.push((owner.clone(), t, h));
            }
        }
    }
    out
}

/// Nearest-first 1:1 pairing of staged nights to published ones, inside [`MATCH_SLACK_S`].
fn pair_nights(nights: &[Night], published: &[(String, i64, f64)]) -> Vec<(usize, usize)> {
    let mut cand: Vec<(i64, usize, usize)> = Vec::new();
    for (i, n) in nights.iter().enumerate() {
        for (j, p) in published.iter().enumerate() {
            let d = (n.start - p.1).abs();
            if n.owner == p.0 && d <= MATCH_SLACK_S {
                cand.push((d, i, j));
            }
        }
    }
    cand.sort();
    let (mut used_n, mut used_p, mut out) = (Vec::new(), Vec::new(), Vec::new());
    for (_, i, j) in cand {
        if !used_n.contains(&i) && !used_p.contains(&j) {
            used_n.push(i);
            used_p.push(j);
            out.push((i, j));
        }
    }
    out.sort();
    out
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
    let mut staged: Vec<Night> = Vec::new();

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
        let (owner, onset) = night_id(d);
        let row = per_owner.entry(owner.clone()).or_default();
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
        staged.push(Night { owner, start: onset, whole: w, deep: dp });
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

    let published = published_hrv();
    println!();
    if published.is_empty() {
        println!("no whoop-hrv.json beside the fixtures — the published-HRV comparison is skipped");
        return;
    }
    let pairs = pair_nights(&staged, &published);
    let (mut ours_deep, mut ours_whole, mut theirs) = (Vec::new(), Vec::new(), Vec::new());
    let (mut dev_deep, mut dev_whole) = (Vec::new(), Vec::new());
    // Every pair, so the window choice can be checked night by night instead of on a median.
    println!("{:<12} {:>10} {:>10} {:>8} {:>12} {:>8}", "onset", "published", "deep-only", "diff", "whole-night", "diff");
    for &(i, j) in &pairs {
        let p = published[j].2;
        let fmt = |v: Option<f64>| v.map_or("-".to_string(), |x| format!("{x:.1}"));
        println!(
            "{:<12} {:>10.1} {:>10} {:>8} {:>12} {:>8}",
            staged[i].start,
            p,
            fmt(staged[i].deep),
            staged[i].deep.map_or("-".to_string(), |v| format!("{:+.1}", v - p)),
            fmt(staged[i].whole),
            staged[i].whole.map_or("-".to_string(), |v| format!("{:+.1}", v - p)),
        );
        if let Some(v) = staged[i].deep {
            ours_deep.push(v);
            theirs.push(p);
            dev_deep.push(v - p);
        }
        if let Some(v) = staged[i].whole {
            ours_whole.push(v);
            dev_whole.push(v - p);
        }
    }
    println!();
    // The two windows are scored on different n: a night with no deep span still has a whole-night
    // value, so reporting one count for both would overstate the deep column's sample.
    println!(
        "vs WHOOP's own published nightly HRV: {} of {} staged nights paired within {} h \
         ({} carry a deep value, {} a whole-night one)",
        pairs.len(), staged.len(), MATCH_SLACK_S / 3600, ours_deep.len(), ours_whole.len(),
    );
    println!(
        "median  published {:.1} ms   deep-only {:.1} ms   whole-night {:.1} ms",
        median(&mut theirs), median(&mut ours_deep), median(&mut ours_whole),
    );
    let mae = |v: &[f64]| v.iter().map(|x| x.abs()).sum::<f64>() / v.len().max(1) as f64;
    let (mae_deep, mae_whole) = (mae(&dev_deep), mae(&dev_whole));
    println!(
        "per-night difference from published: deep-only median {:+.1} ms (MAE {:.1}), whole-night median {:+.1} ms (MAE {:.1})",
        median(&mut dev_deep), mae_deep, median(&mut dev_whole), mae_whole,
    );
    println!("both wrists at once, not both methods on one wrist: WHOOP is worn left, noop right.");
}
