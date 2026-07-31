//! Which sleep window the nightly HRV should be measured over, on real nights.
//!
//!   cargo run --release -p physio-algo --example hrv_window
//!
//! Stages each fixture night with the shipped recipe, then scores the same beats three ways: every 5-min
//! bucket of the night, only buckets centred in deep sleep, and only buckets in the LAST deep run. Also
//! reports how often a night yields no deep bucket at all, which is the coverage cost of the deep window,
//! and pairs the wearer with an export against WHOOP's own published nightly HRV for the same nights.
//!
//! `stage_v2` ONLY, and it changes nothing here: `refine_wake` rewrites wake seconds to light and can
//! create or remove neither a deep nor a REM second, so every deep-window statistic below is identical
//! on both paths. Pinned by `refine::tests::deep_and_rem_seconds_are_untouched`.

mod common;

use common::{dirs_of, median, read_accel, read_hr, read_published_hrv, read_rr};


use physio_algo::hrv::HrvReadiness;
use physio_algo::sleep::{
    params::Params, prepare_v2, stage_v2_prepared, AccelSample, HrSample, SleepInput, SleepStage,
};

/// How far a staged night's start may sit from a published sleep onset and still be the same night.
const MATCH_SLACK_S: i64 = 4 * 3600;

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
    let dirs = dirs_of("ours");
    let params = Params::default();

    let (mut whole, mut deep_only, mut last_sws) = (Vec::new(), Vec::new(), Vec::new());
    let (mut nights, mut no_deep, mut deep_under_5min) = (0usize, 0usize, 0usize);
    let mut shifts: Vec<f64> = Vec::new();
    let mut per_owner: std::collections::BTreeMap<String, Owner> = std::collections::BTreeMap::new();
    let mut staged: Vec<Night> = Vec::new();

    for d in &dirs {
        let hr: Vec<HrSample> = read_hr(d);
        let accel: Vec<AccelSample> = read_accel(d);
        let rr = read_rr(d);
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
        let (owner, onset) = common::night_id(d);
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

    let published = read_published_hrv();
    println!();
    if published.is_empty() {
        println!("no whoop-hrv.json beside the fixtures — the published-HRV comparison is skipped");
        return;
    }
    let keys_n: Vec<(String, i64)> = staged.iter().map(|n| (n.owner.clone(), n.start)).collect();
    let keys_p: Vec<(String, i64)> = published.iter().map(|p| (p.0.clone(), p.1)).collect();
    let pairs = common::pair_nearest(&keys_n, &keys_p, MATCH_SLACK_S);
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
