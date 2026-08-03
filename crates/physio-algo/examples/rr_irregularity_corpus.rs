//! Print the R-R irregularity indices for a synthetic regular series, a synthetic irregular one, and
//! every night of a real R-R fixture, so the separation between them is a measurement rather than a claim.
//!
//! ```text
//! cargo run -p physio-algo --example rr_irregularity_corpus -- [fixture.json]
//! ```
//!
//! The fixture is `{ nights: [{ date, rr: [[unix, ms], ...] }] }`. Each night is reported as stored and
//! again with rescaled second copies removed, because the stored form of this corpus carries them.

use std::collections::HashMap;

use physio_algo::rr_irregularity::{
    ScreenState,
    assess_segments, quality, IrregularityIndices, IrregularityReading, SEGMENT_BEATS,
};
use physio_algo::stats::median;
use serde_json::Value;

const DEFAULT_FIXTURE: &str = "../whoop-data/harnesses/rr-real-fixture.json";

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| DEFAULT_FIXTURE.to_string());
    println!("== synthetic ==");
    report("regular", &synthetic_regular(300));
    report("irregular", &synthetic_irregular(300, 4242));

    let raw = match std::fs::read_to_string(&path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("no fixture at {path}: {e}");
            std::process::exit(2);
        }
    };
    let doc: Value = serde_json::from_str(&raw).expect("fixture is json");
    let nights = doc["nights"].as_array().expect("nights array");
    println!("\n== real: {path} ({} nights) ==", nights.len());
    for night in nights {
        let date = night["date"].as_str().unwrap_or("?");
        let beats: Vec<(u32, u16)> = night["rr"]
            .as_array()
            .map(|rows| {
                rows.iter()
                    .filter_map(|r| {
                        let a = r.as_array()?;
                        Some((a.first()?.as_u64()? as u32, a.get(1)?.as_u64()? as u16))
                    })
                    .collect()
            })
            .unwrap_or_default();
        report(&format!("{date} stored"), &beats);
        let cleaned = drop_rescaled_copies(&beats);
        report(&format!("{date} de-copied"), &cleaned);
        // The shipped screen on the same beats. This is the only real-hardware evidence there is about
        // how often it fires on a wearer with no known irregular rhythm.
        println!("{:24} screen stored     {}", "", summarise(&physio_algo::rr_irregularity::screen(&beats)));
        println!("{:24} screen de-copied  {}", "", summarise(&physio_algo::rr_irregularity::screen(&cleaned)));
    }
}

/// One line per screen result, with the episode indices when there are any.
fn summarise(state: &ScreenState) -> String {
    match state {
        ScreenState::Calibrating { have, need } => format!("Calibrating {have}/{need} beats"),
        ScreenState::Regular { windows_assessed, windows_irregular } => {
            format!("Regular ({windows_irregular} irregular of {windows_assessed} assessed windows)")
        }
        ScreenState::Inconclusive { reason } => format!("Inconclusive {reason:?}"),
        ScreenState::IrregularEpisodes { episodes, windows_assessed } => format!(
            "EPISODES {} of {windows_assessed} assessed windows: {:?}",
            episodes.len(),
            episodes
                .iter()
                .map(|e| format!(
                    "{}w {}s {:?} cosen {:.2} resid {:.2} ectopic {:.2} cov {:.2} rescaled {:.3} occ {:.2}",
                    e.windows, e.dur_s, e.confidence, e.cosen, e.residual_cosen, e.ectopic_fraction, e.coverage, e.rescaled_fraction, e.cell_occupancy
                ))
                .collect::<Vec<String>>()
        ),
    }
}

/// Drop every beat that is `round(other * RESCALE_RATIO)` of another beat in the same second or the one
/// before, keeping the original. The corpus's own duplication, removed so the underlying rhythm shows.
fn drop_rescaled_copies(beats: &[(u32, u16)]) -> Vec<(u32, u16)> {
    let mut by_sec: HashMap<u32, Vec<u16>> = HashMap::new();
    for &(t, v) in beats {
        by_sec.entry(t).or_default().push(v);
    }
    beats
        .iter()
        .copied()
        .filter(|&(t, v)| {
            !(0..=quality::RESCALE_LAG_S).any(|lag| {
                t.checked_sub(lag)
                    .and_then(|s| by_sec.get(&s))
                    .is_some_and(|src| {
                        src.iter().any(|&s| {
                            s != v && (f64::from(s) * quality::RESCALE_RATIO).round() as u16 == v
                        })
                    })
            })
        })
        .collect()
}

/// Segment the series at [`SEGMENT_BEATS`] and print the median of each index across the assessed
/// segments, with the count that were refused and why. Every index is a short-segment statistic, so a
/// per-night single number would not be the published one.
fn report(label: &str, beats: &[(u32, u16)]) {
    let segments = assess_segments(beats, SEGMENT_BEATS);
    let mut assessed: Vec<IrregularityIndices> = Vec::new();
    let mut refused: Vec<String> = Vec::new();
    for (_, reading) in &segments {
        match reading {
            IrregularityReading::Assessed(i) => assessed.push(*i),
            IrregularityReading::Inconclusive { reason, .. } => refused.push(format!("{reason:?}")),
        }
    }
    if assessed.is_empty() {
        println!("{label:24} {}/{} segments assessed  refused {:?}", 0, segments.len(), tally(&refused));
        return;
    }
    let med = |f: fn(&IrregularityIndices) -> Option<f64>| {
        let vs: Vec<f64> = assessed.iter().filter_map(f).collect();
        if vs.is_empty() { "-".to_string() } else { format!("{:.4}", median(&vs)) }
    };
    println!(
        "{label:24} seg {:>4}/{:<4} rmssd/mean {:>7}  entropy {:>7}  tpr {:>7}  sampen {:>7}  \
         cosen {:>7}  sd1 {:>7}  sd2 {:>7}  narea {:>7}  occ {:>7}  oor {:.3} ectopic {:.3} \
         dup {:.3} rescaled {:.3} cov {:.2} refused {:?}",
        assessed.len(),
        segments.len(),
        med(|i| i.rmssd_over_mean),
        med(|i| i.shannon_entropy),
        med(|i| i.turning_point_ratio),
        med(|i| i.sample_entropy),
        med(|i| i.cosen),
        med(|i| i.poincare.map(|p| p.sd1)),
        med(|i| i.poincare.map(|p| p.sd2)),
        med(|i| i.poincare.map(|p| p.normalised_area)),
        med(|i| i.poincare.map(|p| p.cell_occupancy)),
        mean_of(&assessed, |i| i.quality.range_rejected_fraction),
        mean_of(&assessed, |i| i.quality.ectopic_rejected_fraction),
        mean_of(&assessed, |i| i.quality.duplicate_fraction),
        mean_of(&assessed, |i| i.quality.rescaled_fraction),
        mean_of(&assessed, |i| i.quality.coverage),
        tally(&refused),
    );
}

fn mean_of(xs: &[IrregularityIndices], f: fn(&IrregularityIndices) -> f64) -> f64 {
    xs.iter().map(f).sum::<f64>() / xs.len() as f64
}

fn tally(reasons: &[String]) -> Vec<(String, usize)> {
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for r in reasons {
        *counts.entry(r.as_str()).or_default() += 1;
    }
    let mut out: Vec<(String, usize)> = counts.into_iter().map(|(k, v)| (k.to_string(), v)).collect();
    out.sort();
    out
}

/// One beat a second at ~1000 ms with a respiratory sway — a regular rhythm.
fn synthetic_regular(n: u32) -> Vec<(u32, u16)> {
    (0..n)
        .map(|i| {
            let sway = (f64::from(i) * 0.25 * std::f64::consts::TAU / 4.0).sin() * 12.0;
            (i, (1000.0 + sway).round() as u16)
        })
        .collect()
}

/// Each interval drawn independently over 600-1400 ms at the same mean rate — an irregular rhythm.
fn synthetic_irregular(n: u32, seed: u64) -> Vec<(u32, u16)> {
    let mut x = seed;
    let mut acc = 0.0f64;
    (0..n)
        .map(|_| {
            x = x.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            let rr = 600 + ((x >> 33) % 800) as u16;
            acc += f64::from(rr) / 1000.0;
            (acc as u32, rr)
        })
        .collect()
}
