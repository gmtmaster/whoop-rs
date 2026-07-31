//! What excluding the report seam does to nightly RMSSD on real nights.
//!
//!   cargo run --release -p physio-algo --example hrv_seam
//!
//! Before = every beat flattened into one chain (`analyze_raw`, which applies the same range and ectopic
//! cleaning but knows no report boundaries). After = `rmssd_gap_aware`, which now breaks contiguity at
//! each report's first beat. Same beats, same cleaning; only the seam differs.
//!
//! Staging is used only to find the deep spans, which `refine_wake` cannot move (it rewrites wake to
//! light), so the seam figures are the same on both paths.

mod common;

use common::{dirs_of, median, read_rr};


use physio_algo::hrv::HrvReadiness;

fn main() {
    let dirs = dirs_of("ours");

    let (mut shifts, mut befores, mut afters) = (Vec::new(), Vec::new(), Vec::new());
    let mut per_owner: std::collections::BTreeMap<String, Vec<(f64, f64)>> = std::collections::BTreeMap::new();
    let (mut seams, mut beats) = (0usize, 0usize);
    for d in &dirs {
        // Beats sharing a second came from one report, which is the grouping the fix restores, and which
        // `read_rr` already applies. `rmssd_gap_aware` wants the pairs, not the run type.
        let reports: Vec<(u32, Vec<u16>)> =
            read_rr(d).into_iter().map(|r| (r.ts as u32, r.intervals)).collect();
        if reports.len() < 300 {
            continue;
        }
        let flat: Vec<u16> = reports.iter().flat_map(|(_, v)| v.iter().copied()).collect();
        seams += reports.len() - 1;
        beats += flat.len();
        let before = HrvReadiness::analyze_raw(&flat, None).rmssd;
        let after = HrvReadiness::rmssd_gap_aware(&reports);
        if let (Some(a), Some(b)) = (before, after) {
            befores.push(a);
            afters.push(b);
            shifts.push(100.0 * (b - a) / a);
            let owner = d.file_name().unwrap().to_string_lossy().split('_').next().unwrap_or("?").to_string();
            per_owner.entry(owner).or_default().push((a, b));
        }
    }
    println!("{} nights", shifts.len());
    println!("{beats} beats, {seams} report seams ({:.1}% of all successive pairs)", 100.0 * seams as f64 / (beats - 1) as f64);
    println!("RMSSD before {:.1} ms   after {:.1} ms", median(&mut befores), median(&mut afters));
    println!();
    // The per-night shift says whether the rule still fires for a wearer; two medians do not, because
    // a median of a median hides which nights moved.
    println!("{:<14} {:>7} {:>10} {:>10} {:>12} {:>8}", "wearer", "nights", "before", "after", "med shift", "moved");
    for (o, v) in &per_owner {
        let mut a: Vec<f64> = v.iter().map(|x| x.0).collect();
        let mut b: Vec<f64> = v.iter().map(|x| x.1).collect();
        let mut s: Vec<f64> = v.iter().map(|x| 100.0 * (x.1 - x.0) / x.0).collect();
        let moved = s.iter().filter(|x| x.abs() > 0.5).count();
        println!(
            "{:<14} {:>7} {:>9.1} {:>10.1} {:>11.1}% {:>5}/{}",
            o, v.len(), median(&mut a), median(&mut b), median(&mut s), moved, v.len(),
        );
    }
    let mut s = shifts.clone();
    println!(
        "per-night shift: median {:+.1}%   p10 {:+.1}%   p90 {:+.1}%",
        median(&mut s),
        { let mut v = shifts.clone(); v.sort_by(|a, b| a.partial_cmp(b).unwrap()); v[v.len() / 10] },
        { let mut v = shifts.clone(); v.sort_by(|a, b| a.partial_cmp(b).unwrap()); v[9 * v.len() / 10] },
    );
}
