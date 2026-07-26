//! Run the changed algorithms over a real backup's exported slices and report what they produce.
//! Not a test: a sanity readout on one user's own data, to check the new code behaves on real streams
//! rather than only on fixtures.
//!
//!   cargo run --release -p physio-algo --example verify_backup -- <slices-dir> <chrono-age>

use std::fs;
use std::path::Path;

use physio_algo::sleep_regularity::{epoch_grid, sleep_regularity_index, MIN_PAIRED_COVERAGE};
use physio_algo::vitality::{self, VitalityInput};

fn spans(path: &Path) -> Vec<(i64, i64)> {
    fs::read_to_string(path)
        .map(|t| {
            t.lines()
                .filter(|l| !l.trim().is_empty())
                .filter_map(|l| {
                    let (a, b) = l.split_once(',')?;
                    Some((a.trim().parse().ok()?, b.trim().parse().ok()?))
                })
                .collect()
        })
        .unwrap_or_default()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = Path::new(args.get(1).map(String::as_str).unwrap_or("slices"));
    let chrono_age: f64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(40.0);

    let asleep = spans(&dir.join("sleep.csv"));
    let covered = spans(&dir.join("cover.csv"));
    println!("sleep spans {}   wear windows {}", asleep.len(), covered.len());

    // ── Sleep Regularity Index over the trailing windows the app would use ────────────────────
    let last = asleep.iter().map(|s| s.1).max().unwrap_or(0);
    const DAY: i64 = 86_400;
    println!("\n{:<8} {:>10} {:>10}   window", "days", "SRI", "verdict");
    for days in [7usize, 8, 10, 14, 21] {
        let first_midnight = (last - (days as i64 - 1) * DAY) / DAY * DAY;
        let grid = epoch_grid(first_midnight, days, &asleep, &covered);
        // How much of the grid is actually observed, so a None is explained rather than mysterious.
        let (mut known, mut total) = (0usize, 0usize);
        for d in &grid {
            total += d.len();
            known += d.iter().filter(|e| e.is_some()).count();
        }
        let cov = known as f64 / total.max(1) as f64;
        match sleep_regularity_index(&grid) {
            Some(sri) => println!("{days:<8} {sri:>10.1} {:>10}   coverage {:.0}%", "scored", cov * 100.0),
            None => println!("{days:<8} {:>10} {:>10}   coverage {:.0}% (gate {:.0}%)",
                "-", "gated", cov * 100.0, MIN_PAIRED_COVERAGE * 100.0),
        }
    }

    // ── Body Age at the shipped doubling time, and what the previous one would have said ──────
    // Drivers are the reference values; only the regularity driver varies, to isolate its effect.
    let base = VitalityInput { chrono_age, resting_hr: Some(52.0), sleep_hours: Some(7.5), steps: Some(9000.0), ..Default::default() };
    println!("\n{:<26} {:>10} {:>12}", "sleep-regularity driver", "body age", "advance");
    let show = |label: &str, input: VitalityInput| match vitality::compute(&input) {
        Some(v) => println!("{label:<26} {:>10.2} {:>+12.2}", v.body_age, v.advance_years),
        None => println!("{label:<26} {:>10} {:>12}", "-", "too few drivers"),
    };
    for sri in [41.0, 55.0, 68.25, 75.0, 90.0] {
        show(&format!("SRI {sri:.0}"), VitalityInput { sleep_regularity_index: Some(sri), ..base });
    }
    for cv in [0.5, 0.75, 0.95] {
        show(&format!("duration proxy {cv:.2}"), VitalityInput { sleep_consistency: Some(cv), ..base });
    }

    println!("
=== workout detection, CURRENT code on the backup's raw streams ===");
    for day in ["2026-07-24", "2026-07-25", "2026-07-26"] {
        workouts_for(dir, day, 70.0, 187.0);
    }
}

/// Run the CURRENT workout detector over a day's raw streams. Raw data is version-independent, so this
/// answers "what would today's code find" on a backup taken from an older build.
#[allow(dead_code)]
fn workouts_for(dir: &Path, day: &str, resting_hr: f64, max_hr: f64) {
    let hr: Vec<physio_algo::HrSample> = fs::read_to_string(dir.join(format!("hr_{day}.csv")))
        .unwrap_or_default().lines().filter_map(|l| {
            let (a, b) = l.split_once(',')?;
            Some(physio_algo::HrSample { ts: a.parse().ok()?, bpm: b.trim().parse::<f64>().ok()? as i32 })
        }).collect();
    let grav: Vec<physio_algo::workout::GravitySample> = fs::read_to_string(dir.join(format!("grav_{day}.csv")))
        .unwrap_or_default().lines().filter_map(|l| {
            let p: Vec<&str> = l.split(',').collect();
            if p.len() < 4 { return None; }
            Some(physio_algo::workout::GravitySample {
                ts: p[0].parse().ok()?, x: p[1].parse().ok()?, y: p[2].parse().ok()?, z: p[3].trim().parse().ok()?,
            })
        }).collect();
    if hr.is_empty() { println!("{day}: no data"); return; }
    let out = physio_algo::workout::detect(&hr, &grav, Some(resting_hr), Some(max_hr), Some(40.0), 80.0, 180.0, "male");
    println!("\n{day}  hr {} grav {}  -> {} workouts", hr.len(), grav.len(), out.len());
    for w in &out {
        let mins = (w.end - w.start) as f64 / 60.0;
        println!("   {:>5.0} min  avgHR {:>5.1}  strain {:>7}  kcal {:>7}",
            mins, w.avg_hr,
            w.strain.map(|s| format!("{s:.1}")).unwrap_or("None".into()),
            w.calories_kcal.map(|k| format!("{k:.0}")).unwrap_or("None".into()));
    }
}
