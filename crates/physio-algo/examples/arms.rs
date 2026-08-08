//! Score one staging change at a time against the only references that can still adjudicate it.
//!
//!   cargo run --release -p physio-algo --example arms
//!
//! Each arm is a single named mutation of `Params::SHIPPED`, never combined. Selection is DREAMT
//! (n=100) and AAUWSS (n=13) 4-class kappa; sleep-accel is printed but its `rr.csv` is empty on all 31
//! nights, so it cannot judge a respiratory term. Stage fractions ride beside every kappa because the
//! failure this whole exercise is about — one healthy night going 6% to 23% awake — does not move kappa.
//!
//! The band's `sleep_state` does NOT select. It marks the sleep PERIOD, not per-epoch sleep, so it
//! cannot see mid-night wake and scoring on it rewards a stager that calls less. What it still gives is
//! the ONSET gap: its first asleep second against our detected onset, independent of anything we compute.
//! Its close sits at our session end by construction, so no offset figure is reported.
//!
//! `jerk_move_mult` is held at SHIPPED on every arm that does not name it, so one `Prepared` per night
//! serves the whole table.

mod common;

use common::{
    dirs_of, kappa4, median, night_id, read_accel, read_band, read_hr, read_meta, read_rr, read_truth,
    stage_at, stage_idx, BAND_ASLEEP, WAKE,
};

use std::collections::BTreeMap;
use std::path::Path;

use physio_algo::sleep::{params::Params, prepare_v2, stage_v2_prepared, Prepared, SleepInput};

const EPOCH: i64 = 30;
const SELECT: [&str; 2] = ["dreamt", "aauwss"];
const REPORT: [&str; 1] = ["sleep-accel"];
const STAGE_NAMES: [&str; 4] = ["wake", "light", "deep", "rem"];

struct Night {
    input: SleepInput,
    w0: i64,
    n: usize,
    truth: BTreeMap<usize, i32>,
    band: Vec<(i64, i32)>,
    owner: String,
}

fn load(dir: &Path, need_truth: bool) -> Option<Night> {
    let (w0, w1, n) = read_meta(dir)?;
    let truth = read_truth(dir);
    if need_truth && truth.is_empty() {
        return None;
    }
    Some(Night {
        input: SleepInput { start: w0, end: w1, hr: read_hr(dir), rr: read_rr(dir), accel: read_accel(dir) },
        w0,
        n,
        truth,
        band: read_band(dir),
        owner: night_id(dir).0.split('_').next().unwrap_or("?").to_string(),
    })
}

/// One label per epoch, read at the epoch midpoint so a segment boundary cannot land between them.
fn labels(night: &Night, prep: &Prepared, p: &Params) -> Vec<usize> {
    let segs = stage_v2_prepared(prep, p);
    (0..night.n)
        .map(|k| {
            let mid = night.w0 + k as i64 * EPOCH + EPOCH / 2;
            stage_idx(stage_at(&segs, mid).unwrap_or_else(|| segs.last().unwrap().stage))
        })
        .collect()
}

#[derive(Default, Clone, Copy)]
struct Score {
    kappa: f64,
    frac: [f64; 4],
}

/// Fractions count LABELLED epochs only. Counting ours over the whole window while the reference counts
/// only what it scored is a denominator mismatch: DREAMT labels 75.7% of a night, and reading its REM
/// error off mixed denominators gave +0.4 pp where the matched figure is +3.2.
fn score(nights: &[Night], preps: &[Prepared], p: &Params) -> Score {
    let (mut cm, mut frac) = ([[0i64; 4]; 4], [0i64; 4]);
    for (night, prep) in nights.iter().zip(preps) {
        let lab = labels(night, prep, p);
        for (k, &t) in &night.truth {
            if *k < lab.len() && (0..4).contains(&t) {
                cm[t as usize][lab[*k]] += 1;
                frac[lab[*k]] += 1;
            }
        }
    }
    let tot = frac.iter().sum::<i64>().max(1) as f64;
    Score { kappa: kappa4(&cm), frac: std::array::from_fn(|i| 100.0 * frac[i] as f64 / tot) }
}

/// Minutes from our detected onset to the band's first asleep second. Positive means the band settles
/// AFTER we call sleep, which is the direction a too-eager onset shows up in.
fn onset_gap(nights: &[Night], preps: &[Prepared], p: &Params) -> Option<f64> {
    let mut gaps: Vec<f64> = Vec::new();
    for (night, prep) in nights.iter().zip(preps) {
        let Some(band_on) = night.band.iter().find(|(_, s)| *s == BAND_ASLEEP).map(|(t, _)| *t) else {
            continue;
        };
        let lab = labels(night, prep, p);
        let mut run = 0;
        let ours = (0..lab.len()).find(|&i| {
            run = if lab[i] == WAKE { 0 } else { run + 1 };
            run == 10
        })?;
        gaps.push((band_on - (night.w0 + (ours as i64 - 9) * EPOCH)) as f64 / 60.0);
    }
    (!gaps.is_empty()).then(|| median(&mut gaps))
}

/// The arms, each a single change. `#987` rows carry upstream's post-revert value; the rest are ours.
fn arms() -> Vec<(&'static str, Params)> {
    let s = Params::SHIPPED;
    let mut out: Vec<(&'static str, Params)> = vec![("SHIPPED", s)];

    // Proves the instrument: an unchanged Params must read exactly zero in every column. Anything else
    // means the deltas below are measuring the harness, not the arm.
    out.push(("NULL SHIPPED again — must be all +0.000", s));

    let mut a = s;
    a.deep_gate_thresh = 0.25;
    out.push(("A/G  deep_gate_thresh 0.40->0.25  #987", a));

    let mut b = s;
    b.transition[1] = [0.00333, 0.90, 0.06667, 0.03];
    out.push(("B    rem exit cost 4.52->3.40", b));

    let mut c = s;
    c.motion_gate_boost = 5.0;
    out.push(("C    motion_gate_boost 4.0->5.0", c));

    for m in [25.0, 30.0, 40.0, 45.0] {
        let mut v = s;
        v.jerk_gate_mult = m;
        out.push((
            Box::leak(format!("C'   jerk_gate_mult 35->{m:.0}").into_boxed_str()),
            v,
        ));
    }

    let mut e = s;
    e.base_rate = [0.18, 0.22, 0.50, 0.10];
    out.push(("E    base priors deep .18 awake .10  #987", e));

    let mut f = s;
    f.jerk_move_mult = 38.0;
    f.jerk_gate_mult = 55.0;
    f.motion_gate_boost = 2.0;
    out.push(("F    motion gate 38/55/2.0  #987", f));

    let mut h = s;
    h.transition[0] = [0.86, 0.007, 0.126, 0.007];
    h.transition[1] = [0.005, 0.88, 0.10, 0.015];
    h.transition[2] = [0.06, 0.06, 0.85, 0.03];
    out.push(("H    deep/rem/light transition rows  #987", h));

    let mut i = s;
    i.awake_deadzone = 0.0;
    out.push(("I    awake_deadzone 0.30->0.0  #987", i));

    let mut j = s;
    j.deep_hrv = -1.1;
    j.deep_hr = 0.0;
    j.deep_motion = -0.5;
    j.rem_hrv = 0.6;
    j.rem_motion = -0.6;
    j.awake_hrv = 0.8;
    j.awake_hr = 0.4;
    out.push(("J    emission coefficients  #987", j));

    // Measured against PSG truth: REM is over-called and deep under-called on both cohorts holding real
    // deep. Every arm above moves deep DOWN or leaves it; these are the untested direction.
    for t in [0.50, 0.60] {
        let mut v = s;
        v.deep_gate_thresh = t;
        out.push((Box::leak(format!("K    deep_gate_thresh 0.40->{t:.2}").into_boxed_str()), v));
    }
    for d in [0.18, 0.22] {
        let mut v = s;
        v.base_rate[0] = d;
        out.push((Box::leak(format!("L    deep prior .15->{d:.2}").into_boxed_str()), v));
    }
    for r in [0.18, 0.15] {
        let mut v = s;
        v.base_rate[1] = r;
        out.push((Box::leak(format!("M    rem prior .22->{r:.2}").into_boxed_str()), v));
    }
    for h in [0.6, 0.4] {
        let mut v = s;
        v.rem_hrv = h;
        out.push((Box::leak(format!("N    rem_hrv 0.8->{h:.1}").into_boxed_str()), v));
    }

    out
}

fn main() {
    let cohorts: Vec<(&str, bool, Vec<Night>)> = SELECT
        .iter()
        .map(|d| (*d, true))
        .chain(REPORT.iter().map(|d| (*d, false)))
        .filter_map(|(ds, selects)| {
            let ns: Vec<Night> = dirs_of(ds).iter().filter_map(|d| load(d, true)).collect();
            (!ns.is_empty()).then_some((ds, selects, ns))
        })
        .collect();
    if cohorts.is_empty() {
        println!("no PSG fixtures — set WHOOP_SLEEP_FIXTURES");
        return;
    }
    let preps: Vec<Vec<Prepared>> = cohorts
        .iter()
        .map(|(_, _, ns)| ns.iter().map(|n| prepare_v2(&n.input, &Params::SHIPPED)).collect())
        .collect();

    let ours: Vec<Night> = dirs_of("ours").iter().filter_map(|d| load(d, false)).collect();
    let ours: Vec<Night> = ours.into_iter().filter(|n| !n.band.is_empty()).collect();
    let ours_preps: Vec<Prepared> =
        ours.iter().map(|n| prepare_v2(&n.input, &Params::SHIPPED)).collect();
    let mut owners: Vec<String> = ours.iter().map(|n| n.owner.clone()).collect();
    owners.sort();
    owners.dedup();

    println!("SELECTION: {SELECT:?}. REPORTED ONLY: {REPORT:?} (rr.csv empty, no respiratory channel).");
    println!(
        "band onset gap over {} `ours` nights carrying band state, {} owners — CONTEXT, never a veto.\n",
        ours.len(),
        owners.len()
    );

    let table = arms();
    let base: Vec<Score> =
        cohorts.iter().zip(&preps).map(|((_, _, ns), pr)| score(ns, pr, &table[0].1)).collect();
    let base_gap = onset_gap(&ours, &ours_preps, &table[0].1);

    print!("{:<42}", "arm");
    for (ds, selects, ns) in &cohorts {
        let w = if *selects { 34 } else { 10 };
        print!("{:>w$}", format!("{ds}{} n={}", if *selects { "*" } else { "" }, ns.len()));
    }
    println!("{:>12}", "band onset");
    print!("{:<42}", "");
    for (_, selects, _) in &cohorts {
        if *selects {
            print!("{:>10}{:>8}{:>8}{:>8}", "d-kappa", "d-deep%", "d-rem%", "d-wake%");
        } else {
            print!("{:>10}", "d-kappa");
        }
    }
    println!("{:>12}", "d-min");

    for (label, p) in table.iter().skip(1) {
        print!("{label:<42}");
        for (i, ((_, selects, ns), pr)) in cohorts.iter().zip(&preps).enumerate() {
            let s = score(ns, pr, p);
            print!("{:>+10.3}", s.kappa - base[i].kappa);
            if *selects {
                print!(
                    "{:>+8.1}{:>+8.1}{:>+8.1}",
                    s.frac[2] - base[i].frac[2],
                    s.frac[3] - base[i].frac[3],
                    s.frac[0] - base[i].frac[0]
                );
            }
        }
        match (onset_gap(&ours, &ours_preps, p), base_gap) {
            (Some(g), Some(b)) => println!("{:>+12.1}", g - b),
            _ => println!("{:>12}", "-"),
        }
    }

    println!("\nabsolute SHIPPED against the reference's OWN fractions — a stage % with no denominator");
    println!("says nothing about direction, and direction is what gate 4 is:");
    for (i, (ds, _, ns)) in cohorts.iter().enumerate() {
        let mut t = [0i64; 4];
        for n in ns {
            for v in n.truth.values() {
                if (0..4).contains(v) {
                    t[*v as usize] += 1;
                }
            }
        }
        let tot = t.iter().sum::<i64>().max(1) as f64;
        print!("   {ds:<14} kappa4 {:.4}   ", base[i].kappa);
        for (k, name) in STAGE_NAMES.iter().enumerate() {
            let truth = 100.0 * t[k] as f64 / tot;
            print!("{name} {:.1}/{:.1} ({:+.1})  ", base[i].frac[k], truth, base[i].frac[k] - truth);
        }
        println!();
    }
    println!("   (ours/truth (delta) per stage)");
    if let Some(b) = base_gap {
        println!("   {:<14} band onset gap {b:+.1} min", "ours");
    }

    paired(&cohorts, &preps, &table);
}

/// Per-SUBJECT kappa deltas on the selection cohorts. A pooled delta of -0.008 sits exactly on the
/// gate's own tolerance, so it has to be read against how many subjects actually moved and which way.
fn paired(cohorts: &[(&str, bool, Vec<Night>)], preps: &[Vec<Prepared>], table: &[(&str, Params)]) {
    println!("\nPAIRED, per subject — pooled kappa alone cannot tell -0.008 from noise");
    print!("{:<42}", "arm");
    for (ds, selects, _) in cohorts {
        if *selects {
            print!("{:>30}", format!("{ds}: median  better/worse"));
        }
    }
    println!();

    for (label, p) in table.iter() {
        print!("{label:<42}");
        for ((ds, selects, ns), pr) in cohorts.iter().zip(preps) {
            let _ = ds;
            if !*selects {
                continue;
            }
            let mut d: Vec<f64> = Vec::new();
            for (night, prep) in ns.iter().zip(pr) {
                let one = |q: &Params| {
                    let lab = labels(night, prep, q);
                    let mut cm = [[0i64; 4]; 4];
                    for (k, &t) in &night.truth {
                        if *k < lab.len() && (0..4).contains(&t) {
                            cm[t as usize][lab[*k]] += 1;
                        }
                    }
                    kappa4(&cm)
                };
                d.push(one(p) - one(&Params::SHIPPED));
            }
            let better = d.iter().filter(|v| **v > 1e-9).count();
            let worse = d.iter().filter(|v| **v < -1e-9).count();
            print!("{:>+18.4}{:>6}/{:<5}", median(&mut d.clone()), better, worse);
        }
        println!();
    }
}
