//! Step I — stage fractions and efficiency, on the window the reference itself reports.
//!
//!   cargo run --release -p physio-algo --example stage_fractions
//!
//! 1  the pairing: which of our nights an export row covers, and how much of its window we can see
//! 2  the fraction table, on `stage_v2` and on the app's `stage_v2` + `refine_wake`
//! 3  a second wearer's export beside David's — a falsifier for a per-wearer quantity, never a target
//! 4  the same fractions against PSG hypnograms, the other reference, with kappa beside them
//! 5  efficiency, against the export's own
//! 6  three bounded rescues on the deep under-call, each scored on BOTH references
//!
//! `ours` nights are rebased onto a synthetic clock, so a night's real onset comes from its directory
//! name; that is what pairs it to an export row, and what re-slices our epochs onto WHOOP's own in-bed
//! window. Its gravity is held forward across gaps, so the refinement's posture check reads optimistically
//! here — every refined figure carries the density census that says how many spans the gate accepted.

mod common;

use common::{
    dirs_of, kappa4, night_id, pair_nearest, pre_retune, read_accel, read_hr, read_meta, read_rr, read_steps,
     stage_idx, RefineCensus,
};

use std::fs;

use physio_algo::sleep::{
    epoch_starts_v2, params::Params, prepare_v2, stage_v2_prepared, AccelSample, Prepared, SleepInput,
    SleepStage, StageSegment, StepSample,
};

const EPOCH_SEC: i64 = 30;
/// How far apart a staged night and an export row may sit and still be the same night.
const MATCH_SLACK_S: i64 = 4 * 3600;
const PSG: [&str; 3] = ["dreamt", "aauwss", "sleep-accel"];
/// Column order everywhere below, and the order `stage_idx` and the PSG truth codes both produce.
const STAGES: [&str; 4] = ["wake", "light", "deep", "rem"];
const DEEP: usize = 2;

struct Night {
    owner: String,
    /// The wearer's own clock at the fixture's window start.
    onset: i64,
    w0: i64,
    span_s: i64,
    input: SleepInput,
    accel: Vec<AccelSample>,
    steps: Vec<StepSample>,
}

/// One night as WHOOP's app reports it: its own in-bed window and its own stage shares.
struct Export {
    owner: String,
    onset: i64,
    wake: i64,
    frac: [f64; 4],
}

/// A PSG night: the streams, and the hypnogram keyed by epoch start.
struct PsgNight {
    input: SleepInput,
    accel: Vec<AccelSample>,
    steps: Vec<StepSample>,
    truth: Vec<(i64, usize)>,
}

/// One gate-accepted night, staged four ways, so both paths are reported on one denominator.
struct Accepted {
    shipped: [f64; 4],
    shipped_refined: [f64; 4],
    pre: [f64; 4],
    pre_refined: [f64; 4],
}

/// One cohort under one recipe: what we stage, what the hypnogram holds, and the agreement between them.
struct PsgScore {
    nights: usize,
    ours: [f64; 4],
    truth: [f64; 4],
    kappa: f64,
}

fn load_ours() -> Vec<Night> {
    let mut out = Vec::new();
    for d in dirs_of("ours") {
        let Some((w0, w1, _)) = read_meta(&d) else { continue };
        let (owner, onset) = night_id(&d);
        let accel = read_accel(&d);
        let hr = read_hr(&d);
        if hr.len() < 120 || accel.len() < 120 {
            continue;
        }
        out.push(Night {
            owner,
            onset,
            w0,
            span_s: w1 - w0,
            steps: read_steps(&d),
            input: SleepInput { start: w0, end: w1, hr, rr: read_rr(&d), accel: accel.clone() },
            accel,
        });
    }
    out
}

/// WHOOP's per-night stage shares, keyed to the in-bed window the app itself reports.
fn load_export() -> Vec<Export> {
    let path = common::fixtures_root().join("whoop-labels.json");
    let Ok(text) = fs::read_to_string(&path) else { return Vec::new() };
    let Ok(root) = serde_json::from_str::<serde_json::Value>(&text) else { return Vec::new() };
    let mut out = Vec::new();
    for (owner, nights) in root.as_object().into_iter().flatten() {
        for n in nights.as_array().into_iter().flatten() {
            let (Some(onset), Some(wake)) = (n["onset_unix"].as_i64(), n["wake_unix"].as_i64()) else {
                continue;
            };
            let frac = ["awake", "light", "deep", "rem"].map(|k| n[k].as_f64().unwrap_or(f64::NAN));
            if wake > onset && frac.iter().all(|v| v.is_finite()) {
                out.push(Export { owner: owner.clone(), onset, wake, frac });
            }
        }
    }
    out.sort_by_key(|e| e.onset);
    out
}

fn load_psg(set: &str) -> Vec<PsgNight> {
    let mut out = Vec::new();
    for d in dirs_of(set) {
        let Some((w0, w1, _)) = read_meta(&d) else { continue };
        let truth: Vec<(i64, usize)> = common::read_csv(&d.join("truth.csv"))
            .iter()
            .filter(|r| (0.0..STAGES.len() as f64).contains(&r[1]))
            .map(|r| (w0 + r[0] as i64 * EPOCH_SEC, r[1] as usize))
            .collect();
        let accel = read_accel(&d);
        let hr = read_hr(&d);
        if truth.is_empty() || hr.len() < 120 || accel.len() < 120 {
            continue;
        }
        out.push(PsgNight {
            steps: read_steps(&d),
            truth,
            input: SleepInput { start: w0, end: w1, hr, rr: read_rr(&d), accel: accel.clone() },
            accel,
        });
    }
    out
}

/// Per-epoch stage index of one staging, on the stager's own grid.
fn labels(segs: &[StageSegment], starts: &[i64]) -> Vec<usize> {
    starts
        .iter()
        .map(|s| {
            let mid = s + EPOCH_SEC / 2;
            stage_idx(
                segs.iter()
                    .find(|g| g.start <= mid && mid < g.end)
                    .or(segs.last())
                    .map(|g| g.stage)
                    .unwrap_or(SleepStage::Light),
            )
        })
        .collect()
}

/// Stage shares over the epochs `keep` accepts, or None when it accepts none.
fn fractions(labels: &[usize], starts: &[i64], keep: impl Fn(i64) -> bool) -> Option<[f64; 4]> {
    let mut n = [0i64; 4];
    for (l, s) in labels.iter().zip(starts) {
        if keep(s + EPOCH_SEC / 2) {
            n[*l] += 1;
        }
    }
    let all: i64 = n.iter().sum();
    (all > 0).then(|| n.map(|v| 100.0 * v as f64 / all as f64))
}

fn mean_rows(rows: &[[f64; 4]]) -> [f64; 4] {
    let mut m = [0.0; 4];
    for r in rows {
        for i in 0..4 {
            m[i] += r[i] / rows.len() as f64;
        }
    }
    m
}

fn sd_rows(rows: &[[f64; 4]]) -> [f64; 4] {
    let m = mean_rows(rows);
    let mut v = [0.0; 4];
    for r in rows {
        for i in 0..4 {
            v[i] += (r[i] - m[i]).powi(2) / (rows.len().max(2) - 1) as f64;
        }
    }
    v.map(f64::sqrt)
}

fn col(rows: &[([f64; 4], [f64; 4])], second: bool) -> Vec<[f64; 4]> {
    rows.iter().map(|r| if second { r.1 } else { r.0 }).collect()
}

fn row(label: &str, n: usize, f: [f64; 4]) {
    println!("{label:<30} {n:>6} {:>7.1} {:>7.1} {:>7.1} {:>7.1}", f[0], f[1], f[2], f[3]);
}

fn header(what: &str) {
    println!("\n{what}");
    println!("{:<30} {:>6} {:>7} {:>7} {:>7} {:>7}", "source", "nights", STAGES[0], STAGES[1], STAGES[2], STAGES[3]);
}

/// One night staged both ways, restricted to the export's window. The refined half is None when the
/// density gate declined, which is not the app's path and must not be pooled with one that is.
fn stage_night(n: &Night, e: &Export, p: &Params, census: &mut RefineCensus) -> (Option<[f64; 4]>, Option<[f64; 4]>) {
    let prep = prepare_v2(&n.input, p);
    let starts = epoch_starts_v2(&prep);
    let segs = stage_v2_prepared(&prep, p);
    let before = census.refined;
    let refined_segs = census.refine(&segs, &n.accel, &n.steps);
    let accepted = census.refined > before;
    // Fixture clock -> the wearer's own: the night was rebased onto `w0` from its real onset.
    let shift = n.onset - n.w0;
    let keep = |mid: i64| {
        let real = mid + shift;
        real >= e.onset && real < e.wake
    };
    let plain = fractions(&labels(&segs, &starts), &starts, keep);
    let refined = fractions(&labels(&refined_segs, &starts), &starts, keep);
    (plain, accepted.then_some(refined).flatten())
}

/// Our staging against one cohort's hypnogram, on the epochs it scores. Features are extracted once by
/// the caller and only re-labelled here, which is sound exactly while `p.jerk_move_mult` is unchanged --
/// the one field `prepare_v2` reads. `assert_prepared_with` holds every caller to that.
fn psg_score(nights: &[PsgNight], prep_of: &[Prepared], p: &Params, census: &mut RefineCensus) -> Option<PsgScore> {
    let (mut ours, mut truth) = (Vec::new(), Vec::new());
    let mut cm = [[0i64; 4]; 4];
    for (n, prep) in nights.iter().zip(prep_of) {
        let starts = epoch_starts_v2(prep);
        let segs = stage_v2_prepared(prep, p);
        census.refine(&segs, &n.accel, &n.steps);
        let lab = labels(&segs, &starts);
        let (mut n_ours, mut n_true) = ([0i64; 4], [0i64; 4]);
        for (t, code) in &n.truth {
            let Some(i) = starts.iter().position(|s| *s <= *t && *t < s + EPOCH_SEC) else { continue };
            n_ours[lab[i]] += 1;
            n_true[*code] += 1;
            cm[*code][lab[i]] += 1;
        }
        let all: i64 = n_true.iter().sum();
        if all > 0 {
            ours.push(n_ours.map(|v| 100.0 * v as f64 / all as f64));
            truth.push(n_true.map(|v| 100.0 * v as f64 / all as f64));
        }
    }
    (!ours.is_empty()).then(|| PsgScore {
        nights: ours.len(),
        ours: mean_rows(&ours),
        truth: mean_rows(&truth),
        kappa: kappa4(&cm),
    })
}

/// Mean of per-night fractions over the paired nights, ours beside WHOOP's, on the unrefined path.
/// Same prepared-once contract as [`psg_score`].
fn export_pair(
    nights: &[Night], prep_of: &[Prepared], export: &[Export], paired: &[(usize, usize)], p: &Params,
) -> ([f64; 4], [f64; 4]) {
    let mut rows = Vec::new();
    for &(i, j) in paired {
        let (n, e) = (&nights[i], &export[j]);
        let prep = &prep_of[i];
        let starts = epoch_starts_v2(prep);
        let shift = n.onset - n.w0;
        let keep = |mid: i64| mid + shift >= e.onset && mid + shift < e.wake;
        if let Some(f) = fractions(&labels(&stage_v2_prepared(prep, p), &starts), &starts, keep) {
            rows.push((f, e.frac));
        }
    }
    (mean_rows(&col(&rows, false)), mean_rows(&col(&rows, true)))
}

/// A rescue config may only change what `stage_v2_prepared` reads, never what `prepare_v2` does, or the
/// prepared-once reuse above would score stale features.
fn assert_prepared_with(base: &Params, p: &Params) {
    assert!(
        (base.jerk_move_mult - p.jerk_move_mult).abs() < f64::EPSILON,
        "config changes jerk_move_mult, so its features cannot be reused"
    );
}

fn main() {
    let shipped = Params::SHIPPED;
    let pre = pre_retune(&shipped);
    let nights = load_ours();
    let export = load_export();
    if nights.is_empty() || export.is_empty() {
        println!("need both `ours` and whoop-labels.json under {}", common::fixtures_root().display());
        return;
    }

    // ── 1. the pairing ──────────────────────────────────────────────────────────────────────────
    let keys_n: Vec<(String, i64)> = nights.iter().map(|n| (n.owner.clone(), n.onset)).collect();
    let keys_e: Vec<(String, i64)> = export.iter().map(|e| (e.owner.clone(), e.onset)).collect();
    let paired = pair_nearest(&keys_n, &keys_e, MATCH_SLACK_S);
    println!("=== 1. the pairing ===");
    println!("{} nights under ours, {} export nights", nights.len(), export.len());
    let mut cover: Vec<f64> = paired
        .iter()
        .map(|&(i, j)| {
            let (n, e) = (&nights[i], &export[j]);
            let (a, b) = (n.onset.max(e.onset), (n.onset + n.span_s).min(e.wake));
            100.0 * (b - a).max(0) as f64 / (e.wake - e.onset) as f64
        })
        .collect();
    cover.sort_by(f64::total_cmp);
    let mut owners: Vec<&str> = paired.iter().map(|&(i, _)| nights[i].owner.as_str()).collect();
    owners.sort_unstable();
    owners.dedup();
    println!(
        "{} paired within {} h, wearer(s) {}. Our window covers the export's in-bed seconds: median {:.1}%, worst {:.1}%",
        paired.len(),
        MATCH_SLACK_S / 3600,
        owners.join("+"),
        cover.get(cover.len() / 2).copied().unwrap_or(0.0),
        cover.first().copied().unwrap_or(0.0),
    );

    // ── 2. the fractions ────────────────────────────────────────────────────────────────────────
    let mut plain: Vec<([f64; 4], [f64; 4])> = Vec::new(); // (shipped, pre-retune) over every paired night
    let mut both: Vec<Accepted> = Vec::new(); // the gate-accepted nights only
    let (mut whoop_all, mut whoop_ref) = (Vec::new(), Vec::new());
    let (mut cs, mut cp) = (RefineCensus::default(), RefineCensus::default());
    for &(i, j) in &paired {
        let (n, e) = (&nights[i], &export[j]);
        let (s_plain, s_ref) = stage_night(n, e, &shipped, &mut cs);
        let (p_plain, p_ref) = stage_night(n, e, &pre, &mut cp);
        let (Some(sp), Some(pp)) = (s_plain, p_plain) else { continue };
        plain.push((sp, pp));
        whoop_all.push(e.frac);
        if let (Some(sr), Some(pr)) = (s_ref, p_ref) {
            both.push(Accepted { shipped: sp, shipped_refined: sr, pre: pp, pre_refined: pr });
            whoop_ref.push(e.frac);
        }
    }
    header(&format!("=== 2. mean of per-night % of the export's in-bed window, {} paired nights ===", plain.len()));
    row("WHOOP export", whoop_all.len(), mean_rows(&whoop_all));
    row("shipped, stage_v2", plain.len(), mean_rows(&col(&plain, false)));
    row("pre-retune, stage_v2", plain.len(), mean_rows(&col(&plain, true)));
    println!("{}", cs.line("shipped over the paired nights"));

    header("=== 2b. the same nights the density gate accepts, so both paths share a denominator ===");
    if both.is_empty() {
        println!("no paired night passes the density gate, so the app's path cannot be scored here");
    } else {
        row("WHOOP export", whoop_ref.len(), mean_rows(&whoop_ref));
        row("shipped, stage_v2", both.len(), mean_rows(&both.iter().map(|r| r.shipped).collect::<Vec<_>>()));
        row("shipped, + refine_wake", both.len(), mean_rows(&both.iter().map(|r| r.shipped_refined).collect::<Vec<_>>()));
        row("pre-retune, stage_v2", both.len(), mean_rows(&both.iter().map(|r| r.pre).collect::<Vec<_>>()));
        row("pre-retune, + refine_wake", both.len(), mean_rows(&both.iter().map(|r| r.pre_refined).collect::<Vec<_>>()));
        println!("   the refinement rewrites wake to light and can touch neither deep nor REM, so the deep");
        println!("   difference between this block and the one above is the NIGHTS, not the path.");
    }

    // ── 3. a second wearer ──────────────────────────────────────────────────────────────────────
    header("=== 3. every export wearer, mean of per-night % (sd beneath) — a falsifier, not a target ===");
    let mut wearers: Vec<&str> = export.iter().map(|e| e.owner.as_str()).collect();
    wearers.sort_unstable();
    wearers.dedup();
    for w in wearers {
        let rows: Vec<[f64; 4]> = export.iter().filter(|e| e.owner == w).map(|e| e.frac).collect();
        row(w, rows.len(), mean_rows(&rows));
        let sd = sd_rows(&rows);
        println!("{:<30} {:>6} {:>7.1} {:>7.1} {:>7.1} {:>7.1}   sd", "", "", sd[0], sd[1], sd[2], sd[3]);
    }

    // ── 4. the other reference ──────────────────────────────────────────────────────────────────
    // Features once per set, re-labelled per config below: `prepare_v2` reads only `jerk_move_mult`, and
    // `assert_prepared_with` refuses any config that moves it.
    let psg: Vec<(&str, Vec<PsgNight>, Vec<Prepared>)> = PSG
        .iter()
        .map(|s| {
            let ns = load_psg(s);
            let prep = ns.iter().map(|n| prepare_v2(&n.input, &shipped)).collect();
            (*s, ns, prep)
        })
        .collect();
    let ours_prep: Vec<Prepared> = nights.iter().map(|n| prepare_v2(&n.input, &shipped)).collect();
    println!("\n=== 4. the same fractions against PSG hypnograms, mean of per-night % ===");
    println!(
        "{:<16} {:>6} {:>29} {:>29} {:>9} {:>7}",
        "cohort", "nights", "ours: wake/light/deep/rem", "PSG: wake/light/deep/rem", "deep err", "kappa4"
    );
    let mut psg_census = RefineCensus::default();
    for (set, ns, prep) in &psg {
        let Some(s) = psg_score(ns, prep, &shipped, &mut psg_census) else { continue };
        println!(
            "{set:<16} {:>6} {:>7.1}{:>7.1}{:>7.1}{:>8.1} {:>7.1}{:>7.1}{:>7.1}{:>8.1} {:>+9.1} {:>7.3}",
            s.nights,
            s.ours[0], s.ours[1], s.ours[2], s.ours[3],
            s.truth[0], s.truth[1], s.truth[2], s.truth[3],
            s.ours[DEEP] - s.truth[DEEP],
            s.kappa,
        );
    }
    println!("{}", psg_census.line("PSG staging"));

    // ── 5. efficiency ───────────────────────────────────────────────────────────────────────────
    println!("\n=== 5. sleep efficiency, asleep as % of the export's in-bed window ===");
    println!("{:<30} {:>8} {:>10}", "source", "nights", "efficiency");
    println!("{:<30} {:>8} {:>10.1}", "WHOOP export", whoop_all.len(), 100.0 - mean_rows(&whoop_all)[0]);
    println!(
        "{:<30} {:>8} {:>10.1}",
        "shipped, stage_v2",
        plain.len(),
        100.0 - mean_rows(&col(&plain, false))[0]
    );
    if !both.is_empty() {
        println!(
            "{:<30} {:>8} {:>10.1}",
            "shipped, + refine_wake",
            both.len(),
            100.0 - mean_rows(&both.iter().map(|r| r.shipped_refined).collect::<Vec<_>>())[0]
        );
    }

    // ── 6. bounded rescue ───────────────────────────────────────────────────────────────────────
    println!("\n=== 6. three bounded rescues on the deep under-call, each on BOTH references ===");
    println!("export deep err = ours - WHOOP over the paired nights; PSG columns are deep err then kappa4");
    print!("{:<34} {:>9}", "candidate", "export");
    for (set, _, _) in &psg {
        print!("{:>18}", set);
    }
    println!();
    let mut families: Vec<(String, Params)> = Vec::new();
    for v in [0.20, 0.25, 0.30] {
        families.push((format!("base_rate deep {v:.2}"), Params { base_rate: [v, 0.22, 0.50, 0.34], ..shipped }));
    }
    for v in [0.50, 0.60, 0.70] {
        families.push((format!("deep_gate_thresh {v:.2}"), Params { deep_gate_thresh: v, ..shipped }));
    }
    for v in [1.6, 2.0, 2.4] {
        families.push((format!("cycle_deep_scale {v:.1}"), Params { cycle_deep_scale: v, ..shipped }));
    }
    for (name, p) in [("SHIPPED".to_string(), shipped)].iter().chain(families.iter()) {
        assert_prepared_with(&shipped, p);
        let (ours, whoop) = export_pair(&nights, &ours_prep, &export, &paired, p);
        print!("{name:<34} {:>+9.1}", ours[DEEP] - whoop[DEEP]);
        for (_, ns, prep) in &psg {
            let mut c = RefineCensus::default();
            match psg_score(ns, prep, p, &mut c) {
                Some(s) => print!("{:>+11.1}{:>7.3}", s.ours[DEEP] - s.truth[DEEP], s.kappa),
                None => print!("{:>18}", "-"),
            }
        }
        println!();
    }
}
