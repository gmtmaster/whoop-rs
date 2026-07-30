//! Step H — the wake refinement: what it converts, what that costs, and which reference says so.
//!
//!   cargo run --release -p physio-algo --example wake_refine
//!
//! 1  the pass reproduced, unrefined against both rules, with the density census beside it
//! 2  what it converts: the band's own call on the seconds it flips, against the wake it keeps
//! 3  the OTHER reference — the WHOOP export's wake fraction and efficiency, which wants the pass ON
//! 4  why no PSG cohort can score it, asked through the shipped gate rather than assumed
//! 5  three bounded rescues over the eligibility rule, each scored on BOTH references
//! 6  the adopted rule second by second: which flips it holds back, and whether they were wrong
//!
//! Every rescue is a delta on [`pre_h`], the rule H opened with, so the evidence for the fix outlives the
//! fix — re-running this after the change still prints what the change was worth. Sections 1, 2, 5 and 6
//! read `continuous` (real seconds, whole wear blocks, the band at 1 Hz, and the only set whose gravity is
//! NOT held forward). Section 3 and the export half of 5 read `ours` re-sliced onto the export's own in-bed
//! window; `ours` gravity IS held forward, so its posture check reads optimistically and the two references
//! are reported apart, never pooled.

mod common;

use common::{
    dirs_of, median, night_id, pair_nearest, read_accel, read_band, read_hr, read_meta, read_rr, read_steps,
    stage_at, BAND_ASLEEP, Export, RefineCensus, TwoClass, WAKE,
};

use physio_algo::sleep::{
    detect_sessions, epoch_starts_v2, motion_density, params::Params, prepare_v2, refine_wake_with,
    stage_v2_prepared, AccelSample, Prepared, RefineParams, SleepInput, SleepStage, StepSample,
    MIN_DENSE_FRACTION,
};

const EPOCH_SEC: i64 = 30;
const PSG: [&str; 3] = ["dreamt", "aauwss", "sleep-accel"];
/// How far apart a staged night and an export row may sit and still be the same night.
const MATCH_SLACK_S: i64 = 4 * 3600;

/// The eligibility rule H opened with: every wake run eligible, the window's edges included. Every delta
/// below is measured against it, so the step's own justification stays reproducible after the change.
fn pre_h() -> RefineParams {
    RefineParams { skip_window_edges: false, ..RefineParams::SHIPPED }
}

// ── the band reference ────────────────────────────────────────────────────────────────────────────

fn row(tc: &TwoClass, label: &str) {
    println!(
        "   {label:<44} {:>7.1}% {:>8.1}% {:>8.3} {:>8.1} {:>10.1}",
        tc.pred_pct(),
        tc.true_pct(),
        tc.kappa(),
        tc.recall(),
        tc.precision()
    );
}

fn header() {
    println!(
        "   {:<44} {:>8} {:>9} {:>8} {:>8} {:>10}",
        "config", "ours w%", "band w%", "kappa2", "recall", "precision"
    );
}

// ── the continuous rig ────────────────────────────────────────────────────────────────────────────

/// One detected span with its staging inputs, the band over it, and the streams the refinement reads.
struct Span {
    prep: Prepared,
    accel: Vec<AccelSample>,
    steps: Vec<StepSample>,
    band: Vec<(i64, i32)>,
    dense: bool,
}

fn continuous_spans(p: &Params) -> Vec<Span> {
    let mut out = Vec::new();
    for d in dirs_of("continuous") {
        let band = read_band(&d);
        let accel = read_accel(&d);
        if band.is_empty() || accel.len() < 120 {
            continue;
        }
        let (hr, rr, steps) = (read_hr(&d), read_rr(&d), read_steps(&d));
        for s in detect_sessions(&hr, &accel, 0, &[], &band, None) {
            let input = SleepInput {
                start: s.start,
                end: s.end,
                hr: hr.iter().filter(|h| h.ts >= s.start && h.ts < s.end).cloned().collect(),
                rr: rr.iter().filter(|r| r.ts >= s.start && r.ts < s.end).cloned().collect(),
                accel: accel.iter().filter(|g| g.ts >= s.start && g.ts < s.end).cloned().collect(),
            };
            if input.hr.len() < 120 || input.accel.len() < 120 {
                continue;
            }
            let span_steps: Vec<StepSample> =
                steps.iter().filter(|t| t.ts >= s.start && t.ts < s.end).cloned().collect();
            let (g, st) = motion_density(s.start, s.end, &input.accel, &span_steps);
            out.push(Span {
                accel: input.accel.clone(),
                steps: span_steps,
                band: band.iter().filter(|(t, _)| *t >= s.start && *t < s.end).cloned().collect(),
                dense: g >= MIN_DENSE_FRACTION && st >= MIN_DENSE_FRACTION,
                prep: prepare_v2(&input, p),
            });
        }
    }
    out
}

// ── 1  the pass reproduced ────────────────────────────────────────────────────────────────────────

fn section_reproduce(spans: &[Span], p: &Params) {
    println!("1  the refinement reproduced, on the band");
    println!("   `analyze` ends with refine_wake, which only ever SHRINKS wake. All three rows are the same");
    println!("   staging over the same seconds; the last two have a pass applied where the gate allows.");
    let (mut raw, mut old, mut new) = (TwoClass::default(), TwoClass::default(), TwoClass::default());
    let (mut raw_d, mut old_d, mut new_d) = (TwoClass::default(), TwoClass::default(), TwoClass::default());
    let mut census = RefineCensus::default();
    let (mut moved_old, mut moved_new) = (0i64, 0i64);
    for s in spans {
        let segs = stage_v2_prepared(&s.prep, p);
        let a = refine_wake_with(&segs, &s.accel, &s.steps, &pre_h());
        let b = census.refine(&segs, &s.accel, &s.steps);
        for &(ts, code) in &s.band {
            let (Some(x), Some(y), Some(z)) = (stage_at(&segs, ts), stage_at(&a, ts), stage_at(&b, ts)) else {
                continue;
            };
            let awake = code != BAND_ASLEEP;
            raw.add(x == SleepStage::Wake, awake);
            old.add(y == SleepStage::Wake, awake);
            new.add(z == SleepStage::Wake, awake);
            if s.dense {
                raw_d.add(x == SleepStage::Wake, awake);
                old_d.add(y == SleepStage::Wake, awake);
                new_d.add(z == SleepStage::Wake, awake);
            }
            moved_old += i64::from(x != y);
            moved_new += i64::from(x != z);
        }
    }
    println!("\n   {} detected spans over the band-carrying blocks", spans.len());
    println!("{}", census.line("the refinement"));
    println!("   band seconds moved: {moved_old} under the pre-H rule, {moved_new} under the shipped one\n");
    header();
    row(&raw, "stage_v2, all spans");
    row(&old, "+ refine_wake pre-H, all spans");
    row(&new, "+ refine_wake SHIPPED, all spans (the app)");
    println!("\n   the same rows over the {} spans the gate accepted — the pass's own effect:", census.refined);
    header();
    row(&raw_d, "stage_v2, gate-accepted");
    row(&old_d, "+ refine_wake pre-H, gate-accepted");
    row(&new_d, "+ refine_wake SHIPPED, gate-accepted");
    println!(
        "   over-call ratio {:.2}x unrefined -> {:.2}x pre-H -> {:.2}x shipped",
        raw_d.pred_pct() / raw_d.true_pct(),
        old_d.pred_pct() / old_d.true_pct(),
        new_d.pred_pct() / new_d.true_pct()
    );
    println!(
        "   kappa2 against the unrefined staging: pre-H {:+.3}, shipped {:+.3}",
        old_d.kappa() - raw_d.kappa(),
        new_d.kappa() - raw_d.kappa()
    );
}

// ── 2  what it converts ───────────────────────────────────────────────────────────────────────────

/// The question H exists for: is the eligibility rule SELECTING the wake we get wrong, or taking a slice
/// of it at random. Compare the band's call on the seconds it flips against the wake it leaves. Measured
/// on the pre-H rule, because that is the defect the fix was derived from.
fn section_converts(spans: &[Span], p: &Params) {
    println!("\n2  what the pre-H pass converts, judged by the band");
    println!("   A rule that removed our FALSE wake would flip seconds the band calls asleep more often");
    println!("   than the wake it keeps. Anything at or below the kept column is removing true wake too.");
    let (mut flip, mut kept) = ((0i64, 0i64), (0i64, 0i64));
    let mut region = [[(0i64, 0i64); 2]; 3]; // [head|interior|tail][flipped|kept]
    let (mut conv_runs, mut kept_runs) = (Vec::new(), Vec::new());
    for s in spans.iter().filter(|s| s.dense) {
        let segs = stage_v2_prepared(&s.prep, p);
        let out = refine_wake_with(&segs, &s.accel, &s.steps, &pre_h());
        let asleep: Vec<i64> = s.band.iter().filter(|(_, c)| *c == BAND_ASLEEP).map(|(t, _)| *t).collect();
        let (Some(&a0), Some(&a1)) = (asleep.first(), asleep.last()) else { continue };
        for &(ts, code) in &s.band {
            let (Some(a), Some(b)) = (stage_at(&segs, ts), stage_at(&out, ts)) else { continue };
            if a != SleepStage::Wake {
                continue;
            }
            let r = if ts < a0 { 0 } else if ts > a1 { 2 } else { 1 };
            let bucket = usize::from(b == SleepStage::Wake); // 0 = flipped, 1 = kept
            let cell = if bucket == 0 { &mut flip } else { &mut kept };
            cell.0 += i64::from(code == BAND_ASLEEP);
            cell.1 += 1;
            region[r][bucket].0 += i64::from(code == BAND_ASLEEP);
            region[r][bucket].1 += 1;
        }
        // Run lengths, so "it converts the long ones" is a number and not a mechanism read off the code.
        for g in segs.iter().filter(|g| g.stage == SleepStage::Wake) {
            let mid = (g.start + g.end) / 2;
            let mins = (g.end - g.start) as f64 / 60.0;
            match stage_at(&out, mid) {
                Some(SleepStage::Wake) => kept_runs.push(mins),
                Some(_) => conv_runs.push(mins),
                None => {}
            }
        }
    }
    let pct = |c: (i64, i64)| 100.0 * c.0 as f64 / c.1.max(1) as f64;
    println!("\n   {:<34} {:>12} {:>18}", "our wake seconds", "seconds", "band says ASLEEP");
    println!("   {:<34} {:>12} {:>17.1}%", "FLIPPED to light by the pass", flip.1, pct(flip));
    println!("   {:<34} {:>12} {:>17.1}%", "KEPT as wake", kept.1, pct(kept));
    println!(
        "   {:<34} {:>12} {:>17.1}%",
        "both (the unrefined wake)",
        flip.1 + kept.1,
        pct((flip.0 + kept.0, flip.1 + kept.1))
    );
    println!("\n   the same split by region of the span — the head and tail sit outside the strap's asleep");
    println!("   run by construction, so a flip there is wrong on every second and needs no judgement call");
    println!(
        "   {:<12} {:>10} {:>14} {:>8} {:>10} {:>14} {:>8}",
        "region", "flipped", "band asleep", "share", "kept", "band asleep", "share"
    );
    for (i, name) in ["head", "interior", "tail"].iter().enumerate() {
        println!(
            "   {name:<12} {:>10} {:>14} {:>7.1}% {:>10} {:>14} {:>7.1}%",
            region[i][0].1,
            region[i][0].0,
            pct(region[i][0]),
            region[i][1].1,
            region[i][1].0,
            pct(region[i][1])
        );
    }
    println!(
        "   {} of the {} flipped seconds sit at an edge, and {} of those are right.",
        region[0][0].1 + region[2][0].1,
        flip.1,
        region[0][0].0 + region[2][0].0
    );
    println!(
        "\n   converted runs: {}, median {:.1} min · kept runs: {}, median {:.1} min",
        conv_runs.len(),
        median(&mut conv_runs.clone()),
        kept_runs.len(),
        median(&mut kept_runs.clone())
    );
    println!(
        "   (a kept run under the {:.0}-min floor was never eligible; one over it failed the locomotion",
        RefineParams::SHIPPED.min_wake_segment_seconds as f64 / 60.0
    );
    println!("    or the posture test.)");
}

// ── 3  the other reference ────────────────────────────────────────────────────────────────────────

/// One `ours` night: its streams, the real onset that pairs it to an export row, and its fixture window.
struct Night {
    owner: String,
    onset: i64,
    w0: i64,
    input: SleepInput,
    accel: Vec<AccelSample>,
    steps: Vec<StepSample>,
}

fn load_ours() -> Vec<Night> {
    let mut out = Vec::new();
    for d in dirs_of("ours") {
        let Some((w0, w1, _)) = read_meta(&d) else { continue };
        let (owner, onset) = night_id(&d);
        let (accel, hr) = (read_accel(&d), read_hr(&d));
        if hr.len() < 120 || accel.len() < 120 {
            continue;
        }
        out.push(Night {
            owner,
            onset,
            w0,
            steps: read_steps(&d),
            input: SleepInput { start: w0, end: w1, hr, rr: read_rr(&d), accel: accel.clone() },
            accel,
        });
    }
    out
}

/// Our wake share over the export's own in-bed window, under one refinement rule. `None` when the density
/// gate declines the night, so a declined night can never be pooled into a figure labelled refined.
fn export_wake_pct(n: &Night, e: &Export, prep: &Prepared, p: &Params, rp: Option<&RefineParams>) -> Option<f64> {
    let segs = stage_v2_prepared(prep, p);
    let segs = match rp {
        None => segs,
        Some(rp) => {
            let (g, s) = motion_density(n.input.start, n.input.end, &n.accel, &n.steps);
            if g < MIN_DENSE_FRACTION || s < MIN_DENSE_FRACTION {
                return None;
            }
            refine_wake_with(&segs, &n.accel, &n.steps, rp)
        }
    };
    let shift = n.onset - n.w0;
    let (mut wake, mut all) = (0i64, 0i64);
    for start in epoch_starts_v2(prep) {
        let mid = start + EPOCH_SEC / 2;
        if mid + shift < e.onset || mid + shift >= e.wake {
            continue;
        }
        let s = segs.iter().find(|g| g.start <= mid && mid < g.end).or(segs.last()).map(|g| g.stage);
        all += 1;
        wake += i64::from(s == Some(SleepStage::Wake));
    }
    (all > 0).then(|| 100.0 * wake as f64 / all as f64)
}

/// Mean of the per-night wake share under `rp`, over the nights that produce one, beside WHOOP's own.
fn export_mean(
    nights: &[Night], preps: &[Prepared], export: &[Export], paired: &[(usize, usize)], p: &Params,
    rp: Option<&RefineParams>,
) -> (usize, f64, f64) {
    let (mut ours, mut whoop) = (Vec::new(), Vec::new());
    for &(i, j) in paired {
        if let Some(w) = export_wake_pct(&nights[i], &export[j], &preps[i], p, rp) {
            ours.push(w);
            whoop.push(export[j].frac[WAKE]);
        }
    }
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len().max(1) as f64;
    (ours.len(), mean(&ours), mean(&whoop))
}

/// The paired nights whose density gate accepts, so every row of section 3 shares one denominator.
fn dense_pairs(
    nights: &[Night], preps: &[Prepared], export: &[Export], paired: &[(usize, usize)], p: &Params,
) -> Vec<(usize, usize)> {
    paired
        .iter()
        .copied()
        .filter(|&(i, j)| {
            export_wake_pct(&nights[i], &export[j], &preps[i], p, Some(&RefineParams::SHIPPED)).is_some()
        })
        .collect()
}

fn section_export(
    nights: &[Night], preps: &[Prepared], export: &[Export], paired: &[(usize, usize)], dense: &[(usize, usize)],
    p: &Params,
) {
    println!("\n3  the other reference: the WHOOP export's own wake fraction and efficiency");
    println!("   The band scores per second and reads placement. The export scores the NIGHT, and it is the");
    println!("   reference the pass was built for. Efficiency is 1 - wake by construction, so these are one");
    println!("   quantity read two ways — the second is the way a user meets it.");
    let (n_all, ours_all, whoop_all) = export_mean(nights, preps, export, paired, p, None);
    let (_, plain, whoop) = export_mean(nights, preps, export, dense, p, None);
    let (n_old, old, _) = export_mean(nights, preps, export, dense, p, Some(&pre_h()));
    let (n_new, new, _) = export_mean(nights, preps, export, dense, p, Some(&RefineParams::SHIPPED));
    println!("\n   {:<40} {:>8} {:>10} {:>12} {:>12}", "path", "nights", "ours w%", "WHOOP w%", "error");
    let row = |label: &str, n: usize, o: f64, w: f64| {
        println!("   {label:<40} {n:>8} {o:>10.1} {w:>12.1} {:>+12.1}", o - w);
    };
    row("stage_v2, every paired night", n_all, ours_all, whoop_all);
    row("stage_v2, gate-accepted nights", dense.len(), plain, whoop);
    row("+ refine_wake pre-H", n_old, old, whoop);
    row("+ refine_wake SHIPPED (the app)", n_new, new, whoop);
    println!(
        "\n   efficiency: WHOOP {:.1}%, stage_v2 {:.1}%, pre-H {:.1}%, SHIPPED {:.1}%",
        100.0 - whoop,
        100.0 - plain,
        100.0 - old,
        100.0 - new
    );
}

// ── 4  why PSG cannot score it ────────────────────────────────────────────────────────────────────

fn section_psg_cannot() {
    println!("\n4  why no PSG cohort can score this step — asked through the shipped gate");
    println!(
        "   {:<16} {:>8} {:>10} {:>12} {:>10} {:>18}",
        "cohort", "nights", "declined", "on gravity", "on steps", "step cadence"
    );
    for set in PSG {
        let (mut n, mut declined, mut on_g, mut on_s) = (0, 0, 0, 0);
        let mut gaps: Vec<f64> = Vec::new();
        for d in dirs_of(set) {
            let Some((w0, w1, _)) = read_meta(&d) else { continue };
            let (accel, steps) = (read_accel(&d), read_steps(&d));
            if accel.len() < 120 {
                continue;
            }
            n += 1;
            let (g, s) = motion_density(w0, w1, &accel, &steps);
            if g < MIN_DENSE_FRACTION || s < MIN_DENSE_FRACTION {
                declined += 1;
                on_g += usize::from(g < MIN_DENSE_FRACTION);
                on_s += usize::from(s < MIN_DENSE_FRACTION);
            }
            for w in steps.windows(2) {
                gaps.push((w[1].ts - w[0].ts) as f64);
            }
        }
        let cadence =
            if gaps.is_empty() { "no step stream".to_string() } else { format!("{:.0} s median", median(&mut gaps)) };
        println!("   {set:<16} {n:>8} {declined:>10} {on_g:>12} {on_s:>10} {cadence:>18}");
    }
    println!("\n   The gate wants a step sample a MINUTE. A cohort with no step stream, or one sampled far");
    println!("   slower, declines on that stream alone — its accelerometer passes. So H's only per-epoch");
    println!("   reference is the band, and its second reference scores nights rather than epochs.");
}

// ── 5  three bounded rescues ──────────────────────────────────────────────────────────────────────

/// The three families, each the eligibility rule read a different way: how long a run has to be, how still
/// it has to be, and whether it may touch the window's edge. One family per mechanism, so the bound of
/// three attempts is visible in the output instead of asserted in a document.
fn rescues() -> Vec<(&'static str, String, RefineParams)> {
    let s = pre_h();
    let mut out = Vec::new();
    for v in [60i64, 120, 150, 600] {
        out.push((
            "width",
            format!("min wake segment {:.1} min (was 5.0)", v as f64 / 60.0),
            RefineParams { min_wake_segment_seconds: v, ..s },
        ));
    }
    for v in [0.90f64, 0.95, 1.00] {
        out.push((
            "stability",
            format!("min stable minutes {v:.2} (was 0.80)"),
            RefineParams { min_stable_minute_fraction: v, ..s },
        ));
    }
    for v in [0.02f64, 0.01] {
        out.push((
            "stability",
            format!("stable variance {v:.2} g2 (was 0.05)"),
            RefineParams { stable_posture_variance_g2: v, ..s },
        ));
    }
    out.push(("edges", "skip the window's first and last segment  ** ADOPTED **".to_string(), RefineParams::SHIPPED));
    out
}

/// Band kappa2 and our wake share over the gate-accepted spans under one refinement rule.
fn band_score(spans: &[Span], p: &Params, rp: Option<&RefineParams>) -> TwoClass {
    let mut tc = TwoClass::default();
    for s in spans.iter().filter(|s| s.dense) {
        let segs = stage_v2_prepared(&s.prep, p);
        let out = match rp {
            Some(rp) => refine_wake_with(&segs, &s.accel, &s.steps, rp),
            None => segs,
        };
        for &(ts, code) in &s.band {
            let Some(g) = stage_at(&out, ts) else { continue };
            tc.add(g == SleepStage::Wake, code != BAND_ASLEEP);
        }
    }
    tc
}

fn section_rescues(
    spans: &[Span], nights: &[Night], preps: &[Prepared], export: &[Export], dense: &[(usize, usize)],
    p: &Params,
) {
    println!("\n5  three bounded rescues over the eligibility rule, each on BOTH references");
    println!("   Every row is a delta on the pre-H rule. A candidate is an improvement only if neither");
    println!("   column is negative: band kappa2 up AND the export's wake error no larger.");
    let base = band_score(spans, p, Some(&pre_h()));
    let (_, base_e, whoop) = export_mean(nights, preps, export, dense, p, Some(&pre_h()));
    let base_err = base_e - whoop;
    println!(
        "\n   pre-H: band kappa2 {:.3}, our band wake {:.1}% · export wake error {base_err:+.1} pp (ours {base_e:.1} vs WHOOP {whoop:.1})",
        base.kappa(),
        base.pred_pct()
    );
    println!(
        "\n   {:<12} {:<46} {:>11} {:>10} {:>13} {:>10}",
        "family", "candidate", "band d-k2", "band w%", "export err", "d-|err|"
    );
    for (family, label, rp) in rescues() {
        let tc = band_score(spans, p, Some(&rp));
        let (_, e, _) = export_mean(nights, preps, export, dense, p, Some(&rp));
        let err = e - whoop;
        println!(
            "   {family:<12} {label:<46} {:>+11.3} {:>9.1}% {err:>+13.1} {:>+10.1}",
            tc.kappa() - base.kappa(),
            tc.pred_pct(),
            err.abs() - base_err.abs()
        );
    }
    let off = band_score(spans, p, None);
    let (_, off_e, _) = export_mean(nights, preps, export, dense, p, None);
    println!(
        "\n   for scale, the pass OFF entirely: band kappa2 {:+.3}, our band wake {:.1}%, export wake error {:+.1} pp",
        off.kappa() - base.kappa(),
        off.pred_pct(),
        off_e - whoop
    );
    println!("   The adopted rule beats BOTH the pre-H pass and no pass at all on the band, and gives up");
    println!("   almost none of the export benefit the pass exists for.");
}

// ── 6  what the adopted rule changed ──────────────────────────────────────────────────────────────

/// A ranking says which row is highest; this says whether the seconds the adopted rule stops flipping are
/// ones the pre-H rule was getting wrong.
fn section_adopted(spans: &[Span], p: &Params) {
    let (mut held, mut shared) = ((0i64, 0i64), (0i64, 0i64));
    for s in spans.iter().filter(|s| s.dense) {
        let segs = stage_v2_prepared(&s.prep, p);
        let a = refine_wake_with(&segs, &s.accel, &s.steps, &pre_h());
        let b = refine_wake_with(&segs, &s.accel, &s.steps, &RefineParams::SHIPPED);
        for &(ts, code) in &s.band {
            let (Some(raw), Some(x), Some(y)) = (stage_at(&segs, ts), stage_at(&a, ts), stage_at(&b, ts)) else {
                continue;
            };
            if raw != SleepStage::Wake || x == SleepStage::Wake {
                continue; // not a second the pre-H rule flips
            }
            let cell = if y == SleepStage::Wake { &mut held } else { &mut shared };
            cell.0 += i64::from(code == BAND_ASLEEP);
            cell.1 += 1;
        }
    }
    let pct = |c: (i64, i64)| 100.0 * c.0 as f64 / c.1.max(1) as f64;
    println!("\n6  the adopted rule second by second — of the seconds the pre-H rule flips:");
    println!("   {:<40} {:>10} {:>18}", "held back as wake", held.1, format!("{:.1}% band asleep", pct(held)));
    println!("   {:<40} {:>10} {:>18}", "still flipped", shared.1, format!("{:.1}% band asleep", pct(shared)));
    println!(
        "   A flip is RIGHT when the band says asleep, so the change trades {} wrong flips away against {}",
        held.1 - held.0,
        held.0
    );
    println!("   right ones, and leaves the {} it still makes at {:.1}% right.", shared.1, pct(shared));
}

fn main() {
    let shipped = Params::SHIPPED;
    let spans = continuous_spans(&shipped);
    if spans.is_empty() {
        println!("no `continuous` spans under {}", common::fixtures_root().display());
        return;
    }
    section_reproduce(&spans, &shipped);
    section_converts(&spans, &shipped);

    let nights = load_ours();
    let export = common::read_export();
    let preps: Vec<Prepared> = nights.iter().map(|n| prepare_v2(&n.input, &shipped)).collect();
    let keys_n: Vec<(String, i64)> = nights.iter().map(|n| (n.owner.clone(), n.onset)).collect();
    let keys_e: Vec<(String, i64)> = export.iter().map(|e| (e.owner.clone(), e.onset)).collect();
    let paired = pair_nearest(&keys_n, &keys_e, MATCH_SLACK_S);
    if paired.is_empty() {
        println!("\nno export pairing, so section 3 and the export half of 5 cannot run");
        return;
    }
    let dense = dense_pairs(&nights, &preps, &export, &paired, &shipped);
    section_export(&nights, &preps, &export, &paired, &dense, &shipped);
    section_psg_cannot();
    section_rescues(&spans, &nights, &preps, &export, &dense, &shipped);
    section_adopted(&spans, &shipped);
}
