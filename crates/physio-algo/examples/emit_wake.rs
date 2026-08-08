//! Step F — the wake over-call, measured on the path the app actually runs.
//!
//!   cargo run --release -p physio-algo --example emit_wake
//!
//! 1  the over-call reproduced, unrefined and then REFINED, with the reference's base rate beside it
//! 2  where the excess sits: the in-bed-awake lead-in, the interior, and our wake run lengths
//! 3  is it reducible: every term feeding the AWAKE emission swept, scored on two-class kappa
//! 4  the cycle prior: F1's onset-anchored guard and F3's onset-anchored clock, and what the anchor is
//! 5  what each candidate costs on PSG truth — 4-class kappa, three cohorts, stated per cohort
//!
//! Section 1 reads `continuous` (real unix seconds, whole wear blocks, the band per second, and the only
//! set whose gravity is NOT held forward — so the refinement's posture check sees real dropouts).
//! Section 3 reads `ours` (window baked in, cheap to re-label). Section 4 reads the three PSG sets.

mod common;

use common::{
    dirs_of, kappa4, median, read_accel, read_band, read_hr, read_meta, read_rr, read_steps, read_truth,
    stage_at, stage_idx, BAND_ASLEEP, RefineCensus, TwoClass, WAKE,
};

use std::collections::BTreeMap;
use std::path::Path;

use physio_algo::sleep::{
    detect_sessions, motion_dense, params::Params, prepare_v2, refine_wake, stage_v2_prepared, AccelSample,
    HrSample, Prepared, RrRun, SleepInput, SleepStage, StageSegment, StepSample,
};

const EPOCH_SEC: i64 = 30;
/// One epoch in minutes, for reporting an epoch index as a time.
const EPOCH_MIN_F: f64 = 0.5;
/// The PSG cohorts, named so every kappa below says which population it came from.
const PSG: [&str; 3] = ["dreamt", "aauwss", "sleep-accel"];
/// The three rescue families of section 3. One finalist from each reaches the ledger, so the bound of
/// three attempts is visible in the output rather than asserted in a document.
const PROPENSITY: &str = "propensity";
const MOTION_GATE: &str = "motion gate";
const BLIP_FILTER: &str = "blip filter";

fn row(tc: &TwoClass, label: &str) {
    println!(
        "   {label:<40} {:>7.1}% {:>9.1}% {:>8.3} {:>9.1} {:>10.1} {:>7.1}",
        tc.pred_pct(),
        tc.true_pct(),
        tc.kappa(),
        tc.recall(),
        tc.precision(),
        tc.f1()
    );
}

fn header() {
    println!(
        "   {:<40} {:>8} {:>10} {:>8} {:>9} {:>10} {:>7}",
        "config", "ours w%", "ref w%", "kappa2", "recall", "precision", "F1"
    );
}

// ── the continuous rig: whole wear blocks, real seconds, the band at 1 Hz ──────────────────────────

struct Block {
    hr: Vec<HrSample>,
    rr: Vec<RrRun>,
    accel: Vec<AccelSample>,
    steps: Vec<StepSample>,
    band: Vec<(i64, i32)>,
}

fn load_block(d: &Path) -> Option<Block> {
    let (band, accel) = (read_band(d), read_accel(d));
    if band.is_empty() || accel.len() < 120 {
        return None;
    }
    Some(Block { hr: read_hr(d), rr: read_rr(d), accel, steps: read_steps(d), band })
}

/// One detected span with its staging, the band over it, and the streams the refinement reads.
struct Span {
    prep: Prepared,
    accel: Vec<AccelSample>,
    steps: Vec<StepSample>,
    band: Vec<(i64, i32)>,
}

fn continuous_spans(p: &Params) -> Vec<Span> {
    let mut out = Vec::new();
    for d in dirs_of("continuous") {
        let Some(b) = load_block(&d) else { continue };
        for s in detect_sessions(&b.hr, &b.accel, 0, &[], &b.band, None) {
            let input = SleepInput {
                start: s.start,
                end: s.end,
                hr: b.hr.iter().filter(|h| h.ts >= s.start && h.ts < s.end).cloned().collect(),
                rr: b.rr.iter().filter(|r| r.ts >= s.start && r.ts < s.end).cloned().collect(),
                accel: b.accel.iter().filter(|g| g.ts >= s.start && g.ts < s.end).cloned().collect(),
            };
            if input.hr.len() < 120 || input.accel.len() < 120 {
                continue;
            }
            out.push(Span {
                accel: input.accel.clone(),
                steps: b.steps.iter().filter(|t| t.ts >= s.start && t.ts < s.end).cloned().collect(),
                band: b.band.iter().filter(|(t, _)| *t >= s.start && *t < s.end).cloned().collect(),
                prep: prepare_v2(&input, p),
            });
        }
    }
    out
}

/// One recipe under test: the coefficients, the blip width the candidate post-pass drops (0 = off), and
/// the rescue family it belongs to, so the ledger reports one finalist per family instead of five of one.
struct Recipe {
    label: String,
    family: &'static str,
    p: Params,
    blip_sec: i64,
}

fn recipe(label: &str, p: Params, blip_sec: i64) -> Recipe {
    family_recipe(label, "-", p, blip_sec)
}

fn family_recipe(label: &str, family: &'static str, p: Params, blip_sec: i64) -> Recipe {
    Recipe { label: label.to_string(), family, p, blip_sec }
}

/// Candidate post-pass: a wake run shorter than `max_sec` becomes its longer neighbour's stage. Aimed at
/// the interior fragmentation only, so a real in-bed-awake stretch is untouched. Off when `max_sec == 0`.
fn drop_wake_blips(segs: &[StageSegment], max_sec: i64) -> Vec<StageSegment> {
    if max_sec <= 0 || segs.len() < 2 {
        return segs.to_vec();
    }
    let mut out: Vec<StageSegment> = Vec::with_capacity(segs.len());
    for (i, s) in segs.iter().enumerate() {
        let mut stage = s.stage;
        if stage == SleepStage::Wake && s.end - s.start < max_sec {
            let prev = i.checked_sub(1).map(|j| segs[j]);
            let next = segs.get(i + 1).copied();
            stage = match (prev, next) {
                (Some(a), Some(b)) => {
                    if a.end - a.start >= b.end - b.start { a.stage } else { b.stage }
                }
                (Some(a), None) => a.stage,
                (None, Some(b)) => b.stage,
                (None, None) => SleepStage::Light,
            };
        }
        match out.last_mut() {
            Some(last) if last.stage == stage => last.end = s.end,
            _ => out.push(StageSegment { start: s.start, end: s.end, stage }),
        }
    }
    out
}

// ── 1  the over-call, unrefined and refined ───────────────────────────────────────────────────────

fn section_refinement(spans: &[Span], p: &Params) {
    println!("1  the wake over-call, before and after the stage the app runs last");
    println!("   `analyze` is detect -> stage -> refine_wake, and refine_wake only ever SHRINKS wake. Every");
    println!("   A-Z wake figure so far staged with stage_v2 alone, because no fixture carried the step");
    println!("   stream its density gate needs. Both rows below are the same staging, scored per band second.");

    let (mut raw, mut ref_) = (TwoClass::default(), TwoClass::default());
    let (mut raw_dense, mut ref_dense) = (TwoClass::default(), TwoClass::default());
    let mut census = RefineCensus::default();
    let mut moved_sec = 0i64;
    let mut steps_rows = 0usize;
    for s in spans {
        let segs = stage_v2_prepared(&s.prep, p);
        // Through the census, not straight to `refine_wake`: a non-empty steps file is not the same thing
        // as a stream dense enough for the gate, and the gate declines in silence.
        let mut probe = RefineCensus::default();
        let fine = probe.refine(&segs, &s.accel, &s.steps);
        let dense = probe.all_refined();
        census.absorb(&probe);
        steps_rows += s.steps.len();
        for &(ts, st) in &s.band {
            let (Some(a), Some(b)) = (stage_at(&segs, ts), stage_at(&fine, ts)) else { continue };
            let strap = st != BAND_ASLEEP;
            raw.add(a == SleepStage::Wake, strap);
            ref_.add(b == SleepStage::Wake, strap);
            if dense {
                raw_dense.add(a == SleepStage::Wake, strap);
                ref_dense.add(b == SleepStage::Wake, strap);
            }
            moved_sec += i64::from(a != b);
        }
    }
    println!("\n   {} detected spans, {} step rows", spans.len(), steps_rows);
    println!("{}", census.line("the over-call comparison"));
    println!("   the refinement moves {moved_sec} band seconds\n");
    header();
    row(&raw, "stage_v2 only (every prior A-Z number)");
    row(&ref_, "stage_v2 + refine_wake (the app's path)");
    println!(
        "\n   over-call ratio {:.2}x unrefined -> {:.2}x refined",
        raw.pred_pct() / raw.true_pct(),
        ref_.pred_pct() / ref_.true_pct()
    );
    // The two rows above pool the declined spans, whose "refined" labels are their unrefined ones. That
    // is what the app holds over these spans, but it understates what the refinement DOES; the pair below
    // is the same comparison over the spans the gate accepted, which is the refinement's own effect.
    println!("\n   the same comparison over the {} spans the gate accepted:", census.refined);
    header();
    row(&raw_dense, "stage_v2, gate-accepted spans only");
    row(&ref_dense, "+ refine_wake, gate-accepted spans only");
    println!(
        "   over-call ratio {:.2}x -> {:.2}x on those spans",
        raw_dense.pred_pct() / raw_dense.true_pct(),
        ref_dense.pred_pct() / ref_dense.true_pct()
    );
}

// ── 2  where the excess sits ──────────────────────────────────────────────────────────────────────

fn section_where(spans: &[Span], p: &Params) {
    println!("\n2  where the excess wake sits inside a span");
    println!("   A detected span opens before the strap calls sleep, so its head is real in-bed-awake time");
    println!("   the reference also calls awake. Splitting head / interior / tail says whether the over-call");
    println!("   is the lead-in or fragmentation of the sleep itself.");
    let (mut head, mut interior, mut tail) = (TwoClass::default(), TwoClass::default(), TwoClass::default());
    let mut runs: Vec<f64> = Vec::new();
    let mut short_runs = 0i64;
    let mut run_sec_total = 0i64;
    for s in spans {
        let fine = refine_wake(&stage_v2_prepared(&s.prep, p), &s.accel, &s.steps);
        let asleep: Vec<i64> = s.band.iter().filter(|(_, st)| *st == BAND_ASLEEP).map(|(t, _)| *t).collect();
        let (a0, a1) = match (asleep.first(), asleep.last()) {
            (Some(a), Some(b)) => (*a, *b),
            _ => continue,
        };
        for &(ts, st) in &s.band {
            let Some(g) = stage_at(&fine, ts) else { continue };
            let bucket = if ts < a0 { &mut head } else if ts > a1 { &mut tail } else { &mut interior };
            bucket.add(g == SleepStage::Wake, st != BAND_ASLEEP);
        }
        // Our own wake segments inside the strap's asleep run: the fragmentation the strap did not see.
        for g in fine.iter().filter(|g| g.stage == SleepStage::Wake && g.start >= a0 && g.end <= a1) {
            let mins = (g.end - g.start) as f64 / 60.0;
            runs.push(mins);
            run_sec_total += g.end - g.start;
            short_runs += i64::from(mins <= 2.5);
        }
    }
    println!();
    header();
    row(&head, "head: span open -> first strap-asleep second");
    row(&interior, "interior: inside the strap's asleep run");
    row(&tail, "tail: after the last strap-asleep second");
    println!("   (head and tail sit outside the strap's asleep run by construction, so the reference reads");
    println!("    100% there and their kappa is meaningless — only our own column says anything.)");
    println!(
        "\n   our wake runs inside the strap's asleep run: {} runs, median {:.1} min, {} of them <= 2.5 min, {:.1} h total",
        runs.len(),
        median(&mut runs.clone()),
        short_runs,
        run_sec_total as f64 / 3600.0
    );
    println!("   refine_wake passes a wake run under 5 min through untouched, so it cannot see these.");
    section_band_codes(spans, p);
}

// ── the reference's own three non-asleep codes ─────────────────────────────────────────────────────

/// The band is FOUR codes (0 wake / 1 still / 2 asleep / 3 up) and every harness folds it to two. The two
/// foldings in this project disagree: `flow_az` counts `still` as awake, `ours`' truth.csv drops it as
/// unlabelled. So part of any "over-call" is which fold the reference used.
fn section_band_codes(spans: &[Span], p: &Params) {
    println!("\n   the reference's own codes inside the same spans, and what we call each");
    println!("   {:<10} {:>12} {:>9} {:>16}", "band code", "seconds", "share", "we say wake");
    let mut per_code = [[0i64; 2]; 4];
    for s in spans {
        let fine = refine_wake(&stage_v2_prepared(&s.prep, p), &s.accel, &s.steps);
        for &(ts, st) in &s.band {
            let Some(g) = stage_at(&fine, ts) else { continue };
            if !(0..4).contains(&st) {
                continue;
            }
            per_code[st as usize][0] += 1;
            per_code[st as usize][1] += i64::from(g == SleepStage::Wake);
        }
    }
    let tot: i64 = per_code.iter().map(|r| r[0]).sum();
    for (i, name) in ["0 wake", "1 still", "2 asleep", "3 up"].iter().enumerate() {
        println!(
            "   {name:<10} {:>12} {:>8.1}% {:>15.1}%",
            per_code[i][0],
            100.0 * per_code[i][0] as f64 / tot.max(1) as f64,
            100.0 * per_code[i][1] as f64 / per_code[i][0].max(1) as f64
        );
    }
    let folded_in = 100.0 * (per_code[0][0] + per_code[1][0] + per_code[3][0]) as f64 / tot.max(1) as f64;
    let folded_out = 100.0 * (per_code[0][0] + per_code[3][0]) as f64 / tot.max(1) as f64;
    println!(
        "   reference wake% is {folded_in:.1} with `still` folded in (flow_az) and {folded_out:.1} with it dropped"
    );
}

// ── 3  the emission sweep, on `ours` ──────────────────────────────────────────────────────────────

struct Night {
    w0: i64,
    n: usize,
    input: SleepInput,
    steps: Vec<StepSample>,
    truth: BTreeMap<usize, i32>,
}

fn load_ours(d: &Path) -> Option<Night> {
    let (w0, w1, n) = read_meta(d)?;
    Some(Night {
        w0,
        n,
        input: SleepInput { start: w0, end: w1, hr: read_hr(d), rr: read_rr(d), accel: read_accel(d) },
        steps: read_steps(d),
        truth: read_truth(d),
    })
}

/// One label per truth epoch, read at its midpoint. `jerk_move_mult` is held fixed across every config
/// below, so one `Prepared` per night serves the whole sweep.
fn ours_labels(n: &Night, prep: &Prepared, r: &Recipe) -> Vec<usize> {
    let segs = drop_wake_blips(&stage_v2_prepared(prep, &r.p), r.blip_sec);
    (0..n.n)
        .map(|k| {
            let mid = n.w0 + k as i64 * EPOCH_SEC + EPOCH_SEC / 2;
            stage_idx(stage_at(&segs, mid).unwrap_or_else(|| segs.last().unwrap().stage))
        })
        .collect()
}

fn ours_score(ns: &[Night], preps: &[Prepared], r: &Recipe) -> (TwoClass, [f64; 4]) {
    let mut tc = TwoClass::default();
    let mut frac = [0i64; 4];
    for (n, pr) in ns.iter().zip(preps) {
        let lab = ours_labels(n, pr, r);
        for &l in &lab {
            frac[l] += 1;
        }
        for (&k, &t) in &n.truth {
            if k < n.n {
                tc.add(lab[k] == WAKE, t == 0);
            }
        }
    }
    let all: i64 = frac.iter().sum();
    let pct = |v: i64| 100.0 * v as f64 / all.max(1) as f64;
    (tc, [pct(frac[0]), pct(frac[1]), pct(frac[2]), pct(frac[3])])
}

/// Every config the wake sweep scores, with the shipped recipe first so a delta is read down a column.
fn wake_candidates(shipped: &Params) -> Vec<Recipe> {
    let mut out = vec![recipe("SHIPPED", *shipped, 0)];
    for br in [0.26f64, 0.18, 0.10] {
        let mut p = *shipped;
        p.base_rate[3] = br;
        out.push(family_recipe(&format!("awake base rate {br:.2} (was 0.34)"), PROPENSITY, p, 0));
    }
    for b in [3.0f64, 2.0, 1.0, 0.0] {
        let mut p = *shipped;
        p.motion_gate_boost = b;
        out.push(family_recipe(&format!("motion gate boost {b:.1} (was 4.0)"), MOTION_GATE, p, 0));
    }
    for m in [0.6f64, 0.4] {
        let mut p = *shipped;
        p.awake_motion = m;
        out.push(family_recipe(&format!("awake motion {m:.1} (was 1.0)"), PROPENSITY, p, 0));
    }
    for dz in [0.8f64, 1.2] {
        let mut p = *shipped;
        p.awake_deadzone = dz;
        out.push(family_recipe(&format!("awake dead zone {dz:.1} (was 0.30)"), PROPENSITY, p, 0));
    }
    for stay in [0.95f64, 0.80, 0.60] {
        let mut p = *shipped;
        p.transition[3] = [0.0, 0.0, 1.0 - stay, stay];
        out.push(family_recipe(&format!("awake self-transition {stay:.2} (was 0.90)"), PROPENSITY, p, 0));
    }
    // Entry into wake from light, which is what a short interior excursion has to pass through.
    for enter in [0.02f64, 0.01, 0.005] {
        let mut p = *shipped;
        p.transition[2] = [0.08, 0.08, 0.84 - enter, enter];
        out.push(family_recipe(&format!("light->awake {enter:.3} (was 0.040)"), PROPENSITY, p, 0));
    }
    // The targeted post-pass: drop only the short interior runs, leave every long wake stretch alone.
    for mins in [1.0f64, 2.0, 3.0, 5.0, 10.0] {
        out.push(family_recipe(
            &format!("drop wake runs under {mins:.0} min"),
            BLIP_FILTER,
            *shipped,
            (mins * 60.0) as i64,
        ));
    }
    out
}

fn section_sweep(ns: &[Night], preps: &[Prepared], shipped: &Params) -> Vec<Recipe> {
    println!("\n3  is the over-call reducible: every term feeding the AWAKE emission, swept");
    println!("   Scored on the 22 band-labelled `ours` nights. kappa2 is the number to read: raw accuracy");
    println!("   rises whenever a config says wake less often, because the reference calls ~94% asleep.");
    println!("   Stage fractions are over all 92 nights, so they answer the WHOOP-export question too.");
    println!();
    header();
    let mut best: Vec<(f64, String, &'static str, Params, i64)> = Vec::new();
    for r in wake_candidates(shipped) {
        let (tc, _) = ours_score(ns, preps, &r);
        row(&tc, &r.label);
        best.push((tc.kappa(), r.label, r.family, r.p, r.blip_sec));
    }
    println!("\n   stage fractions over all 92 nights (wake / light / deep / REM), WHOOP export = 6.1 / 49.0 / 20.1 / 24.8");
    let mut rows: Vec<Recipe> = vec![recipe("SHIPPED", *shipped, 0)];
    best.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    for fam in [PROPENSITY, MOTION_GATE, BLIP_FILTER] {
        if let Some((_, label, _, p, blip)) = best.iter().find(|c| c.2 == fam) {
            rows.push(family_recipe(&format!("{fam} best: {label}"), fam, *p, *blip));
        }
    }
    for r in &rows {
        let (tc, frac) = ours_score(ns, preps, r);
        println!(
            "   {:<48} {:>6.1}{:>7.1}{:>7.1}{:>7.1}   kappa2 {:.3}",
            r.label, frac[0], frac[1], frac[2], frac[3], tc.kappa()
        );
    }
    rows
}

// ── 5  the PSG cost of every candidate ────────────────────────────────────────────────────────────

struct PsgNight {
    w0: i64,
    n: usize,
    input: SleepInput,
    truth: BTreeMap<usize, i32>,
}

fn load_psg(d: &Path) -> Option<PsgNight> {
    let (w0, w1, n) = read_meta(d)?;
    let truth = read_truth(d);
    if truth.is_empty() {
        return None;
    }
    Some(PsgNight {
        w0,
        n,
        input: SleepInput { start: w0, end: w1, hr: read_hr(d), rr: read_rr(d), accel: read_accel(d) },
        truth,
    })
}

/// 4-class kappa plus the wake fraction, per cohort, so no number is quoted without its population.
fn psg_row(nights: &[PsgNight], preps: &[Prepared], r: &Recipe) -> (f64, f64, f64) {
    let mut cm = [[0i64; 4]; 4];
    let (mut wake, mut all) = (0i64, 0i64);
    for (n, pr) in nights.iter().zip(preps) {
        let segs = drop_wake_blips(&stage_v2_prepared(pr, &r.p), r.blip_sec);
        for (&k, &t) in &n.truth {
            if k >= n.n || !(0..4).contains(&t) {
                continue;
            }
            let mid = n.w0 + k as i64 * EPOCH_SEC + EPOCH_SEC / 2;
            let l = stage_idx(stage_at(&segs, mid).unwrap_or_else(|| segs.last().unwrap().stage));
            cm[t as usize][l] += 1;
            all += 1;
            wake += i64::from(l == WAKE);
        }
    }
    let truth_wake: i64 = nights.iter().map(|n| n.truth.values().filter(|&&v| v == 0).count() as i64).sum();
    (kappa4(&cm), 100.0 * wake as f64 / all.max(1) as f64, 100.0 * truth_wake as f64 / all.max(1) as f64)
}

fn section_psg(cands: &[Recipe], shipped: &Params, ours: &[(String, TwoClass)]) {
    println!("\n5  what each candidate costs on PSG truth — 4-class kappa, per cohort");
    println!("   SELECTION IS PSG-ONLY. `ours` is the band's SLEEP-PERIOD marker, not a hypnogram, and ranking");
    println!("   on it rewards under-calling wake — so it appears below as context (recall, wake share) only.");
    let sets: Vec<(&str, Vec<PsgNight>, Vec<Prepared>)> = PSG
        .iter()
        .filter_map(|ds| {
            let nights: Vec<PsgNight> = dirs_of(ds).iter().filter_map(|d| load_psg(d)).collect();
            if nights.is_empty() {
                return None;
            }
            let preps = nights.iter().map(|n| prepare_v2(&n.input, shipped)).collect();
            Some((*ds, nights, preps))
        })
        .collect();
    print!("\n   {:<48}", "config");
    for (ds, n, _) in &sets {
        print!("{:>24}", format!("{ds} (n={})", n.len()));
    }
    println!();
    print!("   {:<48}", "");
    for _ in &sets {
        print!("{:>10}{:>8}{:>6}", "kappa4", "ours w%", "true");
    }
    println!();
    let mut ledger: Vec<(String, TwoClass, Vec<f64>)> = Vec::new();
    for r in cands {
        print!("   {:<48}", r.label);
        let mut ks = Vec::new();
        for (_, nights, preps) in &sets {
            let (k, w, tw) = psg_row(nights, preps, r);
            print!("{k:>10.3}{w:>8.1}{tw:>6.1}");
            ks.push(k);
        }
        println!();
        let tc = ours.iter().find(|(l, _)| *l == r.label).map(|(_, t)| *t).unwrap_or_default();
        ledger.push((r.label.clone(), tc, ks));
    }

    println!("\n   the ledger: every candidate as a DELTA against SHIPPED");
    println!("   ACCEPT RULE: no PSG column negative. The band columns do not veto — a recall gain rides on");
    println!("   the wake share beside it, so only a gain at a flat share is real.");
    let base = ledger.first().cloned();
    let Some((_, base_tc, base_ks)) = base else { return };
    print!("   {:<48}", "candidate");
    for (ds, _, _) in &sets {
        print!("{:>18}", format!("{ds} d-k4"));
    }
    println!("{:>16}{:>14}", "band d-recall", "band d-wake%");
    for (label, tc, ks) in ledger.iter().skip(1) {
        print!("   {label:<48}");
        for (i, k) in ks.iter().enumerate() {
            print!("{:>+18.3}", k - base_ks[i]);
        }
        println!("{:>+16.1}{:>+14.1}", tc.recall() - base_tc.recall(), tc.pred_pct() - base_tc.pred_pct());
    }
}

// ── 4  the cycle prior: the onset-anchored guard and the onset-anchored clock ──────────────────────

/// F1's guard and F3's clock, each alone and together, against both references. The guard is the same
/// magnitude graded over minutes past onset instead of a step at a fraction of the window; the clock
/// re-anchors the whole time-of-night input, which is the wider of the two.
fn cycle_candidates(shipped: &Params) -> Vec<Recipe> {
    let guard = Params { cycle_rem_early_penalty: 4.0, cycle_rem_onset_minutes: 60.0, ..*shipped };
    vec![
        recipe("SHIPPED", *shipped, 0),
        recipe("F1 onset-anchored guard K=4 M0=60", guard, 0),
        recipe("F3 onset-anchored clock", Params { cycle_clock_from_onset: true, ..*shipped }, 0),
        recipe("F1 + F3", Params { cycle_clock_from_onset: true, ..guard }, 0),
        recipe("REM ramp cap 0.7 (on F1)", Params { cycle_rem_ramp_cap: 0.7, ..guard }, 0),
    ]
}

fn section_cycle(ns: &[Night], preps: &[Prepared], shipped: &Params) -> Vec<Recipe> {
    println!("\n4  the cycle prior — F1's onset-anchored guard and F3's onset-anchored clock");
    println!("   Both replace a WINDOW-relative reading of time-of-night with an ONSET-relative one, which is");
    println!("   why E handed them here. Scored on the band first, then on PSG in section 4's table.");
    println!();
    header();
    let cands = cycle_candidates(shipped);
    for r in &cands {
        let (tc, _) = ours_score(ns, preps, r);
        row(&tc, &r.label);
    }
    println!("\n   stage fractions over all 92 nights (wake / light / deep / REM), WHOOP export = 6.1 / 49.0 / 20.1 / 24.8");
    for r in &cands {
        let (_, frac) = ours_score(ns, preps, r);
        println!("   {:<48} {:>6.1}{:>7.1}{:>7.1}{:>7.1}", r.label, frac[0], frac[1], frac[2], frac[3]);
    }
    section_onset_quality(ns, preps, shipped);
    cands
}

/// What the onset anchor actually lands on. The library takes it from a first staging pass with the guard
/// off, then reads time-of-night from it — so if that pass calls the in-bed-awake lead-in asleep, the
/// anchor sits at the window start and re-anchoring cannot move anything.
fn section_onset_quality(ns: &[Night], preps: &[Prepared], shipped: &Params) {
    // The library's first pass is the shipped recipe with the early-REM guard contributing nothing.
    let probe = recipe("probe", Params { cycle_rem_early_penalty: 0.0, ..*shipped }, 0);
    let (mut found, mut band_onset, mut delta) = (Vec::new(), Vec::new(), Vec::new());
    let mut none = 0usize;
    for (n, pr) in ns.iter().zip(preps) {
        let Some(&first_asleep) = n.truth.iter().find(|(_, &t)| t == 1).map(|(k, _)| k) else { continue };
        let lab = ours_labels(n, pr, &probe);
        let mut run = 0usize;
        let mut onset = None;
        for (i, &l) in lab.iter().enumerate() {
            run = if l == WAKE { 0 } else { run + 1 };
            if run == 10 {
                onset = Some(i + 1 - 10);
                break;
            }
        }
        match onset {
            Some(o) => {
                found.push(o as f64 * EPOCH_MIN_F);
                band_onset.push(first_asleep as f64 * EPOCH_MIN_F);
                delta.push((o as f64 - first_asleep as f64) * EPOCH_MIN_F);
            }
            None => none += 1,
        }
    }
    println!(
        "\n   the anchor itself, over {} band-labelled nights ({} with no sustained run at all)",
        found.len(),
        none
    );
    println!(
        "   median minutes from window open to: our probe onset {:.1}, the strap's first asleep second {:.1}",
        median(&mut found.clone()),
        median(&mut band_onset.clone())
    );
    println!(
        "   median signed gap (ours - strap) {:+.1} min; our probe lands in the first 5 min on {} of {}",
        median(&mut delta.clone()),
        found.iter().filter(|m| **m < 5.0).count(),
        found.len()
    );
}

fn main() {
    let shipped = Params::SHIPPED;
    let spans = continuous_spans(&shipped);
    section_refinement(&spans, &shipped);
    section_where(&spans, &shipped);

    let ns: Vec<Night> = dirs_of("ours").iter().filter_map(|d| load_ours(d)).collect();
    let preps: Vec<Prepared> = ns.iter().map(|n| prepare_v2(&n.input, &shipped)).collect();
    // A non-empty steps file is not a dense one, so the count that matters is the SHIPPED gate's answer.
    let with_steps = ns.iter().filter(|n| !n.steps.is_empty()).count();
    let dense = ns
        .iter()
        .filter(|n| motion_dense(n.input.start, n.input.end, &n.input.accel, &n.steps))
        .count();
    println!(
        "\n   (`ours`: {} nights, {} with a non-empty steps.csv, {} DENSE enough for the refinement, \
         {} band-labelled)",
        ns.len(),
        with_steps,
        dense,
        ns.iter().filter(|n| !n.truth.is_empty()).count()
    );
    println!("   Section 3's sweep is stage_v2 ONLY — it compares emission recipes against each other, and");
    println!("   `ours` gravity is held forward across gaps, so its posture check cannot be trusted here.");
    let mut cands = section_sweep(&ns, &preps, &shipped);
    let cycle = section_cycle(&ns, &preps, &shipped);
    cands.extend(cycle.into_iter().filter(|r| r.label != "SHIPPED"));
    // The band score of every finalist, so the ledger can put it beside the PSG cost of the same row.
    let ours: Vec<(String, TwoClass)> =
        cands.iter().map(|r| (r.label.clone(), ours_score(&ns, &preps, r).0)).collect();
    section_psg(&cands, &shipped, &ours);
}
