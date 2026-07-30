//! Step Z — does a sleep error propagate, amplify, or get absorbed on its way to Recovery?
//!
//!   cargo run --release -p physio-algo --example sleep_to_charge
//!
//! 1  the chain, named: which sleep outputs reach Rest and Charge, and at what weight
//! 2  the unperturbed population, so every delta below has a level beside it
//! 3  the transfer: a known error injected into the hypnogram, measured at Rest and at Charge
//! 4  the two channels separated, because the smaller-weighted one is not the bigger one
//! 5  the chain's OWN measured errors, converted into points of Charge
//!
//! A perturbation relabels a stated share of the night's epochs and nothing else. Everything a single
//! night does not set — the personal sleep need, the driver baselines, the consistency term — is computed
//! once from the unperturbed corpus and held fixed, so what moves is the sleep error and only that.
//! Resting HR is a property of the span, not of the staging, so its driver is inert here by construction.

mod common;

use common::{dirs_of, night_id, read_accel, read_hr, read_meta, read_rr, read_steps, stage_idx};

use physio_algo::hrv::HrvReadiness;
use physio_algo::recovery::{recovery, DriverBaseline, RecoveryInput, W_HRV, W_SLEEP};
use physio_algo::rest::{personal_sleep_need_hours, rest};
use physio_algo::resting_hr::session_resting_hr;
use physio_algo::sleep::{
    epoch_starts_v2, motion_density, params::Params, prepare_v2, refine_wake, stage_v2_prepared, AccelSample,
    SleepInput, SleepStage, StageSegment, StepSample, MIN_DENSE_FRACTION,
};

const EPOCH_SEC: i64 = 30;
const STAGES: [SleepStage; 4] = [SleepStage::Wake, SleepStage::Light, SleepStage::Deep, SleepStage::Rem];

/// One staged night, reduced to what the downstream scores read.
struct Scored {
    asleep_s: f64,
    efficiency: f64,
    deep_s: f64,
    rem_s: f64,
    hrv: Option<f64>,
    rhr: Option<i32>,
}

struct Night {
    owner: String,
    input: SleepInput,
    accel: Vec<AccelSample>,
    steps: Vec<StepSample>,
    /// The same heart rate in the shape `resting_hr` takes. Sleep and the score modules carry their own
    /// sample types, so the conversion happens once here rather than per perturbation.
    rhr_hr: Vec<physio_algo::HrSample>,
    refined: bool,
}

fn load_ours() -> Vec<Night> {
    let mut out = Vec::new();
    for d in dirs_of("ours") {
        let Some((w0, w1, _)) = read_meta(&d) else { continue };
        let (owner, _) = night_id(&d);
        let (accel, hr, rr) = (read_accel(&d), read_hr(&d), read_rr(&d));
        if hr.len() < 120 || accel.len() < 120 || rr.is_empty() {
            continue;
        }
        let steps = read_steps(&d);
        let (g, s) = motion_density(w0, w1, &accel, &steps);
        let rhr_hr = hr.iter().map(|h| physio_algo::HrSample { ts: h.ts, bpm: h.bpm as i32 }).collect();
        out.push(Night {
            owner,
            refined: g >= MIN_DENSE_FRACTION && s >= MIN_DENSE_FRACTION,
            steps,
            rhr_hr,
            input: SleepInput { start: w0, end: w1, hr, rr, accel: accel.clone() },
            accel,
        });
    }
    out
}

/// Epoch labels rebuilt into contiguous segments over `[starts[0], end)`.
fn segments_of(labels: &[usize], starts: &[i64], end: i64) -> Vec<StageSegment> {
    let mut out: Vec<StageSegment> = Vec::new();
    for (i, &l) in labels.iter().enumerate() {
        let (a, b) = (starts[i], starts.get(i + 1).copied().unwrap_or(end));
        match out.last_mut() {
            Some(last) if last.stage == STAGES[l] => last.end = b,
            _ => out.push(StageSegment { start: a, end: b, stage: STAGES[l] }),
        }
    }
    out
}

/// Relabel `pp` percent of the night's epochs between `from` and `to`, spread evenly over the eligible
/// ones rather than taken off the front, so the perturbation does not become one long block.
fn perturb(labels: &[usize], pp: f64, from: usize, to: usize) -> Vec<usize> {
    let mut out = labels.to_vec();
    let want = (pp.abs() / 100.0 * labels.len() as f64).round() as usize;
    let eligible: Vec<usize> = labels.iter().enumerate().filter(|(_, &l)| l == from).map(|(i, _)| i).collect();
    if want == 0 || eligible.is_empty() {
        return out;
    }
    let take = want.min(eligible.len());
    for k in 0..take {
        out[eligible[k * eligible.len() / take]] = to;
    }
    out
}

fn score(n: &Night, segs: &[StageSegment]) -> Scored {
    let secs = |want: SleepStage| -> f64 {
        segs.iter().filter(|s| s.stage == want).map(|s| (s.end - s.start) as f64).sum()
    };
    let (wake, light, deep, rem) =
        (secs(SleepStage::Wake), secs(SleepStage::Light), secs(SleepStage::Deep), secs(SleepStage::Rem));
    let in_bed = wake + light + deep + rem;
    let beats: Vec<(u32, u16)> =
        n.input.rr.iter().flat_map(|r| r.intervals.iter().map(move |&ms| (r.ts as u32, ms))).collect();
    let deep_spans: Vec<(u32, u32)> = segs
        .iter()
        .filter(|s| s.stage == SleepStage::Deep)
        .map(|s| (s.start as u32, s.end as u32))
        .collect();
    Scored {
        asleep_s: light + deep + rem,
        efficiency: if in_bed > 0.0 { (light + deep + rem) / in_bed } else { 0.0 },
        deep_s: deep,
        rem_s: rem,
        hrv: HrvReadiness::windowed_avg_hrv_deep(
            n.input.start as u32,
            n.input.end as u32,
            &beats,
            &deep_spans,
        ),
        rhr: session_resting_hr(n.input.start, n.input.end, &n.rhr_hr),
    }
}

/// Everything the wearer brings to a night rather than the night bringing it. Computed once from the
/// unperturbed corpus and held fixed, so a perturbation moves the night and never the person.
struct Person {
    need_h: f64,
    hrv: DriverBaseline,
    rhr: DriverBaseline,
}

fn baseline(values: &[f64]) -> DriverBaseline {
    let mean = values.iter().sum::<f64>() / values.len().max(1) as f64;
    let dev = values.iter().map(|v| (v - mean).abs()).sum::<f64>() / values.len().max(1) as f64;
    DriverBaseline { mean, spread: dev.max(1e-6) }
}

fn rest_of(s: &Scored, p: &Person) -> Option<f64> {
    rest(s.asleep_s, s.efficiency, s.deep_s, s.rem_s, Some(p.need_h), None)
}

/// Charge from one night's sleep. `hrv_from` and `rest_from` are separate so a channel can be frozen at
/// the unperturbed night's value while the other moves — which is what section 4 does.
fn charge_of(hrv_from: &Scored, rest_from: &Scored, p: &Person) -> Option<f64> {
    let (Some(hrv), Some(rhr)) = (hrv_from.hrv, rest_from.rhr) else { return None };
    recovery(&RecoveryInput {
        hrv,
        rhr: rhr as f64,
        hrv_baseline: Some(p.hrv),
        rhr_baseline: Some(p.rhr),
        sleep_perf: rest_of(rest_from, p).map(|r| r / 100.0),
        ..Default::default()
    })
}

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len().max(1) as f64
}

/// One perturbation applied to every night: the mean level of each output, and the mean signed change.
struct Transfer {
    n: usize,
    d_deep_pp: f64,
    d_wake_pp: f64,
    d_hrv: f64,
    d_rest: f64,
    d_charge: f64,
    d_charge_hrv_only: f64,
    d_charge_rest_only: f64,
}

fn run(nights: &[Night], base: &[(Scored, Vec<usize>, Vec<i64>)], p: &Person, pp: f64, from: usize, to: usize) -> Transfer {
    let (mut dd, mut dw, mut dh, mut dr, mut dc, mut dch, mut dcr) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for (n, (b, labels, starts)) in nights.iter().zip(base) {
        let moved = perturb(labels, pp, from, to);
        let s = score(n, &segments_of(&moved, starts, n.input.end));
        // Relabelling cannot change the window, so both stagings share one denominator.
        let span = in_bed_of(b).max(1.0);
        dd.push(100.0 * (s.deep_s - b.deep_s) / span);
        dw.push(100.0 * ((in_bed_of(&s) - s.asleep_s) - (in_bed_of(b) - b.asleep_s)) / span);
        if let (Some(x), Some(y)) = (s.hrv, b.hrv) {
            dh.push(x - y);
        }
        if let (Some(x), Some(y)) = (rest_of(&s, p), rest_of(b, p)) {
            dr.push(x - y);
        }
        if let (Some(x), Some(y)) = (charge_of(&s, &s, p), charge_of(b, b, p)) {
            dc.push(x - y);
        }
        if let (Some(x), Some(y)) = (charge_of(&s, b, p), charge_of(b, b, p)) {
            dch.push(x - y);
        }
        if let (Some(x), Some(y)) = (charge_of(b, &s, p), charge_of(b, b, p)) {
            dcr.push(x - y);
        }
    }
    Transfer {
        n: dc.len(),
        d_deep_pp: mean(&dd),
        d_wake_pp: mean(&dw),
        d_hrv: mean(&dh),
        d_rest: mean(&dr),
        d_charge: mean(&dc),
        d_charge_hrv_only: mean(&dch),
        d_charge_rest_only: mean(&dcr),
    }
}

fn in_bed_of(s: &Scored) -> f64 {
    if s.efficiency > 0.0 { s.asleep_s / s.efficiency } else { s.asleep_s }
}

fn main() {
    let p_shipped = Params::SHIPPED;
    let nights = load_ours();
    if nights.is_empty() {
        println!("no `ours` nights under {}", common::fixtures_root().display());
        return;
    }

    // The shipped path per night: stage, refine where the gate allows, then reduce to the score inputs.
    let mut base: Vec<(Scored, Vec<usize>, Vec<i64>)> = Vec::new();
    let mut refined = 0usize;
    for n in &nights {
        let prep = prepare_v2(&n.input, &p_shipped);
        let starts = epoch_starts_v2(&prep);
        let segs = stage_v2_prepared(&prep, &p_shipped);
        let segs = if n.refined {
            refined += 1;
            refine_wake(&segs, &n.accel, &n.steps)
        } else {
            segs
        };
        let labels: Vec<usize> = starts
            .iter()
            .map(|s| {
                let mid = s + EPOCH_SEC / 2;
                stage_idx(segs.iter().find(|g| g.start <= mid && mid < g.end).or(segs.last()).unwrap().stage)
            })
            .collect();
        base.push((score(n, &segments_of(&labels, &starts, n.input.end)), labels, starts));
    }

    println!("=== 1. the chain, and it is two channels rather than one ===");
    println!("   sleep -> asleep hours, efficiency, deep and REM seconds -> rest() -> Rest 0-100");
    println!("   Rest/100 is Charge's `sleep_perf` driver at weight {W_SLEEP:.2}.");
    println!("   sleep ALSO picks the deep spans the nightly HRV is averaged over, and HRV is Charge's");
    println!("   dominant driver at weight {W_HRV:.2} — {:.1}x the sleep term. Resting HR is a property of", W_HRV / W_SLEEP);
    println!("   the span rather than the staging, so its driver cannot move here.");

    // The person, from the unperturbed corpus. One wearer dominates `ours`, so this is pooled and said so.
    let hrvs: Vec<f64> = base.iter().filter_map(|(b, _, _)| b.hrv).collect();
    let rhrs: Vec<f64> = base.iter().filter_map(|(b, _, _)| b.rhr.map(f64::from)).collect();
    let asleep_h: Vec<f64> = base.iter().map(|(b, _, _)| b.asleep_s / 3600.0).collect();
    let person = Person {
        need_h: personal_sleep_need_hours(&asleep_h),
        hrv: baseline(&hrvs),
        rhr: baseline(&rhrs),
    };
    let mut owners: Vec<&str> = nights.iter().map(|n| n.owner.as_str()).collect();
    owners.sort_unstable();
    owners.dedup();

    println!("\n=== 2. the unperturbed population, {} nights, {} refined by the density gate ===", nights.len(), refined);
    println!("   wearers: {}", owners.join(", "));
    let rests: Vec<f64> = base.iter().filter_map(|(b, _, _)| rest_of(b, &person)).collect();
    let charges: Vec<f64> = base.iter().filter_map(|(b, _, _)| charge_of(b, b, &person)).collect();
    println!(
        "   asleep {:.2} h · efficiency {:.1}% · deep {:.1}% · REM {:.1}% · deep-window HRV {:.1} ms",
        mean(&asleep_h),
        100.0 * mean(&base.iter().map(|(b, _, _)| b.efficiency).collect::<Vec<_>>()),
        100.0 * mean(&base.iter().map(|(b, _, _)| b.deep_s / in_bed_of(b)).collect::<Vec<_>>()),
        100.0 * mean(&base.iter().map(|(b, _, _)| b.rem_s / in_bed_of(b)).collect::<Vec<_>>()),
        mean(&hrvs)
    );
    println!(
        "   personal sleep need {:.2} h · Rest {:.1} (n={}) · Charge {:.1} (n={})",
        person.need_h,
        mean(&rests),
        rests.len(),
        mean(&charges),
        charges.len()
    );

    println!("\n=== 3. the transfer: a known hypnogram error, measured at Rest and at Charge ===");
    println!("   `injected` is what the relabelling asks for; `deep` and `wake` are what the night's own");
    println!("   stage shares actually moved, which is smaller when a night runs out of eligible epochs.");
    println!(
        "   {:<28} {:>9} {:>9} {:>10} {:>10} {:>11} {:>8}",
        "perturbation", "deep pp", "wake pp", "HRV ms", "d-Rest", "d-Charge", "gain"
    );
    let cases: Vec<(&str, f64, usize, usize)> = vec![
        ("light -> wake  1 pp", 1.0, 1, 0),
        ("light -> wake  2 pp", 2.0, 1, 0),
        ("light -> wake  5 pp", 5.0, 1, 0),
        ("light -> wake 10 pp", 10.0, 1, 0),
        ("wake  -> light 5 pp", 5.0, 0, 1),
        ("light -> deep  1 pp", 1.0, 1, 2),
        ("light -> deep  2 pp", 2.0, 1, 2),
        ("light -> deep  5 pp", 5.0, 1, 2),
        ("light -> deep 10 pp", 10.0, 1, 2),
        ("deep  -> light 5 pp", 5.0, 2, 1),
    ];
    let mut rows: Vec<(String, Transfer)> = Vec::new();
    for (label, pp, from, to) in &cases {
        let t = run(&nights, &base, &person, *pp, *from, *to);
        let moved = t.d_deep_pp.abs().max(t.d_wake_pp.abs());
        println!(
            "   {label:<28} {:>+9.2} {:>+9.2} {:>+10.2} {:>+10.2} {:>+11.2} {:>8.2}",
            t.d_deep_pp,
            t.d_wake_pp,
            t.d_hrv,
            t.d_rest,
            t.d_charge,
            if moved > 0.0 { t.d_charge.abs() / moved } else { f64::NAN }
        );
        rows.push((label.to_string(), t));
    }
    println!("\n   `gain` is |d-Charge| in points per percentage point of stage error actually moved.");
    println!("   Above 1.0 the chain amplifies; below 1.0 it absorbs.");

    println!("\n=== 4. the same deltas with one channel frozen at the unperturbed night ===");
    println!(
        "   {:<28} {:>12} {:>14} {:>14} {:>10}",
        "perturbation", "d-Charge", "HRV channel", "Rest channel", "sum"
    );
    for (label, t) in &rows {
        println!(
            "   {label:<28} {:>+12.2} {:>+14.2} {:>+14.2} {:>+10.2}",
            t.d_charge, t.d_charge_hrv_only, t.d_charge_rest_only, t.d_charge_hrv_only + t.d_charge_rest_only
        );
    }
    println!("\n   The two channels are not additive to the last decimal — Charge is a logistic — but the");
    println!("   split says which one carries the error, and it is not the one named `sleep`.");

    println!("\n=== 5. the chain's own measured errors, in points of Charge ===");
    println!("   Each row takes an error this project MEASURED and asks what it is worth downstream.");
    println!("   {:<44} {:>11} {:>11}", "measured error", "d-Rest", "d-Charge");
    let owned: Vec<(&str, f64, usize, usize)> = vec![
        ("I: deep under-called 6.6 pp vs the export", 6.6, 1, 2),
        ("H: wake over-called 0.9 pp, shipped", 0.9, 1, 0),
        ("H: wake over-called 1.8 pp, unrefined", 1.8, 1, 0),
        ("E: window sensitivity, 5.2% of labels", 5.2, 1, 0),
    ];
    for (label, pp, from, to) in &owned {
        let t = run(&nights, &base, &person, *pp, *from, *to);
        println!("   {label:<44} {:>+11.2} {:>+11.2}", t.d_rest, t.d_charge);
    }
    println!("\n   n = {} nights on every row.", rows[0].1.n);
}
