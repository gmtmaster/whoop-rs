//! Step G — the Viterbi decoder, isolated from the emissions feeding it.
//!
//!   cargo run --release -p physio-algo --example viterbi_decode
//!
//! 1  isolation: the decoder against exhaustive enumeration, and against two independent oracles at
//!    corpus scale, on both real reference families
//! 2  decoder error vs MODEL error: does the model score the reference's own path above the path we
//!    return, and if not, WHICH half of the model prefers ours — the per-epoch evidence or the transitions
//! 3  the two axes sized against each other: emissions pinned and the matrix swept, matrix pinned and the
//!    emissions ablated. Whichever spread is larger owns the error
//! 4  the transition matrix and the base rates against something real: PSG hypnograms at 30 s, the band's
//!    own two-class call, and the PSG stage fractions
//! 5  three bounded rescues on the matrix, bounded at three, each scored on all four references
//!
//! Sections 1-3 read `continuous` (real unix seconds, the band at 1 Hz, `steps.csv`, and the only set
//! whose gravity is not held forward) and the three PSG cohorts. **The PSG sets carry no `steps.csv`, so
//! `refine_wake` declines on them by its own density gate** — section 1 verifies that rather than assuming
//! it, and every PSG number below is therefore the unrefined staging, stated as such.
//!
//! Every refinement here runs through `common::RefineCensus`, so a cohort whose step stream the density
//! gate declines is reported as declined rather than pooled into a refined number.

mod common;

use common::{kappa4, RefineCensus};

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use physio_algo::sleep::{
    decode_v2, detect_sessions, emissions_v2, epoch_starts_v2, params::Params, prepare_v2, segments_v2,
    AccelSample, HrSample, Prepared, RrRun, SleepInput, SleepStage, StageSegment, StepSample,
};

const EPOCH_SEC: i64 = 30;
const BAND_ASLEEP: i32 = 2;
/// Emission/transition columns, mirroring `STAGE_ORDER`.
const DEEP: usize = 0;
const REM: usize = 1;
const LIGHT: usize = 2;
const AWAKE: usize = 3;
/// The floor a zero transition is clamped to before logging — the decoder's own, repeated so the
/// independently-written scorer below shares no code with the thing it scores.
const T_FLOOR: f64 = 1e-9;
/// Emission penalty that forbids a column without making the log-score infinite.
const FORBID: f64 = -1e9;
const PSG: [&str; 3] = ["dreamt", "aauwss", "sleep-accel"];
/// Emission column of a PSG truth code, indexed 0 wake / 1 light / 2 deep / 3 rem.
const TRUTH_TO_COL: [usize; 4] = [AWAKE, LIGHT, DEEP, REM];

// ── fixtures ──────────────────────────────────────────────────────────────────────────────────────

fn root(set: &str) -> PathBuf {
    common::fixtures_root().join(set)
}

fn read_csv(path: &Path) -> Vec<Vec<f64>> {
    fs::read_to_string(path)
        .map(|t| {
            t.lines()
                .filter(|l| !l.trim().is_empty())
                .map(|l| l.split(',').map(|c| c.trim().parse::<f64>().unwrap()).collect())
                .collect()
        })
        .unwrap_or_default()
}

fn dirs_of(set: &str) -> Vec<PathBuf> {
    let mut d: Vec<PathBuf> = fs::read_dir(root(set))
        .map(|rd| rd.filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.is_dir()).collect())
        .unwrap_or_default();
    d.sort();
    d
}

fn read_rr(path: &Path) -> Vec<RrRun> {
    let mut rr: Vec<RrRun> = Vec::new();
    for row in read_csv(path) {
        let (ts, ms) = (row[0] as i64, row[1] as u16);
        match rr.last_mut() {
            Some(l) if l.ts == ts => l.intervals.push(ms),
            _ => rr.push(RrRun { ts, intervals: vec![ms] }),
        }
    }
    rr
}

fn read_steps(path: &Path) -> Vec<StepSample> {
    read_csv(path)
        .iter()
        .map(|r| StepSample {
            ts: r[0] as i64,
            counter: r[1] as u16,
            activity_class: (r[2] >= 0.0).then(|| r[2] as u8),
        })
        .collect()
}

/// One detected span off a continuous block: the prepared features, the streams the refinement reads, and
/// the band per second and per epoch.
struct Span {
    prep: Prepared,
    accel: Vec<AccelSample>,
    steps: Vec<StepSample>,
    band: Vec<(i64, i32)>,
    /// Per prepared epoch: `Some(true)` = the band called this epoch non-asleep, `None` = no band data.
    band_epoch: Vec<Option<bool>>,
}

fn continuous_spans(p: &Params) -> Vec<Span> {
    let mut out = Vec::new();
    for d in dirs_of("continuous") {
        let band: Vec<(i64, i32)> =
            read_csv(&d.join("band.csv")).iter().map(|r| (r[0] as i64, r[1] as i32)).collect();
        if band.is_empty() {
            continue;
        }
        let accel: Vec<AccelSample> = read_csv(&d.join("gravity.csv"))
            .iter()
            .map(|r| AccelSample { ts: r[0] as i64, x: r[1], y: r[2], z: r[3] })
            .collect();
        if accel.len() < 120 {
            continue;
        }
        let hr: Vec<HrSample> =
            read_csv(&d.join("hr.csv")).iter().map(|r| HrSample { ts: r[0] as i64, bpm: r[1] as u16 }).collect();
        let rr = read_rr(&d.join("rr.csv"));
        let steps = read_steps(&d.join("steps.csv"));
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
            let span_band: Vec<(i64, i32)> =
                band.iter().filter(|(t, _)| *t >= s.start && *t < s.end).cloned().collect();
            let prep = prepare_v2(&input, p);
            let band_epoch = epoch_band_call(&epoch_starts_v2(&prep), &span_band);
            out.push(Span {
                accel: input.accel.clone(),
                steps: steps.iter().filter(|t| t.ts >= s.start && t.ts < s.end).cloned().collect(),
                band: span_band,
                band_epoch,
                prep,
            });
        }
    }
    out
}

/// The band's majority call over each prepared epoch. `still` and `up` fold to non-asleep, the same fold
/// `flow_az` and `emit_wake` use, so a number here is comparable with theirs.
fn epoch_band_call(starts: &[i64], band: &[(i64, i32)]) -> Vec<Option<bool>> {
    let mut by: BTreeMap<i64, (i64, i64)> = BTreeMap::new();
    for &(ts, st) in band {
        let e = by.entry(ts - ts.rem_euclid(EPOCH_SEC)).or_insert((0, 0));
        e.0 += 1;
        e.1 += i64::from(st != BAND_ASLEEP);
    }
    starts
        .iter()
        .map(|s| by.get(&(s - s.rem_euclid(EPOCH_SEC))).map(|&(n, wake)| 2 * wake > n))
        .collect()
}

struct PsgNight {
    name: String,
    input: SleepInput,
    /// PSG code per epoch start, aligned by time rather than by index.
    truth: BTreeMap<i64, usize>,
}

fn load_psg(set: &str) -> Vec<PsgNight> {
    let mut out = Vec::new();
    for d in dirs_of(set) {
        let Ok(meta) = fs::read_to_string(d.join("meta.txt")) else { continue };
        let m: Vec<i64> = meta.split_whitespace().filter_map(|x| x.parse().ok()).collect();
        if m.len() < 4 {
            continue;
        }
        let truth: BTreeMap<i64, usize> = read_csv(&d.join("truth.csv"))
            .iter()
            .filter(|r| (0.0..4.0).contains(&r[1]))
            .map(|r| (m[1] + r[0] as i64 * EPOCH_SEC, r[1] as usize))
            .collect();
        if truth.is_empty() {
            continue;
        }
        let accel: Vec<AccelSample> = read_csv(&d.join("gravity.csv"))
            .iter()
            .map(|r| AccelSample { ts: r[0] as i64, x: r[1], y: r[2], z: r[3] })
            .collect();
        let hr: Vec<HrSample> =
            read_csv(&d.join("hr.csv")).iter().map(|r| HrSample { ts: r[0] as i64, bpm: r[1] as u16 }).collect();
        out.push(PsgNight {
            name: d.file_name().unwrap().to_string_lossy().to_string(),
            input: SleepInput { start: m[1], end: m[2], hr, rr: read_rr(&d.join("rr.csv")), accel },
            truth,
        });
    }
    out
}

// ── the model, written independently of the decoder ───────────────────────────────────────────────

/// Emission and transition halves of one path's log-score, kept apart so a gap between two paths can say
/// WHICH half of the model prefers the wrong one. Shares no code with `decode_v2`.
fn score_parts(em: &[[f64; 4]], t: &[[f64; 4]; 4], path: &[usize]) -> (f64, f64) {
    let mut ev = em[0][path[0]];
    let mut tr = 0.0;
    for i in 1..path.len() {
        tr += t[path[i - 1]][path[i]].max(T_FLOOR).ln();
        ev += em[i][path[i]];
    }
    (ev, tr)
}

fn score(em: &[[f64; 4]], t: &[[f64; 4]; 4], path: &[usize]) -> f64 {
    let (ev, tr) = score_parts(em, t, path);
    ev + tr
}

/// Best path and score by exhaustive enumeration of all 4^n — the oracle for a handful of epochs.
fn brute_force(em: &[[f64; 4]], t: &[[f64; 4]; 4]) -> (Vec<usize>, f64) {
    let n = em.len();
    let mut best = (Vec::new(), f64::NEG_INFINITY);
    for code in 0..4usize.pow(n as u32) {
        let path: Vec<usize> = (0..n).map(|i| (code >> (2 * i)) & 3).collect();
        let s = score(em, t, &path);
        if s > best.1 {
            best = (path, s);
        }
    }
    best
}

/// Deterministic LCG in [-4, 4): the same sequences on every run, spread comparably to the log-transitions.
fn lcg(state: &mut u64) -> f64 {
    *state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
    ((*state >> 33) as f64) / ((1u64 << 31) as f64) * 8.0 - 4.0
}

fn cols(labels: &[SleepStage]) -> Vec<usize> {
    labels
        .iter()
        .map(|&s| match s {
            SleepStage::Deep => DEEP,
            SleepStage::Rem => REM,
            SleepStage::Light => LIGHT,
            SleepStage::Wake => AWAKE,
        })
        .collect()
}

/// Per-epoch argmax of the emissions — the decoder deleted, leaving only the evidence.
fn argmax_path(em: &[[f64; 4]]) -> Vec<usize> {
    em.iter().map(|r| (0..4).max_by(|a, b| r[*a].total_cmp(&r[*b])).unwrap()).collect()
}

/// Decode the reversed sequence under the transposed matrix. Maximises the same objective by a different
/// recursion, so an equal best score is an independent confirmation that the search is exact.
fn reverse_decode(em: &[[f64; 4]], t: &[[f64; 4]; 4]) -> Vec<usize> {
    let mut rev: Vec<[f64; 4]> = em.to_vec();
    rev.reverse();
    let mut tt = [[0.0f64; 4]; 4];
    for a in 0..4 {
        for b in 0..4 {
            tt[a][b] = t[b][a];
        }
    }
    let mut path = cols(&decode_v2(&rev, &tt));
    path.reverse();
    path
}

/// True when no single-epoch change to `path` raises its score — a necessary condition for optimality that
/// costs 3n instead of 4^n, so it runs on every real span.
fn locally_optimal(em: &[[f64; 4]], t: &[[f64; 4]; 4], path: &[usize]) -> bool {
    let base = score(em, t, path);
    let mut probe = path.to_vec();
    for i in 0..path.len() {
        let keep = probe[i];
        for c in 0..4 {
            if c == keep {
                continue;
            }
            probe[i] = c;
            if score(em, t, &probe) > base + 1e-9 {
                probe[i] = keep;
                return false;
            }
        }
        probe[i] = keep;
    }
    true
}

/// The best path that agrees with a two-class reference: the columns the reference rules out are forbidden
/// rather than removed, so the result is still a path the model can score.
fn constrained_decode(em: &[[f64; 4]], t: &[[f64; 4]; 4], call: &[Option<bool>]) -> Vec<usize> {
    let mut masked = em.to_vec();
    for (row, c) in masked.iter_mut().zip(call) {
        match c {
            Some(true) => {
                row[DEEP] = FORBID;
                row[REM] = FORBID;
                row[LIGHT] = FORBID;
            }
            Some(false) => row[AWAKE] = FORBID,
            None => {}
        }
    }
    cols(&decode_v2(&masked, t))
}

// ── scoreboards ───────────────────────────────────────────────────────────────────────────────────

#[derive(Default, Clone, Copy)]
struct TwoClass {
    n: i64,
    pred_wake: i64,
    true_wake: i64,
    hit: i64,
}

impl TwoClass {
    fn add(&mut self, pred_wake: bool, true_wake: bool) {
        self.n += 1;
        self.pred_wake += i64::from(pred_wake);
        self.true_wake += i64::from(true_wake);
        self.hit += i64::from(pred_wake && true_wake);
    }
    fn wake_pct(&self) -> f64 {
        100.0 * self.pred_wake as f64 / self.n.max(1) as f64
    }
    fn kappa(&self) -> f64 {
        let n = self.n.max(1) as f64;
        let (pw, tw) = (self.pred_wake as f64, self.true_wake as f64);
        let agree = (self.hit as f64 + (n - pw - tw + self.hit as f64)) / n;
        let expect = (pw * tw + (n - pw) * (n - tw)) / (n * n);
        if expect >= 1.0 { 0.0 } else { (agree - expect) / (1.0 - expect) }
    }
}

fn stage_at(segs: &[StageSegment], ts: i64) -> Option<SleepStage> {
    segs.iter().find(|g| g.start <= ts && ts < g.end).map(|g| g.stage)
}

/// How a candidate produces a path. A matrix swap has two honest readings and they are not the same
/// change: re-decoding pinned emissions is the decoder alone, while re-staging also moves the probe pass
/// the onset-anchored prior runs.
enum Kind {
    Restage(Box<Params>),
    Decode([[f64; 4]; 4]),
}

struct Cfg {
    label: String,
    kind: Kind,
}

fn restage(label: &str, p: Params) -> Cfg {
    Cfg { label: label.to_string(), kind: Kind::Restage(Box::new(p)) }
}

fn decode_only(label: &str, t: [[f64; 4]; 4]) -> Cfg {
    Cfg { label: label.to_string(), kind: Kind::Decode(t) }
}

fn path_of(prep: &Prepared, kind: &Kind) -> Vec<usize> {
    match kind {
        Kind::Restage(p) => cols(&decode_v2(&emissions_v2(prep, p), &p.transition)),
        Kind::Decode(t) => cols(&decode_v2(&emissions_v2(prep, &Params::SHIPPED), t)),
    }
}

fn labels_of(path: &[usize]) -> Vec<SleepStage> {
    path.iter().map(|&c| physio_algo::sleep::STAGE_ORDER[c]).collect()
}

/// Band two-class kappa over the app's real path — decode, tile, then `refine_wake` — plus the census, so
/// a span the density gate declined cannot pass for a refined one. "Changed" and "refined" are different
/// questions: a gate that declines and a refinement that finds nothing to do both leave the labels alone.
fn score_band(spans: &[Span], kind: &Kind) -> (TwoClass, RefineCensus) {
    let (mut tc, mut census) = (TwoClass::default(), RefineCensus::default());
    for s in spans {
        let segs = segments_v2(&s.prep, &labels_of(&path_of(&s.prep, kind)));
        let fine = census.refine(&segs, &s.accel, &s.steps);
        for &(ts, st) in &s.band {
            if let Some(l) = stage_at(&fine, ts) {
                tc.add(l == SleepStage::Wake, st != BAND_ASLEEP);
            }
        }
    }
    (tc, census)
}

/// 4-class kappa against PSG truth. Unrefined: no PSG cohort carries a step stream, so `refine_wake`
/// declines — section 1 verifies that instead of assuming it.
fn score_psg(nights: &[PsgNight], preps: &[Prepared], kind: &Kind) -> f64 {
    let mut cm = [[0i64; 4]; 4];
    for (n, prep) in nights.iter().zip(preps) {
        let path = path_of(prep, kind);
        for (i, &st) in epoch_starts_v2(prep).iter().enumerate() {
            if let Some(&t) = n.truth.get(&st) {
                cm[TRUTH_TO_COL[t]][path[i]] += 1;
            }
        }
    }
    kappa4(&cm)
}

// ── 1  the decoder in isolation ───────────────────────────────────────────────────────────────────

/// A second matrix beside the shipped one, with a hard zero row so the floor is exercised.
const PROBE_T: [[f64; 4]; 4] = [
    [0.50, 0.20, 0.20, 0.10],
    [0.10, 0.40, 0.40, 0.10],
    [0.25, 0.25, 0.25, 0.25],
    [0.00, 0.00, 0.50, 0.50],
];

fn section_isolation(spans: &[Span], psg: &[(&str, Vec<PsgNight>, Vec<Prepared>)]) {
    println!("1  the decoder in isolation — is the returned path the maximum-likelihood one");
    println!("   A decoder is a pure function of (emissions, transitions), so it needs no corpus to check.");
    println!("   The oracle is exhaustive enumeration; the scorer is written separately from the decoder.");

    // (a) exhaustive parity over crafted sequences, under two matrices.
    let mut st = 0xbeef_1234_u64;
    let (mut cases, mut optimal, mut same_path) = (0usize, 0usize, 0usize);
    for case in 0..200 {
        let t = if case % 2 == 0 { Params::SHIPPED.transition } else { PROBE_T };
        let em: Vec<[f64; 4]> =
            (0..8).map(|_| [lcg(&mut st), lcg(&mut st), lcg(&mut st), lcg(&mut st)]).collect();
        let got = cols(&decode_v2(&em, &t));
        let (want, best) = brute_force(&em, &t);
        cases += 1;
        optimal += usize::from(score(&em, &t, &got) == best);
        same_path += usize::from(got == want);
    }
    println!("\n   (a) exhaustive enumeration, 8 epochs = 65,536 paths per case, two matrices alternating");
    println!("       cases {cases}   path score equals the enumerated best {optimal}   identical path {same_path}");

    // (b) every injected path, at a margin that dominates the transitions and at one that does not.
    let t = Params::SHIPPED.transition;
    let (mut strong, mut weak, mut total) = (0usize, 0usize, 0usize);
    for code in 0..4096usize {
        let want: Vec<usize> = (0..6).map(|i| (code >> (2 * i)) & 3).collect();
        for (margin, hits) in [(200.0f64, &mut strong), (1.0f64, &mut weak)] {
            let em: Vec<[f64; 4]> = want
                .iter()
                .map(|&s| {
                    let mut row = [0.0; 4];
                    row[s] = margin;
                    row
                })
                .collect();
            *hits += usize::from(cols(&decode_v2(&em, &t)) == want);
        }
        total += 1;
    }
    println!("\n   (b) injected paths recovered, all 4^6 over 6 epochs (recovering one proves nothing)");
    println!("       total {total}   margin 200 (dominates any transition) {strong}   margin 1.0 {weak}");
    println!("       The weak row is the point of the decoder: at a margin the sticky diagonal can outvote,");
    println!("       it is SUPPOSED to disagree with the emissions. A decoder recovering 4096 of 4096 there");
    println!("       would be a per-epoch argmax.");

    // (c) two independent oracles at corpus scale, on every real span.
    println!("\n   (c) at corpus scale, where 4^n is unreachable: two checks that need no enumeration");
    println!("       {:<26}{:>8}{:>10}{:>14}{:>16}", "corpus", "seqs", "epochs", "reverse=fwd", "locally optimal");
    let mut rows: Vec<(String, usize, usize, usize, usize)> = Vec::new();
    let mut cont = (0usize, 0usize, 0usize, 0usize);
    for s in spans {
        let em = emissions_v2(&s.prep, &Params::SHIPPED);
        if em.is_empty() {
            continue;
        }
        let p = cols(&decode_v2(&em, &t));
        cont.0 += 1;
        cont.1 += em.len();
        cont.2 += usize::from(score(&em, &t, &p) == score(&em, &t, &reverse_decode(&em, &t)));
        cont.3 += usize::from(locally_optimal(&em, &t, &p));
    }
    rows.push(("continuous spans".into(), cont.0, cont.1, cont.2, cont.3));
    for (name, nights, preps) in psg {
        let mut acc = (0usize, 0usize, 0usize, 0usize);
        for prep in preps {
            let em = emissions_v2(prep, &Params::SHIPPED);
            if em.is_empty() {
                continue;
            }
            let p = cols(&decode_v2(&em, &t));
            acc.0 += 1;
            acc.1 += em.len();
            acc.2 += usize::from(score(&em, &t, &p) == score(&em, &t, &reverse_decode(&em, &t)));
            acc.3 += usize::from(locally_optimal(&em, &t, &p));
        }
        rows.push((format!("{name} nights ({})", nights.len()), acc.0, acc.1, acc.2, acc.3));
    }
    for (n, seqs, eps, rev, loc) in &rows {
        println!("       {n:<26}{seqs:>8}{eps:>10}{rev:>14}{loc:>16}");
    }

    // (d) the refinement's gate on each family, stated rather than assumed.
    let (_, census) = score_band(spans, &Kind::Restage(Box::new(Params::SHIPPED)));
    let mut psg_census = RefineCensus::default();
    for (_, nights, preps) in psg {
        // The cohort's OWN gravity, and no steps because no cohort has a `steps.csv` — so the census says
        // whether it is the step stream alone that declines or the gravity as well.
        for (night, prep) in nights.iter().zip(preps) {
            let segs = segments_v2(prep, &labels_of(&path_of(prep, &Kind::Restage(Box::new(Params::SHIPPED)))));
            psg_census.refine(&segs, &night.input.accel, &[]);
        }
    }
    println!("\n   (d) which family runs the app's last stage, verified not assumed");
    println!("{}", census.line("continuous"));
    println!("{}", psg_census.line("PSG, no steps.csv"));
    println!("       so every PSG kappa below is the unrefined staging. Stated, because the gate is silent.");
}

// ── 2  decoder error against model error ──────────────────────────────────────────────────────────

/// The longest run of consecutive prepared epochs that all carry a reference label, as (offset, len).
fn labelled_run(starts: &[i64], has: impl Fn(usize) -> bool) -> (usize, usize) {
    let (mut best, mut cur) = ((0usize, 0usize), (0usize, 0usize));
    for i in 0..starts.len() {
        let contiguous = i > 0 && starts[i] == starts[i - 1] + EPOCH_SEC;
        if has(i) && (contiguous || cur.1 == 0) {
            if cur.1 == 0 {
                cur = (i, 1);
            } else {
                cur.1 += 1;
            }
        } else if has(i) {
            cur = (i, 1);
        } else {
            cur = (0, 0);
        }
        if cur.1 > best.1 {
            best = cur;
        }
    }
    best
}

struct GapRow {
    label: String,
    nights: usize,
    epochs: usize,
    /// Nights where the reference path scores at least as high as our decoded one.
    ref_wins: usize,
    /// Log units per epoch by which the model prefers its own path, split into its two halves.
    ev_gap: f64,
    tr_gap: f64,
    /// Epochs where our path and the reference's differ.
    disagree: usize,
    /// The sequence whose per-epoch gap is SMALLEST — the closest the reference ever came to outscoring us.
    closest: (String, f64),
}

fn gap_row(label: &str, sets: Vec<(String, Vec<[f64; 4]>, Vec<usize>)>) -> GapRow {
    let t = Params::SHIPPED.transition;
    let mut row = GapRow {
        label: label.to_string(),
        nights: 0,
        epochs: 0,
        ref_wins: 0,
        ev_gap: 0.0,
        tr_gap: 0.0,
        disagree: 0,
        closest: (String::from("-"), f64::INFINITY),
    };
    for (name, em, reference) in sets {
        if em.len() < 2 {
            continue;
        }
        // Re-decode the sub-sequence: the global optimum restricted to a window is not the window's
        // optimum, so comparing against the restriction would blame the decoder for the cut.
        let ours = cols(&decode_v2(&em, &t));
        let (oe, ot) = score_parts(&em, &t, &ours);
        let (re, rt) = score_parts(&em, &t, &reference);
        row.nights += 1;
        row.epochs += em.len();
        row.ref_wins += usize::from(re + rt >= oe + ot);
        row.ev_gap += oe - re;
        row.tr_gap += ot - rt;
        row.disagree += ours.iter().zip(&reference).filter(|(a, b)| a != b).count();
        let per_epoch = (oe + ot - re - rt) / em.len() as f64;
        if per_epoch < row.closest.1 {
            row.closest = (name, per_epoch);
        }
    }
    row
}

fn section_gap(spans: &[Span], psg: &[(&str, Vec<PsgNight>, Vec<Prepared>)]) {
    println!("\n\n2  decoder error or model error — which one the disagreements belong to");
    println!("   The decoder maximises evidence + transitions. If the REFERENCE's own path scored higher and");
    println!("   we returned something else, the search failed. If our path scores higher, the search did its");
    println!("   job and the model prefers the wrong answer — and the gap says which half of the model does.");

    let mut rows: Vec<GapRow> = Vec::new();
    for (name, nights, preps) in psg {
        let sets: Vec<(String, Vec<[f64; 4]>, Vec<usize>)> = nights
            .iter()
            .zip(preps)
            .filter_map(|(n, prep)| {
                let em = emissions_v2(prep, &Params::SHIPPED);
                let starts = epoch_starts_v2(prep);
                let (off, len) = labelled_run(&starts, |i| n.truth.contains_key(&starts[i]));
                (len >= 20).then(|| {
                    let reference: Vec<usize> =
                        (off..off + len).map(|i| TRUTH_TO_COL[n.truth[&starts[i]]]).collect();
                    (n.name.clone(), em[off..off + len].to_vec(), reference)
                })
            })
            .collect();
        rows.push(gap_row(&format!("{name} PSG hypnogram"), sets));
    }
    // The band is two-class, so its "reference path" is the best 4-class path that agrees with it.
    let t = Params::SHIPPED.transition;
    let band_sets: Vec<(String, Vec<[f64; 4]>, Vec<usize>)> = spans
        .iter()
        .filter_map(|s| {
            let em = emissions_v2(&s.prep, &Params::SHIPPED);
            let starts = epoch_starts_v2(&s.prep);
            let (off, len) = labelled_run(&starts, |i| s.band_epoch[i].is_some());
            (len >= 20).then(|| {
                let sub = em[off..off + len].to_vec();
                let call: Vec<Option<bool>> = s.band_epoch[off..off + len].to_vec();
                let reference = constrained_decode(&sub, &t, &call);
                (format!("span@{}", starts[off]), sub, reference)
            })
        })
        .collect();
    rows.push(gap_row("band, best path that agrees", band_sets));

    println!("\n   {:<28}{:>7}{:>9}{:>10}{:>12}{:>12}{:>11}", "reference", "seqs", "epochs", "ref>=us",
        "evidence/ep", "transit./ep", "disagree%");
    for r in &rows {
        let e = r.epochs.max(1) as f64;
        println!(
            "   {:<28}{:>7}{:>9}{:>10}{:>12.3}{:>12.3}{:>10.1}%",
            r.label,
            r.nights,
            r.epochs,
            r.ref_wins,
            r.ev_gap / e,
            r.tr_gap / e,
            100.0 * r.disagree as f64 / e
        );
    }
    for r in &rows {
        println!("       closest call on {:<28} {:>8.3} log units per epoch", r.closest.0, r.closest.1);
    }
    println!("   `ref scores >=` is the decoder's own failure count: 0 means the search never returned a path");
    println!("   the reference beat. The two gap columns are the same total split by half of the model — the");
    println!("   larger one is the half that prefers our wrong path.");

    // The decoder's total work: how far the path sits from the evidence alone.
    let mut moved_cont = (0usize, 0usize);
    for s in spans {
        let em = emissions_v2(&s.prep, &Params::SHIPPED);
        if em.is_empty() {
            continue;
        }
        let (a, b) = (argmax_path(&em), cols(&decode_v2(&em, &t)));
        moved_cont.0 += a.iter().zip(&b).filter(|(x, y)| x != y).count();
        moved_cont.1 += em.len();
    }
    let (tc_ship, _) = score_band(spans, &Kind::Restage(Box::new(Params::SHIPPED)));
    let (tc_argmax, _) = score_band_paths(spans, argmax_path);
    println!("\n   the decoder's own contribution, measured by deleting it (per-epoch argmax of the same");
    println!("   emissions): it moves {} of {} continuous epochs ({:.1}%), and those moves are worth",
        moved_cont.0, moved_cont.1, 100.0 * moved_cont.0 as f64 / moved_cont.1.max(1) as f64);
    println!("   band kappa2 {:.3} -> {:.3} ({:+.3}), wake {:.1}% -> {:.1}%.", tc_argmax.kappa(),
        tc_ship.kappa(), tc_ship.kappa() - tc_argmax.kappa(), tc_argmax.wake_pct(), tc_ship.wake_pct());

    // What the smoothing already does to the short wake runs H2 is about.
    let (mut a_short, mut a_all, mut d_short, mut d_all) = (0usize, 0usize, 0usize, 0usize);
    for s in spans {
        let em = emissions_v2(&s.prep, &Params::SHIPPED);
        if em.is_empty() {
            continue;
        }
        let (sa, aa) = wake_runs(&argmax_path(&em));
        let (sd, ad) = wake_runs(&cols(&decode_v2(&em, &t)));
        a_short += sa;
        a_all += aa;
        d_short += sd;
        d_all += ad;
    }
    println!("\n   for H2, whose defect is the short wake runs the refinement cannot see: the decoder is");
    println!("   ALREADY smoothing them. Wake runs <= 2.5 min over the same spans, argmax {a_short} of");
    println!("   {a_all} -> decoded {d_short} of {d_all}. The survivors outlived a path search built to");
    println!("   remove them, so they are emission-driven, not a missing smoother.");
}

/// Wake runs at most 5 epochs (2.5 min) long, and wake runs of any length.
fn wake_runs(path: &[usize]) -> (usize, usize) {
    let (mut short, mut all, mut run) = (0usize, 0usize, 0usize);
    for i in 0..=path.len() {
        if i < path.len() && path[i] == AWAKE {
            run += 1;
            continue;
        }
        if run > 0 {
            all += 1;
            short += usize::from(run <= 5);
        }
        run = 0;
    }
    (short, all)
}

/// Band score from a caller-supplied path over pinned shipped emissions.
fn score_band_paths(spans: &[Span], f: impl Fn(&[[f64; 4]]) -> Vec<usize>) -> (TwoClass, RefineCensus) {
    let (mut tc, mut census) = (TwoClass::default(), RefineCensus::default());
    for s in spans {
        let em = emissions_v2(&s.prep, &Params::SHIPPED);
        if em.is_empty() {
            continue;
        }
        let segs = segments_v2(&s.prep, &labels_of(&f(&em)));
        let fine = census.refine(&segs, &s.accel, &s.steps);
        for &(ts, st) in &s.band {
            if let Some(l) = stage_at(&fine, ts) {
                tc.add(l == SleepStage::Wake, st != BAND_ASLEEP);
            }
        }
    }
    (tc, census)
}

// ── 3  the two axes, sized against each other ─────────────────────────────────────────────────────

/// `T_ij^gamma` renormalised per row: gamma 0 is uniform (no smoothing at all), 1 is shipped, above 1 is
/// stickier. One knob that spans "decoder off" to "decoder dominant".
fn sharpen(t: &[[f64; 4]; 4], gamma: f64) -> [[f64; 4]; 4] {
    let mut out = [[0.0f64; 4]; 4];
    for (o, r) in out.iter_mut().zip(t) {
        let raw: Vec<f64> = r.iter().map(|v| v.max(T_FLOOR).powf(gamma)).collect();
        let sum: f64 = raw.iter().sum();
        for (oc, v) in o.iter_mut().zip(&raw) {
            *oc = v / sum;
        }
    }
    out
}

fn section_axes(spans: &[Span], psg: &[(&str, Vec<PsgNight>, Vec<Prepared>)], base: &[Vec<usize>]) {
    println!("\n\n3  the two axes sized against each other, on the same references");
    println!("   Held-fixed emissions with the matrix swept is the DECODER's leverage; the shipped matrix");
    println!("   with the emissions ablated is the EMISSIONS'. The wider kappa spread owns the error.");

    let sh = Params::SHIPPED;
    let mut cfgs: Vec<Cfg> = Vec::new();
    for g in [0.0f64, 0.25, 0.5, 1.0, 2.0, 4.0] {
        let label = match g {
            0.0 => "decoder: gamma 0 (uniform = off)".to_string(),
            1.0 => "decoder: gamma 1 (SHIPPED)".to_string(),
            _ => format!("decoder: gamma {g}"),
        };
        cfgs.push(decode_only(&label, sharpen(&sh.transition, g)));
    }
    cfgs.push(restage("emissions: no cycle prior", Params {
        cycle_deep_scale: 0.0,
        cycle_rem_scale: 0.0,
        cycle_rem_early_penalty: 0.0,
        ..sh
    }));
    cfgs.push(restage("emissions: no motion", Params { deep_motion: 0.0, rem_motion: 0.0, awake_motion: 0.0,
        motion_gate_boost: 0.0, ..sh }));
    cfgs.push(restage("emissions: cardiac z-scores only", Params {
        deep_motion: 0.0,
        rem_motion: 0.0,
        awake_motion: 0.0,
        motion_gate_boost: 0.0,
        resp_weight: 0.0,
        deep_gate_slope: 0.0,
        cycle_deep_scale: 0.0,
        cycle_rem_scale: 0.0,
        cycle_rem_early_penalty: 0.0,
        ..sh
    }));
    cfgs.push(restage("emissions: no base rate", Params { base_rate: [0.25; 4], ..sh }));
    cfgs.push(restage("emissions: no deep gate", Params { deep_gate_slope: 0.0, ..sh }));

    println!("\n   {:<34}{:>9}{:>9}{:>9}{:>9}{:>9}{:>10}", "config", "band k2", "band w%", "dreamt",
        "aauwss", "sl-accel", "moved%");
    let mut band_dec: Vec<f64> = Vec::new();
    let mut band_emi: Vec<f64> = Vec::new();
    for c in &cfgs {
        let (tc, _) = score_band(spans, &c.kind);
        let mut ks = Vec::new();
        for (_, nights, preps) in psg {
            ks.push(score_psg(nights, preps, &c.kind));
        }
        let (mut moved, mut total) = (0usize, 0usize);
        for (s, b) in spans.iter().zip(base) {
            let p = path_of(&s.prep, &c.kind);
            moved += p.iter().zip(b).filter(|(x, y)| x != y).count();
            total += p.len();
        }
        println!(
            "   {:<34}{:>9.3}{:>8.1}%{:>9.3}{:>9.3}{:>9.3}{:>9.1}%",
            c.label,
            tc.kappa(),
            tc.wake_pct(),
            ks[0],
            ks[1],
            ks[2],
            100.0 * moved as f64 / total.max(1) as f64
        );
        if matches!(c.kind, Kind::Decode(_)) {
            band_dec.push(tc.kappa());
        } else {
            band_emi.push(tc.kappa());
        }
    }
    let spread = |v: &[f64]| v.iter().cloned().fold(f64::NEG_INFINITY, f64::max) - v.iter().cloned().fold(f64::INFINITY, f64::min);
    println!("\n   band kappa2 spread: decoder axis {:.3} over 6 matrices, emission axis {:.3} over 5",
        spread(&band_dec), spread(&band_emi));
    println!("   `moved%` is against the shipped path, and the ~8% grid-phase noise floor applies to it.");
}

// ── 4  the matrix and the base rates against something real ───────────────────────────────────────

/// Row-normalised 30 s transition counts from a set of reference hypnograms, in emission-column order.
fn empirical(sets: &[(Vec<i64>, BTreeMap<i64, usize>)]) -> ([[f64; 4]; 4], [i64; 4], i64) {
    let mut counts = [[0i64; 4]; 4];
    let mut occupancy = [0i64; 4];
    for (starts, truth) in sets {
        for w in starts.windows(2) {
            let (a, b) = (truth.get(&w[0]), truth.get(&w[1]));
            if let (Some(&x), Some(&y)) = (a, b) {
                if w[1] == w[0] + EPOCH_SEC {
                    counts[TRUTH_TO_COL[x]][TRUTH_TO_COL[y]] += 1;
                }
            }
        }
        for (_, &c) in truth.iter() {
            occupancy[TRUTH_TO_COL[c]] += 1;
        }
    }
    let total: i64 = counts.iter().flatten().sum();
    let mut out = [[0.0f64; 4]; 4];
    for (o, r) in out.iter_mut().zip(&counts) {
        let s: i64 = r.iter().sum();
        for (oc, &v) in o.iter_mut().zip(r) {
            *oc = if s == 0 { 0.0 } else { v as f64 / s as f64 };
        }
    }
    (out, occupancy, total)
}

fn print_matrix(name: &str, t: &[[f64; 4]; 4]) {
    let names = ["deep", "rem", "light", "awake"];
    println!("   {name}");
    println!("       {:<8}{:>9}{:>9}{:>9}{:>9}", "from\\to", "deep", "rem", "light", "awake");
    for (i, r) in t.iter().enumerate() {
        println!("       {:<8}{:>9.4}{:>9.4}{:>9.4}{:>9.4}", names[i], r[0], r[1], r[2], r[3]);
    }
}

fn section_reference_matrix(spans: &[Span], psg: &[(&str, Vec<PsgNight>, Vec<Prepared>)]) -> [[f64; 4]; 4] {
    println!("\n\n4  the transition matrix and the base rates against a measured one");
    println!("   The shipped matrix and base rates are hand-set constants with no recorded derivation. PSG");
    println!("   hypnograms at 30 s give a real 4x4; the band's own call gives a two-class one for free.");

    let mut all: Vec<(Vec<i64>, BTreeMap<i64, usize>)> = Vec::new();
    println!("\n   per-cohort 30 s transition rates, and the pooled one below");
    println!("   {:<14}{:>8}{:>10}{:>10}{:>10}{:>10}{:>10}", "cohort", "nights", "pairs", "P(deep|deep)",
        "P(rem|rem)", "P(lt|lt)", "P(aw|aw)");
    for (name, nights, preps) in psg {
        let sets: Vec<(Vec<i64>, BTreeMap<i64, usize>)> = nights
            .iter()
            .zip(preps)
            .map(|(n, prep)| (epoch_starts_v2(prep), n.truth.clone()))
            .collect();
        let (t, _, pairs) = empirical(&sets);
        println!(
            "   {:<14}{:>8}{:>10}{:>13.3}{:>10.3}{:>10.3}{:>10.3}",
            name, nights.len(), pairs, t[DEEP][DEEP], t[REM][REM], t[LIGHT][LIGHT], t[AWAKE][AWAKE]
        );
        all.extend(sets);
    }
    let (pooled, occ, pairs) = empirical(&all);
    let sh = Params::SHIPPED.transition;
    println!(
        "   {:<14}{:>8}{:>10}{:>13.3}{:>10.3}{:>10.3}{:>10.3}",
        "POOLED PSG", all.len(), pairs, pooled[DEEP][DEEP], pooled[REM][REM], pooled[LIGHT][LIGHT],
        pooled[AWAKE][AWAKE]
    );
    println!(
        "   {:<14}{:>8}{:>10}{:>13.3}{:>10.3}{:>10.3}{:>10.3}",
        "SHIPPED", "-", "-", sh[DEEP][DEEP], sh[REM][REM], sh[LIGHT][LIGHT], sh[AWAKE][AWAKE]
    );
    println!();
    print_matrix("SHIPPED", &sh);
    print_matrix("POOLED PSG, measured", &pooled);

    // Base rates against the PSG occupancy the same hypnograms give.
    let tot: i64 = occ.iter().sum();
    let br = Params::SHIPPED.base_rate;
    let brs: f64 = br.iter().sum();
    println!("\n   base rates: the recipe adds ln(base_rate) to EVERY epoch, so it is a per-epoch bias, not");
    println!("   an HMM start prior. Against the same pooled hypnograms:");
    println!("   {:<20}{:>10}{:>10}{:>10}{:>10}", "", "deep", "rem", "light", "awake");
    println!(
        "   {:<20}{:>10.3}{:>10.3}{:>10.3}{:>10.3}",
        "shipped base_rate", br[0], br[1], br[2], br[3]
    );
    println!(
        "   {:<20}{:>10.3}{:>10.3}{:>10.3}{:>10.3}",
        "shipped, normalised", br[0] / brs, br[1] / brs, br[2] / brs, br[3] / brs
    );
    println!(
        "   {:<20}{:>10.3}{:>10.3}{:>10.3}{:>10.3}",
        "pooled PSG fraction",
        occ[0] as f64 / tot as f64,
        occ[1] as f64 / tot as f64,
        occ[2] as f64 / tot as f64,
        occ[3] as f64 / tot as f64
    );
    println!("   The shipped four sum to {brs:.2}, not 1. That is harmless for the argmax — a constant shift");
    println!("   on all four columns of every epoch cancels — but it means the numbers are not readable as");
    println!("   probabilities, and the normalised row is what to compare with PSG.");

    // The band's two-class rate, which needs no PSG at all.
    let (mut stay_sleep, mut from_sleep, mut stay_wake, mut from_wake) = (0i64, 0i64, 0i64, 0i64);
    for s in spans {
        for w in s.band_epoch.windows(2) {
            if let (Some(a), Some(b)) = (w[0], w[1]) {
                if a {
                    from_wake += 1;
                    stay_wake += i64::from(b);
                } else {
                    from_sleep += 1;
                    stay_sleep += i64::from(!b);
                }
            }
        }
    }
    // Fold the shipped matrix to two classes, weighting the three sleep rows by their normalised base rate.
    let wsum: f64 = br[DEEP] + br[REM] + br[LIGHT];
    let shipped_stay_sleep: f64 = [DEEP, REM, LIGHT]
        .iter()
        .map(|&r| br[r] / wsum * (sh[r][DEEP] + sh[r][REM] + sh[r][LIGHT]))
        .sum();
    println!("\n   the band's own two-class rate over the continuous spans, per 30 s epoch:");
    println!("       P(asleep -> asleep) band {:.4} of {} pairs   shipped, base-rate folded {:.4}",
        stay_sleep as f64 / from_sleep.max(1) as f64, from_sleep, shipped_stay_sleep);
    println!("       P(non-asleep -> non-asleep) band {:.4} of {} pairs   shipped awake row {:.4}",
        stay_wake as f64 / from_wake.max(1) as f64, from_wake, sh[AWAKE][AWAKE]);
    pooled
}

// ── 5  three bounded rescues on the matrix ────────────────────────────────────────────────────────

/// Keep the shipped off-diagonal SHAPE and take only the stickiness from a measured matrix: each row's
/// off-diagonals are rescaled to fit whatever the diagonal leaves.
fn diagonal_from(shipped: &[[f64; 4]; 4], measured: &[[f64; 4]; 4]) -> [[f64; 4]; 4] {
    let mut out = *shipped;
    for i in 0..4 {
        let d = measured[i][i];
        let off: f64 = (0..4).filter(|&j| j != i).map(|j| shipped[i][j]).sum();
        for j in 0..4 {
            out[i][j] = if j == i {
                d
            } else if off > 0.0 {
                shipped[i][j] / off * (1.0 - d)
            } else {
                (1.0 - d) / 3.0
            };
        }
    }
    out
}

fn blend(a: &[[f64; 4]; 4], b: &[[f64; 4]; 4], w: f64) -> [[f64; 4]; 4] {
    let mut out = [[0.0f64; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            out[i][j] = (1.0 - w) * a[i][j] + w * b[i][j];
        }
    }
    out
}

fn section_rescue(spans: &[Span], psg: &[(&str, Vec<PsgNight>, Vec<Prepared>)], pooled: &[[f64; 4]; 4]) {
    println!("\n\n5  three bounded rescues on the matrix (rule 9), the failures reported too");
    println!("   Each is a FULL re-stage, because shipping a matrix also moves the probe pass the");
    println!("   onset-anchored prior runs — the decode-only readings are in section 3.");

    let sh = Params::SHIPPED;
    let cands = vec![
        restage("SHIPPED", sh),
        restage("1  the pooled PSG matrix", Params { transition: *pooled, ..sh }),
        decode_only("1b the same, decode only", *pooled),
        restage("2  50/50 with the PSG matrix", Params { transition: blend(&sh.transition, pooled, 0.5), ..sh }),
        restage("3  PSG stickiness, shipped shape",
            Params { transition: diagonal_from(&sh.transition, pooled), ..sh }),
    ];
    println!("\n   {:<34}{:>9}{:>9}{:>9}{:>9}{:>9}", "candidate", "band k2", "band w%", "dreamt", "aauwss",
        "sl-accel");
    let mut base: Option<(f64, [f64; 3])> = None;
    for c in &cands {
        let (tc, _) = score_band(spans, &c.kind);
        let mut ks = [0.0f64; 3];
        for (i, (_, nights, preps)) in psg.iter().enumerate() {
            ks[i] = score_psg(nights, preps, &c.kind);
        }
        match base {
            None => {
                println!("   {:<34}{:>9.3}{:>8.1}%{:>9.3}{:>9.3}{:>9.3}", c.label, tc.kappa(), tc.wake_pct(),
                    ks[0], ks[1], ks[2]);
                base = Some((tc.kappa(), ks));
            }
            Some((bk, bks)) => println!(
                "   {:<34}{:>9.3}{:>8.1}%{:>9.3}{:>9.3}{:>9.3}   d {:+.3} {:+.3} {:+.3} {:+.3}",
                c.label, tc.kappa(), tc.wake_pct(), ks[0], ks[1], ks[2],
                tc.kappa() - bk, ks[0] - bks[0], ks[1] - bks[1], ks[2] - bks[2]
            ),
        }
    }
}

fn main() {
    let sh = Params::SHIPPED;
    println!("Step G — the Viterbi decoder in isolation, then against both reference families");
    println!("fixtures: {}\n", root("continuous").display());

    let spans = continuous_spans(&sh);
    let psg: Vec<(&str, Vec<PsgNight>, Vec<Prepared>)> = PSG
        .iter()
        .filter_map(|s| {
            let nights = load_psg(s);
            if nights.is_empty() {
                return None;
            }
            let preps = nights.iter().map(|n| prepare_v2(&n.input, &sh)).collect();
            Some((*s, nights, preps))
        })
        .collect();
    assert!(!spans.is_empty(), "no continuous spans: check WHOOP_SLEEP_FIXTURES");
    assert_eq!(3, psg.len(), "all three PSG cohorts must load, or a kappa below is a different population");

    let base: Vec<Vec<usize>> = spans.iter().map(|s| path_of(&s.prep, &Kind::Restage(Box::new(sh)))).collect();

    section_isolation(&spans, &psg);
    section_gap(&spans, &psg);
    section_axes(&spans, &psg, &base);
    let pooled = section_reference_matrix(&spans, &psg);
    section_rescue(&spans, &psg, &pooled);
}
