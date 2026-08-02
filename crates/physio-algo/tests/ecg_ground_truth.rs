//! Ground truth for the decode sweep: re-encode real overnight ECG at a layout and a rate the harness
//! knows and the sweep does not, across the whole corpus and a grid that varies one parameter at a time,
//! and count what came back.
//!
//! Measurement only — nothing here is a gate. Convergence on a WRONG answer is counted apart from a
//! refusal to converge, because a sweep that fails loudly is usable and one that succeeds confidently on
//! garbage is not.
//!
//!   cargo test --release -p physio-algo --test ecg_ground_truth -- --ignored --nocapture --test-threads=1

mod ecg_corpus;
#[path = "ecg_corpus/encode.rs"]
#[allow(dead_code)]
mod encode;

use std::collections::BTreeMap;
use std::time::Instant;

use ecg_corpus::{
    add_hum, ecg_fixtures, gaussian_like, matched_to, mean, ppg_fixtures, resample, sawtooth_like,
    shuffled, Fixture,
};
use encode::{detector_consensus, pulse_train};

use physio_algo::ecg::sweep::layout::{encode as pack, BitOrder, Layout, LayoutShape};
use physio_algo::ecg::sweep::{sweep_split, SweepConfig, SweepOutcome, SweepReport};

const EPOCH_MS: f64 = 30_000.0;
const WINDOWS: usize = 3;
const PTT_MS: f64 = 220.0;
const PTT_JITTER_MS: f64 = 18.0;

/// How the waveform is written into the field. A real converter is one of these three, and they are not
/// equally recoverable — see `signedness` in the report.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Coding {
    /// Two's complement, centred on zero.
    Signed,
    /// Unsigned, centred on mid-rail: a signed read of it wraps at every midpoint crossing.
    UnsignedMid,
    /// Unsigned, centred on the quarter-rail and never reaching the top bit: a signed read of it is
    /// bit-for-bit the same waveform, so signedness is not recoverable here by construction.
    UnsignedLow,
}

struct Truth {
    name: &'static str,
    layout: Layout,
    fs_hz: f64,
    hum: f64,
    coding: Coding,
}

fn lay(bits: u8, signed: bool, order: BitOrder, start_bit: usize, stride_bits: usize) -> Layout {
    Layout { bits, signed, order, start_bit, stride_bits }
}

fn t(
    name: &'static str,
    layout: Layout,
    fs_hz: f64,
    hum: f64,
    coding: Coding,
) -> Truth {
    Truth { name, layout, fs_hz, hum, coding }
}

/// The layout grid: one parameter moved at a time off a fixed 400 Hz base, plus the cases that exist to
/// be unrecoverable.
fn layout_grid() -> Vec<Truth> {
    use BitOrder::{LsbFirst as LE, MsbFirst as BE};
    use Coding::*;
    vec![
        t("W16 s LE dense start0", lay(16, true, LE, 0, 16), 400.0, 0.0, Signed),
        t("W24 s LE dense start0", lay(24, true, LE, 0, 24), 400.0, 0.0, Signed),
        t("W32 s LE dense start0", lay(32, true, LE, 0, 32), 400.0, 0.0, Signed),
        t("W18 s LE dense start0 (non-aligned)", lay(18, true, LE, 0, 18), 400.0, 0.0, Signed),
        t("W18 s LE in 24-bit container", lay(18, true, LE, 0, 24), 400.0, 0.0, Signed),
        t("W16 s BE dense start0", lay(16, true, BE, 0, 16), 400.0, 0.0, Signed),
        t("W24 s BE dense start0", lay(24, true, BE, 0, 24), 400.0, 0.0, Signed),
        t("W32 s BE dense start0", lay(32, true, BE, 0, 32), 400.0, 0.0, Signed),
        t("W18 s BE dense start8 (1-byte header)", lay(18, true, BE, 8, 18), 400.0, 0.0, Signed),
        t("W16 s LE dense start24 (3-byte header)", lay(16, true, LE, 24, 16), 400.0, 0.0, Signed),
        t("W16 s LE dense start56 (7-byte header)", lay(16, true, LE, 56, 16), 400.0, 0.0, Signed),
        t("W16 s BE 2-ch interleave start32", lay(16, true, BE, 32, 32), 400.0, 0.0, Signed),
        t("W24 s LE 3-ch interleave start24", lay(24, true, LE, 24, 72), 400.0, 0.0, Signed),
        t("W18 s LE 2-ch start18 (non-aligned offset)", lay(18, true, LE, 18, 36), 400.0, 0.0, Signed),
        t("W16 u LE dense start0 mid-rail", lay(16, false, LE, 0, 16), 400.0, 0.0, UnsignedMid),
        t("W16 u LE dense start0 quarter-rail", lay(16, false, LE, 0, 16), 400.0, 0.0, UnsignedLow),
        // Deliberately outside the enumerated start set: an 18-bit dense field whose first sample begins
        // at bit 18. `candidates()` only reaches a non-byte-aligned start when the stride says a frame
        // interleaves, so this rule cannot be expressed. It is here to see the sweep fail rather than
        // land somewhere confident.
        t("W18 s LE dense start18 (UNREACHABLE)", lay(18, true, LE, 18, 18), 400.0, 0.0, Signed),
    ]
}

/// The rate grid: one fixed, easy layout at every searched rate, with and without power-line hum. Three of
/// the nine rates are the 1/1024-second partner of another, and that is exactly what the hum column tests.
fn rate_grid() -> Vec<Truth> {
    let base = lay(16, true, BitOrder::LsbFirst, 0, 16);
    let names: [(&'static str, f64); 9] = [
        ("R128", 128.0),
        ("R200", 200.0),
        ("R250", 250.0),
        ("R256", 256.0),
        ("R400", 400.0),
        ("R500", 500.0),
        ("R512", 512.0),
        ("R1000", 1000.0),
        ("R1024", 1024.0),
    ];
    let mut out = Vec::new();
    for (n, fs) in names {
        out.push(Truth { name: n, layout: base, fs_hz: fs, hum: 0.0, coding: Coding::Signed });
    }
    for (n, fs) in names {
        let hum_name: &'static str = Box::leak(format!("{n}+hum").into_boxed_str());
        out.push(Truth { name: hum_name, layout: base, fs_hz: fs, hum: 0.30, coding: Coding::Signed });
    }
    out
}

// ---------------------------------------------------------------------------------------------------
// encoding

/// Scale into `bits`-wide counts under `coding`. The gain is arbitrary; nothing may read it as mV.
fn quantise(x: &[f64], bits: u8, coding: Coding) -> Vec<i64> {
    let m = mean(x);
    let span = x.iter().map(|v| (v - m).abs()).fold(0.0f64, f64::max).max(1e-12);
    let full = ((1i64 << (bits - 1)) - 1) as f64;
    match coding {
        Coding::Signed => x.iter().map(|v| (((v - m) / span) * full * 0.9).round() as i64).collect(),
        // Mid-rail: the DC sits exactly on the sign bit, so a signed read wraps constantly.
        Coding::UnsignedMid => {
            let mid = 1i64 << (bits - 1);
            x.iter().map(|v| mid + (((v - m) / span) * full * 0.9).round() as i64).collect()
        }
        // Quarter-rail: the top bit never sets, so a signed read is identical.
        Coding::UnsignedLow => {
            let q = 1i64 << (bits - 2);
            x.iter().map(|v| q + (((v - m) / span) * full * 0.45).round() as i64).collect()
        }
    }
}

/// A byte buffer carrying `wave` (already at the target rate) under `layout`, with `filler` written into
/// the unclaimed slots of an interleaved frame.
fn encode_counts(wave: &[f64], layout: &Layout, coding: Coding, filler: Option<&[f64]>) -> Vec<u8> {
    let counts = quantise(wave, layout.bits, coding);
    let len_bytes = (layout.start_bit + counts.len() * layout.stride_bits).div_ceil(8) + 8;
    let mut bytes = pack(layout, &counts, len_bytes);
    if let Some(f) = filler {
        if layout.stride_bits >= 2 * layout.bits as usize {
            let other = Layout { start_bit: layout.start_bit + layout.bits as usize, ..*layout };
            let filled = pack(&other, &quantise(f, layout.bits, coding), len_bytes);
            for (b, o) in bytes.iter_mut().zip(filled.iter()) {
                *b |= o;
            }
        }
    }
    bytes
}

/// One subject's epoch written under one truth, plus the optical beat train that belongs with it. The
/// beats come off the ORIGINAL 200 Hz samples, so their times are ground truth independent of the rate the
/// bytes were written at.
fn stream(f: &Fixture, tr: &Truth) -> (Vec<u8>, Vec<f64>) {
    let mut wave = resample(&f.samples, f.fs_hz, tr.fs_hz);
    if tr.hum > 0.0 {
        wave = add_hum(&wave, tr.fs_hz, 50.0, tr.hum);
    }
    let filler = resample(&shuffled(&f.samples, 0xA5A5), f.fs_hz, tr.fs_hz);
    let bytes = encode_counts(&wave, &tr.layout, tr.coding, Some(&filler));
    let beats = pulse_train(&detector_consensus(&f.samples, f.fs_hz), f.fs_hz, PTT_MS, PTT_JITTER_MS, 7);
    (bytes, beats)
}

// ---------------------------------------------------------------------------------------------------
// outcome classification

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Verdict {
    /// Converged, reported shape and rate both equal the truth.
    Exact,
    /// Converged on the right rate, reported a different shape but listed the truth as an alias.
    Alias,
    /// Converged on something that is not the truth. The category that matters.
    Wrong,
    /// Refused: the two rate anchors contradicted each other.
    RefusedDisagreement,
    /// Refused: the solved rate is one 1/1024-second conversion off a searched rate.
    RefusedUnitError,
    /// Did not converge.
    Stalled,
}

struct Case {
    verdict: Verdict,
    reported: Option<(LayoutShape, f64)>,
    detail: String,
    ms: f64,
}

fn classify(r: &SweepReport, tr: &Truth) -> Case {
    match &r.outcome {
        SweepOutcome::Converged { shape, fs_hz, .. } => {
            let exact = *shape == tr.layout.shape() && *fs_hz == tr.fs_hz;
            let alias = *fs_hz == tr.fs_hz && r.alias_shapes.contains(&tr.layout.shape());
            let v = if exact {
                Verdict::Exact
            } else if alias {
                Verdict::Alias
            } else {
                Verdict::Wrong
            };
            Case {
                verdict: v,
                reported: Some((*shape, *fs_hz)),
                detail: format!("{} at {fs_hz} Hz", short(shape)),
                ms: 0.0,
            }
        }
        SweepOutcome::Disagreement { mains_fs_hz, ppg_fs_hz, .. } => Case {
            verdict: Verdict::RefusedDisagreement,
            reported: None,
            detail: format!("mains {mains_fs_hz:.1} vs ppg {ppg_fs_hz:.1}"),
            ms: 0.0,
        },
        SweepOutcome::SuspectedUnitError(u) => Case {
            verdict: Verdict::RefusedUnitError,
            reported: None,
            detail: format!("would be {} Hz, rr_long {}", u.true_fs_hz, u.rr_reported_long),
            ms: 0.0,
        },
        SweepOutcome::Searching { reason, best_quality } => Case {
            verdict: Verdict::Stalled,
            reported: None,
            detail: format!("{reason:?} best q {best_quality:?}"),
            ms: 0.0,
        },
    }
}

fn short(s: &LayoutShape) -> String {
    format!(
        "{}b{}{}/{}",
        s.bits,
        if s.signed { "s" } else { "u" },
        if s.order == BitOrder::LsbFirst { "le" } else { "be" },
        s.stride_bits
    )
}

fn pct(a: usize, b: usize) -> f64 {
    if b == 0 {
        0.0
    } else {
        100.0 * a as f64 / b as f64
    }
}

fn quantile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let i = ((sorted.len() - 1) as f64 * q).round() as usize;
    sorted[i]
}

/// Per-truth tally plus the per-parameter columns.
#[derive(Default)]
struct Tally {
    n: usize,
    exact: usize,
    alias: usize,
    wrong: usize,
    stalled: usize,
    refused_unit: usize,
    refused_disagree: usize,
    /// Parameter recovered among the CONVERGED cases only.
    conv: usize,
    bits_ok: usize,
    signed_ok: usize,
    order_ok: usize,
    stride_ok: usize,
    fs_ok: usize,
    ms: Vec<f64>,
    notes: Vec<String>,
}

impl Tally {
    fn add(&mut self, c: &Case, tr: &Truth, subject: &str) {
        self.n += 1;
        self.ms.push(c.ms);
        match c.verdict {
            Verdict::Exact => self.exact += 1,
            Verdict::Alias => self.alias += 1,
            Verdict::Wrong => {
                self.wrong += 1;
                self.notes.push(format!("WRONG  {subject}: {}", c.detail));
            }
            Verdict::Stalled => {
                self.stalled += 1;
                self.notes.push(format!("stall  {subject}: {}", c.detail));
            }
            Verdict::RefusedUnitError => {
                self.refused_unit += 1;
                self.notes.push(format!("refuse {subject}: unit error, {}", c.detail));
            }
            Verdict::RefusedDisagreement => {
                self.refused_disagree += 1;
                self.notes.push(format!("refuse {subject}: anchors disagree, {}", c.detail));
            }
        }
        if let Some((s, fs)) = c.reported {
            let want = tr.layout.shape();
            self.conv += 1;
            self.bits_ok += usize::from(s.bits == want.bits);
            self.signed_ok += usize::from(s.signed == want.signed);
            self.order_ok += usize::from(s.order == want.order);
            self.stride_ok += usize::from(s.stride_bits == want.stride_bits);
            self.fs_ok += usize::from(fs == tr.fs_hz);
        }
    }
}

fn run_grid(title: &str, truths: &[Truth]) {
    let fixtures = ecg_fixtures();
    let cfg = SweepConfig::default();
    println!("\n================ {title} ================");
    println!("subjects {}  windows {WINDOWS}  epoch {EPOCH_MS} ms", fixtures.len());

    let mut all_ms: Vec<f64> = Vec::new();
    let mut grand = Tally::default();
    let mut confusion: BTreeMap<String, usize> = BTreeMap::new();

    for tr in truths {
        let mut tally = Tally::default();
        for f in &fixtures {
            let (bytes, beats) = stream(f, tr);
            let start = Instant::now();
            let r = sweep_split(&bytes, &beats, EPOCH_MS, WINDOWS, &cfg);
            let ms = start.elapsed().as_secs_f64() * 1000.0;
            let mut c = classify(&r, tr);
            c.ms = ms;
            all_ms.push(ms);
            if c.verdict == Verdict::Wrong {
                let (s, fs) = c.reported.unwrap();
                *confusion
                    .entry(format!(
                        "{} @{} Hz  ->  {} @{} Hz",
                        short(&tr.layout.shape()),
                        tr.fs_hz,
                        short(&s),
                        fs
                    ))
                    .or_default() += 1;
            }
            tally.add(&c, tr, &f.subject);
            grand.add(&c, tr, &f.subject);
        }
        let mut ms = tally.ms.clone();
        ms.sort_by(f64::total_cmp);
        println!(
            "\n{:<44} exact {:>2}  alias {:>2}  WRONG {:>2}  stall {:>2}  refuse(unit {} anchors {})  n={}",
            tr.name,
            tally.exact,
            tally.alias,
            tally.wrong,
            tally.stalled,
            tally.refused_unit,
            tally.refused_disagree,
            tally.n
        );
        println!(
            "  per-parameter among {} converged: bits {:.0}%  signed {:.0}%  order {:.0}%  stride {:.0}%  fs {:.0}%   |  {:.0}/{:.0}/{:.0} ms med/p90/max",
            tally.conv,
            pct(tally.bits_ok, tally.conv),
            pct(tally.signed_ok, tally.conv),
            pct(tally.order_ok, tally.conv),
            pct(tally.stride_ok, tally.conv),
            pct(tally.fs_ok, tally.conv),
            quantile(&ms, 0.5),
            quantile(&ms, 0.9),
            quantile(&ms, 1.0)
        );
        for n in &tally.notes {
            println!("    {n}");
        }
    }

    all_ms.sort_by(f64::total_cmp);
    println!("\n---- {title}: totals over {} cases ----", grand.n);
    println!(
        "  exact {} ({:.1}%)  alias {} ({:.1}%)  WRONG {} ({:.1}%)  stalled {} ({:.1}%)  refused-unit {} ({:.1}%)  refused-anchors {} ({:.1}%)",
        grand.exact,
        pct(grand.exact, grand.n),
        grand.alias,
        pct(grand.alias, grand.n),
        grand.wrong,
        pct(grand.wrong, grand.n),
        grand.stalled,
        pct(grand.stalled, grand.n),
        grand.refused_unit,
        pct(grand.refused_unit, grand.n),
        grand.refused_disagree,
        pct(grand.refused_disagree, grand.n)
    );
    println!(
        "  converged {} of {} ({:.1}%); of those bits {:.1}%  signed {:.1}%  order {:.1}%  stride {:.1}%  fs {:.1}% correct",
        grand.conv,
        grand.n,
        pct(grand.conv, grand.n),
        pct(grand.bits_ok, grand.conv),
        pct(grand.signed_ok, grand.conv),
        pct(grand.order_ok, grand.conv),
        pct(grand.stride_ok, grand.conv),
        pct(grand.fs_ok, grand.conv)
    );
    println!(
        "  wall clock per {WINDOWS}-window sweep: min {:.0}  p25 {:.0}  median {:.0}  p75 {:.0}  p90 {:.0}  max {:.0} ms",
        quantile(&all_ms, 0.0),
        quantile(&all_ms, 0.25),
        quantile(&all_ms, 0.5),
        quantile(&all_ms, 0.75),
        quantile(&all_ms, 0.9),
        quantile(&all_ms, 1.0)
    );
    if confusion.is_empty() {
        println!("  confusion: no wrong convergence in this grid");
    } else {
        println!("  confusion (truth -> what it said), count:");
        for (k, v) in &confusion {
            println!("    {v:>2}x  {k}");
        }
    }
}

#[test]
#[ignore = "ground truth over the whole corpus; run with --release"]
fn ground_truth_layout_grid() {
    run_grid("LAYOUT GRID", &layout_grid());
}

#[test]
#[ignore = "ground truth over the whole corpus; run with --release"]
fn ground_truth_rate_grid() {
    run_grid("RATE GRID", &rate_grid());
}

/// The negatives. Every convergence here is a false one and is printed with the parameters it chose.
#[test]
#[ignore = "ground truth negatives over the whole corpus; run with --release"]
fn ground_truth_negatives() {
    let fixtures = ecg_fixtures();
    let ppg = &ppg_fixtures()[0];
    let cfg = SweepConfig::default();
    let truths = vec![
        t(
            "W16 s LE dense 400 Hz",
            lay(16, true, BitOrder::LsbFirst, 0, 16),
            400.0,
            0.0,
            Coding::Signed,
        ),
        t(
            "W24 s LE dense hdr3 512 Hz",
            lay(24, true, BitOrder::LsbFirst, 24, 24),
            512.0,
            0.0,
            Coding::Signed,
        ),
        t(
            "W18 s BE dense hdr1 256 Hz",
            lay(18, true, BitOrder::MsbFirst, 8, 18),
            256.0,
            0.0,
            Coding::Signed,
        ),
    ];

    println!("\n================ NEGATIVES ================");
    println!("subjects {}  truth layouts {}  ppg source: subject {} at {} Hz", fixtures.len(), truths.len(), ppg.subject, ppg.fs_hz);
    let mut totals: BTreeMap<&str, (usize, usize)> = BTreeMap::new();
    let mut all_ms: Vec<f64> = Vec::new();

    for tr in &truths {
        for f in &fixtures {
            let beats = pulse_train(&detector_consensus(&f.samples, f.fs_hz), f.fs_hz, PTT_MS, PTT_JITTER_MS, 7);
            let period = {
                let b = detector_consensus(&f.samples, f.fs_hz);
                (b[b.len() - 1] - b[0]) as f64 / (b.len() - 1) as f64
            };
            let cases: Vec<(&str, Vec<f64>)> = vec![
                ("gaussian", gaussian_like(&f.samples, 11)),
                ("sawtooth", sawtooth_like(&f.samples, period)),
                ("constant", vec![f.samples[0]; f.samples.len()]),
                ("ppg-real", matched_to(&resample(&ppg.samples, ppg.fs_hz, f.fs_hz), &f.samples)),
                ("shuffled", shuffled(&f.samples, 13)),
            ];
            for (name, wave) in cases {
                let w = resample(&wave, f.fs_hz, tr.fs_hz);
                let bytes = encode_counts(&w, &tr.layout, tr.coding, None);
                let start = Instant::now();
                let r = sweep_split(&bytes, &beats, EPOCH_MS, WINDOWS, &cfg);
                all_ms.push(start.elapsed().as_secs_f64() * 1000.0);
                let e = totals.entry(name).or_default();
                e.1 += 1;
                if let SweepOutcome::Converged { shape, fs_hz, .. } = r.outcome {
                    e.0 += 1;
                    println!(
                        "  FALSE CONVERGENCE  {name}  subject {}  under {}  ->  {} at {fs_hz} Hz  (margin {:.3}, windows {}/{})",
                        f.subject,
                        tr.name,
                        short(&shape),
                        r.margin,
                        r.windows_agreed,
                        r.windows_required
                    );
                }
            }
        }
    }

    all_ms.sort_by(f64::total_cmp);
    println!("\n---- negatives: false convergences ----");
    for (k, (bad, n)) in &totals {
        println!("  {k:<10} {bad} of {n}  ({:.1}%)", pct(*bad, *n));
    }
    let bad: usize = totals.values().map(|v| v.0).sum();
    let n: usize = totals.values().map(|v| v.1).sum();
    println!("  TOTAL      {bad} of {n}  ({:.1}%)", pct(bad, n));
    println!(
        "  wall clock: median {:.0}  p90 {:.0}  max {:.0} ms",
        quantile(&all_ms, 0.5),
        quantile(&all_ms, 0.9),
        quantile(&all_ms, 1.0)
    );
}

/// Why the wrong convergences happen, measured rather than argued.
#[test]
#[ignore = "diagnosis of the confusion cases; run with --release"]
fn diagnose_confusions() {
    use physio_algo::ecg::sweep::sweep_window;
    let fixtures = ecg_fixtures();
    let cfg = SweepConfig::default();

    // (1) Can two leaderboard entries in one window ever share a class but differ in shape? That
    // conjunction is exactly what `alias_shapes` filters on.
    let tr = t("W32 s LE dense", lay(32, true, BitOrder::LsbFirst, 0, 32), 400.0, 0.0, Coding::Signed);
    let mut pairs_same_class_diff_shape = 0usize;
    let mut entries = 0usize;
    let mut windows = 0usize;
    for f in &fixtures {
        let (bytes, beats) = stream(f, &tr);
        let chunk = bytes.len() / WINDOWS;
        let w = sweep_window(&bytes[..chunk], &beats, &cfg);
        windows += 1;
        entries += w.leaderboard.len();
        for a in &w.leaderboard {
            for b in &w.leaderboard {
                if a.class == b.class && a.layout.shape() != b.layout.shape() {
                    pairs_same_class_diff_shape += 1;
                }
            }
        }
    }
    println!(
        "\n[1] alias_shapes source: {windows} windows, {entries} leaderboard entries, \
         {pairs_same_class_diff_shape} same-class-different-shape pairs"
    );

    // (2) What is in the leader's class for the 32-bit truth, and does the truth width sit there?
    let f = &fixtures[0];
    for tr in [
        t("W32 s LE dense", lay(32, true, BitOrder::LsbFirst, 0, 32), 400.0, 0.0, Coding::Signed),
        t("W24 s BE dense", lay(24, true, BitOrder::MsbFirst, 0, 24), 400.0, 0.0, Coding::Signed),
    ] {
        let (bytes, beats) = stream(f, &tr);
        let chunk = bytes.len() / WINDOWS;
        let w = sweep_window(&bytes[..chunk], &beats, &cfg);
        println!("\n[2] {} subject {}: leaderboard", tr.name, f.subject);
        for c in &w.leaderboard {
            println!(
                "    cls{} ans{} aliases{} {:>6.1} Hz q{:.4} pass{} | {} start{}  {}",
                c.class,
                c.answer,
                c.aliases,
                c.fs_hz,
                c.quality,
                u8::from(c.passes),
                short(&c.layout.shape()),
                c.layout.start_bit,
                if c.layout.shape() == tr.layout.shape() { "<= TRUTH WIDTH" } else { "" }
            );
        }
        // Is the true reading rule even in the surviving set, and how does its roughness compare?
        let truth_x = tr.layout.decode(&bytes[..chunk]);
        let ts = physio_algo::ecg::sweep::layout_stats(&truth_x).unwrap();
        let rep = w.leaderboard.first().unwrap();
        let rs = physio_algo::ecg::sweep::layout_stats(&rep.layout.decode(&bytes[..chunk])).unwrap();
        println!(
            "    truth roughness {:.9} vs reported {:.9}   (relative gap {:.3e}, representative tolerance 1e-4)",
            ts.roughness,
            rs.roughness,
            (rs.roughness - ts.roughness).abs() / ts.roughness.max(1e-30)
        );
    }

    // (3) Subject 02 never converges under any layout. Which index refuses it?
    let tr = t("W16 s LE dense", lay(16, true, BitOrder::LsbFirst, 0, 16), 400.0, 0.0, Coding::Signed);
    println!("\n[3] per-subject index picture at the TRUE layout and rate");
    for f in &fixtures {
        let (bytes, beats) = stream(f, &tr);
        let chunk = bytes.len() / WINDOWS;
        let w = sweep_window(&bytes[..chunk], &beats, &cfg);
        let truth = w
            .leaderboard
            .iter()
            .find(|c| c.layout.shape() == tr.layout.shape() && c.fs_hz == tr.fs_hz);
        match truth {
            Some(c) => println!(
                "    {}: pass{} b{:.2}/x{:.2} k{:?} p{:?} tmpl{:?} hr{:?}  verdict {:?}",
                f.subject,
                u8::from(c.passes),
                c.ecg.b_sqi,
                c.ecg.b_excess,
                c.ecg.k_sqi.map(|v| (v * 10.0).round() / 10.0),
                c.ecg.p_sqi.map(|v| (v * 1000.0).round() / 1000.0),
                c.ecg.template_sqi.map(|v| (v * 1000.0).round() / 1000.0),
                c.ecg.mean_hr_bpm.map(|v| v.round()),
                c.ecg.verdict
            ),
            None => println!("    {}: the true layout+rate is not on the leaderboard at all", f.subject),
        }
    }

    // (4) The 3-channel interleave stalls on every subject. Where does the truth die?
    let tr = t("W24 s LE 3-ch start24", lay(24, true, BitOrder::LsbFirst, 24, 72), 400.0, 0.0, Coding::Signed);
    println!("\n[4] 3-channel interleave, truth on the leaderboard?");
    for f in fixtures.iter().take(4) {
        let (bytes, beats) = stream(f, &tr);
        let chunk = bytes.len() / WINDOWS;
        let w = sweep_window(&bytes[..chunk], &beats, &cfg);
        let truth = w.leaderboard.iter().find(|c| c.layout.shape() == tr.layout.shape() && c.fs_hz == tr.fs_hz);
        println!(
            "    {}: survived {} classes {} scored {}  truth-on-board {}  leader {} @{} Hz pass{} q{:.3}  margin {:.3}",
            f.subject,
            w.layouts_survived,
            w.classes,
            w.scored,
            truth.map_or("no".into(), |c| format!("yes pass{} q{:.3}", u8::from(c.passes), c.quality)),
            short(&w.leaderboard[0].layout.shape()),
            w.leaderboard[0].fs_hz,
            u8::from(w.leaderboard[0].passes),
            w.leaderboard[0].quality,
            w.margin
        );
    }
}

/// What "NoCandidatePassed" actually means when it is reported: did nothing pass, or did the windows
/// disagree about what passed?
#[test]
#[ignore = "diagnosis of the stall reasons; run with --release"]
fn diagnose_stalls() {
    use physio_algo::ecg::sweep::sweep_window;
    let fixtures = ecg_fixtures();
    let cfg = SweepConfig::default();
    let cases = vec![
        t("W24 s LE 3-ch start24", lay(24, true, BitOrder::LsbFirst, 24, 72), 400.0, 0.0, Coding::Signed),
        t("W16 s LE dense start0", lay(16, true, BitOrder::LsbFirst, 0, 16), 400.0, 0.0, Coding::Signed),
        t("W16 s BE 2-ch start32", lay(16, true, BitOrder::MsbFirst, 32, 32), 400.0, 0.0, Coding::Signed),
    ];
    let (mut stalls, mut every_window_passed, mut keys_disagreed) = (0usize, 0usize, 0usize);
    for tr in &cases {
        println!("\n[5] {}", tr.name);
        for f in &fixtures {
            let (bytes, beats) = stream(f, tr);
            let r = sweep_split(&bytes, &beats, EPOCH_MS, WINDOWS, &cfg);
            let stalled = matches!(&r.outcome, SweepOutcome::Searching { .. });
            if !stalled {
                continue;
            }
            stalls += 1;
            // Re-cut the same way sweep_split does and read each window's first passing candidate.
            let chunk = bytes.len() / WINDOWS;
            let per: Vec<Option<(String, usize, f64)>> = (0..WINDOWS)
                .map(|k| {
                    let b0 = k * chunk;
                    let t0 = EPOCH_MS * k as f64 / WINDOWS as f64;
                    let t1 = EPOCH_MS * (k + 1) as f64 / WINDOWS as f64;
                    let bb: Vec<f64> = beats.iter().filter(|t| **t >= t0 && **t < t1).map(|t| t - t0).collect();
                    let w = sweep_window(&bytes[b0..b0 + chunk], &bb, &cfg);
                    w.leaderboard
                        .iter()
                        .find(|c| c.passes)
                        .map(|c| (short(&c.layout.shape()), c.layout.start_bit, c.fs_hz))
                })
                .collect();
            let all_passed = per.iter().all(|p| p.is_some());
            let distinct: std::collections::BTreeSet<String> =
                per.iter().flatten().map(|(s, _, fs)| format!("{s}@{fs}")).collect();
            if all_passed {
                every_window_passed += 1;
                if distinct.len() > 1 {
                    keys_disagreed += 1;
                }
            }
            let reason = match &r.outcome {
                SweepOutcome::Searching { reason, .. } => format!("{reason:?}"),
                _ => "-".into(),
            };
            println!(
                "    {}: reported {reason:<20} | per-window first-passing: {:?}",
                f.subject,
                per.iter()
                    .map(|p| p.as_ref().map_or("NONE".to_string(), |(s, sb, fs)| format!("{s}@{fs}(start{sb})")))
                    .collect::<Vec<_>>()
            );
        }
    }
    println!(
        "\n[5] summary: {stalls} stalls; in {every_window_passed} of them EVERY window had a passing \
         candidate, and in {keys_disagreed} of those the windows named DIFFERENT shapes/rates"
    );
}

/// Which gate each negative dies at, and by how much. A pipeline that rejects everything says nothing
/// about whether each threshold has ever rejected anything.
#[test]
#[ignore = "per-threshold rejection audit; run with --release"]
fn diagnose_negative_rejection_path() {
    use physio_algo::ecg::score::score;
    use physio_algo::ecg::sweep::layout_stats;
    let fixtures = ecg_fixtures();
    let ppg = &ppg_fixtures()[0];
    let cfg = SweepConfig::default();
    let tr = t("W16 s LE dense 400 Hz", lay(16, true, BitOrder::LsbFirst, 0, 16), 400.0, 0.0, Coding::Signed);

    let mut rough: BTreeMap<&str, Vec<f64>> = BTreeMap::new();
    let mut kurt: BTreeMap<&str, Vec<f64>> = BTreeMap::new();
    let mut survived: BTreeMap<&str, usize> = BTreeMap::new();
    let mut fails: BTreeMap<&str, [usize; 5]> = BTreeMap::new();
    let mut n: BTreeMap<&str, usize> = BTreeMap::new();

    for f in &fixtures {
        let period = {
            let b = detector_consensus(&f.samples, f.fs_hz);
            (b[b.len() - 1] - b[0]) as f64 / (b.len() - 1) as f64
        };
        let cases: Vec<(&str, Vec<f64>)> = vec![
            ("real-ecg", f.samples.clone()),
            ("gaussian", gaussian_like(&f.samples, 11)),
            ("sawtooth", sawtooth_like(&f.samples, period)),
            ("constant", vec![f.samples[0]; f.samples.len()]),
            ("ppg-real", matched_to(&resample(&ppg.samples, ppg.fs_hz, f.fs_hz), &f.samples)),
            ("shuffled", shuffled(&f.samples, 13)),
        ];
        for (name, wave) in cases {
            *n.entry(name).or_default() += 1;
            let w = resample(&wave, f.fs_hz, tr.fs_hz);
            let bytes = encode_counts(&w, &tr.layout, tr.coding, None);
            let x = tr.layout.decode(&bytes);
            let Some(s) = layout_stats(&x) else {
                rough.entry(name).or_default().push(f64::NAN);
                kurt.entry(name).or_default().push(f64::NAN);
                continue;
            };
            rough.entry(name).or_default().push(s.roughness);
            kurt.entry(name).or_default().push(s.kurtosis);
            if s.roughness > cfg.max_roughness || s.kurtosis < cfg.min_kurtosis {
                continue;
            }
            *survived.entry(name).or_default() += 1;
            let sc = score(&x, tr.fs_hz);
            let v = sc.verdict;
            let e = fails.entry(name).or_default();
            e[0] += usize::from(!v.b_ok);
            e[1] += usize::from(!v.k_ok);
            e[2] += usize::from(!v.p_ok);
            e[3] += usize::from(!v.template_ok);
            e[4] += usize::from(!v.hr_ok);
        }
    }

    println!("\n[6] rejection path at the TRUE layout and rate, {} subjects each", fixtures.len());
    println!("    (prune: roughness <= {:.2}, kurtosis >= {:.2})", cfg.max_roughness, cfg.min_kurtosis);
    for name in ["real-ecg", "gaussian", "sawtooth", "constant", "ppg-real", "shuffled"] {
        let r: Vec<f64> = rough[name].iter().copied().filter(|v| v.is_finite()).collect();
        let k: Vec<f64> = kurt[name].iter().copied().filter(|v| v.is_finite()).collect();
        let sv = survived.get(name).copied().unwrap_or(0);
        let z = [0usize; 5];
        let e = fails.get(name).unwrap_or(&z);
        println!(
            "    {name:<9} n={:<3} roughness {:>7.4}..{:>7.4}  kurtosis {:>7.2}..{:>7.2}  survived prune {sv:>2}  \
             then refused by b{} k{} p{} tmpl{} hr{}",
            n[name],
            r.iter().copied().fold(f64::INFINITY, f64::min),
            r.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            k.iter().copied().fold(f64::INFINITY, f64::min),
            k.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            e[0], e[1], e[2], e[3], e[4]
        );
    }
}

/// Is the "constant" negative actually constant by the time it is decoded?
#[test]
#[ignore = "harness self-check; run with --release"]
fn diagnose_constant_negative() {
    let f = &ecg_fixtures()[0];
    let tr = t("W16 s LE dense 400 Hz", lay(16, true, BitOrder::LsbFirst, 0, 16), 400.0, 0.0, Coding::Signed);
    let flat = vec![f.samples[0]; f.samples.len()];
    println!("\n[7] source constant value {:.6}, {} samples, distinct source values {}",
        f.samples[0], flat.len(), flat.iter().map(|v| v.to_bits()).collect::<std::collections::BTreeSet<_>>().len());
    let w = resample(&flat, f.fs_hz, tr.fs_hz);
    let m = mean(&w);
    let span = w.iter().map(|v| (v - m).abs()).fold(0.0f64, f64::max);
    println!("    after resample: {} samples, mean {:.17}, max |v-mean| {:.3e}  (span floor is 1e-12)", w.len(), m, span);
    let counts = quantise(&w, 16, Coding::Signed);
    let lo = counts.iter().copied().min().unwrap();
    let hi = counts.iter().copied().max().unwrap();
    println!("    quantised counts: min {lo}  max {hi}  distinct {}", counts.iter().collect::<std::collections::BTreeSet<_>>().len());
    let bytes = encode_counts(&w, &tr.layout, tr.coding, None);
    let x = tr.layout.decode(&bytes);
    let nz = x.iter().filter(|v| **v != 0.0).count();
    println!("    decoded: {} samples, {} non-zero, min {:.0} max {:.0}", x.len(), nz,
        x.iter().copied().fold(f64::INFINITY, f64::min), x.iter().copied().fold(f64::NEG_INFINITY, f64::max));
}
