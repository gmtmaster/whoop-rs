//! Sweep instrument for the irregular-rhythm screen's six candidate constants, over the RECORD-SPLIT
//! full PhysioNet R-R corpus in `whoop-data/harnesses/rhythm-full/`.
//!
//! It changes NOTHING. `screen.rs` and `cosen.rs` are read-only here: this file carries a parameterised
//! REPLAY of `screen()` and proves the replay reproduces the shipped function exactly, segment for
//! segment, before any constant is moved. A sweep against a replay that had drifted would measure the
//! replay.
//!
//! ```text
//! cargo run --release -p physio-algo --example rr_screen_sweep -- <split-dir> <mode>
//! ```
//!
//! `<split-dir>` is `.../rhythm-full/train` or `.../rhythm-full/test`. Modes: `verify`, `baseline`,
//! `sweep1`, `joint`, `point`, `records`.
//!
//! **Wellness estimate, never medical or diagnostic.** Every rate below is the share of stretches this
//! screen would report as irregular. It is a screen for irregularity, not a diagnosis of any condition,
//! and the corpus is chest-lead Holter ECG rather than wrist PPG.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use physio_algo::hrv::HrvReadiness;
use physio_algo::rr_irregularity::cosen::{cosen_with, COSEN_M};
use physio_algo::rr_irregularity::{
    assess, quality,
    screen::{self, ScreenState},
    IrregularityReading,
};
use physio_algo::stats::median;

/// Seconds a beat in the synthetic clock, the same convention `tests/rr_irregularity_rhythm.rs` uses so
/// that every input-quality gate is inert and only the rhythm logic is under test.
const STAMP_STEP_S: u32 = 3;

/// Beats in one corpus segment. Fixed by the emitter, and it bounds what a sweep can reach: no episode
/// is possible from a segment shorter than `w + (min_windows - 1) * s`.
const SEGMENT_BEATS: usize = 256;

#[derive(Clone)]
struct Segment {
    class: String,
    record: String,
    start_beat: usize,
    rr: Vec<u16>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Params {
    w: usize,
    s: usize,
    r_ms: f64,
    cosen_floor: f64,
    resid_floor: f64,
    min_windows: usize,
    min_assessed: f64,
}

/// The shipped operating point, read off `screen.rs` / `cosen.rs`. Asserted against the real constants
/// in `verify` so a drift in either direction is loud.
const SHIPPED: Params = Params {
    w: 32,
    s: 8,
    r_ms: 30.0,
    cosen_floor: -1.28,
    resid_floor: -1.10,
    min_windows: 24,
    min_assessed: 0.50,
};

/// The two arms the shipped gate scores, and the ceiling or floor each carries.
const POSITIVE: [(&str, f64); 2] = [("AfibAudited", 0.55), ("AfibMachine", 0.50)];
const NEGATIVE: [(&str, f64); 7] = [
    ("SinusNsr", 0.02),
    ("SinusAfdb", 0.02),
    ("SinusMit", 0.02),
    ("EctopyLight", 0.10),
    ("EctopyHeavy", 0.10),
    ("EctopyBigeminal", 0.10),
    // Not a "specificity" class in the gate's own words, but it carries a hard ceiling of its own in
    // `flutter_is_missed_and_that_is_recorded_here_rather_than_hidden`, so a sweep must respect it.
    ("Flutter", 0.15),
];

fn main() {
    let mut args = std::env::args().skip(1);
    let dir = PathBuf::from(args.next().expect("split dir"));
    let mode = args.next().unwrap_or_else(|| "baseline".to_string());
    let segs = load(&dir);
    eprintln!("loaded {} segments from {}", segs.len(), dir.display());
    match mode.as_str() {
        "verify" => verify(&segs),
        "baseline" => baseline(&segs),
        "sweep1" => sweep_one_at_a_time(&segs),
        "joint" => joint(&segs),
        "point" => point(&segs, &args.collect::<Vec<String>>()),
        "records" => records(&segs),
        "recordeval" => recordeval(&segs, &args.collect::<Vec<String>>()),
        "recordsweep" => recordsweep(&segs),
        "recordjoint" => recordjoint(&segs),
        "jackknife" => jackknife(&segs, &args.collect::<Vec<String>>()),
        "removal" => removal(&segs, &args.collect::<Vec<String>>()),
        other => panic!("unknown mode {other}"),
    }
}

// ---------------------------------------------------------------------------------------------
// corpus
// ---------------------------------------------------------------------------------------------

fn load(dir: &Path) -> Vec<Segment> {
    let mut paths: Vec<PathBuf> = fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "txt"))
        .collect();
    paths.sort();
    assert!(!paths.is_empty(), "no .txt in {}", dir.display());
    let mut out = Vec::new();
    for path in &paths {
        let text = fs::read_to_string(path).unwrap();
        let mut class = String::new();
        for line in text.lines() {
            if let Some(rest) = line.strip_prefix('#') {
                let mut it = rest.split_whitespace();
                if let (Some("class"), Some(v)) = (it.next(), it.next()) {
                    class = v.to_string();
                }
                continue;
            }
            let mut it = line.split_whitespace();
            let (Some(record), Some(start), Some(_prem)) = (it.next(), it.next(), it.next()) else {
                continue;
            };
            let rr: Vec<u16> = it.map(|v| v.parse().unwrap()).collect();
            assert_eq!(rr.len(), SEGMENT_BEATS, "segment length changed in {}", path.display());
            out.push(Segment {
                class: class.clone(),
                record: record.to_string(),
                start_beat: start.parse().unwrap(),
                rr,
            });
        }
    }
    out
}

fn stamp(rr: &[u16]) -> Vec<(u32, u16)> {
    rr.iter().enumerate().map(|(i, &v)| (i as u32 * STAMP_STEP_S, v)).collect()
}

// ---------------------------------------------------------------------------------------------
// the parameterised replay
// ---------------------------------------------------------------------------------------------

/// One window's inputs, everything that does NOT depend on the swept `r_ms` or on any threshold.
struct Win {
    assessed: bool,
    contiguous: bool,
    /// Range-filter survivors, which is what `assess` computes every index over.
    ranged: Vec<u16>,
    /// Malik survivors of `ranged` — the residual view the ectopy veto is taken on.
    clean: Vec<u16>,
}

/// Windows of one segment at a given `(w, s)`, plus the beat count so the calibration floor is checkable.
struct SegWins {
    wins: Vec<Win>,
    beats: usize,
}

fn window_cache(segs: &[Segment], w: usize, s: usize) -> Vec<SegWins> {
    segs.iter()
        .map(|seg| {
            let beats = stamp(&seg.rr);
            let mut wins = Vec::new();
            let mut start = 0usize;
            while start + w <= beats.len() {
                let slice = &beats[start..start + w];
                let assessed = matches!(assess(slice), IrregularityReading::Assessed(_));
                let contiguous = slice
                    .windows(2)
                    .all(|p| p[1].0.saturating_sub(p[0].0) <= screen::MAX_BEAT_GAP_S);
                let ranged = quality::ranged(slice);
                let clean = HrvReadiness::clean_rr(&ranged);
                wins.push(Win { assessed, contiguous, ranged, clean });
                start += s;
            }
            SegWins { wins, beats: beats.len() }
        })
        .collect()
}

/// `(cosen, residual_cosen)` per window at one tolerance. The only layer a change of `r_ms` touches.
fn cosen_layer(cache: &[SegWins], r_ms: f64) -> Vec<Vec<(Option<f64>, Option<f64>)>> {
    cache
        .iter()
        .map(|sw| {
            sw.wins
                .iter()
                .map(|win| {
                    if !win.assessed {
                        return (None, None);
                    }
                    (cosen_with(&win.ranged, COSEN_M, r_ms), cosen_with(&win.clean, COSEN_M, r_ms))
                })
                .collect()
        })
        .collect()
}

/// The `[start, end)` window spans the replay would REPORT as episodes. Mirrors `screen()` line for
/// line; `reported` is just "is this list non-empty".
fn episode_spans(sw: &SegWins, layer: &[(Option<f64>, Option<f64>)], p: &Params) -> Vec<(usize, usize)> {
    let min_screen_beats = p.w + (p.min_windows - 1) * p.s;
    if sw.beats < min_screen_beats {
        return Vec::new(); // Calibrating
    }
    let n = sw.wins.len();
    let assessed = sw.wins.iter().filter(|x| x.assessed).count();
    if (assessed as f64) < p.min_assessed * n as f64 {
        return Vec::new(); // Inconclusive / PoorInputQuality
    }
    let irregular: Vec<bool> = sw
        .wins
        .iter()
        .zip(layer)
        .map(|(win, (c, _))| win.contiguous && win.assessed && c.is_some_and(|c| c >= p.cosen_floor))
        .collect();
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < n {
        if !irregular[i] {
            i += 1;
            continue;
        }
        let start = i;
        while i < n && irregular[i] {
            i += 1;
        }
        if i - start < p.min_windows {
            continue;
        }
        let resid: Vec<f64> = layer[start..i].iter().filter_map(|(_, r)| *r).collect();
        if resid.is_empty() {
            continue; // `episode()` returns None: no residual view, so ectopy is not ruled out
        }
        if median(&resid) >= p.resid_floor {
            out.push((start, i));
        }
    }
    out
}

fn reported(sw: &SegWins, layer: &[(Option<f64>, Option<f64>)], p: &Params) -> bool {
    !episode_spans(sw, layer, p).is_empty()
}

// ---------------------------------------------------------------------------------------------
// scoring
// ---------------------------------------------------------------------------------------------

/// `class -> (reported, n)` for one parameter point.
type Rates = BTreeMap<String, (usize, usize)>;

fn rates(segs: &[Segment], cache: &[SegWins], layer: &[Vec<(Option<f64>, Option<f64>)>], p: &Params) -> Rates {
    let mut out: Rates = BTreeMap::new();
    for (i, seg) in segs.iter().enumerate() {
        let e = out.entry(seg.class.clone()).or_default();
        e.1 += 1;
        e.0 += usize::from(reported(&cache[i], &layer[i], p));
    }
    out
}

fn share(r: &Rates, c: &str) -> f64 {
    r.get(c).map_or(f64::NAN, |&(h, n)| if n == 0 { f64::NAN } else { h as f64 / n as f64 })
}

/// Pooled fibrillation sensitivity over both positive arms — the headline number.
fn sensitivity(r: &Rates) -> f64 {
    let (mut h, mut n) = (0usize, 0usize);
    for (c, _) in POSITIVE {
        if let Some(&(hit, tot)) = r.get(c) {
            h += hit;
            n += tot;
        }
    }
    if n == 0 { f64::NAN } else { h as f64 / n as f64 }
}

/// Every breach of the shipped gate's floors and ceilings, in the gate's own thresholds.
fn breaches(r: &Rates) -> Vec<String> {
    let mut out = Vec::new();
    for (c, floor) in POSITIVE {
        let v = share(r, c);
        if v.is_finite() && v < floor {
            out.push(format!("{c} {v:.4} < {floor:.2}"));
        }
    }
    for (c, ceil) in NEGATIVE {
        let v = share(r, c);
        if v.is_finite() && v > ceil {
            out.push(format!("{c} {v:.4} > {ceil:.2}"));
        }
    }
    out
}

fn line(tag: &str, r: &Rates) -> String {
    let mut s = format!("{tag}\tsens {:.4}", sensitivity(r));
    for (c, _) in POSITIVE {
        s += &format!("\t{c} {:.4}", share(r, c));
    }
    for (c, _) in NEGATIVE {
        s += &format!("\t{c} {:.4}", share(r, c));
    }
    let b = breaches(r);
    s += &format!("\t{}", if b.is_empty() { "GATE-PASS".to_string() } else { format!("BREACH[{}]", b.join("; ")) });
    s
}

// ---------------------------------------------------------------------------------------------
// modes
// ---------------------------------------------------------------------------------------------

/// The replay must reproduce the shipped `screen()` exactly, on every segment, before anything is swept.
fn verify(segs: &[Segment]) {
    assert_eq!(SHIPPED.w, screen::WINDOW_BEATS);
    assert_eq!(SHIPPED.s, screen::STEP_BEATS);
    assert_eq!(SHIPPED.min_windows, screen::MIN_EPISODE_WINDOWS);
    assert_eq!(SHIPPED.cosen_floor, screen::COSEN_IRREGULAR_FLOOR);
    assert_eq!(SHIPPED.resid_floor, screen::EPISODE_RESIDUAL_COSEN_FLOOR);
    assert_eq!(SHIPPED.min_assessed, screen::MIN_ASSESSED_FRACTION);
    assert_eq!(SHIPPED.r_ms, physio_algo::rr_irregularity::COSEN_R_MS);
    let cache = window_cache(segs, SHIPPED.w, SHIPPED.s);
    let layer = cosen_layer(&cache, SHIPPED.r_ms);
    let mut mismatch = 0usize;
    for (i, seg) in segs.iter().enumerate() {
        let real = matches!(screen::screen(&stamp(&seg.rr)), ScreenState::IrregularEpisodes { .. });
        let replayed = reported(&cache[i], &layer[i], &SHIPPED);
        if real != replayed {
            mismatch += 1;
            if mismatch <= 5 {
                println!("MISMATCH {} {} start {} real {real} replay {replayed}", seg.class, seg.record, seg.start_beat);
            }
        }
    }
    println!("replay vs shipped screen(): {} segments, {mismatch} mismatches", segs.len());
    assert_eq!(mismatch, 0, "the replay does not reproduce the shipped screen");
    println!("{}", line("SHIPPED", &rates(segs, &cache, &layer, &SHIPPED)));
}

fn baseline(segs: &[Segment]) {
    let cache = window_cache(segs, SHIPPED.w, SHIPPED.s);
    let layer = cosen_layer(&cache, SHIPPED.r_ms);
    let r = rates(segs, &cache, &layer, &SHIPPED);
    println!("{}", line("SHIPPED", &r));
    println!("\nper class, reported of total:");
    for (c, (h, n)) in &r {
        println!("  {c:<18} {h:>5} of {n:>5}   {:.4}", *h as f64 / *n as f64);
    }
}

/// Every candidate walked on its own, wide enough that a cliff cannot be read as a plateau.
fn sweep_one_at_a_time(segs: &[Segment]) {
    // --- thresholds: one cache, one layer, many points ---
    let cache = window_cache(segs, SHIPPED.w, SHIPPED.s);
    let layer = cosen_layer(&cache, SHIPPED.r_ms);

    println!("\n##### COSEN_IRREGULAR_FLOOR (shipped -1.28) #####");
    let mut v = -1.80f64;
    while v <= -0.60 + 1e-9 {
        let p = Params { cosen_floor: round2(v), ..SHIPPED };
        println!("{}", line(&format!("cosen_floor={:.2}", p.cosen_floor), &rates(segs, &cache, &layer, &p)));
        v += 0.04;
    }

    println!("\n##### EPISODE_RESIDUAL_COSEN_FLOOR (shipped -1.10) #####");
    let mut v = -2.20f64;
    while v <= -0.40 + 1e-9 {
        let p = Params { resid_floor: round2(v), ..SHIPPED };
        println!("{}", line(&format!("resid_floor={:.2}", p.resid_floor), &rates(segs, &cache, &layer, &p)));
        v += 0.05;
    }

    println!("\n##### MIN_EPISODE_WINDOWS (shipped 24) #####");
    for m in 4..=32usize {
        let p = Params { min_windows: m, ..SHIPPED };
        println!("{}", line(&format!("min_windows={m}"), &rates(segs, &cache, &layer, &p)));
    }

    println!("\n##### COSEN_R_MS (shipped 30.0) #####");
    for r10 in (50..=1200).step_by(50) {
        let r_ms = f64::from(r10) / 10.0;
        let lay = cosen_layer(&cache, r_ms);
        let p = Params { r_ms, ..SHIPPED };
        println!("{}", line(&format!("cosen_r_ms={r_ms:.1}"), &rates(segs, &cache, &lay, &p)));
    }

    println!("\n##### WINDOW_BEATS (shipped 32) #####");
    for w in (14..=72).step_by(2) {
        let c = window_cache(segs, w, SHIPPED.s);
        let lay = cosen_layer(&c, SHIPPED.r_ms);
        let p = Params { w, ..SHIPPED };
        println!("{}", line(&format!("window_beats={w}"), &rates(segs, &c, &lay, &p)));
    }

    println!("\n##### STEP_BEATS (shipped 8) #####");
    for s in 1..=16usize {
        let c = window_cache(segs, SHIPPED.w, s);
        let lay = cosen_layer(&c, SHIPPED.r_ms);
        let p = Params { s, ..SHIPPED };
        println!("{}", line(&format!("step_beats={s}"), &rates(segs, &c, &lay, &p)));
    }
}

/// The coupled trio swept together, with the two floors moved alongside them. A gain that only survives
/// one-at-a-time is an artefact of the search order.
fn joint(segs: &[Segment]) {
    println!("\n##### JOINT: window x step x min_windows, thresholds shipped #####");
    for w in [24usize, 28, 32, 36, 40, 48] {
        for s in [4usize, 6, 8, 10, 12] {
            let c = window_cache(segs, w, s);
            let lay = cosen_layer(&c, SHIPPED.r_ms);
            for m in [12usize, 16, 20, 24, 28, 32, 40, 48] {
                let p = Params { w, s, min_windows: m, ..SHIPPED };
                // Reject any point whose calibration floor is longer than a corpus segment: it cannot
                // report anything at all and its 0.0000 is an artefact of the fixture, not a result.
                let feasible = w + (m - 1) * s <= SEGMENT_BEATS;
                println!(
                    "{}\tfeasible {feasible}\tmin_screen_beats {}",
                    line(&format!("w={w} s={s} m={m}"), &rates(segs, &c, &lay, &p)),
                    w + (m - 1) * s
                );
            }
        }
    }

    println!("\n##### JOINT: the two floors together, geometry shipped #####");
    let c = window_cache(segs, SHIPPED.w, SHIPPED.s);
    let lay = cosen_layer(&c, SHIPPED.r_ms);
    let mut cf = -1.60f64;
    while cf <= -0.90 + 1e-9 {
        let mut rf = -1.80f64;
        while rf <= -0.70 + 1e-9 {
            let p = Params { cosen_floor: round2(cf), resid_floor: round2(rf), ..SHIPPED };
            println!("{}", line(&format!("cf={:.2} rf={:.2}", p.cosen_floor, p.resid_floor), &rates(segs, &c, &lay, &p)));
            rf += 0.05;
        }
        cf += 0.05;
    }

    println!("\n##### JOINT: r_ms x cosen_floor (the tolerance shifts the whole COSEn scale) #####");
    for r10 in (150..=700).step_by(50) {
        let r_ms = f64::from(r10) / 10.0;
        let l = cosen_layer(&c, r_ms);
        let mut cf = -1.80f64;
        while cf <= -0.70 + 1e-9 {
            let p = Params { r_ms, cosen_floor: round2(cf), ..SHIPPED };
            println!("{}", line(&format!("r={r_ms:.1} cf={:.2}", p.cosen_floor), &rates(segs, &c, &l, &p)));
            cf += 0.05;
        }
    }

    println!("\n##### JOINT: r_ms x both floors x min_windows, a coarse full-factorial #####");
    for r10 in [200u32, 250, 300, 350, 400, 500] {
        let r_ms = f64::from(r10) / 10.0;
        let l = cosen_layer(&c, r_ms);
        for cfi in [-150i32, -140, -130, -128, -120, -110, -100] {
            for rfi in [-160i32, -140, -120, -110, -100, -90, -80] {
                for m in [20usize, 22, 24, 26, 28] {
                    let p = Params {
                        r_ms,
                        cosen_floor: f64::from(cfi) / 100.0,
                        resid_floor: f64::from(rfi) / 100.0,
                        min_windows: m,
                        ..SHIPPED
                    };
                    println!(
                        "{}",
                        line(
                            &format!("r={r_ms:.1} cf={:.2} rf={:.2} m={m}", p.cosen_floor, p.resid_floor),
                            &rates(segs, &c, &l, &p)
                        )
                    );
                }
            }
        }
    }
}

/// One named point, given as `w s r_ms cosen_floor resid_floor min_windows`.
fn point(segs: &[Segment], args: &[String]) {
    let p = Params {
        w: args[0].parse().unwrap(),
        s: args[1].parse().unwrap(),
        r_ms: args[2].parse().unwrap(),
        cosen_floor: args[3].parse().unwrap(),
        resid_floor: args[4].parse().unwrap(),
        min_windows: args[5].parse().unwrap(),
        min_assessed: SHIPPED.min_assessed,
    };
    let c = window_cache(segs, p.w, p.s);
    let l = cosen_layer(&c, p.r_ms);
    let r = rates(segs, &c, &l, &p);
    println!("{p:?}");
    println!("{}", line("POINT", &r));
    for (cl, (h, n)) in &r {
        println!("  {cl:<18} {h:>5} of {n:>5}   {:.4}", *h as f64 / *n as f64);
    }
    // And the shipped point on the same corpus, for a side-by-side that shares every other choice.
    let cs = window_cache(segs, SHIPPED.w, SHIPPED.s);
    let ls = cosen_layer(&cs, SHIPPED.r_ms);
    println!("{}", line("SHIPPED", &rates(segs, &cs, &ls, &SHIPPED)));
}

/// Per-RECORD reporting at the shipped point: how concentrated the positives are, which is what bounds
/// how much any sensitivity difference can be trusted.
fn records(segs: &[Segment]) {
    let cache = window_cache(segs, SHIPPED.w, SHIPPED.s);
    let layer = cosen_layer(&cache, SHIPPED.r_ms);
    let mut per: BTreeMap<(String, String), (usize, usize)> = BTreeMap::new();
    for (i, seg) in segs.iter().enumerate() {
        let e = per.entry((seg.class.clone(), seg.record.clone())).or_default();
        e.1 += 1;
        e.0 += usize::from(reported(&cache[i], &layer[i], &SHIPPED));
    }
    for ((class, record), (h, n)) in &per {
        println!("{class:<18} {record:<8} {h:>5} of {n:>5}   {:.4}", *h as f64 / *n as f64);
    }
}

fn round2(v: f64) -> f64 {
    (v * 100.0).round() / 100.0
}

// ---------------------------------------------------------------------------------------------
// protocol B: whole RECORDS, window level
// ---------------------------------------------------------------------------------------------
//
// Protocol A scores 256-beat stretches, as the shipped gate does, and cannot price the ectopy veto:
// at that length the duration floor already suppresses ectopy. This one rebuilds whole records from
// their consecutive segments and scores WINDOWS, the shape production meets.

/// One record: the whole beat series, and the class of each 256-beat segment in it.
struct Record {
    source_class_of_segment: Vec<String>,
    rr: Vec<u16>,
}

fn build_records(segs: &[Segment]) -> BTreeMap<String, Record> {
    let mut by: BTreeMap<String, Vec<&Segment>> = BTreeMap::new();
    for s in segs {
        by.entry(s.record.clone()).or_default().push(s);
    }
    by.into_iter()
        .map(|(name, mut list)| {
            list.sort_by_key(|s| s.start_beat);
            for (i, s) in list.iter().enumerate() {
                assert_eq!(s.start_beat, i * SEGMENT_BEATS, "record {name} is not contiguous");
            }
            let rr = list.iter().flat_map(|s| s.rr.iter().copied()).collect();
            let source_class_of_segment = list.iter().map(|s| s.class.clone()).collect();
            (name, Record { source_class_of_segment, rr })
        })
        .collect()
}

/// One record's windows at a fixed `(w, s)`, plus each window's class label.
struct RecWins {
    sw: SegWins,
    classes: Vec<String>,
}

fn record_cache(records: &BTreeMap<String, Record>, w: usize, s: usize) -> Vec<RecWins> {
    records
        .values()
        .map(|rec| {
            let beats = stamp(&rec.rr);
            let mut wins = Vec::new();
            let mut classes = Vec::new();
            let mut start = 0usize;
            while start + w <= beats.len() {
                let slice = &beats[start..start + w];
                let assessed = matches!(assess(slice), IrregularityReading::Assessed(_));
                let contiguous = slice
                    .windows(2)
                    .all(|q| q[1].0.saturating_sub(q[0].0) <= screen::MAX_BEAT_GAP_S);
                let ranged = quality::ranged(slice);
                let clean = HrvReadiness::clean_rr(&ranged);
                wins.push(Win { assessed, contiguous, ranged, clean });
                // A window's class is the class of the segment its FIRST beat falls in. Segments are
                // >=95 % pure for their class, so this is the label the corpus supports.
                classes.push(rec.source_class_of_segment[start / SEGMENT_BEATS].clone());
                start += s;
            }
            RecWins { sw: SegWins { wins, beats: beats.len() }, classes }
        })
        .collect()
}

fn record_layer(cache: &[RecWins], r_ms: f64) -> Vec<Vec<(Option<f64>, Option<f64>)>> {
    cache
        .iter()
        .map(|rw| {
            rw.sw
                .wins
                .iter()
                .map(|w| {
                    if !w.assessed {
                        (None, None)
                    } else {
                        (cosen_with(&w.ranged, COSEN_M, r_ms), cosen_with(&w.clean, COSEN_M, r_ms))
                    }
                })
                .collect()
        })
        .collect()
}

fn record_rates_cached(
    cache: &[RecWins],
    layer: &[Vec<(Option<f64>, Option<f64>)>],
    p: &Params,
) -> Rates {
    let mut out: Rates = BTreeMap::new();
    for (rw, lay) in cache.iter().zip(layer) {
        let spans = episode_spans(&rw.sw, lay, p);
        for (i, class) in rw.classes.iter().enumerate() {
            let e = out.entry(class.clone()).or_default();
            e.1 += 1;
            e.0 += usize::from(spans.iter().any(|&(a, b)| i >= a && i < b));
        }
    }
    out
}

/// Window-level rates over whole records at one parameter point.
fn record_rates(records: &BTreeMap<String, Record>, p: &Params) -> Rates {
    let cache = record_cache(records, p.w, p.s);
    let layer = record_layer(&cache, p.r_ms);
    record_rates_cached(&cache, &layer, p)
}

/// The surviving constant walked at RECORD level, then the coupled trio moved jointly around it.
fn recordsweep(segs: &[Segment]) {
    let records = build_records(segs);
    let cache = record_cache(&records, SHIPPED.w, SHIPPED.s);
    let layer = record_layer(&cache, SHIPPED.r_ms);
    let windows: usize = cache.iter().map(|c| c.sw.wins.len()).sum();
    println!("{} records, {windows} windows at w{} s{}", cache.len(), SHIPPED.w, SHIPPED.s);

    println!("\n##### RECORD LEVEL: COSEN_IRREGULAR_FLOOR walked (shipped -1.28) #####");
    let mut cf = -1.70f64;
    while cf <= -1.00 + 1e-9 {
        let p = Params { cosen_floor: round2(cf), ..SHIPPED };
        println!("{}", line(&format!("cf={:.2}", p.cosen_floor), &record_rates_cached(&cache, &layer, &p)));
        cf += 0.02;
    }

    println!("\n##### RECORD LEVEL: EPISODE_RESIDUAL_COSEN_FLOOR walked (shipped -1.10) #####");
    let mut rf = -1.60f64;
    while rf <= -0.80 + 1e-9 {
        let p = Params { resid_floor: round2(rf), ..SHIPPED };
        println!("{}", line(&format!("rf={:.2}", p.resid_floor), &record_rates_cached(&cache, &layer, &p)));
        rf += 0.05;
    }

    println!("\n##### RECORD LEVEL: MIN_EPISODE_WINDOWS walked (shipped 24) #####");
    for m in [8usize, 12, 16, 20, 22, 24, 26, 28, 32, 40, 48, 64] {
        let p = Params { min_windows: m, ..SHIPPED };
        println!("{}", line(&format!("m={m}"), &record_rates_cached(&cache, &layer, &p)));
    }

    println!("\n##### RECORD LEVEL JOINT: cosen_floor x residual_floor #####");
    let mut cf = -1.60f64;
    while cf <= -1.00 + 1e-9 {
        let mut rf = -1.40f64;
        while rf <= -0.90 + 1e-9 {
            let p = Params { cosen_floor: round2(cf), resid_floor: round2(rf), ..SHIPPED };
            println!(
                "{}",
                line(&format!("cf={:.2} rf={:.2}", p.cosen_floor, p.resid_floor), &record_rates_cached(&cache, &layer, &p))
            );
            rf += 0.05;
        }
        cf += 0.04;
    }

    println!("\n##### RECORD LEVEL JOINT: cosen_floor x min_windows #####");
    for m in [16usize, 20, 24, 28, 32, 40] {
        let mut cf = -1.60f64;
        while cf <= -1.00 + 1e-9 {
            let p = Params { cosen_floor: round2(cf), min_windows: m, ..SHIPPED };
            println!(
                "{}",
                line(&format!("cf={:.2} m={m}", p.cosen_floor), &record_rates_cached(&cache, &layer, &p))
            );
            cf += 0.04;
        }
    }

    println!("\n##### RECORD LEVEL JOINT: r_ms x cosen_floor (does a wider tolerance ever pay?) #####");
    for r10 in [250u32, 300, 350, 400, 450, 500] {
        let r_ms = f64::from(r10) / 10.0;
        let l = record_layer(&cache, r_ms);
        let mut cf = -1.80f64;
        while cf <= -0.80 + 1e-9 {
            let p = Params { r_ms, cosen_floor: round2(cf), ..SHIPPED };
            println!(
                "{}",
                line(&format!("r={r_ms:.1} cf={:.2}", p.cosen_floor), &record_rates_cached(&cache, &l, &p))
            );
            cf += 0.05;
        }
    }
}

/// Per-record counts for one point: `record -> class -> (reported windows, windows)`.
fn per_record(
    names: &[String],
    cache: &[RecWins],
    layer: &[Vec<(Option<f64>, Option<f64>)>],
    p: &Params,
) -> BTreeMap<String, Rates> {
    let mut out: BTreeMap<String, Rates> = BTreeMap::new();
    for ((name, rw), lay) in names.iter().zip(cache).zip(layer) {
        let spans = episode_spans(&rw.sw, lay, p);
        let slot = out.entry(name.clone()).or_default();
        for (i, class) in rw.classes.iter().enumerate() {
            let e = slot.entry(class.clone()).or_default();
            e.1 += 1;
            e.0 += usize::from(spans.iter().any(|&(a, b)| i >= a && i < b));
        }
    }
    out
}

fn pool<'a>(it: impl Iterator<Item = &'a Rates>) -> Rates {
    let mut out: Rates = BTreeMap::new();
    for r in it {
        for (c, (h, n)) in r {
            let e = out.entry(c.clone()).or_default();
            e.0 += h;
            e.1 += n;
        }
    }
    out
}

/// Is a claimed sensitivity gain bigger than what one record moves? Leave-one-record-out, at record
/// level, on the candidate against the shipped point.
fn jackknife(segs: &[Segment], args: &[String]) {
    let cand = Params {
        w: args[0].parse().unwrap(),
        s: args[1].parse().unwrap(),
        r_ms: args[2].parse().unwrap(),
        cosen_floor: args[3].parse().unwrap(),
        resid_floor: args[4].parse().unwrap(),
        min_windows: args[5].parse().unwrap(),
        min_assessed: SHIPPED.min_assessed,
    };
    let records = build_records(segs);
    let names: Vec<String> = records.keys().cloned().collect();
    let cs = record_cache(&records, SHIPPED.w, SHIPPED.s);
    let ls = record_layer(&cs, SHIPPED.r_ms);
    let shipped = per_record(&names, &cs, &ls, &SHIPPED);
    let (cc, lc);
    let candidate = if (cand.w, cand.s) == (SHIPPED.w, SHIPPED.s) && cand.r_ms == SHIPPED.r_ms {
        per_record(&names, &cs, &ls, &cand)
    } else {
        cc = record_cache(&records, cand.w, cand.s);
        lc = record_layer(&cc, cand.r_ms);
        per_record(&names, &cc, &lc, &cand)
    };

    println!("candidate {cand:?}");
    println!("{}", line("SHIPPED  ", &pool(shipped.values())));
    println!("{}", line("CANDIDATE", &pool(candidate.values())));
    let base = sensitivity(&pool(shipped.values()));
    let cand_s = sensitivity(&pool(candidate.values()));
    println!("\npooled fibrillation sensitivity: shipped {base:.4}  candidate {cand_s:.4}  delta {:+.4}", cand_s - base);

    println!("\nper record, fibrillation-bearing only (shipped -> candidate):");
    let mut improved = 0usize;
    let mut worsened = 0usize;
    for name in &names {
        let sh = sensitivity(&shipped[name]);
        let cd = sensitivity(&candidate[name]);
        if !sh.is_finite() {
            continue;
        }
        let pos: usize = POSITIVE.iter().filter_map(|(c, _)| shipped[name].get(*c)).map(|&(_, n)| n).sum();
        if cd > sh {
            improved += 1;
        } else if cd < sh {
            worsened += 1;
        }
        println!("  {name:<8} {pos:>6} positive windows   {sh:.4} -> {cd:.4}   {:+.4}", cd - sh);
    }
    println!("  records improved {improved}, worsened {worsened}");

    println!("\nleave-one-record-out on the pooled delta:");
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    let mut worst = String::new();
    for drop in &names {
        let sh = sensitivity(&pool(names.iter().filter(|n| *n != drop).map(|n| &shipped[n])));
        let cd = sensitivity(&pool(names.iter().filter(|n| *n != drop).map(|n| &candidate[n])));
        let d = cd - sh;
        if d < lo {
            lo = d;
            worst = drop.clone();
        }
        hi = hi.max(d);
    }
    println!("  delta ranges {lo:+.4} to {hi:+.4} over the 63 leave-one-out folds (worst fold drops {worst})");

    println!("\nleave-one-record-out on the two ceilings that bind:");
    for class in ["EctopyBigeminal", "EctopyHeavy"] {
        for (label, m) in [("SHIPPED", &shipped), ("CANDIDATE", &candidate)] {
            let mut worst = 0.0f64;
            for drop in &names {
                let r = pool(names.iter().filter(|n| *n != drop).map(|n| &m[n]));
                let v = share(&r, class);
                if v.is_finite() {
                    worst = worst.max(v);
                }
            }
            let full = share(&pool(m.values()), class);
            println!("  {class:<16} {label:<10} full {full:.4}   worst leave-one-out {worst:.4}   ceiling 0.10");
        }
    }
}


/// The `the_ectopy_discriminator_separates_where_the_scatter_statistics_cannot` gate, replicated at an
/// arbitrary pair of floors: of the stretches clearing the irregularity floor, what share the veto removes.
fn removal(segs: &[Segment], args: &[String]) {
    let (cf, rf): (f64, f64) = (args[0].parse().unwrap(), args[1].parse().unwrap());
    let cache = window_cache(segs, SHIPPED.w, SHIPPED.s);
    let layer = cosen_layer(&cache, SHIPPED.r_ms);
    let mut out: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    for (i, seg) in segs.iter().enumerate() {
        let cs: Vec<f64> = layer[i].iter().filter_map(|(c, _)| *c).collect();
        let rs: Vec<f64> = layer[i].iter().filter_map(|(_, r)| *r).collect();
        if cs.is_empty() || median(&cs) < cf {
            continue;
        }
        let e = out.entry(seg.class.clone()).or_default();
        e.1 += 1;
        e.0 += usize::from(rs.is_empty() || median(&rs) < rf);
    }
    println!("floors cf={cf:.2} rf={rf:.2}");
    let sh = |c: &str| out.get(c).map_or(f64::NAN, |&(a, b)| a as f64 / b as f64);
    for (c, (a, b)) in &out {
        println!("  {c:<18} eligible {b:>4}, veto removed {a:>4}   {:.4}", *a as f64 / *b as f64);
    }
    let afib = sh("AfibMachine");
    println!("  GATE: bigeminal {:.4} > afib+0.25 {:.4} -> {}", sh("EctopyBigeminal"), afib + 0.25, sh("EctopyBigeminal") > afib + 0.25);
    println!("  GATE: heavy     {:.4} > afib+0.25 {:.4} -> {}", sh("EctopyHeavy"), afib + 0.25, sh("EctopyHeavy") > afib + 0.25);
    println!("  GATE: afib      {afib:.4} < 0.35 -> {}", afib < 0.35);
}

/// The coupled trio swept jointly at RECORD level: window, step and the episode length together.
fn recordjoint(segs: &[Segment]) {
    let records = build_records(segs);
    for w in [24usize, 28, 32, 36, 40, 48] {
        for s in [4usize, 6, 8, 10, 12] {
            let cache = record_cache(&records, w, s);
            let layer = record_layer(&cache, SHIPPED.r_ms);
            for m in [12usize, 16, 20, 24, 32, 48] {
                for cf in [-1.44f64, -1.36, -1.28, -1.20] {
                    let p = Params { w, s, min_windows: m, cosen_floor: cf, ..SHIPPED };
                    println!(
                        "{}\tepisode_beats {}",
                        line(&format!("w={w} s={s} m={m} cf={cf:.2}"), &record_rates_cached(&cache, &layer, &p)),
                        w + (m - 1) * s
                    );
                }
            }
        }
    }
}

/// Shipped against a named list of candidates, at RECORD level, plus the veto ablation that protocol A
/// is structurally unable to see.
fn recordeval(segs: &[Segment], args: &[String]) {
    let records = build_records(segs);
    let total: usize = records.values().map(|r| r.rr.len()).sum();
    println!("{} records, {total} beats", records.len());
    let mut points: Vec<(String, Params)> = vec![
        ("SHIPPED".to_string(), SHIPPED),
        // The ectopy veto switched off entirely: no residual COSEn can fall below this.
        ("VETO-OFF (rf=-9)".to_string(), Params { resid_floor: -9.0, ..SHIPPED }),
    ];
    for c in args.chunks(6) {
        if c.len() < 6 {
            break;
        }
        let p = Params {
            w: c[0].parse().unwrap(),
            s: c[1].parse().unwrap(),
            r_ms: c[2].parse().unwrap(),
            cosen_floor: c[3].parse().unwrap(),
            resid_floor: c[4].parse().unwrap(),
            min_windows: c[5].parse().unwrap(),
            min_assessed: SHIPPED.min_assessed,
        };
        points.push((format!("w{} s{} r{} cf{} rf{} m{}", p.w, p.s, p.r_ms, p.cosen_floor, p.resid_floor, p.min_windows), p));
    }
    for (name, p) in &points {
        let r = record_rates(&records, p);
        println!("{}", line(name, &r));
    }
}
