//! Verification instrument: run the SHIPPED `rr_irregularity::screen` over every corpus that landed and
//! report the full confusion matrix, PPV/NPV, the abstention rate, and the 30-second restriction.
//!
//! ```text
//! cargo run --release -p physio-algo --example rr_screen_eval -- [DATASETS_DIR]
//! ```

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use physio_algo::ecg::detect_pan_tompkins;
use physio_algo::rr_irregularity::{
    assess, profile, cosen, screen::{self, ScreenState}, IrregularityReading,
};
use physio_algo::stats::median;

mod rr_corpus;
use rr_corpus::Class;

const DEFAULT_DATASETS: &str = "../whoop-data/datasets";

fn main() {
    let root = PathBuf::from(
        std::env::args().nth(1).unwrap_or_else(|| DEFAULT_DATASETS.to_string()),
    );
    if std::env::args().any(|a| a == "--a-only") {
        physionet_windows(&root);
        return;
    }
    if std::env::args().any(|a| a == "--own") {
        own();
        return;
    }
    if std::env::args().any(|a| a == "--cinc-only") {
        cinc(&root);
        return;
    }
    physionet_windows(&root);
    physionet_thirty_seconds(&root);
    cinc(&root);
}

/// Stamp one beat a second, the convention both existing harnesses use for corpus beats.
fn stamp1(rr: &[u16]) -> Vec<(u32, u16)> {
    rr.iter().enumerate().map(|(i, &v)| (i as u32, v)).collect()
}

/// Rebuild a record's beat series from its sliding windows.
fn series(rec: &rr_corpus::Record) -> Vec<u16> {
    let Some(first) = rec.windows.first() else { return Vec::new() };
    let mut rr = first.rr.clone();
    for w in rec.windows.iter().skip(1) {
        rr.extend_from_slice(&w.rr[w.rr.len() - screen::STEP_BEATS..]);
    }
    rr
}

fn paths(root: &Path, db: &str) -> Vec<PathBuf> {
    let dir = root.join(db).join("rr");
    let Ok(rd) = fs::read_dir(&dir) else { return Vec::new() };
    let mut p: Vec<PathBuf> =
        rd.map(|e| e.unwrap().path()).filter(|x| x.extension().is_some_and(|e| e == "txt")).collect();
    p.sort();
    p
}

const DBS: [&str; 3] = ["physionet-afdb", "physionet-nsrdb", "physionet-mitdb"];

/// Window-level confusion matrix of the shipped screen over the whole PhysioNet corpus, per database and
/// pooled, plus the record-level state distribution (how often it abstains).
fn physionet_windows(root: &Path) {
    println!("\n########## A. PhysioNet, shipped screen(), window level ##########");
    // (db, class, reported)
    let mut rows: Vec<(String, Class, bool)> = Vec::new();
    let mut states: BTreeMap<String, BTreeMap<String, usize>> = BTreeMap::new();
    let mut refusals: BTreeMap<String, usize> = BTreeMap::new();
    let mut win_total = 0usize;
    let mut win_refused = 0usize;
    let mut bands: BTreeMap<String, (usize, usize, usize)> = BTreeMap::new();
    let mut notable: Vec<String> = Vec::new();
    for db in DBS {
        for p in paths(root, db) {
            let rec = rr_corpus::windows(&p, screen::WINDOW_BEATS, screen::STEP_BEATS);
            if rec.windows.is_empty() {
                continue;
            }
            let rr = series(&rec);
            let beats = stamp1(&rr);
            let state = screen::screen(&beats);
            let key = match &state {
                ScreenState::Calibrating { .. } => "Calibrating",
                ScreenState::Regular { .. } => "Regular",
                ScreenState::IrregularEpisodes { .. } => "IrregularEpisodes",
                ScreenState::Inconclusive { reason } => match reason {
                    screen::ScreenRefusal::PoorInputQuality { .. } => "Inconclusive/PoorInputQuality",
                    screen::ScreenRefusal::EctopyNotExcluded { .. } => "Inconclusive/EctopyNotExcluded",
                },
            };
            *states.entry(db.to_string()).or_default().entry(key.to_string()).or_default() += 1;
            let eps = match &state {
                ScreenState::IrregularEpisodes { episodes, .. } => episodes.clone(),
                _ => Vec::new(),
            };
            for e in &eps {
                let slot = bands.entry(format!("{:?}", e.confidence)).or_insert((0usize, 0usize, 0usize));
                slot.0 += 1;
                for w in &rec.windows {
                    let (a, b) = (w.start_beat as u32, (w.start_beat + w.rr.len() - 1) as u32);
                    if a >= e.start_unix && b <= e.end_unix {
                        slot.1 += 1;
                        slot.2 += usize::from(w.class.is_positive());
                    }
                }
            }
            let pos = rec.windows.iter().filter(|w| w.class.is_positive()).count();
            if pos > 0 && eps.is_empty() {
                notable.push(format!("MISSED {db}/{} — {pos} fibrillation windows of {}, state {state:?}", rec.name, rec.windows.len()));
            }
            if pos == 0 && !eps.is_empty() {
                notable.push(format!("FALSE  {db}/{} — {} episodes, longest {} windows, no fibrillation in the record", rec.name, eps.len(), eps.iter().map(|e| e.windows).max().unwrap_or(0)));
            }
            for w in &rec.windows {
                let (a, b) = (w.start_beat as u32, (w.start_beat + w.rr.len() - 1) as u32);
                let inside = eps.iter().any(|e| a >= e.start_unix && b <= e.end_unix);
                rows.push((db.to_string(), w.class, inside));
                win_total += 1;
                if let IrregularityReading::Inconclusive { reason, .. } = assess(&stamp1(&w.rr)) {
                    win_refused += 1;
                    *refusals.entry(format!("{reason:?}")).or_default() += 1;
                }
            }
        }
    }
    println!("windows {win_total}");
    matrix("POOLED (afdb+nsrdb+mitdb)", rows.iter().map(|(_, c, r)| (*c, *r)));
    for db in DBS {
        matrix(db, rows.iter().filter(|(d, _, _)| d == db).map(|(_, c, r)| (*c, *r)));
    }
    println!("\nper-class reported-window rate (the number that hurts is the ectopy one):");
    for c in [
        Class::AfibAudited, Class::AfibMachine, Class::SinusNsr, Class::SinusAfdb, Class::SinusMit,
        Class::EctopyLight, Class::EctopyHeavy, Class::EctopyBigeminal, Class::Flutter,
        Class::Junctional, Class::Paced, Class::Mixed,
    ] {
        let list: Vec<&(String, Class, bool)> = rows.iter().filter(|(_, cl, _)| *cl == c).collect();
        if list.is_empty() {
            continue;
        }
        let hit = list.iter().filter(|(_, _, r)| *r).count();
        println!("  {:<18} {hit:>6} of {:>6}   {:.4}", format!("{c:?}"), list.len(), hit as f64 / list.len() as f64);
    }
    let total_eps: usize = bands.values().map(|v| v.0).sum();
    println!("\nepisodes by confidence band: total {total_eps}");
    for (band, (n, w, p)) in &bands {
        println!("  {band:<9} {n:>4} episodes, {:.4} of their {w} windows were fibrillation", *p as f64 / *w as f64);
    }
    println!("\nrecords worth naming:");
    for l in &notable {
        println!("  {l}");
    }
    println!("\nrecord-level state, per database:");
    for (db, m) in &states {
        let n: usize = m.values().sum();
        print!("  {db:<20} n {n:>3}  ");
        for (k, v) in m {
            print!("{k} {v}   ");
        }
        println!();
    }
    println!("\nwindow-level assess() refusals: {win_refused} of {win_total} ({:.4})", win_refused as f64 / win_total as f64);
    for (r, n) in &refusals {
        println!("  {r:<40} {n}");
    }
}

/// Sensitivity / specificity / PPV / NPV plus the four cells, over the two labelled arms.
fn matrix(label: &str, it: impl Iterator<Item = (Class, bool)>) {
    let (mut tp, mut fp, mut tn, mut fnn) = (0usize, 0usize, 0usize, 0usize);
    for (c, reported) in it {
        match (c.is_positive(), c.is_negative(), reported) {
            (true, _, true) => tp += 1,
            (true, _, false) => fnn += 1,
            (_, true, true) => fp += 1,
            (_, true, false) => tn += 1,
            _ => {}
        }
    }
    let d = |a: usize, b: usize| if a + b == 0 { f64::NAN } else { a as f64 / (a + b) as f64 };
    println!(
        "\n{label}\n  TP {tp}  FP {fp}  TN {tn}  FN {fnn}\n  sensitivity {:.4}  specificity {:.4}  PPV {:.4}  NPV {:.4}",
        d(tp, fnn), d(tn, fp), d(tp, fp), d(tn, fnn)
    );
}

/// The product is a 30-second reading. Cut every record into consecutive 30 s spans of BEAT TIME and run
/// the shipped screen on each, then the same spans through the window-level rule the screen is built on.
fn physionet_thirty_seconds(root: &Path) {
    println!("\n########## B. PhysioNet restricted to 30-second spans ##########");
    for span_s in [30.0f64, 60.0, 300.0] {
        let mut states: BTreeMap<String, usize> = BTreeMap::new();
        let mut rule: Vec<(Class, bool)> = Vec::new();
        let mut beats_per: Vec<usize> = Vec::new();
        for db in DBS {
            for p in paths(root, db) {
                let rec = rr_corpus::windows(&p, screen::WINDOW_BEATS, screen::STEP_BEATS);
                if rec.windows.is_empty() {
                    continue;
                }
                let rr = series(&rec);
                // Class of a span: the class of the window whose start_beat it contains, taken from the
                // corpus labels the harness already derives.
                let mut i = 0usize;
                while i < rr.len() {
                    let mut ms = 0.0f64;
                    let start = i;
                    while i < rr.len() && ms < span_s * 1000.0 {
                        ms += f64::from(rr[i]);
                        i += 1;
                    }
                    if ms < span_s * 900.0 {
                        break; // trailing partial span
                    }
                    let span = &rr[start..i];
                    beats_per.push(span.len());
                    let key = match screen::screen(&stamp1(span)) {
                        ScreenState::Calibrating { .. } => "Calibrating",
                        ScreenState::Regular { .. } => "Regular",
                        ScreenState::IrregularEpisodes { .. } => "IrregularEpisodes",
                        ScreenState::Inconclusive { .. } => "Inconclusive",
                    };
                    *states.entry(key.to_string()).or_default() += 1;
                    // The window-level rule, with the duration floor removed: the most the indices can
                    // say about a span this short.
                    let class = rec
                        .windows
                        .iter()
                        .find(|w| w.start_beat >= start && w.start_beat < i)
                        .map(|w| w.class);
                    if let Some(class) = class {
                        rule.push((class, span_rule(span)));
                    }
                }
            }
        }
        beats_per.sort_unstable();
        let n: usize = states.values().sum();
        println!(
            "\n--- {span_s:.0} s spans: {n} spans, beats per span median {} (min {} max {})",
            beats_per.get(beats_per.len() / 2).copied().unwrap_or(0),
            beats_per.first().copied().unwrap_or(0),
            beats_per.last().copied().unwrap_or(0),
        );
        println!("  MIN_SCREEN_BEATS = {}", screen::MIN_SCREEN_BEATS);
        for (k, v) in &states {
            println!("  shipped screen(): {k:<20} {v:>7}  {:.4}", *v as f64 / n as f64);
        }
        matrix(&format!("{span_s:.0} s, duration floor REMOVED (median COSEn + residual rule)"), rule.iter().copied());
        for c in [Class::EctopyLight, Class::EctopyHeavy, Class::EctopyBigeminal, Class::Flutter] {
            let list: Vec<&(Class, bool)> = rule.iter().filter(|(cl, _)| *cl == c).collect();
            if list.is_empty() {
                continue;
            }
            let hit = list.iter().filter(|(_, r)| *r).count();
            println!("  {:<18} {hit:>5} of {:>5}   {:.4}", format!("{c:?}"), list.len(), hit as f64 / list.len() as f64);
        }
    }
}

/// The screen's own two tests applied to one short span, with the duration floor removed: median COSEn
/// over the span's windows at or above the irregularity floor, and median residual COSEn at or above the
/// ectopy floor. This is the strongest honest statement the indices support on a 30-second reading.
fn span_rule(rr: &[u16]) -> bool {
    let mut cs = Vec::new();
    let mut rs = Vec::new();
    let mut s = 0usize;
    while s + screen::WINDOW_BEATS <= rr.len() {
        let w = &rr[s..s + screen::WINDOW_BEATS];
        if let Some(c) = cosen(w) {
            cs.push(c);
        }
        if let Some(r) = profile(w).and_then(|p| p.residual_cosen) {
            rs.push(r);
        }
        s += screen::STEP_BEATS;
    }
    if cs.is_empty() {
        // Too few beats for even one window: the span cannot be scored at all.
        return false;
    }
    median(&cs) >= screen::COSEN_IRREGULAR_FLOOR
        && !rs.is_empty()
        && median(&rs) >= screen::EPISODE_RESIDUAL_COSEN_FLOOR
}

/// CinC 2017: single handheld lead, 9-61 s, four classes. R-R comes from the project's own Pan-Tompkins,
/// since the Challenge ships no beat annotations.
fn cinc(root: &Path) {
    println!("\n########## C. PhysioNet/CinC Challenge 2017 (single lead, ~30 s) ##########");
    let dir = root.join("physionet-challenge2017");
    let Ok(index) = fs::read_to_string(dir.join("index.tsv")) else {
        println!("  no corpus at {}", dir.display());
        return;
    };
    let Ok(blob) = fs::read(dir.join("signals.i16")) else {
        println!("  no signals.i16");
        return;
    };
    let mut states: BTreeMap<String, BTreeMap<String, usize>> = BTreeMap::new();
    let mut rule: Vec<(String, bool)> = Vec::new();
    let mut gated: Vec<(String, Option<bool>)> = Vec::new();
    let mut beats_by: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    let mut scorable: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    let mut lines = index.lines();
    lines.next();
    for line in lines {
        let f: Vec<&str> = line.split('\t').collect();
        if f.len() < 7 {
            continue;
        }
        let (class, n, off, fs_hz, gain) = (
            f[1].to_string(),
            f[2].parse::<usize>().unwrap(),
            f[3].parse::<usize>().unwrap(),
            f[4].parse::<f64>().unwrap(),
            f[5].parse::<f64>().unwrap(),
        );
        let samples: Vec<f64> = (0..n)
            .map(|i| {
                let b = off + i * 2;
                f64::from(i16::from_le_bytes([blob[b], blob[b + 1]])) / gain
            })
            .collect();
        let peaks = detect_pan_tompkins(&samples, fs_hz);
        let rr: Vec<u16> = peaks
            .windows(2)
            .map(|w| ((w[1] - w[0]) as f64 * 1000.0 / fs_hz).round().clamp(0.0, 65535.0) as u16)
            .collect();
        beats_by.entry(class.clone()).or_default().push(rr.len());
        let key = match screen::screen(&stamp1(&rr)) {
            ScreenState::Calibrating { .. } => "Calibrating",
            ScreenState::Regular { .. } => "Regular",
            ScreenState::IrregularEpisodes { .. } => "IrregularEpisodes",
            ScreenState::Inconclusive { .. } => "Inconclusive",
        };
        *states.entry(class.clone()).or_default().entry(key.to_string()).or_default() += 1;
        // Scorable = at least one 32-beat window survived assess(); the abstention that is not the
        // duration floor.
        let mut ok = 0usize;
        let mut tried = 0usize;
        let mut s = 0usize;
        while s + screen::WINDOW_BEATS <= rr.len() {
            tried += 1;
            if matches!(assess(&stamp1(&rr[s..s + screen::WINDOW_BEATS])), IrregularityReading::Assessed(_)) {
                ok += 1;
            }
            s += screen::STEP_BEATS;
        }
        let e = scorable.entry(class.clone()).or_default();
        e.1 += 1;
        e.0 += usize::from(tried > 0 && ok > 0);
        rule.push((class.clone(), span_rule(&rr)));
        gated.push((class, gated_rule(&rr)));
    }
    println!("\nrecords by class, beats detected (Pan-Tompkins at 300 Hz):");
    for (c, v) in &beats_by {
        let mut v = v.clone();
        v.sort_unstable();
        let over = v.iter().filter(|&&b| b >= screen::MIN_SCREEN_BEATS).count();
        println!(
            "  class {c:<3} n {:>5}  beats median {:>4} max {:>4}   records reaching MIN_SCREEN_BEATS ({}) : {over}",
            v.len(), v[v.len() / 2], v[v.len() - 1], screen::MIN_SCREEN_BEATS
        );
    }
    println!("\nshipped screen() state by class:");
    for (c, m) in &states {
        let n: usize = m.values().sum();
        print!("  class {c:<3} n {n:>5}  ");
        for (k, v) in m {
            print!("{k} {v} ({:.3})   ", *v as f64 / n as f64);
        }
        println!();
    }
    println!("\nwindows that survived assess(): records with at least one, by class");
    for (c, (ok, n)) in &scorable {
        println!("  class {c:<3} {ok:>5} of {n:>5}   {:.4}   abstain {:.4}", *ok as f64 / *n as f64, 1.0 - *ok as f64 / *n as f64);
    }
    println!("\nthe screen's own rule, duration floor removed, per class:");
    for c in ["N", "A", "O", "~"] {
        let list: Vec<&(String, bool)> = rule.iter().filter(|(cl, _)| cl == c).collect();
        if list.is_empty() {
            continue;
        }
        let hit = list.iter().filter(|(_, r)| *r).count();
        println!("  class {c:<3} flagged {hit:>5} of {:>5}   {:.4}", list.len(), hit as f64 / list.len() as f64);
    }
    let cell = |want_pos: &dyn Fn(&str) -> bool, want_neg: &dyn Fn(&str) -> bool| {
        let (mut tp, mut fp, mut tn, mut fnn) = (0usize, 0usize, 0usize, 0usize);
        for (c, r) in &rule {
            if want_pos(c) {
                if *r { tp += 1 } else { fnn += 1 }
            } else if want_neg(c) {
                if *r { fp += 1 } else { tn += 1 }
            }
        }
        let d = |a: usize, b: usize| if a + b == 0 { f64::NAN } else { a as f64 / (a + b) as f64 };
        println!(
            "  TP {tp} FP {fp} TN {tn} FN {fnn}  sens {:.4} spec {:.4} PPV {:.4} NPV {:.4}",
            d(tp, fnn), d(tn, fp), d(tp, fp), d(tn, fnn)
        );
    };
    println!("\nA vs N only:");
    cell(&|c| c == "A", &|c| c == "N");
    println!("A vs N + O (the honest population — other rhythm is a negative here):");
    cell(&|c| c == "A", &|c| c == "N" || c == "O");
    println!("A vs N + O + ~ (everything a wrist would meet):");
    cell(&|c| c == "A", &|c| c != "A");

    println!("
--- QUALITY-GATED: only windows that survived assess() count; no scorable window = Inconclusive");
    for c in ["N", "A", "O", "~"] {
        let list: Vec<&(String, Option<bool>)> = gated.iter().filter(|(cl, _)| cl == c).collect();
        if list.is_empty() {
            continue;
        }
        let inc = list.iter().filter(|(_, r)| r.is_none()).count();
        let flagged = list.iter().filter(|(_, r)| *r == Some(true)).count();
        let conclusive = list.len() - inc;
        println!(
            "  class {c:<3} n {:>5}  Inconclusive {inc:>5} ({:.4})   flagged {flagged:>4} of the {conclusive} conclusive ({:.4})   flagged of all ({:.4})",
            list.len(),
            inc as f64 / list.len() as f64,
            if conclusive == 0 { f64::NAN } else { flagged as f64 / conclusive as f64 },
            flagged as f64 / list.len() as f64
        );
    }
    let gcell = |label: &str, want_neg: &dyn Fn(&str) -> bool| {
        let (mut tp, mut fp, mut tn, mut fnn) = (0usize, 0usize, 0usize, 0usize);
        for (c, r) in &gated {
            let Some(r) = r else { continue };
            if c == "A" {
                if *r { tp += 1 } else { fnn += 1 }
            } else if want_neg(c) {
                if *r { fp += 1 } else { tn += 1 }
            }
        }
        let d = |a: usize, b: usize| if a + b == 0 { f64::NAN } else { a as f64 / (a + b) as f64 };
        println!(
            "  {label}: TP {tp} FP {fp} TN {tn} FN {fnn}  sens {:.4} spec {:.4} PPV {:.4} NPV {:.4}",
            d(tp, fnn), d(tn, fp), d(tp, fp), d(tn, fnn)
        );
    };
    gcell("A vs N (conclusive only)", &|c| c == "N");
    gcell("A vs N+O (conclusive only)", &|c| c == "N" || c == "O");
    gcell("A vs N+O+~ (conclusive only)", &|c| c != "A");
}

/// The screen's two tests over the windows that SURVIVED `assess()`. `None` when no window did — the
/// honest abstention, kept apart from a negative answer.
fn gated_rule(rr: &[u16]) -> Option<bool> {
    let mut cs = Vec::new();
    let mut rs = Vec::new();
    let mut s = 0usize;
    while s + screen::WINDOW_BEATS <= rr.len() {
        let w = &rr[s..s + screen::WINDOW_BEATS];
        if let IrregularityReading::Assessed(i) = assess(&stamp1(w)) {
            if let Some(c) = i.cosen {
                cs.push(c);
            }
            if let Some(r) = i.ectopy.and_then(|e| e.residual_cosen) {
                rs.push(r);
            }
        }
        s += screen::STEP_BEATS;
    }
    if cs.is_empty() {
        return None;
    }
    Some(
        median(&cs) >= screen::COSEN_IRREGULAR_FLOOR
            && !rs.is_empty()
            && median(&rs) >= screen::EPISODE_RESIDUAL_COSEN_FLOOR,
    )
}

/// David's own strap R-R: print every reported episode's span, then the beats inside it.
fn own() {
    let path = "../whoop-data/harnesses/rr-real-fixture.json";
    let raw = fs::read_to_string(path).expect("fixture");
    // Minimal scan of the fixture's `[unix, ms]` pairs per night, without a json dependency change.
    for chunk in raw.split("\"date\":").skip(1) {
        let date: String = chunk.chars().skip(1).take(10).collect();
        let Some(rrpos) = chunk.find("\"rr\":") else { continue };
        let body = &chunk[rrpos..];
        let end = body.find("}]").map(|e| e + 2).unwrap_or(body.len());
        let mut beats: Vec<(u32, u16)> = Vec::new();
        let mut nums: Vec<u64> = Vec::new();
        let mut cur = String::new();
        for ch in body[..end].chars() {
            if ch.is_ascii_digit() {
                cur.push(ch);
            } else {
                if !cur.is_empty() {
                    nums.push(cur.parse().unwrap_or(0));
                    cur.clear();
                }
                if ch == ']' && nums.len() >= 2 {
                    beats.push((nums[0] as u32, nums[1].min(65535) as u16));
                    nums.clear();
                } else if ch == '[' {
                    nums.clear();
                }
            }
        }
        if beats.len() < 100 {
            continue;
        }
        let state = screen::screen(&beats);
        let ScreenState::IrregularEpisodes { episodes, .. } = &state else {
            println!("{date}  {} beats  {:?}", beats.len(), std::mem::discriminant(&state));
            continue;
        };
        for e in episodes {
            let inside: Vec<(u32, u16)> =
                beats.iter().copied().filter(|&(t, _)| t >= e.start_unix && t <= e.end_unix).collect();
            let mut secs: BTreeMap<u32, Vec<u16>> = BTreeMap::new();
            for &(t, v) in &inside {
                secs.entry(t).or_default().push(v);
            }
            let multi = secs.values().filter(|v| v.len() > 1).count();
            let impossible = secs
                .values()
                .filter(|v| v.len() > 1 && v.iter().copied().min().unwrap() < 600 && v.iter().copied().max().unwrap() > 1400)
                .count();
            let beat_ms: f64 = inside.iter().map(|&(_, v)| f64::from(v)).sum();
            println!(
                "
{date} EPISODE {}..{} ({} s), {} beats over {} distinct seconds",
                e.start_unix, e.end_unix, e.dur_s, inside.len(), secs.len()
            );
            println!(
                "  beat-time {:.0} s over clock {} s = coverage {:.3}   seconds with >1 beat {multi}   seconds holding both a <600 ms and a >1400 ms interval {impossible}",
                beat_ms / 1000.0, e.dur_s, beat_ms / 1000.0 / f64::from(e.dur_s.max(1))
            );
            println!("  reported: {:?} cosen {:.2} residual {:.2} ectopic {:.2} coverage {:.2}", e.confidence, e.cosen, e.residual_cosen, e.ectopic_fraction, e.coverage);
            let head: Vec<String> = inside.iter().take(24).map(|(t, v)| format!("{}:{v}", t % 100000)).collect();
            println!("  first 24 beats  {}", head.join(" "));
        }
    }
}
