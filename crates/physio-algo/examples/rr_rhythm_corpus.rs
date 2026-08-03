//! Measure the R-R irregularity indices over the PhysioNet R-R corpus, by rhythm class, so the
//! separation between a sustained irregular rhythm and an ectopy burst is a measurement rather than a
//! claim.
//!
//! ```text
//! cargo run --release -p physio-algo --example rr_rhythm_corpus -- [DATASETS_DIR] [--beats N] [--emit DIR]
//! ```
//!
//! Reads `physionet-{afdb,nsrdb,mitdb}/rr/*.txt` (`rr_ms beat rhythm` per line). Every record of every
//! database is read; nothing is sampled or selected. Indices are computed on the bare R-R series: the
//! corpus never passed through this project's storage, so the duplication gates that guard real device
//! data would only measure the whole-second stamp this harness derives. `--emit` writes the distilled
//! per-class fixture the acceptance test carries.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use physio_algo::rr_irregularity::{
    screen::{self, ScreenState},
    cosen, poincare, profile, rmssd_over_mean_rr, sample_entropy, shannon_entropy_drr,
    turning_point_ratio, COSEN_M, COSEN_R_MS,
};
use physio_algo::stats::median;

mod rr_corpus;
use rr_corpus::{segments, Class, Segment};

const DEFAULT_DATASETS: &str = "../whoop-data/datasets";
/// Segment lengths reported side by side: the length a 30 s reading can support, and Dash's own.
const REPORT_BEATS: [usize; 2] = [32, 128];
/// Beats in one committed fixture stretch. Over `screen::MIN_SCREEN_BEATS`, so an episode is possible.
const FIXTURE_BEATS: usize = 256;

fn main() {
    let mut args = std::env::args().skip(1);
    let mut root = PathBuf::from(DEFAULT_DATASETS);
    let mut emit: Option<PathBuf> = None;
    let mut beats: Vec<usize> = REPORT_BEATS.to_vec();
    while let Some(a) = args.next() {
        match a.as_str() {
            "--emit" => emit = args.next().map(PathBuf::from),
            "--beats" => beats = vec![args.next().and_then(|v| v.parse().ok()).expect("--beats N")],
            other => root = PathBuf::from(other),
        }
    }
    for n in &beats {
        let segs = load(&root, *n);
        println!("\n===== {} segments of {n} beats, from {} =====", segs.len(), root.display());
        per_class(&segs);
        sweep(&segs);
    }
    // The committed fixture carries whole stretches, not single windows: the screen needs more beats than
    // one window before it can form an episode at all, so a window-length fixture could not exercise it.
    if let Some(dir) = &emit {
        emit_fixture(&load(&root, FIXTURE_BEATS), dir);
    }
    episodes(&root);
    shipped(&root);
}

/// Run the SHIPPED `screen()` over every record and score its episodes. The ablation above works on raw
/// index rows; this is the only number that describes the code that runs, and the two are not the same
/// thing until they are checked against each other.
///
/// Corpus beats are stamped one per second. They carry no wall clock this project produced, so the
/// storage-duplication gates inside `assess` would only be measuring the stamp this harness invented;
/// one beat per second makes them inert and leaves the rhythm logic under test. Episode DURATIONS from
/// this run are therefore in beats, not seconds.
fn shipped(root: &Path) {
    let mut reported: Vec<(Class, bool)> = Vec::new();
    let (mut flagged_records, mut records_total) = (0usize, 0usize);
    let mut per_db: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    let mut episode_windows: Vec<usize> = Vec::new();
    let mut by_conf: BTreeMap<String, (usize, usize, usize)> = BTreeMap::new();
    let mut notable: Vec<String> = Vec::new();
    let mut ectopic: Vec<f64> = Vec::new();
    for db in ["physionet-afdb", "physionet-nsrdb", "physionet-mitdb"] {
        let dir = root.join(db).join("rr");
        let Ok(rd) = fs::read_dir(&dir) else { continue };
        let mut paths: Vec<PathBuf> =
            rd.map(|e| e.unwrap().path()).filter(|p| p.extension().is_some_and(|x| x == "txt")).collect();
        paths.sort();
        for p in &paths {
            let rec = rr_corpus::windows(p, screen::WINDOW_BEATS, screen::STEP_BEATS);
            let Some(first) = rec.windows.first() else { continue };
            // Rebuild the record's beat series from the windows, then stamp one beat a second.
            let mut rr: Vec<u16> = first.rr.clone();
            for w in rec.windows.iter().skip(1) {
                rr.extend_from_slice(&w.rr[w.rr.len() - screen::STEP_BEATS..]);
            }
            let beats: Vec<(u32, u16)> = rr.iter().enumerate().map(|(i, &v)| (i as u32, v)).collect();
            let state = screen::screen(&beats);
            let eps = match &state {
                ScreenState::IrregularEpisodes { episodes, .. } => episodes.clone(),
                _ => Vec::new(),
            };
            records_total += 1;
            let entry = per_db.entry(db.to_string()).or_default();
            entry.1 += 1;
            if !eps.is_empty() {
                flagged_records += 1;
                entry.0 += 1;
            }
            let inside = |a: u32, b: u32| eps.iter().any(|e| a >= e.start_unix && b <= e.end_unix);
            let mut pos_windows = 0usize;
            for w in &rec.windows {
                let (a, b) = (w.start_beat as u32, (w.start_beat + w.rr.len() - 1) as u32);
                reported.push((w.class, inside(a, b)));
                pos_windows += usize::from(w.class.is_positive());
            }
            // Each episode's confidence band, with how much of it was actually fibrillation, so the band
            // has a measured meaning rather than only an ordering.
            for e in &eps {
                episode_windows.push(e.windows);
                ectopic.push(e.ectopic_fraction);
                let slot = by_conf.entry(format!("{:?}", e.confidence)).or_default();
                slot.0 += 1;
                for w in &rec.windows {
                    let (a, b) = (w.start_beat as u32, (w.start_beat + w.rr.len() - 1) as u32);
                    if a >= e.start_unix && b <= e.end_unix {
                        slot.1 += 1;
                        slot.2 += usize::from(w.class.is_positive());
                    }
                }
            }
            if pos_windows > 0 && eps.is_empty() {
                notable.push(format!("MISSED {db}/{} ({pos_windows} fibrillation windows, {state:?})", rec.name));
            }
            if pos_windows == 0 && !eps.is_empty() {
                notable.push(format!(
                    "FALSE  {db}/{} ({} episodes, longest {} windows)",
                    rec.name,
                    eps.len(),
                    eps.iter().map(|e| e.windows).max().unwrap_or(0)
                ));
            }
        }
    }
    let rate = |want: &dyn Fn(Class) -> bool| {
        let list: Vec<&(Class, bool)> = reported.iter().filter(|(c, _)| want(*c)).collect();
        if list.is_empty() {
            f64::NAN
        } else {
            list.iter().filter(|(_, r)| *r).count() as f64 / list.len() as f64
        }
    };
    println!("\n===== SHIPPED screen(): {} windows over {records_total} records =====", reported.len());
    println!("sensitivity (fibrillation windows inside a reported episode) {:.3}", rate(&Class::is_positive));
    println!("specificity (pooled negatives)                               {:.3}", 1.0 - rate(&Class::is_negative));
    for c in NEG_CLASSES {
        println!("  false-positive {:<17} {:.4}   n {}", format!("{c:?}"), rate(&|x| x == c), reported.iter().filter(|(cl, _)| *cl == c).count());
    }
    for c in [Class::Flutter, Class::Junctional, Class::Paced, Class::Mixed] {
        println!("  reported (neither arm) {:<9} {:.4}   n {}", format!("{c:?}"), rate(&|x| x == c), reported.iter().filter(|(cl, _)| *cl == c).count());
    }
    println!("records with at least one episode: {flagged_records} of {records_total}");
    for (db, (hit, n)) in &per_db {
        println!("  {db:<24} {hit} of {n}");
    }
    episode_windows.sort_unstable();
    if !episode_windows.is_empty() {
        println!(
            "episodes {}: windows min {} median {} max {}",
            episode_windows.len(),
            episode_windows[0],
            episode_windows[episode_windows.len() / 2],
            episode_windows[episode_windows.len() - 1]
        );
    }
    ectopic.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if !ectopic.is_empty() {
        let q = |f: f64| ectopic[((ectopic.len() - 1) as f64 * f) as usize];
        println!(
            "episode ectopic fraction: min {:.3} p50 {:.3} p90 {:.3} p99 {:.3} max {:.3}",
            q(0.0), q(0.5), q(0.9), q(0.99), q(1.0)
        );
    }
    for (band, (n, windows, positive)) in &by_conf {
        println!(
            "  confidence {band:<9} {n:>4} episodes, {:.3} of their {windows} windows were fibrillation",
            *positive as f64 / *windows as f64
        );
    }
    for line in &notable {
        println!("  {line}");
    }
}

/// What persistence buys. A window is reported only inside a run of `min_run` consecutive firing windows
/// of the same record, which is the shipped rule: a sustained irregular rhythm holds, a burst does not.
/// Bigeminy can also hold for minutes, so persistence alone cannot separate it — that is the ectopy
/// discriminator's job, and this table is where the two are read together.
fn episodes(root: &Path) {
    let (window, step) = (32usize, 8usize);
    let mut records: Vec<rr_corpus::Record> = Vec::new();
    for db in ["physionet-afdb", "physionet-nsrdb", "physionet-mitdb"] {
        let dir = root.join(db).join("rr");
        let Ok(rd) = fs::read_dir(&dir) else { continue };
        let mut paths: Vec<PathBuf> =
            rd.map(|e| e.unwrap().path()).filter(|p| p.extension().is_some_and(|x| x == "txt")).collect();
        paths.sort();
        for p in &paths {
            records.push(rr_corpus::windows(p, window, step));
        }
    }
    let scored: Vec<Vec<Row>> = records
        .iter()
        .map(|r| r.windows.iter().map(|s| (s.class, row(&s.rr))).collect())
        .collect();
    let total: usize = scored.iter().map(Vec::len).sum();
    println!("\n===== episodes: {total} windows of {window} beats, step {step}, over {} records =====", records.len());
    println!("ablation: the most sensitive rule whose worst per-class reported rate clears the ceiling");
    println!("{:>7} {:>10} {:>7} {:>9} {:>7} {:>6}   per-class reported-window rate", "ceiling", "rule", "cosen", "residCOS", "min_run", "sens");
    let cosen_cuts: Vec<f64> = (0..24).map(|i| -2.0 + f64::from(i) * 0.08).collect();
    let resid_cuts: Vec<f64> = (0..18).map(|i| -2.2 + f64::from(i) * 0.10).collect();
    // What the ectopy veto is worth AT THE SHIPPED CONSTANTS, which is the only comparison that
    // describes the code that runs. The search below re-optimises the cut for each variant and so
    // answers a different question.
    println!("
at the shipped constants (cosen {:.2}, min_run {}), veto on and off:", screen::COSEN_IRREGULAR_FLOOR, screen::MIN_EPISODE_WINDOWS);
    for veto in [Veto::None, Veto::PerEpisode] {
        let flags = flag(&scored, screen::COSEN_IRREGULAR_FLOOR, screen::EPISODE_RESIDUAL_COSEN_FLOOR, screen::MIN_EPISODE_WINDOWS, veto);
        let per: Vec<String> = NEG_CLASSES
            .iter()
            .map(|c| format!("{:?} {:.4}", c, rate(&scored, &flags, &|x| x == *c)))
            .collect();
        println!("  {:<12} sens {:.3}   {}", format!("{veto:?}"), rate(&scored, &flags, &Class::is_positive), per.join("  "));
    }
    for ceiling in [0.20, 0.10, 0.05, 0.02] {
        for veto in [Veto::None, Veto::PerWindow, Veto::PerEpisode] {
            let mut best: Option<(f64, f64, f64, usize)> = None;
            for &ca in &cosen_cuts {
                for &cb in if veto == Veto::None { &[f64::NEG_INFINITY][..] } else { &resid_cuts } {
                    for min_run in [1usize, 2, 4, 6, 8, 12, 16, 24] {
                        let flags = flag(&scored, ca, cb, min_run, veto);
                        let sens = rate(&scored, &flags, &Class::is_positive);
                        if best.is_some_and(|b| sens <= b.0) {
                            continue;
                        }
                        let worst = NEG_CLASSES
                            .iter()
                            .map(|c| rate(&scored, &flags, &|x| x == *c))
                            .fold(0.0f64, f64::max);
                        if worst <= ceiling {
                            best = Some((sens, ca, cb, min_run));
                        }
                    }
                }
            }
            let Some((sens, ca, cb, min_run)) = best else {
                println!("{ceiling:>7.2} {veto:>10?}   nothing clears it");
                continue;
            };
            let flags = flag(&scored, ca, cb, min_run, veto);
            let per: Vec<String> = NEG_CLASSES
                .iter()
                .map(|c| format!("{:?} {:.3}", c, rate(&scored, &flags, &|x| x == *c)))
                .collect();
            println!(
                "{ceiling:>7.2} {veto:>10?} {ca:>7.2} {:>9} {min_run:>7} {sens:>6.3}   {}",
                if cb.is_finite() { format!("{cb:.2}") } else { "-".into() },
                per.join("  ")
            );
        }
    }
}

/// Where the ectopy discriminator is applied: nowhere, to each window before the run is formed, or to the
/// finished run as a whole.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Veto {
    None,
    PerWindow,
    PerEpisode,
}

/// Per record: fire on `cosen >= ca`, apply the veto, and keep only the flags inside a run of `min_run`.
/// A missing residual never survives a veto: without that view the ectopy cannot be ruled out, and not
/// reporting is the conservative answer.
fn flag(scored: &[Vec<Row>], ca: f64, cb: f64, min_run: usize, veto: Veto) -> Vec<Vec<bool>> {
    scored
        .iter()
        .map(|rec| {
            let ok = |r: &Row| r.1[13].is_finite() && r.1[13] >= cb;
            let raw: Vec<bool> = rec
                .iter()
                .map(|r| r.1[4] >= ca && (veto != Veto::PerWindow || ok(r)))
                .collect();
            let mut out = persist(&raw, min_run);
            if veto == Veto::PerEpisode {
                for (s, e) in runs(&out) {
                    let mut v: Vec<f64> = rec[s..e].iter().map(|r| r.1[13]).filter(|x| x.is_finite()).collect();
                    let keep = !v.is_empty() && {
                        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
                        v[v.len() / 2] >= cb
                    };
                    if !keep {
                        out[s..e].fill(false);
                    }
                }
            }
            out
        })
        .collect()
}

/// The `[start, end)` spans of consecutive set flags.
fn runs(flags: &[bool]) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < flags.len() {
        if !flags[i] {
            i += 1;
            continue;
        }
        let s = i;
        while i < flags.len() && flags[i] {
            i += 1;
        }
        out.push((s, i));
    }
    out
}

/// Share of the windows a predicate selects that were reported.
fn rate(scored: &[Vec<Row>], flags: &[Vec<bool>], want: &dyn Fn(Class) -> bool) -> f64 {
    let (mut hit, mut n) = (0usize, 0usize);
    for (rec, fl) in scored.iter().zip(flags) {
        for (r, &f) in rec.iter().zip(fl) {
            if want(r.0) {
                n += 1;
                hit += usize::from(f);
            }
        }
    }
    if n == 0 {
        f64::NAN
    } else {
        hit as f64 / n as f64
    }
}

/// Keep only the flags inside a run of at least `min_run` consecutive set flags.
fn persist(raw: &[bool], min_run: usize) -> Vec<bool> {
    let mut out = vec![false; raw.len()];
    let mut i = 0;
    while i < raw.len() {
        if !raw[i] {
            i += 1;
            continue;
        }
        let start = i;
        while i < raw.len() && raw[i] {
            i += 1;
        }
        if i - start >= min_run {
            out[start..i].fill(true);
        }
    }
    out
}

/// The negative classes, reported one at a time: pooled specificity is dominated by the 74k sinus
/// segments and would hide a detector that fires on every ectopy burst.
const NEG_CLASSES: [Class; 6] = [
    Class::SinusNsr,
    Class::SinusAfdb,
    Class::SinusMit,
    Class::EctopyLight,
    Class::EctopyHeavy,
    Class::EctopyBigeminal,
];

type Row = (Class, [f64; 14]);

/// Positives are the two fibrillation classes, negatives the sinus and ectopy ones. Each index is swept
/// alone, then the best pairs are swept as a two-condition rule, which is what the detector ships.
fn sweep(segs: &[Segment]) {
    let scored: Vec<Row> = segs.iter().map(|s| (s.class, row(&s.rr))).collect();
    let pos: Vec<&Row> = scored.iter().filter(|(c, _)| c.is_positive()).collect();
    let neg: Vec<&Row> = scored.iter().filter(|(c, _)| c.is_negative()).collect();
    println!("\npositives {} negatives {}", pos.len(), neg.len());
    println!("{:<9} {:>4} {:>8} {:>6} {:>6} {:>6}   per-class false positives", "index", "dir", "cut", "sens", "spec", "J");
    for (k, head) in HEADS.iter().enumerate() {
        for dir in [1.0f64, -1.0] {
            let Some((cut, sens, spec)) = best_cut(&pos, &neg, k, dir) else { continue };
            if sens + spec - 1.0 <= 0.0 {
                continue;
            }
            report(head, dir, cut, sens, spec, &scored, &|r: &Row| fires(r, k, dir, cut));
        }
    }

    // The operating point that matters is not the one maximising J: pooled J is decided by the 74k sinus
    // segments, and a rule can win it while firing on half the ectopy. Constrain every negative class
    // instead, then take the most sensitive rule that clears the ceiling.
    for ceiling in [0.10, 0.05, 0.02] {
        println!("\ntwo-condition rules, max sensitivity with EVERY negative class under {ceiling:.2}");
        for &(a, b) in &[(4usize, 13), (4, 11), (7, 13), (3, 11), (4, 8), (13, 7)] {
            grid(&scored, &pos, a, b, ceiling);
        }
        for k in [4usize, 13, 11, 7, 3] {
            single(&scored, &pos, k, ceiling);
        }
    }
}

/// Worst false-positive rate across the negative classes, and the pooled one.
fn worst_fp(scored: &[Row], hit: &dyn Fn(&Row) -> bool) -> (f64, f64) {
    let mut worst = 0.0f64;
    for c in NEG_CLASSES {
        let list: Vec<&Row> = scored.iter().filter(|(cl, _)| *cl == c).collect();
        worst = worst.max(share(&list, hit));
    }
    let all: Vec<&Row> = scored.iter().filter(|(c, _)| c.is_negative()).collect();
    (worst, share(&all, hit))
}

fn single(scored: &[Row], pos: &[&Row], k: usize, ceiling: f64) {
    let mut best: Option<(f64, f64)> = None;
    for cut in candidates(pos, k) {
        let hit = |r: &Row| fires(r, k, 1.0, cut);
        let sens = share(pos, &hit);
        if worst_fp(scored, &hit).0 <= ceiling && best.is_none_or(|b| sens > b.0) {
            best = Some((sens, cut));
        }
    }
    let Some((sens, cut)) = best else {
        println!("{:<9} {:>4} {:>8} {:>6}   no cut clears the ceiling", HEADS[k], ">=", "-", "-");
        return;
    };
    let hit = move |r: &Row| fires(r, k, 1.0, cut);
    report(HEADS[k], 1.0, cut, sens, 1.0 - worst_fp(scored, &hit).1, scored, &hit);
}

/// 41 candidate cuts spanning the positive arm, so the grid comes from the data rather than a guess.
fn candidates(pos: &[&Row], k: usize) -> Vec<f64> {
    let mut v: Vec<f64> = pos.iter().map(|(_, r)| r[k]).filter(|x| x.is_finite()).collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if v.is_empty() {
        return Vec::new();
    }
    (0..=40).map(|i| v[(i * (v.len() - 1)) / 40]).collect()
}

/// True when index `k` is present and past `cut` in direction `dir`. A non-finite index never fires: an
/// index its own beat-count floor ruled out cannot be evidence of anything.
fn fires(r: &Row, k: usize, dir: f64, cut: f64) -> bool {
    r.1[k].is_finite() && dir * r.1[k] >= dir * cut
}

/// The cut maximising Youden's J, searched over percentiles of the positive arm so the grid comes from
/// the data rather than from a guess.
fn best_cut(pos: &[&Row], neg: &[&Row], k: usize, dir: f64) -> Option<(f64, f64, f64)> {
    let mut vals: Vec<f64> = pos.iter().map(|(_, r)| r[k]).filter(|v| v.is_finite()).collect();
    if vals.is_empty() {
        return None;
    }
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut best = (f64::NEG_INFINITY, 0.0, 0.0, 0.0);
    for i in 0..=200 {
        let cut = vals[(i * (vals.len() - 1)) / 200];
        let sens = share(pos, &|r: &Row| fires(r, k, dir, cut));
        let spec = 1.0 - share(neg, &|r: &Row| fires(r, k, dir, cut));
        if sens + spec - 1.0 > best.0 {
            best = (sens + spec - 1.0, cut, sens, spec);
        }
    }
    Some((best.1, best.2, best.3))
}

/// Two conditions in AND: the most sensitive pair of cuts whose worst per-class false-positive rate stays
/// under `ceiling`. Index 8 (`origin`) is read the other way, since a high steady background is evidence
/// AGAINST a sustained irregular rhythm.
fn grid(scored: &[Row], pos: &[&Row], ka: usize, kb: usize, ceiling: f64) {
    let dir = |k: usize| if k == 8 { -1.0 } else { 1.0 };
    let (da, db) = (dir(ka), dir(kb));
    let mut best: Option<(f64, f64, f64)> = None;
    for &ca in &candidates(pos, ka) {
        for &cb in &candidates(pos, kb) {
            let hit = |r: &Row| fires(r, ka, da, ca) && fires(r, kb, db, cb);
            let sens = share(pos, &hit);
            if best.is_some_and(|b| sens <= b.0) || worst_fp(scored, &hit).0 > ceiling {
                continue;
            }
            best = Some((sens, ca, cb));
        }
    }
    let Some((sens, ca, cb)) = best else {
        println!("{:<9} and {:<9}   no pair of cuts clears the ceiling", HEADS[ka], HEADS[kb]);
        return;
    };
    let hit = move |r: &Row| fires(r, ka, da, ca) && fires(r, kb, db, cb);
    println!("{:<9} {:>4} {ca:>8.3}", HEADS[ka], if da > 0.0 { ">=" } else { "<=" });
    report(HEADS[kb], db, cb, sens, 1.0 - worst_fp(scored, &hit).1, scored, &hit);
}

fn report(head: &str, dir: f64, cut: f64, sens: f64, spec: f64, scored: &[Row], hit: &dyn Fn(&Row) -> bool) {
    let per: Vec<String> = NEG_CLASSES
        .iter()
        .map(|c| {
            let list: Vec<&Row> = scored.iter().filter(|(cl, _)| cl == c).collect();
            format!("{:?} {:.3}", c, share(&list, hit))
        })
        .collect();
    let d = if dir > 0.0 { ">=" } else { "<=" };
    println!(
        "{head:<9} {d:>4} {cut:>8.3} {sens:>6.3} {spec:>6.3} {:>6.3}   {}",
        sens + spec - 1.0,
        per.join("  ")
    );
}

fn share(xs: &[&Row], hit: &dyn Fn(&Row) -> bool) -> f64 {
    xs.iter().filter(|r| hit(r)).count() as f64 / xs.len().max(1) as f64
}

fn load(root: &Path, segment_beats: usize) -> Vec<Segment> {
    let mut all = Vec::new();
    for db in ["physionet-afdb", "physionet-nsrdb", "physionet-mitdb"] {
        let dir = root.join(db).join("rr");
        let mut paths: Vec<PathBuf> = match fs::read_dir(&dir) {
            Ok(rd) => {
                rd.map(|e| e.unwrap().path()).filter(|p| p.extension().is_some_and(|x| x == "txt")).collect()
            }
            Err(e) => {
                eprintln!("no corpus at {}: {e}", dir.display());
                std::process::exit(2);
            }
        };
        paths.sort();
        for p in &paths {
            all.extend(segments(p, segment_beats));
        }
    }
    all
}

/// Every index of one segment, in the print order of the table.
fn row(rr: &[u16]) -> [f64; 14] {
    let p = poincare(rr);
    let e = profile(rr);
    [
        rmssd_over_mean_rr(rr).unwrap_or(f64::NAN),
        shannon_entropy_drr(rr).unwrap_or(f64::NAN),
        turning_point_ratio(rr).unwrap_or(f64::NAN),
        sample_entropy(rr, COSEN_M, COSEN_R_MS).unwrap_or(f64::NAN),
        cosen(rr).unwrap_or(f64::NAN),
        p.map_or(f64::NAN, |p| p.sd1),
        p.map_or(f64::NAN, |p| p.normalised_area),
        p.map_or(f64::NAN, |p| p.cell_occupancy),
        e.map_or(f64::NAN, |e| e.origin_fraction),
        e.map_or(f64::NAN, |e| e.alternation_fraction),
        e.map_or(f64::NAN, |e| e.ectopic_fraction),
        e.and_then(|e| e.residual_sample_entropy).unwrap_or(f64::NAN),
        // How far the entropy falls once the premature beats are gone.
        e.map_or(f64::NAN, |e| match (e.sample_entropy, e.residual_sample_entropy) {
            (Some(a), Some(b)) => a - b,
            _ => f64::NAN,
        }),
        e.and_then(|e| e.residual_cosen).unwrap_or(f64::NAN),
    ]
}

const HEADS: [&str; 14] = [
    "rmssd/mn", "entropy", "tpr", "sampen", "cosen", "sd1", "narea", "occ", "origin", "altern",
    "ectopic", "residSE", "SEdrop", "residCOS",
];

/// Per class: segment count, the median of every index, and the median premature-beat share.
fn per_class(segs: &[Segment]) {
    let mut by: BTreeMap<Class, Vec<&Segment>> = BTreeMap::new();
    for s in segs {
        by.entry(s.class).or_default().push(s);
    }
    print!("{:<18} {:>6} {:>5}", "class", "segs", "ectop");
    for h in HEADS {
        print!(" {h:>7}");
    }
    println!();
    for (class, list) in &by {
        let rows: Vec<[f64; 14]> = list.iter().map(|s| row(&s.rr)).collect();
        let ectopy: Vec<f64> = list.iter().map(|s| s.premature_fraction).collect();
        print!("{:<18} {:>6} {:>5.3}", format!("{class:?}"), list.len(), median(&ectopy));
        for k in 0..HEADS.len() {
            print!(" {:>7}", fmt(&column(&rows, k)));
        }
        println!();
    }
}

fn column(rows: &[[f64; 14]], k: usize) -> Vec<f64> {
    rows.iter().map(|r| r[k]).filter(|x| x.is_finite()).collect()
}

fn fmt(v: &[f64]) -> String {
    if v.is_empty() {
        "-".into()
    } else {
        format!("{:.3}", median(v))
    }
}

/// Write the distilled fixture: a fixed stride through each class, so the committed subset is a sample of
/// the corpus rather than a selection from it.
fn emit_fixture(segs: &[Segment], dir: &Path) {
    fs::create_dir_all(dir).expect("fixture dir");
    let mut by: BTreeMap<Class, Vec<&Segment>> = BTreeMap::new();
    for s in segs {
        by.entry(s.class).or_default().push(s);
    }
    for (class, list) in &by {
        // Mixed and Paced belong to neither arm, so a gate cannot say anything about them.
        if matches!(class, Class::Mixed | Class::Paced) {
            continue;
        }
        let want = 60usize.min(list.len());
        let stride = (list.len() / want).max(1);
        let taken: Vec<&&Segment> = list.iter().step_by(stride).take(want).collect();
        let mut out = String::new();
        out.push_str("# physionet r-r rhythm segments, distilled\n");
        out.push_str("# source PhysioNet afdb / nsrdb / mitdb, ODC-BY 1.0\n");
        out.push_str(&format!("# class {class:?}\n"));
        out.push_str(&format!("# segment_beats {}\n", taken.first().map_or(0, |s| s.rr.len())));
        out.push_str(&format!("# selection every {stride}th of {} segments in this class\n", list.len()));
        out.push_str("# columns record start_beat premature_fraction rr_ms...\n");
        out.push_str("# lead chest, ambulatory Holter ECG - NOT wrist PPG\n");
        for s in &taken {
            out.push_str(&format!("{} {} {:.4}", s.record, s.start_beat, s.premature_fraction));
            for v in &s.rr {
                out.push_str(&format!(" {v}"));
            }
            out.push('\n');
        }
        let path = dir.join(format!("{}.txt", format!("{class:?}").to_lowercase()));
        fs::write(&path, out).expect("write fixture");
        println!("wrote {} ({} of {} segments)", path.display(), taken.len(), list.len());
    }
}
