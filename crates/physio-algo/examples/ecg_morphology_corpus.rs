//! Print the three ECG morphology measures for every labelled waveform window in the PhysioNet
//! corpus, grouped by the rhythm the recording is annotated with, so the separation between them is a
//! measurement rather than a claim.
//!
//! ```text
//! cargo run --release -p physio-algo --example ecg_morphology_corpus -- [DATASETS_DIR] [--rows]
//! ```
//!
//! Windows are the 60 s fixtures under `physionet-{afdb,nsrdb,mitdb}/raw/`, one integer sample per
//! line under `# key value` headers. The corpus lives outside this repo, so this is an example and not
//! a test: a test pointing at it would pass by skipping on a clean checkout, which this project has
//! been bitten by before.
//!
//! Every recording is chest or handheld ECG. NONE is wrist PPG or a wrist electrode, so a separation
//! here is necessary and not sufficient — it says nothing about the noise a wrist front end adds.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use physio_algo::ecg::morphology::{morphology, AtrialBandEvidence, BeatConsistency, PWaveFinding};
use physio_algo::ecg::detect_pan_tompkins;
use physio_algo::stats::median;

const DEFAULT_DATASETS: &str = "../whoop-data/datasets";
const DATABASES: [&str; 3] = ["physionet-afdb", "physionet-nsrdb", "physionet-mitdb"];

struct Window {
    database: String,
    record: String,
    rhythm: String,
    fs_hz: f64,
    samples: Vec<f64>,
}

/// One window's measures, flattened for the per-class medians. A measure that did not compute is left
/// out of its column rather than filled with a stand-in.
struct Row {
    finding: String,
    present_fraction: Option<f64>,
    consistency: Option<f64>,
    amplitude_ratio: Option<f64>,
    noise_ratio: Option<f64>,
    atrial_ratio: Option<f64>,
    segment_ms: Option<f64>,
    beat_consistency: Option<f64>,
    beats: usize,
}

fn main() {
    let mut args = std::env::args().skip(1);
    let mut root = PathBuf::from(DEFAULT_DATASETS);
    let mut per_row = false;
    for a in args.by_ref() {
        if a == "--rows" {
            per_row = true;
        } else {
            root = PathBuf::from(a);
        }
    }

    let windows = load(&root);
    if windows.is_empty() {
        eprintln!("no windows under {}", root.display());
        std::process::exit(2);
    }
    println!("{} windows from {}", windows.len(), root.display());

    let mut by_class: BTreeMap<String, Vec<Row>> = BTreeMap::new();
    if per_row {
        println!(
            "\n{:>16} {:>7} {:>6} {:>5} {:>22} {:>7} {:>7} {:>7} {:>8} {:>7} {:>7}",
            "record", "rhythm", "fs", "beats", "p-wave", "frac", "consis", "amp/R", "noise/R", "atrial", "tmpl"
        );
    }
    for w in &windows {
        let peaks = detect_pan_tompkins(&w.samples, w.fs_hz);
        let m = morphology(&w.samples, w.fs_hz, &peaks);
        let finding = describe(&m.p_wave.finding);
        let (atrial_ratio, segment_ms) = match m.atrial_band {
            AtrialBandEvidence::Measured(a) => (Some(a.ratio), Some(a.median_segment_ms)),
            AtrialBandEvidence::Indeterminate(_) => (None, None),
        };
        let beat_consistency = match m.beat_consistency {
            BeatConsistency::Measured { correlation, .. } => Some(correlation),
            BeatConsistency::Indeterminate => None,
        };
        let row = Row {
            finding,
            present_fraction: m.p_wave.present_fraction,
            consistency: m.p_wave.consistency,
            amplitude_ratio: m.p_wave.amplitude_ratio,
            noise_ratio: m.p_wave.noise_ratio,
            atrial_ratio,
            segment_ms,
            beat_consistency,
            beats: peaks.len(),
        };
        if per_row {
            println!(
                "{:>16} {:>7} {:>6.0} {:>5} {:>22} {} {} {} {} {} {}",
                format!("{}/{}", short_db(&w.database), w.record),
                w.rhythm,
                w.fs_hz,
                row.beats,
                row.finding,
                cell(row.present_fraction, 7, 3),
                cell(row.consistency, 7, 3),
                cell(row.amplitude_ratio, 7, 3),
                cell(row.noise_ratio, 8, 4),
                cell(row.atrial_ratio, 7, 3),
                cell(row.beat_consistency, 7, 3),
            );
        }
        by_class.entry(class_of(w)).or_default().push(row);
    }

    println!("\n== P-wave finding, by annotated rhythm ==");
    println!("{:>22} {:>7} {:>9} {:>8} {:>8} {:>7} {:>7} {:>8}", "class", "windows", "present", "absent", "indet", "frac", "amp/R", "noise/R");
    for (class, rows) in &by_class {
        let n = rows.len();
        let present = rows.iter().filter(|r| r.finding == "Present").count();
        let absent = rows.iter().filter(|r| r.finding == "Absent").count();
        println!(
            "{class:>22} {n:>7} {present:>9} {absent:>8} {:>8} {} {} {}",
            n - present - absent,
            cell(med(rows, |r| r.present_fraction), 7, 3),
            cell(med(rows, |r| r.amplitude_ratio), 7, 3),
            cell(med(rows, |r| r.noise_ratio), 8, 4),
        );
    }

    println!("\n== why a window was indeterminate ==");
    let mut reasons: BTreeMap<(String, String), usize> = BTreeMap::new();
    for (class, rows) in &by_class {
        for r in rows.iter().filter(|r| r.finding != "Present" && r.finding != "Absent") {
            *reasons.entry((class.clone(), r.finding.clone())).or_default() += 1;
        }
    }
    for ((class, reason), n) in &reasons {
        println!("{class:>22} {reason:>22} {n:>4}");
    }

    println!("\n== atrial band (4-9 Hz share of the TP segment) and beat-template consistency ==");
    println!("{:>22} {:>7} {:>9} {:>10} {:>10} {:>8}", "class", "windows", "measured", "ratio", "segment ms", "tmpl");
    for (class, rows) in &by_class {
        let measured = rows.iter().filter(|r| r.atrial_ratio.is_some()).count();
        println!(
            "{class:>22} {:>7} {measured:>9} {} {} {}",
            rows.len(),
            cell(med(rows, |r| r.atrial_ratio), 10, 4),
            cell(med(rows, |r| r.segment_ms), 10, 0),
            cell(med(rows, |r| r.beat_consistency), 8, 3),
        );
    }
}

/// The class a window is reported under: its own rhythm annotation where it has one. nsrdb carries no
/// rhythm markers at all, so its windows are labelled at the DATABASE level and named that way rather
/// than being folded in with annotated normal sinus.
fn class_of(w: &Window) -> String {
    if w.rhythm == "?" {
        format!("{} (db label)", short_db(&w.database))
    } else {
        format!("{} {}", short_db(&w.database), w.rhythm)
    }
}

fn short_db(db: &str) -> &str {
    db.strip_prefix("physionet-").unwrap_or(db)
}

/// `Present` / `Absent`, or the name of the limit that made the finding indeterminate.
fn describe(f: &PWaveFinding) -> String {
    match f {
        PWaveFinding::Present => "Present".to_string(),
        PWaveFinding::Absent => "Absent".to_string(),
        PWaveFinding::Indeterminate(l) => {
            format!("{l:?}").split([' ', '{']).next().unwrap_or("?").trim().to_string()
        }
    }
}

fn med(rows: &[Row], pick: impl Fn(&Row) -> Option<f64>) -> Option<f64> {
    let v: Vec<f64> = rows.iter().filter_map(pick).collect();
    (!v.is_empty()).then(|| median(&v))
}

fn cell(v: Option<f64>, width: usize, places: usize) -> String {
    match v {
        Some(x) => format!("{x:>width$.places$}"),
        None => format!("{:>width$}", "-"),
    }
}

fn load(root: &Path) -> Vec<Window> {
    let mut out = Vec::new();
    for db in DATABASES {
        let dir = root.join(db).join("raw");
        let Ok(entries) = fs::read_dir(&dir) else {
            eprintln!("skipping {}: not readable", dir.display());
            continue;
        };
        let mut paths: Vec<PathBuf> =
            entries.filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.extension().is_some_and(|x| x == "txt")).collect();
        paths.sort();
        out.extend(paths.iter().map(|p| read_window(db, p)));
    }
    out
}

fn read_window(database: &str, path: &Path) -> Window {
    let text = fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let (mut fs_hz, mut scale, mut record, mut rhythm) = (0.0f64, 1.0f64, String::new(), "?".to_string());
    let mut samples = Vec::new();
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix('#') {
            let mut it = rest.split_whitespace();
            match (it.next(), it.next()) {
                (Some("fs_hz"), Some(v)) => fs_hz = v.parse().unwrap(),
                (Some("scale"), Some(v)) => scale = v.parse().unwrap(),
                (Some("record"), Some(v)) => record = v.to_string(),
                (Some("rhythm"), Some(v)) => rhythm = v.to_string(),
                _ => {}
            }
        } else if !line.trim().is_empty() {
            samples.push(line.trim().parse::<f64>().unwrap() * scale);
        }
    }
    assert!(fs_hz > 0.0 && !samples.is_empty(), "unusable window {}", path.display());
    Window { database: database.to_string(), record, rhythm, fs_hz, samples }
}
