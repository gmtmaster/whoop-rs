//! Reader for the PhysioNet R-R fixtures, and the rhythm class of one segment.
//!
//! Fixture lines are `rr_ms beat rhythm`, `#` headers. `beat` is the symbol of the beat that ENDS the
//! interval and `rhythm` the rhythm token in force at it. Beat times are the cumulative interval sum, so
//! the whole-second stamp a segment carries is the beat's own time and not an invented one.

use std::fs;
use std::path::Path;

/// WFDB symbols for a beat that arrives EARLY. These, not the rhythm token, are what an irregularity
/// index sees as a burst of scatter inside an otherwise sinus segment.
pub const PREMATURE_SYMBOLS: [&str; 5] = ["A", "a", "J", "S", "V"];

/// The rhythm class of one segment. `Afib*` are the sustained irregular positives; `Sinus*` and
/// `Ectopy*` are the negatives, split by how many of their beats arrive early, because that share — not
/// the rhythm token — is what drives the false positives.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Class {
    /// mitdb atrial fibrillation, beats audited by the database's cardiologists.
    AfibAudited,
    /// afdb atrial fibrillation, beats from an unaudited automated detector.
    AfibMachine,
    /// afdb atrial flutter — regular by conduction, and the honest false negative of this whole approach.
    Flutter,
    /// afdb junctional rhythm.
    Junctional,
    /// nsrdb, database-level normal sinus, no rhythm markers of its own.
    SinusNsr,
    /// afdb sinus spans; its beats are machine-detected, so its premature share is not measurable.
    SinusAfdb,
    /// mitdb sinus under [`ECTOPY_CLEAN`] premature beats.
    SinusMit,
    /// Occasional premature beats — the everyday healthy case, and the commonest false positive.
    EctopyLight,
    /// A heavy burst of premature beats.
    EctopyHeavy,
    /// Two beats in five or more arrive early: bigeminy and trigeminy, the extreme of alternating ectopy.
    EctopyBigeminal,
    /// Paced rhythm: regular by pacemaker, and not a sinus negative.
    Paced,
    /// Spans two rhythms, or is labelled something else; reported, never scored as either arm.
    Mixed,
}

impl Class {
    /// A sustained irregular rhythm — what the screen is meant to catch.
    pub fn is_positive(self) -> bool {
        matches!(self, Class::AfibAudited | Class::AfibMachine)
    }

    /// A rhythm the screen must NOT report. Flutter is in neither arm: it is a real irregular-rhythm
    /// condition this approach cannot see, so counting it either way would flatter the result. Paced and
    /// junctional are likewise excluded — neither is the population this screen runs on.
    pub fn is_negative(self) -> bool {
        matches!(
            self,
            Class::SinusNsr
                | Class::SinusAfdb
                | Class::SinusMit
                | Class::EctopyLight
                | Class::EctopyHeavy
                | Class::EctopyBigeminal
        )
    }
}

/// One scored segment of a record.
// Shared by several example binaries; each reads a different subset, so dead-code is per-target here.
#[allow(dead_code)]
pub struct Segment {
    pub record: String,
    pub class: Class,
    pub start_beat: usize,
    /// R-R in whole ms, rounded the way the strap delivers them.
    pub rr: Vec<u16>,
    /// Intervals ending on a [`PREMATURE_SYMBOLS`] beat, over intervals.
    pub premature_fraction: f64,
}

struct Row {
    rr_ms: f64,
    beat: String,
    rhythm: String,
}

/// One record, its windows in recording order — the form the episode evaluation needs, since a run of
/// firing windows is only meaningful inside one continuous recording.
pub struct Record {
    pub name: String,
    pub windows: Vec<Segment>,
}

/// Every non-overlapping `segment_beats`-length segment of one record file; a trailing partial segment is
/// dropped so every segment carries the same statistical length.
#[allow(dead_code)]
pub fn segments(path: &Path, segment_beats: usize) -> Vec<Segment> {
    windows(path, segment_beats, segment_beats).windows
}

/// Sliding windows of `window_beats` advancing `step_beats` at a time, in recording order.
pub fn windows(path: &Path, window_beats: usize, step_beats: usize) -> Record {
    let text = fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let (mut db, mut record) = (String::new(), String::new());
    let mut rows: Vec<Row> = Vec::new();
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix('#') {
            let mut it = rest.split_whitespace();
            match (it.next(), it.next()) {
                (Some("database"), Some(v)) => db = v.to_string(),
                (Some("record"), Some(v)) => record = v.to_string(),
                _ => {}
            }
            continue;
        }
        let mut it = line.split_whitespace();
        let (Some(rr), Some(beat), Some(rhythm)) = (it.next(), it.next(), it.next()) else {
            continue;
        };
        rows.push(Row { rr_ms: rr.parse().unwrap(), beat: beat.to_string(), rhythm: rhythm.to_string() });
    }

    // Beat times are the running interval sum, so a window's stamp is the beat's own time.
    let mut clock_ms = vec![0.0f64; rows.len() + 1];
    for (i, r) in rows.iter().enumerate() {
        clock_ms[i + 1] = clock_ms[i] + r.rr_ms;
    }
    let mut out = Vec::new();
    let mut start = 0usize;
    while start + window_beats <= rows.len() {
        let chunk = &rows[start..start + window_beats];
        let rr: Vec<u16> = chunk.iter().map(|r| r.rr_ms.round().clamp(0.0, 65535.0) as u16).collect();
        let premature_fraction = fraction(chunk, |r| PREMATURE_SYMBOLS.contains(&r.beat.as_str()));
        out.push(Segment {
            record: record.clone(),
            class: classify(&db, chunk, premature_fraction),
            start_beat: start,
            rr,
            premature_fraction,
        });
        start += step_beats.max(1);
    }
    Record { name: record, windows: out }
}

fn fraction(rows: &[Row], f: impl Fn(&Row) -> bool) -> f64 {
    rows.iter().filter(|r| f(r)).count() as f64 / rows.len() as f64
}

/// Dominant rhythm token and the share of the segment holding it.
fn dominant(rows: &[Row]) -> (&str, f64) {
    let mut best = ("", 0usize);
    for r in rows {
        let n = rows.iter().filter(|o| o.rhythm == r.rhythm).count();
        if n > best.1 {
            best = (r.rhythm.as_str(), n);
        }
    }
    (best.0, best.1 as f64 / rows.len() as f64)
}

/// Purity a rhythm token must hold across the segment before the segment is called that rhythm.
const PURE: f64 = 0.95;
/// Premature-beat share at or below which a segment counts as clean sinus.
pub const ECTOPY_CLEAN: f64 = 0.02;
/// Share above which occasional ectopy becomes a burst.
pub const ECTOPY_HEAVY: f64 = 0.15;
/// Share at which the ectopy is alternating: bigeminy is 0.5, trigeminy 0.33.
pub const ECTOPY_BIGEMINAL: f64 = 0.30;

/// Class one segment. Ectopy is read from the BEAT symbols, not from the rhythm token: mitdb's bigeminy
/// and trigeminy tokens rarely hold pure across 128 beats, while the premature beats they are made of are
/// counted exactly.
fn classify(db: &str, rows: &[Row], premature: f64) -> Class {
    if fraction(rows, |r| r.beat == "/" || r.beat == "f") > 0.5 {
        return Class::Paced;
    }
    let (token, share) = dominant(rows);
    // Any fibrillation or flutter at all disqualifies a segment from the negative arm.
    let any_af = rows.iter().any(|r| r.rhythm == "AFIB" || r.rhythm == "AFL");
    if db == "nsrdb" {
        return Class::SinusNsr;
    }
    if share >= PURE {
        match (db, token) {
            ("afdb", "AFIB") => return Class::AfibMachine,
            ("mitdb", "AFIB") => return Class::AfibAudited,
            (_, "AFL") => return Class::Flutter,
            ("afdb", "J") => return Class::Junctional,
            _ => {}
        }
    }
    if any_af {
        return Class::Mixed;
    }
    if premature > ECTOPY_BIGEMINAL {
        return Class::EctopyBigeminal;
    }
    if premature > ECTOPY_HEAVY {
        return Class::EctopyHeavy;
    }
    if premature > ECTOPY_CLEAN {
        return Class::EctopyLight;
    }
    match db {
        "afdb" => Class::SinusAfdb,
        "mitdb" => Class::SinusMit,
        _ => Class::Mixed,
    }
}
