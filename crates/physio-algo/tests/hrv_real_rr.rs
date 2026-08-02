//! Acceptance gate for the displayed HRV numbers — nightly RMSSD, session avgHrv, SDNN, pNN50, meanNN
//! — over 500 labelled 256-beat R-R stretches (128,000 real intervals) instead of hand-made beats.
//!
//! The `hrv.rs` unit gates pin the same numbers on 5-8 synthetic beats computed by hand. The range
//! filter never fires on that input and its one ectopic beat clears `ECTOPIC_THRESHOLD` threefold, so
//! `RR_MIN_MS`, `RR_MAX_MS` and `ECTOPIC_THRESHOLD` are unreachable from them. On this corpus both
//! filters fire, and `the_cleaning_chain_is_actually_reached_by_this_corpus` asserts that they do.
//!
//! Scope. The stretches are the tracked, openly licensed PhysioNet set `rr_irregularity_rhythm.rs`
//! uses (afdb / nsrdb / mitdb, ODC-BY 1.0). They are chest-lead ambulatory ECG, not wrist PPG, and
//! carry no wall-clock, so beats are stamped by cumulative R-R time: this pins the cleaning and
//! formula chain over physiological beat sequences, never the strap's optical front end.
//!
//! Every number here is a wellness estimate, never medical or diagnostic. The fixtures are TRACKED and
//! this file PANICS when they are missing: an `assume`-style skip on an absent fixture reports a PASS.
//!
//!   cargo test -p physio-algo --test hrv_real_rr -- --nocapture

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use physio_algo::hrv::{clean_counts, HrvReadiness};

/// One labelled stretch of consecutive R-R intervals (ms).
struct Stretch {
    class: String,
    rr: Vec<u16>,
}

/// The failure message for an absent corpus: these fixtures are tracked, so a miss means an incomplete
/// checkout, and this test has no skip path.
fn unusable(dir: &Path, why: &str) -> String {
    format!(
        "rhythm R-R fixture directory unusable: {} ({why}).\n\
         It is TRACKED, so a clean checkout carries it - restore it from git rather than skipping.\n\
         To rebuild from source: `python tools/physionet_to_fixture.py fetch` then `convert`, then \
         `cargo run --release -p physio-algo --example rr_rhythm_corpus -- --emit {}`.",
        dir.display(),
        dir.display()
    )
}

fn fixtures() -> Vec<Stretch> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/rhythm_rr");
    let entries = fs::read_dir(&dir).unwrap_or_else(|e| panic!("{}", unusable(&dir, &e.to_string())));
    let mut paths: Vec<PathBuf> = entries
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "txt"))
        .collect();
    paths.sort();
    assert!(!paths.is_empty(), "{}", unusable(&dir, "it holds no .txt fixture"));
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
            // columns: record start_beat premature_fraction rr_ms...
            let mut it = line.split_whitespace();
            let (Some(_record), Some(_start), Some(_prem)) = (it.next(), it.next(), it.next()) else {
                continue;
            };
            out.push(Stretch { class: class.clone(), rr: it.map(|v| v.parse().unwrap()).collect() });
        }
    }
    out
}

/// The classes, in the order every table below prints them.
const CLASSES: [&str; 10] = [
    "AfibAudited",
    "AfibMachine",
    "EctopyBigeminal",
    "EctopyHeavy",
    "EctopyLight",
    "Flutter",
    "Junctional",
    "SinusAfdb",
    "SinusMit",
    "SinusNsr",
];

/// Group a stretch into per-second reports by cumulative beat time, the shape
/// [`HrvReadiness::rmssd_gap_aware`] and the nightly path take. A beat is filed under the second its
/// R-R interval ENDS in, which is how a strap that reports once a second would file it.
fn per_second_reports(rr: &[u16]) -> Vec<(u32, Vec<u16>)> {
    let mut out: Vec<(u32, Vec<u16>)> = Vec::new();
    let mut cumulative_ms: u64 = 0;
    for &v in rr {
        cumulative_ms += u64::from(v);
        let sec = (cumulative_ms / 1000) as u32;
        match out.last_mut() {
            Some((t, beats)) if *t == sec => beats.push(v),
            _ => out.push((sec, vec![v])),
        }
    }
    out
}

/// The same stamping, flattened to the `(unix, rr)` pairs [`HrvReadiness::windowed_avg_hrv`] takes.
fn stamped(rr: &[u16]) -> Vec<(u32, u16)> {
    per_second_reports(rr).into_iter().flat_map(|(t, b)| b.into_iter().map(move |v| (t, v))).collect()
}

/// Median of a slice, by value. Panics on an empty slice: an empty class is a corpus fault, not a
/// number to paper over.
fn median(xs: &[f64]) -> f64 {
    assert!(!xs.is_empty(), "median of an empty sample");
    let mut v = xs.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 1 { v[n / 2] } else { (v[n / 2 - 1] + v[n / 2]) / 2.0 }
}

/// Every displayed HRV number for one class. The four per-stretch numbers are medians over that
/// class's stretches; `avg_hrv` is the one number a single 256-beat stretch cannot produce, so it is
/// measured over the class's stretches laid end to end (see [`class_avg_hrv`]).
struct ClassHrv {
    stretches: usize,
    beats: usize,
    /// Survivors of the 300-2000 ms range filter, as a share of the input beats.
    ranged_share: f64,
    /// Survivors of range + Malik ectopic, as a share of the input beats.
    clean_share: f64,
    /// Nightly-path gap-aware RMSSD (ms) — `hrv_rmssd_gap_aware` across the FFI.
    rmssd: f64,
    /// Run-variant RMSSD (ms) — `hrv_rmssd`, the only path that reaches `MAX_BEAT_DELTA_MS`.
    rmssd_run: f64,
    /// Uncleaned RMSSD (ms) — `hrv_rmssd_plain`, the rolling-chart and stress-onset input.
    rmssd_plain: f64,
    /// Session avgHrv (ms): the mean of per-300 s-bucket gap-aware RMSSD over the class timeline.
    avg_hrv: f64,
    /// Deep-sleep-windowed avgHrv (ms) over the same timeline with alternate buckets marked deep.
    avg_hrv_deep: f64,
    /// How many 300 s buckets contributed to `avg_hrv`.
    buckets: usize,
    sdnn: f64,
    pnn50: f64,
    mean_nn: f64,
    /// Stretches where `analyze_raw` refused (under MIN_BEATS clean survivors).
    refused: usize,
}

/// Session avgHrv over a whole class, and how many buckets scored. One 256-beat stretch spans ~200 s
/// against a 300 s bucket, so it yields a single bucket and `windowed_avg_hrv` degenerates to
/// `rmssd_gap_aware`; laying the class end to end is what exercises the multi-bucket mean.
fn class_avg_hrv(mine: &[&Stretch]) -> (f64, f64, usize) {
    let concatenated: Vec<u16> = mine.iter().flat_map(|s| s.rr.iter().copied()).collect();
    let beats = stamped(&concatenated);
    let (start, end) = (beats.first().unwrap().0, beats.last().unwrap().0);
    let all = HrvReadiness::windowed_buckets(start, end, &beats);
    let buckets = all.iter().filter(|b| b.rmssd.is_some()).count();
    // Alternate buckets stand in for deep-sleep spans: the deep path's own job is to KEEP some buckets
    // and drop others, and a synthetic every-other mask exercises exactly that without inventing a
    // sleep claim about ambulatory ECG.
    let deep: Vec<(u32, u32)> =
        all.iter().enumerate().filter(|(i, _)| i % 2 == 0).map(|(_, b)| (b.start, b.start + 300)).collect();
    (
        HrvReadiness::windowed_avg_hrv(start, end, &beats).expect("a class timeline yields buckets"),
        HrvReadiness::windowed_avg_hrv_deep(start, end, &beats, &deep).expect("alternate buckets are deep"),
        buckets,
    )
}

/// Every displayed HRV number over one set of stretches.
fn measure(mine: &[&Stretch]) -> ClassHrv {
    let (mut n_input, mut n_ranged, mut n_clean) = (0usize, 0usize, 0usize);
    let (mut rmssd, mut rmssd_run, mut rmssd_plain) = (Vec::new(), Vec::new(), Vec::new());
    let (mut sdnn, mut pnn50, mut mean_nn) = (Vec::new(), Vec::new(), Vec::new());
    let mut refused = 0usize;
    for s in mine {
        let c = clean_counts(&s.rr);
        n_input += c.n_input as usize;
        n_ranged += c.n_ranged as usize;
        n_clean += c.n_clean as usize;

        if let Some(v) = HrvReadiness::rmssd_gap_aware(&per_second_reports(&s.rr)) {
            rmssd.push(v);
        }
        if let Some(v) = HrvReadiness::rmssd(&s.rr) {
            rmssd_run.push(v);
        }
        if let Some(v) = HrvReadiness::rmssd_plain(&s.rr) {
            rmssd_plain.push(v);
        }
        // The nightly/session path, so no spot rejected-fraction gate.
        let a = HrvReadiness::analyze_raw(&s.rr, None);
        match (a.sdnn, a.pnn50, a.mean_nn) {
            (Some(sd), Some(p), Some(m)) => {
                sdnn.push(sd);
                pnn50.push(p);
                mean_nn.push(m);
            }
            _ => refused += 1,
        }
    }
    let (avg_hrv, avg_hrv_deep, buckets) = class_avg_hrv(mine);
    ClassHrv {
        stretches: mine.len(),
        beats: n_input,
        ranged_share: n_ranged as f64 / n_input as f64,
        clean_share: n_clean as f64 / n_input as f64,
        rmssd: median(&rmssd),
        rmssd_run: median(&rmssd_run),
        rmssd_plain: median(&rmssd_plain),
        avg_hrv,
        avg_hrv_deep,
        buckets,
        sdnn: median(&sdnn),
        pnn50: median(&pnn50),
        mean_nn: median(&mean_nn),
        refused,
    }
}

/// One row per class plus `ALL`, the whole-corpus row every other gate quotes.
fn per_class() -> BTreeMap<String, ClassHrv> {
    let all = fixtures();
    let mut out = BTreeMap::new();
    for class in CLASSES {
        let mine: Vec<&Stretch> = all.iter().filter(|s| s.class == class).collect();
        assert!(!mine.is_empty(), "class {class} missing from the fixtures");
        out.insert(class.to_string(), measure(&mine));
    }
    out.insert("ALL".to_string(), measure(&all.iter().collect::<Vec<_>>()));
    out
}

#[test]
fn print_every_displayed_hrv_number_per_class() {
    println!("every number is a wellness estimate, never medical or diagnostic");
    println!(
        "{:<16} {:>4} {:>7} {:>8} {:>8} {:>10} {:>9} {:>9} {:>10} {:>10} {:>5} {:>10} {:>8} {:>9} {:>4}",
        "class", "n", "beats", "ranged", "clean", "rmssd", "rmssdRun", "rmssdPln", "avgHrv", "avgHrvDp",
        "bkts", "sdnn", "pnn50", "meanNN", "ref"
    );
    for (class, c) in per_class() {
        println!(
            "{:<16} {:>4} {:>7} {:>8.5} {:>8.5} {:>10.7} {:>9.6} {:>9.6} {:>10.7} {:>10.7} {:>5} {:>10.7} {:>8.5} {:>9.4} {:>4}",
            class,
            c.stretches,
            c.beats,
            c.ranged_share,
            c.clean_share,
            c.rmssd,
            c.rmssd_run,
            c.rmssd_plain,
            c.avg_hrv,
            c.avg_hrv_deep,
            c.buckets,
            c.sdnn,
            c.pnn50,
            c.mean_nn,
            c.refused
        );
    }
}

/// Re-emits the `PINS` table below at full f64 precision, so a deliberate re-measure is a paste rather
/// than a transcription. Never a substitute for reading WHY a number moved.
#[test]
#[ignore]
fn emit_the_pins_table() {
    for (class, c) in per_class() {
        println!(
            "        Pin {{ class: \"{}\", rmssd: {:?}, rmssd_run: {:?}, rmssd_plain: {:?}, avg_hrv: {:?}, avg_hrv_deep: {:?}, sdnn: {:?}, pnn50: {:?}, mean_nn: {:?}, buckets: {}, refused: {} }},",
            class, c.rmssd, c.rmssd_run, c.rmssd_plain, c.avg_hrv, c.avg_hrv_deep, c.sdnn, c.pnn50,
            c.mean_nn, c.buckets, c.refused
        );
    }
}

#[test]
fn the_cohort_is_the_one_these_numbers_were_measured_on() {
    // Asserting the cohort is half the gate. A fixture file silently truncated, or one class dropped,
    // moves every median below without failing any of them by itself.
    let all = fixtures();
    assert_eq!(all.len(), 500, "stretch count changed: the pinned medians below are not comparable");
    let beats: usize = all.iter().map(|s| s.rr.len()).sum();
    assert_eq!(beats, 128_000, "beat count changed: {beats}");
    assert!(all.iter().all(|s| s.rr.len() == 256), "every stretch is 256 beats by construction");
    let classes: std::collections::BTreeSet<&str> = all.iter().map(|s| s.class.as_str()).collect();
    assert_eq!(classes.len(), CLASSES.len(), "class count changed: {classes:?}");
    for c in CLASSES {
        assert!(classes.contains(c), "class {c} missing");
    }
}

#[test]
fn the_cleaning_chain_is_actually_reached_by_this_corpus() {
    // The claim the synthetic gates cannot make. If either filter never fires, its constants are
    // unreachable from here too and the pins below would be worth no more than the hand-made ones.
    let per = per_class();
    let per: BTreeMap<String, ClassHrv> = per.into_iter().filter(|(k, _)| k != "ALL").collect();
    let ranged_drop: f64 = 1.0 - per.values().map(|c| c.ranged_share * c.beats as f64).sum::<f64>()
        / per.values().map(|c| c.beats as f64).sum::<f64>();
    let ectopic_drop: f64 = per.values().map(|c| (c.ranged_share - c.clean_share) * c.beats as f64).sum::<f64>()
        / per.values().map(|c| c.beats as f64).sum::<f64>();
    println!("corpus-wide: range filter drops {ranged_drop:.5}, Malik ectopic drops {ectopic_drop:.5}");
    assert!(
        ranged_drop > 0.0,
        "the 300-2000 ms range filter never fires on this corpus, so RR_MIN_MS / RR_MAX_MS stay unreachable"
    );
    assert!(
        ectopic_drop > 0.01,
        "the Malik ectopic filter drops only {ectopic_drop:.5} of beats, too little to pin ECTOPIC_THRESHOLD"
    );
    // And it must fire on the class named after the thing it removes, not only on fibrillation.
    let bigeminal = &per["EctopyBigeminal"];
    assert!(
        bigeminal.ranged_share - bigeminal.clean_share > 0.10,
        "Malik drops only {:.4} of bigeminal beats",
        bigeminal.ranged_share - bigeminal.clean_share
    );
}

/// One class's pinned row. Re-emit the whole table with `-- --ignored emit_the_pins_table`.
struct Pin {
    class: &'static str,
    rmssd: f64,
    rmssd_run: f64,
    rmssd_plain: f64,
    avg_hrv: f64,
    avg_hrv_deep: f64,
    sdnn: f64,
    pnn50: f64,
    mean_nn: f64,
    buckets: usize,
    refused: usize,
}

#[test]
fn every_displayed_hrv_number_reproduces_on_real_rr() {
    // Measured 2026-08-02 on the 500-stretch cohort asserted above. Tolerance 1e-9: these are
    // deterministic replays of committed integers, not a sample statistic that may drift.
    const PINS: &[Pin] = &[
        Pin { class: "ALL", rmssd: 35.48921280381816, rmssd_run: 47.18823933995884, rmssd_plain: 126.74649997842525, avg_hrv: 46.74367726233443, avg_hrv_deep: 46.3476203874652, sdnn: 42.945050875993616, pnn50: 11.76470588235294, mean_nn: 692.0, buckets: 313, refused: 1 },
        Pin { class: "AfibAudited", rmssd: 77.66265573129787, rmssd_run: 101.73720978095771, rmssd_plain: 213.35059979145254, avg_hrv: 80.23198132531199, avg_hrv_deep: 79.08787364192699, sdnn: 84.18050975347695, pnn50: 53.48837209302325, mean_nn: 681.1635514018692, buckets: 17, refused: 0 },
        Pin { class: "AfibMachine", rmssd: 75.8270422796061, rmssd_run: 101.5156845821151, rmssd_plain: 196.8789499045669, avg_hrv: 85.37605237320112, avg_hrv_deep: 84.46221803859333, sdnn: 100.11396067519445, pnn50: 50.0, mean_nn: 624.4782608695652, buckets: 35, refused: 0 },
        Pin { class: "EctopyBigeminal", rmssd: 56.39296013739629, rmssd_run: 80.58523972238686, rmssd_plain: 296.7910779302523, avg_hrv: 64.31338175167826, avg_hrv_deep: 62.625348736126874, sdnn: 65.33625350253318, pnn50: 30.0, mean_nn: 686.0869565217391, buckets: 22, refused: 1 },
        Pin { class: "EctopyHeavy", rmssd: 52.184991916210585, rmssd_run: 80.83025802411433, rmssd_plain: 238.10458239366739, avg_hrv: 55.56521107510068, avg_hrv_deep: 57.50787746164652, sdnn: 49.2316463406298, pnn50: 27.43362831858407, mean_nn: 680.873417721519, buckets: 25, refused: 0 },
        Pin { class: "EctopyLight", rmssd: 34.210331027993746, rmssd_run: 41.948618273477095, rmssd_plain: 98.93443483049225, avg_hrv: 35.612092991858276, avg_hrv_deep: 34.6969986493109, sdnn: 32.19357078579205, pnn50: 8.523740221939239, mean_nn: 746.7638561560341, buckets: 39, refused: 0 },
        Pin { class: "Flutter", rmssd: 25.907826899998263, rmssd_run: 31.421363145512174, rmssd_plain: 37.0770289933122, avg_hrv: 27.08543886866652, avg_hrv_deep: 22.24229733743721, sdnn: 18.786277950863717, pnn50: 7.363188134142862, mean_nn: 413.3500919681177, buckets: 16, refused: 0 },
        Pin { class: "Junctional", rmssd: 27.37797434087289, rmssd_run: 72.03886194570231, rmssd_plain: 240.8477215557523, avg_hrv: 30.591102806484248, avg_hrv_deep: 31.273434052241328, sdnn: 21.52991497321612, pnn50: 6.323884860470226, mean_nn: 647.0356895356895, buckets: 34, refused: 0 },
        Pin { class: "SinusAfdb", rmssd: 22.907723565900177, rmssd_run: 23.70594366248632, rmssd_plain: 72.31762299226057, avg_hrv: 32.18714277440019, avg_hrv_deep: 31.69261468343763, sdnn: 29.362081512515, pnn50: 1.9607843137254901, mean_nn: 867.597433299561, buckets: 43, refused: 0 },
        Pin { class: "SinusMit", rmssd: 32.281904491870065, rmssd_run: 32.718340979482505, rmssd_plain: 41.61002700024646, avg_hrv: 41.27370768089851, avg_hrv_deep: 40.233153010867234, sdnn: 37.142797429527114, pnn50: 10.196078431372548, mean_nn: 865.384765625, buckets: 46, refused: 0 },
        Pin { class: "SinusNsr", rmssd: 29.04121882828393, rmssd_run: 29.04121882828393, rmssd_plain: 31.670738940587995, avg_hrv: 37.5599441677424, avg_hrv_deep: 40.94316766147781, sdnn: 55.423508432211335, pnn50: 8.431372549019606, mean_nn: 779.423828125, buckets: 41, refused: 0 },
    ];
    assert_eq!(PINS.len(), CLASSES.len() + 1, "every class plus the ALL row must be pinned");
    let per = per_class();
    for p in PINS {
        let c = &per[p.class];
        let close = |got: f64, want: f64, what: &str| {
            assert!((got - want).abs() < 1e-9, "{} {what}: got {got:.12}, pinned {want:.12}", p.class);
        };
        close(c.rmssd, p.rmssd, "nightly RMSSD (gap-aware)");
        close(c.rmssd_run, p.rmssd_run, "run-variant RMSSD");
        close(c.rmssd_plain, p.rmssd_plain, "plain RMSSD");
        close(c.avg_hrv, p.avg_hrv, "session avgHrv");
        close(c.avg_hrv_deep, p.avg_hrv_deep, "deep-window avgHrv");
        close(c.sdnn, p.sdnn, "SDNN");
        close(c.pnn50, p.pnn50, "pNN50");
        close(c.mean_nn, p.mean_nn, "meanNN");
        assert_eq!(c.buckets, p.buckets, "{} scoring 300 s buckets", p.class);
        assert_eq!(c.refused, p.refused, "{} stretches analyze_raw refused", p.class);
    }
}
