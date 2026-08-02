//! Acceptance gate for the irregular-rhythm screen: run it over labelled R-R stretches and check what it
//! reports, what it refuses, and what it is known to miss.
//!
//! The fixtures are 256-beat stretches converted from PhysioNet afdb / nsrdb / mitdb (ODC-BY 1.0) by
//! `examples/rr_rhythm_corpus.rs --emit`, taken at a fixed stride through each class so the committed set
//! is a sample of the corpus rather than a selection from it. They are committed, and this file PANICS if
//! they are missing rather than skipping: an `assume`-style skip on a missing fixture reports a pass, and
//! this project has lost gates that way before.
//!
//! The class counts are pinned exactly, not by a floor. A floor is something a shrinking corpus slides
//! under, and every share below is a ratio whose denominator is one of those counts.
//!
//! Every share here is compared against the SAME acceptance thresholds by [`acceptance`], which is also
//! applied to three do-nothing scorers. A screen that always answers, never answers, or answers on beat
//! scatter alone has to fail it; that is what makes the shipped numbers a result rather than a reading.
//!
//! The full corpus is 382,466 windows over 91 records and lives in `whoop-data`, which is not a git
//! repository. `examples/rr_rhythm_corpus.rs` is the instrument for it; this file is the part that
//! travels with the code.
//!
//!   cargo test -p physio-algo --test rr_irregularity_rhythm -- --nocapture

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use physio_algo::rr_irregularity::screen::MAX_BEAT_GAP_S;
use physio_algo::rr_irregularity::{
    assess, quality, screen, IrregularityReading, Refusal, ScreenState,
};

/// One labelled stretch of consecutive R-R intervals.
struct Stretch {
    class: String,
    record: String,
    premature_fraction: f64,
    rr: Vec<u16>,
}

/// The exact class counts of the committed fixture. Pinned rather than floored: every share below is a
/// ratio over one of these, so a corpus that quietly shrank would move the shares without failing a floor.
const COHORT: [(&str, usize); 10] = [
    ("AfibAudited", 27),
    ("AfibMachine", 60),
    ("EctopyBigeminal", 32),
    ("EctopyHeavy", 39),
    ("EctopyLight", 60),
    ("Flutter", 42),
    ("Junctional", 60),
    ("SinusAfdb", 60),
    ("SinusMit", 60),
    ("SinusNsr", 60),
];

/// The failure message for an absent corpus: these fixtures are tracked, so a miss means an incomplete
/// checkout, and this test has no skip path.
fn unusable(dir: &Path, why: &str) -> String {
    format!(
        "rhythm R-R fixture directory unusable: {} ({why}).\n\
         It is TRACKED, so a clean checkout carries it - restore it from git rather than skipping.\n\
         To rebuild from source: `python tools/physionet_to_fixture.py fetch` then `convert` (pulls \
         afdb / nsrdb / mitdb from https://physionet.org/content/, open access, ODC-BY 1.0), then \
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
            let mut it = line.split_whitespace();
            let (Some(record), Some(_start), Some(prem)) = (it.next(), it.next(), it.next()) else {
                continue;
            };
            out.push(Stretch {
                class: class.clone(),
                record: record.to_string(),
                premature_fraction: prem.parse().unwrap(),
                rr: it.map(|v| v.parse().unwrap()).collect(),
            });
        }
    }
    let mut counted: BTreeMap<&str, usize> = BTreeMap::new();
    for s in &out {
        *counted.entry(COHORT.iter().map(|(c, _)| *c).find(|c| *c == s.class).unwrap_or("UNKNOWN")).or_default() += 1;
    }
    let want: BTreeMap<&str, usize> = COHORT.iter().copied().collect();
    assert_eq!(counted, want, "the fixture cohort changed; every share in this file is a ratio over it");
    assert_eq!(out.len(), COHORT.iter().map(|(_, n)| n).sum::<usize>());
    out
}

/// Seconds between successive beats in the synthetic clock this file stamps its stretches with.
///
/// These stretches never passed through this project's storage, so the input-quality gates inside
/// `assess` would only be measuring a stamp this test invented. Three seconds is the smallest spacing
/// that makes ALL of them inert, which
/// [`the_synthetic_clock_makes_every_input_quality_gate_inert`] measures rather than assumes:
///
/// - over [`quality::RESCALE_LAG_S`], so no beat can be read as a rescaled copy of another;
/// - one beat per second, so no `(second, value)` pair can repeat;
/// - slow enough that beat-time never exceeds elapsed time, which is what one beat a second did NOT
///   achieve: at that spacing any window averaging over ~1,114 ms was refused as impossible coverage;
/// - at or under [`MAX_BEAT_GAP_S`], so `contiguous` still holds and no run is broken by the clock.
const STAMP_STEP_S: u32 = 3;

/// Stamp the stretch on the synthetic clock, leaving the rhythm logic as the only thing under test.
fn stamp(rr: &[u16]) -> Vec<(u32, u16)> {
    rr.iter().enumerate().map(|(i, &v)| (i as u32 * STAMP_STEP_S, v)).collect()
}

fn reported(s: &Stretch) -> bool {
    matches!(screen(&stamp(&s.rr)), ScreenState::IrregularEpisodes { .. })
}

/// Share of each class a scorer reports as an episode.
fn rates_of(score: &dyn Fn(&Stretch) -> bool) -> BTreeMap<String, (usize, usize)> {
    let mut out: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    for s in fixtures() {
        let e = out.entry(s.class.clone()).or_default();
        e.1 += 1;
        e.0 += usize::from(score(&s));
    }
    out
}

/// Share of each class the SHIPPED screen reports as an episode.
fn rates() -> BTreeMap<String, (usize, usize)> {
    rates_of(&reported)
}

/// The one set of acceptance thresholds this file judges anything by. Returns every breach rather than
/// the first, so a scorer that fails both arms is seen to fail both.
///
/// Audited beats and machine beats are separate arms on purpose: afdb's beat times come from an
/// unaudited detector, so agreement across the two is what makes the result a property of the rhythm.
/// The sensitivity floors sit below the 0.772 the full corpus gives because a 256-beat stretch holds 29
/// windows against a 24-window minimum: six interrupted windows anywhere in it cost the whole episode.
fn acceptance(r: &BTreeMap<String, (usize, usize)>) -> Vec<String> {
    let share = |c: &str| {
        let (hit, n) = r.get(c).unwrap_or_else(|| panic!("class {c} missing from the fixtures"));
        *hit as f64 / *n as f64
    };
    let mut breaches = Vec::new();
    for (c, floor) in [("AfibAudited", 0.55), ("AfibMachine", 0.50)] {
        if share(c) < floor {
            breaches.push(format!("{c} reported {:.4}, under the {floor:.2} floor", share(c)));
        }
    }
    // Sinus from three independent sources, then the ectopy classes: premature beats out-scatter
    // fibrillation, so a screen built on scatter alone fires hardest on the people who least need it.
    for (c, ceiling) in [
        ("SinusNsr", 0.02),
        ("SinusAfdb", 0.02),
        ("SinusMit", 0.02),
        ("EctopyLight", 0.10),
        ("EctopyHeavy", 0.10),
        ("EctopyBigeminal", 0.10),
    ] {
        if share(c) > ceiling {
            breaches.push(format!("{c} reported {:.4}, over the {ceiling:.2} ceiling", share(c)));
        }
    }
    breaches
}

#[test]
fn print_the_reported_share_of_every_class() {
    for (class, (hit, n)) in rates() {
        println!("{class:<16} reported {hit:>3} of {n:>3}   {:.3}", hit as f64 / n as f64);
    }
}

#[test]
fn the_shipped_screen_clears_the_sensitivity_floors_and_the_specificity_ceilings() {
    let breaches = acceptance(&rates());
    assert!(breaches.is_empty(), "{}", breaches.join("\n"));
}

#[test]
fn a_scorer_that_does_nothing_cannot_clear_the_same_thresholds() {
    // The gate above is only worth its name if something fails it. Three do-nothing scorers, judged by
    // exactly the same `acceptance`: the shipped screen is the only one that clears it.
    //
    // Scatter-only is the one that matters. It is the screen this module exists NOT to be, and it fails
    // BOTH arms: it under-reports fibrillation and fires hardest on bigeminal ectopy.
    let never = rates_of(&|_| false);
    let always = rates_of(&|_| true);
    let scatter = rates_of(&scatter_only);
    for (name, r) in [("never report", &never), ("always report", &always), ("scatter only", &scatter)] {
        let breaches = acceptance(r);
        println!("null '{name}': {} breach(es)", breaches.len());
        for b in &breaches {
            println!("    {b}");
        }
        assert!(!breaches.is_empty(), "null scorer '{name}' cleared the acceptance thresholds");
    }
    let sh = |r: &BTreeMap<String, (usize, usize)>, c: &str| {
        let (h, n) = r[c];
        h as f64 / n as f64
    };
    assert!(sh(&scatter, "EctopyBigeminal") > 0.10, "scatter-only must breach the ectopy ceiling");
    assert!(sh(&scatter, "AfibAudited") < 0.55, "scatter-only must also miss fibrillation");
    // And it must be WORSE on ectopy than on fibrillation, which is the premise of the whole module.
    assert!(sh(&scatter, "EctopyBigeminal") > sh(&scatter, "AfibMachine"));
}

/// A scatter-only screen: fire on a window whose RMSSD over mean R-R reaches the corpus fibrillation
/// median, then apply the shipped duration rule and nothing else.
fn scatter_only(s: &Stretch) -> bool {
    const FIBRILLATION_MEDIAN_RMSSD_OVER_MEAN: f64 = 0.289;
    let mut fired = Vec::new();
    let mut start = 0usize;
    while start + screen::WINDOW_BEATS <= s.rr.len() {
        let w = &s.rr[start..start + screen::WINDOW_BEATS];
        fired.push(
            physio_algo::rr_irregularity::rmssd_over_mean_rr(w)
                .is_some_and(|v| v >= FIBRILLATION_MEDIAN_RMSSD_OVER_MEAN),
        );
        start += screen::STEP_BEATS;
    }
    longest_run(&fired) >= screen::MIN_EPISODE_WINDOWS
}

fn longest_run(flags: &[bool]) -> usize {
    let (mut best, mut cur) = (0usize, 0usize);
    for &f in flags {
        cur = if f { cur + 1 } else { 0 };
        best = best.max(cur);
    }
    best
}

#[test]
fn the_synthetic_clock_makes_every_input_quality_gate_inert() {
    // The gate that would have caught the defect this file used to carry. Stamping one beat a second
    // made the coverage gate fire on any window slower than about 54 bpm, and it deleted 6.5 % of the
    // EctopyBigeminal windows - the class whose ceiling has the thinnest margin - before the rhythm
    // logic ever saw them. A deleted window also breaks a run, so the deletions could only ever make
    // the specificity ceilings easier to clear.
    let mut refusals: BTreeMap<String, usize> = BTreeMap::new();
    let mut windows = 0usize;
    for s in fixtures() {
        let beats = stamp(&s.rr);
        let mut start = 0usize;
        while start + screen::WINDOW_BEATS <= beats.len() {
            windows += 1;
            if let IrregularityReading::Inconclusive { reason, .. } =
                assess(&beats[start..start + screen::WINDOW_BEATS])
            {
                *refusals.entry(format!("{reason:?}")).or_default() += 1;
            }
            start += screen::STEP_BEATS;
        }
    }
    println!("{windows} windows at {STAMP_STEP_S} s a beat, refusals {refusals:?}");
    // Both are consts, so these are compile-time: a stamp that broke either could never be built.
    const { assert!(STAMP_STEP_S > quality::RESCALE_LAG_S, "a beat could be read as a rescaled copy") };
    const { assert!(STAMP_STEP_S <= MAX_BEAT_GAP_S, "a clock coarser than the gap rule breaks every run") };
    // The three clock-derived refusals must never fire: they would be measuring this file's stamp.
    for reason in [
        format!("{:?}", Refusal::RepeatedBeats),
        format!("{:?}", Refusal::RescaledCopies),
        format!("{:?}", Refusal::ImpossibleCoverage),
    ] {
        assert_eq!(refusals.get(&reason), None, "the stamp is producing {reason}: {refusals:?}");
    }
    // What is left is a property of the R-R VALUES, not of the clock, so it is pinned rather than banned:
    // one window in the corpus has over a fifth of its beats outside the physiological range.
    let out_of_range = refusals.values().sum::<usize>();
    assert_eq!(out_of_range, 2, "value-derived refusals moved: {refusals:?}");
    // 500 stretches of 256 beats, 29 windows each.
    assert_eq!(windows, 14_500, "window count changed");
}

#[test]
fn ectopy_out_scatters_fibrillation_and_is_still_not_reported() {
    // The main job, and the measurement behind it. The ceilings themselves are asserted by
    // `the_shipped_screen_clears_...`; what this pins is the premise they only matter under.
    let all = fixtures();
    let med_ratio = |c: &str| {
        let mut v: Vec<f64> = all
            .iter()
            .filter(|s| s.class == c)
            .filter_map(|s| physio_algo::rr_irregularity::rmssd_over_mean_rr(&s.rr))
            .collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    };
    let (bigeminal, afib) = (med_ratio("EctopyBigeminal"), med_ratio("AfibMachine"));
    println!("median RMSSD/mean: bigeminal {bigeminal:.3}, machine fibrillation {afib:.3}");
    assert!(
        bigeminal > afib,
        "the premise of this whole module: bigeminal scatter {bigeminal:.3} must exceed fibrillation {afib:.3}"
    );
}

#[test]
fn the_ectopy_discriminator_separates_where_the_scatter_statistics_cannot() {
    // The ceilings above do NOT prove the discriminator does anything: raising
    // `EPISODE_RESIDUAL_COSEN_FLOOR` out of the way leaves them passing, because a 256-beat stretch holds
    // only 29 windows and the 24-window duration floor already suppresses ectopy on its own. Measured
    // here: with the veto removed entirely, EctopyBigeminal still reports 2 of 32. The discriminator
    // earns its place over whole records, and this is the test that pins the mechanism it works by.
    // What the veto does is NOT widen the median gap - measured on these fixtures the median gap
    // narrows, 0.996 on plain COSEn against 0.862 on the residual. It works on the OVERLAP: of the
    // stretches irregular enough to clear the floor, it removes far more of the ectopy than of the
    // fibrillation. That is the claim, and it is the one worth pinning.
    let per_class = removal_shares();
    for (class, (removed, eligible)) in &per_class {
        println!("{class:<16} cleared the irregularity floor {eligible:>3}, veto removed {removed:>3}");
    }
    let share = |c: &str| {
        let (removed, eligible) = per_class[c];
        assert!(eligible > 0, "{c} never clears the irregularity floor, so the veto is untested on it");
        removed as f64 / eligible as f64
    };
    // Measured here: fibrillation 10 of 59 removed (0.169), bigeminal ectopy 5 of 10 (0.500), heavy
    // ectopy 9 of 13 (0.692) - a three- to fourfold differential. The floor is set below the smallest of
    // those margins because the bigeminal arm is only ten stretches.
    let afib = share("AfibMachine");
    for c in ["EctopyBigeminal", "EctopyHeavy"] {
        assert!(
            share(c) > afib + 0.25,
            "the veto must remove {c} ({:.3}) far more than fibrillation ({afib:.3})",
            share(c)
        );
    }
    assert!(afib < 0.35, "the veto must leave most fibrillation alone, removed {afib:.3}");
}

/// Per class: how many stretches clear the irregularity floor on median COSEn, and how many of those the
/// residual floor then removes. Mirrors what an episode is judged by — the medians over its windows.
fn removal_shares() -> BTreeMap<String, (usize, usize)> {
    let mut out: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    for s in fixtures() {
        let mut cosen = Vec::new();
        let mut residual = Vec::new();
        let mut start = 0usize;
        while start + screen::WINDOW_BEATS <= s.rr.len() {
            let w = &s.rr[start..start + screen::WINDOW_BEATS];
            if let Some(c) = physio_algo::rr_irregularity::cosen(w) {
                cosen.push(c);
            }
            if let Some(r) = physio_algo::rr_irregularity::profile(w).and_then(|p| p.residual_cosen) {
                residual.push(r);
            }
            start += screen::STEP_BEATS;
        }
        let med = |mut v: Vec<f64>| {
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            (!v.is_empty()).then(|| v[v.len() / 2])
        };
        let (Some(c), r) = (med(cosen), med(residual)) else { continue };
        if c < screen::COSEN_IRREGULAR_FLOOR {
            continue;
        }
        let e = out.entry(s.class.clone()).or_default();
        e.1 += 1;
        // A missing residual counts as removed: without that view the ectopy cannot be ruled out.
        e.0 += usize::from(r.is_none_or(|r| r < screen::EPISODE_RESIDUAL_COSEN_FLOOR));
    }
    out
}

#[test]
fn flutter_is_missed_and_that_is_recorded_here_rather_than_hidden() {
    // Flutter with fixed conduction is REGULAR, so an R-R screen cannot see it. It is a real rhythm
    // condition and a real false negative. This test exists so the gap is a checked fact, not a footnote.
    let r = rates();
    let (hit, n) = r["Flutter"];
    assert!(hit as f64 / n as f64 <= 0.15, "flutter reported {hit} of {n}");
}

#[test]
fn a_stretch_shorter_than_an_episode_can_only_calibrate() {
    let all = fixtures();
    let s = &all[0];
    let short = &s.rr[..s.rr.len().min(screen::MIN_SCREEN_BEATS - 1)];
    assert!(matches!(screen(&stamp(short)), ScreenState::Calibrating { .. }));
    // A 30 s reading at 70 bpm is about 35 beats. It cannot support the word "episode", and says so.
    assert!(matches!(
        screen(&stamp(&s.rr[..35])),
        ScreenState::Calibrating { have: 35, .. }
    ));
}

#[test]
fn every_reported_episode_carries_the_indices_behind_it() {
    let mut checked = 0usize;
    for s in fixtures().iter().filter(|s| s.class.starts_with("Afib")) {
        let ScreenState::IrregularEpisodes { episodes, windows_assessed } = screen(&stamp(&s.rr)) else {
            continue;
        };
        assert!(windows_assessed > 0);
        for e in &episodes {
            assert!(e.windows >= screen::MIN_EPISODE_WINDOWS, "{} {e:?}", s.record);
            assert!(e.cosen >= screen::COSEN_IRREGULAR_FLOOR, "{e:?}");
            assert!(e.residual_cosen >= screen::EPISODE_RESIDUAL_COSEN_FLOOR, "{e:?}");
            assert!(e.end_unix > e.start_unix && e.dur_s > 0, "{e:?}");
            // Reported and never decided on, because it ranks ectopy above fibrillation.
            assert!(e.rmssd_over_mean.is_finite() && e.cell_occupancy.is_finite(), "{e:?}");
            checked += 1;
        }
    }
    assert!(checked >= 20, "only {checked} episodes to inspect");
}

#[test]
fn the_premature_beat_labels_and_the_classes_agree() {
    // A guard on the fixture itself: if the emitter's class boundaries ever drift from the premature-beat
    // shares they were derived from, every number above would move without any test noticing.
    for s in fixtures() {
        match s.class.as_str() {
            "SinusMit" | "SinusNsr" => assert!(s.premature_fraction <= 0.02, "{} {}", s.class, s.record),
            "EctopyBigeminal" => assert!(s.premature_fraction > 0.30, "{} {}", s.class, s.record),
            "EctopyHeavy" => assert!(s.premature_fraction > 0.15, "{} {}", s.class, s.record),
            _ => {}
        }
    }
}
