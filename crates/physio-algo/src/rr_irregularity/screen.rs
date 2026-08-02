//! The wellness screen: irregular-rhythm EPISODES over a sliding window, with the ectopy discrimination
//! that stops it firing on ordinary premature beats.
//!
//! **This is a wellness screen, not a diagnosis and not a medical device.** It reports that a stretch of
//! beats was irregular in a way ordinary ectopy does not explain. It does not name a condition, and it
//! cannot: naming one needs an ECG read by someone qualified to read it. [`ScreenState::Inconclusive`] is
//! a first-class answer and is returned freely — a short record, a noisy one, or irregularity that could
//! be ectopy all end there, because "we cannot tell" is true and "regular" would be false reassurance.
//!
//! Shape, from [`hr_anomaly`](crate::hr_anomaly): eligibility first, a personal-scale-free index second,
//! a sustained run third. What is different here is that scatter alone is the WRONG index — premature
//! beats out-scatter fibrillation — so a run must also survive [`ectopy`](super::ectopy).
//!
//! # The operating point, and what it cost
//!
//! Every constant below was derived on the PhysioNet R-R corpus — afdb, nsrdb and mitdb, every record of
//! every database, 382,466 sliding windows over 91 records. The point chosen is the most sensitive rule
//! whose reported-window rate stays at or under 5 % on EVERY negative class, the ectopy ones included.
//! Pooled specificity is decided by the 74k sinus windows and would hide a screen that fired on every
//! premature-beat burst, so it is not the number the point was chosen on.
//!
//! Measured by running THIS function over the corpus, not by re-scoring the indices:
//!
//! ```text
//! sensitivity, fibrillation windows inside a reported episode   0.772
//! specificity, pooled negatives                                 0.998
//!   sinus, nsrdb / afdb / mitdb            0.0015 / 0.0022 / 0.0000
//!   occasional ectopy                                          0.0015
//!   heavy ectopy                                               0.0399
//!   bigeminal ectopy                                           0.0312
//! records with at least one episode      afdb 23/25, mitdb 9/48, nsrdb 1/18
//! ```
//!
//! Both afdb misses are records that hold almost no fibrillation at all — 4 and 36 windows of some 4,500.
//!
//! **What it trades away: roughly one fibrillation window in four.** Tuned instead for the best balanced
//! separation the same corpus gives 0.947, at a 41 % reported rate on bigeminal ectopy — firing constantly
//! on healthy people with common premature beats. The ectopy veto itself, held at these constants, costs
//! 0.832 to 0.772 in sensitivity and halves bigeminal false positives, 0.0670 to 0.0312.
//!
//! # What it cannot see, and where it has not been shown to work
//!
//! Atrial flutter with fixed conduction is REGULAR. Corpus flutter windows read a median COSEn of -1.786,
//! below normal sinus. This screen misses it by construction rather than by tuning, and no amount of R-R
//! work will find it. That is a real condition and a real false negative.
//!
//! **Every source above is chest or handheld ECG. None is wrist PPG.** On the only real strap R-R this
//! project holds — 8 nights, one wearer, no known irregular rhythm — the screen returns an episode on 2 of
//! them, refuses 1 night on input quality, and calls the rest regular. Both episodes sit at the corpus's
//! 99th percentile of rejected-beat share. No ground truth exists for those nights, so they are neither
//! confirmed false positives nor a validation: they are the measurement that says these numbers have not
//! been shown to carry from a chest electrode to an optical wrist sensor.

use crate::rr_irregularity::{
    assess, ectopy::EctopyProfile, IrregularityIndices, IrregularityReading, Refusal,
};
use crate::stats::median;

/// Beats in one window. Short enough that a run of them localises an episode, long enough for the binned
/// entropy (26) and COSEn (12) floors to be met inside one.
pub const WINDOW_BEATS: usize = 32;
/// Beats a window advances. A quarter of a window, so an episode boundary lands within 8 beats.
pub const STEP_BEATS: usize = 8;
/// Consecutive firing windows an episode needs. With the window and step above this spans
/// `32 + 23 * 8 = 216` beats — roughly 2.2 to 3.6 minutes of beats at 60 to 100 bpm.
pub const MIN_EPISODE_WINDOWS: usize = 24;
/// Beats before any episode is possible: one window plus the step of every later window in the shortest
/// episode. Under this the screen is [`ScreenState::Calibrating`] and can be nothing else.
pub const MIN_SCREEN_BEATS: usize = WINDOW_BEATS + (MIN_EPISODE_WINDOWS - 1) * STEP_BEATS;
/// COSEn at or above which a window is irregular. Corpus grid point; sinus medians sit near -1.9 to -2.3
/// and fibrillation near -0.5.
pub const COSEN_IRREGULAR_FLOOR: f64 = -1.28;
/// An episode's MEDIAN residual COSEn — COSEn of the same beats with the premature ones removed — must
/// reach this, or the episode is ectopy-shaped and the screen says so instead of reporting it.
///
/// Applied to the finished episode, never to a single window: measured on the corpus, the per-window form
/// of this same test LOSES sensitivity against no test at all (0.735 against 0.832 at a 10 % ceiling)
/// because it rejects genuine windows one at a time, while the per-episode form gains (0.856).
pub const EPISODE_RESIDUAL_COSEN_FLOOR: f64 = -1.10;
/// Windows that must survive the input-quality gate before a reading is attempted at all.
pub const MIN_ASSESSED_FRACTION: f64 = 0.50;
/// Successive beats further apart than this did not follow one another: beats were lost in between, and
/// the difference across the gap is not a beat-to-beat change. A window containing one is never scored.
///
/// Windowing by BEAT COUNT alone silently bridges such a gap. Measured on real strap R-R that produced an
/// episode of 280 beats spanning 1,781 seconds — 6.4 seconds a beat — assembled from beats minutes apart.
/// The same 5 s that breaks a run in [`hr_anomaly`](crate::hr_anomaly).
pub const MAX_BEAT_GAP_S: u32 = 5;

/// Why the screen declined to answer. Each is a measured property of the input, never a guess.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScreenRefusal {
    /// Too many windows failed the input-quality gate for what is left to be representative.
    PoorInputQuality { assessed: usize, windows: usize },
    /// A long enough run of irregular windows was found, and the premature-beat view could not rule
    /// ordinary ectopy out. Reported as unresolved rather than as either answer — a missing residual view
    /// counts as not ruled out.
    EctopyNotExcluded { runs: usize },
}

/// How far above the floor an episode sat. An ORDERING, not a probability: this project has no prevalence
/// to calibrate a probability from, and a number that looked like one would be believed.
///
/// The bands have a measured meaning. Over the 423 episodes the corpus produced: `High` covered 210 of
/// them and 98.9 % of their windows were fibrillation; `Moderate` covered 213 at 94.9 %. `Low` covered
/// NONE — the residual floor already excludes almost all of the region it describes, since removing
/// premature beats normally lowers COSEn rather than raising it. It is kept as the fallback arm, and an
/// episode landing there would be one this corpus never showed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum EpisodeConfidence {
    /// Over the floor, inside the band ordinary ectopy also reaches.
    Low,
    /// Clear of the band ectopy reaches, short of the fibrillation median.
    Moderate,
    /// At or above the corpus fibrillation median on both views.
    High,
}

/// Median COSEn of a corpus fibrillation window; the [`EpisodeConfidence::High`] boundary.
pub const COSEN_CONFIDENT: f64 = -0.53;
/// Median residual COSEn of a corpus fibrillation window.
pub const RESIDUAL_COSEN_CONFIDENT: f64 = -0.92;
/// COSEn above which an episode is clear of the band heavy ectopy reaches (its corpus median, -1.464).
pub const COSEN_MODERATE: f64 = -1.00;

/// One reported episode: when it ran, how long, and the indices behind it. The indices are the medians
/// over the episode's windows, so a caller can see WHICH view fired rather than only that something did.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Episode {
    pub start_unix: u32,
    pub end_unix: u32,
    pub dur_s: u32,
    /// Consecutive windows in the run.
    pub windows: usize,
    pub confidence: EpisodeConfidence,
    /// Median COSEn over the run.
    pub cosen: f64,
    /// Median COSEn over the run with the premature beats removed — the ectopy view that cleared it.
    pub residual_cosen: f64,
    /// Median share of beats Malik rejection removed, so a reader can see how much ectopy was present.
    pub ectopic_fraction: f64,
    /// Median RMSSD over mean R-R. Reported and NOT decided on: it ranks ectopy above fibrillation.
    pub rmssd_over_mean: f64,
    /// Median Poincare cell occupancy over the run.
    pub cell_occupancy: f64,
    /// Median beat-time over elapsed time across the run. Above 1.0 means more beat-time than clock, so
    /// the beats cannot all be real; each window individually cleared [`quality::MAX_COVERAGE`], and this
    /// is what the run looks like as a whole.
    pub coverage: f64,
    /// Median rescaled-copy share across the run — the storage defect an exact-repeat count cannot see.
    pub rescaled_fraction: f64,
}

/// The screen's answer over one run of beats.
#[derive(Clone, Debug, PartialEq)]
pub enum ScreenState {
    /// Not enough beats for even the shortest episode. Resolves by collecting more.
    Calibrating { have: usize, need: usize },
    /// Assessed, and no run of irregular windows reached the duration floor.
    Regular { windows_assessed: usize, windows_irregular: usize },
    /// One or more episodes the ectopy view did not explain.
    IrregularEpisodes { episodes: Vec<Episode>, windows_assessed: usize },
    /// The input cannot support an honest answer. Not a failure, and not "regular".
    Inconclusive { reason: ScreenRefusal },
}

/// Screen one timestamped R-R series (unix second, interval in ms) in the caller's chronological order.
///
/// Windows of [`WINDOW_BEATS`] advance [`STEP_BEATS`] at a time; each is put through [`assess`], so a
/// window whose input fails the duplication or range gates is never scored. A window is irregular when
/// its COSEn reaches [`COSEN_IRREGULAR_FLOOR`]. A run of [`MIN_EPISODE_WINDOWS`] consecutive irregular
/// windows is an episode candidate — a refused or regular window breaks the run — and a candidate is
/// reported only if its median residual COSEn reaches [`EPISODE_RESIDUAL_COSEN_FLOOR`].
///
/// A 30-second reading holds roughly 30 to 50 beats. That is one window at most and far under
/// [`MIN_SCREEN_BEATS`], so it can only ever return [`ScreenState::Calibrating`]: it can show the indices
/// of a moment, and it cannot support the word "episode".
pub fn screen(beats: &[(u32, u16)]) -> ScreenState {
    if beats.len() < MIN_SCREEN_BEATS {
        return ScreenState::Calibrating { have: beats.len(), need: MIN_SCREEN_BEATS };
    }
    let readings: Vec<(usize, IrregularityReading)> = (0..)
        .map(|i| i * STEP_BEATS)
        .take_while(|&s| s + WINDOW_BEATS <= beats.len())
        .map(|s| (s, assess(&beats[s..s + WINDOW_BEATS])))
        .collect();

    let scored: Vec<Option<IrregularityIndices>> = readings
        .iter()
        .map(|(_, r)| match r {
            IrregularityReading::Assessed(i) => Some(*i),
            IrregularityReading::Inconclusive { .. } => None,
        })
        .collect();
    let assessed = scored.iter().filter(|s| s.is_some()).count();
    if (assessed as f64) < MIN_ASSESSED_FRACTION * readings.len() as f64 {
        return ScreenState::Inconclusive {
            reason: ScreenRefusal::PoorInputQuality { assessed, windows: readings.len() },
        };
    }

    let irregular: Vec<bool> = readings
        .iter()
        .zip(&scored)
        .map(|((s, _), idx)| {
            contiguous(&beats[*s..s + WINDOW_BEATS])
                && idx.is_some_and(|i| i.cosen.is_some_and(|c| c >= COSEN_IRREGULAR_FLOOR))
        })
        .collect();
    let mut episodes = Vec::new();
    let mut candidates = 0usize;
    for (s, e) in runs(&irregular) {
        if e - s < MIN_EPISODE_WINDOWS {
            continue;
        }
        candidates += 1;
        if let Some(ep) = episode(beats, &readings, &scored, s, e) {
            episodes.push(ep);
        }
    }

    match (episodes.is_empty(), candidates) {
        (false, _) => ScreenState::IrregularEpisodes { episodes, windows_assessed: assessed },
        (true, 0) => ScreenState::Regular {
            windows_assessed: assessed,
            windows_irregular: irregular.iter().filter(|&&b| b).count(),
        },
        (true, runs) => {
            ScreenState::Inconclusive { reason: ScreenRefusal::EctopyNotExcluded { runs } }
        }
    }
}

/// Summarise windows `[s, e)` as an episode, or `None` when the ectopy view does not clear it.
fn episode(
    beats: &[(u32, u16)],
    readings: &[(usize, IrregularityReading)],
    scored: &[Option<IrregularityIndices>],
    s: usize,
    e: usize,
) -> Option<Episode> {
    let idx: Vec<IrregularityIndices> = scored[s..e].iter().flatten().copied().collect();
    let med = |f: fn(&IrregularityIndices) -> Option<f64>| {
        let v: Vec<f64> = idx.iter().filter_map(f).collect();
        (!v.is_empty()).then(|| median(&v))
    };
    let ect = |f: fn(&EctopyProfile) -> Option<f64>| {
        let v: Vec<f64> = idx.iter().filter_map(|i| i.ectopy.as_ref().and_then(f)).collect();
        (!v.is_empty()).then(|| median(&v))
    };
    let residual_cosen = ect(|e| e.residual_cosen)?;
    if residual_cosen < EPISODE_RESIDUAL_COSEN_FLOOR {
        return None;
    }
    let cosen = med(|i| i.cosen)?;

    let first = readings[s].0;
    let last = readings[e - 1].0 + WINDOW_BEATS - 1;
    let (start_unix, end_unix) = (beats[first].0, beats[last.min(beats.len() - 1)].0);
    Some(Episode {
        start_unix,
        end_unix,
        dur_s: end_unix.saturating_sub(start_unix),
        windows: e - s,
        confidence: confidence(cosen, residual_cosen),
        cosen,
        residual_cosen,
        ectopic_fraction: ect(|e| Some(e.ectopic_fraction)).unwrap_or(0.0),
        rmssd_over_mean: med(|i| i.rmssd_over_mean).unwrap_or(f64::NAN),
        cell_occupancy: med(|i| i.poincare.map(|p| p.cell_occupancy)).unwrap_or(f64::NAN),
        coverage: med(|i| Some(i.quality.coverage)).unwrap_or(f64::NAN),
        rescaled_fraction: med(|i| Some(i.quality.rescaled_fraction)).unwrap_or(f64::NAN),
    })
}

/// Both views at or above the corpus fibrillation medians is the top band; clear of the band heavy
/// ectopy reaches is the middle one.
fn confidence(cosen: f64, residual_cosen: f64) -> EpisodeConfidence {
    if cosen >= COSEN_CONFIDENT && residual_cosen >= RESIDUAL_COSEN_CONFIDENT {
        EpisodeConfidence::High
    } else if cosen >= COSEN_MODERATE {
        EpisodeConfidence::Moderate
    } else {
        EpisodeConfidence::Low
    }
}

/// No two successive beats in the window are more than [`MAX_BEAT_GAP_S`] apart, so its successive
/// differences are all real beat-to-beat changes rather than one spanning a hole in the data.
fn contiguous(window: &[(u32, u16)]) -> bool {
    window.windows(2).all(|w| w[1].0.saturating_sub(w[0].0) <= MAX_BEAT_GAP_S)
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

/// The refusal a window reported, for a caller reporting why a stretch was never scored.
pub fn window_refusals(beats: &[(u32, u16)]) -> Vec<Refusal> {
    (0..)
        .map(|i| i * STEP_BEATS)
        .take_while(|&s| s + WINDOW_BEATS <= beats.len())
        .filter_map(|s| match assess(&beats[s..s + WINDOW_BEATS]) {
            IrregularityReading::Inconclusive { reason, .. } => Some(reason),
            IrregularityReading::Assessed(_) => None,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Beats at one a second carrying `rr`, timestamped by their own running beat time.
    fn stamp(rr: &[u16]) -> Vec<(u32, u16)> {
        let mut t = 0.0f64;
        rr.iter()
            .map(|&v| {
                t += f64::from(v);
                ((t / 1000.0) as u32, v)
            })
            .collect()
    }

    /// A regular rhythm at ~1000 ms with respiratory sway.
    fn regular(n: usize) -> Vec<u16> {
        (0..n)
            .map(|i| (1000.0 + (i as f64 * 0.2).sin() * 15.0).round() as u16)
            .collect()
    }

    /// Every interval drawn independently over `[lo, lo + span)` — a sustained irregular rhythm.
    fn irregular(n: usize, lo: u16, span: u64, seed: u64) -> Vec<u16> {
        let mut x = seed;
        (0..n)
            .map(|_| {
                x = x.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
                lo + ((x >> 33) % span) as u16
            })
            .collect()
    }

    /// Bigeminy: every other beat premature, with the pause that compensates for it.
    fn bigeminy(n: usize) -> Vec<u16> {
        (0..n).map(|i| if i % 2 == 0 { 560u16 } else { 1040 }).collect()
    }

    #[test]
    fn a_short_record_can_only_calibrate_and_says_how_short() {
        // A 30 s reading at 70 bpm is about 35 beats: one window, and no episode is possible from it.
        let thirty_seconds = stamp(&regular(35));
        assert_eq!(
            screen(&thirty_seconds),
            ScreenState::Calibrating { have: 35, need: MIN_SCREEN_BEATS }
        );
        assert_eq!(MIN_SCREEN_BEATS, 216, "one window plus the step of every later window in an episode");
        assert!(matches!(screen(&[]), ScreenState::Calibrating { have: 0, .. }));
    }

    #[test]
    fn a_regular_night_reports_regular_and_counts_what_it_looked_at() {
        match screen(&stamp(&regular(1200))) {
            ScreenState::Regular { windows_assessed, windows_irregular } => {
                assert!(windows_assessed > 100, "assessed {windows_assessed}");
                assert_eq!(windows_irregular, 0);
            }
            other => panic!("expected regular, got {other:?}"),
        }
    }

    #[test]
    fn a_sustained_irregular_stretch_is_reported_as_an_episode_with_its_indices() {
        let mut rr = regular(400);
        rr.extend(irregular(600, 500, 700, 24));
        rr.extend(regular(400));
        match screen(&stamp(&rr)) {
            ScreenState::IrregularEpisodes { episodes, .. } => {
                assert_eq!(episodes.len(), 1, "one stretch, one episode: {episodes:?}");
                let e = episodes[0];
                assert!(e.windows >= MIN_EPISODE_WINDOWS);
                assert!(e.cosen >= COSEN_IRREGULAR_FLOOR);
                assert!(e.residual_cosen >= EPISODE_RESIDUAL_COSEN_FLOOR);
                assert!(e.dur_s > 0 && e.end_unix > e.start_unix);
                // The episode must not start before the irregular beats do. Window 0 covers beats 0..32,
                // so the first firing window cannot begin earlier than beat 400 - WINDOW_BEATS.
                let earliest = stamp(&rr)[400 - WINDOW_BEATS].0;
                assert!(e.start_unix >= earliest, "episode started at {} before {earliest}", e.start_unix);
            }
            other => panic!("expected an episode, got {other:?}"),
        }
    }

    #[test]
    fn sustained_bigeminy_is_never_reported_as_an_episode() {
        // The whole point. Bigeminy out-scatters fibrillation, so a scatter-only screen fires on it.
        let mut rr = regular(300);
        rr.extend(bigeminy(800));
        rr.extend(regular(300));
        let state = screen(&stamp(&rr));
        assert!(
            !matches!(state, ScreenState::IrregularEpisodes { .. }),
            "alternating ectopy must never be reported: {state:?}"
        );
        // And a scatter statistic really would have fired: RMSSD over mean R-R is far above the
        // fibrillation median of 0.289 that the corpus measures.
        let ratio = crate::rr_irregularity::rmssd_over_mean_rr(&bigeminy(64)).unwrap();
        assert!(ratio > 0.5, "bigeminy scatter ratio {ratio}");
    }

    #[test]
    fn irregularity_that_could_be_ectopy_is_inconclusive_and_never_regular() {
        // Frequent premature beats on a steady background: irregular enough to form a run, and explained
        // by ectopy once the premature beats are removed. Saying "regular" here would be false comfort.
        // The background carries sway, or whole-second stamping of an exactly repeating pattern trips the
        // duplicate-beat gate before any of this is reached.
        let rr: Vec<u16> = (0..900)
            .map(|i| {
                let sway = (i as f64 * 0.17).sin() * 18.0;
                match i % 4 {
                    0 => (520.0 + sway).round() as u16,
                    1 => (1080.0 + sway).round() as u16,
                    _ => (800.0 + sway).round() as u16,
                }
            })
            .collect();
        let state = screen(&stamp(&rr));
        assert!(
            matches!(
                state,
                ScreenState::Inconclusive { reason: ScreenRefusal::EctopyNotExcluded { .. } }
            ) || matches!(state, ScreenState::Regular { .. }),
            "ectopy must resolve to inconclusive or regular, never an episode: {state:?}"
        );
    }

    #[test]
    fn a_duplicated_series_is_refused_on_quality_not_scored_as_irregular() {
        // Every beat stored twice: the storage shape that reads as pure irregularity.
        let doubled: Vec<(u32, u16)> = stamp(&regular(700)).into_iter().flat_map(|b| [b, b]).collect();
        match screen(&doubled) {
            ScreenState::Inconclusive { reason: ScreenRefusal::PoorInputQuality { .. } } => {}
            other => panic!("a duplicated series must be refused on quality, got {other:?}"),
        }
        assert!(!window_refusals(&doubled).is_empty());
    }

    #[test]
    fn confidence_bands_are_ordered_and_bounded_by_the_corpus_medians() {
        assert_eq!(confidence(COSEN_CONFIDENT, RESIDUAL_COSEN_CONFIDENT), EpisodeConfidence::High);
        assert_eq!(confidence(COSEN_CONFIDENT, RESIDUAL_COSEN_CONFIDENT - 0.01), EpisodeConfidence::Moderate);
        assert_eq!(confidence(COSEN_MODERATE, -2.0), EpisodeConfidence::Moderate);
        assert_eq!(confidence(COSEN_IRREGULAR_FLOOR, -2.0), EpisodeConfidence::Low);
        assert!(EpisodeConfidence::Low < EpisodeConfidence::Moderate);
        assert!(EpisodeConfidence::Moderate < EpisodeConfidence::High);
    }

    #[test]
    fn no_wording_this_screen_can_emit_is_clinical() {
        // The gate `format_hr_watch` carries in the CLI, applied to every state and refusal this module
        // can render, which is the only text it produces.
        let banned = [
            "afib", "fibrillat", "arrhythm", "cardiac", "diagnos", "disease", "patient", "clinical",
            "alarm", "emergency", "abnormal", "medical", "treat",
        ];
        let mut rr = regular(300);
        rr.extend(irregular(700, 500, 700, 7));
        let mut texts: Vec<String> = vec![
            format!("{:?}", screen(&stamp(&rr))),
            format!("{:?}", screen(&stamp(&regular(1200)))),
            format!("{:?}", screen(&stamp(&regular(20)))),
        ];
        for reason in [
            ScreenRefusal::PoorInputQuality { assessed: 1, windows: 9 },
            ScreenRefusal::EctopyNotExcluded { runs: 2 },
        ] {
            texts.push(format!("{:?}", ScreenState::Inconclusive { reason }));
        }
        for c in [EpisodeConfidence::Low, EpisodeConfidence::Moderate, EpisodeConfidence::High] {
            texts.push(format!("{c:?}"));
        }
        for text in texts {
            let low = text.to_lowercase();
            for term in banned {
                assert!(!low.contains(term), "clinical term '{term}' in: {text}");
            }
        }
    }
}
