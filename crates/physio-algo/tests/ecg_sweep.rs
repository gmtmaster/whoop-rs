//! The decode sweep against ground truth: re-encode real overnight ECG at a layout and a sample rate the
//! test knows and the sweep does not, then measure whether it gets them back.
//!
//! The corpus is the 13 AAUWSS subjects, one 30 s epoch each at a measured 200 Hz, resampled to the
//! target rate and packed bit-exactly. Recovery is therefore measured, not argued. Negatives are matched
//! to the positive they came from, so nothing separates on amplitude or duration, and a sweep that
//! accepted one would be worse than useless — they are asserted as hard failures, not reported as ranges.
//!
//! The default run asserts a fixed, stated set of SIX cases so it stays quick. The whole 65-case matrix
//! is `the_recovery_rate_over_the_whole_corpus`, which asserts a floor on it; that is the only place the
//! recovery rate may be quoted from, and it is an EXACT-SHAPE rate.
//!
//! Every subject-named claim binds by subject id. Bound by array position, a claim survives the corpus
//! changing underneath it and goes on reading as though it were about the subject it names.
//!
//!   cargo test -p physio-algo --test ecg_sweep -- --nocapture
//!   cargo test --release -p physio-algo --test ecg_sweep -- --ignored --nocapture

mod ecg_corpus;
#[path = "ecg_corpus/encode.rs"]
mod encode;

use std::time::Instant;

use ecg_corpus::{add_hum, ecg_fixtures, gaussian_like, ppg_fixtures, resample, sawtooth_like, shuffled, Fixture};
use encode::{detector_consensus, encode_at, encode_stream, ppg_beats_ms, pulse_train};

use physio_algo::ecg::sweep::layout::{BitOrder, Layout};
use physio_algo::ecg::sweep::{
    layout_stats, sweep_split, sweep_window, Attribution, StallReason, SweepConfig, SweepOutcome,
    SweepReport, DEFAULT_RATES_HZ, UNSEARCHED_RATES_HZ,
};

/// The epoch length every fixture carries, in ms. Used to cut the optical beat list at the same
/// fractions the byte buffer is cut at.
const EPOCH_MS: f64 = 30_000.0;
/// Independent windows a candidate must survive.
const WINDOWS: usize = 3;
/// The transit time the pulse train is built with, and the per-beat spread around it.
const PTT_MS: f64 = 220.0;
const PTT_JITTER_MS: f64 = 18.0;

/// One ground-truth encoding: a reading rule, the rate it was written at, and how much power-line hum
/// rides on it.
struct Truth {
    name: &'static str,
    layout: Layout,
    fs_hz: f64,
    hum: f64,
}

fn lay(bits: u8, signed: bool, order: BitOrder, start_bit: usize, stride_bits: usize) -> Layout {
    Layout { bits, signed, order, start_bit, stride_bits }
}

/// The truth set. 512 and 256 are 1/1024-second partners of another searched rate and carry hum for that
/// reason — see `a_conversion_partner_is_refused_when_there_is_no_second_anchor`.
fn truths() -> Vec<Truth> {
    vec![
        Truth {
            name: "16-bit signed LE, dense, no header, 400 Hz",
            layout: lay(16, true, BitOrder::LsbFirst, 0, 16),
            fs_hz: 400.0,
            hum: 0.0,
        },
        Truth {
            name: "24-bit signed LE, dense, 3-byte header, 512 Hz + hum",
            layout: lay(24, true, BitOrder::LsbFirst, 24, 24),
            fs_hz: 512.0,
            hum: 0.30,
        },
        Truth {
            name: "18-bit signed MSB-first, dense, 1-byte header, 256 Hz + hum",
            layout: lay(18, true, BitOrder::MsbFirst, 8, 18),
            fs_hz: 256.0,
            hum: 0.30,
        },
        Truth {
            name: "16-bit signed BE, 2-channel interleave, 200 Hz",
            layout: lay(16, true, BitOrder::MsbFirst, 32, 32),
            fs_hz: 200.0,
            hum: 0.0,
        },
        Truth {
            name: "18-bit signed LE in a 24-bit container, 128 Hz",
            layout: lay(18, true, BitOrder::LsbFirst, 0, 24),
            fs_hz: 128.0,
            hum: 0.0,
        },
    ]
}

/// A byte buffer carrying `f`'s epoch under `t`, plus the optical beat train that goes with it.
///
/// Resample FIRST, then add the hum at the rate the sweep will read. Adding it at the fixture's own
/// 200 Hz put the 2nd harmonic exactly on Nyquist, where it is identically zero, and the 3rd above it,
/// where it folds back onto the fundamental with the opposite sign — leaving a bare 50 Hz tone at two
/// thirds of the requested amplitude and no harmonic structure at all.
fn stream(f: &Fixture, t: &Truth) -> (Vec<u8>, Vec<f64>) {
    let filler = resample(&shuffled(&f.samples, 0xA5A5), f.fs_hz, t.fs_hz);
    let mut wave = resample(&f.samples, f.fs_hz, t.fs_hz);
    if t.hum > 0.0 {
        wave = add_hum(&wave, t.fs_hz, 50.0, t.hum);
    }
    let bytes = encode_at(&wave, &t.layout, Some(&filler));
    let beats = pulse_train(&detector_consensus(&f.samples, f.fs_hz), f.fs_hz, PTT_MS, PTT_JITTER_MS, 7);
    (bytes, beats)
}

/// Every subject, with the cohort size checked: every claim below names a subject, and a claim bound to
/// an array position instead reads the same after the corpus changes underneath it.
fn corpus() -> Vec<Fixture> {
    let fx = ecg_fixtures();
    assert_eq!(fx.len(), SUBJECTS, "AAUWSS ships {SUBJECTS} ECG subjects; found {}", fx.len());
    fx
}

/// The subject with this id, by id and never by index.
fn subject<'a>(fx: &'a [Fixture], id: &str) -> &'a Fixture {
    fx.iter()
        .find(|f| f.subject == id)
        .unwrap_or_else(|| panic!("no subject {id} in {:?}", fx.iter().map(|f| &f.subject).collect::<Vec<_>>()))
}

/// One named subject, for the tests that need exactly one.
fn one(id: &str) -> Fixture {
    let mut fx = corpus();
    let i = fx.iter().position(|f| f.subject == id).unwrap_or_else(|| panic!("no subject {id}"));
    fx.swap_remove(i)
}

fn cfg() -> SweepConfig {
    SweepConfig::default()
}

/// ECG subjects in the corpus. The whole matrix is [`SUBJECTS`] x the truth set.
const SUBJECTS: usize = 13;

/// The (truth index, subject id) cases the DEFAULT run asserts: **6 of the 65-case matrix**. Fixed and
/// stated rather than picked per run. The recovery rate over all 65 is asserted by
/// `the_recovery_rate_over_the_whole_corpus`, which is `#[ignore]`d for runtime and is the only place
/// the headline rate may be quoted from.
const RECOVERED_CASES: [(usize, &str); 6] =
    [(0, "01"), (0, "03"), (0, "04"), (1, "03"), (3, "01"), (4, "01")];

fn recovered(r: &SweepReport, t: &Truth) -> bool {
    match r.outcome {
        SweepOutcome::Converged { shape, fs_hz, .. } => {
            fs_hz == t.fs_hz && (shape == t.layout.shape() || r.alias_shapes.contains(&t.layout.shape()))
        }
        _ => false,
    }
}

#[test]
fn the_sweep_recovers_the_six_stated_cases_of_the_sixty_five_case_matrix() {
    let fixtures = corpus();
    let truths = truths();
    assert_eq!(truths.len() * fixtures.len(), 65, "the matrix this file's headline rate is quoted over");
    assert_eq!(RECOVERED_CASES.len(), 6, "and what this test actually covers");
    for (ti, id) in RECOVERED_CASES {
        let (t, f) = (&truths[ti], subject(&fixtures, id));
        let (bytes, beats) = stream(f, t);
        let r = sweep_split(&bytes, &beats, EPOCH_MS, WINDOWS, &cfg());
        assert!(recovered(&r, t), "{} subject {}: {:?}\n{:?}", t.name, f.subject, r.outcome, r.conditions);
        // The rate comes back exactly; the reading rule only up to its alias set — an arm that has never
        // once been taken, see `the_report_counts_the_readings_it_cannot_tell_apart_but_never_names_them`.
        let SweepOutcome::Converged { shape, fs_hz, .. } = r.outcome else { unreachable!() };
        assert_eq!(fs_hz, t.fs_hz);
        assert!(r.margin >= cfg().min_margin && r.windows_agreed == WINDOWS);
        assert_eq!(shape, t.layout.shape(), "recovery has only ever been exact-shape");
    }
}

#[test]
fn the_report_counts_the_readings_it_cannot_tell_apart_but_never_names_them() {
    // `Candidate::aliases` counts the other reading rules in the leader's waveform class; `alias_shapes`
    // is meant to list them, and `recovered` accepts the truth appearing in that list. Measured over the
    // whole 65-case matrix: `alias_shapes` is empty in 65 of 65 reports while the leader reports
    // `aliases > 0` in 50 of them. The two cannot agree as the search stands — only one member of a
    // class, its representative, reaches the leaderboard, so the filter that builds `alias_shapes`
    // (same class, different shape) has nothing left to match.
    //
    // Pinned so that if this is ever fixed, this line is what says so. The alias arm of `recovered` is
    // dead until it is, and the corpus recovery rate is an EXACT-shape rate, not a rate up to aliasing.
    let fixtures = corpus();
    let truths = truths();
    let (mut counted, mut named, mut cases) = (0usize, 0usize, 0usize);
    for (ti, id) in RECOVERED_CASES {
        let (t, f) = (&truths[ti], subject(&fixtures, id));
        let (bytes, beats) = stream(f, t);
        let r = sweep_split(&bytes, &beats, EPOCH_MS, WINDOWS, &cfg());
        cases += 1;
        counted += usize::from(r.leader.as_ref().is_some_and(|c| c.aliases > 0));
        named += usize::from(!r.alias_shapes.is_empty());
    }
    println!("{cases} cases: leader reports aliases>0 in {counted}, alias_shapes non-empty in {named}");
    assert!(counted > 0, "no case has an alias to name, so this proves nothing - pick other cases");
    assert_eq!(named, 0, "alias_shapes is populated now - the alias arm of `recovered` is live again");
}

#[test]
fn matched_negatives_never_converge() {
    let fixtures = corpus();
    let f = subject(&fixtures, "01");
    let t = &truths()[0];
    let period = {
        let beats = detector_consensus(&f.samples, f.fs_hz);
        (beats[beats.len() - 1] - beats[0]) as f64 / (beats.len() - 1) as f64
    };
    let cases: Vec<(&str, Vec<f64>)> = vec![
        ("gaussian", gaussian_like(&f.samples, 11)),
        ("shuffled", shuffled(&f.samples, 13)),
        ("sawtooth", sawtooth_like(&f.samples, period)),
        ("constant", vec![f.samples[0]; f.samples.len()]),
    ];
    let beats = pulse_train(&detector_consensus(&f.samples, f.fs_hz), f.fs_hz, PTT_MS, PTT_JITTER_MS, 7);
    for (name, wave) in cases {
        let bytes = encode_stream(&wave, f.fs_hz, t.fs_hz, &t.layout, None);
        let r = sweep_split(&bytes, &beats, EPOCH_MS, WINDOWS, &cfg());
        assert!(r.converged().is_none(), "{name} converged: {:?}", r.outcome);
    }
}

#[test]
fn the_rate_free_prune_rejects_a_misaligned_reading_of_a_real_stream() {
    let f = &one("01");
    let t = &truths()[1];
    let (bytes, _) = stream(f, t);
    let right = layout_stats(&t.layout.decode(&bytes)).unwrap();
    // One bit out of alignment mixes each sample's high bits into the next.
    let skewed = Layout { start_bit: t.layout.start_bit + 1, ..t.layout };
    let wrong = layout_stats(&skewed.decode(&bytes)).unwrap();
    assert!(right.roughness < cfg().max_roughness, "true layout roughness {}", right.roughness);
    assert!(wrong.roughness > cfg().max_roughness, "misaligned roughness {}", wrong.roughness);
    // And nothing that rough ever reaches the leaderboard.
    let chunk = &bytes[..bytes.len() / WINDOWS];
    let w = sweep_window(chunk, &[], &cfg());
    for c in &w.leaderboard {
        assert!(layout_stats(&c.layout.decode(chunk)).unwrap().roughness <= cfg().max_roughness);
    }
}

#[test]
fn a_lone_window_is_never_enough_however_good_it_looks() {
    let f = &one("01");
    let t = &truths()[0];
    let (bytes, beats) = stream(f, t);
    let one = sweep_split(&bytes, &beats, EPOCH_MS, 1, &cfg());
    assert!(!one.conditions.stable, "a single window cannot be stable");
    assert!(one.converged().is_none(), "{:?}", one.outcome);
    let three = sweep_split(&bytes, &beats, EPOCH_MS, WINDOWS, &cfg());
    assert!(three.converged().is_some());
}

#[test]
fn the_optical_channel_is_what_separates_two_rates_a_hair_apart() {
    // 500 and 512 Hz differ by 2.4 %. Over a ten-second window that is 240 ms of accumulated drift
    // against a 40 ms per-beat tolerance, so the beats stop lining up — while every morphology index,
    // which knows nothing about wall-clock time, reads the same at both.
    let f = &one("03");
    let t = &truths()[1];
    let (bytes, beats) = stream(f, t);
    let w = sweep_window(&bytes[..bytes.len() / WINDOWS], &beats, &cfg());
    let at = |fs: f64| w.leaderboard.iter().find(|c| c.fs_hz == fs);
    let (Some(right), Some(near)) = (at(t.fs_hz), at(500.0)) else {
        panic!("both rates must be scored to be compared: {:?}", w.leaderboard.iter().map(|c| c.fs_hz).collect::<Vec<_>>())
    };
    assert!((right.ecg.b_sqi - near.ecg.b_sqi).abs() < 0.05, "morphology alone cannot tell them apart");
    let (a, b) = (right.ppg.unwrap(), near.ppg.unwrap());
    assert!(a.match_fraction > 0.9, "the true rate must line up beat by beat: {a:?}");
    assert!(b.match_fraction < 0.7, "the neighbour must not: {b:?}");
    assert!(right.passes && !near.passes);
}

#[test]
fn a_conversion_partner_is_refused_when_there_is_no_second_anchor() {
    // 500 = 512 × 1000/1024 exactly, and both are plausible converter rates. With only the optical beats
    // there is no way to tell a true 500 from a 512 read through one 1/1024-second conversion, so the
    // sweep says so instead of picking. Three of the nine searched rates sit in such a pair.
    // Nothing is filtered or injected here. The corpus itself supplies both cases: subject 01 carries too
    // little power-line interference for the anchor to find, and subject 03 carries enough.
    let t = Truth { name: "16-bit signed LE, dense, 500 Hz", fs_hz: 500.0, hum: 0.0, ..truths().swap_remove(0) };
    let fixtures = corpus();
    let at = |id: &str| {
        let f = subject(&fixtures, id);
        let bytes = encode_at(&resample(&f.samples, f.fs_hz, t.fs_hz), &t.layout, None);
        let beats = pulse_train(&detector_consensus(&f.samples, f.fs_hz), f.fs_hz, PTT_MS, PTT_JITTER_MS, 7);
        sweep_split(&bytes, &beats, EPOCH_MS, WINDOWS, &cfg())
    };

    let r = at("01");
    assert!(r.anchors.mains_fs_hz.is_none(), "subject 01 has no usable line peak");
    assert!(matches!(r.outcome, SweepOutcome::SuspectedUnitError(_)), "{:?}", r.outcome);
    assert_eq!(r.anchors.unit_error.map(|u| u.true_fs_hz), Some(512.0));
    assert!(r.converged().is_none());

    let r = at("03");
    assert!(r.anchors.mains_fs_hz.is_some(), "subject 03 does carry a line peak");
    assert!(recovered(&r, &t), "{:?} {:?}", r.outcome, r.anchors);
}

#[test]
fn two_anchors_that_contradict_each_other_stop_the_sweep() {
    // Hum written as though the stream ran at 500 Hz when it really runs at 400. Which rung of that hum's
    // own ladder the power-line anchor lands on is its business; the claim here is that it contradicts
    // the beats, and that a contradiction is terminal rather than something to split the difference on.
    let f = &one("01");
    let t = &truths()[0];
    let wave = resample(&f.samples, f.fs_hz, t.fs_hz);
    let bytes = encode_at(&add_hum(&wave, 500.0, 50.0, 0.30), &t.layout, None);
    let beats = pulse_train(&detector_consensus(&f.samples, f.fs_hz), f.fs_hz, PTT_MS, PTT_JITTER_MS, 7);
    let r = sweep_split(&bytes, &beats, EPOCH_MS, WINDOWS, &cfg());
    match r.outcome {
        SweepOutcome::Disagreement { mains_fs_hz, ppg_fs_hz, relative_difference } => {
            assert!(relative_difference > cfg().rate_agreement_tolerance);
            assert!(mains_fs_hz > 0.0 && ppg_fs_hz > 0.0);
            println!("power line {mains_fs_hz:.1} Hz against beats {ppg_fs_hz:.1} Hz");
        }
        other => panic!("a contradiction must be terminal, got {other:?}"),
    }
    assert!(r.converged().is_none());
}

#[test]
fn a_real_optical_channel_agrees_on_rate_and_that_is_not_enough() {
    // Subject 01's ECG and PPG fixtures are the same 30 s of the same night. Their RATES agree closely,
    // and the sweep still refuses, because the per-beat timing does not. That is the design working: an
    // average heart rate is weak evidence, and plenty of wrong answers can produce a plausible one.
    //
    // The limit here is the harness, not the data. A real caller is handed R-R by the strap; this test
    // has to find the pulses itself in a 64 Hz sleeping-wrist trace, and a peak detector at that rate
    // misplaces individual beats by hundreds of ms even when it gets the rate right. So this measures
    // what rate agreement alone is worth, which is the point, and it claims nothing stronger.
    let ecg = &one("01");
    let ppg = ppg_fixtures();
    assert_eq!(ppg.len(), 1, "one PPG subject by availability, not by selection");
    let ppg = &ppg[0];
    assert_eq!(ppg.subject, ecg.subject, "the two fixtures must be the same subject");
    let beats = ppg_beats_ms(&ppg.samples, ppg.fs_hz);
    let gap = |v: &[f64]| (v[v.len() - 1] - v[0]) / (v.len() - 1) as f64;
    let r_true: Vec<f64> = detector_consensus(&ecg.samples, ecg.fs_hz)
        .iter()
        .map(|&p| p as f64 / ecg.fs_hz * 1000.0)
        .collect();
    let (ppg_bpm, ecg_bpm) = (60_000.0 / gap(&beats), 60_000.0 / gap(&r_true));
    println!("real PPG {ppg_bpm:.1} bpm over {} beats, ECG {ecg_bpm:.1} bpm over {}", beats.len(), r_true.len());
    assert!((ppg_bpm - ecg_bpm).abs() / ecg_bpm < 0.05, "the two rates must agree: {ppg_bpm} vs {ecg_bpm}");

    let t = &truths()[0];
    let (bytes, _) = stream(ecg, t);
    let r = sweep_split(&bytes, &beats, EPOCH_MS, WINDOWS, &cfg());
    assert!(r.converged().is_none(), "rate agreement alone must not be enough: {:?}", r.outcome);
    let best = r.windows[0].leaderboard.first().expect("something must still be scored");
    let a = best.ppg.expect("and it must still report what the optical channel did");
    assert!(a.match_fraction < cfg().min_ppg_match, "per-beat matching: {a:?}");
    println!("best per-beat match {:.2} at {} ms, under the {} gate", a.match_fraction, a.offset_ms, cfg().min_ppg_match);
}

#[test]
fn a_stall_is_attributed_to_decode_or_to_contact() {
    let f = &one("01");
    let t = &truths()[0];
    let (bytes, beats) = stream(f, t);
    // Nothing decodable, but a steady optical rate: the problem is the decode.
    let noise: Vec<u8> = (0..40_000u32).map(|i| (i.wrapping_mul(2_654_435_761) >> 24) as u8).collect();
    let r = sweep_split(&noise, &beats, EPOCH_MS, WINDOWS, &cfg());
    assert!(r.converged().is_none(), "{:?}", r.outcome);
    assert_eq!(r.attribution, Attribution::Decode);
    assert!(
        matches!(
            r.outcome,
            SweepOutcome::Searching {
                reason: StallReason::NoLayoutSurvived | StallReason::NoCandidatePassed,
                ..
            }
        ),
        "{:?}",
        r.outcome
    );

    // A decodable stream but an erratic optical rate: the problem is contact.
    // Ascending, but with the intervals a dropped and a doubled beat leave behind.
    let mut erratic = vec![120.0];
    for k in 0..40 {
        erratic.push(erratic[k] + if k % 2 == 0 { 350.0 } else { 1600.0 });
    }
    let r = sweep_split(&bytes, &erratic, EPOCH_MS, WINDOWS, &cfg());
    assert_eq!(r.attribution, Attribution::Contact, "{:?}", r.outcome);
    // With no optical channel at all the two cannot be separated, and that is also said out loud.
    assert_eq!(sweep_split(&bytes, &[], 0.0, WINDOWS, &cfg()).attribution, Attribution::Unknown);
}

#[test]
fn nothing_in_the_report_is_an_amplitude() {
    // Every index is scale-invariant, so multiplying the stream by anything cannot change the verdict.
    // That property is what lets the sweep run on counts of unknown worth — and it is the reason no
    // counts-per-mV can come out of it.
    let f = &one("01");
    let t = &truths()[0];
    let (bytes, beats) = stream(f, t);
    let a = sweep_split(&bytes, &beats, EPOCH_MS, WINDOWS, &cfg());
    let scaled: Vec<f64> = f.samples.iter().map(|v| v * 37.0).collect();
    let bytes = encode_stream(&scaled, f.fs_hz, t.fs_hz, &t.layout, Some(&shuffled(&f.samples, 0xA5A5)));
    let b = sweep_split(&bytes, &beats, EPOCH_MS, WINDOWS, &cfg());
    assert_eq!(a.outcome, b.outcome, "a gain must not change the answer");
    assert_eq!(a.leader.map(|c| c.quality), b.leader.map(|c| c.quality));
}

#[test]
fn the_searched_rate_set_is_stated_and_so_is_what_is_not_searched() {
    for r in DEFAULT_RATES_HZ {
        assert!((physio_algo::ecg::MIN_FS_HZ..=physio_algo::ecg::MAX_FS_HZ).contains(&r), "{r} is unusable");
    }
    for r in UNSEARCHED_RATES_HZ {
        assert!(r > physio_algo::ecg::MAX_FS_HZ, "{r} is searchable and should not be on that list");
    }
    assert!(DEFAULT_RATES_HZ.contains(&500.0) && DEFAULT_RATES_HZ.contains(&512.0));
}

#[test]
fn degenerate_input_is_a_named_stall_not_a_panic() {
    let c = cfg();
    for bytes in [vec![], vec![0u8; 3], vec![0xFFu8; 5000], vec![0u8; 60_000]] {
        let r = sweep_split(&bytes, &[], 0.0, WINDOWS, &c);
        assert!(r.converged().is_none());
        assert!(matches!(r.outcome, SweepOutcome::Searching { .. }), "{:?}", r.outcome);
    }
    assert!(sweep_split(&[0u8; 60_000], &[f64::NAN, 1.0, 2.0], 100.0, 0, &c).converged().is_none());
}

/// Exact-shape recoveries the 65-case matrix must still reach. Measured: 41 of 65 (0.6308).
///
/// It was 47 of 65 (0.7231) while the hum was injected at the fixture's 200 Hz, where its 2nd harmonic
/// is annihilated on Nyquist and its 3rd folds back onto the fundamental; a hum written at the rate the
/// sweep reads costs 6 of the 65. The floor is one case under what is measured, not a round number.
const MIN_RECOVERED_OF_65: usize = 40;

#[test]
#[ignore = "the whole 65-case matrix, ~10 s; run with --release"]
fn the_recovery_rate_over_the_whole_corpus() {
    // The one place this file's headline recovery rate may be quoted from, and it is asserted here
    // rather than printed. `the_sweep_recovers_the_six_stated_cases...` covers 6 of these 65.
    let fixtures = corpus();
    println!("subjects {}, rates searched {DEFAULT_RATES_HZ:?}", fixtures.len());
    let (mut total_recovered, mut total_cases, mut total_wrong) = (0usize, 0usize, 0usize);
    for t in truths() {
        let (mut converged, mut wrong, mut aliased, mut stalls) = (0usize, 0usize, 0usize, Vec::new());
        let (mut ms_total, mut margins, mut survivors, mut classes, mut scored) =
            (0.0f64, Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let mut wrongs = Vec::new();
        for f in &fixtures {
            let (bytes, beats) = stream(f, &t);
            let start = Instant::now();
            let r = sweep_split(&bytes, &beats, EPOCH_MS, WINDOWS, &cfg());
            ms_total += start.elapsed().as_secs_f64() * 1000.0;
            for w in &r.windows {
                survivors.push(w.layouts_survived);
                classes.push(w.classes);
                scored.push(w.scored);
            }
            match r.outcome {
                SweepOutcome::Converged { shape, fs_hz, .. } => {
                    if recovered(&r, &t) {
                        converged += 1;
                        margins.push(r.margin);
                        // Counted from the report's own alias list, not from `shape != truth`. The old
                        // form could only ever print zero: `recovered` is reached by the exact-shape arm,
                        // which makes `shape != truth` false by construction.
                        aliased += usize::from(shape != t.layout.shape() && r.alias_shapes.contains(&t.layout.shape()));
                    } else {
                        wrong += 1;
                        wrongs.push(format!("wrong {}: {shape:?} at {fs_hz} Hz", f.subject));
                    }
                }
                other => stalls.push(format!("stalled {}: {other:?}", f.subject)),
            }
        }
        total_recovered += converged;
        total_wrong += wrong;
        total_cases += fixtures.len();
        println!(
            "\n{}\n  recovered {converged}/{} ({aliased} of them as an alias)  wrong {wrong}  stalled {}\
             \n  {:.0} ms per sweep of {WINDOWS} windows",
            t.name,
            fixtures.len(),
            stalls.len(),
            ms_total / fixtures.len() as f64
        );
        println!(
            "  survivors {:?}..{:?}  classes {:?}..{:?}  scorings {:?}..{:?}",
            survivors.iter().min(),
            survivors.iter().max(),
            classes.iter().min(),
            classes.iter().max(),
            scored.iter().min(),
            scored.iter().max()
        );
        if !margins.is_empty() {
            println!("  worst margin {:.3}", margins.iter().copied().fold(f64::INFINITY, f64::min));
        }
        for s in wrongs.iter().chain(stalls.iter()) {
            println!("  {s}");
        }
    }
    println!(
        "\nWHOLE MATRIX: recovered {total_recovered}/{total_cases} ({:.4})  wrong {total_wrong}",
        total_recovered as f64 / total_cases as f64
    );
    assert_eq!(total_cases, 65, "the matrix the rate is quoted over");
    assert!(
        total_recovered >= MIN_RECOVERED_OF_65,
        "recovery fell to {total_recovered}/65, under the {MIN_RECOVERED_OF_65} floor"
    );
    // A converged-but-wrong answer is worse than a stall: it is a decode nobody would question.
    assert!(total_wrong <= 1, "{total_wrong} cases converged on the wrong answer");
}

#[test]
#[ignore = "characterisation: what the rate-free prune sees; run with --release"]
fn characterise_the_prune() {
    let fixtures = ecg_fixtures();
    for t in truths() {
        let (mut true_r, mut other_r) = (Vec::new(), Vec::new());
        for f in fixtures.iter().take(4) {
            let (bytes, _) = stream(f, &t);
            for l in physio_algo::ecg::sweep::layout::candidates() {
                if l.sample_count(bytes.len()) < cfg().min_samples {
                    continue;
                }
                let Some(s) = layout_stats(&l.decode(&bytes)) else { continue };
                if l.shape() == t.layout.shape() && l.start_bit == t.layout.start_bit {
                    true_r.push(s.roughness);
                } else {
                    other_r.push(s.roughness);
                }
            }
        }
        other_r.sort_by(f64::total_cmp);
        let below = other_r.iter().filter(|v| **v <= cfg().max_roughness).count();
        println!(
            "{}\n  true roughness {:.4}..{:.4}   other rules: {} of {} below {:.2}, lowest {:.4}",
            t.name,
            true_r.iter().copied().fold(f64::INFINITY, f64::min),
            true_r.iter().copied().fold(0.0f64, f64::max),
            below,
            other_r.len(),
            cfg().max_roughness,
            other_r.first().copied().unwrap_or(f64::NAN)
        );
    }
}

#[test]
#[ignore = "characterisation: what competes with the truth; run with --release"]
fn characterise_the_leaderboard() {
    let f = &one("01");
    for t in truths() {
        let (bytes, beats) = stream(f, &t);
        let w = sweep_window(&bytes[..bytes.len() / WINDOWS], &beats, &cfg());
        println!("\n{}  truth {:?}", t.name, t.layout);
        for c in &w.leaderboard {
            let truth = c.layout.shape() == t.layout.shape() && c.fs_hz == t.fs_hz;
            println!(
                "  {} ans{} cls{} {:>6.1} Hz q{:.3} pass{} b{:.2}/x{:.2} k{:?} p{:?} tm{:?} hr{:?} | {}b {}{} start{} stride{} | ppg {:?}",
                if truth { "TRUE " } else { "     " },
                c.answer,
                c.class,
                c.fs_hz,
                c.quality,
                u8::from(c.passes),
                c.ecg.b_sqi,
                c.ecg.b_excess,
                c.ecg.k_sqi.map(|v| (v * 10.0).round() / 10.0),
                c.ecg.p_sqi.map(|v| (v * 100.0).round() / 100.0),
                c.ecg.template_sqi.map(|v| (v * 100.0).round() / 100.0),
                c.ecg.mean_hr_bpm.map(|v| v.round()),
                c.layout.bits,
                if c.layout.signed { "s" } else { "u" },
                if c.layout.order == BitOrder::LsbFirst { "le" } else { "be" },
                c.layout.start_bit,
                c.layout.stride_bits,
                c.ppg.map(|p| (p.offset_ms, p.matched, p.ppg_beats, (p.match_fraction * 100.0).round(), p.fs_solved_hz.map(|v| v.round())))
            );
        }
    }
}

#[test]
#[ignore = "characterisation: one window's wall clock; run with --release"]
fn characterise_one_window_wall_clock() {
    let f = &one("01");
    for t in truths() {
        let (bytes, beats) = stream(f, &t);
        let chunk = bytes.len() / WINDOWS;
        let start = Instant::now();
        let w = sweep_window(&bytes[..chunk], &beats, &cfg());
        let ms = start.elapsed().as_secs_f64() * 1000.0;
        println!(
            "{}\n  {chunk} bytes -> {} of {} rules decoded, {} survived, {} classes, {} scorings, {ms:.1} ms",
            t.name, w.layouts_decoded, w.layouts_enumerated, w.layouts_survived, w.classes, w.scored
        );
    }
}
