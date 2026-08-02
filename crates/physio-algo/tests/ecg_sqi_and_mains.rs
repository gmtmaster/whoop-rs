//! Signal-quality gate and mains anchor over real overnight ECG, with ground truth.
//!
//! Positives are the 13 AAUWSS subjects; negatives are matched to them (same length, mean and SD) so a
//! separation cannot come from amplitude. The mains test resamples each subject to a rate the anchor
//! is not told, injects 50 Hz hum, and checks the rate that comes back — recovery is measured against
//! a known answer rather than asserted to be plausible.
//!
//! One finding shapes the whole file: **this corpus carries real 50 Hz mains**, so the anchor recovers
//! its true 200 Hz rate from the untouched recordings. The hum-free case therefore has to be built by
//! notching it out rather than by assuming a clinical amplifier is clean.
//!
//! Run it in release; the scan is O(N²) in the window length and there are hundreds of them:
//!   cargo test -p physio-algo --release --test ecg_sqi_and_mains -- --nocapture

mod ecg_corpus;

use ecg_corpus::*;
use physio_algo::ecg::mains::{MainsAnchor, MainsUnavailable, MAINS_HZ_EU};
use physio_algo::ecg::score::{
    score, EcgScore, BAS_SQI_MIN, B_SQI_MIN, K_SQI_MIN, P_SQI_MAX, P_SQI_MIN, TEMPLATE_SQI_MIN,
};
use physio_algo::ecg::{mains_anchor, Periodogram};

/// Windows the indices are measured over. Ten seconds holds ten beats at 60 bpm and gives 0.1 Hz
/// spectral resolution at 200 Hz, and three of them fit in one 30 s epoch.
const WINDOW_S: f64 = 10.0;
/// The same length for the mains scan. Longer is more precise but the scan cost grows as N².
const MAINS_WINDOW_S: f64 = 10.0;

fn chunks(samples: &[f64], fs_hz: f64, seconds: f64) -> Vec<Vec<f64>> {
    let n = (seconds * fs_hz) as usize;
    samples.chunks(n).filter(|c| c.len() == n).map(|c| c.to_vec()).collect()
}

/// The five indices in a fixed order, with a missing one as NaN so it cannot quietly vanish.
fn indices(s: &EcgScore) -> [f64; 5] {
    [
        s.b_sqi,
        s.k_sqi.unwrap_or(f64::NAN),
        s.p_sqi.unwrap_or(f64::NAN),
        s.bas_sqi.unwrap_or(f64::NAN),
        s.template_sqi.unwrap_or(f64::NAN),
    ]
}

const INDEX_NAMES: [&str; 5] = ["bSQI", "kSQI", "pSQI", "basSQI", "tmplSQI"];

fn column(rows: &[[f64; 5]], k: usize) -> Vec<f64> {
    rows.iter().map(|r| r[k]).filter(|v| v.is_finite()).collect()
}

/// Which index rejected a window, in the order the verdict reports them.
fn failed_indices(s: &EcgScore) -> Vec<&'static str> {
    let v = &s.verdict;
    [("bSQI", v.b_ok), ("kSQI", v.k_ok), ("pSQI", v.p_ok), ("basSQI", v.bas_ok), ("tmplSQI", v.template_ok), ("HR", v.hr_ok)]
        .iter()
        .filter(|(_, ok)| !ok)
        .map(|(n, _)| *n)
        .collect()
}

/// Per-negative-kind roll-up: how many windows, how many slipped through, every index value, and which
/// index rejected each window.
#[derive(Default)]
struct NegativeTally {
    name: &'static str,
    windows: usize,
    accepted: usize,
    rows: Vec<[f64; 5]>,
    failed: Vec<&'static str>,
}

/// The matched negative set for one subject, each named.
fn negatives(f: &Fixture, ppg: &Fixture, seed: u64) -> Vec<(&'static str, Vec<f64>)> {
    let beats = score(&f.samples, f.fs_hz).beats.max(1);
    let period = f.samples.len() as f64 / beats as f64;
    let ppg_up = resample(&ppg.samples, ppg.fs_hz, f.fs_hz);
    vec![
        ("gaussian", gaussian_like(&f.samples, seed)),
        ("shuffled", shuffled(&f.samples, seed)),
        ("sawtooth", sawtooth_like(&f.samples, period)),
        ("ppg", matched_to(&ppg_up, &f.samples)),
    ]
}

#[test]
fn the_gate_accepts_real_ecg_and_no_matched_negative() {
    let fx = ecg_fixtures();
    assert_eq!(fx.len(), 13, "AAUWSS ships 13 ECG subjects; found {}", fx.len());
    let ppg = ppg_fixtures();
    assert_eq!(ppg.len(), 1, "one published PPG subject; found {}", ppg.len());

    let (mut pos, mut neg) = (Vec::new(), Vec::new());
    let (mut pos_accept, mut neg_accept, mut pos_windows) = (0usize, 0usize, 0usize);
    let mut by_negative: Vec<NegativeTally> = Vec::new();

    println!(
        "\n{:>5} {:>4} {:>6} {:>7} {:>6} {:>7} {:>7} {:>6} {:>4} verdict",
        "subj", "win", "bSQI", "kSQI", "pSQI", "basSQI", "tmplSQI", "bpm", "pass"
    );
    for f in &fx {
        for (w, win) in chunks(&f.samples, f.fs_hz, WINDOW_S).iter().enumerate() {
            let s = score(win, f.fs_hz);
            println!(
                "{:>5} {:>4} {:>6.3} {:>7.2} {:>6.3} {:>7.3} {:>7.3} {:>6.1} {:>4} {} {:?}",
                f.subject,
                w,
                s.b_sqi,
                s.k_sqi.unwrap_or(f64::NAN),
                s.p_sqi.unwrap_or(f64::NAN),
                s.bas_sqi.unwrap_or(f64::NAN),
                s.template_sqi.unwrap_or(f64::NAN),
                s.mean_hr_bpm.unwrap_or(f64::NAN),
                s.verdict.passed,
                if s.verdict.accepted { "ACCEPT" } else { "reject" },
                failed_indices(&s)
            );
            pos.push(indices(&s));
            pos_windows += 1;
            pos_accept += usize::from(s.verdict.accepted);
        }
    }

    for (k, f) in fx.iter().enumerate() {
        for (name, sig) in negatives(f, &ppg[0], 0x5C1_0000u64.wrapping_add(k as u64)) {
            let slot = match by_negative.iter().position(|t| t.name == name) {
                Some(i) => i,
                None => {
                    by_negative.push(NegativeTally { name, ..NegativeTally::default() });
                    by_negative.len() - 1
                }
            };
            for win in chunks(&sig, f.fs_hz, WINDOW_S) {
                let s = score(&win, f.fs_hz);
                by_negative[slot].windows += 1;
                by_negative[slot].accepted += usize::from(s.verdict.accepted);
                by_negative[slot].rows.push(indices(&s));
                by_negative[slot].failed.extend(failed_indices(&s));
                neg.push(indices(&s));
                neg_accept += usize::from(s.verdict.accepted);
                if s.verdict.accepted {
                    println!("  ACCEPTED NEGATIVE {name} subject {}: {s:?}", f.subject);
                }
            }
        }
    }

    println!("\n{:>9} {:>9} {:>9} {:>9} | {:>9} {:>9}", "index", "pos_min", "pos_med", "pos_max", "neg_min", "neg_max");
    for (i, name) in INDEX_NAMES.iter().enumerate() {
        let (p, n) = (column(&pos, i), column(&neg, i));
        println!(
            "{:>9} {:>9.3} {:>9.3} {:>9.3} | {:>9.3} {:>9.3}",
            name,
            min_of(&p),
            median_of(&p),
            max_of(&p),
            min_of(&n),
            max_of(&n)
        );
    }

    // Which index does the work against which negative. An index that never rejects anything is not a
    // detector, and one that rejects every negative on its own would make the other four redundant.
    println!("\n{:>10} {:>7} {:>9}  rejected by (count)", "negative", "windows", "accepted");
    for t in &by_negative {
        let tally: Vec<(&str, usize)> = ["bSQI", "kSQI", "pSQI", "basSQI", "tmplSQI", "HR"]
            .into_iter()
            .map(|idx| (idx, t.failed.iter().filter(|v| **v == idx).count()))
            .filter(|(_, c)| *c > 0)
            .collect();
        let ranges: Vec<String> = (0..5)
            .map(|i| format!("{}={:.2}..{:.2}", INDEX_NAMES[i], min_of(&column(&t.rows, i)), max_of(&column(&t.rows, i))))
            .collect();
        println!("{:>10} {:>7} {:>9}  {tally:?}", t.name, t.windows, t.accepted);
        println!("{:>10}  {}", " ", ranges.join("  "));
    }
    println!("\naccepted: {pos_accept}/{pos_windows} positive windows, {neg_accept}/{} negative windows", neg.len());
    println!(
        "gate: bSQI>={B_SQI_MIN} kSQI>={K_SQI_MIN} pSQI in {P_SQI_MIN}..={P_SQI_MAX} tmpl>={TEMPLATE_SQI_MIN} (basSQI>={BAS_SQI_MIN} reported, not gated)"
    );

    assert_eq!(neg_accept, 0, "a negative was accepted — the gate is not a gate");
    assert!(
        pos_accept * 10 >= pos_windows * 9,
        "only {pos_accept}/{pos_windows} real-ECG windows accepted; a gate this strict rejects the signal it is for"
    );
    // Every gated index has to earn its place by rejecting something.
    for idx in ["bSQI", "kSQI", "pSQI", "tmplSQI", "HR"] {
        let rejects: usize = by_negative.iter().map(|t| t.failed.iter().filter(|v| **v == idx).count()).sum();
        assert!(rejects > 0, "{idx} never rejected a negative — it is not doing anything");
    }
}

/// The characterisation sweep: 13 subjects x 5 rates x 7 hum amplitudes, 455 scans. Ignored by
/// default because the scan is O(N²) and this costs ~100 s in the debug profile against 16 s for the
/// rest of the file. Run it after any change to the anchor:
///   cargo test -p physio-algo --release --test ecg_sqi_and_mains -- --ignored --nocapture
#[test]
#[ignore = "455 O(N^2) scans, ~100 s in debug; run with --release"]
fn mains_anchor_recovers_a_resampled_rate_and_says_where_it_stops() {
    let fx = ecg_fixtures();
    let rates = [150.0, 200.0, 256.0, 333.0, 500.0];
    let amps = [0.50, 0.20, 0.10, 0.05, 0.02, 0.01, 0.0];

    println!("\n{:>6} {:>7} {:>8} {:>7} {:>9} {:>9} {:>9}", "amp", "runs", "unavail", "wrong", "med_err", "p90_err", "max_err");
    let mut lowest_clean = f64::NAN;
    for &amp in &amps {
        let (mut errs, mut unavailable, mut wrong, mut runs) = (Vec::new(), 0usize, 0usize, 0usize);
        for f in &fx {
            for &fs in &rates {
                let base = resample(&f.samples, f.fs_hz, fs);
                let win = match chunks(&base, fs, MAINS_WINDOW_S).into_iter().next() {
                    Some(w) => w,
                    None => continue,
                };
                runs += 1;
                match mains_anchor(&add_hum(&win, fs, MAINS_HZ_EU, amp)) {
                    MainsAnchor::Found(fix) => {
                        let err = (fix.fs_hz - fs).abs();
                        if err > 1.0 {
                            wrong += 1;
                        }
                        errs.push(err);
                    }
                    MainsAnchor::Unavailable(_) => unavailable += 1,
                }
            }
        }
        errs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p90 = if errs.is_empty() { f64::NAN } else { errs[(errs.len() * 9 / 10).min(errs.len() - 1)] };
        println!(
            "{amp:>6.2} {runs:>7} {unavailable:>8} {wrong:>7} {:>9.3} {p90:>9.3} {:>9.3}",
            median_of(&errs),
            max_of(&errs)
        );
        if unavailable == 0 && wrong == 0 && amp > 0.0 {
            lowest_clean = amp;
        }
    }
    println!("\nlowest injected hum with 0 unavailable and 0 wrong: {lowest_clean} x SD");
    println!("(amp 0.00 is the corpus' own mains, which is why it still resolves at all)");
    // Frozen: the anchor has to survive down to a fifth of the signal's own SD. Below that it starts
    // returning nothing or the wrong harmonic, and that boundary is the result, not a bug to hide.
    assert!(lowest_clean <= 0.2, "the anchor now needs {lowest_clean} x SD of hum; it used to work at 0.2");

    // Frozen from the measured run: at half the signal's own SD every subject at every rate comes back
    // within a hertz. A regression here means the anchor stopped anchoring.
    for f in &fx {
        for &fs in &rates {
            let base = resample(&f.samples, f.fs_hz, fs);
            let win = chunks(&base, fs, MAINS_WINDOW_S).into_iter().next().unwrap();
            let fix = match mains_anchor(&add_hum(&win, fs, MAINS_HZ_EU, 0.5)) {
                MainsAnchor::Found(fix) => fix,
                other => panic!("subject {} at {fs} Hz: {other:?}", f.subject),
            };
            assert!((fix.fs_hz - fs).abs() < 1.0, "subject {} at {fs} Hz: got {} Hz ({fix:?})", f.subject, fix.fs_hz);
        }
    }
}

#[test]
fn the_anchor_reads_the_corpus_own_mains_and_loses_it_when_notched() {
    // These recordings carry real 50 Hz interference, so the anchor recovers their true 200 Hz with no
    // injection at all — the strongest evidence available that the method works on real hardware data.
    let fx = ecg_fixtures();
    let (mut found, mut absent) = (Vec::new(), 0usize);
    for f in &fx {
        match mains_anchor(&f.samples) {
            MainsAnchor::Found(fix) => {
                println!(
                    "subject {}: {:.3} Hz (true {:.0}), prominence {:.1} dB, harmonics {}, conf {:.2}",
                    f.subject, fix.fs_hz, f.fs_hz, fix.prominence_db, fix.harmonics_confirmed, fix.confidence
                );
                assert!((fix.fs_hz - f.fs_hz).abs() < 1.0, "subject {}: {fix:?}", f.subject);
                found.push(fix.fs_hz);
            }
            MainsAnchor::Unavailable(MainsUnavailable::NoPeak { best_prominence_db }) => {
                println!("subject {}: no anchor, best prominence {best_prominence_db:.1} dB", f.subject);
                absent += 1;
            }
            MainsAnchor::Unavailable(other) => panic!("subject {}: {other:?}", f.subject),
        }
    }
    println!("{}/{} subjects anchored on their own hum, {absent} had none strong enough", found.len(), fx.len());
    assert!(found.len() * 2 > fx.len(), "only {} of {} subjects anchored", found.len(), fx.len());

    // Now the production case: a front end that notches the line. A two-sample comb has an exact zero
    // at a quarter of the sample rate, which at 200 Hz is exactly 50 Hz (and takes 150 Hz with it).
    // Nothing is left to anchor on, and the honest answer is that there is no answer.
    let mut still_found = Vec::new();
    for f in &fx {
        let notched: Vec<f64> = (2..f.samples.len()).map(|i| f.samples[i] + f.samples[i - 2]).collect();
        match mains_anchor(&notched) {
            MainsAnchor::Unavailable(reason) => println!("subject {} notched: {reason:?}", f.subject),
            MainsAnchor::Found(fix) => {
                println!("subject {} notched: STILL FOUND {fix:?}", f.subject);
                still_found.push(f.subject.clone());
            }
        }
    }
    assert!(still_found.is_empty(), "notching 50 Hz out did not remove the anchor for {still_found:?}");
}

#[test]
fn a_loud_in_band_tone_captures_the_anchor_and_that_is_reported_not_hidden() {
    // A narrow tone is a narrow tone. Nothing in a spectrum says a 20 Hz artefact is not mains, so the
    // anchor answers to it — but the true peak survives as the runner-up with a stated margin, which is
    // the whole reason a runner-up is on the struct.
    let f = &ecg_fixtures()[0];
    let fs = f.fs_hz;
    let s = sd(&f.samples);
    let loud: Vec<f64> = f
        .samples
        .iter()
        .enumerate()
        .map(|(i, v)| v + 3.0 * s * (2.0 * std::f64::consts::PI * 20.0 * i as f64 / fs).sin())
        .collect();
    match mains_anchor(&loud) {
        MainsAnchor::Found(fix) => {
            println!("loud 20 Hz tone: {:.1} Hz, runner-up {:?}, margin {:.1} dB", fix.fs_hz, fix.runner_up_fs_hz, fix.margin_db);
            assert!((fix.fs_hz - 500.0).abs() < 5.0, "20 Hz at 200 Hz reads as 500: {fix:?}");
            let runner = fix.runner_up_fs_hz.expect("the real 200 Hz peak must still be a candidate");
            assert!((runner - 200.0).abs() < 2.0, "runner-up should be the true rate, got {runner}");
            // The morphology side is what refuses it: at 500 Hz these beats imply 2.5x the heart rate.
            assert!(!score(&loud, fix.fs_hz).verdict.accepted, "the wrong rate must fail the SQI gate");
        }
        MainsAnchor::Unavailable(reason) => panic!("expected the artefact to win, got {reason:?}"),
    }

    // And the spectral primitive places a tone where it belongs, on real data.
    let pg = Periodogram::new(&add_hum(&f.samples, fs, MAINS_HZ_EU, 0.5));
    let at_mains = pg.power_at(MAINS_HZ_EU / fs);
    let off_mains = pg.power_at(MAINS_HZ_EU / fs + 20.0 * pg.bin_width());
    assert!(at_mains > 100.0 * off_mains, "hum {at_mains:e} vs neighbour {off_mains:e}");
}
