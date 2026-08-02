//! Morphology gate: run the three ECG-only measures over real overnight ECG and over matched
//! negatives, and check what each can and cannot establish.
//!
//! Positives are the 13 AAUWSS subjects, committed in-tree so this runs on a clean checkout. Negatives
//! are matched to the subject they came from — same length, mean and standard deviation — so nothing
//! here can separate on amplitude or duration alone.
//!
//! The labelled-rhythm measurement lives in `examples/ecg_morphology_corpus.rs` instead, because that
//! corpus sits outside the repo and a test pointing at it would pass by skipping.
//!
//!   cargo test -p physio-algo --test ecg_morphology -- --nocapture

mod ecg_corpus;

use ecg_corpus::*;
use physio_algo::ecg::detect_pan_tompkins;
use physio_algo::ecg::morphology::p_wave::{P_MIN_BEATS, P_MIN_DEFLECTION_RATIO};
use physio_algo::ecg::morphology::{
    atrial_band, morphology, p_wave, AtrialBandEvidence, BeatConsistency, PWaveFinding,
};

/// Windows the measures are taken over. Thirty seconds is one AAUWSS epoch and ~30 beats at a sleeping
/// rate, which is the count the ensemble noise floor is sized against.
fn peaks_of(x: &[f64], fs_hz: f64) -> Vec<usize> {
    detect_pan_tompkins(x, fs_hz)
}

#[test]
fn real_sleeping_ecg_carries_a_p_wave_on_every_subject() {
    let fx = ecg_fixtures();
    assert_eq!(fx.len(), 13, "AAUWSS ships 13 ECG subjects; found {}", fx.len());

    println!("\n{:>5} {:>6} {:>7} {:>7} {:>7} {:>8} {:>6}", "subj", "beats", "frac", "consis", "amp/R", "noise/R", "conf");
    let (mut fracs, mut amps) = (Vec::new(), Vec::new());
    for f in &fx {
        let peaks = peaks_of(&f.samples, f.fs_hz);
        let r = p_wave(&f.samples, f.fs_hz, &peaks);
        println!(
            "{:>5} {:>6} {:>7.3} {:>7.3} {:>7.3} {:>8.4} {:>6.2}",
            f.subject,
            r.beats_examined,
            r.present_fraction.unwrap_or(f64::NAN),
            r.consistency.unwrap_or(f64::NAN),
            r.amplitude_ratio.unwrap_or(f64::NAN),
            r.noise_ratio.unwrap_or(f64::NAN),
            r.confidence
        );
        assert_eq!(r.finding, PWaveFinding::Present, "subject {}: {r:?}", f.subject);
        assert!(r.beats_examined >= P_MIN_BEATS, "subject {}: {r:?}", f.subject);
        fracs.push(r.present_fraction.unwrap());
        amps.push(r.amplitude_ratio.unwrap());
    }
    let worst_frac = min_of(&fracs);
    let worst_amp = min_of(&amps);
    println!("real ECG: worst present fraction {worst_frac:.3}, worst deflection {worst_amp:.3} of R");
    // The anti-dust floor must sit clear of the real distribution or it starts reporting genuine P
    // waves as absent — which an earlier value of it did, on two of these subjects. This asserts the
    // margin, not just the verdict.
    assert!(worst_amp > 2.0 * P_MIN_DEFLECTION_RATIO, "worst real deflection {worst_amp:.3}");
    assert!(worst_frac >= 0.90, "worst real present fraction {worst_frac:.3}");
}

#[test]
fn noise_is_rejected_but_a_pulse_waveform_is_not() {
    // A measure that has never refused anything is not a measure — and one that refuses everything is
    // not one either, so what it does NOT reject is asserted here too.
    //
    // The PPG negative is reported Present, and that is a real limitation rather than a bug: this
    // measure asks "is there a consistent deflection ahead of each peak", and a pulse waveform has one
    // (the foot of the pulse) in exactly that position. It presumes its input is already established as
    // an ECG — `ecg::score` is what establishes that, and it rejects a PPG on its band ratio. Pinned so
    // that if a change ever fixes it, this line is what says so.
    println!("\n{:>5} | {:>26} | {:>26} | {:>26}", "subj", "gaussian", "shuffled", "matched PPG");
    let ppg = ppg_fixtures().pop().expect("one PPG subject");
    let mut ppg_present = 0usize;
    for (k, f) in ecg_fixtures().iter().enumerate() {
        let seed = 0xEC6_1000 + k as u64;
        let negatives: [(&str, Vec<f64>); 3] = [
            ("gaussian", gaussian_like(&f.samples, seed)),
            ("shuffled", shuffled(&f.samples, seed)),
            ("ppg", matched_to(&resample(&ppg.samples, ppg.fs_hz, f.fs_hz), &f.samples)),
        ];
        let mut cells = Vec::new();
        for (name, sig) in negatives {
            let r = p_wave(&sig, f.fs_hz, &peaks_of(&sig, f.fs_hz));
            if name == "ppg" {
                ppg_present += usize::from(r.finding == PWaveFinding::Present);
            } else {
                assert_ne!(r.finding, PWaveFinding::Present, "subject {} {name}: {r:?}", f.subject);
            }
            cells.push(format!("{:>26}", format!("{:?}", r.finding)));
        }
        println!("{:>5} | {} | {} | {}", f.subject, cells[0], cells[1], cells[2]);
    }
    println!("matched PPG read Present on {ppg_present} of 13 - the documented limitation");
    assert!(ppg_present > 0, "the PPG limitation is gone - re-derive it before deleting this test");
}

#[test]
fn stripping_the_pr_segment_flips_the_finding_and_leaves_the_qrs_alone() {
    // The one manipulation that targets exactly what this measure claims to see: replace each PR window
    // with a straight line plus the record's own noise, and nothing else. If the finding did not move,
    // the measure would be reading something other than the PR segment.
    let mut flipped = 0usize;
    for (k, f) in ecg_fixtures().iter().enumerate() {
        let peaks = peaks_of(&f.samples, f.fs_hz);
        let before = p_wave(&f.samples, f.fs_hz, &peaks);
        let after = p_wave(&flatten_pr(&f.samples, f.fs_hz, &peaks, 0xEC6_2000 + k as u64), f.fs_hz, &peaks);
        let consistency_before = morphology(&f.samples, f.fs_hz, &peaks).beat_consistency;
        println!("subject {}: {:?} -> {:?}", f.subject, before.finding, after.finding);
        assert_eq!(before.finding, PWaveFinding::Present, "subject {}", f.subject);
        assert_ne!(after.finding, PWaveFinding::Present, "subject {}: {after:?}", f.subject);
        if after.finding == PWaveFinding::Absent {
            flipped += 1;
        }
        // The QRS window is untouched, so beat-template consistency must not have moved with it. This is
        // the independence the module refuses to fuse away.
        assert!(matches!(consistency_before, BeatConsistency::Measured { .. }), "subject {}", f.subject);
    }
    println!("{flipped} of 13 flipped all the way to Absent");
    assert_eq!(flipped, 13, "only {flipped} of 13 reached Absent");
}

#[test]
fn the_atrial_band_ratio_tracks_energy_put_into_its_band_and_nowhere_else() {
    // NO GROUND TRUTH EXISTS for this index — no threshold is asserted, and none is defined. So what is
    // established here is that it REACTS, which is the strongest claim available without a reference.
    //
    // `ratio` is P(4-9 Hz) over P(1-30 Hz), a sub-band over a band that contains it, so it lies in
    // [0, 1] by construction: asserting that range proves nothing at all, and an implementation that
    // returned any fixed number would satisfy it. Instead a tone is put inside the band and then outside
    // it, using the SAME peaks both times so beat detection is not a variable.
    println!("\n{:>5} {:>9} {:>11} {:>8} {:>9} {:>9}", "subj", "segments", "segment ms", "ratio", "+6Hz", "+20Hz");
    let (mut measured, mut rose, mut fell) = (0usize, 0usize, 0usize);
    for f in ecg_fixtures() {
        let peaks = peaks_of(&f.samples, f.fs_hz);
        let ratio_of = |sig: &[f64]| match atrial_band(sig, f.fs_hz, &peaks) {
            AtrialBandEvidence::Measured(a) => Some(a.ratio),
            AtrialBandEvidence::Indeterminate(_) => None,
        };
        match atrial_band(&f.samples, f.fs_hz, &peaks) {
            AtrialBandEvidence::Measured(a) => {
                let in_band = ratio_of(&with_tone(&f.samples, f.fs_hz, 6.0, 0.30));
                let out_band = ratio_of(&with_tone(&f.samples, f.fs_hz, 20.0, 0.30));
                println!(
                    "{:>5} {:>9} {:>11.0} {:>8.4} {:>9} {:>9}",
                    f.subject,
                    a.segments,
                    a.median_segment_ms,
                    a.ratio,
                    in_band.map_or("-".into(), |v| format!("{v:.4}")),
                    out_band.map_or("-".into(), |v| format!("{v:.4}"))
                );
                assert!(a.median_segment_ms > 0.0 && a.segments > 0, "subject {}: {a:?}", f.subject);
                measured += 1;
                rose += usize::from(in_band.is_some_and(|v| v > a.ratio));
                fell += usize::from(out_band.is_some_and(|v| v < a.ratio));
            }
            AtrialBandEvidence::Indeterminate(l) => println!("{:>5} {l:?}", f.subject),
        }
        // No beats, no quiet segment between them, no ratio. The refusal is the answer.
        assert!(matches!(
            atrial_band(&f.samples, f.fs_hz, &[]),
            AtrialBandEvidence::Indeterminate(_)
        ));
    }
    println!("{measured} measured: {rose} rose on a 6 Hz tone, {fell} fell on a 20 Hz one");
    assert!(measured > 0, "nothing was measured, so nothing was tested");
    // Every one, both ways. A constant, a mean or any index blind to its own band fails this.
    assert_eq!(rose, measured, "only {rose} of {measured} rose on a tone inside the 4-9 Hz band");
    assert_eq!(fell, measured, "only {fell} of {measured} fell on a tone inside the reference band only");
}

/// The record plus a pure tone at `hz`, scaled to `amp_ratio` of the record's own spread.
fn with_tone(x: &[f64], fs_hz: f64, hz: f64, amp_ratio: f64) -> Vec<f64> {
    let amp = amp_ratio * sd(x);
    x.iter()
        .enumerate()
        .map(|(i, v)| v + amp * (2.0 * std::f64::consts::PI * hz * i as f64 / fs_hz).sin())
        .collect()
}

/// Replace every PR window with a straight line between its own end points plus the record's own
/// high-frequency noise, leaving the QRS, the T wave and the TP segment exactly as they were.
///
/// The line, rather than a flat level, is what makes this a clean manipulation: a constant leaves a
/// step at each boundary, and after the measure's low-pass that step is the same shape on every beat,
/// so it reads as a consistent deflection and the finding does not move. Measured while writing this —
/// a flat replacement left 1 of 13 still reporting Present at a deflection of 0.011 of R.
fn flatten_pr(x: &[f64], fs_hz: f64, peaks: &[usize], seed: u64) -> Vec<f64> {
    let mut out = x.to_vec();
    let lo = (0.260 * fs_hz).round() as usize;
    let hi = (0.060 * fs_hz).round() as usize;
    let mut rng = Rng(seed);
    for &p in peaks {
        if p < lo || p >= x.len() {
            continue;
        }
        let (start, end) = (p - lo, p - hi);
        // The local high-frequency spread, so the replacement is not flatter than the record it sits in.
        let steps: Vec<f64> = x[start..end].windows(2).map(|w| w[1] - w[0]).collect();
        let scale = if steps.is_empty() { 0.0 } else { sd(&steps) / 2f64.sqrt() };
        let (a, b) = (x[start], x[end]);
        let span = (end - start) as f64;
        for (k, v) in out[start..end].iter_mut().enumerate() {
            *v = a + (b - a) * k as f64 / span + scale * rng.gaussian();
        }
    }
    out
}
