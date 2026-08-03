//! The gates. Four kinds, meant to be read together:
//!
//! 1. the oracle's dot decoder pinned against literal Unicode codepoints, and cross-checked dot for
//!    dot against the shipping canvas — two independently written bit tables that must agree;
//! 2. a hand-typed frame the oracle reads correctly with no code having drawn it;
//! 3. round-trips: the SHIPPING strip renderer turns millivolts and seconds into characters, and the
//!    oracle measures the characters back;
//! 4. the same round-trips at deliberately WRONG scales, which the oracle has to reject.
//!
//! Section 3 used to run through a stand-in rasteriser written from the spec. It now drives
//! `ecg_render` itself, so what these numbers gate is the tool. They stay HAND-DERIVED from the spec:
//! re-deriving one from the renderer would make sections 4 and 5 pass on a broken scale.

use crate::braille::Canvas;
use crate::ecg_render::demo::FIXTURE_COUNTS_PER_MV;
use crate::ecg_render::frame;
use crate::ecg_render::plan::{
    fit, Request, Terminal, ASSUMED_SAMPLE_RATE_HZ as RATE_HZ, GUTTER_COLS,
};
use crate::ecg_render::renderer::{EcgRenderer, Sample};

use super::decode::{self, DotGrid, Extract};
use super::oracle::{self, ClaimedScale};
use super::synth::{self, Signal};

/// Reference scale throughout: 1 dot/mm, 10 mm/mV, 25 mm/s. Ten dots per millivolt, 12.5 columns per
/// second, so a 5-second strip is 63 columns.
fn reference() -> ClaimedScale {
    ClaimedScale::reference()
}

/// Render a signal through the strip renderer at a claimed scale, and return the strip's plot region
/// as the reader sees it: found by this module's own structural block finder, windowed past the
/// gutter. The grid is off, so every dot in the returned rows is trace.
fn plot(sig: &Signal, claim: ClaimedScale) -> Vec<String> {
    let req = Request {
        duration_s: sig.duration_s(),
        strip_s: sig.duration_s(),
        mm_per_s: claim.mm_per_s,
        mm_per_mv: claim.mm_per_mv,
        strip_mm: (sig.peak_to_peak_mv() * claim.mm_per_mv * 2.0).max(20.0),
        dots_per_mm_x: Some(claim.dots_per_mm),
        counts_per_mv: Some(FIXTURE_COUNTS_PER_MV),
        sample_rate_hz: sig.sample_rate_hz,
        grid: false,
        ..Request::default()
    };
    let plan = fit(&req, Terminal::new(4000, 4000)).expect("the harness terminal holds any of these");
    let mut r = EcgRenderer::new(plan);
    let samples: Vec<Sample> =
        sig.samples_mv.iter().map(|mv| Sample::new(mv * FIXTURE_COUNTS_PER_MV, true)).collect();
    r.push(&samples);
    r.finish();
    let lines = frame::frame(&r, false, None, 1);
    let block = decode::braille_row_blocks(&lines).first().cloned().expect("a strip was drawn");
    lines[block].iter().map(|l| l.chars().skip(GUTTER_COLS).collect()).collect()
}

fn grid_of(sig: &Signal, claim: ClaimedScale) -> DotGrid {
    decode::from_lines(&plot(sig, claim), Extract::All)
}

// ---- 1. the decoder, against Unicode and against the shipping encoder ------------------------

#[test]
fn braille_bits_are_the_unicode_layout() {
    // Codepoint -> the single dot it names, straight out of the Unicode block. External ground truth,
    // so an encoder and a decoder sharing a wrong table could not both pass.
    let pins = [
        ('\u{2801}', 0usize, 0usize),
        ('\u{2802}', 1, 0),
        ('\u{2804}', 2, 0),
        ('\u{2840}', 3, 0),
        ('\u{2808}', 0, 1),
        ('\u{2810}', 1, 1),
        ('\u{2820}', 2, 1),
        ('\u{2880}', 3, 1),
    ];
    for (ch, row, col) in pins {
        let dots = decode::cell_dots(ch);
        assert_eq!(decode::dot_bit(row, col), Some((ch as u32 - decode::BRAILLE_BASE).trailing_zeros()));
        for (r, dot_row) in dots.iter().enumerate() {
            for (c, on) in dot_row.iter().enumerate() {
                assert_eq!(*on, (r, c) == (row, col), "U+{:04X} dot ({r},{c})", ch as u32);
            }
        }
    }
    assert_eq!(decode::cell_dots('\u{28FF}'), [[true; 2]; 4], "U+28FF is all eight dots");
    assert_eq!(decode::cell_dots('\u{2800}'), [[false; 2]; 4], "U+2800 is the blank cell");
    assert_eq!(decode::cell_dots('x'), [[false; 2]; 4], "a non-braille character carries no dots");
    assert_eq!(decode::dot_bit(4, 0), None, "there is no fifth row in a cell");
}

#[test]
fn the_oracle_decoder_agrees_with_the_shipping_encoder_on_every_dot() {
    // Two independently written bit tables. Light one dot at a time on the canvas the renderer uses,
    // read the characters back with the oracle's own decoder, require the same position.
    let (w, h) = (7usize, 9usize);
    for y in 0..h {
        for x in 0..w {
            let mut c = Canvas::new(w, h);
            assert!(c.set(x as i32, y as i32));
            let grid = decode::from_lines(&c.render(), Extract::All);
            assert_eq!(grid.count(), 1, "one dot lit at ({x},{y})");
            assert!(grid.get(x, y), "the decoder puts the dot back at ({x},{y})");
        }
    }
}

#[test]
fn styled_cells_separate_the_grid_from_the_trace() {
    // A renderer draws its background grid in a dim SGR span; the trace stays default.
    let lines = vec![format!("\x1b[2m{}\x1b[0m{}", '\u{28FF}', '\u{2801}')];
    assert_eq!(decode::from_lines(&lines, Extract::All).count(), 9);
    assert_eq!(decode::from_lines(&lines, Extract::Unstyled).count(), 1);
    assert_eq!(decode::from_lines(&lines, Extract::Styled).count(), 8);
    // The escape must not consume a column: the unstyled dot is in cell 1, i.e. dot x = 2.
    assert!(decode::from_lines(&lines, Extract::Unstyled).get(2, 0));
}

#[test]
fn a_region_windows_out_the_banner_and_the_footer() {
    let lines: Vec<String> = ["ECG — UNCALIBRATED", "\u{2801}\u{2804}", "25 mm/s"].iter().map(|s| s.to_string()).collect();
    assert_eq!(decode::braille_row_blocks(&lines), vec![1..2]);
    let all = decode::from_lines(&lines, Extract::All);
    assert_eq!(all.count(), 2, "the text rows contribute no dots");
    let region = decode::from_lines_region(&lines, 1..2, 1..2, Extract::All);
    assert_eq!(region.count(), 1, "only the second cell of the strip row");
    assert!(region.get(0, 2), "U+2804 is dot row 2, and the window moved it to column 0");
}

// ---- 2. a frame nothing drew -----------------------------------------------------------------

#[test]
fn a_hand_typed_frame_measures_one_millivolt() {
    // Three cell rows. Dots at (0,0) and (0,10): row 10 is cell row 2, dot row 2 = U+2804. Counted by
    // hand, so this path never touches an encoder at all.
    let frame: Vec<String> = ["\u{2801}", "\u{2800}", "\u{2804}"].iter().map(|s| s.to_string()).collect();
    let grid = decode::from_lines(&frame, Extract::All);
    let amp = oracle::measure_amplitude(&grid, reference()).expect("dots present");
    assert_eq!((amp.top_row, amp.bottom_row), (0, 10));
    assert_eq!(amp.peak_to_peak_dots, 10);
    assert!((amp.peak_to_peak_mm - 10.0).abs() < 1e-12, "10 dots at 1 dot/mm is 10 mm");
    assert!((amp.peak_to_peak_mv - 1.0).abs() < 1e-12, "10 mm at 10 mm/mV is 1 mV");
}

// ---- 3. the synthetic signals have the dimensions they claim ---------------------------------

#[test]
fn calibration_pulse_is_one_millivolt_and_one_hertz() {
    let sig = synth::calibration_pulse(RATE_HZ, 5.0);
    assert_eq!(sig.len(), 2500);
    assert!((sig.duration_s() - 5.0).abs() < 1e-12);
    assert!((sig.peak_to_peak_mv() - 1.0).abs() < 1e-12);
    // Starts low, so the first rising edge is at 0.5 s and there are five in five seconds.
    assert_eq!(sig.samples_mv[0], 0.0);
    assert_eq!(sig.samples_mv[250], 1.0, "high from t = 0.5 s");
    assert_eq!(sig.samples_mv.windows(2).filter(|w| w[1] > w[0]).count(), 5);
}

#[test]
fn square_wave_peak_to_peak_is_its_amplitude() {
    for amp in [0.5, 1.0, 2.5] {
        assert!((synth::square_wave(RATE_HZ, 2.0, amp, 2.0).peak_to_peak_mv() - amp).abs() < 1e-12);
    }
}

#[test]
fn counts_need_a_supplied_scale() {
    let sig = synth::calibration_pulse(RATE_HZ, 1.0);
    assert_eq!(sig.to_counts(1000.0)[250], 1000, "1 mV at 1000 counts/mV");
    assert_eq!(sig.to_counts(1000.0)[0], 0);
}

#[test]
fn pqrst_peak_to_peak_is_one_and_a_quarter_r() {
    let r = 1.2;
    let sig = synth::pqrst(RATE_HZ, 4.0, 60.0, r);
    let hi = sig.samples_mv.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let lo = sig.samples_mv.iter().copied().fold(f64::INFINITY, f64::min);
    assert!((hi - r).abs() < 0.01 * r, "R peak is {hi}, wanted {r}");
    assert!((lo + 0.25 * r).abs() < 0.01 * r, "S trough is {lo}, wanted {}", -0.25 * r);
    assert!((sig.peak_to_peak_mv() - 1.25 * r).abs() < 0.01 * r, "documented 1.25 x R");
}

#[test]
fn flat_and_noise_are_the_no_contact_conditions() {
    assert_eq!(synth::flat(RATE_HZ, 1.0, 0.0).peak_to_peak_mv(), 0.0);
    let a = synth::noise_baseline(RATE_HZ, 1.0, 0.05, 7);
    let b = synth::noise_baseline(RATE_HZ, 1.0, 0.05, 7);
    let c = synth::noise_baseline(RATE_HZ, 1.0, 0.05, 8);
    assert_eq!(a.samples_mv, b.samples_mv, "a seed names a waveform");
    assert_ne!(a.samples_mv, c.samples_mv);
    let rms = (a.samples_mv.iter().map(|v| v * v).sum::<f64>() / a.len() as f64).sqrt();
    assert!((rms - 0.05).abs() < 0.05 * 0.15, "rms {rms} within 15% of the 0.05 asked for");
}

#[test]
fn snr_noise_refuses_a_flat_base() {
    let flat = synth::flat(RATE_HZ, 1.0, 0.0);
    assert!(synth::add_noise_snr(&flat, 20.0, 1).is_none(), "an SNR against zero power is undefined");

    let base = synth::calibration_pulse(RATE_HZ, 2.0);
    let noisy = synth::add_noise_snr(&base, 20.0, 1).expect("a square wave carries power");
    let sig_rms = (base.samples_mv.iter().map(|v| v * v).sum::<f64>() / base.len() as f64).sqrt();
    let err: Vec<f64> = noisy.samples_mv.iter().zip(&base.samples_mv).map(|(n, b)| n - b).collect();
    let noise_rms = (err.iter().map(|v| v * v).sum::<f64>() / err.len() as f64).sqrt();
    let achieved_db = 20.0 * (sig_rms / noise_rms).log10();
    assert!((achieved_db - 20.0).abs() < 1.5, "achieved {achieved_db} dB, asked 20");
}

#[test]
fn intermittent_spans_alternate_and_cover_the_signal() {
    let trace = synth::calibration_pulse(RATE_HZ, 5.0);
    let inter = synth::intermittent_contact(&trace, 1.0, 0.5, 0.4, 3);
    assert_eq!(inter.signal.len(), trace.len());
    assert!(inter.spans[0].contact, "starts in contact");

    let mut cursor = 0usize;
    for (i, span) in inter.spans.iter().enumerate() {
        assert_eq!(span.start_sample, cursor, "span {i} is contiguous");
        assert_eq!(span.contact, i % 2 == 0, "spans alternate");
        assert!((span.start_s - span.start_sample as f64 / RATE_HZ).abs() < 1e-12);
        if span.contact {
            assert_eq!(
                &inter.signal.samples_mv[span.start_sample..span.end_sample],
                &trace.samples_mv[span.start_sample..span.end_sample],
                "a contact span is the real trace, untouched"
            );
        }
        cursor = span.end_sample;
    }
    assert_eq!(cursor, trace.len(), "the spans cover the whole signal");
    assert_eq!(inter.spans.iter().filter(|s| s.contact).count(), 4);
}

// ---- 4. the round trip: draw at a claimed scale, measure it back out of the characters --------

#[test]
fn the_reference_scale_restates_the_spec() {
    // Every figure the spec quotes, derived from the claim alone. A slip between dot columns and cell
    // columns shows up here as a clean factor of two, before it can reach a measurement.
    let c = reference();
    assert!((c.claimed_dots_per_mv() - 10.0).abs() < 1e-12, "10 mm/mV at 1 dot/mm is 10 dots/mV");
    assert!((c.claimed_s_per_dot_column() - 0.04).abs() < 1e-12);
    assert!((c.claimed_s_per_cell_column() - 0.08).abs() < 1e-12);
    assert!((c.claimed_cell_columns_per_second() - 12.5).abs() < 1e-9, "12.5 columns a second");
    assert_eq!((5.0 * c.claimed_cell_columns_per_second()).ceil() as usize, 63, "a 5 s strip is 63 columns");
}

#[test]
fn oracle_recovers_one_millivolt_and_one_hertz_from_the_characters() {
    let sig = synth::calibration_pulse(RATE_HZ, 5.0);
    let claim = reference();
    let lines = plot(&sig, claim);
    // Hand-derived from the spec: 25 dots/s x 5 s = 126 dot columns = 63 braille columns.
    assert_eq!(lines[0].chars().count(), 63, "a 5-second strip is 63 columns");
    let grid = decode::from_lines(&lines, Extract::All);

    let amp = oracle::measure_amplitude(&grid, claim).expect("dots present");
    assert_eq!(amp.peak_to_peak_dots, 10, "1 mV at 10 mm/mV and 1 dot/mm is 10 dots");
    assert!((amp.peak_to_peak_mv - 1.0).abs() < 1e-12, "measured {} mV", amp.peak_to_peak_mv);

    let time = oracle::measure_time(&grid, 1.0, claim).expect("edges present");
    assert_eq!(time.cycles, 4, "five rising edges over five seconds");
    assert!((time.dot_columns_per_cycle - 25.0).abs() < 1e-12, "{} dots/cycle", time.dot_columns_per_cycle);
    assert!((time.cell_columns_per_cycle - 12.5).abs() < 1e-12, "12.5 cells a second at 25 mm/s");
    assert!((time.s_per_dot_column - claim.claimed_s_per_dot_column()).abs() < 1e-12);
    assert!((time.s_per_cell_column - 0.08).abs() < 1e-9, "2 dots / 1 dot per mm / 25 mm/s = 0.08 s");
    assert!((time.mm_per_s_depicted - 25.0).abs() < 1e-9, "depicts {} mm/s", time.mm_per_s_depicted);

    let v = oracle::check(&grid, claim, 1.0, 1.0, 1e-6).expect("both axes measurable");
    assert!(v.agrees, "the picture agrees with its claim: {v:?}");
}

#[test]
fn a_pqrst_measures_back_to_its_own_peak_to_peak() {
    let sig = synth::pqrst(RATE_HZ, 5.0, 60.0, 1.2);
    let claim = reference();
    let amp = oracle::measure_amplitude(&grid_of(&sig, claim), claim).expect("dots present");
    let expected = sig.peak_to_peak_mv();
    // One dot is 0.1 mV at this scale; rounding at both ends can cost one and a half.
    assert!(
        (amp.peak_to_peak_mv - expected).abs() <= 0.15,
        "picture {} mV vs signal {expected} mV",
        amp.peak_to_peak_mv
    );
}

// ---- 5. the negative tests: the oracle must FAIL a wrongly-scaled picture ---------------------

#[test]
fn a_wrong_dots_per_mm_claim_is_caught() {
    let grid = grid_of(&synth::calibration_pulse(RATE_HZ, 5.0), reference());
    // The picture is unchanged; only the claim is wrong. Twice the dots per mm halves every reading.
    let wrong = ClaimedScale { dots_per_mm: 2.0, ..reference() };
    let amp = oracle::measure_amplitude(&grid, wrong).expect("dots present");
    assert!((amp.peak_to_peak_mv - 0.5).abs() < 1e-12, "reports what it sees: {} mV", amp.peak_to_peak_mv);

    let v = oracle::check(&grid, wrong, 1.0, 1.0, 0.01).expect("both axes measurable");
    assert!(!v.agrees, "a halved picture must not pass as 1 mV: {v:?}");
    assert!((v.amplitude_error - 0.5).abs() < 1e-12);
    assert!((v.time_error - 0.5).abs() < 1e-9);
}

#[test]
fn a_silently_rescaled_amplitude_is_caught() {
    // The renderer drew at 20 mm/mV while the footer still says 10 — a plausible, wrongly-scaled trace.
    let sig = synth::calibration_pulse(RATE_HZ, 5.0);
    let grid = grid_of(&sig, ClaimedScale { mm_per_mv: 20.0, ..reference() });

    let amp = oracle::measure_amplitude(&grid, reference()).expect("dots present");
    assert_eq!(amp.peak_to_peak_dots, 20, "the picture really is 20 dots tall");
    assert!((amp.peak_to_peak_mv - 2.0).abs() < 1e-12);

    let v = oracle::check(&grid, reference(), 1.0, 1.0, 0.01).expect("both axes measurable");
    assert!(!v.agrees, "a doubled trace must not pass: {v:?}");
    assert!((v.amplitude_error - 1.0).abs() < 1e-12, "reported 2 mV where 1 was fed");
    assert!(v.time_error < 0.01, "only the amplitude axis moved: {v:?}");
}

#[test]
fn a_silently_rescaled_sweep_speed_is_caught() {
    // The terminal was too narrow, so the renderer quietly drew at 50 mm/s under a 25 mm/s label.
    let sig = synth::calibration_pulse(RATE_HZ, 5.0);
    let grid = grid_of(&sig, ClaimedScale { mm_per_s: 50.0, ..reference() });

    let time = oracle::measure_time(&grid, 1.0, reference()).expect("edges present");
    assert!((time.dot_columns_per_cycle - 50.0).abs() < 1e-12, "{} dots/cycle", time.dot_columns_per_cycle);
    assert!((time.mm_per_s_depicted - 50.0).abs() < 1e-9, "depicts {} mm/s", time.mm_per_s_depicted);

    let v = oracle::check(&grid, reference(), 1.0, 1.0, 0.01).expect("both axes measurable");
    assert!(!v.agrees, "a doubled sweep must not pass under a 25 mm/s label: {v:?}");
    assert!((v.time_error - 1.0).abs() < 1e-9);
    assert!(v.amplitude_error < 1e-9, "only the time axis moved: {v:?}");
}

#[test]
fn an_unmeasurable_picture_is_not_a_passing_one() {
    let blank = DotGrid::new(20, 8);
    assert!(oracle::measure_amplitude(&blank, reference()).is_none());
    assert!(oracle::check(&blank, reference(), 1.0, 1.0, 0.01).is_none());
    // A flat line has an amplitude but no edges, so time cannot be measured and there is no verdict.
    let flat = grid_of(&synth::flat(RATE_HZ, 5.0, 0.0), reference());
    assert!(oracle::measure_amplitude(&flat, reference()).is_some());
    assert!(oracle::measure_time(&flat, 1.0, reference()).is_none());
    assert!(oracle::check(&flat, reference(), 1.0, 1.0, 0.01).is_none());
    let pulse = grid_of(&synth::calibration_pulse(RATE_HZ, 5.0), reference());
    assert!(oracle::measure_time(&pulse, 0.0, reference()).is_none(), "a period of zero is no period");
}

#[test]
fn relative_error_never_reads_zero_as_agreement() {
    assert_eq!(oracle::relative_error(0.0, 0.0), 0.0);
    assert!(oracle::relative_error(0.3, 0.0).is_infinite(), "something measured against nothing disagrees");
}

#[test]
fn a_minimum_run_rejects_a_single_column_glitch() {
    // A clean two-cycle square in dot space, with a two-column spike lifted out of the first low run.
    let mut grid = DotGrid::new(40, 14);
    for x in 0..40 {
        let high = (x / 10) % 2 == 1 || (4..6).contains(&x);
        grid.plot(x, if high { 2 } else { 12 });
    }
    assert_eq!(oracle::find_edges(&grid).len(), 5, "the glitch reads as two extra edges");
    assert_eq!(oracle::find_edges_min_run(&grid, 3).len(), 3, "a three-column run absorbs it");
}

// ---- 6. the anti-circularity guard ------------------------------------------------------------

#[test]
fn the_oracle_imports_std_and_the_dot_decoder_only() {
    // An assertion that re-derives its expected value with the code under test proves nothing. This is
    // the mechanical half of keeping the oracle out of the renderer's reach. It covers the two
    // measuring files; this test module is deliberately exempt, because driving the real canvas is
    // what makes the round trips worth anything.
    //
    // What it does NOT cover: a copy-paste of the renderer's arithmetic into these files, since there
    // would be no import to catch. `braille_bits_are_the_unicode_layout` and
    // `the_oracle_decoder_agrees_with_the_shipping_encoder_on_every_dot` are what cover the bit table.
    const ALLOWED: &str = "super::decode::{DotGrid, DOTS_W}";
    for (name, src) in [("oracle.rs", include_str!("oracle.rs")), ("decode.rs", include_str!("decode.rs"))] {
        let code: String = src.lines().map(|l| l.split("//").next().unwrap_or("")).collect::<Vec<_>>().join("\n");
        let uses: Vec<&str> = code.lines().map(str::trim).filter(|l| l.starts_with("use ")).collect();
        for line in &uses {
            let path = line.trim_start_matches("use ").trim_end_matches(';').trim();
            assert!(
                path.starts_with("std::") || path == ALLOWED,
                "{name} imports `{path}`; the oracle may reach std and the dot decoder, nothing else"
            );
        }
        assert!(!code.contains("crate::"), "{name} reaches into the crate root");
        for banned in ["render", "Render", "canvas", "Canvas", "Layer", "mask_char", "dots_set"] {
            assert!(!code.contains(banned), "{name} names `{banned}` in code, so it can reach the renderer");
        }
    }
    // The anchor: the allowlist has to be able to fail. If oracle.rs stopped importing the decoder the
    // constant would be stale and silently permissive, so assert the one import it does have.
    assert!(include_str!("oracle.rs").contains(&format!("use {ALLOWED};")), "the allowlist still matches");
}
