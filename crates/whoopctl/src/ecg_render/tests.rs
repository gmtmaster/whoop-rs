//! The gates for the strip renderer. The scale ones all work the same way: render, then hand the
//! CHARACTERS to `ecg_oracle`, which measures them back and knows nothing about the arithmetic that
//! placed them. Expected values are hand-derived from the target scale, never recomputed from the
//! code under test.
//!
//! `renders_one_millivolt_as_ten_dots_and_one_second_as_twenty_five_columns` is the anchor: a 1 mV /
//! 1 Hz square at 25 mm/s and 10 mm/mV must measure exactly 1.000 mV and 25.0 dot columns per cycle
//! off the picture. The negatives beside it assert specific WRONG numbers, so an expectation quietly
//! re-derived from the renderer would make them fail, not pass.

use crate::ecg_oracle::{decode, oracle, synth};
use crate::ecg_oracle::decode::Extract;
use crate::ecg_oracle::oracle::ClaimedScale;

use super::demo::{self, DemoSignal, FIXTURE_COUNTS_PER_MV};
use super::driver::{self, OutputMode};
use super::frame;
use super::plan::{
    fit, FitError, FitNote, Geometry, Plan, Provenance, Request, Terminal,
    ASSUMED_SAMPLE_RATE_HZ as RATE_HZ, GUTTER_COLS,
};
use super::renderer::{EcgRenderer, Sample, STYLE_LEADOFF, STYLE_MAJOR, STYLE_TRACE};
use super::vertical::VerticalBasis;

/// The reference request: one 5 s strip, 25 mm/s, calibrated at the fixture's own counts-per-mV.
fn calibrated_request() -> Request {
    Request {
        duration_s: 5.0,
        strip_s: 5.0,
        counts_per_mv: Some(FIXTURE_COUNTS_PER_MV),
        sample_rate_hz: RATE_HZ,
        grid: false,
        ..Request::default()
    }
}

fn samples_mv(sig: &synth::Signal, contact: bool) -> Vec<Sample> {
    sig.samples_mv.iter().map(|mv| Sample::new(mv * FIXTURE_COUNTS_PER_MV, contact)).collect()
}

fn render(plan: Plan, data: &[Sample]) -> EcgRenderer {
    let mut r = EcgRenderer::new(plan);
    r.push(data);
    r.finish();
    r
}

/// The strip's plot region as the reader sees it: the frame, windowed by the gutter, found by the
/// oracle's own structural block finder rather than by anything the renderer told it.
fn strip_from_frame(r: &EcgRenderer, strip: usize, colour: bool) -> Vec<String> {
    let lines = frame::frame(r, colour, None, r.plan().strips);
    let blocks = decode::braille_row_blocks(&lines);
    lines[blocks[strip].clone()]
        .iter()
        .map(|l| l.chars().skip(GUTTER_COLS).collect::<String>())
        .collect()
}

// ---- the aspect-ratio reasoning, checked rather than taken ------------------------------------

/// A braille cell is 2 dots wide and 4 tall inside a terminal cell about twice as tall as it is wide,
/// so a dot's own aspect is HALF the terminal cell's. The single-dots-per-mm claim therefore holds at
/// a cell aspect of exactly 2 and nowhere else, which is why the aspect is a parameter with an
/// ASSUMED tag rather than a constant.
#[test]
fn one_dots_per_mm_serves_both_axes_only_when_the_cell_is_twice_as_tall_as_wide() {
    let square = Geometry { dots_per_mm_x: 1.0, cell_aspect: 2.0 };
    assert!((square.dot_aspect() - 1.0).abs() < 1e-12, "a 2:1 cell gives a square dot");
    assert!((square.dots_per_mm_y() - 1.0).abs() < 1e-12, "so one figure serves both axes");
    assert!(square.is_square_dot());

    // The realistic span of monospace fonts, and what assuming 2.0 would cost on each.
    for (cell_aspect, want_dot_aspect) in [(1.8, 0.9), (2.0, 1.0), (2.4, 1.2)] {
        let g = Geometry { dots_per_mm_x: 1.0, cell_aspect };
        assert!((g.dot_aspect() - want_dot_aspect).abs() < 1e-12, "cell {cell_aspect} → dot {want_dot_aspect}");
        assert!((g.dots_per_mm_y() - 1.0 / want_dot_aspect).abs() < 1e-12);
        assert_eq!(g.is_square_dot(), (want_dot_aspect - 1.0f64).abs() < 1e-9);
    }

    // A non-square dot must move the vertical axis, or a 20% font difference would be a 20% amplitude
    // error nothing reported.
    let tall = Geometry { dots_per_mm_x: 1.0, cell_aspect: 2.4 };
    assert!(tall.dots_per_mm_y() < 0.85, "a taller cell needs fewer dots per mm down: {}", tall.dots_per_mm_y());
    assert!(!tall.is_square_dot());
}

#[test]
fn a_non_square_dot_is_declared_in_the_banner() {
    let plan = fit(&Request { cell_aspect: 2.4, ..calibrated_request() }, Terminal::new(80, 60)).unwrap();
    let banner = frame::banner(&plan).join("\n");
    assert!(banner.contains("cell aspect 2.40 (ASSUMED)"), "{banner}");
    assert!(banner.contains("NOT square"), "{banner}");

    let square = fit(&calibrated_request(), Terminal::new(80, 60)).unwrap();
    assert!(frame::banner(&square).join("\n").contains("(square)"));
}

/// At a non-square dot the two axes must carry DIFFERENT dots-per-mm, measured off the picture.
#[test]
fn a_non_square_dot_keeps_both_axes_true() {
    let cell_aspect = 2.4; // dot aspect 1.2, so 1/1.2 dots per mm down
    let plan = fit(
        &Request { cell_aspect, strip_mm: 40.0, ..calibrated_request() },
        Terminal::new(80, 60),
    )
    .unwrap();
    let dots_y = plan.geom.dots_per_mm_y();
    let r = render(plan, &samples_mv(&synth::calibration_pulse(RATE_HZ, 5.0), true));
    let grid = decode::from_lines(&strip_from_frame(&r, 0, false), Extract::All);

    // Time is measured under the x claim, amplitude under the y claim: one number for both would be
    // wrong here by exactly the dot aspect.
    let x_claim = ClaimedScale { dots_per_mm: plan.geom.dots_per_mm_x, mm_per_mv: 10.0, mm_per_s: 25.0 };
    let y_claim = ClaimedScale { dots_per_mm: dots_y, ..x_claim };
    let time = oracle::measure_time(&grid, 1.0, x_claim).expect("edges");
    let amp = oracle::measure_amplitude(&grid, y_claim).expect("dots");
    assert!((time.mm_per_s_depicted - 25.0).abs() < 0.01, "{} mm/s", time.mm_per_s_depicted);
    // A dot is 1.2 mm tall here, so 0.12 mV; rounding at both ends can cost one and a half of them.
    assert!((amp.peak_to_peak_mv - 1.0).abs() <= 0.18, "{} mV", amp.peak_to_peak_mv);

    // The same picture read under the WRONG (x) claim on the y axis is out by exactly the dot aspect.
    let wrong = oracle::measure_amplitude(&grid, x_claim).expect("dots");
    assert!(
        (wrong.peak_to_peak_mv / amp.peak_to_peak_mv - dots_y).abs() < 1e-12,
        "one dots-per-mm for both axes is wrong by the dot aspect: {} vs {}",
        wrong.peak_to_peak_mv,
        amp.peak_to_peak_mv,
    );
    assert!(wrong.peak_to_peak_mv < 0.95, "and it under-reports: {} mV", wrong.peak_to_peak_mv);
}

// ---- the anchor: a known square wave measures its own scale back ------------------------------

#[test]
fn renders_one_millivolt_as_ten_dots_and_one_second_as_twenty_five_columns() {
    let plan = fit(&calibrated_request(), Terminal::new(80, 60)).unwrap();
    // Hand-derived from 25 mm/s at 1 dot/mm: 125 dots plus the endpoint = 126 dots = 63 columns.
    assert!((plan.geom.dots_per_mm_x - 1.0).abs() < 1e-12, "80 columns leaves room for exactly 1 dot/mm");
    assert_eq!(plan.strip_dot_w, 126);
    assert_eq!(plan.plot_cols(), 63, "a 5 s strip is 63 terminal columns");
    assert_eq!(plan.strips, 1);

    let r = render(plan, &samples_mv(&synth::calibration_pulse(RATE_HZ, 5.0), true));
    let lines = strip_from_frame(&r, 0, false);
    assert_eq!(lines[0].chars().count(), 63);

    let claim = ClaimedScale::reference();
    let grid = decode::from_lines(&lines, Extract::All);
    let amp = oracle::measure_amplitude(&grid, claim).expect("dots present");
    assert_eq!(amp.peak_to_peak_dots, 10, "1 mV at 10 mm/mV and 1 dot/mm is 10 dots");
    assert!((amp.peak_to_peak_mv - 1.0).abs() < 1e-12, "measured {} mV", amp.peak_to_peak_mv);

    let time = oracle::measure_time(&grid, 1.0, claim).expect("edges present");
    assert_eq!(time.cycles, 4, "five rising edges over five seconds");
    assert!((time.dot_columns_per_cycle - 25.0).abs() < 1e-12, "{} dots/cycle", time.dot_columns_per_cycle);
    assert!((time.cell_columns_per_cycle - 12.5).abs() < 1e-12, "12.5 terminal columns a second");
    assert!((time.s_per_cell_column - 0.08).abs() < 1e-9);
    assert!((time.mm_per_s_depicted - 25.0).abs() < 1e-9, "depicts {} mm/s", time.mm_per_s_depicted);

    let v = oracle::check(&grid, claim, 1.0, 1.0, 1e-6).expect("both axes measurable");
    assert!(v.agrees, "the picture agrees with its claim: {v:?}");
}

/// The same signal at four terminal widths. The mm/s never moves; only the dot resolution does, and
/// the fit says so.
#[test]
fn the_scale_holds_at_every_terminal_width() {
    let claim = ClaimedScale::reference();
    for (cols, want_k) in [(80usize, 1.0f64), (100, 1.0), (140, 2.0), (200, 3.0)] {
        let plan = fit(&calibrated_request(), Terminal::new(cols, 80)).unwrap();
        assert!((plan.geom.dots_per_mm_x - want_k).abs() < 1e-12, "{cols} columns → {} dots/mm", plan.geom.dots_per_mm_x);
        assert!(plan.plot_cols() + GUTTER_COLS <= cols, "the frame fits in {cols}");

        let r = render(plan, &samples_mv(&synth::calibration_pulse(RATE_HZ, 5.0), true));
        let grid = decode::from_lines(&strip_from_frame(&r, 0, false), Extract::All);
        let at_k = ClaimedScale { dots_per_mm: want_k, ..claim };

        let amp = oracle::measure_amplitude(&grid, at_k).expect("dots");
        assert_eq!(amp.peak_to_peak_dots, (10.0 * want_k) as usize, "{cols} columns");
        assert!((amp.peak_to_peak_mv - 1.0).abs() < 1e-9, "{cols} columns measured {} mV", amp.peak_to_peak_mv);

        let time = oracle::measure_time(&grid, 1.0, at_k).expect("edges");
        assert!((time.dot_columns_per_cycle - 25.0 * want_k).abs() < 1e-9, "{cols} columns");
        assert!((time.mm_per_s_depicted - 25.0).abs() < 1e-9, "{cols} columns depicts {} mm/s — the sweep speed must not move", time.mm_per_s_depicted);
    }
}

/// A picture drawn at 20 mm/mV under a 10 mm/mV claim reports 2 mV, and the time axis stays clean.
/// The oracle's job, restated against the shipping renderer.
#[test]
fn a_doubled_amplitude_scale_is_caught_by_the_oracle() {
    let plan = fit(
        &Request { mm_per_mv: 20.0, strip_mm: 40.0, ..calibrated_request() },
        Terminal::new(80, 80),
    )
    .unwrap();
    let r = render(plan, &samples_mv(&synth::calibration_pulse(RATE_HZ, 5.0), true));
    let grid = decode::from_lines(&strip_from_frame(&r, 0, false), Extract::All);

    let amp = oracle::measure_amplitude(&grid, ClaimedScale::reference()).expect("dots");
    assert_eq!(amp.peak_to_peak_dots, 20, "the picture really is 20 dots tall");
    let v = oracle::check(&grid, ClaimedScale::reference(), 1.0, 1.0, 0.01).expect("measurable");
    assert!(!v.agrees, "a doubled trace must not pass a 10 mm/mV claim: {v:?}");
    assert!((v.amplitude_error - 1.0).abs() < 1e-12, "reported 2 mV where 1 was fed");
    assert!(v.time_error < 0.01, "only the amplitude axis moved: {v:?}");
}

/// The same for the sweep: 50 mm/s under a 25 mm/s claim is 50 dot columns a cycle.
#[test]
fn a_doubled_sweep_speed_is_caught_by_the_oracle() {
    let plan = fit(&Request { mm_per_s: 50.0, ..calibrated_request() }, Terminal::new(140, 80)).unwrap();
    assert!((plan.geom.dots_per_mm_x - 1.0).abs() < 1e-12);
    let r = render(plan, &samples_mv(&synth::calibration_pulse(RATE_HZ, 5.0), true));
    let grid = decode::from_lines(&strip_from_frame(&r, 0, false), Extract::All);

    let time = oracle::measure_time(&grid, 1.0, ClaimedScale::reference()).expect("edges");
    assert!((time.dot_columns_per_cycle - 50.0).abs() < 1e-12, "{} dots/cycle", time.dot_columns_per_cycle);
    let v = oracle::check(&grid, ClaimedScale::reference(), 1.0, 1.0, 0.01).expect("measurable");
    assert!(!v.agrees, "a doubled sweep must not pass a 25 mm/s claim: {v:?}");
    assert!((v.time_error - 1.0).abs() < 1e-9);
    assert!(v.amplitude_error < 1e-9, "only the time axis moved: {v:?}");
}

// ---- the grid must not disturb the trace -------------------------------------------------------

#[test]
fn the_grid_moves_no_trace_dot() {
    let data = samples_mv(&synth::pqrst(RATE_HZ, 5.0, 62.0, 1.2), true);
    let off = render(fit(&calibrated_request(), Terminal::new(80, 80)).unwrap(), &data);
    let on = render(fit(&Request { grid: true, ..calibrated_request() }, Terminal::new(80, 80)).unwrap(), &data);
    assert_eq!(on.trace_lines(0), off.trace_lines(0), "the grid layer must not touch the trace layer");
    assert!(on.strip_lines(0) != off.strip_lines(0), "and the grid really is drawn");
}

#[test]
fn the_grid_is_dim_and_the_trace_carries_no_escape() {
    let plan = fit(&Request { grid: true, ..calibrated_request() }, Terminal::new(80, 80)).unwrap();
    let r = render(plan, &samples_mv(&synth::calibration_pulse(RATE_HZ, 5.0), true));
    let styles: Vec<_> = r.strip_spans(0).into_iter().flatten().map(|s| s.style).collect();
    assert!(styles.contains(&STYLE_MAJOR), "the 5 mm grid is drawn");
    assert!(styles.contains(&STYLE_TRACE), "the trace is drawn");
    assert_eq!(frame::escape(STYLE_MAJOR), "\x1b[2m", "the grid is dim");
    assert_eq!(frame::escape(STYLE_TRACE), "", "the trace carries no escape, so a decoder can isolate it");

    // With colour on, the unstyled cells are exactly the trace's.
    let painted = strip_from_frame(&r, 0, true);
    let trace_only = decode::from_lines(&painted, Extract::Unstyled);
    let grid_only = decode::from_lines(&painted, Extract::Styled);
    assert!(trace_only.count() > 0 && grid_only.count() > 0);
}

// ---- the uncalibrated path ---------------------------------------------------------------------

#[test]
fn without_a_counts_per_mv_the_axis_is_counts_and_both_banner_and_footer_say_so() {
    let plan = fit(&Request { counts_per_mv: None, ..calibrated_request() }, Terminal::new(80, 80)).unwrap();
    assert!(!plan.amplitude.is_calibrated());
    assert_eq!(plan.dots_per_mv(), None, "no mm/mV is claimed, so there is no dots-per-mV to give");

    let banner = frame::banner(&plan).join("\n");
    assert!(banner.contains("UNCALIBRATED"), "banner: {banner}");
    assert!(banner.contains("RAW COUNTS"), "banner: {banner}");
    assert!(banner.contains("NO mm/mV is claimed"), "banner: {banner}");
    assert!(!banner.contains("mm/mV ·"), "the banner must not quote an mm/mV: {banner}");

    let r = render(plan, &samples_mv(&synth::calibration_pulse(RATE_HZ, 5.0), true));
    let report = r.report();
    let footer = frame::footer(&plan, &report, None).join("\n");
    assert!(footer.contains(frame::UNCALIBRATED_BANNER), "footer: {footer}");
    assert!(footer.contains("counts/mm"), "footer: {footer}");
    assert!(footer.contains("NOT millivolts"), "footer: {footer}");

    // Autoscaled: the 1 mV range is stretched over 80% of the 24 mm window = 19.2 mm, which lands on
    // 20 dots once each end rounds. Hand-derived, and deliberately NOT 10 — an uncalibrated picture
    // must never accidentally come out at the true scale.
    let grid = decode::from_lines(&strip_from_frame(&r, 0, false), Extract::All);
    let amp = oracle::measure_amplitude(&grid, ClaimedScale::reference()).expect("dots");
    assert_eq!(amp.peak_to_peak_dots, 20, "1000 counts stretched over 0.8 x 24 mm at 1 dot/mm");
    match report.vertical {
        Some(VerticalBasis::Autoscaled { counts_per_mm }) => {
            assert!((counts_per_mm - 1000.0 / 19.2).abs() < 1e-9, "1000 counts over 19.2 mm: {counts_per_mm}")
        }
        other => panic!("expected an autoscaled basis, got {other:?}"),
    }
}

#[test]
fn a_flat_input_claims_no_vertical_scale() {
    let plan = fit(&Request { counts_per_mv: None, ..calibrated_request() }, Terminal::new(80, 80)).unwrap();
    let flat = vec![Sample::new(1234.0, true); (RATE_HZ * 5.0) as usize];
    let r = render(plan, &flat);
    let report = r.report();
    assert_eq!(report.vertical, Some(VerticalBasis::Flat));
    let footer = frame::footer(&plan, &report, None).join("\n");
    assert!(footer.contains("NO vertical scale is claimed"), "{footer}");

    let grid = decode::from_lines(&strip_from_frame(&r, 0, false), Extract::All);
    let amp = oracle::measure_amplitude(&grid, ClaimedScale::reference()).expect("dots");
    assert_eq!(amp.peak_to_peak_dots, 0, "a flat input draws a flat line, at the midline");
}

/// A calibrated frame must not carry the uncalibrated warning, or the warning would mean nothing.
#[test]
fn a_calibrated_frame_carries_no_uncalibrated_warning() {
    let plan = fit(&calibrated_request(), Terminal::new(80, 80)).unwrap();
    let r = render(plan, &samples_mv(&synth::calibration_pulse(RATE_HZ, 5.0), true));
    let whole = frame::frame(&r, false, Some("test"), plan.strips).join("\n");
    assert!(!whole.contains("UNCALIBRATED"), "{whole}");
    assert!(whole.contains("10.0 mm/mV"), "{whole}");
    assert!(whole.contains("1000.000 counts/mV (SUPPLIED)"), "{whole}");
}

// ---- narrow terminals: shorten or refuse, never rescale -----------------------------------------

#[test]
fn a_narrow_terminal_shortens_the_strip_and_never_the_sweep_speed() {
    // 48 columns: 40 for the plot = 80 dots. A 5 s strip needs 126, so the strip is cut.
    let plan = fit(&calibrated_request(), Terminal::new(48, 80)).unwrap();
    match plan.note {
        FitNote::ShorterStrip { asked_s, got_s } => {
            assert!((asked_s - 5.0).abs() < 1e-12);
            assert!((got_s - 3.1).abs() < 1e-12, "79 dots at 25 mm/s is 3.16 s, floored to {got_s}");
        }
        other => panic!("expected a shortened strip, got {other:?}"),
    }
    assert!((plan.mm_per_s - 25.0).abs() < 1e-12, "the sweep speed is untouched");
    assert!((plan.geom.dots_per_mm_x - 1.0).abs() < 1e-12);
    assert_eq!(plan.strips, 2, "5 s of signal at a 3.1 s strip is two strips");
    assert!(plan.plot_cols() + GUTTER_COLS <= 48);

    let banner = frame::banner(&plan).join("\n");
    assert!(banner.contains("SHORTENED to 3.1 s"), "{banner}");
    assert!(banner.contains("The mm/s was NOT changed"), "{banner}");

    // And the picture really still sweeps at 25 mm/s: one cycle is 25 dot columns.
    let r = render(plan, &samples_mv(&synth::calibration_pulse(RATE_HZ, 5.0), true));
    let grid = decode::from_lines(&strip_from_frame(&r, 0, false), Extract::All);
    let time = oracle::measure_time(&grid, 1.0, ClaimedScale::reference()).expect("edges");
    assert!((time.dot_columns_per_cycle - 25.0).abs() < 1e-9, "{} dots/cycle", time.dot_columns_per_cycle);
    assert!((time.mm_per_s_depicted - 25.0).abs() < 1e-9, "a shortened strip is NOT a rescaled one");
}

#[test]
fn a_terminal_too_narrow_for_the_shortest_strip_is_refused() {
    let err = fit(&calibrated_request(), Terminal::new(20, 80)).expect_err("20 columns cannot hold a 1 s strip");
    match err {
        FitError::TooNarrow { have_cols, need_cols, min_strip_s, mm_per_s } => {
            assert_eq!(have_cols, 20);
            // 1 s at 25 mm/s and 1 dot/mm = 26 dots = 13 columns, plus the 8-column gutter.
            assert_eq!(need_cols, GUTTER_COLS + 13);
            assert!((min_strip_s - 1.0).abs() < 1e-12);
            assert!((mm_per_s - 25.0).abs() < 1e-12);
        }
        other => panic!("expected TooNarrow, got {other:?}"),
    }
    let text = err.to_string();
    assert!(text.contains("Refusing rather than drawing at a different mm/s"), "{text}");
    // The boundary: one column more and it fits, at the SAME mm/s.
    let ok = fit(&calibrated_request(), Terminal::new(GUTTER_COLS + 13, 80)).expect("the boundary fits");
    assert!((ok.mm_per_s - 25.0).abs() < 1e-12);
    assert!(matches!(ok.note, FitNote::ShorterStrip { got_s, .. } if (got_s - 1.0).abs() < 1e-12));
}

#[test]
fn a_bad_request_is_refused_rather_than_repaired() {
    for bad in [
        Request { mm_per_s: 0.0, ..calibrated_request() },
        Request { strip_s: -1.0, ..calibrated_request() },
        Request { duration_s: 0.0, ..calibrated_request() },
        Request { counts_per_mv: Some(0.0), ..calibrated_request() },
        Request { cell_aspect: 0.0, ..calibrated_request() },
        Request { strip_mm: 0.0, ..calibrated_request() },
        Request { sample_rate_hz: 0.0, ..calibrated_request() },
    ] {
        assert!(matches!(fit(&bad, Terminal::new(120, 60)), Err(FitError::BadRequest(_))), "{bad:?}");
    }
}

// ---- lead-off -----------------------------------------------------------------------------------

#[test]
fn lead_off_spans_are_hatched_marked_and_never_interpolated() {
    let plan = fit(&Request { grid: false, ..calibrated_request() }, Terminal::new(80, 80)).unwrap();
    let sig = synth::calibration_pulse(RATE_HZ, 5.0);
    let mut data = samples_mv(&sig, true);
    // Two seconds of the middle lost, sample-exact: 2.0 s to 3.0 s.
    let (lo, hi) = ((2.0 * RATE_HZ) as usize, (3.0 * RATE_HZ) as usize);
    for s in data.iter_mut().take(hi).skip(lo) {
        s.contact = false;
        s.counts = 9_000_000.0; // garbage, as a floating lead reads
    }
    let r = render(plan, &data);

    let report = r.report();
    assert_eq!(report.lead_off.len(), 1);
    assert!((report.lead_off[0].start_s - 2.0).abs() < 1e-9);
    assert!((report.lead_off[0].end_s - 3.0).abs() < 1e-9);
    assert!((report.lead_off_s - 1.0).abs() < 1e-9);
    assert_eq!(report.clipped, 0, "lead-off samples are not drawn, so they cannot clip");

    // The columns 2.0-3.0 s covers: 25 dots a second at 1 dot/mm.
    let (first, last) = (plan.dot_column(2.0), plan.dot_column(3.0));
    assert_eq!((first, last), (50, 75));
    let trace = decode::from_lines(&r.trace_lines(0), Extract::All);
    let hatch = decode::from_lines(&r.lead_off_lines(0), Extract::All);
    for x in first..=last {
        assert!(trace.column_rows(x).is_empty(), "column {x} must carry no trace");
        assert!(!hatch.column_rows(x).is_empty(), "column {x} must carry the lead-off hatch");
    }
    // Outside the span the trace is intact and the hatch is absent.
    for x in [first - 2, last + 2] {
        assert!(!trace.column_rows(x).is_empty(), "column {x} still carries the trace");
        assert!(hatch.column_rows(x).is_empty(), "column {x} carries no hatch");
    }

    // The hatch spans the whole strip height, so it can never read as a flat signal.
    let rows: Vec<usize> = (first..=last).flat_map(|x| hatch.column_rows(x)).collect();
    assert!(rows.iter().copied().min().unwrap() < 2);
    assert!(rows.iter().copied().max().unwrap() > plan.strip_dot_h - 3);

    let styles: Vec<_> = r.strip_spans(0).into_iter().flatten().map(|s| s.style).collect();
    assert!(styles.contains(&STYLE_LEADOFF), "the hatch gets its own style");
    assert_eq!(frame::escape(STYLE_LEADOFF), "\x1b[31m");
    assert_ne!(frame::escape(STYLE_LEADOFF), frame::escape(STYLE_TRACE));

    let footer = frame::footer(&plan, &report, None).join("\n");
    assert!(footer.contains("lead-off: 1 span, 1.00 s total"), "{footer}");
    assert!(footer.contains("never interpolated"), "{footer}");
    assert!(footer.contains("2.00-3.00 s"), "{footer}");
}

/// Lead-off samples must not set the vertical scale, or one floating lead would flatten the trace.
#[test]
fn lead_off_samples_do_not_prime_the_vertical_scale() {
    let plan = fit(&Request { counts_per_mv: None, ..calibrated_request() }, Terminal::new(80, 80)).unwrap();
    let sig = synth::calibration_pulse(RATE_HZ, 5.0);
    let mut data = samples_mv(&sig, true);
    for s in data.iter_mut().take(200) {
        s.contact = false;
        s.counts = 5_000_000.0;
    }
    let r = render(plan, &data);
    match r.report().vertical {
        // The 5,000,000-count garbage would have given ~312,500 counts/mm had it been used.
        Some(VerticalBasis::Autoscaled { counts_per_mm }) => {
            assert!(counts_per_mm < 100.0, "primed on the real trace, not the garbage: {counts_per_mm}")
        }
        other => panic!("expected an autoscaled basis, got {other:?}"),
    }
    assert!(r.report().primed_on_contact);
}

#[test]
fn priming_entirely_on_lead_off_is_declared() {
    let plan = fit(&Request { counts_per_mv: None, ..calibrated_request() }, Terminal::new(80, 80)).unwrap();
    let data = vec![Sample::new(10.0, false); 2500];
    let r = render(plan, &data);
    let report = r.report();
    assert!(!report.primed_on_contact);
    let footer = frame::footer(&plan, &report, None).join("\n");
    assert!(footer.contains("primed on LEAD-OFF samples"), "{footer}");
}

// ---- clipping is reported, not accommodated ------------------------------------------------------

#[test]
fn samples_outside_the_window_clip_and_are_counted_rather_than_rescaled() {
    let plan = fit(&calibrated_request(), Terminal::new(80, 80)).unwrap();
    let mut data = samples_mv(&synth::calibration_pulse(RATE_HZ, 5.0), true);
    // A late 10 mV excursion, far outside the 20 mm (2 mV) window.
    for s in data.iter_mut().skip(2000) {
        s.counts = 10.0 * FIXTURE_COUNTS_PER_MV;
    }
    let r = render(plan, &data);
    let report = r.report();
    assert_eq!(report.clipped, 500, "every late sample fell outside the window");
    assert!((report.counts_per_dot.unwrap() - 100.0).abs() < 1e-9, "the scale did not move to fit them");

    let grid = decode::from_lines(&strip_from_frame(&r, 0, false), Extract::All);
    let amp = oracle::measure_amplitude(&grid, ClaimedScale::reference()).expect("dots");
    assert!(amp.peak_to_peak_dots <= plan.strip_dot_h, "clipped to the window, not squeezed into it");
    let footer = frame::footer(&plan, &report, None).join("\n");
    assert!(footer.contains("clipped: 500 samples"), "{footer}");
    assert!(footer.contains("the scale was NOT changed"), "{footer}");
}

// ---- progressive drawing --------------------------------------------------------------------------

/// Pushing in chunks must give the same picture as pushing once, or a redraw would depend on the
/// arrival pattern. Also the reason a redraw is cheap: nothing is reallocated per push.
#[test]
fn chunked_pushes_draw_the_same_picture_as_one_push() {
    let plan = fit(&Request { duration_s: 30.0, ..calibrated_request() }, Terminal::new(80, 200)).unwrap();
    assert_eq!(plan.strips, 6, "30 s at a 5 s strip is six strips, derived");
    let data = samples_mv(&synth::pqrst(RATE_HZ, 30.0, 62.0, 1.2), true);

    let one = render(plan, &data);
    let mut many = EcgRenderer::new(plan);
    for chunk in data.chunks(37) {
        many.push(chunk);
    }
    many.finish();

    for strip in 0..plan.strips {
        assert_eq!(many.strip_lines(strip), one.strip_lines(strip), "strip {strip}");
    }
    assert_eq!(many.report(), one.report());
}

#[test]
fn strips_complete_in_order_as_the_stream_advances() {
    let plan = fit(&Request { duration_s: 30.0, ..calibrated_request() }, Terminal::new(80, 200)).unwrap();
    let data = samples_mv(&synth::calibration_pulse(RATE_HZ, 30.0), true);
    let mut r = EcgRenderer::new(plan);
    assert_eq!(r.strips_complete(), 0);
    let mut seen = Vec::new();
    for chunk in data.chunks(250) {
        r.push(chunk);
        seen.push(r.strips_complete());
    }
    r.finish();
    assert_eq!(r.strips_complete(), 6);
    assert!(seen.windows(2).all(|w| w[1] >= w[0]), "a completed strip never un-completes: {seen:?}");
    assert!(seen.contains(&1) && seen.contains(&3), "strips land one at a time: {seen:?}");
}

/// Nothing is drawn until the vertical mapping is frozen — the scale can never move under dots that
/// are already on screen.
#[test]
fn nothing_is_drawn_before_the_scale_is_frozen() {
    let plan = fit(&calibrated_request(), Terminal::new(80, 80)).unwrap();
    assert_eq!(plan.prime_samples, 500, "one second at 500 Hz");
    let data = samples_mv(&synth::calibration_pulse(RATE_HZ, 5.0), true);
    let mut r = EcgRenderer::new(plan);
    r.push(&data[..499]);
    assert_eq!(r.report().vertical, None, "still priming");
    assert_eq!(r.report().drawn, 0, "and nothing drawn");
    assert_eq!(decode::from_lines(&r.trace_lines(0), Extract::All).count(), 0);
    assert!(frame::footer(&plan, &r.report(), None).join("\n").contains("still priming"));

    r.push(&data[499..600]);
    assert!(r.report().vertical.is_some(), "the window closed and the scale froze");
    assert_eq!(r.report().drawn, 600, "the buffered samples were then drawn at that scale");
    assert!(decode::from_lines(&r.trace_lines(0), Extract::All).count() > 0);
}

/// A stream that ends before the priming window closes still renders, from what arrived.
#[test]
fn a_short_stream_still_renders() {
    let plan = fit(&calibrated_request(), Terminal::new(80, 80)).unwrap();
    let data = samples_mv(&synth::calibration_pulse(RATE_HZ, 0.6), true);
    let mut r = EcgRenderer::new(plan);
    r.push(&data);
    assert_eq!(r.report().drawn, 0, "300 samples is short of the 500-sample window");
    r.finish();
    assert_eq!(r.report().drawn, 300);
    assert!(decode::from_lines(&r.trace_lines(0), Extract::All).count() > 0);
}

// ---- output mode ---------------------------------------------------------------------------------

#[test]
fn the_output_mode_is_chosen_and_stated() {
    let plan = fit(&Request { duration_s: 30.0, ..calibrated_request() }, Terminal::new(80, 200)).unwrap();
    let rows = frame::frame_rows(&plan);

    let (mode, why) = driver::choose_mode(Terminal::new(80, 200), rows, true);
    assert_eq!(mode, OutputMode::InPlace);
    assert!(mode.describe(&why).contains("in-place progressive redraw"), "{why}");

    let (mode, why) = driver::choose_mode(Terminal::new(80, 200), rows, false);
    assert_eq!(mode, OutputMode::Sequential, "a pipe is not a terminal");
    assert!(mode.describe(&why).contains("stdout is not a terminal"), "{why}");

    let (mode, why) = driver::choose_mode(Terminal::new(80, 12), rows, true);
    assert_eq!(mode, OutputMode::Sequential, "a frame taller than the terminal cannot redraw in place");
    assert!(why.contains("the frame is"), "{why}");
}

#[test]
fn a_frame_never_exceeds_its_row_bound() {
    for (duration, strip_s, counts) in
        [(30.0, 5.0, None), (30.0, 5.0, Some(FIXTURE_COUNTS_PER_MV)), (10.0, 2.5, None), (5.0, 5.0, None)]
    {
        let req = Request { duration_s: duration, strip_s, counts_per_mv: counts, ..calibrated_request() };
        let plan = fit(&req, Terminal::new(80, 200)).unwrap();
        let mut data = samples_mv(&synth::calibration_pulse(RATE_HZ, duration), true);
        for s in data.iter_mut().take(100) {
            s.contact = false;
        }
        let r = render(plan, &data);
        let rows = frame::frame(&r, false, Some("a mode note"), plan.strips).len();
        assert!(rows <= frame::frame_rows(&plan), "{rows} rows exceeds the bound {}", frame::frame_rows(&plan));
    }
}

/// The sequential path emits the banner, every strip and the footer, and nothing is lost when no
/// strip ever completes mid-stream.
#[test]
fn the_sequential_path_emits_every_strip_then_the_footer() {
    let plan = fit(&Request { duration_s: 10.0, ..calibrated_request() }, Terminal::new(80, 200)).unwrap();
    let data = samples_mv(&synth::calibration_pulse(RATE_HZ, 10.0), true);
    let mut r = EcgRenderer::new(plan);
    let mut painter = driver::Painter::new(OutputMode::Sequential, "test".into(), false);
    let mut out: Vec<u8> = Vec::new();
    for chunk in data.chunks(500) {
        r.push(chunk);
        painter.paint(&mut out, &r).unwrap();
    }
    r.finish();
    painter.finish(&mut out, &r).unwrap();

    let text = String::from_utf8(out).unwrap();
    assert!(!text.contains('\x1b'), "the sequential path emits no escapes at all");
    assert_eq!(text.matches("achieved x:").count(), 1, "one footer");
    assert_eq!(text.matches("ECG — 10.0 s").count(), 1, "one banner");
    let blocks = decode::braille_row_blocks(&text.lines().map(str::to_string).collect::<Vec<_>>());
    assert_eq!(blocks.len(), 2, "both strips were emitted");
}

#[test]
fn the_in_place_path_rewinds_by_exactly_what_it_wrote() {
    let plan = fit(&calibrated_request(), Terminal::new(80, 80)).unwrap();
    let data = samples_mv(&synth::calibration_pulse(RATE_HZ, 5.0), true);
    let mut r = EcgRenderer::new(plan);
    let mut painter = driver::Painter::new(OutputMode::InPlace, "test".into(), false);
    let mut out: Vec<u8> = Vec::new();
    for chunk in data.chunks(1250) {
        r.push(chunk);
        painter.paint(&mut out, &r).unwrap();
    }
    let text = String::from_utf8(out).unwrap();
    let rows = text.matches("\x1b[K").count();
    let rewound: usize = text
        .split("\x1b[")
        .filter_map(|s| s.strip_suffix("A").or_else(|| s.split('A').next().filter(|p| s.starts_with(&format!("{p}A")))))
        .filter_map(|n| n.parse::<usize>().ok())
        .sum();
    let painted_per_frame = frame::frame(&r, false, Some("x"), plan.strips).len();
    assert_eq!(rows % painted_per_frame, 0, "every redraw wrote a whole frame");
    assert_eq!(rewound, rows - painted_per_frame, "each redraw rewound exactly the last frame");
    assert!(!text.contains("\x1b[2J"), "no clear-screen — that is what flickers");
}

// ---- the terminal size, and its provenance --------------------------------------------------------

#[test]
fn the_terminal_size_carries_where_it_came_from() {
    let supplied = driver::terminal(Some(123), Some(45));
    assert_eq!((supplied.cols, supplied.rows), (123, 45));
    assert_eq!(supplied.cols_from, Provenance::Supplied);
    assert_eq!(supplied.rows_from, Provenance::Supplied);

    // No flag and (in a test harness) no COLUMNS/LINES: the fallback must be labelled, never silent.
    let fallback = driver::terminal(None, None);
    for from in [fallback.cols_from, fallback.rows_from] {
        assert!(matches!(from, Provenance::Environment | Provenance::Assumed), "{from:?}");
    }
    assert_eq!(Provenance::Assumed.tag(), "ASSUMED");
}

// ---- the demo -------------------------------------------------------------------------------------

#[test]
fn the_demo_streams_a_signal_of_known_dimensions() {
    let data = demo::samples(DemoSignal::Pulse, RATE_HZ, 5.0, false);
    assert_eq!(data.len(), 2500);
    assert!(data.iter().all(|s| s.contact));
    let (lo, hi) = data.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(l, h), s| (l.min(s.counts), h.max(s.counts)));
    assert!((hi - lo - FIXTURE_COUNTS_PER_MV).abs() < 1e-9, "1 mV at the fixture's own counts-per-mV");

    let with_lead_off = demo::samples(DemoSignal::Pulse, RATE_HZ, 5.0, true);
    assert!(with_lead_off.iter().any(|s| !s.contact), "lead-off is actually injected");
    assert!(with_lead_off.iter().any(|s| s.contact));
    assert_eq!(DemoSignal::parse("ecg"), Some(DemoSignal::Ecg));
    assert_eq!(DemoSignal::parse("nonsense"), None);
}

/// End to end through the demo runner, into a pipe: it must produce a frame and refuse nothing.
#[test]
fn the_demo_runs_end_to_end_into_a_pipe() {
    let req = Request {
        duration_s: 10.0,
        counts_per_mv: Some(FIXTURE_COUNTS_PER_MV),
        sample_rate_hz: RATE_HZ,
        ..Request::default()
    };
    let opts = demo::DemoOptions { signal: DemoSignal::Ecg, lead_off: true, speed: 0.0, colour: false };
    let mut out: Vec<u8> = Vec::new();
    let plan = demo::run(&req, Terminal::new(100, 200), &opts, &mut out).expect("a 100-column terminal fits");
    assert_eq!(plan.strips, 2);
    let text = String::from_utf8(out).unwrap();
    assert!(text.contains("FIXTURE, not a device constant"), "{text}");
    assert!(text.contains("lead-off:"), "{text}");
    assert!(text.contains("achieved x: 25.000 mm/s"), "{text}");
}
