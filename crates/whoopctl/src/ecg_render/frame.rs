//! Banner, footer and frame assembly. The text here is the contract with the reader: every number it
//! quotes is the number the picture was drawn at, and every number nobody has read off the device
//! carries its provenance tag beside it.
//!
//! While no counts-per-mV is supplied the amplitude axis is raw ADC counts, and that is stated in the
//! banner AND the footer on every render — never in a verbose mode, never only in a capture file.

use crate::braille::{Span, StyleId, BASE_STYLE};

use super::grid;
use super::plan::{AmplitudePlan, FitNote, Plan, GUTTER_COLS};
use super::renderer::{EcgRenderer, Report, STYLE_LEADOFF, STYLE_MAJOR, STYLE_MINOR};
use super::vertical::VerticalBasis;

/// The one-line warning, repeated top and bottom while the axis is uncalibrated.
pub const UNCALIBRATED_BANNER: &str =
    "*** UNCALIBRATED — the amplitude axis is RAW ADC COUNTS, not millivolts. No counts-per-mV is known. ***";

/// The escape a style is painted with. The trace carries none, so a decoder can isolate it.
pub fn escape(style: StyleId) -> &'static str {
    match style {
        STYLE_MINOR | STYLE_MAJOR => "\x1b[2m",
        STYLE_LEADOFF => "\x1b[31m",
        _ => "",
    }
}

/// Paint one row of spans. Unstyled runs are emitted bare, so nothing but the grid and the lead-off
/// hatch is ever wrapped.
pub fn paint(row: &[Span], colour: bool) -> String {
    let mut out = String::new();
    for span in row {
        let esc = if colour { escape(span.style) } else { "" };
        if esc.is_empty() || span.style == BASE_STYLE {
            out.push_str(&span.text);
        } else {
            out.push_str(esc);
            out.push_str(&span.text);
            out.push_str("\x1b[0m");
        }
    }
    out
}

/// The banner: what the picture claims, before any of it is drawn.
pub fn banner(plan: &Plan) -> Vec<String> {
    let g = plan.geom;
    let mut lines = vec![format!(
        "ECG — {:.1} s as {} strip{} of {:.1} s{}",
        plan.duration_s,
        plan.strips,
        if plan.strips == 1 { "" } else { "s" },
        plan.strip_s,
        match plan.amplitude {
            AmplitudePlan::Uncalibrated => "   *** UNCALIBRATED — y axis is RAW COUNTS ***",
            AmplitudePlan::Calibrated { .. } => "",
        },
    )];
    lines.push(format!(
        "  x: {:.1} mm/s · {:.2} dot/mm · {:.4} s per dot column · strip = {} dots = {} columns",
        plan.mm_per_s,
        g.dots_per_mm_x,
        plan.s_per_dot_column(),
        plan.strip_dot_w,
        plan.plot_cols(),
    ));
    lines.push(match plan.amplitude {
        AmplitudePlan::Calibrated { counts_per_mv, mm_per_mv } => format!(
            "  y: {:.1} mm/mV · {:.2} dot/mm · {:.2} dots/mV · {:.3} counts/mV ({}) · window {:.1} mm = {:.2} mV",
            mm_per_mv,
            g.dots_per_mm_y(),
            plan.dots_per_mv().unwrap_or_default(),
            counts_per_mv,
            super::plan::Provenance::Supplied.tag(),
            plan.strip_mm,
            plan.strip_mm / mm_per_mv,
        ),
        AmplitudePlan::Uncalibrated => format!(
            "  y: UNCALIBRATED — no counts-per-mV supplied, so NO mm/mV is claimed · {:.2} dot/mm · window {:.1} mm",
            g.dots_per_mm_y(),
            plan.strip_mm,
        ),
    });
    lines.push(format!(
        "  sample rate {:.1} Hz ({}) · cell aspect {:.2} ({}) → dot aspect {:.3}{}",
        plan.sample_rate_hz,
        plan.sample_rate_from.tag(),
        g.cell_aspect,
        plan.cell_aspect_from.tag(),
        g.dot_aspect(),
        if g.is_square_dot() { " (square)" } else { " (NOT square — the axes use separate dots/mm)" },
    ));
    if let Some(note) = fit_note(plan) {
        lines.push(format!("  {note}"));
    }
    lines
}

/// What the fit did, when it did anything. Silence would be the dishonest option.
fn fit_note(plan: &Plan) -> Option<String> {
    match plan.note {
        FitNote::AsAsked => None,
        FitNote::FinerDots { dots_per_mm_x } => Some(format!(
            "fit: the terminal had room for {dots_per_mm_x:.0} dots/mm — finer dots, SAME {:.1} mm/s",
            plan.mm_per_s
        )),
        FitNote::ShorterStrip { asked_s, got_s } => Some(format!(
            "fit: terminal too narrow for a {asked_s:.1} s strip at {:.1} mm/s — SHORTENED to {got_s:.1} s. \
             The mm/s was NOT changed.",
            plan.mm_per_s
        )),
    }
}

/// The footer: what the picture actually is, measured.
pub fn footer(plan: &Plan, report: &Report, mode: Option<&str>) -> Vec<String> {
    let mut lines = vec![format!(
        "  achieved x: {:.3} mm/s at {:.2} dot/mm ({:.4} s per dot column, {:.4} s per terminal column)",
        plan.mm_per_s,
        plan.geom.dots_per_mm_x,
        plan.s_per_dot_column(),
        plan.s_per_dot_column() * crate::braille::CELL_DOTS_W as f64,
    )];
    lines.push(format!("  achieved y: {}", achieved_y(report)));
    if plan.grid {
        lines.push(format!(
            "  grid: {:.0} mm major lines (dim); {:.0} mm minor {}",
            grid::MAJOR_MM,
            grid::MINOR_MM,
            grid::minor_note(plan),
        ));
    } else {
        lines.push("  grid: not drawn".to_string());
    }
    lines.push(format!(
        "  sample rate {:.1} Hz ({}) · {} samples in, {} drawn",
        plan.sample_rate_hz,
        plan.sample_rate_from.tag(),
        report.samples,
        report.drawn,
    ));
    lines.push(lead_off_line(report));
    lines.push(format!(
        "  clipped: {} sample{} fell outside the {:.1} mm window — the scale was NOT changed to fit them",
        report.clipped,
        if report.clipped == 1 { "" } else { "s" },
        plan.strip_mm,
    ));
    if !report.primed_on_contact {
        lines.push("  ⚠ the vertical mapping was primed on LEAD-OFF samples — no in-contact data arrived first".into());
    }
    if let Some(mode) = mode {
        lines.push(format!("  mode: {mode}"));
    }
    if !plan.amplitude.is_calibrated() {
        lines.push(UNCALIBRATED_BANNER.to_string());
    }
    lines.push("  wellness estimate, never medical.".to_string());
    lines
}

fn achieved_y(report: &Report) -> String {
    match report.vertical {
        None => "not yet frozen — still priming, nothing drawn".to_string(),
        Some(VerticalBasis::Calibrated { counts_per_mv, mm_per_mv }) => format!(
            "{mm_per_mv:.1} mm/mV from {counts_per_mv:.3} counts/mV ({}), {:.3} counts per dot, baseline {:.1} counts",
            super::plan::Provenance::Supplied.tag(),
            report.counts_per_dot.unwrap_or_default(),
            report.centre_counts.unwrap_or_default(),
        ),
        Some(VerticalBasis::Autoscaled { counts_per_mm }) => format!(
            "UNCALIBRATED — autoscaled to {counts_per_mm:.3} counts/mm ({:.3} counts per dot), centre {:.1} counts. \
             {:.0} mm is {:.1} counts, NOT millivolts",
            report.counts_per_dot.unwrap_or_default(),
            report.centre_counts.unwrap_or_default(),
            grid::MAJOR_MM,
            counts_per_mm * grid::MAJOR_MM,
        ),
        Some(VerticalBasis::Flat) => format!(
            "UNCALIBRATED — the priming window had no range, so NO vertical scale is claimed \
             (flat trace at the midline, centre {:.1} counts)",
            report.centre_counts.unwrap_or_default(),
        ),
    }
}

fn lead_off_line(report: &Report) -> String {
    if report.lead_off.is_empty() {
        return "  lead-off: none — every column carries in-contact samples".to_string();
    }
    let spans: Vec<String> =
        report.lead_off.iter().take(8).map(|s| format!("{:.2}-{:.2} s", s.start_s, s.end_s)).collect();
    let more = report.lead_off.len().saturating_sub(spans.len());
    format!(
        "  lead-off: {} span{}, {:.2} s total — hatched, samples NOT drawn, never interpolated: {}{}",
        report.lead_off.len(),
        if report.lead_off.len() == 1 { "" } else { "s" },
        report.lead_off_s,
        spans.join(", "),
        if more > 0 { format!(", +{more} more") } else { String::new() },
    )
}

/// The gutter cell for one row of a strip: the strip's start time on its first row, blanks below.
pub fn gutter(plan: &Plan, strip: usize, row: usize) -> String {
    if row == 0 {
        format!("{:>width$.1}s ", strip as f64 * plan.strip_s, width = GUTTER_COLS - 2)
    } else {
        " ".repeat(GUTTER_COLS)
    }
}

/// The whole frame: banner, every strip with its gutter, footer. `strips` limits how many are shown,
/// so the sequential mode can emit them as they complete.
pub fn frame(r: &EcgRenderer, colour: bool, mode: Option<&str>, strips: usize) -> Vec<String> {
    let plan = r.plan();
    let mut out = banner(plan);
    for strip in 0..strips.min(plan.strips) {
        out.push(String::new());
        out.extend(strip_rows(r, strip, colour));
    }
    out.push(String::new());
    out.extend(footer(plan, &r.report(), mode));
    out
}

/// One strip's terminal rows, gutter included.
pub fn strip_rows(r: &EcgRenderer, strip: usize, colour: bool) -> Vec<String> {
    let plan = r.plan();
    r.strip_spans(strip)
        .iter()
        .enumerate()
        .map(|(row, spans)| format!("{}{}", gutter(plan, strip, row), paint(spans, colour)))
        .collect()
}

/// Upper bound on the rows the full frame occupies, for deciding whether it can be redrawn in place.
/// `a_frame_never_exceeds_its_row_bound` keeps the footer allowance honest.
pub fn frame_rows(plan: &Plan) -> usize {
    banner(plan).len() + plan.strips * (1 + plan.strip_rows()) + 1 + MAX_FOOTER_ROWS
}

/// Six fixed footer lines plus the four conditional ones.
const MAX_FOOTER_ROWS: usize = 10;
