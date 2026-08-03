//! The progressive renderer: push samples, redraw. Strips and their layers are allocated once by
//! [`EcgRenderer::new`] and drawn INTO, so a push costs the dots it adds and nothing is reallocated
//! per sample. A redraw is a fresh read of those layers, at whatever rate the driver picks.
//!
//! Layers back to front: minor grid, major grid, lead-off hatch, trace. The frontmost layer with a dot
//! in a cell owns that cell's style, so the trace stays legible over the grid.

use crate::braille::{Canvas, LayerStack, StyleId};

use super::grid;
use super::plan::Plan;
use super::vertical::{VerticalBasis, VerticalMap};

pub const STYLE_MINOR: StyleId = 1;
pub const STYLE_MAJOR: StyleId = 2;
pub const STYLE_LEADOFF: StyleId = 3;
pub const STYLE_TRACE: StyleId = 4;

/// One ADC sample and the AFE's own contact flag for it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sample {
    pub counts: f64,
    pub contact: bool,
}

impl Sample {
    pub fn new(counts: f64, contact: bool) -> Self {
        Sample { counts, contact }
    }
}

/// A closed lead-off span, in seconds from the start of the recording.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LeadOffSpan {
    pub start_s: f64,
    pub end_s: f64,
}

/// What actually happened, for the footer. Every field here is measured, none assumed.
#[derive(Clone, Debug, PartialEq)]
pub struct Report {
    pub samples: usize,
    pub drawn: usize,
    pub clipped: usize,
    pub lead_off: Vec<LeadOffSpan>,
    pub lead_off_s: f64,
    pub strips_complete: usize,
    /// `None` until the priming window has closed and the vertical mapping is frozen.
    pub vertical: Option<VerticalBasis>,
    pub centre_counts: Option<f64>,
    pub counts_per_dot: Option<f64>,
    pub primed_on_contact: bool,
}

/// The dot column being accumulated. A column takes the min and max of its samples so a narrow R peak
/// survives decimation; one lead-off sample makes the whole column lead-off.
#[derive(Clone, Copy, Debug)]
struct Column {
    strip: usize,
    x: usize,
    lo: i32,
    hi: i32,
    last: i32,
    has_contact: bool,
    lead_off: bool,
}

pub struct EcgRenderer {
    plan: Plan,
    strips: Vec<LayerStack>,
    trace_idx: usize,
    leadoff_idx: usize,
    vmap: Option<VerticalMap>,
    prime: Vec<Sample>,
    consumed: usize,
    drawn: usize,
    clipped: usize,
    accum: Option<Column>,
    prev: Option<(usize, usize, i32)>,
    lead_off: Vec<LeadOffSpan>,
    open_lead_off: Option<f64>,
}

impl EcgRenderer {
    /// Allocate every strip and its layers up front. Nothing here depends on the samples.
    pub fn new(plan: Plan) -> Self {
        let major = plan.grid.then(|| grid::major(&plan));
        let minor = grid::minor(&plan);
        let strips = (0..plan.strips)
            .map(|_| {
                let mut stack = LayerStack::new(plan.strip_dot_w, plan.strip_dot_h);
                if let Some(m) = &minor {
                    stack.push(STYLE_MINOR, m.clone());
                }
                if let Some(m) = &major {
                    stack.push(STYLE_MAJOR, m.clone());
                }
                stack.push_blank(STYLE_LEADOFF);
                stack.push_blank(STYLE_TRACE);
                stack
            })
            .collect::<Vec<_>>();
        let leadoff_idx = strips.first().map_or(0, |s| s.layer_count() - 2);
        let trace_idx = leadoff_idx + 1;
        EcgRenderer {
            plan,
            strips,
            trace_idx,
            leadoff_idx,
            vmap: None,
            prime: Vec::new(),
            consumed: 0,
            drawn: 0,
            clipped: 0,
            accum: None,
            prev: None,
            lead_off: Vec::new(),
            open_lead_off: None,
        }
    }

    pub fn plan(&self) -> &Plan {
        &self.plan
    }

    /// Feed samples in stream order. Nothing is drawn until the priming window has closed and the
    /// vertical mapping is frozen; the buffered samples are then drawn at that frozen scale.
    pub fn push(&mut self, samples: &[Sample]) {
        self.consumed += samples.len();
        if self.vmap.is_none() {
            self.prime.extend_from_slice(samples);
            if self.prime.len() < self.plan.prime_samples {
                return;
            }
            self.freeze();
            let buffered = std::mem::take(&mut self.prime);
            self.draw(&buffered);
            return;
        }
        self.draw(samples);
    }

    /// End of stream: prime from whatever arrived if the window never filled, then flush the column
    /// in progress. Idempotent.
    pub fn finish(&mut self) {
        if self.vmap.is_none() {
            self.freeze();
            let buffered = std::mem::take(&mut self.prime);
            self.draw(&buffered);
        }
        self.flush_column();
        if let Some(start_s) = self.open_lead_off.take() {
            self.lead_off.push(LeadOffSpan { start_s, end_s: self.drawn as f64 / self.plan.sample_rate_hz });
        }
    }

    /// Strips whose whole time span has been drawn — what the sequential output mode emits.
    pub fn strips_complete(&self) -> usize {
        let done = (self.drawn as f64 / self.plan.sample_rate_hz / self.plan.strip_s).floor();
        (done.max(0.0) as usize).min(self.plan.strips)
    }

    pub fn report(&self) -> Report {
        let mut lead_off = self.lead_off.clone();
        if let Some(start_s) = self.open_lead_off {
            lead_off.push(LeadOffSpan { start_s, end_s: self.drawn as f64 / self.plan.sample_rate_hz });
        }
        Report {
            samples: self.consumed,
            drawn: self.drawn,
            clipped: self.clipped,
            lead_off_s: lead_off.iter().map(|s| s.end_s - s.start_s).sum(),
            lead_off,
            strips_complete: self.strips_complete(),
            vertical: self.vmap.map(|m| m.basis()),
            centre_counts: self.vmap.map(|m| m.centre_counts()),
            counts_per_dot: self.vmap.map(|m| m.counts_per_dot()),
            primed_on_contact: self.vmap.is_none_or(|m| m.primed_on_contact()),
        }
    }

    /// One strip's plot region as terminal rows, styles applied by the caller's mapping.
    pub fn strip_spans(&self, strip: usize) -> Vec<Vec<crate::braille::Span>> {
        self.strips.get(strip).map(LayerStack::render_spans).unwrap_or_default()
    }

    /// One strip's plot region as plain rows — the pipeable form, no escapes.
    pub fn strip_lines(&self, strip: usize) -> Vec<String> {
        self.strips.get(strip).map(LayerStack::render).unwrap_or_default()
    }

    /// One strip's TRACE layer alone. Used to prove the grid moves no trace dot; the scale itself is
    /// measured off the full strip.
    pub fn trace_lines(&self, strip: usize) -> Vec<String> {
        self.strips
            .get(strip)
            .and_then(|s| s.layer(self.trace_idx))
            .map(Canvas::render)
            .unwrap_or_default()
    }

    /// One strip's LEAD-OFF layer alone.
    pub fn lead_off_lines(&self, strip: usize) -> Vec<String> {
        self.strips
            .get(strip)
            .and_then(|s| s.layer(self.leadoff_idx))
            .map(Canvas::render)
            .unwrap_or_default()
    }

    /// Freeze on the first `prime_samples`, NOT on everything buffered — a push large enough to carry
    /// the whole recording would otherwise scale the picture differently from the same samples
    /// arriving in chunks, and the arrival pattern must not be visible in the output.
    fn freeze(&mut self) {
        let window = &self.prime[..self.prime.len().min(self.plan.prime_samples)];
        let counts: Vec<f64> = window.iter().map(|s| s.counts).collect();
        let contact: Vec<bool> = window.iter().map(|s| s.contact).collect();
        self.vmap = Some(VerticalMap::freeze(&self.plan, &counts, &contact));
    }

    fn draw(&mut self, samples: &[Sample]) {
        let Some(vmap) = self.vmap else { return };
        let rate = self.plan.sample_rate_hz;
        for s in samples {
            let i = self.drawn;
            self.drawn += 1;
            let t = i as f64 / rate;
            if t >= self.plan.duration_s {
                continue;
            }
            self.log_contact(t, s.contact);
            let strip = ((t / self.plan.strip_s).floor().max(0.0) as usize).min(self.plan.strips - 1);
            let x = self.plan.dot_column(t - strip as f64 * self.plan.strip_s);

            match self.accum {
                Some(c) if c.strip == strip && c.x == x => {}
                _ => {
                    self.flush_column();
                    self.accum =
                        Some(Column { strip, x, lo: i32::MAX, hi: i32::MIN, last: 0, has_contact: false, lead_off: false });
                }
            }
            let col = self.accum.as_mut().expect("just set");
            if !s.contact {
                col.lead_off = true;
                continue;
            }
            let row = vmap.row(s.counts);
            if !vmap.in_window(row) {
                self.clipped += 1;
            }
            col.lo = col.lo.min(row);
            col.hi = col.hi.max(row);
            col.last = row;
            col.has_contact = true;
        }
    }

    /// Track lead-off spans in seconds. A span opens on the first bad sample and closes on the first
    /// good one, so it is reported as measured rather than rounded to columns.
    fn log_contact(&mut self, t: f64, contact: bool) {
        match (contact, self.open_lead_off) {
            (false, None) => self.open_lead_off = Some(t),
            (true, Some(start_s)) => {
                self.lead_off.push(LeadOffSpan { start_s, end_s: t });
                self.open_lead_off = None;
            }
            _ => {}
        }
    }

    /// Draw the accumulated column. A lead-off column gets the hatch and BREAKS the trace, so no line
    /// is ever interpolated across a contact loss.
    fn flush_column(&mut self) {
        let Some(col) = self.accum.take() else { return };
        if col.lead_off {
            self.hatch(col.strip, col.x);
            self.prev = None;
            return;
        }
        if !col.has_contact {
            self.prev = None;
            return;
        }
        let (mut top, mut bottom) = (col.lo, col.hi);
        if let Some((ps, px, prow)) = self.prev {
            if ps == col.strip && px + 1 == col.x {
                top = top.min(prow);
                bottom = bottom.max(prow);
            }
        }
        let idx = self.trace_idx;
        if let Some(canvas) = self.strips.get_mut(col.strip).and_then(|s| s.layer_mut(idx)) {
            canvas.line(col.x as i32, top, col.x as i32, bottom);
        }
        self.prev = Some((col.strip, col.x, col.last));
    }

    /// A full-height diagonal hatch over one column: unmistakably not a trace, and impossible to read
    /// as a signal that happened to be flat.
    fn hatch(&mut self, strip: usize, x: usize) {
        let (h, idx) = (self.plan.strip_dot_h, self.leadoff_idx);
        let Some(canvas) = self.strips.get_mut(strip).and_then(|s| s.layer_mut(idx)) else { return };
        for y in (0..h).filter(|y| (x + y).is_multiple_of(3)) {
            canvas.set(x as i32, y as i32);
        }
    }
}
