//! The millimetre grid, drawn dim behind the trace so the scale is checkable by eye rather than
//! merely claimed. It is a grid of SCREEN millimetres: 5 mm across is always 0.2 s at 25 mm/s, and
//! 5 mm down is 0.5 mV only when a counts-per-mV has calibrated the axis.

use crate::braille::Canvas;

use super::plan::Plan;

/// Major grid pitch, in millimetres.
pub const MAJOR_MM: f64 = 5.0;
/// Minor grid pitch, in millimetres.
pub const MINOR_MM: f64 = 1.0;

/// The 5 mm grid, vertical anchored at t = 0 and horizontal at the strip's midline. The lines are
/// stippled — every second dot — so the pitch stays countable without the mesh burying the trace at
/// braille resolution.
pub fn major(plan: &Plan) -> Canvas {
    let (w, h) = (plan.strip_dot_w, plan.strip_dot_h);
    let mut c = Canvas::new(w, h);
    for x in ticks(0.0, MAJOR_MM * plan.geom.dots_per_mm_x, w) {
        for y in (0..h as i32).step_by(2) {
            c.set(x, y);
        }
    }
    for y in ticks(h as f64 / 2.0, MAJOR_MM * plan.geom.dots_per_mm_y(), h) {
        for x in (0..w as i32).step_by(2) {
            c.set(x, y);
        }
    }
    c
}

/// The 1 mm grid as intersection dots, or `None` below 2 dots/mm where 1 mm lines would fill the strip.
pub fn minor(plan: &Plan) -> Option<Canvas> {
    if !plan.minor_grid {
        return None;
    }
    let (w, h) = (plan.strip_dot_w, plan.strip_dot_h);
    let mut c = Canvas::new(w, h);
    let xs = ticks(0.0, MINOR_MM * plan.geom.dots_per_mm_x, w);
    let ys = ticks(h as f64 / 2.0, MINOR_MM * plan.geom.dots_per_mm_y(), h);
    for x in &xs {
        for y in &ys {
            c.set(*x, *y);
        }
    }
    Some(c)
}

/// Why the 1 mm grid is or is not drawn, for the footer to state.
pub fn minor_note(plan: &Plan) -> &'static str {
    if !plan.grid {
        "grid off"
    } else if plan.minor_grid {
        "drawn as intersection dots"
    } else {
        "not drawn (needs >= 2 dots/mm on both axes)"
    }
}

/// Grid positions `anchor + k * step`, rounded, that land inside `[0, limit)`. An unusable step (finer
/// than one dot) yields nothing rather than a solid fill.
fn ticks(anchor: f64, step: f64, limit: usize) -> Vec<i32> {
    if limit == 0 || step < 1.0 || !step.is_finite() {
        return Vec::new();
    }
    let n = (limit as f64 / step).ceil() as i32 + 1;
    let mut out: Vec<i32> = (-n..=n)
        .map(|k| (anchor + k as f64 * step).round())
        .filter(|v| *v >= 0.0 && (*v as usize) < limit)
        .map(|v| v as i32)
        .collect();
    out.dedup();
    out
}
