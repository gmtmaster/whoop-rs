//! The amplitude mapping, FROZEN from a priming window before the first dot is drawn.
//!
//! A mapping that tracked the data would rescale dots already on screen — a silent rescale under an
//! unchanged label, which is the failure the renderer exists to refuse. So it is decided once, from
//! the first `prime_samples` in-contact samples, and everything after it that falls outside the window
//! is CLIPPED and counted, never accommodated.

use super::plan::{AmplitudePlan, Plan};

/// What the vertical axis means. `Autoscaled` and `Flat` both claim NO millivolts.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum VerticalBasis {
    /// A counts-per-mV was supplied, so the axis is millivolts at a fixed mm/mV.
    Calibrated { counts_per_mv: f64, mm_per_mv: f64 },
    /// No counts-per-mV: raw ADC counts, stretched to the window. Not millivolts.
    Autoscaled { counts_per_mm: f64 },
    /// The priming window had no range at all, so no vertical scale is claimed.
    Flat,
}

/// Counts to dot rows, and the basis it was frozen on.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VerticalMap {
    centre_counts: f64,
    counts_per_dot: f64,
    mid_row: f64,
    height: usize,
    basis: VerticalBasis,
    /// Whether the priming window contained any in-contact sample. False means it was frozen on
    /// lead-off data, which the footer has to say.
    primed_on_contact: bool,
}

impl VerticalMap {
    /// Freeze the mapping. `primed` is the priming window; `contact` flags it sample for sample.
    pub fn freeze(plan: &Plan, primed: &[f64], contact: &[bool]) -> Self {
        let usable: Vec<f64> = primed
            .iter()
            .zip(contact.iter().chain(std::iter::repeat(&true)))
            .filter(|(_, c)| **c)
            .map(|(v, _)| *v)
            .collect();
        let primed_on_contact = !usable.is_empty();
        let window: &[f64] = if primed_on_contact { &usable } else { primed };

        let mid_row = plan.strip_dot_h as f64 / 2.0;
        let dots_y = plan.geom.dots_per_mm_y();
        let (lo, hi) = min_max(window);
        let mut map = VerticalMap {
            centre_counts: 0.0,
            counts_per_dot: 1.0,
            mid_row,
            height: plan.strip_dot_h,
            basis: VerticalBasis::Flat,
            primed_on_contact,
        };

        match plan.amplitude {
            AmplitudePlan::Calibrated { counts_per_mv, mm_per_mv } => {
                map.centre_counts = median(window);
                map.counts_per_dot = counts_per_mv / (mm_per_mv * dots_y);
                map.basis = VerticalBasis::Calibrated { counts_per_mv, mm_per_mv };
            }
            AmplitudePlan::Uncalibrated => {
                let range = hi - lo;
                map.centre_counts = (lo + hi) / 2.0;
                if range > 0.0 && range.is_finite() {
                    map.counts_per_dot = range / plan.autoscale_dots();
                    map.basis = VerticalBasis::Autoscaled { counts_per_mm: map.counts_per_dot * dots_y };
                }
            }
        }
        map
    }

    /// Dot row of a sample. May land outside the strip — the caller counts that as a clip rather than
    /// moving the scale to fit it.
    pub fn row(&self, counts: f64) -> i32 {
        let offset = (counts - self.centre_counts) / self.counts_per_dot;
        (self.mid_row - offset).round() as i32
    }

    /// The strip's own midline, where a flat baseline sits.
    pub fn mid_row(&self) -> i32 {
        self.mid_row.round() as i32
    }

    pub fn in_window(&self, row: i32) -> bool {
        row >= 0 && (row as usize) < self.height
    }

    pub fn basis(&self) -> VerticalBasis {
        self.basis
    }

    pub fn centre_counts(&self) -> f64 {
        self.centre_counts
    }

    pub fn counts_per_dot(&self) -> f64 {
        self.counts_per_dot
    }

    pub fn primed_on_contact(&self) -> bool {
        self.primed_on_contact
    }
}

fn min_max(v: &[f64]) -> (f64, f64) {
    v.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(l, h), x| (l.min(*x), h.max(*x)))
}

/// Median of a window, 0.0 for an empty one. Used only to CENTRE a calibrated trace; it never sets a
/// scale, so an empty window costs a centred baseline and no invented number.
fn median(v: &[f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = s.len();
    if n % 2 == 1 {
        s[n / 2]
    } else {
        (s[n / 2 - 1] + s[n / 2]) / 2.0
    }
}
