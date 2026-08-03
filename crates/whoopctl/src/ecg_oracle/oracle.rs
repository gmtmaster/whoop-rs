//! The measurement oracle: read the scale back OUT of a rendered picture.
//!
//! Everything here works from a [`DotGrid`] and plain numbers. It imports `std` and the dot decoder
//! and nothing else — no renderer type, no renderer constant, no renderer function — so a measurement
//! can never be the renderer's own arithmetic handed back. An assertion that re-derives its expected
//! value with the code under test proves nothing, and that is the mistake this module is shaped to
//! refuse. `the_oracle_imports_std_and_the_dot_decoder_only` holds the line mechanically.
//!
//! Row 0 is the top of the picture, so a SMALLER row index is a HIGHER trace.

use super::decode::{DotGrid, DOTS_W};

/// Dots across one terminal cell. A measurement is made in DOT columns, which is what a dot grid is
/// indexed in; this is only ever used to restate one in the cells a width budget is spent in.
pub const DOTS_PER_CELL_COLUMN: f64 = DOTS_W as f64;

/// The scale a picture is CLAIMED to have been drawn at.
///
/// The oracle only ever divides by these. It derives no scale of its own, so a renderer that quietly
/// drew at some other scale shows up as a measurement that disagrees with its claim.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClaimedScale {
    pub dots_per_mm: f64,
    pub mm_per_mv: f64,
    pub mm_per_s: f64,
}

impl ClaimedScale {
    /// The 1 dot/mm, 10 mm/mV, 25 mm/s reference: 10 dots per mV, 12.5 columns per second.
    pub fn reference() -> Self {
        ClaimedScale { dots_per_mm: 1.0, mm_per_mv: 10.0, mm_per_s: 25.0 }
    }

    /// Seconds one DOT column would cover IF the claim held — what a measurement is checked against.
    pub fn claimed_s_per_dot_column(&self) -> f64 {
        (1.0 / self.dots_per_mm) / self.mm_per_s
    }

    /// The same per terminal cell: 0.08 s at the reference scale, so 12.5 cells a second.
    pub fn claimed_s_per_cell_column(&self) -> f64 {
        self.claimed_s_per_dot_column() * DOTS_PER_CELL_COLUMN
    }

    /// Terminal cells one second would occupy IF the claim held — the width budget.
    pub fn claimed_cell_columns_per_second(&self) -> f64 {
        1.0 / self.claimed_s_per_cell_column()
    }

    /// Dots one millivolt would stand IF the claim held.
    pub fn claimed_dots_per_mv(&self) -> f64 {
        self.mm_per_mv * self.dots_per_mm
    }
}

/// What a picture actually depicts vertically, measured from its dots.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AmplitudeReading {
    /// Centre-to-centre distance between the highest and lowest trace dot. A 10-dot excursion touches
    /// 11 dot centres, so this is `bottom - top`, not a count of dots.
    pub peak_to_peak_dots: usize,
    pub peak_to_peak_mm: f64,
    /// The millivolts the picture depicts under the claimed scale. Meaningful only once a
    /// counts-per-mV has calibrated the samples that were fed in; until then the axis is arbitrary.
    pub peak_to_peak_mv: f64,
    pub top_row: usize,
    pub bottom_row: usize,
    pub columns_with_trace: usize,
}

/// Measure the height of the trace. `None` when the picture carries no dots at all.
pub fn measure_amplitude(grid: &DotGrid, claim: ClaimedScale) -> Option<AmplitudeReading> {
    let (top_row, bottom_row) = extent(grid)?;
    let peak_to_peak_dots = bottom_row - top_row;
    let peak_to_peak_mm = peak_to_peak_dots as f64 / claim.dots_per_mm;
    Some(AmplitudeReading {
        peak_to_peak_dots,
        peak_to_peak_mm,
        peak_to_peak_mv: peak_to_peak_mm / claim.mm_per_mv,
        top_row,
        bottom_row,
        columns_with_trace: (0..grid.width()).filter(|x| !grid.column_rows(*x).is_empty()).count(),
    })
}

/// Highest and lowest set row in the whole picture.
pub fn extent(grid: &DotGrid) -> Option<(usize, usize)> {
    let mut found = None;
    for y in 0..grid.height() {
        if (0..grid.width()).any(|x| grid.get(x, y)) {
            found = Some(match found {
                None => (y, y),
                Some((t, _)) => (t, y),
            });
        }
    }
    found
}

/// A level transition in the picture. `rising` = the trace moved UP the screen.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Edge {
    pub column: usize,
    pub rising: bool,
}

/// Square-wave edges, from the midline of the picture's own vertical extent.
///
/// A column is HIGH when its median set row sits above the midline. The transition column of a drawn
/// vertical stroke straddles the midline and lands one column late on a rise and on time on a fall —
/// a constant bias, so rise-to-rise spacing is unaffected.
pub fn find_edges(grid: &DotGrid) -> Vec<Edge> {
    find_edges_min_run(grid, 1)
}

/// Edges that survive a minimum run at each level — rejects the flapping a noisy trace produces.
pub fn find_edges_min_run(grid: &DotGrid, min_run: usize) -> Vec<Edge> {
    let Some((top, bottom)) = extent(grid) else {
        return Vec::new();
    };
    let mid = (top + bottom) as f64 / 2.0;
    let levels: Vec<(usize, bool)> = (0..grid.width())
        .filter_map(|x| column_median_row(grid, x).map(|m| (x, m < mid)))
        .collect();

    let mut runs = run_lengths(&levels);
    absorb_short_runs(&mut runs, min_run.max(1));
    runs.iter().skip(1).map(|(col, high, _)| Edge { column: *col, rising: *high }).collect()
}

/// Median of the set rows in a column, `None` when the column is empty.
pub fn column_median_row(grid: &DotGrid, x: usize) -> Option<f64> {
    let rows = grid.column_rows(x);
    match rows.len() {
        0 => None,
        n if n % 2 == 1 => Some(rows[n / 2] as f64),
        n => Some((rows[n / 2 - 1] + rows[n / 2]) as f64 / 2.0),
    }
}

/// Run-length encode the level sequence into (first column, level, length).
fn run_lengths(levels: &[(usize, bool)]) -> Vec<(usize, bool, usize)> {
    let mut runs: Vec<(usize, bool, usize)> = Vec::new();
    for (col, high) in levels {
        match runs.last_mut() {
            Some(last) if last.1 == *high => last.2 += 1,
            _ => runs.push((*col, *high, 1)),
        }
    }
    runs
}

/// Fold runs shorter than `min_run` into the run before them, then re-merge neighbours that match.
fn absorb_short_runs(runs: &mut Vec<(usize, bool, usize)>, min_run: usize) {
    let mut i = 1;
    while i < runs.len() {
        if runs[i].2 < min_run {
            let len = runs[i].2;
            runs[i - 1].2 += len;
            runs.remove(i);
            if i < runs.len() && runs[i].1 == runs[i - 1].1 {
                let len = runs[i].2;
                runs[i - 1].2 += len;
                runs.remove(i);
            }
        } else {
            i += 1;
        }
    }
}

/// What a picture actually depicts horizontally, measured from the spacing of its own edges.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TimeReading {
    /// Whole cycles spanned by the first and last rising edge.
    pub cycles: usize,
    /// DOT columns per cycle — the measurement, in the units a dot grid is indexed in.
    pub dot_columns_per_cycle: f64,
    /// The same restated in terminal cells: 12.5 per second at the reference scale.
    pub cell_columns_per_cycle: f64,
    pub s_per_dot_column: f64,
    pub s_per_cell_column: f64,
    /// The mm/s the picture depicts, under the claimed dots-per-mm. If this is not the claimed mm/s,
    /// the renderer rescaled.
    pub mm_per_s_depicted: f64,
}

/// Measure time from the rising edges of a signal whose period is known.
///
/// `None` when fewer than two rising edges are visible, or the period is not positive — there is then
/// no distance to measure and no number to report.
pub fn measure_time(grid: &DotGrid, known_period_s: f64, claim: ClaimedScale) -> Option<TimeReading> {
    let edges = find_edges(grid);
    measure_time_from_edges(&edges, known_period_s, claim)
}

/// The same measurement from an already-found edge list, for a signal whose edges need custom filtering.
pub fn measure_time_from_edges(edges: &[Edge], known_period_s: f64, claim: ClaimedScale) -> Option<TimeReading> {
    if known_period_s <= 0.0 {
        return None;
    }
    let rising: Vec<usize> = edges.iter().filter(|e| e.rising).map(|e| e.column).collect();
    let (&first, &last) = (rising.first()?, rising.last()?);
    let cycles = rising.len() - 1;
    if cycles == 0 || last <= first {
        return None;
    }
    let dot_columns_per_cycle = (last - first) as f64 / cycles as f64;
    let s_per_dot_column = known_period_s / dot_columns_per_cycle;
    Some(TimeReading {
        cycles,
        dot_columns_per_cycle,
        cell_columns_per_cycle: dot_columns_per_cycle / DOTS_PER_CELL_COLUMN,
        s_per_dot_column,
        s_per_cell_column: s_per_dot_column * DOTS_PER_CELL_COLUMN,
        mm_per_s_depicted: (1.0 / claim.dots_per_mm) / s_per_dot_column,
    })
}

/// The verdict of a picture against its claim, both axes as relative errors.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScaleVerdict {
    pub measured_mv: f64,
    pub expected_mv: f64,
    pub amplitude_error: f64,
    pub measured_mm_per_s: f64,
    pub claimed_mm_per_s: f64,
    pub time_error: f64,
    pub agrees: bool,
}

/// Does the picture depict the amplitude and period it was fed, at the scale it claims?
///
/// `tol` is the relative tolerance both axes must be inside. `None` when either axis could not be
/// measured — an unmeasurable picture is not a passing one, and the caller has to say so.
pub fn check(
    grid: &DotGrid,
    claim: ClaimedScale,
    expected_mv: f64,
    known_period_s: f64,
    tol: f64,
) -> Option<ScaleVerdict> {
    let amp = measure_amplitude(grid, claim)?;
    let time = measure_time(grid, known_period_s, claim)?;
    let amplitude_error = relative_error(amp.peak_to_peak_mv, expected_mv);
    let time_error = relative_error(time.mm_per_s_depicted, claim.mm_per_s);
    Some(ScaleVerdict {
        measured_mv: amp.peak_to_peak_mv,
        expected_mv,
        amplitude_error,
        measured_mm_per_s: time.mm_per_s_depicted,
        claimed_mm_per_s: claim.mm_per_s,
        time_error,
        agrees: amplitude_error <= tol && time_error <= tol,
    })
}

/// |measured - expected| / |expected|, and infinite when the expectation is zero and the measurement
/// is not — never a silent 0.0 that would read as agreement.
pub fn relative_error(measured: f64, expected: f64) -> f64 {
    if expected == 0.0 {
        return if measured == 0.0 { 0.0 } else { f64::INFINITY };
    }
    (measured - expected).abs() / expected.abs()
}
