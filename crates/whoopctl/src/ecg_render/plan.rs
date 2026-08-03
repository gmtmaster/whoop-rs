//! What the picture claims, and whether the terminal can honour it.
//!
//! Every number a label will quote is decided here, once, before a dot is drawn. The fit may give the
//! picture FINER dots or a SHORTER strip; it may never change mm/s or mm/mV, because a rescaled trace
//! under an unchanged label is the one failure this whole module exists to refuse.

use crate::braille::{CELL_DOTS_H, CELL_DOTS_W};

/// Left margin holding each strip's start time, e.g. `  25.0s `.
pub const GUTTER_COLS: usize = 8;

/// The shortest strip a narrow terminal may be cut down to before the render is refused.
pub const MIN_STRIP_S: f64 = 1.0;

/// Strip lengths are cut to this granularity, so a shortened strip is still a round number.
const STRIP_STEP_S: f64 = 0.1;

/// Fraction of the strip window an autoscaled (uncalibrated) trace is mapped into.
const AUTOSCALE_FILL: f64 = 0.8;

/// Beyond this the dots are finer than any eye needs and the frame stops fitting anything else.
const MAX_DOTS_PER_MM: f64 = 8.0;

/// Where a number came from. Printed next to the number itself, in the default output.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Provenance {
    /// Given by the operator on the command line.
    Supplied,
    /// Read from the environment.
    Environment,
    /// Nobody knows it yet. Every use of it must carry this tag.
    Assumed,
}

impl Provenance {
    pub fn tag(self) -> &'static str {
        match self {
            Provenance::Supplied => "SUPPLIED",
            Provenance::Environment => "FROM ENV",
            Provenance::Assumed => "ASSUMED",
        }
    }
}

/// Dot geometry. A braille cell is 2 dots wide and 4 tall, so a dot's own aspect is HALF the terminal
/// cell's: at the usual cell height/width of 2 the dot is square and one dots-per-mm serves both axes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Geometry {
    pub dots_per_mm_x: f64,
    /// Terminal cell height / width. Nobody can measure this from inside a terminal, so it is assumed.
    pub cell_aspect: f64,
}

impl Geometry {
    /// Dot height / dot width.
    pub fn dot_aspect(self) -> f64 {
        self.cell_aspect * CELL_DOTS_W as f64 / CELL_DOTS_H as f64
    }

    /// Dots per mm DOWN the screen. A taller dot needs fewer of them to span the same millimetre.
    pub fn dots_per_mm_y(self) -> f64 {
        self.dots_per_mm_x / self.dot_aspect()
    }

    /// Whether the two axes share one dots-per-mm figure.
    pub fn is_square_dot(self) -> bool {
        (self.dot_aspect() - 1.0).abs() < 1e-9
    }
}

/// The amplitude axis. There is no default counts-per-mV: without one the axis is raw ADC counts and
/// no mm/mV is claimed anywhere.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AmplitudePlan {
    Calibrated { counts_per_mv: f64, mm_per_mv: f64 },
    Uncalibrated,
}

impl AmplitudePlan {
    pub fn is_calibrated(self) -> bool {
        matches!(self, AmplitudePlan::Calibrated { .. })
    }
}

/// The sample rate assumed when nobody supplies one. ASSUMED, not read off the device — `Request`
/// tags it `Provenance::Assumed` so every frame prints it as such.
pub const ASSUMED_SAMPLE_RATE_HZ: f64 = 500.0;

/// What the caller asks for, before the terminal has had its say.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Request {
    pub duration_s: f64,
    pub strip_s: f64,
    pub mm_per_s: f64,
    pub mm_per_mv: f64,
    /// Height of one strip's window, in millimetres of screen.
    pub strip_mm: f64,
    pub cell_aspect: f64,
    pub cell_aspect_from: Provenance,
    /// `None` fits the finest whole dots-per-mm the width allows.
    pub dots_per_mm_x: Option<f64>,
    /// `None` leaves the amplitude axis uncalibrated.
    pub counts_per_mv: Option<f64>,
    pub sample_rate_hz: f64,
    pub sample_rate_from: Provenance,
    /// Seconds of samples used to freeze the vertical mapping before anything is drawn.
    pub prime_s: f64,
    pub grid: bool,
}

impl Default for Request {
    fn default() -> Self {
        Request {
            duration_s: 30.0,
            strip_s: 5.0,
            mm_per_s: 25.0,
            mm_per_mv: 10.0,
            strip_mm: 24.0,
            cell_aspect: 2.0,
            cell_aspect_from: Provenance::Assumed,
            dots_per_mm_x: None,
            counts_per_mv: None,
            sample_rate_hz: ASSUMED_SAMPLE_RATE_HZ,
            sample_rate_from: Provenance::Assumed,
            prime_s: 1.0,
            grid: true,
        }
    }
}

/// The terminal the frame has to live in.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Terminal {
    pub cols: usize,
    pub rows: usize,
    pub cols_from: Provenance,
    pub rows_from: Provenance,
}

impl Terminal {
    pub fn new(cols: usize, rows: usize) -> Self {
        Terminal { cols, rows, cols_from: Provenance::Supplied, rows_from: Provenance::Supplied }
    }
}

/// What the fit had to do to the request. Never a change of mm/s or mm/mV.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FitNote {
    AsAsked,
    /// The terminal was wide enough for more dots per millimetre — finer resolution, same mm/s.
    FinerDots { dots_per_mm_x: f64 },
    /// The terminal was too narrow, so the STRIP was cut. The sweep speed is untouched.
    ShorterStrip { asked_s: f64, got_s: f64 },
}

/// Why a frame could not be produced. Refusing is the correct outcome; rescaling silently is not.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FitError {
    TooNarrow { have_cols: usize, need_cols: usize, min_strip_s: f64, mm_per_s: f64 },
    BadRequest(&'static str),
}

impl std::fmt::Display for FitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FitError::TooNarrow { have_cols, need_cols, min_strip_s, mm_per_s } => write!(
                f,
                "terminal is {have_cols} columns; {need_cols} are needed for the shortest useful strip \
                 ({min_strip_s:.1} s at {mm_per_s:.1} mm/s). Refusing rather than drawing at a different \
                 mm/s under a {mm_per_s:.1} mm/s label — widen the terminal, or pass --width",
            ),
            FitError::BadRequest(why) => write!(f, "cannot render: {why}"),
        }
    }
}

impl std::error::Error for FitError {}

/// The frozen render plan: sizes in dots, and every label's number.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Plan {
    pub geom: Geometry,
    pub mm_per_s: f64,
    pub amplitude: AmplitudePlan,
    pub duration_s: f64,
    pub strip_s: f64,
    pub strips: usize,
    pub strip_dot_w: usize,
    pub strip_dot_h: usize,
    pub strip_mm: f64,
    pub sample_rate_hz: f64,
    pub sample_rate_from: Provenance,
    pub cell_aspect_from: Provenance,
    pub grid: bool,
    pub minor_grid: bool,
    pub prime_samples: usize,
    pub note: FitNote,
}

impl Plan {
    /// Terminal columns one strip's plot region occupies.
    pub fn plot_cols(&self) -> usize {
        self.strip_dot_w.div_ceil(CELL_DOTS_W)
    }

    /// Terminal rows one strip occupies.
    pub fn strip_rows(&self) -> usize {
        self.strip_dot_h.div_ceil(CELL_DOTS_H)
    }

    /// Dot column of a time offset WITHIN a strip.
    pub fn dot_column(&self, t_in_strip_s: f64) -> usize {
        let x = (t_in_strip_s * self.mm_per_s * self.geom.dots_per_mm_x).round();
        (x.max(0.0) as usize).min(self.strip_dot_w.saturating_sub(1))
    }

    /// Seconds one dot column covers — 0.04 s at 25 mm/s and 1 dot/mm.
    pub fn s_per_dot_column(&self) -> f64 {
        1.0 / (self.mm_per_s * self.geom.dots_per_mm_x)
    }

    /// Dots per millivolt, or `None` while the axis is uncalibrated — there is no figure to give.
    pub fn dots_per_mv(&self) -> Option<f64> {
        match self.amplitude {
            AmplitudePlan::Calibrated { mm_per_mv, .. } => Some(mm_per_mv * self.geom.dots_per_mm_y()),
            AmplitudePlan::Uncalibrated => None,
        }
    }

    /// Dots an autoscaled trace's primed peak-to-peak is mapped onto.
    pub fn autoscale_dots(&self) -> f64 {
        (self.strip_mm * AUTOSCALE_FILL * self.geom.dots_per_mm_y()).max(1.0)
    }
}

/// Fit a request to a terminal: finer dots if there is room, a shorter strip if there is not, and a
/// refusal if even the shortest strip will not fit. The sweep speed and the mm/mV are never touched.
pub fn fit(req: &Request, term: Terminal) -> Result<Plan, FitError> {
    for (ok, why) in [
        (req.duration_s > 0.0, "duration must be positive"),
        (req.strip_s > 0.0, "strip length must be positive"),
        (req.mm_per_s > 0.0, "mm/s must be positive"),
        (req.strip_mm > 0.0, "the strip window must be positive"),
        (req.cell_aspect > 0.0, "cell aspect must be positive"),
        (req.sample_rate_hz > 0.0, "sample rate must be positive"),
        (req.counts_per_mv.is_none_or(|c| c > 0.0), "counts-per-mV must be positive"),
        (req.dots_per_mm_x.is_none_or(|k| k > 0.0), "dots-per-mm must be positive"),
        (req.mm_per_mv > 0.0, "mm/mV must be positive"),
    ] {
        if !ok {
            return Err(FitError::BadRequest(why));
        }
    }

    let avail_dots = term.cols.saturating_sub(GUTTER_COLS) * CELL_DOTS_W;
    let dots_for = |strip_s: f64, k: f64| (strip_s * req.mm_per_s * k).round() as usize + 1;

    // Resolution first: more dots per millimetre is a sharper picture at the SAME mm/s.
    let (k, mut note) = match req.dots_per_mm_x {
        Some(k) => (k, FitNote::AsAsked),
        None => {
            let mut k = 1.0f64;
            while k < MAX_DOTS_PER_MM && dots_for(req.strip_s, k + 1.0) <= avail_dots {
                k += 1.0;
            }
            (k, if k > 1.0 { FitNote::FinerDots { dots_per_mm_x: k } } else { FitNote::AsAsked })
        }
    };

    // Still too wide: cut the STRIP, never the scale.
    let mut strip_s = req.strip_s;
    if dots_for(strip_s, k) > avail_dots {
        let room = avail_dots.saturating_sub(1) as f64;
        let max_s = (room / (req.mm_per_s * k) / STRIP_STEP_S).floor() * STRIP_STEP_S;
        if max_s < MIN_STRIP_S {
            return Err(FitError::TooNarrow {
                have_cols: term.cols,
                need_cols: GUTTER_COLS + dots_for(MIN_STRIP_S, k).div_ceil(CELL_DOTS_W),
                min_strip_s: MIN_STRIP_S,
                mm_per_s: req.mm_per_s,
            });
        }
        strip_s = max_s;
        note = FitNote::ShorterStrip { asked_s: req.strip_s, got_s: strip_s };
    }

    let geom = Geometry { dots_per_mm_x: k, cell_aspect: req.cell_aspect };
    let dots_y = geom.dots_per_mm_y();
    if dots_y <= 0.0 || !dots_y.is_finite() {
        return Err(FitError::BadRequest("cell aspect leaves no vertical resolution"));
    }
    let strip_dot_h = (req.strip_mm * dots_y).round().max(CELL_DOTS_H as f64) as usize;

    Ok(Plan {
        geom,
        mm_per_s: req.mm_per_s,
        amplitude: match req.counts_per_mv {
            Some(counts_per_mv) => AmplitudePlan::Calibrated { counts_per_mv, mm_per_mv: req.mm_per_mv },
            None => AmplitudePlan::Uncalibrated,
        },
        duration_s: req.duration_s,
        strip_s,
        strips: (req.duration_s / strip_s).ceil().max(1.0) as usize,
        strip_dot_w: dots_for(strip_s, k),
        strip_dot_h,
        strip_mm: req.strip_mm,
        sample_rate_hz: req.sample_rate_hz,
        sample_rate_from: req.sample_rate_from,
        cell_aspect_from: req.cell_aspect_from,
        grid: req.grid,
        minor_grid: req.grid && k >= 2.0 && dots_y >= 2.0,
        prime_samples: (req.prime_s * req.sample_rate_hz).round().max(1.0) as usize,
        note,
    })
}
