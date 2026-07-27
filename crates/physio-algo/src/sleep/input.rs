//! Protocol-free inputs to the sleep stager. Each is a plain per-sample value with a wall-clock unix
//! second timestamp — decoupled from any device record or wire type. All timestamps are unix SECONDS.

/// One heart-rate sample: `bpm` at second `ts`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HrSample {
    pub ts: i64,
    pub bpm: u16,
}

/// A run of consecutive R-R intervals sharing one timestamp anchor.
///
/// A single physiological record/notification reports several beats under one whole-second `ts`; the
/// `intervals` are the successive beat-to-beat gaps in milliseconds, in emission order. Keeping them as a
/// run (not one flattened `(ts, rr_ms)` list) preserves intra-second ordering, which the RSA respiration
/// term and gap-aware RMSSD both read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RrRun {
    pub ts: i64,
    pub intervals: Vec<u16>,
}

/// One 3-axis accelerometer / gravity-vector sample (unit: g). Motion and jerk features derive from this.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AccelSample {
    pub ts: i64,
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

/// One step-counter sample: a wrap-aware u16 `counter` and an optional activity class (1=walk, 2=run) at
/// second `ts`. Motion-aware wake refinement reads the per-minute walk-class tick cadence from these.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StepSample {
    pub ts: i64,
    pub counter: u16,
    pub activity_class: Option<u8>,
}

/// The full input bundle for one detected in-bed span. `[start, end]` are wall-clock unix seconds; the
/// stager tiles that span with 30 s epochs and reads only samples inside it (plus a small feature pad).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SleepInput {
    pub start: i64,
    pub end: i64,
    pub hr: Vec<HrSample>,
    pub rr: Vec<RrRun>,
    pub accel: Vec<AccelSample>,
}
