//! The decode sweep: search the unknown packet layout and sample rate, and let ECG morphology plus the
//! optical channel's own beats pick the winner.
//!
//! The dependency is inverted on purpose. Rather than waiting to be told how to decode the stream, every
//! plausible reading rule is applied to the same bytes and the resulting waveforms are scored; the rule
//! that produces something a QRS detector pair, the published quality indices and the wrist pulse all
//! agree is an ECG IS the answer. Physiology is the reference standard.
//!
//! **Counts-per-mV is not in the search space and never will be.** Morphology bounds a single-lead QRS
//! to roughly half a millivolt to two, which varies about fourfold between people and with electrode
//! placement, so a sweep that "found" a counts-per-mV would only be centring an assumption inside that
//! range and dressing it as a measurement. Nothing here returns, infers or implies an amplitude in mV;
//! every statistic used is invariant to the amplitude scale, which is exactly what lets it work on a
//! stream whose units are unknown.
//!
//! What the sweep does solve, and how:
//!
//! - **layout** — [`layout::candidates`] enumerates field width, signedness, bit order, first-bit
//!   position and stride, including non-byte-aligned widths. A wrong rule mixes one sample's high bits
//!   into the next and comes out near-white, which [`prune`] rejects for the cost of one pass.
//! - **sample rate** — two anchors that fail for unrelated reasons. The optical beats have to line up
//!   with the R peaks BEAT BY BEAT at a wrist transit time ([`ppg::ppg_agreement`]), which a wrong rate
//!   cannot fake — an average heart rate can, which is why average rate is not the claim here. The
//!   power-line interference solves the rate again ([`crate::ecg::mains`]) with no physiology involved
//!   at all. When they disagree the sweep returns [`SweepOutcome::Disagreement`] and stops. It never
//!   averages them: two methods that contradict each other mean an assumption is broken, not that the
//!   truth is in between.
//!
//! Acceptance needs all three of a passing score, a stated margin over the best genuinely different
//! answer, and the same answer in every window — and all three numbers are returned, because a winner
//! that barely edges out a competitor is a coin flip and must not read as success.

pub mod layout;
pub mod ppg;
pub mod prune;
mod search;
mod verdict;

pub use layout::{BitOrder, Layout, LayoutShape};
pub use ppg::{ppg_agreement, suspected_unit_error, PpgAgreement, UnitErrorSuspicion};
pub use prune::{cluster, layout_stats, roughness, same_time_base, LayoutStats};
pub use search::sweep_window;
pub use verdict::{sweep, sweep_split};

use crate::ecg::score::EcgScore;

/// Sample rates searched, in Hz. The converter's rate register is a single byte, so it is an enum code
/// and not a frequency; this list is the set of rates such a code plausibly selects — the binary
/// decimation chain (128/256/512/1024) and the decimal divisions of a 1 kHz base (200/250/400/500/1000).
/// **500 Hz is one entry among nine, not a default.** It is in the list because it is plausible, and it
/// wins only if the data says so.
pub const DEFAULT_RATES_HZ: [f64; 9] = [128.0, 200.0, 250.0, 256.0, 400.0, 500.0, 512.0, 1000.0, 1024.0];

/// Rates that are plausible for the converter but outside the span the detectors support, recorded so
/// their absence from [`DEFAULT_RATES_HZ`] is a stated limit rather than a silent one. Every window in
/// every stage is sized in seconds, and above [`crate::ecg::MAX_FS_HZ`] those windows stop being
/// validated by anything measured here.
pub const UNSEARCHED_RATES_HZ: [f64; 2] = [2048.0, 4096.0];

/// Every threshold the sweep applies. All are operating points measured on one corpus, not constants of
/// nature; see the crate's ECG integration tests for the ranges each was taken from.
#[derive(Clone, Debug, PartialEq)]
pub struct SweepConfig {
    pub rates_hz: Vec<f64>,
    /// Reject a reading rule whose `1 − r₁` exceeds this. DERIVED: real ECG at 128-1024 Hz measures far
    /// below it and a misaligned field sits near 1.0.
    pub max_roughness: f64,
    /// Reject a reading rule below this kurtosis. A ramp, a counter and a sawtooth all sit near 1.8.
    pub min_kurtosis: f64,
    /// Fewest decoded samples a rule must yield to be scored at all.
    pub min_samples: usize,
    /// `|r|` at which two decoded waveforms are the same answer rather than two.
    pub class_min_r: f64,
    /// Detector-agreement excess a candidate must clear, over and above the gate in
    /// [`crate::ecg::score`]. Raw agreement rises with detection density alone, and a wrong layout is
    /// exactly the case that produces dense spurious peaks.
    pub min_b_excess: f64,
    /// Fraction of the scarcer beat series that must match the optical beats one-to-one. Must stay above
    /// a half: a rate wrong by a factor of k still matches exactly 1 beat in k, precisely and with a
    /// perfect linear fit, so a lower gate would let a doubled rate through.
    pub min_ppg_match: f64,
    /// Relative tolerance on the implied-vs-optical heart rate used to prune before scoring.
    pub hr_prune_tolerance: f64,
    /// How far the leader must beat the best genuinely different answer, in quality units.
    pub min_margin: f64,
    /// Relative difference at which the two rate anchors are called a disagreement.
    pub rate_agreement_tolerance: f64,
    /// Relative window for the 1/1024-second unit-error check.
    pub unit_error_tolerance: f64,
    /// Candidates kept per window for display.
    pub top_n: usize,
}

impl Default for SweepConfig {
    fn default() -> Self {
        SweepConfig {
            rates_hz: DEFAULT_RATES_HZ.to_vec(),
            max_roughness: 0.60,
            min_kurtosis: 2.00,
            min_samples: 1024,
            class_min_r: 0.99,
            min_b_excess: 0.40,
            min_ppg_match: 0.70,
            hr_prune_tolerance: 0.20,
            min_margin: 0.10,
            rate_agreement_tolerance: 0.02,
            unit_error_tolerance: 0.01,
            top_n: 8,
        }
    }
}

/// One window's bytes and the optical beat times inside it, in ms from that window's first sample.
#[derive(Clone, Copy, Debug)]
pub struct WindowInput<'a> {
    pub bytes: &'a [u8],
    /// Empty when the optical channel is not available; the sweep then runs on morphology alone and
    /// says so in [`SweepReport::anchors`].
    pub ppg_beats_ms: &'a [f64],
}

/// One scored decode hypothesis.
#[derive(Clone, Debug, PartialEq)]
pub struct Candidate {
    pub layout: Layout,
    /// Which waveform class this rule belongs to. Rules in one class produce the same waveform up to
    /// scale and sign and are not distinguishable by anything in this module.
    pub class: usize,
    /// Other rules in the same class — the answers this one cannot be told apart from.
    pub aliases: usize,
    /// Which ANSWER this candidate is, once decimation is accounted for. A class read at half the rate
    /// with every other sample covers the same seconds and is the same answer with information thrown
    /// away, so it shares this index; only a different `answer` is a rival for the margin.
    pub answer: usize,
    pub fs_hz: f64,
    /// The ranking scalar: the mean of the available evidence terms — the detector-agreement excess, and
    /// the optical-beat agreement excess when there is one. Ranking only. `passes` is the conjunction,
    /// and nothing is accepted on `quality`.
    pub quality: f64,
    pub ecg: EcgScore,
    pub ppg: Option<PpgAgreement>,
    /// The rate the matched beat pairs fit, which need not be `fs_hz`. `None` when nothing paired.
    pub fs_from_ppg: Option<f64>,
    /// Every acceptance condition this candidate meets on its own window.
    pub passes: bool,
}

/// What a stalled sweep can be blamed on. Costs nothing to compute and turns dead waiting into a
/// diagnosis: a live, plausible optical rate with no decode found means the problem is the decode; an
/// absent or erratic optical rate means the problem is contact.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Attribution {
    Decode,
    Contact,
    /// No optical beats were supplied, so the two cannot be separated.
    Unknown,
}

/// One window's search, in the order the acceptance argument is built.
#[derive(Clone, Debug, PartialEq)]
pub struct WindowReport {
    pub bytes: usize,
    pub layouts_enumerated: usize,
    /// Rules that decoded at least [`SweepConfig::min_samples`] samples.
    pub layouts_decoded: usize,
    /// Rules that survived the rate-free prune.
    pub layouts_survived: usize,
    /// Distinct waveforms among the survivors.
    pub classes: usize,
    /// Full scorings actually run — class representatives times rates that got past the cheap gate.
    pub scored: usize,
    /// Best first, truncated to [`SweepConfig::top_n`].
    pub leaderboard: Vec<Candidate>,
    /// Quality gap between the leader and the best candidate of a different class.
    pub margin: f64,
    pub attribution: Attribution,
}

/// The two rate anchors, side by side and never merged.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Anchors {
    /// From the power-line peak, if there is one. Absent on a stream the front end has notched.
    pub mains_fs_hz: Option<f64>,
    pub mains_confidence: Option<f64>,
    /// From the two beat series, if the optical channel was supplied.
    pub ppg_fs_hz: Option<f64>,
    /// `|mains − ppg| / ppg`, when both exist.
    pub relative_difference: Option<f64>,
    /// Set when the optically solved rate sits one 1/1024-second conversion away from a searched rate.
    pub unit_error: Option<UnitErrorSuspicion>,
}

/// The three acceptance conditions, each reported whether or not it is met.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Acceptance {
    /// The leader passes every index in every window.
    pub scored: bool,
    /// It beats the best genuinely different answer by [`SweepConfig::min_margin`].
    pub margin_ok: bool,
    /// It is the same answer in every window.
    pub stable: bool,
    /// The two rate anchors do not contradict each other.
    pub anchors_agree: bool,
}

/// Why the sweep has not converged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StallReason {
    NoWindows,
    /// Nothing survived the rate-free prune — no reading rule produced a waveform at all.
    NoLayoutSurvived,
    /// Waveforms exist but none passed the indices at any searched rate.
    NoCandidatePassed,
    /// A leader exists but sits too close to a genuinely different answer.
    MarginTooSmall,
    /// A leader exists but is not the same answer in every window.
    Unstable,
}

/// The verdict. `Converged` is the only value that means the parameters are known.
#[derive(Clone, Debug, PartialEq)]
pub enum SweepOutcome {
    Converged {
        layout: Layout,
        shape: LayoutShape,
        fs_hz: f64,
    },
    /// The two rate anchors contradict each other. Deliberately terminal: an assumption is broken, and
    /// the midpoint of two contradictory measurements is not a third measurement.
    Disagreement {
        mains_fs_hz: f64,
        ppg_fs_hz: f64,
        relative_difference: f64,
    },
    /// The optically solved rate is one 1/1024-second conversion away from a searched rate. Reported
    /// instead of a result, because this failure produces a plausible number and would otherwise read
    /// as confirmation.
    SuspectedUnitError(UnitErrorSuspicion),
    Searching {
        reason: StallReason,
        best_quality: Option<f64>,
    },
}

/// Everything the sweep decided and everything it decided it on.
#[derive(Clone, Debug, PartialEq)]
pub struct SweepReport {
    pub windows: Vec<WindowReport>,
    /// The leader's best-scoring rule, from the window it scored highest in.
    pub leader: Option<Candidate>,
    /// The best candidate of a genuinely different class, from the same window.
    pub runner_up: Option<Candidate>,
    /// The leader's worst per-window margin over a different class — the honest one, not the best one.
    pub margin: f64,
    pub windows_agreed: usize,
    pub windows_required: usize,
    /// Rules the leader cannot be told apart from, in its own window.
    pub alias_shapes: Vec<LayoutShape>,
    pub conditions: Acceptance,
    pub anchors: Anchors,
    pub attribution: Attribution,
    pub outcome: SweepOutcome,
}

impl SweepReport {
    pub fn converged(&self) -> Option<(Layout, f64)> {
        match self.outcome {
            SweepOutcome::Converged { layout, fs_hz, .. } => Some((layout, fs_hz)),
            _ => None,
        }
    }
}
