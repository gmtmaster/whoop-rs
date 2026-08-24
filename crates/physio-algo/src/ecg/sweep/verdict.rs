//! Folding consecutive windows into a verdict: which answer led, by how much, in how many windows, and
//! what the two rate anchors say about the rate it claims.
//!
//! Nothing here re-scores anything — that is [`super::search`]. This is only the argument built on top of
//! the per-window leaderboards, and it is deliberately the part that says no: a contradiction between the
//! anchors, a coincidence that would read as confirmation, a margin too thin to be a result, or an answer
//! that did not survive every window each stop the sweep on its own.

use crate::ecg::MainsConfig;
use crate::ecg::mains;
use crate::ecg::sweep::layout::LayoutShape;
use crate::ecg::sweep::ppg::suspected_unit_error;
use crate::ecg::sweep::search::{attribution, sweep_window};
use crate::ecg::sweep::{
    Acceptance, Anchors, Attribution, Candidate, StallReason, SweepConfig, SweepOutcome,
    SweepReport, WindowInput, WindowReport,
};

/// Search consecutive windows and decide. Convergence needs all three acceptance conditions and both
/// rate anchors to be consistent; every one of those numbers is in the report whether it was met or not.
pub fn sweep(windows: &[WindowInput], cfg: &SweepConfig) -> SweepReport {
    let reports: Vec<WindowReport> = windows
        .iter()
        .map(|w| sweep_window(w.bytes, w.ppg_beats_ms, cfg))
        .collect();
    let attribution = reports
        .first()
        .map_or(Attribution::Unknown, |r| r.attribution);
    if reports.is_empty() {
        return stalled(reports, StallReason::NoWindows, attribution);
    }
    if reports.iter().all(|r| r.layouts_survived == 0) {
        return stalled(reports, StallReason::NoLayoutSurvived, attribution);
    }

    // A window's class numbering is its own, so the answer is keyed on the reading rule's SHAPE and the
    // rate. Where the first whole sample begins is a property of where the buffer was cut, not of the
    // stream, and requiring it to repeat would make an arbitrary cut look like instability.
    let mut keys: Vec<(LayoutShape, u64)> = Vec::new();
    for r in &reports {
        // Only what each window itself put on top. Every passing candidate would put a decimation of the
        // leader into the running as if it were a separate contender for the answer.
        if let Some(c) = r.leaderboard.iter().find(|c| c.passes) {
            let k = (c.layout.shape(), c.fs_hz.to_bits());
            if !keys.contains(&k) {
                keys.push(k);
            }
        }
    }
    let best_key = keys
        .iter()
        .map(|k| {
            let hits: Vec<&Candidate> = reports.iter().filter_map(|r| find(r, k)).collect();
            let passed = hits.iter().filter(|c| c.passes).count();
            let mean = hits.iter().map(|c| c.quality).sum::<f64>() / hits.len().max(1) as f64;
            (*k, passed, mean)
        })
        // Most windows passed, then best mean quality, then the densest reading: within one answer the
        // decimated views score identically, and the one that discards nothing is the one to report.
        .max_by(|a, b| {
            a.1.cmp(&b.1)
                .then_with(|| a.2.total_cmp(&b.2))
                .then_with(|| f64::from_bits(a.0.1).total_cmp(&f64::from_bits(b.0.1)))
        });

    let Some((key, agreed, _)) = best_key else {
        let best = reports
            .iter()
            .filter_map(|r| r.leaderboard.first())
            .map(|c| c.quality)
            .fold(f64::NAN, f64::max);
        let mut out = stalled(reports, StallReason::NoCandidatePassed, attribution);
        out.outcome = SweepOutcome::Searching {
            reason: StallReason::NoCandidatePassed,
            best_quality: best.is_finite().then_some(best),
        };
        return out;
    };

    // The leader is reported from the window it did best in; the margin from the window it did WORST in.
    let leader_window = reports
        .iter()
        .enumerate()
        .filter_map(|(i, r)| find(r, &key).map(|c| (i, c)))
        .max_by(|a, b| a.1.quality.total_cmp(&b.1.quality));
    let (best_idx, leader) = match leader_window {
        Some((i, c)) => (i, c.clone()),
        None => return stalled(reports, StallReason::NoCandidatePassed, attribution),
    };
    let runner_up = other_answer(&reports[best_idx], &leader).cloned();
    let margin = reports
        .iter()
        .filter_map(|r| {
            let c = find(r, &key)?;
            Some(match other_answer(r, c) {
                Some(o) => c.quality - o.quality,
                None => f64::INFINITY,
            })
        })
        .fold(f64::INFINITY, f64::min);

    let alias_shapes = alias_shapes(&reports[best_idx], &leader);
    let anchors = anchors(windows[best_idx].bytes, &leader, cfg);
    let stable = reports.len() >= 2 && agreed == reports.len();
    let anchors_agree = anchors
        .relative_difference
        .is_none_or(|d| d <= cfg.rate_agreement_tolerance);
    let conditions = Acceptance {
        scored: leader.passes && agreed == reports.len(),
        margin_ok: margin >= cfg.min_margin,
        stable,
        anchors_agree,
    };

    let outcome = decide(&conditions, &anchors, &leader);
    SweepReport {
        windows: reports,
        leader: Some(leader),
        runner_up,
        margin,
        windows_agreed: agreed,
        windows_required: windows.len(),
        alias_shapes,
        conditions,
        anchors,
        attribution,
        outcome,
    }
}

/// Split one buffer into `windows` consecutive equal parts and sweep them. `capture_ms` is the whole
/// buffer's duration, used only to cut the optical beat list at the same fractions the bytes are cut at;
/// a constant-rate stream spends equal time in equal byte spans, so the two cuts coincide.
pub fn sweep_split(
    bytes: &[u8],
    ppg_beats_ms: &[f64],
    capture_ms: f64,
    windows: usize,
    cfg: &SweepConfig,
) -> SweepReport {
    let n = windows.max(1);
    let chunk = bytes.len() / n;
    if chunk == 0 {
        return stalled(
            Vec::new(),
            StallReason::NoWindows,
            attribution(ppg_beats_ms),
        );
    }
    let beats: Vec<Vec<f64>> = (0..n)
        .map(|k| {
            let (t0, t1) = (
                capture_ms * k as f64 / n as f64,
                capture_ms * (k + 1) as f64 / n as f64,
            );
            ppg_beats_ms
                .iter()
                .filter(|t| **t >= t0 && **t < t1)
                .map(|t| t - t0)
                .collect()
        })
        .collect();
    let inputs: Vec<WindowInput> = (0..n)
        .map(|k| WindowInput {
            bytes: &bytes[k * chunk..(k + 1) * chunk],
            ppg_beats_ms: &beats[k],
        })
        .collect();
    sweep(&inputs, cfg)
}

fn find<'a>(r: &'a WindowReport, key: &(LayoutShape, u64)) -> Option<&'a Candidate> {
    r.leaderboard
        .iter()
        .find(|c| c.layout.shape() == key.0 && c.fs_hz.to_bits() == key.1)
}

/// The best candidate in this window that is a genuinely different answer — a different waveform, or a
/// rate that is not an exact decimation of this one.
fn other_answer<'a>(r: &'a WindowReport, c: &Candidate) -> Option<&'a Candidate> {
    r.leaderboard.iter().find(|o| o.answer != c.answer)
}

fn alias_shapes(r: &WindowReport, leader: &Candidate) -> Vec<LayoutShape> {
    let mut out: Vec<LayoutShape> = r
        .leaderboard
        .iter()
        .filter(|c| c.class == leader.class && c.layout.shape() != leader.layout.shape())
        .map(|c| c.layout.shape())
        .collect();
    out.sort_unstable();
    out.dedup();
    out
}

fn anchors(bytes: &[u8], leader: &Candidate, cfg: &SweepConfig) -> Anchors {
    let x = leader.layout.decode(bytes);
    let fs_max = cfg
        .rates_hz
        .iter()
        .copied()
        .fold(0.0f64, f64::max)
        .max(crate::ecg::MAX_FS_HZ);
    let m = mains::mains_anchor_with(
        &x,
        MainsConfig {
            fs_max_hz: fs_max,
            ..MainsConfig::default()
        },
    );
    let fix = m.fix();
    let ppg_fs = leader.fs_from_ppg;
    let diff = match (fix.map(|f| f.fs_hz), ppg_fs) {
        (Some(a), Some(b)) if b > 0.0 => Some((a - b).abs() / b),
        _ => None,
    };
    Anchors {
        mains_fs_hz: fix.map(|f| f.fs_hz),
        mains_confidence: fix.map(|f| f.confidence),
        ppg_fs_hz: ppg_fs,
        relative_difference: diff,
        unit_error: ppg_fs
            .and_then(|f| suspected_unit_error(f, &cfg.rates_hz, cfg.unit_error_tolerance)),
    }
}

/// The verdict, in the order the failures have to be reported: a contradiction between the two anchors
/// outranks everything, then the unit-error coincidence, then the three acceptance conditions.
fn decide(c: &Acceptance, a: &Anchors, leader: &Candidate) -> SweepOutcome {
    if let (Some(m), Some(p), Some(d)) = (a.mains_fs_hz, a.ppg_fs_hz, a.relative_difference) {
        if !c.anchors_agree {
            return SweepOutcome::Disagreement {
                mains_fs_hz: m,
                ppg_fs_hz: p,
                relative_difference: d,
            };
        }
    }
    // The 1/1024 coincidence is only terminal when there is no second opinion. When the power-line
    // anchor exists and agrees, two independent methods have already ruled the scaling error out; this
    // is the case that anchor earns its keep in.
    if let Some(u) = a.unit_error {
        if a.mains_fs_hz.is_none() {
            return SweepOutcome::SuspectedUnitError(u);
        }
    }
    let reason = if !c.scored {
        Some(StallReason::NoCandidatePassed)
    } else if !c.margin_ok {
        Some(StallReason::MarginTooSmall)
    } else if !c.stable {
        Some(StallReason::Unstable)
    } else {
        None
    };
    match reason {
        Some(reason) => SweepOutcome::Searching {
            reason,
            best_quality: Some(leader.quality),
        },
        None => SweepOutcome::Converged {
            layout: leader.layout,
            shape: leader.layout.shape(),
            fs_hz: leader.fs_hz,
        },
    }
}

fn stalled(
    windows: Vec<WindowReport>,
    reason: StallReason,
    attribution: Attribution,
) -> SweepReport {
    SweepReport {
        windows,
        leader: None,
        runner_up: None,
        margin: 0.0,
        windows_agreed: 0,
        windows_required: 0,
        alias_shapes: Vec::new(),
        conditions: Acceptance {
            scored: false,
            margin_ok: false,
            stable: false,
            anchors_agree: true,
        },
        anchors: Anchors::default(),
        attribution,
        outcome: SweepOutcome::Searching {
            reason,
            best_quality: None,
        },
    }
}
