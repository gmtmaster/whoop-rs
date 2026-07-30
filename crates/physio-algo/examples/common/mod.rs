//! Shared fixture loading for the analysis harnesses, and the refinement guard.
//!
//! Every stream a harness reads has one reader here, `steps.csv` included — the app runs
//! detect -> stage -> refine_wake, and a harness with no step stream stages only the first two.
//! [`RefineCensus`] is the only route to the refinement: it counts which side of the density gate each
//! span fell on, so an unrefined span can never be pooled into a figure labelled refined.

#![allow(dead_code)]

use std::fs;
use std::path::{Path, PathBuf};

use physio_algo::sleep::{
    motion_density, refine_wake, AccelSample, HrSample, RrRun, StageSegment, StepSample, MIN_DENSE_FRACTION,
};

/// The de-duplicated corpus. The raw `fixtures_multi` root holds each beat twice on 16 of its 92 `ours`
/// nights and 6 of its 64 `continuous` blocks, so defaulting to it would hand back a doubled R-R stream
/// with no error. `WHOOP_SLEEP_FIXTURES` overrides.
const DEFAULT_ROOT: &str = "C:/Users/DavidGillot/Projects/whoop/sleep-benchmark/fixtures_multi_clean";

pub fn fixtures_root() -> PathBuf {
    std::env::var("WHOOP_SLEEP_FIXTURES").map(PathBuf::from).unwrap_or_else(|_| PathBuf::from(DEFAULT_ROOT))
}

pub fn root(set: &str) -> PathBuf {
    fixtures_root().join(set)
}

/// The fixture directories of one set, sorted, so two harnesses iterate in the same order.
pub fn dirs_of(set: &str) -> Vec<PathBuf> {
    let mut d: Vec<PathBuf> = fs::read_dir(root(set))
        .map(|rd| rd.filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.is_dir()).collect())
        .unwrap_or_default();
    d.sort();
    d
}

pub fn read_csv(path: &Path) -> Vec<Vec<f64>> {
    fs::read_to_string(path)
        .map(|t| {
            t.lines()
                .filter(|l| !l.trim().is_empty())
                .map(|l| l.split(',').map(|c| c.trim().parse::<f64>().unwrap()).collect())
                .collect()
        })
        .unwrap_or_default()
}

pub fn read_hr(dir: &Path) -> Vec<HrSample> {
    read_csv(&dir.join("hr.csv")).iter().map(|r| HrSample { ts: r[0] as i64, bpm: r[1] as u16 }).collect()
}

pub fn read_accel(dir: &Path) -> Vec<AccelSample> {
    read_csv(&dir.join("gravity.csv"))
        .iter()
        .map(|r| AccelSample { ts: r[0] as i64, x: r[1], y: r[2], z: r[3] })
        .collect()
}

/// Consecutive same-timestamp rows of `rr.csv` grouped into one run, which is how the strap reported them.
pub fn read_rr(dir: &Path) -> Vec<RrRun> {
    let mut rr: Vec<RrRun> = Vec::new();
    for row in read_csv(&dir.join("rr.csv")) {
        let (ts, ms) = (row[0] as i64, row[1] as u16);
        match rr.last_mut() {
            Some(l) if l.ts == ts => l.intervals.push(ms),
            _ => rr.push(RrRun { ts, intervals: vec![ms] }),
        }
    }
    rr
}

/// The strap's own `sleep_state`, 1 Hz, as `(ts, code)`.
pub fn read_band(dir: &Path) -> Vec<(i64, i32)> {
    read_csv(&dir.join("band.csv")).iter().map(|r| (r[0] as i64, r[1] as i32)).collect()
}

/// The strap's own step stream. `-1` in column 3 means the class was not decoded, which is not "still".
/// A missing or empty file returns empty, which is exactly what makes the refinement decline in silence.
pub fn read_steps(dir: &Path) -> Vec<StepSample> {
    read_csv(&dir.join("steps.csv"))
        .iter()
        .map(|r| StepSample { ts: r[0] as i64, counter: r[1] as u16, activity_class: (r[2] >= 0.0).then(|| r[2] as u8) })
        .collect()
}

/// Which side of the refinement's density gate each span fell on, for one reported figure.
///
/// The gate wants a gravity sample twice a minute and a step sample once a minute over 80% of the
/// minutes, and it declines with no error. A figure mixing declined spans with refined ones is neither
/// path, so every harness that refines counts through this and prints [`RefineCensus::line`].
#[derive(Default, Clone, Copy)]
pub struct RefineCensus {
    /// Spans the gate accepted, so the refinement really ran.
    pub refined: usize,
    /// Spans the gate declined: staged by `stage_v2` alone, which is NOT the path the app runs.
    pub declined: usize,
    /// Refined spans whose labels actually moved.
    pub changed: usize,
    /// Of the declines, how many were the GRAVITY stream and how many the STEP stream. A gravity decline
    /// would happen on a device too; a step decline can be the fixture's gap rather than the strap's.
    pub declined_gravity: usize,
    pub declined_steps: usize,
}

impl RefineCensus {
    /// Refine one span's staging, counting the gate's answer. Returns the segments the app would hold.
    pub fn refine(&mut self, segs: &[StageSegment], grav: &[AccelSample], steps: &[StepSample]) -> Vec<StageSegment> {
        let (Some(f), Some(l)) = (segs.first(), segs.last()) else {
            self.declined += 1;
            return segs.to_vec();
        };
        if l.end <= f.start {
            self.declined += 1;
            return segs.to_vec();
        }
        let (g, s) = motion_density(f.start, l.end, grav, steps);
        if g < MIN_DENSE_FRACTION || s < MIN_DENSE_FRACTION {
            self.declined += 1;
            self.declined_gravity += usize::from(g < MIN_DENSE_FRACTION);
            self.declined_steps += usize::from(s < MIN_DENSE_FRACTION);
            return segs.to_vec();
        }
        self.refined += 1;
        let out = refine_wake(segs, grav, steps);
        self.changed += usize::from(out != segs);
        out
    }

    /// Fold another census in, so a per-span probe can be accumulated into a per-figure total.
    pub fn absorb(&mut self, other: &RefineCensus) {
        self.refined += other.refined;
        self.declined += other.declined;
        self.changed += other.changed;
        self.declined_gravity += other.declined_gravity;
        self.declined_steps += other.declined_steps;
    }

    pub fn total(&self) -> usize {
        self.refined + self.declined
    }

    /// True when every span counted went through the refinement, so the figure beside it is the app's path.
    pub fn all_refined(&self) -> bool {
        self.declined == 0 && self.refined > 0
    }

    /// One line naming the split. Print it beside any figure this census fed.
    pub fn line(&self, what: &str) -> String {
        let tail = if self.declined == 0 {
            " — refined throughout".to_string()
        } else {
            format!(
                " ({} on gravity, {} on steps) — MIXED, so this figure is neither path alone",
                self.declined_gravity, self.declined_steps
            )
        };
        format!(
            "   {what}: {} of {} spans REFINED ({} moved), {} DECLINED by the density gate{tail}",
            self.refined,
            self.total(),
            self.changed,
            self.declined,
        )
    }

    /// Stop a mixed figure being reported as refined. Use where the harness has already restricted itself
    /// to spans carrying a dense stream, so a decline means the restriction is wrong.
    pub fn require_all_refined(&self, what: &str) {
        assert!(
            self.all_refined(),
            "{what}: {} of {} spans were declined by the density gate, so this figure is a mixture",
            self.declined,
            self.total()
        );
    }
}
