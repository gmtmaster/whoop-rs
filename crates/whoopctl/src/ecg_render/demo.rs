//! The offline demo: a synthetic waveform of KNOWN dimensions, streamed through the renderer at
//! wall-clock speed. No BLE, no strap — the R16 packet layout is not known, so there is nothing to
//! decode yet and guessing one would mean rewriting it.
//!
//! The generator emits millivolts, so the demo has to encode them as counts. [`FIXTURE_COUNTS_PER_MV`]
//! is a property of the FIXTURE, not of any device: pass it back in with `--counts-per-mv` to see the
//! calibrated axis, leave it out to see the uncalibrated one the real strap will give.

use std::io::Write;
use std::time::{Duration, Instant};

use crate::ecg_oracle::synth;

use super::driver::{self, Painter};
use super::plan::{fit, FitError, Plan, Request, Terminal};
use super::renderer::{EcgRenderer, Sample};

/// The synthetic fixture's own counts-per-mV. NOT a device constant — the strap's conversion has not
/// been read, and nothing in the renderer defaults to this or to any other value.
pub const FIXTURE_COUNTS_PER_MV: f64 = 1000.0;

/// Which waveform the demo streams.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DemoSignal {
    /// The 1 mV / 1 Hz calibration pulse: 10 mm tall, 25 mm per cycle at the true scale.
    Pulse,
    /// A synthetic PQRST at 62 bpm.
    Ecg,
}

impl DemoSignal {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "pulse" | "square" => Some(DemoSignal::Pulse),
            "ecg" | "pqrst" => Some(DemoSignal::Ecg),
            _ => None,
        }
    }

    fn label(self) -> &'static str {
        match self {
            DemoSignal::Pulse => "1 mV / 1 Hz calibration pulse",
            DemoSignal::Ecg => "synthetic PQRST, 62 bpm, 1.2 mV R",
        }
    }
}

/// Build the demo samples. Lead-off, when asked for, replaces the trace with noise for 0.8 s in every
/// 4 s and flags those samples — which is what makes the marked-not-interpolated rule testable.
pub fn samples(signal: DemoSignal, rate_hz: f64, duration_s: f64, lead_off: bool) -> Vec<Sample> {
    let base = match signal {
        DemoSignal::Pulse => synth::calibration_pulse(rate_hz, duration_s),
        DemoSignal::Ecg => synth::pqrst(rate_hz, duration_s, 62.0, 1.2),
    };
    if !lead_off {
        return base.samples_mv.iter().map(|mv| Sample::new(mv * FIXTURE_COUNTS_PER_MV, true)).collect();
    }
    let inter = synth::intermittent_contact(&base, 3.2, 0.8, 0.35, 0xE0C);
    let mut out: Vec<Sample> = inter
        .signal
        .samples_mv
        .iter()
        .map(|mv| Sample::new(mv * FIXTURE_COUNTS_PER_MV, true))
        .collect();
    for span in inter.spans.iter().filter(|s| !s.contact) {
        for s in out.iter_mut().take(span.end_sample).skip(span.start_sample) {
            s.contact = false;
        }
    }
    out
}

/// Everything the demo needs that the plan does not carry.
pub struct DemoOptions {
    pub signal: DemoSignal,
    pub lead_off: bool,
    /// Wall-clock multiplier. 1.0 streams in real time; 0.0 renders as fast as it can.
    pub speed: f64,
    pub colour: bool,
}

/// Fit, stream and paint. Returns the plan actually used so the caller can report it.
pub fn run<W: Write>(req: &Request, term: Terminal, opts: &DemoOptions, out: &mut W) -> Result<Plan, FitError> {
    let plan = fit(req, term)?;
    let data = samples(opts.signal, plan.sample_rate_hz, plan.duration_s, opts.lead_off);

    writeln!(
        out,
        "demo: {} encoded at {FIXTURE_COUNTS_PER_MV:.0} counts/mV (FIXTURE, not a device constant){}",
        opts.signal.label(),
        if opts.lead_off { " · lead-off injected 0.8 s in every 4 s" } else { "" },
    )
    .ok();

    let is_tty = driver::stdout_is_terminal();
    let (mode, why) = driver::choose_mode(term, super::frame::frame_rows(&plan), is_tty);
    let mut painter = Painter::new(mode, why, opts.colour);
    let mut renderer = EcgRenderer::new(plan);

    // One redraw per chunk, not per sample: the canvases are drawn into, only the frame is re-read.
    let chunk = ((plan.sample_rate_hz / REDRAW_HZ).round().max(1.0)) as usize;
    let started = Instant::now();
    for (i, block) in data.chunks(chunk).enumerate() {
        renderer.push(block);
        if opts.speed > 0.0 {
            let due = Duration::from_secs_f64((i + 1) as f64 * chunk as f64 / plan.sample_rate_hz / opts.speed);
            if let Some(wait) = due.checked_sub(started.elapsed()) {
                std::thread::sleep(wait);
            }
        }
        painter.paint(out, &renderer).ok();
    }
    renderer.finish();
    painter.finish(out, &renderer).ok();
    Ok(plan)
}

/// Redraws a second. Fast enough to look live, slow enough that the frame is not the cost.
const REDRAW_HZ: f64 = 20.0;
