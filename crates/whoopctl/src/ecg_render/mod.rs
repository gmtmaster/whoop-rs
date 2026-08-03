//! The terminal ECG strip renderer: samples in, a braille frame out, at a scale that is TRUE and
//! STATED, or not drawn at all.
//!
//! A wrong scale draws a correctly-shaped, plausible, wrongly-scaled trace and nothing about it looks
//! wrong, so three rules run through every file here:
//!
//! - the scale never changes silently. A terminal too narrow for the asked-for mm/s gets a SHORTER
//!   strip or a refusal, and either way the frame says which;
//! - the amplitude axis is UNCALIBRATED until a counts-per-mV is supplied. There is no default guess,
//!   and while it is uncalibrated the banner and the footer both say so, on every render;
//! - anything nobody has read off the device — the sample rate, the terminal's cell aspect, the
//!   terminal size — is printed with an ASSUMED tag beside the number, in the default output.
//!
//! Layout is derived, never hardcoded: 30 s at a 5 s strip is six strips because 30/5 is six.
//! Drawing sits on [`crate::braille`]; [`crate::ecg_oracle`] measures the result back out of the
//! characters and is what proves the scale rather than restating it.

pub mod demo;
pub mod driver;
pub mod frame;
pub mod grid;
pub mod plan;
pub mod renderer;
pub mod vertical;

#[cfg(test)]
mod tests;

pub use driver::{OutputMode, Painter};
pub use plan::{fit, AmplitudePlan, FitError, FitNote, Geometry, Plan, Provenance, Request, Terminal};
pub use renderer::{EcgRenderer, LeadOffSpan, Report, Sample};
pub use vertical::{VerticalBasis, VerticalMap};
