//! Offline test support for the terminal ECG renderer: a synthetic waveform of known dimensions, and
//! an oracle that measures the scale back out of a rendered frame.
//!
//! The pair exists to catch one failure. A wrong scale draws a correctly-shaped, plausible,
//! wrongly-scaled trace, and nothing about it looks wrong — so the renderer's claim is checked against
//! the picture's own characters rather than against the renderer's own arithmetic. [`oracle`] reaches
//! nothing but `std` and [`decode`], and [`decode`] carries a second, independent braille bit table on
//! purpose. Both rules are enforced by test, not by convention.
//!
//! No BLE, no hardware, no strap: the packet layout and the amplitude scale are still unknown, so every
//! number here is either supplied by the caller or measured from characters.

pub mod decode;
pub mod oracle;
pub mod synth;

#[cfg(test)]
mod tests;
