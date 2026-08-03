//! Library face of the CLI: the offline drawing and measuring halves, reachable from tests and from
//! the `ecg` subcommand without linking a radio.
//!
//! [`braille`] draws — a dot canvas and layer composition. [`ecg_render`] turns samples into a scaled
//! ECG strip frame on top of it. [`ecg_oracle`] measures a drawn frame back and carries its own
//! braille decoder on purpose: an oracle that shared the drawing side's bit table could not catch a
//! wrong one. Keep them independent.

pub mod braille;
pub mod ecg_oracle;
pub mod ecg_render;
