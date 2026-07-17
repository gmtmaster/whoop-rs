//! Brand-neutral physiological algorithms — the analytics layer that turns decoded physiological
//! signals into wellness estimates. Protocol-free by design: every entry point takes plain values
//! (R-R runs, PPG samples, accel/motion, per-epoch fields), never a wire frame or a device record.
//! Pure and deterministic — no BLE, no async, no IO. Absent signal returns `None`, never a fabricated
//! number. Outputs are wellness estimates, never medical advice.
//!
//! Modules are organised by physiological domain. `sleep` is populated first; the remaining domains are
//! stubs ported in later batches.

pub mod calibration;
pub mod hrv;
pub mod ppg;
pub mod recovery;
pub mod sleep;
pub mod spo2;
pub mod strain;
