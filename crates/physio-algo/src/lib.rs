//! Brand-neutral physiological algorithms — the analytics layer that turns decoded physiological
//! signals into wellness estimates. The core entry points take plain values (R-R runs, PPG samples,
//! accel/motion, per-epoch fields); a few history adapters accept an already-decoded `HistoryRecord`
//! slice — never a wire frame, never BLE. Pure and deterministic: no async, no IO. Absent signal
//! returns `None`, never a fabricated number. Outputs are wellness estimates, never medical advice.
//!
//! Modules are organised by physiological domain. `sleep`, `ppg`, `hrv`, `spo2`, `calibration`,
//! `hr_anomaly` and the shared `stats` are populated; `recovery` and `strain` are stubs ported later.

pub mod calibration;
pub mod hr_anomaly;
pub mod hrv;
pub mod ppg;
pub mod recovery;
pub mod sleep;
pub mod spo2;
pub mod stats;
pub mod strain;

pub use calibration::Calibration;
pub use hr_anomaly::{HrWatch, HrWatchState};
pub use hrv::{HrvReadiness, HrvReadinessResult, ReadinessTier, SECS_PER_DAY};
pub use spo2::{RollingReading, Spo2};
pub use stats::{linear_fit, pearson, LinearFit};
