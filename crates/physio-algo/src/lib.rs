//! Brand-neutral physiological algorithms — the analytics layer that turns decoded physiological
//! signals into wellness estimates. The core entry points take plain values (R-R runs, PPG samples,
//! accel/motion, per-epoch fields); a few history adapters accept an already-decoded `HistoryRecord`
//! slice — never a wire frame, never BLE. Pure and deterministic: no async, no IO. Absent signal
//! returns `None`, never a fabricated number. Outputs are wellness estimates, never medical advice.
//!
//! Modules are organised by physiological domain: `sleep`, `ppg`, `hrv`, `spo2`, `calibration`,
//! `hr_anomaly`, `recovery`, `strain`, `stress`, `resting_hr`, `hr_zones`, `vo2max`,
//! `respiratory_rate`, `imu_features` and the shared `stats`.

pub mod baselines;
pub mod calibration;
pub mod hr_anomaly;
pub mod hr_zones;
pub mod hrv;
pub mod imu_features;
pub mod ppg;
pub mod recovery;
pub mod respiratory_rate;
pub mod rest;
pub mod resting_hr;
pub mod sleep;
pub mod sleep_debt;
pub mod spo2;
pub mod stats;
pub mod strain;
pub mod stress;
pub mod stress_onset;
pub mod vo2max;

pub use calibration::Calibration;
pub use hr_anomaly::{HrWatch, HrWatchState};
pub use hrv::{HrvReadiness, HrvReadinessResult, ReadinessTier, SECS_PER_DAY};
pub use spo2::{RollingReading, Spo2};
pub use stats::{linear_fit, pearson, LinearFit};
