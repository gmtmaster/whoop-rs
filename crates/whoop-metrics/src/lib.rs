//! Derived wellness metrics over decoded records — the analytics layer above the codec. Pure,
//! deterministic, no BLE, no IO. Every metric returns `None` rather than a fabricated number when the
//! signal is absent. Outputs are wellness estimates, never medical.

pub mod calibration;
pub mod hr_anomaly;
pub mod hrv_readiness;
pub mod spo2;
pub mod stats;

pub use calibration::Calibration;
pub use hr_anomaly::{HrWatch, HrWatchState};
pub use hrv_readiness::{HrvReadiness, HrvReadinessResult, ReadinessTier, SECS_PER_DAY};
pub use spo2::{RollingReading, Spo2};
pub use stats::{linear_fit, pearson, LinearFit};
