//! Thin compatibility shim. The derived-metric implementations now live in `physio-algo`; this crate
//! re-exports them so existing `whoop_metrics::…` paths keep resolving. New code should depend on
//! `physio-algo` directly.

pub use physio_algo::{calibration, hr_anomaly, hrv as hrv_readiness, ppg as ppg_hr, spo2, stats};

pub use physio_algo::{
    linear_fit, pearson, Calibration, HrWatch, HrWatchState, HrvReadiness, HrvReadinessResult, LinearFit,
    ReadinessTier, RollingReading, Spo2, SECS_PER_DAY,
};
