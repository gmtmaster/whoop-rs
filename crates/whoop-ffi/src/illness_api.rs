//! Illness-watch baseline policy over the FFI: the window/gap/trust numbers and the z-score itself,
//! so the daily banner and the heads-up suite read one policy instead of each carrying a window.

use physio_algo::illness;

/// The trailing-baseline policy every illness signal is scored against.
#[derive(uniffi::Record)]
pub struct IllnessBaselineCfgInfo {
    /// Recent nights excluded from the baseline, counted back from the scored day.
    pub gap_nights: u32,
    /// Nights the baseline averages, ending `gap_nights` before the scored day.
    pub window_nights: u32,
    /// Usable nights the window needs before a reading is trusted.
    pub min_nights: u32,
}

/// The one baseline policy, read by every illness path so none can hold a second copy of it.
#[uniffi::export]
pub fn illness_baseline_cfg() -> IllnessBaselineCfgInfo {
    IllnessBaselineCfgInfo {
        gap_nights: illness::BASELINE_GAP_NIGHTS as u32,
        window_nights: illness::BASELINE_WINDOW_NIGHTS as u32,
        min_nights: illness::MIN_BASELINE_NIGHTS as u32,
    }
}

/// Per-night z of a chronological (oldest first) daily signal against its own trailing baseline.
/// A night with no value, or too few usable nights behind it, comes back `None`.
#[uniffi::export]
pub fn illness_baseline_z_series(values: Vec<Option<f64>>) -> Vec<Option<f64>> {
    illness::baseline_z_series(&values)
}
