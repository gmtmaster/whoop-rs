//! Illness-watch baseline policy: the one trailing window every illness signal is compared against,
//! and the z-score over it. Consumed by the daily banner and the heads-up suite so neither can hold
//! its own window.

use crate::stats;

/// Nights between the scored day and the end of its baseline window. A multi-day illness would
/// otherwise sit inside the mean it is measured against and damp its own deviation.
pub const BASELINE_GAP_NIGHTS: usize = 3;

/// Trailing nights the baseline averages, ending [`BASELINE_GAP_NIGHTS`] before the scored day.
pub const BASELINE_WINDOW_NIGHTS: usize = 30;

/// Usable nights the window must hold before a z is trusted; below it the signal is still learning.
pub const MIN_BASELINE_NIGHTS: usize = 14;

/// Spread floor in signal units, so a near-constant window cannot divide by ~0 and explode the z.
pub const SD_FLOOR: f64 = 1e-6;

/// Half-open index range of the baseline window for day `i` of a chronological series: the window
/// ends [`BASELINE_GAP_NIGHTS`] before `i` and runs back [`BASELINE_WINDOW_NIGHTS`] nights.
pub fn baseline_window(i: usize) -> std::ops::Range<usize> {
    let end = i.saturating_sub(BASELINE_GAP_NIGHTS);
    end.saturating_sub(BASELINE_WINDOW_NIGHTS)..end
}

/// z of `values[i]` against its baseline window's mean and sample SD (n − 1). `None` when the day
/// has no value or the window holds fewer than [`MIN_BASELINE_NIGHTS`] usable nights.
pub fn baseline_z_at(values: &[Option<f64>], i: usize) -> Option<f64> {
    let value = *values.get(i)?.as_ref()?;
    let r = baseline_window(i);
    let xs: Vec<f64> = values[r.start..r.end.min(values.len())]
        .iter()
        .flatten()
        .copied()
        .collect();
    if xs.len() < MIN_BASELINE_NIGHTS {
        return None;
    }
    let sd = stats::sample_sd(&xs).max(SD_FLOOR);
    Some((value - stats::mean(&xs)) / sd)
}

/// [`baseline_z_at`] for every day of a chronological (oldest first) series.
pub fn baseline_z_series(values: &[Option<f64>]) -> Vec<Option<f64>> {
    (0..values.len())
        .map(|i| baseline_z_at(values, i))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The gap is the whole point: today and the two nights before it must not be in the window.
    #[test]
    fn window_excludes_the_gap_nights() {
        let r = baseline_window(46);
        assert_eq!(r, 13..43);
        assert_eq!(r.len(), BASELINE_WINDOW_NIGHTS);
        assert!(!r.contains(&43) && !r.contains(&45) && !r.contains(&46));
    }

    /// Short histories clamp instead of panicking, and stay below the trust gate.
    #[test]
    fn short_history_yields_no_z() {
        let vs: Vec<Option<f64>> = (0..16).map(|i| Some(50.0 + i as f64)).collect();
        assert_eq!(baseline_window(2), 0..0);
        assert!(baseline_z_at(&vs, 15).is_none()); // window is 0..12, only 12 usable nights
        assert!(baseline_z_at(&vs, 0).is_none());
    }

    /// The defect the gap fixes: a run of ill nights entering its own baseline damps the z below the
    /// 2.0 the engines fire at. Same series, same day, gapped 2.33 vs gapless 1.71.
    #[test]
    fn multi_day_rise_clears_the_fire_threshold_only_with_the_gap() {
        // 52 calm nights (55 ± 1 alternating), then an 8-night rise to 62.
        let mut vs: Vec<Option<f64>> = (0..52)
            .map(|i| Some(if i % 2 == 0 { 54.0 } else { 56.0 }))
            .collect();
        vs.extend((0..8).map(|_| Some(62.0)));
        let i = vs.len() - 1;
        let z = baseline_z_at(&vs, i).unwrap();
        assert!(z > 2.0, "gapped z was {z}");

        // Same day against the gapless window: three more ill nights in the baseline, and it goes quiet.
        let gapless: Vec<f64> = vs[i - BASELINE_WINDOW_NIGHTS..i]
            .iter()
            .flatten()
            .copied()
            .collect();
        let z0 = (62.0 - stats::mean(&gapless)) / stats::sample_sd(&gapless).max(SD_FLOOR);
        assert!(z0 < 2.0, "gapless z was {z0}");
    }

    /// Missing nights are skipped, never zero-filled, and the trust gate counts usable ones only.
    #[test]
    fn missing_nights_are_skipped_not_zero_filled() {
        let mut vs: Vec<Option<f64>> = (0..48)
            .map(|i| Some(if i % 2 == 0 { 54.0 } else { 56.0 }))
            .collect();
        let i = vs.len() - 1;
        vs[i] = Some(60.0);
        let full = baseline_z_at(&vs, i).unwrap();
        for k in baseline_window(i).take(10) {
            vs[k] = None;
        }
        let holey = baseline_z_at(&vs, i).unwrap();
        assert!((full - holey).abs() < 1.0);
        // Below the gate once too few usable nights remain.
        for k in baseline_window(i).take(20) {
            vs[k] = None;
        }
        assert!(baseline_z_at(&vs, i).is_none());
    }

    #[test]
    fn flat_window_is_floored_not_infinite() {
        let mut vs: Vec<Option<f64>> = (0..48).map(|_| Some(55.0)).collect();
        let i = vs.len() - 1;
        vs[i] = Some(55.0);
        assert_eq!(baseline_z_at(&vs, i), Some(0.0));
        assert_eq!(baseline_z_series(&vs).len(), 48);
    }
}
