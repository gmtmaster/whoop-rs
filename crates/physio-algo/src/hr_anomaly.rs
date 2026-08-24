//! Wellness HR watch: a sustained heart rate out of the wearer's personal at-rest band, from the decoded
//! per-second HR. A fitness-tracker-style nudge, never a diagnosis and never real-time (whoop-rs reads
//! offloaded history). It flags only ELEVATED-at-rest; low HR is deliberately not flagged (resting
//! athletes sit low, and there is no arrest detector on this hardware). Wellness only, never medical.

use crate::stats::percentile;
use crate::worn::worn_state;
use whoop_protocol::HistoryRecord;

/// Eligible at-rest samples needed before a personal baseline (and any nudge) is trustworthy.
const MIN_BASELINE_SAMPLES: usize = 600;
/// An elevated run must hold this many seconds of contiguous eligible records to count.
const SUSTAIN_S: u32 = 300;
/// Records more than this many seconds apart break a run (an offload gap can't stitch two periods).
const MAX_GAP_S: u32 = 5;
/// Elevated = HR at or above personal resting HR plus this margin, but never below the absolute floor.
const ELEV_MARGIN: u8 = 45;
const HIGH_ABS: u8 = 100;
/// Minimum PPG `signal_quality` for a record to be trusted (the "good" band boundary).
const QUAL_MIN: u8 = 192;
/// Personal resting HR = this low percentile of eligible at-rest HR.
const RESTING_PCT: f64 = 0.10;

/// The watch result over one drain's records.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum HrWatchState {
    /// Not enough trustworthy at-rest samples to form a personal baseline yet.
    Calibrating { have: usize, need: usize },
    /// Assessed, no sustained elevated-at-rest run.
    Normal,
    /// A sustained run of elevated HR while at rest — a wellness nudge, not a diagnosis.
    ElevatedAtRest {
        peak_bpm: u8,
        start_unix: u32,
        dur_s: u32,
    },
}

pub struct HrWatch;

impl HrWatch {
    /// Evaluate the watch over history records: keep only at-rest, good-signal, on-wrist records with a
    /// heart rate, derive a personal resting HR, then report the first sustained elevated-at-rest run.
    pub fn evaluate(history: &[HistoryRecord]) -> HrWatchState {
        let mut rest: Vec<(u32, u8)> = history.iter().filter_map(eligible).collect();
        if rest.len() < MIN_BASELINE_SAMPLES {
            return HrWatchState::Calibrating {
                have: rest.len(),
                need: MIN_BASELINE_SAMPLES,
            };
        }
        rest.sort_by_key(|&(t, _)| t);

        let mut hrs: Vec<f64> = rest.iter().map(|&(_, hr)| hr as f64).collect();
        hrs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let resting = percentile(&hrs, RESTING_PCT).round() as u8;
        let threshold = resting.saturating_add(ELEV_MARGIN).max(HIGH_ABS);

        // A contiguous elevated run, tracked as (start, last, peak); a below-threshold record or a gap ends it.
        let mut run: Option<(u32, u32, u8)> = None;
        let mut prev = 0u32;
        for &(t, hr) in &rest {
            if hr >= threshold {
                run = match run {
                    Some((s, _, p)) if t.saturating_sub(prev) <= MAX_GAP_S => {
                        Some((s, t, p.max(hr)))
                    }
                    _ => Some((t, t, hr)),
                };
                if let Some((s, l, p)) = run {
                    if l.saturating_sub(s) >= SUSTAIN_S {
                        return HrWatchState::ElevatedAtRest {
                            peak_bpm: p,
                            start_unix: s,
                            dur_s: l - s,
                        };
                    }
                }
            } else {
                run = None;
            }
            prev = t;
        }
        HrWatchState::Normal
    }
}

/// A record is eligible if it is at rest (still or asleep), not proven off-wrist, has a trustworthy PPG
/// signal, and carries a non-zero heart rate. Wear comes from [`worn_state`], never from the off-wrist
/// bit alone — that bit reads clear on real off-wrist records.
fn eligible(h: &HistoryRecord) -> Option<(u32, u8)> {
    let at_rest = h.activity_class == Some(0) || h.sleep_state == Some(2);
    // The band's own per-second verdict on its beat detection, alongside the confidence floor.
    let good =
        h.signal_quality.is_some_and(|q| q >= QUAL_MIN) && h.optical_signal_poor != Some(true);
    let on_wrist = !worn_state(h).is_off();
    let hr = h.heart_rate?;
    (at_rest && good && on_wrist && hr > 0).then_some((h.unix, hr))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(unix: u32, hr: u8) -> HistoryRecord {
        HistoryRecord {
            version: 18,
            unix,
            heart_rate: Some(hr),
            activity_class: Some(0),
            signal_quality: Some(255),
            signal_flags: Some(0),
            ..Default::default()
        }
    }

    #[test]
    fn below_min_samples_is_calibrating() {
        let h: Vec<_> = (0..100).map(|i| rec(i, 60)).collect();
        assert_eq!(
            HrWatch::evaluate(&h),
            HrWatchState::Calibrating {
                have: 100,
                need: MIN_BASELINE_SAMPLES
            }
        );
    }

    #[test]
    fn sustained_elevated_at_rest_fires() {
        let mut h: Vec<_> = (0..600).map(|i| rec(i, 60)).collect();
        h.extend((600..1000).map(|i| rec(i, 120)));
        match HrWatch::evaluate(&h) {
            HrWatchState::ElevatedAtRest {
                peak_bpm, dur_s, ..
            } => {
                assert_eq!(peak_bpm, 120);
                assert!(dur_s >= SUSTAIN_S);
            }
            other => panic!("expected elevated, got {other:?}"),
        }
    }

    #[test]
    fn a_brief_spike_does_not_fire() {
        let mut h: Vec<_> = (0..600).map(|i| rec(i, 60)).collect();
        h.extend((600..610).map(|i| rec(i, 120))); // 10 s only
        h.extend((610..1000).map(|i| rec(i, 60)));
        assert_eq!(HrWatch::evaluate(&h), HrWatchState::Normal);
    }

    #[test]
    fn an_off_wrist_run_is_never_eligible_though_its_off_wrist_bit_is_clear() {
        let mut h: Vec<_> = (0..600).map(|i| rec(i, 60)).collect();
        h.extend((600..1000).map(|i| {
            let mut r = rec(i, 150);
            r.optical_signal_poor = Some(true); // baselines both 0 → decoded None; the flags bit stays clear
            r
        }));
        assert_eq!(HrWatch::evaluate(&h), HrWatchState::Normal);
    }

    #[test]
    fn a_band_flagged_poor_second_is_never_eligible() {
        let h: Vec<_> = (0..1000)
            .map(|i| {
                let mut r = rec(i, 60);
                r.optical_baseline_a = Some(101); // worn, but the band distrusts its own beat detection
                r.optical_signal_poor = Some(true);
                r
            })
            .collect();
        assert_eq!(
            HrWatch::evaluate(&h),
            HrWatchState::Calibrating {
                have: 0,
                need: MIN_BASELINE_SAMPLES
            }
        );
    }

    #[test]
    fn a_high_hr_while_moving_is_never_eligible() {
        let mut h: Vec<_> = (0..600).map(|i| rec(i, 60)).collect();
        h.extend((600..1000).map(|i| {
            let mut r = rec(i, 150);
            r.activity_class = Some(2); // running, not at rest
            r
        }));
        assert_eq!(HrWatch::evaluate(&h), HrWatchState::Normal);
    }
}
