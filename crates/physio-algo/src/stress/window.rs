//! One windowed autonomic-stress formula, and the day and night derivations that share it. Each
//! bucket is z-scored against the window set's OWN calm quartiles (Q25 HR, Q75 RMSSD) and squashed
//! onto the shared 0–3 scale, so a reading is relative to that window and needs no cross-day
//! history. Replacing [`windowed_stress`] replaces both derivations at once.

use super::{band_of, mean_opt, population_std, squash, StressBand, HIGH_BAND_FLOOR, SD_FLOOR};
use crate::stats::percentile;

/// Bucket count below which the calm reference falls back to the plain mean.
const CALM_QUARTILE_MIN_COUNT: usize = 4;
const SECONDS_PER_MINUTE: i64 = 60;
/// The day derivation: one-hour buckets over local 06:00–21:59, three trailing buckets to sustain.
const HOUR_SECONDS: i64 = 3_600;
const WAKING_HOURS: (i32, i32) = (6, 22);
const SUSTAINED_BUCKETS: usize = 3;

/// One bucket's aggregates. `mean_hr` is `None` below the caller's sample gate, `rmssd` `None` on
/// insufficient clean R-R.
#[derive(Clone, Copy, Debug)]
pub struct HourPoint {
    /// Local hour (0-23) the bucket starts in.
    pub hour: i32,
    pub mean_hr: Option<f64>,
    pub rmssd: Option<f64>,
}

/// One scored bucket.
#[derive(Clone, Copy, Debug)]
pub struct ScoredHour {
    pub hour: i32,
    pub mean_hr: f64,
    pub rmssd: Option<f64>,
    pub stress: f64,
}

/// How a window set is selected and how long each bucket lasts. `bucket_seconds` only converts
/// scored buckets to band minutes today: [`HourPoint`] is keyed by hour-of-day, so a sub-hour
/// bucket needs a bucket index before the knob can be turned.
#[derive(Clone, Copy, Debug)]
pub struct StressWindowCfg {
    pub bucket_seconds: i64,
    /// Local hour-of-day `[start, end)` to keep, or `None` to score every bucket given.
    pub hours: Option<(i32, i32)>,
    /// Trailing high buckets that raise `sustained_high`.
    pub sustained_buckets: usize,
}

/// A scored window set: the per-bucket scores, the summary, and the minutes spent in each band.
#[derive(Clone, Debug)]
pub struct StressWindows {
    /// Scored buckets in input order; a bucket without a mean HR is dropped, not invented.
    pub buckets: Vec<ScoredHour>,
    /// Mean across the scored buckets, `None` when none scored.
    pub mean: Option<f64>,
    /// Hour of the highest-scoring bucket; the last one on a tie.
    pub peak_hour: Option<i32>,
    /// True when the last `sustained_buckets` scored buckets are all in the high band.
    pub sustained_high: bool,
    /// Length of the trailing high run.
    pub sustained_run: usize,
    /// Minutes spent in each band, `bucket_seconds` per scored bucket.
    pub low_minutes: i64,
    pub medium_minutes: i64,
    pub high_minutes: i64,
}

impl StressWindows {
    fn empty() -> Self {
        Self {
            buckets: Vec::new(), mean: None, peak_hour: None,
            sustained_high: false, sustained_run: 0,
            low_minutes: 0, medium_minutes: 0, high_minutes: 0,
        }
    }

    /// Share of the scored window spent in the high band, as a percentage. `None` when nothing
    /// scored — a window with no signal has no share, which is not the same as zero.
    pub fn high_share_pct(&self) -> Option<f64> {
        let total = self.low_minutes + self.medium_minutes + self.high_minutes;
        if total == 0 { return None; }
        Some(self.high_minutes as f64 * 100.0 / total as f64)
    }
}

/// Score a set of buckets for autonomic activation against their own calm quartiles. The reference
/// is built from the SAME buckets that get scored, so the two can never drift apart.
pub fn windowed_stress(points: &[HourPoint], cfg: StressWindowCfg) -> StressWindows {
    let selected: Vec<&HourPoint> = match cfg.hours {
        Some((start, end)) => points.iter().filter(|h| (start..end).contains(&h.hour)).collect(),
        None => points.iter().collect(),
    };
    if selected.is_empty() {
        return StressWindows::empty();
    }
    let hr_vals: Vec<f64> = selected.iter().filter_map(|h| h.mean_hr).collect();
    let rmssd_vals: Vec<f64> = selected.iter().filter_map(|h| h.rmssd).collect();

    let calm_hr = calm_reference(&hr_vals, true);
    let calm_rmssd = calm_reference(&rmssd_vals, false);
    let hr_mean = mean_opt(&hr_vals);
    let sd_hr = population_std(&hr_vals, hr_mean);
    let rmssd_mean = mean_opt(&rmssd_vals);
    let sd_rmssd = population_std(&rmssd_vals, rmssd_mean);

    let mut scored: Vec<ScoredHour> = Vec::new();
    for h in &selected {
        let Some(mean_hr) = h.mean_hr else { continue };
        let mut raw = 0.0;
        if let Some(ref_hr) = calm_hr {
            if sd_hr > SD_FLOOR {
                raw += (mean_hr - ref_hr) / sd_hr;
            }
        }
        if let (Some(r), Some(ref_r)) = (h.rmssd, calm_rmssd) {
            if sd_rmssd > SD_FLOOR {
                raw += (ref_r - r) / sd_rmssd;
            }
        }
        scored.push(ScoredHour { hour: h.hour, mean_hr, rmssd: h.rmssd, stress: squash(raw) });
    }
    let mut run = 0;
    for s in scored.iter().rev() {
        if s.stress >= HIGH_BAND_FLOOR { run += 1 } else { break; }
    }
    let sustained = run >= cfg.sustained_buckets;
    let mean = mean_opt(&scored.iter().map(|s| s.stress).collect::<Vec<f64>>());
    let peak_hour = scored
        .iter()
        .max_by(|a, b| a.stress.partial_cmp(&b.stress).unwrap())
        .map(|s| s.hour);

    let per_bucket = cfg.bucket_seconds / SECONDS_PER_MINUTE;
    let minutes = |band: StressBand| {
        scored.iter().filter(|s| band_of(s.stress) == band).count() as i64 * per_bucket
    };
    StressWindows {
        low_minutes: minutes(StressBand::Low),
        medium_minutes: minutes(StressBand::Medium),
        high_minutes: minutes(StressBand::High),
        mean,
        peak_hour,
        sustained_high: sustained,
        sustained_run: run,
        buckets: scored,
    }
}

/// Waking-hour stress across a day. Scores local 06:00–21:59 only: sleep is the calmest stretch of
/// the day, and letting it into the reference drags the calm anchor under every waking hour.
pub fn daytime_stress(hours: &[HourPoint]) -> StressWindows {
    windowed_stress(hours, StressWindowCfg {
        bucket_seconds: HOUR_SECONDS,
        hours: Some(WAKING_HOURS),
        sustained_buckets: SUSTAINED_BUCKETS,
    })
}

/// Stress across one sleep window. The caller passes only the buckets inside the span, because a
/// night crosses midnight and one hour-of-day range cannot say "22:00 to 06:00".
pub fn sleep_stress(hours: &[HourPoint]) -> StressWindows {
    windowed_stress(hours, StressWindowCfg {
        bucket_seconds: HOUR_SECONDS,
        hours: None,
        sustained_buckets: SUSTAINED_BUCKETS,
    })
}

/// The window set's calm end: the lower quartile when calm is low (HR), the upper when calm is
/// high (RMSSD). Falls back to the plain mean below four values.
fn calm_reference(xs: &[f64], calm_is_low: bool) -> Option<f64> {
    if xs.is_empty() { return None; }
    if xs.len() < CALM_QUARTILE_MIN_COUNT { return mean_opt(xs); }
    let mut s = xs.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Some(percentile(&s, if calm_is_low { 0.25 } else { 0.75 }))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One real worn day (24 hourly buckets, every hour above the 300-sample gate), banked so the
    /// day derivation is pinned to figures the shipped code produced before it was refactored.
    const REAL_DAY: [(i32, f64, f64); 24] = [
        (0, 69.66055555555556, 31.853136716696262),
        (1, 74.61222222222223, 28.580312554873604),
        (2, 72.715, 32.34110853233119),
        (3, 77.725, 23.600810436828418),
        (4, 71.61805555555556, 31.82855520534681),
        (5, 74.88333333333334, 31.901111810600415),
        (6, 76.93638888888889, 33.41835448743116),
        (7, 79.58055555555555, 32.92988783546757),
        (8, 81.86, 39.194740739980375),
        (9, 88.1025, 33.65074165294286),
        (10, 80.35863964273446, 46.149393178731735),
        (11, 81.62106703146375, 47.29722304948207),
        (12, 82.69511846988296, 49.17354525293673),
        (13, 97.03333333333333, 38.656262968200465),
        (14, 90.94305555555556, 43.17652239350124),
        (15, 86.31833333333333, 48.90791445549015),
        (16, 82.34055555555555, 64.58555699274339),
        (17, 87.78472222222223, 47.96100682387079),
        (18, 87.66666666666667, 38.86665627111457),
        (19, 85.71916666666667, 31.259841796659202),
        (20, 84.63601000277855, 31.958253315144663),
        (21, 88.18227285357044, 37.03831655466921),
        (22, 94.05666666666667, 39.54099430851305),
        (23, 85.92416666666666, 48.71628845225568),
    ];

    fn real_day() -> Vec<HourPoint> {
        REAL_DAY.iter()
            .map(|&(hour, hr, rmssd)| HourPoint { hour, mean_hr: Some(hr), rmssd: Some(rmssd) })
            .collect()
    }

    fn hours(range: std::ops::Range<i32>, hr: f64, rmssd: f64) -> Vec<HourPoint> {
        range.map(|h| HourPoint { hour: h, mean_hr: Some(hr), rmssd: Some(rmssd) }).collect()
    }

    // ── the day derivation, pinned bit-for-bit ──

    #[test]
    fn real_day_reproduces_the_shipped_scores() {
        // Every scored hour of one real day, to the last bit. A change here is a behaviour change.
        const EXPECTED: [(i32, f64); 16] = [
            (6, 1.9487038705569648),
            (7, 2.3195649234236337),
            (8, 2.1777912507317003),
            (9, 2.8469281331909126),
            (10, 1.3888917940633005),
            (11, 1.4863940057677225),
            (12, 1.4910399303170887),
            (13, 2.955978700967047),
            (14, 2.7524779713325853),
            (15, 2.0544234686818097),
            (16, 0.39772007276957405),
            (17, 2.301539910923305),
            (18, 2.7075310445390244),
            (19, 2.811642863863905),
            (20, 2.749485844900177),
            (21, 2.781870795005547),
        ];
        let r = daytime_stress(&real_day());
        assert_eq!(r.buckets.len(), EXPECTED.len());
        for (got, &(hour, stress)) in r.buckets.iter().zip(EXPECTED.iter()) {
            assert_eq!(got.hour, hour);
            assert_eq!(got.stress.to_bits(), stress.to_bits(), "hour {hour} moved");
        }
        assert_eq!(r.mean.unwrap().to_bits(), 2.1982490363146434f64.to_bits());
        assert_eq!(r.peak_hour, Some(13));
        assert!(r.sustained_high);
        assert_eq!(r.sustained_run, 5);
    }

    #[test]
    fn real_day_bands_tally_to_the_scored_hours() {
        let r = daytime_stress(&real_day());
        // 16 scored hours at 60 min each: 1 low (16:00), 4 medium, 11 high.
        assert_eq!((r.low_minutes, r.medium_minutes, r.high_minutes), (60, 240, 660));
        assert_eq!(
            r.low_minutes + r.medium_minutes + r.high_minutes,
            r.buckets.len() as i64 * 60,
        );
    }

    #[test]
    fn real_day_high_share_matches_its_band_minutes() {
        let r = daytime_stress(&real_day());
        // 660 of 960 scored minutes are high.
        assert_eq!(r.high_share_pct().unwrap().to_bits(), (660.0f64 * 100.0 / 960.0).to_bits());
    }

    #[test]
    fn unscored_window_has_no_high_share() {
        // Zero minutes is not "0% high": nothing was measured, so the share is absent.
        assert!(sleep_stress(&[]).high_share_pct().is_none());
    }

    #[test]
    fn flat_day_scores_neutral() {
        // All hours identical → zero spread → all z-scores zero → neutral 1.5
        let r = daytime_stress(&hours(6..22, 70.0, 40.0));
        assert!((r.mean.unwrap() - 1.5).abs() < 0.1, "flat day → neutral, got {}", r.mean.unwrap());
        assert!(!r.sustained_high);
    }

    #[test]
    fn spiky_afternoon_scores_high_and_sustained() {
        let mut points = hours(6..14, 70.0, 40.0);
        points.extend(hours(14..18, 90.0, 25.0));
        let r = daytime_stress(&points);
        assert!(r.sustained_high, "3+ spiky hours should trigger sustained high");
        assert!(r.peak_hour.is_some());
    }

    #[test]
    fn below_hr_gate_excluded() {
        let points = vec![HourPoint { hour: 10, mean_hr: None, rmssd: Some(40.0) }];
        let r = daytime_stress(&points);
        assert!(r.buckets.is_empty());
        assert!(r.mean.is_none());
        assert_eq!((r.low_minutes, r.medium_minutes, r.high_minutes), (0, 0, 0));
    }

    #[test]
    fn sleep_hours_excluded() {
        let points = vec![HourPoint { hour: 2, mean_hr: Some(60.0), rmssd: Some(50.0) }];
        let r = daytime_stress(&points);
        assert!(r.buckets.is_empty(), "2am should be excluded from waking window");
    }

    // ── the night derivation ──

    #[test]
    fn night_scores_the_hours_the_day_window_drops() {
        // The same real day's night buckets, which `daytime_stress` refuses by hour-of-day.
        let night: Vec<HourPoint> = real_day()
            .into_iter()
            .filter(|h| !(6..22).contains(&h.hour))
            .collect();
        assert_eq!(night.len(), 8);
        assert!(daytime_stress(&night).buckets.is_empty());
        let r = sleep_stress(&night);
        assert_eq!(r.buckets.len(), 8);
        assert_eq!(r.buckets.first().unwrap().hour, 0);
        assert_eq!(r.buckets.last().unwrap().hour, 23);
        assert_eq!(r.low_minutes + r.medium_minutes + r.high_minutes, 8 * 60);
    }

    #[test]
    fn night_crossing_midnight_keeps_every_bucket() {
        // 22:00 → 05:00 in wall order; an hour-of-day range cannot express it, `None` can.
        let spans: Vec<i32> = vec![22, 23, 0, 1, 2, 3, 4, 5];
        let points: Vec<HourPoint> = spans.iter()
            .map(|&h| HourPoint { hour: h, mean_hr: Some(60.0 + h as f64 % 3.0), rmssd: Some(50.0) })
            .collect();
        let r = sleep_stress(&points);
        assert_eq!(r.buckets.len(), spans.len());
        assert_eq!(r.buckets.iter().map(|b| b.hour).collect::<Vec<_>>(), spans);
    }

    #[test]
    fn uniformly_stressful_night_still_reads_neutral() {
        // The reference is the night's own calm quartile, so a flat high-HR night has nothing to
        // look stressed against. The caveat, asserted rather than described.
        let points: Vec<HourPoint> = (0..8)
            .map(|h| HourPoint { hour: h, mean_hr: Some(95.0), rmssd: Some(12.0) })
            .collect();
        let r = sleep_stress(&points);
        assert!((r.mean.unwrap() - 1.5).abs() < 1e-12);
        assert_eq!(r.high_minutes, 0);
    }

    #[test]
    fn empty_input_scores_nothing() {
        let r = sleep_stress(&[]);
        assert!(r.buckets.is_empty() && r.mean.is_none() && r.peak_hour.is_none());
        assert_eq!(r.sustained_run, 0);
    }

    // ── the core's knobs ──

    #[test]
    fn hours_none_and_a_covering_range_agree() {
        let points = hours(6..22, 70.0, 40.0);
        let a = windowed_stress(&points, StressWindowCfg {
            bucket_seconds: HOUR_SECONDS, hours: None, sustained_buckets: SUSTAINED_BUCKETS,
        });
        let b = daytime_stress(&points);
        assert_eq!(a.buckets.len(), b.buckets.len());
        for (x, y) in a.buckets.iter().zip(b.buckets.iter()) {
            assert_eq!(x.stress.to_bits(), y.stress.to_bits());
        }
    }

    #[test]
    fn bucket_seconds_scales_the_band_minutes() {
        let points = hours(6..22, 70.0, 40.0);
        let five_min = windowed_stress(&points, StressWindowCfg {
            bucket_seconds: 300, hours: Some(WAKING_HOURS), sustained_buckets: SUSTAINED_BUCKETS,
        });
        let hourly = daytime_stress(&points);
        assert_eq!(hourly.medium_minutes, 16 * 60);
        assert_eq!(five_min.medium_minutes, 16 * 5);
        // Only the minutes move: the scores themselves are unchanged by the bucket width.
        for (x, y) in five_min.buckets.iter().zip(hourly.buckets.iter()) {
            assert_eq!(x.stress.to_bits(), y.stress.to_bits());
        }
    }

    #[test]
    fn sustained_buckets_gates_the_run() {
        let mut points = hours(6..14, 70.0, 40.0);
        points.extend(hours(14..18, 90.0, 25.0));
        let strict = windowed_stress(&points, StressWindowCfg {
            bucket_seconds: HOUR_SECONDS, hours: Some(WAKING_HOURS), sustained_buckets: 99,
        });
        assert!(!strict.sustained_high);
        assert_eq!(strict.sustained_run, daytime_stress(&points).sustained_run);
    }
}
