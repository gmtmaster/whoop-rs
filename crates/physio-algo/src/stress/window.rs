//! One windowed autonomic-stress formula, and the day and night derivations that share it. Each
//! bucket is z-scored against the window set's OWN calm quartiles (Q25 HR, Q75 RMSSD) and squashed
//! onto the shared 0–3 scale, so a reading is relative to that window and needs no cross-day
//! history. Replacing [`windowed_stress`] replaces both derivations at once.

use super::{band_of, mean_opt, population_std, squash, StressBand, HIGH_BAND_FLOOR, SD_FLOOR};
use crate::stats::percentile;

/// Bucket count below which the calm reference falls back to the plain mean.
const CALM_QUARTILE_MIN_COUNT: usize = 4;
const SECONDS_PER_MINUTE: i64 = 60;
const MILLIS_PER_SECOND: i64 = 1_000;
/// The day and night derivations both bucket by the hour; three trailing buckets sustain.
const HOUR_SECONDS: i64 = 3_600;
const SUSTAINED_BUCKETS: usize = 3;
/// Dynamic accel (g) at or above which a bucket is movement, not rest. [`crate::stress_onset`]
/// gates on this same constant and the same `>=` edge, so both stress paths refuse the same motion.
pub const ACTIVITY_GATE_G: f64 = 0.15;

/// A `[start, end)` span in epoch ms — one sleep or nap window a bucket can fall inside.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpanMs {
    pub start_ms: i64,
    pub end_ms: i64,
}

/// Why a bucket was held out of the score. A suppressed bucket is a known state, not a gap: the
/// caller can say "no reading, you were moving" instead of drawing a hole.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Suppression {
    /// Bucket motion reached [`ACTIVITY_GATE_G`]. Checked first, so a moving bucket inside a span
    /// reads as activity rather than sleep.
    Active,
    /// Bucket overlaps a sleep or nap span.
    Asleep,
}

/// One bucket kept out of both the score and the calm reference, with its reason.
#[derive(Clone, Copy, Debug)]
pub struct SuppressedBucket {
    pub start_ms: i64,
    pub hour: i32,
    pub suppression: Suppression,
}

/// One bucket's aggregates. `mean_hr` is `None` below the caller's sample gate, `rmssd` `None` on
/// insufficient clean R-R, `motion_g` `None` when the accel channel is absent for the bucket.
#[derive(Clone, Copy, Debug)]
pub struct HourPoint {
    /// Epoch ms the bucket starts at; with `bucket_seconds` this fixes its span on the clock.
    pub start_ms: i64,
    /// Local hour (0-23) the bucket starts in.
    pub hour: i32,
    pub mean_hr: Option<f64>,
    pub rmssd: Option<f64>,
    /// Mean dynamic accel over the bucket, in g.
    pub motion_g: Option<f64>,
}

/// One scored bucket.
#[derive(Clone, Copy, Debug)]
pub struct ScoredHour {
    pub start_ms: i64,
    pub hour: i32,
    pub mean_hr: f64,
    pub rmssd: Option<f64>,
    pub stress: f64,
}

/// How a window set is selected and how long each bucket lasts. `bucket_seconds` is load-bearing:
/// it sets the band minutes AND, with [`HourPoint::start_ms`], each bucket's span for the sleep
/// overlap test, so a sub-hour bucket is now a real setting.
#[derive(Clone, Copy, Debug)]
pub struct StressWindowCfg<'a> {
    pub bucket_seconds: i64,
    /// Sleep + nap spans; any bucket overlapping one is suppressed before scoring. Empty scores
    /// every bucket, which is what the night derivation wants.
    pub sleep_spans: &'a [SpanMs],
    /// Motion gate in g, or `None` to score regardless of movement.
    pub activity_gate_g: Option<f64>,
    /// Trailing high buckets that raise `sustained_high`.
    pub sustained_buckets: usize,
}

/// A scored window set: the per-bucket scores, the buckets held out and why, the summary, and the
/// minutes spent in each band.
#[derive(Clone, Debug)]
pub struct StressWindows {
    /// Scored buckets in input order; a bucket without a mean HR is dropped, not invented.
    pub buckets: Vec<ScoredHour>,
    /// Buckets excluded by the activity or sleep gate, in input order.
    pub suppressed: Vec<SuppressedBucket>,
    /// Mean across the scored buckets, `None` when none scored.
    pub mean: Option<f64>,
    /// Hour of the highest-scoring bucket; the last one on a tie.
    pub peak_hour: Option<i32>,
    /// Start of that same peak bucket — the sub-hour answer `peak_hour` cannot give.
    pub peak_start_ms: Option<i64>,
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
            buckets: Vec::new(), suppressed: Vec::new(), mean: None,
            peak_hour: None, peak_start_ms: None,
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

    /// Suppressed buckets carrying `reason` — the count behind "you were moving" or "you were asleep".
    pub fn suppressed_count(&self, reason: Suppression) -> usize {
        self.suppressed.iter().filter(|s| s.suppression == reason).count()
    }
}

/// Why this bucket cannot enter the score, or `None` to keep it. Activity is tested before sleep,
/// so a moving bucket inside a nap span is reported as movement.
fn suppression_of(h: &HourPoint, cfg: &StressWindowCfg) -> Option<Suppression> {
    if let Some(gate) = cfg.activity_gate_g {
        if h.motion_g.is_some_and(|m| m >= gate) {
            return Some(Suppression::Active);
        }
    }
    let bucket_end = h.start_ms + cfg.bucket_seconds.max(0) * MILLIS_PER_SECOND;
    let overlaps = cfg
        .sleep_spans
        .iter()
        .any(|s| h.start_ms < s.end_ms && s.start_ms < bucket_end);
    overlaps.then_some(Suppression::Asleep)
}

/// Score a set of buckets for autonomic activation against their own calm quartiles. Suppressed
/// buckets leave BEFORE the reference is built, so a workout never anchors the calm end. The
/// reference is built from the SAME buckets that get scored, so the two can never drift apart.
pub fn windowed_stress(points: &[HourPoint], cfg: StressWindowCfg) -> StressWindows {
    let mut selected: Vec<&HourPoint> = Vec::with_capacity(points.len());
    let mut suppressed: Vec<SuppressedBucket> = Vec::new();
    for h in points {
        match suppression_of(h, &cfg) {
            Some(suppression) => suppressed.push(SuppressedBucket {
                start_ms: h.start_ms, hour: h.hour, suppression,
            }),
            None => selected.push(h),
        }
    }
    if selected.is_empty() {
        return StressWindows { suppressed, ..StressWindows::empty() };
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
        scored.push(ScoredHour {
            start_ms: h.start_ms, hour: h.hour, mean_hr, rmssd: h.rmssd, stress: squash(raw),
        });
    }
    let mut run = 0;
    for s in scored.iter().rev() {
        if s.stress >= HIGH_BAND_FLOOR { run += 1 } else { break; }
    }
    let sustained = run >= cfg.sustained_buckets;
    let mean = mean_opt(&scored.iter().map(|s| s.stress).collect::<Vec<f64>>());
    let peak = scored.iter().max_by(|a, b| a.stress.partial_cmp(&b.stress).unwrap());

    let per_bucket = cfg.bucket_seconds / SECONDS_PER_MINUTE;
    let minutes = |band: StressBand| {
        scored.iter().filter(|s| band_of(s.stress) == band).count() as i64 * per_bucket
    };
    StressWindows {
        low_minutes: minutes(StressBand::Low),
        medium_minutes: minutes(StressBand::Medium),
        high_minutes: minutes(StressBand::High),
        mean,
        peak_hour: peak.map(|s| s.hour),
        peak_start_ms: peak.map(|s| s.start_ms),
        sustained_high: sustained,
        sustained_run: run,
        suppressed,
        buckets: scored,
    }
}

/// Waking stress across a day. Sleep is the calmest stretch of the day and a workout the loudest,
/// so both leave before the calm anchor is built — the spans and the motion channel decide, not
/// the clock.
pub fn daytime_stress(hours: &[HourPoint], sleep_spans: &[SpanMs]) -> StressWindows {
    windowed_stress(hours, StressWindowCfg {
        bucket_seconds: HOUR_SECONDS,
        sleep_spans,
        activity_gate_g: Some(ACTIVITY_GATE_G),
        sustained_buckets: SUSTAINED_BUCKETS,
    })
}

/// Stress across one sleep window. The caller passes only the buckets inside the span, because a
/// night crosses midnight; nothing is suppressed, since sleep and stillness are the point here.
pub fn sleep_stress(hours: &[HourPoint]) -> StressWindows {
    windowed_stress(hours, StressWindowCfg {
        bucket_seconds: HOUR_SECONDS,
        sleep_spans: &[],
        activity_gate_g: None,
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

    /// Fixture day base (epoch ms) with local == UTC, so hour `h` starts at `DAY_START_MS + h × 1 h`.
    const DAY_START_MS: i64 = 1_753_920_000_000;
    const HOUR_MS: i64 = 3_600_000;

    /// One real worn day's HR and RMSSD (24 hourly buckets, every hour above the 300-sample gate),
    /// banked so the day derivation is pinned to figures the shipped code produced before it was
    /// refactored. Its motion channel is [`REAL_DAY_MOTION_G`], banked separately.
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

    /// A real worn day's motion channel: mean on-strap dynamic accel (g) per hourly bucket, all 24
    /// buckets ≥ 3475 of 3600 seconds. Without it every bucket read `motion_g: None`, so
    /// [`ACTIVITY_GATE_G`] was never evaluated on the pinned day and could not move any figure here.
    const REAL_DAY_MOTION_G: [f64; 24] = [
        0.021853, 0.006737, 0.010248, 0.009181, 0.008856, 0.008557,
        0.015612, 0.011207, 0.016108, 0.015935, 0.093877, 0.068195,
        0.111046, 0.085727, 0.142547, 0.039923, 0.129114, 0.119880,
        0.110810, 0.097985, 0.107272, 0.118880, 0.086236, 0.079337,
    ];

    /// The pinned day as the shipped path sees it: HR, RMSSD **and** motion on every bucket.
    fn real_day() -> Vec<HourPoint> {
        REAL_DAY.iter()
            .map(|&(hour, hr, rmssd)| HourPoint {
                motion_g: Some(REAL_DAY_MOTION_G[hour as usize]),
                ..point(hour, Some(hr), Some(rmssd))
            })
            .collect()
    }

    fn point(hour: i32, mean_hr: Option<f64>, rmssd: Option<f64>) -> HourPoint {
        HourPoint {
            start_ms: DAY_START_MS + hour as i64 * HOUR_MS,
            hour,
            mean_hr,
            rmssd,
            motion_g: None,
        }
    }

    fn span(from_hour: i64, to_hour: i64) -> SpanMs {
        SpanMs { start_ms: DAY_START_MS + from_hour * HOUR_MS, end_ms: DAY_START_MS + to_hour * HOUR_MS }
    }

    /// The retired clock window `WAKING_HOURS = (6, 22)` as spans — the control that holds the
    /// scores still across the refactor, NOT this day's sleep: [`REAL_DAY`] carries no sleep record,
    /// and hour 22's 94.06 bpm is its second-highest HR. Spans-beat-the-clock is a separate test.
    fn waking_window_as_spans() -> Vec<SpanMs> {
        vec![span(0, 6), span(22, 24)]
    }

    fn hours(range: std::ops::Range<i32>, hr: f64, rmssd: f64) -> Vec<HourPoint> {
        range.map(|h| point(h, Some(hr), Some(rmssd))).collect()
    }

    // ── the day derivation, pinned bit-for-bit ──

    #[test]
    fn real_day_reproduces_the_shipped_scores() {
        // Every scored hour of one real day, to the last bit, with the night given as sleep spans.
        // A change here is a behaviour change.
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
        let day = real_day();
        // The input, before the output: a bucket without motion skips the activity gate silently and
        // scores identically, so only the INPUT can catch the channel going absent again.
        assert!(day.iter().all(|h| h.motion_g.is_some()), "the pinned day lost its motion channel");
        let r = daytime_stress(&day, &waking_window_as_spans());
        assert_eq!(r.buckets.len(), EXPECTED.len());
        for (got, &(hour, stress)) in r.buckets.iter().zip(EXPECTED.iter()) {
            assert_eq!(got.hour, hour);
            assert_eq!(got.start_ms, DAY_START_MS + hour as i64 * HOUR_MS);
            assert_eq!(got.stress.to_bits(), stress.to_bits(), "hour {hour} moved");
        }
        assert_eq!(r.mean.unwrap().to_bits(), 2.1982490363146434f64.to_bits());
        assert_eq!(r.peak_hour, Some(13));
        assert_eq!(r.peak_start_ms, Some(DAY_START_MS + 13 * HOUR_MS));
        assert!(r.sustained_high);
        assert_eq!(r.sustained_run, 5);
        assert_eq!(r.suppressed.len(), 8);
        assert_eq!(r.suppressed_count(Suppression::Asleep), 8);
        // Named, not inferred: the gate ran on all 24 buckets and suppressed none of them.
        assert_eq!(r.suppressed_count(Suppression::Active), 0);
    }

    #[test]
    fn the_real_day_never_reaches_the_activity_gate() {
        // The headroom the golden day leaves ACTIVITY_GATE_G, asserted rather than assumed. Its
        // busiest hour is 14:00 at 0.142547 g — below 0.15, so no real hour here is suppressed as
        // movement, and the gate would have to drop under that value before this day moves.
        let busiest = REAL_DAY_MOTION_G.iter().copied().fold(f64::MIN, f64::max);
        assert_eq!(busiest.to_bits(), 0.142547f64.to_bits());
        assert!(busiest < ACTIVITY_GATE_G, "{busiest} g would suppress an hour of the golden day");
        // The margin is 4.97% of the gate, so a -10% move to 0.135 crosses it and a +10% does not.
        assert!(0.135 < busiest && busiest < 0.165);
        let at = |g: f64| windowed_stress(&real_day(), StressWindowCfg {
            bucket_seconds: HOUR_SECONDS, sleep_spans: &waking_window_as_spans(),
            activity_gate_g: Some(g), sustained_buckets: SUSTAINED_BUCKETS,
        });
        // Lowered to 0.135 the day loses 14:00 to movement, so the calm quartiles are rebuilt from
        // 15 buckets and the mean moves 0.0277. Raised to 0.165 nothing changes: this day can only
        // falsify the constant downwards, which is the measurement, not a target.
        let lowered = at(0.135);
        assert_eq!(lowered.suppressed_count(Suppression::Active), 1);
        assert_eq!(lowered.suppressed_count(Suppression::Asleep), 8);
        assert_eq!(lowered.buckets.len(), 15);
        assert!(lowered.buckets.iter().all(|b| b.hour != 14));
        assert_eq!(lowered.mean.unwrap().to_bits(), 2.1705370801496633f64.to_bits());
        assert_eq!((lowered.low_minutes, lowered.medium_minutes, lowered.high_minutes), (60, 240, 600));
        let raised = at(0.165);
        assert_eq!(raised.suppressed_count(Suppression::Active), 0);
        assert_eq!(raised.mean.unwrap().to_bits(), 2.1982490363146434f64.to_bits());
    }

    #[test]
    fn real_day_bands_tally_to_the_scored_hours() {
        let r = daytime_stress(&real_day(), &waking_window_as_spans());
        // 16 scored hours at 60 min each: 1 low (16:00), 4 medium, 11 high.
        assert_eq!((r.low_minutes, r.medium_minutes, r.high_minutes), (60, 240, 660));
        assert_eq!(
            r.low_minutes + r.medium_minutes + r.high_minutes,
            r.buckets.len() as i64 * 60,
        );
    }

    #[test]
    fn real_day_high_share_matches_its_band_minutes() {
        let r = daytime_stress(&real_day(), &waking_window_as_spans());
        // 660 of 960 scored minutes are high.
        assert_eq!(r.high_share_pct().unwrap().to_bits(), (660.0f64 * 100.0 / 960.0).to_bits());
    }

    #[test]
    fn a_flat_day_cannot_reproduce_the_golden_day() {
        // The null the golden day is pinned against: the SAME 24 buckets with the SAME motion and
        // the SAME spans, but every HR and RMSSD replaced by the day's own mean. A scorer that
        // never reads the two channels can only produce this, and it lands on the neutral midpoint
        // with no high minutes at all. Without this arm the golden figures are only self-consistent.
        let hr_mean = REAL_DAY.iter().map(|r| r.1).sum::<f64>() / REAL_DAY.len() as f64;
        let rmssd_mean = REAL_DAY.iter().map(|r| r.2).sum::<f64>() / REAL_DAY.len() as f64;
        let flat: Vec<HourPoint> = REAL_DAY
            .iter()
            .map(|&(hour, _, _)| HourPoint {
                motion_g: Some(REAL_DAY_MOTION_G[hour as usize]),
                ..point(hour, Some(hr_mean), Some(rmssd_mean))
            })
            .collect();
        let null = daytime_stress(&flat, &waking_window_as_spans());
        let real = daytime_stress(&real_day(), &waking_window_as_spans());
        // Same 16 buckets held, same 8 suppressed: only the channels differ.
        assert_eq!(null.buckets.len(), real.buckets.len());
        assert_eq!(null.suppressed_count(Suppression::Asleep), 8);
        assert!(null.buckets.iter().all(|b| b.stress.to_bits() == squash(0.0).to_bits()));
        assert_eq!(null.mean.unwrap().to_bits(), squash(0.0).to_bits());
        assert_eq!((null.low_minutes, null.medium_minutes, null.high_minutes), (0, 960, 0));
        assert_eq!(null.high_share_pct(), Some(0.0));
        assert_eq!(null.sustained_run, 0);
        assert!(!null.sustained_high);
        // …and every headline figure of the golden day is somewhere else.
        assert_ne!(null.mean.unwrap().to_bits(), real.mean.unwrap().to_bits());
        assert_ne!(null.high_share_pct(), real.high_share_pct());
        assert_ne!(null.sustained_run, real.sustained_run);
    }

    #[test]
    fn the_golden_day_scored_as_a_night_reproduces_its_own_shipped_scores() {
        // The night derivation on the same real day: the 8 hours the waking spans hold out, scored
        // by `sleep_stress` against their OWN calm quartiles. Bit-exact, with the flat-night null
        // beside it — a scorer blind to the two channels reads neutral and has no high minutes.
        const EXPECTED: [(i32, f64); 8] = [
            (0, 1.4733469641345744),
            (1, 2.2322992736077323),
            (2, 1.7160720744458653),
            (3, 2.693717620268321),
            (4, 1.6649947239615413),
            (5, 1.9592109448513932),
            (22, 2.65148521046022),
            (23, 1.2620996267839222),
        ];
        let night: Vec<HourPoint> =
            real_day().into_iter().filter(|h| !(6..22).contains(&h.hour)).collect();
        let r = sleep_stress(&night);
        assert_eq!(r.buckets.len(), EXPECTED.len());
        for (got, &(hour, stress)) in r.buckets.iter().zip(EXPECTED.iter()) {
            assert_eq!(got.hour, hour);
            assert_eq!(got.stress.to_bits(), stress.to_bits(), "night hour {hour} moved");
        }
        assert_eq!(r.mean.unwrap().to_bits(), 1.9566533048141959f64.to_bits());
        assert_eq!(r.peak_hour, Some(3));
        assert_eq!((r.low_minutes, r.medium_minutes, r.high_minutes), (0, 300, 180));
        assert_eq!(r.high_share_pct(), Some(37.5));
        assert_eq!(r.sustained_run, 0);

        let hr_mean = mean_opt(&night.iter().filter_map(|h| h.mean_hr).collect::<Vec<_>>()).unwrap();
        let rmssd_mean = mean_opt(&night.iter().filter_map(|h| h.rmssd).collect::<Vec<_>>()).unwrap();
        let flat: Vec<HourPoint> = night
            .iter()
            .map(|h| HourPoint { mean_hr: Some(hr_mean), rmssd: Some(rmssd_mean), ..*h })
            .collect();
        let null = sleep_stress(&flat);
        assert!(null.buckets.iter().all(|b| b.stress.to_bits() == squash(0.0).to_bits()));
        assert_eq!(null.high_minutes, 0);
        assert_ne!(null.mean.unwrap().to_bits(), r.mean.unwrap().to_bits());
        assert_ne!(null.high_share_pct(), r.high_share_pct());
    }

    #[test]
    fn a_non_clock_span_set_moves_the_golden_day() {
        // The same golden day under a night no hour-of-day range can express: 23:10 the evening
        // before → 06:40, crossing midnight and ending mid-bucket. Every headline figure moves, so
        // the spans really do select the window and the pinned day is not the clock in disguise.
        let night = [SpanMs {
            start_ms: DAY_START_MS - 50 * 60_000,
            end_ms: DAY_START_MS + 6 * HOUR_MS + 40 * 60_000,
        }];
        let r = daytime_stress(&real_day(), &night);
        assert_eq!(r.suppressed_count(Suppression::Asleep), 7, "00:00–06:00 go, 06:00 by 40 min");
        assert_eq!(r.buckets.iter().map(|b| b.hour).collect::<Vec<_>>(), (7..24).collect::<Vec<_>>());
        assert_eq!(r.mean.unwrap().to_bits(), 2.233697117932028f64.to_bits());
        assert_eq!((r.low_minutes, r.medium_minutes, r.high_minutes), (60, 240, 720));
        // The run is the most span-sensitive figure: 5 under the clock window, 0 here. 22:00 does
        // score high (2.9170), but the day now ends on 23:00 at 1.9951 — 0.0049 under the floor.
        assert_eq!(r.sustained_run, 0);
        assert!(!r.sustained_high);
        assert_eq!(r.buckets.last().unwrap().stress.to_bits(), 1.9951226376307365f64.to_bits());
    }

    #[test]
    fn unscored_window_has_no_high_share() {
        // Zero minutes is not "0% high": nothing was measured, so the share is absent.
        assert!(sleep_stress(&[]).high_share_pct().is_none());
    }

    #[test]
    fn flat_day_scores_neutral() {
        // All hours identical → zero spread → all z-scores zero → neutral 1.5
        let r = daytime_stress(&hours(6..22, 70.0, 40.0), &waking_window_as_spans());
        assert!((r.mean.unwrap() - 1.5).abs() < 0.1, "flat day → neutral, got {}", r.mean.unwrap());
        assert!(!r.sustained_high);
    }

    #[test]
    fn spiky_afternoon_scores_high_and_sustained() {
        let mut points = hours(6..14, 70.0, 40.0);
        points.extend(hours(14..18, 90.0, 25.0));
        let r = daytime_stress(&points, &waking_window_as_spans());
        assert!(r.sustained_high, "3+ spiky hours should trigger sustained high");
        assert!(r.peak_hour.is_some());
    }

    #[test]
    fn below_hr_gate_excluded() {
        let points = vec![point(10, None, Some(40.0))];
        let r = daytime_stress(&points, &waking_window_as_spans());
        assert!(r.buckets.is_empty());
        assert!(r.mean.is_none());
        assert_eq!((r.low_minutes, r.medium_minutes, r.high_minutes), (0, 0, 0));
        // A missing HR is a gap, not a suppression: nothing was excluded, there was nothing to score.
        assert!(r.suppressed.is_empty());
    }

    // ── the calm reference, and the short-window fallback ──

    #[test]
    fn the_calm_reference_switches_from_the_mean_to_the_quartile_at_four_values() {
        // Below CALM_QUARTILE_MIN_COUNT the anchor is the plain mean; at it, the quartiles take
        // over. Three values and four are asserted against BOTH rules, so a reference stuck on
        // either one fails: the mean answer and the quartile answer differ on both sides.
        let three = [60.0, 62.0, 100.0];
        let four = [60.0, 62.0, 100.0, 102.0];
        assert_eq!(three.len(), CALM_QUARTILE_MIN_COUNT - 1);
        assert_eq!(four.len(), CALM_QUARTILE_MIN_COUNT);

        assert_eq!(calm_reference(&three, true), mean_opt(&three));
        assert_eq!(calm_reference(&three, false), mean_opt(&three));
        assert_ne!(calm_reference(&three, true).unwrap(), percentile(&three, 0.25));
        assert_ne!(calm_reference(&three, false).unwrap(), percentile(&three, 0.75));
        // Below four the fallback stops telling the two ends apart: one mean answers for both.
        assert_eq!(calm_reference(&three, true), calm_reference(&three, false));

        assert_eq!(calm_reference(&four, true).unwrap(), percentile(&four, 0.25));
        assert_eq!(calm_reference(&four, false).unwrap(), percentile(&four, 0.75));
        assert_ne!(calm_reference(&four, true), mean_opt(&four));
        assert_ne!(calm_reference(&four, false), mean_opt(&four));
        assert_ne!(calm_reference(&four, true), calm_reference(&four, false));

        assert_eq!(calm_reference(&[], true), None);
    }

    #[test]
    fn a_three_bucket_nap_is_scored_against_its_mean_not_its_quartile() {
        // The short-window fallback end to end: a 3-bucket nap through `sleep_stress`. Both
        // candidate references are computed here and the scores checked against the mean one, so a
        // reference that reached for the quartile on a window this short would fail every bucket.
        let hr = [60.0, 62.0, 100.0];
        let nap: Vec<HourPoint> =
            hr.iter().enumerate().map(|(i, &h)| point(13 + i as i32, Some(h), Some(40.0))).collect();
        let r = sleep_stress(&nap);
        assert_eq!(r.buckets.len(), 3);

        let mean = mean_opt(&hr).unwrap();
        let sd = population_std(&hr, Some(mean));
        // A flat RMSSD channel has no spread, so only the HR term reaches the score.
        for (b, &h) in r.buckets.iter().zip(hr.iter()) {
            assert_eq!(b.stress.to_bits(), squash((h - mean) / sd).to_bits(), "hour {}", b.hour);
            assert_ne!(b.stress.to_bits(), squash((h - percentile(&hr, 0.25)) / sd).to_bits());
        }
        assert_eq!(r.mean.unwrap().to_bits(), 1.465216959752496f64.to_bits());
    }

    // ── the sleep gate: spans, not the clock ──

    #[test]
    fn sleep_span_suppresses_and_names_the_bucket() {
        let points = vec![point(2, Some(60.0), Some(50.0))];
        let r = daytime_stress(&points, &waking_window_as_spans());
        assert!(r.buckets.is_empty(), "a bucket inside a sleep span is not scored");
        assert_eq!(r.suppressed.len(), 1);
        assert_eq!(r.suppressed[0].suppression, Suppression::Asleep);
        assert_eq!(r.suppressed[0].hour, 2);
        assert_eq!(r.suppressed[0].start_ms, DAY_START_MS + 2 * HOUR_MS);
    }

    #[test]
    fn an_afternoon_nap_is_suppressed_mid_day() {
        // A 14:00–15:30 nap: mid-afternoon sleep leaves the score, and both overlapped buckets go.
        let points = hours(6..22, 70.0, 40.0);
        let spans = [span(14, 15), SpanMs { start_ms: DAY_START_MS + 15 * HOUR_MS, end_ms: DAY_START_MS + 15 * HOUR_MS + HOUR_MS / 2 }];
        let r = daytime_stress(&points, &spans);
        assert_eq!(r.suppressed_count(Suppression::Asleep), 2, "14:00 and the partly-overlapped 15:00");
        assert!(r.buckets.iter().all(|b| b.hour != 14 && b.hour != 15));
    }

    #[test]
    fn a_bucket_touching_a_span_edge_is_kept() {
        // Spans are `[start, end)`; a bucket that ends exactly where sleep starts never overlaps it.
        let points = vec![point(5, Some(70.0), Some(40.0)), point(6, Some(70.0), Some(40.0))];
        let r = daytime_stress(&points, &[span(6, 8)]);
        assert_eq!(r.suppressed.len(), 1);
        assert_eq!(r.suppressed[0].hour, 6);
        assert_eq!(r.buckets.len(), 1);
        assert_eq!(r.buckets[0].hour, 5);
    }

    #[test]
    fn night_derivation_scores_inside_its_own_span() {
        // `sleep_stress` passes no spans, so the same buckets `daytime_stress` refuses all score.
        let night: Vec<HourPoint> = real_day().into_iter().filter(|h| !(6..22).contains(&h.hour)).collect();
        assert_eq!(night.len(), 8);
        assert!(daytime_stress(&night, &waking_window_as_spans()).buckets.is_empty());
        let r = sleep_stress(&night);
        assert_eq!(r.buckets.len(), 8);
        assert!(r.suppressed.is_empty());
    }

    // ── the activity gate ──

    #[test]
    fn a_workout_bucket_never_reaches_the_calm_reference() {
        // Without the gate the 18:00 workout hour is scored AND anchors the calm quartiles; with it,
        // the reference is built from the desk hours alone and every remaining score moves.
        let mut points = hours(6..22, 70.0, 40.0);
        points[8] = HourPoint { motion_g: Some(0.9), mean_hr: Some(150.0), rmssd: Some(12.0), ..points[8] };
        let gated = daytime_stress(&points, &waking_window_as_spans());
        let ungated = windowed_stress(&points, StressWindowCfg {
            bucket_seconds: HOUR_SECONDS, sleep_spans: &waking_window_as_spans(),
            activity_gate_g: None, sustained_buckets: SUSTAINED_BUCKETS,
        });
        assert_eq!(gated.suppressed_count(Suppression::Active), 1);
        assert!(ungated.suppressed.is_empty());
        assert_eq!(gated.buckets.len() + 1, ungated.buckets.len());
        // Every calm hour is identical, so the gated day is flat-neutral; the ungated one is dragged.
        assert!(gated.buckets.iter().all(|b| (b.stress - 1.5).abs() < 1e-12));
        assert!(ungated.buckets.iter().any(|b| (b.stress - 1.5).abs() > 0.1));
    }

    #[test]
    fn the_activity_gate_is_inclusive_at_its_edge() {
        let at = vec![HourPoint { motion_g: Some(ACTIVITY_GATE_G), ..point(10, Some(70.0), Some(40.0)) }];
        let below = vec![HourPoint { motion_g: Some(ACTIVITY_GATE_G - 1e-9), ..point(10, Some(70.0), Some(40.0)) }];
        assert_eq!(daytime_stress(&at, &[]).suppressed_count(Suppression::Active), 1);
        assert!(daytime_stress(&below, &[]).suppressed.is_empty());
    }

    #[test]
    fn motion_wins_over_sleep_when_a_bucket_is_both() {
        let points = vec![HourPoint { motion_g: Some(0.5), ..point(2, Some(60.0), Some(50.0)) }];
        let r = daytime_stress(&points, &waking_window_as_spans());
        assert_eq!(r.suppressed[0].suppression, Suppression::Active);
    }

    #[test]
    fn an_absent_motion_channel_suppresses_nothing() {
        let r = daytime_stress(&hours(6..22, 70.0, 40.0), &[]);
        assert!(r.suppressed.is_empty());
        assert_eq!(r.buckets.len(), 16);
    }

    #[test]
    fn a_fully_suppressed_window_still_reports_why() {
        let r = daytime_stress(&hours(0..6, 60.0, 50.0), &waking_window_as_spans());
        assert!(r.buckets.is_empty() && r.mean.is_none() && r.peak_start_ms.is_none());
        assert_eq!(r.suppressed_count(Suppression::Asleep), 6);
        assert_eq!((r.low_minutes, r.medium_minutes, r.high_minutes), (0, 0, 0));
    }

    // ── the night derivation ──

    #[test]
    fn night_crossing_midnight_keeps_every_bucket() {
        // 22:00 → 05:00 in wall order, ordered by start_ms rather than hour-of-day.
        let spans: Vec<i32> = vec![22, 23, 0, 1, 2, 3, 4, 5];
        let points: Vec<HourPoint> = spans.iter().enumerate()
            .map(|(i, &h)| HourPoint {
                start_ms: DAY_START_MS + 22 * HOUR_MS + i as i64 * HOUR_MS,
                ..point(h, Some(60.0 + h as f64 % 3.0), Some(50.0))
            })
            .collect();
        let r = sleep_stress(&points);
        assert_eq!(r.buckets.len(), spans.len());
        assert_eq!(r.buckets.iter().map(|b| b.hour).collect::<Vec<_>>(), spans);
    }

    #[test]
    fn uniformly_stressful_night_still_reads_neutral() {
        // The reference is the night's own calm quartile, so a flat high-HR night has nothing to
        // look stressed against. The caveat, asserted rather than described.
        let points: Vec<HourPoint> = (0..8).map(|h| point(h, Some(95.0), Some(12.0))).collect();
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
    fn no_gates_and_a_covering_span_set_agree() {
        let points = hours(6..22, 70.0, 40.0);
        let a = windowed_stress(&points, StressWindowCfg {
            bucket_seconds: HOUR_SECONDS, sleep_spans: &[],
            activity_gate_g: None, sustained_buckets: SUSTAINED_BUCKETS,
        });
        let b = daytime_stress(&points, &waking_window_as_spans());
        assert_eq!(a.buckets.len(), b.buckets.len());
        for (x, y) in a.buckets.iter().zip(b.buckets.iter()) {
            assert_eq!(x.stress.to_bits(), y.stress.to_bits());
        }
    }

    #[test]
    fn bucket_seconds_sizes_the_band_minutes_and_the_sleep_test() {
        // Quarter-hour buckets: the minutes shrink to match, and a 15-min span now catches exactly
        // one bucket where an hourly bucket would have straddled it.
        let points: Vec<HourPoint> = (0..8)
            .map(|i| HourPoint {
                start_ms: DAY_START_MS + 10 * HOUR_MS + i * 900_000,
                ..point(10, Some(70.0), Some(40.0))
            })
            .collect();
        let quarter = StressWindowCfg {
            bucket_seconds: 900,
            sleep_spans: &[SpanMs {
                start_ms: DAY_START_MS + 10 * HOUR_MS + 900_000,
                end_ms: DAY_START_MS + 10 * HOUR_MS + 1_800_000,
            }],
            activity_gate_g: None,
            sustained_buckets: SUSTAINED_BUCKETS,
        };
        let r = windowed_stress(&points, quarter);
        assert_eq!(r.suppressed.len(), 1, "only the 10:15 bucket lies inside the span");
        assert_eq!(r.buckets.len(), 7);
        assert_eq!(r.medium_minutes, 7 * 15);
        assert_eq!(r.peak_start_ms, Some(r.buckets.last().unwrap().start_ms));
    }

    #[test]
    fn sustained_buckets_is_the_threshold_the_trailing_run_must_reach() {
        // The run is measured, then the threshold walked across it. Only the `99` half of this was
        // asserted before, and `!sustained_high` at 99 is true for a detector that never sustains.
        let mut points = hours(6..14, 70.0, 40.0);
        points.extend(hours(14..18, 90.0, 25.0));
        let spans = waking_window_as_spans();
        let at = |n: usize| windowed_stress(&points, StressWindowCfg {
            bucket_seconds: HOUR_SECONDS, sleep_spans: &spans,
            activity_gate_g: Some(ACTIVITY_GATE_G), sustained_buckets: n,
        });
        // 12 scored hours: eight desk hours at the neutral 1.5, then four spiky ones above the floor.
        let run = at(SUSTAINED_BUCKETS).sustained_run;
        assert_eq!(run, 4);
        assert_eq!(at(SUSTAINED_BUCKETS).buckets.len(), 12);
        assert!(at(SUSTAINED_BUCKETS).buckets[..8].iter().all(|b| b.stress < HIGH_BAND_FLOOR));
        assert!(at(SUSTAINED_BUCKETS).buckets[8..].iter().all(|b| b.stress >= HIGH_BAND_FLOOR));
        // The threshold is `>=`, so it flips exactly between the run and the run plus one. A
        // detector that always answers true fails at run+1; one that always answers false fails at
        // run, so neither constant can pass this pair.
        assert!(at(run).sustained_high, "a run of {run} must satisfy a threshold of {run}");
        assert!(!at(run + 1).sustained_high, "a run of {run} must not satisfy {}", run + 1);
        assert!(!at(99).sustained_high);
        // The run counts TRAILING high buckets only: one calm hour at the end ends it, and the four
        // high hours before it no longer count.
        let mut trailing_calm = points.clone();
        trailing_calm.push(point(18, Some(70.0), Some(40.0)));
        let ended = daytime_stress(&trailing_calm, &spans);
        assert_eq!(ended.sustained_run, 0);
        assert!(!ended.sustained_high);
        assert_eq!(ended.high_minutes, 4 * 60, "the four high hours are still in the band tally");
    }
}
