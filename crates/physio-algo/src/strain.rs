//! Strain / cardiovascular Effort (0–100) from an HR series: Karvonen %HRR per sample, TRIMP
//! accumulation (Edwards 5-zone or Banister exponential), then a logarithmic map onto [0, 100].
//! Pure; a light or absent series scores an honest 0 and too-little data returns `None`.

pub const MIN_READINGS: usize = 600;
pub const MIN_SPARSE_READINGS: usize = 20;
pub const MIN_SPAN_SECONDS: i64 = 600;
pub const MAX_STRAIN: f64 = 100.0;
pub const STRAIN_DENOMINATOR: f64 = 7201.0;
pub const FALLBACK_SAMPLE_MIN: f64 = 1.0 / 60.0;
pub const DEFAULT_AGE: i32 = 30;
pub const HRMAX_MIN_SAMPLES: usize = 600;
pub const HRMAX_PERCENTILE: f64 = 99.5;
pub const BANISTER_SCALE: f64 = 0.64;
pub const BANISTER_B_MEN: f64 = 1.92;
pub const BANISTER_B_WOMEN: f64 = 1.67;

/// Duration credited to a lone sample: no neighbour on either side to derive an interval from, so
/// one minute is assumed.
pub const LONE_SAMPLE_SECONDS: f64 = 60.0;

/// The other 0–N Day Strain axis an Effort score is read on, when the reader asks for it.
pub const WHOOP_DAY_STRAIN_MAX: f64 = 21.0;

/// Day Strain → Effort, the multiplier an import boundary applies so imported history lands on the
/// same axis as a computed Effort.
pub const WHOOP_DAY_STRAIN_TO_EFFORT: f64 = MAX_STRAIN / WHOOP_DAY_STRAIN_MAX;

/// Effort → Day Strain, for a display that asks for the other axis. Multiplying by this is NOT the
/// same operation as dividing by [`WHOOP_DAY_STRAIN_TO_EFFORT`]; only the division inverts an import
/// exactly, so an export boundary divides.
pub const EFFORT_TO_WHOOP_DAY_STRAIN: f64 = WHOOP_DAY_STRAIN_MAX / MAX_STRAIN;

/// An Effort score on the axis the reader chose: Day Strain when `whoop_axis`, else the native
/// 0–[`MAX_STRAIN`] value unchanged.
pub fn effort_on_axis(value: f64, whoop_axis: bool) -> f64 {
    if whoop_axis { value * EFFORT_TO_WHOOP_DAY_STRAIN } else { value }
}

/// Edwards cut-offs as (%HRR threshold, weight), highest-first.
const EDWARDS_ZONES: [(f64, i64); 5] = [(90.0, 5), (80.0, 4), (70.0, 3), (60.0, 2), (50.0, 1)];

pub use crate::hr_gap::{GapAccounting, GapPosition, GapVerdict};
pub use crate::hr_sample::HrSample;

/// TRIMP accumulation method.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Method {
    Edwards,
    Banister,
}

/// Denominator-fit failure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StrainError {
    TooFewPairs,
    Degenerate,
}

/// Tanaka HRmax = 208 − 0.7 × age (gender-independent).
pub fn tanaka_hrmax(age: f64) -> f64 {
    208.0 - 0.7 * age
}

/// Classic 220 − age; last-resort fallback.
pub fn default_max_hr(age: i32) -> i32 {
    220 - age
}

/// [`stats::percentile`] on the 0..100 scale, for the constants here that are written as percents.
/// The unit is in the name because the two take the SAME argument on different scales, and a
/// bare `percentile` imported from either would compile against both.
///
/// [`stats::percentile`]: crate::stats::percentile
pub fn percentile_pct(sorted_values: &[f64], pct: f64) -> f64 {
    crate::stats::percentile(sorted_values, pct / 100.0)
}

/// Last-resort HRmax (bpm) when the chain has nothing: no caller value, no observed peak, no age. The
/// one fallback in the tree, so no call site holds a second and no two paths gate a person differently.
pub const FALLBACK_HRMAX: f64 = 220.0;

/// The one HRmax any displayed number is scored against: caller → observed peak → Tanaka(age) →
/// [`FALLBACK_HRMAX`], with the source that won ("caller"/"observed"/"tanaka"/"fallback"). Callers that
/// must distinguish "no HRmax at all" read [`estimate_hrmax`] instead, which reports `unknown`.
pub fn resolve_hrmax(caller: Option<f64>, hr_history: &[f64], age: Option<f64>) -> (f64, &'static str) {
    if let Some(m) = caller {
        return (m, "caller");
    }
    match estimate_hrmax(hr_history, age) {
        (_, "unknown") => (FALLBACK_HRMAX, "fallback"),
        resolved => resolved,
    }
}

/// Personalized HRmax from a trailing HR series → (bpm, source ∈ observed/tanaka/unknown).
pub fn estimate_hrmax(hr_history: &[f64], age: Option<f64>) -> (f64, &'static str) {
    let tanaka = age.map(tanaka_hrmax);
    if hr_history.len() >= HRMAX_MIN_SAMPLES {
        let mut sorted = hr_history.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let observed = percentile_pct(&sorted, HRMAX_PERCENTILE);
        return match tanaka {
            None => (observed, "observed"),
            Some(t) if observed >= t => (observed, "observed"),
            Some(t) => (t, "tanaka"),
        };
    }
    match tanaka {
        Some(t) => (t, "tanaka"),
        None => (0.0, "unknown"),
    }
}

/// Karvonen %HRR, clamped [0, 100].
pub fn pct_hrr(bpm: f64, resting_hr: f64, hr_reserve: f64) -> f64 {
    ((bpm - resting_hr) / hr_reserve * 100.0).clamp(0.0, 100.0)
}

/// Edwards 5-zone weight (0–5) from %HRR (unclamped; extremes agree with the clamped path).
pub fn zone_weight(bpm: f64, resting_hr: f64, hr_reserve: f64) -> i64 {
    let pct = (bpm - resting_hr) / hr_reserve * 100.0;
    for (threshold, weight) in EDWARDS_ZONES {
        if pct >= threshold {
            return weight;
        }
    }
    0
}

/// Per-sample duration (minutes) from the first two timestamps; 1 s fallback.
pub fn sample_duration_minutes(hr: &[HrSample]) -> f64 {
    if hr.len() < 2 {
        return FALLBACK_SAMPLE_MIN;
    }
    let delta_s = (hr[1].ts - hr[0].ts).abs() as f64;
    if delta_s > 0.0 {
        delta_s / 60.0
    } else {
        FALLBACK_SAMPLE_MIN
    }
}

pub fn banister_trimp(hr: &[HrSample], resting_hr: f64, hr_reserve: f64, sample_dur_min: f64, b: f64) -> f64 {
    let mut acc = 0.0;
    for s in hr {
        let x = pct_hrr(s.bpm as f64, resting_hr, hr_reserve) / 100.0;
        if x > 0.0 {
            acc += sample_dur_min * x * BANISTER_SCALE * (b * x).exp();
        }
    }
    acc
}

/// Per-interval Edwards TRIMP: each sample's zone weight times its own interval-derived duration
/// (minutes). A sample owns half of each adjacent gap; the end samples have one unmeasured half,
/// granted only within the tighter lead/trail ceiling of [`crate::hr_gap`].
pub fn edwards_trimp_interval(hr: &[HrSample], resting_hr: f64, hr_reserve: f64) -> f64 {
    edwards_trimp_accounted(hr, resting_hr, hr_reserve).0
}

/// [`edwards_trimp_interval`] plus the provenance of the seconds it billed, so a caller can tell how
/// much of the load rests on bridged rather than measured time.
pub fn edwards_trimp_accounted(hr: &[HrSample], resting_hr: f64, hr_reserve: f64) -> (f64, GapAccounting) {
    let n = hr.len();
    let mut acct = GapAccounting::default();
    if n == 0 {
        return (0.0, acct);
    }
    let gap = |a: usize, b: usize| (hr[a].ts - hr[b].ts).unsigned_abs() as f64;
    let mut total = 0.0;
    for (i, sample) in hr.iter().enumerate() {
        let w = zone_weight(sample.bpm as f64, resting_hr, hr_reserve) as f64;
        let seconds = if n == 1 {
            acct.add(LONE_SAMPLE_SECONDS, GapPosition::Trailing);
            LONE_SAMPLE_SECONDS
        } else if i == 0 {
            let fwd = gap(1, 0);
            half(fwd, GapPosition::Interior, &mut acct) + half(fwd, GapPosition::Leading, &mut acct)
        } else if i == n - 1 {
            let bwd = gap(i, i - 1);
            half(bwd, GapPosition::Interior, &mut acct) + half(bwd, GapPosition::Trailing, &mut acct)
        } else {
            half(gap(i + 1, i), GapPosition::Interior, &mut acct)
                + half(gap(i, i - 1), GapPosition::Interior, &mut acct)
        };
        total += w * seconds / 60.0;
    }
    (total, acct)
}

/// One sample's half-share of a gap, filed under the verdict of the WHOLE gap so a refused gap can
/// never be half-credited. Returns the seconds this sample may bill.
fn half(gap_seconds: f64, position: GapPosition, acct: &mut GapAccounting) -> f64 {
    if gap_seconds <= 0.0 {
        return 0.0;
    }
    let share = gap_seconds / 2.0;
    match crate::hr_gap::classify(gap_seconds, position) {
        GapVerdict::Cadence => acct.measured_seconds += share,
        GapVerdict::Bridge => acct.bridged_seconds += share,
        GapVerdict::Refuse => {
            acct.refused_seconds += share;
            return 0.0;
        }
    }
    share
}

/// Map accumulated TRIMP onto [0, 100] via 100 × ln(TRIMP+1) / ln(D), 2 dp. TRIMP ≤ 0 → 0.
pub fn trimp_to_strain(trimp: f64, denominator: f64) -> f64 {
    if trimp <= 0.0 {
        return 0.0;
    }
    let value = MAX_STRAIN * (trimp + 1.0).ln() / denominator.ln();
    (value * 100.0).round() / 100.0
}

/// Calibrate D from (TRIMP, reference_strain) pairs via the through-origin least-squares line:
/// ln(D) = maxStrain × Σ(x²) / Σ(xy), x = ln(TRIMP+1). Reference strains are on the 0–100 scale.
pub fn fit_strain_denominator(pairs: &[(f64, f64)]) -> Result<f64, StrainError> {
    let usable: Vec<(f64, f64)> = pairs.iter().copied().filter(|(t, s)| *t > 0.0 && *s > 0.0).collect();
    if usable.len() < 2 {
        return Err(StrainError::TooFewPairs);
    }
    let mut sum_xx = 0.0;
    let mut sum_xy = 0.0;
    for (trimp, strain) in usable {
        let x = (trimp + 1.0).ln();
        sum_xx += x * x;
        sum_xy += x * strain;
    }
    if !(sum_xy > 0.0 && sum_xx > 0.0) {
        return Err(StrainError::Degenerate);
    }
    Ok((MAX_STRAIN * sum_xx / sum_xy).exp())
}

/// Cardiovascular Effort (0–100) from an HR series, or `None` when there isn't enough data
/// (fewer than [`MIN_READINGS`] samples AND under [`MIN_SPAN_SECONDS`] of coverage) or HRR ≤ 0.
pub fn strain(
    hr: &[HrSample],
    max_hr: Option<f64>,
    resting_hr: f64,
    method: Method,
    sex: &str,
    denominator: f64,
) -> Option<f64> {
    let eff_max = max_hr.unwrap_or_else(|| default_max_hr(DEFAULT_AGE) as f64);
    let enough_data = if hr.len() >= MIN_READINGS {
        true
    } else if hr.len() >= MIN_SPARSE_READINGS {
        let max = hr.iter().map(|s| s.ts).max().unwrap_or(0);
        let min = hr.iter().map(|s| s.ts).min().unwrap_or(0);
        max - min >= MIN_SPAN_SECONDS
    } else {
        false
    };
    if !enough_data || eff_max <= resting_hr {
        return None;
    }

    let hr_reserve = eff_max - resting_hr;
    let trimp = match method {
        Method::Banister => {
            let sample_dur = sample_duration_minutes(hr);
            let b = if sex.to_lowercase().starts_with('f') { BANISTER_B_WOMEN } else { BANISTER_B_MEN };
            banister_trimp(hr, resting_hr, hr_reserve, sample_dur, b)
        }
        Method::Edwards => edwards_trimp_interval(hr, resting_hr, hr_reserve),
    };
    Some(trimp_to_strain(trimp, denominator))
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 0.01;

    fn hr_constant(bpm: i32, n: usize) -> Vec<HrSample> {
        (0..n).map(|i| HrSample { ts: i as i64, bpm }).collect()
    }

    fn hr_every(bpm: i32, n: usize, step_s: i64) -> Vec<HrSample> {
        (0..n).map(|i| HrSample { ts: i as i64 * step_s, bpm }).collect()
    }

    fn eff(hr: &[HrSample], max_hr: f64, resting_hr: f64) -> Option<f64> {
        strain(hr, Some(max_hr), resting_hr, Method::Edwards, "male", STRAIN_DENOMINATOR)
    }

    /// The published values, then the same values rebuilt from the closed form, then the two shapes
    /// a do-nothing map could take: a constant answer and a straight line through the same anchors.
    #[test]
    fn trimp_to_strain_goldens() {
        assert_eq!(trimp_to_strain(0.0, STRAIN_DENOMINATOR), 0.0);
        assert_eq!(trimp_to_strain(-5.0, STRAIN_DENOMINATOR), 0.0);
        assert!((trimp_to_strain(100.0, STRAIN_DENOMINATOR) - 51.96).abs() < EPS);
        assert!((trimp_to_strain(500.0, STRAIN_DENOMINATOR) - 69.99).abs() < EPS);
        assert!((trimp_to_strain(1000.0, STRAIN_DENOMINATOR) - 77.78).abs() < EPS);
        assert!((trimp_to_strain(3600.0, STRAIN_DENOMINATOR) - 92.20).abs() < EPS);
        assert!((trimp_to_strain(7200.0, STRAIN_DENOMINATOR) - 100.0).abs() < EPS);

        // 100 x ln(TRIMP+1) / ln(D), independent of the implementation.
        for trimp in [100.0f64, 500.0, 1000.0, 3600.0, 7200.0] {
            let closed = MAX_STRAIN * (trimp + 1.0).ln() / STRAIN_DENOMINATOR.ln();
            let got = trimp_to_strain(trimp, STRAIN_DENOMINATOR);
            assert!((got - (closed * 100.0).round() / 100.0).abs() < 1e-9, "{trimp}: {got} vs {closed}");
        }

        // A constant scorer holding the 100-TRIMP answer is 40 points out at the top of the range.
        let constant = trimp_to_strain(100.0, STRAIN_DENOMINATOR);
        assert!((trimp_to_strain(7200.0, STRAIN_DENOMINATOR) - constant).abs() > 40.0);
        // A straight line through (0,0) and (7200,100) reads 1.39 where the log map reads 51.96.
        let linear = 100.0 * 100.0 / 7200.0;
        assert!((constant - linear).abs() > 45.0, "the map is logarithmic, not linear");
    }

    /// rest 60, max 160 → HRR 100, so %HRR = bpm − 60. 600 samples 1 s apart bill 1 s each, giving
    /// TRIMP = 600 × weight / 60 = 10 × weight, which is rebuilt here rather than assumed.
    #[test]
    fn edwards_zone_goldens() {
        let v115 = eff(&hr_constant(115, 600), 160.0, 60.0).unwrap(); // zone 1 (50-59%): weight=1
        assert!((v115 - 27.0).abs() < EPS, "115bpm got {v115}");
        let v135 = eff(&hr_constant(135, 600), 160.0, 60.0).unwrap(); // zone 3 (70-79%): weight=3
        assert!((v135 - 38.66).abs() < EPS, "135bpm got {v135}");
        let v155 = eff(&hr_constant(155, 600), 160.0, 60.0).unwrap(); // zone 5 (90-100%): weight=5
        assert!((v155 - 44.27).abs() < EPS, "155bpm got {v155}");

        for (bpm, weight) in [(115, 1.0), (135, 3.0), (155, 5.0)] {
            assert_eq!(zone_weight(bpm as f64, 60.0, 100.0), weight as i64);
            let expected = trimp_to_strain(10.0 * weight, STRAIN_DENOMINATOR);
            let got = eff(&hr_constant(bpm, 600), 160.0, 60.0).unwrap();
            assert!((got - expected).abs() < 1e-9, "{bpm}bpm: {got} vs rebuilt {expected}");
        }

        // Null: a scorer that ignores HR reads the resting series, which is an honest 0 — so a
        // constant scorer cannot sit anywhere that satisfies all three goldens at once.
        assert_eq!(eff(&hr_constant(60, 600), 160.0, 60.0).unwrap(), 0.0);
        assert!(v155 > v135 && v135 > v115, "strain must rise with intensity");
    }

    #[test]
    fn interval_method_handles_cadence_transition() {
        // First gap 30s, remaining 599 samples at 1s — old method inflated, new method correct.
        let mut hr = Vec::new();
        hr.push(HrSample { ts: 0, bpm: 135 });
        for i in 1..600 {
            hr.push(HrSample { ts: 30 + i as i64 - 1, bpm: 135 });
        }
        let v = eff(&hr, 160.0, 60.0).unwrap();
        // Each sample gets its own interval (1s for most, 30s for first). Sum of gaps ≈ 30+599*1=629s.
        // weight=3, total = 3 × 629/60 = 31.45 min. TRIMP = 31.45, strain ≈ ...
        assert!(v < 40.0, "cadence transition should not inflate, got {v}");
        assert!(v > 20.0, "cadence transition should score, got {v}");
    }

    #[test]
    fn interval_method_refuses_a_dropout_past_the_interior_ceiling() {
        // Two clusters with a 1-hour gap: past the 1800 s interior ceiling, so neither bracketing
        // sample bills any of it and the hour is recorded as refused, not credited.
        let mut hr = Vec::new();
        for i in 0..600 { hr.push(HrSample { ts: i as i64, bpm: 135 }); }
        let gap_start = 3600i64;
        for i in 0..600 { hr.push(HrSample { ts: gap_start + i as i64, bpm: 135 }); }
        let v = eff(&hr, 160.0, 60.0).unwrap();
        assert!(v > 0.0, "gapped day still scores");
        let (_, acct) = edwards_trimp_accounted(&hr, 60.0, 100.0);
        // 3001 s gap, split as two half-shares, both refused.
        assert!((acct.refused_seconds - 3001.0).abs() < 1e-9, "got {}", acct.refused_seconds);
        assert_eq!(acct.bridged_seconds, 0.0);
    }

    #[test]
    fn end_gaps_use_a_tighter_ceiling_than_an_interior_one() {
        // The same 450 s gap: leading it is within the 600 s lead ceiling, trailing it is past the
        // 300 s trail ceiling. A single flat cap scores these two series identically.
        let lead = [
            HrSample { ts: 0, bpm: 115 },
            HrSample { ts: 450, bpm: 115 },
            HrSample { ts: 451, bpm: 115 },
        ];
        let trail = [
            HrSample { ts: 0, bpm: 115 },
            HrSample { ts: 1, bpm: 115 },
            HrSample { ts: 451, bpm: 115 },
        ];
        // weight 1 at 55 %HRR, so TRIMP is billed seconds / 60.
        let lead_trimp = edwards_trimp_interval(&lead, 60.0, 100.0);
        let trail_trimp = edwards_trimp_interval(&trail, 60.0, 100.0);
        assert!((lead_trimp - 676.5 / 60.0).abs() < 1e-9, "lead {lead_trimp}");
        assert!((trail_trimp - 451.5 / 60.0).abs() < 1e-9, "trail {trail_trimp}");
        assert!(trail_trimp < lead_trimp, "a trailing gap must be trusted less than a leading one");
    }

    #[test]
    fn single_interval_sample_does_not_crash() {
        let hr = vec![HrSample { ts: 0, bpm: 120 }];
        assert!(eff(&hr, 160.0, 60.0).is_none()); // <600 samples
        // but the function itself shouldn't crash on 1 sample
        let (trimp, acct) = edwards_trimp_accounted(&hr, 60.0, 100.0);
        assert!(trimp > 0.0);
        // A lone sample's minute has no measured interval behind it at all.
        assert_eq!(acct.measured_seconds, 0.0);
        assert_eq!(acct.bridged_seconds, LONE_SAMPLE_SECONDS);
    }

    #[test]
    fn a_regular_stream_bills_nothing_to_bridged_time() {
        let (_, acct) = edwards_trimp_accounted(&hr_constant(135, 600), 60.0, 100.0);
        assert_eq!(acct.bridged_seconds, 0.0);
        assert_eq!(acct.refused_seconds, 0.0);
        assert!((acct.measured_seconds - 600.0).abs() < 1e-9, "got {}", acct.measured_seconds);
    }

    #[test]
    fn uniform_stream_credits_one_second_per_sample() {
        // 600 samples 1 s apart at zone-3 intensity: every gap is 1 s, so TRIMP = 600 × 3 × 1/60 = 30 min.
        let uniform = hr_constant(135, 600);
        let trimp = edwards_trimp_interval(&uniform, 60.0, 100.0);
        assert!((trimp - 30.0).abs() < 0.1, "got {trimp}");
    }

    #[test]
    fn null_when_too_few_or_invalid_hrr() {
        assert!(eff(&hr_constant(135, 599), 160.0, 60.0).is_none());
        assert!(eff(&hr_constant(135, 600), 60.0, 60.0).is_none());
    }

    #[test]
    fn sparse_stream_scores_once_it_spans_enough_time() {
        let sparse = hr_every(155, 30, 30);
        assert!(sparse.last().unwrap().ts - sparse.first().unwrap().ts >= MIN_SPAN_SECONDS);
        assert!(eff(&sparse, 160.0, 60.0).is_some());
    }

    #[test]
    fn sparse_stream_null_under_sample_floor() {
        let too_few = hr_every(155, 5, 200);
        assert!(eff(&too_few, 160.0, 60.0).is_none());
    }

    #[test]
    fn light_day_honestly_scores_zero() {
        assert_eq!(eff(&hr_constant(105, 1200), 184.0, 60.0).unwrap(), 0.0);
        assert_eq!(eff(&hr_every(105, 40, 30), 184.0, 60.0).unwrap(), 0.0);
    }

    #[test]
    fn sparse_stream_scores_real_workout() {
        let s = eff(&hr_every(175, 40, 30), 184.0, 60.0);
        assert!(s.is_some() && s.unwrap() > 0.0);
    }

    /// The non-default TRIMP method, selectable over the FFI. Pins the exponential by value against
    /// its closed form, then the three ways a stand-in could pass a bare `high > low`: routing to
    /// Edwards, ignoring the sex coefficient, and ignoring HR.
    #[test]
    fn banister_reproduces_its_exponential_form_and_reads_the_sex_coefficient() {
        assert!((BANISTER_SCALE - 0.64).abs() < 1e-9, "shipped scale");
        assert!((BANISTER_B_MEN - 1.92).abs() < 1e-9, "shipped male exponent");
        assert!((BANISTER_B_WOMEN - 1.67).abs() < 1e-9, "shipped female exponent");

        // 600 samples 1 s apart at 150 bpm against (60, 190): x = 90/130 %HRR as a fraction.
        let series = hr_constant(150, 600);
        let (rest, reserve) = (60.0f64, 130.0f64);
        let x = pct_hrr(150.0, rest, reserve) / 100.0;
        let dur_min = sample_duration_minutes(&series);
        assert!((dur_min - 1.0 / 60.0).abs() < 1e-12, "1 Hz series bills a 60th of a minute");

        let closed = 600.0 * dur_min * x * BANISTER_SCALE * (BANISTER_B_MEN * x).exp();
        let measured = banister_trimp(&series, rest, reserve, dur_min, BANISTER_B_MEN);
        assert!((measured - 16.740_048_787_290_35).abs() < 1e-9, "male TRIMP {measured}");
        assert!((measured - closed).abs() < 1e-9, "TRIMP {measured} vs closed form {closed}");

        let male = strain(&series, Some(190.0), rest, Method::Banister, "male", STRAIN_DENOMINATOR).unwrap();
        let female = strain(&series, Some(190.0), rest, Method::Banister, "female", STRAIN_DENOMINATOR).unwrap();
        assert!((male - 32.38).abs() < EPS, "male {male}");
        assert!((female - 30.55).abs() < EPS, "female {female}");
        assert!(female < male, "the lower female exponent must lower the score");
        assert!(
            (female - trimp_to_strain(
                banister_trimp(&series, rest, reserve, dur_min, BANISTER_B_WOMEN), STRAIN_DENOMINATOR)).abs() < 1e-9
        );

        // Null: routing Banister to Edwards would read 34.28 on this same series.
        let edwards = strain(&series, Some(190.0), rest, Method::Edwards, "male", STRAIN_DENOMINATOR).unwrap();
        assert!((edwards - 34.28).abs() < EPS, "edwards {edwards}");
        assert!((edwards - male).abs() > 1.5, "Banister must not be Edwards under another name");

        // Null: a series pinned at the resting value carries no load under either exponent.
        let flat = hr_constant(60, 600);
        assert_eq!(banister_trimp(&flat, rest, reserve, dur_min, BANISTER_B_MEN), 0.0);
        assert_eq!(strain(&flat, Some(190.0), rest, Method::Banister, "male", STRAIN_DENOMINATOR), Some(0.0));

        // Monotonic in intensity, which is the only thing the old gate checked.
        let low = strain(&hr_constant(120, 600), Some(184.0), 60.0, Method::Banister, "male", STRAIN_DENOMINATOR);
        let high = strain(&hr_constant(175, 600), Some(184.0), 60.0, Method::Banister, "male", STRAIN_DENOMINATOR);
        assert!(high.unwrap() > low.unwrap());
    }

    /// The two helpers plus the EMPTY-history fallbacks only — both calls pass `&[]`, so the
    /// personalised branch never runs here. That one is `observed_hrmax_needs_a_long_history`.
    #[test]
    fn hrmax_and_percentile() {
        assert!((tanaka_hrmax(30.0) - 187.0).abs() < 1e-9);
        let sorted: Vec<f64> = (0..=100).map(|v| v as f64).collect();
        assert!((percentile_pct(&sorted, 50.0) - 50.0).abs() < 1e-9);
        let (bpm, src) = estimate_hrmax(&[], Some(40.0));
        assert_eq!(src, "tanaka");
        assert!((bpm - tanaka_hrmax(40.0)).abs() < 1e-9, "got {bpm}");
        assert_eq!(estimate_hrmax(&[], None), (0.0, "unknown"));
    }

    /// One chain, one fallback: whatever a caller knows, [`resolve_hrmax`] answers with a number and a
    /// source, so no call site needs a literal of its own. Two paths holding 190 and 220 for the same
    /// person put their activity gates 41 bpm apart and billed the same second two ways.
    #[test]
    fn resolve_hrmax_always_answers_so_no_caller_needs_its_own_fallback() {
        assert_eq!(resolve_hrmax(Some(184.0), &[], None), (184.0, "caller"));
        assert_eq!(resolve_hrmax(Some(184.0), &vec![200.0; HRMAX_MIN_SAMPLES], Some(35.0)).1, "caller");
        assert_eq!(resolve_hrmax(None, &[], Some(40.0)), (tanaka_hrmax(40.0), "tanaka"));
        assert_eq!(resolve_hrmax(None, &vec![200.0; HRMAX_MIN_SAMPLES], None), (200.0, "observed"));
        // Nothing known at all still resolves, and to the ONE constant.
        assert_eq!(resolve_hrmax(None, &[], None), (FALLBACK_HRMAX, "fallback"));
        assert!((FALLBACK_HRMAX - 220.0).abs() < 1e-9, "shipped last-resort HRmax");
        assert!(estimate_hrmax(&[], None).0 == 0.0, "the chain under it still reports 'no HRmax'");
    }

    /// `low` bpm for all but the last `peak_n` of `n` samples, which sit at `peak`.
    fn hr_history(n: usize, low: f64, peak_n: usize, peak: f64) -> Vec<f64> {
        let mut v = vec![low; n - peak_n];
        v.append(&mut vec![peak; peak_n]);
        v
    }

    /// The personalised branch behind the "observed" source a session reports, and the tie-break
    /// that lets a real history outrank Tanaka. The two constants are pinned by value: a fixture
    /// sized from them would move with them and assert nothing.
    #[test]
    fn observed_hrmax_needs_a_long_history() {
        assert_eq!(HRMAX_MIN_SAMPLES, 600, "shipped sample floor");
        assert!((HRMAX_PERCENTILE - 99.5).abs() < 1e-9, "shipped percentile");
        let tanaka_40 = tanaka_hrmax(40.0);

        // 1% of 600 samples at 190 clears the 99.5th percentile.
        let peak_190 = hr_history(600, 120.0, 6, 190.0);
        let (bpm, src) = estimate_hrmax(&peak_190, None);
        assert_eq!(src, "observed");
        assert!((bpm - 190.0).abs() < EPS, "got {bpm}");

        // With an age the higher of the two wins, and keeps its own label.
        let (bpm, src) = estimate_hrmax(&peak_190, Some(40.0));
        assert_eq!(src, "observed", "190 must outrank Tanaka {tanaka_40}");
        assert!((bpm - 190.0).abs() < EPS, "got {bpm}");

        let peak_170 = hr_history(600, 120.0, 6, 170.0);
        let (bpm, src) = estimate_hrmax(&peak_170, Some(40.0));
        assert_eq!(src, "tanaka", "170 is under Tanaka {tanaka_40}");
        assert!((bpm - tanaka_40).abs() < EPS, "got {bpm}");

        // 99.5 and not 100: three spikes in 600 samples barely move the estimate.
        let spikes = hr_history(600, 120.0, 3, 220.0);
        let (bpm, src) = estimate_hrmax(&spikes, None);
        assert_eq!(src, "observed");
        assert!((bpm - 120.5).abs() < EPS, "an artefact must not become HRmax; got {bpm}");

        // One sample under the floor and the branch is off, however high the history runs.
        let short = hr_history(599, 120.0, 599, 190.0);
        assert_eq!(estimate_hrmax(&short, Some(40.0)).1, "tanaka");
        assert_eq!(estimate_hrmax(&short, None), (0.0, "unknown"));
    }

    /// [`fit_strain_denominator`] has no caller in this workspace and no FFI export: every shipped
    /// score uses the fixed [`STRAIN_DENOMINATOR`]. This gate covers the fit as an unused library
    /// function, so a green run says nothing about any number a wearer sees.
    #[test]
    fn unused_denominator_fit_recovers_its_seed_and_refuses_uninformative_pairs() {
        assert!((STRAIN_DENOMINATOR - 7201.0).abs() < 1e-9, "the denominator every shipped score uses");

        // Round-trip: strains generated from D=7201 refit back to ~7201 (through-origin fit).
        let pairs: Vec<(f64, f64)> = [100.0, 500.0, 1000.0, 3600.0]
            .iter()
            .map(|&t| (t, trimp_to_strain(t, STRAIN_DENOMINATOR)))
            .collect();
        let d = fit_strain_denominator(&pairs).unwrap();
        assert!((d - STRAIN_DENOMINATOR).abs() / STRAIN_DENOMINATOR < 0.01, "got {d}");
        assert_eq!(fit_strain_denominator(&[(100.0, 50.0)]), Err(StrainError::TooFewPairs));

        // Null: reference strains that carry no information still return an Ok denominator, 99x the
        // seed. The fit does not refuse it, so a caller must never read Ok as "calibrated".
        let flat: Vec<(f64, f64)> = [100.0, 500.0, 1000.0, 3600.0].iter().map(|&t| (t, 50.0)).collect();
        let flat_d = fit_strain_denominator(&flat).unwrap();
        assert!((flat_d - 713_381.98).abs() < 0.1, "uninformative fit {flat_d}");
        assert!(flat_d > STRAIN_DENOMINATOR * 50.0, "and it is nowhere near the seed");

        // Null: the pairing shuffled. Same TRIMPs, same strains, wrong correspondence.
        let shuffled = [
            (100.0, trimp_to_strain(3600.0, STRAIN_DENOMINATOR)),
            (3600.0, trimp_to_strain(100.0, STRAIN_DENOMINATOR)),
        ];
        let shuffled_d = fit_strain_denominator(&shuffled).unwrap();
        assert!((shuffled_d - 32_297.586_246).abs() < 1e-3, "shuffled fit {shuffled_d}");
        assert!(shuffled_d > STRAIN_DENOMINATOR * 4.0);
    }

    /// The two axis ratios, pinned to the bits the frontend produced before it read them here.
    #[test]
    fn day_strain_axis_ratios_are_exact() {
        assert_eq!(EFFORT_TO_WHOOP_DAY_STRAIN, 0.21);
        assert_eq!(WHOOP_DAY_STRAIN_TO_EFFORT, 4.761904761904762);
        assert_eq!(WHOOP_DAY_STRAIN_TO_EFFORT * EFFORT_TO_WHOOP_DAY_STRAIN, 1.0);
        assert_eq!(1.0 / WHOOP_DAY_STRAIN_TO_EFFORT, EFFORT_TO_WHOOP_DAY_STRAIN);
        assert_eq!(MAX_STRAIN * EFFORT_TO_WHOOP_DAY_STRAIN, WHOOP_DAY_STRAIN_MAX);
    }

    /// Multiplying by the ratio and dividing by its inverse are different operations; only the
    /// division inverts an import exactly, so this pins which one `effort_on_axis` performs.
    #[test]
    fn effort_on_axis_multiplies_and_leaves_the_native_scale_alone() {
        assert_eq!(effort_on_axis(76.193, true), 16.000_529_999_999_998);
        assert_ne!(76.193 * EFFORT_TO_WHOOP_DAY_STRAIN, 76.193 / WHOOP_DAY_STRAIN_TO_EFFORT);
        assert_eq!(effort_on_axis(50.0, true), 10.5);
        assert_eq!(effort_on_axis(12.3, true), 2.583);
        assert_eq!(effort_on_axis(0.0, true), 0.0);
        assert_eq!(effort_on_axis(76.193, false), 76.193);
        assert_eq!(effort_on_axis(100.0, false), 100.0);
    }
}
