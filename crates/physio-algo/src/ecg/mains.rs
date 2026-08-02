//! The mains anchor: recover an unknown sample rate from the power-line interference in the samples.
//!
//! European mains is exactly 50 Hz. If a narrow peak sits at normalised frequency `f` cycles per
//! sample, then `fs = 50 / f`, with no firmware constant involved anywhere. That is the whole idea, and
//! it is worth stating precisely what it can and cannot deliver.
//!
//! **It resolves a harmonic mistaken for the fundamental.** Picking the 100 Hz peak yields an `fs`
//! twice too high. The test that rejects it is [`is_harmonic_of`]: a lower, comparably prominent peak
//! `g` with `fold(k·g)` landing on the candidate. It has to be fold-aware, because a harmonic past
//! Nyquist no longer sits at an integer multiple of the observed fundamental.
//!
//! **It provably does NOT resolve aliasing of the fundamental, and the harmonics make that worse
//! rather than better.** A component at normalised frequency `a` is observed at `fold(a) =
//! |a − round(a)|`, and for any integer `k`, `fold(k·fold(a)) = fold(k·a)` identically: `fold(a) =
//! ±a + m`, so `k·fold(a) = ±k·a + k·m`, and both the sign flip and the integer offset are invisible
//! to `fold`. The harmonics of an aliased fundamental therefore land exactly where an un-aliased
//! hypothesis predicts them. Measured here: 50/100/150 Hz sampled at 80 Hz folds to 0.375/0.25/0.125,
//! which is a textbook 1:2:3 ladder based at 0.125 — so this returns 400 Hz, with both harmonics
//! confirming, from a stream actually sampled at 80. Harmonic agreement does not merely fail to
//! detect the alias; it endorses the wrong answer.
//!
//! So `fs = mains / f` holds only under the assumption `fs > 2 × mains`, and **that assumption cannot
//! be tested from the spectrum**. It has to be closed from outside: at 80 Hz a QRS is barely
//! representable, so the morphology indices in `ecg::score` reject that candidate on their own. The
//! anchor is one of two independent methods and is not self-validating.
//!
//! [`MainsFix::alias_fs_hz`] lists the rates that fold the mains fundamental onto the peak that was
//! picked. When the peak really is the fundamental's image that list contains the truth; when the
//! fundamental was itself aliased past a harmonic, as above, it does not.
//!
//! **It needs hum to exist.** A stream the strap has already filtered may carry none at all — a 50 Hz
//! notch is standard in ECG front ends, and the filtered stream is the likely case here. Then this
//! returns [`MainsAnchor::Unavailable`], never a guess. The anchor is a raw-stream instrument.

use crate::ecg::spectrum::Periodogram;
use crate::signal::find_peaks;
use crate::stats::median;

/// European power-line frequency (Hz). Exact by grid regulation, which is what makes it a time standard.
pub const MAINS_HZ_EU: f64 = 50.0;

/// Grid points per DFT bin in the scan. The Hann main lobe is four bins wide, so four points per bin
/// puts >= 16 samples across a peak — enough for the parabolic refinement to be meaningful.
const OVERSAMPLE: usize = 4;
/// Half-width of the neighbourhood the local noise floor is taken over, and the guard around the peak
/// that is excluded from it, both in grid points.
const FLOOR_HALF_WIDTH: usize = 30 * OVERSAMPLE;
const FLOOR_GUARD: usize = 3 * OVERSAMPLE;
/// Prominence a predicted harmonic must reach to count as confirming.
const HARMONIC_CONFIRM_DB: f64 = 6.0;
/// Prominence at which the confidence term saturates.
const PROMINENCE_SATURATION_DB: f64 = 30.0;
/// Rates below this are not reported in the alias ladder — nothing samples an ECG that slowly.
const MIN_LADDER_FS_HZ: f64 = 20.0;
/// Shortest window a rate is read from. One DFT bin is `mains / (f² · N)` Hz of `fs`, so at 512
/// samples and `f = 0.1` the floor is already 10 Hz of `fs`; shorter than this the answer is a bin
/// index, not a rate.
const MIN_MAINS_SAMPLES: usize = 512;

/// Search settings. `fs_min_hz` must exceed `2 × mains_hz`: at or below it the fundamental itself
/// aliases and, per the module note, that is not recoverable from the spectrum.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MainsConfig {
    pub mains_hz: f64,
    pub fs_min_hz: f64,
    pub fs_max_hz: f64,
    /// Peak-to-local-floor contrast a candidate must reach. DERIVED from the corpus, not published:
    /// over 13 subjects the real 50 Hz peak that gave the right rate ran 17.6-37.2 dB, while the one
    /// peak that gave a wrong rate sat at 12.1 dB. The default sits in that gap. n = 13, so treat it
    /// as an operating point rather than a constant of nature.
    pub min_prominence_db: f64,
    /// How far the winner must beat the best candidate implying a different rate, in dB.
    pub min_margin_db: f64,
}

impl Default for MainsConfig {
    fn default() -> Self {
        MainsConfig {
            mains_hz: MAINS_HZ_EU,
            fs_min_hz: 110.0,
            fs_max_hz: 1000.0,
            min_prominence_db: 15.0,
            min_margin_db: 3.0,
        }
    }
}

/// A recovered rate, with everything the decision rested on kept visible.
#[derive(Clone, Debug, PartialEq)]
pub struct MainsFix {
    /// Peak position in cycles per sample, parabolically refined.
    pub normalised_freq: f64,
    /// `mains_hz / normalised_freq`, valid only if the fundamental is not aliased.
    pub fs_hz: f64,
    /// 0..1 from prominence and how many harmonics confirmed. Not a probability.
    pub confidence: f64,
    pub prominence_db: f64,
    /// Prominence at the predicted 2nd and 3rd harmonic positions; `None` when the predicted position
    /// is too near DC or Nyquist to be resolved.
    pub harmonic_db: [Option<f64>; 2],
    /// How many of those two cleared [`HARMONIC_CONFIRM_DB`]. Zero means the peak is a lone tone that
    /// merely sits where mains would: the rate is reported, but nothing corroborates it.
    pub harmonics_confirmed: usize,
    /// The next candidate implying a different rate, and how far below the winner it sits.
    pub runner_up_fs_hz: Option<f64>,
    pub margin_db: f64,
    /// One DFT bin expressed in Hz of `fs` at this window length — the resolution floor of the
    /// estimate, not a measured error.
    pub fs_bin_hz: f64,
    /// Rates whose mains fundamental folds onto this peak, highest first; `fs_hz` is the first. It
    /// contains the truth only if the peak really is the fundamental's image — see the module note.
    pub alias_fs_hz: Vec<f64>,
}

/// Why no rate was returned.
#[derive(Clone, Debug, PartialEq)]
pub enum MainsUnavailable {
    /// Too short to resolve a peak at the configured search range.
    TooFewSamples,
    /// Nothing in the search range stands far enough above its local floor — the filtered-stream case.
    NoPeak { best_prominence_db: f64 },
    /// Two or more candidates imply different rates and are too close to separate.
    Ambiguous { candidates_fs_hz: Vec<f64>, margin_db: f64 },
}

/// A rate, or a named reason there is none.
#[derive(Clone, Debug, PartialEq)]
pub enum MainsAnchor {
    Found(MainsFix),
    Unavailable(MainsUnavailable),
}

impl MainsAnchor {
    pub fn fix(&self) -> Option<&MainsFix> {
        match self {
            MainsAnchor::Found(f) => Some(f),
            MainsAnchor::Unavailable(_) => None,
        }
    }
}

/// Recover the sample rate from mains interference, with [`MainsConfig::default`].
pub fn mains_anchor(samples: &[f64]) -> MainsAnchor {
    mains_anchor_with(samples, MainsConfig::default())
}

/// Recover the sample rate from mains interference. Never guesses: every path that cannot separate a
/// peak returns [`MainsAnchor::Unavailable`] with the reason.
pub fn mains_anchor_with(samples: &[f64], cfg: MainsConfig) -> MainsAnchor {
    use MainsUnavailable::*;
    if !cfg.mains_hz.is_finite() || cfg.mains_hz <= 0.0 || cfg.fs_min_hz <= 2.0 * cfg.mains_hz || cfg.fs_max_hz <= cfg.fs_min_hz {
        return MainsAnchor::Unavailable(TooFewSamples);
    }
    if samples.len() < MIN_MAINS_SAMPLES {
        return MainsAnchor::Unavailable(TooFewSamples);
    }
    let pg = Periodogram::new(samples);
    if pg.is_empty() {
        return MainsAnchor::Unavailable(TooFewSamples);
    }
    let bin = pg.bin_width();
    let step = bin / OVERSAMPLE as f64;
    // Candidates live in [mains/fs_max, mains/fs_min]; the scan reaches a third of the way below that
    // so a candidate's own subharmonic is visible, and stops short of Nyquist where a peak cannot be
    // resolved from its own mirror.
    let cand_lo = (cfg.mains_hz / cfg.fs_max_hz).max(4.0 * bin);
    let cand_hi = (cfg.mains_hz / cfg.fs_min_hz).min(0.5 - 2.0 * bin);
    let scan_lo = (cand_lo / 3.0).max(2.0 * bin);
    if cand_hi <= cand_lo || scan_lo >= cand_hi {
        return MainsAnchor::Unavailable(TooFewSamples);
    }
    let scan = pg.scan(scan_lo, cand_hi, step);
    if scan.len() < 4 * FLOOR_GUARD {
        return MainsAnchor::Unavailable(TooFewSamples);
    }
    let powers: Vec<f64> = scan.iter().map(|(_, p)| *p).collect();

    let peak_idx = find_peaks(&powers, FLOOR_GUARD, f64::NEG_INFINITY);
    let mut peaks: Vec<(f64, f64)> = peak_idx
        .iter()
        .map(|&i| (refine(&scan, i, step), prominence_db(powers[i], local_floor(&powers, i))))
        .collect();
    peaks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let best_overall = peaks.first().map(|p| p.1).unwrap_or(f64::NEG_INFINITY);
    // Fundamentals only: inside the candidate band, prominent enough, and with no stronger peak at
    // half or a third of the frequency (which would make this one that peak's harmonic).
    let mut fundamentals: Vec<(f64, f64)> = peaks
        .iter()
        .copied()
        .filter(|&(f, prom)| {
            f >= cand_lo && f <= cand_hi && prom >= cfg.min_prominence_db && !is_harmonic_of(&peaks, f, cfg.min_prominence_db, bin)
        })
        .collect();
    if fundamentals.is_empty() {
        return MainsAnchor::Unavailable(NoPeak { best_prominence_db: round3(best_overall) });
    }
    fundamentals.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let (f0, prom) = fundamentals[0];
    let fs = cfg.mains_hz / f0;
    // A runner-up is only a competitor if it implies a materially different rate; two grid points on
    // the same peak are not two answers.
    let runner = fundamentals[1..]
        .iter()
        .find(|(f, _)| (f - f0).abs() > 4.0 * bin)
        .copied();
    let margin = match runner {
        Some((_, p)) => prom - p,
        None => f64::INFINITY,
    };
    if let Some((rf, _)) = runner {
        if margin < cfg.min_margin_db {
            return MainsAnchor::Unavailable(Ambiguous {
                candidates_fs_hz: vec![round3(fs), round3(cfg.mains_hz / rf)],
                margin_db: round3(margin),
            });
        }
    }

    let harmonic_db =
        [harmonic_prominence(&pg, f0, 2.0, step, bin), harmonic_prominence(&pg, f0, 3.0, step, bin)];
    let confirmed = harmonic_db.iter().filter(|d| d.is_some_and(|v| v >= HARMONIC_CONFIRM_DB)).count();
    let resolvable = harmonic_db.iter().filter(|d| d.is_some()).count();
    MainsAnchor::Found(MainsFix {
        normalised_freq: f0,
        fs_hz: fs,
        confidence: confidence(prom, confirmed, resolvable, cfg.min_prominence_db),
        prominence_db: round3(prom),
        harmonic_db: [harmonic_db[0].map(round3), harmonic_db[1].map(round3)],
        harmonics_confirmed: confirmed,
        runner_up_fs_hz: runner.map(|(rf, _)| round3(cfg.mains_hz / rf)),
        margin_db: round3(margin),
        fs_bin_hz: round3(cfg.mains_hz * bin / (f0 * f0)),
        alias_fs_hz: alias_ladder(cfg.mains_hz, f0),
    })
}

/// Where a component at normalised frequency `a` is observed after sampling: the distance to the
/// nearest integer, in `[0, 0.5]`.
pub fn fold(a: f64) -> f64 {
    (a - a.round()).abs()
}

/// Every sample rate whose mains fundamental folds to `f`, highest first. The head is the un-aliased
/// reading; the tail is what this method cannot rule out.
fn alias_ladder(mains_hz: f64, f: f64) -> Vec<f64> {
    let mut out = Vec::new();
    for k in 0..4 {
        for a in [k as f64 + f, k as f64 - f] {
            if a > 0.0 {
                let fs = mains_hz / a;
                if fs >= MIN_LADDER_FS_HZ && !out.iter().any(|v: &f64| (v - fs).abs() < 1e-6) {
                    out.push(fs);
                }
            }
        }
    }
    out.sort_by(|a, b| b.partial_cmp(a).unwrap());
    out.iter().map(|v| round3(*v)).collect()
}

/// Median power over a neighbourhood, excluding a guard band around the peak itself.
fn local_floor(powers: &[f64], idx: usize) -> f64 {
    let lo = idx.saturating_sub(FLOOR_HALF_WIDTH);
    let hi = (idx + FLOOR_HALF_WIDTH).min(powers.len() - 1);
    let around: Vec<f64> = (lo..=hi)
        .filter(|&i| i.abs_diff(idx) > FLOOR_GUARD)
        .map(|i| powers[i])
        .collect();
    if around.is_empty() {
        0.0
    } else {
        median(&around)
    }
}

fn prominence_db(peak: f64, floor: f64) -> f64 {
    if peak <= 0.0 || floor <= 0.0 {
        return f64::NEG_INFINITY;
    }
    10.0 * (peak / floor).log10()
}

/// Parabolic refinement of a peak position over log power; falls back to the grid point when the fit
/// is not concave.
fn refine(scan: &[(f64, f64)], idx: usize, step: f64) -> f64 {
    let f0 = scan[idx].0;
    if idx == 0 || idx + 1 >= scan.len() {
        return f0;
    }
    let lg = |p: f64| if p > 0.0 { p.ln() } else { f64::NEG_INFINITY };
    let (y0, y1, y2) = (lg(scan[idx - 1].1), lg(scan[idx].1), lg(scan[idx + 1].1));
    let denom = y0 - 2.0 * y1 + y2;
    if !denom.is_finite() || denom >= 0.0 {
        return f0;
    }
    f0 + step * (0.5 * (y0 - y2) / denom).clamp(-1.0, 1.0)
}

/// `true` when some lower peak `g` is itself a credible fundamental (prominence at or above the accept
/// floor) and `fold(k·g) ≈ f` for k of 2 or 3 — the signature of `f` being that peak's harmonic.
///
/// The comparison is against the accept floor, not against the candidate's own prominence, because
/// prominence is measured against a LOCAL noise floor: a harmonic sitting in the quiet part of the
/// spectrum routinely out-scores the stronger fundamental buried in the signal's own band. Measured on
/// one subject at 500 Hz the 100 Hz harmonic reached 77 dB against the fundamental's 35.
///
/// Fold-aware on purpose. Once a harmonic passes Nyquist it no longer sits at an integer multiple of
/// the observed fundamental (50 Hz at 256 sits at 0.195, its 3rd harmonic folds to 0.414, and
/// 0.414/3 = 0.138 is nowhere), so a plain `f/2, f/3` test misses exactly the case that matters. The
/// `g < f` clause settles the mutual case — at `f = 0.2` the 2nd harmonic at 0.4 folds back onto 0.2,
/// so without it a fundamental would reject itself.
fn is_harmonic_of(peaks: &[(f64, f64)], f: f64, min_prom_db: f64, bin: f64) -> bool {
    peaks.iter().any(|&(g, gp)| {
        g < f - 2.0 * bin
            && gp >= min_prom_db
            && (2..=3).any(|k| (fold(k as f64 * g) - f).abs() <= k as f64 * 2.0 * bin)
    })
}

/// Prominence at the `k`-th harmonic of `f0`, after folding it back below Nyquist. `None` when the
/// folded position sits within two bins of DC or Nyquist, or when it lands back on `f0` itself — at
/// `f0 = 0.25` the 3rd harmonic folds to 0.25, and reporting the fundamental's own peak as its
/// corroboration is a confirmation of nothing.
fn harmonic_prominence(pg: &Periodogram, f0: f64, k: f64, step: f64, bin: f64) -> Option<f64> {
    let f = fold(k * f0);
    if f <= 2.0 * bin || f >= 0.5 - 2.0 * bin || (f - f0).abs() <= 4.0 * bin {
        return None;
    }
    let half = FLOOR_HALF_WIDTH;
    let lo = (f - half as f64 * step).max(bin);
    let hi = (f + half as f64 * step).min(0.5);
    let local = pg.scan(lo, hi, step);
    if local.len() < 4 * FLOOR_GUARD {
        return None;
    }
    // The harmonic's position inherits k times the fundamental's error, so take the strongest point
    // within a few bins of the prediction rather than the exact one.
    let peak = local
        .iter()
        .filter(|(lf, _)| (lf - f).abs() <= 3.0 * bin)
        .map(|(_, p)| *p)
        .fold(0.0f64, f64::max);
    let floor: Vec<f64> = local.iter().filter(|(lf, _)| (lf - f).abs() > 3.0 * bin).map(|(_, p)| *p).collect();
    if floor.is_empty() {
        return None;
    }
    Some(prominence_db(peak, median(&floor)))
}

/// Prominence above the accept threshold, saturating, discounted by how the harmonics went.
///
/// "Resolvable" and "confirmed" are different things. At `f = 0.25` (any `fs` of four times mains) the
/// 2nd harmonic sits exactly on Nyquist and the 3rd folds back onto the fundamental, so neither CAN
/// corroborate; that geometry gets the neutral factor rather than the penalty an absent-but-lookable
/// harmonic earns.
fn confidence(prom_db: f64, confirmed: usize, resolvable: usize, min_db: f64) -> f64 {
    let span = (PROMINENCE_SATURATION_DB - min_db).max(1.0);
    let base = ((prom_db - min_db) / span).clamp(0.0, 1.0);
    let factor = match (resolvable, confirmed) {
        (0, _) => 0.8,
        (r, c) if c == r => 1.0,
        (_, 0) => 0.6,
        _ => 0.8,
    };
    round3(base * factor)
}

fn round3(v: f64) -> f64 {
    if v.is_finite() {
        (v * 1000.0).round() / 1000.0
    } else {
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn lcg(seed: u64) -> impl FnMut() -> f64 {
        let mut s = seed;
        move || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        }
    }

    /// Broadband noise plus mains at `mains_hz` with its 2nd and 3rd harmonic at 1/2 and 1/3 amplitude.
    fn hum(n: usize, fs: f64, mains_hz: f64, amp: f64, seed: u64) -> Vec<f64> {
        let mut rnd = lcg(seed);
        (0..n)
            .map(|i| {
                let t = i as f64 / fs;
                let mut v = rnd();
                for (k, scale) in [(1.0, 1.0), (2.0, 0.5), (3.0, 1.0 / 3.0)] {
                    v += amp * scale * (2.0 * PI * k * mains_hz * t).sin();
                }
                v
            })
            .collect()
    }

    #[test]
    fn recovers_a_known_rate_from_injected_hum() {
        for fs in [200.0, 250.0, 256.0, 500.0, 512.0] {
            let x = hum(4096, fs, 50.0, 0.5, 3);
            let fix = mains_anchor(&x).fix().cloned().unwrap_or_else(|| panic!("no fix at {fs} Hz"));
            let err = (fix.fs_hz - fs).abs();
            assert!(err < 0.5, "fs {fs}: got {} (err {err:.3} Hz)", fix.fs_hz);
            assert!(fix.confidence > 0.5, "fs {fs}: confidence {}", fix.confidence);
        }
    }

    #[test]
    fn no_hum_is_unavailable_not_a_guess() {
        // The filtered-stream case: broadband noise with no line component at all.
        let mut rnd = lcg(9);
        let x: Vec<f64> = (0..4096).map(|_| rnd()).collect();
        match mains_anchor(&x) {
            MainsAnchor::Unavailable(MainsUnavailable::NoPeak { best_prominence_db }) => {
                assert!(best_prominence_db < MainsConfig::default().min_prominence_db);
            }
            other => panic!("noise must not yield a rate: {other:?}"),
        }
        // And a real signal with the line notched out of it likewise.
        let mut rnd = lcg(10);
        let notched: Vec<f64> = (0..4096)
            .map(|i| rnd() + 2.0 * (2.0 * PI * 10.0 * i as f64 / 200.0).sin())
            .collect();
        assert!(mains_anchor(&notched).fix().is_none(), "a 10 Hz tone is not mains");
    }

    #[test]
    fn a_harmonic_loses_to_its_own_fundamental() {
        // Both present at fs = 500. The 100 Hz peak sits at f = 0.2 and would read as fs = 250, which
        // is inside the search range, so the wrong answer is available and has to lose on merit.
        let fs = 500.0;
        let fix = mains_anchor(&hum(4096, fs, 50.0, 0.5, 5)).fix().cloned().unwrap();
        assert!((fix.fs_hz - fs).abs() < 1.0, "harmonic beat the fundamental: {fix:?}");
        assert_eq!(fix.harmonics_confirmed, 2);
    }

    #[test]
    fn a_lone_tone_is_reported_but_nothing_corroborates_it() {
        // Only 100 Hz at fs = 500, no fundamental and no other harmonic. Nothing in the spectrum can
        // say whether this is the 2nd harmonic of 50 Hz at 500, or 50 Hz itself at 250 — the method's
        // premise is that the peak IS the fundamental, so it reads 250 and says so with no support.
        let mut rnd = lcg(5);
        let only_2nd: Vec<f64> = (0..4096).map(|i| rnd() + 0.5 * (2.0 * PI * 100.0 * i as f64 / 500.0).sin()).collect();
        let fix = mains_anchor(&only_2nd).fix().cloned().unwrap();
        assert!((fix.fs_hz - 250.0).abs() < 1.0, "{fix:?}");
        assert_eq!(fix.harmonics_confirmed, 0, "a lone tone must corroborate nothing: {fix:?}");
        assert!(fix.confidence <= 0.6, "unconfirmed must be discounted: {fix:?}");
    }

    #[test]
    fn aliasing_of_the_fundamental_produces_a_confident_wrong_rate() {
        // fold(k*fold(a)) == fold(k*a) for every k: an aliased ladder is indistinguishable from a real
        // one, so the harmonic check cannot detect this and in fact endorses the wrong answer.
        for &a in &[0.13, 0.42, 0.625, 1.3, 2.9] {
            for k in 1..6 {
                let (lhs, rhs) = (fold(k as f64 * fold(a)), fold(k as f64 * a));
                assert!((lhs - rhs).abs() < 1e-12, "a={a} k={k}: {lhs} vs {rhs}");
            }
        }
        // 50/100/150 Hz sampled at 80 Hz folds to 0.375 / 0.25 / 0.125 — a clean 1:2:3 ladder based at
        // 0.125, i.e. exactly what un-aliased mains at 400 Hz looks like.
        let fix = mains_anchor(&hum(4096, 80.0, 50.0, 0.5, 4)).fix().cloned().unwrap();
        assert!((fix.fs_hz - 400.0).abs() < 2.0, "expected the alias to read as 400 Hz: {fix:?}");
        assert_eq!(fix.harmonics_confirmed, 2, "and to be fully corroborated: {fix:?}");
        // Both harmonics confirm at 27 and 33 dB, above the 24 dB of the peak taken as fundamental —
        // the ladder is upside down, and nothing in the method looks at that.
        assert!(fix.confidence > 0.6, "and confident: {fix:?}");
        assert!(!fix.alias_fs_hz.iter().any(|v| (v - 80.0).abs() < 0.5), "the ladder cannot recover 80 Hz either");
        // The rate is only refused from outside: at 80 Hz sampling a QRS is not representable, so the
        // morphology indices are what close this hole.
        assert!(!crate::ecg::score(&hum(4096, 80.0, 50.0, 0.5, 4), 400.0).verdict.accepted);
    }

    #[test]
    fn degenerate_inputs_are_named_reasons() {
        assert_eq!(mains_anchor(&[]), MainsAnchor::Unavailable(MainsUnavailable::TooFewSamples));
        assert_eq!(mains_anchor(&[1.0; 64]), MainsAnchor::Unavailable(MainsUnavailable::TooFewSamples));
        let bad = MainsConfig { fs_min_hz: 90.0, ..MainsConfig::default() };
        assert_eq!(
            mains_anchor_with(&[0.0; 4096], bad),
            MainsAnchor::Unavailable(MainsUnavailable::TooFewSamples),
            "fs_min at or below 2x mains is refused, not silently searched"
        );
    }
}
