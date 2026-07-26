//! Small pure statistics shared by the metrics: mean, sample/population SD, OLS slope, median, median
//! sample gap, percentile, and the robust pulsatile amplitude (p95 − p5).

/// Arithmetic mean; `0.0` for an empty slice.
pub fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    xs.iter().sum::<f64>() / xs.len() as f64
}

/// Sample standard deviation (n − 1); `0.0` for fewer than two points.
pub fn sample_sd(xs: &[f64]) -> f64 {
    if xs.len() < 2 {
        return 0.0;
    }
    let m = mean(xs);
    let var = xs.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (xs.len() - 1) as f64;
    var.sqrt()
}

/// Population standard deviation (divide by n); `0.0` for an empty slice. The per-night / per-window
/// spread the z-scorers use, as distinct from the n−1 [`sample_sd`] the baselines use.
pub fn population_sd(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    let m = mean(xs);
    let var = xs.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / xs.len() as f64;
    if var < 0.0 {
        0.0
    } else {
        var.sqrt()
    }
}

/// OLS slope of `ys` over x = 0, 1, 2, …; `0.0` for fewer than two points or a degenerate x-spread.
pub fn least_squares_slope(ys: &[f64]) -> f64 {
    if ys.len() < 2 {
        return 0.0;
    }
    let mean_x = (ys.len() - 1) as f64 / 2.0;
    let mean_y = mean(ys);
    let (mut num, mut den) = (0.0, 0.0);
    for (i, &y) in ys.iter().enumerate() {
        let dx = i as f64 - mean_x;
        num += dx * (y - mean_y);
        den += dx * dx;
    }
    if den == 0.0 {
        0.0
    } else {
        num / den
    }
}

/// OLS line of `ys` over x = 0, 1, 2, … as `(slope, intercept)`. `(0.0, mean)` for fewer than two points.
/// The single source for both the slope (`least_squares_slope`) and a full linear detrend.
pub fn least_squares_line(ys: &[f64]) -> (f64, f64) {
    let slope = least_squares_slope(ys);
    let mean_x = ys.len().saturating_sub(1) as f64 / 2.0;
    (slope, mean(ys) - slope * mean_x)
}

/// Median: the middle on odd counts, the mean of the two middles on even counts; `0.0` when empty.
pub fn median(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    let mut s = xs.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = s.len();
    if n % 2 == 1 {
        s[n / 2]
    } else {
        (s[n / 2 - 1] + s[n / 2]) / 2.0
    }
}

/// Median spacing (s) between consecutive timestamps, restricted to plausible `(0, 300)` gaps, floored at
/// 1.0; `fallback` when no plausible gap exists. Not a true median: takes the upper-middle after sort.
/// The one sample-cadence estimate every per-sample duration credit is derived from.
pub fn median_gap_s(times: &[i64], fallback: f64) -> f64 {
    if times.len() < 2 {
        return fallback;
    }
    let mut gaps: Vec<f64> = Vec::new();
    for w in times.windows(2) {
        let g = (w[1] - w[0]) as f64;
        if g > 0.0 && g < 300.0 {
            gaps.push(g);
        }
    }
    if gaps.is_empty() {
        return fallback;
    }
    gaps.sort_by(|a, b| a.partial_cmp(b).unwrap());
    gaps[gaps.len() / 2].max(1.0)
}

/// Linear-interpolated percentile over an ascending-sorted slice; `p` in 0..=1; `0.0` when empty.
pub fn percentile(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        return sorted[0];
    }
    let rank = p * (n - 1) as f64;
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    if lo == hi {
        return sorted[lo];
    }
    let frac = rank - lo as f64;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

/// Robust pulsatile amplitude of a window: p95 − p5, so a lone spike moves neither tail.
pub fn amplitude(xs: &[f64]) -> f64 {
    let mut s = xs.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    percentile(&s, 0.95) - percentile(&s, 0.05)
}

/// Pearson correlation of two equal-length series; `None` for < 2 pairs or a zero-variance series. The
/// per-strap "is this field signal against a known reference" number — computed, never assumed.
pub fn pearson(xs: &[f64], ys: &[f64]) -> Option<f64> {
    let n = xs.len().min(ys.len());
    if n < 2 {
        return None;
    }
    let (mx, my) = (mean(&xs[..n]), mean(&ys[..n]));
    let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
    for i in 0..n {
        let (dx, dy) = (xs[i] - mx, ys[i] - my);
        sxy += dx * dy;
        sxx += dx * dx;
        syy += dy * dy;
    }
    if sxx <= 0.0 || syy <= 0.0 {
        return None;
    }
    Some(sxy / (sxx * syy).sqrt())
}

/// A per-strap linear calibration `reference ≈ scale·field + offset`, with the `r` that says how far to
/// trust it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LinearFit {
    pub scale: f64,
    pub offset: f64,
    pub r: f64,
}

/// Least-squares fit of `reference` onto `field` (`ref ≈ scale·field + offset`) plus the Pearson `r`.
/// `None` for < 2 pairs or a field with no spread. How the client derives a device-specific coefficient
/// from one strap's own captures instead of hardcoding another strap's number.
pub fn linear_fit(field: &[f64], reference: &[f64]) -> Option<LinearFit> {
    let n = field.len().min(reference.len());
    if n < 2 {
        return None;
    }
    let (mf, mr) = (mean(&field[..n]), mean(&reference[..n]));
    let (mut sfr, mut sff) = (0.0, 0.0);
    for i in 0..n {
        let df = field[i] - mf;
        sfr += df * (reference[i] - mr);
        sff += df * df;
    }
    if sff <= 0.0 {
        return None;
    }
    let scale = sfr / sff;
    Some(LinearFit { scale, offset: mr - scale * mf, r: pearson(&field[..n], &reference[..n])? })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mean_sd_slope_basic() {
        assert_eq!(mean(&[2.0, 4.0, 6.0]), 4.0);
        assert!((sample_sd(&[2.0, 4.0, 6.0]) - 2.0).abs() < 1e-12);
        assert!((least_squares_slope(&[1.0, 2.0, 3.0, 4.0]) - 1.0).abs() < 1e-12);
        assert_eq!(least_squares_slope(&[5.0]), 0.0);
    }

    #[test]
    fn least_squares_line_matches_slope_and_recovers_intercept() {
        // y = 2x + 3 over x = 0..4 → slope 2, intercept 3.
        let ys = [3.0, 5.0, 7.0, 9.0, 11.0];
        let (slope, intercept) = least_squares_line(&ys);
        assert!((slope - 2.0).abs() < 1e-12 && (intercept - 3.0).abs() < 1e-12);
        assert!((slope - least_squares_slope(&ys)).abs() < 1e-12);
        assert_eq!(least_squares_line(&[42.0]), (0.0, 42.0)); // < 2 points → (0, mean)
    }

    #[test]
    fn median_odd_even() {
        assert_eq!(median(&[3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median(&[4.0, 1.0, 3.0, 2.0]), 2.5);
        assert_eq!(median(&[]), 0.0); // empty is 0, never an index panic
    }

    #[test]
    fn population_sd_divides_by_n() {
        // [2,4,6]: mean 4, squared devs 4+0+4 = 8 -> /3 = 2.667, sqrt ~1.633 (vs 2.0 for sample_sd).
        assert!((population_sd(&[2.0, 4.0, 6.0]) - (8.0f64 / 3.0).sqrt()).abs() < 1e-12);
        assert_eq!(population_sd(&[]), 0.0);
        assert_eq!(population_sd(&[7.0]), 0.0);
    }

    #[test]
    fn median_gap_uses_upper_middle_and_drops_implausible() {
        assert_eq!(median_gap_s(&[100], 60.0), 60.0); // too few to time -> fallback
        assert_eq!(median_gap_s(&[0, 1, 2, 3], 60.0), 1.0);
        assert_eq!(median_gap_s(&[0, 2, 402], 60.0), 2.0); // the 400 s gap is excluded
        assert_eq!(median_gap_s(&[0, 900], 1.0), 1.0); // no plausible gap -> fallback
    }

    #[test]
    fn percentile_empty_is_zero_not_a_panic() {
        assert_eq!(percentile(&[], 0.5), 0.0);
        assert_eq!(percentile(&[42.0], 0.9), 42.0);
    }

    #[test]
    fn amplitude_is_p95_minus_p5() {
        let win: Vec<f64> = std::iter::repeat_n(98.0, 10).chain(std::iter::repeat_n(102.0, 10)).collect();
        assert!((amplitude(&win) - 4.0).abs() < 1e-9);
    }

    #[test]
    fn linear_fit_recovers_a_known_line() {
        // reference = 2·field + 3, perfectly correlated.
        let field = [1.0, 2.0, 3.0, 4.0, 5.0];
        let reference: Vec<f64> = field.iter().map(|&x| 2.0 * x + 3.0).collect();
        assert!((pearson(&field, &reference).unwrap() - 1.0).abs() < 1e-12);
        let fit = linear_fit(&field, &reference).unwrap();
        assert!((fit.scale - 2.0).abs() < 1e-9 && (fit.offset - 3.0).abs() < 1e-9 && (fit.r - 1.0).abs() < 1e-9);
        // A flat field has no spread — nothing to calibrate.
        assert!(linear_fit(&[5.0, 5.0, 5.0], &[1.0, 2.0, 3.0]).is_none());
    }
}
