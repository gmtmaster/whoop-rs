//! Small pure statistics shared by the metrics: mean, sample SD, OLS slope, median, percentile, and the
//! robust pulsatile amplitude (p95 − p5).

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

/// Median: the middle on odd counts, the mean of the two middles on even counts. Caller ensures non-empty.
pub fn median(xs: &[f64]) -> f64 {
    let mut s = xs.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = s.len();
    if n % 2 == 1 {
        s[n / 2]
    } else {
        (s[n / 2 - 1] + s[n / 2]) / 2.0
    }
}

/// Linear-interpolated percentile over an ascending-sorted slice; `p` in 0..=1. Caller ensures non-empty.
pub fn percentile(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
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
    fn median_odd_even() {
        assert_eq!(median(&[3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median(&[4.0, 1.0, 3.0, 2.0]), 2.5);
    }

    #[test]
    fn amplitude_is_p95_minus_p5() {
        let win: Vec<f64> = std::iter::repeat_n(98.0, 10).chain(std::iter::repeat_n(102.0, 10)).collect();
        assert!((amplitude(&win) - 4.0).abs() < 1e-9);
    }
}
