//! Poincare (Lorenz) plot dispersion: each R-R interval against the next, summarised by the two axes of
//! the fitted ellipse and by how much of the plane the points actually occupy.
//!
//! Brennan, Palaniswami & Kamen, "Do existing measures of Poincare plot geometry reflect nonlinear
//! features of heart rate variability?", IEEE Trans Biomed Eng 48(11):1342-1347, 2001, for the SD1 / SD2
//! identities. The occupancy measure follows the box-counting reading of the plot used for irregular-rhythm
//! screening (Park et al., "Atrial fibrillation detection by heart rate variability in Poincare plot",
//! BioMedical Engineering OnLine 8:38, 2009): a regular rhythm draws a narrow cigar along the identity
//! line, an irregular one a diffuse cloud, and the count of occupied cells separates them where SD1 and
//! SD2 alone can be matched by a wide but orderly rhythm.

use crate::rr_irregularity::indices::mean_rr_ms;
use crate::stats::sample_sd;

/// Beats needed for a plot: enough successive pairs for a sample SD of the differences.
pub const POINCARE_MIN_BEATS: usize = 8;
/// Grid cell for the occupancy count (ms square). Wide enough that the quantisation of R-R to whole
/// milliseconds cannot on its own scatter a cigar across cells.
pub const POINCARE_CELL_MS: f64 = 25.0;

/// One Poincare plot. `sd1` / `sd2` / `ellipse_area_ms2` are in milliseconds; `normalised_area` divides
/// the area by the squared mean R-R so two people at different heart rates compare. `ratio` is `sd1/sd2`
/// and is `None` on a degenerate plot with no long-axis spread.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Poincare {
    pub sd1: f64,
    pub sd2: f64,
    pub ratio: Option<f64>,
    pub ellipse_area_ms2: f64,
    pub normalised_area: f64,
    /// Occupied [`POINCARE_CELL_MS`] cells over plotted points, in `0.0..=1.0`. A cigar stacks many
    /// points into few cells and reads low; a diffuse cloud gives almost every point its own cell and
    /// reads near 1.0. It rises with scatter and FALLS with record length, so compare it only between
    /// series of similar length.
    pub cell_occupancy: f64,
}

/// Poincare descriptors of an R-R series (ms), in beat order. `None` under [`POINCARE_MIN_BEATS`] or on a
/// non-positive mean R-R. Never panics on constant input: that is a real plot, a single point, and it
/// reports zeros with `ratio` `None`.
pub fn poincare(rr_ms: &[u16]) -> Option<Poincare> {
    if rr_ms.len() < POINCARE_MIN_BEATS {
        return None;
    }
    let mean = mean_rr_ms(rr_ms)?;
    let values: Vec<f64> = rr_ms.iter().map(|&v| f64::from(v)).collect();
    let diffs: Vec<f64> = values.windows(2).map(|w| w[1] - w[0]).collect();

    // SD1 is the spread across the identity line, SD2 the spread along it: SD1^2 = SDSD^2 / 2 and
    // SD2^2 = 2 SDNN^2 - SD1^2.
    let sdnn = sample_sd(&values);
    let sd1 = (sample_sd(&diffs).powi(2) / 2.0).max(0.0).sqrt();
    let sd2 = (2.0 * sdnn * sdnn - sd1 * sd1).max(0.0).sqrt();
    let area = std::f64::consts::PI * sd1 * sd2;

    Some(Poincare {
        sd1,
        sd2,
        ratio: (sd2 > 0.0).then_some(sd1 / sd2),
        ellipse_area_ms2: area,
        normalised_area: area / (mean * mean),
        cell_occupancy: cell_occupancy(&values),
    })
}

/// Distinct [`POINCARE_CELL_MS`] cells the `(rr[i], rr[i+1])` points fall in, over the number of points.
fn cell_occupancy(values: &[f64]) -> f64 {
    let points = values.len().saturating_sub(1);
    if points == 0 {
        return 0.0;
    }
    let cell = |v: f64| (v / POINCARE_CELL_MS).floor() as i64;
    let mut seen: Vec<(i64, i64)> = values
        .windows(2)
        .map(|w| (cell(w[0]), cell(w[1])))
        .collect();
    seen.sort_unstable();
    seen.dedup();
    seen.len() as f64 / points as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `n` R-R values uniform over `[lo, lo + span)` ms from a seeded LCG.
    fn scatter(n: usize, lo: u16, span: u64, seed: u64) -> Vec<u16> {
        let mut x = seed;
        (0..n)
            .map(|_| {
                x = x
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                lo + ((x >> 33) % span) as u16
            })
            .collect()
    }

    #[test]
    fn a_metronome_is_a_single_point() {
        let p = poincare(&[800u16; 40]).unwrap();
        assert_eq!((p.sd1, p.sd2, p.ellipse_area_ms2), (0.0, 0.0, 0.0));
        assert_eq!(
            p.ratio, None,
            "a plot with no long axis has no ratio, not a division by zero"
        );
        assert!(
            (p.cell_occupancy - 1.0 / 39.0).abs() < 1e-12,
            "39 points in one cell"
        );
    }

    #[test]
    fn sd1_is_the_short_axis_of_an_alternating_series() {
        // +/-10 about 800: every successive difference is +/-20, so SDSD ~ 20 and SD1 ~ 20/sqrt(2).
        let z: Vec<u16> = (0..60)
            .map(|i| if i % 2 == 0 { 790u16 } else { 810 })
            .collect();
        let p = poincare(&z).unwrap();
        assert!((p.sd1 - 20.0 / 2f64.sqrt()).abs() < 0.5, "sd1 {}", p.sd1);
        // A pure alternation puts ALL of its variance across the identity line, so the long axis is
        // exactly zero and there is no ratio to report — the degenerate case, answered as None.
        assert!(p.sd2 < 1e-6, "sd2 {}", p.sd2);
        assert_eq!(p.ratio, None);
        // Add a slow drift underneath and the long axis appears; SD1 still dominates it.
        let drifting: Vec<u16> = (0..60)
            .map(|i| 790 + 20 * (i % 2) as u16 + (i / 4) as u16)
            .collect();
        let d = poincare(&drifting).unwrap();
        assert!(d.sd2 > 0.0 && d.ratio.unwrap() > 1.0, "{d:?}");
    }

    #[test]
    fn a_slow_ramp_is_a_cigar_and_a_wide_scatter_is_a_cloud() {
        // A ramp moves along the identity line: SD2 large, SD1 tiny, ratio far below 1.
        let ramp: Vec<u16> = (0..120).map(|i| 700 + i as u16 * 2).collect();
        let cigar = poincare(&ramp).unwrap();
        assert!(
            cigar.ratio.unwrap() < 0.05,
            "a cigar must be long and thin: {cigar:?}"
        );

        let cloud = poincare(&scatter(120, 600, 400, 987)).unwrap();
        assert!(
            cloud.normalised_area > cigar.normalised_area * 10.0,
            "cloud {cloud:?} cigar {cigar:?}"
        );
        assert!(
            cloud.cell_occupancy > cigar.cell_occupancy,
            "the cloud must occupy more cells: cloud {} cigar {}",
            cloud.cell_occupancy,
            cigar.cell_occupancy
        );
    }

    #[test]
    fn short_and_degenerate_input_returns_none() {
        assert_eq!(poincare(&[]), None);
        assert_eq!(poincare(&[800u16; POINCARE_MIN_BEATS - 1]), None);
        assert_eq!(poincare(&[0u16; 40]), None); // no mean to normalise by
    }
}
