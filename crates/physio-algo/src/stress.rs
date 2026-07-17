//! Baevsky Stress Index (SI) — a histogram-based autonomic-balance proxy over an R-R series.
//! `SI = AMo / (2 * Mo * MxDMn)`: Mo is the modal R-R (s), AMo the modal bin's share (%), MxDMn the
//! R-R range (s). A tall, narrow, low-range histogram (rigid, sympathetic) reads high; a broad, flat
//! one reads low. R-R is cleaned first (range band + Malik ectopic). Wellness only, never medical.

/// Histogram bin width in seconds (Baevsky's 50 ms cardiointervalography grid).
const BIN_WIDTH_SEC: f64 = 0.05;
/// Minimum clean intervals before an SI is computed.
pub const MIN_BEATS: usize = 20;

/// R-R keep-band (ms); intervals outside are dropouts/ectopics.
const RR_MIN_MS: f64 = 300.0;
const RR_MAX_MS: f64 = 2000.0;
/// Malik ectopic rejection: beat dropped if it deviates over 20% from its local median.
const ECTOPIC_THRESHOLD: f64 = 0.20;
/// Half-width (beats) of the centred median window; a 5-beat window at radius 2.
const ECTOPIC_WINDOW_RADIUS: usize = 2;

/// Intermediate histogram terms behind an SI, exposed so a caller can show the "why".
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StressComponents {
    pub mo_sec: f64,
    pub amo_percent: f64,
    pub mxdmn_sec: f64,
    pub si: f64,
}

/// Baevsky Stress Index from a raw R-R series (ms). `None` when too few clean beats survive or the
/// range is degenerate (all-equal beats → MxDMn 0 → an honest `None`, not infinity).
pub fn stress_index_raw(rr_ms: &[f64]) -> Option<f64> {
    components_raw(rr_ms).map(|c| c.si)
}

/// Full SI components from a raw R-R series (ms). Pure and deterministic.
pub fn components_raw(rr_ms: &[f64]) -> Option<StressComponents> {
    let clean = clean_rr(rr_ms);
    if clean.len() < MIN_BEATS {
        return None;
    }
    let sec: Vec<f64> = clean.iter().map(|v| v / 1000.0).collect();
    let min_v = sec.iter().copied().fold(f64::INFINITY, f64::min);
    let max_v = sec.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mxdmn = max_v - min_v;
    if mxdmn <= 0.0 {
        return None;
    }

    let bin_count = ((mxdmn / BIN_WIDTH_SEC).floor() as usize + 1).max(1);
    let mut counts = vec![0usize; bin_count];
    for &v in &sec {
        let mut idx = ((v - min_v) / BIN_WIDTH_SEC).floor() as isize;
        if idx < 0 {
            idx = 0;
        }
        let mut idx = idx as usize;
        if idx >= bin_count {
            idx = bin_count - 1;
        }
        counts[idx] += 1;
    }
    // Modal bin: highest count; ties resolve to the lowest index (deterministic across platforms).
    let mut mode_idx = 0usize;
    let mut mode_count = counts[0];
    for (i, &c) in counts.iter().enumerate().skip(1) {
        if c > mode_count {
            mode_count = c;
            mode_idx = i;
        }
    }
    let mo = min_v + (mode_idx as f64 + 0.5) * BIN_WIDTH_SEC;
    let amo = mode_count as f64 / sec.len() as f64 * 100.0;
    if mo <= 0.0 {
        return None;
    }
    let si = amo / (2.0 * mo * mxdmn);
    Some(StressComponents { mo_sec: mo, amo_percent: amo, mxdmn_sec: mxdmn, si })
}

/// Full clean: range band then Malik ectopic rejection, order preserved.
fn clean_rr(rr: &[f64]) -> Vec<f64> {
    let ranged: Vec<f64> = rr.iter().copied().filter(|&v| (RR_MIN_MS..=RR_MAX_MS).contains(&v)).collect();
    reject_ectopic(&ranged)
}

/// Drop any beat deviating over `ECTOPIC_THRESHOLD` from its local median; short series pass through.
fn reject_ectopic(nn: &[f64]) -> Vec<f64> {
    if nn.len() <= ECTOPIC_WINDOW_RADIUS {
        return nn.to_vec();
    }
    let mut kept = Vec::with_capacity(nn.len());
    for i in 0..nn.len() {
        let lo = i.saturating_sub(ECTOPIC_WINDOW_RADIUS);
        let hi = (i + ECTOPIC_WINDOW_RADIUS).min(nn.len() - 1);
        let mut neighbours: Vec<f64> = Vec::with_capacity(hi - lo);
        for (j, &v) in nn.iter().enumerate().take(hi + 1).skip(lo) {
            if j != i {
                neighbours.push(v);
            }
        }
        if neighbours.len() < 2 {
            kept.push(nn[i]);
            continue;
        }
        let med = median(&neighbours);
        if med <= 0.0 {
            kept.push(nn[i]);
            continue;
        }
        if (nn[i] - med).abs() / med <= ECTOPIC_THRESHOLD {
            kept.push(nn[i]);
        }
    }
    kept
}

/// Median of a slice (sorts a copy). Empty → 0.
fn median(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut s = values.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = s.len();
    if n % 2 == 1 {
        s[n / 2]
    } else {
        0.5 * (s[n / 2 - 1] + s[n / 2])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GOLDEN: [f64; 22] = [
        700.0, 720.0, 740.0, 760.0, 780.0, 800.0, 820.0, 840.0, 860.0, 800.0, 800.0, 800.0, 800.0,
        820.0, 780.0, 800.0, 810.0, 790.0, 800.0, 800.0, 805.0, 795.0,
    ];

    #[test]
    fn golden_stress_index_hand_computed() {
        let comp = components_raw(&GOLDEN).expect("scorable");
        assert!((comp.mxdmn_sec - 0.16).abs() < 1e-9);
        assert!((comp.mo_sec - 0.825).abs() < 1e-9);
        assert!((comp.amo_percent - 59.09090909090909).abs() < 1e-9);
        assert!((comp.si - 223.82920110192836).abs() < 1e-9);
        assert!((stress_index_raw(&GOLDEN).unwrap() - 223.82920110192836).abs() < 1e-9);
    }

    #[test]
    fn tighter_histogram_raises_si() {
        let broad: Vec<f64> = (0..30).map(|it| 700.0 + (it % 11) as f64 * 18.0).collect();
        let rigid: Vec<f64> = (0..30).map(|it| if it % 6 == 0 { 810.0 } else { 800.0 }).collect();
        let si_broad = stress_index_raw(&broad).expect("broad scorable");
        let si_rigid = stress_index_raw(&rigid).expect("rigid scorable");
        assert!(si_rigid > si_broad, "a rigid, tightly-clustered rhythm has a higher SI");
    }

    #[test]
    fn too_few_beats_returns_none() {
        let rr = vec![800.0; MIN_BEATS - 1];
        assert!(stress_index_raw(&rr).is_none());
    }

    #[test]
    fn degenerate_range_returns_none() {
        let rr = vec![800.0; 30];
        assert!(stress_index_raw(&rr).is_none());
    }
}
