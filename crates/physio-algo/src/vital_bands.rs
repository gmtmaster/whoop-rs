//! Banding for the health-monitor vitals: in-range against the wearer's own trailing baseline once
//! it is trusted, against the typical-adult window before that and again whenever it goes stale.
//! A metric config's physiological bounds stay an outer guard either way, never the in-range band —
//! using them as one reads a normal personal HRV of 35 ms as permanently out of range.
//! Wellness estimates, never clinical cut points.

use crate::baselines::{self, BaselineStatus, MetricCfg};
use crate::recovery::z_score;

/// |z| at or below this is in range against the personal baseline: about 95% of the wearer's own
/// normal nights, where the |z| ≤ 1 of a deviation read would flag about a third of them.
pub const SIGMA_K: f64 = 2.0;

/// At or above this a skin-temp reading is an absolute wrist temperature; below it, a ±°C deviation.
/// No real wrist temperature sits under it and no real deviation reaches it.
pub const SKIN_TEMP_ABSOLUTE_MIN_C: f64 = 20.0;

/// An inclusive typical-adult window, the fallback an untrusted baseline bands against.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TypicalRange {
    pub min: f64,
    pub max: f64,
}

impl TypicalRange {
    pub const fn new(min: f64, max: f64) -> Self {
        Self { min, max }
    }
    pub fn contains(&self, v: f64) -> bool {
        self.min <= v && v <= self.max
    }
}

pub const RESP_TYPICAL_RPM: TypicalRange = TypicalRange::new(12.0, 20.0);
pub const SPO2_TYPICAL_PCT: TypicalRange = TypicalRange::new(95.0, 100.0);
pub const RHR_TYPICAL_BPM: TypicalRange = TypicalRange::new(40.0, 60.0);
pub const HRV_TYPICAL_MS: TypicalRange = TypicalRange::new(40.0, 120.0);
pub const SKIN_ABS_TYPICAL_C: TypicalRange = TypicalRange::new(33.0, 36.0);
pub const SKIN_DEV_TYPICAL_C: TypicalRange = TypicalRange::new(-0.6, 0.6);

/// The typical window for a vital key, or `None` when the key names no vital.
pub fn typical_range(vital: &str) -> Option<TypicalRange> {
    Some(match vital {
        "resp" => RESP_TYPICAL_RPM,
        "spo2" => SPO2_TYPICAL_PCT,
        "rhr" => RHR_TYPICAL_BPM,
        "hrv" => HRV_TYPICAL_MS,
        "skin_abs" => SKIN_ABS_TYPICAL_C,
        "skin_dev" => SKIN_DEV_TYPICAL_C,
        _ => return None,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Band {
    InRange,
    OutOfRange,
    NoData,
}

impl Band {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::InRange => "inRange",
            Self::OutOfRange => "outOfRange",
            Self::NoData => "noData",
        }
    }
}

/// Which yardstick judged the reading.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Basis {
    Personal,
    Population,
}

impl Basis {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Personal => "personal",
            Self::Population => "population",
        }
    }
}

/// A banded reading, with the valid-night count the basis was decided on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BandResult {
    pub band: Band,
    pub basis: Basis,
    pub nights: i32,
}

fn population(value: f64, range: TypicalRange, nights: i32) -> BandResult {
    BandResult {
        band: if range.contains(value) {
            Band::InRange
        } else {
            Band::OutOfRange
        },
        basis: Basis::Population,
        nights,
    }
}

/// Band one vital. `history` is nightly values oldest first, EXCLUDING the displayed day, with
/// `None` for a missing night. A `None` `cfg` disables the personal path entirely, leaving an
/// absolute window that is meaningful regardless of personal history.
pub fn band(
    value: Option<f64>,
    history: &[Option<f64>],
    population_range: TypicalRange,
    cfg: Option<&MetricCfg>,
) -> BandResult {
    let Some(v) = value else {
        return BandResult {
            band: Band::NoData,
            basis: Basis::Population,
            nights: 0,
        };
    };
    let Some(cfg) = cfg else {
        return population(v, population_range, 0);
    };
    let state = baselines::fold_history(history, cfg);
    // Outer guard: a value outside the physiological bounds is out of range however wide the
    // personal spread happens to be.
    if !(cfg.min_val <= v && v <= cfg.max_val) {
        return BandResult {
            band: Band::OutOfRange,
            basis: Basis::Population,
            nights: state.n_valid,
        };
    }
    if state.status == BaselineStatus::Trusted {
        let z = z_score(v, state.baseline, state.spread);
        return BandResult {
            band: if z.abs() <= SIGMA_K {
                Band::InRange
            } else {
                Band::OutOfRange
            },
            basis: Basis::Personal,
            nights: state.n_valid,
        };
    }
    population(v, population_range, state.n_valid)
}

/// Whether a skin-temp value is an absolute wrist temperature rather than a ±°C deviation.
pub fn is_absolute_skin_temp(v: f64) -> bool {
    v >= SKIN_TEMP_ABSOLUTE_MIN_C
}

/// Keep only the history entries of the same kind as `value`, blanking the rest, so a baseline is
/// never folded across the absolute and deviation scales at once.
pub fn skin_temp_history(value: f64, history: &[Option<f64>]) -> Vec<Option<f64>> {
    let absolute = is_absolute_skin_temp(value);
    history
        .iter()
        .map(|v| v.filter(|&x| is_absolute_skin_temp(x) == absolute))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hrv_cfg() -> MetricCfg {
        MetricCfg::hrv()
    }

    #[test]
    fn null_value_is_no_data() {
        let r = band(None, &[Some(50.0)], HRV_TYPICAL_MS, Some(&hrv_cfg()));
        assert_eq!(r.band, Band::NoData);
        assert_eq!(r.nights, 0);
    }

    /// The motivating case: a personal-normal 35 ms HRV under a 40–120 population window. Below the
    /// trust gate it is still judged against the population, so out of range.
    #[test]
    fn low_hrv_below_the_trust_gate_bands_against_the_population() {
        let hist: Vec<Option<f64>> = vec![Some(35.0); 10];
        let r = band(Some(35.0), &hist, HRV_TYPICAL_MS, Some(&hrv_cfg()));
        assert_eq!(r.band, Band::OutOfRange);
        assert_eq!(r.basis, Basis::Population);
        assert_eq!(r.nights, 10);
    }

    /// At the trust gate the same 35 ms is in range against the wearer's own baseline.
    #[test]
    fn low_hrv_at_the_trust_gate_bands_personally() {
        let hist: Vec<Option<f64>> = vec![Some(35.0); 14];
        let r = band(Some(35.0), &hist, HRV_TYPICAL_MS, Some(&hrv_cfg()));
        assert_eq!(r.band, Band::InRange);
        assert_eq!(r.basis, Basis::Personal);
        assert_eq!(r.nights, 14);
    }

    #[test]
    fn a_big_personal_deviation_is_out_of_range() {
        let hist: Vec<Option<f64>> = vec![Some(35.0); 30];
        let r = band(Some(70.0), &hist, HRV_TYPICAL_MS, Some(&hrv_cfg()));
        assert_eq!(r.band, Band::OutOfRange);
        assert_eq!(r.basis, Basis::Personal);
    }

    /// Just inside the gate in σ-space (spread × 1.253 ≈ σ) stays in range.
    #[test]
    fn just_inside_two_sigma_is_in_range() {
        let hist: Vec<Option<f64>> = vec![Some(35.0); 30];
        let state = baselines::fold_history(&hist, &hrv_cfg());
        let edge = state.baseline + 1.99 * 1.253 * state.spread;
        assert_eq!(
            band(Some(edge), &hist, HRV_TYPICAL_MS, Some(&hrv_cfg())).band,
            Band::InRange
        );
    }

    #[test]
    fn an_implausible_value_is_out_of_range_even_on_a_trusted_baseline() {
        let hist: Vec<Option<f64>> = vec![Some(35.0); 30];
        let r = band(Some(300.0), &hist, HRV_TYPICAL_MS, Some(&hrv_cfg()));
        assert_eq!(r.band, Band::OutOfRange);
        assert_eq!(r.basis, Basis::Population);
    }

    #[test]
    fn no_cfg_stays_population_only() {
        let r = band(Some(93.0), &[], SPO2_TYPICAL_PCT, None);
        assert_eq!(r.band, Band::OutOfRange);
        assert_eq!(r.basis, Basis::Population);
        assert_eq!(
            band(Some(97.0), &[], SPO2_TYPICAL_PCT, None).band,
            Band::InRange
        );
    }

    #[test]
    fn missing_nights_do_not_count_toward_trust() {
        let mut hist: Vec<Option<f64>> = vec![Some(35.0); 13];
        hist.extend(std::iter::repeat_n(None, 10));
        assert_eq!(
            band(Some(35.0), &hist, HRV_TYPICAL_MS, Some(&hrv_cfg())).basis,
            Basis::Population
        );
    }

    #[test]
    fn a_stale_baseline_falls_back_to_the_population() {
        let mut hist: Vec<Option<f64>> = vec![Some(35.0); 20];
        hist.extend(std::iter::repeat_n(None, 20));
        assert_eq!(
            band(Some(35.0), &hist, HRV_TYPICAL_MS, Some(&hrv_cfg())).basis,
            Basis::Population
        );
    }

    #[test]
    fn typical_windows_are_the_ones_the_screens_draw() {
        assert_eq!(typical_range("resp"), Some(TypicalRange::new(12.0, 20.0)));
        assert_eq!(typical_range("spo2"), Some(TypicalRange::new(95.0, 100.0)));
        assert_eq!(typical_range("rhr"), Some(TypicalRange::new(40.0, 60.0)));
        assert_eq!(typical_range("hrv"), Some(TypicalRange::new(40.0, 120.0)));
        assert_eq!(
            typical_range("skin_abs"),
            Some(TypicalRange::new(33.0, 36.0))
        );
        assert_eq!(
            typical_range("skin_dev"),
            Some(TypicalRange::new(-0.6, 0.6))
        );
        assert_eq!(typical_range("nope"), None);
    }

    #[test]
    fn range_edges_are_inclusive() {
        assert!(RESP_TYPICAL_RPM.contains(12.0));
        assert!(RESP_TYPICAL_RPM.contains(20.0));
        assert!(!RESP_TYPICAL_RPM.contains(11.999));
        assert!(!RESP_TYPICAL_RPM.contains(20.001));
    }

    #[test]
    fn skin_temp_history_partitions_the_two_scales() {
        let mixed = [Some(34.1), Some(0.2), None, Some(33.8), Some(-0.1)];
        assert_eq!(
            skin_temp_history(0.3, &mixed),
            vec![None, Some(0.2), None, None, Some(-0.1)]
        );
        assert_eq!(
            skin_temp_history(34.0, &mixed),
            vec![Some(34.1), None, None, Some(33.8), None]
        );
    }

    #[test]
    fn the_skin_temp_kind_threshold_is_inclusive() {
        assert!(is_absolute_skin_temp(20.0));
        assert!(is_absolute_skin_temp(34.0));
        assert!(!is_absolute_skin_temp(19.999));
        assert!(!is_absolute_skin_temp(-0.3));
    }
}
