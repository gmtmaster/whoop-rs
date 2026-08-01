//! Rolling sleep-debt ledger: Σ(slept − need) over a capped trailing window of nights with
//! usable data. Pure; skips nights without sleep data (never zero-fills).

pub const DEFAULT_WINDOW_NIGHTS: usize = 14;
const DEFAULT_NEED_HOURS: f64 = 8.0;
pub const ON_TARGET_BAND_MIN: f64 = 30.0;

/// One night's contribution to the ledger.
#[derive(Debug, Clone, PartialEq)]
pub struct DebtNight {
    pub day: String,
    pub slept_min: f64,
    pub delta_min: f64,
}

/// The rolling debt ledger over the capped trailing window.
#[derive(Debug, Clone, PartialEq)]
pub struct DebtLedger {
    pub balance_min: f64,
    pub nights: Vec<DebtNight>,
    pub need_min: f64,
}

impl DebtLedger {
    pub fn night_count(&self) -> usize {
        self.nights.len()
    }
    pub fn is_on_target(&self) -> bool {
        self.balance_min.abs() < ON_TARGET_BAND_MIN
    }
}

/// Round to 1 dp, half-away-from-zero.
fn round1(v: f64) -> f64 {
    let scaled = v * 10.0;
    let rounded = if scaled < 0.0 { (scaled - 0.5).ceil() } else { (scaled + 0.5).floor() };
    rounded / 10.0
}

/// Build the ledger from a chronological series of `(day, total_sleep_min?)` rows.
/// Rows with `None` or `≤0` sleep are skipped. The window caps the most-recent COUNTED nights.
pub fn ledger(series: &[(String, Option<f64>)], need_hours: Option<f64>, window: Option<usize>) -> DebtLedger {
    let need_min = need_hours.unwrap_or(DEFAULT_NEED_HOURS).max(0.0) * 60.0;
    let cap = window.unwrap_or(DEFAULT_WINDOW_NIGHTS).max(1);

    let usable: Vec<&(String, Option<f64>)> = series.iter().filter(|(_, s)| s.unwrap_or(0.0) > 0.0).collect();
    let windowed = if usable.len() > cap { &usable[usable.len() - cap..] } else { &usable[..] };

    let mut nights = Vec::with_capacity(windowed.len());
    let mut balance = 0.0;
    for (day, slept) in windowed {
        let slept_min = slept.unwrap_or(0.0);
        let delta = slept_min - need_min;
        balance += delta;
        nights.push(DebtNight { day: day.clone(), slept_min, delta_min: delta });
    }
    DebtLedger { balance_min: round1(balance), nights, need_min }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(day: &str, min: f64) -> (String, Option<f64>) {
        (day.to_string(), Some(min))
    }

    #[test]
    fn empty_series_returns_zero() {
        let l = ledger(&[], None, None);
        assert_eq!(l.balance_min, 0.0);
        assert_eq!(l.night_count(), 0);
    }

    #[test]
    fn surplus_offsets_deficit() {
        let series = [s("d1", 360.0), s("d2", 600.0)];
        let l = ledger(&series, Some(8.0), Some(14));
        // need = 480 min. d1: 360-480=-120, d2: 600-480=+120 → balance=0
        assert!((l.balance_min).abs() < 0.1, "got {}", l.balance_min);
    }

    #[test]
    fn null_skipped_not_zeroed() {
        let series = [("d1".to_string(), None), s("d2", 240.0)];
        let l = ledger(&series, Some(8.0), Some(14));
        assert_eq!(l.night_count(), 1); // only d2
        assert!(l.balance_min < 0.0);
    }

    #[test]
    fn window_caps_most_recent() {
        let mut series = Vec::new();
        for i in 0..20 {
            series.push((format!("d{i}"), Some(500.0)));
        }
        let l = ledger(&series, Some(8.0), Some(14));
        assert_eq!(l.night_count(), 14);
        assert_eq!(l.nights[0].day, "d6"); // oldest in window
        assert_eq!(l.nights[13].day, "d19"); // newest
    }

    #[test]
    fn on_target_band() {
        let l = DebtLedger { balance_min: 15.0, nights: vec![], need_min: 480.0 };
        assert!(l.is_on_target());
        let l2 = DebtLedger { balance_min: -45.0, nights: vec![], need_min: 480.0 };
        assert!(!l2.is_on_target());
    }
}
