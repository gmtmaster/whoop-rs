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

    fn nights_of(mins: &[Option<f64>]) -> Vec<(String, Option<f64>)> {
        mins.iter().enumerate().map(|(i, m)| (format!("d{i}"), *m)).collect()
    }

    /// One row of the truth table: an input, and the balance and counted-night total it must yield.
    struct Row {
        name: &'static str,
        series: Vec<(String, Option<f64>)>,
        need: Option<f64>,
        window: Option<usize>,
        balance: f64,
        count: usize,
    }

    fn row(
        name: &'static str,
        mins: &[Option<f64>],
        need: Option<f64>,
        window: Option<usize>,
        balance: f64,
        count: usize,
    ) -> Row {
        Row { name, series: nights_of(mins), need, window, balance, count }
    }

    /// A ledger is exact arithmetic, not physiology, so every figure below is computed, not observed.
    fn table() -> Vec<Row> {
        vec![
            row("nothing reported", &[], None, None, 0.0, 0),
            row("a surplus cancels an equal deficit", &[Some(360.0), Some(600.0)], Some(8.0), Some(14), 0.0, 2),
            row("two short nights", &[Some(360.0), Some(360.0)], Some(8.0), Some(14), -240.0, 2),
            row("two long nights", &[Some(600.0), Some(600.0)], Some(8.0), Some(14), 240.0, 2),
            row("the default need", &[Some(420.0)], None, None, -60.0, 1),
            row("a night with no data", &[None, Some(240.0)], Some(8.0), Some(14), -240.0, 1),
            row("a shorter need", &[Some(420.0)], Some(6.0), None, 60.0, 1),
        ]
    }

    fn reproduces(scorer: impl Fn(&[(String, Option<f64>)], Option<f64>, Option<usize>) -> (f64, usize)) -> bool {
        table().into_iter().all(|r| scorer(&r.series, r.need, r.window) == (r.balance, r.count))
    }

    #[test]
    fn the_shipped_ledger_reproduces_the_table_and_three_do_nothing_ledgers_do_not() {
        for r in table() {
            let l = ledger(&r.series, r.need, r.window);
            assert_eq!(l.balance_min, r.balance, "{}: balance", r.name);
            assert_eq!(l.night_count(), r.count, "{}: counted nights", r.name);
        }
        assert!(reproduces(|se, n, w| {
            let l = ledger(se, n, w);
            (l.balance_min, l.night_count())
        }));
        // One always reports level, one sums the sleep without ever subtracting the need, one
        // zero-fills the nights that reported nothing. Each must miss at least one row.
        type Null = fn(&[(String, Option<f64>)], Option<f64>, Option<usize>) -> (f64, usize);
        let nulls: [(&str, Null); 3] = [
            ("always level", |se, _, _| (0.0, se.iter().filter(|(_, m)| m.is_some()).count())),
            ("never subtracts the need", |se, _, _| {
                let u: Vec<f64> = se.iter().filter_map(|(_, m)| *m).collect();
                (u.iter().sum(), u.len())
            }),
            ("zero-fills the missing nights", |se, n, _| {
                let need = n.unwrap_or(DEFAULT_NEED_HOURS) * 60.0;
                (se.iter().map(|(_, m)| m.unwrap_or(0.0) - need).sum(), se.len())
            }),
        ];
        for (name, null) in nulls {
            assert!(!reproduces(null), "{name} reproduced every row; the table cannot tell it apart");
        }
    }

    #[test]
    fn the_default_need_is_eight_hours_and_the_default_path_uses_it() {
        assert_eq!(DEFAULT_NEED_HOURS, 8.0);
        assert_eq!(DEFAULT_WINDOW_NIGHTS, 14);
        let l = ledger(&[s("d1", 420.0)], None, None);
        assert_eq!(l.need_min, DEFAULT_NEED_HOURS * 60.0);
        assert_eq!(l.need_min, 480.0);
        assert_eq!(l.nights[0].delta_min, -60.0);
        assert_eq!(l.balance_min, -60.0);
    }

    #[test]
    fn the_balance_is_the_exact_sum_of_the_nightly_deltas() {
        let l = ledger(&[s("d1", 360.0), s("d2", 600.0)], Some(8.0), Some(14));
        assert_eq!(l.need_min, 480.0);
        assert_eq!(l.nights[0].delta_min, -120.0);
        assert_eq!(l.nights[1].delta_min, 120.0);
        assert_eq!(l.balance_min, 0.0, "a surplus cancels an equal deficit");
        assert_eq!(ledger(&[s("d1", 360.0), s("d2", 360.0)], Some(8.0), Some(14)).balance_min, -240.0);
        assert_eq!(ledger(&[s("d1", 600.0), s("d2", 600.0)], Some(8.0), Some(14)).balance_min, 240.0);
    }

    #[test]
    fn the_need_is_configurable_and_floored_at_zero() {
        assert_eq!(ledger(&[s("d1", 420.0)], Some(6.0), None).balance_min, 60.0);
        let l = ledger(&[s("d1", 420.0)], Some(-3.0), None);
        assert_eq!(l.need_min, 0.0, "a negative need is clamped, never a bonus");
        assert_eq!(l.balance_min, 420.0);
    }

    #[test]
    fn the_balance_rounds_to_one_decimal_half_away_from_zero() {
        assert_eq!(ledger(&[s("d1", 480.25)], Some(8.0), None).balance_min, 0.3);
        assert_eq!(ledger(&[s("d1", 479.75)], Some(8.0), None).balance_min, -0.3);
    }

    #[test]
    fn a_night_without_usable_sleep_is_skipped_never_zero_filled() {
        let series = vec![
            ("d1".to_string(), None),
            ("d2".to_string(), Some(0.0)),
            ("d3".to_string(), Some(-5.0)),
            s("d4", 240.0),
        ];
        let l = ledger(&series, Some(8.0), Some(14));
        assert_eq!(l.night_count(), 1, "only d4 reported usable sleep");
        assert_eq!(l.nights[0].day, "d4");
        assert_eq!(l.balance_min, -240.0, "zero-filling the other three would read -1920");
    }

    #[test]
    fn the_window_caps_counted_nights_not_calendar_nights() {
        let all: Vec<(String, Option<f64>)> = (0..20).map(|i| (format!("d{i}"), Some(500.0))).collect();
        let l = ledger(&all, Some(8.0), Some(14));
        assert_eq!(l.night_count(), 14);
        assert_eq!(l.nights[0].day, "d6");
        assert_eq!(l.nights[13].day, "d19");
        assert_eq!(l.balance_min, 280.0, "14 nights 20 min over the need");

        // Nights that reported nothing are dropped first, so they never consume a window slot.
        let mixed: Vec<(String, Option<f64>)> =
            (0..20).map(|i| (format!("d{i}"), (i % 2 == 0).then_some(500.0))).collect();
        let m = ledger(&mixed, Some(8.0), Some(14));
        assert_eq!(m.night_count(), 10, "only the ten reporting nights count");
        assert_eq!(m.nights[0].day, "d0");
        assert_eq!(m.nights[9].day, "d18");
        let capped = ledger(&mixed, Some(8.0), Some(3));
        assert_eq!(capped.night_count(), 3);
        assert_eq!(capped.nights[0].day, "d14");
        assert_eq!(capped.balance_min, 60.0);
    }

    #[test]
    fn the_on_target_band_is_thirty_minutes_either_side_of_level() {
        assert_eq!(ON_TARGET_BAND_MIN, 30.0);
        assert!(ledger(&[s("d1", 450.1)], Some(8.0), None).is_on_target(), "29.9 min short is on target");
        assert!(ledger(&[s("d1", 509.9)], Some(8.0), None).is_on_target());
        assert!(!ledger(&[s("d1", 450.0)], Some(8.0), None).is_on_target(), "the band is exclusive at 30");
        assert!(!ledger(&[s("d1", 510.0)], Some(8.0), None).is_on_target());
        assert!(ledger(&[], None, None).is_on_target(), "an empty ledger is level, not off target");
    }
}
