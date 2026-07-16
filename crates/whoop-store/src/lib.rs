//! Per-(person, strap) calibration store. Persists each drain's nightly metrics to SQLite keyed on the
//! person wearing the strap AND the strap serial, so a new person or a new strap starts a fresh
//! calibration period (you never calibrate against someone else's data). A metric's baseline finalizes
//! once its (person, strap) reaches the WHOOP milestone from `whoop_metrics::calibration`.

use rusqlite::{params, Connection};
use whoop_metrics::{calibration, stats, HrvReadiness};
use whoop_protocol::{HistoryRecord, Record};

/// One calendar day's summary for a (person, strap), segmented from a drain.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NightMetrics {
    pub day: u32,
    pub spo2_median: Option<f64>,
    pub rmssd_ms: Option<f64>,
    pub hr_min: Option<u8>,
    pub hr_max: Option<u8>,
    pub records: u32,
}

/// A metric's calibration state for a (person, strap): still gathering nights, or a finalized baseline.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CalState {
    Calibrating { have: usize, need: u32 },
    Baseline { value: f64, nights: usize },
}

#[derive(thiserror::Error, Debug)]
pub enum StoreError {
    #[error("sqlite: {0}")]
    Sqlite(#[from] rusqlite::Error),
}

const SCHEMA: &str = "\
CREATE TABLE IF NOT EXISTS strap_night(
  person TEXT NOT NULL, strap TEXT NOT NULL, day INTEGER NOT NULL,
  spo2_median REAL, rmssd_ms REAL, hr_min INTEGER, hr_max INTEGER, records INTEGER NOT NULL,
  PRIMARY KEY(person, strap, day));
CREATE TABLE IF NOT EXISTS baseline(
  person TEXT NOT NULL, strap TEXT NOT NULL, metric TEXT NOT NULL,
  value REAL NOT NULL, nights INTEGER NOT NULL, finalized_at INTEGER NOT NULL DEFAULT (unixepoch()),
  PRIMARY KEY(person, strap, metric));";

pub struct Store {
    conn: Connection,
}

impl Store {
    /// Open (creating the schema) a store file; use `:memory:` semantics via `open_memory` for tests.
    pub fn open(path: &str) -> Result<Self, StoreError> {
        let conn = Connection::open(path)?;
        conn.execute_batch(SCHEMA)?;
        Ok(Store { conn })
    }

    pub fn open_memory() -> Result<Self, StoreError> {
        let conn = Connection::open_in_memory()?;
        conn.execute_batch(SCHEMA)?;
        Ok(Store { conn })
    }

    /// Segment a drain's records into nights and upsert them under (person, strap). Returns nights written.
    pub fn ingest(&self, person: &str, strap: &str, records: &[Record]) -> Result<usize, StoreError> {
        let nights = segment(records);
        for n in &nights {
            self.conn.execute(
                "INSERT INTO strap_night(person,strap,day,spo2_median,rmssd_ms,hr_min,hr_max,records)
                 VALUES(?1,?2,?3,?4,?5,?6,?7,?8)
                 ON CONFLICT(person,strap,day) DO UPDATE SET
                   spo2_median=excluded.spo2_median, rmssd_ms=excluded.rmssd_ms,
                   hr_min=excluded.hr_min, hr_max=excluded.hr_max, records=excluded.records",
                params![person, strap, n.day, n.spo2_median, n.rmssd_ms, n.hr_min, n.hr_max, n.records],
            )?;
        }
        Ok(nights.len())
    }

    /// Distinct nights recorded for (person, strap).
    pub fn night_count(&self, person: &str, strap: &str) -> Result<usize, StoreError> {
        let n: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM strap_night WHERE person=?1 AND strap=?2",
            params![person, strap],
            |r| r.get(0),
        )?;
        Ok(n as usize)
    }

    /// The nightly values of a numeric column for (person, strap), oldest → newest, skipping NULLs.
    fn column(&self, person: &str, strap: &str, col: &str) -> Result<Vec<f64>, StoreError> {
        let sql =
            format!("SELECT {col} FROM strap_night WHERE person=?1 AND strap=?2 AND {col} IS NOT NULL ORDER BY day");
        let mut stmt = self.conn.prepare(&sql)?;
        let rows = stmt.query_map(params![person, strap], |r| r.get::<_, f64>(0))?;
        Ok(rows.filter_map(Result::ok).collect())
    }

    /// Calibration state of one metric for (person, strap): calibrating until `milestone.full` nights,
    /// then a finalized (and persisted) baseline = the median of that metric's nightly values.
    pub fn calibrate(
        &self,
        person: &str,
        strap: &str,
        metric: &str,
        col: &str,
        milestone: calibration::Calibration,
    ) -> Result<CalState, StoreError> {
        let nights = self.column(person, strap, col)?;
        if (nights.len() as u32) < milestone.full {
            return Ok(CalState::Calibrating { have: nights.len(), need: milestone.full });
        }
        let value = stats::median(&nights);
        self.conn.execute(
            "INSERT INTO baseline(person,strap,metric,value,nights) VALUES(?1,?2,?3,?4,?5)
             ON CONFLICT(person,strap,metric) DO UPDATE SET
               value=excluded.value, nights=excluded.nights, finalized_at=unixepoch()",
            params![person, strap, metric, value, nights.len()],
        )?;
        Ok(CalState::Baseline { value, nights: nights.len() })
    }

    /// SpO2 calibration state for (person, strap), gated on the blood-oxygen milestone.
    pub fn spo2_state(&self, person: &str, strap: &str) -> Result<CalState, StoreError> {
        self.calibrate(person, strap, "spo2", "spo2_median", calibration::BLOOD_OXYGEN)
    }

    /// HRV baseline state for (person, strap), gated on the recovery-score milestone.
    pub fn hrv_state(&self, person: &str, strap: &str) -> Result<CalState, StoreError> {
        self.calibrate(person, strap, "hrv", "rmssd_ms", calibration::RECOVERY_SCORE)
    }
}

/// Group history records by calendar day into one summary per day.
fn segment(records: &[Record]) -> Vec<NightMetrics> {
    let mut by_day: std::collections::BTreeMap<u32, DayAcc> = std::collections::BTreeMap::new();
    for r in records {
        if let Record::History(h) = r {
            by_day.entry(h.unix / 86_400).or_default().push(h);
        }
    }
    by_day.into_iter().map(|(day, acc)| acc.finish(day)).collect()
}

#[derive(Default)]
struct DayAcc {
    spo2: Vec<f64>,
    rr: Vec<u16>,
    hr_min: Option<u8>,
    hr_max: Option<u8>,
    records: u32,
}

impl DayAcc {
    fn push(&mut self, h: &HistoryRecord) {
        if let Some(p) = h.spo2_pct {
            self.spo2.push(p as f64);
        }
        self.rr.extend(&h.rr_intervals);
        if let Some(hr) = h.heart_rate {
            self.hr_min = Some(self.hr_min.map_or(hr, |m| m.min(hr)));
            self.hr_max = Some(self.hr_max.map_or(hr, |m| m.max(hr)));
        }
        self.records += 1;
    }

    fn finish(self, day: u32) -> NightMetrics {
        NightMetrics {
            day,
            spo2_median: (!self.spo2.is_empty()).then(|| stats::median(&self.spo2)),
            rmssd_ms: HrvReadiness::rmssd(&self.rr),
            hr_min: self.hr_min,
            hr_max: self.hr_max,
            records: self.records,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hist(unix: u32, spo2: Option<u8>) -> Record {
        Record::History(HistoryRecord { version: 18, unix, spo2_pct: spo2, heart_rate: Some(60), ..Default::default() })
    }

    #[test]
    fn ingest_segments_by_day_and_keys_on_person_strap() {
        let s = Store::open_memory().unwrap();
        // person 1, strap A: two days (day 0 twice + day 1)
        s.ingest("1", "A", &[hist(0, Some(97)), hist(100, Some(98)), hist(86_401, Some(96))]).unwrap();
        assert_eq!(s.night_count("1", "A").unwrap(), 2);
        // a DIFFERENT person on the SAME strap is a separate key — a fresh calibration period
        s.ingest("2", "A", &[hist(0, Some(90))]).unwrap();
        assert_eq!(s.night_count("2", "A").unwrap(), 1);
        assert_eq!(s.night_count("1", "A").unwrap(), 2); // person 1 untouched
    }

    #[test]
    fn spo2_calibrates_then_baselines_at_the_milestone() {
        let s = Store::open_memory().unwrap();
        assert!(matches!(s.spo2_state("1", "A").unwrap(), CalState::Calibrating { have: 0, .. }));
        // blood-oxygen unlocks after one recovery, so one night with SpO2 finalizes the baseline.
        s.ingest("1", "A", &[hist(0, Some(97)), hist(50, Some(99))]).unwrap();
        match s.spo2_state("1", "A").unwrap() {
            CalState::Baseline { value, nights } => {
                assert_eq!(nights, 1);
                assert!((value - 98.0).abs() < 1e-9); // median(97, 99)
            }
            other => panic!("expected baseline, got {other:?}"),
        }
    }
}
