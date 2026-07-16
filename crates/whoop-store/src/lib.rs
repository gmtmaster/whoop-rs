//! Per-(person, strap) calibration store. Persists each drain's nightly metrics to SQLite keyed on the
//! person wearing the strap AND the strap serial, so a new person or a new strap starts a fresh
//! calibration period (you never calibrate against someone else's data). A metric's baseline finalizes
//! once its (person, strap) reaches the WHOOP milestone from `whoop_metrics::calibration`. Nightly RMSSD is
//! gap-aware and artifact-corrected (successive differences pool only within time-contiguous, plausible
//! R-R runs, dropping ectopic/missed-beat jumps). A per-strap
//! linear fit (`reference ≈ scale·field + offset`) can also be finalized and persisted from paired columns.
//! Known limit: nights are bucketed by UTC calendar day, so a sleep that straddles UTC midnight is split
//! across two rows (the one beat-pair at the boundary is dropped). A physiological-day cutover is a TODO.

use rusqlite::{params, Connection, OptionalExtension};
use whoop_metrics::{calibration, stats, HrvReadiness, LinearFit};
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

/// A per-strap linear-fit state: gathering paired nights, enough nights but no spread to fit, or a
/// finalized (and persisted) coefficient.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FitState {
    Calibrating { have: usize, need: u32 },
    Unfittable { nights: usize },
    Fit { fit: LinearFit, nights: usize },
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
  PRIMARY KEY(person, strap, metric));
CREATE TABLE IF NOT EXISTS fit_baseline(
  person TEXT NOT NULL, strap TEXT NOT NULL, name TEXT NOT NULL,
  scale REAL NOT NULL, offset REAL NOT NULL, r REAL NOT NULL, nights INTEGER NOT NULL,
  finalized_at INTEGER NOT NULL DEFAULT (unixepoch()),
  PRIMARY KEY(person, strap, name));";

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

    /// Paired nightly (field, reference) values for (person, strap) where both columns are non-null,
    /// oldest → newest. Column names come from code constants, not user input (same pattern as `column`).
    fn pairs(&self, person: &str, strap: &str, field: &str, reference: &str) -> Result<Vec<(f64, f64)>, StoreError> {
        let sql = format!(
            "SELECT {field}, {reference} FROM strap_night
             WHERE person=?1 AND strap=?2 AND {field} IS NOT NULL AND {reference} IS NOT NULL ORDER BY day"
        );
        let mut stmt = self.conn.prepare(&sql)?;
        let rows = stmt.query_map(params![person, strap], |r| Ok((r.get::<_, f64>(0)?, r.get::<_, f64>(1)?)))?;
        Ok(rows.filter_map(Result::ok).collect())
    }

    /// Fit a device-relative field column against a reference column over the accumulated (person, strap)
    /// nights, persisting the per-strap coefficient (`reference ≈ scale·field + offset`, plus `r`) once
    /// `milestone.full` paired nights exist. This is how a raw sensor field earns a strap-specific
    /// scale/offset from its own captures instead of borrowing another strap's number. Still `Calibrating`
    /// if there aren't enough nights, or if the field never varied enough to fit.
    pub fn calibrate_fit(
        &self,
        person: &str,
        strap: &str,
        name: &str,
        field: &str,
        reference: &str,
        milestone: calibration::Calibration,
    ) -> Result<FitState, StoreError> {
        let pairs = self.pairs(person, strap, field, reference)?;
        if (pairs.len() as u32) < milestone.full {
            return Ok(FitState::Calibrating { have: pairs.len(), need: milestone.full });
        }
        let (fields, refs): (Vec<f64>, Vec<f64>) = pairs.into_iter().unzip();
        let Some(fit) = stats::linear_fit(&fields, &refs) else {
            return Ok(FitState::Unfittable { nights: fields.len() });
        };
        self.conn.execute(
            "INSERT INTO fit_baseline(person,strap,name,scale,offset,r,nights) VALUES(?1,?2,?3,?4,?5,?6,?7)
             ON CONFLICT(person,strap,name) DO UPDATE SET
               scale=excluded.scale, offset=excluded.offset, r=excluded.r, nights=excluded.nights,
               finalized_at=unixepoch()",
            params![person, strap, name, fit.scale, fit.offset, fit.r, fields.len()],
        )?;
        Ok(FitState::Fit { fit, nights: fields.len() })
    }

    /// The persisted per-strap fit for (person, strap, name), if one has finalized.
    pub fn fit(&self, person: &str, strap: &str, name: &str) -> Result<Option<(LinearFit, usize)>, StoreError> {
        let got = self
            .conn
            .query_row(
                "SELECT scale, offset, r, nights FROM fit_baseline WHERE person=?1 AND strap=?2 AND name=?3",
                params![person, strap, name],
                |r| Ok((LinearFit { scale: r.get(0)?, offset: r.get(1)?, r: r.get(2)? }, r.get::<_, i64>(3)? as usize)),
            )
            .optional()?;
        Ok(got)
    }
}

/// Group history records by calendar day into one summary per day.
fn segment(records: &[Record]) -> Vec<NightMetrics> {
    let mut by_day: std::collections::BTreeMap<u32, DayAcc> = std::collections::BTreeMap::new();
    for r in records {
        if let Record::History(h) = r {
            by_day.entry(h.unix / whoop_metrics::SECS_PER_DAY).or_default().push(h);
        }
    }
    by_day.into_iter().map(|(day, acc)| acc.finish(day)).collect()
}

#[derive(Default)]
struct DayAcc {
    spo2: Vec<f64>,
    beats: Vec<(u32, Vec<u16>)>,
    hr_min: Option<u8>,
    hr_max: Option<u8>,
    records: u32,
}

impl DayAcc {
    fn push(&mut self, h: &HistoryRecord) {
        if let Some(p) = h.spo2_pct {
            self.spo2.push(p as f64);
        }
        if !h.rr_intervals.is_empty() {
            self.beats.push((h.unix, h.rr_intervals.clone()));
        }
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
            rmssd_ms: HrvReadiness::rmssd_gap_aware(&self.beats),
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

    fn hr_hist(unix: u32, hr: u8) -> Record {
        Record::History(HistoryRecord { version: 18, unix, heart_rate: Some(hr), ..Default::default() })
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

    #[test]
    fn nightly_rmssd_is_gap_aware_not_inflated_across_offload_gaps() {
        // A drain where two contiguous records of steady beats are followed by a far-away single 1400 ms
        // beat. The persisted nightly RMSSD must stay small, not blow up on the cross-gap jump.
        let s = Store::open_memory().unwrap();
        let rr = |t: u32, rr: Vec<u16>| {
            Record::History(HistoryRecord { version: 18, unix: t, heart_rate: Some(60), rr_intervals: rr, ..Default::default() })
        };
        s.ingest("1", "A", &[rr(0, vec![600, 610]), rr(1, vec![605, 615]), rr(10_000, vec![1400])]).unwrap();
        let persisted: f64 = s
            .conn
            .query_row("SELECT rmssd_ms FROM strap_night WHERE person='1' AND strap='A'", [], |r| r.get(0))
            .unwrap();
        assert!(persisted < 20.0, "persisted nightly RMSSD should be gap-aware, got {persisted}");
    }

    #[test]
    fn calibrate_fit_recovers_a_per_strap_line_and_persists_it() {
        let s = Store::open_memory().unwrap();
        // Three nights; each night hr_max = 2·hr_min + 10 (a synthetic device-relative → reference line).
        for (day, lo) in [(0u32, 50u8), (1, 55), (2, 60)] {
            let t = day * 86_400;
            s.ingest("1", "A", &[hr_hist(t, lo), hr_hist(t + 10, 2 * lo + 10)]).unwrap();
        }
        let milestone = calibration::RECOVERY_SCORE; // full = 3
        match s.calibrate_fit("1", "A", "hr_span", "hr_min", "hr_max", milestone).unwrap() {
            FitState::Fit { fit, nights } => {
                assert_eq!(nights, 3);
                assert!((fit.scale - 2.0).abs() < 1e-6 && (fit.offset - 10.0).abs() < 1e-6 && (fit.r - 1.0).abs() < 1e-6);
            }
            other => panic!("expected fit, got {other:?}"),
        }
        // persisted and read back
        let (fit, n) = s.fit("1", "A", "hr_span").unwrap().unwrap();
        assert_eq!(n, 3);
        assert!((fit.scale - 2.0).abs() < 1e-6);
        // a different (person, strap) with one night is below the milestone → calibrating, nothing persisted
        s.ingest("2", "B", &[hr_hist(0, 50), hr_hist(10, 110)]).unwrap();
        assert!(matches!(
            s.calibrate_fit("2", "B", "hr_span", "hr_min", "hr_max", milestone).unwrap(),
            FitState::Calibrating { have: 1, need: 3 }
        ));
        assert!(s.fit("2", "B", "hr_span").unwrap().is_none());
    }
}
