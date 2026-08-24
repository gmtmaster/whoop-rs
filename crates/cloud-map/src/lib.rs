//! Maps decoded `whoop-protocol` history records onto the six protocol-free shapes
//! `noop-backend`'s `POST /api/v1/sync/batch` accepts, and computes the stable cloud
//! `sourceId` every record carries.
//!
//! This crate is the one place that boundary is drawn - Java (`noop-backend`) never
//! sees a WHOOP packet, and no client (mobile, `whoopctl`, a future desktop bridge)
//! reimplements this mapping. It depends only on `whoop-protocol`'s already-decoded
//! output structs, not on BLE or the offload state machine, so it composes with any
//! transport.
//!
//! # sourceId
//!
//! WHOOP 5.0/MG's v18 history records carry `record_index` - "a dense per-second
//! emission counter... orders and de-duplicates records independently of arrival,
//! which no timestamp can do within a second" (see `whoop-protocol`). That is real
//! protocol-level identity, not a client-invented one, so it is priority 1 here.
//! WHOOP 4.0 records carry no such counter, but are one-per-second, so `unix` alone
//! is a sufficient, stable fallback (priority 2). Both are direct, deterministic
//! constructions from already-stable decoded fields - there is no hashing anywhere
//! in this crate, because every stream here always has at least `unix`.
//!
//! sourceId shape: `"{device_serial}:{stream}:{record_index_or_unix}"`. Computing it
//! here (not per-client) is what makes "the same physical WHOOP record decoded twice
//! gets the same sourceId" a property of the protocol mapping instead of a promise
//! every client has to keep separately.

use whoop_protocol::records::HistoryRecord;

/// The minimal per-second shape this crate needs to do the mapping - deliberately
/// narrower than `whoop_protocol::records::HistoryRecord` so any decoded
/// representation of "one second" can feed it, not just that exact struct. In
/// particular `whoop-ffi`'s uniffi-exposed `HistorySummary` (a different Rust type
/// with the same fields, generated for the FFI boundary) converts into this too, so
/// the mapping logic lives here exactly once regardless of which crate decoded the
/// bytes.
#[derive(Clone, Debug, Default)]
pub struct DecodedSecond {
    pub unix: i64,
    pub record_index: Option<u32>,
    pub heart_rate: Option<u8>,
    pub rr_intervals: Vec<u16>,
    pub gravity: Option<[f32; 3]>,
    pub steps: Option<u16>,
    pub activity_class: Option<u8>,
    pub sleep_state: Option<u8>,
    pub signal_flags: Option<u8>,
}

impl From<&HistoryRecord> for DecodedSecond {
    fn from(r: &HistoryRecord) -> Self {
        DecodedSecond {
            unix: i64::from(r.unix),
            record_index: r.record_index,
            heart_rate: r.heart_rate,
            rr_intervals: r.rr_intervals.clone(),
            gravity: r.gravity,
            steps: r.steps,
            activity_class: r.activity_class,
            sleep_state: r.sleep_state,
            signal_flags: r.signal_flags,
        }
    }
}

/// Which of noop-backend's six raw streams a record belongs to. `as_str()` matches
/// the JSON field names on `POST /api/v1/sync/batch` exactly, so it doubles as
/// documentation of the wire contract.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CloudStream {
    HeartRate,
    RrRuns,
    Accelerometer,
    Steps,
    WristOff,
    BandSleepState,
}

impl CloudStream {
    pub fn as_str(self) -> &'static str {
        match self {
            CloudStream::HeartRate => "heartRate",
            CloudStream::RrRuns => "rrRuns",
            CloudStream::Accelerometer => "accelerometer",
            CloudStream::Steps => "steps",
            CloudStream::WristOff => "wristOff",
            CloudStream::BandSleepState => "bandSleepState",
        }
    }
}

/// The stable cloud sourceId for one record. `record_index` (5.0/MG v18) is
/// preferred when present; `unix` (always present) is the fallback used on 4.0 and
/// on any record that didn't carry one. Never a hash - both inputs are already
/// stable, protocol-level identity.
pub fn source_id(device_serial: &str, stream: CloudStream, unix: i64, record_index: Option<u32>) -> String {
    match record_index {
        Some(idx) => format!("{device_serial}:{}:{idx}", stream.as_str()),
        None => format!("{device_serial}:{}:{unix}", stream.as_str()),
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct CloudHeartRateRecord {
    pub source_id: String,
    pub unix: i64,
    pub bpm: u8,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CloudRrRun {
    pub source_id: String,
    pub unix: i64,
    /// Ordered RR interval list (ms-ticks), exactly as decoded - order is
    /// physiologically meaningful and must never be re-sorted.
    pub intervals: Vec<u16>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CloudAccelRecord {
    pub source_id: String,
    pub unix: i64,
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CloudStepRecord {
    pub source_id: String,
    pub unix: i64,
    pub counter: u16,
    pub activity_class: Option<u8>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CloudWristOffRecord {
    pub source_id: String,
    pub start: i64,
    pub end: i64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CloudBandSleepStateRecord {
    pub source_id: String,
    pub unix: i64,
    pub state: u8,
}

/// One mapped batch, shaped exactly like `POST /api/v1/sync/batch`'s body minus
/// `deviceId`/`batchId` (the caller's transport layer owns those). Every field maps
/// 1:1 onto that endpoint's arrays.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CloudRawBatch {
    pub heart_rate: Vec<CloudHeartRateRecord>,
    pub rr_runs: Vec<CloudRrRun>,
    pub accelerometer: Vec<CloudAccelRecord>,
    pub steps: Vec<CloudStepRecord>,
    pub wrist_off: Vec<CloudWristOffRecord>,
    pub band_sleep_state: Vec<CloudBandSleepStateRecord>,
}

/// v18's `signal_flags` bit 4 = off-wrist (see `whoop-protocol::records::HistoryRecord`).
const OFF_WRIST_BIT: u8 = 0x10;

/// Maps a batch of already-decoded history records (any mix of 4.0/5.0 versions, as
/// long as they share one physical device) onto the six cloud streams.
///
/// `records` should be in chronological order (as they come off `sync_history`/the
/// offload drain) - wrist-off interval derivation depends on it. Records missing a
/// given field simply don't contribute to that stream; nothing is fabricated.
///
/// # Wrist-off derivation
/// `signal_flags` bit 4 is a per-second on/off-wrist flag, not an interval - this
/// function edge-detects it into `[start, end)` intervals. An interval's sourceId is
/// derived from its *start* record only (the real decoded transition), not its end,
/// so re-observing the same off-wrist period with a longer tail on a later,
/// more-complete decode reuses the same sourceId - the backend's dedupe then keeps
/// whichever end was uploaded first. Rare in practice (this only matters for an
/// interval still open at the edge of a partial decode) and noted here rather than
/// hidden.
pub fn map_history_records(device_serial: &str, records: &[HistoryRecord]) -> CloudRawBatch {
    let seconds: Vec<DecodedSecond> = records.iter().map(DecodedSecond::from).collect();
    map_decoded_seconds(device_serial, &seconds)
}

/// Same mapping as [`map_history_records`], but over the crate-agnostic
/// [`DecodedSecond`] shape - this is what a caller with its own decoded-record type
/// (e.g. `whoop-ffi`'s uniffi `HistorySummary`) converts into and calls directly,
/// instead of duplicating the field-by-field mapping logic below.
pub fn map_decoded_seconds(device_serial: &str, records: &[DecodedSecond]) -> CloudRawBatch {
    let mut batch = CloudRawBatch::default();
    let mut wrist_off_start: Option<(i64, Option<u32>)> = None;

    for record in records {
        let unix = record.unix;
        let idx = record.record_index;

        if let Some(bpm) = record.heart_rate {
            batch.heart_rate.push(CloudHeartRateRecord {
                source_id: source_id(device_serial, CloudStream::HeartRate, unix, idx),
                unix,
                bpm,
            });
        }

        if !record.rr_intervals.is_empty() {
            batch.rr_runs.push(CloudRrRun {
                source_id: source_id(device_serial, CloudStream::RrRuns, unix, idx),
                unix,
                intervals: record.rr_intervals.clone(),
            });
        }

        if let Some(g) = record.gravity {
            batch.accelerometer.push(CloudAccelRecord {
                source_id: source_id(device_serial, CloudStream::Accelerometer, unix, idx),
                unix,
                x: g[0],
                y: g[1],
                z: g[2],
            });
        }

        if let Some(counter) = record.steps {
            batch.steps.push(CloudStepRecord {
                source_id: source_id(device_serial, CloudStream::Steps, unix, idx),
                unix,
                counter,
                activity_class: record.activity_class,
            });
        }

        if let Some(state) = record.sleep_state {
            batch.band_sleep_state.push(CloudBandSleepStateRecord {
                source_id: source_id(device_serial, CloudStream::BandSleepState, unix, idx),
                unix,
                state,
            });
        }

        if let Some(flags) = record.signal_flags {
            let off_wrist = flags & OFF_WRIST_BIT != 0;
            match (off_wrist, wrist_off_start) {
                (true, None) => wrist_off_start = Some((unix, idx)),
                (false, Some((start, start_idx))) => {
                    batch.wrist_off.push(CloudWristOffRecord {
                        source_id: source_id(device_serial, CloudStream::WristOff, start, start_idx),
                        start,
                        end: unix,
                    });
                    wrist_off_start = None;
                }
                _ => {}
            }
        }
    }

    batch
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(unix: u32, record_index: Option<u32>) -> HistoryRecord {
        HistoryRecord { version: 18, unix, record_index, ..Default::default() }
    }

    #[test]
    fn source_id_prefers_record_index_over_unix() {
        let with_index = source_id("SN1", CloudStream::HeartRate, 100, Some(42));
        let without_index = source_id("SN1", CloudStream::HeartRate, 100, None);
        assert_eq!(with_index, "SN1:heartRate:42");
        assert_eq!(without_index, "SN1:heartRate:100");
        assert_ne!(with_index, without_index);
    }

    #[test]
    fn source_id_is_deterministic_for_the_same_inputs() {
        let a = source_id("SN1", CloudStream::RrRuns, 100, Some(7));
        let b = source_id("SN1", CloudStream::RrRuns, 100, Some(7));
        assert_eq!(a, b);
    }

    #[test]
    fn source_id_differs_across_streams_and_devices() {
        let hr = source_id("SN1", CloudStream::HeartRate, 100, Some(7));
        let rr = source_id("SN1", CloudStream::RrRuns, 100, Some(7));
        let other_device = source_id("SN2", CloudStream::HeartRate, 100, Some(7));
        assert_ne!(hr, rr);
        assert_ne!(hr, other_device);
    }

    #[test]
    fn maps_heart_rate_and_rr_from_one_record() {
        let mut r = record(1_700_000_000, Some(5));
        r.heart_rate = Some(62);
        r.rr_intervals = vec![950, 960, 940];
        let batch = map_history_records("SN1", &[r]);
        assert_eq!(batch.heart_rate.len(), 1);
        assert_eq!(batch.heart_rate[0].bpm, 62);
        assert_eq!(batch.rr_runs.len(), 1);
        assert_eq!(batch.rr_runs[0].intervals, vec![950, 960, 940]); // order preserved
        assert_eq!(batch.rr_runs[0].source_id, "SN1:rrRuns:5");
    }

    #[test]
    fn absent_fields_contribute_nothing_to_their_stream() {
        let r = record(1_700_000_000, Some(1)); // heart_rate, rr, etc. all default/empty
        let batch = map_history_records("SN1", &[r]);
        assert!(batch.heart_rate.is_empty());
        assert!(batch.rr_runs.is_empty());
        assert!(batch.accelerometer.is_empty());
        assert!(batch.steps.is_empty());
        assert!(batch.band_sleep_state.is_empty());
    }

    #[test]
    fn wrist_off_edge_detects_a_closed_interval() {
        let mut on1 = record(100, Some(1));
        on1.signal_flags = Some(0x00);
        let mut off1 = record(101, Some(2));
        off1.signal_flags = Some(0x10); // off-wrist bit set
        let mut off2 = record(102, Some(3));
        off2.signal_flags = Some(0x10);
        let mut on2 = record(103, Some(4));
        on2.signal_flags = Some(0x00); // back on-wrist

        let batch = map_history_records("SN1", &[on1, off1, off2, on2]);
        assert_eq!(batch.wrist_off.len(), 1);
        assert_eq!(batch.wrist_off[0].start, 101);
        assert_eq!(batch.wrist_off[0].end, 103);
        assert_eq!(batch.wrist_off[0].source_id, "SN1:wristOff:2"); // keyed off the start record
    }

    #[test]
    fn wrist_off_still_open_at_batch_end_emits_nothing() {
        let mut on1 = record(100, Some(1));
        on1.signal_flags = Some(0x00);
        let mut off1 = record(101, Some(2));
        off1.signal_flags = Some(0x10);
        // Batch ends still off-wrist - no closing record, so no interval yet.
        let batch = map_history_records("SN1", &[on1, off1]);
        assert!(batch.wrist_off.is_empty());
    }

    #[test]
    fn same_records_decoded_twice_yield_identical_source_ids() {
        let mut r1 = record(1_700_000_000, Some(9));
        r1.heart_rate = Some(70);
        let mut r2 = record(1_700_000_000, Some(9));
        r2.heart_rate = Some(70);

        let first = map_history_records("SN1", &[r1]);
        let second = map_history_records("SN1", &[r2]);
        assert_eq!(first.heart_rate[0].source_id, second.heart_rate[0].source_id);
    }
}
