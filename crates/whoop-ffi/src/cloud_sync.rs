//! uniffi surface for `cloud-map`: turns already-decoded `HistorySummary` records
//! into the six protocol-free shapes `noop-backend`'s `POST /api/v1/sync/batch`
//! expects, each carrying a stable `sourceId`. All the actual mapping/identity logic
//! lives in the `cloud-map` crate (shared with `whoopctl` and any other Rust
//! caller) - this module is only the FFI shape plus the `HistorySummary` ->
//! `cloud_map::DecodedSecond` bridge, so the logic is never duplicated per client.
//!
//! The mobile-side loop this is built for: accumulate the `HistorySummary` values
//! from `WhoopCodec::feed`'s `Step::Record` results for one offload drain (or since
//! the last successful upload), then call `map_history_to_cloud_batch` once with the
//! device's serial number (read via `Response::Hello`/`BatteryPack`, standard GATT,
//! or `Response::Hello.device_name` — whatever the platform's identity read
//! surfaces) to get a ready-to-serialize batch. Swift/Kotlin never sees a WHOOP
//! packet field name past this point.

use crate::*;

fn to_decoded_second(s: &HistorySummary) -> cloud_map::DecodedSecond {
    cloud_map::DecodedSecond {
        unix: i64::from(s.unix),
        record_index: s.record_index,
        heart_rate: s.heart_rate,
        rr_intervals: s.rr_intervals.clone(),
        gravity: s.gravity.as_ref().and_then(|g| match g.as_slice() {
            [x, y, z] => Some([*x, *y, *z]),
            _ => None,
        }),
        steps: s.steps,
        activity_class: s.activity_class,
        sleep_state: s.sleep_state,
        signal_flags: s.signal_flags,
    }
}

/// One heart-rate record ready for `POST /api/v1/sync/batch`'s `heartRate` array.
#[derive(uniffi::Record)]
pub struct CloudHeartRateRecord {
    pub source_id: String,
    pub unix: i64,
    pub bpm: u8,
}

impl From<cloud_map::CloudHeartRateRecord> for CloudHeartRateRecord {
    fn from(r: cloud_map::CloudHeartRateRecord) -> Self {
        CloudHeartRateRecord { source_id: r.source_id, unix: r.unix, bpm: r.bpm }
    }
}

/// One RR run ready for the `rrRuns` array. `intervals` order is preserved exactly
/// as decoded - it is physiologically meaningful and must never be re-sorted.
#[derive(uniffi::Record)]
pub struct CloudRrRun {
    pub source_id: String,
    pub unix: i64,
    pub intervals: Vec<u16>,
}

impl From<cloud_map::CloudRrRun> for CloudRrRun {
    fn from(r: cloud_map::CloudRrRun) -> Self {
        CloudRrRun { source_id: r.source_id, unix: r.unix, intervals: r.intervals }
    }
}

/// One accelerometer (gravity) record ready for the `accelerometer` array.
#[derive(uniffi::Record)]
pub struct CloudAccelRecord {
    pub source_id: String,
    pub unix: i64,
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl From<cloud_map::CloudAccelRecord> for CloudAccelRecord {
    fn from(r: cloud_map::CloudAccelRecord) -> Self {
        CloudAccelRecord { source_id: r.source_id, unix: r.unix, x: r.x, y: r.y, z: r.z }
    }
}

/// One step-counter record ready for the `steps` array.
#[derive(uniffi::Record)]
pub struct CloudStepRecord {
    pub source_id: String,
    pub unix: i64,
    pub counter: u16,
    pub activity_class: Option<u8>,
}

impl From<cloud_map::CloudStepRecord> for CloudStepRecord {
    fn from(r: cloud_map::CloudStepRecord) -> Self {
        CloudStepRecord { source_id: r.source_id, unix: r.unix, counter: r.counter, activity_class: r.activity_class }
    }
}

/// One off-wrist interval ready for the `wristOff` array. Derived from `signal_flags`
/// bit 4's on/off edges - see `cloud-map` for the derivation and its sourceId caveat.
#[derive(uniffi::Record)]
pub struct CloudWristOffRecord {
    pub source_id: String,
    pub start: i64,
    pub end: i64,
}

impl From<cloud_map::CloudWristOffRecord> for CloudWristOffRecord {
    fn from(r: cloud_map::CloudWristOffRecord) -> Self {
        CloudWristOffRecord { source_id: r.source_id, start: r.start, end: r.end }
    }
}

/// One band sleep-state record ready for the `bandSleepState` array.
#[derive(uniffi::Record)]
pub struct CloudBandSleepStateRecord {
    pub source_id: String,
    pub unix: i64,
    pub state: u8,
}

impl From<cloud_map::CloudBandSleepStateRecord> for CloudBandSleepStateRecord {
    fn from(r: cloud_map::CloudBandSleepStateRecord) -> Self {
        CloudBandSleepStateRecord { source_id: r.source_id, unix: r.unix, state: r.state }
    }
}

/// One mapped batch - shaped 1:1 onto `POST /api/v1/sync/batch`'s body (minus
/// `deviceId`/`batchId`, which the client's transport layer owns). Only streams
/// `whoop-rs` has actually decoded appear here: accelerometer/steps/wristOff/
/// bandSleepState are only populated where the underlying `HistorySummary` carried
/// that field (5.0/MG v18) - nothing is fabricated for 4.0 or an unmapped version.
#[derive(uniffi::Record)]
pub struct CloudRawBatch {
    pub heart_rate: Vec<CloudHeartRateRecord>,
    pub rr_runs: Vec<CloudRrRun>,
    pub accelerometer: Vec<CloudAccelRecord>,
    pub steps: Vec<CloudStepRecord>,
    pub wrist_off: Vec<CloudWristOffRecord>,
    pub band_sleep_state: Vec<CloudBandSleepStateRecord>,
}

impl From<cloud_map::CloudRawBatch> for CloudRawBatch {
    fn from(b: cloud_map::CloudRawBatch) -> Self {
        CloudRawBatch {
            heart_rate: b.heart_rate.into_iter().map(Into::into).collect(),
            rr_runs: b.rr_runs.into_iter().map(Into::into).collect(),
            accelerometer: b.accelerometer.into_iter().map(Into::into).collect(),
            steps: b.steps.into_iter().map(Into::into).collect(),
            wrist_off: b.wrist_off.into_iter().map(Into::into).collect(),
            band_sleep_state: b.band_sleep_state.into_iter().map(Into::into).collect(),
        }
    }
}

/// Maps a batch of `HistorySummary` records (accumulated from `WhoopCodec::feed`'s
/// `Step::Record` results, in chronological order) onto noop-backend's raw sync
/// contract, with a stable, deterministic `sourceId` on every record. `device_serial`
/// should be the strap's GATT serial (0x2A25) - the same value stays stable across
/// reconnects and app restarts, which is what makes the sourceId stable too.
#[uniffi::export]
pub fn map_history_to_cloud_batch(device_serial: String, summaries: Vec<HistorySummary>) -> CloudRawBatch {
    let seconds: Vec<cloud_map::DecodedSecond> = summaries.iter().map(to_decoded_second).collect();
    cloud_map::map_decoded_seconds(&device_serial, &seconds).into()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn summary(unix: u32, record_index: Option<u32>) -> HistorySummary {
        HistorySummary {
            version: 18,
            unix,
            heart_rate: None,
            rr_intervals: Vec::new(),
            gravity: None,
            skin_temp_c: None,
            skin_temp_raw: None,
            spo2_red: None,
            spo2_ir: None,
            spo2_pct: None,
            resp_raw: None,
            steps: None,
            activity_class: None,
            sleep_state: None,
            signal_flags: None,
            signal_quality: None,
            dynamic_acceleration_g: None,
            optical_baseline_a: None,
            optical_baseline_b: None,
            optical_amp_a: None,
            optical_amp_b: None,
            optical_signal_poor: None,
            record_index,
            temp_aux_1_raw: None,
            temp_aux_2_raw: None,
            sleep_state_raw: None,
            raw_u8_28: None,
            raw_u8_29: None,
            raw_u16_30: None,
            raw_f32_105: None,
            raw_u16_26: None,
            unpinned: None,
        }
    }

    #[test]
    fn maps_a_history_summary_batch_with_stable_source_ids() {
        let mut s = summary(1_700_000_000, Some(5));
        s.heart_rate = Some(58);
        s.gravity = Some(vec![0.01, 0.02, 0.98]);

        let batch = map_history_to_cloud_batch("SN-ABC".into(), vec![s]);
        assert_eq!(batch.heart_rate.len(), 1);
        assert_eq!(batch.heart_rate[0].bpm, 58);
        assert_eq!(batch.heart_rate[0].source_id, "SN-ABC:heartRate:5");
        assert_eq!(batch.accelerometer.len(), 1);
        assert_eq!(batch.accelerometer[0].z, 0.98);
    }

    #[test]
    fn malformed_gravity_vector_is_dropped_not_guessed() {
        let mut s = summary(1_700_000_001, Some(6));
        s.gravity = Some(vec![0.5]); // not 3 components - can't happen from a real decode, but guard it
        let batch = map_history_to_cloud_batch("SN-ABC".into(), vec![s]);
        assert!(batch.accelerometer.is_empty());
    }
}
