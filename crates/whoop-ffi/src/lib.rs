//! uniffi surface over the pure WHOOP codec. The mobile apps do BLE natively (Android BluetoothGatt,
//! iOS CoreBluetooth) — including the universal easy-connect (attach to an already-connected band) — and
//! feed notification bytes here; this returns decoded records and the exact frames to write back. No
//! async, no radio crosses the FFI: the sans-IO `Offload` is what makes one Rust core serve both apps.

uniffi::setup_scaffolding!();

use std::sync::{Arc, Mutex};

use whoop_protocol::deframe::DeframerMap;
use whoop_protocol::hello::GEN5_CLIENT_HELLO;
use whoop_protocol::{config, framing, records, Channel, Family, Offload, OffloadStep, Record};

#[derive(uniffi::Enum, Clone, Copy)]
pub enum Gen {
    Gen4,
    Gen5,
}

impl From<Gen> for Family {
    fn from(g: Gen) -> Self {
        match g {
            Gen::Gen4 => Family::Gen4,
            Gen::Gen5 => Family::Gen5,
        }
    }
}

/// The logical notify channel a native notification arrived on (mapped from its characteristic UUID).
#[derive(uniffi::Enum, Clone, Copy)]
pub enum Chan {
    CmdNotify,
    Events,
    Data,
    Console,
}

impl From<Chan> for Channel {
    fn from(c: Chan) -> Self {
        match c {
            Chan::CmdNotify => Channel::CmdNotify,
            Chan::Events => Channel::Events,
            Chan::Data => Channel::Data,
            Chan::Console => Channel::Console,
        }
    }
}

/// A per-second history summary handed to the app (v18 on 5.0/MG, v5/v24 on 4.0).
#[derive(uniffi::Record, Clone)]
pub struct HistorySummary {
    pub version: u8,
    pub unix: u32,
    pub heart_rate: Option<u8>,
    pub rr_intervals: Vec<u16>,
    pub skin_temp_c: Option<f32>,
    pub sleep_state: Option<u8>,
}

impl From<records::HistoryRecord> for HistorySummary {
    fn from(h: records::HistoryRecord) -> Self {
        HistorySummary {
            version: h.version,
            unix: h.unix,
            heart_rate: h.heart_rate,
            rr_intervals: h.rr_intervals,
            skin_temp_c: h.skin_temp_c,
            sleep_state: h.sleep_state,
        }
    }
}

/// One action the app must take while draining history.
#[derive(uniffi::Enum)]
pub enum Step {
    /// A decoded record — persist it.
    Record { summary: HistorySummary },
    /// A raw 24 Hz optical buffer (v26): a single AC-coupled waveform (no red/IR pair).
    Ppg { unix: u32, record_id: Option<u16>, samples: Vec<i16> },
    /// A raw 6-axis IMU buffer (v21): 100 Hz accel + gyro, interleaved x,y,z per sample (raw i16 LSB —
    /// scale accel by 1/4096 for g, gyro by 2000/32768 for deg/s).
    Imu { unix: u32, sample_rate_hz: u16, accel: Vec<i16>, gyro: Vec<i16> },
    /// Write these bytes (confirmed) to the command characteristic — the mandatory HISTORY_END ACK.
    Ack { frame: Vec<u8> },
    /// The drain finished.
    Complete,
}

fn to_step(s: OffloadStep) -> Step {
    match s {
        OffloadStep::Record(Record::History(h)) => Step::Record { summary: h.into() },
        OffloadStep::Record(Record::Ppg(p)) => Step::Ppg { unix: p.unix, record_id: p.record_id, samples: p.samples },
        OffloadStep::Record(Record::Imu(m)) => Step::Imu {
            unix: m.unix,
            sample_rate_hz: m.sample_rate_hz,
            accel: m.accel.into_iter().flatten().collect(),
            gyro: m.gyro.into_iter().flatten().collect(),
        },
        OffloadStep::Ack(frame) => Step::Ack { frame },
        OffloadStep::Complete => Step::Complete,
    }
}

/// The stateful codec the app drives: one per connected band. Interior-mutable so it presents the
/// `&self` methods uniffi objects require while owning the reassembler + offload state.
#[derive(uniffi::Object)]
pub struct WhoopCodec {
    family: Family,
    inner: Mutex<Inner>,
}

struct Inner {
    deframers: DeframerMap,
    offload: Offload,
}

#[uniffi::export]
impl WhoopCodec {
    #[uniffi::constructor]
    pub fn new(gen: Gen) -> Arc<Self> {
        let family: Family = gen.into();
        Arc::new(WhoopCodec {
            family,
            inner: Mutex::new(Inner { deframers: DeframerMap::new(family), offload: Offload::new(family) }),
        })
    }

    /// The GEN5 bond-open frame — write it (confirmed) after connecting to trigger the just-works bond.
    pub fn client_hello(&self) -> Vec<u8> {
        GEN5_CLIENT_HELLO.to_vec()
    }

    /// The SEND_HISTORICAL_DATA frame that starts a drain.
    pub fn offload_start(&self) -> Vec<u8> {
        self.inner.lock().unwrap().offload.start_frame()
    }

    /// The 16 R22 deep-data enable frames (write each confirmed, ~80 ms apart). Reversible.
    pub fn r22_frames(&self) -> Vec<Vec<u8>> {
        config::r22_frames(0)
    }

    /// Feed one native notification (its channel + bytes): reassembles frames and drives the offload,
    /// returning the steps to perform (persist records, write ACKs, stop on Complete).
    pub fn feed(&self, chan: Chan, bytes: Vec<u8>) -> Vec<Step> {
        let mut inner = self.inner.lock().unwrap();
        let frames = inner.deframers.push(chan.into(), &bytes);
        let mut out = Vec::new();
        for f in frames {
            for step in inner.offload.on_frame(&f) {
                out.push(to_step(step));
            }
        }
        out
    }

    /// Decode a single complete history frame (for offline replay of a captured file).
    pub fn decode_history(&self, raw: Vec<u8>) -> Option<HistorySummary> {
        let frame = framing::decode(self.family, &raw).ok()?;
        match records::decode(&frame)? {
            Record::History(h) => Some(h.into()),
            _ => None,
        }
    }

    /// Drop any buffered partial frames — call on (re)connect.
    pub fn reset(&self) {
        self.inner.lock().unwrap().deframers.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::{Chan, Gen, Step, WhoopCodec};
    use whoop_protocol::hello::GEN5_CLIENT_HELLO;
    use whoop_protocol::{command, framing, Family};

    fn v18_frame() -> Vec<u8> {
        let mut p = vec![0u8; 40];
        p[4..8].copy_from_slice(&1_784_000_000u32.to_le_bytes()); // unix @ inner 7
        p[11] = 96; // hr @ inner 14
        framing::encode(Family::Gen5, 47, 18, 0, &p)
    }

    fn v21_imu_frame() -> Vec<u8> {
        let mut p = vec![0u8; 1229];
        p[13..15].copy_from_slice(&100u16.to_le_bytes()); // count_a @ inner 16
        p[17..19].copy_from_slice(&4096i16.to_le_bytes()); // ax[0]
        p[619..621].copy_from_slice(&100u16.to_le_bytes()); // count_b @ inner 622
        framing::encode(Family::Gen5, 47, 21, 0, &p)
    }

    #[test]
    fn feed_v18_yields_a_record_step() {
        let codec = WhoopCodec::new(Gen::Gen5);
        let steps = codec.feed(Chan::Data, v18_frame());
        assert_eq!(steps.len(), 1);
        match &steps[0] {
            Step::Record { summary } => {
                assert_eq!(summary.heart_rate, Some(96));
                assert_eq!(summary.unix, 1_784_000_000);
            }
            _ => panic!("expected Record"),
        }
    }

    #[test]
    fn feed_v21_yields_an_imu_step() {
        let codec = WhoopCodec::new(Gen::Gen5);
        let steps = codec.feed(Chan::Data, v21_imu_frame());
        assert_eq!(steps.len(), 1);
        match &steps[0] {
            Step::Imu { sample_rate_hz, accel, gyro, .. } => {
                assert_eq!(*sample_rate_hz, 100);
                assert_eq!(accel.len(), 300);
                assert_eq!(gyro.len(), 300);
                assert_eq!(accel[0], 4096);
            }
            _ => panic!("expected Imu"),
        }
    }

    #[test]
    fn decode_history_roundtrips_v18() {
        let codec = WhoopCodec::new(Gen::Gen5);
        assert_eq!(codec.decode_history(v18_frame()).unwrap().heart_rate, Some(96));
    }

    #[test]
    fn r22_frames_are_sixteen_set_configs() {
        let frames = WhoopCodec::new(Gen::Gen5).r22_frames();
        assert_eq!(frames.len(), 16);
        for f in &frames {
            assert_eq!(framing::decode(Family::Gen5, f).unwrap().cmd(), command::SET_CONFIG);
        }
    }

    #[test]
    fn client_hello_is_the_bond_frame() {
        assert_eq!(WhoopCodec::new(Gen::Gen5).client_hello(), GEN5_CLIENT_HELLO.to_vec());
    }

    #[test]
    fn reset_drops_a_buffered_partial_frame() {
        let codec = WhoopCodec::new(Gen::Gen5);
        let full = v18_frame();
        let _ = codec.feed(Chan::Data, full[..5].to_vec()); // partial
        codec.reset();
        assert_eq!(codec.feed(Chan::Data, full).len(), 1); // clean frame still decodes
    }
}
