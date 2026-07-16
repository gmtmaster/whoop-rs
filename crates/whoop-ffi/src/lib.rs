//! uniffi surface over the pure WHOOP codec. The mobile apps do BLE natively (Android BluetoothGatt,
//! iOS CoreBluetooth) — including the universal easy-connect (attach to an already-connected band) — and
//! feed notification bytes here; this returns decoded records and the exact frames to write back. No
//! async, no radio crosses the FFI: the sans-IO `Offload` is what makes one Rust core serve both apps.

uniffi::setup_scaffolding!();

use std::sync::{Arc, Mutex};

use whoop_metrics::{ppg_hr, HrvReadiness, Spo2};
use whoop_protocol::deframe::DeframerMap;
use whoop_protocol::hello::GEN5_CLIENT_HELLO;
use whoop_protocol::{
    alarm, command, config, console, framing, haptic, live, records, response, Channel, Family, Offload,
    OffloadStep, PacketType, Record,
};

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
    pub gravity: Option<Vec<f32>>, // [x, y, z]
    pub skin_temp_c: Option<f32>,
    pub skin_temp_raw: Option<u16>, // raw register; the consumer applies its family/device-specific °C scale
    pub spo2_red: Option<u16>, // 4.0 v24 raw red ADC
    pub spo2_ir: Option<u16>,  // 4.0 v24 raw IR ADC
    pub spo2_pct: Option<u8>,  // 5.0 v18 computed %, sleep-only
    pub resp_raw: Option<u16>, // 4.0 v24 raw respiration ADC
    pub steps: Option<u16>,
    pub activity_class: Option<u8>,
    pub sleep_state: Option<u8>,
    pub signal_flags: Option<u8>,   // 5.0 v18 PPG SIGPROC bitfield (bit 4 = off-wrist)
    pub signal_quality: Option<u8>, // 5.0 v18 PPG confidence, 255 = clean
}

impl From<records::HistoryRecord> for HistorySummary {
    fn from(h: records::HistoryRecord) -> Self {
        HistorySummary {
            version: h.version,
            unix: h.unix,
            heart_rate: h.heart_rate,
            rr_intervals: h.rr_intervals,
            gravity: h.gravity.map(|g| g.to_vec()),
            skin_temp_c: h.skin_temp_c,
            skin_temp_raw: h.skin_temp_raw,
            spo2_red: h.spo2.map(|(r, _)| r),
            spo2_ir: h.spo2.map(|(_, i)| i),
            spo2_pct: h.spo2_pct,
            resp_raw: h.resp_raw,
            steps: h.steps,
            activity_class: h.activity_class,
            sleep_state: h.sleep_state,
            signal_flags: h.signal_flags,
            signal_quality: h.signal_quality,
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

/// A live-notify frame decoded off the realtime / event / console channels (separate from the offload path).
#[derive(uniffi::Enum)]
pub enum Live {
    Realtime { unix: u32, heart_rate: u8, rr_intervals: Vec<u16> },
    R22 { unix: u32, hr_ch1: u8, hr_ch2: Option<u8>, accel: Vec<f32> },
    Event { number: u8, unix: u32 },
    Battery { soc_percent: f32, millivolts: u16, charging: bool },
    Console { text: String },
}

/// A decoded command response (identity, battery, clock, data range, firmware).
#[derive(uniffi::Enum)]
pub enum Response {
    Battery { percent: f32 },
    Clock { unix: u32 },
    Hello { device_name: String, fw_version: Option<Vec<u8>> },
    DataRange { oldest: u32, newest: u32 },
    Version { fw: Vec<u32> },
    Other { cmd: u8, result: Option<u8> },
}

/// One haptic buzz pulse; a clock chime is a sequence of these (write a buzz per pulse).
#[derive(uniffi::Record)]
pub struct Pulse {
    pub duration_ms: u32,
    pub gap_ms: u32,
}

/// One raw PPG sample for `ppg_hr` (wall-clock second + raw ADC value).
#[derive(uniffi::Record)]
pub struct PpgSample {
    pub ts: i64,
    pub value: i32,
}

/// A derived HR estimate from `ppg_hr`.
#[derive(uniffi::Record)]
pub struct PpgEstimate {
    pub ts: i64,
    pub bpm: i32,
    pub conf: f64,
}

/// One record's R-R for gap-aware RMSSD (the app builds these from a `HistorySummary`'s unix + rr_intervals).
#[derive(uniffi::Record)]
pub struct RrRun {
    pub unix: u32,
    pub rr: Vec<u16>,
}

#[derive(uniffi::Enum)]
pub enum ReadinessTier {
    Primed,
    Normal,
    Suppressed,
}

impl From<whoop_metrics::ReadinessTier> for ReadinessTier {
    fn from(t: whoop_metrics::ReadinessTier) -> Self {
        match t {
            whoop_metrics::ReadinessTier::Primed => ReadinessTier::Primed,
            whoop_metrics::ReadinessTier::Normal => ReadinessTier::Normal,
            whoop_metrics::ReadinessTier::Suppressed => ReadinessTier::Suppressed,
        }
    }
}

/// An HRV-readiness reading (log-domain baseline vs the personal normal band).
#[derive(uniffi::Record)]
pub struct HrvReadinessInfo {
    pub tier: ReadinessTier,
    pub baseline7_ms: f64,
    pub normal_low_ms: f64,
    pub normal_high_ms: f64,
    pub overreaching_watch: bool,
}

fn result_to_u8(r: whoop_protocol::event::ResultCode) -> u8 {
    use whoop_protocol::event::ResultCode::*;
    match r {
        Failure => 0,
        Success => 1,
        Pending => 2,
        Unsupported => 3,
        Unknown(x) => x,
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

    /// Decode a single live-notify frame (realtime HR/R-R, on-wrist r22 biometric, event/battery, or
    /// console text). Stateless; the app routes live-channel frames here, offload stays in `feed`.
    pub fn decode_live(&self, raw: Vec<u8>) -> Option<Live> {
        let f = framing::decode(self.family, &raw).ok()?;
        match f.packet().canonical() {
            PacketType::RealtimeData => live::realtime(&f).map(|r| Live::Realtime {
                unix: r.timestamp,
                heart_rate: r.heart_rate,
                rr_intervals: r.rr_intervals,
            }),
            PacketType::Event => live::battery_event(&f)
                .map(|b| Live::Battery { soc_percent: b.soc_percent, millivolts: b.millivolts, charging: b.charging })
                .or_else(|| live::event(&f).map(|e| Live::Event { number: e.number, unix: e.timestamp })),
            PacketType::ConsoleLogs => console::text(&f).map(|text| Live::Console { text }),
            PacketType::HistoricalData if matches!(f.cmd(), 0x80 | 0x82) => live::r22_live(&f).map(|r| Live::R22 {
                unix: r.timestamp,
                hr_ch1: r.hr_ch1,
                hr_ch2: r.hr_ch2,
                accel: r.accel.to_vec(),
            }),
            _ => None,
        }
    }

    /// Decode a single command-response frame (identity/battery/clock/data-range/firmware).
    pub fn decode_response(&self, raw: Vec<u8>) -> Option<Response> {
        use response::CommandResponse as C;
        let f = framing::decode(self.family, &raw).ok()?;
        Some(match response::decode(&f)? {
            C::Battery { percent } => Response::Battery { percent },
            C::Clock { unix } => Response::Clock { unix },
            C::Hello { device_name, fw_version } => {
                Response::Hello { device_name, fw_version: fw_version.map(|v| v.to_vec()) }
            }
            C::DataRange { oldest, newest } => Response::DataRange { oldest, newest },
            C::VersionInfo { fw } => Response::Version { fw: fw.to_vec() },
            C::Other { cmd, result } => Response::Other { cmd, result: result.map(result_to_u8) },
        })
    }

    // --- Command frames: return the bytes for the app to write (confirmed); the FFI never writes. ---

    /// Cleanly abort an in-flight history drain.
    pub fn offload_abort(&self) -> Vec<u8> {
        self.inner.lock().unwrap().offload.abort_frame()
    }

    /// GET_HELLO (identity + firmware), family-forked. The `[0x00]` payload matches the client's default
    /// no-arg body (frame-identical to empty on Gen5 after padding; it is the trailing byte Gen4 expects).
    pub fn get_hello_frame(&self, seq: u8) -> Vec<u8> {
        match self.family {
            Family::Gen5 => framing::command(self.family, seq, command::GET_HELLO, &[0x01]),
            Family::Gen4 => framing::command(self.family, seq, command::GET_HELLO_HARVARD, &[0x00]),
        }
    }

    /// GET_BATTERY_LEVEL (also the bond-establishing write on 5/MG).
    pub fn get_battery_frame(&self, seq: u8) -> Vec<u8> {
        framing::command(self.family, seq, command::GET_BATTERY_LEVEL, &[0x00])
    }

    /// GET_DATA_RANGE (the oldest/newest banked seconds).
    pub fn get_data_range_frame(&self, seq: u8) -> Vec<u8> {
        framing::command(self.family, seq, command::GET_DATA_RANGE, &[0x00])
    }

    /// Stop the live type-43 raw flood (sent during the handshake).
    pub fn stop_raw_flood_frame(&self, seq: u8) -> Vec<u8> {
        framing::command(self.family, seq, command::SEND_R10_R11_REALTIME, &[0])
    }

    /// Toggle the realtime HR/R-R stream on the vendor channel.
    pub fn toggle_realtime_hr_frame(&self, seq: u8, on: bool) -> Vec<u8> {
        framing::command(self.family, seq, command::TOGGLE_REALTIME_HR, &[u8::from(on)])
    }

    /// Warm-reboot the strap (data kept). Confirmation-gated in the app.
    pub fn reboot_frame(&self, seq: u8) -> Vec<u8> {
        framing::command(self.family, seq, command::REBOOT_STRAP, &[])
    }

    /// One-shot 5/MG buzz.
    pub fn buzz_frame(&self, seq: u8) -> Vec<u8> {
        haptic::maverick_buzz_frame(seq)
    }

    /// SET_DEVICE_CONFIG to advertise standard 0x180D HR (Garmin/Edge). Opt-in gated in the app.
    pub fn broadcast_hr_frame(&self, seq: u8, on: bool) -> Vec<u8> {
        config::device_frame(seq, "whoop_live_hr_in_adv_ind_pkt", if on { b'1' } else { b'0' })
    }

    /// SET_CONFIG for one runtime-named feature flag. Opt-in / deep-data gated in the app.
    pub fn set_config_frame(&self, seq: u8, name: String, value: u8) -> Vec<u8> {
        config::feature_frame_named(seq, &name, value)
    }

    /// SET_ALARM_TIME (5/MG, 20-byte body). `wake_epoch_ms` is resolved app-side. Experimental, UI-gated.
    pub fn alarm_set_frame(&self, seq: u8, wake_epoch_ms: u64, alarm_id: u8) -> Vec<u8> {
        framing::command(self.family, seq, command::SET_ALARM_TIME, &alarm::build(wake_epoch_ms, alarm_id))
    }

    /// DISABLE_ALARM (5/MG form).
    pub fn alarm_disable_frame(&self, seq: u8) -> Vec<u8> {
        framing::command(self.family, seq, command::DISABLE_ALARM, &alarm::disable_rev2())
    }
}

/// The haptic pulses for a clock chime (write a buzz per pulse, spaced by its gap). Pure encoder.
#[uniffi::export]
pub fn haptic_clock_pulses(hour: u32, minute: u32, is_24h: bool) -> Vec<Pulse> {
    haptic::pulses(hour, minute, is_24h)
        .into_iter()
        .map(|p| Pulse { duration_ms: p.duration_ms, gap_ms: p.gap_ms })
        .collect()
}

/// HR from a v26 optical PPG buffer (24 Hz autocorrelation).
#[uniffi::export]
pub fn ppg_hr(samples: Vec<PpgSample>) -> Vec<PpgEstimate> {
    let s: Vec<ppg_hr::Sample> = samples.into_iter().map(|p| ppg_hr::Sample { ts: p.ts, value: p.value }).collect();
    ppg_hr::estimate(&s)
        .into_iter()
        .map(|e| PpgEstimate { ts: e.ts, bpm: e.bpm, conf: e.conf })
        .collect()
}

/// Gap-aware, artifact-corrected nightly RMSSD (ms) from per-record R-R runs.
#[uniffi::export]
pub fn hrv_rmssd_gap_aware(runs: Vec<RrRun>) -> Option<f64> {
    let beats: Vec<(u32, Vec<u16>)> = runs.into_iter().map(|r| (r.unix, r.rr)).collect();
    HrvReadiness::rmssd_gap_aware(&beats)
}

/// HRV-readiness over a nightly RMSSD series (oldest → newest; `None` slots = missing nights).
#[uniffi::export]
pub fn hrv_readiness(nightly_rmssd: Vec<Option<f64>>) -> Option<HrvReadinessInfo> {
    HrvReadiness::evaluate(&nightly_rmssd).map(|r| HrvReadinessInfo {
        tier: r.tier.into(),
        baseline7_ms: r.baseline7_ms,
        normal_low_ms: r.normal_low_ms,
        normal_high_ms: r.normal_high_ms,
        overreaching_watch: r.overreaching_watch,
    })
}

/// SpO2 (%) from a 4.0 paired red/IR window (ratio-of-ratios). `None` if not pulsatile.
#[uniffi::export]
pub fn spo2_from_paired(red: Vec<f64>, ir: Vec<f64>) -> Option<f64> {
    Spo2::from_paired(&red, &ir)
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

    #[test]
    fn decode_history_surfaces_the_wide_fields() {
        let mut p = vec![0u8; 90];
        p[4..8].copy_from_slice(&1_784_000_000u32.to_le_bytes()); // unix @ inner 7
        p[11] = 96; // hr @ inner 14
        p[46..48].copy_from_slice(&123u16.to_le_bytes()); // steps @ inner 49
        p[52] = 2; // activity_class @ inner 55
        p[62..64].copy_from_slice(&3345u16.to_le_bytes()); // skin temp @ inner 65 (33.45 °C)
        p[71] = 97; // spo2_pct @ inner 74
        let s = WhoopCodec::new(Gen::Gen5)
            .decode_history(framing::encode(Family::Gen5, 47, 18, 0, &p))
            .unwrap();
        assert_eq!(s.heart_rate, Some(96));
        assert_eq!(s.steps, Some(123));
        assert_eq!(s.activity_class, Some(2));
        assert_eq!(s.skin_temp_raw, Some(3345));
        assert_eq!(s.spo2_pct, Some(97));
    }

    #[test]
    fn command_builders_carry_the_right_opcodes() {
        let c = WhoopCodec::new(Gen::Gen5);
        let cases: [(Vec<u8>, u8); 4] = [
            (c.get_battery_frame(0), command::GET_BATTERY_LEVEL),
            (c.get_data_range_frame(0), command::GET_DATA_RANGE),
            (c.reboot_frame(0), command::REBOOT_STRAP),
            (c.set_config_frame(0, "enable_r22_packets".into(), b'2'), command::SET_CONFIG),
        ];
        for (frame, op) in cases {
            assert_eq!(framing::decode(Family::Gen5, &frame).unwrap().cmd(), op);
        }
    }

    #[test]
    fn metrics_free_fns_compute() {
        let runs = vec![super::RrRun { unix: 0, rr: vec![600, 610, 605] }];
        assert!(super::hrv_rmssd_gap_aware(runs).unwrap() < 20.0);
        assert!(super::hrv_readiness(vec![Some(50.0); 5]).is_some());
        assert!(super::ppg_hr(vec![]).is_empty());
    }
}
