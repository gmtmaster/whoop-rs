//! uniffi surface over the pure WHOOP codec. The mobile apps do BLE natively (Android BluetoothGatt,
//! iOS CoreBluetooth) — including the universal easy-connect (attach to an already-connected band) — and
//! feed notification bytes here; this returns decoded records and the exact frames to write back. No
//! async, no radio crosses the FFI: the sans-IO `Offload` is what makes one Rust core serve both apps.

uniffi::setup_scaffolding!();

use std::sync::{Arc, Mutex};

use physio_algo::{
    hr_zones, imu_features, ppg as ppg_hr, recovery, respiratory_rate, resting_hr, sleep, strain, stress, vo2max,
    HrvReadiness, Spo2,
};
use whoop_protocol::deframe::DeframerMap;
use whoop_protocol::hello::GEN5_CLIENT_HELLO;
use whoop_protocol::{
    advertising, alarm, clock, command, config, console, framing, haptic, live, records, response, Channel,
    Family, Offload, OffloadStep, PacketType, Record,
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
    /// A raw multi-wavelength optical buffer (v20): ~25 Hz × 6 channels, raw 20-bit signed ADC counts,
    /// flattened channel-major (25 samples of ch0, then ch1, …). Channels [0,1] green, [2,3] ambient,
    /// [4,5] red/IR (inferred). SpO2 % is not here — it rides the v18 summary.
    Optical { unix: u32, sample_rate_hz: u16, channels: u8, samples: Vec<i32> },
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
        OffloadStep::Record(Record::Optical(o)) => Step::Optical {
            unix: o.unix,
            sample_rate_hz: o.sample_rate_hz,
            channels: o.channels.len() as u8,
            samples: o.channels.into_iter().flatten().collect(),
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
    /// An EVENT frame. `battery_*` are Some only for BATTERY_LEVEL (raw deci-% + mV + charging — the
    /// consumer divides the deci-% in f64). `payload_hex` is the Gen5 opaque residual (lowercase hex).
    Event {
        number: u8,
        unix: u32,
        battery_soc_deci: Option<u16>,
        battery_millivolts: Option<u16>,
        battery_charging: Option<bool>,
        payload_hex: Option<String>,
    },
    Console { text: String },
}

/// A decoded command response (identity, battery, clock, data range, firmware). Every variant carries
/// `resp_cmd` (the command byte replied to) and `result` (the status code, `None` on 4.0) so the live
/// COMMAND_RESPONSE handshake can gate on them.
#[derive(uniffi::Enum)]
pub enum Response {
    Battery { resp_cmd: u8, result: Option<u8>, percent: f64 },
    Clock { resp_cmd: u8, result: Option<u8>, unix: u32 },
    Hello { resp_cmd: u8, result: Option<u8>, device_name: String, fw_version: Option<Vec<u8>> },
    DataRange { resp_cmd: u8, result: Option<u8>, oldest: u32, newest: u32 },
    Version { resp_cmd: u8, result: Option<u8>, fw: Vec<u32> },
    ExtendedBattery { resp_cmd: u8, result: Option<u8>, millivolts: u16, remaining_mah: u16, current_ma: i16 },
    BatteryPack { resp_cmd: u8, result: Option<u8>, serial: String, soc_pct: f64, millivolts: u16, pack_id: u32 },
    Other { resp_cmd: u8, result: Option<u8> },
}

/// METADATA offload-state fields: `meta_type` (1 start / 2 end / 3 complete) drives the drain, and for a
/// HISTORY_END `unix` + `trim_cursor` are the ack/advance cursor. `crc_ok` lets the app gate on a
/// checksum-valid frame (a forged HISTORY_END must not advance the trim over unstored data).
#[derive(uniffi::Record)]
pub struct MetadataInfo {
    pub meta_type: u8,
    pub unix: u32,
    pub trim_cursor: u32,
    pub crc_ok: bool,
}

/// A single-frame v26 24 Hz PPG waveform (24 i16 samples, single wavelength).
#[derive(uniffi::Record)]
pub struct PpgFrame {
    pub unix: u32,
    pub record_id: Option<u16>,
    pub samples: Vec<i16>,
}

/// A single-frame v21 100 Hz 6-axis IMU buffer: accel + gyro interleaved x,y,z per sample (raw i16 LSB —
/// scale accel by 1/4096 for g, gyro by 2000/32768 for deg/s).
#[derive(uniffi::Record)]
pub struct ImuFrame {
    pub unix: u32,
    pub sample_rate_hz: u16,
    pub accel: Vec<i16>,
    pub gyro: Vec<i16>,
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

impl From<physio_algo::ReadinessTier> for ReadinessTier {
    fn from(t: physio_algo::ReadinessTier) -> Self {
        match t {
            physio_algo::ReadinessTier::Primed => ReadinessTier::Primed,
            physio_algo::ReadinessTier::Normal => ReadinessTier::Normal,
            physio_algo::ReadinessTier::Suppressed => ReadinessTier::Suppressed,
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

/// A staged sleep stage. String forms are `"wake" | "light" | "deep" | "rem"` for cross-platform parity.
#[derive(uniffi::Enum, Clone, Copy)]
pub enum SleepStage {
    Wake,
    Light,
    Deep,
    Rem,
}

impl From<sleep::SleepStage> for SleepStage {
    fn from(s: sleep::SleepStage) -> Self {
        match s {
            sleep::SleepStage::Wake => SleepStage::Wake,
            sleep::SleepStage::Light => SleepStage::Light,
            sleep::SleepStage::Deep => SleepStage::Deep,
            sleep::SleepStage::Rem => SleepStage::Rem,
        }
    }
}

/// A contiguous run of one stage over `[start, end]` wall-clock unix seconds.
#[derive(uniffi::Record, Clone)]
pub struct SleepSegment {
    pub start: i64,
    pub end: i64,
    pub stage: SleepStage,
}

/// One HR sample for the stager (`bpm` at unix second `ts`).
#[derive(uniffi::Record, Clone)]
pub struct SleepHrSample {
    pub ts: i64,
    pub bpm: u16,
}

/// A run of consecutive R-R intervals (ms) sharing one unix-second anchor `ts`.
#[derive(uniffi::Record, Clone)]
pub struct SleepRrRun {
    pub ts: i64,
    pub intervals: Vec<u16>,
}

/// One 3-axis accel / gravity sample (g) at unix second `ts`.
#[derive(uniffi::Record, Clone)]
pub struct SleepAccelSample {
    pub ts: i64,
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

/// One raw respiration-ADC sample at unix second `ts` (accepted for parity; V2 recovers RSA from R-R).
#[derive(uniffi::Record, Clone)]
pub struct SleepRespSample {
    pub ts: i64,
    pub raw: i32,
}

/// The protocol-free input bundle for one detected in-bed span (`[start, end]` unix seconds).
#[derive(uniffi::Record, Clone)]
pub struct SleepInput {
    pub start: i64,
    pub end: i64,
    pub hr: Vec<SleepHrSample>,
    pub rr: Vec<SleepRrRun>,
    pub accel: Vec<SleepAccelSample>,
    pub resp: Vec<SleepRespSample>,
}

impl From<SleepInput> for sleep::SleepInput {
    fn from(i: SleepInput) -> Self {
        sleep::SleepInput {
            start: i.start,
            end: i.end,
            hr: i.hr.into_iter().map(|h| sleep::HrSample { ts: h.ts, bpm: h.bpm }).collect(),
            rr: i.rr.into_iter().map(|r| sleep::RrRun { ts: r.ts, intervals: r.intervals }).collect(),
            accel: i.accel.into_iter().map(|a| sleep::AccelSample { ts: a.ts, x: a.x, y: a.y, z: a.z }).collect(),
            resp: i.resp.into_iter().map(|r| sleep::RespSample { ts: r.ts, raw: r.raw }).collect(),
        }
    }
}

fn to_sleep_segments(segs: Vec<sleep::StageSegment>) -> Vec<SleepSegment> {
    segs.into_iter()
        .map(|s| SleepSegment { start: s.start, end: s.end, stage: s.stage.into() })
        .collect()
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

    /// Decode a single complete history frame (for offline replay of a captured file). A bad-CRC frame is
    /// rejected here (None) — a forged/corrupt record must not be stored past the trim.
    pub fn decode_history(&self, raw: Vec<u8>) -> Option<HistorySummary> {
        let frame = framing::decode(self.family, &raw).ok()?;
        if !frame.crc_ok {
            return None;
        }
        match records::decode(&frame)? {
            Record::History(h) => Some(h.into()),
            _ => None,
        }
    }

    /// Decode a single METADATA frame's offload-state fields (meta_type + unix + trim_cursor + crc_ok), so
    /// the app's offload state machine can recognise HISTORY_END/COMPLETE. `None` off a non-metadata frame.
    pub fn decode_metadata(&self, raw: Vec<u8>) -> Option<MetadataInfo> {
        let f = framing::decode(self.family, &raw).ok()?;
        if !matches!(f.packet().canonical(), PacketType::Metadata) {
            return None;
        }
        let m = live::metadata(&f)?;
        Some(MetadataInfo { meta_type: m.meta_type, unix: m.unix, trim_cursor: m.trim_cursor, crc_ok: f.crc_ok })
    }

    /// Decode a single v26 PPG waveform frame (the 24 optical samples) for offline replay / the deep buffers.
    /// Bad-CRC frames are rejected; `decode_ppg` version-gates, so a v18/v24 frame yields None (not 24 samples).
    pub fn decode_ppg_frame(&self, raw: Vec<u8>) -> Option<PpgFrame> {
        let f = framing::decode(self.family, &raw).ok()?;
        if !f.crc_ok {
            return None;
        }
        let p = records::decode_ppg(&f)?;
        Some(PpgFrame { unix: p.unix, record_id: p.record_id, samples: p.samples })
    }

    /// Decode a single v21 6-axis IMU buffer frame (accel/gyro columns) for the deep-buffer diagnostics.
    pub fn decode_imu_frame(&self, raw: Vec<u8>) -> Option<ImuFrame> {
        let f = framing::decode(self.family, &raw).ok()?;
        let m = records::decode_imu(&f)?;
        Some(ImuFrame {
            unix: m.unix,
            sample_rate_hz: m.sample_rate_hz,
            accel: m.accel.into_iter().flatten().collect(),
            gyro: m.gyro.into_iter().flatten().collect(),
        })
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
            PacketType::Event => live::event(&f).map(|e| {
                let batt = live::battery_event(&f);
                Live::Event {
                    number: e.number,
                    unix: e.timestamp,
                    battery_soc_deci: batt.map(|b| b.soc_deci),
                    battery_millivolts: batt.map(|b| b.millivolts),
                    battery_charging: batt.map(|b| b.charging),
                    payload_hex: live::event_payload_hex(&f),
                }
            }),
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
        let (resp_cmd, result) = response::resp_status(&f);
        let result = result.map(result_to_u8);
        Some(match response::decode(&f)? {
            C::Battery { percent } => Response::Battery { resp_cmd, result, percent },
            C::Clock { unix } => Response::Clock { resp_cmd, result, unix },
            C::Hello { device_name, fw_version } => {
                Response::Hello { resp_cmd, result, device_name, fw_version: fw_version.map(|v| v.to_vec()) }
            }
            C::DataRange { oldest, newest } => Response::DataRange { resp_cmd, result, oldest, newest },
            C::VersionInfo { fw } => Response::Version { resp_cmd, result, fw: fw.to_vec() },
            C::ExtendedBattery { millivolts, remaining_mah, current_ma } => {
                Response::ExtendedBattery { resp_cmd, result, millivolts, remaining_mah, current_ma }
            }
            C::BatteryPack { serial, soc_pct, millivolts, pack_id } => {
                Response::BatteryPack { resp_cmd, result, serial, soc_pct, millivolts, pack_id }
            }
            C::Other { .. } => Response::Other { resp_cmd, result },
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

    /// Generic outbound COMMAND builder for opcodes without a dedicated method (get-clock/version,
    /// alarm-readback, run-alarm, stop-haptics, historical-data-result, plus the gated set-clock/adv-name
    /// the app allow-lists). Refuses the genuinely-destructive set (trim/DFU) so it can't be built here.
    pub fn command_frame(&self, seq: u8, cmd: u8, payload: Vec<u8>) -> Option<Vec<u8>> {
        if command::is_destructive(cmd) {
            return None;
        }
        Some(framing::command(self.family, seq, cmd, &payload))
    }

    /// SET_CLOCK — the 8-byte form newer firmware latches. Gated in the app.
    pub fn set_clock_frame(&self, seq: u8, now_unix: u32) -> Vec<u8> {
        framing::command(self.family, seq, command::SET_CLOCK, &clock::set_clock_payload(now_unix))
    }

    /// SET_CLOCK — the legacy 9-byte form older 4.0 firmware needs (a no-op on newer). Sent alongside the
    /// 8-byte form so either firmware latches.
    pub fn set_clock_legacy_frame(&self, seq: u8, now_unix: u32) -> Vec<u8> {
        framing::command(self.family, seq, command::SET_CLOCK, &clock::set_clock_payload_legacy(now_unix))
    }

    /// SET_ALARM_TIME — the WHOOP 4.0 9-byte body (minute-precision). `wake_epoch_secs` is resolved
    /// app-side. Experimental, UI-gated.
    pub fn alarm_set_frame_gen4(&self, seq: u8, wake_epoch_secs: u32) -> Vec<u8> {
        framing::command(self.family, seq, command::SET_ALARM_TIME, &alarm::whoop4_build(wake_epoch_secs))
    }

    /// RUN_HAPTICS_PATTERN — the 4.0 preset buzz `[pattern_id][loops][0][0][0]` (pattern 2 = graduated
    /// alarm buzz). On 5/MG the app remaps to the maverick buzz instead.
    pub fn run_haptics_frame(&self, seq: u8, pattern_id: u8, loops: u8) -> Vec<u8> {
        framing::command(self.family, seq, command::RUN_HAPTICS_PATTERN, &haptic::run_haptics_pattern(pattern_id, loops))
    }

    /// SET_ADVERTISING_NAME — rename the 4.0 strap's BLE advertising name (clamped to 24 UTF-8 bytes). The
    /// strap reboots to apply. Gated in the app.
    pub fn advertising_name_frame(&self, seq: u8, name: String) -> Vec<u8> {
        framing::command(self.family, seq, command::SET_ADVERTISING_NAME, &advertising::advertising_name_payload(&name))
    }

    /// SET_ADVERTISING_NAME (5/MG) — rename the strap's puffin BLE advertising name (clamped to 24 UTF-8
    /// bytes). The strap reboots to apply. Hardware-unverified (opcode/b3 inferred). Gated in the app.
    pub fn advertising_name_frame_gen5(&self, seq: u8, name: String) -> Vec<u8> {
        framing::command(self.family, seq, command::SET_ADVERTISING_NAME, &advertising::advertising_name_payload_gen5(&name))
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

/// Newest plausible unix banked, scanning EVERY byte offset of a GET_DATA_RANGE frame and preferring the
/// newest non-future word (falls back to newest-any). This is the sync gate — it REPLACES the fixed-offset
/// `Response::DataRange` newest read.
#[uniffi::export]
pub fn data_range_newest(frame: Vec<u8>, wall_now_unix: u64, future_skew_seconds: u64) -> Option<u32> {
    response::data_range_scan_newest(&frame, wall_now_unix, future_skew_seconds)
}

/// Oldest plausible unix banked (backlog depth), scanning only the aligned-from-7 grid (asymmetric with
/// the newest scan by design, to dodge a WHOOP-4 straddle word).
#[uniffi::export]
pub fn data_range_oldest(frame: Vec<u8>) -> Option<u32> {
    response::data_range_scan_oldest(&frame)
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

/// Windowed session avgHrv (ms): the mean of per-5-min-bucket gap-aware RMSSD over `[start, end]`, the
/// app's stored `SleepSession.avgHrv`. `runs` are the session's per-record `(unix, rr)` in chronological
/// order; buckets tumble from `start`. `None` when no bucket yields a value.
#[uniffi::export]
pub fn hrv_windowed_avg(start: u32, end: u32, runs: Vec<RrRun>) -> Option<f64> {
    let beats: Vec<(u32, u16)> = runs
        .into_iter()
        .flat_map(|r| {
            let unix = r.unix;
            r.rr.into_iter().map(move |v| (unix, v))
        })
        .collect();
    HrvReadiness::windowed_avg_hrv(start, end, &beats)
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

/// Sleep hypnogram via V2 (cardiorespiratory) — the 5.0/MG default. Per-30 s-epoch stage segments over
/// the detected in-bed span. Pure and deterministic.
#[uniffi::export]
pub fn stage_sleep_v2(input: SleepInput) -> Vec<SleepSegment> {
    to_sleep_segments(sleep::stage_v2(&input.into()))
}

/// Sleep hypnogram via V1 (Cole-Kripke) — the 4.0 recipe and the session-detection source of truth.
#[uniffi::export]
pub fn stage_sleep_v1(input: SleepInput) -> Vec<SleepSegment> {
    to_sleep_segments(sleep::stage_v1(&input.into()))
}

/// One heart-rate tick (unix seconds + bpm) shared by strain, recovery, resting-HR and zones.
#[derive(uniffi::Record)]
pub struct HrTick {
    pub ts: i64,
    pub bpm: i32,
}

/// A personal baseline driver (mean + spread) for recovery z-scoring.
#[derive(uniffi::Record)]
pub struct DriverBaselineInfo {
    pub mean: f64,
    pub spread: f64,
}

impl From<DriverBaselineInfo> for recovery::DriverBaseline {
    fn from(b: DriverBaselineInfo) -> Self {
        recovery::DriverBaseline { mean: b.mean, spread: b.spread }
    }
}

/// Nightly recovery drivers. Optional terms drop and the weights renormalise.
#[derive(uniffi::Record)]
pub struct RecoveryDrivers {
    pub hrv: f64,
    pub rhr: f64,
    pub resp: Option<f64>,
    pub hrv_baseline: Option<DriverBaselineInfo>,
    pub rhr_baseline: Option<DriverBaselineInfo>,
    pub resp_baseline: Option<DriverBaselineInfo>,
    pub sleep_perf: Option<f64>,
    pub skin_temp_dev: Option<f64>,
    pub hrv_baseline_usable: bool,
    pub recovery_index_slope: Option<f64>,
    pub effort_baseline: Option<DriverBaselineInfo>,
    pub prior_day_effort: Option<f64>,
}

/// Recovery "Charge" score in [0, 100]. `None` at cold-start or when no driver is available.
#[uniffi::export]
pub fn recovery_score(d: RecoveryDrivers) -> Option<f64> {
    recovery::recovery(&recovery::RecoveryInput {
        hrv: d.hrv,
        rhr: d.rhr,
        resp: d.resp,
        hrv_baseline: d.hrv_baseline.map(Into::into),
        rhr_baseline: d.rhr_baseline.map(Into::into),
        resp_baseline: d.resp_baseline.map(Into::into),
        sleep_perf: d.sleep_perf,
        skin_temp_dev: d.skin_temp_dev,
        hrv_baseline_usable: d.hrv_baseline_usable,
        recovery_index_slope: d.recovery_index_slope,
        effort_baseline: d.effort_baseline.map(Into::into),
        prior_day_effort: d.prior_day_effort,
    })
}

/// Recovery colour band ("red" | "yellow" | "green") for a score.
#[uniffi::export]
pub fn recovery_band(score: f64) -> String {
    recovery::band(score).to_string()
}

/// Overnight HR-decline slope (bpm/hour) — the recovery-index driver.
#[uniffi::export]
pub fn recovery_index_slope(hr: Vec<HrTick>, start: i64, end: i64) -> Option<f64> {
    let s: Vec<recovery::HrSample> = hr.into_iter().map(|h| recovery::HrSample { ts: h.ts, bpm: h.bpm }).collect();
    recovery::recovery_index_slope(&s, start, end)
}

/// Count of nights carrying a usable nightly HRV — the calibration-progress count.
#[uniffi::export]
pub fn recovery_banked_nights(nightly_hrv: Vec<Option<f64>>) -> u32 {
    recovery::banked_nights(&nightly_hrv, recovery::HRV_MIN_MS, recovery::HRV_MAX_MS) as u32
}

/// TRIMP accumulation method for strain.
#[derive(uniffi::Enum)]
pub enum StrainMethod {
    Edwards,
    Banister,
}

impl From<StrainMethod> for strain::Method {
    fn from(m: StrainMethod) -> Self {
        match m {
            StrainMethod::Edwards => strain::Method::Edwards,
            StrainMethod::Banister => strain::Method::Banister,
        }
    }
}

/// Cardiovascular Effort (0–100) from an HR series. `None` without enough data or when HRR ≤ 0.
#[uniffi::export]
pub fn strain_score(
    hr: Vec<HrTick>,
    max_hr: Option<f64>,
    resting_hr: f64,
    method: StrainMethod,
    sex: String,
    denominator: f64,
) -> Option<f64> {
    let s: Vec<strain::HrSample> = hr.into_iter().map(|h| strain::HrSample { ts: h.ts, bpm: h.bpm }).collect();
    strain::strain(&s, max_hr, resting_hr, method.into(), &sex, denominator)
}

/// The default strain denominator (log-map scale onto 0–100).
#[uniffi::export]
pub fn strain_default_denominator() -> f64 {
    strain::STRAIN_DENOMINATOR
}

/// Baevsky Stress Index histogram terms behind an SI.
#[derive(uniffi::Record)]
pub struct StressComponentsInfo {
    pub mo_sec: f64,
    pub amo_percent: f64,
    pub mxdmn_sec: f64,
    pub si: f64,
}

/// Baevsky Stress Index from a raw R-R series (ms). `None` on too-few beats or a degenerate range.
#[uniffi::export]
pub fn stress_index(rr_ms: Vec<f64>) -> Option<f64> {
    stress::stress_index_raw(&rr_ms)
}

/// Full SI components from a raw R-R series (ms).
#[uniffi::export]
pub fn stress_components(rr_ms: Vec<f64>) -> Option<StressComponentsInfo> {
    stress::components_raw(&rr_ms).map(|c| StressComponentsInfo {
        mo_sec: c.mo_sec,
        amo_percent: c.amo_percent,
        mxdmn_sec: c.mxdmn_sec,
        si: c.si,
    })
}

/// Lowest 5-min tumbling-window mean bpm floor over `[start, end]`. `None` with no samples.
#[uniffi::export]
pub fn session_resting_hr(start: i64, end: i64, hr: Vec<HrTick>) -> Option<i32> {
    let s: Vec<resting_hr::HrSample> = hr.into_iter().map(|h| resting_hr::HrSample { ts: h.ts, bpm: h.bpm }).collect();
    resting_hr::session_resting_hr(start, end, &s)
}

/// Daily resting HR = min of the per-session floors.
#[uniffi::export]
pub fn daily_resting_hr(session_floors: Vec<Option<i32>>) -> Option<i32> {
    resting_hr::daily_resting_hr(&session_floors)
}

/// A single HR zone as a bpm interval `[lower, upper)` plus its %HRmax band.
#[derive(uniffi::Record)]
pub struct HrZoneInfo {
    pub number: u8,
    pub lower: f64,
    pub upper: f64,
    pub lower_pct: f64,
    pub upper_pct: f64,
}

/// Five HR zones, the max HR they were built from, and its source ("tanaka" | "manual").
#[derive(uniffi::Record)]
pub struct HrZoneSetInfo {
    pub zones: Vec<HrZoneInfo>,
    pub max_hr: f64,
    pub source: String,
}

/// Seconds in each of the five zones (index 0 == Zone 1) plus time below Zone 1.
#[derive(uniffi::Record)]
pub struct TimeInZoneInfo {
    pub seconds: Vec<f64>,
    pub below_zone1: f64,
}

fn zone_set_to_ffi(z: hr_zones::HrZoneSet) -> HrZoneSetInfo {
    HrZoneSetInfo {
        zones: z
            .zones
            .iter()
            .map(|zn| HrZoneInfo {
                number: zn.number,
                lower: zn.lower,
                upper: zn.upper,
                lower_pct: zn.lower_pct,
                upper_pct: zn.upper_pct,
            })
            .collect(),
        max_hr: z.max_hr,
        source: z.source,
    }
}

/// Age-derived (Tanaka) HR zones, or a manual max-HR override.
#[uniffi::export]
pub fn hr_zones_for_age(age: f64, max_hr_override: Option<f64>) -> HrZoneSetInfo {
    zone_set_to_ffi(hr_zones::zones_for_age(age, max_hr_override))
}

/// Seconds spent in each HR zone over an HR series, using age-derived (or override) zones.
#[uniffi::export]
pub fn hr_time_in_zone(hr: Vec<HrTick>, age: f64, max_hr_override: Option<f64>) -> TimeInZoneInfo {
    let zs = hr_zones::zones_for_age(age, max_hr_override);
    let s: Vec<hr_zones::HrSample> = hr.into_iter().map(|h| hr_zones::HrSample { ts: h.ts, bpm: h.bpm }).collect();
    let t = hr_zones::time_in_zone(&s, &zs);
    TimeInZoneInfo { seconds: t.seconds.to_vec(), below_zone1: t.below_zone1 }
}

/// A computed Fitness Age with the inputs to present it. `vo2max` is filled only with a waist.
#[derive(uniffi::Record)]
pub struct FitnessAgeInfo {
    pub vo2max: Option<f64>,
    pub fitness_age: f64,
    pub chrono_age: f64,
    pub delta_years: f64,
    pub band_years: f64,
    pub lower_confidence: bool,
}

/// Non-exercise VO2max estimate (ml/kg/min) from the waist-circumference model. Wellness only.
#[uniffi::export]
pub fn vo2max_estimate(age: f64, sex: String, waist_cm: f64, resting_hr: f64, pa_index: f64) -> f64 {
    vo2max::estimate_vo2max(age, &sex, waist_cm, resting_hr, pa_index)
}

/// Full Fitness Age. `None` only if RHR or age is missing.
#[uniffi::export]
pub fn fitness_age_compute(
    age: f64,
    sex: String,
    resting_hr: f64,
    pa_index: f64,
    waist_cm: Option<f64>,
    lower_confidence: bool,
) -> Option<FitnessAgeInfo> {
    vo2max::compute(age, &sex, resting_hr, pa_index, waist_cm, lower_confidence).map(|r| FitnessAgeInfo {
        vo2max: r.vo2max,
        fitness_age: r.fitness_age,
        chrono_age: r.chrono_age,
        delta_years: r.delta_years,
        band_years: r.band_years,
        lower_confidence: r.lower_confidence,
    })
}

/// One R-R beat for respiratory-rate estimation (unix seconds + interval ms).
#[derive(uniffi::Record)]
pub struct RrBeat {
    pub ts: i64,
    pub rr_ms: u16,
}

/// Respiratory rate (breaths/min) from R-R via RSA. `None` when the signal is too thin.
#[uniffi::export]
pub fn resp_rate_from_rr(beats: Vec<RrBeat>, start: i64, end: i64) -> Option<f64> {
    let b: Vec<(i64, u16)> = beats.into_iter().map(|x| (x.ts, x.rr_ms)).collect();
    respiratory_rate::resp_rate_from_rr(&b, start, end)
}

/// One raw IMU sample: 3-axis accelerometer (g) + 3-axis gyroscope (deg/s).
#[derive(uniffi::Record)]
pub struct ImuSampleInfo {
    pub ax: f64,
    pub ay: f64,
    pub az: f64,
    pub gx: f64,
    pub gy: f64,
    pub gz: f64,
}

/// Compact activity features over a window of raw IMU samples.
#[derive(uniffi::Record)]
pub struct ImuFeaturesInfo {
    pub accel_energy_g: f64,
    pub gyro_energy_dps: f64,
    pub jerk_rms: f64,
    pub cadence_hz: Option<f64>,
    pub cadence_strength: f64,
    pub sample_count: u32,
}

/// Accel/gyro energy, jerk and gait-band cadence over IMU samples at `sample_rate_hz`.
#[uniffi::export]
pub fn imu_features(samples: Vec<ImuSampleInfo>, sample_rate_hz: i32) -> ImuFeaturesInfo {
    let s: Vec<imu_features::ImuSample> = samples
        .into_iter()
        .map(|x| imu_features::ImuSample { ax: x.ax, ay: x.ay, az: x.az, gx: x.gx, gy: x.gy, gz: x.gz })
        .collect();
    let f = imu_features::extract(&s, sample_rate_hz);
    ImuFeaturesInfo {
        accel_energy_g: f.accel_energy_g,
        gyro_energy_dps: f.gyro_energy_dps,
        jerk_rms: f.jerk_rms,
        cadence_hz: f.cadence_hz,
        cadence_strength: f.cadence_strength,
        sample_count: f.sample_count as u32,
    }
}

#[cfg(test)]
mod tests {
    use super::{Chan, Gen, Response, Step, WhoopCodec};
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
    fn decode_live_folds_battery_into_event() {
        use super::Live;
        // Battery EVENT: payload = inner[3..]; inner 4/13/17/22 = payload 1/10/14/19.
        let mut p = vec![0u8; 24];
        p[1..5].copy_from_slice(&1_784_000_000u32.to_le_bytes());
        p[10..12].copy_from_slice(&812u16.to_le_bytes()); // 81.2%
        p[14..16].copy_from_slice(&4100u16.to_le_bytes());
        p[19] = 1;
        let wire = framing::encode(Family::Gen5, 48, 0, 3, &p); // cmd 3 = BATTERY_LEVEL
        match WhoopCodec::new(Gen::Gen5).decode_live(wire).unwrap() {
            Live::Event { number, unix, battery_soc_deci, battery_millivolts, battery_charging, payload_hex } => {
                assert_eq!(number, 3);
                assert_eq!(unix, 1_784_000_000);
                assert_eq!(battery_soc_deci, Some(812));
                assert_eq!(battery_millivolts, Some(4100));
                assert_eq!(battery_charging, Some(true));
                assert!(payload_hex.is_some());
            }
            _ => panic!("expected Live::Event"),
        }
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
        let win = vec![super::RrRun { unix: 100, rr: vec![800, 810, 820, 815, 805] }];
        assert!((super::hrv_windowed_avg(100, 400, win).unwrap() - 9.013878188659973).abs() < 1e-12);
    }

    #[test]
    fn decode_metadata_reads_a_real_history_end() {
        let raw = whoop_protocol::bytes::from_hex(
            "aa011c00010023d1319102b949596a705d3b000000fdba010010000000000000f269faec",
        )
        .unwrap();
        let m = WhoopCodec::new(Gen::Gen5).decode_metadata(raw).unwrap();
        assert_eq!(m.meta_type, 2);
        assert_eq!(m.unix, 1_784_236_473);
        assert_eq!(m.trim_cursor, 113_405);
        assert!(m.crc_ok);
    }

    #[test]
    fn decode_metadata_rejects_a_non_metadata_frame() {
        assert!(WhoopCodec::new(Gen::Gen5).decode_metadata(v18_frame()).is_none());
    }

    #[test]
    fn decode_ppg_frame_reads_a_real_v26() {
        let raw = whoop_protocol::bytes::from_hex(
            "aa015000010035412f1a804b047b019452596aae0701004b8503006bfdcffd36fe50fe12ff73ff6dff42ffa7ffc9fff9ffe5ff5c005a007a00f20089003000dbfd2efd0bfe3ffeaefe3affc06c213c50070001001ddc65fe",
        )
        .unwrap();
        let p = WhoopCodec::new(Gen::Gen5).decode_ppg_frame(raw).unwrap();
        assert_eq!(p.record_id, Some(1099));
        assert_eq!(p.unix, 1_784_238_740);
        assert_eq!(p.samples.len(), 24);
        assert_eq!(p.samples[0], -661);
    }

    #[test]
    fn decode_history_rejects_a_bad_crc_frame() {
        // A real worn v18 frame from the protocol fixture; flipping one CRC byte must drop it (None), so a
        // corrupt/forged record can never be stored past the trim.
        let mut raw = whoop_protocol::bytes::from_hex(
            "aa01740001003fb12f1280733d8401b69f266a66460066025a0265020000000000007b0a8d656463ff0012163cf6a439bf2924fd3ed763fe3e3200aa000000000000000000f7000901f10b0007010c020c00000000000000000000000000000000000000000000000100656f1e1e0000009d61a7c00000003e862817",
        )
        .unwrap();
        let codec = WhoopCodec::new(Gen::Gen5);
        assert!(codec.decode_history(raw.clone()).is_some()); // good CRC decodes
        *raw.last_mut().unwrap() ^= 0xFF; // trash the CRC32 tail
        assert!(codec.decode_history(raw).is_none());
    }

    #[test]
    fn decode_ppg_frame_rejects_a_non_v26_frame() {
        // A v18 history frame carries no PPG waveform: the version gate returns None, not 24 samples.
        assert!(WhoopCodec::new(Gen::Gen5).decode_ppg_frame(v18_frame()).is_none());
    }

    #[test]
    fn decode_imu_frame_reads_a_synthetic_v21() {
        let f = WhoopCodec::new(Gen::Gen5).decode_imu_frame(v21_imu_frame()).unwrap();
        assert_eq!(f.sample_rate_hz, 100);
        assert_eq!(f.accel.len(), 300);
        assert_eq!(f.gyro.len(), 300);
        assert_eq!(f.accel[0], 4096);
    }

    #[test]
    fn data_range_scan_free_fns_pin_a_real_frame() {
        let frame = whoop_protocol::bytes::from_hex(
            "aa014c00010032d124f22204010140bb0100f9ba010001bb0100f9ba010010000000000002006a00000088ff1d001432b869cc4c00004549596ab83e00004549596ab83e0000ae49596aeb1100000000d0da9256",
        )
        .unwrap();
        assert_eq!(super::data_range_newest(frame.clone(), 1_784_236_480, 3600), Some(1_784_236_462));
        assert_eq!(super::data_range_oldest(frame), Some(1_778_385_408));
    }

    #[test]
    fn generic_command_frame_builds_and_refuses_destructive() {
        let c = WhoopCodec::new(Gen::Gen5);
        let f = c.command_frame(3, command::GET_ALARM_TIME, vec![0x00]).unwrap();
        assert_eq!(framing::decode(Family::Gen5, &f).unwrap().cmd(), command::GET_ALARM_TIME);
        // The genuinely-destructive set can't be built even through the generic door.
        assert!(c.command_frame(0, command::FORCE_TRIM, vec![]).is_none());
        assert!(c.command_frame(0, command::ENTER_BLE_DFU, vec![]).is_none());
    }

    #[test]
    fn gen4_encoder_frames_carry_the_right_opcodes_and_bodies() {
        let c = WhoopCodec::new(Gen::Gen4);
        let checks: [(Vec<u8>, u8); 5] = [
            (c.set_clock_frame(0, 1_784_000_000), command::SET_CLOCK),
            (c.set_clock_legacy_frame(0, 1_784_000_000), command::SET_CLOCK),
            (c.alarm_set_frame_gen4(0, 1_784_000_000), command::SET_ALARM_TIME),
            (c.run_haptics_frame(0, 2, 3), command::RUN_HAPTICS_PATTERN),
            (c.advertising_name_frame(0, "noop".into()), command::SET_ADVERTISING_NAME),
        ];
        for (frame, op) in checks {
            assert_eq!(framing::decode(Family::Gen4, &frame).unwrap().cmd(), op);
        }
        // The 8-byte vs legacy 9-byte set-clock bodies differ in length only.
        assert_eq!(framing::decode(Family::Gen4, &c.set_clock_frame(0, 1)).unwrap().payload().len(), 8);
        assert_eq!(framing::decode(Family::Gen4, &c.set_clock_legacy_frame(0, 1)).unwrap().payload().len(), 9);
    }

    #[test]
    fn decode_response_surfaces_cmd_and_result_on_every_variant() {
        let codec = WhoopCodec::new(Gen::Gen5);
        // A GET_DATA_RANGE reply: the success gate needs both the command byte and the status.
        let mut p = vec![0u8; 12];
        p[1] = 1; // result = SUCCESS
        p[3..7].copy_from_slice(&1_784_000_000u32.to_le_bytes());
        p[7..11].copy_from_slice(&1_784_200_000u32.to_le_bytes());
        let wire = framing::encode(Family::Gen5, 36, 0, command::GET_DATA_RANGE, &p);
        match codec.decode_response(wire).unwrap() {
            Response::DataRange { resp_cmd, result, oldest, newest } => {
                assert_eq!(resp_cmd, command::GET_DATA_RANGE);
                assert_eq!(result, Some(1));
                assert_eq!((oldest, newest), (1_784_000_000, 1_784_200_000));
            }
            _ => panic!("expected DataRange"),
        }
        // An unmodelled command still surfaces its cmd + status through Other.
        let mut p = vec![0u8; 8];
        p[1] = 2; // result = PENDING
        let wire = framing::encode(Family::Gen5, 36, 0, command::REBOOT_STRAP, &p);
        match codec.decode_response(wire).unwrap() {
            Response::Other { resp_cmd, result } => {
                assert_eq!(resp_cmd, command::REBOOT_STRAP);
                assert_eq!(result, Some(2));
            }
            _ => panic!("expected Other"),
        }
    }

    #[test]
    fn stage_sleep_v2_tiles_a_synthetic_night() {
        use super::{stage_sleep_v2, SleepAccelSample, SleepHrSample, SleepInput, SleepRrRun, SleepStage};
        let start = 1_749_517_200i64;
        let dur = 90 * 60 * 4;
        let (mut hr, mut rr, mut accel) = (Vec::new(), Vec::new(), Vec::new());
        for i in 0..dur {
            let ts = start + i;
            let ph = (i / (90 * 60)) as usize;
            accel.push(SleepAccelSample { ts, x: 0.0, y: 0.0, z: 1.0 });
            let bpm: i64 = [50, 55, 58, 68][ph];
            hr.push(SleepHrSample { ts, bpm: bpm as u16 });
            rr.push(SleepRrRun { ts, intervals: vec![(60_000 / bpm) as u16] });
        }
        let input = SleepInput { start, end: start + dur, hr, rr, accel, resp: Vec::new() };
        let segs = stage_sleep_v2(input);
        assert!(!segs.is_empty());
        assert_eq!(segs.first().unwrap().start, start);
        assert_eq!(segs.last().unwrap().end, start + dur);
        // Contiguous tiling, each a valid stage.
        for w in segs.windows(2) {
            assert_eq!(w[0].end, w[1].start);
        }
        assert!(matches!(segs[0].stage, SleepStage::Wake | SleepStage::Light | SleepStage::Deep | SleepStage::Rem));
    }

    #[test]
    fn gen5_advertising_name_frame_has_the_puffin_body() {
        let c = WhoopCodec::new(Gen::Gen5);
        let frame = c.advertising_name_frame_gen5(0, "band".into());
        let f = framing::decode(Family::Gen5, &frame).unwrap();
        assert_eq!(f.cmd(), command::SET_ADVERTISING_NAME);
        // b3 selector 0x01, reserved 0x00, name, NUL terminator (GEN5 pads the inner to a 4-byte boundary).
        assert_eq!(&f.payload()[..7], b"\x01\x00band\x00");
    }
}
