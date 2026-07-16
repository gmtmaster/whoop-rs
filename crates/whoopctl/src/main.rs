//! WHOOP BLE CLI over the btleplug transport. Read-safe by default; the write actions (r22on/buzz/
//! reboot/send) are the gated ones from whoop-client. All the on-band behaviour is validated here, not
//! in unit tests — a connect proves nothing about a real strap.

use std::io::Write as _;
use std::path::{Path, PathBuf};

use anyhow::Result;
use clap::{Parser, Subcommand};

use std::collections::BTreeMap;

use ble_btleplug::{scan_whoops, BtleplugTransport};
use whoop_client::{capture_line, WhoopClient};
use whoop_metrics::{HrvReadiness, Spo2};
use whoop_protocol::bytes::to_hex;
use whoop_protocol::{records, Family, Frame, HistoryRecord, PacketType, Record};

#[derive(Parser)]
#[command(name = "whoopctl", about = "WHOOP BLE tool")]
struct Cli {
    /// Generation: 5 (5.0/MG, default) or 4 (4.0).
    #[arg(long, default_value_t = 5)]
    gen: u8,
    /// Target band by advertised name (else the first WHOOP found).
    #[arg(long)]
    name: Option<String>,
    /// Target band by BLE address (surest when several WHOOPs are in range).
    #[arg(long)]
    address: Option<String>,
    /// Target band by serial, full or suffix (connects to each candidate and reads its serial).
    #[arg(long)]
    sn: Option<String>,
    /// Who is wearing the strap. Calibration is per (person, strap): a new person, or the same person on
    /// a new strap, starts a fresh calibration period — you never calibrate against someone else's data.
    #[arg(long, default_value = "default")]
    person: String,
    /// Calibration store (SQLite) path.
    #[arg(long, default_value = "whoop-cal.db")]
    db: PathBuf,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// List WHOOP bands in range.
    Scan {
        #[arg(long, default_value_t = 6)]
        secs: u64,
    },
    /// Connect (no bond) and read the standard identity chars (name/serial/fw). Read-only.
    Identify,
    /// Connect, bond, read identity/battery/data-range.
    Info,
    /// Connect, bond, read banked history, decode it to JSON Lines. Keeps the strap intact by default.
    Sync {
        /// Write decoded records here as JSON Lines.
        #[arg(long, default_value = "whoop-sync.jsonl")]
        out: PathBuf,
        /// Also write EVERY reassembled frame (lossless, including undecodable ones) here — the capture
        /// tap for offline field-layout analysis of unmapped records.
        #[arg(long)]
        raw: Option<PathBuf>,
        /// Drain the FULL history and advance the strap's trim cursor (the records leave the strap; the
        /// WHOOP app can no longer offload them). Default reads the first chunk and keeps everything.
        #[arg(long)]
        wipe: bool,
    },
    /// Connect, bond, stream frames (optionally one packet type).
    Monitor {
        #[arg(long)]
        r#type: Option<u8>,
        #[arg(long, default_value_t = 20)]
        secs: u64,
    },
    /// Connect, bond, stream live heart rate from the standard HR characteristic (2a37).
    Hr {
        #[arg(long, default_value_t = 15)]
        secs: u64,
        /// First enable the strap's HR broadcast (reversible) so 2a37 streams.
        #[arg(long)]
        broadcast: bool,
    },
    /// Send one command opcode (+ hex payload). Destructive/config-write opcodes are refused.
    Send { op: String, payload: Option<String> },
    /// Connect, bond, read history (keep-only, never trims), and compute derived metrics: SpO2 (4.0
    /// paired red/IR, or the 5.0/MG v26 channel ranking) and HRV-readiness from the R-R.
    Metrics,
    /// Enable R22, start the raw AFE stream, and capture type-43 REALTIME_RAW_DATA frames (SpO2 red/IR hunt).
    Raw {
        #[arg(long, default_value_t = 20)]
        secs: u64,
        #[arg(long, default_value = "raw-stream.jsonl")]
        out: PathBuf,
    },
    /// Enable the R22 deep-data streams (16 SET_CONFIG flags; reversible).
    R22on,
    /// Fire the one-shot buzz.
    Buzz,
    /// Warm-reboot the strap (data kept).
    Reboot,
    /// Ingest a previously captured raw JSON Lines file into the calibration store (no band needed).
    Ingest {
        /// A raw-capture file (from `sync --raw`).
        capture: PathBuf,
    },
}

fn family(gen: u8) -> Family {
    if gen == 4 {
        Family::Gen4
    } else {
        Family::Gen5
    }
}

fn make_client(cli: &Cli) -> WhoopClient<BtleplugTransport> {
    let fam = family(cli.gen);
    let mut transport = BtleplugTransport::new(whoop_client::service(fam));
    if let Some(address) = &cli.address {
        transport = transport.with_target_address(address.clone());
    } else if let Some(sn) = &cli.sn {
        transport = transport.with_target_serial(sn.clone());
    } else if let Some(name) = &cli.name {
        transport = transport.with_target_name(name.clone());
    }
    WhoopClient::new(transport, fam)
}

async fn connect(cli: &Cli) -> Result<WhoopClient<BtleplugTransport>> {
    let mut client = make_client(cli);
    client.connect_and_bond().await?;
    Ok(client)
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let fam = family(cli.gen);

    match &cli.cmd {
        Cmd::Scan { secs } => {
            for b in scan_whoops(whoop_client::service(fam), *secs).await? {
                let f = |s: &str| if s.is_empty() { "?".to_string() } else { s.to_string() };
                println!(
                    "{}  rssi={:?}  sn={}  hw={}  fw={}  model={}  name=\"{}\"",
                    b.address, b.rssi, f(&b.serial), f(&b.hardware), f(&b.firmware), f(&b.model), f(&b.name)
                );
            }
        }
        Cmd::Identify => {
            let mut client = make_client(&cli);
            for (label, value) in client.identify().await? {
                println!("{label}: {value}");
            }
            client.disconnect().await.ok();
        }
        Cmd::Info => {
            let mut client = connect(&cli).await?;
            for resp in client.info().await? {
                println!("{resp:?}");
            }
            client.disconnect().await.ok();
        }
        Cmd::Sync { out, raw, wipe } => {
            let mut client = connect(&cli).await?;
            if *wipe {
                if let Err(e) = guard_wipe(&client).await {
                    client.disconnect().await.ok();
                    return Err(e);
                }
            }
            let outcome = drain_to_jsonl(&mut client, out, raw.as_deref(), *wipe).await;
            let strap = client.serial().await.or_else(|| cli.sn.clone()).or_else(|| cli.address.clone())
                .unwrap_or_else(|| "unknown".into());
            client.disconnect().await.ok();
            let (records, stats) = outcome?;
            report_sync(&records, out, *wipe);
            report_frames(&stats, raw.as_deref());
            persist_calibration(&cli, &strap, &records);
        }
        Cmd::Monitor { r#type, secs } => {
            let mut client = connect(&cli).await?;
            for f in client.monitor(*secs, *r#type).await? {
                println!("{} seq={} cmd={} crc={} {}", f.packet().name(), f.seq(), f.cmd(), f.crc_ok, to_hex(f.raw()));
            }
            client.disconnect().await.ok();
        }
        Cmd::Hr { secs, broadcast } => {
            let mut client = connect(&cli).await?;
            if *broadcast {
                client.set_broadcast_hr(true).await?;
            }
            let bpms = client.heart_rate(*secs).await?;
            if bpms.is_empty() {
                println!("no HR on 2a37 — the strap isn't broadcasting HR (try --broadcast)");
            } else {
                let (lo, hi) = (bpms.iter().min().unwrap(), bpms.iter().max().unwrap());
                println!("live HR: {} samples, {lo}-{hi} bpm, latest {} bpm", bpms.len(), bpms.last().unwrap());
            }
            client.disconnect().await.ok();
        }
        Cmd::Send { op, payload } => {
            let op = u8::from_str_radix(op.trim_start_matches("0x"), 16)?;
            let pay = payload.as_deref().map(parse_hex).transpose()?.unwrap_or_default();
            let client = connect(&cli).await?;
            client.send_raw(op, &pay).await?;
            println!("sent {op:#04x}");
            client.disconnect().await.ok();
        }
        Cmd::Metrics => {
            let mut client = connect(&cli).await?;
            let records = client.sync_history_with(false, |_| Ok(())).await?; // keep-mode: never trims
            client.disconnect().await.ok();
            report_metrics(&records, fam);
        }
        Cmd::Raw { secs, out } => {
            let mut client = connect(&cli).await?;
            let file = std::fs::File::create(out)?;
            let mut writer = std::io::BufWriter::new(file);
            let mut counts: BTreeMap<&'static str, u64> = BTreeMap::new();
            client
                .raw_stream(*secs, |f| {
                    *counts.entry(f.packet().name()).or_default() += 1;
                    let _ = writeln!(writer, "{}", capture_line(now_ms(), "raw", "stream", &f, false));
                })
                .await?;
            writer.flush()?;
            println!("raw stream: {} frame types", counts.len());
            for (name, n) in &counts {
                println!("  {name}: {n}");
            }
            println!("saved to {}", out.display());
            client.disconnect().await.ok();
        }
        Cmd::R22on => {
            let client = connect(&cli).await?;
            println!("sent {} R22 flags", client.enable_r22().await?);
            client.disconnect().await.ok();
        }
        Cmd::Buzz => {
            let client = connect(&cli).await?;
            client.buzz().await?;
            println!("buzzed");
            client.disconnect().await.ok();
        }
        Cmd::Reboot => {
            let client = connect(&cli).await?;
            client.reboot().await?;
            println!("reboot sent");
            client.disconnect().await.ok();
        }
        Cmd::Ingest { capture } => {
            let (strap, records) = decode_capture(capture, fam)?;
            let strap = strap.or_else(|| cli.sn.clone()).unwrap_or_else(|| "unknown".into());
            println!("decoded {} history records from {}", records.len(), capture.display());
            persist_calibration(&cli, &strap, &records);
        }
    }
    Ok(())
}

/// Decode a raw-capture JSON Lines file into history records (+ the strap serial from `session_id`), to
/// backfill the calibration store from a past drain without a band.
fn decode_capture(path: &Path, fam: Family) -> Result<(Option<String>, Vec<Record>)> {
    let text = std::fs::read_to_string(path)?;
    let mut records = Vec::new();
    let mut strap = None;
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let v: serde_json::Value = serde_json::from_str(line)?;
        if strap.is_none() {
            strap = v.get("session_id").and_then(|s| s.as_str()).map(str::to_string);
        }
        let Some(hex) = v.get("hex").and_then(|h| h.as_str()) else { continue };
        let Ok(bytes) = parse_hex(hex) else { continue };
        let Ok(frame) = whoop_protocol::framing::decode(fam, &bytes) else { continue };
        if frame.packet().canonical() == PacketType::HistoricalData {
            if let Some(rec) = records::decode(&frame) {
                records.push(rec);
            }
        }
    }
    Ok((strap, records))
}

/// Refuse `--wipe` when the strap serial matches the local `WHOOPCTL_PROTECT` allowlist (comma-separated
/// suffixes) or can't be read to check it. Environment-local guard; empty by default.
async fn guard_wipe(client: &WhoopClient<BtleplugTransport>) -> Result<()> {
    let protect: Vec<String> = std::env::var("WHOOPCTL_PROTECT")
        .unwrap_or_default()
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    match client.serial().await {
        Some(s) if protect.iter().any(|p| s.to_ascii_uppercase().ends_with(&p.to_ascii_uppercase())) => {
            anyhow::bail!("refusing --wipe: strap {s} is protected by WHOOPCTL_PROTECT; read-keep only")
        }
        None if !protect.is_empty() => {
            anyhow::bail!("refusing --wipe: could not read the strap serial to check WHOOPCTL_PROTECT")
        }
        Some(s) => eprintln!("wiping strap {s}: full drain, trim cursor advances"),
        None => eprintln!("wiping: serial unread (no WHOOPCTL_PROTECT set)"),
    }
    Ok(())
}

/// Drain history to `out` as JSON Lines (records) and optionally `raw` (every frame), flushing each line
/// to the OS file before its chunk's ACK. Also tallies a frame histogram for the unmapped-field report.
async fn drain_to_jsonl(
    client: &mut WhoopClient<BtleplugTransport>,
    out: &Path,
    raw: Option<&Path>,
    wipe: bool,
) -> Result<(Vec<Record>, FrameStats)> {
    let file = std::fs::File::create(out)?;
    let mut writer = std::io::BufWriter::new(file);
    let mut raw_writer = raw.map(|p| std::fs::File::create(p).map(std::io::BufWriter::new)).transpose()?;
    let session = client.serial().await.unwrap_or_else(|| "whoopctl".to_string());
    let mut stats = FrameStats::default();
    let records = client
        .sync_history_capturing(
            wipe,
            |frame| {
                stats.observe(frame);
                if let Some(w) = raw_writer.as_mut() {
                    w.write_all(capture_line(now_ms(), &session, "offload", frame, true).as_bytes())?;
                    w.write_all(b"\n")?;
                    w.flush()?;
                }
                Ok(())
            },
            |rec| {
                serde_json::to_writer(&mut writer, rec).map_err(std::io::Error::other)?;
                writer.write_all(b"\n")?;
                writer.flush()
            },
        )
        .await?;
    Ok((records, stats))
}

fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

/// Frame-type / history-version tallies over an offload, tracking which history versions failed to decode.
#[derive(Default)]
struct FrameStats {
    total: u64,
    crc_bad: u64,
    by_type: BTreeMap<&'static str, u64>,
    hist_versions: BTreeMap<u8, u64>,
    hist_undecoded: BTreeMap<u8, u64>,
}

impl FrameStats {
    fn observe(&mut self, f: &Frame) {
        self.total += 1;
        *self.by_type.entry(f.packet().name()).or_default() += 1;
        if !f.crc_ok {
            self.crc_bad += 1;
        }
        if f.packet().canonical() == PacketType::HistoricalData {
            let v = f.version();
            *self.hist_versions.entry(v).or_default() += 1;
            if f.crc_ok && records::decode(f).is_none() {
                *self.hist_undecoded.entry(v).or_default() += 1;
            }
        }
    }
}

/// Print the frame composition of the buffer and flag any history version that didn't fully decode.
fn report_frames(stats: &FrameStats, raw: Option<&Path>) {
    if stats.total == 0 {
        return;
    }
    println!("\nframe buffer: {} frames ({} crc-bad)", stats.total, stats.crc_bad);
    for (name, n) in &stats.by_type {
        println!("  {name}: {n}");
    }
    if !stats.hist_versions.is_empty() {
        println!("history versions:");
        for (v, n) in &stats.hist_versions {
            let un = stats.hist_undecoded.get(v).copied().unwrap_or(0);
            let tag = if un == 0 { "mapped" } else { "UNMAPPED" };
            println!("  v{v}: {n} frames, {un} undecoded  [{tag}]");
        }
    }
    if let Some(p) = raw {
        println!("raw frames saved to {} (lossless capture for offline field analysis)", p.display());
    }
}

/// Compute + print the derived metrics from a keep-mode drain: SpO2 (4.0 paired red/IR only) and
/// HRV-readiness from the R-R.
fn report_metrics(records: &[Record], fam: Family) {
    let history: Vec<HistoryRecord> = records.iter().filter_map(|r| match r {
        Record::History(h) => Some(h.clone()),
        _ => None,
    }).collect();
    let ppg = records.iter().filter(|r| matches!(r, Record::Ppg(_))).count();

    println!("metrics from {} records ({} history, {} v26 ppg)", records.len(), history.len(), ppg);

    if fam == Family::Gen4 {
        match Spo2::from_history(&history) {
            Some(pct) => println!("SpO2 (4.0 paired red/IR): {pct:.1}%"),
            None => println!("SpO2 (4.0): no pulsatile red/IR window (is the band worn?)"),
        }
    } else {
        print_spo2_5(&history);
    }

    let nightly = nightly_rmssd(&history);
    let nights = nightly.iter().filter(|n| n.is_some()).count();
    match HrvReadiness::evaluate(&nightly) {
        Some(r) => println!(
            "HRV-readiness: {:?}  (7-night baseline {:.0} ms, normal {:.0}-{:.0} ms)",
            r.tier, r.baseline7_ms, r.normal_low_ms, r.normal_high_ms
        ),
        None => println!(
            "HRV-readiness: calibrating ({nights} night(s) of R-R; needs {})",
            whoop_metrics::calibration::RECOVERY_SCORE.unlock
        ),
    }
}

/// Persist the drain's nights under (person, strap) and report each metric's calibration state. A new
/// person, or the same person on a new strap, is a fresh key — so it never mixes another wearer's data.
fn persist_calibration(cli: &Cli, strap: &str, records: &[Record]) {
    let store = match whoop_store::Store::open(&cli.db.to_string_lossy()) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("calibration store: {e}");
            return;
        }
    };
    match store.ingest(&cli.person, strap, records) {
        Ok(n) => println!("calibration: +{n} night(s) for person '{}' on strap {strap} → {}", cli.person, cli.db.display()),
        Err(e) => {
            eprintln!("calibration ingest: {e}");
            return;
        }
    }
    report_cal_state("SpO2 baseline (%)", store.spo2_state(&cli.person, strap));
    report_cal_state("HRV baseline (ms)", store.hrv_state(&cli.person, strap));
}

fn report_cal_state(label: &str, state: Result<whoop_store::CalState, whoop_store::StoreError>) {
    use whoop_store::CalState;
    match state {
        Ok(CalState::Baseline { value, nights }) => println!("  {label}: {value:.1} ({nights} night(s), calibrated)"),
        Ok(CalState::Calibrating { have, need }) => println!("  {label}: calibrating ({have}/{need} night(s))"),
        Err(e) => eprintln!("  {label}: {e}"),
    }
}

/// The 5.0/MG SpO2 summary from the v18 spo2_pct field (the sleep-only computed percent).
fn print_spo2_5(history: &[HistoryRecord]) {
    let spo2: Vec<u8> = history.iter().filter_map(|h| h.spo2_pct).collect();
    let (Some(&lo), Some(&hi)) = (spo2.iter().min(), spo2.iter().max()) else {
        println!("SpO2 (5.0/MG): no sleep readings in this drain (generated on-wrist during sleep)");
        return;
    };
    let mean = spo2.iter().map(|&v| v as u32).sum::<u32>() as f32 / spo2.len() as f32;
    println!("SpO2 (5.0/MG): {} sleep readings, {lo}-{hi}% (mean {mean:.1}%)", spo2.len());
}

/// Per-day RMSSD (ms) from the R-R carried in history records, oldest → newest, for HRV-readiness.
fn nightly_rmssd(history: &[HistoryRecord]) -> Vec<Option<f64>> {
    let mut by_day: BTreeMap<u32, Vec<u16>> = BTreeMap::new();
    for h in history {
        if !h.rr_intervals.is_empty() {
            by_day.entry(h.unix / 86_400).or_default().extend(&h.rr_intervals);
        }
    }
    by_day.values().map(|rr| HrvReadiness::rmssd(rr)).collect()
}

/// Print a one-look summary of a decode: counts by kind, HR range, R-R total, time span, and where saved.
fn report_sync(records: &[Record], out: &Path, wiped: bool) {
    let (mut hist, mut ppg, mut imu) = (0u32, 0u32, 0u32);
    let (mut hr_lo, mut hr_hi) = (u8::MAX, 0u8);
    let mut rr = 0usize;
    let (mut t0, mut t1) = (u32::MAX, 0u32);
    for r in records {
        let unix = match r {
            Record::History(h) => {
                hist += 1;
                if let Some(hr) = h.heart_rate {
                    hr_lo = hr_lo.min(hr);
                    hr_hi = hr_hi.max(hr);
                }
                rr += h.rr_intervals.len();
                h.unix
            }
            Record::Ppg(p) => {
                ppg += 1;
                p.unix
            }
            Record::Imu(i) => {
                imu += 1;
                i.unix
            }
        };
        if unix > 0 {
            t0 = t0.min(unix);
            t1 = t1.max(unix);
        }
    }
    println!("{} records  (history {hist}, ppg {ppg}, imu {imu})", records.len());
    if hr_hi > 0 {
        println!("heart rate {hr_lo}-{hr_hi} bpm, {rr} R-R intervals");
    }
    let spo2: Vec<u8> = records.iter().filter_map(|r| match r {
        Record::History(h) => h.spo2_pct,
        _ => None,
    }).collect();
    if let (Some(lo), Some(hi)) = (spo2.iter().min(), spo2.iter().max()) {
        println!("SpO2 (5.0 sleep): {} readings, {lo}-{hi}%", spo2.len());
    }
    if t0 != u32::MAX {
        println!("span unix {t0}..{t1} ({}s)", t1 - t0);
    }
    println!("saved to {}", out.display());
    if !wiped {
        println!("strap NOT trimmed (default keep; --wipe drains the full history)");
    }
}

fn parse_hex(s: &str) -> Result<Vec<u8>> {
    let s = s.trim();
    anyhow::ensure!(s.len().is_multiple_of(2), "odd hex length");
    (0..s.len())
        .step_by(2)
        .map(|i| -> Result<u8> { Ok(u8::from_str_radix(&s[i..i + 2], 16)?) })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{family, parse_hex, Family};

    #[test]
    fn family_maps_gen_number() {
        assert_eq!(family(4), Family::Gen4);
        assert_eq!(family(5), Family::Gen5);
        assert_eq!(family(9), Family::Gen5); // anything but 4 → 5 (the default)
    }

    #[test]
    fn parse_hex_decodes_and_rejects_odd_length() {
        assert_eq!(parse_hex("0a1bFF").unwrap(), vec![0x0a, 0x1b, 0xff]);
        assert!(parse_hex("abc").is_err());
    }
}
