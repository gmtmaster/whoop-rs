//! WHOOP BLE CLI over the btleplug transport. Read-safe by default; the write actions (r22on/buzz/
//! reboot/send) are the gated ones from whoop-client. All the on-band behaviour is validated here, not
//! in unit tests — a connect proves nothing about a real strap.

use std::collections::BTreeMap;
use std::io::Write as _;
use std::path::Path;

use anyhow::Result;
use clap::Parser;

use ble_btleplug::{scan_whoops, BtleplugTransport};
use whoop_client::{capture_line, WhoopClient};
use whoop_protocol::bytes::to_hex;
use whoop_protocol::{command, response, Family, Frame, Record};

mod cli;
mod flash;
mod report;

use cli::{Cli, Cmd};
use report::{hr_watch, persist_calibration, report_frames, report_metrics, report_sync, FrameStats};

fn family(gen: u8) -> Family {
    if gen == 4 {
        Family::Gen4
    } else {
        Family::Gen5
    }
}

pub(crate) fn make_client(cli: &Cli) -> WhoopClient<BtleplugTransport> {
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

pub(crate) async fn connect(cli: &Cli) -> Result<WhoopClient<BtleplugTransport>> {
    let mut client = make_client(cli);
    client.connect_and_bond().await?;
    Ok(client)
}

/// Connect to one named band, whatever the global selectors say — an explicit `--address` still wins.
/// Used where attaching to the wrong strap is itself the hazard.
pub(crate) async fn connect_serial(cli: &Cli, serial: &str) -> Result<WhoopClient<BtleplugTransport>> {
    let fam = family(cli.gen);
    let mut transport = BtleplugTransport::new(whoop_client::service(fam));
    transport = match &cli.address {
        Some(address) => transport.with_target_address(address.clone()),
        None => transport.with_target_serial(serial.to_string()),
    };
    let mut client = WhoopClient::new(transport, fam);
    client.connect_and_bond().await?;
    Ok(client)
}

/// Read GET_BODY_LOCATION_AND_STATUS and print the strap's own wear block. The location/status codes
/// are unmapped, so the raw bytes are what a wrist write is judged against.
async fn print_body_location(client: &mut WhoopClient<BtleplugTransport>, when: &str) -> Result<()> {
    let frames = client.probe(command::GET_BODY_LOCATION_AND_STATUS, &[0x00], 2).await?;
    match frames.iter().find_map(response::body_location) {
        Some(b) => println!("body location ({when}): sub={} location={} status={}", b.sub, b.location, b.status),
        None => println!("body location ({when}): no reply in {} frames", frames.len()),
    }
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let fam = family(cli.gen);

    match &cli.cmd {
        Cmd::Scan { secs } => {
            for b in scan_whoops(&whoop_client::all_services(), *secs).await? {
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
        Cmd::Pack => {
            use whoop_protocol::response::CommandResponse;
            let mut client = connect(&cli).await?;
            for resp in client.battery_pack().await? {
                match resp {
                    CommandResponse::NoBatteryPack => println!("no battery pack attached"),
                    CommandResponse::BatteryPack { serial, soc_pct, bt_addr } => {
                        println!("pack {serial}  soc={soc_pct:.1}%  addr={bt_addr}");
                    }
                    other => println!("{other:?}"),
                }
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
            let mut client = connect(&cli).await?;
            for f in client.probe(op, &pay, 2).await? {
                println!("{} cmd={} crc={} {}", f.packet().name(), f.cmd(), f.crc_ok, to_hex(f.raw()));
            }
            client.disconnect().await.ok();
        }
        Cmd::Metrics => {
            let mut client = connect(&cli).await?;
            let records = client.sync_history_with(false, |_| Ok(())).await?; // keep-mode: never trims
            client.disconnect().await.ok();
            report_metrics(&records, fam, cli.hr_watch);
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
        Cmd::Wrist { set } => {
            let side = match set.as_deref() {
                None => None,
                Some("left") => Some(false),
                Some("right") => Some(true),
                Some(other) => anyhow::bail!("wrist must be left or right, got {other}"),
            };
            let mut client = connect(&cli).await?;
            print_body_location(&mut client, "before").await?;
            if let Some(right) = side {
                client.select_wrist(right).await?;
                println!("sent SELECT_WRIST {}", if right { "right" } else { "left" });
                print_body_location(&mut client, "after").await?;
            }
            client.disconnect().await.ok();
        }
        Cmd::UndoTrim { agree } => {
            let mut client = connect(&cli).await?;
            let frames = if *agree {
                println!("sending the reset: the strap rewinds its trim and read pointers to oldest");
                client.undo_trim().await
            } else {
                println!(
                    "dry run — sending the inert payload the strap declines, so no pointer moves.\n\
                     with --i-agree-to-possible-loss-of-data this would ask the strap to rewind its trim\n\
                     and read pointers to oldest; that is a pointer move and cannot un-erase flash."
                );
                client.undo_trim_dry_run().await
            };
            client.disconnect().await.ok();
            report_trim_reply(&frames?);
        }
        Cmd::R22on { with_v9, report } => {
            let v9 = with_v9.as_deref().map(|v| v.as_bytes()[0]);
            let mut client = connect(&cli).await?;
            if *report || v9.is_some() {
                for (name, result) in client.enable_r22_reporting(v9).await? {
                    let shown = result.map_or_else(|| "no reply".to_string(), |r| format!("{r:?}"));
                    println!("  {name:<32} {shown}");
                }
            } else {
                println!("sent {} R22 flags", client.enable_r22().await?);
            }
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
            if cli.hr_watch {
                hr_watch(&records);
            }
        }
        Cmd::DecodeCapture { file } => {
            let (strap, records) = decode_capture(file, fam)?;
            report::decode_report(file, strap.as_deref(), &records);
        }
        Cmd::Ecg(args) => render_ecg(args)?,
        Cmd::Flash(args) => flash::run(&cli, args).await?,
    }
    Ok(())
}

/// Draw the ECG strips. Offline only: the R16 packet layout and the counts-per-mV are unread, so
/// there is nothing to decode off a strap yet and building the session first would mean guessing.
fn render_ecg(args: &cli::EcgArgs) -> Result<()> {
    use whoopctl::ecg_render::{demo, driver, fit, Request};

    if !args.demo {
        anyhow::bail!(
            "live ECG capture is not implemented: the R16 packet layout and the strap's counts-per-mV \
             have not been read, so any decode would be a guess. Use --demo to render the synthetic \
             waveform at the true scale."
        );
    }
    let signal = demo::DemoSignal::parse(&args.demo_signal)
        .ok_or_else(|| anyhow::anyhow!("unknown --demo-signal {:?}; try pulse or ecg", args.demo_signal))?;

    let req = Request {
        duration_s: args.duration,
        strip_s: args.strip_seconds,
        mm_per_s: args.mm_per_s,
        mm_per_mv: args.mm_per_mv,
        strip_mm: args.strip_mm,
        cell_aspect: args.cell_aspect,
        dots_per_mm_x: args.dots_per_mm,
        counts_per_mv: args.counts_per_mv,
        sample_rate_hz: args.sample_rate,
        grid: !args.no_grid,
        ..Request::default()
    };
    let term = driver::terminal(args.width, args.height);
    // Refuse loudly rather than draw at a scale the label does not match.
    fit(&req, term).map_err(|e| anyhow::anyhow!("{e}"))?;

    let opts = demo::DemoOptions {
        signal,
        lead_off: args.demo_lead_off,
        speed: args.speed.max(0.0),
        colour: !args.no_colour && driver::stdout_is_terminal(),
    };
    println!(
        "terminal {}x{} ({} / {})",
        term.cols,
        term.rows,
        term.cols_from.tag(),
        term.rows_from.tag(),
    );
    let mut out = std::io::stdout();
    demo::run(&req, term, &opts, &mut out).map_err(|e| anyhow::anyhow!("{e}"))?;
    Ok(())
}

/// Read a raw-capture JSON Lines file and decode it into history records (+ the strap serial), to backfill
/// the calibration store from a past drain without a band. The line/frame decode is owned by whoop-client.
fn decode_capture(path: &Path, fam: Family) -> Result<(Option<String>, Vec<Record>)> {
    let text = std::fs::read_to_string(path)?;
    Ok(whoop_client::decode_capture(&text, fam))
}

/// Print whatever the strap answered an `undo-trim` write with. Empty is the interesting case: the frame
/// was written, so silence says the handler produced no reply, not that the write failed.
fn report_trim_reply(frames: &[Frame]) {
    if frames.is_empty() {
        println!("no reply — the command was written, the strap answered nothing");
        return;
    }
    println!("strap answered with {} frame(s):", frames.len());
    for f in frames {
        println!("  {} cmd={} crc={} {}", f.packet().name(), f.cmd(), f.crc_ok, to_hex(f.raw()));
    }
}

/// Serial suffixes the local `WHOOPCTL_PROTECT` names, comma-separated. Empty by default; it can only
/// ever narrow what a write reaches, never widen it.
pub(crate) fn protect_list() -> Vec<String> {
    std::env::var("WHOOPCTL_PROTECT")
        .unwrap_or_default()
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

/// Refuse `--wipe` when the strap serial matches the local `WHOOPCTL_PROTECT` allowlist or can't be
/// read to check it.
async fn guard_wipe(client: &WhoopClient<BtleplugTransport>) -> Result<()> {
    let protect = protect_list();
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

fn parse_hex(s: &str) -> Result<Vec<u8>> {
    whoop_protocol::bytes::from_hex(s.trim()).ok_or_else(|| anyhow::anyhow!("invalid hex"))
}

#[cfg(test)]
mod tests {
    use super::{family, parse_hex};
    use whoop_protocol::Family;

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
