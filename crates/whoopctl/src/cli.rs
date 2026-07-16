//! The clap command-line surface: global options and the subcommand set.

use std::path::PathBuf;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "whoopctl", about = "WHOOP BLE tool")]
pub(crate) struct Cli {
    /// Generation: 5 (5.0/MG, default) or 4 (4.0).
    #[arg(long, default_value_t = 5)]
    pub(crate) gen: u8,
    /// Target band by advertised name (else the first WHOOP found).
    #[arg(long)]
    pub(crate) name: Option<String>,
    /// Target band by BLE address (surest when several WHOOPs are in range).
    #[arg(long)]
    pub(crate) address: Option<String>,
    /// Target band by serial, full or suffix (connects to each candidate and reads its serial).
    #[arg(long)]
    pub(crate) sn: Option<String>,
    /// Who is wearing the strap. Calibration is per (person, strap): a new person, or the same person on
    /// a new strap, starts a fresh calibration period — you never calibrate against someone else's data.
    #[arg(long, default_value = "default")]
    pub(crate) person: String,
    /// Calibration store (SQLite) path.
    #[arg(long, default_value = "whoop-cal.db")]
    pub(crate) db: PathBuf,
    /// Opt in to the wellness HR watch: a display-only nudge on a sustained elevated at-rest heart rate.
    /// Retrospective and never medical; no device write, no buzz.
    #[arg(long, default_value_t = false)]
    pub(crate) hr_watch: bool,
    #[command(subcommand)]
    pub(crate) cmd: Cmd,
}

#[derive(Subcommand)]
pub(crate) enum Cmd {
    /// List WHOOP bands in range.
    Scan {
        #[arg(long, default_value_t = 6)]
        secs: u64,
    },
    /// Connect (no bond) and read the standard identity chars (name/serial/fw). Read-only.
    Identify,
    /// Connect, bond, read identity/battery/data-range.
    Info,
    /// Connect, bond, read the 5.0 battery-pack fuel gauge (serial, SOC, mV, pack-id). 5.0 only.
    Pack,
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
