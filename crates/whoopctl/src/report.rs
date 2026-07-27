//! Human-readable CLI output: drain/sync summaries, the frame histogram, derived-metric lines, the
//! wellness HR-watch line, and calibration-store persistence + status.

use std::collections::BTreeMap;
use std::path::Path;

use physio_algo::{HrWatch, HrWatchState, HrvReadiness, Spo2};
use whoop_protocol::{records, Family, Frame, HistoryRecord, PacketType, Record};

use crate::cli::Cli;

/// Report the v20/v21 breakdown + sanity of a deep-buffer capture. IMU sanity = the gravity shell
/// (|accel| ≈ 4096 LSB = 1 g within ±15% across all samples); optical sanity = the green channel (0)
/// carrying a varying waveform while the dark reference (2/3) stays flat.
pub(crate) fn decode_report(path: &Path, strap: Option<&str>, records: &[Record]) {
    let (mut hist, mut ppg, mut imu, mut optical) = (0u32, 0u32, 0u32, 0u32);
    let (mut imu_shell, mut imu_dims) = (0u32, (0usize, 0usize));
    let (mut opt_pulse, mut opt_flat, mut opt_dims) = (0u32, 0u32, (0usize, 0usize));

    for r in records {
        match r {
            Record::History(_) => hist += 1,
            Record::Ppg(_) => ppg += 1,
            Record::Imu(m) => {
                imu += 1;
                imu_dims = (m.accel.len(), 6);
                let ok = !m.accel.is_empty()
                    && m.accel.iter().all(|a| {
                        let mag = ((a[0] as f64).powi(2) + (a[1] as f64).powi(2) + (a[2] as f64).powi(2)).sqrt();
                        (3482.0..4710.0).contains(&mag) // 4096 ± 15%
                    });
                if ok {
                    imu_shell += 1;
                }
            }
            Record::Optical(o) => {
                optical += 1;
                opt_dims = (o.channels.len(), o.channels.first().map_or(0, |c| c.len()));
                if span(o.channels.first()) > 8 {
                    opt_pulse += 1; // green varies = a real waveform
                }
                if o.channels.get(2).is_some_and(|c| span(Some(c)) <= 8) {
                    opt_flat += 1; // dark/ambient reference is flat
                }
            }
        }
    }

    println!("decode-capture: {}  strap={}", path.display(), strap.unwrap_or("?"));
    println!("  {} frames decoded — history {hist}, ppg {ppg}, imu(v21) {imu}, optical(v20) {optical}", records.len());
    if imu > 0 {
        println!(
            "  v21 IMU:     {imu} buffers × {}×{}-axis; gravity-shell OK {imu_shell}/{imu} ({}%)",
            imu_dims.0, imu_dims.1, pct(imu_shell, imu),
        );
    }
    if optical > 0 {
        println!(
            "  v20 optical: {optical} buffers × {}ch×{}; green varies {opt_pulse}/{optical} ({}%), dark flat {opt_flat}/{optical} ({}%)",
            opt_dims.0, opt_dims.1, pct(opt_pulse, optical), pct(opt_flat, optical),
        );
    }
    if imu == 0 && optical == 0 {
        println!("  ⚠ NO v20/v21 deep buffers here — only per-second v18/v26. Band wasn't worn+asleep under R22, or the backlog was already drained.");
    } else {
        let imu_ok = imu == 0 || imu_shell * 2 >= imu;
        let opt_ok = optical == 0 || opt_pulse * 2 >= optical;
        println!("  => decoders {}", if imu_ok && opt_ok { "VALIDATED on real bytes ✅" } else { "decoded, but sanity WEAK — check the layout ⚠" });
    }
}

/// Min→max span of a channel's samples (0 if empty/None).
fn span(ch: Option<&Vec<i32>>) -> i32 {
    match ch {
        Some(c) if !c.is_empty() => {
            let (mn, mx) = c.iter().fold((i32::MAX, i32::MIN), |(a, b), &v| (a.min(v), b.max(v)));
            mx.saturating_sub(mn)
        }
        _ => 0,
    }
}

fn pct(n: u32, d: u32) -> u32 {
    (n * 100).checked_div(d).unwrap_or(0)
}

/// Frame-type / history-version tallies over an offload, tracking which history versions failed to decode.
#[derive(Default)]
pub(crate) struct FrameStats {
    total: u64,
    crc_bad: u64,
    by_type: BTreeMap<&'static str, u64>,
    hist_versions: BTreeMap<u8, u64>,
    hist_undecoded: BTreeMap<u8, u64>,
}

impl FrameStats {
    pub(crate) fn observe(&mut self, f: &Frame) {
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
pub(crate) fn report_frames(stats: &FrameStats, raw: Option<&Path>) {
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

/// The history records of a decode, cloned out of the mixed record stream.
fn history_of(records: &[Record]) -> Vec<HistoryRecord> {
    records.iter().filter_map(|r| match r {
        Record::History(h) => Some(h.clone()),
        _ => None,
    }).collect()
}

/// Print the wellness HR-watch line for a decode (opt-in; filters history first).
pub(crate) fn hr_watch(records: &[Record]) {
    hr_watch_line(&history_of(records));
}

fn hr_watch_line(history: &[HistoryRecord]) {
    println!("{}", format_hr_watch(HrWatch::evaluate(history)));
}

/// Compute + print the derived metrics from a keep-mode drain: SpO2 (4.0 paired red/IR only) and
/// HRV-readiness from the R-R.
pub(crate) fn report_metrics(records: &[Record], fam: Family, hr_watch: bool) {
    let history = history_of(records);
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

    let nightly = HrvReadiness::nightly_rmssd(&history);
    let nights = nightly.iter().filter(|n| n.is_some()).count();
    match HrvReadiness::evaluate(&nightly) {
        Some(r) => println!(
            "HRV-readiness: {:?}  (7-night baseline {:.0} ms, normal {:.0}-{:.0} ms)",
            r.tier, r.baseline7_ms, r.normal_low_ms, r.normal_high_ms
        ),
        None => println!(
            "HRV-readiness: calibrating ({nights} night(s) of R-R; needs {})",
            physio_algo::calibration::RECOVERY_SCORE.unlock
        ),
    }

    if hr_watch {
        hr_watch_line(&history);
    }
}

/// Render the opt-in HR watch as a display-only line. Wellness wording only — the guard test forbids any
/// clinical term so this can never read as a diagnosis.
pub(crate) fn format_hr_watch(state: HrWatchState) -> String {
    match state {
        HrWatchState::Calibrating { have, need } => {
            format!("HR watch: building your at-rest baseline ({have}/{need} at-rest samples)")
        }
        HrWatchState::Normal => "HR watch: at-rest heart rate within your usual range".to_string(),
        HrWatchState::ElevatedAtRest { peak_bpm, dur_s, .. } => format!(
            "HR watch: sustained higher-than-usual heart rate while at rest ({peak_bpm} bpm for {} min) \
             — a wellness nudge, not medical",
            dur_s / 60
        ),
    }
}

/// Persist the drain's nights under (person, strap) and report each metric's calibration state. A new
/// person, or the same person on a new strap, is a fresh key — so it never mixes another wearer's data.
pub(crate) fn persist_calibration(cli: &Cli, strap: &str, records: &[Record]) {
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

/// Print a one-look summary of a decode: counts by kind, HR range, R-R total, time span, and where saved.
pub(crate) fn report_sync(records: &[Record], out: &Path, wiped: bool) {
    let (mut hist, mut ppg, mut imu, mut optical) = (0u32, 0u32, 0u32, 0u32);
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
            Record::Optical(o) => {
                optical += 1;
                o.unix
            }
        };
        if unix > 0 {
            t0 = t0.min(unix);
            t1 = t1.max(unix);
        }
    }
    println!("{} records  (history {hist}, ppg {ppg}, imu {imu}, optical {optical})", records.len());
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

#[cfg(test)]
mod tests {
    use super::format_hr_watch;
    use physio_algo::HrWatchState;

    #[test]
    fn hr_watch_line_carries_no_clinical_terms() {
        // Every wording the HR watch can emit must read as wellness, never as a diagnosis.
        let banned = ["afib", "fibrillat", "arrhythm", "cardiac", "ecg", "ekg", "diagnos", "alarm", "emergency"];
        let lines = [
            format_hr_watch(HrWatchState::Calibrating { have: 10, need: 600 }),
            format_hr_watch(HrWatchState::Normal),
            format_hr_watch(HrWatchState::ElevatedAtRest { peak_bpm: 118, start_unix: 0, dur_s: 480 }),
        ];
        for line in lines {
            let low = line.to_lowercase();
            for term in banned {
                assert!(!low.contains(term), "clinical term '{term}' in HR-watch line: {line}");
            }
        }
    }

    #[test]
    fn hr_watch_elevated_line_reports_bpm_and_minutes() {
        let line = format_hr_watch(HrWatchState::ElevatedAtRest { peak_bpm: 118, start_unix: 0, dur_s: 480 });
        assert!(line.contains("118 bpm") && line.contains("8 min"));
    }
}
