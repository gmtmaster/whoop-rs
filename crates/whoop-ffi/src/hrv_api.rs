//! Beat-interval analysis: RMSSD and its variants, frequency domain, readiness, and the
//! respiration rate derived from the same intervals.

use crate::*;

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

/// RMSSD (ms) of raw R-R values. `None` for <2 beats (filtered, drops deltas >200ms).
#[uniffi::export]
pub fn hrv_rmssd(rr_ms: Vec<u16>) -> Option<f64> {
    HrvReadiness::rmssd(&rr_ms)
}

/// Plain unfiltered RMSSD (no artifact rejection); the raw counterpart to `hrv_rmssd`.
#[uniffi::export]
pub fn hrv_rmssd_plain(rr_ms: Vec<u16>) -> Option<f64> {
    HrvReadiness::rmssd_plain(&rr_ms)
}

/// Range-filter R-R values, keeping only 300–2000 ms.
#[uniffi::export]
pub fn hrv_range_filter(rr_ms: Vec<u16>) -> Vec<u16> {
    HrvReadiness::range_filter(&rr_ms)
}

/// Standard deviation of NN intervals (ms), sample std (ddof=1). `None` for <2 values.
#[uniffi::export]
pub fn hrv_sdnn(rr_ms: Vec<u16>) -> Option<f64> {
    HrvReadiness::sdnn(&rr_ms)
}

/// Full HRV analysis over a raw R-R capture (ms): cleaned RMSSD/SDNN/pNN50/meanNN + input/clean counts.
/// Fields are `None` (and `n_clean` 0) below the 20-clean-beat floor, or when `max_rejected_fraction` (the
/// spot honesty gate) is set and exceeded. The nightly path passes `None` (no gate).
#[derive(uniffi::Record)]
pub struct HrvAnalysisInfo {
    pub rmssd: Option<f64>,
    pub sdnn: Option<f64>,
    pub mean_nn: Option<f64>,
    pub pnn50: Option<f64>,
    pub n_input: u32,
    pub n_clean: u32,
}

/// Clean-and-analyze a raw R-R series in one call (the app's full spot/nightly HRV analysis path).
#[uniffi::export]
pub fn hrv_analyze_raw(rr_ms: Vec<u16>, max_rejected_fraction: Option<f64>) -> HrvAnalysisInfo {
    let a = HrvReadiness::analyze_raw(&rr_ms, max_rejected_fraction);
    HrvAnalysisInfo {
        rmssd: a.rmssd,
        sdnn: a.sdnn,
        mean_nn: a.mean_nn,
        pnn50: a.pnn50,
        n_input: a.n_input,
        n_clean: a.n_clean,
    }
}

/// Frequency-domain HRV bands (ms²): LF / HF / LF-HF / total power. `lf`/`lfhf` are `None` under the 250 s
/// LF span gate; the whole record is absent (`None`) under the 60 s HF gate or the 20-beat floor.
#[derive(uniffi::Record)]
pub struct HrvBandsInfo {
    pub lf: Option<f64>,
    pub hf: f64,
    pub lfhf: Option<f64>,
    pub total_power: f64,
}

/// Frequency-domain HRV over a time-ordered R-R series (ms) via the Lomb-Scargle periodogram.
#[uniffi::export]
pub fn hrv_freq_domain(rr_ms: Vec<u16>) -> Option<HrvBandsInfo> {
    physio_algo::hrv_freq::freq_domain(&rr_ms).map(|b| HrvBandsInfo {
        lf: b.lf,
        hf: b.hf,
        lfhf: b.lfhf,
        total_power: b.total_power,
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

/// Deep-sleep-windowed session avgHrv (ms): per-5min-bucket RMSSD like [hrv_windowed_avg], keeping only
/// buckets whose center falls inside a deep-sleep (SWS/N3) span. Takes the full segment list; filters for
/// `SleepStage::Deep` internally. `None` when no deep bucket yields a value.
#[uniffi::export]
pub fn hrv_windowed_avg_deep(
    start: u32,
    end: u32,
    runs: Vec<RrRun>,
    segments: Vec<SleepSegment>,
) -> Option<f64> {
    let beats: Vec<(u32, u16)> = runs.iter().flat_map(|r| r.rr.iter().map(|b| (r.unix, *b))).collect();
    let deep_spans: Vec<(u32, u32)> = segments
        .iter()
        .filter(|s| matches!(s.stage, SleepStage::Deep))
        .map(|s| (s.start as u32, s.end as u32))
        .collect();
    HrvReadiness::windowed_avg_hrv_deep(start, end, &beats, &deep_spans)
}

// ── Rest (sleep performance composite) ─────────────────────────────────────

/// Nocturnal RMSSD age norm (ms) — the reference the HRV driver is scored against.
#[uniffi::export]
pub fn vitality_rmssd_norm(for_age: f64) -> f64 {
    vitality::rmssd_norm(for_age)
}
