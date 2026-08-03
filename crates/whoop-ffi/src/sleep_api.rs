//! Sleep: staging a night, picking the main night, naps, debt and regularity.

use crate::*;

pub(crate) fn to_night_blocks(blocks: &[MainNightBlock]) -> Vec<sleep::NightBlock> {
    blocks.iter().map(|b| sleep::NightBlock { start: b.start, end: b.end }).collect()
}

fn to_scored_blocks(blocks: &[MainNightScoredBlock]) -> Vec<sleep::ScoredNightBlock> {
    blocks
        .iter()
        .map(|b| sleep::ScoredNightBlock { onset: b.onset, asleep_s: b.asleep_s, in_bed_s: b.in_bed_s })
        .collect()
}

/// Detect + stage a night's streams: one call carves the in-bed spans and returns one session each.
#[uniffi::export]
pub fn analyze_sleep(streams: SleepStreams) -> Vec<SleepSession> {
    sleep::analyze(&streams.into()).into_iter().map(SleepSession::from).collect()
}

#[uniffi::export]
pub fn main_night_index(blocks: Vec<MainNightBlock>, offset_s: i64, habitual_midsleep_sec: Option<i64>) -> Option<u32> {
    sleep::main_night_index(&to_night_blocks(&blocks), offset_s, habitual_midsleep_sec).map(|i| i as u32)
}

#[uniffi::export]
pub fn main_night_group_indices(
    blocks: Vec<MainNightBlock>,
    offset_s: i64,
    habitual_midsleep_sec: Option<i64>,
) -> Option<Vec<u32>> {
    sleep::main_night_group_indices(&to_night_blocks(&blocks), offset_s, habitual_midsleep_sec)
        .map(|v| v.into_iter().map(|i| i as u32).collect())
}

#[uniffi::export]
pub fn main_night_selection(
    blocks: Vec<MainNightBlock>,
    offset_s: i64,
    habitual_midsleep_sec: Option<i64>,
) -> Option<MainNightSel> {
    sleep::main_night_selection(&to_night_blocks(&blocks), offset_s, habitual_midsleep_sec).map(|s| MainNightSel {
        index: s.index as u32,
        reason: s.reason.into(),
        asleep_sec: s.asleep_sec,
    })
}

/// The day's main night scored on DECODED stage time rather than the clock span.
#[uniffi::export]
pub fn main_night_index_scored(
    blocks: Vec<MainNightScoredBlock>,
    offset_s: i64,
    habitual_midsleep_sec: Option<i64>,
) -> Option<u32> {
    sleep::main_night_index_scored(&to_scored_blocks(&blocks), offset_s, habitual_midsleep_sec).map(|i| i as u32)
}

/// The main-night group scored on DECODED stage time: one scorer, fed what the hypnogram holds instead
/// of the clock span, so the stages path and the detected path cannot name different nights.
#[uniffi::export]
pub fn main_night_group_indices_scored(
    blocks: Vec<MainNightScoredBlock>,
    offset_s: i64,
    habitual_midsleep_sec: Option<i64>,
) -> Option<Vec<u32>> {
    sleep::main_night_group_indices_scored(&to_scored_blocks(&blocks), offset_s, habitual_midsleep_sec)
        .map(|v| v.into_iter().map(|i| i as u32).collect())
}

/// The scored main-night pick plus why it won; `asleep_sec` is the winner's decoded asleep time.
#[uniffi::export]
pub fn main_night_selection_scored(
    blocks: Vec<MainNightScoredBlock>,
    offset_s: i64,
    habitual_midsleep_sec: Option<i64>,
) -> Option<MainNightSel> {
    sleep::main_night_selection_scored(&to_scored_blocks(&blocks), offset_s, habitual_midsleep_sec).map(|s| {
        MainNightSel { index: s.index as u32, reason: s.reason.into(), asleep_sec: s.asleep_sec }
    })
}

#[uniffi::export]
pub fn bridged_night_groups(blocks: Vec<MainNightBlock>, offset_s: i64) -> Vec<SleepBridgedGroup> {
    sleep::bridged_night_groups(&to_night_blocks(&blocks), offset_s)
        .into_iter()
        .map(|g| SleepBridgedGroup {
            indices: g.indices.into_iter().map(|i| i as u32).collect(),
            gaps: g.gaps.into_iter().map(|(s, e)| SleepGap { start: s, end: e }).collect(),
        })
        .collect()
}

#[uniffi::export]
pub fn habitual_midsleep_sec(history: Vec<SleepHistoryBlock>, offset_s: i64, min_days: u32) -> Option<i64> {
    let h: Vec<sleep::HistoryBlock> = history
        .into_iter()
        .map(|b| sleep::HistoryBlock { start: b.start, end: b.end, day_key: b.day_key })
        .collect();
    sleep::habitual_midsleep_sec(&h, offset_s, min_days as usize)
}

/// One local day of the habitual-midsleep series; `midsleep_sec` is None where the trailing window holds
/// fewer than the day floor.
#[derive(uniffi::Record)]
pub struct HabitualMidsleepDay {
    pub day: String,
    pub midsleep_sec: Option<i64>,
}

/// The habitual midsleep PER DAY over a trailing `window_days` window, ascending, so the consistency
/// band bends with the habit instead of sitting flat. A day whose window is too thin carries None.
#[uniffi::export]
pub fn habitual_midsleep_series(
    history: Vec<SleepHistoryBlock>,
    offset_s: i64,
    min_days: u32,
    window_days: u32,
) -> Vec<HabitualMidsleepDay> {
    let h: Vec<sleep::HistoryBlock> = history
        .into_iter()
        .map(|b| sleep::HistoryBlock { start: b.start, end: b.end, day_key: b.day_key })
        .collect();
    sleep::habitual_midsleep_series(&h, offset_s, min_days as usize, window_days as usize)
        .into_iter()
        .map(|(day, midsleep_sec)| HabitualMidsleepDay { day, midsleep_sec })
        .collect()
}

/// Stage one already-detected in-bed span with the V2 recipe + motion-aware wake refinement (the
/// single-span edit self-heal path). Per-30 s-epoch stage segments over `[start, end]`.
#[uniffi::export]
pub fn stage_sleep_refined(input: SleepInput, steps: Vec<SleepStepSample>) -> Vec<SleepSegment> {
    let steps: Vec<sleep::StepSample> = steps
        .into_iter()
        .map(|s| sleep::StepSample { ts: s.ts, counter: s.counter, activity_class: s.activity_class })
        .collect();
    to_sleep_segments(sleep::stage_refined(&input.into(), &steps))
}

/// Tri-state nap verdict.
#[derive(uniffi::Enum)]
pub enum NapVerdictInfo {
    Nap,
    None,
    Inconclusive,
}

/// A proposed nap to offer for review. `confidence` orders the UI only, never a medical claim.
#[derive(uniffi::Record)]
pub struct NapCandidateInfo {
    pub start: i64,
    pub end: i64,
    pub mean_hr: Option<i32>,
    pub confidence: f64,
}

/// The nap verdict + (only when `Nap`) the candidate to review.
#[derive(uniffi::Record)]
pub struct NapDecisionInfo {
    pub verdict: NapVerdictInfo,
    pub candidate: Option<NapCandidateInfo>,
}

/// User-tunable nap thresholds.
#[derive(uniffi::Record)]
pub struct NapConfigInfo {
    pub enabled: bool,
    pub min_nap_minutes: i32,
    pub max_nap_minutes: i32,
    pub still_threshold_g: f64,
    pub hr_settle_margin_bpm: i32,
    pub smooth_window_seconds: f64,
}

/// Classify one candidate window for a short nap (tri-state, conservative — only PROPOSES a review card).
#[uniffi::export]
pub fn nap_evaluate(
    gravity: Vec<WorkoutGravitySample>,
    hr: Vec<HrTick>,
    resting_hr: Option<i32>,
    config: NapConfigInfo,
) -> NapDecisionInfo {
    let grav: Vec<workout::GravitySample> = gravity
        .into_iter()
        .map(|g| workout::GravitySample { ts: g.ts, x: g.x, y: g.y, z: g.z })
        .collect();
    let hr_samples = to_hr(hr);
    let cfg = physio_algo::nap::NapConfig {
        enabled: config.enabled,
        min_nap_minutes: config.min_nap_minutes,
        max_nap_minutes: config.max_nap_minutes,
        still_threshold_g: config.still_threshold_g,
        hr_settle_margin_bpm: config.hr_settle_margin_bpm,
        smooth_window_seconds: config.smooth_window_seconds,
    };
    let d = physio_algo::nap::evaluate(&grav, &hr_samples, resting_hr, &cfg);
    NapDecisionInfo {
        verdict: match d.verdict {
            physio_algo::nap::NapVerdict::Nap => NapVerdictInfo::Nap,
            physio_algo::nap::NapVerdict::None => NapVerdictInfo::None,
            physio_algo::nap::NapVerdict::Inconclusive => NapVerdictInfo::Inconclusive,
        },
        candidate: d.candidate.map(|c| NapCandidateInfo {
            start: c.start,
            end: c.end,
            mean_hr: c.mean_hr,
            confidence: c.confidence,
        }),
    }
}

/// One night fed into the debt ledger: local `day` key + `slept_min` (None = no data, skipped not zeroed).
#[derive(uniffi::Record)]
pub struct DebtNightInput {
    pub day: String,
    pub slept_min: Option<f64>,
}

/// One night's contribution to the ledger.
#[derive(uniffi::Record)]
pub struct DebtNightInfo {
    pub day: String,
    pub slept_min: f64,
    pub delta_min: f64,
}

/// The rolling debt ledger over the capped trailing window of nights with data.
#[derive(uniffi::Record)]
pub struct DebtLedgerInfo {
    pub balance_min: f64,
    pub nights: Vec<DebtNightInfo>,
    pub need_min: f64,
}

/// Rolling sleep-debt ledger: Σ(slept − need) over the last `window` (default 14) nights with data.
/// `need_hours` defaults to 8 h. Nights with no sleep are skipped, never zero-filled.
#[uniffi::export]
pub fn sleep_debt_ledger(series: Vec<DebtNightInput>, need_hours: Option<f64>, window: Option<u32>) -> DebtLedgerInfo {
    let s: Vec<(String, Option<f64>)> = series.into_iter().map(|n| (n.day, n.slept_min)).collect();
    let l = sleep_debt::ledger(&s, need_hours, window.map(|w| w as usize));
    DebtLedgerInfo {
        balance_min: l.balance_min,
        nights: l
            .nights
            .into_iter()
            .map(|n| DebtNightInfo { day: n.day, slept_min: n.slept_min, delta_min: n.delta_min })
            .collect(),
        need_min: l.need_min,
    }
}

// ── Daily stress (autonomic, RHR + HRV vs baseline) ────────────────────────

/// Sleep Regularity Index (-100..100) from asleep spans and the wear windows they sit in. Non-wear is
/// UNKNOWN, never awake. `None` below the day-pair and coverage gates.
#[uniffi::export]
pub fn sleep_regularity_index(
    first_local_midnight: i64,
    days: u32,
    asleep: Vec<TimeSpan>,
    covered: Vec<TimeSpan>,
) -> Option<f64> {
    let to_pairs = |v: Vec<TimeSpan>| -> Vec<(i64, i64)> {
        v.into_iter().map(|s| (s.start, s.end)).collect()
    };
    let grid = sleep_regularity::epoch_grid(
        first_local_midnight, days as usize, &to_pairs(asleep), &to_pairs(covered),
    );
    sleep_regularity::sleep_regularity_index(&grid)
}

/// Personal sleep need (hours) = mean of recent nightly asleep hours, floored at 7.5. For the Rest
/// score's sleep-need input.
#[uniffi::export]
pub fn personal_sleep_need_hours(recent_asleep_hours: Vec<f64>) -> f64 {
    rest::personal_sleep_need_hours(&recent_asleep_hours)
}
