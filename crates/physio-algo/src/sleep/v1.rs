//! V1 (Cole-Kripke) sleep stager — the 4.0 path. Actigraphy sleep/wake from summed gravity jerk, then a
//! percentile-band cardiac classifier (low HR + high parasympathetic tone → deep; high HR + high
//! HR-variability → REM via the no-respiration fallback), median smoothing, physiology re-imposition, and a
//! sub-3-min fragment merge.
//!
//! Mirrors the shipped app's V1 staging path (detection lives elsewhere; this consumes an already-detected
//! `[start, end]` span). The respiration channel is optional; with none present the no-resp path runs (REM
//! only via still + high HR + high HR-variability). Outputs are wellness estimates, never medical advice.

use super::common::{
    convolve_reflect, find_peaks, gaussian_kernel, median, percentile, population_std, range_filter,
    rmssd_raw, stage_depth_rank,
};
use super::input::{AccelSample, HrSample, SleepInput};
use super::{SleepStage, StageSegment};

const EPOCH_S: f64 = 30.0;
const CK_WEIGHTS: [f64; 7] = [106.0, 54.0, 58.0, 76.0, 230.0, 74.0, 67.0];
const CK_COUNT_DIVISOR: f64 = 100.0;
const CK_COUNT_CLIP: f64 = 300.0;
const CK_SCALE: f64 = 0.001;
const CK_BACK: i64 = 4;
const MOVE_DELTA_THRESHOLD_G: f64 = 0.01;
const HR_DOG_SIGMA1_S: f64 = 120.0;
const HR_DOG_SIGMA2_S: f64 = 600.0;
const STAGE_HR_LOW_PCT: f64 = 25.0;
const STAGE_HR_HIGH_PCT: f64 = 70.0;
const STAGE_HRV_HIGH_PCT: f64 = 70.0;
const STAGE_HR_VAR_HIGH_PCT: f64 = 65.0;
const STAGE_WAKE_MOVE_FRAC: f64 = 0.15;
const STAGE_STILL_MOVE_FRAC: f64 = 0.10;
const SMOOTH_EPOCHS: usize = 5;
const NO_REM_AFTER_ONSET_MIN: f64 = 15.0;
const DEEP_FIRST_FRACTION: f64 = 1.0 / 3.0;
const CARDIAC_SPARSE_EPOCH_FRAC: f64 = 0.5;
const ONSET_PERSIST_EPOCHS: i64 = 3;
const STAGE_RRV_HIGH_PCT: f64 = 65.0;
const STAGE_RRV_LOW_PCT: f64 = 50.0;
const FEATURE_WINDOW_S: f64 = 5.0 * 60.0;
const FRAGMENT_MERGE_MIN: f64 = 3.0;

struct EpochGrid {
    n: usize,
    edges: Vec<f64>,
    counts: Vec<f64>,
    move_frac: Vec<f64>,
    hr: Vec<f64>,
    rr: Vec<Vec<f64>>,
    resp: Vec<Vec<f64>>,
}

struct EpochFeatures {
    hr: f64,
    hr_var: f64,
    rmssd: f64,
    rrv: f64,
    move_frac: f64,
    ck_sleep: bool,
    clock: f64,
}

/// Stage a detected in-bed span with the V1 recipe; segments tile `[start, end]`.
pub fn stage(input: &SleepInput) -> Vec<StageSegment> {
    let start = input.start;
    let end = input.end;

    let g_seg: Vec<&AccelSample> = input.accel.iter().filter(|g| g.ts >= start && g.ts <= end).collect();
    if g_seg.len() < 2 {
        return vec![StageSegment { start, end, stage: SleepStage::Light }];
    }
    let g_deltas = gravity_deltas(&g_seg);
    let g_times: Vec<f64> = g_seg.iter().map(|g| g.ts as f64).collect();

    let hr_seg: Vec<&HrSample> = input.hr.iter().filter(|h| h.ts >= start && h.ts <= end).collect();
    let mut rr_seg: Vec<(f64, f64)> = Vec::new();
    for run in &input.rr {
        if run.ts >= start && run.ts <= end {
            for &ms in &run.intervals {
                rr_seg.push((run.ts as f64, ms as f64));
            }
        }
    }
    let resp_seg: Vec<(f64, f64)> = input
        .resp
        .iter()
        .filter(|r| r.ts >= start && r.ts <= end)
        .map(|r| (r.ts as f64, r.raw as f64))
        .collect();

    let grid = build_epoch_grid(start as f64, end as f64, &g_times, &g_deltas, &hr_seg, &rr_seg, &resp_seg);
    if grid.n == 0 {
        return vec![StageSegment { start, end, stage: SleepStage::Light }];
    }

    let rescaled: Vec<f64> = grid.counts.iter().map(|c| (c / CK_COUNT_DIVISOR).min(CK_COUNT_CLIP)).collect();
    let ck_flags = cole_kripke(&rescaled);
    let (onset, final_wake) = onset_and_final_wake(&ck_flags);
    let dog = dog_hr_variability(&grid.hr);
    let feats = extract_features(&grid, &ck_flags, &dog, onset, final_wake);

    let mut labels = classify_epochs(&feats);
    labels = smooth_labels(&labels);
    labels = reimpose_physiology(&labels, &feats, onset, final_wake);
    labels = merge_fragments(&labels);

    for (i, lab) in labels.iter_mut().enumerate() {
        let i = i as i64;
        if i < onset || i > final_wake {
            *lab = SleepStage::Wake;
        }
    }

    let mut segments: Vec<StageSegment> = Vec::new();
    for (i, &stage) in labels.iter().enumerate() {
        let seg_start = grid.edges[i].round() as i64;
        let seg_end = grid.edges[i + 1].round() as i64;
        match segments.last_mut() {
            Some(last) if last.stage == stage => last.end = seg_end,
            _ => segments.push(StageSegment { start: seg_start, end: seg_end, stage }),
        }
    }
    if let Some(last) = segments.last_mut() {
        last.end = end;
    }
    segments
}

fn gravity_deltas(grav: &[&AccelSample]) -> Vec<f64> {
    let mut deltas = Vec::with_capacity(grav.len());
    for i in 0..grav.len() {
        if i == 0 {
            deltas.push(0.0);
        } else {
            let p = grav[i - 1];
            let r = grav[i];
            let dx = p.x - r.x;
            let dy = p.y - r.y;
            let dz = p.z - r.z;
            deltas.push((dx * dx + dy * dy + dz * dz).sqrt());
        }
    }
    deltas
}

fn build_epoch_grid(
    start: f64,
    end: f64,
    grav_times: &[f64],
    grav_deltas: &[f64],
    hr: &[&HrSample],
    rr: &[(f64, f64)],
    resp: &[(f64, f64)],
) -> EpochGrid {
    if end <= start {
        return EpochGrid {
            n: 0,
            edges: vec![start],
            counts: vec![],
            move_frac: vec![],
            hr: vec![],
            rr: vec![],
            resp: vec![],
        };
    }
    let n = (((end - start) / EPOCH_S).ceil() as usize).max(1);
    let mut edges = vec![0.0f64; n + 1];
    for (i, e) in edges.iter_mut().enumerate() {
        *e = start + i as f64 * EPOCH_S;
    }
    edges[n] = edges[n].max(end);

    let mut counts = vec![0.0f64; n];
    let mut move_n = vec![0i64; n];
    let mut grav_n = vec![0i64; n];
    let mut hr_sum = vec![0.0f64; n];
    let mut hr_cnt = vec![0i64; n];
    let mut rr_buckets: Vec<Vec<f64>> = vec![Vec::new(); n];
    let mut resp_buckets: Vec<Vec<f64>> = vec![Vec::new(); n];

    let idx = |ts: f64| -> Option<usize> {
        if ts < start || ts >= end {
            if ts == end {
                return Some(n - 1);
            }
            return None;
        }
        Some((((ts - start) / EPOCH_S) as usize).min(n - 1))
    };

    for k in 0..grav_times.len() {
        if let Some(i) = idx(grav_times[k]) {
            counts[i] += grav_deltas[k];
            grav_n[i] += 1;
            if grav_deltas[k] >= MOVE_DELTA_THRESHOLD_G {
                move_n[i] += 1;
            }
        }
    }
    for r in hr {
        if let Some(i) = idx(r.ts as f64) {
            hr_sum[i] += r.bpm as f64;
            hr_cnt[i] += 1;
        }
    }
    for (ts, ms) in rr {
        if let Some(i) = idx(*ts) {
            rr_buckets[i].push(*ms);
        }
    }
    for (ts, raw) in resp {
        if let Some(i) = idx(*ts) {
            resp_buckets[i].push(*raw);
        }
    }

    let hr_mean: Vec<f64> = (0..n).map(|i| if hr_cnt[i] > 0 { hr_sum[i] / hr_cnt[i] as f64 } else { f64::NAN }).collect();
    let move_frac: Vec<f64> = (0..n).map(|i| if grav_n[i] > 0 { move_n[i] as f64 / grav_n[i] as f64 } else { 1.0 }).collect();

    EpochGrid { n, edges, counts, move_frac, hr: hr_mean, rr: rr_buckets, resp: resp_buckets }
}

/// Respiratory rate (breaths/min) + RRV (s) from a raw resp window: detrend by the window mean, peak-pick,
/// keep breath intervals in the 1.5–12 s band. `(NaN, NaN)` when the signal is too short or flat.
fn resp_rate_and_rrv(resp_raw: &[f64]) -> (f64, f64) {
    let nan = f64::NAN;
    if resp_raw.len() < 8 {
        return (nan, nan);
    }
    let mean = resp_raw.iter().sum::<f64>() / resp_raw.len() as f64;
    let x: Vec<f64> = resp_raw.iter().map(|v| v - mean).collect();
    if x.iter().all(|v| v.abs() < 1e-12) {
        return (nan, nan);
    }
    if population_std(&x) <= 0.0 {
        return (nan, nan);
    }
    // dt = 1 s → minimum peak spacing of 2 samples.
    let peaks = find_peaks(&x, 2, 0.0);
    if peaks.len() < 3 {
        return (nan, nan);
    }
    let mut intervals: Vec<f64> = Vec::new();
    for i in 1..peaks.len() {
        let iv = (peaks[i] - peaks[i - 1]) as f64;
        if (1.5..=12.0).contains(&iv) {
            intervals.push(iv);
        }
    }
    if intervals.len() < 2 {
        return (nan, nan);
    }
    (60.0 / median(&intervals), population_std(&intervals))
}

fn cole_kripke(rescaled: &[f64]) -> Vec<bool> {
    let n = rescaled.len() as i64;
    let mut flags = Vec::with_capacity(rescaled.len());
    for i in 0..n {
        let mut si = 0.0;
        for (k, &w) in CK_WEIGHTS.iter().enumerate() {
            let j = i - CK_BACK + k as i64;
            let a = if j >= 0 && j < n { rescaled[j as usize] } else { 0.0 };
            si += w * a;
        }
        si *= CK_SCALE;
        flags.push(si < 1.0);
    }
    flags
}

fn onset_and_final_wake(ck_flags: &[bool]) -> (i64, i64) {
    let n = ck_flags.len() as i64;
    if n == 0 {
        return (0, 0);
    }
    let mut onset: Option<i64> = None;
    let mut run = 0i64;
    for (i, &s) in ck_flags.iter().enumerate() {
        run = if s { run + 1 } else { 0 };
        if run >= ONSET_PERSIST_EPOCHS {
            onset = Some(i as i64 - ONSET_PERSIST_EPOCHS + 1);
            break;
        }
    }
    let mut final_wake: Option<i64> = None;
    for i in (0..n).rev() {
        if ck_flags[i as usize] {
            final_wake = Some(i);
            break;
        }
    }
    let o = onset.unwrap_or(0);
    let mut f = final_wake.unwrap_or(n - 1);
    if f < o {
        f = n - 1;
    }
    (o, f)
}

fn dog_hr_variability(hr_per_epoch: &[f64]) -> Vec<f64> {
    let n = hr_per_epoch.len();
    if n == 0 {
        return Vec::new();
    }
    let mask: Vec<usize> = (0..n).filter(|&i| !hr_per_epoch[i].is_nan()).collect();
    if mask.is_empty() {
        return vec![0.0; n];
    }
    let first = mask[0];
    let last = *mask.last().unwrap();
    let mut filled = vec![0.0f64; n];
    for i in 0..n {
        if !hr_per_epoch[i].is_nan() {
            filled[i] = hr_per_epoch[i];
            continue;
        }
        if i <= first {
            filled[i] = hr_per_epoch[first];
            continue;
        }
        if i >= last {
            filled[i] = hr_per_epoch[last];
            continue;
        }
        let mut lo = first;
        let mut hi = last;
        for &m in &mask {
            if m <= i {
                lo = m;
            }
            if m >= i {
                hi = m;
                break;
            }
        }
        if hi == lo {
            filled[i] = hr_per_epoch[lo];
        } else {
            let frac = (i - lo) as f64 / (hi - lo) as f64;
            filled[i] = hr_per_epoch[lo] + frac * (hr_per_epoch[hi] - hr_per_epoch[lo]);
        }
    }
    let k1 = gaussian_kernel(HR_DOG_SIGMA1_S, EPOCH_S);
    let k2 = gaussian_kernel(HR_DOG_SIGMA2_S, EPOCH_S);
    let g1 = convolve_reflect(&filled, &k1);
    let g2 = convolve_reflect(&filled, &k2);
    (0..n).map(|i| g1[i] - g2[i]).collect()
}

fn extract_features(
    grid: &EpochGrid,
    ck_flags: &[bool],
    dog: &[f64],
    onset: i64,
    final_wake: i64,
) -> Vec<EpochFeatures> {
    let n = grid.n;
    let half_w = (FEATURE_WINDOW_S / EPOCH_S / 2.0).round() as usize;
    let span = (final_wake - onset).max(1) as f64;

    let mut feats = Vec::with_capacity(n);
    for i in 0..n {
        let lo = i.saturating_sub(half_w);
        let hi = (i + half_w + 1).min(n);

        let win_hr: Vec<f64> = (lo..hi).map(|j| grid.hr[j]).filter(|v| !v.is_nan()).collect();
        let hr_mean = if win_hr.is_empty() { f64::NAN } else { win_hr.iter().sum::<f64>() / win_hr.len() as f64 };

        let win_dog: Vec<f64> = (lo..hi).map(|j| if dog.is_empty() { 0.0 } else { dog[j] }).collect();
        let hr_var = if win_dog.len() >= 2 { population_std(&win_dog) } else { f64::NAN };

        let mut win_rr: Vec<f64> = Vec::new();
        for j in lo..hi {
            win_rr.extend_from_slice(&grid.rr[j]);
        }
        let filtered_rr = range_filter(&win_rr);
        let rmssd = if filtered_rr.len() >= 5 { rmssd_raw(&filtered_rr).unwrap_or(f64::NAN) } else { f64::NAN };

        let mut win_resp: Vec<f64> = Vec::new();
        for j in lo..hi {
            win_resp.extend_from_slice(&grid.resp[j]);
        }
        let (_resp_rate, rrv) = resp_rate_and_rrv(&win_resp);

        let clock = ((i as i64 - onset) as f64 / span).clamp(0.0, 1.0);
        feats.push(EpochFeatures {
            hr: hr_mean,
            hr_var,
            rmssd,
            rrv,
            move_frac: grid.move_frac[i],
            ck_sleep: if i < ck_flags.len() { ck_flags[i] } else { true },
            clock,
        });
    }
    feats
}

fn classify_epochs(features: &[EpochFeatures]) -> Vec<SleepStage> {
    let n = features.len();
    if n == 0 {
        return Vec::new();
    }
    let any_sleep = features.iter().any(|f| f.ck_sleep);
    let sleep_feats: Vec<&EpochFeatures> = if any_sleep {
        features.iter().filter(|f| f.ck_sleep).collect()
    } else {
        features.iter().collect()
    };
    let hr_lo = percentile(&sleep_feats.iter().map(|f| f.hr).collect::<Vec<_>>(), STAGE_HR_LOW_PCT);
    let hr_hi = percentile(&sleep_feats.iter().map(|f| f.hr).collect::<Vec<_>>(), STAGE_HR_HIGH_PCT);
    let rmssd_hi = percentile(&sleep_feats.iter().map(|f| f.rmssd).collect::<Vec<_>>(), STAGE_HRV_HIGH_PCT);
    let hrvar_hi = percentile(&sleep_feats.iter().map(|f| f.hr_var).collect::<Vec<_>>(), STAGE_HR_VAR_HIGH_PCT);
    let rrv_hi = percentile(&sleep_feats.iter().map(|f| f.rrv).collect::<Vec<_>>(), STAGE_RRV_HIGH_PCT);
    let rrv_lo = percentile(&sleep_feats.iter().map(|f| f.rrv).collect::<Vec<_>>(), STAGE_RRV_LOW_PCT);
    let cardiac_sparse = is_cardiac_sparse(&sleep_feats);

    features
        .iter()
        .map(|f| classify_one(f, hr_lo, hr_hi, rmssd_hi, hrvar_hi, rrv_hi, rrv_lo, cardiac_sparse))
        .collect()
}

fn is_cardiac_sparse(sleep_feats: &[&EpochFeatures]) -> bool {
    if sleep_feats.is_empty() {
        return false;
    }
    let sparse = sleep_feats.iter().filter(|f| !f.rmssd.is_finite()).count();
    sparse as f64 >= CARDIAC_SPARSE_EPOCH_FRAC * sleep_feats.len() as f64
}

#[allow(clippy::too_many_arguments)]
fn classify_one(
    f: &EpochFeatures,
    hr_lo: Option<f64>,
    hr_hi: Option<f64>,
    rmssd_hi: Option<f64>,
    hrvar_hi: Option<f64>,
    rrv_hi: Option<f64>,
    rrv_lo: Option<f64>,
    cardiac_sparse: bool,
) -> SleepStage {
    let has_hr = f.hr.is_finite();
    let hr_low = has_hr && hr_lo.is_some_and(|t| f.hr <= t);
    let hr_high = has_hr && hr_hi.is_some_and(|t| f.hr >= t);
    // A missing per-epoch RMSSD is treated as pro-deep (not deep-blocking), mirroring the sparse-R-R path.
    let parasymp_ok = !f.rmssd.is_finite() || rmssd_hi.is_some_and(|t| f.rmssd >= t);
    let hrvar_high = f.hr_var.is_finite() && hrvar_hi.is_some_and(|t| f.hr_var >= t);
    let cardiac_activated = hr_high || hrvar_high;
    let cardiac_activated_for_wake = if cardiac_sparse { hr_high } else { cardiac_activated };
    // Missing respiration (NaN RRV) is treated as regular (pro-deep); the no-resp REM fallback then requires
    // both cardiac signals.
    let rrv_irregular = f.rrv.is_finite() && rrv_hi.is_some_and(|t| f.rrv >= t);
    let rrv_regular = !f.rrv.is_finite() || rrv_lo.is_some_and(|t| f.rrv <= t);
    let still = f.move_frac <= STAGE_STILL_MOVE_FRAC;
    let moving = f.move_frac >= STAGE_WAKE_MOVE_FRAC;

    if moving && (cardiac_activated_for_wake || !has_hr) {
        return SleepStage::Wake;
    }
    if still && parasymp_ok && hr_low && rrv_regular {
        return SleepStage::Deep;
    }
    if still && cardiac_activated && rrv_irregular {
        return SleepStage::Rem;
    }
    if still && hr_high && hrvar_high && !f.rrv.is_finite() {
        return SleepStage::Rem;
    }
    SleepStage::Light
}

fn smooth_labels(labels: &[SleepStage]) -> Vec<SleepStage> {
    let n = labels.len();
    let mut w = SMOOTH_EPOCHS;
    if n == 0 || w <= 1 {
        return labels.to_vec();
    }
    if w.is_multiple_of(2) {
        w += 1;
    }
    let half = w / 2;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let lo = i.saturating_sub(half);
        let hi = (i + half + 1).min(n);
        // Insertion-order tally so a tie resolves to the first-seen winner (matching the Kotlin twin).
        let mut order: Vec<SleepStage> = Vec::new();
        let mut counts: Vec<i32> = Vec::new();
        for &s in &labels[lo..hi] {
            match order.iter().position(|&o| o == s) {
                Some(p) => counts[p] += 1,
                None => {
                    order.push(s);
                    counts.push(1);
                }
            }
        }
        let best = *counts.iter().max().unwrap();
        let winners: Vec<SleepStage> = order.iter().zip(&counts).filter(|(_, &c)| c == best).map(|(&s, _)| s).collect();
        out.push(if winners.contains(&labels[i]) { labels[i] } else { winners[0] });
    }
    out
}

fn reimpose_physiology(labels: &[SleepStage], features: &[EpochFeatures], onset: i64, final_wake: i64) -> Vec<SleepStage> {
    let mut out = labels.to_vec();
    let no_rem_epochs = (NO_REM_AFTER_ONSET_MIN * 60.0 / EPOCH_S).round() as i64;
    let has_early_deep = labels
        .iter()
        .enumerate()
        .any(|(i, &l)| l == SleepStage::Deep && features[i].clock <= DEEP_FIRST_FRACTION);
    for (i, f) in features.iter().enumerate() {
        let ii = i as i64;
        if ii < onset || ii > final_wake {
            continue;
        }
        if out[i] == SleepStage::Rem && (ii - onset) < no_rem_epochs {
            out[i] = SleepStage::Light;
        }
        if out[i] == SleepStage::Deep && f.clock > DEEP_FIRST_FRACTION && has_early_deep {
            out[i] = SleepStage::Light;
        }
    }
    out
}

fn merge_fragments(labels: &[SleepStage]) -> Vec<SleepStage> {
    let n = labels.len();
    let threshold = (FRAGMENT_MERGE_MIN * 60.0 / EPOCH_S).round() as i64;
    if n == 0 || threshold <= 1 {
        return labels.to_vec();
    }
    // Collapse to contiguous runs.
    let mut runs: Vec<(SleepStage, i64)> = Vec::new();
    for &s in labels {
        match runs.last_mut() {
            Some(last) if last.0 == s => last.1 += 1,
            _ => runs.push((s, 1)),
        }
    }
    if runs.len() < 2 {
        return labels.to_vec();
    }

    let mut merged: Vec<(SleepStage, i64)> = Vec::new();
    let mut i = 0usize;
    while i < runs.len() {
        let current = runs[i];
        if current.1 >= threshold {
            merged.push(current);
            i += 1;
            continue;
        }
        let has_prev = !merged.is_empty();
        let has_next = i + 1 < runs.len();
        if has_prev && has_next && merged.last().unwrap().0 == runs[i + 1].0 {
            let add = current.1 + runs[i + 1].1;
            merged.last_mut().unwrap().1 += add;
            i += 2;
        } else if has_prev && has_next {
            let prev = *merged.last().unwrap();
            let next = runs[i + 1];
            let winner = if prev.1 > next.1 {
                prev.0
            } else if next.1 > prev.1 {
                next.0
            } else if stage_depth_rank(prev.0) <= stage_depth_rank(next.0) {
                prev.0
            } else {
                next.0
            };
            if winner == prev.0 {
                merged.last_mut().unwrap().1 += current.1;
                i += 1;
            } else {
                runs[i + 1] = (next.0, next.1 + current.1);
                i += 1;
            }
        } else if has_next {
            runs[i + 1] = (runs[i + 1].0, runs[i + 1].1 + current.1);
            i += 1;
        } else if has_prev {
            merged.last_mut().unwrap().1 += current.1;
            i += 1;
        } else {
            merged.push(current);
            i += 1;
        }
    }

    let mut out = Vec::with_capacity(n);
    for (s, l) in merged {
        for _ in 0..l {
            out.push(s);
        }
    }
    out
}
