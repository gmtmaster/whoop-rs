//! V2 (cardiorespiratory) sleep stager — the 5.0/MG default. Per-night z-scored HR / HR-variability /
//! motion emissions, a deep gate on the 11-min HR-flatness percentile, a soft sleep-cycle prior, a
//! self-calibrating jerk wake gate, an R-R RSA respiration term, and Viterbi transition smoothing. The
//! per-epoch coefficients are fixed a-priori from sleep physiology + population base rates.
//!
//! Mirrors the shipped app's V2 recipe. Absent signal scores the neutral centre, so a sparse channel never
//! blocks a stage. Outputs are wellness estimates, never medical advice.

use std::collections::HashMap;
use std::f64::consts::PI;

use super::common::{flatten_rr, median, ZScore};
use super::params::Params;
use super::input::{AccelSample, HrSample, SleepInput};
use super::{SleepStage, StageSegment};

// Stage indices for the emission/transition arrays: order [deep, rem, light, awake].
const DEEP: usize = 0;
const REM: usize = 1;
const LIGHT: usize = 2;
const AWAKE: usize = 3;

/// Deep-eligibility HR-flatness percentile gate of the shipped recipe.
pub const DEEP_GATE_THRESH: f64 = Params::SHIPPED.deep_gate_thresh;

/// One 30 s epoch's recipe features. `None` means "no measurement" (scored neutral).
struct Epoch {
    start: i64,
    hr: Option<f64>,
    hr_var: Option<f64>,
    hr_flat11: Option<f64>,
    move_frac: Option<f64>,
    jerk_max: f64,
    resp_reg: Option<f64>,
    clock: f64,
    jerk_scale: f64,
}

/// Stage a detected in-bed span with the shipped V2 recipe; segments tile `[start, end]`.
pub fn stage(input: &SleepInput) -> Vec<StageSegment> {
    stage_with(input, &Params::SHIPPED)
}

/// One night's epoch features, already extracted. Only `jerk_move_mult` changes these, so a sweep over
/// the emission/prior/transition axes can extract once and re-label many times.
pub struct Prepared {
    feats: Vec<Epoch>,
    start: i64,
    end: i64,
}

/// Extract a night's epoch features under `p`. Pair with [`stage_prepared`].
pub fn prepare(input: &SleepInput, p: &Params) -> Prepared {
    let mut grav = input.accel.clone();
    grav.sort_by_key(|g| g.ts);
    let mut hr = input.hr.clone();
    hr.sort_by_key(|h| h.ts);
    let mut rr = flatten_rr(&input.rr);
    rr.sort_by_key(|a| a.0);
    let feats = features(input.start, input.end, &grav, &hr, &rr, p);
    Prepared { feats, start: input.start, end: input.end }
}

/// Label already-extracted features under `p`; segments tile the prepared span.
pub fn stage_prepared(prep: &Prepared, p: &Params) -> Vec<StageSegment> {
    if prep.feats.is_empty() {
        return vec![StageSegment { start: prep.start, end: prep.end, stage: SleepStage::Light }];
    }
    let labels = stage_epochs(&prep.feats, p);
    segments_from(&prep.feats, &labels, prep.start, prep.end)
}

/// Stage with an explicit recipe. The tuning path; `stage` is what everything else calls.
pub fn stage_with(input: &SleepInput, p: &Params) -> Vec<StageSegment> {
    let start = input.start;
    let end = input.end;

    let mut grav = input.accel.clone();
    grav.sort_by_key(|g| g.ts);
    let mut hr = input.hr.clone();
    hr.sort_by_key(|h| h.ts);
    let mut rr = flatten_rr(&input.rr);
    rr.sort_by_key(|a| a.0);

    let feats = features(start, end, &grav, &hr, &rr, p);
    if feats.is_empty() {
        return vec![StageSegment { start, end, stage: SleepStage::Light }];
    }
    let labels = stage_epochs(&feats, p);
    segments_from(&feats, &labels, start, end)
}

/// Tile `[start, end]` from one label per epoch, merging equal-stage neighbours.
fn segments_from(feats: &[Epoch], labels: &[SleepStage], start: i64, end: i64) -> Vec<StageSegment> {
    let mut segments: Vec<StageSegment> = Vec::new();
    let n = feats.len();
    for (i, f) in feats.iter().enumerate() {
        let stage = labels[i];
        let seg_start = if i == 0 { start } else { f.start };
        let seg_end = if i == n - 1 { end } else { feats[i + 1].start };
        match segments.last_mut() {
            Some(last) if last.stage == stage => last.end = seg_end,
            _ => segments.push(StageSegment { start: seg_start, end: seg_end, stage }),
        }
    }
    segments
}

fn features(
    start: i64,
    end: i64,
    grav: &[AccelSample],
    hr: &[HrSample],
    rr: &[(i64, f64)],
    p: &Params,
) -> Vec<Epoch> {
    if end <= start {
        return Vec::new();
    }
    let span = (end - start).max(1) as f64;

    // Per-second HR mean.
    let mut hr_sum: HashMap<i64, f64> = HashMap::new();
    let mut hr_cnt: HashMap<i64, i64> = HashMap::new();
    for s in hr {
        *hr_sum.entry(s.ts).or_insert(0.0) += s.bpm as f64;
        *hr_cnt.entry(s.ts).or_insert(0) += 1;
    }
    let mut sec_hr: HashMap<i64, f64> = HashMap::with_capacity(hr_sum.len());
    for (k, v) in &hr_sum {
        sec_hr.insert(*k, v / hr_cnt[k] as f64);
    }

    // Per-second gravity mean (x, y, z).
    let mut gx: HashMap<i64, f64> = HashMap::new();
    let mut gy: HashMap<i64, f64> = HashMap::new();
    let mut gz: HashMap<i64, f64> = HashMap::new();
    let mut gc: HashMap<i64, i64> = HashMap::new();
    for g in grav {
        *gx.entry(g.ts).or_insert(0.0) += g.x;
        *gy.entry(g.ts).or_insert(0.0) += g.y;
        *gz.entry(g.ts).or_insert(0.0) += g.z;
        *gc.entry(g.ts).or_insert(0) += 1;
    }
    let mut sec_g: HashMap<i64, (f64, f64, f64)> = HashMap::with_capacity(gc.len());
    for (k, c) in &gc {
        let d = *c as f64;
        sec_g.insert(*k, (gx[k] / d, gy[k] / d, gz[k] / d));
    }

    // R-R bucketed by second (for the RSA window).
    let mut rr_by: HashMap<i64, Vec<f64>> = HashMap::new();
    for (ts, ms) in rr {
        rr_by.entry(*ts).or_default().push(*ms);
    }

    // Prefix sums over the integer-second HR axis → O(1) windowed population std.
    let (axis_lo, sum_px, sum_sq_px, cnt_px) = if sec_hr.is_empty() {
        (0i64, vec![0.0], vec![0.0], vec![0i64])
    } else {
        let lo = *sec_hr.keys().min().unwrap();
        let hi = *sec_hr.keys().max().unwrap();
        let len = (hi - lo + 1) as usize;
        let mut sp = vec![0.0; len + 1];
        let mut sqp = vec![0.0; len + 1];
        let mut cp = vec![0i64; len + 1];
        for i in 0..len {
            let v = sec_hr.get(&(lo + i as i64)).copied();
            sp[i + 1] = sp[i] + v.unwrap_or(0.0);
            sqp[i + 1] = sqp[i] + v.map(|x| x * x).unwrap_or(0.0);
            cp[i + 1] = cp[i] + if v.is_some() { 1 } else { 0 };
        }
        (lo, sp, sqp, cp)
    };
    let axis_hi_excl = axis_lo + (cnt_px.len() as i64 - 1);

    let std_of_seconds = |lo: i64, hi: i64| -> Option<f64> {
        if cnt_px.len() <= 1 {
            return None;
        }
        let q_lo = lo.clamp(axis_lo, axis_hi_excl);
        let q_hi = hi.clamp(axis_lo, axis_hi_excl);
        if q_hi <= q_lo {
            return None;
        }
        let a = (q_lo - axis_lo) as usize;
        let b = (q_hi - axis_lo) as usize;
        let n = cnt_px[b] - cnt_px[a];
        if n < 2 {
            return None;
        }
        let sum = sum_px[b] - sum_px[a];
        let sum_sq = sum_sq_px[b] - sum_sq_px[a];
        let mean = sum / n as f64;
        let variance = sum_sq / n as f64 - mean * mean;
        Some(if variance < 0.0 { 0.0 } else { variance }.sqrt())
    };

    // PASS 1 — per-epoch quantities except moveFrac; pool every per-second jerk.
    struct Raw {
        start: i64,
        hr: Option<f64>,
        hr_var: Option<f64>,
        hr_flat11: Option<f64>,
        jerks: Vec<f64>,
        gap_sec: i64,
        jerk_max: f64,
        resp_reg: Option<f64>,
        clock: f64,
    }
    let mut raws: Vec<Raw> = Vec::new();
    let mut all_jerks: Vec<f64> = Vec::new();
    let first_e = ((start + 29) / 30) * 30;
    let mut e = first_e;
    while e < end {
        let mut hrs: Vec<f64> = Vec::new();
        let mut gseq: Vec<(f64, f64, f64)> = Vec::new();
        let mut s = e;
        while s < e + 30 {
            if let Some(v) = sec_hr.get(&s) {
                hrs.push(*v);
            }
            if let Some(v) = sec_g.get(&s) {
                gseq.push(*v);
            }
            s += 1;
        }
        if hrs.is_empty() && gseq.is_empty() {
            e += 30;
            continue;
        }

        let mut jerks: Vec<f64> = Vec::new();
        let mut i = 1usize;
        while i < gseq.len().max(1) {
            let a = gseq[i - 1];
            let b = gseq[i];
            let dx = a.0 - b.0;
            let dy = a.1 - b.1;
            let dz = a.2 - b.2;
            jerks.push((dx * dx + dy * dy + dz * dz).sqrt());
            i += 1;
        }
        all_jerks.extend_from_slice(&jerks);
        let jerk_max = jerks.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let jerk_max = if jerks.is_empty() { 0.0 } else { jerk_max };

        let hr_mean = if hrs.is_empty() { None } else { Some(hrs.iter().sum::<f64>() / hrs.len() as f64) };
        let hr_var = std_of_seconds(e - 150, e + 30 + 150);
        let hr_flat11 = std_of_seconds(e - 330, e + 30 + 360);

        let mut beats: Vec<(f64, f64)> = Vec::new();
        let mut bs = e - 90;
        while bs < e + 120 {
            if let Some(vs) = rr_by.get(&bs) {
                for v in vs {
                    beats.push((bs as f64, v.clamp(300.0, 2000.0)));
                }
            }
            bs += 1;
        }
        beats.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then(a.1.partial_cmp(&b.1).unwrap()));
        let resp_reg = resp_regularity(&beats);

        raws.push(Raw {
            start: e,
            hr: hr_mean,
            hr_var,
            hr_flat11,
            gap_sec: (gseq.len() as i64 - 1).max(1),
            jerk_max,
            resp_reg,
            clock: (e + 15 - start) as f64 / span,
            jerks,
        });
        e += 30;
    }

    let jerk_scale = if all_jerks.is_empty() { 1e-6 } else { median(&all_jerks) };
    let move_thr = jerk_scale * p.jerk_move_mult;

    // PASS 2 — move fraction against the night-relative threshold.
    raws.into_iter()
        .map(|r| {
            // No gravity in the epoch = no motion evidence; absent, not "perfectly still".
            let observed = !r.jerks.is_empty();
            let moves = r.jerks.iter().filter(|&&j| j > move_thr).count();
            Epoch {
                start: r.start,
                hr: r.hr,
                hr_var: r.hr_var,
                hr_flat11: r.hr_flat11,
                move_frac: observed.then(|| moves as f64 / r.gap_sec as f64),
                jerk_max: r.jerk_max,
                resp_reg: r.resp_reg,
                clock: r.clock,
                jerk_scale,
            }
        })
        .collect()
}

/// RSA respiration regularity: tachogram → 4 Hz resample → detrend → band-limited DFT peak/sum over the
/// 0.15–0.40 Hz band. Higher = more regular breathing. `None` when there are too few beats.
fn resp_regularity(beats: &[(f64, f64)]) -> Option<f64> {
    if beats.len() < 12 {
        return None;
    }
    let t0 = beats[0].0;
    let t_n = beats[beats.len() - 1].0;
    if t_n <= t0 {
        return None;
    }
    let n = ((t_n - t0) / 0.25 - 1e-9).ceil() as i64;
    if n < 16 {
        return None;
    }
    let n = n as usize;

    let mut y = vec![0.0f64; n];
    let mut seg = 0usize;
    for (i, yi) in y.iter_mut().enumerate() {
        let t = t0 + 0.25 * i as f64;
        while seg < beats.len() - 2 && beats[seg + 1].0 < t {
            seg += 1;
        }
        let ta = beats[seg].0;
        let tb = beats[seg + 1].0;
        let va = beats[seg].1;
        let vb = beats[seg + 1].1;
        *yi = if tb <= ta { va } else { va + ((t - ta) / (tb - ta)).clamp(0.0, 1.0) * (vb - va) };
    }
    let mean = y.iter().sum::<f64>() / n as f64;
    for v in y.iter_mut() {
        *v -= mean;
    }

    let k_lo = (0.15 * 0.25 * n as f64).ceil() as i64;
    let k_hi = (0.40 * 0.25 * n as f64).floor() as i64;
    if k_hi < k_lo || k_lo < 0 {
        return None;
    }
    let mut max_p = 0.0;
    let mut sum_p = 0.0;
    for k in k_lo..=k_hi {
        let mut re = 0.0;
        let mut im = 0.0;
        let w = -2.0 * PI * k as f64 / n as f64;
        for (j, &yj) in y.iter().enumerate() {
            let a = w * j as f64;
            re += yj * a.cos();
            im += yj * a.sin();
        }
        let p = re * re + im * im;
        sum_p += p;
        if p > max_p {
            max_p = p;
        }
    }
    if sum_p == 0.0 {
        None
    } else {
        Some(max_p / sum_p)
    }
}

/// Soft sleep-cycle prior: deep concentrated early (decays), REM suppressed in the first ~12% then rising.
fn cycle_prior(c: f64, p: &Params) -> [f64; 4] {
    let mut pr = [0.0; 4];
    pr[DEEP] = p.cycle_deep_scale * (1.0 - c / p.cycle_deep_decay).max(0.0);
    pr[REM] = p.cycle_rem_scale * c - if c < p.cycle_rem_early_frac { p.cycle_rem_early_penalty } else { 0.0 };
    pr
}

/// Motion-quiescent: movement was OBSERVED and was none, with peak jerk at/below the night floor × the
/// gate multiplier. An epoch with no gravity cannot be quiescent — absence is not stillness.
fn motion_quiescent(f: &Epoch, p: &Params) -> bool {
    f.move_frac.is_some_and(|m| m <= 0.0) && f.jerk_max <= f.jerk_scale * p.jerk_gate_mult
}

fn dz(z: f64, deadzone: f64) -> f64 {
    if deadzone <= 0.0 {
        z
    } else if z > deadzone {
        z - deadzone
    } else if z < -deadzone {
        z + deadzone
    } else {
        0.0
    }
}

/// Viterbi most-likely path over the per-epoch log-emissions with the sticky transition matrix and a
/// uniform start. Ties resolve to the earlier stage (deep < rem < light < awake).
#[allow(clippy::needless_range_loop)]
fn viterbi(em_seq: &[[f64; 4]], transition: &[[f64; 4]; 4]) -> Vec<SleepStage> {
    if em_seq.is_empty() {
        return Vec::new();
    }
    let mut log_t = [[0.0f64; 4]; 4];
    for (fi, row) in transition.iter().enumerate() {
        for (ti, &v) in row.iter().enumerate() {
            log_t[fi][ti] = v.max(1e-9).ln();
        }
    }
    let mut v = em_seq[0];
    let mut back: Vec<[usize; 4]> = Vec::new();
    for em in &em_seq[1..] {
        let mut new_v = [0.0f64; 4];
        let mut bp = [0usize; 4];
        for s in 0..4 {
            let mut best_prev = 0usize;
            let mut best_val = v[0] + log_t[0][s];
            for p in 1..4 {
                let value = v[p] + log_t[p][s];
                if value > best_val {
                    best_val = value;
                    best_prev = p;
                }
            }
            new_v[s] = best_val + em[s];
            bp[s] = best_prev;
        }
        v = new_v;
        back.push(bp);
    }
    let mut last = 0usize;
    let mut last_v = v[0];
    for s in 1..4 {
        if v[s] > last_v {
            last_v = v[s];
            last = s;
        }
    }
    let mut path = vec![last];
    for bp in back.iter().rev() {
        last = bp[last];
        path.push(last);
    }
    path.reverse();
    path.into_iter().map(idx_to_stage).collect()
}

fn idx_to_stage(i: usize) -> SleepStage {
    match i {
        DEEP => SleepStage::Deep,
        REM => SleepStage::Rem,
        LIGHT => SleepStage::Light,
        _ => SleepStage::Wake,
    }
}

/// Run the full recipe over a night's epochs and return one stage label per epoch. All normalisation
/// (z-scores, the HR-flatness percentile) is within the night.
fn stage_epochs(feats: &[Epoch], p: &Params) -> Vec<SleepStage> {
    if feats.is_empty() {
        return Vec::new();
    }
    let blp = p.base_log_prior();
    let zhr = ZScore::build(&feats.iter().map(|f| f.hr).collect::<Vec<_>>());
    let zhv = ZScore::build(&feats.iter().map(|f| f.hr_var).collect::<Vec<_>>());
    let zmv = ZScore::build(&feats.iter().map(|f| f.move_frac).collect::<Vec<_>>());
    let zrg = ZScore::build(&feats.iter().map(|f| f.resp_reg).collect::<Vec<_>>());

    let mut fsorted: Vec<f64> = feats.iter().filter_map(|f| f.hr_flat11).collect();
    fsorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let fpct = |value: Option<f64>| -> f64 {
        match value {
            Some(v) if !fsorted.is_empty() => {
                // bisect_right / n
                let mut lo = 0usize;
                let mut hi = fsorted.len();
                while lo < hi {
                    let mid = (lo + hi) / 2;
                    if fsorted[mid] <= v {
                        lo = mid + 1;
                    } else {
                        hi = mid;
                    }
                }
                lo as f64 / fsorted.len() as f64
            }
            _ => 0.5,
        }
    };

    let mut seq: Vec<[f64; 4]> = Vec::with_capacity(feats.len());
    for f in feats {
        let zhrv = zhr.apply(f.hr);
        let zhvv = zhv.apply(f.hr_var);
        let zmvv = zmv.apply(f.move_frac);
        let gate = p.deep_gate_slope * (fpct(f.hr_flat11) - p.deep_gate_thresh).max(0.0);
        let awake_cardiac0 = p.awake_hrv * dz(zhvv, p.awake_deadzone) + p.awake_hr * dz(zhrv, p.awake_deadzone);
        let awake_cardiac = if motion_quiescent(f, p) { awake_cardiac0.min(0.0) } else { awake_cardiac0 };

        let mut em = [0.0f64; 4];
        em[DEEP] = p.deep_hrv * zhvv + p.deep_hr * zhrv + p.deep_motion * zmvv - gate + blp[DEEP];
        em[REM] = p.rem_hrv * zhvv + p.rem_motion * zmvv + p.rem_hr * zhrv + blp[REM];
        em[LIGHT] = blp[LIGHT];
        em[AWAKE] = p.awake_motion * zmvv + awake_cardiac + blp[AWAKE];

        let pr = cycle_prior(f.clock, p);
        for (s, p) in pr.iter().enumerate() {
            em[s] += p;
        }
        if f.jerk_max > f.jerk_scale * p.jerk_gate_mult {
            em[AWAKE] += p.motion_gate_boost;
        }
        if let Some(rg) = f.resp_reg {
            let z = zrg.apply(Some(rg));
            em[DEEP] += p.resp_weight * z;
            em[REM] -= p.resp_weight * z;
        }
        seq.push(em);
    }
    viterbi(&seq, &p.transition)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sleep::input::RrRun;

    /// One minute of HR with gravity over only the first half: the epochs that saw the accelerometer
    /// carry a motion reading, the epochs that did not carry `None`.
    fn half_blind_epochs() -> Vec<Epoch> {
        let start = 1_749_513_600i64;
        let hr: Vec<HrSample> = (0..60).map(|i| HrSample { ts: start + i, bpm: 55 }).collect();
        let accel: Vec<AccelSample> =
            (0..30).map(|i| AccelSample { ts: start + i, x: 0.0, y: 0.0, z: 1.0 }).collect();
        let input = SleepInput { start, end: start + 60, hr, rr: Vec::<RrRun>::new(), accel, resp: Vec::new() };
        let mut grav = input.accel.clone();
        grav.sort_by_key(|g| g.ts);
        features(input.start, input.end, &grav, &input.hr, &[], &Params::SHIPPED)
    }

    #[test]
    fn an_epoch_without_gravity_reports_no_motion_reading() {
        let feats = half_blind_epochs();
        assert_eq!(feats.len(), 2, "two 30 s epochs");
        assert_eq!(feats[0].move_frac, Some(0.0), "gravity present and still -> a measured zero");
        assert_eq!(feats[1].move_frac, None, "no gravity -> no reading, not a measured zero");
    }

    #[test]
    fn an_epoch_without_gravity_is_not_motion_quiescent() {
        // The quiescent gate clamps the awake-cardiac term, so asserting it on absent motion would let a
        // dropout suppress wake. Only an epoch that actually SAW a still wrist may be quiescent.
        let feats = half_blind_epochs();
        assert!(motion_quiescent(&feats[0], &Params::SHIPPED), "an observed still wrist is quiescent");
        assert!(!motion_quiescent(&feats[1], &Params::SHIPPED), "an absent accelerometer cannot assert stillness");
    }

    #[test]
    fn an_absent_motion_reading_z_scores_neutral() {
        // The z-scorer must place "no reading" at the centre, not at the bottom of the night's motion
        // distribution, so a dropout never reads as the stillest stretch of the night.
        let z = ZScore::build(&[Some(0.0), Some(4.0), Some(8.0), None]);
        assert_eq!(z.apply(None), 0.0);
        assert!(z.apply(Some(0.0)) < 0.0, "a measured zero sits below the night mean");
        assert!(z.apply(None) > z.apply(Some(0.0)), "absence must not out-still a measured zero");
    }
}
