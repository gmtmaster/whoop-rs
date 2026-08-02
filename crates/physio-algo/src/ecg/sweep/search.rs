//! The search itself: prune, group, score, then fold consecutive windows into a verdict.
//!
//! Ordered by cost. Every reading rule is decoded and judged on two rate-free statistics first, because
//! that is one pass over the samples and it removes almost everything; only the distinct waveforms that
//! survive are carried into the per-rate stage, and only the ones whose implied heart rate is possible
//! reach the spectrum and the beat template. The counts at each stage are reported, not hidden — a sweep
//! that has to run live during a calibration is a design failure if it takes a minute per window.

use crate::ecg::score::score;
use crate::ecg::sweep::layout::{self, Layout};
use crate::ecg::sweep::ppg::{self, ppg_agreement};
use crate::ecg::sweep::prune::{cluster, layout_stats, same_time_base};
use crate::ecg::sweep::{Attribution, Candidate, SweepConfig, WindowReport};
use crate::ecg::score::{MAX_HR_BPM, MIN_HR_BPM};
use crate::ecg::{detect_pan_tompkins, usable_rate};
use crate::stats::median;

/// Search one window's bytes. This is the live unit: a caller rendering progress calls it per window and
/// draws the leaderboard it returns.
pub fn sweep_window(bytes: &[u8], ppg_beats_ms: &[f64], cfg: &SweepConfig) -> WindowReport {
    let layouts = layout::candidates();
    let mut decoded = 0usize;
    let mut kept: Vec<(Layout, Vec<f64>, f64)> = Vec::new();
    for l in &layouts {
        if l.sample_count(bytes.len()) < cfg.min_samples {
            continue;
        }
        decoded += 1;
        let x = l.decode(bytes);
        let Some(stats) = layout_stats(&x) else { continue };
        if stats.roughness > cfg.max_roughness || stats.kurtosis < cfg.min_kurtosis {
            continue;
        }
        kept.push((*l, x, stats.roughness));
    }

    let waves: Vec<&[f64]> = kept.iter().map(|(_, x, _)| x.as_slice()).collect();
    let classes = cluster(&waves, cfg.class_min_r);
    let n_classes = classes.iter().copied().max().map_or(0, |m| m + 1);
    let ppg_bpm = beat_rate_bpm(ppg_beats_ms);
    let ppg_usable = ppg_beats_ms.len() >= ppg::PPG_MIN_BEATS;

    let mut leaderboard: Vec<(Candidate, usize)> = Vec::new();
    let mut scored = 0usize;
    for class in 0..n_classes {
        let Some(rep) = representative(&kept, &classes, class) else { continue };
        let aliases = classes.iter().filter(|&&c| c == class).count() - 1;
        let (l, x, _) = &kept[rep];
        for &fs in &cfg.rates_hz {
            if !usable_rate(fs) {
                continue;
            }
            let peaks = detect_pan_tompkins(x, fs);
            let Some(hr) = beat_rate_bpm_samples(&peaks, fs) else { continue };
            if !(MIN_HR_BPM..=MAX_HR_BPM).contains(&hr) {
                continue;
            }
            if let Some(p) = ppg_bpm {
                if (hr - p).abs() / p > cfg.hr_prune_tolerance {
                    continue;
                }
            }
            scored += 1;
            let ecg = score(x, fs);
            let agreement = ppg_agreement(&peaks, x.len(), fs, ppg_beats_ms);
            // Ranking averages the evidence; ACCEPTANCE conjoins it. The two do different jobs: an
            // average orders candidates by how much evidence there is in total, while a conjunction
            // refuses to let a strong axis pay for a weak one. Ranking on the conjunction instead
            // buried the term that separates two rates under the term that cannot see a rate at all.
            let quality = match agreement {
                Some(a) => 0.5 * (ecg.b_excess + a.excess),
                None => ecg.b_excess,
            };
            let ppg_ok = match agreement {
                Some(a) => a.match_fraction >= cfg.min_ppg_match && a.offset_plausible,
                None => !ppg_usable,
            };
            let passes = ecg.verdict.accepted && ecg.b_excess >= cfg.min_b_excess && ppg_ok;
            leaderboard.push((
                Candidate {
                    layout: *l,
                    class,
                    aliases,
                    answer: 0,
                    fs_hz: fs,
                    quality,
                    ecg,
                    ppg: agreement,
                    fs_from_ppg: agreement.and_then(|a| a.fs_solved_hz),
                    passes,
                },
                rep,
            ));
        }
    }

    // Highest rate first among equal scores: within one answer that is the reading that discards the
    // fewest samples, and it becomes the answer's representative below.
    leaderboard.sort_by(|(a, _), (b, _)| {
        b.quality
            .total_cmp(&a.quality)
            .then_with(|| b.fs_hz.total_cmp(&a.fs_hz))
            .then_with(|| order_key(a).cmp(&order_key(b)))
    });
    let mut reps: Vec<(usize, f64)> = Vec::new();
    for (cand, w) in leaderboard.iter_mut() {
        let found = reps.iter().position(|&(rw, rfs)| {
            same_time_base(&kept[rw].1, rfs, &kept[*w].1, cand.fs_hz, cfg.class_min_r)
        });
        cand.answer = match found {
            Some(a) => a,
            None => {
                reps.push((*w, cand.fs_hz));
                reps.len() - 1
            }
        };
    }
    // Re-order so each ANSWER is represented by its densest reading. Within one answer the decimated
    // views score within a thousandth of the full-rate one and either may come out on top by chance;
    // reporting the one that throws samples away because it won a coin toss is not a result.
    let mut leaderboard: Vec<Candidate> = leaderboard.into_iter().map(|(c, _)| c).collect();
    let best_of: Vec<f64> = (0..=leaderboard.iter().map(|c| c.answer).max().unwrap_or(0))
        .map(|a| leaderboard.iter().filter(|c| c.answer == a).map(|c| c.quality).fold(f64::NEG_INFINITY, f64::max))
        .collect();
    leaderboard.sort_by(|a, b| {
        best_of[b.answer]
            .total_cmp(&best_of[a.answer])
            .then_with(|| a.answer.cmp(&b.answer))
            .then_with(|| b.fs_hz.total_cmp(&a.fs_hz))
            .then_with(|| order_key(a).cmp(&order_key(b)))
    });
    let margin = match leaderboard.first() {
        Some(a) => match leaderboard.iter().find(|o| o.answer != a.answer) {
            Some(b) => a.quality - b.quality,
            None => f64::INFINITY,
        },
        None => 0.0,
    };
    leaderboard.truncate(cfg.top_n.max(1));

    WindowReport {
        bytes: bytes.len(),
        layouts_enumerated: layouts.len(),
        layouts_decoded: decoded,
        layouts_survived: kept.len(),
        classes: n_classes,
        scored,
        leaderboard,
        margin,
        attribution: attribution(ppg_beats_ms),
    }
}

/// Which of decode and contact a stall should be blamed on, from the optical channel alone.
pub(super) fn attribution(ppg_beats_ms: &[f64]) -> Attribution {
    if ppg_beats_ms.is_empty() {
        return Attribution::Unknown;
    }
    let Some(bpm) = beat_rate_bpm(ppg_beats_ms) else { return Attribution::Contact };
    if !(MIN_HR_BPM..=MAX_HR_BPM).contains(&bpm) {
        return Attribution::Contact;
    }
    // Erratic is judged on the beat intervals themselves: a contact loss shows up as intervals that are
    // multiples or fragments of the true one, which moves the spread far more than any real rhythm does.
    let gaps: Vec<f64> = ppg_beats_ms.windows(2).map(|w| w[1] - w[0]).filter(|g| g.is_finite() && *g > 0.0).collect();
    if gaps.len() < 3 {
        return Attribution::Contact;
    }
    let m = median(&gaps);
    let spread = median(&gaps.iter().map(|g| (g - m).abs()).collect::<Vec<_>>());
    if m <= 0.0 || spread / m > ERRATIC_BEAT_SPREAD {
        Attribution::Contact
    } else {
        Attribution::Decode
    }
}

/// Median absolute beat-interval deviation, over the median interval, above which the optical channel is
/// called erratic. A dropped or doubled beat moves this statistic by a factor; real beat-to-beat
/// variability does not reach it.
const ERRATIC_BEAT_SPREAD: f64 = 0.25;

fn beat_rate_bpm(beats_ms: &[f64]) -> Option<f64> {
    if beats_ms.len() < ppg::PPG_MIN_BEATS {
        return None;
    }
    let gaps: Vec<f64> = beats_ms.windows(2).map(|w| w[1] - w[0]).filter(|g| g.is_finite() && *g > 0.0).collect();
    if gaps.is_empty() {
        return None;
    }
    let m = median(&gaps);
    (m > 0.0).then(|| 60_000.0 / m)
}

fn beat_rate_bpm_samples(peaks: &[usize], fs_hz: f64) -> Option<f64> {
    if peaks.len() < ppg::PPG_MIN_BEATS {
        return None;
    }
    let gaps: Vec<f64> = peaks.windows(2).map(|w| (w[1] - w[0]) as f64).collect();
    let m = median(&gaps);
    (m > 0.0).then(|| 60.0 * fs_hz / m)
}

/// The member of a class to score and to report: the smoothest reading, then the narrowest field, then
/// the densest packing, then the earliest start.
///
/// A class is one waveform, and its members correlate above 0.9999, but they are not all the same claim.
/// Measured on the corpus, two kinds of near-miss sit in a class with the truth: a 16-bit read of the top
/// of a true 24-bit field, which throws away eight low bits, and a 32-bit read of a true 16-bit field in
/// a two-channel frame, which sweeps in sixteen bits of the neighbour. Both add sample-to-sample
/// discontinuity the true reading does not have, so the smoothest member is the one that neither
/// truncates nor borrows. The margin between them can fall below what an f64 sum of a few thousand terms
/// can resolve, and then the width tie-break decides and every alternative stays in `alias_shapes`.
fn representative(kept: &[(Layout, Vec<f64>, f64)], classes: &[usize], class: usize) -> Option<usize> {
    let members = || (0..classes.len()).filter(|&i| classes[i] == class);
    let floor = members().map(|i| kept[i].2).fold(f64::INFINITY, f64::min);
    if !floor.is_finite() {
        return None;
    }
    // Quantise before comparing. The gap between a truncating read and a borrowing one can be a part in
    // 10^8 of the roughness, which is below what this corpus resolves and below what stays put from one
    // window to the next; treating that as a preference made the reported field width flicker between
    // windows and read as instability. Ties fall to the narrowest field, which is the smaller claim.
    members()
        .min_by_key(|&i| {
            let step = (floor.abs() * REPRESENTATIVE_ROUGHNESS_TOLERANCE).max(f64::MIN_POSITIVE);
            let bucket = ((kept[i].2 - floor) / step).round().clamp(0.0, u32::MAX as f64) as u32;
            let l = kept[i].0;
            (bucket, l.bits, l.stride_bits, l.start_bit)
        })
}

/// Relative roughness difference below which two readings of one class are not ranked against each other.
const REPRESENTATIVE_ROUGHNESS_TOLERANCE: f64 = 1e-4;

fn order_key(c: &Candidate) -> (u64, u8, bool, layout::BitOrder, usize, usize) {
    (c.fs_hz.to_bits(), c.layout.bits, c.layout.signed, c.layout.order, c.layout.start_bit, c.layout.stride_bits)
}

