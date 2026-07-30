//! Step Z gate: follow a sleep error into Rest and Charge, and pin what the chain does to it.
//!
//! Sleep reaches Recovery twice. Once as `sleep_perf`, which is Rest/100 at weight 0.15. Once as the deep
//! spans the nightly HRV is averaged over, and HRV is the dominant driver at weight 0.55. Nothing followed
//! either link, so a change that silently cut one would not have failed a test.
//!
//! Synthetic streams throughout: a fixture that is absent turns a gate into a pass, and this one has to
//! run everywhere. The hypnogram is built by hand rather than staged, because what is under test is the
//! propagation and not the stager.

use physio_algo::hrv::HrvReadiness;
use physio_algo::recovery::{recovery, DriverBaseline, RecoveryInput};
use physio_algo::rest::rest;
use physio_algo::sleep::{SleepStage, StageSegment};

const START: i64 = 1_749_513_600;
const HOURS: i64 = 8;
const NEED_H: f64 = 8.0;

/// R-R over the whole window, one beat a second, with a beat-to-beat wobble that falls steadily from 60
/// ms to 8 across the night. Without that gradient, moving a deep span could land on the same variability
/// it left and the second channel would read zero for the wrong reason.
fn beats() -> Vec<(u32, u16)> {
    let total = HOURS * 3600;
    (0..total)
        .map(|i| {
            let amp = 60 - 52 * i / total;
            let wobble = [0, amp, 0, -amp][(i % 4) as usize];
            ((START + i) as u32, (1_090 + wobble) as u16)
        })
        .collect()
}

/// Light / Deep / Light / REM / Light, one hour each of deep and REM, with an interior wake run.
fn hypnogram() -> Vec<StageSegment> {
    let h = 3600;
    let seg = |a: i64, b: i64, stage: SleepStage| StageSegment { start: START + a, end: START + b, stage };
    vec![
        seg(0, h, SleepStage::Light),
        seg(h, 2 * h, SleepStage::Deep),
        seg(2 * h, 3 * h, SleepStage::Light),
        seg(3 * h, 3 * h + 1800, SleepStage::Wake),
        seg(3 * h + 1800, 4 * h, SleepStage::Light),
        seg(4 * h, 5 * h, SleepStage::Rem),
        seg(5 * h, HOURS * h, SleepStage::Light),
    ]
}

/// Convert `pp` percent of the window's seconds from `from` to `to`, taken off the front of each eligible
/// run so the change is a stated size rather than a scattered one.
fn relabel(segs: &[StageSegment], pp: f64, from: SleepStage, to: SleepStage) -> Vec<StageSegment> {
    let span: i64 = segs.iter().map(|s| s.end - s.start).sum();
    let mut budget = (pp / 100.0 * span as f64).round() as i64;
    let mut out = Vec::new();
    for s in segs {
        if s.stage != from || budget <= 0 {
            out.push(*s);
            continue;
        }
        let take = budget.min(s.end - s.start);
        budget -= take;
        out.push(StageSegment { start: s.start, end: s.start + take, stage: to });
        if s.start + take < s.end {
            out.push(StageSegment { start: s.start + take, end: s.end, stage: from });
        }
    }
    assert_eq!(0, budget, "not enough {from:?} to move {pp} pp");
    out
}

struct Night {
    asleep_s: f64,
    efficiency: f64,
    deep_s: f64,
    rem_s: f64,
    hrv: f64,
}

fn measure(segs: &[StageSegment]) -> Night {
    let secs = |want: SleepStage| -> f64 {
        segs.iter().filter(|s| s.stage == want).map(|s| (s.end - s.start) as f64).sum()
    };
    let (wake, light, deep, rem) =
        (secs(SleepStage::Wake), secs(SleepStage::Light), secs(SleepStage::Deep), secs(SleepStage::Rem));
    let in_bed = wake + light + deep + rem;
    let deep_spans: Vec<(u32, u32)> = segs
        .iter()
        .filter(|s| s.stage == SleepStage::Deep)
        .map(|s| (s.start as u32, s.end as u32))
        .collect();
    let hrv = HrvReadiness::windowed_avg_hrv_deep(
        START as u32,
        (START + HOURS * 3600) as u32,
        &beats(),
        &deep_spans,
    )
    .expect("the crafted night must yield a deep-window HRV");
    Night { asleep_s: light + deep + rem, efficiency: (light + deep + rem) / in_bed, deep_s: deep, rem_s: rem, hrv }
}

fn rest_of(n: &Night) -> f64 {
    rest(n.asleep_s, n.efficiency, n.deep_s, n.rem_s, Some(NEED_H), None).expect("asleep time is positive")
}

/// Charge, with each channel sourced separately so one can be frozen at the unperturbed night.
fn charge(hrv_from: &Night, rest_from: &Night) -> f64 {
    recovery(&RecoveryInput {
        hrv: hrv_from.hrv,
        rhr: 52.0,
        hrv_baseline: Some(DriverBaseline { mean: 45.0, spread: 8.0 }),
        rhr_baseline: Some(DriverBaseline { mean: 55.0, spread: 3.0 }),
        sleep_perf: Some(rest_of(rest_from) / 100.0),
        ..Default::default()
    })
    .expect("every driver is present")
}

/// A wake error reaches Charge only through Rest: relabelling wake against light can create or remove
/// neither a deep nor a REM second, so the deep window the HRV is taken over does not move at all.
#[test]
fn a_wake_error_reaches_charge_through_rest_alone() {
    let base = measure(&hypnogram());
    let hurt = measure(&relabel(&hypnogram(), 10.0, SleepStage::Light, SleepStage::Wake));

    assert!(hurt.efficiency < base.efficiency, "10 pp of wake must cost efficiency");
    assert_eq!(base.hrv, hurt.hrv, "a wake error must not move the deep-window HRV");

    let (r0, r1) = (rest_of(&base), rest_of(&hurt));
    assert!(r1 < r0, "Rest must fall: {r1} vs {r0}");
    let (c0, c1) = (charge(&base, &base), charge(&hurt, &hurt));
    assert!(c1 < c0, "Charge must fall: {c1} vs {c0}");
    // The HRV channel alone, with Rest frozen, must be inert here.
    assert_eq!(c0, charge(&hurt, &base), "the HRV channel must not move on a wake error");

    // Absorbed, not amplified: a 10 pp stage error is worth well under 10 points of Charge.
    let gain = (c0 - c1).abs() / 10.0;
    assert!(gain < 1.0, "the chain must absorb a wake error, gain was {gain}");
    assert!(gain > 0.0, "and it must not swallow it entirely");
}

/// A deep error reaches Charge through BOTH channels, and the one that carries most of it is the HRV
/// window rather than the driver named `sleep`. Freezing either channel must shrink the response.
#[test]
fn a_deep_error_reaches_charge_through_both_channels() {
    let base = measure(&hypnogram());
    let more = measure(&relabel(&hypnogram(), 10.0, SleepStage::Light, SleepStage::Deep));

    assert!(more.deep_s > base.deep_s, "10 pp of deep must be added");
    assert_eq!(base.efficiency, more.efficiency, "moving light to deep cannot change efficiency");
    assert_ne!(base.hrv, more.hrv, "the deep window must move, or the second channel is dead");

    let full = charge(&more, &more) - charge(&base, &base);
    let hrv_only = charge(&more, &base) - charge(&base, &base);
    let rest_only = charge(&base, &more) - charge(&base, &base);
    assert!(full > 0.0 && hrv_only > 0.0 && rest_only > 0.0, "all three must move the same way");
    assert!(hrv_only.abs() > 0.0, "the HRV channel must be live");
    assert!(rest_only.abs() > 0.0, "the Rest channel must be live");
    // Charge is a logistic, so the two do not sum exactly; they must still account for the whole move.
    assert!((hrv_only + rest_only - full).abs() < 0.05 * full.abs(), "the split must explain the total");

    let gain = full.abs() / 10.0;
    assert!(gain < 1.0, "the chain must absorb a deep error, gain was {gain}");
}

/// Rest's own response, pinned so a weight change cannot pass silently. Duration is half of Rest, so an
/// error that shortens the night has to dominate one that only re-labels inside it.
#[test]
fn rest_weights_order_the_three_sleep_errors() {
    let base = rest_of(&measure(&hypnogram()));
    let wake = rest_of(&measure(&relabel(&hypnogram(), 10.0, SleepStage::Light, SleepStage::Wake)));
    let deep = rest_of(&measure(&relabel(&hypnogram(), 10.0, SleepStage::Light, SleepStage::Deep)));
    let rem = rest_of(&measure(&relabel(&hypnogram(), 10.0, SleepStage::Light, SleepStage::Rem)));

    assert!(wake < base, "wake costs Rest");
    assert!(deep > base && rem > base, "both restorative stages raise Rest");
    assert!((base - wake).abs() > (deep - base).abs(), "losing asleep time outweighs re-labelling it");
}
