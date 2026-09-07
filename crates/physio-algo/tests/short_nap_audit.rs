//! Read-only forensic regression over an externally supplied audit bundle.
//! Run with `NAP_AUDIT_DIR=/path/to/extracted/nap-audit-2026-09-05-06 cargo test ... --ignored`.

use std::fs;
use std::path::{Path, PathBuf};

use physio_algo::nightly_physiology::RrDeviceGeneration;
use physio_algo::sleep::{
    AccelSample, HrSample, RrRun, SessionAcceptancePath, SessionRole, ShortNapOriginalRejection,
    ShortNapShadowVerdict, SleepStreams, StepSample, analyze_for_rr_generation,
    analyze_for_rr_generation_with_short_nap_promotion,
    analyze_for_rr_generation_with_short_nap_shadow, assign_session_roles,
};

const NAP_A_START: i64 = 1_788_621_707;
const NAP_A_END: i64 = 1_788_624_916;
const NAP_B_START: i64 = 1_788_695_035;
const NAP_B_END: i64 = 1_788_702_815;

fn audit_root() -> PathBuf {
    PathBuf::from(
        std::env::var("NAP_AUDIT_DIR").expect("NAP_AUDIT_DIR must point to the extracted bundle"),
    )
}

fn data_lines(path: &Path) -> impl Iterator<Item = String> {
    fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()))
        .lines()
        .skip(1)
        .map(str::to_string)
        .collect::<Vec<_>>()
        .into_iter()
}

fn source_epoch(line: &str) -> i64 {
    line.rsplit(',')
        .next()
        .and_then(|source| source.split(':').nth(2))
        .and_then(|value| value.parse().ok())
        .unwrap_or_else(|| panic!("missing source epoch in {line}"))
}

fn streams(day: &str) -> SleepStreams {
    let root = audit_root().join(day);
    let hr = data_lines(&root.join("heart_rate_samples.csv"))
        .map(|line| {
            let fields: Vec<&str> = line.split(',').collect();
            HrSample {
                ts: source_epoch(&line),
                bpm: fields[4].parse().unwrap(),
            }
        })
        .collect();
    let accel = data_lines(&root.join("accelerometer_samples.csv"))
        .map(|line| {
            let fields: Vec<&str> = line.split(',').collect();
            AccelSample {
                ts: source_epoch(&line),
                x: fields[4].parse().unwrap(),
                y: fields[5].parse().unwrap(),
                z: fields[6].parse().unwrap(),
            }
        })
        .collect();
    let rr = data_lines(&root.join("rr_runs.csv"))
        .map(|line| RrRun {
            ts: source_epoch(&line),
            intervals: vec![800],
        })
        .collect();
    let steps = data_lines(&root.join("step_samples.csv"))
        .map(|line| {
            let fields: Vec<&str> = line.split(',').collect();
            StepSample {
                ts: source_epoch(&line),
                counter: fields[4].parse().unwrap(),
                activity_class: fields[5].parse().ok(),
            }
        })
        .collect();
    let band_sleep_state = data_lines(&root.join("band_sleep_state_samples.csv"))
        .map(|line| {
            let fields: Vec<&str> = line.split(',').collect();
            (source_epoch(&line), fields[4].parse().unwrap())
        })
        .collect();
    SleepStreams {
        hr,
        rr,
        accel,
        steps,
        tz_offset_s: 2 * 3_600,
        wrist_off: Vec::new(),
        band_sleep_state,
    }
}

#[test]
#[ignore = "requires the external read-only nap audit bundle"]
fn nap_a_keeps_its_detector_boundaries_and_remains_shadow_only() {
    let result = analyze_for_rr_generation_with_short_nap_shadow(
        &streams("sep05-missed"),
        RrDeviceGeneration::Whoop5Mg,
    );

    assert!(result.sessions.is_empty());
    assert_eq!(result.short_nap_diagnostics.len(), 1);
    let diagnostic = &result.short_nap_diagnostics[0];
    assert_eq!((diagnostic.start, diagnostic.end), (NAP_A_START, NAP_A_END));
    assert_eq!(diagnostic.duration_seconds, 53 * 60 + 29);
    assert_eq!(
        diagnostic.original_rejection_reason,
        ShortNapOriginalRejection::BaseMinimumDuration
    );
    assert_eq!(diagnostic.verdict, ShortNapShadowVerdict::Eligible);
    println!("{diagnostic:#?}");
}

#[test]
#[ignore = "requires the external read-only nap audit bundle"]
fn nap_b_remains_exactly_on_the_normal_path() {
    let streams = streams("sep06-detected");
    let production = analyze_for_rr_generation(&streams, RrDeviceGeneration::Whoop5Mg);
    let result =
        analyze_for_rr_generation_with_short_nap_shadow(&streams, RrDeviceGeneration::Whoop5Mg);

    assert_eq!(result.sessions, production);
    assert_eq!(result.sessions.len(), 1);
    assert_eq!(
        (result.sessions[0].start, result.sessions[0].end),
        (NAP_B_START, NAP_B_END)
    );
    assert!(result.short_nap_diagnostics.is_empty());
}

// ---------------------------------------------------------------------------------------------
// Phase B: promotion + canonical role, over the SAME real audit bundle as above.
// ---------------------------------------------------------------------------------------------

/// Phase B regression test A (spec section 11.A): on the real Nap A raw input, the promotion path
/// must (1) preserve the exact Phase A candidate boundaries and duration, (2) preserve the exact
/// Phase A ELIGIBLE verdict (the evaluator itself is untouched), (3) promote it into exactly one
/// canonical session with `role == NAP`, and (4) leave every other accepted-session output
/// (respiratory rate / skin temp / main-night inputs are asserted at the noop-engine layer, not
/// here — see noop-engine's own Phase B tests) otherwise unaffected: zero other sessions, zero
/// duplicate shadow/canonical session for the same candidate.
#[test]
#[ignore = "requires the external read-only nap audit bundle"]
fn nap_a_promotes_to_exactly_one_canonical_nap_with_unchanged_boundaries_and_verdict() {
    let streams = streams("sep05-missed");
    let shadow = analyze_for_rr_generation_with_short_nap_shadow(&streams, RrDeviceGeneration::Whoop5Mg);
    let promoted =
        analyze_for_rr_generation_with_short_nap_promotion(&streams, RrDeviceGeneration::Whoop5Mg);

    // Phase A's own boundaries/verdict are untouched by Phase B (same evaluator, same candidate).
    assert_eq!(shadow.short_nap_diagnostics.len(), 1);
    let diagnostic = &shadow.short_nap_diagnostics[0];
    assert_eq!((diagnostic.start, diagnostic.end), (NAP_A_START, NAP_A_END));
    assert_eq!(diagnostic.duration_seconds, 53 * 60 + 29);
    assert_eq!(diagnostic.verdict, ShortNapShadowVerdict::Eligible);

    // Phase B: exactly one promoted canonical session, same boundaries, no duplicates.
    assert_eq!(promoted.short_nap_diagnostics, shadow.short_nap_diagnostics);
    assert_eq!(promoted.sessions.len(), 1, "no duplicate shadow/canonical session");
    let session = &promoted.sessions[0];
    assert_eq!((session.start, session.end), (NAP_A_START, NAP_A_END));
    assert_eq!(session.acceptance_path, SessionAcceptancePath::ShortNap);

    let roles = assign_session_roles(&promoted.sessions, 2 * 3_600);
    assert_eq!(roles, vec![SessionRole::Nap]);
}

/// Phase B regression test B (spec section 11.B), CORRECTED post-root-cause-fix: Nap B keeps its
/// exact existing canonical boundaries and its Normal acceptance path (the Phase A shadow path
/// still never fires for it). In THIS single-session fixture it is also the day's only Normal-path
/// session — but unlike the original Phase B cut, that no longer trivially wins MAIN_SLEEP.
///
/// Root cause (see `assign_session_roles`'s doc in `sleep/mod.rs`): this fixture is exactly what
/// noop-engine's noon-to-noon compute-date window hands `assign_session_roles` for Nap B's compute
/// date in production — Nap B (13:43:55-15:53:35 local) lands in a DIFFERENT noon-to-noon bucket
/// than the real Sep 6->7 main sleep (which ends before noon and so belongs to the PREVIOUS
/// compute-date's window), so the two never compete in the same call. `main_night_group_indices_
/// scored` is a ranker, not a classifier: given one candidate it always returns that candidate as
/// the `OnlyBlock` winner, evidence or not. `assign_session_roles` now additionally requires the
/// winning group's onset to fall overnight (`detect::is_overnight_onset`, the same signal
/// `mainnight`'s own night-tail bridging already relies on) before trusting a "winner" as
/// MAIN_SLEEP. Nap B's onset (13:43 local) is squarely inside the daytime band, so it fails that
/// gate and now reads FRAGMENT — never silently NAP, which still requires an explicit `ShortNap`
/// promotion (Phase A's shadow path does not fire for Nap B, confirmed below).
#[test]
#[ignore = "requires the external read-only nap audit bundle"]
fn nap_b_keeps_exact_boundaries_normal_path_and_is_fragment_not_main_sleep() {
    let streams = streams("sep06-detected");
    let production = analyze_for_rr_generation(&streams, RrDeviceGeneration::Whoop5Mg);
    let promoted =
        analyze_for_rr_generation_with_short_nap_promotion(&streams, RrDeviceGeneration::Whoop5Mg);

    assert_eq!(promoted.sessions, production);
    assert_eq!(promoted.sessions.len(), 1);
    assert_eq!(
        (promoted.sessions[0].start, promoted.sessions[0].end),
        (NAP_B_START, NAP_B_END)
    );
    assert_eq!(promoted.sessions[0].acceptance_path, SessionAcceptancePath::Normal);
    assert!(promoted.short_nap_diagnostics.is_empty(), "shadow path must not fire for Nap B");

    let roles = assign_session_roles(&promoted.sessions, 2 * 3_600);
    // Being the day's ONLY session, Nap B would trivially win the main-night `OnlyBlock` group —
    // but its onset (13:43 local) has no overnight evidence, so the gate keeps it FRAGMENT rather
    // than converting "only candidate in this isolated compute-date bucket" into MAIN_SLEEP.
    assert_eq!(roles, vec![SessionRole::Fragment]);
}
