//! Read-only forensic regression over an externally supplied audit bundle.
//! Run with `NAP_AUDIT_DIR=/path/to/extracted/nap-audit-2026-09-05-06 cargo test ... --ignored`.

use std::fs;
use std::path::{Path, PathBuf};

use physio_algo::nightly_physiology::RrDeviceGeneration;
use physio_algo::sleep::{
    analyze_for_rr_generation, analyze_for_rr_generation_with_short_nap_shadow, AccelSample,
    HrSample, RrRun, ShortNapOriginalRejection, ShortNapShadowVerdict, SleepStreams, StepSample,
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
