//! C6 nightly-HRV regression on the 14 frozen WHOOP reference nights (10 exploration Sep 12-21,
//! 4 held-out Oct 6/7/8/10; `hrv-research/VALIDATION_PLAN.md`).
//!
//! The raw RR of these nights is personal physiological data and is NOT committed. The fixture is
//! produced by `hrv-research/tools/make_c6_fixture.py` and read from `$HRV_C6_REFERENCE_DIR`:
//!   sessions.csv  id,date,split,start,end,c6_expected_ms,whoop_hrv_ms
//!   rr.csv        ts,intervals            (intervals: `|`-separated ms; only these sessions' spans)
//! Run with: `HRV_C6_REFERENCE_DIR=... cargo test -p physio-algo --test c6_reference_nights -- --ignored`
//!
//! Asserts, through the full `nightly_physiology` entry point:
//!   1. every night reproduces the validated research C6 value exactly (|d| <= 1e-9 ms);
//!   2. agreement with WHOOP stays within the frozen result: MAE <= 1.5 ms exploration,
//!      <= 2.5 ms held-out, every night within +/-10 %.
use physio_algo::nightly_physiology::{nightly_physiology, HrvMeasurementMode, ALGORITHM_VERSION_V5};
use physio_algo::sleep::RrRun;
use std::path::PathBuf;

fn dir() -> Option<PathBuf> {
    std::env::var_os("HRV_C6_REFERENCE_DIR").map(PathBuf::from)
}

#[test]
#[ignore = "needs the private reference fixture: set HRV_C6_REFERENCE_DIR and run with --ignored"]
fn c6_reproduces_the_14_reference_nights() {
    let d = dir().expect("HRV_C6_REFERENCE_DIR must point at the private C6 reference fixture");
    let rr_text = std::fs::read_to_string(d.join("rr.csv")).expect("rr.csv");
    let mut rr: Vec<RrRun> = rr_text
        .lines()
        .skip(1)
        .filter_map(|l| {
            let (ts, iv) = l.split_once(',')?;
            let intervals: Vec<u16> = if iv.is_empty() { vec![] } else { iv.split('|').map(|x| x.parse().unwrap()).collect() };
            Some(RrRun { ts: ts.parse().unwrap(), intervals })
        })
        .collect();
    rr.sort_by_key(|r| r.ts);
    let sessions = std::fs::read_to_string(d.join("sessions.csv")).expect("sessions.csv");
    let (mut n, mut ex_err, mut ho_err, mut ex_n, mut ho_n) = (0, 0.0, 0.0, 0, 0);
    for l in sessions.lines().skip(1) {
        let f: Vec<&str> = l.split(',').collect();
        let (date, split) = (f[1], f[2]);
        let (start, end): (i64, i64) = (f[3].parse().unwrap(), f[4].parse().unwrap());
        let (expected, whoop): (f64, f64) = (f[5].parse().unwrap(), f[6].parse().unwrap());
        let night = nightly_physiology(start, end, &[], &[], &rr, &[], &[]);
        let got = night.hrv.rmssd_ms.unwrap_or_else(|| panic!("{date}: C6 refused"));
        assert_eq!(night.hrv.measurement_mode, Some(HrvMeasurementMode::PrimaryWholeNight));
        assert_eq!(night.algorithm_version, ALGORITHM_VERSION_V5);
        assert!((got - expected).abs() <= 1e-9, "{date}: C6 {got} != validated {expected}");
        assert!((got / whoop - 1.0).abs() <= 0.10, "{date}: C6 {got} not within 10 % of WHOOP {whoop}");
        match split {
            "exploration" => { ex_err += (got - whoop).abs(); ex_n += 1 }
            "heldout" => { ho_err += (got - whoop).abs(); ho_n += 1 }
            other => panic!("unknown split {other}"),
        }
        n += 1;
    }
    assert_eq!((n, ex_n, ho_n), (14, 10, 4), "fixture must hold the 14 frozen nights");
    let (ex_mae, ho_mae) = (ex_err / ex_n as f64, ho_err / ho_n as f64);
    eprintln!("C6 vs WHOOP: exploration MAE {ex_mae:.3} ms, held-out MAE {ho_mae:.3} ms");
    assert!(ex_mae <= 1.5 && ho_mae <= 2.5, "MAE regressed: {ex_mae} / {ho_mae}");
}
