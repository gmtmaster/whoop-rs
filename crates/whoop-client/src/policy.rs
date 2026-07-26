//! Pure offload-cadence + reconnect policy (no BLE, no clock): a strap that keeps offloading nothing
//! (off-wrist / not banking) or whose RTC reads future-dated stops being re-polled every 90 s, without
//! ever delaying a user- or connection-driven sync.

/// What prompted a historical-offload attempt.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackfillTrigger {
    /// The repeating 900 s timer while connected + bonded.
    Periodic,
    /// A (re)connect / bond confirmation.
    Connect,
    /// The app became active (foreground).
    Foreground,
    /// The user tapped "Sync now".
    Manual,
    /// An incoming strap EVENT packet.
    Strap,
    /// An immediate back-to-back continuation of a deep oldest-first backlog. Un-floored by design.
    AutoContinue,
}

pub const PERIODIC_FLOOR_S: f64 = 900.0;
pub const EVENT_FLOOR_S: f64 = 90.0;
pub const EMPTY_BACKOFF_THRESHOLD: i32 = 3;
pub const MAX_EMPTY_BACKOFF: f64 = 4.0;

/// Whether an offload should run now. `empty_streak` = consecutive completed-but-empty offloads (past
/// the threshold the automatic triggers stretch their floor, doubling per empty up to the cap);
/// `clock_untrusted` = the strap RTC reads future-dated (skip the automatic triggers entirely).
pub fn should_run(
    trigger: BackfillTrigger,
    now_s: f64,
    last_backfill_s: Option<f64>,
    empty_streak: i32,
    clock_untrusted: bool,
) -> bool {
    let Some(last) = last_backfill_s else { return true };
    let elapsed = now_s - last;
    let backoff = if empty_streak >= EMPTY_BACKOFF_THRESHOLD {
        2f64.powi(empty_streak - EMPTY_BACKOFF_THRESHOLD + 1).min(MAX_EMPTY_BACKOFF)
    } else {
        1.0
    };
    match trigger {
        BackfillTrigger::Manual | BackfillTrigger::AutoContinue => true,
        BackfillTrigger::Connect | BackfillTrigger::Foreground => elapsed >= EVENT_FLOOR_S,
        BackfillTrigger::Strap => !clock_untrusted && elapsed >= EVENT_FLOOR_S * backoff,
        BackfillTrigger::Periodic => !clock_untrusted && elapsed >= PERIODIC_FLOOR_S * backoff,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manual_never_floored() {
        assert!(should_run(BackfillTrigger::Manual, 100.0, Some(99.9), 10, true));
    }

    #[test]
    fn periodic_respects_floor_and_backoff() {
        // Under the floor → no.
        assert!(!should_run(BackfillTrigger::Periodic, 800.0, Some(0.0), 0, false));
        // Over the floor → yes.
        assert!(should_run(BackfillTrigger::Periodic, 1000.0, Some(0.0), 0, false));
        // Empty-streak past threshold stretches the floor (streak 3 → 2×): 1000 < 1800 → no.
        assert!(!should_run(BackfillTrigger::Periodic, 1000.0, Some(0.0), 3, false));
    }

    #[test]
    fn clock_untrusted_skips_automatic() {
        assert!(!should_run(BackfillTrigger::Periodic, 100000.0, Some(0.0), 0, true));
        assert!(!should_run(BackfillTrigger::Strap, 100000.0, Some(0.0), 0, true));
        // …but a connect pass still runs.
        assert!(should_run(BackfillTrigger::Connect, 100.0, Some(0.0), 0, true));
    }
}
