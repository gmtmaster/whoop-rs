//! DEPRECATED — superseded by `paired_sleep_benchmark.rs`, the generalized canonical replay adapter for
//! `dev-docs/paired-sleep-benchmark/` (works for any night, not just Night 1). This file could not be
//! deleted from this session (the connected-folder mount here refuses removal), so it's reduced to a
//! redirect stub rather than left as a second, silently-diverging implementation.
//!
//! Use instead:
//!   WHOOP_NIGHT_DIR=... WHOOP_NIGHT_ID=2026-09-01 \
//!   WHOOP_SESSION_START="2026-08-31 22:07:47+00" WHOOP_SESSION_END="2026-09-01 06:47:16+00" \
//!   cargo run --release -p physio-algo --example paired_sleep_benchmark

fn main() {
    eprintln!(
        "night1_replay is deprecated -- run `cargo run -p physio-algo --example paired_sleep_benchmark` \
         instead (see this file's module doc for the env vars)."
    );
    std::process::exit(1);
}
