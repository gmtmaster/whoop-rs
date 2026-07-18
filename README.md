# whoop-rs

A from-scratch **Rust WHOOP BLE client** for the WHOOP 4.0 ("Harvard", GEN4) and 5.0 / MG
("Maverick", GEN5) bands. One pure wire codec, a generic BLE core, and a WHOOP client on top —
reusable as a desktop CLI today and as the shared core of an Android and iOS app.

Own-device, right-to-repair, **non-commercial**. Firmware is **read-only, never flashed**. Every
derived health number is a wellness estimate, **never medical**.

## What it does

- **Scan and identify** any WHOOP in range (4.0 and 5.0 / MG), unbonded, by address / serial / name.
- **Sync history** off the band to JSON Lines — HR, R-R, gravity, skin-temp, steps, activity,
  sleep-state, and the 5.0 / MG sleep **SpO₂** — decoding record versions v18 / v24 / v26 and the
  100 Hz raw 6-axis IMU deep buffer. Keep-mode by default; `--wipe` drains the full backlog (and still
  never deletes on the strap — it only advances the read pointer).
- **Live monitor** realtime HR / R-R / events, read the battery + the pack fuel-gauge, **buzz**,
  **reboot**, and unlock the deep **R22** biometric streams — all reversible, gated writes.
- **Derived metrics**, pure (no BLE, no IO): gap-aware artifact-corrected HRV / RMSSD, SpO₂,
  HR-from-PPG, HR zones, resting HR, respiration, stress, VO₂-max, recovery, strain, sleep staging,
  the WHOOP calibration timeline, and a per-strap `linear_fit`.
- **One core, native radios.** A uniffi FFI (`whoop-ffi`) exposes the whole codec + metrics to Kotlin
  and Swift, so a mobile app can drop its own decoder and just feed notification bytes in.

## Quick start

```bash
cd whoop-rs
cargo build
cargo run -p whoopctl -- scan
```

```bash
# identify a band (unbonded), then bond and read its info
whoopctl identify --address <MAC>
whoopctl info --sn <serial-suffix>

# sync last night's history to JSON Lines (keeps the strap's copy)
whoopctl sync --sn <serial-suffix> > night.jsonl

# unlock + drain the deep biometric streams (R22 opt-in)
whoopctl r22on --sn <serial-suffix>
whoopctl sync --wipe --raw capture.jsonl --sn <serial-suffix>

# buzz the band / read the battery pack fuel-gauge
whoopctl buzz --sn <serial-suffix>
whoopctl pack --sn <serial-suffix>
```

Target a band with `--address` (surest), `--sn` (full serial or a suffix), or `--name`. `scan` and
`identify` work over an unbonded connection; the command channel (`info` / `sync` / `buzz` / …) needs
the LE bond.

## Workspace

Nine crates, dependencies pointing strictly inward — the codec knows nothing about BLE or async, the
BLE core knows nothing about WHOOP, and only one crate ever links a radio.

| Crate | Role |
|---|---|
| `whoop-protocol` | Pure sans-IO wire codec: framing, CRC, records (v18 / v24 / v26 / IMU), the offload state machine, config / alarm / haptic builders |
| `physio-algo` | Pure derived metrics + scoring: HRV, resting HR, respiration, HR zones, VO₂-max, recovery, strain, stress, SpO₂, PPG-HR, sleep staging, IMU features, calibration timeline, `linear_fit` |
| `whoop-metrics` | Compatibility re-export shim over `physio-algo` |
| `ble-core` | The `BleTransport` async trait + a neutral notification / error type + `MockTransport` |
| `ble-btleplug` | The btleplug 0.12 backend behind `BleTransport` (the only crate that links a radio) |
| `whoop-client` | `WhoopClient<T>` — bond, history sync, monitor, gated writes, capture, backfill policy |
| `whoop-store` | SQLite per-(person, strap) nightly persistence + milestone-gated baselines |
| `whoopctl` | The clap CLI over the real radio |
| `whoop-ffi` | The uniffi surface → Kotlin + Swift from one Rust source |

Full design + the byte-level wire-protocol map: **[`docs/architecture.md`](docs/architecture.md)**.

## Safety

The write surface is gated in `whoop-protocol` + `whoop-client`:

- **`FORBIDDEN`** opcodes (firmware-load, trim, DFU, config-write, set-clock, adv-name, wrist-select)
  are refused on the blind `send` path; the legitimate ones (reboot, R22) have dedicated methods that
  a UI opt-in gates above the client.
- **`DESTRUCTIVE`** (force-trim, DFU) is never sent at all.
- **History sync never deletes the strap's data.** The ACK only advances the read pointer; only
  FORCE-TRIM deletes, and it lives in `FORBIDDEN`. A CRC-bad `HISTORY_END` can't move the pointer over
  unstored data.
- `whoopctl` refuses `--wipe` on any serial matching the local `WHOOPCTL_PROTECT` allowlist, and when
  the serial can't be read to verify.

## Status

The codec, the btleplug backend, robustness, and the `whoopctl` CLI are done; the mobile FFI compiles
on host. Validated on real 5.0 bands end to end — scan / identify / bond / info / buzz, and a full
overnight `--wipe` drain (35,310 records, 11.6 h) that decoded v18 and **located the sleep SpO₂**
(363 readings, 95-100%), plus v26 raw-PPG and v21 IMU decoders verified against real captures.
`cargo test` and `cargo clippy --all-targets` are green.

Next: a physiological-day nightly bucket, a worn R22 capture (the v20 optical layout, live IMU
streams), and the Android app (`cargo-ndk` + native BLE + the `WhoopCodec` FFI), then iOS (Mac-gated).

## Build notes

Pinned to the **MSVC** toolchain on Windows (btleplug's WinRT deps need the MSVC linker); on a fresh
clone run `rustup override set stable-x86_64-pc-windows-msvc` (requires the VS Build Tools). The pure
core builds on any toolchain. The workspace currently `[patch.crates-io]`es btleplug to a sibling
`../btleplug` fork (a WinRT `add_peripheral` by-address fix) until that change lands upstream.

## License

[PolyForm Noncommercial 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0/). Own-device
interoperability with hardware you own, not for commercial use. Firmware stays read-only; the health
outputs are wellness estimates and never medical.
