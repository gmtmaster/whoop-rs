# whoop-rs — agent instructions

**The working contract is [`CLAUDE.md`](CLAUDE.md). Read it, then `docs/architecture.md`.**

This file used to be a second copy of that contract and drifted from it: it still named a
`whoop-metrics` crate that no longer exists, counted 7 crates against 8, and quoted a 66-test suite
against 425. One contract file, so there is nothing left to diverge.

Three rules that hold whatever you were asked to do:

- **Firmware is read-only, never flashed.** Health outputs are wellness estimates, never medical.
- **Gated writes only.** `command::FORBIDDEN` / `DESTRUCTIVE` refuse firmware-load, trim, DFU,
  config-write, set-clock, adv-name and wrist-select on the blind path.
- **Nothing pushed, no PR, no `git init`-and-push without explicit approval.**
