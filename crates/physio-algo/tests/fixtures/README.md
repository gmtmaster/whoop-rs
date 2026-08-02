# physio-algo test fixtures — what is tracked, and why

Every file under this directory is **tracked**: 24 fixtures, 1,046 KiB, plus this README. Nothing here reproduces a
restricted corpus, so nothing here is ignored. The rule that decides it, and the ignore lines that
enforce it, are in the repo `.gitignore`.

| directory | files | size | source | licence | verdict |
|---|---|---|---|---|---|
| `aauwss_ecg` | 13 | 513 KiB | AAUWSS v1.1, Zenodo `10.5281/zenodo.16919071` (open access) | CC-BY-4.0 | track |
| `aauwss_ppg` | 1 | 8 KiB | same record, `ppg` pickles | CC-BY-4.0 | track |
| `rhythm_rr` | 10 | 525 KiB | PhysioNet `afdb` / `nsrdb` / `mitdb` 1.0.0 (open access) | ODC-BY 1.0 | track |

Both licences permit redistribution of derived works provided attribution travels with the copy, so
every file carries its source and licence in its own `#` header — the AAUWSS ones name the authors
(Djanian, Nielsen, Nielsen, Bruun 2025), the PhysioNet ones name the databases and ODC-BY 1.0. The
converters write those lines, so a regenerated fixture keeps them.

Each file is a small derivation, not a copy of the corpus: one 30 s epoch per AAUWSS subject
(the middle one, chosen by rule so no epoch is picked for quality), and a fixed-stride sample of
256-beat R-R stretches per rhythm class out of 382,466 windows.

## What may never be tracked

A fixture derived from a **restricted** corpus stays out of git however small it is. **DREAMT is
PhysioNet credentialed-access**, so no DREAMT-derived fixture may be committed — reproducing its
content in a public repo is redistribution regardless of file size. `.gitignore` reserves
`tests/fixtures/restricted/` and `tests/fixtures/dreamt*/` for that material so it can never be added
by accident, and the DREAMT-backed parity gates read their input from outside this repo
(`sleep-benchmark/`, `whoop-data/`) and are `#[ignore]`d for that reason.

## If a fixture is missing

Six test files read this directory, carrying 32 live tests (plus 12 `#[ignore]`d), and none of them
skips on absence: an `assume`-style skip on a missing
fixture reports a pass, and this project has lost gates that way. The loaders panic with the fetch
route instead — `tests/ecg_corpus/mod.rs` for AAUWSS, `tests/rr_irregularity_rhythm.rs` for the rhythm
set. Regeneration:

```
python tools/aauwss_ecg_to_fixture.py           # needs the Zenodo pickles unpacked under whoop-data
python tools/aauwss_ppg_to_fixture.py
python tools/physionet_to_fixture.py fetch      # headers + annotations only, ~tens of MB
python tools/physionet_to_fixture.py convert
cargo run --release -p physio-algo --example rr_rhythm_corpus -- --emit crates/physio-algo/tests/fixtures/rhythm_rr
```
