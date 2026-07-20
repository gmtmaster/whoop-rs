# Algorithms — formulas, location, verification

Every formula NOOP ships. Status icons: ✅ verified (golden tests or DREAMT) · ⚡ verified (parity test) · 🔧 formula correct but code path leak · ❌ unverified.

---

## 1. Sleep detection

```
gravity_deltas = L2(Δx, Δy, Δz) per sample
still_sample  = rolling_window(gravity_deltas) → fraction < 0.01g ≥ 0.70
runs          = collapse(still_samples), break on class change or >20 min gap
                sparse gravity: HR-vouched gap bridge (≤baseline×1.05 across gap)
merged        = absorb runs <15 min into neighbours
bridged       = sparse-only: merge sleep runs ≤90 min apart when HR stays in sleep band
gate loop:
  reject ≤60 min, >16 h
  reject median HR > baseline × 1.05 (×1.30 if deeply motion-quiescent)
  reject off-wrist fraction ≥ 50%
  daytime [11:00,20:00) local: reject unless ≥90 min + resting HR ≤ baseline × 0.95
  morning-stillness (≤3 h after overnight wake): also band-state ≥60% asleep or RHR≤baseline×0.90
  night-continuation chain: overnight-onset run anchors, subsequent ≤90 min from tail = kept
output: DetectedSpan { start, end, resting_hr }
```

| Location | Status |
|---|---|
| `whoop-rs/crates/physio-algo/src/sleep/detect.rs` | ✅ 30 synthetic tests |
| `noop-tan/…/analytics/SleepStager.kt` | 🔧 **DEAD PATH — not called since 2026-07-20** |
| `noop-tan/…/analytics/AnalyticsEngine.kt` → `RustSleepStager.analyze()` | ✅ FFI routed 2026-07-20 |

---

## 2. Sleep staging V2 (cardiorespiratory)

```
Per 30 s epoch, z-scored within-night:

EMISSION:
  deep  = −0.8·z(hr_var) + 0.5·z(HR) − 0.1·z(move_frac) − deep_gate + 0.6·z(resp_reg) + ln(0.15)
  rem   = +0.8·z(hr_var) − 0.4·z(move_frac) + 0.4·z(HR) − 0.6·z(resp_reg) + ln(0.22)
  light = ln(0.50)
  wake  = 1.0·z(move_frac) + 0.5·dz(z(hr_var)) + 0.6·dz(z(HR)) + motion_gate_boost + ln(0.34)

  dz(x)     = deadzone: x in [−0.30,+0.30] → 0
  deep_gate = 5 × max(0, hr_flat11_percentile − 0.40)
  motion_quiescent → clamps cardiac wake term to ≤0
  jerk_max > night_median_jerk × 35 → +4.0 on wake
  resp_reg = R-R tachogram → 4 Hz resample → detrend → DFT peak/sum 0.15–0.40 Hz

CYCLE PRIOR:
  deep = 1.2 × (1 − clock/0.55), decays to 0 after 55 % of night
  rem  = 1.0 × clock − (clock < 0.12 ? 3.0 : 0)

VITERBI (4-state): deep→deep 0.76, rem→rem 0.92, light→light 0.80, wake→wake 0.90
```

| Location | Status |
|---|---|
| `whoop-rs/crates/physio-algo/src/sleep/v2.rs` | ✅ DREAMT κ=0.325 (n=100) · frozen golden test |
| `noop-tan/…/analytics/SleepStager.kt` (V2 path) | 🔧 dead since detection → Rust |

---

## 3. Motion-aware wake refinement

```
Per wake segment ≥5 min, density-gated (≥80 % minutes have ≥2 grav + ≥1 step sample):
  no locomotion (walk ticks) → posture-stable ≥80 % of minutes → burst minutes stay wake, rest → light
  locomotion present → segment stays wake
```

| Location | Status |
|---|---|
| `whoop-rs/crates/physio-algo/src/sleep/refine.rs` | ✅ unit tests |

---

## 4. Main-night selection

```
score(block) = asleep_minutes + alignment_bonus

alignment_bonus:
  target = habitual_midsleep (circular mean, ≥14 days) ?? 03:30 local
  bonus = 90 min within ±2 h, linear decay to 0 at ±5 h

bridge: adjacent blocks <60 min apart → merge; overnight block 60–90 min → merge
reason: OnlyBlock | Longest | LongestNearUsual | AlignedToUsual
```

| Location | Status |
|---|---|
| `whoop-rs/crates/physio-algo/src/sleep/mainnight.rs` | ✅ 15+ tests (cold-start, habitual, biphasic, nap-vs-night) |
| `noop-tan/…/analytics/SleepStageTotals.kt` | 🔧 thin FFI wrapper (delegates to Rust), mirror code still present |

---

## 5. Effort / Strain

```
HR_reserve = HRmax − RHR
%HRR = clamp((HR − RHR) / HR_reserve × 100, 0, 100)

Edwards zones: [90,100)%→5, [80,90)%→4, [70,80)%→3, [60,70)%→2, [50,60)%→1, <50%→0

Per-interval TRIMP (since 2026-07-20):
  Each sample's zone_weight × its own interval_gap (capped at 20 min dropout).
  First sample → forward gap, last → backward gap, middle → avg(fwd, bwd).

Effort = 100 × ln(TRIMP + 1) / ln(7201)
```

| Location | Status |
|---|---|
| `whoop-rs/crates/physio-algo/src/strain.rs` | ✅ 14 tests (golden, cadence transition, dropout cap, uniform agreement) |
| `noop-tan/…/analytics/StrainScorer.kt` | ✅ delegated to RustScores 2026-07-20 (6 callers → per-interval fix) |

---

## 6. Charge / Recovery

```
For each nightly metric x, EWMA baseline (μ = center, s = spread):
  σ = 1.253 × s
  z(x) = (x − μ) / max(σ, 1e−9)

Term             Formula                        Weight
HRV              z(HRV)                         0.55   higher → better
Resting HR       z(RHR_baseline, RHR_current)  0.20   lower → better
Respiration      z(resp_baseline, resp_current) 0.05   lower → better
Rest quality     (Rest/100 − 0.85) / 0.12      0.15   higher → better
Skin temp        −|deviation_C| / 1.0           0.05   near baseline → better

Only present terms enter; weights renormalize.
composite_z = Σ(term_z × weight) / Σ(weights)
Charge = clamp(100 / (1 + exp(−1.6·(composite_z + 0.20))), 0, 100)
```

| Location | Status |
|---|---|
| `whoop-rs/crates/physio-algo/src/recovery.rs` | ✅ parity tests |
| `noop-tan/…/analytics/IntelligenceEngine.kt` | ✅ skin temp ordering + chronological baselines fixed 2026-07-20 |

---

## 7. Personal baselines (EWMA)

```
Per metric (HRV, RHR, respiration, skin temp, Effort):
  center α = 1 − exp(ln(0.5) / 14)   (14-night half-life)
  spread α = 1 − exp(ln(0.5) / 21)   (21-night half-life)

  <4 nights: Calibrating (unusable)
  4−13: Provisional (usable)
  ≥14: Trusted
  >14 missing after usable: Stale

  Cold-start (first 8): center α=3-night half-life, winsor ×2.5, no hard-outlier reject
  Steady-state: accepted values clamped to 3×spread, beyond 5× seen but not folded
```

| Location | Status |
|---|---|
| `noop-tan/…/analytics/Baselines.kt` | 🔧 **Kotlin-only, not in whoop-rs** |

---

## 8. Daily Stress

```
Against up to 30 prior (RHR, HRV) rows:
  zRHR = (RHR_today − μ_RHR) / max(σ_RHR, 0.0001)
  zHRV = (μ_HRV − HRV_today) / max(σ_HRV, 0.0001)
  Stress = 3 / (1 + exp(−(zRHR + zHRV)))

Bands: low [0,1) · medium [1,2) · high [2,3]
```

| Location | Status |
|---|---|
| `whoop-rs/crates/physio-algo/src/stress.rs` | ✅ Rust 2026-07-20 (5 tests) · 14-day baseline gate · Kotlin StressModel still called from UI |
| `whoop-rs/crates/physio-algo/src/stress.rs` | Baevsky SI only, not daily Stress |

---

## 9. Daytime Stress

```
Per hour 06:00−21:59, ≥300 HR rows:
  calm_HR  = Q25(hourly means)     calm_HRV = Q75(hourly RMSSDs)
  zHR   = (mean_HR − calm_HR) / σ_HR     zHRV = (calm_HRV − RMSSD) / σ_HRV
  Stress = 3 / (1 + exp(−(zHR + zHRV)))

No exercise gate.
```

| Location | Status |
|---|---|
| `noop-tan/…/analytics/DaytimeStress.kt` | ❌ Kotlin-only · no exercise gate · reads only `my-whoop` source |

---

## 10. R-R / HRV

```
range_filter: keep 300–2000 ms
RMSSD = √(mean((rr[i+1] − rr[i])²))
windowed_avg_HRV = mean of 5-min tumbling-window RMSSDs over session
Gap-aware: gaps > 3×median-RR split the run, RMSSD computed per gap-free segment
```

| Location | Status |
|---|---|
| `whoop-rs/crates/physio-algo/src/hrv.rs` | ✅ parity tests · real-data agreement fixtures |
| `noop-tan/…/analytics/HrvAnalyzer.kt` | 🔧 **12 callers still use Kotlin mirror** |

---

## 11. Resting HR

```
Session resting HR = lowest 5-min rolling-mean HR across the session.
```

| Location | Status |
|---|---|
| `whoop-rs/crates/physio-algo/src/resting_hr.rs` | ✅ parity tests |
| `noop-tan/…/analytics/AnalyticsEngine.kt` | ⚡ routed through `RustScores.dailyRestingHr()` |

---

## 12. Respiration rate (RSA)

```
Median R-R-derived RSA estimate per session.
```

| Location | Status |
|---|---|
| `whoop-rs/crates/physio-algo/src/respiratory_rate.rs` | ✅ parity tests |
| `noop-tan/…/analytics/AnalyticsEngine.kt` | 🔧 Kotlin also computes independently |

---

## 13. Baevsky Stress Index

```
SI = AMo / (2 × Mo × MxDMn)
  AMo = mode amplitude (histogram peak %)   Mo = mode RR (s)   MxDMn = RR range (s)
```

| Location | Status |
|---|---|
| `whoop-rs/crates/physio-algo/src/stress.rs` | ✅ golden test |
| `noop-tan/…/analytics/StressIndex.kt` | 🔧 duplicated |

---

## 14. Live stress onset detector

```
Rolling R-R + HR window: RMSSD decline + HR stability/rise → onset event.
Suppressed by: HR zone (exercise), recent motion, low R-R count.
```

| Location | Status |
|---|---|
| `noop-tan/…/analytics/StressOnsetDetector.kt` | ❌ Kotlin-only |

---

## 15. Rest quality composite

```
Rest = f(main-night stages, efficiency, duration, prior Rest)
Exact formula embedded in AnalyticsEngine.computeRest.
```

| Location | Status |
|---|---|
| `whoop-rs/crates/physio-algo/src/rest.rs` | ✅ Rust 2026-07-20 (5 tests) · Kotlin RestScorer still called, FFI not yet wired |

---

## 16. Sleep debt

```
Rolling 14-night window total sleep vs 8 h fixed need.
```

| Location | Status |
|---|---|
| `whoop-rs/crates/physio-algo/src/sleep_debt.rs` | ✅ Rust 2026-07-20 (5 tests) · Kotlin SleepDebt.kt still called |

---

## 17. HR zones

```
HRmax = user override ?? tanaka(208−0.7×age) ?? 220−age
Zones: each 10 %-HRR band from 50 % up.
```

| Location | Status |
|---|---|
| `whoop-rs/crates/physio-algo/src/hr_zones.rs` | ✅ parity tests |
| `noop-tan/…/analytics/HrZones.kt` | 🔧 Kotlin mirror |

---

## Fix status

| # | Formula | Status |
|---|---|---|
| ✅ | Sleep detection → `analyzeSleep` FFI | Routed 2026-07-20 |
| ✅ | Effort per-interval integration | Fixed 2026-07-20 (whoop-rs) |
| ✅ | Charge skin temp ordering | Fixed 2026-07-20 (Kotlin) |
| 🔧 | Effort call sites → Rust FFI | 16 Kotlin callers on old StrainScorer |
| 🔧 | HRV → Rust FFI | 12 Kotlin callers on old HrvAnalyzer |
| ✅ | Charge chronological baselines | Incremental fold in pass-2 2026-07-20 |
| 🔧 | R-R beat order | `ORDER BY ts, seq` in Room query 2026-07-20 |
| 🔧 | Baselines → whoop-rs | 400 lines Kotlin |
| ❌ | Daily Stress → whoop-rs | No baseline gate, not in Rust |
| ❌ | Daytime Stress → whoop-rs | No exercise gate, not in Rust |
| ❌ | Rest formula → whoop-rs | Not extracted, no parity test |
| ❌ | Sleep debt → whoop-rs | Kotlin-only |
| ❌ | Onset detector → whoop-rs | Kotlin-only |
| ❌ | Daytime Stress reads source | Hardcoded `my-whoop`, no active-device union |
