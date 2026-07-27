//! Motion-derived output: IMU features, stress onset, steps, calories, workouts and the
//! circadian activity series.

use crate::*;

/// One raw IMU sample: 3-axis accelerometer (g) + 3-axis gyroscope (deg/s).
#[derive(uniffi::Record)]
pub struct ImuSampleInfo {
    pub ax: f64,
    pub ay: f64,
    pub az: f64,
    pub gx: f64,
    pub gy: f64,
    pub gz: f64,
}

/// Compact activity features over a window of raw IMU samples.
#[derive(uniffi::Record)]
pub struct ImuFeaturesInfo {
    pub accel_energy_g: f64,
    pub gyro_energy_dps: f64,
    pub jerk_rms: f64,
    pub cadence_hz: Option<f64>,
    pub cadence_strength: f64,
    pub sample_count: u32,
}

/// Accel/gyro energy, jerk and gait-band cadence over IMU samples at `sample_rate_hz`.
#[uniffi::export]
pub fn imu_features(samples: Vec<ImuSampleInfo>, sample_rate_hz: i32) -> ImuFeaturesInfo {
    let s: Vec<imu_features::ImuSample> = samples
        .into_iter()
        .map(|x| imu_features::ImuSample { ax: x.ax, ay: x.ay, az: x.az, gx: x.gx, gy: x.gy, gz: x.gz })
        .collect();
    let f = imu_features::extract(&s, sample_rate_hz);
    ImuFeaturesInfo {
        accel_energy_g: f.accel_energy_g,
        gyro_energy_dps: f.gyro_energy_dps,
        jerk_rms: f.jerk_rms,
        cadence_hz: f.cadence_hz,
        cadence_strength: f.cadence_strength,
        sample_count: f.sample_count as u32,
    }
}

// ── Stress onset detector ──────────────────────────────────────────────────

/// Persisted state for the onset detector, serializable across FFI.
#[derive(uniffi::Record)]
pub struct OnsetStateInfo {
    pub baseline_rmssd: f64,
    pub was_below: bool,
    pub last_fire_at: i64,
}

/// The decision returned each evaluation, with the state to persist for the next one.
#[derive(uniffi::Record)]
pub struct OnsetDecisionInfo {
    pub should_nudge: bool,
    pub reason: String,
    pub fast_rmssd: Option<f64>,
    pub baseline_rmssd: Option<f64>,
    pub next_state: OnsetStateInfo,
}

pub(crate) fn reason_str(r: stress_onset::OnsetReason) -> String {
    match r {
        stress_onset::OnsetReason::Onset => "onset",
        stress_onset::OnsetReason::Disabled => "disabled",
        stress_onset::OnsetReason::InsufficientData => "insufficient_data",
        stress_onset::OnsetReason::NoDip => "no_dip",
        stress_onset::OnsetReason::NotAnEdge => "not_an_edge",
        stress_onset::OnsetReason::ExerciseGated => "exercise_gated",
        stress_onset::OnsetReason::Suppressed => "suppressed",
    }.to_string()
}

#[allow(clippy::too_many_arguments)]
#[uniffi::export]
pub fn stress_onset_evaluate(
    rr_buffer: Vec<u16>,
    current_hr: Option<f64>,
    recent_motion_g: Option<f64>,
    session_active: bool,
    state: OnsetStateInfo,
    enabled: bool,
    auto_nudge: bool,
    quiet_hours_enabled: bool,
    quiet_start_min: i32,
    quiet_end_min: i32,
    now_sec: i64,
    tz_offset_sec: i64,
) -> OnsetDecisionInfo {
    let s = stress_onset::OnsetState {
        baseline_rmssd: state.baseline_rmssd,
        was_below: state.was_below,
        last_fire_at: state.last_fire_at,
    };
    let d = stress_onset::evaluate(&rr_buffer, current_hr, recent_motion_g, session_active, s,
        enabled, auto_nudge, quiet_hours_enabled, quiet_start_min, quiet_end_min, now_sec, tz_offset_sec);
    OnsetDecisionInfo {
        should_nudge: d.should_nudge,
        reason: reason_str(d.reason),
        fast_rmssd: d.fast_rmssd,
        baseline_rmssd: d.baseline_rmssd,
        next_state: OnsetStateInfo {
            baseline_rmssd: d.next_state.baseline_rmssd,
            was_below: d.next_state.was_below,
            last_fire_at: d.next_state.last_fire_at,
        },
    }
}

/// Raw wrap-aware motion-tick total from step counter samples. None with fewer than 2 samples
/// or no forward movement. The caller applies its stepTicksPerStep calibration.
#[uniffi::export]
pub fn steps_counter(samples: Vec<SleepStepSample>) -> Option<u64> {
    let s: Vec<sleep::StepSample> = samples
        .into_iter()
        .map(|st| sleep::StepSample { ts: st.ts, counter: st.counter, activity_class: st.activity_class })
        .collect();
    steps::steps_in_window(&s).map(|v| v as u64)
}

// ── Calories (Keytel + Harris-Benedict) ────────────────────────────────────

/// Whole-day energy estimate (kcal) from HR samples. Each sample = one second.
#[uniffi::export]
pub fn calories_estimate_day(
    hr: Vec<HrTick>,
    weight_kg: f64,
    height_cm: f64,
    age: f64,
    sex: String,
    hrmax: f64,
    resting_hr: f64,
) -> f64 {
    let samples = to_hr(hr);
    calories::estimate_day_calories(&samples, weight_kg, height_cm, age, &sex, hrmax, resting_hr)
}

/// Bout energy estimate (kcal, kJ) from HR samples. Each sample weighted by elapsed time to next.
#[uniffi::export]
pub fn calories_estimate_bout(
    hr: Vec<HrTick>,
    weight_kg: f64,
    height_cm: f64,
    age: f64,
    sex: String,
    hrmax: f64,
    resting_hr: f64,
) -> Vec<f64> {
    let samples = to_hr(hr);
    let (kcal, kj) = calories::estimate_bout_calories(&samples, weight_kg, height_cm, age, &sex, hrmax, resting_hr);
    vec![kcal, kj]
}

// ── Workout detection ──────────────────────────────────────────────────────

/// A gravity/accel sample for workout detection.
#[derive(uniffi::Record)]
pub struct WorkoutGravitySample {
    pub ts: i64,
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

/// One detected workout session.
#[derive(uniffi::Record)]
pub struct ZoneTimeEntry {
    pub zone: i32,
    pub pct: f64,
}

/// One detected workout session.
#[derive(uniffi::Record)]
pub struct WorkoutSession {
    pub start: i64,
    pub end: i64,
    pub avg_hr: f64,
    pub peak_hr: i32,
    pub strain: Option<f64>,
    pub duration_s: f64,
    pub zone_time: Vec<ZoneTimeEntry>,
    pub avg_hrr_pct: Option<f64>,
    /// Effective HRmax the zone maths used (bpm), and where it came from
    /// ("caller" / "observed" / "tanaka" / "unknown").
    pub hrmax: Option<f64>,
    pub hrmax_source: String,
    pub calories_kcal: Option<f64>,
    pub calories_kj: Option<f64>,
}

/// Detect workout sessions from HR + gravity streams. Returns one session per detected bout.
#[allow(clippy::too_many_arguments)]
#[uniffi::export]
pub fn workout_detect(
    hr: Vec<HrTick>,
    gravity: Vec<WorkoutGravitySample>,
    resting_hr: Option<f64>,
    max_hr: Option<f64>,
    age: Option<f64>,
    weight_kg: f64,
    height_cm: f64,
    sex: String,
) -> Vec<WorkoutSession> {
    let hr_samples = to_hr(hr);
    let grav_samples: Vec<workout::GravitySample> = gravity
        .into_iter()
        .map(|g| workout::GravitySample { ts: g.ts, x: g.x, y: g.y, z: g.z })
        .collect();
    let sessions = workout::detect(
        &hr_samples, &grav_samples,
        resting_hr, max_hr, age,
        weight_kg, height_cm, &sex,
    );
    sessions.into_iter().map(|s| WorkoutSession {
        start: s.start,
        end: s.end,
        avg_hr: s.avg_hr,
        peak_hr: s.peak_hr,
        strain: s.strain,
        duration_s: s.duration_s,
        zone_time: s.zone_time_pct.into_iter().map(|(z, p)| ZoneTimeEntry { zone: z, pct: p }).collect(),
        avg_hrr_pct: s.avg_hrr_pct,
        hrmax: s.hrmax,
        hrmax_source: s.hrmax_source,
        calories_kcal: s.calories_kcal,
        calories_kj: s.calories_kj,
    }).collect()
}

/// One motion-intensity sample: the L2 magnitude of the gravity change vs the previous sample.
#[derive(uniffi::Record)]
pub struct ActivityPointInfo {
    pub ts: i64,
    pub intensity: f64,
}

/// Trailing rolling mean of the motion intensities over `window_s` — the smoothed spine the stillness
/// and sedentary gates threshold against.
#[uniffi::export]
pub fn smoothed_intensity(motion: Vec<ActivityPointInfo>, window_s: f64) -> Vec<f64> {
    let pts: Vec<workout::ActivityPoint> = motion
        .into_iter()
        .map(|p| workout::ActivityPoint { ts: p.ts, intensity: p.intensity })
        .collect();
    workout::smoothed_intensity(&pts, window_s)
}

/// Per-record motion-intensity series from a gravity stream — the shared motion spine the workout,
/// nap and sedentary readings all measure against.
#[uniffi::export]
pub fn activity_series(gravity: Vec<WorkoutGravitySample>) -> Vec<ActivityPointInfo> {
    let g: Vec<workout::GravitySample> = gravity
        .into_iter()
        .map(|s| workout::GravitySample { ts: s.ts, x: s.x, y: s.y, z: s.z })
        .collect();
    workout::activity_series(&g)
        .into_iter()
        .map(|p| ActivityPointInfo { ts: p.ts, intensity: p.intensity })
        .collect()
}

// ── Nap detection ──────────────────────────────────────────────────────────

/// One raw activity sample: unix seconds + the on-chip gravity-removed motion magnitude (g).
#[derive(uniffi::Record, Clone, Copy)]
pub struct ActivitySample {
    pub unix: i64,
    pub activity: f64,
}

/// A circadian Rhythm Age result plus the cosinor fit it came from.
#[derive(uniffi::Record, Clone, Copy)]
pub struct RhythmAgeInfo {
    pub cosinor_age_years: f64,
    pub advance_years: f64,
    pub mesor: f64,
    pub amplitude: f64,
    pub acrophase_hours: f64,
    pub relative_amplitude: f64,
}

pub(crate) fn to_pairs(samples: Vec<ActivitySample>) -> Vec<(i64, f64)> {
    samples.into_iter().map(|s| (s.unix, s.activity)).collect()
}

/// Circadian Rhythm Age from raw (unix, activity) samples + tz offset + chronological age + sex: bins the
/// samples per LOCAL hour, fits a single-component cosinor, then the Gompertz biological-age transform.
/// `None` when the fit is degenerate (< 3 populated hours). v1 is a RELATIVE index; the activity scale is
/// not calibrated to the model's mg-ENMO training units.
#[uniffi::export]
pub fn rhythm_age_from_samples(
    samples: Vec<ActivitySample>,
    tz_offset_seconds: i64,
    chronological_age: f64,
    sex: SexInput,
) -> Option<RhythmAgeInfo> {
    // Scale on-chip activity (g) → the model's mg-ENMO training units before the fit; without it the
    // mesor/amplitude terms collapse and the age is wrong.
    let scale = biological_age::ACTIVITY_TO_MG_ENMO_SCALE;
    let pairs: Vec<(i64, f64)> = to_pairs(samples).into_iter().map(|(t, a)| (t, a * scale)).collect();
    let bins = circadian::hourly_bins(&pairs, tz_offset_seconds);
    let fit = circadian::cosinor(&bins)?;
    // A real rhythm only: below the relative-amplitude floor the fit is flat/noise, so no age is fabricated.
    if fit.relative_amplitude() < circadian::MIN_RELATIVE_AMPLITUDE {
        return None;
    }
    let age = biological_age::cosinor_age(
        fit.mesor,
        fit.amplitude,
        fit.acrophase_radians(),
        chronological_age,
        sex.into(),
    )?;
    Some(RhythmAgeInfo {
        cosinor_age_years: age.cosinor_age_years,
        advance_years: age.advance_years,
        mesor: fit.mesor,
        amplitude: fit.amplitude,
        acrophase_hours: fit.acrophase_hours,
        relative_amplitude: fit.relative_amplitude(),
    })
}
