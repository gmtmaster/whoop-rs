//! Sleep Regularity Index — how reliably the wearer is in the same state (asleep or awake) at the same
//! clock time from one day to the next. Timing regularity, not duration regularity: a person who sleeps
//! exactly eight hours a night but starts at 22:00 one day and 02:00 the next is regular by duration and
//! irregular by this measure, and it is the irregular group the mortality cohorts identify.

/// Epoch length (s) the index is computed on. Matches the sleep stager's grid, so a hypnogram feeds it
/// directly with no resampling.
pub const EPOCH_SECONDS: i64 = 30;

/// Epochs in one day at [`EPOCH_SECONDS`].
pub const EPOCHS_PER_DAY: usize = (24 * 60 * 60 / EPOCH_SECONDS) as usize;

/// Days of paired coverage required before an index is offered. Below this the estimate swings wildly on
/// a single odd night.
pub const MIN_PAIRED_DAYS: usize = 5;

/// Fraction of epoch pairs that must be observed (both days known) before the index is trusted. Non-wear
/// is UNKNOWN, never "awake" — counting a removed strap as wakefulness would read not wearing it as
/// irregularity, which is the opposite of what the measure is for.
pub const MIN_PAIRED_COVERAGE: f64 = 0.70;

/// One epoch's known state, or `None` where the strap was not reporting.
pub type EpochState = Option<bool>;

/// Sleep Regularity Index over a day-major grid of epoch states, oldest day first. Each inner slice is
/// one day of [`EPOCHS_PER_DAY`] epochs aligned to local midnight.
///
/// `SRI = -100 + (200 / P) * matches`, where `P` is the number of epoch pairs observed on both days.
/// +100 is a perfectly repeating schedule, 0 is chance, negative is anti-phase. `None` when fewer than
/// [`MIN_PAIRED_DAYS`] day-pairs or under [`MIN_PAIRED_COVERAGE`] of the pairs are observed.
///
/// Defined by Phillips et al., Scientific Reports 2017 (PMID 28607474) and validated in older adults by
/// Lunsford-Avery et al., Scientific Reports 2018 (PMID 30242174).
pub fn sleep_regularity_index(days: &[Vec<EpochState>]) -> Option<f64> {
    if days.len() < MIN_PAIRED_DAYS + 1 {
        return None;
    }
    let (mut matches, mut pairs, mut possible) = (0usize, 0usize, 0usize);
    for w in days.windows(2) {
        let (today, tomorrow) = (&w[0], &w[1]);
        let n = today.len().min(tomorrow.len());
        possible += n;
        for i in 0..n {
            if let (Some(a), Some(b)) = (today[i], tomorrow[i]) {
                pairs += 1;
                if a == b {
                    matches += 1;
                }
            }
        }
    }
    if pairs == 0 || possible == 0 {
        return None;
    }
    if (pairs as f64 / possible as f64) < MIN_PAIRED_COVERAGE {
        return None;
    }
    Some(-100.0 + 200.0 * matches as f64 / pairs as f64)
}

/// Longest gap between consecutive samples still treated as continuous wear. Past it the strap was not
/// reporting and the time is unknown, not awake.
pub const MAX_WEAR_GAP_SECONDS: i64 = 300;

/// Coverage windows from a sample series: consecutive timestamps closer than [`MAX_WEAR_GAP_SECONDS`]
/// form one span. Taking min..max of a day instead would swallow every mid-day gap as worn time, which
/// for this index means silently scoring a removed strap as wakefulness.
pub fn coverage_spans(timestamps: &[i64], max_gap_s: i64) -> Vec<(i64, i64)> {
    let mut ts: Vec<i64> = timestamps.to_vec();
    ts.sort_unstable();
    ts.dedup();
    let mut out: Vec<(i64, i64)> = Vec::new();
    for t in ts {
        match out.last_mut() {
            Some(last) if t - last.1 <= max_gap_s => last.1 = t + 1,
            _ => out.push((t, t + 1)),
        }
    }
    out
}

/// Build the day-major epoch grid the index needs from asleep spans and the wear windows they sit in.
///
/// `asleep` are `[start, end)` unix-second spans the stager called sleep. `covered` are the spans the
/// strap was actually reporting over; an epoch outside every one of them is `None` (unknown), NOT awake.
/// The grid starts at `first_local_midnight` and runs `days` days.
pub fn epoch_grid(
    first_local_midnight: i64,
    days: usize,
    asleep: &[(i64, i64)],
    covered: &[(i64, i64)],
) -> Vec<Vec<EpochState>> {
    let in_any = |t: i64, spans: &[(i64, i64)]| spans.iter().any(|&(s, e)| t >= s && t < e);
    (0..days)
        .map(|d| {
            let day_start = first_local_midnight + d as i64 * 24 * 60 * 60;
            (0..EPOCHS_PER_DAY)
                .map(|i| {
                    let t = day_start + i as i64 * EPOCH_SECONDS;
                    if in_any(t, covered) {
                        Some(in_any(t, asleep))
                    } else {
                        None
                    }
                })
                .collect()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const DAY: i64 = 24 * 60 * 60;

    /// A night of `hours` starting at `start_hour` on day `d`, as a `[start, end)` span.
    fn night(d: i64, start_hour: f64, hours: f64) -> (i64, i64) {
        let start = d * DAY + (start_hour * 3600.0) as i64;
        (start, start + (hours * 3600.0) as i64)
    }

    /// Nights for days -1..days, so day 0 carries the morning block of the night that started before the
    /// grid. Without it day 0 is a genuinely different pattern and every schedule looks slightly irregular.
    fn nights(days: i64, start_hour: f64, hours: f64) -> Vec<(i64, i64)> {
        (-1..days).map(|d| night(d, start_hour, hours)).collect()
    }

    fn full_cover(days: i64) -> Vec<(i64, i64)> {
        vec![(0, days * DAY)]
    }

    /// A fully covered grid of `days` identical 23:00-07:00 nights.
    fn steady(days: usize) -> Vec<Vec<EpochState>> {
        epoch_grid(
            0,
            days,
            &nights(days as i64, 23.0, 8.0),
            &full_cover(days as i64),
        )
    }

    /// [`steady`] with every epoch of day 0 blanked and `blank_tail` epochs off the last day. Each
    /// of those days sits in exactly one pair window, so this removes `EPOCHS_PER_DAY + blank_tail`
    /// observed pairs from the `(days - 1) * EPOCHS_PER_DAY` possible, leaving `possible` untouched.
    fn thin_coverage(days: usize, blank_tail: usize) -> Vec<Vec<EpochState>> {
        let mut grid = steady(days);
        for e in grid[0].iter_mut() {
            *e = None;
        }
        for e in grid[days - 1].iter_mut().take(blank_tail) {
            *e = None;
        }
        grid
    }

    /// The epochs that must be blanked off the last day to sit exactly on [`MIN_PAIRED_COVERAGE`]
    /// in a `days`-day grid whose day 0 is already blank.
    fn tail_at_the_coverage_floor(days: usize) -> usize {
        let possible = (days - 1) * EPOCHS_PER_DAY;
        possible - EPOCHS_PER_DAY - (MIN_PAIRED_COVERAGE * possible as f64) as usize
    }

    /// A grid with one bedtime 4 h later than the other seven.
    fn one_late_night() -> Vec<Vec<EpochState>> {
        let mut asleep = nights(8, 23.0, 8.0);
        asleep[5] = night(4, 3.0, 8.0); // index 5 == day 4, the vec starts at -1
        epoch_grid(0, 8, &asleep, &full_cover(8))
    }

    /// A grid asleep all day on even days and awake all day on odd ones: every pair disagrees.
    fn inverted() -> Vec<Vec<EpochState>> {
        let asleep: Vec<(i64, i64)> = (0..8)
            .filter(|d| d % 2 == 0)
            .map(|d| (d * DAY, d * DAY + DAY))
            .collect();
        epoch_grid(0, 8, &asleep, &full_cover(8))
    }

    /// One row of the truth table: a grid, and the value the index must yield on it.
    struct Row {
        name: &'static str,
        grid: Vec<Vec<EpochState>>,
        want: Option<f64>,
    }

    fn row(name: &'static str, grid: Vec<Vec<EpochState>>, want: Option<f64>) -> Row {
        Row { name, grid, want }
    }

    fn table() -> Vec<Row> {
        let floor_tail = tail_at_the_coverage_floor(MIN_PAIRED_DAYS + 1);
        vec![
            row("an identical schedule", steady(8), Some(100.0)),
            row("a perfect inversion", inverted(), Some(-100.0)),
            row("one late night", one_late_night(), Some(71.42857142857142)),
            row("no days at all", vec![], None),
            row("one paired day short", steady(MIN_PAIRED_DAYS), None),
            row(
                "at the paired-day minimum",
                steady(MIN_PAIRED_DAYS + 1),
                Some(100.0),
            ),
            row(
                "at the coverage floor",
                thin_coverage(MIN_PAIRED_DAYS + 1, floor_tail),
                Some(100.0),
            ),
            row(
                "one pair under the floor",
                thin_coverage(MIN_PAIRED_DAYS + 1, floor_tail + 1),
                None,
            ),
        ]
    }

    fn reproduces(scorer: impl Fn(&[Vec<EpochState>]) -> Option<f64>) -> bool {
        table()
            .into_iter()
            .all(|r| match (scorer(&r.grid), r.want) {
                (Some(got), Some(w)) => (got - w).abs() < 1e-9,
                (None, None) => true,
                _ => false,
            })
    }

    /// The index formula with both availability gates removed, so it answers on any grid.
    fn ungated(days: &[Vec<EpochState>]) -> Option<f64> {
        let (mut matches, mut pairs) = (0usize, 0usize);
        for w in days.windows(2) {
            let (today, tomorrow) = (&w[0], &w[1]);
            for (a, b) in today.iter().zip(tomorrow) {
                if let (Some(a), Some(b)) = (a, b) {
                    pairs += 1;
                    matches += usize::from(a == b);
                }
            }
        }
        (pairs > 0).then(|| -100.0 + 200.0 * matches as f64 / pairs as f64)
    }

    #[test]
    fn the_shipped_index_reproduces_the_table_and_three_do_nothing_scorers_do_not() {
        for r in table() {
            let (name, want) = (r.name, r.want);
            match (sleep_regularity_index(&r.grid), want) {
                (Some(got), Some(w)) => {
                    assert!((got - w).abs() < 1e-9, "{name}: got {got}, want {w}")
                }
                (None, None) => {}
                (got, w) => panic!("{name}: got {got:?}, want {w:?}"),
            }
        }
        assert!(reproduces(sleep_regularity_index));
        // One refuses everything, one calls every schedule perfect, one runs the formula with both
        // availability gates removed. Each must miss at least one row.
        type Null = fn(&[Vec<EpochState>]) -> Option<f64>;
        let nulls: [(&str, Null); 3] = [
            ("always none", |_| None),
            ("always perfect", |_| Some(100.0)),
            ("no availability gate", ungated),
        ];
        for (name, null) in nulls {
            assert!(
                !reproduces(null),
                "{name} reproduced every row; the table cannot tell it apart"
            );
        }
    }

    #[test]
    fn an_identical_schedule_every_night_is_perfectly_regular() {
        let grid = epoch_grid(0, 8, &nights(8, 23.0, 8.0), &full_cover(8));
        let sri = sleep_regularity_index(&grid).unwrap();
        assert!((sri - 100.0).abs() < 1e-9, "got {sri}");
    }

    #[test]
    fn a_shifting_bedtime_reads_irregular_even_at_constant_duration() {
        // THE POINT OF THIS METRIC: every night is exactly 8 h, so duration variance is zero and the old
        // 1 - CV measure would call this perfectly regular. The start time alternates by 4 h, which the
        // index sees.
        let asleep: Vec<(i64, i64)> = (-1..8)
            .map(|d| night(d, if d.rem_euclid(2) == 0 { 22.0 } else { 2.0 }, 8.0))
            .collect();
        let grid = epoch_grid(0, 8, &asleep, &full_cover(8));
        let sri = sleep_regularity_index(&grid).unwrap();
        assert!(
            sri < 70.0,
            "a 4 h alternating bedtime should not read as regular, got {sri}"
        );

        let steady_sri =
            sleep_regularity_index(&epoch_grid(0, 8, &nights(8, 22.0, 8.0), &full_cover(8)))
                .unwrap();
        assert!(
            steady_sri > sri,
            "steady {steady_sri} must beat shifting {sri}"
        );
    }

    #[test]
    fn a_perfectly_inverted_schedule_is_negative() {
        // Asleep all day on even days, awake all day on odd days: every pair disagrees.
        let asleep: Vec<(i64, i64)> = (0..8)
            .filter(|d| d % 2 == 0)
            .map(|d| (d * DAY, d * DAY + DAY))
            .collect();
        let grid = epoch_grid(0, 8, &asleep, &full_cover(8));
        let sri = sleep_regularity_index(&grid).unwrap();
        assert!((sri + 100.0).abs() < 1e-9, "got {sri}");
    }

    #[test]
    fn non_wear_is_unknown_and_never_counted_as_awake() {
        // Identical nights, but the strap only reported for the first four days. The observed pairs still
        // agree, so the index stays high instead of being dragged down by the gap.
        let asleep = nights(8, 23.0, 8.0);
        let grid = epoch_grid(0, 8, &asleep, &[(0, 5 * DAY)]);
        // Coverage is 4 of 7 pairs = 0.57, under the gate.
        assert_eq!(sleep_regularity_index(&grid), None);

        // Widen the wear window and the same schedule scores.
        let worn = epoch_grid(0, 8, &asleep, &[(0, 7 * DAY)]);
        assert!(sleep_regularity_index(&worn).unwrap() > 99.0);
    }

    #[test]
    fn coverage_spans_split_on_a_real_gap() {
        // Two clusters an hour apart: one span each, never one span swallowing the gap.
        let mut ts: Vec<i64> = (0..600).collect();
        ts.extend(4200..4800);
        let spans = coverage_spans(&ts, MAX_WEAR_GAP_SECONDS);
        assert_eq!(spans, vec![(0, 600), (4200, 4800)]);
        // A gap under the threshold stays one span.
        let close: Vec<i64> = (0..100).chain(200..300).collect();
        assert_eq!(coverage_spans(&close, MAX_WEAR_GAP_SECONDS), vec![(0, 300)]);
        assert!(coverage_spans(&[], MAX_WEAR_GAP_SECONDS).is_empty());
    }

    #[test]
    fn the_paired_day_minimum_is_the_edge_between_none_and_a_score() {
        assert_eq!(MIN_PAIRED_DAYS, 5);
        assert_eq!(sleep_regularity_index(&[]), None);
        // `n` days carry `n - 1` pair windows, so the minimum needs one more day than pairs.
        let short = steady(MIN_PAIRED_DAYS);
        assert_eq!(short.len(), MIN_PAIRED_DAYS);
        assert_eq!(
            sleep_regularity_index(&short),
            None,
            "one paired day short must refuse"
        );
        let sri = sleep_regularity_index(&steady(MIN_PAIRED_DAYS + 1))
            .expect("the minimum itself scores");
        assert!((sri - 100.0).abs() < 1e-9, "got {sri}");
    }

    #[test]
    fn the_coverage_floor_is_inclusive_at_seventy_percent_of_the_epoch_pairs() {
        assert_eq!(MIN_PAIRED_COVERAGE, 0.70);
        let days = MIN_PAIRED_DAYS + 1;
        let tail = tail_at_the_coverage_floor(days);
        let sri =
            sleep_regularity_index(&thin_coverage(days, tail)).expect("the floor itself scores");
        assert!((sri - 100.0).abs() < 1e-9, "got {sri}");
        assert_eq!(
            sleep_regularity_index(&thin_coverage(days, tail + 1)),
            None,
            "one observed pair under the floor must refuse"
        );
        assert!(sleep_regularity_index(&thin_coverage(days, tail - 1)).is_some());
    }

    #[test]
    fn a_single_late_night_dents_but_does_not_destroy_the_index() {
        let sri = sleep_regularity_index(&one_late_night()).unwrap();
        assert!(
            sri > 50.0 && sri < 100.0,
            "one odd night should dent, not destroy: {sri}"
        );
        assert!((sri - 71.42857142857142).abs() < 1e-9, "got {sri}");
    }
}
