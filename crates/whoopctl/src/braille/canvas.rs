//! The dot grid: set/clear/test a dot, draw a clipped polyline, render braille rows. Sits under
//! `LayerStack`, which composes several of these.

use super::{dot_bit, mask_char, CELL_DOTS_H, CELL_DOTS_W};

/// A grid of braille dots, (0, 0) at the top left. Every write outside the grid is clipped, never a
/// panic, and the grid is stored one mask byte per cell so rendering is a straight scan.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Canvas {
    dot_w: usize,
    dot_h: usize,
    cell_w: usize,
    cell_h: usize,
    cells: Vec<u8>,
}

impl Canvas {
    /// A blank canvas sized in DOTS. The cell grid is the dot grid rounded up, so a partial cell on
    /// the right or bottom edge still exists and its surplus dots are unreachable.
    pub fn new(dot_w: usize, dot_h: usize) -> Self {
        let (cell_w, cell_h) = (dot_w.div_ceil(CELL_DOTS_W), dot_h.div_ceil(CELL_DOTS_H));
        Self { dot_w, dot_h, cell_w, cell_h, cells: vec![0; cell_w * cell_h] }
    }

    /// Size in dots, as constructed.
    pub fn dot_size(&self) -> (usize, usize) {
        (self.dot_w, self.dot_h)
    }

    /// Size in terminal cells: columns and rows of the rendered output.
    pub fn cell_size(&self) -> (usize, usize) {
        (self.cell_w, self.cell_h)
    }

    /// Light a dot. False means it fell outside the grid and nothing was drawn.
    pub fn set(&mut self, x: i32, y: i32) -> bool {
        self.set_at(x as i64, y as i64)
    }

    /// Unlight a dot. False means it fell outside the grid.
    pub fn clear(&mut self, x: i32, y: i32) -> bool {
        match self.index(x as i64, y as i64) {
            Some((i, bit)) => {
                self.cells[i] &= !bit;
                true
            }
            None => false,
        }
    }

    /// Whether a dot is lit. Anything outside the grid is unlit.
    pub fn get(&self, x: i32, y: i32) -> bool {
        self.index(x as i64, y as i64).is_some_and(|(i, bit)| self.cells[i] & bit != 0)
    }

    /// Unlight every dot, keeping the size.
    pub fn clear_all(&mut self) {
        self.cells.fill(0);
    }

    /// How many dots are lit.
    pub fn dots_set(&self) -> usize {
        self.cells.iter().map(|c| c.count_ones() as usize).sum()
    }

    /// Draw the 8-connected line between two dots, clipped to the grid. Work is bounded by the
    /// canvas, not by how far outside it the endpoints are.
    pub fn line(&mut self, x0: i32, y0: i32, x1: i32, y1: i32) {
        if self.dot_w == 0 || self.dot_h == 0 {
            return;
        }
        let (x0, y0) = (i64::from(x0), i64::from(y0));
        let (dx, dy) = (i64::from(x1) - x0, i64::from(y1) - y0);
        let n = dx.abs().max(dy.abs());
        if n == 0 {
            self.set_at(x0, y0);
            return;
        }
        let (Some((sx0, sx1)), Some((sy0, sy1))) = (
            step_span(x0, dx, n, self.dot_w as i64 - 1),
            step_span(y0, dy, n, self.dot_h as i64 - 1),
        ) else {
            return;
        };
        for i in sx0.max(sy0)..=sx1.min(sy1) {
            self.set_at(x0 + step_offset(i, dx, n), y0 + step_offset(i, dy, n));
        }
    }

    /// Draw a run of points as connected line segments. A bare scatter of dots reads as broken at
    /// any real sweep speed, so the trace is always a polyline.
    pub fn polyline(&mut self, points: &[(i32, i32)]) {
        match points {
            [] => {}
            [(x, y)] => {
                self.set(*x, *y);
            }
            _ => {
                for pair in points.windows(2) {
                    self.line(pair[0].0, pair[0].1, pair[1].0, pair[1].1);
                }
            }
        }
    }

    /// OR another canvas's dots into this one over the region they share. Sizes need not match.
    pub fn overlay(&mut self, other: &Canvas) {
        for cy in 0..self.cell_h.min(other.cell_h) {
            for cx in 0..self.cell_w.min(other.cell_w) {
                self.cells[cy * self.cell_w + cx] |= other.cells[cy * other.cell_w + cx];
            }
        }
    }

    /// The raw dot mask of a cell. Cells outside the grid read as empty.
    pub fn cell_mask(&self, cx: usize, cy: usize) -> u8 {
        if cx >= self.cell_w || cy >= self.cell_h {
            return 0;
        }
        self.cells[cy * self.cell_w + cx]
    }

    /// The braille character of a cell. Cells outside the grid read as blank.
    pub fn cell_char(&self, cx: usize, cy: usize) -> char {
        mask_char(self.cell_mask(cx, cy))
    }

    /// One string per terminal row, each exactly `cell_size().0` characters. Empty cells are the
    /// blank braille pattern, not a space, so columns stay aligned; nothing is trimmed and no
    /// escape is emitted.
    pub fn render(&self) -> Vec<String> {
        (0..self.cell_h)
            .map(|cy| (0..self.cell_w).map(|cx| self.cell_char(cx, cy)).collect())
            .collect()
    }

    fn set_at(&mut self, x: i64, y: i64) -> bool {
        match self.index(x, y) {
            Some((i, bit)) => {
                self.cells[i] |= bit;
                true
            }
            None => false,
        }
    }

    /// Cell index and mask bit of a dot, or None if it is outside the grid.
    fn index(&self, x: i64, y: i64) -> Option<(usize, u8)> {
        let (x, y) = (usize::try_from(x).ok()?, usize::try_from(y).ok()?);
        if x >= self.dot_w || y >= self.dot_h {
            return None;
        }
        Some((y / CELL_DOTS_H * self.cell_w + x / CELL_DOTS_W, dot_bit(x, y)))
    }
}

/// The i-th step's offset along one axis, `i` in `[0, n]` and `n = max(|dx|, |dy|)`. Rounding half
/// away from zero keeps the major axis exact and the minor axis moving by 0 or 1, so the line is
/// 8-connected.
fn step_offset(i: i64, d: i64, n: i64) -> i64 {
    let (num, den) = (i128::from(i) * i128::from(d), i128::from(n));
    let q = if num >= 0 { (2 * num + den) / (2 * den) } else { -((-2 * num + den) / (2 * den)) };
    q as i64
}

/// The steps whose offset can land within `[0, lim]`, clamped to `[0, n]`. Widened by one step per
/// side so rounding can never clip a visible dot; the caller still clips each dot it draws.
fn step_span(v0: i64, d: i64, n: i64, lim: i64) -> Option<(i64, i64)> {
    if lim < 0 {
        return None;
    }
    if d == 0 {
        return (0..=lim).contains(&v0).then_some((0, n));
    }
    let (n, d) = (i128::from(n), i128::from(d));
    let (lo_v, hi_v) = (i128::from(-v0) - 1, i128::from(lim - v0) + 1);
    let (lo, hi) = if d > 0 {
        (floor_div(lo_v * n, d), ceil_div(hi_v * n, d))
    } else {
        (floor_div(hi_v * n, d), ceil_div(lo_v * n, d))
    };
    let (lo, hi) = (lo.max(0), hi.min(n));
    (lo <= hi).then_some((lo as i64, hi as i64))
}

fn floor_div(num: i128, den: i128) -> i128 {
    let (q, r) = (num / den, num % den);
    if r != 0 && (r < 0) != (den < 0) { q - 1 } else { q }
}

fn ceil_div(num: i128, den: i128) -> i128 {
    let (q, r) = (num / den, num % den);
    if r != 0 && (r < 0) == (den < 0) { q + 1 } else { q }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every lit dot of a canvas, in row-major order.
    fn lit(c: &Canvas) -> Vec<(i32, i32)> {
        let (w, h) = c.dot_size();
        (0..h as i32)
            .flat_map(|y| (0..w as i32).map(move |x| (x, y)))
            .filter(|&(x, y)| c.get(x, y))
            .collect()
    }

    #[test]
    fn dot_size_rounds_up_to_whole_cells() {
        for (dots, cells) in [((0, 0), (0, 0)), ((1, 1), (1, 1)), ((2, 4), (1, 1)), ((3, 5), (2, 2)), ((80, 12), (40, 3))] {
            let c = Canvas::new(dots.0, dots.1);
            assert_eq!(c.dot_size(), dots);
            assert_eq!(c.cell_size(), cells, "{dots:?} dots");
            assert_eq!(c.render().len(), cells.1);
            assert!(c.render().iter().all(|r| r.chars().count() == cells.0));
        }
    }

    #[test]
    fn a_zero_size_canvas_draws_nothing_and_never_panics() {
        for dims in [(0, 0), (0, 8), (8, 0)] {
            let mut c = Canvas::new(dims.0, dims.1);
            assert!(!c.set(0, 0));
            assert!(!c.clear(0, 0));
            assert!(!c.get(0, 0));
            c.line(-5, -5, 5, 5);
            c.polyline(&[(0, 0), (3, 3), (9, 1)]);
            c.overlay(&Canvas::new(16, 16));
            assert_eq!(c.dots_set(), 0);
            assert!(c.render().iter().all(String::is_empty) || c.render().is_empty());
        }
    }

    #[test]
    fn out_of_bounds_writes_are_clipped_not_panics() {
        let mut c = Canvas::new(6, 6);
        let outside = [(-1, 0), (0, -1), (6, 0), (0, 6), (i32::MIN, i32::MIN), (i32::MAX, i32::MAX), (-1, -1)];
        for (x, y) in outside {
            assert!(!c.set(x, y), "({x}, {y}) is outside");
            assert!(!c.clear(x, y), "({x}, {y}) is outside");
            assert!(!c.get(x, y), "({x}, {y}) is outside");
        }
        assert_eq!(c.dots_set(), 0);
        assert!(c.set(5, 5) && c.get(5, 5) && c.dots_set() == 1);
        assert!(c.clear(5, 5) && !c.get(5, 5) && c.dots_set() == 0);
    }

    /// The surplus dots of a partial edge cell must stay unreachable, or a write would appear to
    /// succeed and land off the declared grid.
    #[test]
    fn a_partial_edge_cell_exposes_no_extra_dots() {
        let mut c = Canvas::new(3, 5);
        assert_eq!(c.cell_size(), (2, 2));
        assert!(!c.set(3, 0), "dot column 3 is padding");
        assert!(!c.set(0, 5), "dot row 5 is padding");
        assert_eq!(c.dots_set(), 0);
    }

    #[test]
    fn clear_all_keeps_the_size() {
        let mut c = Canvas::new(9, 9);
        c.line(0, 0, 8, 8);
        assert!(c.dots_set() > 0);
        c.clear_all();
        assert_eq!(c.dots_set(), 0);
        assert_eq!(c.cell_size(), (5, 3));
    }

    #[test]
    fn a_line_is_eight_connected_and_hits_both_ends() {
        let cases = [(0, 0, 19, 3), (0, 3, 19, 0), (19, 3, 0, 0), (0, 0, 3, 19), (0, 0, 19, 19), (7, 2, 7, 17), (2, 7, 17, 7)];
        for (x0, y0, x1, y1) in cases {
            let mut c = Canvas::new(20, 20);
            c.line(x0, y0, x1, y1);
            assert!(c.get(x0, y0) && c.get(x1, y1), "endpoints of {:?}", (x0, y0, x1, y1));
            let pts = lit(&c);
            // one dot per major-axis step, each 8-adjacent to the last
            let n = (x1 - x0).abs().max((y1 - y0).abs());
            assert_eq!(pts.len() as i32, n + 1, "step count of {:?}", (x0, y0, x1, y1));
            let mut ordered = pts.clone();
            ordered.sort_by_key(|&(x, y)| if (x1 - x0).abs() >= (y1 - y0).abs() { (x, y) } else { (y, x) });
            for w in ordered.windows(2) {
                let (dx, dy) = ((w[1].0 - w[0].0).abs(), (w[1].1 - w[0].1).abs());
                assert!(dx <= 1 && dy <= 1 && dx + dy > 0, "gap between {:?} and {:?}", w[0], w[1]);
            }
        }
    }

    #[test]
    fn a_degenerate_line_is_a_single_dot() {
        let mut c = Canvas::new(8, 8);
        c.line(3, 4, 3, 4);
        assert_eq!(lit(&c), vec![(3, 4)]);
    }

    /// The exact dots of two known segments. The clip tests are self-consistency checks and pass
    /// under any monotone rounding rule, so the rule itself is pinned here.
    #[test]
    fn line_rasterisation_is_pinned() {
        let mut steep = Canvas::new(16, 24);
        steep.line(11, 10, 12, 20);
        assert_eq!(
            lit(&steep),
            vec![(11, 10), (11, 11), (11, 12), (11, 13), (11, 14), (12, 15), (12, 16), (12, 17), (12, 18), (12, 19), (12, 20)],
        );

        let mut shallow = Canvas::new(16, 8);
        shallow.line(0, 0, 10, 3);
        assert_eq!(
            lit(&shallow),
            vec![(0, 0), (1, 0), (2, 1), (3, 1), (4, 1), (5, 2), (6, 2), (7, 2), (8, 2), (9, 3), (10, 3)],
        );
    }

    /// `line` skips steps that cannot land on the grid, so it must agree exactly with a reference
    /// that walks every step and clips each dot on its own. Rounding is shared with the reference on
    /// purpose — what is under test is which steps get visited, which is where a fast clip loses dots.
    #[test]
    fn line_agrees_with_a_brute_force_reference() {
        fn lcg(state: &mut u64) -> u64 {
            *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            *state >> 33
        }
        fn reference(w: usize, h: usize, x0: i32, y0: i32, x1: i32, y1: i32) -> Canvas {
            let mut c = Canvas::new(w, h);
            let (dx, dy) = (i64::from(x1) - i64::from(x0), i64::from(y1) - i64::from(y0));
            let n = dx.abs().max(dy.abs()).max(1);
            for i in 0..=n {
                c.set_at(i64::from(x0) + step_offset(i, dx, n), i64::from(y0) + step_offset(i, dy, n));
            }
            c
        }
        let (mut seed, mut checked) = (0x2800u64, 0usize);
        for _ in 0..20_000 {
            let (w, h) = (1 + lcg(&mut seed) as usize % 12, 1 + lcg(&mut seed) as usize % 12);
            let mut coord = || -60 + (lcg(&mut seed) % 121) as i32;
            let (x0, y0, x1, y1) = (coord(), coord(), coord(), coord());
            let mut fast = Canvas::new(w, h);
            fast.line(x0, y0, x1, y1);
            assert_eq!(fast, reference(w, h, x0, y0, x1, y1), "{w}x{h} segment {:?}", (x0, y0, x1, y1));
            checked += 1;
        }
        assert_eq!(checked, 20_000);
    }

    /// Clipping must remove exactly the dots that fall outside and no others: the same segment on a
    /// big canvas, cropped, has to equal the small canvas. The last four are counterexamples found
    /// by search — each loses an edge dot if the step range is not widened for rounding.
    #[test]
    fn clipping_keeps_exactly_the_visible_dots() {
        let (w, h) = (23, 17);
        let cases = [
            (-40, -30, 60, 40), (-40, 8, 60, 8), (11, -50, 11, 50), (-5, 20, 30, -9),
            (100, 100, -100, -100), (-1, -1, 1, 1), (0, 40, 22, -20), (-200, 3, 5, 3),
            (-2, 3, 48, -12), (4, 57, 1, -34), (29, 2, -29, 3), (1, 0, 44, -21),
        ];
        for (x0, y0, x1, y1) in cases {
            let mut small = Canvas::new(w, h);
            small.line(x0, y0, x1, y1);

            // Redraw on a canvas big enough to hold the whole segment, offset so nothing clips.
            let (off_x, off_y) = (200, 200);
            let mut big = Canvas::new(w + 2 * off_x, h + 2 * off_y);
            big.line(x0 + off_x as i32, y0 + off_y as i32, x1 + off_x as i32, y1 + off_y as i32);
            let expected: Vec<(i32, i32)> = lit(&big)
                .into_iter()
                .map(|(x, y)| (x - off_x as i32, y - off_y as i32))
                .filter(|&(x, y)| (0..w as i32).contains(&x) && (0..h as i32).contains(&y))
                .collect();

            assert_eq!(lit(&small), expected, "segment {:?}", (x0, y0, x1, y1));
        }
    }

    /// A segment spanning the whole i32 range must terminate on canvas-sized work, not iterate it.
    #[test]
    fn an_enormous_segment_terminates() {
        let mut c = Canvas::new(40, 12);
        c.line(i32::MIN, 6, i32::MAX, 6);
        assert_eq!(lit(&c), (0..40).map(|x| (x, 6)).collect::<Vec<_>>());

        let mut c = Canvas::new(40, 12);
        c.line(i32::MIN, i32::MIN, i32::MAX, i32::MAX);
        assert!(c.dots_set() > 0, "the diagonal crosses the grid");

        let mut c = Canvas::new(40, 12);
        c.line(i32::MIN, i32::MIN, i32::MAX / 2, i32::MIN + 1);
        assert_eq!(c.dots_set(), 0, "that segment never reaches the grid");
    }

    #[test]
    fn a_canvas_larger_than_any_terminal_renders_whole() {
        let mut c = Canvas::new(4000, 400);
        assert_eq!(c.cell_size(), (2000, 100));
        c.line(0, 0, 3999, 399);
        let rows = c.render();
        assert_eq!(rows.len(), 100);
        assert!(rows.iter().all(|r| r.chars().count() == 2000));
        assert_eq!(c.dots_set(), 4000);
    }

    #[test]
    fn polyline_joins_its_points_and_tolerates_short_input() {
        let mut c = Canvas::new(16, 16);
        c.polyline(&[]);
        assert_eq!(c.dots_set(), 0);
        c.polyline(&[(2, 2)]);
        assert_eq!(lit(&c), vec![(2, 2)]);

        let mut joined = Canvas::new(16, 16);
        joined.polyline(&[(0, 0), (5, 5), (10, 0), (15, 8)]);
        let mut segments = Canvas::new(16, 16);
        segments.line(0, 0, 5, 5);
        segments.line(5, 5, 10, 0);
        segments.line(10, 0, 15, 8);
        assert_eq!(joined, segments);
    }

    #[test]
    fn overlay_unions_dots_over_the_shared_region() {
        let mut base = Canvas::new(16, 8);
        base.line(0, 0, 15, 0);
        let mut top = Canvas::new(16, 8);
        top.line(0, 4, 15, 4);
        base.overlay(&top);
        assert_eq!(base.dots_set(), 32);
        assert!(base.get(7, 0) && base.get(7, 4));

        // A smaller layer contributes only where it exists, and a larger one is cropped.
        let mut small = Canvas::new(4, 4);
        small.set(1, 1);
        let mut wide = Canvas::new(16, 8);
        wide.overlay(&small);
        assert_eq!(lit(&wide), vec![(1, 1)]);
        let mut narrow = Canvas::new(4, 4);
        narrow.overlay(&base);
        assert_eq!(narrow.dots_set(), 4, "only the first cell column and row survive");
    }

    #[test]
    fn cells_outside_the_grid_read_as_blank() {
        let c = Canvas::new(4, 4);
        assert_eq!(c.cell_mask(99, 0), 0);
        assert_eq!(c.cell_mask(0, 99), 0);
        assert_eq!(c.cell_char(99, 99), '\u{2800}');
    }

    #[test]
    fn rendered_rows_are_pure_braille_with_no_escapes() {
        let mut c = Canvas::new(60, 16);
        c.polyline(&[(0, 8), (15, 1), (30, 15), (45, 4), (59, 8)]);
        for row in c.render() {
            assert!(row.chars().all(|ch| ('\u{2800}'..='\u{28FF}').contains(&ch)), "row: {row}");
            assert!(!row.contains('\u{1b}'));
        }
    }

    /// A square wave of a known height and period must measure that height and period back off the
    /// grid. This is the offline stand-in for a calibration signal: the scale is checked against the
    /// drawn dots, never against the arithmetic that placed them.
    #[test]
    fn a_square_wave_measures_its_own_height_and_period() {
        let (amp, half, width, height) = (10i32, 12i32, 72i32, 30i32); // 10 dots tall, 24 per cycle
        let mid = height / 2;
        let (top, bottom) = (mid - amp / 2, mid + amp / 2);
        let mut c = Canvas::new(width as usize, height as usize);
        c.polyline(
            &(0..width)
                .map(|i| (i, if (i / half) % 2 == 0 { top } else { bottom }))
                .collect::<Vec<_>>(),
        );
        let column = |x: i32| (0..height).filter(|&y| c.get(x, y)).collect::<Vec<i32>>();
        let extent = |ys: &[i32]| ys.last().copied().unwrap_or(0) - ys.first().copied().unwrap_or(0);

        // Height: peak to peak over the whole trace, and again across each transition on its own.
        let all: Vec<i32> = (0..width).flat_map(&column).collect();
        assert_eq!((all.iter().max(), all.iter().min()), (Some(&bottom), Some(&top)));
        assert_eq!(extent(&[top, bottom]), amp, "the wave stands exactly the amplitude tall");

        // A near-vertical edge spans two columns, so a transition is a RUN of multi-dot columns.
        let multi: Vec<i32> = (0..width).filter(|&x| column(x).len() > 1).collect();
        let starts: Vec<i32> = multi.iter().copied().filter(|x| !multi.contains(&(x - 1))).collect();
        assert_eq!(starts.len(), 5, "five edges fit in three cycles: {multi:?}");
        for edge in &starts {
            let mut pair = column(*edge);
            pair.extend(column(edge + 1));
            pair.sort_unstable();
            assert_eq!(extent(&pair), amp, "edge at {edge} rises the full amplitude");
        }
        for pair in starts.windows(2) {
            assert_eq!(pair[1] - pair[0], half, "half a period between edges");
        }

        // Period: the flats sit at alternating levels, and repeat one full cycle later.
        for k in 0..width / half {
            let x = k * half + half / 2;
            let want = if k % 2 == 0 { top } else { bottom };
            assert_eq!(column(x), vec![want], "flat run {k} at x={x}");
            if x + 2 * half < width {
                assert_eq!(column(x + 2 * half), vec![want], "one cycle after x={x}");
            }
        }
    }
}
