//! Composition: canvases stacked back-to-front, each tagged with an opaque style id. A dim grid
//! goes in behind a bright trace and the two compose here. The style never reaches the characters —
//! composition hands back spans, or a parallel style grid, for the caller to wrap.

use super::Canvas;

/// An opaque style tag. The caller maps it to a colour or intensity; nothing here emits an escape.
pub type StyleId = u8;

/// The style of a cell no layer drew in. Callers should map it to "no styling".
pub const BASE_STYLE: StyleId = 0;

/// A run of adjacent cells on one row sharing a style.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Span {
    pub style: StyleId,
    pub text: String,
}

/// Canvases composed back-to-front: the first pushed sits behind, the last in front. A cell shows
/// the union of every layer's dots and the style of the frontmost layer with a dot in it.
#[derive(Clone, Debug)]
pub struct LayerStack {
    dot_w: usize,
    dot_h: usize,
    layers: Vec<(StyleId, Canvas)>,
}

impl LayerStack {
    /// An empty stack sized in DOTS. Composition is done at this size whatever the layers measure.
    pub fn new(dot_w: usize, dot_h: usize) -> Self {
        Self { dot_w, dot_h, layers: Vec::new() }
    }

    /// Add a drawn canvas as the new front layer, returning its index. A layer of a different size
    /// contributes only over the region it shares with the stack.
    pub fn push(&mut self, style: StyleId, canvas: Canvas) -> usize {
        self.layers.push((style, canvas));
        self.layers.len() - 1
    }

    /// Add an empty front layer at the stack's own size, returning its index. Pair with
    /// `layer_mut` to draw into it over time, as a progressive trace does.
    pub fn push_blank(&mut self, style: StyleId) -> usize {
        self.push(style, Canvas::new(self.dot_w, self.dot_h))
    }

    /// A layer by index, for reading.
    pub fn layer(&self, idx: usize) -> Option<&Canvas> {
        self.layers.get(idx).map(|(_, c)| c)
    }

    /// A layer by index, for drawing into after it was pushed.
    pub fn layer_mut(&mut self, idx: usize) -> Option<&mut Canvas> {
        self.layers.get_mut(idx).map(|(_, c)| c)
    }

    /// The style a layer was pushed with.
    pub fn style_of(&self, idx: usize) -> Option<StyleId> {
        self.layers.get(idx).map(|(s, _)| *s)
    }

    /// How many layers are stacked.
    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    /// Size in dots, as constructed.
    pub fn dot_size(&self) -> (usize, usize) {
        (self.dot_w, self.dot_h)
    }

    /// Size in terminal cells: columns and rows of the rendered output.
    pub fn cell_size(&self) -> (usize, usize) {
        Canvas::new(self.dot_w, self.dot_h).cell_size()
    }

    /// Every layer's dots unioned into one canvas of the stack's size.
    pub fn flatten(&self) -> Canvas {
        let mut out = Canvas::new(self.dot_w, self.dot_h);
        for (_, layer) in &self.layers {
            out.overlay(layer);
        }
        out
    }

    /// One unstyled string per terminal row — the pipeable form.
    pub fn render(&self) -> Vec<String> {
        self.flatten().render()
    }

    /// One style per cell, in the same shape as `render`. A cell no layer drew in is `BASE_STYLE`.
    pub fn style_grid(&self) -> Vec<Vec<StyleId>> {
        let (cell_w, cell_h) = self.cell_size();
        (0..cell_h)
            .map(|cy| {
                (0..cell_w)
                    .map(|cx| {
                        self.layers
                            .iter()
                            .rev()
                            .find(|(_, l)| l.cell_mask(cx, cy) != 0)
                            .map_or(BASE_STYLE, |(s, _)| *s)
                    })
                    .collect()
            })
            .collect()
    }

    /// One row of style-tagged runs per terminal row. Concatenating a row's `text` gives exactly
    /// the same characters as `render`, so styling can never change what was drawn.
    pub fn render_spans(&self) -> Vec<Vec<Span>> {
        let (flat, styles) = (self.flatten(), self.style_grid());
        styles
            .iter()
            .enumerate()
            .map(|(cy, row_styles)| {
                let mut row: Vec<Span> = Vec::new();
                for (cx, &style) in row_styles.iter().enumerate() {
                    let ch = flat.cell_char(cx, cy);
                    match row.last_mut() {
                        Some(span) if span.style == style => span.text.push(ch),
                        _ => row.push(Span { style, text: ch.to_string() }),
                    }
                }
                row
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GRID_STYLE: StyleId = 1;
    const TRACE: StyleId = 2;

    /// A dim 5-dot grid behind a bright trace: the classic two-layer case. The height is chosen so
    /// one cell row falls between two grid lines, leaving cells no layer draws in.
    fn grid_and_trace() -> LayerStack {
        let (w, h) = (40, 24);
        let mut stack = LayerStack::new(w, h);
        let mut grid = Canvas::new(w, h);
        for x in (0..w as i32).step_by(5) {
            grid.line(x, 0, x, h as i32 - 1);
        }
        for y in (0..h as i32).step_by(5) {
            grid.line(0, y, w as i32 - 1, y);
        }
        let mut trace = Canvas::new(w, h);
        trace.polyline(&[(0, 12), (10, 12), (12, 4), (14, 21), (16, 12), (39, 12)]);
        stack.push(GRID_STYLE, grid);
        stack.push(TRACE, trace);
        stack
    }

    #[test]
    fn an_empty_stack_renders_blank_rows_of_the_right_shape() {
        let stack = LayerStack::new(40, 16);
        assert_eq!(stack.layer_count(), 0);
        assert_eq!(stack.cell_size(), (20, 4));
        assert_eq!(stack.render(), vec!["\u{2800}".repeat(20); 4]);
        assert_eq!(stack.style_grid(), vec![vec![BASE_STYLE; 20]; 4]);
        assert_eq!(stack.flatten().dots_set(), 0);
    }

    #[test]
    fn push_returns_an_index_that_reads_back() {
        let mut stack = LayerStack::new(20, 8);
        let a = stack.push_blank(GRID_STYLE);
        let b = stack.push_blank(TRACE);
        assert_eq!((a, b), (0, 1));
        assert_eq!(stack.style_of(a), Some(GRID_STYLE));
        assert_eq!(stack.style_of(b), Some(TRACE));
        assert_eq!(stack.style_of(2), None);
        assert!(stack.layer(2).is_none() && stack.layer_mut(2).is_none());
        assert_eq!(stack.layer(a).map(Canvas::dot_size), Some((20, 8)));
    }

    #[test]
    fn a_layer_can_be_drawn_into_after_it_is_pushed() {
        let mut stack = LayerStack::new(20, 8);
        let trace = stack.push_blank(TRACE);
        for x in 0..20i32 {
            stack.layer_mut(trace).expect("index came from push").set(x, 4);
        }
        assert_eq!(stack.flatten().dots_set(), 20);
    }

    #[test]
    fn flatten_unions_every_layer() {
        let stack = grid_and_trace();
        let (grid, trace) = (stack.layer(0).unwrap(), stack.layer(1).unwrap());
        let (flat, (w, h)) = (stack.flatten(), stack.dot_size());
        for y in 0..h as i32 {
            for x in 0..w as i32 {
                assert_eq!(flat.get(x, y), grid.get(x, y) || trace.get(x, y), "({x}, {y})");
            }
        }
        assert!(flat.dots_set() > 0);
    }

    #[test]
    fn the_frontmost_layer_with_a_dot_owns_the_cell_style() {
        let stack = grid_and_trace();
        let styles = stack.style_grid();
        let (grid, trace) = (stack.layer(0).unwrap(), stack.layer(1).unwrap());
        let (cell_w, cell_h) = stack.cell_size();
        assert_eq!(styles.len(), cell_h);
        for (cy, row) in styles.iter().enumerate() {
            assert_eq!(row.len(), cell_w);
            for (cx, &got) in row.iter().enumerate() {
                let want = match (trace.cell_mask(cx, cy) != 0, grid.cell_mask(cx, cy) != 0) {
                    (true, _) => TRACE,
                    (false, true) => GRID_STYLE,
                    (false, false) => BASE_STYLE,
                };
                assert_eq!(got, want, "cell ({cx}, {cy})");
            }
        }
        // The trace really does cover grid cells, or the test above proves nothing.
        assert!(styles.iter().flatten().any(|&s| s == TRACE));
        assert!(styles.iter().flatten().any(|&s| s == GRID_STYLE));
        assert!(styles.iter().flatten().any(|&s| s == BASE_STYLE));
    }

    #[test]
    fn spans_carry_the_same_characters_as_a_plain_render() {
        let stack = grid_and_trace();
        let plain = stack.render();
        let spans = stack.render_spans();
        assert_eq!(spans.len(), plain.len());
        for (row, line) in spans.iter().zip(&plain) {
            assert_eq!(row.iter().map(|s| s.text.as_str()).collect::<String>(), *line);
            assert!(row.iter().all(|s| !s.text.is_empty()), "no empty span");
            for pair in row.windows(2) {
                assert_ne!(pair[0].style, pair[1].style, "adjacent runs must differ");
            }
        }
    }

    #[test]
    fn spans_hold_no_escape_sequences() {
        for row in grid_and_trace().render_spans() {
            for span in row {
                assert!(span.text.chars().all(|c| ('\u{2800}'..='\u{28FF}').contains(&c)), "{:?}", span.text);
            }
        }
    }

    #[test]
    fn a_span_style_matches_the_style_grid_cell_by_cell() {
        let stack = grid_and_trace();
        let styles = stack.style_grid();
        for (cy, row) in stack.render_spans().iter().enumerate() {
            let flat: Vec<StyleId> = row.iter().flat_map(|s| std::iter::repeat_n(s.style, s.text.chars().count())).collect();
            assert_eq!(flat, styles[cy], "row {cy}");
        }
    }

    #[test]
    fn a_layer_smaller_than_the_stack_contributes_over_its_own_area_only() {
        let mut stack = LayerStack::new(40, 16);
        let mut small = Canvas::new(8, 8);
        small.line(0, 0, 7, 7);
        stack.push(TRACE, small);
        assert_eq!(stack.cell_size(), (20, 4));
        assert_eq!(stack.flatten().dots_set(), 8);
        let styles = stack.style_grid();
        assert!(styles[0][0] == TRACE && styles[3][19] == BASE_STYLE);
    }

    #[test]
    fn a_layer_larger_than_the_stack_is_cropped() {
        let mut stack = LayerStack::new(8, 8);
        let mut big = Canvas::new(40, 16);
        big.line(0, 0, 39, 15);
        stack.push(TRACE, big);
        assert_eq!(stack.render().len(), 2);
        assert!(stack.render().iter().all(|r| r.chars().count() == 4));
        assert!(stack.flatten().dots_set() > 0);
    }

    #[test]
    fn a_zero_size_stack_composes_to_nothing() {
        let mut stack = LayerStack::new(0, 0);
        stack.push_blank(TRACE);
        assert_eq!(stack.cell_size(), (0, 0));
        assert!(stack.render().is_empty());
        assert!(stack.style_grid().is_empty());
        assert!(stack.render_spans().is_empty());
    }
}
