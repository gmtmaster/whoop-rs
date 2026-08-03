//! Braille dot-matrix canvas for terminal plots. A cell is 2 dots wide x 4 dots tall and a terminal
//! cell is roughly twice as tall as it is wide, so a dot comes out approximately square and one
//! dots-per-unit figure serves both axes. Nothing here is domain-specific: coordinates are dots and
//! a layer's style is an opaque id the caller maps, so no escape sequence is ever baked into the text.

mod canvas;
mod layers;

pub use canvas::Canvas;
pub use layers::{LayerStack, Span, StyleId, BASE_STYLE};

/// Dots per braille cell.
pub const CELL_DOTS_W: usize = 2;
pub const CELL_DOTS_H: usize = 4;

/// The empty pattern. The low 8 bits of a codepoint in this block are its dot mask.
const BRAILLE_BASE: u32 = 0x2800;

/// Dot (x, y) within a cell -> its mask bit. Braille numbers dots down the columns — 1, 2, 3 then 7
/// on the left and 4, 5, 6 then 8 on the right — and dot N is bit N-1, so the fourth row is the
/// high pair and the mask is NOT row-major.
const DOT_BITS: [[u8; CELL_DOTS_H]; CELL_DOTS_W] = [
    [0x01, 0x02, 0x04, 0x40],
    [0x08, 0x10, 0x20, 0x80],
];

/// The mask bit of a dot at (x, y) within its cell.
pub fn dot_bit(x: usize, y: usize) -> u8 {
    DOT_BITS[x % CELL_DOTS_W][y % CELL_DOTS_H]
}

/// The braille character for a dot mask. Every one of the 256 masks lands inside the block.
pub fn mask_char(mask: u8) -> char {
    char::from_u32(BRAILLE_BASE + mask as u32).expect("the braille block holds 256 scalar values")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The table, checked against the dot numbering itself rather than against a copy of the table.
    #[test]
    fn dot_table_matches_the_braille_dot_numbering() {
        let numbering = [
            (0, 0, 1), (0, 1, 2), (0, 2, 3), (0, 3, 7),
            (1, 0, 4), (1, 1, 5), (1, 2, 6), (1, 3, 8),
        ];
        for (x, y, dot) in numbering {
            assert_eq!(dot_bit(x, y), 1u8 << (dot - 1), "dot {dot} sits at ({x}, {y})");
        }
    }

    /// Each dot position pinned to a literal codepoint. `every_pattern_renders_to_its_codepoint`
    /// cannot do this: it builds its mask from the same table it checks, so any bijective table
    /// passes it. Only this test and the one above fail on a row-major table.
    #[test]
    fn each_dot_position_lands_on_its_literal_codepoint() {
        let pinned = [
            (0, 0, '\u{2801}'), (0, 1, '\u{2802}'), (0, 2, '\u{2804}'), (0, 3, '\u{2840}'),
            (1, 0, '\u{2808}'), (1, 1, '\u{2810}'), (1, 2, '\u{2820}'), (1, 3, '\u{2880}'),
        ];
        for (x, y, want) in pinned {
            let mut c = Canvas::new(CELL_DOTS_W, CELL_DOTS_H);
            assert!(c.set(x, y));
            assert_eq!(c.render()[0].chars().next(), Some(want), "the dot at ({x}, {y})");
        }
        // Two dots in the same column, then both columns of the bottom row, then a full cell.
        let combos = [
            (vec![(0, 0), (0, 3)], '\u{2841}'),
            (vec![(0, 3), (1, 3)], '\u{28C0}'),
            (vec![(1, 0), (1, 1), (1, 2)], '\u{2838}'),
            ((0..2).flat_map(|x| (0..4).map(move |y| (x, y))).collect(), '\u{28FF}'),
        ];
        for (dots, want) in combos {
            let mut c = Canvas::new(CELL_DOTS_W, CELL_DOTS_H);
            for (x, y) in &dots {
                assert!(c.set(*x, *y));
            }
            assert_eq!(c.render()[0].chars().next(), Some(want), "dots {dots:?}");
        }
    }

    /// All 256 patterns, each drawn dot by dot on a one-cell canvas, must land on their own
    /// codepoint. Proves mask composition, cell indexing and that all 256 are distinct — but NOT
    /// the table itself; the two tests above pin that.
    #[test]
    fn every_pattern_renders_to_its_codepoint() {
        let mut seen = std::collections::BTreeSet::new();
        for mask in 0u8..=u8::MAX {
            let mut c = Canvas::new(CELL_DOTS_W, CELL_DOTS_H);
            for y in 0..CELL_DOTS_H {
                for x in 0..CELL_DOTS_W {
                    if mask & dot_bit(x, y) != 0 {
                        assert!(c.set(x as i32, y as i32), "({x}, {y}) is inside a full cell");
                    }
                }
            }
            assert_eq!(c.cell_mask(0, 0), mask, "mask {mask:#04x} round-trips through the grid");

            let rows = c.render();
            assert_eq!(rows.len(), 1, "one cell row");
            let ch = rows[0].chars().next().expect("one cell");
            assert_eq!(rows[0].chars().count(), 1, "one cell wide");
            assert_eq!(ch as u32, BRAILLE_BASE + mask as u32, "mask {mask:#04x}");
            assert_eq!(ch, mask_char(mask));
            seen.insert(ch);
        }
        assert_eq!(seen.len(), 256, "every pattern is a distinct character");
    }

    /// A dot's position must survive the round trip in both directions, on a multi-cell grid.
    #[test]
    fn every_dot_of_a_multi_cell_grid_round_trips() {
        let (w, h) = (7, 9);
        for y in 0..h {
            for x in 0..w {
                let mut c = Canvas::new(w, h);
                assert!(c.set(x as i32, y as i32));
                let lit: Vec<(usize, usize)> = (0..h)
                    .flat_map(|yy| (0..w).map(move |xx| (xx, yy)))
                    .filter(|&(xx, yy)| c.get(xx as i32, yy as i32))
                    .collect();
                assert_eq!(lit, vec![(x, y)], "only ({x}, {y}) is lit");
            }
        }
    }
}
