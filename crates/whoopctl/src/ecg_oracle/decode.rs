//! Braille (U+2800) DECODER, in dot space only. One way: characters in, dots out.
//!
//! This is the oracle's own reader and it is a deliberate second implementation of the bit table the
//! drawing side holds. Sharing that table would mean a wrong one could never be caught — the encoder
//! and the decoder would agree with each other while both disagreed with Unicode. So the table here is
//! written independently, pinned against literal codepoints, and cross-checked dot-for-dot against the
//! shipping canvas in the test module.
//!
//! Nothing here knows what a dot measures: no millimetres, no seconds, no volts. Row 0 is the TOP.

/// First codepoint of the braille block; the dot bits are added to it.
pub const BRAILLE_BASE: u32 = 0x2800;
/// Dots across one cell.
pub const DOTS_W: usize = 2;
/// Dots down one cell.
pub const DOTS_H: usize = 4;

/// Bit index of dot (row, col) within a cell, row-major from the top.
const BIT: [[u32; DOTS_W]; DOTS_H] = [[0, 3], [1, 4], [2, 5], [6, 7]];

/// The cell bit for a dot position, or `None` outside a cell.
pub fn dot_bit(row: usize, col: usize) -> Option<u32> {
    BIT.get(row)?.get(col).copied()
}

/// Is this character a braille pattern?
pub fn is_braille(ch: char) -> bool {
    (BRAILLE_BASE..BRAILLE_BASE + 0x100).contains(&(ch as u32))
}

/// Decode one character into its eight dot flags, row-major from the top. Non-braille is all-clear.
pub fn cell_dots(ch: char) -> [[bool; DOTS_W]; DOTS_H] {
    let mut out = [[false; DOTS_W]; DOTS_H];
    if !is_braille(ch) {
        return out;
    }
    let bits = ch as u32 - BRAILLE_BASE;
    for (r, row) in BIT.iter().enumerate() {
        for (c, bit) in row.iter().enumerate() {
            out[r][c] = bits & (1 << bit) != 0;
        }
    }
    out
}

/// Which characters of a frame contribute dots.
///
/// A renderer draws its background grid in a styled (dim) span and the trace in the default style, so
/// `Unstyled` isolates the trace without knowing anything about the renderer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Extract {
    /// Every braille cell, styled or not.
    All,
    /// Only cells outside an SGR span — the trace, when the grid is drawn dim.
    Unstyled,
    /// Only cells inside an SGR span — the grid, when the grid is drawn dim.
    Styled,
}

/// A dot bitmap recovered from braille cells. Row 0 is the TOP, so a row index that falls is a trace
/// that rises.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DotGrid {
    width: usize,
    height: usize,
    set: Vec<bool>,
}

impl DotGrid {
    /// An all-clear grid of `width` x `height` dots.
    pub fn new(width: usize, height: usize) -> Self {
        DotGrid { width, height, set: vec![false; width * height] }
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    /// Is the dot at (x, y) set? Out-of-range reads are clear, never a panic.
    pub fn get(&self, x: usize, y: usize) -> bool {
        x < self.width && y < self.height && self.set[y * self.width + x]
    }

    /// Set the dot at (x, y). Out-of-range writes are dropped.
    pub fn plot(&mut self, x: usize, y: usize) {
        if x < self.width && y < self.height {
            self.set[y * self.width + x] = true;
        }
    }

    /// Total dots set.
    pub fn count(&self) -> usize {
        self.set.iter().filter(|s| **s).count()
    }

    /// The set rows of column `x`, top-first.
    pub fn column_rows(&self, x: usize) -> Vec<usize> {
        (0..self.height).filter(|y| self.get(x, *y)).collect()
    }
}

/// Decode the braille cells of a frame into a dot grid, under an extraction policy.
///
/// ANSI escapes are consumed without occupying a column; every other character occupies one column and
/// a non-braille one contributes no dots, so a banner or footer line reads as empty dot rows.
pub fn from_lines(lines: &[String], take: Extract) -> DotGrid {
    let cells: Vec<Vec<(char, bool)>> = lines.iter().map(|l| visible_cells(l)).collect();
    from_cells(&cells, take)
}

/// Decode only a rectangular window of CELLS — the plot region of a frame that also carries a banner,
/// a footer, or several stacked strips.
pub fn from_lines_region(
    lines: &[String],
    rows: std::ops::Range<usize>,
    cols: std::ops::Range<usize>,
    take: Extract,
) -> DotGrid {
    let cells: Vec<Vec<(char, bool)>> = lines
        .get(rows)
        .unwrap_or_default()
        .iter()
        .map(|l| visible_cells(l).into_iter().skip(cols.start).take(cols.len()).collect())
        .collect();
    from_cells(&cells, take)
}

fn from_cells(cells: &[Vec<(char, bool)>], take: Extract) -> DotGrid {
    let cols = cells.iter().map(Vec::len).max().unwrap_or(0);
    let mut grid = DotGrid::new(cols * DOTS_W, cells.len() * DOTS_H);
    for (cr, line) in cells.iter().enumerate() {
        for (cc, (ch, styled)) in line.iter().enumerate() {
            let wanted = match take {
                Extract::All => true,
                Extract::Unstyled => !styled,
                Extract::Styled => *styled,
            };
            if !wanted {
                continue;
            }
            for (r, row) in cell_dots(*ch).iter().enumerate() {
                for (c, on) in row.iter().enumerate() {
                    if *on {
                        grid.plot(cc * DOTS_W + c, cr * DOTS_H + r);
                    }
                }
            }
        }
    }
    grid
}

/// Contiguous runs of lines holding at least one braille cell — the strips of a stacked frame, found
/// structurally rather than from any knowledge of the layout.
pub fn braille_row_blocks(lines: &[String]) -> Vec<std::ops::Range<usize>> {
    let mut out: Vec<std::ops::Range<usize>> = Vec::new();
    let mut start = None;
    for (i, line) in lines.iter().enumerate() {
        let has = visible_cells(line).iter().any(|(ch, _)| is_braille(*ch));
        match (has, start) {
            (true, None) => start = Some(i),
            (false, Some(s)) => {
                out.push(s..i);
                start = None;
            }
            _ => {}
        }
    }
    if let Some(s) = start {
        out.push(s..lines.len());
    }
    out
}

/// Split a line into its visible characters, each flagged with whether an SGR span was open at it.
/// `\x1b[0m` and `\x1b[m` close the span; any other SGR opens one.
fn visible_cells(line: &str) -> Vec<(char, bool)> {
    let mut out = Vec::new();
    let mut styled = false;
    let mut chars = line.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch != '\x1b' {
            out.push((ch, styled));
            continue;
        }
        if chars.peek() != Some(&'[') {
            continue;
        }
        chars.next();
        let mut params = String::new();
        for c in chars.by_ref() {
            if c.is_ascii_alphabetic() {
                if c == 'm' {
                    styled = !matches!(params.as_str(), "" | "0");
                }
                break;
            }
            params.push(c);
        }
    }
    out
}
