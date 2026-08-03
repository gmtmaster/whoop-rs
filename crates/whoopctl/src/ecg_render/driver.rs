//! Where the frame goes. Two modes, and the frame always says which one it chose.
//!
//! A terminal gets an in-place redraw: the cursor walks back up and each row is rewritten, so the
//! trace grows without a clear-screen flicker. Anything else — a pipe, a file, a terminal too short
//! to hold the frame — gets completed strips in order, which is the same picture without the cursor.

use std::io::{IsTerminal, Write};

use super::frame;
use super::plan::{Provenance, Terminal};
use super::renderer::EcgRenderer;

/// How the frame is delivered.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutputMode {
    /// The whole frame is rewritten in place as samples arrive.
    InPlace,
    /// Each strip is printed once it is complete, then the footer.
    Sequential,
}

impl OutputMode {
    pub fn describe(self, why: &str) -> String {
        match self {
            OutputMode::InPlace => format!("in-place progressive redraw ({why})"),
            OutputMode::Sequential => format!("completed strips in order ({why})"),
        }
    }
}

/// Pick the mode, and say why. A frame taller than the terminal cannot be redrawn in place — the
/// cursor would walk off the scrollback — so it falls back rather than corrupting itself.
pub fn choose_mode(term: Terminal, frame_rows: usize, is_tty: bool) -> (OutputMode, String) {
    if !is_tty {
        return (OutputMode::Sequential, "stdout is not a terminal".to_string());
    }
    if frame_rows + 1 > term.rows {
        return (
            OutputMode::Sequential,
            format!("the frame is {frame_rows} rows and the terminal is {}", term.rows),
        );
    }
    (OutputMode::InPlace, format!("stdout is a terminal, {frame_rows} rows fit in {}", term.rows))
}

/// The terminal size, from the flags, then the environment, then a stated assumption. There is no
/// portable way to ask a terminal without linking one, so the fallback is labelled, not hidden.
pub fn terminal(width: Option<usize>, height: Option<usize>) -> Terminal {
    let env = |k: &str| std::env::var(k).ok().and_then(|v| v.trim().parse::<usize>().ok()).filter(|n| *n > 0);
    let pick = |flag: Option<usize>, key: &str, default: usize| match flag.filter(|n| *n > 0) {
        Some(n) => (n, Provenance::Supplied),
        None => match env(key) {
            Some(n) => (n, Provenance::Environment),
            None => (default, Provenance::Assumed),
        },
    };
    let (cols, cols_from) = pick(width, "COLUMNS", 80);
    let (rows, rows_from) = pick(height, "LINES", 24);
    Terminal { cols, rows, cols_from, rows_from }
}

pub fn stdout_is_terminal() -> bool {
    std::io::stdout().is_terminal()
}

/// Paints a renderer's frame, tracking how many rows it last wrote so the next redraw can land on top
/// of them. Holds no canvas of its own.
pub struct Painter {
    mode: OutputMode,
    why: String,
    colour: bool,
    painted_rows: usize,
    strips_emitted: usize,
    banner_emitted: bool,
}

impl Painter {
    pub fn new(mode: OutputMode, why: String, colour: bool) -> Self {
        Painter { mode, why, colour, painted_rows: 0, strips_emitted: 0, banner_emitted: false }
    }

    pub fn mode(&self) -> OutputMode {
        self.mode
    }

    /// Redraw for the current state. In place: rewrite every row. Sequential: emit the banner once,
    /// then any strip that has completed since the last call.
    pub fn paint<W: Write>(&mut self, out: &mut W, r: &EcgRenderer) -> std::io::Result<()> {
        let note = self.mode.describe(&self.why);
        match self.mode {
            OutputMode::InPlace => {
                if self.painted_rows > 0 {
                    write!(out, "\x1b[{}A", self.painted_rows)?;
                }
                let lines = frame::frame(r, self.colour, Some(&note), r.plan().strips);
                for line in &lines {
                    write!(out, "\r\x1b[K{line}\n")?;
                }
                self.painted_rows = lines.len();
            }
            OutputMode::Sequential => {
                if !self.banner_emitted {
                    for line in frame::banner(r.plan()) {
                        writeln!(out, "{line}")?;
                    }
                    self.banner_emitted = true;
                }
                while self.strips_emitted < r.strips_complete() {
                    writeln!(out)?;
                    for line in frame::strip_rows(r, self.strips_emitted, self.colour) {
                        writeln!(out, "{line}")?;
                    }
                    self.strips_emitted += 1;
                }
            }
        }
        out.flush()
    }

    /// The last paint: in place this is one more redraw, sequentially it flushes the strips that never
    /// completed and then the footer.
    pub fn finish<W: Write>(&mut self, out: &mut W, r: &EcgRenderer) -> std::io::Result<()> {
        let note = self.mode.describe(&self.why);
        self.paint(out, r)?;
        if self.mode == OutputMode::Sequential {
            while self.strips_emitted < r.plan().strips {
                writeln!(out)?;
                for line in frame::strip_rows(r, self.strips_emitted, self.colour) {
                    writeln!(out, "{line}")?;
                }
                self.strips_emitted += 1;
            }
            writeln!(out)?;
            for line in frame::footer(r.plan(), &r.report(), Some(&note)) {
                writeln!(out, "{line}")?;
            }
        }
        out.flush()
    }
}
