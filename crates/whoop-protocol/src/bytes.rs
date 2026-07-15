//! Byte-slice helpers shared across the codec: bounds-checked little-endian reads (out-of-range → `None`,
//! so a truncated frame decodes to absent fields), the R-R interval extractor, and lowercase-hex encoding.

use std::fmt::Write;

pub fn u8_at(b: &[u8], i: usize) -> Option<u8> {
    b.get(i).copied()
}

pub fn u16_at(b: &[u8], i: usize) -> Option<u16> {
    b.get(i..i + 2).map(|s| u16::from_le_bytes([s[0], s[1]]))
}

pub fn u32_at(b: &[u8], i: usize) -> Option<u32> {
    b.get(i..i + 4).map(|s| u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
}

pub fn i16_at(b: &[u8], i: usize) -> Option<i16> {
    b.get(i..i + 2).map(|s| i16::from_le_bytes([s[0], s[1]]))
}

pub fn f32_at(b: &[u8], i: usize) -> Option<f32> {
    b.get(i..i + 4).map(|s| f32::from_le_bytes([s[0], s[1], s[2], s[3]]))
}

/// Read the R-R intervals of a record: a `u8` count at `count_off` (clamped to `max` slots), then that
/// many consecutive `u16` LE from `first`, dropping any zero (a zero R-R = an empty slot). Out-of-range
/// reads are skipped. Historical layouts have 4 fixed slots (`max = 4`); the realtime burst is unbounded.
pub fn rr_intervals(b: &[u8], count_off: usize, first: usize, max: usize) -> Vec<u16> {
    let count = (u8_at(b, count_off).unwrap_or(0) as usize).min(max);
    let mut rr = Vec::new();
    for i in 0..count {
        if let Some(v) = u16_at(b, first + i * 2) {
            if v != 0 {
                rr.push(v);
            }
        }
    }
    rr
}

/// Lowercase hex of a byte slice, no separators — the capture-JSONL / debug-frame encoding.
pub fn to_hex(b: &[u8]) -> String {
    let mut s = String::with_capacity(b.len() * 2);
    for x in b {
        let _ = write!(s, "{x:02x}");
    }
    s
}
