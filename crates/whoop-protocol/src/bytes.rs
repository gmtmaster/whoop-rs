//! Byte-slice helpers shared across the codec: bounds-checked little-endian reads (out-of-range → `None`,
//! so a truncated frame decodes to absent fields), the R-R interval extractor, and lowercase-hex encoding.

use std::fmt::Write;

pub fn u8_at(b: &[u8], i: usize) -> Option<u8> {
    b.get(i).copied()
}

/// A byte where 0 means "field absent" (an unpopulated slot), not a literal zero.
pub fn nonzero_u8_at(b: &[u8], i: usize) -> Option<u8> {
    u8_at(b, i).filter(|&v| v > 0)
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

/// The inner-relative offset-7 unix timestamp — the shared read for every historical/live record.
pub fn unix_at(b: &[u8]) -> Option<u32> {
    u32_at(b, 7)
}

/// A `[marker][body]` buffer — the b3-prefixed command-payload shape.
pub fn prefixed(marker: u8, body: &[u8]) -> Vec<u8> {
    let mut v = Vec::with_capacity(1 + body.len());
    v.push(marker);
    v.extend_from_slice(body);
    v
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

/// Bytes from a hex string (either case, no separators), the inverse of `to_hex`; `None` on an odd length
/// or a non-hex digit.
pub fn from_hex(s: &str) -> Option<Vec<u8>> {
    if !s.len().is_multiple_of(2) {
        return None;
    }
    (0..s.len()).step_by(2).map(|i| u8::from_str_radix(&s[i..i + 2], 16).ok()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hex_round_trips_and_rejects_bad_input() {
        assert_eq!(from_hex("0a1bff").unwrap(), vec![0x0a, 0x1b, 0xff]);
        assert_eq!(from_hex(&to_hex(&[0xde, 0xad, 0xbe, 0xef])).unwrap(), vec![0xde, 0xad, 0xbe, 0xef]);
        assert_eq!(from_hex("0A1BFF").unwrap(), vec![0x0a, 0x1b, 0xff]); // upper-case
        assert!(from_hex("abc").is_none()); // odd length
        assert!(from_hex("zz").is_none()); // non-hex
    }
}
