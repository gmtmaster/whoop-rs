//! Where a sample sits inside a buffer of raw packet bytes, and the enumeration of the plausible
//! answers. A [`Layout`] is a pure reading rule — a field width, a signedness, a bit order, a first-bit
//! position and a bit stride — applied to bytes the caller has already collected.
//!
//! Widths are not assumed byte-aligned. An 18-bit converter word is either packed densely, so the
//! second sample starts inside a byte, or carried in the next byte-aligned container with the data at
//! one end of it; both are enumerated. The bit order covers byte endianness for free: reading a
//! byte-aligned field LSB-first out of consecutive bytes reconstructs a little-endian word and MSB-first
//! a big-endian one, so one model spans both.
//!
//! **No amplitude scale is applied or implied.** [`Layout::decode`] returns raw converter counts as
//! `f64`. Nothing in this module knows what a count is worth.

/// Which end of the field is read first. For a byte-aligned width this is exactly byte endianness.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum BitOrder {
    /// Least-significant bit first — little-endian for a byte-aligned field.
    LsbFirst,
    /// Most-significant bit first — big-endian for a byte-aligned field.
    MsbFirst,
}

/// The structure of a reading rule without its phase. Two windows of the same stream share a shape but
/// need not share a `start_bit`: where the first whole sample begins depends on where the buffer was
/// cut, so phase is a property of the buffer and shape is the property of the stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct LayoutShape {
    pub bits: u8,
    pub signed: bool,
    pub order: BitOrder,
    pub stride_bits: usize,
}

/// One complete reading rule.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Layout {
    /// Width of the sample field in bits.
    pub bits: u8,
    /// Two's complement when set, plain unsigned otherwise.
    pub signed: bool,
    pub order: BitOrder,
    /// Bit position of the first sample's first bit, counted from the start of the buffer.
    pub start_bit: usize,
    /// Bit distance between consecutive samples; equal to `bits` for dense packing, larger when the
    /// field sits in a wider container or the stream interleaves other channels.
    pub stride_bits: usize,
}

/// Sample widths searched. 16/24/32 are the byte-aligned converter words; 18 is the non-aligned one,
/// and it is in the list precisely because assuming byte alignment is how it would be missed.
pub const WIDTHS_BITS: [u8; 4] = [16, 18, 24, 32];
/// Longest packet header the first sample is allowed to sit behind, in bytes.
pub const MAX_HEADER_BYTES: usize = 8;
/// Most channels a frame may interleave before the wanted one repeats.
pub const MAX_INTERLEAVE: usize = 4;

impl Layout {
    pub fn shape(&self) -> LayoutShape {
        LayoutShape { bits: self.bits, signed: self.signed, order: self.order, stride_bits: self.stride_bits }
    }

    /// How many whole samples this rule reads out of `len_bytes`.
    pub fn sample_count(&self, len_bytes: usize) -> usize {
        let bits = self.bits as usize;
        let total = len_bytes.saturating_mul(8);
        if bits == 0 || self.stride_bits < bits || self.start_bit + bits > total {
            return 0;
        }
        1 + (total - self.start_bit - bits) / self.stride_bits
    }

    /// Every whole sample, as raw converter counts. Never panics: a rule that does not fit the buffer
    /// reads nothing.
    pub fn decode(&self, bytes: &[u8]) -> Vec<f64> {
        let n = self.sample_count(bytes.len());
        (0..n).map(|i| self.value_at(bytes, self.start_bit + i * self.stride_bits)).collect()
    }

    /// One field starting at absolute bit `start`. The caller has already bounded `start`.
    fn value_at(&self, bytes: &[u8], start: usize) -> f64 {
        let bits = self.bits as usize;
        let mask = if bits >= 64 { u64::MAX } else { (1u64 << bits) - 1 };
        let byte = start >> 3;
        let skew = start & 7;
        let raw = match self.order {
            BitOrder::LsbFirst => (word_le(bytes, byte) >> skew) & mask,
            BitOrder::MsbFirst => (word_be(bytes, byte) >> (64 - skew - bits)) & mask,
        };
        if self.signed {
            let sign = 1u64 << (bits - 1);
            (((raw ^ sign) as i64) - (sign as i64)) as f64
        } else {
            raw as f64
        }
    }
}

/// Eight bytes from `at`, zero-padded past the end, least-significant byte first.
fn word_le(bytes: &[u8], at: usize) -> u64 {
    let mut w = 0u64;
    for k in 0..8 {
        if let Some(&b) = bytes.get(at + k) {
            w |= (b as u64) << (8 * k);
        }
    }
    w
}

/// Eight bytes from `at`, zero-padded past the end, most-significant byte first.
fn word_be(bytes: &[u8], at: usize) -> u64 {
    let mut w = 0u64;
    for k in 0..8 {
        w = (w << 8) | *bytes.get(at + k).unwrap_or(&0) as u64;
    }
    w
}

/// Every reading rule the sweep considers, deterministic in order.
///
/// The stride set is each width's own dense packing plus its byte-aligned container (24 bits for an
/// 18-bit field), each repeated for up to [`MAX_INTERLEAVE`] channels. The start set is every byte
/// boundary up to [`MAX_HEADER_BYTES`] plus, when a frame interleaves, the position of each later
/// channel inside it — which for a non-aligned width is not a byte boundary and would otherwise be
/// unreachable.
pub fn candidates() -> Vec<Layout> {
    let mut out = Vec::new();
    for &bits in &WIDTHS_BITS {
        let w = bits as usize;
        let containers = if w.is_multiple_of(8) { vec![w] } else { vec![w, w.div_ceil(8) * 8] };
        let mut seen_stride = Vec::new();
        for &c in &containers {
            for k in 1..=MAX_INTERLEAVE {
                let stride = c * k;
                if seen_stride.contains(&stride) {
                    continue;
                }
                seen_stride.push(stride);
                let mut starts: Vec<usize> = (0..=MAX_HEADER_BYTES).map(|b| b * 8).collect();
                for m in 1..k {
                    let s = m * c;
                    if !starts.contains(&s) {
                        starts.push(s);
                    }
                }
                starts.sort_unstable();
                for &start_bit in &starts {
                    for signed in [false, true] {
                        for order in [BitOrder::LsbFirst, BitOrder::MsbFirst] {
                            out.push(Layout { bits, signed, order, start_bit, stride_bits: stride });
                        }
                    }
                }
            }
        }
    }
    out
}

/// Pack `values` back into bytes under `layout`, wrapping each to the field width. The inverse of
/// [`Layout::decode`] and the only way to build a buffer whose true layout is known.
pub fn encode(layout: &Layout, values: &[i64], len_bytes: usize) -> Vec<u8> {
    let mut bytes = vec![0u8; len_bytes];
    let bits = layout.bits as usize;
    for (i, &v) in values.iter().enumerate() {
        let start = layout.start_bit + i * layout.stride_bits;
        if start + bits > len_bytes * 8 {
            break;
        }
        let mask = if bits >= 64 { u64::MAX } else { (1u64 << bits) - 1 };
        let raw = (v as u64) & mask;
        for k in 0..bits {
            let bit = match layout.order {
                BitOrder::LsbFirst => (raw >> k) & 1,
                BitOrder::MsbFirst => (raw >> (bits - 1 - k)) & 1,
            };
            if bit == 1 {
                let b = start + k;
                let shift = match layout.order {
                    BitOrder::LsbFirst => b & 7,
                    BitOrder::MsbFirst => 7 - (b & 7),
                };
                bytes[b >> 3] |= 1 << shift;
            }
        }
    }
    bytes
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lay(bits: u8, signed: bool, order: BitOrder, start_bit: usize, stride_bits: usize) -> Layout {
        Layout { bits, signed, order, start_bit, stride_bits }
    }

    #[test]
    fn byte_aligned_reads_match_the_platform_word() {
        // 0x1234 then 0xABCD, little-endian.
        let bytes = [0x34u8, 0x12, 0xCD, 0xAB];
        let le = lay(16, false, BitOrder::LsbFirst, 0, 16);
        assert_eq!(le.decode(&bytes), vec![0x1234 as f64, 0xABCD as f64]);
        let be = lay(16, false, BitOrder::MsbFirst, 0, 16);
        assert_eq!(be.decode(&bytes), vec![0x3412 as f64, 0xCDAB as f64]);
        // Signed reads the same bits as two's complement: 0xABCD is negative.
        let s = lay(16, true, BitOrder::LsbFirst, 0, 16);
        assert_eq!(s.decode(&bytes), vec![0x1234 as f64, 0xABCD as f64 - 65536.0]);
    }

    #[test]
    fn an_eighteen_bit_field_round_trips_through_a_byte_boundary() {
        // Dense 18-bit packing: sample 1 starts at bit 18, inside byte 2.
        let values: Vec<i64> = vec![1, -1, 131071, -131072, 42, -42];
        for order in [BitOrder::LsbFirst, BitOrder::MsbFirst] {
            let l = lay(18, true, order, 0, 18);
            let bytes = encode(&l, &values, 32);
            let got = l.decode(&bytes);
            assert_eq!(&got[..values.len()], &values.iter().map(|&v| v as f64).collect::<Vec<_>>()[..]);
        }
        // The same field carried in a 24-bit container, data at the low end, is a different rule.
        let dense = lay(18, true, BitOrder::LsbFirst, 0, 18);
        let padded = lay(18, true, BitOrder::LsbFirst, 0, 24);
        let bytes = encode(&padded, &values, 32);
        assert_ne!(dense.decode(&bytes)[1], values[1] as f64);
        assert_eq!(padded.decode(&bytes)[1], values[1] as f64);
    }

    #[test]
    fn interleave_and_header_select_different_channels() {
        // Two 16-bit channels, wanted one second, after a 3-byte header.
        let want: Vec<i64> = vec![10, 20, 30];
        let other: Vec<i64> = vec![-1000, -2000, -3000];
        let mut bytes = vec![0xFFu8; 3];
        for i in 0..3 {
            bytes.extend_from_slice(&(other[i] as i16).to_le_bytes());
            bytes.extend_from_slice(&(want[i] as i16).to_le_bytes());
        }
        let l = lay(16, true, BitOrder::LsbFirst, 3 * 8 + 16, 32);
        assert_eq!(l.decode(&bytes)[..3], [10.0, 20.0, 30.0]);
        let o = lay(16, true, BitOrder::LsbFirst, 3 * 8, 32);
        assert_eq!(o.decode(&bytes)[..3], [-1000.0, -2000.0, -3000.0]);
    }

    #[test]
    fn a_rule_that_does_not_fit_reads_nothing_rather_than_panicking() {
        assert_eq!(lay(32, true, BitOrder::LsbFirst, 0, 32).sample_count(3), 0);
        assert!(lay(32, true, BitOrder::LsbFirst, 0, 32).decode(&[1, 2, 3]).is_empty());
        assert!(lay(16, false, BitOrder::LsbFirst, 200, 16).decode(&[1, 2, 3, 4]).is_empty());
        // A stride narrower than the field is not a packing, it is nonsense.
        assert_eq!(lay(24, false, BitOrder::LsbFirst, 0, 16).sample_count(64), 0);
        // Reading right up to the end pads with zeros rather than indexing off it.
        let l = lay(24, false, BitOrder::MsbFirst, 0, 24);
        assert_eq!(l.decode(&[0xFF, 0xFF, 0xFF]), vec![16_777_215.0]);
    }

    #[test]
    fn the_candidate_set_is_deterministic_distinct_and_covers_the_named_space() {
        let c = candidates();
        let mut seen = c.clone();
        seen.sort_by_key(|l| (l.bits, l.signed, l.order, l.start_bit, l.stride_bits));
        seen.dedup();
        assert_eq!(seen.len(), c.len(), "candidates must not repeat a rule");
        assert_eq!(c, candidates(), "enumeration order must be stable");
        for &bits in &WIDTHS_BITS {
            assert!(c.iter().any(|l| l.bits == bits && l.stride_bits == bits as usize));
        }
        // The 18-bit-in-24 container and the non-aligned channel start both have to be reachable.
        assert!(c.iter().any(|l| l.bits == 18 && l.stride_bits == 24));
        assert!(c.iter().any(|l| l.bits == 18 && l.start_bit == 18));
        assert!(c.iter().all(|l| l.stride_bits >= l.bits as usize));
    }
}
