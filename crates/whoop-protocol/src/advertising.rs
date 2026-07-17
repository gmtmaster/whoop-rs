//! SET_ADVERTISING_NAME payload (4.0 strap rename). The strap reboots to apply, so the new name appears
//! on the next connect. Reversible.

/// Body `[0x00, 0x00] + UTF-8 name + [0x00]`, the name clamped to 24 UTF-8 bytes on a char boundary so it
/// can't overflow the advertising packet (never splits a multibyte char).
pub fn advertising_name_payload(name: &str) -> Vec<u8> {
    let mut end = name.len();
    while end > 24 {
        end -= 1;
        while !name.is_char_boundary(end) {
            end -= 1;
        }
    }
    let clamped = &name[..end];
    let mut out = Vec::with_capacity(3 + clamped.len());
    out.extend_from_slice(&[0, 0]);
    out.extend_from_slice(clamped.as_bytes());
    out.push(0);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wraps_short_name() {
        assert_eq!(advertising_name_payload("noop"), b"\x00\x00noop\x00");
    }

    #[test]
    fn clamps_to_24_utf8_bytes() {
        let p = advertising_name_payload(&"a".repeat(40));
        // [0,0] + 24 bytes + [0] = 27
        assert_eq!(p.len(), 27);
        assert_eq!(&p[2..26], &b"a".repeat(24)[..]);
        assert_eq!(*p.last().unwrap(), 0);
    }

    #[test]
    fn never_splits_a_multibyte_char() {
        // Each 'é' is 2 UTF-8 bytes; 13 of them = 26 bytes, must clamp to 12 (24 bytes) not straddle 13th.
        let p = advertising_name_payload(&"é".repeat(13));
        assert_eq!(p.len(), 2 + 24 + 1);
        // The clamped middle must be valid UTF-8 (no half char).
        assert!(std::str::from_utf8(&p[2..p.len() - 1]).is_ok());
    }
}
