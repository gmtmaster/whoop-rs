//! SET_ADVERTISING_NAME payload (strap rename). The strap reboots to apply, so the new name appears on the
//! next connect. Reversible. The 4.0 and 5/MG bodies share a shape; 5/MG leads with the puffin `b3` selector
//! byte every 5/MG set/name command carries.

/// Advertising-name cap in UTF-8 bytes so the body can't overflow the advertising packet.
const MAX_NAME_BYTES: usize = 24;

/// Clamp `name` to [`MAX_NAME_BYTES`] on a char boundary (never splits a multibyte char).
fn clamp(name: &str) -> &str {
    let mut end = name.len();
    while end > MAX_NAME_BYTES {
        end -= 1;
        while !name.is_char_boundary(end) {
            end -= 1;
        }
    }
    &name[..end]
}

/// Body `[lead, 0x00] + UTF-8 name + [0x00]`, the name clamped to 24 UTF-8 bytes.
fn body(lead: u8, name: &str) -> Vec<u8> {
    let clamped = clamp(name);
    let mut out = Vec::with_capacity(3 + clamped.len());
    out.extend_from_slice(&[lead, 0]);
    out.extend_from_slice(clamped.as_bytes());
    out.push(0);
    out
}

/// 4.0 (Harvard) body `[0x00, 0x00] + name + [0x00]`.
pub fn advertising_name_payload(name: &str) -> Vec<u8> {
    body(0x00, name)
}

/// 5/MG (puffin) body `[0x01, 0x00] + name + [0x00]`: the leading `0x01` is the `b3` selector every 5/MG
/// set/name command carries; the rest mirrors the 4.0 body. Hardware-unverified — the opcode and `b3` are a
/// best-supported inference, not yet strap-confirmed.
pub fn advertising_name_payload_gen5(name: &str) -> Vec<u8> {
    body(0x01, name)
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

    #[test]
    fn gen5_leads_with_the_b3_selector() {
        assert_eq!(advertising_name_payload_gen5("noop"), b"\x01\x00noop\x00");
    }

    #[test]
    fn gen5_shares_the_clamp_and_terminator() {
        let p = advertising_name_payload_gen5(&"a".repeat(40));
        assert_eq!(p.len(), 27);
        assert_eq!(p[0], 0x01);
        assert_eq!(&p[2..26], &b"a".repeat(24)[..]);
        assert_eq!(*p.last().unwrap(), 0);
    }
}
