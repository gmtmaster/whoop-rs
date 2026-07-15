//! CONSOLE_LOGS (type 50) text extraction. The strap narrates its firmware console over this channel
//! (history-sync progress, "RTC timestamp invalid; not saving", the persistent-config table). A record
//! header precedes the text; matching the shipped decoder, the readable text is payload[10..] filtered
//! to printable ASCII + newlines.

use crate::packet::Frame;

const MAX_TEXT: usize = 2048;
const HEADER: usize = 10;

/// Extract the printable console text from a CONSOLE_LOGS frame, or `None` if there is no body.
pub fn text(f: &Frame) -> Option<String> {
    let p = f.payload();
    if p.len() <= HEADER {
        return None;
    }
    let mut s = String::new();
    for &b in &p[HEADER..] {
        if b == 0x0a {
            s.push('\n');
        } else if (32..=126).contains(&b) {
            s.push(b as char);
        }
        if s.len() >= MAX_TEXT {
            break;
        }
    }
    (!s.is_empty()).then_some(s)
}
