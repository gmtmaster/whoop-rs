//! FORCE_TRIM payloads. Two LE u32: `[0..4]` = trim page, `[4..8]` = wrap count. Only the two builders
//! here construct one — every other pair of words on that opcode is a real trim, so no caller supplies
//! the numbers. `whoop_client::WhoopClient::undo_trim` is the only sender.

/// Both words: reset the trim and read pointers back to oldest. Pointers only — it cannot un-erase flash.
pub const RESET_TO_OLDEST: u32 = 0xFDFD_FDFD;
/// Both words: erase all banked history. Named so the reset can be told apart from it; nothing builds it.
pub const TRIM_ALL: u32 = 0xFEFE_FEFE;
/// First word: the strap declines the request. The inert probe — reaches the handler, moves nothing.
pub const IGNORE: u32 = 0xFFFF_FFFF;

// The reset and the erase differ by one bit per byte, so a transposition would be silent. Pin them
// apart, and pin which is which, at compile time.
const _: () = assert!(RESET_TO_OLDEST != TRIM_ALL);
const _: () = assert!(RESET_TO_OLDEST.to_le_bytes()[0] == 0xFD && TRIM_ALL.to_le_bytes()[0] == 0xFE);

/// The reset payload — eight `0xFD` bytes.
pub fn reset_to_oldest() -> [u8; 8] {
    words(RESET_TO_OLDEST, RESET_TO_OLDEST)
}

/// The inert payload — first word `0xFFFFFFFF`, which the strap declines. The positive control for the
/// reset: same opcode, same handler, no pointer moved.
pub fn inert_probe() -> [u8; 8] {
    words(IGNORE, IGNORE)
}

/// `[page LE][wrap LE]`. Private: a payload nobody chose is the whole safety property here.
fn words(page: u32, wrap: u32) -> [u8; 8] {
    let mut out = [0u8; 8];
    out[..4].copy_from_slice(&page.to_le_bytes());
    out[4..].copy_from_slice(&wrap.to_le_bytes());
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reset_is_eight_fd_bytes() {
        assert_eq!(reset_to_oldest(), [0xFD; 8]);
    }

    #[test]
    fn reset_is_not_the_erase() {
        assert_ne!(RESET_TO_OLDEST, TRIM_ALL);
        assert_ne!(reset_to_oldest(), words(TRIM_ALL, TRIM_ALL));
        assert_eq!(words(TRIM_ALL, TRIM_ALL), [0xFE; 8]); // the neighbour, spelled out
    }

    #[test]
    fn inert_probe_leads_with_the_ignore_sentinel() {
        let p = inert_probe();
        assert_eq!(&p[..4], &0xFFFF_FFFFu32.to_le_bytes());
        assert_ne!(p, [0xFD; 8]);
        assert_ne!(p, [0xFE; 8]);
    }

    #[test]
    fn words_lays_page_then_wrap_little_endian() {
        assert_eq!(words(0x0403_0201, 0x0807_0605), [1, 2, 3, 4, 5, 6, 7, 8]);
    }
}
