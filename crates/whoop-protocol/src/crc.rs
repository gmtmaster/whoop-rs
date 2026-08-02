//! The three wire checksums. Params are hardware-verified; do not "simplify".
//!  - crc8       : GEN4 header check over the 2 declared-length bytes. poly 0x07, MSB-first, no xorout.
//!  - crc16_modbus: GEN5 header check over the 6-byte header. poly 0xA001, reflected, init 0xFFFF.
//!  - crc32_zlib : inner-record trailer, both generations. standard zlib (poly 0xEDB88320, xorout all-ones).

pub fn crc8(data: &[u8]) -> u8 {
    let mut crc: u8 = 0x00;
    for &b in data {
        crc ^= b;
        for _ in 0..8 {
            crc = if crc & 0x80 != 0 { (crc << 1) ^ 0x07 } else { crc << 1 };
        }
    }
    crc
}

pub fn crc16_modbus(data: &[u8]) -> u16 {
    let mut crc: u16 = 0xFFFF;
    for &b in data {
        crc ^= b as u16;
        for _ in 0..8 {
            crc = if crc & 1 != 0 { (crc >> 1) ^ 0xA001 } else { crc >> 1 };
        }
    }
    crc
}

pub fn crc32_zlib(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFF_FFFF;
    for &b in data {
        crc ^= b as u32;
        for _ in 0..8 {
            crc = if crc & 1 != 0 { (crc >> 1) ^ 0xEDB8_8320 } else { crc >> 1 };
        }
    }
    crc ^ 0xFFFF_FFFF
}

#[cfg(test)]
mod tests {
    use super::*;

    // Known answers from the real CLIENT_HELLO frame.
    #[test]
    fn client_hello_header_crc16() {
        assert_eq!(crc16_modbus(&[0xAA, 0x01, 0x08, 0x00, 0x00, 0x01]), 0x71E6);
    }

    #[test]
    fn client_hello_inner_crc32() {
        assert_eq!(crc32_zlib(&[0x23, 0x01, 0x91, 0x01]), 0x8D5C_3E36);
    }

    /// Standard check values over the ASCII digits, one per parameter set. Catches a wrong
    /// polynomial, init, reflection or xorout without needing a frame.
    #[test]
    fn each_checksum_matches_its_standard_check_value() {
        assert_eq!(crc8(b"123456789"), 0xF4, "crc8 poly 0x07, init 0, MSB-first, no xorout");
        assert_eq!(crc16_modbus(b"123456789"), 0x4B37, "crc16 poly 0xA001 reflected, init 0xFFFF");
        assert_eq!(crc32_zlib(b"123456789"), 0xCBF4_3926, "crc32 zlib");
    }

    /// GEN4 header: crc8 over the 2 declared-length bytes is the check byte that follows them.
    /// Three real declared lengths, each with the byte the strap actually sent.
    #[test]
    fn crc8_reproduces_real_gen4_header_check_bytes() {
        for (len, want) in [([0x10u8, 0x00], 0x57u8), ([0x50, 0x00], 0x0C), ([0x64, 0x00], 0xA1)] {
            assert_eq!(crc8(&len), want, "gen4 header check byte over declared length {len:02x?}");
        }
        // The all-zero length is the one vector a do-nothing checksum survives, so it proves nothing
        // on its own; it is kept only to pin the degenerate case.
        assert_eq!(crc8(&[0x00, 0x00]), 0x00);
    }

    /// The null arm: a checksum that always returns 0 must fail. Each function has to separate
    /// inputs, and every single-bit flip in the covered bytes has to change the output.
    #[test]
    fn a_constant_checksum_and_a_blind_bit_flip_both_fail() {
        assert_ne!(crc8(&[0x10, 0x00]), 0, "a constant-zero crc8 would pass here");
        assert_ne!(crc16_modbus(&[0xAA, 0x01, 0x08, 0x00, 0x00, 0x01]), 0);
        assert_ne!(crc32_zlib(&[0x23, 0x01, 0x91, 0x01]), 0);

        let base8 = [0x10u8, 0x00];
        for bit in 0..16 {
            let mut d = base8;
            d[bit / 8] ^= 1 << (bit % 8);
            assert_ne!(crc8(&d), crc8(&base8), "crc8 missed a flip of bit {bit}");
        }
        let base16 = [0xAAu8, 0x01, 0x08, 0x00, 0x00, 0x01];
        for bit in 0..48 {
            let mut d = base16;
            d[bit / 8] ^= 1 << (bit % 8);
            assert_ne!(crc16_modbus(&d), crc16_modbus(&base16), "crc16 missed a flip of bit {bit}");
        }
        let base32 = [0x23u8, 0x01, 0x91, 0x01];
        for bit in 0..32 {
            let mut d = base32;
            d[bit / 8] ^= 1 << (bit % 8);
            assert_ne!(crc32_zlib(&d), crc32_zlib(&base32), "crc32 missed a flip of bit {bit}");
        }
    }
}
