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

    #[test]
    fn crc8_over_length_bytes() {
        // GEN4 header: crc8 over the 2 declared-length bytes lands at frame[3].
        assert_eq!(crc8(&[0x00, 0x00]), 0x00);
    }
}
