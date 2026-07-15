//! The GEN5 bond-open frame. Not a magic blob — it is a type-35 COMMAND carrying GET_HELLO(145) with a
//! `[0x01]` payload; `framing::encode(Gen5, 35, 1, 145, &[1])` reproduces it byte-for-byte (see tests).
//! Written (confirmed) to the cmd-write characteristic to trigger the just-works bond.

pub const GEN5_CLIENT_HELLO: [u8; 16] = [
    0xAA, 0x01, 0x08, 0x00, 0x00, 0x01, 0xE6, 0x71, 0x23, 0x01, 0x91, 0x01, 0x36, 0x3E, 0x5C, 0x8D,
];
