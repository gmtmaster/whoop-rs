//! Codec error type. Structural decode failures only — a CRC mismatch is NOT an error, it surfaces as
//! `Frame::crc_ok == false` so the caller can log-and-drop without unwinding.

#[derive(thiserror::Error, Debug, PartialEq, Eq)]
pub enum ProtocolError {
    #[error("frame too short: have {have}, need {need}")]
    TooShort { have: usize, need: usize },
    #[error("bad start-of-frame byte")]
    BadSof,
    #[error("bad declared length")]
    BadLength,
    #[error("empty inner record")]
    EmptyInner,
}
