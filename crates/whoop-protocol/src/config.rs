//! Persistent config / feature-flag writes (GEN5 only). SET_CONFIG writes one 40-byte feature flag,
//! SET_DEVICE_CONFIG one 33-byte device value; the R22 sequence unlocks the deep biometric streams.
//! Reversible. Frames are built via `framing::encode`.

use crate::bytes::prefixed;
use crate::command;
use crate::family::Family;
use crate::framing;

/// One persistent flag and the value the official app writes for it (`b'1'` or `b'2'`).
pub struct Flag {
    pub name: &'static str,
    pub value: u8,
}

/// The ordered enable_r22 sequence (16 flags).
pub const R22_SEQUENCE: [Flag; 16] = [
    Flag { name: "enable_r22_packets", value: b'2' },
    Flag { name: "enable_r22_v2_packets", value: b'2' },
    Flag { name: "enable_r22_v3_packets", value: b'2' },
    Flag { name: "enable_r22_v4_packets", value: b'1' },
    Flag { name: "enable_r22_v5_packets", value: b'2' },
    Flag { name: "enable_r22_v6_packets", value: b'2' },
    Flag { name: "enable_r22_v8_packets", value: b'2' },
    Flag { name: "make_hrfm_visible", value: b'2' },
    Flag { name: "disable_pip_r26_packets", value: b'2' },
    Flag { name: "wear_detect_bias", value: b'2' },
    Flag { name: "hr_ch_switching", value: b'2' },
    Flag { name: "ir_hw_switching", value: b'2' },
    Flag { name: "enable_passive_strap_fit_gen5", value: b'1' },
    Flag { name: "enable_sig11_during_sleep", value: b'2' },
    Flag { name: "dorset_inhibit_wpt", value: b'2' },
    Flag { name: "enable_sig12", value: b'1' },
];

/// The R22 sub-version that appeared after the 16-flag sequence was transcribed. Its value is not
/// known from any capture; `b'2'` matches every other R22 flag but the base and v4.
pub const R22_V9: Flag = Flag { name: "enable_r22_v9_packets", value: b'2' };

/// Config flags present in firmware but not sent by the R22 sequence (untried).
pub const FIRMWARE_ONLY_FLAGS: [&str; 5] = [
    "whoop_live_2_hrm_devices",
    "enable_raw_data_w_ecg",
    "general_ab_test",
    "enable_pdaf_walk_det",
    "enable_maverick_model",
];

/// A named config body: `name` (≤32B, NUL-padded) at [0..32] + `value` @32, in a `len`-byte buffer
/// (40 = SET_CONFIG with 7 trailing zeros, 33 = SET_DEVICE_CONFIG without).
fn named_body(len: usize, name: &str, value: u8) -> Vec<u8> {
    let mut b = vec![0u8; len];
    let n = name.as_bytes();
    let take = n.len().min(32);
    b[..take].copy_from_slice(&n[..take]);
    b[32] = value;
    b
}

/// SET_CONFIG 40-byte body: name (≤32B, NUL-padded) + value @32 + 7 zero.
pub fn feature_body(name: &str, value: u8) -> Vec<u8> {
    named_body(40, name, value)
}

/// SET_DEVICE_CONFIG 33-byte body: name (≤32B, NUL-padded) + value @32 (no trailing 7 zeros).
pub fn device_body(name: &str, value: u8) -> Vec<u8> {
    named_body(33, name, value)
}

/// Wrap a config body as a GEN5 COMMAND frame: `0x01` prefix + `body`, opcode `cmd`.
fn config_frame(seq: u8, cmd: u8, body: &[u8]) -> Vec<u8> {
    framing::command(Family::Gen5, seq, cmd, &prefixed(0x01, body))
}

/// A ready-to-write SET_CONFIG frame for one feature flag (0x01 prefix + 40-byte body).
pub fn feature_frame(seq: u8, flag: &Flag) -> Vec<u8> {
    config_frame(seq, command::SET_CONFIG, &feature_body(flag.name, flag.value))
}

/// A SET_CONFIG frame for a runtime-named feature flag (the arbitrary-flag builder the FFI needs).
pub fn feature_frame_named(seq: u8, name: &str, value: u8) -> Vec<u8> {
    config_frame(seq, command::SET_CONFIG, &feature_body(name, value))
}

/// A ready-to-write SET_DEVICE_CONFIG frame (0x01 prefix + 33-byte body).
pub fn device_frame(seq: u8, name: &str, value: u8) -> Vec<u8> {
    config_frame(seq, command::SET_DEVICE_CONFIG, &device_body(name, value))
}

/// The full 16-frame R22 enable sequence, seq-numbered from `start_seq` (wrapping). The single source
/// for both the client (which writes each frame) and the FFI (which hands them to the app to write).
pub fn r22_frames(start_seq: u8) -> Vec<Vec<u8>> {
    r22_frames_ext(start_seq, None)
}

/// [`r22_frames`] plus [`R22_V9`] at `v9` when given. The base 16 are byte-identical either way, so an
/// added flag can only ever appear last.
pub fn r22_frames_ext(start_seq: u8, v9: Option<u8>) -> Vec<Vec<u8>> {
    r22_flags_ext(v9)
        .iter()
        .enumerate()
        .map(|(i, flag)| feature_frame(start_seq.wrapping_add(i as u8), flag))
        .collect()
}

/// The flags [`r22_frames_ext`] would send, in order.
pub fn r22_flags_ext(v9: Option<u8>) -> Vec<Flag> {
    let mut v: Vec<Flag> = R22_SEQUENCE.iter().map(|f| Flag { name: f.name, value: f.value }).collect();
    if let Some(value) = v9 {
        v.push(Flag { name: R22_V9.name, value });
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn r22_body_shape() {
        let b = feature_body("enable_r22_packets", b'2');
        assert_eq!(b.len(), 40);
        assert_eq!(&b[..18], b"enable_r22_packets");
        assert_eq!(b[18], 0); // NUL pad
        assert_eq!(b[32], b'2');
    }

    #[test]
    fn r22_frame_decodes_as_set_config() {
        let wire = feature_frame(0, &R22_SEQUENCE[0]);
        let f = framing::decode(Family::Gen5, &wire).unwrap();
        assert!(f.crc_ok);
        assert_eq!(f.cmd(), command::SET_CONFIG);
        assert_eq!(f.payload()[0], 0x01);
        assert_eq!(&f.payload()[1..19], b"enable_r22_packets");
    }
}
