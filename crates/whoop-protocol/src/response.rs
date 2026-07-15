//! COMMAND_RESPONSE (type 36) body decoders. The reply body forks per generation (GEN4 deci-percent
//! battery vs GEN5 direct percent; GEN5-only hello name/firmware), so it dispatches on family here.
//! Offsets are payload-relative (payload = inner[3..]).

use crate::bytes::{u16_at, u32_at, u8_at};
use crate::command;
use crate::event::ResultCode;
use crate::family::Family;
use crate::packet::Frame;

#[derive(Clone, Debug, PartialEq)]
pub enum CommandResponse {
    Battery { percent: f32 },
    Clock { unix: u32 },
    Hello { device_name: String, fw_version: Option<[u8; 4]> },
    DataRange { oldest: u32, newest: u32 },
    VersionInfo { fw: [u32; 4] },
    Other { cmd: u8, result: Option<ResultCode> },
}

pub fn decode(f: &Frame) -> Option<CommandResponse> {
    let p = f.payload();
    let cmd = f.cmd();
    match f.family {
        Family::Gen5 => decode_gen5(cmd, p),
        Family::Gen4 => decode_gen4(cmd, p),
    }
}

fn decode_gen5(cmd: u8, p: &[u8]) -> Option<CommandResponse> {
    match cmd {
        command::GET_BATTERY_LEVEL => Some(CommandResponse::Battery { percent: u8_at(p, 13)? as f32 }),
        command::GET_HELLO => Some(CommandResponse::Hello {
            device_name: ascii_z(p, 16),
            fw_version: match u8_at(p, 93) {
                Some(50) => Some([u8_at(p, 93)?, u8_at(p, 94)?, u8_at(p, 95)?, u8_at(p, 96)?]),
                _ => None,
            },
        }),
        command::GET_DATA_RANGE => Some(CommandResponse::DataRange { oldest: u32_at(p, 3)?, newest: u32_at(p, 7)? }),
        _ => Some(CommandResponse::Other { cmd, result: u8_at(p, 1).map(ResultCode::from_u8) }),
    }
}

fn decode_gen4(cmd: u8, p: &[u8]) -> Option<CommandResponse> {
    match cmd {
        command::GET_BATTERY_LEVEL => Some(CommandResponse::Battery { percent: u16_at(p, 2)? as f32 / 10.0 }),
        command::GET_CLOCK_GEN4 => Some(CommandResponse::Clock { unix: u32_at(p, 2)? }),
        command::REPORT_VERSION_INFO => Some(CommandResponse::VersionInfo {
            fw: [u32_at(p, 3)?, u32_at(p, 7)?, u32_at(p, 11)?, u32_at(p, 15)?],
        }),
        _ => Some(CommandResponse::Other { cmd, result: None }),
    }
}

/// Printable-ASCII string from `p[start]` up to a NUL or the first non-printable byte.
fn ascii_z(p: &[u8], start: usize) -> String {
    let mut s = String::new();
    for &b in p.get(start..).unwrap_or(&[]) {
        if b == 0 || !(32..=126).contains(&b) {
            break;
        }
        s.push(b as char);
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::framing;

    #[test]
    fn gen5_battery_is_direct_percent() {
        let mut p = vec![0u8; 20];
        p[1] = 1; // result = SUCCESS
        p[13] = 93; // battery @ payload 13
        let wire = framing::encode(Family::Gen5, 36, 0, command::GET_BATTERY_LEVEL, &p);
        let f = framing::decode(Family::Gen5, &wire).unwrap();
        assert_eq!(decode(&f), Some(CommandResponse::Battery { percent: 93.0 }));
    }

    #[test]
    fn gen5_hello_reads_device_name() {
        let mut p = vec![0u8; 40];
        p[16..26].copy_from_slice(b"5AG0546409");
        let wire = framing::encode(Family::Gen5, 36, 0, command::GET_HELLO, &p);
        let f = framing::decode(Family::Gen5, &wire).unwrap();
        match decode(&f).unwrap() {
            CommandResponse::Hello { device_name, .. } => assert_eq!(device_name, "5AG0546409"),
            other => panic!("expected Hello, got {other:?}"),
        }
    }

    #[test]
    fn gen4_battery_is_deci_percent() {
        let mut p = vec![0u8; 8];
        p[2..4].copy_from_slice(&812u16.to_le_bytes()); // 81.2 %
        let wire = framing::encode(Family::Gen4, 36, 0, command::GET_BATTERY_LEVEL, &p);
        let f = framing::decode(Family::Gen4, &wire).unwrap();
        assert_eq!(decode(&f), Some(CommandResponse::Battery { percent: 81.2 }));
    }
}
