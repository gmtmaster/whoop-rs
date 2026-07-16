//! COMMAND_RESPONSE (type 36) body decoders. The reply body forks per generation (GEN4 deci-percent
//! battery vs GEN5 direct percent; GEN5-only hello name/firmware), so it dispatches on family here.
//! Offsets are payload-relative (payload = inner[3..]).

use crate::bytes::{i16_at, u16_at, u32_at, u8_at};
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
    /// 4.0 strap fuel gauge (GET_EXTENDED_BATTERY_INFO). Voltage, remaining capacity (remaining/full = the
    /// SOC), and the signed instantaneous current.
    ExtendedBattery { millivolts: u16, remaining_mah: u16, current_ma: i16 },
    /// 5.0 battery-pack fuel gauge (GET_BATTERY_PACK_INFO). `millivolts` is the pack voltage; `pack_id` is a
    /// per-pack 32-bit id distinct from `serial`.
    BatteryPack { serial: String, soc_pct: f32, millivolts: u16, pack_id: u32 },
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
        command::GET_BATTERY_LEVEL => Some(CommandResponse::Battery { percent: u8_at(p, 2)? as f32 }),
        command::GET_HELLO => Some(CommandResponse::Hello {
            device_name: ascii_z(p, 16),
            // Gate on the "5.x" marker; the `?` stays inside the closure so a truncated fw block yields
            // fw_version None without discarding the already-decoded serial.
            fw_version: u8_at(p, 93).filter(|&v| v == 50).and_then(|_| {
                Some([u8_at(p, 93)?, u8_at(p, 94)?, u8_at(p, 95)?, u8_at(p, 96)?])
            }),
        }),
        command::GET_DATA_RANGE => Some(CommandResponse::DataRange { oldest: u32_at(p, 3)?, newest: u32_at(p, 7)? }),
        // Pack fuel gauge: pack-id u32@4, mV u16@8, ASCII serial @10 (NUL-terminated), SOC u16@26 (raw/10).
        // The `?` is closure-scoped so a short/unsupported reply degrades to Other, not a lost response.
        command::GET_BATTERY_PACK_INFO => (|| {
            Some(CommandResponse::BatteryPack {
                pack_id: u32_at(p, 4)?,
                millivolts: u16_at(p, 8)?,
                serial: ascii_z(p, 10),
                soc_pct: u16_at(p, 26)? as f32 / 10.0,
            })
        })()
        .or_else(|| Some(CommandResponse::Other { cmd, result: u8_at(p, 1).map(ResultCode::from_u8) })),
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
        // Fuel gauge pinned by 3-strap correlation vs SOC: mV@7, remaining-capacity@13 (remaining/full=SOC),
        // signed current@3.
        command::GET_EXTENDED_BATTERY_INFO => Some(CommandResponse::ExtendedBattery {
            millivolts: u16_at(p, 7)?,
            remaining_mah: u16_at(p, 13)?,
            current_ma: i16_at(p, 3)?,
        }),
        command::GET_HELLO_HARVARD => {
            let serial = ascii_z(p, 16);
            // The serial is followed by a variable-length ASCII-hex session token, then a u32 status block
            // whose 4th..7th words are the firmware. Skip the token (its trailing byte is non-hex) to find
            // the block, and gate on a plausible major so a wrong offset yields None.
            let mut block = 16 + serial.len() + 1;
            while p.get(block).is_some_and(|&b| b.is_ascii_hexdigit()) {
                block += 1;
            }
            let fw = u32_at(p, block + 12).filter(|&v| (1..=99).contains(&v)).and_then(|maj| {
                Some([maj as u8, u32_at(p, block + 16)? as u8, u32_at(p, block + 20)? as u8, u32_at(p, block + 24)? as u8])
            });
            Some(CommandResponse::Hello { device_name: serial, fw_version: fw })
        }
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
        p[2] = 93; // battery % @ payload 2 (real 5.0 layout: 035=100, 206=89)
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
    fn gen5_hello_reads_firmware_and_keeps_serial_on_gate_miss() {
        let mut p = vec![0u8; 100];
        p[16..26].copy_from_slice(b"5AG0268206");
        p[93..97].copy_from_slice(&[50, 40, 1, 0]);
        let decode_p = |p: &[u8]| {
            let wire = framing::encode(Family::Gen5, 36, 0, command::GET_HELLO, p);
            decode(&framing::decode(Family::Gen5, &wire).unwrap()).unwrap()
        };
        assert_eq!(
            decode_p(&p),
            CommandResponse::Hello { device_name: "5AG0268206".into(), fw_version: Some([50, 40, 1, 0]) }
        );
        // A non-5.x marker drops firmware but must keep the serial (the fw `?` is closure-scoped).
        p[93] = 40;
        assert_eq!(
            decode_p(&p),
            CommandResponse::Hello { device_name: "5AG0268206".into(), fw_version: None }
        );
    }

    #[test]
    fn gen5_battery_pack_decodes_serial_soc_mv_id() {
        let mut p = vec![0u8; 40];
        p[4..8].copy_from_slice(&0x1122_3344u32.to_le_bytes()); // pack-id
        p[8..10].copy_from_slice(&3700u16.to_le_bytes()); // mV
        p[10..23].copy_from_slice(b"WBBTEST123456"); // serial, NUL at p[23]
        p[26..28].copy_from_slice(&875u16.to_le_bytes()); // SOC 87.5%
        let wire = framing::encode(Family::Gen5, 36, 0, command::GET_BATTERY_PACK_INFO, &p);
        let f = framing::decode(Family::Gen5, &wire).unwrap();
        assert_eq!(
            decode(&f),
            Some(CommandResponse::BatteryPack {
                serial: "WBBTEST123456".into(),
                soc_pct: 87.5,
                millivolts: 3700,
                pack_id: 0x1122_3344,
            })
        );
    }

    #[test]
    fn gen4_extended_battery_reads_mv_capacity_current() {
        let mut p = vec![0u8; 30];
        p[3..5].copy_from_slice(&(-532i16).to_le_bytes()); // current
        p[7..9].copy_from_slice(&4363u16.to_le_bytes()); // mV
        p[13..15].copy_from_slice(&2017u16.to_le_bytes()); // remaining mAh
        let wire = framing::encode(Family::Gen4, 36, 0, command::GET_EXTENDED_BATTERY_INFO, &p);
        let f = framing::decode(Family::Gen4, &wire).unwrap();
        assert_eq!(
            decode(&f),
            Some(CommandResponse::ExtendedBattery { millivolts: 4363, remaining_mah: 2017, current_ma: -532 })
        );
    }

    #[test]
    fn gen4_hello_reads_serial_and_firmware() {
        let mut p = vec![0u8; 120];
        p[16..25].copy_from_slice(b"TEST12345"); // serial, NUL-terminated by the zero at p[25]
        let token = b"8e2782b74f40284c"; // ASCII-hex session token (any length); the block follows it
        p[26..26 + token.len()].copy_from_slice(token);
        let block = 26 + token.len(); // first non-hex byte after the token
        p[block + 12] = 41;
        p[block + 16] = 17;
        p[block + 20] = 6;
        let wire = framing::encode(Family::Gen4, 36, 0, command::GET_HELLO_HARVARD, &p);
        let f = framing::decode(Family::Gen4, &wire).unwrap();
        match decode(&f).unwrap() {
            CommandResponse::Hello { device_name, fw_version } => {
                assert_eq!(device_name, "TEST12345");
                assert_eq!(fw_version, Some([41, 17, 6, 0]));
            }
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
