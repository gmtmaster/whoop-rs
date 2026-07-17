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
    Battery { percent: f64 },
    Clock { unix: u32 },
    Hello { device_name: String, fw_version: Option<[u8; 4]> },
    DataRange { oldest: u32, newest: u32 },
    VersionInfo { fw: [u32; 4] },
    /// 4.0 strap fuel gauge (GET_EXTENDED_BATTERY_INFO). Voltage, remaining capacity (remaining/full = the
    /// SOC), and the signed instantaneous current.
    ExtendedBattery { millivolts: u16, remaining_mah: u16, current_ma: i16 },
    /// 5.0 battery-pack fuel gauge (GET_BATTERY_PACK_INFO). `millivolts` is the pack voltage; `pack_id` is a
    /// per-pack 32-bit id distinct from `serial`.
    BatteryPack { serial: String, soc_pct: f64, millivolts: u16, pack_id: u32 },
    Other { cmd: u8, result: Option<ResultCode> },
}

/// The plausible unix window a banked-history word must fall in (≈2023-11 .. 2030-03).
const PLAUSIBLE_LO: u32 = 1_700_000_000;
const PLAUSIBLE_HI: u32 = 1_900_000_000;

fn word_le(frame: &[u8], i: usize) -> u32 {
    u32::from_le_bytes([frame[i], frame[i + 1], frame[i + 2], frame[i + 3]])
}

/// Newest plausible unix banked by the strap, scanning EVERY byte offset of a GET_DATA_RANGE frame (the
/// newest-record word isn't on a fixed grid). Prefers the newest word that is NOT implausibly future
/// (> wall_now + skew) so a garbage future word can't latch and stall auto-sync, falling back to the
/// newest-any word so a genuinely future-dated RTC still surfaces. `None` = too short / no plausible word.
/// Replaces the fixed-offset `CommandResponse::DataRange` read for the sync gate.
pub fn data_range_scan_newest(frame: &[u8], wall_now_unix: u64, future_skew_seconds: u64) -> Option<u32> {
    let cutoff = wall_now_unix.saturating_add(future_skew_seconds);
    let mut newest_not_future: Option<u32> = None;
    let mut newest_any: Option<u32> = None;
    let mut i = 0;
    while i + 4 <= frame.len() {
        let w = word_le(frame, i);
        if (PLAUSIBLE_LO..=PLAUSIBLE_HI).contains(&w) {
            newest_any = Some(newest_any.map_or(w, |m| m.max(w)));
            if u64::from(w) <= cutoff {
                newest_not_future = Some(newest_not_future.map_or(w, |m| m.max(w)));
            }
        }
        i += 1;
    }
    newest_not_future.or(newest_any)
}

/// Oldest plausible unix banked (start of history), scanning ONLY the 4-byte grid aligned from offset 7 —
/// deliberately asymmetric with the newest scan: the minimum is fragile (an any-offset scan surfaces a
/// spurious WHOOP-4 straddle word that would hijack it), the maximum is not. `None` if no distinct word.
pub fn data_range_scan_oldest(frame: &[u8]) -> Option<u32> {
    let mut oldest: Option<u32> = None;
    let mut i = 7;
    while i + 4 <= frame.len() {
        let w = word_le(frame, i);
        if (PLAUSIBLE_LO..=PLAUSIBLE_HI).contains(&w) {
            oldest = Some(oldest.map_or(w, |m| m.min(w)));
        }
        i += 4;
    }
    oldest
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
        command::GET_BATTERY_LEVEL => Some(CommandResponse::Battery { percent: u8_at(p, 2)? as f64 }),
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
                soc_pct: u16_at(p, 26)? as f64 / 10.0,
            })
        })()
        .or_else(|| Some(CommandResponse::Other { cmd, result: u8_at(p, 1).map(ResultCode::from_u8) })),
        _ => Some(CommandResponse::Other { cmd, result: u8_at(p, 1).map(ResultCode::from_u8) }),
    }
}

fn decode_gen4(cmd: u8, p: &[u8]) -> Option<CommandResponse> {
    match cmd {
        command::GET_BATTERY_LEVEL => Some(CommandResponse::Battery { percent: u16_at(p, 2)? as f64 / 10.0 }),
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
    fn data_range_scan_pins_newest_and_oldest_on_a_real_frame() {
        let frame = crate::bytes::from_hex(
            "aa014c00010032d124f22204010140bb0100f9ba010001bb0100f9ba010010000000000002006a00000088ff1d001432b869cc4c00004549596ab83e00004549596ab83e0000ae49596aeb1100000000d0da9256",
        )
        .unwrap();
        // Wall well ahead of the banked newest so nothing is future-skewed.
        assert_eq!(super::data_range_scan_newest(&frame, 1_784_236_480, 3600), Some(1_784_236_462));
        // Oldest scans the aligned-from-7 grid; the newest word does not sit on it, so the min is the
        // deep-backlog start, not the recent word.
        assert_eq!(super::data_range_scan_oldest(&frame), Some(1_778_385_408));
    }

    #[test]
    fn data_range_newest_prefers_not_future_but_falls_back() {
        // Two plausible words: an in-window one and a future-skewed one.
        let mut frame = vec![0u8; 12];
        frame[0..4].copy_from_slice(&1_784_000_000u32.to_le_bytes());
        frame[6..10].copy_from_slice(&1_850_000_000u32.to_le_bytes()); // implausibly future vs wall
        // Prefer the non-future word so a garbage future word can't latch.
        assert_eq!(super::data_range_scan_newest(&frame, 1_784_000_100, 3600), Some(1_784_000_000));
        // With no non-future word, fall back to the newest-any (a genuine future-dated RTC surfaces).
        let mut only_future = vec![0u8; 4];
        only_future.copy_from_slice(&1_850_000_000u32.to_le_bytes());
        assert_eq!(super::data_range_scan_newest(&only_future, 1_784_000_100, 3600), Some(1_850_000_000));
    }

    #[test]
    fn data_range_oldest_skips_the_off_grid_straddle_word() {
        // A plausible word only at an off-grid offset (not aligned from 7) must NOT hijack the minimum.
        let mut frame = vec![0u8; 15];
        frame[6..10].copy_from_slice(&1_750_000_000u32.to_le_bytes()); // straddle, off the from-7 grid
        assert_eq!(super::data_range_scan_oldest(&frame), None);
        // Same word on the grid (offset 7) is accepted.
        let mut on_grid = vec![0u8; 15];
        on_grid[7..11].copy_from_slice(&1_750_000_000u32.to_le_bytes());
        assert_eq!(super::data_range_scan_oldest(&on_grid), Some(1_750_000_000));
    }

    #[test]
    fn gen4_battery_is_deci_percent() {
        let mut p = vec![0u8; 8];
        p[2..4].copy_from_slice(&812u16.to_le_bytes()); // 81.2 %
        let wire = framing::encode(Family::Gen4, 36, 0, command::GET_BATTERY_LEVEL, &p);
        let f = framing::decode(Family::Gen4, &wire).unwrap();
        assert_eq!(decode(&f), Some(CommandResponse::Battery { percent: 81.2 }));
    }

    #[test]
    fn gen4_battery_divides_in_f64() {
        // Deci-% divided in f64 lands on the exact stored Double: 999 -> 99.9 (f32 would give
        // 99.90000152587891). This is the adjudicated precision improvement over the old f32 path.
        let mut p = vec![0u8; 8];
        p[2..4].copy_from_slice(&999u16.to_le_bytes());
        let wire = framing::encode(Family::Gen4, 36, 0, command::GET_BATTERY_LEVEL, &p);
        let f = framing::decode(Family::Gen4, &wire).unwrap();
        assert_eq!(decode(&f), Some(CommandResponse::Battery { percent: 99.9 }));
        assert_ne!((999.0f32 / 10.0) as f64, 99.9); // the old f32-domain division was NOT exact
    }
}
