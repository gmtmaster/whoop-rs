//! The battery pack's identity, as the STRAP reports it — the pack never talks to the phone. Two
//! sources of the same values: the CRC-gated type-54 event the strap forwards from its NFC poller,
//! and the same fields printed on the strap's console, which arrives in chunks that split a line
//! mid-value. Both fill one `PackIdentity`; only a complete, well-formed value is ever taken.

use crate::bytes::{to_hex, u16_at};
use crate::packet::{Frame, PacketType};

/// Charge in tenths of a percent divided into a percent.
pub const SOC_DECI_PER_PCT: f64 = 10.0;

/// Type-54 event carrying the pack's whole HW-information block.
const HW_INFO_EVENT: u8 = 20;
/// Type-54 event carrying only the pack's charge.
const SOC_EVENT: u8 = 2;
/// Bytes before an event's own body: `[pad][unix u32][ticks u32][pad]`.
const EVENT_HEADER: usize = 10;
/// HW-info body: family, serial, address, hw rev, four firmware octets, colorway, charge.
const HW_INFO_BODY: usize = 31;
const SERIAL_LEN: usize = 16;
const ADDR_LEN: usize = 6;
/// Charge is a percent in tenths, so nothing above this is a charge.
const SOC_DECI_MAX: u16 = 1000;

/// What the strap knows about the attached pack. Each field is independently optional: one source
/// carries a subset, and an unreadable value leaves the field alone rather than clearing it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PackIdentity {
    pub firmware: Option<String>,
    pub serial: Option<String>,
    pub bt_addr: Option<String>,
    pub hw_family: Option<u8>,
    pub hw_version: Option<u8>,
    pub colorway: Option<u8>,
    /// Charge in tenths of a percent. A CROSS-CHECK on the command reply's charge, never its
    /// replacement — the reply is the one the app reads.
    pub soc_deci_pct: Option<u16>,
}

impl PackIdentity {
    pub fn is_empty(&self) -> bool {
        *self == Self::default()
    }

    /// Take every value `other` carries, leaving a field it does not know alone. True when a field
    /// gained a value or changed one.
    pub fn absorb(&mut self, other: &PackIdentity) -> bool {
        let mut changed = take(&mut self.firmware, &other.firmware);
        changed |= take(&mut self.serial, &other.serial);
        changed |= take(&mut self.bt_addr, &other.bt_addr);
        changed |= take(&mut self.hw_family, &other.hw_family);
        changed |= take(&mut self.hw_version, &other.hw_version);
        changed |= take(&mut self.colorway, &other.colorway);
        changed |= take(&mut self.soc_deci_pct, &other.soc_deci_pct);
        changed
    }
}

fn take<T: Clone + PartialEq>(dst: &mut Option<T>, src: &Option<T>) -> bool {
    match src {
        Some(v) if dst.as_ref() != Some(v) => {
            *dst = Some(v.clone());
            true
        }
        _ => false,
    }
}

/// Pack identity from a type-54 `PUFFIN_EVENTS_FROM_STRAP` frame. A bad CRC, another packet type or
/// an event that is not about the pack yields nothing.
pub fn identity_from_event(f: &Frame) -> Option<PackIdentity> {
    if !f.crc_ok || f.packet() != PacketType::PuffinEventsFromStrap {
        return None;
    }
    let body = f.payload().get(EVENT_HEADER..)?;
    let mut id = PackIdentity::default();
    match f.cmd() {
        HW_INFO_EVENT => {
            if body.len() < HW_INFO_BODY {
                return None;
            }
            let serial_end = 1 + SERIAL_LEN;
            let addr_end = serial_end + ADDR_LEN;
            id.hw_family = Some(body[0]);
            id.serial = ascii_z(&body[1..serial_end]);
            id.bt_addr = Some(to_hex(&body[serial_end..addr_end]));
            id.hw_version = Some(body[addr_end]);
            id.firmware = Some(firmware_string(&body[addr_end + 1..addr_end + 5]));
            id.colorway = Some(body[addr_end + 5]);
            id.soc_deci_pct = u16_at(body, addr_end + 6).filter(|&s| s <= SOC_DECI_MAX);
        }
        SOC_EVENT => id.soc_deci_pct = u16_at(body, 0).filter(|&s| s <= SOC_DECI_MAX),
        _ => return None,
    }
    (!id.is_empty()).then_some(id)
}

/// The strap's `BATTERY_PACK_INFO` event body: the pack block, unprompted, in the same fields the
/// pack command answers with. The three bytes after the charge are NOT decoded — two values have
/// been seen and neither is explained, so naming them would be a guess.
const INFO_BODY: usize = ADDR_LEN + SERIAL_LEN + 2;

/// The pack block a `BATTERY_PACK_INFO` event carries, or `None` when the strap zeroed it — which is
/// the strap saying there is no pack, exactly as the zeroed command reply does.
fn identity_from_info_event(body: &[u8]) -> Option<PackIdentity> {
    if body.len() < INFO_BODY {
        return None;
    }
    let serial_end = ADDR_LEN + SERIAL_LEN;
    let addr = &body[..ADDR_LEN];
    let serial = ascii_z(&body[ADDR_LEN..serial_end]);
    let soc = u16_at(body, serial_end).filter(|&s| s <= SOC_DECI_MAX);
    if addr.iter().all(|&b| b == 0) && serial.is_none() && soc.unwrap_or(0) == 0 {
        return None;
    }
    Some(PackIdentity {
        serial,
        bt_addr: Some(to_hex(addr)),
        soc_deci_pct: soc,
        ..Default::default()
    })
}

/// What one frame said about the pack. `Detached` covers both the strap's own removal signal and a
/// pack block it zeroed: each is the strap stating there is no pack.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PackSignal {
    Attached,
    Detached,
    /// The frame carried values, already merged into the reader's identity.
    Identity,
    /// A pack event named from the pack's own vocabulary but not decoded here, with its raw body so
    /// a capture can pin the layout.
    Undecoded { name: &'static str, body: String },
}

/// The pack's own event vocabulary, carried in the `cmd` byte of a type-54 frame. Only the two the
/// codec decodes are absent here; the rest are named so an arriving one is identifiable in a log.
fn pack_event_name(cmd: u8) -> Option<&'static str> {
    Some(match cmd {
        1 => "ERROR",
        3 => "USB_CONNECTED",
        4 => "USB_DISCONNECTED",
        5 => "CHARGING_ON",
        6 => "CHARGING_OFF",
        7 => "BLE_CONNECTED",
        8 => "BLE_DISCONNECTED",
        9 => "DOUBLE_TAP",
        10 => "STRAP_DETECTED",
        11 => "STRAP_REMOVED",
        12 => "TRIM_ALL_DATA_START",
        13 => "TRIM_ALL_DATA_END",
        14 => "BOOT_REPORT",
        15 => "SHIPMODE_SET",
        16 => "SHIPMODE_CLEAR",
        17 => "EXTENDED_FG_INFO",
        18 => "BPK_BATTERY_HEALTH",
        19 => "WPT_HEALTH",
        21 => "REBOOT_REASON",
        22 => "MODULE_FAILURE_REASON",
        23 => "WPT_RESET",
        50 => "PUFF_LOG_PACKET",
        _ => return None,
    })
}

/// Read one frame for anything it says about the pack, merging any values into `id`. Presence is a
/// fact the strap states outright, so it is reported whether or not a value moved.
///
/// The two channels have SEPARATE vocabularies and must never be folded together: 21/22 are the
/// strap's attach/detach on type 48, while on type 54 the same numbers are the pack's own
/// `REBOOT_REASON` and `MODULE_FAILURE_REASON`.
pub fn read_frame(f: &Frame, id: &mut PackIdentity) -> Option<PackSignal> {
    if !f.crc_ok {
        return None;
    }
    let mut merge = |got: PackIdentity| {
        id.absorb(&got);
        PackSignal::Identity
    };
    match f.packet() {
        PacketType::Event => match f.cmd() {
            crate::event::BATTERY_PACK_CONNECTED => Some(PackSignal::Attached),
            crate::event::BATTERY_PACK_REMOVED => Some(PackSignal::Detached),
            crate::event::BATTERY_PACK_INFO => Some(
                identity_from_info_event(f.payload().get(EVENT_HEADER..)?)
                    .map_or(PackSignal::Detached, &mut merge),
            ),
            _ => None,
        },
        PacketType::PuffinEventsFromStrap => Some(match identity_from_event(f) {
            Some(got) => merge(got),
            None => PackSignal::Undecoded {
                name: pack_event_name(f.cmd())?,
                body: to_hex(f.payload().get(EVENT_HEADER..).unwrap_or(&[])),
            },
        }),
        _ => None,
    }
}

/// A NUL-terminated printable-ASCII field, or `None` when it is empty or holds a non-printable byte.
fn ascii_z(b: &[u8]) -> Option<String> {
    let text = &b[..b.iter().position(|&c| c == 0).unwrap_or(b.len())];
    if text.is_empty() || !text.iter().all(|&c| (32..=126).contains(&c)) {
        return None;
    }
    Some(text.iter().map(|&c| c as char).collect())
}

fn firmware_string(o: &[u8]) -> String {
    format!("{}.{}.{}.{}", o[0], o[1], o[2], o[3])
}

/// The console marker every pack line carries, after the strap's own `<task>, <ticks>: ` prefix.
const MARKER: &str = "NFC COMMS: ";
/// Longer than any line the strap prints; a line reaching it lost its newline and is dropped.
const MAX_LINE: usize = 512;

/// Reassembles the strap's chunked console text and reads a line only once its newline has arrived,
/// so a value split across two chunks is never read half-formed. Feed `console::text` of each
/// CONSOLE_LOGS frame in arrival order.
#[derive(Debug, Default)]
pub struct ConsoleScan {
    pending: String,
    overflowed: bool,
    id: PackIdentity,
}

impl ConsoleScan {
    pub fn new() -> Self {
        Self::default()
    }

    /// Append one console chunk and read every line its newline completed. True when a field gained
    /// a value or changed one. A chunk ending mid-value contributes nothing until the rest arrives.
    pub fn push(&mut self, chunk: &str) -> bool {
        let mut changed = false;
        for ch in chunk.chars() {
            if ch == '\n' {
                let line = std::mem::take(&mut self.pending);
                if self.overflowed {
                    self.overflowed = false;
                } else {
                    changed |= self.read_line(&line);
                }
            } else if self.pending.len() >= MAX_LINE {
                self.pending.clear();
                self.overflowed = true;
            } else {
                self.pending.push(ch);
            }
        }
        changed
    }

    pub fn identity(&self) -> &PackIdentity {
        &self.id
    }

    fn read_line(&mut self, line: &str) -> bool {
        let Some(at) = line.find(MARKER) else { return false };
        let Some((key, val)) = line[at + MARKER.len()..].split_once(':') else { return false };
        let (key, val) = (key.trim(), val.trim());
        let mut got = PackIdentity::default();
        match key {
            "FW version" => got.firmware = console_firmware(val),
            "HW version" => got.hw_version = val.parse().ok(),
            "HW family" => got.hw_family = val.parse().ok(),
            "Colorway" => got.colorway = val.parse().ok(),
            "BT addr" => got.bt_addr = console_addr(val),
            "BPK SoC" | "Unwrapping BPK SoC" => {
                got.soc_deci_pct = val.parse().ok().filter(|&s| s <= SOC_DECI_MAX);
            }
            _ => return false,
        }
        self.id.absorb(&got)
    }
}

/// Exactly four numeric octets, re-joined so the string matches the one the wire event builds. A
/// truncated `3.30.5` is three parts and yields nothing.
fn console_firmware(v: &str) -> Option<String> {
    let o: Vec<u8> = v.split('.').map(|p| p.parse::<u8>()).collect::<Result<_, _>>().ok()?;
    (o.len() == 4).then(|| firmware_string(&o))
}

/// Exactly `ADDR_LEN` bytes of hex, lowercased; a half-arrived address is short and yields nothing.
fn console_addr(v: &str) -> Option<String> {
    (v.len() == ADDR_LEN * 2 && v.bytes().all(|c| c.is_ascii_hexdigit()))
        .then(|| v.to_ascii_lowercase())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::family::Family;
    use crate::framing;

    /// The strap's console arrives in fixed 50-character chunks; these fourteen are one real pack
    /// attach verbatim, so `FW version: 3.30.5` ends chunk 4 and its `.0` opens chunk 5.
    const CHUNKS: [&str; 14] = [
        "42, 309949629: LISTENER: RTC sent to battery pack ",
        "poller - 1786003267\n 42, 309949669: NFC COMMS: HW ",
        "family : 12\n 42, 309949669: NFC COMMS: BT addr : f",
        "7381d2e3161\n 42, 309949669: NFC COMMS: HW version ",
        ": 13\n 42, 309949669: NFC COMMS: FW version: 3.30.5",
        ".0\n 42, 309949669: NFC COMMS: Colorway : 1\n 42, 30",
        "9949669: NFC COMMS: BPK SoC : 733\n 42, 309949669: ",
        "NFC COMMS: Unwrapping BPK HW information event : 2",
        "0\n 42, 309950329: LISTENER: Charging On\n 42, 30995",
        "0329: LISTENER: Battery Pack Installed\n 42, 309950",
        "329: LP5562: Setting current array to blue 0, gree",
        "n 175, red 0\n 42, 309955029: NFC COMMS: Unwrapping",
        " BPK SoC : 732 \n 42, 309975089: NFC COMMS: Unwrapp",
        "ing BPK SoC : 730 \n 42, 309995239: NFC COMMS: Unwr",
    ];

    /// The type-54 HW-information event captured from the same attach, CRC32-valid on the wire.
    const HW_INFO_FRAME: &str = "aa0130000102aa8036961400433f746ab83e2000010c574242354150303132363339350000\
                                 00f7381d2e31610d031e050001dd020e2eb457";
    /// The type-54 charge-only event from the same attach.
    const SOC_FRAME: &str = "aa0114000102a07036cd02005740746a1e05040001bf0200e32c7a9b";

    fn frame(hex: &str) -> Frame {
        let bytes = crate::bytes::from_hex(&hex.replace([' ', '\n'], "")).unwrap();
        framing::decode(Family::Gen5, &bytes).unwrap()
    }

    fn scan(chunks: &[&str]) -> ConsoleScan {
        let mut s = ConsoleScan::new();
        for c in chunks {
            s.push(c);
        }
        s
    }

    #[test]
    fn real_chunked_attach_yields_every_field() {
        let id = scan(&CHUNKS).identity().clone();
        assert_eq!(id.firmware.as_deref(), Some("3.30.5.0"));
        assert_eq!(id.bt_addr.as_deref(), Some("f7381d2e3161"));
        assert_eq!(id.hw_family, Some(12));
        assert_eq!(id.hw_version, Some(13));
        assert_eq!(id.colorway, Some(1));
        assert_eq!(id.soc_deci_pct, Some(730));
    }

    /// The chunk boundary the whole reassembly exists for: `3.30.5` is a complete-looking value on a
    /// frame of its own, and must not be read until the newline after `.0`.
    #[test]
    fn a_value_split_across_chunks_is_never_read_half_formed() {
        assert_eq!(scan(&CHUNKS[..5]).identity().firmware, None, "3.30.5 read as a firmware");
        assert_eq!(scan(&CHUNKS[..6]).identity().firmware.as_deref(), Some("3.30.5.0"));
        assert_eq!(scan(&CHUNKS[..3]).identity().bt_addr, None, "half an address read as one");
        assert_eq!(scan(&CHUNKS[..4]).identity().bt_addr.as_deref(), Some("f7381d2e3161"));
    }

    /// A stream that stops mid-value never emits: the last line has no newline, so it is not read.
    #[test]
    fn a_stream_ending_mid_value_emits_nothing() {
        let mut s = ConsoleScan::new();
        assert!(!s.push(" 42, 1: NFC COMMS: FW version: 3.30.5"));
        assert!(s.identity().is_empty());
        assert!(!s.push(".0"));
        assert!(s.identity().is_empty(), "a complete value was read before its newline");
        assert!(s.push("\n"), "the newline is what releases the value");
        assert_eq!(s.identity().firmware.as_deref(), Some("3.30.5.0"));
    }

    /// One character at a time is the worst split there is, and must reach the same answer.
    #[test]
    fn byte_at_a_time_matches_the_native_chunking() {
        let whole: String = CHUNKS.concat();
        let mut s = ConsoleScan::new();
        for ch in whole.chars() {
            s.push(&ch.to_string());
        }
        assert_eq!(*s.identity(), *scan(&CHUNKS).identity());
    }

    #[test]
    fn malformed_and_unrelated_lines_leave_the_last_good_value_standing() {
        let mut s = scan(&CHUNKS);
        let before = s.identity().clone();
        s.push(" 42, 1: NFC COMMS: FW version: 3.30\n");
        s.push(" 42, 1: NFC COMMS: FW version: x.y.z.w\n");
        s.push(" 42, 1: NFC COMMS: BT addr : f7381d2e31\n");
        s.push(" 42, 1: NFC COMMS: BT addr : zzzzzzzzzzzz\n");
        s.push(" 42, 1: NFC COMMS: Unwrapping BPK HW information event : 20\n");
        s.push(" 42, 1: BLE: History burst success. Trim: 0x00000007:0000ce20\n");
        assert_eq!(*s.identity(), before);
    }

    /// A line that never ends is dropped, and its remainder must not be read as a line of its own.
    #[test]
    fn a_line_without_a_newline_is_dropped_not_buffered_forever() {
        let mut s = ConsoleScan::new();
        s.push(&"x".repeat(MAX_LINE * 3));
        s.push(" 42, 1: NFC COMMS: FW version: 9.9.9.9\n");
        assert_eq!(s.identity().firmware, None, "an overflowed line's tail was read");
        s.push(" 42, 1: NFC COMMS: FW version: 9.9.9.9\n");
        assert_eq!(s.identity().firmware.as_deref(), Some("9.9.9.9"));
    }

    #[test]
    fn the_hw_information_event_decodes_the_same_pack() {
        let id = identity_from_event(&frame(HW_INFO_FRAME)).unwrap();
        assert_eq!(id.firmware.as_deref(), Some("3.30.5.0"));
        assert_eq!(id.serial.as_deref(), Some("WBB5AP0126395"));
        assert_eq!(id.bt_addr.as_deref(), Some("f7381d2e3161"));
        assert_eq!(id.hw_family, Some(12));
        assert_eq!(id.hw_version, Some(13));
        assert_eq!(id.colorway, Some(1));
        assert_eq!(id.soc_deci_pct, Some(733));
    }

    /// The two sources are independent encodings of one attach, so every field they share must agree.
    #[test]
    fn the_event_and_the_console_agree_field_for_field() {
        let wire = identity_from_event(&frame(HW_INFO_FRAME)).unwrap();
        let text = scan(&CHUNKS[..8]).identity().clone();
        assert_eq!(text.firmware, wire.firmware);
        assert_eq!(text.bt_addr, wire.bt_addr);
        assert_eq!(text.hw_family, wire.hw_family);
        assert_eq!(text.hw_version, wire.hw_version);
        assert_eq!(text.colorway, wire.colorway);
        assert_eq!(text.soc_deci_pct, wire.soc_deci_pct);
    }

    #[test]
    fn the_charge_event_carries_only_the_charge() {
        let id = identity_from_event(&frame(SOC_FRAME)).unwrap();
        assert_eq!(id.soc_deci_pct, Some(703));
        assert_eq!(id.firmware, None);
        assert_eq!(id.serial, None);
    }

    #[test]
    fn a_corrupt_or_unrelated_frame_yields_nothing() {
        let mut bytes = crate::bytes::from_hex(HW_INFO_FRAME).unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 0xFF;
        let bad = framing::decode(Family::Gen5, &bytes).unwrap();
        assert!(!bad.crc_ok);
        assert!(identity_from_event(&bad).is_none(), "a bad CRC decoded into an identity");

        let console = framing::encode(Family::Gen5, PacketType::ConsoleLogs.to_u8(), 0, 0, &[1, 2, 3, 4]);
        assert!(identity_from_event(&framing::decode(Family::Gen5, &console).unwrap()).is_none());
    }

    /// A HW-info event cut short must yield nothing rather than a partly-filled identity. Built by
    /// hand, since `framing::encode` pads an inner back to a 4-byte boundary and would restore the
    /// length this is testing the absence of.
    fn unpadded_gen5(inner: &[u8]) -> Frame {
        let decl = (inner.len() + 4) as u16;
        let mut f = vec![0xAA, 0x01];
        f.extend_from_slice(&decl.to_le_bytes());
        f.extend_from_slice(&[0x00, 0x01]);
        f.extend_from_slice(&crate::crc::crc16_modbus(&f[0..6]).to_le_bytes());
        f.extend_from_slice(inner);
        f.extend_from_slice(&crate::crc::crc32_zlib(inner).to_le_bytes());
        framing::decode(Family::Gen5, &f).unwrap()
    }

    #[test]
    fn a_truncated_hw_information_event_yields_nothing() {
        let full = crate::bytes::from_hex(HW_INFO_FRAME).unwrap();
        let inner = &full[8..full.len() - 4];
        for cut in 1..inner.len() {
            let f = unpadded_gen5(&inner[..cut]);
            assert!(f.crc_ok);
            assert!(identity_from_event(&f).is_none(), "a {cut}-byte inner produced an identity");
        }
        assert!(identity_from_event(&unpadded_gen5(inner)).is_some());
    }

    /// Real frames from two straps: the strap's own attach/detach, and its unprompted pack block in
    /// both the filled and the zeroed form.
    const ATTACHED: &str = "aa0110000100208130e41500cfd7576a140e0000d3b899e3";
    const DETACHED: &str = "aa011000010020813018160047d8576a701d00008065c557";
    const INFO_PRESENT: &str = "aa012c0001002cd130e56d00cfd7576aae271c0001ccb7a6dc16095742423541\
                               50303131333635350000004903010c00ba269563";
    const INFO_ZEROED: &str = "aa012c0001002cd1308c6d00208d526af5681c00010000000000000000000000\
                               00000000000000000000000000010e005b84085f";
    /// The one real `WPT_HEALTH` frame we hold. Its body is not decoded — one sample cannot pin a
    /// layout — so it must surface named and raw rather than as an invented value.
    const WPT_HEALTH: &str = "aa0118000102a320366713007d94526a7a340800010042010100000037d1d87e";

    fn read(hex: &str) -> (Option<PackSignal>, PackIdentity) {
        let mut id = PackIdentity::default();
        let sig = read_frame(&frame(hex), &mut id);
        (sig, id)
    }

    #[test]
    fn the_strap_states_presence_outright() {
        assert_eq!(read(ATTACHED).0, Some(PackSignal::Attached));
        assert_eq!(read(DETACHED).0, Some(PackSignal::Detached));
    }

    /// A zeroed pack block is the strap saying there is no pack, exactly as the zeroed command reply
    /// is — never a pack with an empty serial at 0%.
    #[test]
    fn the_unprompted_pack_block_carries_values_or_states_absence() {
        let (sig, id) = read(INFO_PRESENT);
        assert_eq!(sig, Some(PackSignal::Identity));
        assert_eq!(id.serial.as_deref(), Some("WBB5AP0113655"));
        assert_eq!(id.bt_addr.as_deref(), Some("ccb7a6dc1609"));
        assert_eq!(id.soc_deci_pct, Some(841));

        let (sig, id) = read(INFO_ZEROED);
        assert_eq!(sig, Some(PackSignal::Detached));
        assert!(id.is_empty(), "a zeroed block filled an identity");
    }

    /// The two channels reuse the same numbers for different things. On type 54, 21 and 22 are the
    /// pack's own reboot/failure reports and must never read as the strap's attach and detach.
    #[test]
    fn the_two_event_vocabularies_are_never_folded_together() {
        for (cmd, name) in [(21u8, "REBOOT_REASON"), (22, "MODULE_FAILURE_REASON")] {
            let wire = framing::encode(
                Family::Gen5,
                PacketType::PuffinEventsFromStrap.to_u8(),
                0,
                cmd,
                &[0u8; 16],
            );
            let sig = read_frame(&framing::decode(Family::Gen5, &wire).unwrap(), &mut PackIdentity::default());
            assert_eq!(sig, Some(PackSignal::Undecoded { name, body: "00000000000000".into() }));
        }
    }

    /// Named from the pack's own vocabulary, body kept raw. Nothing decodes it: one frame cannot
    /// pin a layout, and inventing one would be worse than reporting the bytes.
    #[test]
    fn an_undecoded_pack_event_surfaces_named_with_its_raw_body() {
        let (sig, id) = read(WPT_HEALTH);
        assert_eq!(
            sig,
            Some(PackSignal::Undecoded { name: "WPT_HEALTH", body: "00420101000000".into() })
        );
        assert!(id.is_empty());
    }

    #[test]
    fn a_frame_with_nothing_to_say_about_the_pack_says_nothing() {
        let hr = framing::encode(Family::Gen5, PacketType::RealtimeData.to_u8(), 0, 0, &[0u8; 8]);
        assert!(read_frame(&framing::decode(Family::Gen5, &hr).unwrap(), &mut PackIdentity::default()).is_none());
        let wrist = framing::encode(Family::Gen5, PacketType::Event.to_u8(), 0, crate::event::WRIST_ON, &[0u8; 12]);
        assert!(read_frame(&framing::decode(Family::Gen5, &wrist).unwrap(), &mut PackIdentity::default()).is_none());
    }

    #[test]
    fn absorb_fills_gaps_and_reports_only_real_change() {
        let mut a = PackIdentity { firmware: Some("3.30.5.0".into()), ..Default::default() };
        let b = PackIdentity { serial: Some("WBB5AP0126395".into()), ..Default::default() };
        assert!(a.absorb(&b));
        assert!(!a.absorb(&b), "absorbing the same values reported a change");
        assert_eq!(a.firmware.as_deref(), Some("3.30.5.0"));
        assert!(!a.absorb(&PackIdentity::default()), "an empty identity cleared a known value");
        assert_eq!(a.serial.as_deref(), Some("WBB5AP0126395"));
    }
}
