//! The battery pack, accumulated across the channels the strap volunteers it on. The reader is
//! stateful because the console arrives in chunks that split a line mid-value; the app feeds it and
//! reads back only complete values, plus the presence the strap states outright.

use crate::*;
use whoop_protocol::pack::{self, ConsoleScan, PackIdentity};

/// What the strap knows about the attached pack. Every field is independently absent until a source
/// carried a complete one.
#[derive(uniffi::Record, Clone, Debug, PartialEq)]
pub struct PackInfo {
    pub firmware: Option<String>,
    pub serial: Option<String>,
    pub bt_addr: Option<String>,
    pub hw_family: Option<u8>,
    pub hw_version: Option<u8>,
    pub colorway: Option<u8>,
    /// The pack's charge as the NFC channel reports it. A CROSS-CHECK on the battery-pack command
    /// reply, never its replacement — the reply is the reading the app displays.
    pub soc_pct: Option<f64>,
}

impl From<&PackIdentity> for PackInfo {
    fn from(i: &PackIdentity) -> Self {
        PackInfo {
            firmware: i.firmware.clone(),
            serial: i.serial.clone(),
            bt_addr: i.bt_addr.clone(),
            hw_family: i.hw_family,
            hw_version: i.hw_version,
            colorway: i.colorway,
            soc_pct: i.soc_deci_pct.map(|s| f64::from(s) / pack::SOC_DECI_PER_PCT),
        }
    }
}

/// What one frame said about the pack. `Detached` is the strap stating there is no pack — its own
/// removal signal, or a pack block it zeroed.
#[derive(uniffi::Enum, Clone, Debug, PartialEq)]
pub enum PackSignal {
    Attached,
    Detached,
    /// The frame carried values, already merged. `changed` is true when this one moved something,
    /// so a caller can log a change without logging the once-a-second charge report.
    Identity { info: PackInfo, changed: bool },
    /// A pack event named from the pack's own vocabulary but not decoded, with its raw body so a
    /// capture can pin the layout.
    Undecoded { name: String, body: String },
}

/// Accumulates the pack for one link: feed every CONSOLE_LOGS text and every frame. Interior-mutable
/// for the `&self` methods uniffi objects require.
#[derive(uniffi::Object)]
pub struct PackReader {
    inner: Mutex<Inner>,
}

#[derive(Default)]
struct Inner {
    console: ConsoleScan,
    id: PackIdentity,
}

#[uniffi::export]
impl PackReader {
    /// One per link. Dropping it drops the half-arrived console line with it, so a reconnect can
    /// never join a stale fragment onto a fresh chunk.
    #[uniffi::constructor]
    pub fn new() -> Arc<Self> {
        Arc::new(PackReader { inner: Mutex::new(Inner::default()) })
    }

    /// Feed one CONSOLE_LOGS text in arrival order. Returns the identity only when this chunk
    /// completed a value that is new or changed; a chunk ending mid-value returns nothing.
    pub fn push_console(&self, chunk: String) -> Option<PackInfo> {
        let mut inner = self.lock();
        if !inner.console.push(&chunk) {
            return None;
        }
        let scanned = inner.console.identity().clone();
        inner.id.absorb(&scanned).then(|| PackInfo::from(&inner.id))
    }

    /// Feed one complete frame. Returns what it said about the pack, or nothing when it said
    /// nothing — including any frame that failed its checksum.
    pub fn push_frame(&self, gen: Gen, bytes: Vec<u8>) -> Option<PackSignal> {
        let f = framing::decode(gen.into(), &bytes).ok()?;
        let mut inner = self.lock();
        let before = inner.id.clone();
        let signal = pack::read_frame(&f, &mut inner.id)?;
        Some(match signal {
            pack::PackSignal::Attached => PackSignal::Attached,
            pack::PackSignal::Detached => PackSignal::Detached,
            pack::PackSignal::Identity => PackSignal::Identity {
                changed: inner.id != before,
                info: PackInfo::from(&inner.id),
            },
            pack::PackSignal::Undecoded { name, body } => {
                PackSignal::Undecoded { name: name.to_string(), body }
            }
        })
    }

    /// Everything read so far on this link, or nothing if no complete value has arrived.
    pub fn info(&self) -> Option<PackInfo> {
        let inner = self.lock();
        (!inner.id.is_empty()).then(|| PackInfo::from(&inner.id))
    }
}

impl PackReader {
    fn lock(&self) -> std::sync::MutexGuard<'_, Inner> {
        self.inner.lock().expect("Mutex poisoned — a prior panic left the pack reader corrupted")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const HW_INFO_FRAME: &str = "aa0130000102aa8036961400433f746ab83e2000010c5742423541503031323633393500\
                                 0000f7381d2e31610d031e050001dd020e2eb457";
    const ATTACHED: &str = "aa0110000100208130e41500cfd7576a140e0000d3b899e3";
    const DETACHED: &str = "aa011000010020813018160047d8576a701d00008065c557";
    const INFO_ZEROED: &str = "aa012c0001002cd1308c6d00208d526af5681c00010000000000000000000000\
                              00000000000000000000000000010e005b84085f";

    fn bytes(hex: &str) -> Vec<u8> {
        whoop_protocol::bytes::from_hex(&hex.replace([' ', '\n'], "")).unwrap()
    }

    #[test]
    fn a_console_value_split_across_two_chunks_surfaces_once_and_whole() {
        let r = PackReader::new();
        assert!(r.push_console(" 42, 1: NFC COMMS: FW version: 3.30.5".into()).is_none());
        assert!(r.info().is_none(), "half a value reached the app");
        let got = r.push_console(".0\n".into()).expect("the completed line reported nothing");
        assert_eq!(got.firmware.as_deref(), Some("3.30.5.0"));
        assert!(r.push_console(" 42, 2: NFC COMMS: FW version: 3.30.5.0\n".into()).is_none());
    }

    #[test]
    fn the_event_fills_what_the_console_cannot_and_the_charge_is_a_percent() {
        let r = PackReader::new();
        let PackSignal::Identity { info, changed } = r.push_frame(Gen::Gen5, bytes(HW_INFO_FRAME)).unwrap()
        else {
            panic!("the hardware-information event did not read as an identity")
        };
        assert!(changed);
        assert_eq!(info.serial.as_deref(), Some("WBB5AP0126395"));
        assert_eq!(info.firmware.as_deref(), Some("3.30.5.0"));
        assert_eq!(info.soc_pct, Some(73.3));

        let PackSignal::Identity { changed, .. } = r.push_frame(Gen::Gen5, bytes(HW_INFO_FRAME)).unwrap()
        else {
            panic!("a repeat event stopped reading as an identity")
        };
        assert!(!changed, "an unchanged event reported a change");
    }

    #[test]
    fn presence_is_reported_from_the_straps_own_signals() {
        let r = PackReader::new();
        assert_eq!(r.push_frame(Gen::Gen5, bytes(ATTACHED)), Some(PackSignal::Attached));
        assert_eq!(r.push_frame(Gen::Gen5, bytes(DETACHED)), Some(PackSignal::Detached));
        assert_eq!(r.push_frame(Gen::Gen5, bytes(INFO_ZEROED)), Some(PackSignal::Detached));
        assert!(r.info().is_none(), "a presence signal invented an identity");
    }

    #[test]
    fn a_frame_that_is_not_about_the_pack_is_ignored() {
        let r = PackReader::new();
        assert!(r.push_frame(Gen::Gen5, GEN5_CLIENT_HELLO.to_vec()).is_none());
        assert!(r.push_frame(Gen::Gen5, vec![0xAA, 0x01]).is_none());
        assert!(r.push_frame(Gen::Gen5, Vec::new()).is_none());
        assert!(r.info().is_none());
    }
}
