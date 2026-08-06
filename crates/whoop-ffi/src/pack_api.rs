//! The battery pack's identity, accumulated across the two channels the strap volunteers it on. The
//! reader is stateful because the console arrives in chunks that split a line mid-value; the app
//! feeds it and reads back only complete values.

use crate::*;
use whoop_protocol::pack::{self, ConsoleScan, PackIdentity};

/// What the strap knows about the attached pack. Every field is independently absent until a source
/// carried a complete one.
#[derive(uniffi::Record, Clone, PartialEq)]
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

/// Accumulates the pack's identity for one link: feed every CONSOLE_LOGS text and every frame, and
/// it reports the moment a complete new value arrived. Interior-mutable for the `&self` methods
/// uniffi objects require.
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

    /// Feed one complete frame. Only a pack event carries anything; every other frame, and any frame
    /// that failed its checksum, returns nothing.
    pub fn push_frame(&self, gen: Gen, bytes: Vec<u8>) -> Option<PackInfo> {
        let f = framing::decode(gen.into(), &bytes).ok()?;
        let got = pack::identity_from_event(&f)?;
        let mut inner = self.lock();
        inner.id.absorb(&got).then(|| PackInfo::from(&inner.id))
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

    fn frame() -> Vec<u8> {
        whoop_protocol::bytes::from_hex(&HW_INFO_FRAME.replace([' ', '\n'], "")).unwrap()
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
        let got = r.push_frame(Gen::Gen5, frame()).unwrap();
        assert_eq!(got.serial.as_deref(), Some("WBB5AP0126395"));
        assert_eq!(got.firmware.as_deref(), Some("3.30.5.0"));
        assert_eq!(got.soc_pct, Some(73.3));
        assert!(r.push_frame(Gen::Gen5, frame()).is_none(), "an unchanged event reported a change");
    }

    #[test]
    fn a_frame_that_is_not_a_pack_event_is_ignored() {
        let r = PackReader::new();
        assert!(r.push_frame(Gen::Gen5, GEN5_CLIENT_HELLO.to_vec()).is_none());
        assert!(r.push_frame(Gen::Gen5, vec![0xAA, 0x01]).is_none());
        assert!(r.push_frame(Gen::Gen5, Vec::new()).is_none());
        assert!(r.info().is_none());
    }
}
