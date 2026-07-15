use std::time::Duration;

use async_trait::async_trait;
use btleplug::api::{BDAddr, Central, Manager as _, Peripheral as _, PeripheralProperties, ScanFilter, WriteType};
use btleplug::platform::{Adapter, Manager, Peripheral, PeripheralId};
use futures::stream::BoxStream;
use futures::StreamExt;
use uuid::Uuid;

use ble_core::{BleError, BleTransport, Notification};

const DISCOVER_TIMEOUT: Duration = Duration::from_secs(12);
const POLL: Duration = Duration::from_millis(400);
/// How long to let the scan populate before reading serials (identity needs a connect, which WinRT
/// dislikes mid-scan — so we gather candidates first, then stop scanning and read them).
const SERIAL_SCAN_WINDOW: Duration = Duration::from_secs(8);

/// A btleplug-backed connection to a device advertising `service`. `target_name` optionally pins which
/// band to attach to (its advertised local name) when several are in range.
/// Standard SIG chars carrying the band's identity, readable without a bond.
const GATT_DEVICE_NAME: Uuid = Uuid::from_u128(0x00002a00_0000_1000_8000_00805f9b34fb);
const GATT_MODEL_NUMBER: Uuid = Uuid::from_u128(0x00002a24_0000_1000_8000_00805f9b34fb);
const GATT_SERIAL_NUMBER: Uuid = Uuid::from_u128(0x00002a25_0000_1000_8000_00805f9b34fb);
const GATT_FIRMWARE_REV: Uuid = Uuid::from_u128(0x00002a26_0000_1000_8000_00805f9b34fb);
const GATT_HARDWARE_REV: Uuid = Uuid::from_u128(0x00002a27_0000_1000_8000_00805f9b34fb);

pub struct BtleplugTransport {
    service: Uuid,
    target_name: Option<String>,
    target_address: Option<String>,
    target_serial: Option<String>,
    // Held for the transport's lifetime: the peripheral keeps only a Weak ref to the adapter, so
    // dropping it would kill disconnect/CentralEvent delivery for a mid-session link drop.
    adapter: Option<Adapter>,
    peripheral: Option<Peripheral>,
}

impl BtleplugTransport {
    pub fn new(service: Uuid) -> Self {
        BtleplugTransport {
            service,
            target_name: None,
            target_address: None,
            target_serial: None,
            adapter: None,
            peripheral: None,
        }
    }

    pub fn with_target_name(mut self, name: impl Into<String>) -> Self {
        self.target_name = Some(name.into());
        self
    }

    /// Pin the band by BLE address (the surest discriminator when several WHOOPs are in range).
    pub fn with_target_address(mut self, address: impl Into<String>) -> Self {
        self.target_address = Some(address.into());
        self
    }

    /// Pin the band by serial (full or suffix). The name/serial isn't advertised, so this connects to
    /// each candidate and reads its serial char — the reliable discriminator when addresses aren't known.
    pub fn with_target_serial(mut self, serial: impl Into<String>) -> Self {
        self.target_serial = Some(serial.into());
        self
    }

    fn peripheral(&self) -> Result<&Peripheral, BleError> {
        self.peripheral.as_ref().ok_or(BleError::NotConnected)
    }

    async fn characteristic(&self, uuid: Uuid) -> Result<btleplug::api::Characteristic, BleError> {
        self.peripheral()?
            .characteristics()
            .into_iter()
            .find(|c| c.uuid == uuid)
            .ok_or(BleError::NoCharacteristic(uuid))
    }
}

fn backend<E: std::fmt::Display>(e: E) -> BleError {
    BleError::Backend(e.to_string())
}

/// Connect, then discover services — retrying discovery, since a co-resident (multi-central) connection
/// can race and hand back an empty characteristic set on the first pass.
async fn connect_and_discover(p: &Peripheral) -> Result<(), BleError> {
    const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
    match tokio::time::timeout(CONNECT_TIMEOUT, p.connect()).await {
        Ok(r) => r.map_err(backend)?,
        Err(_) => return Err(BleError::Backend("connect timed out".into())),
    }
    for _ in 0..4 {
        p.discover_services().await.map_err(backend)?;
        if !p.characteristics().is_empty() {
            return Ok(());
        }
        tokio::time::sleep(Duration::from_millis(300)).await;
    }
    Ok(())
}

/// Acquire the first BLE adapter (the machine's default controller).
async fn first_adapter() -> Result<Adapter, BleError> {
    Manager::new()
        .await
        .map_err(backend)?
        .adapters()
        .await
        .map_err(backend)?
        .into_iter()
        .next()
        .ok_or(BleError::NoAdapter)
}

/// The untargeted-WHOOP heuristic: advertises the vendor service, or has a "WHOOP…" local name. Used
/// when no exact name is pinned (the pinned case demands an exact local-name match instead).
fn matches_whoop(props: &PeripheralProperties, service: Uuid) -> bool {
    props.services.contains(&service) || props.local_name.as_deref().is_some_and(|n| n.starts_with("WHOOP"))
}

/// A band seen during a scan, with the identity read back from GATT (empty if the connect/read failed).
pub struct Scanned {
    pub address: String,
    pub rssi: Option<i16>,
    pub name: String,
    pub serial: String,
    pub model: String,
    pub firmware: String,
    pub hardware: String,
}

/// Scan for `secs`, then for each WHOOP found connect briefly (no bond) and read its identity from the
/// standard GATT chars — the name/serial/hardware aren't in the advertisement, only over a connection.
pub async fn scan_whoops(service: Uuid, secs: u64) -> Result<Vec<Scanned>, BleError> {
    let adapter = first_adapter().await?;
    adapter.start_scan(ScanFilter { services: vec![service] }).await.map_err(backend)?;
    tokio::time::sleep(Duration::from_secs(secs)).await;
    let peripherals = adapter.peripherals().await.map_err(backend)?;
    let _ = adapter.stop_scan().await;

    let mut out = Vec::new();
    for p in peripherals {
        let Some(props) = p.properties().await.map_err(backend)? else { continue };
        if !matches_whoop(&props, service) {
            continue;
        }
        let mut band = Scanned {
            address: props.address.to_string(),
            rssi: props.rssi,
            name: props.local_name.unwrap_or_default(),
            serial: String::new(),
            model: String::new(),
            firmware: String::new(),
            hardware: String::new(),
        };
        if p.connect().await.is_ok() && p.discover_services().await.is_ok() {
            let name = read_char(&p, GATT_DEVICE_NAME).await;
            if !name.is_empty() {
                band.name = name;
            }
            band.serial = read_char(&p, GATT_SERIAL_NUMBER).await;
            band.model = read_char(&p, GATT_MODEL_NUMBER).await;
            band.firmware = read_char(&p, GATT_FIRMWARE_REV).await;
            band.hardware = read_char(&p, GATT_HARDWARE_REV).await;
            let _ = p.disconnect().await;
        }
        out.push(band);
    }
    Ok(out)
}

/// Read one standard-string GATT char, or "" if it isn't present / can't be read.
async fn read_char(p: &Peripheral, uuid: Uuid) -> String {
    match p.characteristics().into_iter().find(|c| c.uuid == uuid) {
        Some(c) => p.read(&c).await.map(|b| ble_core::gatt_string(&b)).unwrap_or_default(),
        None => String::new(),
    }
}

/// Poll the adapter until a peripheral advertising `service` (and matching `target_name`, if set) shows
/// up, or time out. btleplug's Linux ScanFilter can leak other apps' matches, so we re-check the service.
async fn discover(adapter: &Adapter, service: Uuid, target_name: &Option<String>) -> Result<Peripheral, BleError> {
    let deadline = tokio::time::Instant::now() + DISCOVER_TIMEOUT;
    loop {
        for p in adapter.peripherals().await.map_err(backend)? {
            let Some(props) = p.properties().await.map_err(backend)? else { continue };
            // A pinned name MUST match exactly — otherwise the service heuristic would attach to the wrong
            // band (several in range). Fall back to the heuristic only when no name is set.
            let matched = match target_name {
                Some(want) => props.local_name.as_deref() == Some(want.as_str()),
                None => matches_whoop(&props, service),
            };
            if matched {
                return Ok(p);
            }
        }
        if tokio::time::Instant::now() >= deadline {
            return Err(BleError::NotFound);
        }
        tokio::time::sleep(POLL).await;
    }
}

/// Find a band by serial among already-scanned candidates: connect to each WHOOP, read its serial/name
/// (unbonded), and return the first whose serial ends with `serial` (so a suffix like "409" works).
/// Non-matches are dropped. Returns already connected + service-discovered. Caller stops the scan first.
async fn discover_by_serial(adapter: &Adapter, service: Uuid, serial: &str) -> Result<Peripheral, BleError> {
    let want = serial.to_ascii_uppercase();
    for p in adapter.peripherals().await.map_err(backend)? {
        let Some(props) = p.properties().await.map_err(backend)? else { continue };
        if !matches_whoop(&props, service) {
            continue;
        }
        if p.connect().await.is_err() {
            continue; // held by another central / out of range — skip
        }
        if p.discover_services().await.is_err() {
            let _ = p.disconnect().await;
            continue;
        }
        match read_serial(&p).await {
            Some(id) if id.to_ascii_uppercase().ends_with(&want) => return Ok(p),
            _ => {
                let _ = p.disconnect().await;
            }
        }
    }
    Err(BleError::NotFound)
}

/// Read the band's serial (or, failing that, its name) from the standard chars — no bond required.
async fn read_serial(p: &Peripheral) -> Option<String> {
    for uuid in [GATT_SERIAL_NUMBER, GATT_DEVICE_NAME] {
        let s = read_char(p, uuid).await;
        if !s.is_empty() {
            return Some(s);
        }
    }
    None
}

#[async_trait]
impl BleTransport for BtleplugTransport {
    async fn connect(&mut self) -> Result<(), BleError> {
        let adapter = first_adapter().await?;
        let peripheral = if let Some(addr) = &self.target_address {
            // Reach the band directly by address (via `add_peripheral`) — no advertisement needed, so a
            // bonded band that's connected to another central (e.g. the phone) is still reachable.
            let bdaddr: BDAddr = addr.parse().map_err(|_| BleError::NotFound)?;
            let p = adapter.add_peripheral(&PeripheralId::from(bdaddr)).await.map_err(backend)?;
            connect_and_discover(&p).await?;
            p
        } else if let Some(serial) = &self.target_serial {
            // Serial isn't advertised — let candidates show up, stop scanning, then connect + read each.
            adapter.start_scan(ScanFilter { services: vec![self.service] }).await.map_err(backend)?;
            tokio::time::sleep(SERIAL_SCAN_WINDOW).await;
            let _ = adapter.stop_scan().await;
            discover_by_serial(&adapter, self.service, serial).await?
        } else {
            adapter.start_scan(ScanFilter { services: vec![self.service] }).await.map_err(backend)?;
            let p = discover(&adapter, self.service, &self.target_name).await?;
            connect_and_discover(&p).await?;
            let _ = adapter.stop_scan().await;
            p
        };

        // MTU is auto-negotiated (btleplug exposes mtu() but no setter); WinRT updates it asynchronously,
        // so poll briefly. Our largest write is the ~48-byte R22 config frame, which needs MTU >= ~51 —
        // warn if it never rises above the 23-byte ATT default.
        let mut mtu = peripheral.mtu();
        for _ in 0..10 {
            if mtu >= 64 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
            mtu = peripheral.mtu();
        }
        if mtu < 64 {
            tracing::warn!("negotiated MTU {mtu} may be too small for config writes (need ~51)");
        }

        self.adapter = Some(adapter);
        self.peripheral = Some(peripheral);
        Ok(())
    }

    // On Windows the command channel needs an OS bond; pair via WinRT if not already bonded. macOS bonds
    // on first encrypted access; Linux needs an external BlueZ agent (default no-op there).
    #[cfg(windows)]
    async fn ensure_paired(&self) -> Result<(), BleError> {
        // WinRT COM objects are !Send, so run the pairing on its own blocking thread + local runtime.
        let address = self.peripheral()?.address().to_string();
        tokio::task::spawn_blocking(move || {
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .map_err(|e| BleError::Pairing(e.to_string()))?
                .block_on(win_pair::ensure_bonded(&address))
        })
        .await
        .map_err(|e| BleError::Pairing(e.to_string()))?
    }

    async fn subscribe(&self, characteristic: Uuid) -> Result<(), BleError> {
        const SUBSCRIBE_TIMEOUT: Duration = Duration::from_secs(5);
        let c = self.characteristic(characteristic).await?;
        match tokio::time::timeout(SUBSCRIBE_TIMEOUT, self.peripheral()?.subscribe(&c)).await {
            Ok(r) => r.map_err(backend),
            Err(_) => Err(BleError::Backend("subscribe timed out".into())),
        }
    }

    async fn write(&self, characteristic: Uuid, data: &[u8], with_response: bool) -> Result<(), BleError> {
        // A confirmed WinRT write can block its own thread waiting for an ATT ack; run it on a blocking
        // thread + local runtime (like ensure_paired) so a stall can't freeze the executor and the timeout
        // can actually fire. The strap still receives the write PDU, so it replies on notify regardless.
        const WRITE_TIMEOUT: Duration = Duration::from_secs(5);
        let c = self.characteristic(characteristic).await?;
        let kind = if with_response { WriteType::WithResponse } else { WriteType::WithoutResponse };
        let peripheral = self.peripheral()?.clone();
        let data = data.to_vec();
        let task = tokio::task::spawn_blocking(move || {
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .map_err(|e| BleError::Backend(e.to_string()))?
                .block_on(peripheral.write(&c, &data, kind))
                .map_err(backend)
        });
        match tokio::time::timeout(WRITE_TIMEOUT, task).await {
            Ok(Ok(r)) => r,
            Ok(Err(join)) => Err(BleError::Backend(format!("write task: {join}"))),
            Err(_) => Err(BleError::Backend("write timed out".into())),
        }
    }

    async fn read(&self, characteristic: Uuid) -> Result<Vec<u8>, BleError> {
        let c = self.characteristic(characteristic).await?;
        self.peripheral()?.read(&c).await.map_err(backend)
    }

    async fn notifications(&self) -> Result<BoxStream<'static, Notification>, BleError> {
        let stream = self.peripheral()?.notifications().await.map_err(backend)?;
        Ok(Box::pin(stream.map(|vn| Notification { uuid: vn.uuid, value: vn.value })))
    }

    async fn disconnect(&self) -> Result<(), BleError> {
        if let Some(p) = &self.peripheral {
            if p.is_connected().await.unwrap_or(false) {
                p.disconnect().await.map_err(backend)?;
            }
        }
        Ok(())
    }
}

/// WinRT LE pairing: the ConfirmOnly ("just works") ceremony WHOOP presents, auto-accepted.
#[cfg(windows)]
mod win_pair {
    use ble_core::BleError;
    use windows::core::Ref;
    use windows::Devices::Bluetooth::BluetoothLEDevice;
    use windows::Devices::Enumeration::{
        DeviceInformationCustomPairing, DevicePairingKinds, DevicePairingRequestedEventArgs, DevicePairingResultStatus,
    };
    use windows::Foundation::TypedEventHandler;

    fn err<E: std::fmt::Display>(e: E) -> BleError {
        BleError::Pairing(e.to_string())
    }

    fn mac_to_u64(mac: &str) -> Option<u64> {
        mac.split(':').try_fold(0u64, |acc, part| Some((acc << 8) | u8::from_str_radix(part.trim(), 16).ok()? as u64))
    }

    pub async fn ensure_bonded(mac: &str) -> Result<(), BleError> {
        let addr = mac_to_u64(mac).ok_or_else(|| BleError::Pairing(format!("bad address {mac}")))?;
        let device = BluetoothLEDevice::FromBluetoothAddressAsync(addr).map_err(err)?.await.map_err(err)?;
        let pairing = device.DeviceInformation().map_err(err)?.Pairing().map_err(err)?;
        if pairing.IsPaired().unwrap_or(false) {
            return Ok(());
        }
        let custom: DeviceInformationCustomPairing = pairing.Custom().map_err(err)?;
        let handler = TypedEventHandler::new(
            |_sender: Ref<DeviceInformationCustomPairing>, args: Ref<DevicePairingRequestedEventArgs>| {
                if let Some(args) = args.as_ref() {
                    args.Accept()?;
                }
                Ok(())
            },
        );
        let token = custom.PairingRequested(&handler).map_err(err)?;
        let result = custom.PairAsync(DevicePairingKinds::ConfirmOnly).map_err(err)?.await.map_err(err)?;
        let _ = custom.RemovePairingRequested(token);
        let status = result.Status().map_err(err)?;
        eprintln!("[pair] {mac} status={status:?}");
        match status {
            DevicePairingResultStatus::Paired | DevicePairingResultStatus::AlreadyPaired => Ok(()),
            other => Err(BleError::Pairing(format!("pairing failed: {other:?}"))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn props(name: Option<&str>, service: Option<Uuid>) -> PeripheralProperties {
        PeripheralProperties {
            local_name: name.map(str::to_string),
            services: service.into_iter().collect(),
            ..Default::default()
        }
    }

    #[test]
    fn matches_whoop_by_service_or_name_prefix() {
        let svc = Uuid::from_u128(0xfd4b0001);
        assert!(matches_whoop(&props(None, Some(svc)), svc)); // advertises the service
        assert!(matches_whoop(&props(Some("WHOOP 5AG…"), None), svc)); // "WHOOP…" name
        assert!(!matches_whoop(&props(Some("Other"), None), svc)); // neither
        assert!(!matches_whoop(&props(None, Some(Uuid::from_u128(2))), svc)); // wrong service
    }
}
