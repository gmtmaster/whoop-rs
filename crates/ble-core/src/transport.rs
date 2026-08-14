use async_trait::async_trait;
use futures::stream::BoxStream;
use uuid::Uuid;

use crate::types::{BleError, Notification};

/// A connected BLE central peripheral. Backends: `ble-btleplug` (real) and `MockTransport` (tests).
/// `notifications` returns an owned `'static` stream so a caller can hold it while still issuing writes
/// on `&self` (the borrow that fights the checker if the stream borrowed `self`).
#[async_trait]
pub trait BleTransport: Send + Sync {
    /// Scan (if needed), connect, negotiate MTU, discover services.
    async fn connect(&mut self) -> Result<(), BleError>;

    /// Establish the OS-level encrypted bond. Default no-op (macOS bonds transparently on first
    /// encrypted access); Linux/Windows backends override this.
    async fn ensure_paired(&self) -> Result<(), BleError> {
        Ok(())
    }

    /// Drop the link. Default no-op; the btleplug backend disconnects the peripheral so the (exclusive-
    /// bond) band isn't left held, which would wedge the next session.
    async fn disconnect(&self) -> Result<(), BleError> {
        Ok(())
    }

    /// Negotiated ATT MTU, when the backend can report one. A confirmed write of `n` bytes needs an MTU
    /// of at least `n + 3`, so a caller with an unusually wide frame can refuse before sending it.
    fn mtu(&self) -> Option<usize> {
        None
    }

    async fn subscribe(&self, characteristic: Uuid) -> Result<(), BleError>;

    async fn write(&self, characteristic: Uuid, data: &[u8], with_response: bool) -> Result<(), BleError>;

    async fn read(&self, characteristic: Uuid) -> Result<Vec<u8>, BleError>;

    /// A merged, owned (`'static`) stream of notifications across every subscribed characteristic — a
    /// caller can hold it while still issuing writes on `&self`.
    async fn notifications(&self) -> Result<BoxStream<'static, Notification>, BleError>;
}
