//! A scriptable transport for hardware-free tests: it replays a fixed notification sequence and records
//! every write, so the whole bond/offload driver can be exercised with no radio.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures::stream::BoxStream;
use uuid::Uuid;

use crate::transport::BleTransport;
use crate::types::{BleError, Notification};

/// Shared log of `(characteristic, bytes)` written to the transport.
type WriteLog = Arc<Mutex<Vec<(Uuid, Vec<u8>)>>>;

#[derive(Clone, Default)]
pub struct MockTransport {
    scripted: Vec<Notification>,
    writes: WriteLog,
}

impl MockTransport {
    pub fn new(scripted: Vec<Notification>) -> Self {
        MockTransport { scripted, writes: Arc::new(Mutex::new(Vec::new())) }
    }

    /// Everything written to the strap so far — assert against this in tests.
    pub fn writes(&self) -> Vec<(Uuid, Vec<u8>)> {
        self.writes.lock().unwrap().clone()
    }
}

#[async_trait]
impl BleTransport for MockTransport {
    async fn connect(&mut self) -> Result<(), BleError> {
        Ok(())
    }

    async fn subscribe(&self, _characteristic: Uuid) -> Result<(), BleError> {
        Ok(())
    }

    async fn write(&self, characteristic: Uuid, data: &[u8], _with_response: bool) -> Result<(), BleError> {
        self.writes.lock().unwrap().push((characteristic, data.to_vec()));
        Ok(())
    }

    async fn read(&self, _characteristic: Uuid) -> Result<Vec<u8>, BleError> {
        Ok(Vec::new())
    }

    async fn notifications(&self) -> Result<BoxStream<'static, Notification>, BleError> {
        Ok(Box::pin(futures::stream::iter(self.scripted.clone())))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;

    #[tokio::test]
    async fn records_writes_and_replays_scripted_notifications() {
        let uuid = Uuid::from_u128(1);
        let note = Notification { uuid, value: vec![1, 2, 3] };
        let mock = MockTransport::new(vec![note.clone()]);
        let handle = mock.clone(); // shares the write log

        mock.write(uuid, &[9, 9], false).await.unwrap();

        let replayed: Vec<_> = mock.notifications().await.unwrap().collect().await;
        assert_eq!(replayed, vec![note]);
        assert_eq!(handle.writes(), vec![(uuid, vec![9, 9])]);
    }
}
