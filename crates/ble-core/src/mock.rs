//! A scriptable transport for hardware-free tests: it replays a fixed notification sequence, records
//! every write, and can answer a write through a responder, so a request/response driver is exercisable
//! with no radio. With no responder attached it behaves exactly as the replay-only form.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures::channel::mpsc::UnboundedSender;
use futures::stream::BoxStream;
use futures::StreamExt;
use uuid::Uuid;

use crate::transport::BleTransport;
use crate::types::{BleError, Notification};

/// Turns one written frame into the notifications it provokes — the failure-injection seam: a wrong
/// opcode, a wrong echo, a refusal, or nothing at all.
pub type Responder = dyn Fn(&[u8]) -> Vec<Notification> + Send + Sync;

/// Shared log of `(characteristic, bytes)` written to the transport.
type WriteLog = Arc<Mutex<Vec<(Uuid, Vec<u8>)>>>;

#[derive(Clone, Default)]
pub struct MockTransport {
    scripted: Vec<Notification>,
    reads: HashMap<Uuid, Vec<u8>>,
    writes: WriteLog,
    responder: Option<Arc<Responder>>,
    listeners: Arc<Mutex<Vec<UnboundedSender<Notification>>>>,
    mtu: Option<usize>,
}

impl MockTransport {
    pub fn new(scripted: Vec<Notification>) -> Self {
        MockTransport { scripted, ..Default::default() }
    }

    /// A transport that answers writes instead of replaying a script.
    pub fn with_responder(responder: Arc<Responder>) -> Self {
        MockTransport { responder: Some(responder), ..Default::default() }
    }

    /// Serve `value` for reads of `characteristic`. Unset characteristics still read empty.
    pub fn with_read(mut self, characteristic: Uuid, value: Vec<u8>) -> Self {
        self.reads.insert(characteristic, value);
        self
    }

    /// Report a negotiated MTU. Unset reads as unknown, which is what the replay-only form has always
    /// been.
    pub fn with_mtu(mut self, mtu: usize) -> Self {
        self.mtu = Some(mtu);
        self
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

    fn mtu(&self) -> Option<usize> {
        self.mtu
    }

    async fn subscribe(&self, _characteristic: Uuid) -> Result<(), BleError> {
        Ok(())
    }

    async fn write(&self, characteristic: Uuid, data: &[u8], _with_response: bool) -> Result<(), BleError> {
        self.writes.lock().unwrap().push((characteristic, data.to_vec()));
        if let Some(responder) = &self.responder {
            let replies = responder(data);
            let mut listeners = self.listeners.lock().unwrap();
            listeners.retain(|tx| !tx.is_closed());
            for note in replies {
                for tx in listeners.iter() {
                    let _ = tx.unbounded_send(note.clone());
                }
            }
        }
        Ok(())
    }

    async fn read(&self, characteristic: Uuid) -> Result<Vec<u8>, BleError> {
        Ok(self.reads.get(&characteristic).cloned().unwrap_or_default())
    }

    async fn notifications(&self) -> Result<BoxStream<'static, Notification>, BleError> {
        let scripted = futures::stream::iter(self.scripted.clone());
        if self.responder.is_none() {
            return Ok(Box::pin(scripted));
        }
        let (tx, rx) = futures::channel::mpsc::unbounded();
        self.listeners.lock().unwrap().push(tx);
        Ok(Box::pin(scripted.chain(rx)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn note(n: u8) -> Notification {
        Notification { uuid: Uuid::from_u128(1), value: vec![n] }
    }

    #[tokio::test]
    async fn records_writes_and_replays_scripted_notifications() {
        let uuid = Uuid::from_u128(1);
        let mock = MockTransport::new(vec![note(3)]);
        let handle = mock.clone(); // shares the write log

        mock.write(uuid, &[9, 9], false).await.unwrap();

        let replayed: Vec<_> = mock.notifications().await.unwrap().collect().await;
        assert_eq!(replayed, vec![note(3)]);
        assert_eq!(handle.writes(), vec![(uuid, vec![9, 9])]);
    }

    /// The replay-only form must still terminate — every existing driver test relies on that.
    #[tokio::test]
    async fn a_script_without_a_responder_ends() {
        let mock = MockTransport::new(vec![note(1), note(2)]);
        let seen: Vec<_> = mock.notifications().await.unwrap().collect().await;
        assert_eq!(seen, vec![note(1), note(2)]);
    }

    /// A responder answers each write, on every stream open at the time.
    #[tokio::test]
    async fn a_responder_answers_every_open_stream() {
        let mock = MockTransport::with_responder(Arc::new(|w: &[u8]| vec![note(w[0] + 1)]));
        let mut first = mock.notifications().await.unwrap();
        let mut second = mock.notifications().await.unwrap();

        mock.write(Uuid::from_u128(1), &[7], false).await.unwrap();
        assert_eq!(first.next().await, Some(note(8)));
        assert_eq!(second.next().await, Some(note(8)));
    }

    #[tokio::test]
    async fn reads_serve_what_was_scripted_and_empty_otherwise() {
        let served = Uuid::from_u128(2);
        let mock = MockTransport::new(Vec::new()).with_read(served, b"ABC123".to_vec());
        assert_eq!(mock.read(served).await.unwrap(), b"ABC123");
        assert!(mock.read(Uuid::from_u128(3)).await.unwrap().is_empty());
    }
}
