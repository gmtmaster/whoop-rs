#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error(transparent)]
    Ble(#[from] ble_core::BleError),
    #[error(transparent)]
    Protocol(#[from] whoop_protocol::ProtocolError),
    #[error("opcode {0:#04x} is forbidden on the explore path (destructive / config-write)")]
    Forbidden(u8),
    #[error("record sink: {0}")]
    Sink(#[from] std::io::Error),
    #[error(transparent)]
    Flash(#[from] crate::flash::FlashFault),
}
