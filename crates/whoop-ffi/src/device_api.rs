//! Which strap a hardware revision names. The revision-prefix table stays in the codec, so a caller
//! reads the classification and chooses its own wording instead of carrying a copy of the prefixes.

use crate::*;

/// The strap as a capability, not as a wire format.
#[derive(uniffi::Enum, Clone, Copy)]
pub enum StrapVariant {
    Whoop4,
    Whoop5,
    WhoopMg,
    /// Identity not read, or a revision prefix nothing has seen. Never resolved to the nearer guess.
    Unknown,
}

impl From<Variant> for StrapVariant {
    fn from(v: Variant) -> Self {
        match v {
            Variant::Whoop4 => StrapVariant::Whoop4,
            Variant::Whoop5 => StrapVariant::Whoop5,
            Variant::WhoopMg => StrapVariant::WhoopMg,
            Variant::Unknown => StrapVariant::Unknown,
        }
    }
}

/// Classify a GATT Hardware Revision string, with the wire generation as the fallback where the
/// revision is absent or unrecognised. A Gen5 strap with no usable revision stays `Unknown`.
#[uniffi::export]
pub fn strap_variant(hardware_rev: Option<String>, family: Gen) -> StrapVariant {
    Variant::classify(hardware_rev.as_deref().unwrap_or(""), family.into()).into()
}
