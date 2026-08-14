//! The plaintext header ahead of a firmware payload, and the full self-consistency check the transfer
//! is gated on. Pure byte reads; `firmware` builds the frames that carry the bytes this describes.

use crate::crc::crc32_zlib;

/// Plaintext header ahead of the compressed payload.
pub const IMAGE_HEADER_LEN: usize = 512;

/// Product marker at [`PRODUCT_OFFSET`]: the 5.0/MG line. The 4.0 line carries 6.
pub const PRODUCT_MAVERICK: u32 = 13;

/// Container marker at [`CONTAINER_OFFSET`]: set on the wrapped image that goes on the wire. A
/// decompressed image carries 1, so this one field separates the two.
pub const CONTAINER_ZBIN: u32 = 5;

/// Header field offsets, file-relative.
const PAYLOAD_CRC_OFFSET: usize = 0x00;
pub(crate) const LEN_OFFSET: usize = 0x04;
const CONTAINER_OFFSET: usize = 0x0C;
const PRODUCT_OFFSET: usize = 0x10;
const BUILT_OFFSET: usize = 0x18;
const VERSION_OFFSET: usize = 0x4C;
const HEADER_CRC_OFFSET: usize = 0x1F8;
const CRC_COPY_OFFSET: usize = 0x1FC;

/// Bytes the header CRC covers, and the width the version string is read from.
const HEADER_CRC_RANGE: std::ops::Range<usize> = 0x08..0x1F8;
const VERSION_MAX: usize = 32;

/// The header as the strap reads it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImageHeader {
    pub payload_crc: u32,
    pub payload_len: u32,
    pub container: u32,
    pub product: u32,
    pub built_unix: u32,
    pub version: String,
    pub header_crc: u32,
}

/// Why an image is not sendable. Reported in check order, so the first mismatch is the one named.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImageFault {
    TooShort,
    LengthMismatch,
    CrcCopyMismatch,
    HeaderCrc,
    PayloadCrc,
    NotAContainer(u32),
    WrongProduct(u32),
}

/// Transfer length: the header plus the payload length the header declares. `None` when `image` is
/// shorter than the header or the declared length overruns it.
pub fn transfer_len(image: &[u8]) -> Option<usize> {
    if image.len() < IMAGE_HEADER_LEN {
        return None;
    }
    let declared = u32::from_le_bytes(image[LEN_OFFSET..LEN_OFFSET + 4].try_into().ok()?) as usize;
    let total = IMAGE_HEADER_LEN.checked_add(declared)?;
    (total <= image.len()).then_some(total)
}

/// Parse and fully check an image: length self-consistency, both CRC-32s, container and product
/// markers. `Ok` means every field the strap reads agrees with itself.
pub fn inspect(image: &[u8]) -> Result<ImageHeader, ImageFault> {
    if image.len() < IMAGE_HEADER_LEN {
        return Err(ImageFault::TooShort);
    }
    let at = |i: usize| u32::from_le_bytes(image[i..i + 4].try_into().unwrap());

    let payload_crc = at(PAYLOAD_CRC_OFFSET);
    if transfer_len(image) != Some(image.len()) {
        return Err(ImageFault::LengthMismatch);
    }
    if at(CRC_COPY_OFFSET) != payload_crc {
        return Err(ImageFault::CrcCopyMismatch);
    }
    let header_crc = at(HEADER_CRC_OFFSET);
    if crc32_zlib(&image[HEADER_CRC_RANGE]) != header_crc {
        return Err(ImageFault::HeaderCrc);
    }
    if crc32_zlib(&image[IMAGE_HEADER_LEN..]) != payload_crc {
        return Err(ImageFault::PayloadCrc);
    }
    let container = at(CONTAINER_OFFSET);
    if container != CONTAINER_ZBIN {
        return Err(ImageFault::NotAContainer(container));
    }
    let product = at(PRODUCT_OFFSET);
    if product != PRODUCT_MAVERICK {
        return Err(ImageFault::WrongProduct(product));
    }
    Ok(ImageHeader {
        payload_crc,
        payload_len: at(LEN_OFFSET),
        container,
        product,
        built_unix: at(BUILT_OFFSET),
        version: version_string(image),
        header_crc,
    })
}

/// The NUL-terminated version field. Masked in every shipped image, so it names a line and never a
/// build — nothing may gate on it.
fn version_string(image: &[u8]) -> String {
    let raw = &image[VERSION_OFFSET..(VERSION_OFFSET + VERSION_MAX).min(IMAGE_HEADER_LEN)];
    let cut = raw.iter().position(|&b| b == 0).unwrap_or(raw.len());
    String::from_utf8_lossy(&raw[..cut]).trim().to_string()
}

#[cfg(test)]
pub(crate) mod fixture {
    use super::*;

    /// A header-shaped buffer declaring `payload` bytes, then that payload. Nothing else is set, so
    /// only [`transfer_len`] holds on it.
    pub(crate) fn sized(payload: usize) -> Vec<u8> {
        let mut v = vec![0u8; IMAGE_HEADER_LEN + payload];
        v[LEN_OFFSET..LEN_OFFSET + 4].copy_from_slice(&(payload as u32).to_le_bytes());
        v
    }

    /// Recompute the header CRC-32 and both copies of the payload CRC-32 in place.
    pub(crate) fn seal(v: &mut [u8]) {
        let payload_crc = crc32_zlib(&v[IMAGE_HEADER_LEN..]);
        v[PAYLOAD_CRC_OFFSET..PAYLOAD_CRC_OFFSET + 4].copy_from_slice(&payload_crc.to_le_bytes());
        v[CRC_COPY_OFFSET..CRC_COPY_OFFSET + 4].copy_from_slice(&payload_crc.to_le_bytes());
        let header_crc = crc32_zlib(&v[HEADER_CRC_RANGE]);
        v[HEADER_CRC_OFFSET..HEADER_CRC_OFFSET + 4].copy_from_slice(&header_crc.to_le_bytes());
    }

    /// A fully self-consistent image: a non-constant body, both markers, a version string, both CRCs.
    pub(crate) fn valid(payload: usize) -> Vec<u8> {
        let mut v = sized(payload);
        for (i, b) in v[IMAGE_HEADER_LEN..].iter_mut().enumerate() {
            *b = (i % 251) as u8;
        }
        v[CONTAINER_OFFSET..CONTAINER_OFFSET + 4].copy_from_slice(&CONTAINER_ZBIN.to_le_bytes());
        v[PRODUCT_OFFSET..PRODUCT_OFFSET + 4].copy_from_slice(&PRODUCT_MAVERICK.to_le_bytes());
        v[VERSION_OFFSET..VERSION_OFFSET + 7].copy_from_slice(b"testver");
        seal(&mut v);
        v
    }

    /// Overwrite one u32 header field and re-seal, so only that field disagrees.
    pub(crate) fn with_field(payload: usize, offset: usize, value: u32) -> Vec<u8> {
        let mut v = valid(payload);
        v[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
        seal(&mut v);
        v
    }
}

#[cfg(test)]
mod tests {
    use super::fixture::{sized, valid, with_field};
    use super::*;

    #[test]
    fn transfer_len_is_the_header_plus_the_declared_payload() {
        assert_eq!(transfer_len(&sized(1000)), Some(IMAGE_HEADER_LEN + 1000));
    }

    #[test]
    fn transfer_len_rejects_a_short_buffer_and_an_overrunning_length() {
        assert_eq!(transfer_len(&[0u8; 16]), None);
        let mut v = sized(1000);
        v[LEN_OFFSET..LEN_OFFSET + 4].copy_from_slice(&u32::MAX.to_le_bytes());
        assert_eq!(transfer_len(&v), None);
    }

    #[test]
    fn inspect_accepts_a_self_consistent_image() {
        let img = valid(1000);
        let h = inspect(&img).unwrap();
        assert_eq!(h.payload_len, 1000);
        assert_eq!(h.container, CONTAINER_ZBIN);
        assert_eq!(h.product, PRODUCT_MAVERICK);
        assert_eq!(h.version, "testver");
        assert_eq!(h.payload_crc, crc32_zlib(&img[IMAGE_HEADER_LEN..]));
    }

    /// Each fault in isolation: one field wrong at a time, everything else re-sealed.
    #[test]
    fn inspect_names_the_first_field_that_disagrees() {
        assert_eq!(inspect(&[0u8; 16]), Err(ImageFault::TooShort));

        let mut short = valid(1000);
        short.pop();
        assert_eq!(inspect(&short), Err(ImageFault::LengthMismatch));

        let mut copy = valid(1000);
        copy[CRC_COPY_OFFSET] ^= 0xFF;
        assert_eq!(inspect(&copy), Err(ImageFault::CrcCopyMismatch));

        let mut header = valid(1000);
        header[BUILT_OFFSET] ^= 0xFF; // inside the header-CRC range, so that CRC no longer holds
        assert_eq!(inspect(&header), Err(ImageFault::HeaderCrc));

        let mut body = valid(1000);
        body[IMAGE_HEADER_LEN] ^= 0xFF;
        assert_eq!(inspect(&body), Err(ImageFault::PayloadCrc));

        assert_eq!(inspect(&with_field(1000, CONTAINER_OFFSET, 1)), Err(ImageFault::NotAContainer(1)));
        assert_eq!(inspect(&with_field(1000, PRODUCT_OFFSET, 6)), Err(ImageFault::WrongProduct(6)));
    }

    /// Every archived image in `WHOOP_ZBIN_DIR` must satisfy its own header. Run after extracting them.
    #[test]
    #[ignore = "needs the archived images, which live outside the repo"]
    fn the_archived_images_all_satisfy_their_own_header() {
        let dir = std::env::var("WHOOP_ZBIN_DIR").expect("set WHOOP_ZBIN_DIR");
        let mut seen = 0;
        for entry in std::fs::read_dir(&dir).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().is_none_or(|e| e != "zbin") {
                continue;
            }
            let img = std::fs::read(&path).unwrap();
            let fault = inspect(&img).err();
            // The 4.0 line differs only in its product marker; everything before that must hold.
            assert!(
                fault.is_none() || matches!(fault, Some(ImageFault::WrongProduct(6))),
                "{path:?}: {fault:?}"
            );
            seen += 1;
        }
        assert!(seen > 0, "no images found in {dir}");
    }
}
