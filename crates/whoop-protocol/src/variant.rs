//! Which strap this actually is, and therefore what it can do.
//!
//! `Family` (Gen4/Gen5) is the WIRE shape and deliberately lumps the 5.0 with the MG, because they
//! frame identically. That is right for framing and wrong for capability: the MG has an ECG
//! electrode and the 5.0 does not, and the two are indistinguishable by family, by model string
//! ("WHOOP 5.0 / MG" is what onboarding assigns every 5-series strap) and by serial prefix (`5A00`,
//! `5AG` and `5AM` are all in David's bond table and the first two are 5.0s).
//!
//! The **hardware revision** does distinguish them, and we already read it — `Scanned::hardware`
//! comes from the standard GATT Hardware Revision char on every scan, and has never been used.
//!
//! Evidence — four straps, two observers, and both variants measured on the SAME firmware version
//! so the difference cannot be a firmware artefact:
//!
//! | strap | model | hardware | source |
//! |---|---|---|---|
//! | `5AM0020360` — David's MG | `MG` | `WS50_r00` | `whoopctl scan`, 2026-08-01, fw 50.40.1.0 |
//! | `5AG0268206` — a 5.0 | `5.0` | `WG50_r52` | same scan, same firmware |
//! | `5A00960910` — David's, on-wrist | `5.0` | `WG50_r45` | `docs/DEVICE-DEBUG-SECURITY-2026-06.md` |
//! | someone else's MG | `MG` | `WS50_r03` | `whoop-research/judasclub/blogpost.md` |
//!
//! The `_rNN` suffix is a board revision and spans r00-r52 across both variants, so **only the
//! prefix carries the variant**. The shipping maverick image contains both `WG50_r` and `WS50_r`,
//! one firmware serving both boards. Firmware refuses ECG on the 4.0 outright ("ECG Control not
//! supported on Goose hardware"), so that arm is not a guess either.
//!
//! The GATT **model** string discriminates too (`MG` vs `5.0`), independently of the revision. That
//! is a cross-check, not a second classifier — see [`Variant::model_agrees`].
//!
//! This lives in Rust because a capability is a classification, and the prefix table is a constant
//! that must have exactly one home. A Kotlin copy of `"WS50"` would be a second source of truth for
//! which strap owns a feature, which is the drift this project keeps finding.

use crate::family::Family;

/// Hardware-revision prefix for the 5.0 board.
const HW_PREFIX_WHOOP5: &str = "WG50";
/// Hardware-revision prefix for the MG board — the only strap with an ECG electrode.
const HW_PREFIX_MG: &str = "WS50";

/// The strap as a capability, not as a wire format.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Variant {
    /// WHOOP 4.0. Firmware refuses ECG on this board.
    Whoop4,
    /// WHOOP 5.0 — Gen5 wire, no ECG electrode.
    Whoop5,
    /// WHOOP MG — Gen5 wire, ECG electrode present.
    WhoopMg,
    /// Identity not read, or a revision prefix we have never seen.
    Unknown,
}

impl Variant {
    /// Classify from the GATT Hardware Revision string, with `Family` as the fallback when the
    /// revision is absent or unrecognised.
    ///
    /// A revision we do not recognise resolves to `Unknown` rather than to the nearest guess. A new
    /// board would otherwise inherit a capability set nobody has checked, and on this axis that
    /// means offering ECG on hardware that may not have the electrode.
    pub fn classify(hardware_rev: &str, family: Family) -> Self {
        let hw = hardware_rev.trim();
        if hw.starts_with(HW_PREFIX_MG) {
            return Variant::WhoopMg;
        }
        if hw.starts_with(HW_PREFIX_WHOOP5) {
            return Variant::Whoop5;
        }
        // No usable revision. Gen4 is still fully determined — there is only one 4.0 board and its
        // firmware refuses ECG regardless — but Gen5 genuinely splits two ways and must not be guessed.
        match family {
            Family::Gen4 => Variant::Whoop4,
            Family::Gen5 => Variant::Unknown,
        }
    }

    /// Whether this strap has an ECG electrode.
    ///
    /// **Fails closed.** `Unknown` is not capable: hiding the entry on a strap that turns out to
    /// support it is a one-line fix once its revision prefix is known, whereas offering it on a
    /// strap without an electrode produces a capture that can only ever be noise — and noise
    /// rendered as a waveform is the failure this whole feature is built to avoid.
    pub fn has_ecg(self) -> bool {
        matches!(self, Variant::WhoopMg)
    }

    /// The wire generation this variant frames as.
    pub fn family(self) -> Option<Family> {
        match self {
            Variant::Whoop4 => Some(Family::Gen4),
            Variant::Whoop5 | Variant::WhoopMg => Some(Family::Gen5),
            Variant::Unknown => None,
        }
    }

    /// Whether the GATT model string agrees with the variant the hardware revision gave.
    ///
    /// The two fields are read from the same device but are independent: `WS50_r00` came with model
    /// `MG`, `WG50_r52` with `5.0`. Agreement is not needed for classification — the revision is
    /// authoritative — but **a disagreement means an assumption is broken**, and a caller that can
    /// see one should say so rather than proceed. `None` when the model string is empty or is the
    /// generic 5-series label, which carries no information either way.
    pub fn model_agrees(self, model: &str) -> Option<bool> {
        let m = model.trim();
        if m.is_empty() || m.eq_ignore_ascii_case("WHOOP 5.0 / MG") {
            return None;
        }
        let says_mg = m.eq_ignore_ascii_case("MG");
        let says_5 = m.eq_ignore_ascii_case("5.0");
        match self {
            Variant::WhoopMg if says_mg => Some(true),
            Variant::Whoop5 if says_5 => Some(true),
            Variant::WhoopMg | Variant::Whoop5 if says_mg || says_5 => Some(false),
            _ => None,
        }
    }

    /// A label for display. `Unknown` deliberately reads as the generic 5-series name rather than
    /// claiming a board, so presentation degrades to what we already showed before this existed.
    pub fn label(self) -> &'static str {
        match self {
            Variant::Whoop4 => "WHOOP 4.0",
            Variant::Whoop5 => "WHOOP 5.0",
            Variant::WhoopMg => "WHOOP MG",
            Variant::Unknown => "WHOOP 5.0 / MG",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every strap we have actually read, with the model string the same device reported.
    /// The first two came off one `whoopctl scan` on 2026-08-01, both on firmware 50.40.1.0.
    const OBSERVED: [(&str, &str, Variant); 4] = [
        ("WS50_r00", "MG", Variant::WhoopMg),  // 5AM0020360, David's MG
        ("WG50_r52", "5.0", Variant::Whoop5),  // 5AG0268206
        ("WG50_r45", "5.0", Variant::Whoop5),  // 5A00960910, read on-wrist
        ("WS50_r03", "MG", Variant::WhoopMg),  // judasclub's MG, fw 50.38.1.0
    ];

    #[test]
    fn every_strap_we_have_read_classifies_as_recorded() {
        for (hw, _, want) in OBSERVED {
            assert_eq!(Variant::classify(hw, Family::Gen5), want, "{hw}");
        }
    }

    #[test]
    fn the_model_string_agrees_on_every_strap_we_have_read() {
        // Two independent GATT fields, same device. If this ever fails, an assumption broke.
        for (hw, model, _) in OBSERVED {
            assert_eq!(Variant::classify(hw, Family::Gen5).model_agrees(model), Some(true), "{hw}/{model}");
        }
        // A genuine contradiction is reported, not swallowed.
        assert_eq!(Variant::classify("WS50_r00", Family::Gen5).model_agrees("5.0"), Some(false));
        // The generic label onboarding assigns carries no information.
        assert_eq!(Variant::classify("WS50_r00", Family::Gen5).model_agrees("WHOOP 5.0 / MG"), None);
        assert_eq!(Variant::classify("WS50_r00", Family::Gen5).model_agrees(""), None);
    }

    #[test]
    fn only_the_mg_offers_ecg() {
        assert!(Variant::classify("WS50_r03", Family::Gen5).has_ecg());
        assert!(!Variant::classify("WG50_r45", Family::Gen5).has_ecg());
        assert!(!Variant::classify("", Family::Gen4).has_ecg());
        // The case that matters most: a 5-series strap whose identity was never read.
        assert!(!Variant::classify("", Family::Gen5).has_ecg());
    }

    #[test]
    fn a_revision_we_have_never_seen_is_unknown_not_a_guess() {
        assert_eq!(Variant::classify("WX99_r01", Family::Gen5), Variant::Unknown);
        assert!(!Variant::classify("WX99_r01", Family::Gen5).has_ecg());
        // A 4.0 is still determined without a revision: one board, and firmware refuses ECG on it.
        assert_eq!(Variant::classify("", Family::Gen4), Variant::Whoop4);
    }

    #[test]
    fn revision_suffix_and_whitespace_do_not_matter() {
        // r03 and r45 are board revisions within a variant; only the prefix carries the variant.
        for rev in ["WS50_r01", "WS50_r03", "WS50_r99", " WS50_r03 "] {
            assert_eq!(Variant::classify(rev, Family::Gen5), Variant::WhoopMg, "{rev}");
        }
    }

    #[test]
    fn family_round_trips_except_where_it_genuinely_cannot() {
        assert_eq!(Variant::Whoop5.family(), Some(Family::Gen5));
        assert_eq!(Variant::WhoopMg.family(), Some(Family::Gen5));
        assert_eq!(Variant::Whoop4.family(), Some(Family::Gen4));
        assert_eq!(Variant::Unknown.family(), None);
    }
}
