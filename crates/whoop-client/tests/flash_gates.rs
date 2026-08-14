//! Every safety gate on the firmware transfer, and the blind-path guards it must not weaken. Each gate
//! test asserts the refusal AND that no firmware opcode reached the strap.

mod common;

use common::{identity, image_with_broken_field, three_chunks, transport, Strap, CONTAINER_OFFSET, SERIAL};

use whoop_client::{Error, FlashArm, FlashFault, FlashOptions, FlashStep, WhoopClient};
use whoop_protocol::command;
use whoop_protocol::Family;

/// Options armed to commit against the rig's strap — the maximum a test can ask for.
fn armed() -> FlashOptions {
    FlashOptions { expect_serial: SERIAL.into(), arm: FlashArm::Commit, ..Default::default() }
}

const FIRMWARE_OPCODES: [u8; 4] = [
    command::START_FIRMWARE_LOAD_NEW,
    command::LOAD_FIRMWARE_DATA_NEW,
    command::PROCESS_FIRMWARE_IMAGE_NEW,
    command::VERIFY_FIRMWARE_IMAGE,
];

fn no_firmware_reached(strap: &Strap) {
    let sent: Vec<u8> = strap.flow().into_iter().filter(|c| FIRMWARE_OPCODES.contains(c)).collect();
    assert!(sent.is_empty(), "a gate let firmware opcodes through: {sent:?}");
}

/// The default arm plans and writes nothing, so a caller that forgets to arm cannot flash.
#[tokio::test(start_paused = true)]
async fn the_default_arm_plans_and_sends_no_firmware_command() {
    let strap = Strap::healthy();
    let mut client = WhoopClient::new(transport(&strap), Family::Gen5);
    let opts = FlashOptions { expect_serial: SERIAL.into(), ..Default::default() };
    assert_eq!(opts.arm, FlashArm::Plan);

    let report = client.flash_firmware(&three_chunks(), &opts, |_| {}).await.unwrap();
    assert_eq!(report.reached, FlashStep::Planned);
    assert_eq!(report.chunks_planned, 3);
    assert_eq!(report.chunks_sent, 0);
    no_firmware_reached(&strap);
}

/// An expected serial that names no single band is refused, empty or merely too short to be more than
/// a wildcard against a suffix match.
#[tokio::test(start_paused = true)]
async fn an_expected_serial_that_names_no_band_is_refused() {
    for expect in ["", "1", "0001"] {
        let strap = Strap::healthy();
        let mut client = WhoopClient::new(transport(&strap), Family::Gen5);
        let opts = FlashOptions { expect_serial: expect.into(), arm: FlashArm::Commit, ..Default::default() };

        let err = client.flash_firmware(&three_chunks(), &opts, |_| {}).await.unwrap_err();
        assert!(matches!(err, Error::Flash(FlashFault::NoExpectedSerial(_))), "{expect:?}: {err}");
        no_firmware_reached(&strap);
    }
}

/// A link too narrow for a full-width chunk is refused before anything is written, and an unknown MTU
/// is not treated as a refusal.
#[tokio::test(start_paused = true)]
async fn a_link_that_cannot_carry_a_chunk_is_refused() {
    let strap = Strap::healthy();
    let mut client = WhoopClient::new(common::transport_with_mtu(&strap, 185), Family::Gen5);
    let err = client.flash_firmware(&three_chunks(), &armed(), |_| {}).await.unwrap_err();

    match err {
        Error::Flash(FlashFault::LinkTooNarrow { have, need }) => {
            assert_eq!(have, 185);
            assert_eq!(need, 247);
        }
        other => panic!("expected LinkTooNarrow, got {other}"),
    }
    assert!(strap.log().is_empty(), "the link is checked before the strap is touched");

    let strap = Strap::healthy();
    let mut client = WhoopClient::new(common::transport_with_mtu(&strap, 247), Family::Gen5);
    let report = client.flash_firmware(&three_chunks(), &armed(), |_| {}).await.unwrap();
    assert_eq!(report.mtu, Some(247));
    assert_eq!(report.reached, FlashStep::Committed);
}

#[tokio::test(start_paused = true)]
async fn a_strap_that_is_not_the_named_band_is_refused() {
    let strap = Strap::healthy();
    let transport = identity(&strap, Some("5A00000002"), Some(common::FIRMWARE));
    let mut client = WhoopClient::new(transport, Family::Gen5);

    let err = client.flash_firmware(&three_chunks(), &armed(), |_| {}).await.unwrap_err();
    assert!(matches!(err, Error::Flash(FlashFault::WrongStrap { .. })), "{err}");
    no_firmware_reached(&strap);
}

#[tokio::test(start_paused = true)]
async fn an_unreadable_serial_is_refused() {
    let strap = Strap::healthy();
    let mut client = WhoopClient::new(identity(&strap, None, Some(common::FIRMWARE)), Family::Gen5);

    let err = client.flash_firmware(&three_chunks(), &armed(), |_| {}).await.unwrap_err();
    assert!(matches!(err, Error::Flash(FlashFault::SerialUnreadable)), "{err}");
    no_firmware_reached(&strap);
}

/// The suffix match is case-insensitive, matching the wipe guard.
#[tokio::test(start_paused = true)]
async fn the_serial_match_is_a_case_insensitive_suffix() {
    let strap = Strap::healthy();
    let mut client = WhoopClient::new(transport(&strap), Family::Gen5);
    let opts = FlashOptions { expect_serial: SERIAL.to_lowercase(), ..Default::default() };

    let report = client.flash_firmware(&three_chunks(), &opts, |_| {}).await.unwrap();
    assert_eq!(report.serial, SERIAL);
}

#[tokio::test(start_paused = true)]
async fn a_strap_on_another_firmware_line_is_refused() {
    let strap = Strap::healthy();
    let mut client = WhoopClient::new(identity(&strap, Some(SERIAL), Some("41.17.4.0")), Family::Gen5);

    let err = client.flash_firmware(&three_chunks(), &armed(), |_| {}).await.unwrap_err();
    assert!(matches!(err, Error::Flash(FlashFault::WrongStrapFirmware(_))), "{err}");
    no_firmware_reached(&strap);
}

/// An unreadable firmware revision refuses too — never "proceed because we could not check".
#[tokio::test(start_paused = true)]
async fn an_unreadable_firmware_revision_is_refused() {
    let strap = Strap::healthy();
    let mut client = WhoopClient::new(identity(&strap, Some(SERIAL), None), Family::Gen5);

    let err = client.flash_firmware(&three_chunks(), &armed(), |_| {}).await.unwrap_err();
    assert!(matches!(err, Error::Flash(FlashFault::WrongStrapFirmware(_))), "{err}");
    no_firmware_reached(&strap);
}

#[tokio::test(start_paused = true)]
async fn a_4_0_connection_is_refused() {
    let strap = Strap::healthy();
    let mut client = WhoopClient::new(transport(&strap), Family::Gen4);

    let err = client.flash_firmware(&three_chunks(), &armed(), |_| {}).await.unwrap_err();
    assert!(matches!(err, Error::Flash(FlashFault::WrongFamily)), "{err}");
    no_firmware_reached(&strap);
}

/// Every image fault surfaces before a write, the decompressed-image case included.
#[tokio::test(start_paused = true)]
async fn an_image_that_fails_its_own_header_is_refused() {
    for image in [
        vec![0u8; 16],                                 // too short
        image_with_broken_field(0x04, 9_999),          // declared length disagrees
        image_with_broken_field(0x1FC, 0xDEAD_BEEF),   // the CRC copy disagrees
        image_with_broken_field(0x1F8, 0xDEAD_BEEF),   // the header CRC disagrees
        image_with_broken_field(CONTAINER_OFFSET, 1),  // a decompressed image, not the container
    ] {
        let strap = Strap::healthy();
        let mut client = WhoopClient::new(transport(&strap), Family::Gen5);
        let err = client.flash_firmware(&image, &armed(), |_| {}).await.unwrap_err();
        assert!(matches!(err, Error::Flash(FlashFault::Image(_))), "{err}");
        // The image is checked first, so nothing at all was written — not even the battery poll.
        assert!(strap.log().is_empty(), "a bad image must be named before the strap is touched");
    }
}

/// The image is checked ahead of the serial, so a wrong file is named even on the wrong strap.
#[tokio::test(start_paused = true)]
async fn a_bad_image_is_named_before_the_wrong_strap() {
    let strap = Strap::healthy();
    let mut client = WhoopClient::new(identity(&strap, Some("5A00000002"), Some(common::FIRMWARE)), Family::Gen5);

    let err = client.flash_firmware(&image_with_broken_field(0x1F8, 7), &armed(), |_| {}).await.unwrap_err();
    assert!(matches!(err, Error::Flash(FlashFault::Image(_))), "{err}");
    assert!(strap.log().is_empty());
}

#[tokio::test(start_paused = true)]
async fn the_battery_floor_holds_and_can_only_be_raised() {
    // Below the built-in floor.
    let strap = Strap::healthy().battery(&[Some(79)]);
    let mut client = WhoopClient::new(transport(&strap), Family::Gen5);
    let err = client.flash_firmware(&three_chunks(), &armed(), |_| {}).await.unwrap_err();
    assert!(matches!(err, Error::Flash(FlashFault::LowBattery { .. })), "{err}");
    no_firmware_reached(&strap);

    // A caller cannot lower it.
    let strap = Strap::healthy().battery(&[Some(79)]);
    let mut client = WhoopClient::new(transport(&strap), Family::Gen5);
    let opts = FlashOptions { min_battery_pct: 10.0, ..armed() };
    let err = client.flash_firmware(&three_chunks(), &opts, |_| {}).await.unwrap_err();
    assert!(matches!(err, Error::Flash(FlashFault::LowBattery { floor, .. }) if floor == 80.0), "{err}");

    // A caller can raise it.
    let strap = Strap::healthy().battery(&[Some(90)]);
    let mut client = WhoopClient::new(transport(&strap), Family::Gen5);
    let opts = FlashOptions { min_battery_pct: 95.0, ..armed() };
    let err = client.flash_firmware(&three_chunks(), &opts, |_| {}).await.unwrap_err();
    assert!(matches!(err, Error::Flash(FlashFault::LowBattery { floor, .. }) if floor == 95.0), "{err}");

    // Exactly at the floor passes the gate.
    let strap = Strap::healthy().battery(&[Some(80)]);
    let mut client = WhoopClient::new(transport(&strap), Family::Gen5);
    let opts = FlashOptions { expect_serial: SERIAL.into(), ..Default::default() };
    assert_eq!(client.flash_firmware(&three_chunks(), &opts, |_| {}).await.unwrap().battery_pct, Some(80.0));
}

#[tokio::test(start_paused = true)]
async fn an_unreadable_battery_is_refused() {
    let strap = Strap::healthy().battery(&[None]);
    let mut client = WhoopClient::new(transport(&strap), Family::Gen5);

    let err = client.flash_firmware(&three_chunks(), &armed(), |_| {}).await.unwrap_err();
    assert!(matches!(err, Error::Flash(FlashFault::BatteryUnreadable)), "{err}");
    no_firmware_reached(&strap);
}

/// The gated door does not relax the blind path: the transfer opcodes stay refused there.
#[tokio::test(start_paused = true)]
async fn the_transfer_opcodes_stay_forbidden_on_the_blind_path() {
    let strap = Strap::healthy();
    let transport = transport(&strap);
    let mut client = WhoopClient::new(transport, Family::Gen5);

    for op in [
        command::START_FIRMWARE_LOAD_NEW,
        command::LOAD_FIRMWARE_DATA_NEW,
        command::PROCESS_FIRMWARE_IMAGE_NEW,
    ] {
        assert!(matches!(client.send_raw(op, &[0x01]).await, Err(Error::Forbidden(_))));
        assert!(matches!(client.probe(op, &[0x01], 0).await, Err(Error::Forbidden(_))));
    }
    assert!(strap.log().is_empty());
}

/// Frozen membership of both lists, so a later edit has to be deliberate.
#[test]
fn the_forbidden_and_destructive_lists_are_unchanged() {
    assert_eq!(
        command::FORBIDDEN,
        &[
            command::SET_CLOCK,
            command::SET_CLOCK_MAVERICK,
            command::FORCE_TRIM,
            command::REBOOT_STRAP,
            command::POWER_CYCLE_STRAP,
            command::ENTER_BLE_DFU,
            command::SET_ADVERTISING_NAME,
            command::SET_DEVICE_CONFIG,
            command::SET_CONFIG,
            command::RESET_FUEL_GAUGE,
            command::SELECT_WRIST,
            command::START_FIRMWARE_LOAD_NEW,
            command::LOAD_FIRMWARE_DATA_NEW,
            command::PROCESS_FIRMWARE_IMAGE_NEW,
        ]
    );
    assert_eq!(
        command::DESTRUCTIVE,
        &[
            command::FORCE_TRIM,
            command::ENTER_BLE_DFU,
            command::START_FIRMWARE_LOAD_NEW,
            command::LOAD_FIRMWARE_DATA_NEW,
            command::PROCESS_FIRMWARE_IMAGE_NEW,
        ]
    );
    // The check command is in neither list, and the radio DFU route has no caller here.
    assert!(!command::is_forbidden(command::VERIFY_FIRMWARE_IMAGE));
    assert!(!command::is_destructive(command::VERIFY_FIRMWARE_IMAGE));
    assert!(command::is_forbidden(command::ENTER_BLE_DFU));
}
