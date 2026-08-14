//! The shape of a clean transfer: command order, chunk offsets, sequence numbers, payload layout, and
//! what each arm level is allowed to send.

mod common;

use common::{three_chunks, transport, Strap, SERIAL};

use whoop_protocol::firmware::CHUNK_AMBIQ;
use whoop_protocol::{command, framing, Family};

use whoop_client::{FlashArm, FlashOptions, FlashStep, Quiesce, WhoopClient};

const ABORT: u8 = command::ABORT_HISTORICAL_TRANSMITS;
const START: u8 = command::START_FIRMWARE_LOAD_NEW;
const LOAD: u8 = command::LOAD_FIRMWARE_DATA_NEW;
const VERIFY: u8 = command::VERIFY_FIRMWARE_IMAGE;
const PROCESS: u8 = command::PROCESS_FIRMWARE_IMAGE_NEW;

fn opts(arm: FlashArm, quiesce: Quiesce) -> FlashOptions {
    FlashOptions { expect_serial: SERIAL.into(), arm, quiesce, ..Default::default() }
}

#[tokio::test(start_paused = true)]
async fn a_committed_run_sends_abort_start_loads_verify_process_in_that_order() {
    let strap = Strap::healthy();
    let mut client = WhoopClient::new(transport(&strap), Family::Gen5);

    let report =
        client.flash_firmware(&three_chunks(), &opts(FlashArm::Commit, Quiesce::AbortOnly), |_| {}).await.unwrap();

    assert_eq!(strap.flow(), vec![ABORT, START, LOAD, LOAD, LOAD, VERIFY, PROCESS]);
    assert_eq!(report.reached, FlashStep::Committed);
    assert_eq!(report.chunks_sent, 3);
    assert_eq!(report.retries, 0);
}

/// The two toggles differ in shape: the HR stream takes a bare off byte, the IMU stream takes the
/// revision marker first. Both are awaited, so only one command is ever outstanding.
#[tokio::test(start_paused = true)]
async fn the_app_identical_quiesce_leads_with_the_two_toggles() {
    let strap = Strap::healthy();
    let handle = transport(&strap);
    let mock = handle.clone();
    let mut client = WhoopClient::new(handle, Family::Gen5);

    let report =
        client.flash_firmware(&three_chunks(), &opts(FlashArm::Commit, Quiesce::AppIdentical), |_| {}).await.unwrap();

    let expected = vec![command::TOGGLE_REALTIME_HR, command::SET_IMU_DATA_STREAM, ABORT, START, LOAD, LOAD, LOAD, VERIFY, PROCESS];
    assert_eq!(strap.flow(), expected);
    assert_eq!(report.stray_frames, 0, "every reply answered an outstanding command");
    assert_eq!(report.reached, FlashStep::Committed);

    let payload = |op: u8| {
        mock.writes()
            .iter()
            .map(|(_, w)| framing::decode(Family::Gen5, w).unwrap())
            .find(|f| f.cmd() == op)
            .map(|f| f.payload().to_vec())
            .unwrap()
    };
    // Trailing zeros are the inner's 4-byte alignment pad, not payload.
    assert_eq!(payload(command::TOGGLE_REALTIME_HR), vec![0x00]);
    assert_eq!(payload(command::SET_IMU_DATA_STREAM)[..2], [0x01, 0x00]);
}

/// A real transfer runs thousands of chunks, so the one-byte sequence counter wraps many times over.
/// The block reserved for the chunks must line up with the frames actually numbered, and the check
/// after them must carry the next seq of all.
#[tokio::test(start_paused = true)]
async fn the_sequence_counter_wraps_across_a_long_transfer() {
    let strap = Strap::healthy();
    let mut client = WhoopClient::new(transport(&strap), Family::Gen5);

    let report = client
        .flash_firmware(&common::wrapping_chunks(), &opts(FlashArm::Commit, Quiesce::AbortOnly), |_| {})
        .await
        .unwrap();
    assert_eq!(report.chunks_sent, 600);

    let log = strap.log();
    for (i, (cmd, seq)) in log.iter().enumerate() {
        let want = log[0].1.wrapping_add(i as u8);
        assert_eq!(*seq, want, "frame {i} (opcode {cmd}) carried seq {seq}, expected {want}");
    }
    let loads: Vec<(u8, u8)> = log.iter().copied().filter(|&(c, _)| c == LOAD).collect();
    let verify = log.iter().find(|&&(c, _)| c == VERIFY).unwrap();
    assert_eq!(verify.1, loads.last().unwrap().1.wrapping_add(1));
}

#[tokio::test(start_paused = true)]
async fn with_no_quiesce_the_run_opens_on_start() {
    let strap = Strap::healthy();
    let mut client = WhoopClient::new(transport(&strap), Family::Gen5);

    client.flash_firmware(&three_chunks(), &opts(FlashArm::Commit, Quiesce::None), |_| {}).await.unwrap();
    assert_eq!(strap.flow(), vec![START, LOAD, LOAD, LOAD, VERIFY, PROCESS]);
}

#[tokio::test(start_paused = true)]
async fn chunk_offsets_are_ascending_contiguous_and_cover_the_whole_image() {
    let strap = Strap::healthy();
    let image = three_chunks();
    let mut client = WhoopClient::new(transport(&strap), Family::Gen5);

    client.flash_firmware(&image, &opts(FlashArm::Commit, Quiesce::AbortOnly), |_| {}).await.unwrap();

    let offsets = strap.load_offsets();
    assert_eq!(offsets, vec![0, CHUNK_AMBIQ as u32, 2 * CHUNK_AMBIQ as u32]);
    let last = strap.last_load_body().unwrap();
    let declared = last[5] as usize;
    assert_eq!(*offsets.last().unwrap() as usize + declared, image.len());
}

/// One command is outstanding at a time and every frame carries the next seq, so a reply can only be
/// matched to the command that asked for it.
#[tokio::test(start_paused = true)]
async fn every_written_frame_carries_the_next_sequence_number() {
    let strap = Strap::healthy();
    let mut client = WhoopClient::new(transport(&strap), Family::Gen5);

    client.flash_firmware(&three_chunks(), &opts(FlashArm::Commit, Quiesce::AbortOnly), |_| {}).await.unwrap();

    let log = strap.log();
    for (i, (cmd, seq)) in log.iter().enumerate() {
        let want = log[0].1.wrapping_add(i as u8);
        assert_eq!(*seq, want, "frame {i} (opcode {cmd}) carried seq {seq}, expected {want}");
    }
}

#[tokio::test(start_paused = true)]
async fn staging_never_sends_the_commit_even_on_a_successful_verify() {
    let strap = Strap::healthy();
    let mut client = WhoopClient::new(transport(&strap), Family::Gen5);

    let report =
        client.flash_firmware(&three_chunks(), &opts(FlashArm::Stage, Quiesce::AbortOnly), |_| {}).await.unwrap();

    assert!(!strap.flow().contains(&PROCESS));
    assert_eq!(report.reached, FlashStep::Verified);
    assert_eq!(report.verify, Some(whoop_protocol::event::ResultCode::Success));
    assert_eq!(report.process, None);
}

/// Planning still polls the battery, since that is one of the gates it exists to evaluate. What it must
/// never write is a command that moves the transfer.
#[tokio::test(start_paused = true)]
async fn planning_sends_only_the_battery_gate_even_with_a_strap_answering() {
    let strap = Strap::healthy();
    let mut client = WhoopClient::new(transport(&strap), Family::Gen5);

    client.flash_firmware(&three_chunks(), &opts(FlashArm::Plan, Quiesce::AbortOnly), |_| {}).await.unwrap();

    let written: Vec<u8> = strap.log().into_iter().map(|(c, _)| c).collect();
    assert_eq!(written, vec![command::GET_BATTERY_LEVEL], "plan wrote {written:?}");
}

/// The revision marker leads every firmware command, and a chunk body is offset, length, then data.
#[tokio::test(start_paused = true)]
async fn the_revision_marker_leads_and_the_chunk_body_is_offset_length_data() {
    let strap = Strap::healthy();
    let handle = transport(&strap);
    let mock = handle.clone();
    let image = three_chunks();
    let mut client = WhoopClient::new(handle, Family::Gen5);

    client.flash_firmware(&image, &opts(FlashArm::Commit, Quiesce::AbortOnly), |_| {}).await.unwrap();

    for (_, wire) in mock.writes() {
        let f = framing::decode(Family::Gen5, &wire).unwrap();
        if [START, LOAD, VERIFY, PROCESS].contains(&f.cmd()) {
            assert_eq!(f.payload()[0], 0x01, "opcode {} lost its revision marker", f.cmd());
        }
    }

    let first = strap.load_offsets()[0];
    assert_eq!(first, 0);
    let last = strap.last_load_body().unwrap();
    assert_eq!(&last[1..5], &(2 * CHUNK_AMBIQ as u32).to_le_bytes());
    assert_eq!(last[5] as usize, CHUNK_AMBIQ);
    assert_eq!(&last[6..6 + CHUNK_AMBIQ], &image[2 * CHUNK_AMBIQ..]);
}
