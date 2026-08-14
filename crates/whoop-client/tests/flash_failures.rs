//! What the transfer does when the strap refuses, goes quiet, answers the wrong thing, or the link
//! drops. The invariant under all of it: nothing is committed unless VERIFY answered success.

mod common;

use common::{three_chunks, transport, Flaky, Quirk, Strap, SERIAL};

use whoop_protocol::command;
use whoop_protocol::event::ResultCode;
use whoop_protocol::Family;

use whoop_client::{Error, FlashArm, FlashFault, FlashOptions, FlashStep, WhoopClient};

const START: u8 = command::START_FIRMWARE_LOAD_NEW;
const LOAD: u8 = command::LOAD_FIRMWARE_DATA_NEW;
const VERIFY: u8 = command::VERIFY_FIRMWARE_IMAGE;
const PROCESS: u8 = command::PROCESS_FIRMWARE_IMAGE_NEW;

fn armed() -> FlashOptions {
    FlashOptions { expect_serial: SERIAL.into(), arm: FlashArm::Commit, ..Default::default() }
}

async fn run(strap: &std::sync::Arc<Strap>) -> Result<whoop_client::FlashReport, Error> {
    let mut client = WhoopClient::new(transport(strap), Family::Gen5);
    client.flash_firmware(&three_chunks(), &armed(), |_| {}).await
}

#[tokio::test(start_paused = true)]
async fn a_refused_start_sends_no_chunk() {
    let strap = Strap::healthy().quirk(START, 0, Quirk::Code(0));
    let err = run(&strap).await.unwrap_err();

    assert!(matches!(err, Error::Flash(FlashFault::Refused { step: FlashStep::Started, .. })), "{err}");
    assert!(!strap.flow().contains(&LOAD));
}

#[tokio::test(start_paused = true)]
async fn a_silent_start_stops_the_run() {
    let strap = Strap::healthy().quirk(START, 0, Quirk::Silence);
    let err = run(&strap).await.unwrap_err();

    assert!(matches!(err, Error::Flash(FlashFault::NoReply { step: FlashStep::Started })), "{err}");
    assert!(!strap.flow().contains(&LOAD));
}

/// An explicit refusal on a chunk aborts at once: the strap's duplicate-address latch would return the
/// same code for a re-send, so retrying could not succeed.
#[tokio::test(start_paused = true)]
async fn a_refused_chunk_aborts_the_whole_transfer() {
    let strap = Strap::healthy().quirk(LOAD, 1, Quirk::Status(0, 0x0B));
    let err = run(&strap).await.unwrap_err();

    match err {
        Error::Flash(FlashFault::ChunkRefused { offset, result, status }) => {
            assert_eq!(offset, 220);
            assert_eq!(result, Some(ResultCode::Failure));
            assert_eq!(status, Some(0x0B));
        }
        other => panic!("expected ChunkRefused, got {other}"),
    }
    assert_eq!(strap.load_offsets(), vec![0, 220], "the third chunk must not be sent");
    assert!(!strap.flow().contains(&VERIFY));
    assert!(!strap.flow().contains(&PROCESS));
}

/// A lost reply re-sends the SAME offset, which the strap's duplicate-address latch accepts.
#[tokio::test(start_paused = true)]
async fn a_lost_chunk_reply_resends_the_same_offset() {
    let strap = Strap::healthy().quirk(LOAD, 1, Quirk::Silence);
    let report = run(&strap).await.unwrap();

    assert_eq!(strap.load_offsets(), vec![0, 220, 220, 440]);
    assert_eq!(report.retries, 1);
    assert_eq!(report.chunks_sent, 3);
    assert_eq!(report.reached, FlashStep::Committed);
}

#[tokio::test(start_paused = true)]
async fn the_retry_budget_is_global_and_stops_the_transfer() {
    let strap = Strap::healthy();
    for nth in 0..8 {
        strap.quirk(LOAD, nth, Quirk::Silence);
    }
    let err = run(&strap).await.unwrap_err();

    assert!(matches!(err, Error::Flash(FlashFault::RetriesExhausted { offset: 0 })), "{err}");
    assert!(!strap.flow().contains(&VERIFY));
    assert!(!strap.flow().contains(&PROCESS));
}

/// PENDING is not a final code: it never completes a wait, so a step answering only PENDING can end
/// only in a timeout. That is a deliberate departure from the app, which counts PENDING as an
/// acknowledgement — waiting can only fail closed, accepting it could advance on an unwritten chunk.
#[tokio::test(start_paused = true)]
async fn a_pending_reply_never_advances_the_machine() {
    let strap = Strap::healthy();
    for nth in 0..8 {
        strap.quirk(LOAD, nth, Quirk::PendingOnly);
    }
    let err = run(&strap).await.unwrap_err();

    assert!(matches!(err, Error::Flash(FlashFault::RetriesExhausted { offset: 0 })), "{err}");
    assert!(!strap.flow().contains(&VERIFY));
}

/// A PENDING and its final reply can arrive in ONE notification, so both come out of a single deframer
/// push. The final one must still be the answer, not a frame dropped behind the interim code.
#[tokio::test(start_paused = true)]
async fn a_pending_followed_by_its_final_reply_in_one_push_is_answered() {
    let strap = Strap::healthy();
    for (cmd, nth) in [(START, 0), (LOAD, 0), (LOAD, 1), (LOAD, 2), (VERIFY, 0), (PROCESS, 0)] {
        strap.quirk(cmd, nth, Quirk::PendingThenFinal);
    }
    let report = run(&strap).await.unwrap();

    assert_eq!(report.reached, FlashStep::Committed);
    assert_eq!(report.retries, 0);
    assert_eq!(report.stray_frames, 0, "the interim reply answered its command too");
}

#[tokio::test(start_paused = true)]
async fn a_frame_answering_nothing_is_counted_and_stepped_over() {
    let strap = Strap::healthy().quirk(START, 0, Quirk::StrayFirst);
    let report = run(&strap).await.unwrap();

    assert!(report.stray_frames > 0);
    assert_eq!(report.reached, FlashStep::Committed);
}

/// The right opcode echoing the wrong origin seq is not this command's reply.
#[tokio::test(start_paused = true)]
async fn a_reply_echoing_the_wrong_sequence_is_not_accepted() {
    let strap = Strap::healthy().quirk(START, 0, Quirk::WrongSeq);
    let err = run(&strap).await.unwrap_err();

    assert!(matches!(err, Error::Flash(FlashFault::NoReply { step: FlashStep::Started })), "{err}");
    assert!(!strap.flow().contains(&LOAD));
}

/// A failed check is a recorded outcome, not an error to work around: the run stops at Loaded with the
/// code the strap gave, and the commit is never built.
#[tokio::test(start_paused = true)]
async fn a_failed_verify_stops_short_of_the_commit() {
    for (code, expected) in [(0u8, ResultCode::Failure), (0x0A, ResultCode::Unknown(0x0A)), (3, ResultCode::Unsupported)] {
        let strap = Strap::healthy().quirk(VERIFY, 0, Quirk::Code(code));
        let report = run(&strap).await.unwrap();

        assert_eq!(report.verify, Some(expected));
        assert_eq!(report.reached, FlashStep::Loaded);
        assert_eq!(report.process, None);
        assert!(!strap.flow().contains(&PROCESS));
        // The chunk handlers reported their status, so it is on the record even though VERIFY has none.
        assert_eq!(report.last_status, Some(0));
        // Exactly one check was sent — there is no retry of it.
        assert_eq!(strap.flow().iter().filter(|&&c| c == VERIFY).count(), 1);
    }
}

#[tokio::test(start_paused = true)]
async fn a_silent_verify_stops_short_of_the_commit() {
    let strap = Strap::healthy().quirk(VERIFY, 0, Quirk::Silence);
    let err = run(&strap).await.unwrap_err();

    assert!(matches!(err, Error::Flash(FlashFault::NoReply { step: FlashStep::Verified })), "{err}");
    assert!(!strap.flow().contains(&PROCESS));
}

/// The strap reboots a few seconds after a successful commit; the driver sends nothing after it.
#[tokio::test(start_paused = true)]
async fn a_successful_commit_ends_the_run_and_writes_nothing_further() {
    let strap = Strap::healthy();
    let report = run(&strap).await.unwrap();

    assert_eq!(report.reached, FlashStep::Committed);
    assert_eq!(report.process, Some(ResultCode::Success));
    assert_eq!(*strap.flow().last().unwrap(), PROCESS);
}

/// A refused commit is recorded, never retried.
#[tokio::test(start_paused = true)]
async fn a_refused_commit_is_recorded_once() {
    let strap = Strap::healthy().quirk(PROCESS, 0, Quirk::Code(0));
    let report = run(&strap).await.unwrap();

    assert_eq!(report.process, Some(ResultCode::Failure));
    assert_eq!(report.reached, FlashStep::Verified);
    assert_eq!(strap.flow().iter().filter(|&&c| c == PROCESS).count(), 1);
}

#[tokio::test(start_paused = true)]
async fn a_link_that_drops_mid_transfer_aborts_before_the_check() {
    let strap = Strap::healthy();
    // battery, abort, start, chunk 0 land; the next write fails.
    let mut client = WhoopClient::new(Flaky::new(transport(&strap), 4), Family::Gen5);

    let err = client.flash_firmware(&three_chunks(), &armed(), |_| {}).await.unwrap_err();
    assert!(matches!(err, Error::Ble(_)), "{err}");
    assert_eq!(strap.load_offsets(), vec![0]);
    assert!(!strap.flow().contains(&VERIFY));
    assert!(!strap.flow().contains(&PROCESS));
}

/// Battery is re-read in the seconds before the commit; a drop below the floor stops it there.
#[tokio::test(start_paused = true)]
async fn a_battery_that_falls_before_the_commit_stops_it() {
    let strap = Strap::healthy().battery(&[Some(95), Some(50)]);
    let err = run(&strap).await.unwrap_err();

    assert!(matches!(err, Error::Flash(FlashFault::LowBattery { .. })), "{err}");
    assert!(!strap.flow().contains(&PROCESS));
    // Everything up to and including the check still happened — only the commit was withheld.
    assert!(strap.flow().contains(&VERIFY));
}

/// An abort returns an error and no report, so the run's last act must be to state how far it got —
/// otherwise the one moment an operator needs the strap's state is the one that never prints it.
#[tokio::test(start_paused = true)]
async fn an_aborted_run_still_reports_where_it_stopped() {
    let strap = Strap::healthy().battery(&[Some(95), Some(50)]);
    let mut client = WhoopClient::new(transport(&strap), Family::Gen5);
    let mut ended = Vec::new();

    let err = client
        .flash_firmware(&three_chunks(), &armed(), |p| {
            if let whoop_client::FlashProgress::Ended { reached, chunks_sent, retries, .. } = p {
                ended.push((reached, chunks_sent, retries));
            }
        })
        .await
        .unwrap_err();

    assert!(matches!(err, Error::Flash(FlashFault::LowBattery { .. })), "{err}");
    assert_eq!(ended, vec![(FlashStep::Verified, 3, 0)]);
}
