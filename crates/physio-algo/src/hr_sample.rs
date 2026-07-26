//! The one heart-rate sample every plain-value metric takes: a wall-clock unix second and an integer
//! bpm. Shared by `recovery`, `resting_hr`, `strain`, `hr_zones`, `calories` and `workout` so a caller
//! builds one series and feeds it to any of them. The sleep stager keeps its own `sleep::HrSample`
//! (u16 bpm) as its protocol-free input type.

/// One HR reading: unix-second `ts` and integer `bpm`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HrSample {
    pub ts: i64,
    pub bpm: i32,
}

impl HrSample {
    pub fn new(ts: i64, bpm: i32) -> Self {
        Self { ts, bpm }
    }
}
