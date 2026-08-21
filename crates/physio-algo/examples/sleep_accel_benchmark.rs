//! Reproducible SleepAccel PSG report and offline-only `deep_hr` ablation.
//!
//! WHOOP_SLEEP_FIXTURES=/path/to/fixture-root cargo run --release -p physio-algo \
//!   --example sleep_accel_benchmark

mod common;

use common::{dirs_of, read_accel, read_hr, read_meta, read_rr, read_truth};
use physio_algo::sleep::{stage_v2_with, Params, SleepInput, SleepStage, StageSegment};

const NAMES: [&str; 4] = ["Wake", "Light", "Deep", "REM"];

#[derive(Clone)]
struct NightResult {
    id: String,
    cm: [[i64; 4]; 4],
    truth_epochs: usize,
    bias: [f64; 7], // TST/Wake/Light/Deep/REM min, efficiency pp, Deep pp
}

fn idx(stage: SleepStage) -> usize {
    match stage {
        SleepStage::Wake => 0,
        SleepStage::Light => 1,
        SleepStage::Deep => 2,
        SleepStage::Rem => 3,
    }
}

fn at(segs: &[StageSegment], timestamp: i64) -> usize {
    idx(segs
        .iter()
        .find(|s| s.start <= timestamp && timestamp < s.end)
        .or(segs.last())
        .expect("non-empty staging")
        .stage)
}

fn kappa(cm: &[[i64; 4]; 4]) -> f64 {
    let n: i64 = cm.iter().flatten().sum();
    if n == 0 {
        return f64::NAN;
    }
    let observed = (0..4).map(|i| cm[i][i]).sum::<i64>() as f64 / n as f64;
    let expected = (0..4)
        .map(|i| cm[i].iter().sum::<i64>() as f64 * cm.iter().map(|r| r[i]).sum::<i64>() as f64)
        .sum::<f64>()
        / (n as f64).powi(2);
    (observed - expected) / (1.0 - expected)
}

fn score(params: Params) -> Vec<NightResult> {
    let mut results = Vec::new();
    for dir in dirs_of("sleep-accel") {
        let Some((w0, w1, n_epochs)) = read_meta(&dir) else {
            continue;
        };
        let truth = read_truth(&dir);
        let input = SleepInput {
            start: w0,
            end: w1,
            hr: read_hr(&dir),
            rr: read_rr(&dir),
            accel: read_accel(&dir),
        };
        let segs = stage_v2_with(&input, &params);
        let mut cm = [[0i64; 4]; 4];
        for (&epoch, &label) in &truth {
            if epoch < n_epochs && (0..4).contains(&label) {
                cm[label as usize][at(&segs, w0 + epoch as i64 * 30 + 15)] += 1;
            }
        }
        let mut truth_n = [0i64; 4];
        let mut pred_n = [0i64; 4];
        for i in 0..4 {
            truth_n[i] = cm[i].iter().sum();
            pred_n[i] = cm.iter().map(|r| r[i]).sum();
        }
        let truth_sleep: i64 = truth_n[1..].iter().sum();
        let pred_sleep: i64 = pred_n[1..].iter().sum();
        let evaluable = truth_n.iter().sum::<i64>();
        let bias = [
            (pred_sleep - truth_sleep) as f64 / 2.0,
            (pred_n[0] - truth_n[0]) as f64 / 2.0,
            (pred_n[1] - truth_n[1]) as f64 / 2.0,
            (pred_n[2] - truth_n[2]) as f64 / 2.0,
            (pred_n[3] - truth_n[3]) as f64 / 2.0,
            100.0 * (pred_sleep - truth_sleep) as f64 / evaluable as f64,
            100.0 * (pred_n[2] - truth_n[2]) as f64 / evaluable as f64,
        ];
        results.push(NightResult {
            id: dir.file_name().unwrap().to_string_lossy().into_owned(),
            cm,
            truth_epochs: n_epochs,
            bias,
        });
    }
    results
}

fn pooled(nights: &[NightResult]) -> [[i64; 4]; 4] {
    let mut cm = [[0i64; 4]; 4];
    for night in nights {
        for i in 0..4 {
            for j in 0..4 {
                cm[i][j] += night.cm[i][j];
            }
        }
    }
    cm
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    let n = values.len();
    if n % 2 == 0 {
        (values[n / 2 - 1] + values[n / 2]) / 2.0
    } else {
        values[n / 2]
    }
}

fn summarize(label: &str, nights: &[NightResult]) {
    let cm = pooled(nights);
    let total: i64 = cm.iter().flatten().sum();
    let accuracy = (0..4).map(|i| cm[i][i]).sum::<i64>() as f64 / total as f64;
    println!("\n=== {label} ===");
    println!(
        "nights={} truth_epochs={} evaluable_epochs={} coverage={:.3}%",
        nights.len(),
        nights.iter().map(|n| n.truth_epochs).sum::<usize>(),
        total,
        100.0 * total as f64 / nights.iter().map(|n| n.truth_epochs).sum::<usize>() as f64
    );
    println!("counts (truth rows, predicted columns Wake Light Deep REM)");
    for i in 0..4 {
        println!(
            "{:<5} {:>7} {:>7} {:>7} {:>7}",
            NAMES[i], cm[i][0], cm[i][1], cm[i][2], cm[i][3]
        );
    }
    println!("row_percent");
    for i in 0..4 {
        let row: i64 = cm[i].iter().sum();
        println!(
            "{:<5} {:>7.2} {:>7.2} {:>7.2} {:>7.2}",
            NAMES[i],
            100.0 * cm[i][0] as f64 / row as f64,
            100.0 * cm[i][1] as f64 / row as f64,
            100.0 * cm[i][2] as f64 / row as f64,
            100.0 * cm[i][3] as f64 / row as f64
        );
    }
    println!("accuracy={accuracy:.6} kappa={:.6}", kappa(&cm));
    println!("stage recall precision f1");
    for i in 0..4 {
        let recall = cm[i][i] as f64 / cm[i].iter().sum::<i64>() as f64;
        let precision = cm[i][i] as f64 / cm.iter().map(|r| r[i]).sum::<i64>() as f64;
        let f1 = 2.0 * recall * precision / (recall + precision);
        println!(
            "{:<5} {:>8.4} {:>9.4} {:>8.4}",
            NAMES[i], recall, precision, f1
        );
    }
    println!("bias_minutes metric mean median mae min max");
    for (metric, column) in ["TST", "Wake", "Light", "Deep", "REM", "Eff_pp", "Deep_pp"]
        .iter()
        .zip(0..7)
    {
        let values: Vec<f64> = nights.iter().map(|n| n.bias[column]).collect();
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let mae = values.iter().map(|v| v.abs()).sum::<f64>() / values.len() as f64;
        let min = values.iter().copied().fold(f64::INFINITY, f64::min);
        let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        println!(
            "{metric:<5} {mean:>8.2} {:>8.2} {mae:>8.2} {min:>8.2} {max:>8.2}",
            median(values)
        );
    }
    let mut outliers = nights.to_vec();
    outliers.sort_by(|a, b| {
        b.bias[..5]
            .iter()
            .map(|v| v.abs())
            .fold(0.0, f64::max)
            .total_cmp(&a.bias[..5].iter().map(|v| v.abs()).fold(0.0, f64::max))
    });
    println!("largest absolute per-stage biases (id: TST Wake Light Deep REM)");
    for n in outliers.iter().take(5) {
        println!(
            "{}: {:.1} {:.1} {:.1} {:.1} {:.1}",
            n.id, n.bias[0], n.bias[1], n.bias[2], n.bias[3], n.bias[4]
        );
    }
}

fn main() {
    for value in [0.5, 0.0, -0.5] {
        let params = Params {
            deep_hr: value,
            ..Params::SHIPPED
        };
        summarize(&format!("deep_hr={value:+.1}"), &score(params));
    }
}
