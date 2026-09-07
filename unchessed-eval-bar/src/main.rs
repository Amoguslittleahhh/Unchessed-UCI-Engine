use std::env;
use std::fs;
use std::time::Instant;

use unchessed_core::eval_bar::{wdl_from_display_cp_exact, wdl_from_display_cp_fast, EvalBar};
use unchessed_core::fen;

fn main() {
    let args = env::args().skip(1).collect::<Vec<_>>();
    if args.first().map(String::as_str) == Some("--benchmark") {
        let path = args
            .get(1)
            .map(String::as_str)
            .unwrap_or("benchmarks/matetrack.epd");
        benchmark(path);
        return;
    }
    let fen_text = args.join(" ");
    let pos = if fen_text.is_empty() || fen_text == "startpos" {
        fen::startpos()
    } else {
        fen::parse(&fen_text).unwrap_or_else(|error| {
            eprintln!("invalid FEN: {error}");
            std::process::exit(2);
        })
    };
    let mut bar = EvalBar::new();
    let sample = bar.sample(&pos);
    println!("static_cp_white={}", sample.static_cp_white);
    println!("display_cp_white={}", sample.display_cp_white);
    println!("wdl_white_draw_black={:?}", sample.wdl_per_mille);
    println!("expected_score={:.6}", sample.expected_score);
    println!("bar_fraction={:.6}", sample.bar_fraction);
    println!("confidence={:.6}", sample.confidence);
    println!(
        "smoothed_cp_white={}",
        bar.smoothed_cp_white().unwrap_or(sample.display_cp_white)
    );
    println!("provenance=unchessed_hce_proxy_not_exact_sfnnv16");
}

fn benchmark(path: &str) {
    let text = fs::read_to_string(path).unwrap_or_else(|error| {
        eprintln!("cannot read benchmark corpus {path}: {error}");
        std::process::exit(2);
    });
    let positions = text
        .lines()
        .filter(|line| !line.trim().is_empty() && !line.trim_start().starts_with('#'))
        .filter_map(|line| {
            let fen4 = line.split(" bm ").next()?.trim();
            let mut fields = fen4.split_whitespace();
            let fen = format!(
                "{} {} {} {} {} {}",
                fields.next()?,
                fields.next()?,
                fields.next()?,
                fields.next()?,
                0,
                1
            );
            fen::parse(&fen).ok()
        })
        .collect::<Vec<_>>();
    assert!(
        !positions.is_empty(),
        "benchmark corpus contains no parseable EPD positions"
    );

    // Exclude one-time tensor construction from steady-state throughput.
    let _ = wdl_from_display_cp_fast(137, &positions[0]);
    let rounds = 10_000usize;
    let start = Instant::now();
    let mut exact_checksum = 0u64;
    for _ in 0..rounds {
        for pos in &positions {
            let wdl = wdl_from_display_cp_exact(137, pos);
            exact_checksum = exact_checksum.wrapping_add(u64::from(wdl[0]));
        }
    }
    let exact_ns = start.elapsed().as_nanos();

    let start = Instant::now();
    let mut fast_checksum = 0u64;
    for _ in 0..rounds {
        for pos in &positions {
            let fast = wdl_from_display_cp_fast(137, pos);
            fast_checksum = fast_checksum.wrapping_add(u64::from(fast[0]));
        }
    }
    let fast_ns = start.elapsed().as_nanos();
    let mut max_delta = 0i32;
    for pos in &positions {
        let fast = wdl_from_display_cp_fast(137, pos);
        let exact = wdl_from_display_cp_exact(137, pos);
        for (a, b) in fast.iter().zip(exact.iter()) {
            max_delta = max_delta.max((i32::from(*a) - i32::from(*b)).abs());
        }
    }
    println!("corpus={path}");
    println!("positions={}", positions.len());
    println!("rounds={rounds}");
    println!("exact_elapsed_ns={exact_ns}");
    println!("fast_elapsed_ns={fast_ns}");
    println!("max_wdl_delta_per_mille={max_delta}");
    println!("checksums_exact={exact_checksum} fast={fast_checksum}");
}
