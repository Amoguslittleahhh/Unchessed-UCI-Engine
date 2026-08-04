//! Training-data generator for the Maia-style human policy net.
//!
//! Reads Lichess PGN dumps, replays every rated non-bullet game with our own
//! movegen, and writes per-rating-bucket binary samples (record v2, 104 bytes):
//!
//!   bytes 0-95   12 x u64 LE bitboards (side-to-move normalized: planes 0-5 =
//!                mover P,N,B,R,Q,K; 6-11 = opponent)
//!   96-97        u16 move (from | to<<6, normalized)
//!   98-99        u16 mover rating
//!   100          castling rights: bit0 mover-K, bit1 mover-Q, bit2 opp-K, bit3 opp-Q
//!   101          en-passant file 0-7, or 0xFF
//!   102          flags: 1 = castle move, 2 = en passant, 4 = promotion
//!   103          padding (0)
//!
//! En-passant and promotion samples bypass the acceptance subsample (they are
//! rare and the net must learn the special rules well).
//!
//! Usage: unchessed-datagen <out_dir> <cap_per_bucket> <accept_prob> <pgn...>
//!
//! NNUE mode instead labels quiet positions with a shallow HCE search and
//! writes 104-byte records:
//!
//!   bytes 0-95   12 x u64 LE bitboards, STM-normalized like above
//!   96-97        i16 LE search score in centipawns, from the STM perspective
//!   98           u8 WDL from the STM perspective (2 = win, 1 = draw, 0 = loss)
//!   99-103       padding (0)
//!
//! Usage: unchessed-datagen nnue <out_file> <worker_id> <n_workers>
//!                               <max_positions> <pgn...>

use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::sync::atomic::AtomicBool;
use std::time::Instant;

use unchessed_core::board::*;
use unchessed_core::eval::Hce;
use unchessed_core::fen;
use unchessed_core::movegen::in_check;
use unchessed_core::san::parse_san;
use unchessed_core::search::{self, Limits};
use unchessed_core::tt::TT;

pub const BUCKETS: [(u16, u16); 4] = [(0, 1299), (1300, 1599), (1600, 1899), (1900, 4000)];

fn bucket_of(rating: u16) -> usize {
    for (i, (lo, hi)) in BUCKETS.iter().enumerate() {
        if rating >= *lo && rating <= *hi {
            return i;
        }
    }
    BUCKETS.len() - 1
}

struct Rng(u64);
impl Rng {
    fn f64(&mut self) -> f64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        (x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Normalize so the mover is always "white": vertical flip for black.
fn write_sample(
    out: &mut BufWriter<File>,
    pos: &Position,
    mv: Move,
    rating: u16,
) -> std::io::Result<()> {
    let flip = matches!(pos.side, Color::Black);
    let (us, them) = (pos.side.idx(), pos.side.flip().idx());
    let mut buf = [0u8; 104];
    for p in 0..6 {
        let mover = if flip {
            pos.bb[us][p].swap_bytes()
        } else {
            pos.bb[us][p]
        };
        let opp = if flip {
            pos.bb[them][p].swap_bytes()
        } else {
            pos.bb[them][p]
        };
        buf[p * 8..p * 8 + 8].copy_from_slice(&mover.to_le_bytes());
        buf[48 + p * 8..48 + p * 8 + 8].copy_from_slice(&opp.to_le_bytes());
    }
    let (from, to) = if flip {
        (mv.from() ^ 56, mv.to() ^ 56)
    } else {
        (mv.from(), mv.to())
    };
    let packed = from as u16 | (to as u16) << 6;
    buf[96..98].copy_from_slice(&packed.to_le_bytes());
    buf[98..100].copy_from_slice(&rating.to_le_bytes());

    // castling rights, mover-relative
    let (mk, mq, ok, oq) = if flip {
        (BK, BQ, WK, WQ)
    } else {
        (WK, WQ, BK, BQ)
    };
    let mut castle = 0u8;
    if pos.castling & mk != 0 {
        castle |= 1;
    }
    if pos.castling & mq != 0 {
        castle |= 2;
    }
    if pos.castling & ok != 0 {
        castle |= 4;
    }
    if pos.castling & oq != 0 {
        castle |= 8;
    }
    buf[100] = castle;
    buf[101] = if pos.ep == NO_EP { 0xFF } else { file_of(pos.ep) };
    let mut flags = 0u8;
    if mv.kind() == MK_CASTLE {
        flags |= 1;
    }
    if mv.kind() == MK_EP {
        flags |= 2;
    }
    if mv.is_promo() {
        flags |= 4;
    }
    buf[102] = flags;
    buf[103] = 0;
    out.write_all(&buf)
}

#[derive(Default)]
struct Headers {
    white_elo: Option<u16>,
    black_elo: Option<u16>,
    base_secs: Option<u32>,
    result: Option<String>,
}

fn parse_header(line: &str, h: &mut Headers) {
    let inner = line.trim_start_matches('[').trim_end_matches(']');
    if let Some(rest) = inner.strip_prefix("WhiteElo ") {
        h.white_elo = rest.trim_matches('"').parse().ok();
    } else if let Some(rest) = inner.strip_prefix("BlackElo ") {
        h.black_elo = rest.trim_matches('"').parse().ok();
    } else if let Some(rest) = inner.strip_prefix("TimeControl ") {
        let tc = rest.trim_matches('"');
        h.base_secs = tc.split('+').next().and_then(|b| b.parse().ok());
    } else if let Some(rest) = inner.strip_prefix("Result ") {
        h.result = Some(rest.trim_matches('"').to_string());
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() >= 2 && args[1] == "nnue" {
        run_nnue(&args);
        return;
    }
    if args.len() >= 2 && args[1] == "book" {
        run_book(&args);
        return;
    }
    if args.len() < 5 {
        eprintln!("usage: {} <out_dir> <cap_per_bucket> <accept_prob> <pgn...>", args[0]);
        eprintln!(
            "   or: {} nnue <out_file> <worker_id> <n_workers> <max_positions> <pgn...>",
            args[0]
        );
        eprintln!(
            "   or: {} book <out.pgn> <max_openings> <min_ply> <max_ply> <pgn...>",
            args[0]
        );
        std::process::exit(1);
    }
    let out_dir = &args[1];
    let cap: u64 = args[2].parse().expect("cap");
    let accept: f64 = args[3].parse().expect("accept_prob");
    std::fs::create_dir_all(out_dir).unwrap();

    let mut writers: Vec<BufWriter<File>> = (0..BUCKETS.len())
        .map(|i| {
            BufWriter::new(File::create(format!("{}/bucket{}.bin", out_dir, i)).unwrap())
        })
        .collect();
    let mut counts = vec![0u64; BUCKETS.len()];
    let mut rng = Rng(0x5EED_CAFE_2024_0720);
    let (mut games, mut used_games, mut skipped_moves) = (0u64, 0u64, 0u64);

    for path in &args[4..] {
        let f = File::open(path).unwrap_or_else(|e| panic!("open {}: {}", path, e));
        let mut r = BufReader::with_capacity(1 << 20, f);
        let mut line = String::new();
        let mut headers = Headers::default();
        let mut movetext = String::new();
        let mut in_moves = false;

        loop {
            line.clear();
            if r.read_line(&mut line).unwrap() == 0 {
                break;
            }
            let t = line.trim();
            if t.starts_with('[') {
                if in_moves {
                    // new game begins
                    process_game(
                        &headers, &movetext, &mut writers, &mut counts, cap, accept,
                        &mut rng, &mut used_games, &mut skipped_moves,
                    );
                    games += 1;
                    headers = Headers::default();
                    movetext.clear();
                    in_moves = false;
                }
                parse_header(t, &mut headers);
            } else if !t.is_empty() {
                in_moves = true;
                movetext.push_str(t);
                movetext.push(' ');
            }
        }
        if in_moves {
            process_game(
                &headers, &movetext, &mut writers, &mut counts, cap, accept, &mut rng,
                &mut used_games, &mut skipped_moves,
            );
            games += 1;
        }
        eprintln!("done {}: games so far {}", path, games);
        if counts.iter().all(|&c| c >= cap) {
            break;
        }
    }

    for w in &mut writers {
        w.flush().unwrap();
    }
    eprintln!(
        "games seen {}, used {}, skipped-move events {}",
        games, used_games, skipped_moves
    );
    for (i, c) in counts.iter().enumerate() {
        eprintln!("bucket{} ({}-{}): {} samples", i, BUCKETS[i].0, BUCKETS[i].1, c);
    }
}

#[allow(clippy::too_many_arguments)]
fn process_game(
    h: &Headers,
    movetext: &str,
    writers: &mut [BufWriter<File>],
    counts: &mut [u64],
    cap: u64,
    accept: f64,
    rng: &mut Rng,
    used_games: &mut u64,
    skipped_moves: &mut u64,
) {
    let (we, be) = match (h.white_elo, h.black_elo) {
        (Some(w), Some(b)) => (w, b),
        _ => return,
    };
    // skip bullet: base time under 3 minutes
    if h.base_secs.map(|b| b < 180).unwrap_or(true) {
        return;
    }
    // if every relevant bucket is full, skip cheaply
    let wb = bucket_of(we);
    let bb = bucket_of(be);
    if counts[wb] >= cap && counts[bb] >= cap {
        return;
    }

    let mut pos = fen::startpos();
    let mut ply = 0u32;
    let mut in_comment = false;
    *used_games += 1;

    for tok in movetext.split_whitespace() {
        if in_comment {
            if tok.ends_with('}') {
                in_comment = false;
            }
            continue;
        }
        if tok.starts_with('{') {
            if !tok.ends_with('}') {
                in_comment = true;
            }
            continue;
        }
        if tok.starts_with('$') || tok == "*" || tok == "1-0" || tok == "0-1" || tok == "1/2-1/2" {
            continue;
        }
        // strip move numbers: "12." "12..." or attached "12.e4"
        let san = tok.trim_start_matches(|c: char| c.is_ascii_digit() || c == '.');
        if san.is_empty() {
            continue;
        }
        let mv = match parse_san(&pos, san) {
            Some(m) => m,
            None => {
                *skipped_moves += 1;
                return; // desynced: abandon rest of game
            }
        };
        let rating = if matches!(pos.side, Color::White) { we } else { be };
        let b = bucket_of(rating);
        // rare special-rule moves always pass the subsample gate
        let special = mv.kind() == MK_EP || mv.is_promo();
        if ply >= 2 && counts[b] < cap && (special || rng.f64() < accept) {
            if write_sample(&mut writers[b], &pos, mv, rating).is_ok() {
                counts[b] += 1;
            }
        }
        pos = pos.make(mv);
        ply += 1;
    }
}

// ---------------------------------------------------------------------------
// NNUE training-data mode
// ---------------------------------------------------------------------------

const NNUE_MIN_ELO: u16 = 1500;
const NNUE_MIN_BASE_SECS: u32 = 180;
const NNUE_MIN_GAME_PLIES: usize = 20;
const NNUE_MIN_PLY: usize = 10;
const NNUE_MAX_PER_GAME: u32 = 12;
const NNUE_PLY_GAP: usize = 4;
const NNUE_ACCEPT: f64 = 0.9;
const NNUE_LABEL_NODES: u64 = 5000;
const NNUE_MAX_ABS_SCORE: i32 = 2000;

fn run_nnue(args: &[String]) {
    if args.len() < 7 {
        eprintln!(
            "usage: {} nnue <out_file> <worker_id> <n_workers> <max_positions> <pgn...>",
            args[0]
        );
        std::process::exit(1);
    }
    let out_file = &args[2];
    let worker_id: u64 = args[3].parse().expect("worker_id");
    let n_workers: u64 = args[4].parse().expect("n_workers");
    assert!(
        n_workers > 0 && worker_id < n_workers,
        "need 0 <= worker_id < n_workers"
    );
    let max_positions: u64 = args[5].parse().expect("max_positions");

    let mut out = BufWriter::new(
        File::create(out_file).unwrap_or_else(|e| panic!("create {}: {}", out_file, e)),
    );
    // persistent per-worker TT, deliberately never cleared between positions
    let tt = TT::new(64);
    let mut rng = Rng(0x5EED_CAFE_2024_0720 ^ worker_id.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    let start = Instant::now();
    let mut samples = 0u64;
    let mut games = 0u64;
    let mut used_games = 0u64;

    'files: for path in &args[6..] {
        let f = File::open(path).unwrap_or_else(|e| panic!("open {}: {}", path, e));
        let mut r = BufReader::with_capacity(1 << 20, f);
        let mut line = String::new();
        let mut headers = Headers::default();
        let mut movetext = String::new();
        let mut in_moves = false;

        loop {
            line.clear();
            if r.read_line(&mut line).unwrap() == 0 {
                break;
            }
            let t = line.trim();
            if t.starts_with('[') {
                if in_moves {
                    // new game begins
                    if games % n_workers == worker_id {
                        process_nnue_game(
                            &headers, &movetext, &mut out, &tt, &mut rng, &mut samples,
                            max_positions, &mut used_games, &start,
                        );
                    }
                    games += 1;
                    headers = Headers::default();
                    movetext.clear();
                    in_moves = false;
                    if samples >= max_positions {
                        break 'files;
                    }
                }
                parse_header(t, &mut headers);
            } else if !t.is_empty() {
                in_moves = true;
                movetext.push_str(t);
                movetext.push(' ');
            }
        }
        if in_moves {
            if games % n_workers == worker_id {
                process_nnue_game(
                    &headers, &movetext, &mut out, &tt, &mut rng, &mut samples,
                    max_positions, &mut used_games, &start,
                );
            }
            games += 1;
        }
        eprintln!(
            "worker {}: done {}: {} samples so far ({} games seen)",
            worker_id, path, samples, games
        );
        if samples >= max_positions {
            break;
        }
    }

    out.flush().unwrap();
    let secs = start.elapsed().as_secs_f64().max(1e-9);
    eprintln!(
        "worker {}: {} samples from {} games ({} seen) in {:.0}s ({:.0} samples/s)",
        worker_id,
        samples,
        used_games,
        games,
        secs,
        samples as f64 / secs
    );
}

#[allow(clippy::too_many_arguments)]
fn process_nnue_game(
    h: &Headers,
    movetext: &str,
    out: &mut BufWriter<File>,
    tt: &TT,
    rng: &mut Rng,
    samples: &mut u64,
    max_positions: u64,
    used_games: &mut u64,
    start: &Instant,
) {
    if *samples >= max_positions {
        return;
    }
    // game-level filters
    let (we, be) = match (h.white_elo, h.black_elo) {
        (Some(w), Some(b)) => (w, b),
        _ => return,
    };
    if we < NNUE_MIN_ELO || be < NNUE_MIN_ELO {
        return;
    }
    if h.base_secs.map(|b| b < NNUE_MIN_BASE_SECS).unwrap_or(true) {
        return;
    }
    // WDL from White's perspective: 2 = white won, 1 = draw, 0 = black won
    let wdl_white: u8 = match h.result.as_deref() {
        Some("1-0") => 2,
        Some("1/2-1/2") => 1,
        Some("0-1") => 0,
        _ => return, // unfinished ("*") or missing result
    };

    // replay the whole game first: desynced SAN abandons it sample-free,
    // and the game-length filter needs the full ply count
    let mut pos = fen::startpos();
    let mut pre: Vec<Position> = Vec::new();
    let mut in_comment = false;
    for tok in movetext.split_whitespace() {
        if in_comment {
            if tok.ends_with('}') {
                in_comment = false;
            }
            continue;
        }
        if tok.starts_with('{') {
            if !tok.ends_with('}') {
                in_comment = true;
            }
            continue;
        }
        if tok.starts_with('$') || tok == "*" || tok == "1-0" || tok == "0-1" || tok == "1/2-1/2" {
            continue;
        }
        let san = tok.trim_start_matches(|c: char| c.is_ascii_digit() || c == '.');
        if san.is_empty() {
            continue;
        }
        let mv = match parse_san(&pos, san) {
            Some(m) => m,
            None => return, // desynced: abandon the game
        };
        pre.push(pos);
        pos = pos.make(mv);
    }
    if pre.len() < NNUE_MIN_GAME_PLIES {
        return;
    }
    *used_games += 1;

    let mut taken = 0u32;
    let mut last_ply: Option<usize> = None;
    for (ply, p) in pre.iter().enumerate() {
        if *samples >= max_positions || taken >= NNUE_MAX_PER_GAME {
            return;
        }
        // position-level filters
        if ply < NNUE_MIN_PLY {
            continue;
        }
        if let Some(lp) = last_ply {
            if ply - lp < NNUE_PLY_GAP {
                continue;
            }
        }
        if in_check(p) {
            continue;
        }
        if rng.f64() >= NNUE_ACCEPT {
            continue;
        }

        // label with a shallow fixed-node HCE search
        let limits = Limits {
            nodes: Some(NNUE_LABEL_NODES),
            ..Default::default()
        };
        let stop = AtomicBool::new(false);
        let lines = search::go(
            p,
            &Hce::default(),
            &limits,
            1,
            tt,
            &stop,
            &[],
            0,
            search::SearchParams::default(),
            1,
            &mut |_| {},
        );
        let l = match lines.first() {
            Some(l) => l,
            None => continue,
        };
        // depth 0 = the search's emergency fallback line; its score is not a label
        if l.depth < 1 {
            continue;
        }
        if search::is_mate_score(l.score) || l.score.abs() >= NNUE_MAX_ABS_SCORE {
            continue;
        }
        // skip tactical best moves: captures, en passant, promotions
        let m = l.mv;
        if p.board[m.to() as usize] != NO_PIECE || m.kind() == MK_EP || m.is_promo() {
            continue;
        }

        if write_nnue_sample(out, p, l.score, wdl_white).is_ok() {
            *samples += 1;
            taken += 1;
            last_ply = Some(ply);
            if *samples % 100_000 == 0 {
                let secs = start.elapsed().as_secs_f64().max(1e-9);
                eprintln!(
                    "nnue samples: {} ({:.0}/s)",
                    *samples,
                    *samples as f64 / secs
                );
            }
        }
    }
}

/// 104-byte NNUE record: STM-normalized bitboards + score + WDL, see header.
fn write_nnue_sample(
    out: &mut BufWriter<File>,
    pos: &Position,
    score_stm: i32,
    wdl_white: u8,
) -> std::io::Result<()> {
    let flip = matches!(pos.side, Color::Black);
    let (us, them) = (pos.side.idx(), pos.side.flip().idx());
    let mut buf = [0u8; 104];
    for p in 0..6 {
        let mover = if flip {
            pos.bb[us][p].swap_bytes()
        } else {
            pos.bb[us][p]
        };
        let opp = if flip {
            pos.bb[them][p].swap_bytes()
        } else {
            pos.bb[them][p]
        };
        buf[p * 8..p * 8 + 8].copy_from_slice(&mover.to_le_bytes());
        buf[48 + p * 8..48 + p * 8 + 8].copy_from_slice(&opp.to_le_bytes());
    }
    // search score is already from the side-to-move's perspective
    let sc = score_stm.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
    buf[96..98].copy_from_slice(&sc.to_le_bytes());
    // WDL flipped to the STM perspective for black-to-move samples
    buf[98] = if flip { 2 - wdl_white } else { wdl_white };
    // bytes 99-103 stay zero (padding)
    out.write_all(&buf)
}

// ---------------------------------------------------------------------------
// Opening book extraction (for SPRT test harness paired-openings)
// ---------------------------------------------------------------------------

const BOOK_MIN_ELO: u16 = 1400;
const BOOK_MIN_BASE_SECS: u32 = 180;

/// Build a diverse PGN opening book by extracting short, deduplicated
/// opening lines from real human games — self-contained (no external book
/// file needed), reusing the same validated SAN parser as the NNUE labeler.
/// A desynced (illegal) line is dropped entirely, same policy as elsewhere.
fn run_book(args: &[String]) {
    if args.len() < 6 {
        eprintln!(
            "usage: {} book <out.pgn> <max_openings> <min_ply> <max_ply> <pgn...>",
            args[0]
        );
        std::process::exit(1);
    }
    let out_path = &args[2];
    let max_openings: usize = args[3].parse().expect("max_openings");
    let min_ply: usize = args[4].parse().expect("min_ply");
    let max_ply: usize = args[5].parse().expect("max_ply");
    assert!(min_ply >= 2 && max_ply >= min_ply, "need 2 <= min_ply <= max_ply");

    let mut out = BufWriter::new(File::create(out_path).unwrap());
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut written = 0usize;
    let mut games = 0u64;

    'files: for path in &args[6..] {
        let f = File::open(path).unwrap_or_else(|e| panic!("open {}: {}", path, e));
        let mut r = BufReader::with_capacity(1 << 20, f);
        let mut line = String::new();
        let mut headers = Headers::default();
        let mut movetext = String::new();
        let mut in_moves = false;

        loop {
            line.clear();
            if r.read_line(&mut line).unwrap() == 0 {
                break;
            }
            let t = line.trim();
            if t.starts_with('[') {
                if in_moves {
                    try_extract_opening(
                        &headers, &movetext, min_ply, max_ply, &mut seen, &mut out,
                        &mut written,
                    );
                    games += 1;
                    headers = Headers::default();
                    movetext.clear();
                    in_moves = false;
                    if written >= max_openings {
                        break 'files;
                    }
                }
                parse_header(t, &mut headers);
            } else if !t.is_empty() {
                in_moves = true;
                movetext.push_str(t);
                movetext.push(' ');
            }
        }
        if in_moves {
            try_extract_opening(
                &headers, &movetext, min_ply, max_ply, &mut seen, &mut out, &mut written,
            );
            games += 1;
        }
        eprintln!("book: {} -> {} unique openings so far ({} games scanned)", path, written, games);
        if written >= max_openings {
            break;
        }
    }
    out.flush().unwrap();
    eprintln!("book: wrote {} unique openings to {}", written, out_path);
}

#[allow(clippy::too_many_arguments)]
fn try_extract_opening(
    h: &Headers,
    movetext: &str,
    min_ply: usize,
    max_ply: usize,
    seen: &mut std::collections::HashSet<String>,
    out: &mut BufWriter<File>,
    written: &mut usize,
) {
    let (we, be) = match (h.white_elo, h.black_elo) {
        (Some(w), Some(b)) => (w, b),
        _ => return,
    };
    if we < BOOK_MIN_ELO || be < BOOK_MIN_ELO {
        return;
    }
    if h.base_secs.map(|b| b < BOOK_MIN_BASE_SECS).unwrap_or(true) {
        return;
    }

    let mut pos = fen::startpos();
    let mut tokens: Vec<String> = Vec::new();
    for tok in movetext.split_whitespace() {
        if tokens.len() >= max_ply {
            break;
        }
        if tok.starts_with('$') || tok == "*" || tok == "1-0" || tok == "0-1" || tok == "1/2-1/2" {
            continue;
        }
        let san = tok.trim_start_matches(|c: char| c.is_ascii_digit() || c == '.');
        if san.is_empty() {
            continue;
        }
        let mv = match parse_san(&pos, san) {
            Some(m) => m,
            None => return, // desynced: drop this game's opening entirely
        };
        tokens.push(san.to_string());
        pos = pos.make(mv);
    }
    if tokens.len() < min_ply {
        return; // game itself too short to yield a usable opening
    }

    let key = tokens.join(" ");
    if !seen.insert(key) {
        return; // duplicate line, already have it
    }

    writeln!(out, "[Event \"Book\"]").unwrap();
    writeln!(out, "[Site \"Unchessed AI\"]").unwrap();
    writeln!(out, "[Result \"*\"]").unwrap();
    writeln!(out).unwrap();
    let mut line = String::new();
    for (i, tok) in tokens.iter().enumerate() {
        if i % 2 == 0 {
            line.push_str(&format!("{}. ", i / 2 + 1));
        }
        line.push_str(tok);
        line.push(' ');
    }
    line.push('*');
    writeln!(out, "{}", line).unwrap();
    writeln!(out).unwrap();
    *written += 1;
}
