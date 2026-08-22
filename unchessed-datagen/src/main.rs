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
//!
//! Aegis v4 mode emits schema-headed 1,088-byte human-policy records with the
//! full promotion-aware legal set, WDL/history/time metadata, and SipHash-2-4
//! game/player pseudonyms. It does not fabricate teacher regrets:
//!
//! Usage: unchessed-datagen policy-v4 <out_file> <128-bit-key-hex>
//!                               <max_positions> <accept_prob> <pgn...>

use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Seek, SeekFrom, Write};
use std::sync::atomic::AtomicBool;
use std::time::Instant;

use unchessed_core::board::*;
use unchessed_core::chessformer::{encode_policy_action, legal_policy_actions};
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
    buf[101] = if pos.ep == NO_EP {
        0xFF
    } else {
        file_of(pos.ep)
    };
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
    white_name: Option<String>,
    black_name: Option<String>,
    site: Option<String>,
    date: Option<String>,
    round: Option<String>,
    base_secs: Option<u32>,
    increment_secs: Option<u32>,
    result: Option<String>,
}

fn parse_header(line: &str, h: &mut Headers) {
    let inner = line.trim_start_matches('[').trim_end_matches(']');
    if let Some(rest) = inner.strip_prefix("WhiteElo ") {
        h.white_elo = rest.trim_matches('"').parse().ok();
    } else if let Some(rest) = inner.strip_prefix("BlackElo ") {
        h.black_elo = rest.trim_matches('"').parse().ok();
    } else if let Some(rest) = inner.strip_prefix("White ") {
        h.white_name = Some(rest.trim_matches('"').to_string());
    } else if let Some(rest) = inner.strip_prefix("Black ") {
        h.black_name = Some(rest.trim_matches('"').to_string());
    } else if let Some(rest) = inner.strip_prefix("Site ") {
        h.site = Some(rest.trim_matches('"').to_string());
    } else if let Some(rest) = inner.strip_prefix("Date ") {
        h.date = Some(rest.trim_matches('"').to_string());
    } else if let Some(rest) = inner.strip_prefix("Round ") {
        h.round = Some(rest.trim_matches('"').to_string());
    } else if let Some(rest) = inner.strip_prefix("TimeControl ") {
        let tc = rest.trim_matches('"');
        let mut fields = tc.split('+');
        h.base_secs = fields.next().and_then(|b| b.parse().ok());
        h.increment_secs = fields.next().and_then(|value| value.parse().ok());
    } else if let Some(rest) = inner.strip_prefix("Result ") {
        h.result = Some(rest.trim_matches('"').to_string());
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() >= 2 && args[1] == "policy-v4" {
        run_policy_v4(&args);
        return;
    }
    if args.len() >= 2 && args[1] == "nnue" {
        run_nnue(&args);
        return;
    }
    if args.len() >= 2 && args[1] == "book" {
        run_book(&args);
        return;
    }
    if args.len() < 5 {
        eprintln!(
            "usage: {} <out_dir> <cap_per_bucket> <accept_prob> <pgn...>",
            args[0]
        );
        eprintln!(
            "   or: {} policy-v4 <out_file> <128-bit-hash-key-hex> <max_positions> <accept_prob> <pgn...>",
            args[0]
        );
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
        .map(|i| BufWriter::new(File::create(format!("{}/bucket{}.bin", out_dir, i)).unwrap()))
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
                        &headers,
                        &movetext,
                        &mut writers,
                        &mut counts,
                        cap,
                        accept,
                        &mut rng,
                        &mut used_games,
                        &mut skipped_moves,
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
                &headers,
                &movetext,
                &mut writers,
                &mut counts,
                cap,
                accept,
                &mut rng,
                &mut used_games,
                &mut skipped_moves,
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
        eprintln!(
            "bucket{} ({}-{}): {} samples",
            i, BUCKETS[i].0, BUCKETS[i].1, c
        );
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
        let rating = if matches!(pos.side, Color::White) {
            we
        } else {
            be
        };
        let b = bucket_of(rating);
        // rare special-rule moves always pass the subsample gate
        let special = mv.kind() == MK_EP || mv.is_promo();
        if ply >= 2
            && counts[b] < cap
            && (special || rng.f64() < accept)
            && write_sample(&mut writers[b], &pos, mv, rating).is_ok()
        {
            counts[b] += 1;
        }
        pos = pos.make(mv);
        ply += 1;
    }
}

// ---------------------------------------------------------------------------
// Hydra Aegis v4 legal-policy data mode
// ---------------------------------------------------------------------------

const AEGIS_V4_HEADER_BYTES: usize = 64;
const AEGIS_V4_RECORD_BYTES: usize = 1088;
const AEGIS_V4_MAX_LEGAL: usize = 218;
const AEGIS_V4_SCHEMA_SHA256: [u8; 32] = [
    0x9b, 0x43, 0x3d, 0x91, 0xa8, 0x6f, 0x73, 0x52, 0xb4, 0x80, 0x75, 0x18, 0x5c, 0x75, 0xa7, 0xa9,
    0x57, 0xc0, 0xc2, 0xef, 0x8a, 0x57, 0x2c, 0xd8, 0x51, 0xe6, 0xc6, 0xf5, 0x59, 0x16, 0x72, 0x17,
];

fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = !0u32;
    for &byte in bytes {
        crc ^= byte as u32;
        for _ in 0..8 {
            crc = (crc >> 1) ^ (0xedb8_8320u32 & 0u32.wrapping_sub(crc & 1));
        }
    }
    !crc
}

fn aegis_v4_header(records: u64) -> [u8; AEGIS_V4_HEADER_BYTES] {
    let mut header = [0u8; AEGIS_V4_HEADER_BYTES];
    header[0..8].copy_from_slice(b"UNCHD4R0");
    header[8..10].copy_from_slice(&4u16.to_le_bytes());
    header[10..12].copy_from_slice(&(AEGIS_V4_HEADER_BYTES as u16).to_le_bytes());
    header[12..14].copy_from_slice(&(AEGIS_V4_RECORD_BYTES as u16).to_le_bytes());
    header[14..16].copy_from_slice(&0x00ffu16.to_le_bytes());
    header[16..20].copy_from_slice(&0x0102_0304u32.to_le_bytes());
    header[20..28].copy_from_slice(&records.to_le_bytes());
    header[28..60].copy_from_slice(&AEGIS_V4_SCHEMA_SHA256);
    let checksum = crc32(&header[..60]);
    header[60..64].copy_from_slice(&checksum.to_le_bytes());
    header
}

#[inline]
fn sip_round(mut v0: u64, mut v1: u64, mut v2: u64, mut v3: u64) -> (u64, u64, u64, u64) {
    v0 = v0.wrapping_add(v1);
    v1 = v1.rotate_left(13) ^ v0;
    v0 = v0.rotate_left(32);
    v2 = v2.wrapping_add(v3);
    v3 = v3.rotate_left(16) ^ v2;
    v0 = v0.wrapping_add(v3);
    v3 = v3.rotate_left(21) ^ v0;
    v2 = v2.wrapping_add(v1);
    v1 = v1.rotate_left(17) ^ v2;
    v2 = v2.rotate_left(32);
    (v0, v1, v2, v3)
}

fn siphash24(key: [u64; 2], bytes: &[u8]) -> u64 {
    let mut v0 = 0x736f_6d65_7073_6575 ^ key[0];
    let mut v1 = 0x646f_7261_6e64_6f6d ^ key[1];
    let mut v2 = 0x6c79_6765_6e65_7261 ^ key[0];
    let mut v3 = 0x7465_6462_7974_6573 ^ key[1];
    let mut chunks = bytes.chunks_exact(8);
    for chunk in &mut chunks {
        let m = u64::from_le_bytes(chunk.try_into().unwrap());
        v3 ^= m;
        (v0, v1, v2, v3) = sip_round(v0, v1, v2, v3);
        (v0, v1, v2, v3) = sip_round(v0, v1, v2, v3);
        v0 ^= m;
    }
    let mut tail = (bytes.len() as u64) << 56;
    for (shift, &byte) in chunks.remainder().iter().enumerate() {
        tail |= (byte as u64) << (8 * shift);
    }
    v3 ^= tail;
    (v0, v1, v2, v3) = sip_round(v0, v1, v2, v3);
    (v0, v1, v2, v3) = sip_round(v0, v1, v2, v3);
    v0 ^= tail;
    v2 ^= 0xff;
    for _ in 0..4 {
        (v0, v1, v2, v3) = sip_round(v0, v1, v2, v3);
    }
    v0 ^ v1 ^ v2 ^ v3
}

fn parse_hash_key(value: &str) -> Option<[u64; 2]> {
    if value.len() != 32 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return None;
    }
    let first = u64::from_str_radix(&value[..16], 16).ok()?;
    let second = u64::from_str_radix(&value[16..], 16).ok()?;
    Some([first, second])
}

fn nonzero_hash(key: [u64; 2], value: &str) -> u64 {
    siphash24(key, value.as_bytes()).max(1)
}

fn policy_time_class(base_secs: Option<u32>) -> u8 {
    match base_secs {
        Some(seconds) if seconds < 180 => 0,
        Some(seconds) if seconds < 600 => 1,
        Some(seconds) if seconds < 1800 => 2,
        Some(_) => 3,
        None => 4,
    }
}

fn run_policy_v4(args: &[String]) {
    if args.len() < 7 {
        eprintln!(
            "usage: {} policy-v4 <out_file> <128-bit-hash-key-hex> <max_positions> <accept_prob> <pgn...>",
            args[0]
        );
        std::process::exit(1);
    }
    let output_path = &args[2];
    let hash_key =
        parse_hash_key(&args[3]).expect("hash key must be exactly 32 hexadecimal digits");
    let maximum: u64 = args[4].parse().expect("max_positions");
    let accept: f64 = args[5].parse().expect("accept_prob");
    assert!(
        (0.0..=1.0).contains(&accept),
        "accept_prob must be in 0..=1"
    );
    let file =
        File::create(output_path).unwrap_or_else(|error| panic!("create {}: {error}", output_path));
    let mut output = BufWriter::new(file);
    output.write_all(&aegis_v4_header(0)).unwrap();
    let mut rng = Rng(0xa3e6_15d4_2026_0820);
    let mut records = 0u64;
    let mut games = 0u64;
    let mut skipped = 0u64;

    'files: for path in &args[6..] {
        let file = File::open(path).unwrap_or_else(|error| panic!("open {path}: {error}"));
        let mut reader = BufReader::with_capacity(1 << 20, file);
        let mut line = String::new();
        let mut headers = Headers::default();
        let mut movetext = String::new();
        let mut in_moves = false;
        loop {
            line.clear();
            if reader.read_line(&mut line).unwrap() == 0 {
                break;
            }
            let trimmed = line.trim();
            if trimmed.starts_with('[') {
                if in_moves {
                    process_policy_v4_game(
                        &headers,
                        &movetext,
                        hash_key,
                        accept,
                        maximum,
                        &mut rng,
                        &mut output,
                        &mut records,
                        &mut skipped,
                    );
                    games += 1;
                    headers = Headers::default();
                    movetext.clear();
                    in_moves = false;
                    if records >= maximum {
                        break 'files;
                    }
                }
                parse_header(trimmed, &mut headers);
            } else if !trimmed.is_empty() {
                in_moves = true;
                movetext.push_str(trimmed);
                movetext.push(' ');
            }
        }
        if in_moves && records < maximum {
            process_policy_v4_game(
                &headers,
                &movetext,
                hash_key,
                accept,
                maximum,
                &mut rng,
                &mut output,
                &mut records,
                &mut skipped,
            );
            games += 1;
        }
        eprintln!("policy-v4: {path}: {records} records from {games} games");
        if records >= maximum {
            break;
        }
    }
    output.flush().unwrap();
    output.seek(SeekFrom::Start(0)).unwrap();
    output.write_all(&aegis_v4_header(records)).unwrap();
    output.flush().unwrap();
    eprintln!(
        "policy-v4: wrote {records} records to {output_path}; {skipped} games skipped/desynced"
    );
}

fn canonical_game_movetext(movetext: &str) -> String {
    let mut output = Vec::new();
    let mut in_comment = false;
    for token in movetext.split_whitespace() {
        if in_comment {
            if token.ends_with('}') {
                in_comment = false;
            }
            continue;
        }
        if token.starts_with('{') {
            if !token.ends_with('}') {
                in_comment = true;
            }
            continue;
        }
        if token.starts_with('$') || matches!(token, "*" | "1-0" | "0-1" | "1/2-1/2") {
            continue;
        }
        let san = token
            .trim_start_matches(|character: char| character.is_ascii_digit() || character == '.');
        if !san.is_empty() {
            output.push(san);
        }
    }
    output.join(" ")
}

#[allow(clippy::too_many_arguments)]
fn process_policy_v4_game(
    headers: &Headers,
    movetext: &str,
    hash_key: [u64; 2],
    accept: f64,
    maximum: u64,
    rng: &mut Rng,
    output: &mut BufWriter<File>,
    records: &mut u64,
    skipped: &mut u64,
) {
    let (white_elo, black_elo, white_name, black_name) = match (
        headers.white_elo,
        headers.black_elo,
        headers.white_name.as_deref(),
        headers.black_name.as_deref(),
    ) {
        (Some(we), Some(be), Some(wn), Some(bn)) => (we, be, wn, bn),
        _ => return,
    };
    let wdl_white = match headers.result.as_deref() {
        Some("1-0") => 2,
        Some("1/2-1/2") => 1,
        Some("0-1") => 0,
        _ => return,
    };
    let white_identity = white_name.trim().to_lowercase();
    let black_identity = black_name.trim().to_lowercase();
    let game_identity = format!(
        "{}|{}|{}|{}|{}|{}|{}",
        headers.site.as_deref().unwrap_or("?").trim().to_lowercase(),
        headers.date.as_deref().unwrap_or("?"),
        headers.round.as_deref().unwrap_or("?"),
        white_identity,
        black_identity,
        headers.result.as_deref().unwrap_or("?"),
        canonical_game_movetext(movetext),
    );
    let game_hash = nonzero_hash(hash_key, &game_identity);
    let white_hash = nonzero_hash(hash_key, &white_identity);
    let black_hash = nonzero_hash(hash_key, &black_identity);
    let mut pos = fen::startpos();
    let mut history = Vec::<Move>::new();
    let mut in_comment = false;
    for token in movetext.split_whitespace() {
        if *records >= maximum {
            return;
        }
        if in_comment {
            if token.ends_with('}') {
                in_comment = false;
            }
            continue;
        }
        if token.starts_with('{') {
            if !token.ends_with('}') {
                in_comment = true;
            }
            continue;
        }
        if token.starts_with('$') || matches!(token, "*" | "1-0" | "0-1" | "1/2-1/2") {
            continue;
        }
        let san = token
            .trim_start_matches(|character: char| character.is_ascii_digit() || character == '.');
        if san.is_empty() {
            continue;
        }
        let mv = match parse_san(&pos, san) {
            Some(value) => value,
            None => {
                *skipped += 1;
                return;
            }
        };
        let special = mv.kind() == MK_EP || mv.is_promo();
        if history.len() >= 2 && (special || rng.f64() < accept) {
            let rating = if pos.side == Color::White {
                white_elo
            } else {
                black_elo
            };
            let player_hash = if pos.side == Color::White {
                white_hash
            } else {
                black_hash
            };
            if write_policy_v4_sample(
                output,
                &pos,
                mv,
                rating,
                if pos.side == Color::White {
                    wdl_white
                } else {
                    2 - wdl_white
                },
                policy_time_class(headers.base_secs),
                headers.increment_secs,
                &history,
                game_hash,
                player_hash,
            )
            .is_ok()
            {
                *records += 1;
            }
        }
        history.push(mv);
        pos = pos.make(mv);
    }
}

#[allow(clippy::too_many_arguments)]
fn write_policy_v4_sample(
    output: &mut BufWriter<File>,
    pos: &Position,
    selected: Move,
    rating: u16,
    wdl: u8,
    time_class: u8,
    increment_secs: Option<u32>,
    history: &[Move],
    game_hash: u64,
    player_hash: u64,
) -> std::io::Result<()> {
    let legal_actions = legal_policy_actions(pos);
    if legal_actions.overflowed()
        || legal_actions.is_empty()
        || legal_actions.len() > AEGIS_V4_MAX_LEGAL
    {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "legal policy action bound exceeded",
        ));
    }
    let selected_action = encode_policy_action(selected, pos.side);
    if legal_actions.find_action(selected_action) != Some(selected) {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "selected move absent from legal policy set",
        ));
    }
    let flip = pos.side == Color::Black;
    let (us, them) = (pos.side.idx(), pos.side.flip().idx());
    let mut record = [0u8; AEGIS_V4_RECORD_BYTES];
    for piece in 0..6 {
        let mover = if flip {
            pos.bb[us][piece].swap_bytes()
        } else {
            pos.bb[us][piece]
        };
        let opponent = if flip {
            pos.bb[them][piece].swap_bytes()
        } else {
            pos.bb[them][piece]
        };
        record[piece * 8..piece * 8 + 8].copy_from_slice(&mover.to_le_bytes());
        record[48 + piece * 8..48 + piece * 8 + 8].copy_from_slice(&opponent.to_le_bytes());
    }
    let normalized_move = selected_action & 0x0fff;
    record[96..98].copy_from_slice(&normalized_move.to_le_bytes());
    record[98] = (selected_action >> 12) as u8;
    record[99] = wdl;
    record[100..102].copy_from_slice(&rating.to_le_bytes());
    let (mk, mq, ok, oq) = if flip {
        (BK, BQ, WK, WQ)
    } else {
        (WK, WQ, BK, BQ)
    };
    record[102] = u8::from(pos.castling & mk != 0)
        | (u8::from(pos.castling & mq != 0) << 1)
        | (u8::from(pos.castling & ok != 0) << 2)
        | (u8::from(pos.castling & oq != 0) << 3);
    record[103] = if pos.ep == NO_EP {
        0xff
    } else {
        file_of(pos.ep)
    };
    record[104] = pos.halfmove.min(255) as u8;
    record[105] = time_class.min(4);
    let mut flags = 0u8;
    if selected.kind() == MK_CASTLE {
        flags |= 1;
    }
    if selected.kind() == MK_EP {
        flags |= 2;
    }
    if selected.is_promo() {
        flags |= 4;
    }
    let history_len = history.len().min(8);
    if history_len != 0 {
        flags |= 1 << 4;
    }
    if increment_secs.is_some() {
        flags |= 1 << 5;
    }
    record[106] = flags;
    record[107] = history_len as u8;
    for (slot, mv) in history.iter().rev().take(8).enumerate() {
        let from = if flip { mv.from() ^ 56 } else { mv.from() } as u16;
        let to = if flip { mv.to() ^ 56 } else { mv.to() } as u16;
        let normalized = from | (to << 6) | (mv.0 & 0xf000);
        record[108 + slot * 2..110 + slot * 2].copy_from_slice(&normalized.to_le_bytes());
    }
    record[124..132].copy_from_slice(&game_hash.to_le_bytes());
    record[132..140].copy_from_slice(&player_hash.to_le_bytes());
    record[148..150].copy_from_slice(&(history.len().min(u16::MAX as usize) as u16).to_le_bytes());
    record[150..154].copy_from_slice(&u32::MAX.to_le_bytes());
    record[154..158].copy_from_slice(
        &increment_secs
            .map(|seconds| seconds.saturating_mul(1000))
            .unwrap_or(u32::MAX)
            .to_le_bytes(),
    );
    record[160..162].copy_from_slice(&(legal_actions.len() as u16).to_le_bytes());
    record[162..164].copy_from_slice(&selected_action.to_le_bytes());
    record[164..166].copy_from_slice(&u16::MAX.to_le_bytes());
    // bytes 166/167: human policy kind and no teacher-regret flag.
    for slot in 0..AEGIS_V4_MAX_LEGAL {
        record[168 + slot * 2..170 + slot * 2].copy_from_slice(&u16::MAX.to_le_bytes());
        record[604 + slot * 2..606 + slot * 2].copy_from_slice(&i16::MAX.to_le_bytes());
    }
    for (slot, entry) in legal_actions.as_slice().iter().enumerate() {
        record[168 + slot * 2..170 + slot * 2].copy_from_slice(&entry.action.to_le_bytes());
    }
    output.write_all(&record)
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
                            &headers,
                            &movetext,
                            &mut out,
                            &tt,
                            &mut rng,
                            &mut samples,
                            max_positions,
                            &mut used_games,
                            &start,
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
                    &headers,
                    &movetext,
                    &mut out,
                    &tt,
                    &mut rng,
                    &mut samples,
                    max_positions,
                    &mut used_games,
                    &start,
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
            if (*samples).is_multiple_of(100_000) {
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
    assert!(
        min_ply >= 2 && max_ply >= min_ply,
        "need 2 <= min_ply <= max_ply"
    );

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
                        &headers,
                        &movetext,
                        min_ply,
                        max_ply,
                        &mut seen,
                        &mut out,
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
                &headers,
                &movetext,
                min_ply,
                max_ply,
                &mut seen,
                &mut out,
                &mut written,
            );
            games += 1;
        }
        eprintln!(
            "book: {} -> {} unique openings so far ({} games scanned)",
            path, written, games
        );
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn siphash_matches_reference_empty_message_vector() {
        let key = [0x0706_0504_0302_0100, 0x0f0e_0d0c_0b0a_0908];
        assert_eq!(siphash24(key, b""), 0x726f_db47_dd0e_0e31);
    }

    #[test]
    fn v4_header_has_frozen_width_and_crc() {
        let header = aegis_v4_header(37);
        assert_eq!(&header[..8], b"UNCHD4R0");
        assert_eq!(u16::from_le_bytes(header[12..14].try_into().unwrap()), 1088);
        assert_eq!(u64::from_le_bytes(header[20..28].try_into().unwrap()), 37);
        assert_eq!(
            u32::from_le_bytes(header[60..64].try_into().unwrap()),
            crc32(&header[..60])
        );
    }

    #[test]
    fn game_identity_movetext_is_stable_across_formatting_and_comments() {
        let a = "1. e4 {clock 10:00} e5 2. Nf3 Nc6 1-0";
        let b = "1.e4 e5\n2.Nf3 $1 Nc6 1-0";
        assert_eq!(canonical_game_movetext(a), "e4 e5 Nf3 Nc6");
        assert_eq!(canonical_game_movetext(a), canonical_game_movetext(b));
    }

    #[test]
    fn v4_writer_emits_complete_sorted_start_position_legal_set() {
        let path = std::env::temp_dir().join(format!(
            "unchessed-v4-record-{}-{}.bin",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        let file = File::create(&path).unwrap();
        let mut output = BufWriter::new(file);
        let pos = fen::startpos();
        let selected = unchessed_core::movegen::parse_uci_move(&pos, "e2e4").unwrap();
        write_policy_v4_sample(
            &mut output,
            &pos,
            selected,
            1500,
            1,
            2,
            Some(5),
            &[],
            11,
            12,
        )
        .unwrap();
        output.flush().unwrap();
        let bytes = fs::read(&path).unwrap();
        fs::remove_file(&path).ok();
        assert_eq!(bytes.len(), AEGIS_V4_RECORD_BYTES);
        let legal_count = u16::from_le_bytes(bytes[160..162].try_into().unwrap()) as usize;
        assert_eq!(legal_count, 20);
        let target = u16::from_le_bytes(bytes[162..164].try_into().unwrap());
        let actions: Vec<u16> = (0..legal_count)
            .map(|slot| {
                u16::from_le_bytes(bytes[168 + slot * 2..170 + slot * 2].try_into().unwrap())
            })
            .collect();
        assert!(actions.windows(2).all(|pair| pair[0] < pair[1]));
        assert!(actions.contains(&target));
        assert_eq!(
            u16::from_le_bytes(
                bytes[168 + legal_count * 2..170 + legal_count * 2]
                    .try_into()
                    .unwrap()
            ),
            u16::MAX
        );
    }
}
