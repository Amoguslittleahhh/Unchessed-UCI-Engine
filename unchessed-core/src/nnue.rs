//! NNUE evaluation. Supports three weight-file versions, dispatched by the
//! version field in the header:
//!   v1: flat 768 = (own/opp) x (piece 0-5) x (square 0-63) features, 512-wide
//!       output layer (SCReLU of each perspective's 256-wide accumulator).
//!   v2: HalfKA king-relative features (45056 = king_sq x 11 non-own-king
//!       planes x 64 squares, no mirroring/bucketing), 1024-wide output layer
//!       (SCReLU + plain ClippedReLU, the SFNNv5 trick). SPRT-FAILED at
//!       -70.3 Elo vs v1 (decisive) -- kept loadable for reference/debugging
//!       only, not used as the default.
//!   v3: HalfKAv2_hm-style features -- 32-bucket horizontal king mirroring,
//!       own king included as an active feature (matching real Stockfish's
//!       PS_KING category), 22528 = 32 buckets x 11 piece types x 64
//!       squares. Same plain 512-wide SCReLU-only output layer as v1 (v2's
//!       SFNNv5 output-head change was deliberately dropped for v3 so this
//!       is a clean, single-variable test of the feature-scheme fix against
//!       v1, not another bundle of changes).
//! Inference is f32 with a full accumulator recompute per eval (no
//! incremental updates yet, any version).
//!
//! Weights file format "UNCHNNUE" (little-endian):
//!   magic "UNCHNNUE" (8 bytes)
//!   u32 version (1, 2, or 3), u32 ft_in, u32 acc (256)
//!   feature-transformer weights: [ft_in rows][256 f32] (one row per feature)
//!   feature-transformer bias: [256 f32]
//!   output weights: [mult*256 f32], mult=2 (v1, v3) or 4 (v2)
//!     v1/v3 order: [STM half, non-STM half]
//!     v2 order: [STM SCReLU, STM ClippedReLU, NSTM SCReLU, NSTM ClippedReLU]
//!   output bias: [1 f32]
//!
//! v1 feature index, from perspective P: a piece of color c, type p, on
//! square s maps to (own/opp)*384 + p*64 + sq, where own/opp is 0 if c == P
//! else 1, and sq is s for a white perspective, s^56 (vertical flip) for
//! black.
//!
//! v2 feature index, from perspective P: let king_sq be P's own king's
//! square in P's frame (s, or s^56 if P is black). A non-own-king piece of
//! color c, type p, on square s (frame-adjusted the same way) contributes
//! king_sq*704 + piece_idx*64 + sq, where piece_idx enumerates the 11
//! non-own-king (color, piece-type) pairs in order: own P,N,B,R,Q (0-4),
//! then opp P,N,B,R,Q,K (5-10). Own king is the anchor and never itself an
//! active feature.
//!
//! v3 feature index, from perspective P (matches nnue-pytorch's
//! HalfKAv2_hm^ exactly, ported from official-stockfish/nnue-pytorch's
//! model/modules/features/halfka_v2_hm.py and Stockfish's own
//! src/nnue/features/half_ka_v2_hm.h): compute own king square in P's frame
//! (as v2), then if that square is on files a-d, file-mirror the WHOLE
//! board (every square s -> s^7) so the king ends up on files e-h. Look up
//! the 32-way king bucket for the (now always e-h) king square via
//! KING_BUCKETS. Non-king pieces (5 own + 5 opp = 10 piece-color pairs)
//! contribute bucket*704 + p_idx*64 + sq, where p_idx = piece_type*2 +
//! (0 if own else 1) (own/opp interleaved per piece type, matching
//! Stockfish's PieceSquareIndex convention). Both kings share ONE 64-wide
//! block at p_idx=10 (features [640,703] within the bucket): the opponent
//! king contributes bucket*704+640+opp_king_sq, and the own king
//! ADDITIONALLY contributes bucket*704+640+own_king_sq (own_king_sq here is
//! exactly the same square used to select the bucket) -- these are always
//! two distinct rows since two kings can never occupy the same square.

use crate::board::*;
use crate::eval::Eval;

pub const ACC: usize = 256;
const FT_IN_V1: usize = 768;
const FT_IN_V2: usize = 64 * 11 * 64; // 45056
const N_PIECE_SQ_V2: usize = 11 * 64; // 704

const N_BUCKETS: usize = 32;
const N_PIECE_SQ_V3: usize = 11 * 64; // 704 (per bucket, export layout)
const FT_IN_V3: usize = N_BUCKETS * N_PIECE_SQ_V3; // 22528

const MAGIC: &[u8; 8] = b"UNCHNNUE";
const SCALE: f32 = 400.0;
const EVAL_CLAMP: i32 = 3000;

// 32 king buckets, identical to Stockfish's src/nnue/features/half_ka_v2_hm.h
// and nnue-pytorch's model/modules/features/halfka_v2_hm.py KingBuckets
// table. Indexed by the ORIENTED (post-mirror) king square 0-63; entries for
// files a-d are unreachable (-1) since mirroring guarantees the king never
// lands there after orientation.
#[rustfmt::skip]
const KING_BUCKETS: [i8; 64] = [
    -1, -1, -1, -1, 31, 30, 29, 28,
    -1, -1, -1, -1, 27, 26, 25, 24,
    -1, -1, -1, -1, 23, 22, 21, 20,
    -1, -1, -1, -1, 19, 18, 17, 16,
    -1, -1, -1, -1, 15, 14, 13, 12,
    -1, -1, -1, -1, 11, 10, 9, 8,
    -1, -1, -1, -1, 7, 6, 5, 4,
    -1, -1, -1, -1, 3, 2, 1, 0,
];

enum Scheme {
    Flat768,
    HalfKa,
    HalfKav2Hm,
}

pub struct Nnue {
    scheme: Scheme,
    /// [ft_in][ACC], row-major: one row of ACC weights per feature index
    ft_w: Vec<f32>,
    /// [ACC]
    ft_b: Vec<f32>,
    /// [mult * ACC]; mult=2 (v1, v3) or 4 (v2), see module doc for layout
    out_w: Vec<f32>,
    out_b: f32,
}

fn read_u32(buf: &[u8], off: &mut usize) -> Result<u32, String> {
    if *off + 4 > buf.len() {
        return Err("truncated UNCHNNUE file".into());
    }
    let v = u32::from_le_bytes(buf[*off..*off + 4].try_into().unwrap());
    *off += 4;
    Ok(v)
}

fn read_f32s(buf: &[u8], off: &mut usize, n: usize) -> Result<Vec<f32>, String> {
    if *off + n * 4 > buf.len() {
        return Err("truncated UNCHNNUE file".into());
    }
    let mut v = Vec::with_capacity(n);
    for i in 0..n {
        let s = *off + i * 4;
        v.push(f32::from_le_bytes(buf[s..s + 4].try_into().unwrap()));
    }
    *off += n * 4;
    Ok(v)
}

#[inline]
fn add_row(acc: &mut [f32], ft_w: &[f32], idx: usize) {
    let row = &ft_w[idx * ACC..(idx + 1) * ACC];
    for (a, w) in acc.iter_mut().zip(row) {
        *a += *w;
    }
}

impl Nnue {
    pub fn load(path: &str) -> Result<Nnue, String> {
        let buf = std::fs::read(path).map_err(|e| format!("open {}: {}", path, e))?;
        Nnue::from_bytes(&buf)
    }

    fn from_bytes(buf: &[u8]) -> Result<Nnue, String> {
        if buf.len() < 20 || &buf[0..8] != MAGIC {
            return Err("not an UNCHNNUE weights file".into());
        }
        let mut off = 8usize;
        let version = read_u32(buf, &mut off)?;
        let (scheme, expected_ft_in, mult) = match version {
            1 => (Scheme::Flat768, FT_IN_V1, 2usize),
            2 => (Scheme::HalfKa, FT_IN_V2, 4usize),
            3 => (Scheme::HalfKav2Hm, FT_IN_V3, 2usize),
            v => return Err(format!("unsupported UNCHNNUE version {}", v)),
        };
        let ft_in = read_u32(buf, &mut off)? as usize;
        let acc = read_u32(buf, &mut off)? as usize;
        if ft_in != expected_ft_in || acc != ACC {
            return Err(format!(
                "unsupported dims {}x{} (expected {}x{} for version {})",
                ft_in, acc, expected_ft_in, ACC, version
            ));
        }
        let expected = 20 + 4 * (ft_in * ACC + ACC + mult * ACC + 1);
        if buf.len() != expected {
            return Err(format!(
                "bad UNCHNNUE file size {} (expected {})",
                buf.len(),
                expected
            ));
        }
        let ft_w = read_f32s(buf, &mut off, ft_in * ACC)?;
        let ft_b = read_f32s(buf, &mut off, ACC)?;
        let out_w = read_f32s(buf, &mut off, mult * ACC)?;
        let out_b = read_f32s(buf, &mut off, 1)?[0];
        Ok(Nnue {
            scheme,
            ft_w,
            ft_b,
            out_w,
            out_b,
        })
    }

    /// Accumulator for one perspective: ft bias + sum of active feature rows.
    fn accumulate(&self, pos: &Position, persp: Color) -> Vec<f32> {
        let mut acc = self.ft_b.clone();
        let white_persp = matches!(persp, Color::White);
        match self.scheme {
            Scheme::Flat768 => {
                for c in 0..2 {
                    let own = if c == persp.idx() { 0 } else { 1 };
                    for p in 0..6 {
                        let mut bb = pos.bb[c][p];
                        while bb != 0 {
                            let s = bb.trailing_zeros() as usize;
                            bb &= bb - 1;
                            let sq = if white_persp { s } else { s ^ 56 };
                            let idx = own * 384 + p * 64 + sq;
                            add_row(&mut acc, &self.ft_w, idx);
                        }
                    }
                }
            }
            Scheme::HalfKa => {
                let own_c = persp.idx();
                let opp_c = 1 - own_c;
                let king_raw = pos.bb[own_c][KING].trailing_zeros() as usize;
                let king_sq = if white_persp { king_raw } else { king_raw ^ 56 };
                let mut piece_idx = 0usize;
                // own pieces first (skipping own king), then all opp pieces --
                // matches the trainer's KEEP_PLANES order [0..4, 6..11].
                for &(c, is_own) in &[(own_c, true), (opp_c, false)] {
                    for p in 0..6 {
                        if is_own && p == KING {
                            continue;
                        }
                        let mut bb = pos.bb[c][p];
                        while bb != 0 {
                            let s = bb.trailing_zeros() as usize;
                            bb &= bb - 1;
                            let sq = if white_persp { s } else { s ^ 56 };
                            let idx = king_sq * N_PIECE_SQ_V2 + piece_idx * 64 + sq;
                            add_row(&mut acc, &self.ft_w, idx);
                        }
                        piece_idx += 1;
                    }
                }
            }
            Scheme::HalfKav2Hm => {
                let own_c = persp.idx();
                let opp_c = 1 - own_c;
                let king_raw = pos.bb[own_c][KING].trailing_zeros() as usize;
                let king_oriented = if white_persp { king_raw } else { king_raw ^ 56 };
                let mirror = (king_oriented % 8) < 4;
                let king_final = if mirror { king_oriented ^ 7 } else { king_oriented };
                let bucket = KING_BUCKETS[king_final] as usize; // always valid post-mirror

                let orient_sq = |s: usize| -> usize {
                    let o = if white_persp { s } else { s ^ 56 };
                    if mirror { o ^ 7 } else { o }
                };

                // Non-king pieces: p_idx = piece_type*2 + (0 own, 1 opp),
                // matching Stockfish's own/opp-interleaved PieceSquareIndex.
                for p in 0..5 {
                    for &(c, is_own) in &[(own_c, true), (opp_c, false)] {
                        let mut bb = pos.bb[c][p];
                        while bb != 0 {
                            let s = bb.trailing_zeros() as usize;
                            bb &= bb - 1;
                            let sq = orient_sq(s);
                            let p_idx = p * 2 + usize::from(!is_own);
                            let idx = bucket * N_PIECE_SQ_V3 + p_idx * 64 + sq;
                            add_row(&mut acc, &self.ft_w, idx);
                        }
                    }
                }
                // Merged king block (p_idx=10, features [640,703] within the
                // bucket): opponent king at its own square, own king at its
                // own (already-computed) square -- always two distinct rows.
                let opp_king_raw = pos.bb[opp_c][KING].trailing_zeros() as usize;
                let opp_king_sq = orient_sq(opp_king_raw);
                add_row(&mut acc, &self.ft_w, bucket * N_PIECE_SQ_V3 + 640 + opp_king_sq);
                add_row(&mut acc, &self.ft_w, bucket * N_PIECE_SQ_V3 + 640 + king_final);
            }
        }
        acc
    }
}

/// SCReLU: clamp to [0, 1] then square.
#[inline]
fn screlu(x: f32) -> f32 {
    let v = x.clamp(0.0, 1.0);
    v * v
}

/// Plain ClippedReLU: clamp to [0, 1]. Used alongside SCReLU in v2 (SFNNv5
/// trick: concatenating both activation shapes of the same accumulator).
#[inline]
fn crelu(x: f32) -> f32 {
    x.clamp(0.0, 1.0)
}

impl Eval for Nnue {
    /// Centipawns from the side-to-move's perspective (negamax-ready).
    fn eval(&self, pos: &Position) -> i32 {
        let acc_stm = self.accumulate(pos, pos.side);
        let acc_nstm = self.accumulate(pos, pos.side.flip());
        let mut out = self.out_b;
        match self.scheme {
            Scheme::Flat768 | Scheme::HalfKav2Hm => {
                for i in 0..ACC {
                    out += screlu(acc_stm[i]) * self.out_w[i];
                    out += screlu(acc_nstm[i]) * self.out_w[ACC + i];
                }
            }
            Scheme::HalfKa => {
                for i in 0..ACC {
                    out += screlu(acc_stm[i]) * self.out_w[i];
                    out += crelu(acc_stm[i]) * self.out_w[ACC + i];
                    out += screlu(acc_nstm[i]) * self.out_w[2 * ACC + i];
                    out += crelu(acc_nstm[i]) * self.out_w[3 * ACC + i];
                }
            }
        }
        ((out * SCALE) as i32).clamp(-EVAL_CLAMP, EVAL_CLAMP)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fen;

    fn xorshift_weights(n: usize, state0: u64) -> Vec<f32> {
        let mut state = state0;
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let unit = (state >> 11) as f64 / (1u64 << 53) as f64;
            out.push((unit as f32) * 0.2 - 0.1); // uniform in [-0.1, 0.1)
        }
        out
    }

    /// Build a deterministic dummy v1 network file image (xorshift weights).
    fn dummy_net_bytes_v1() -> Vec<u8> {
        let n_weights = FT_IN_V1 * ACC + ACC + 2 * ACC + 1;
        let mut buf = Vec::with_capacity(20 + 4 * n_weights);
        buf.extend_from_slice(MAGIC);
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&(FT_IN_V1 as u32).to_le_bytes());
        buf.extend_from_slice(&(ACC as u32).to_le_bytes());
        for w in xorshift_weights(n_weights, 0x1234_5678_9ABC_DEF0) {
            buf.extend_from_slice(&w.to_le_bytes());
        }
        buf
    }

    /// Build a deterministic dummy v2 (HalfKA) network file image.
    fn dummy_net_bytes_v2() -> Vec<u8> {
        let n_weights = FT_IN_V2 * ACC + ACC + 4 * ACC + 1;
        let mut buf = Vec::with_capacity(20 + 4 * n_weights);
        buf.extend_from_slice(MAGIC);
        buf.extend_from_slice(&2u32.to_le_bytes());
        buf.extend_from_slice(&(FT_IN_V2 as u32).to_le_bytes());
        buf.extend_from_slice(&(ACC as u32).to_le_bytes());
        for w in xorshift_weights(n_weights, 0x0FED_CBA9_8765_4321) {
            buf.extend_from_slice(&w.to_le_bytes());
        }
        buf
    }

    /// Build a deterministic dummy v3 (HalfKAv2_hm) network file image.
    fn dummy_net_bytes_v3() -> Vec<u8> {
        let n_weights = FT_IN_V3 * ACC + ACC + 2 * ACC + 1;
        let mut buf = Vec::with_capacity(20 + 4 * n_weights);
        buf.extend_from_slice(MAGIC);
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&(FT_IN_V3 as u32).to_le_bytes());
        buf.extend_from_slice(&(ACC as u32).to_le_bytes());
        for w in xorshift_weights(n_weights, 0x9E37_79B9_7F4A_7C15) {
            buf.extend_from_slice(&w.to_le_bytes());
        }
        buf
    }

    /// Write the dummy net to a uniquely named temp file and Nnue::load it.
    fn load_dummy(tag: &str, bytes: Vec<u8>) -> Nnue {
        let path = std::env::temp_dir().join(format!(
            "unchessed_nnue_test_{}_{}.bin",
            std::process::id(),
            tag
        ));
        std::fs::write(&path, bytes).unwrap();
        let net = Nnue::load(path.to_str().unwrap()).unwrap();
        let _ = std::fs::remove_file(&path);
        net
    }

    /// Color-mirror a FEN: piece colors swapped, board flipped vertically
    /// (square ^56), side flipped, castling rights swapped, ep mirrored.
    fn color_mirror_fen(f: &str) -> String {
        let parts: Vec<&str> = f.split_whitespace().collect();
        let swap = |c: char| -> char {
            if c.is_ascii_uppercase() {
                c.to_ascii_lowercase()
            } else if c.is_ascii_lowercase() {
                c.to_ascii_uppercase()
            } else {
                c
            }
        };
        let placement: Vec<String> = parts[0]
            .split('/')
            .rev()
            .map(|rank| rank.chars().map(swap).collect())
            .collect();
        let side = if parts[1] == "w" { "b" } else { "w" };
        let castling = if parts[2] == "-" {
            "-".to_string()
        } else {
            let swapped: Vec<char> = parts[2].chars().map(swap).collect();
            let mut out = String::new();
            for want in ['K', 'Q', 'k', 'q'] {
                if swapped.contains(&want) {
                    out.push(want);
                }
            }
            out
        };
        let ep = if parts[3] == "-" {
            "-".to_string()
        } else {
            let b = parts[3].as_bytes();
            let rank = b[1] - b'0';
            format!("{}{}", b[0] as char, 9 - rank)
        };
        format!(
            "{} {} {} {} {} {}",
            placement.join("/"),
            side,
            castling,
            ep,
            parts.get(4).unwrap_or(&"0"),
            parts.get(5).unwrap_or(&"1"),
        )
    }

    const MIRROR_FENS: [&str; 5] = [
        // startpos (symmetric baseline)
        fen::START_FEN,
        // asymmetric middlegame, black castled short, white king in the center
        "r2q1rk1/pp2ppbp/2np1np1/8/3NP3/2N1B3/PPPQ1PPP/R3KB1R w KQ - 5 9",
        // both castled, opposite sides (white short, black long)
        "2kr3r/ppp2ppp/2nqbn2/3pp3/8/2N1PN2/PPPPBPPP/R1BQ1RK1 w - - 6 9",
        // en-passant square set, black to move next after mirror
        "rnbqkb1r/pp1p1ppp/5n2/2pPp3/8/8/PPP1PPPP/RNBQKBNR w KQkq e6 0 4",
        // lopsided material, black to move
        "r4rk1/1pp2ppp/p1np1n2/8/2B1P3/2N5/PPP2PPP/2KR3R b - - 0 12",
    ];

    /// CRITICAL correctness property, checked for ALL weight-file versions.
    /// The eval is STM-relative, and the color-mirror swaps both the pieces
    /// AND the side to move, so the mover faces the identical relative
    /// position: the two evals must be EQUAL. (They would be negations of
    /// each other only for a white-relative eval, or for a mirror that did
    /// not flip the side to move.) This is the standard
    /// `eval(pos) == eval(colorflip(pos))` NNUE feature-index sanity check;
    /// a tolerance of 1 cp absorbs f32 summation-order noise.
    #[test]
    fn color_mirror_symmetry_v1() {
        let net = load_dummy("mirror_v1", dummy_net_bytes_v1());
        for f in MIRROR_FENS {
            let pos = fen::parse(f).unwrap();
            let mir = fen::parse(&color_mirror_fen(f)).unwrap();
            let e1 = net.eval(&pos);
            let e2 = net.eval(&mir);
            assert!((e1 - e2).abs() <= 1, "mirror mismatch for {}: {} vs {}", f, e1, e2);
        }
    }

    #[test]
    fn color_mirror_symmetry_v2() {
        let net = load_dummy("mirror_v2", dummy_net_bytes_v2());
        for f in MIRROR_FENS {
            let pos = fen::parse(f).unwrap();
            let mir = fen::parse(&color_mirror_fen(f)).unwrap();
            let e1 = net.eval(&pos);
            let e2 = net.eval(&mir);
            assert!((e1 - e2).abs() <= 1, "mirror mismatch for {}: {} vs {}", f, e1, e2);
        }
    }

    #[test]
    fn color_mirror_symmetry_v3() {
        let net = load_dummy("mirror_v3", dummy_net_bytes_v3());
        for f in MIRROR_FENS {
            let pos = fen::parse(f).unwrap();
            let mir = fen::parse(&color_mirror_fen(f)).unwrap();
            let e1 = net.eval(&pos);
            let e2 = net.eval(&mir);
            assert!((e1 - e2).abs() <= 1, "mirror mismatch for {}: {} vs {}", f, e1, e2);
        }
    }

    #[test]
    fn eval_is_not_degenerate() {
        for (tag, bytes) in [
            ("v1", dummy_net_bytes_v1()),
            ("v2", dummy_net_bytes_v2()),
            ("v3", dummy_net_bytes_v3()),
        ] {
            let net = Nnue::from_bytes(&bytes).unwrap();
            let a = net.eval(
                &fen::parse("r2q1rk1/pp2ppbp/2np1np1/8/3NP3/2N1B3/PPPQ1PPP/R3KB1R w KQ - 5 9")
                    .unwrap(),
            );
            let b = net.eval(&fen::startpos());
            assert_ne!(a, b, "[{}] dummy net evals should differ across positions", tag);
            assert!(a.abs() <= EVAL_CLAMP && b.abs() <= EVAL_CLAMP);
        }
    }

    #[test]
    fn load_rejects_garbage() {
        assert!(Nnue::from_bytes(b"NOTANNUE").is_err());
        assert!(Nnue::from_bytes(b"UNCHNNUE").is_err()); // truncated header
        // right magic, wrong dims
        let mut buf = Vec::new();
        buf.extend_from_slice(MAGIC);
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&512u32.to_le_bytes());
        buf.extend_from_slice(&(ACC as u32).to_le_bytes());
        assert!(Nnue::from_bytes(&buf).is_err());
        // right header, wrong body length
        let mut buf = Vec::new();
        buf.extend_from_slice(MAGIC);
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&(FT_IN_V1 as u32).to_le_bytes());
        buf.extend_from_slice(&(ACC as u32).to_le_bytes());
        buf.extend_from_slice(&[0u8; 64]);
        assert!(Nnue::from_bytes(&buf).is_err());
        // unsupported version
        let mut buf = Vec::new();
        buf.extend_from_slice(MAGIC);
        buf.extend_from_slice(&4u32.to_le_bytes());
        buf.extend_from_slice(&(FT_IN_V1 as u32).to_le_bytes());
        buf.extend_from_slice(&(ACC as u32).to_le_bytes());
        assert!(Nnue::from_bytes(&buf).is_err());
    }
}
