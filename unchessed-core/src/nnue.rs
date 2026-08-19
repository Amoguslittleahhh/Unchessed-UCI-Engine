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
//! Inference is f32. v1 and v3 (the deployed default) update their
//! accumulators incrementally on each move via `Eval::update_state`
//! (add/remove only the feature rows a move actually changed; a full
//! rebuild is used only when a perspective's own king moves, since every
//! v3 feature is indexed relative to the king's bucket). v2 always takes
//! the full-recompute path -- it's kept loadable for reference only and
//! isn't worth the complexity for a scheme nothing ships as the default.
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
use crate::eval::{Eval, EvalState, NnueEvalState};

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

#[inline]
fn sub_row(acc: &mut [f32], ft_w: &[f32], idx: usize) {
    let row = &ft_w[idx * ACC..(idx + 1) * ACC];
    for (a, w) in acc.iter_mut().zip(row) {
        *a -= *w;
    }
}

/// Per-perspective context needed to compute a single piece's feature-row
/// index incrementally, without re-deriving it from scratch each time.
/// Valid only as long as this perspective's OWN king hasn't moved since it
/// was computed -- callers must full-rebuild instead when it has.
/// v3 (HalfKAv2_hm) only -- v2 (HalfKa) always takes the full-recompute
/// path in `update_state` instead of using this.
#[derive(Clone, Copy)]
struct KingFrame {
    bucket: usize,
    mirror: bool,
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

    /// King-relative context for perspective `persp`, read from `pos`. Only
    /// meaningful for schemes whose feature index depends on the king
    /// square (v2, v3) -- callers of `feature_row` for v1 ignore it.
    fn king_frame(&self, pos: &Position, persp: Color) -> KingFrame {
        let white_persp = matches!(persp, Color::White);
        let king_raw = pos.bb[persp.idx()][KING].trailing_zeros() as usize;
        let king_oriented = if white_persp { king_raw } else { king_raw ^ 56 };
        let mirror = (king_oriented % 8) < 4;
        let king_final = if mirror { king_oriented ^ 7 } else { king_oriented };
        KingFrame {
            bucket: KING_BUCKETS[king_final] as usize,
            mirror,
        }
    }

    /// Feature-row index for one (color, piece-type, square) contribution
    /// from perspective `persp`, given `persp`'s current (unmoved-king)
    /// `frame`. Mirrors `accumulate`'s per-scheme formulas exactly -- this
    /// and `accumulate` must never be allowed to drift apart, since that
    /// would silently desync incremental updates from a full recompute
    /// without either one crashing. Only called for Flat768/HalfKAv2_hm;
    /// v2 always takes the full-recompute path in `update_state` instead.
    fn feature_row(&self, persp: Color, frame: KingFrame, c: usize, p: usize, sq: usize) -> usize {
        let white_persp = matches!(persp, Color::White);
        let own = c == persp.idx();
        match self.scheme {
            Scheme::Flat768 => {
                let own_flag = if own { 0 } else { 1 };
                let s = if white_persp { sq } else { sq ^ 56 };
                own_flag * 384 + p * 64 + s
            }
            Scheme::HalfKav2Hm => {
                let s0 = if white_persp { sq } else { sq ^ 56 };
                let s = if frame.mirror { s0 ^ 7 } else { s0 };
                if p == KING {
                    // Both kings share the same merged block, keyed only by
                    // their own oriented/mirrored square -- see
                    // `accumulate`'s HalfKAv2_hm branch.
                    frame.bucket * N_PIECE_SQ_V3 + 640 + s
                } else {
                    let p_idx = p * 2 + usize::from(!own);
                    frame.bucket * N_PIECE_SQ_V3 + p_idx * 64 + s
                }
            }
            Scheme::HalfKa => unreachable!("v2 uses full recompute, not incremental feature_row"),
        }
    }

    /// Update `acc` in place for perspective `persp`, given the piece
    /// bitboards changed from `before` to `after`. Diffing the bitboards
    /// (rather than threading move-kind metadata through) means captures,
    /// en passant, castling, and promotion all fall out correctly for
    /// free: whatever squares actually gained or lost a piece get exactly
    /// one row added or removed, regardless of why.
    fn apply_diff(&self, before: &Position, after: &Position, persp: Color, acc: &mut [f32; ACC]) {
        let frame = self.king_frame(after, persp);
        for c in 0..2 {
            for p in 0..6 {
                let removed = before.bb[c][p] & !after.bb[c][p];
                let added = after.bb[c][p] & !before.bb[c][p];
                let mut bb = removed;
                while bb != 0 {
                    let sq = bb.trailing_zeros() as usize;
                    bb &= bb - 1;
                    sub_row(acc, &self.ft_w, self.feature_row(persp, frame, c, p, sq));
                }
                let mut bb = added;
                while bb != 0 {
                    let sq = bb.trailing_zeros() as usize;
                    bb &= bb - 1;
                    add_row(acc, &self.ft_w, self.feature_row(persp, frame, c, p, sq));
                }
            }
        }
    }

    fn combine(&self, acc_stm: &[f32], acc_nstm: &[f32]) -> i32 {
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
        self.combine(&acc_stm, &acc_nstm)
    }

    fn initial_state(&self, pos: &Position) -> EvalState {
        let mut acc = [[0.0f32; ACC]; 2];
        acc[Color::White.idx()].copy_from_slice(&self.accumulate(pos, Color::White));
        acc[Color::Black.idx()].copy_from_slice(&self.accumulate(pos, Color::Black));
        EvalState {
            nnue: NnueEvalState { acc },
        }
    }

    fn update_state(
        &self,
        before: &Position,
        after: &Position,
        _mv: Move,
        state: &EvalState,
    ) -> EvalState {
        if matches!(self.scheme, Scheme::HalfKa) {
            // v2 (HalfKA, SPRT-failed, kept loadable for reference only)
            // isn't worth incremental-update complexity for a scheme
            // nothing ships as the default -- full recompute is correct
            // and this path is never hot in practice.
            return self.initial_state(after);
        }
        let mut acc = state.nnue.acc;
        for &persp in &[Color::White, Color::Black] {
            let idx = persp.idx();
            let own_king_moved = before.bb[persp.idx()][KING] != after.bb[persp.idx()][KING];
            if matches!(self.scheme, Scheme::HalfKav2Hm) && own_king_moved {
                // Every one of this perspective's active features is
                // indexed relative to its own king's bucket, which just
                // changed -- only a full rebuild is correct here.
                acc[idx] = [0.0; ACC];
                acc[idx].copy_from_slice(&self.accumulate(after, persp));
                continue;
            }
            self.apply_diff(before, after, persp, &mut acc[idx]);
        }
        EvalState {
            nnue: NnueEvalState { acc },
        }
    }

    fn eval_with_state(&self, pos: &Position, state: &EvalState) -> i32 {
        let acc_stm = &state.nnue.acc[pos.side.idx()];
        let acc_nstm = &state.nnue.acc[pos.side.flip().idx()];
        self.combine(acc_stm, acc_nstm)
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

    /// Asserts two states agree to within float-summation-order noise --
    /// not bit-exact, since incremental add/remove and a from-scratch sum
    /// aren't required to hit IEEE-754 float addition in the same order.
    fn assert_states_close(a: &EvalState, b: &EvalState, tol: f32, ctx: &str) {
        for persp in 0..2 {
            for i in 0..ACC {
                let (x, y) = (a.nnue.acc[persp][i], b.nnue.acc[persp][i]);
                assert!(
                    (x - y).abs() <= tol,
                    "{}: perspective {} accumulator[{}] diverged: incremental {} vs full-refresh {}",
                    ctx,
                    persp,
                    i,
                    x,
                    y
                );
            }
        }
    }

    /// One (fen, uci move) pair, applied via update_state and compared
    /// against a from-scratch initial_state on the resulting position.
    fn check_incremental_step(net: &Nnue, fen: &str, uci: &str, ctx: &str) {
        let before = fen::parse(fen).unwrap();
        let mv = crate::movegen::parse_uci_move(&before, uci)
            .unwrap_or_else(|| panic!("{}: '{}' illegal in '{}'", ctx, uci, fen));
        let after = before.make(mv);
        let before_state = net.initial_state(&before);
        let incremental = net.update_state(&before, &after, mv, &before_state);
        let full_refresh = net.initial_state(&after);
        assert_states_close(&incremental, &full_refresh, 0.01, ctx);
    }

    /// CRITICAL correctness property: an incrementally-updated accumulator
    /// must match a full recompute on the resulting position, for every
    /// move-type special case (each of which changes feature rows for
    /// different, easy-to-get-wrong reasons -- a capture removes an extra
    /// row nowhere near the mover's destination, en passant removes a row
    /// at neither the origin nor the destination, promotion changes which
    /// piece-type row gets added, and castling moves two pieces at once
    /// while also (for the mover's own perspective) invalidating the
    /// entire king-bucket frame).
    #[test]
    fn incremental_accumulators_match_full_refresh_for_special_moves() {
        let cases: &[(&str, &str, &str)] = &[
            ("quiet move", fen::START_FEN, "e2e4"),
            (
                "capture",
                "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
                "f3e5",
            ),
            (
                "en passant",
                "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3",
                "e5d6",
            ),
            (
                "kingside castling",
                "rnbqk2r/pppp1ppp/5n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
                "e1g1",
            ),
            (
                "queenside castling",
                "r3k2r/pppqbppp/2np1n2/4p3/4P3/2NPBN2/PPPQ1PPP/R3K2R w KQkq - 6 7",
                "e1c1",
            ),
            ("promotion (quiet)", "8/4P1k1/8/8/8/8/6K1/8 w - - 0 1", "e7e8q"),
            (
                "promotion + capture",
                "4r2k/3P4/8/8/8/8/6K1/8 w - - 0 1",
                "d7e8q",
            ),
            (
                "king move (non-castling)",
                "8/8/8/4k3/8/4K3/8/8 w - - 0 1",
                "e3d3",
            ),
        ];
        for (tag, bytes) in [
            ("v1", dummy_net_bytes_v1()),
            ("v3", dummy_net_bytes_v3()),
        ] {
            let net = Nnue::from_bytes(&bytes).unwrap();
            for &(name, fen_str, uci) in cases {
                check_incremental_step(&net, fen_str, uci, &format!("[{}] {}", tag, name));
            }
        }
    }

    /// The same property, but over a longer, non-trivial move sequence
    /// (opening theory including a capture and castling both sides), state
    /// carried incrementally the whole way rather than rebuilt each step
    /// -- catches drift that a single-move test can't (e.g. an off-by-one
    /// in which frame a later move should be diffed against).
    #[test]
    fn incremental_accumulators_match_full_refresh_over_move_tree() {
        let moves = [
            "e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6", "e1g1", "f8e7",
            "f1e1", "b7b5", "a4b3", "d7d6", "c2c3", "e8g8", "h2h3", "c6a5", "b3c2", "c7c5",
        ];
        for (tag, bytes) in [
            ("v1", dummy_net_bytes_v1()),
            ("v3", dummy_net_bytes_v3()),
        ] {
            let net = Nnue::from_bytes(&bytes).unwrap();
            let mut pos = fen::startpos();
            let mut state = net.initial_state(&pos);
            for (i, uci) in moves.iter().enumerate() {
                let mv = crate::movegen::parse_uci_move(&pos, uci)
                    .unwrap_or_else(|| panic!("[{}] ply {}: '{}' illegal", tag, i, uci));
                let next = pos.make(mv);
                state = net.update_state(&pos, &next, mv, &state);
                let full_refresh = net.initial_state(&next);
                assert_states_close(
                    &state,
                    &full_refresh,
                    0.01,
                    &format!("[{}] after ply {} ({})", tag, i + 1, uci),
                );
                pos = next;
            }
        }
    }
}
