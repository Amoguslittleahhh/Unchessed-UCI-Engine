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
use crate::cpu;
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
    /// [out_buckets * mult * ACC]; mult=2 (v1, v3) or 4 (v2), see module
    /// doc for layout. v4 (and only v4) has 8 piece-count output buckets:
    /// `(popcount(occupied) - 1) / 4`, Stockfish's standard output-head
    /// bucketing (docs/research-notes-moe-2507.11181.md: the fixed
    /// hand-specified router that keeps MoE's benefit without its failure
    /// modes).
    out_w: Vec<f32>,
    /// [out_buckets]; v1-v3 files have a single bias.
    out_b: Vec<f32>,
    /// 1 for v1-v3, 8 for v4.
    out_buckets: usize,
}

/// Piece-count output bucket (v4): `(pieces - 1) / 4`, 0..=7. Matches
/// Stockfish's half-ka output-bucket selection; a legal position has 2..=32
/// pieces, so the clamp is belt-and-braces, not a reachable correction.
#[inline]
fn output_bucket(pieces: usize) -> usize {
    ((pieces - 1) / 4).min(7)
}

#[inline]
fn occupied_count(pos: &Position) -> usize {
    let mut occ: u64 = 0;
    for c in 0..2 {
        for p in 0..6 {
            occ |= pos.bb[c][p];
        }
    }
    occ.count_ones() as usize
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

// ---------------------------------------------------------------------------
// SIMD kernels
//
// The accumulator update (`add_row`/`sub_row`) and the output layer
// (`combine`) are the two hot spots: the former runs on every make, the
// latter on essentially every search node. Both are pure elementwise
// float work, so AVX2+FMA vectorizes them exactly -- the only difference
// from the scalar path is float summation order in `combine`'s reduction,
// which is why the existing 1cp-tolerance parity tests still gate this.
//
// Dispatch is resolved once into a `bool` at load time rather than per
// call, and the scalar paths below remain the behavioral reference (and
// the only path on non-x86).
// ---------------------------------------------------------------------------

#[inline]
fn add_row_scalar(acc: &mut [f32], row: &[f32]) {
    for (a, w) in acc.iter_mut().zip(row) {
        *a += *w;
    }
}

#[inline]
fn sub_row_scalar(acc: &mut [f32], row: &[f32]) {
    for (a, w) in acc.iter_mut().zip(row) {
        *a -= *w;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn add_row_avx2(acc: &mut [f32], row: &[f32]) {
    use std::arch::x86_64::*;
    let n = acc.len().min(row.len());
    let a = acc.as_mut_ptr();
    let w = row.as_ptr();
    let mut i = 0usize;
    while i + 8 <= n {
        let v = _mm256_add_ps(_mm256_loadu_ps(a.add(i)), _mm256_loadu_ps(w.add(i)));
        _mm256_storeu_ps(a.add(i), v);
        i += 8;
    }
    while i < n {
        *a.add(i) += *w.add(i);
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn sub_row_avx2(acc: &mut [f32], row: &[f32]) {
    use std::arch::x86_64::*;
    let n = acc.len().min(row.len());
    let a = acc.as_mut_ptr();
    let w = row.as_ptr();
    let mut i = 0usize;
    while i + 8 <= n {
        let v = _mm256_sub_ps(_mm256_loadu_ps(a.add(i)), _mm256_loadu_ps(w.add(i)));
        _mm256_storeu_ps(a.add(i), v);
        i += 8;
    }
    while i < n {
        *a.add(i) -= *w.add(i);
        i += 1;
    }
}

#[inline]
fn add_row(acc: &mut [f32], ft_w: &[f32], idx: usize) {
    let row = &ft_w[idx * ACC..(idx + 1) * ACC];
    #[cfg(target_arch = "x86_64")]
    if cpu::has_avx2() {
        // SAFETY: guarded by a runtime AVX2 check; the kernel only
        // touches `acc`/`row` within their shared length.
        unsafe {
            add_row_avx2(acc, row);
        }
        return;
    }
    add_row_scalar(acc, row);
}

#[inline]
fn sub_row(acc: &mut [f32], ft_w: &[f32], idx: usize) {
    let row = &ft_w[idx * ACC..(idx + 1) * ACC];
    #[cfg(target_arch = "x86_64")]
    if cpu::has_avx2() {
        // SAFETY: as above.
        unsafe {
            sub_row_avx2(acc, row);
        }
        return;
    }
    sub_row_scalar(acc, row);
}

/// Horizontal sum of an AVX2 register.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn hsum256(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    let lo = _mm256_castps256_ps128(v);
    let hi = _mm256_extractf128_ps(v, 1);
    let s = _mm_add_ps(lo, hi);
    let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    let s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 0x55));
    _mm_cvtss_f32(s)
}

/// `sum(screlu(acc[i]) * w[i])` over `ACC` lanes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn screlu_dot_avx2(acc: &[f32], w: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let zero = _mm256_setzero_ps();
    let one = _mm256_set1_ps(1.0);
    let mut sum = _mm256_setzero_ps();
    let a = acc.as_ptr();
    let p = w.as_ptr();
    let n = acc.len().min(w.len());
    let mut i = 0usize;
    while i + 8 <= n {
        let v = _mm256_loadu_ps(a.add(i));
        let c = _mm256_min_ps(_mm256_max_ps(v, zero), one);
        sum = _mm256_fmadd_ps(_mm256_mul_ps(c, c), _mm256_loadu_ps(p.add(i)), sum);
        i += 8;
    }
    let mut out = hsum256(sum);
    while i < n {
        out += screlu(*a.add(i)) * *p.add(i);
        i += 1;
    }
    out
}

/// `sum(crelu(acc[i]) * w[i])` over `ACC` lanes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn crelu_dot_avx2(acc: &[f32], w: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let zero = _mm256_setzero_ps();
    let one = _mm256_set1_ps(1.0);
    let mut sum = _mm256_setzero_ps();
    let a = acc.as_ptr();
    let p = w.as_ptr();
    let n = acc.len().min(w.len());
    let mut i = 0usize;
    while i + 8 <= n {
        let v = _mm256_loadu_ps(a.add(i));
        let c = _mm256_min_ps(_mm256_max_ps(v, zero), one);
        sum = _mm256_fmadd_ps(c, _mm256_loadu_ps(p.add(i)), sum);
        i += 8;
    }
    let mut out = hsum256(sum);
    while i < n {
        out += crelu(*a.add(i)) * *p.add(i);
        i += 1;
    }
    out
}

#[inline]
fn screlu_dot(acc: &[f32], w: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if cpu::has_avx2_fma() {
        // SAFETY: guarded by a runtime AVX2+FMA check.
        return unsafe { screlu_dot_avx2(acc, w) };
    }
    let n = acc.len().min(w.len());
    let mut out = 0.0;
    for i in 0..n {
        out += screlu(acc[i]) * w[i];
    }
    out
}

#[inline]
fn crelu_dot(acc: &[f32], w: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if cpu::has_avx2_fma() {
        // SAFETY: guarded by a runtime AVX2+FMA check.
        return unsafe { crelu_dot_avx2(acc, w) };
    }
    let n = acc.len().min(w.len());
    let mut out = 0.0;
    for i in 0..n {
        out += crelu(acc[i]) * w[i];
    }
    out
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
        let (scheme, expected_ft_in, mult, out_buckets) = match version {
            1 => (Scheme::Flat768, FT_IN_V1, 2usize, 1usize),
            2 => (Scheme::HalfKa, FT_IN_V2, 4usize, 1usize),
            3 => (Scheme::HalfKav2Hm, FT_IN_V3, 2usize, 1usize),
            4 => (Scheme::HalfKav2Hm, FT_IN_V3, 2usize, 8usize),
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
        let expected =
            20 + 4 * (ft_in * ACC + ACC + out_buckets * mult * ACC + out_buckets);
        if buf.len() != expected {
            return Err(format!(
                "bad UNCHNNUE file size {} (expected {})",
                buf.len(),
                expected
            ));
        }
        let ft_w = read_f32s(buf, &mut off, ft_in * ACC)?;
        let ft_b = read_f32s(buf, &mut off, ACC)?;
        let out_w = read_f32s(buf, &mut off, out_buckets * mult * ACC)?;
        let out_b = read_f32s(buf, &mut off, out_buckets)?;
        Ok(Nnue {
            scheme,
            ft_w,
            ft_b,
            out_w,
            out_b,
            out_buckets,
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

    fn combine(&self, acc_stm: &[f32], acc_nstm: &[f32], pieces: usize) -> i32 {
        let bucket = if self.out_buckets == 1 {
            0
        } else {
            output_bucket(pieces)
        };
        let mult = if matches!(self.scheme, Scheme::HalfKa) { 4 } else { 2 };
        let base = bucket * mult * ACC;
        let mut out = self.out_b[bucket];
        match self.scheme {
            Scheme::Flat768 | Scheme::HalfKav2Hm => {
                out += screlu_dot(&acc_stm[..ACC], &self.out_w[base..base + ACC]);
                out += screlu_dot(&acc_nstm[..ACC], &self.out_w[base + ACC..base + 2 * ACC]);
            }
            Scheme::HalfKa => {
                out += screlu_dot(&acc_stm[..ACC], &self.out_w[base..base + ACC]);
                out += crelu_dot(&acc_stm[..ACC], &self.out_w[base + ACC..base + 2 * ACC]);
                out += screlu_dot(&acc_nstm[..ACC], &self.out_w[base + 2 * ACC..base + 3 * ACC]);
                out += crelu_dot(&acc_nstm[..ACC], &self.out_w[base + 3 * ACC..base + 4 * ACC]);
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
        self.combine(&acc_stm, &acc_nstm, occupied_count(pos))
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
        // The piece count (and hence the v4 bucket) changes only on
        // captures; it is derived from `pos` here, so `update_state`
        // needs no bucket bookkeeping.
        self.combine(acc_stm, acc_nstm, occupied_count(pos))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fen;

    /// `KING_BUCKETS` must match Stockfish's `half_ka_v2_hm.h` exactly.
    ///
    /// This table decides which of the 32 buckets every feature index lands
    /// in, so a single wrong entry silently routes a king square to the wrong
    /// 704-feature block. The net still loads, still evaluates, and still
    /// passes the existing colour-mirror tests -- it is just quietly reading
    /// the wrong weights for some king positions. That is exactly the kind of
    /// bug worth a cheap structural test.
    ///
    /// The expected values are transcribed from Stockfish's own source. Our
    /// convention differs in one respect: the board is horizontally mirrored
    /// before lookup so the king is always on files e-h, which means only
    /// those 32 entries are ever read and the a-d half is `-1`. Stockfish
    /// stores a full, horizontally symmetric table instead. Comparing only
    /// the e-h half is therefore the correct comparison, not a weakening of
    /// it -- and `king_buckets_cover_only_the_mirrored_half` pins the rest.
    #[test]
    fn king_buckets_match_stockfish_half_ka_v2_hm() {
        // Stockfish src/nnue/features/half_ka_v2_hm.h, KingBuckets, a1..h8.
        #[rustfmt::skip]
        const STOCKFISH: [i8; 64] = [
            28, 29, 30, 31, 31, 30, 29, 28,
            24, 25, 26, 27, 27, 26, 25, 24,
            20, 21, 22, 23, 23, 22, 21, 20,
            16, 17, 18, 19, 19, 18, 17, 16,
            12, 13, 14, 15, 15, 14, 13, 12,
             8,  9, 10, 11, 11, 10,  9,  8,
             4,  5,  6,  7,  7,  6,  5,  4,
             0,  1,  2,  3,  3,  2,  1,  0,
        ];

        // Sanity-check the reference itself before trusting it: Stockfish's
        // table must be horizontally symmetric, which is the property that
        // makes mirroring to one half lossless.
        for sq in 0..64usize {
            let mirrored = (sq / 8) * 8 + (7 - sq % 8);
            assert_eq!(
                STOCKFISH[sq], STOCKFISH[mirrored],
                "reference table is not horizontally symmetric at {sq}"
            );
        }

        for sq in 0..64usize {
            if sq % 8 >= 4 {
                assert_eq!(
                    KING_BUCKETS[sq], STOCKFISH[sq],
                    "bucket mismatch at square {sq} (file {})",
                    sq % 8
                );
            }
        }
    }

    /// The mirrored half must be a clean bijection onto 0..32.
    ///
    /// Two independent failure modes this catches: a duplicated bucket (two
    /// king squares sharing one weight block, halving effective capacity for
    /// those squares) and an out-of-range index (which would read past the
    /// intended block).
    #[test]
    fn king_buckets_cover_only_the_mirrored_half() {
        let mut seen = [false; N_BUCKETS];
        let mut valid = 0;
        for sq in 0..64usize {
            let bucket = KING_BUCKETS[sq];
            if sq % 8 < 4 {
                assert_eq!(bucket, -1, "file a-d square {sq} must be unreachable");
                continue;
            }
            assert!(bucket >= 0, "file e-h square {sq} must have a bucket");
            let bucket = bucket as usize;
            assert!(bucket < N_BUCKETS, "bucket {bucket} out of range at {sq}");
            assert!(!seen[bucket], "bucket {bucket} used twice, at square {sq}");
            seen[bucket] = true;
            valid += 1;
        }
        assert_eq!(valid, N_BUCKETS, "expected exactly {N_BUCKETS} live squares");
        assert!(seen.iter().all(|&b| b), "some bucket is never produced");
    }

    /// The v3 feature layout must be internally consistent.
    ///
    /// `FT_IN_V3` is validated against the file header on load, so if these
    /// constants drifted apart the net would fail to load rather than
    /// misbehave -- but the arithmetic is worth pinning next to the table it
    /// depends on.
    #[test]
    fn v3_feature_dimensions_are_consistent() {
        assert_eq!(N_PIECE_SQ_V3, 11 * 64);
        assert_eq!(FT_IN_V3, N_BUCKETS * N_PIECE_SQ_V3);
        assert_eq!(FT_IN_V3, 22528);
    }

    /// The SIMD kernels must agree with the scalar reference they replaced.
    ///
    /// `add_row`/`sub_row` are elementwise and must be *bit-exact*.
    /// `screlu_dot`/`crelu_dot` reduce, so they may differ from the scalar
    /// sum only by float accumulation order -- bounded here well below the
    /// 1cp tolerance the eval parity tests already allow.
    #[test]
    fn simd_kernels_match_scalar_reference() {
        let acc_a = xorshift_weights(ACC, 0x1234_5678_9abc_def1);
        let acc_b = xorshift_weights(ACC, 0x0fed_cba9_8765_4321);
        let weights = xorshift_weights(4 * ACC, 0xdead_beef_cafe_0001);

        // Elementwise add/sub must be exact.
        let mut simd_acc = acc_a.clone();
        let mut scalar_acc = acc_a.clone();
        let row = &weights[..ACC];
        add_row_scalar(&mut scalar_acc, row);
        {
            let mut tmp = simd_acc.clone();
            #[cfg(target_arch = "x86_64")]
            if cpu::has_avx2() {
                unsafe { add_row_avx2(&mut tmp, row) };
            } else {
                add_row_scalar(&mut tmp, row);
            }
            #[cfg(not(target_arch = "x86_64"))]
            add_row_scalar(&mut tmp, row);
            simd_acc = tmp;
        }
        for i in 0..ACC {
            assert_eq!(
                simd_acc[i].to_bits(),
                scalar_acc[i].to_bits(),
                "add_row lane {i} must be bit-exact"
            );
        }
        sub_row_scalar(&mut scalar_acc, row);
        {
            let mut tmp = simd_acc.clone();
            #[cfg(target_arch = "x86_64")]
            if cpu::has_avx2() {
                unsafe { sub_row_avx2(&mut tmp, row) };
            } else {
                sub_row_scalar(&mut tmp, row);
            }
            #[cfg(not(target_arch = "x86_64"))]
            sub_row_scalar(&mut tmp, row);
            simd_acc = tmp;
        }
        for i in 0..ACC {
            assert_eq!(
                simd_acc[i].to_bits(),
                scalar_acc[i].to_bits(),
                "sub_row lane {i} must be bit-exact"
            );
        }

        // Reductions: allow only summation-order noise.
        let scalar_screlu: f32 = (0..ACC).map(|i| screlu(acc_a[i]) * weights[i]).sum();
        let scalar_crelu: f32 = (0..ACC).map(|i| crelu(acc_b[i]) * weights[ACC + i]).sum();
        let got_screlu = screlu_dot(&acc_a[..ACC], &weights[..ACC]);
        let got_crelu = crelu_dot(&acc_b[..ACC], &weights[ACC..2 * ACC]);
        assert!(
            (got_screlu - scalar_screlu).abs() < 1e-3,
            "screlu_dot {got_screlu} vs scalar {scalar_screlu}"
        );
        assert!(
            (got_crelu - scalar_crelu).abs() < 1e-3,
            "crelu_dot {got_crelu} vs scalar {scalar_crelu}"
        );
    }

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

    /// Build a deterministic dummy v4 (8 piece-count output buckets)
    /// network file image.
    fn dummy_net_bytes_v4() -> Vec<u8> {
        let n_weights = FT_IN_V3 * ACC + ACC + 8 * 2 * ACC + 8;
        let mut buf = Vec::with_capacity(20 + 4 * n_weights);
        buf.extend_from_slice(MAGIC);
        buf.extend_from_slice(&4u32.to_le_bytes());
        buf.extend_from_slice(&(FT_IN_V3 as u32).to_le_bytes());
        buf.extend_from_slice(&(ACC as u32).to_le_bytes());
        for w in xorshift_weights(n_weights, 0xBEEF_CAFE_1234_5678) {
            buf.extend_from_slice(&w.to_le_bytes());
        }
        buf
    }

    /// A v4 net whose feature tables and output weights are all zero and
    /// whose per-bucket biases are 0..=7: the evaluation is exactly
    /// `bucket * SCALE`, so the piece-count bucket selection is directly
    /// observable.
    fn bucket_probe_net() -> Nnue {
        let mut buf = Vec::with_capacity(
            20 + 4 * (FT_IN_V3 * ACC + ACC + 8 * 2 * ACC + 8),
        );
        buf.extend_from_slice(MAGIC);
        buf.extend_from_slice(&4u32.to_le_bytes());
        buf.extend_from_slice(&(FT_IN_V3 as u32).to_le_bytes());
        buf.extend_from_slice(&(ACC as u32).to_le_bytes());
        for _ in 0..FT_IN_V3 * ACC + ACC + 8 * 2 * ACC {
            buf.extend_from_slice(&0f32.to_le_bytes());
        }
        for b in 0..8u32 {
            buf.extend_from_slice(&(b as f32).to_le_bytes());
        }
        Nnue::from_bytes(&buf).expect("v4 probe net")
    }

    #[test]
    fn v4_output_bucket_selection_by_piece_count() {
        let net = bucket_probe_net();
        // (fen, expected piece count, expected bucket)
        let cases: &[(&str, usize, usize)] = &[
            // 3 pieces -> (3-1)/4 = 0
            ("7k/8/8/8/8/8/8/K3R3 w - - 0 1", 3, 0),
            // 5 -> 1
            ("7k/8/8/8/4P3/4P3/8/K3R3 w - - 0 1", 5, 1),
            // 11 -> 2
            ("7k/8/P7/PPPP4/PPP5/8/8/K3R3 w - - 0 1", 11, 2),
            // 15 -> 3
            ("7k/8/PPP5/PPP5/PPP5/PPP5/8/K3R3 w - - 0 1", 15, 3),
            // 18 -> 4
            ("7k/PPP5/PPP5/PPP5/PPP5/PPP5/8/K3R3 w - - 0 1", 18, 4),
            // 21 -> 5
            ("7k/PPP5/PPP5/PPP5/PPP5/PPP5/PPP5/K3R3 w - - 0 1", 21, 5),
            // 26 -> 6
            ("7k/PPP5/PPP5/PPP5/PPP5/PPP5/PPPPPPPP/K3R3 w - - 0 1", 26, 6),
            // 32 (startpos) -> 7
            (fen::START_FEN, 32, 7),
        ];
        for (fen_text, pieces, bucket) in cases {
            let pos = fen::parse(fen_text)
                .unwrap_or_else(|e| panic!("probe fen must parse ({}): {}", fen_text, e));
            assert_eq!(
                occupied_count(&pos),
                *pieces,
                "piece count for {}",
                fen_text
            );
            let want = (*bucket as f32) * SCALE;
            assert_eq!(
                net.eval(&pos) as f32,
                want,
                "bucket eval for {} ({} pieces)",
                fen_text,
                pieces
            );
            // state path must agree (bucket derived from pos there too)
            let state = net.initial_state(&pos);
            assert_eq!(
                net.eval_with_state(&pos, &state) as f32,
                want,
                "state bucket eval for {}",
                fen_text
            );
        }
    }

    #[test]
    fn v4_net_loads_and_incremental_survives_captures() {
        let net = load_dummy("v4", dummy_net_bytes_v4());
        assert_eq!(net.out_buckets, 8);
        let start = fen::parse(fen::START_FEN).unwrap();
        let s0 = net.eval(&start);
        assert!(s0.abs() <= EVAL_CLAMP);
        let state = net.initial_state(&start);
        assert_eq!(net.eval_with_state(&start, &state), s0);
        check_incremental_step(&net, fen::START_FEN, "e2e4", "v4 quiet");
        check_incremental_step(
            &net,
            "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
            "f3e5",
            "v4 capture",
        );
    }

    /// A capture that crosses a bucket boundary (29 -> 28 pieces:
    /// bucket 7 -> 6) must change the evaluation by exactly the
    /// per-bucket bias delta on the probe net, through both the
    /// full-refresh and the incremental state paths.
    #[test]
    fn v4_bucket_change_on_capture_changes_output() {
        let net = bucket_probe_net();
        let before = fen::parse("3rkbnr/ppp1pppp/8/1p6/PP6/8/1PPPPPPP/RNBQKB1R b - - 0 1")
            .expect("capture probe fen");
        assert_eq!(occupied_count(&before), 29);
        assert_eq!(net.eval(&before) as f32, 7.0 * SCALE);
        let mv = crate::movegen::parse_uci_move(&before, "b5a4").expect("b5a4");
        let after = before.make(mv);
        assert_eq!(occupied_count(&after), 28);
        let full = net.initial_state(&after);
        assert_eq!(net.eval_with_state(&after, &full) as f32, 6.0 * SCALE);
        let incremental =
            net.update_state(&before, &after, mv, &net.initial_state(&before));
        assert_eq!(net.eval_with_state(&after, &incremental) as f32, 6.0 * SCALE);
    }

    /// End-to-end ABI cross-check against the Python trainer: loads a
    /// trainer-exported v4 net (path from NNUE_CROSSCHECK_NET, test skipped
    /// when unset) and prints the raw output for the start position.
    #[test]
    fn v4_crosscheck_exported_net() {
        let Ok(path) = std::env::var("NNUE_CROSSCHECK_NET") else {
            return;
        };
        let net = Nnue::load(&path).expect("trainer export must load as v4");
        let pos = fen::parse(fen::START_FEN).expect("startpos");
        let state = net.initial_state(&pos);
        let acc_stm = &state.nnue.acc[pos.side.idx()];
        let acc_nstm = &state.nnue.acc[pos.side.flip().idx()];
        let pieces = occupied_count(&pos);
        let bkt = output_bucket(pieces);
        let base = bkt * 2 * ACC;
        let raw = net.out_b[bkt]
            + screlu_dot(acc_stm, &net.out_w[base..base + ACC])
            + screlu_dot(acc_nstm, &net.out_w[base + ACC..base + 2 * ACC]);
        let cp = net.eval(&pos);
        println!(
            "CROSSCHECK rust raw={:.6} cp={} bucket={} pieces={} out_buckets={}",
            raw, cp, bkt, pieces, net.out_buckets
        );
        assert_eq!(net.eval(&pos), net.eval_with_state(&pos, &state));
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
