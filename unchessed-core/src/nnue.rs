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
//! Weight files remain f32, then the feature transformer is quantized at load
//! time to int16 (scale 511) and accumulated with AVX-512BW, AVX2, or scalar
//! dispatch. Active features and accumulators are stack-resident, removing the
//! former two heap allocations per eval. Search carries a ply-indexed state;
//! normal moves update changed rows incrementally and own-king bucket changes
//! trigger a perspective refresh.
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

#[derive(Clone, Copy)]
enum Scheme {
    Flat768,
    HalfKa,
    HalfKav2Hm,
}

#[derive(Clone, Copy, Debug)]
enum AccumulatorBackend {
    Scalar,
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    Avx2,
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    Avx512,
}

const QA: f32 = 511.0;
const MAX_ACTIVE_FEATURES: usize = 32;

pub struct Nnue {
    scheme: Scheme,
    /// Quantized [ft_in][ACC], row-major.
    ft_w: Vec<i16>,
    /// Quantized [ACC].
    ft_b: [i16; ACC],
    /// Output head remains f32; it is a tiny fraction of inference cost and
    /// retaining scalar order avoids extra reassociation drift.
    out_w: Vec<f32>,
    out_b: f32,
    backend: AccumulatorBackend,
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
        let value = f32::from_le_bytes(buf[s..s + 4].try_into().unwrap());
        if !value.is_finite() {
            return Err("UNCHNNUE contains a non-finite weight".into());
        }
        v.push(value);
    }
    *off += n * 4;
    Ok(v)
}

fn quantize(value: f32) -> i16 {
    (value * QA).round().clamp(i16::MIN as f32, i16::MAX as f32) as i16
}

fn select_backend() -> AccumulatorBackend {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512bw") {
            return AccumulatorBackend::Avx512;
        }
        if std::is_x86_feature_detected!("avx2") {
            return AccumulatorBackend::Avx2;
        }
    }
    AccumulatorBackend::Scalar
}

fn accumulate_scalar(weights: &[i16], bias: &[i16; ACC], indices: &[usize]) -> [i16; ACC] {
    let mut accumulator = *bias;
    for &index in indices {
        let row = &weights[index * ACC..(index + 1) * ACC];
        for (value, weight) in accumulator.iter_mut().zip(row) {
            *value += *weight;
        }
    }
    accumulator
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn accumulate_avx2(weights: &[i16], bias: &[i16; ACC], indices: &[usize]) -> [i16; ACC] {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let mut accumulator = *bias;
    for &index in indices {
        let row = weights.as_ptr().add(index * ACC);
        for offset in (0..ACC).step_by(16) {
            let a = _mm256_loadu_si256(accumulator.as_ptr().add(offset) as *const __m256i);
            let w = _mm256_loadu_si256(row.add(offset) as *const __m256i);
            _mm256_storeu_si256(
                accumulator.as_mut_ptr().add(offset) as *mut __m256i,
                _mm256_add_epi16(a, w),
            );
        }
    }
    accumulator
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn accumulate_avx512(weights: &[i16], bias: &[i16; ACC], indices: &[usize]) -> [i16; ACC] {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let mut accumulator = *bias;
    for &index in indices {
        let row = weights.as_ptr().add(index * ACC);
        for offset in (0..ACC).step_by(32) {
            let a = _mm512_loadu_si512(accumulator.as_ptr().add(offset) as *const __m512i);
            let w = _mm512_loadu_si512(row.add(offset) as *const __m512i);
            _mm512_storeu_si512(
                accumulator.as_mut_ptr().add(offset) as *mut __m512i,
                _mm512_add_epi16(a, w),
            );
        }
    }
    accumulator
}

impl Nnue {
    pub fn load(path: &str) -> Result<Nnue, String> {
        let buf = std::fs::read(path).map_err(|e| format!("open {}: {}", path, e))?;
        Nnue::from_bytes(&buf)
    }

    pub fn backend_name(&self) -> &'static str {
        match self.backend {
            AccumulatorBackend::Scalar => "int16-scalar",
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            AccumulatorBackend::Avx2 => "int16-avx2",
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            AccumulatorBackend::Avx512 => "int16-avx512bw",
        }
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

        let ft_w_f32 = read_f32s(buf, &mut off, ft_in * ACC)?;
        let ft_b_f32 = read_f32s(buf, &mut off, ACC)?;
        let out_w = read_f32s(buf, &mut off, mult * ACC)?;
        let out_b = read_f32s(buf, &mut off, 1)?[0];
        let max_weight = ft_w_f32.iter().map(|value| value.abs()).fold(0.0, f32::max);
        let max_bias = ft_b_f32.iter().map(|value| value.abs()).fold(0.0, f32::max);
        let conservative_bound = (max_bias + MAX_ACTIVE_FEATURES as f32 * max_weight) * QA;
        if conservative_bound > i16::MAX as f32 {
            return Err(format!(
                "quantized accumulator may overflow i16 (bound {:.0} > {})",
                conservative_bound,
                i16::MAX
            ));
        }
        let ft_w = ft_w_f32.into_iter().map(quantize).collect();
        let ft_b_vec: Vec<i16> = ft_b_f32.into_iter().map(quantize).collect();
        let ft_b: [i16; ACC] = ft_b_vec.try_into().map_err(|_| "bad FT bias length")?;
        Ok(Nnue {
            scheme,
            ft_w,
            ft_b,
            out_w,
            out_b,
            backend: select_backend(),
        })
    }

    fn push_feature(indices: &mut [usize; MAX_ACTIVE_FEATURES], len: &mut usize, index: usize) {
        debug_assert!(*len < MAX_ACTIVE_FEATURES);
        indices[*len] = index;
        *len += 1;
    }

    fn active_features(
        &self,
        pos: &Position,
        persp: Color,
    ) -> ([usize; MAX_ACTIVE_FEATURES], usize) {
        let mut indices = [0usize; MAX_ACTIVE_FEATURES];
        let mut len = 0usize;
        let white_persp = matches!(persp, Color::White);
        match self.scheme {
            Scheme::Flat768 => {
                for color in 0..2 {
                    let own = usize::from(color != persp.idx());
                    for piece in 0..6 {
                        let mut pieces = pos.bb[color][piece];
                        while pieces != 0 {
                            let square = pieces.trailing_zeros() as usize;
                            pieces &= pieces - 1;
                            let oriented = if white_persp { square } else { square ^ 56 };
                            Self::push_feature(
                                &mut indices,
                                &mut len,
                                own * 384 + piece * 64 + oriented,
                            );
                        }
                    }
                }
            }
            Scheme::HalfKa => {
                let own_color = persp.idx();
                let opponent = 1 - own_color;
                let king_raw = pos.bb[own_color][KING].trailing_zeros() as usize;
                let king_square = if white_persp { king_raw } else { king_raw ^ 56 };
                let mut piece_index = 0usize;
                for &(color, is_own) in &[(own_color, true), (opponent, false)] {
                    for piece in 0..6 {
                        if is_own && piece == KING {
                            continue;
                        }
                        let mut pieces = pos.bb[color][piece];
                        while pieces != 0 {
                            let square = pieces.trailing_zeros() as usize;
                            pieces &= pieces - 1;
                            let oriented = if white_persp { square } else { square ^ 56 };
                            Self::push_feature(
                                &mut indices,
                                &mut len,
                                king_square * N_PIECE_SQ_V2 + piece_index * 64 + oriented,
                            );
                        }
                        piece_index += 1;
                    }
                }
            }
            Scheme::HalfKav2Hm => {
                let own_color = persp.idx();
                let opponent = 1 - own_color;
                let king_raw = pos.bb[own_color][KING].trailing_zeros() as usize;
                let king_oriented = if white_persp { king_raw } else { king_raw ^ 56 };
                let mirror = (king_oriented % 8) < 4;
                let king_final = if mirror {
                    king_oriented ^ 7
                } else {
                    king_oriented
                };
                let bucket = KING_BUCKETS[king_final] as usize;
                let orient = |square: usize| {
                    let oriented = if white_persp { square } else { square ^ 56 };
                    if mirror {
                        oriented ^ 7
                    } else {
                        oriented
                    }
                };
                for piece in 0..5 {
                    for &(color, is_own) in &[(own_color, true), (opponent, false)] {
                        let mut pieces = pos.bb[color][piece];
                        while pieces != 0 {
                            let square = pieces.trailing_zeros() as usize;
                            pieces &= pieces - 1;
                            let piece_index = piece * 2 + usize::from(!is_own);
                            Self::push_feature(
                                &mut indices,
                                &mut len,
                                bucket * N_PIECE_SQ_V3 + piece_index * 64 + orient(square),
                            );
                        }
                    }
                }
                let opponent_king = pos.bb[opponent][KING].trailing_zeros() as usize;
                Self::push_feature(
                    &mut indices,
                    &mut len,
                    bucket * N_PIECE_SQ_V3 + 640 + orient(opponent_king),
                );
                Self::push_feature(
                    &mut indices,
                    &mut len,
                    bucket * N_PIECE_SQ_V3 + 640 + king_final,
                );
            }
        }
        (indices, len)
    }

    fn accumulate_indices(&self, indices: &[usize]) -> [i16; ACC] {
        match self.backend {
            AccumulatorBackend::Scalar => accumulate_scalar(&self.ft_w, &self.ft_b, indices),
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            AccumulatorBackend::Avx2 => unsafe { accumulate_avx2(&self.ft_w, &self.ft_b, indices) },
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            AccumulatorBackend::Avx512 => unsafe {
                accumulate_avx512(&self.ft_w, &self.ft_b, indices)
            },
        }
    }

    fn accumulate(&self, pos: &Position, perspective: Color) -> [i16; ACC] {
        let (indices, len) = self.active_features(pos, perspective);
        self.accumulate_indices(&indices[..len])
    }

    fn feature_index(
        &self,
        pos: &Position,
        perspective: Color,
        color: Color,
        piece: usize,
        square: usize,
    ) -> Option<usize> {
        let white = perspective == Color::White;
        match self.scheme {
            Scheme::Flat768 => {
                let own = usize::from(color != perspective);
                let oriented = if white { square } else { square ^ 56 };
                Some(own * 384 + piece * 64 + oriented)
            }
            Scheme::HalfKa => {
                if color == perspective && piece == KING {
                    return None;
                }
                let king_raw = pos.bb[perspective.idx()][KING].trailing_zeros() as usize;
                let king_square = if white { king_raw } else { king_raw ^ 56 };
                let piece_index = if color == perspective {
                    piece
                } else {
                    5 + piece
                };
                let oriented = if white { square } else { square ^ 56 };
                Some(king_square * N_PIECE_SQ_V2 + piece_index * 64 + oriented)
            }
            Scheme::HalfKav2Hm => {
                let king_raw = pos.bb[perspective.idx()][KING].trailing_zeros() as usize;
                let king_oriented = if white { king_raw } else { king_raw ^ 56 };
                let mirror = king_oriented % 8 < 4;
                let king_final = if mirror {
                    king_oriented ^ 7
                } else {
                    king_oriented
                };
                let bucket = KING_BUCKETS[king_final] as usize;
                let mut oriented = if white { square } else { square ^ 56 };
                if mirror {
                    oriented ^= 7;
                }
                let piece_index = if piece == KING {
                    10
                } else {
                    piece * 2 + usize::from(color != perspective)
                };
                Some(bucket * N_PIECE_SQ_V3 + piece_index * 64 + oriented)
            }
        }
    }

    fn apply_row(&self, accumulator: &mut [i16; ACC], index: usize, sign: i32) {
        let row = &self.ft_w[index * ACC..(index + 1) * ACC];
        for (value, &weight) in accumulator.iter_mut().zip(row) {
            let updated = *value as i32 + sign * weight as i32;
            debug_assert!((i16::MIN as i32..=i16::MAX as i32).contains(&updated));
            *value = updated as i16;
        }
    }

    fn update_accumulator(
        &self,
        before: &Position,
        after: &Position,
        perspective: Color,
        parent: &[i16; ACC],
    ) -> [i16; ACC] {
        let own_king_moved = before.king_sq(perspective) != after.king_sq(perspective);
        if own_king_moved && !matches!(self.scheme, Scheme::Flat768) {
            return self.accumulate(after, perspective);
        }

        let mut accumulator = *parent;
        for color in [Color::White, Color::Black] {
            for piece in 0..6 {
                let mut removed = before.bb[color.idx()][piece] & !after.bb[color.idx()][piece];
                while removed != 0 {
                    let square = removed.trailing_zeros() as usize;
                    removed &= removed - 1;
                    if let Some(index) =
                        self.feature_index(before, perspective, color, piece, square)
                    {
                        self.apply_row(&mut accumulator, index, -1);
                    }
                }
                let mut added = after.bb[color.idx()][piece] & !before.bb[color.idx()][piece];
                while added != 0 {
                    let square = added.trailing_zeros() as usize;
                    added &= added - 1;
                    if let Some(index) =
                        self.feature_index(after, perspective, color, piece, square)
                    {
                        self.apply_row(&mut accumulator, index, 1);
                    }
                }
            }
        }
        accumulator
    }

    fn evaluate_accumulators(
        &self,
        pos: &Position,
        white_accumulator: &[i16; ACC],
        black_accumulator: &[i16; ACC],
    ) -> i32 {
        let (stm, nstm) = if pos.side == Color::White {
            (white_accumulator, black_accumulator)
        } else {
            (black_accumulator, white_accumulator)
        };
        let mut out = self.out_b;
        match self.scheme {
            Scheme::Flat768 | Scheme::HalfKav2Hm => {
                for i in 0..ACC {
                    out += screlu(stm[i] as f32 / QA) * self.out_w[i];
                    out += screlu(nstm[i] as f32 / QA) * self.out_w[ACC + i];
                }
            }
            Scheme::HalfKa => {
                for i in 0..ACC {
                    let stm_value = stm[i] as f32 / QA;
                    let nstm_value = nstm[i] as f32 / QA;
                    out += screlu(stm_value) * self.out_w[i];
                    out += crelu(stm_value) * self.out_w[ACC + i];
                    out += screlu(nstm_value) * self.out_w[2 * ACC + i];
                    out += crelu(nstm_value) * self.out_w[3 * ACC + i];
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
    fn eval(&self, pos: &Position) -> i32 {
        let white = self.accumulate(pos, Color::White);
        let black = self.accumulate(pos, Color::Black);
        self.evaluate_accumulators(pos, &white, &black)
    }

    fn initial_state(&self, pos: &Position) -> EvalState {
        EvalState {
            nnue: NnueEvalState {
                accumulator: [
                    self.accumulate(pos, Color::White),
                    self.accumulate(pos, Color::Black),
                ],
            },
            is_nnue: true,
        }
    }

    fn update_state(
        &self,
        before: &Position,
        after: &Position,
        _mv: Move,
        state: &EvalState,
    ) -> EvalState {
        if !state.is_nnue {
            return self.initial_state(after);
        }
        EvalState {
            nnue: NnueEvalState {
                accumulator: [
                    self.update_accumulator(
                        before,
                        after,
                        Color::White,
                        &state.nnue.accumulator[Color::White.idx()],
                    ),
                    self.update_accumulator(
                        before,
                        after,
                        Color::Black,
                        &state.nnue.accumulator[Color::Black.idx()],
                    ),
                ],
            },
            is_nnue: true,
        }
    }

    fn eval_with_state(&self, pos: &Position, state: &EvalState) -> i32 {
        if state.is_nnue {
            self.evaluate_accumulators(
                pos,
                &state.nnue.accumulator[Color::White.idx()],
                &state.nnue.accumulator[Color::Black.idx()],
            )
        } else {
            self.eval(pos)
        }
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
            assert!(
                (e1 - e2).abs() <= 1,
                "mirror mismatch for {}: {} vs {}",
                f,
                e1,
                e2
            );
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
            assert!(
                (e1 - e2).abs() <= 1,
                "mirror mismatch for {}: {} vs {}",
                f,
                e1,
                e2
            );
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
            assert!(
                (e1 - e2).abs() <= 1,
                "mirror mismatch for {}: {} vs {}",
                f,
                e1,
                e2
            );
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
            assert_ne!(
                a, b,
                "[{}] dummy net evals should differ across positions",
                tag
            );
            assert!(a.abs() <= EVAL_CLAMP && b.abs() <= EVAL_CLAMP);
        }
    }

    #[test]
    fn incremental_accumulators_match_full_refresh_for_special_moves() {
        let net = Nnue::from_bytes(&dummy_net_bytes_v3()).unwrap();
        let mut pos = fen::startpos();
        let mut state = net.initial_state(&pos);
        for uci in [
            "e2e4", "a7a6", "e4e5", "d7d5", "e5d6", // en passant
            "e7d6", "g1f3", "b8c6", "f1e2", "g8f6", "e1g1", // castle
        ] {
            let mv = crate::movegen::parse_uci_move(&pos, uci).unwrap();
            let next = pos.make(mv);
            state = net.update_state(&pos, &next, mv, &state);
            assert_eq!(
                net.eval_with_state(&next, &state),
                net.eval(&next),
                "incremental mismatch after {}",
                uci
            );
            pos = next;
        }

        let promo = fen::parse("7k/P7/8/8/8/8/8/7K w - - 0 1").unwrap();
        let promo_state = net.initial_state(&promo);
        let mv = crate::movegen::parse_uci_move(&promo, "a7a8q").unwrap();
        let next = promo.make(mv);
        let next_state = net.update_state(&promo, &next, mv, &promo_state);
        assert_eq!(net.eval_with_state(&next, &next_state), net.eval(&next));
    }

    #[test]
    fn incremental_accumulators_match_full_refresh_over_move_tree() {
        fn walk(net: &Nnue, pos: &Position, state: EvalState, depth: u32) {
            assert_eq!(net.eval_with_state(pos, &state), net.eval(pos));
            if depth == 0 {
                return;
            }
            let moves = crate::movegen::legal(pos);
            for &mv in moves.as_slice() {
                let next = pos.make(mv);
                let next_state = net.update_state(pos, &next, mv, &state);
                walk(net, &next, next_state, depth - 1);
            }
        }
        let net = Nnue::from_bytes(&dummy_net_bytes_v3()).unwrap();
        let pos = fen::startpos();
        let state = net.initial_state(&pos);
        walk(&net, &pos, state, 3);
    }

    #[test]
    fn selected_simd_accumulator_matches_scalar_exactly() {
        let net = Nnue::from_bytes(&dummy_net_bytes_v3()).unwrap();
        for fen_text in MIRROR_FENS {
            let pos = fen::parse(fen_text).unwrap();
            for perspective in [Color::White, Color::Black] {
                let (indices, len) = net.active_features(&pos, perspective);
                let scalar = accumulate_scalar(&net.ft_w, &net.ft_b, &indices[..len]);
                let selected = net.accumulate_indices(&indices[..len]);
                assert_eq!(selected, scalar, "backend {}", net.backend_name());
            }
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
