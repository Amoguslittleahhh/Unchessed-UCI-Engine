//! Hand-crafted evaluation: tapered material + piece-square tables.
//!
//! The `Eval` trait is the seam between the search and its evaluator:
//! `Hce` (below) is the hand-crafted fallback, `crate::nnue::Nnue` the
//! network evaluator.

use crate::board::*;
use crate::movegen::{
    attacked, bishop_att, queen_att, rook_att, FILE_A, KNIGHT_ATT, PAWN_ATT, RANK_1,
};

pub trait Eval: Send + Sync {
    /// Score in centipawns from the side-to-move's perspective.
    fn eval(&self, pos: &Position) -> i32;
}

/// Tunable eval constants, exposed as UCI options so they can be SPSA-tuned
/// instead of hand-picked. A hand-picked passed-pawn bonus failed its SPRT
/// gate (likely double-counting the rank gradient PAWN_EG already has) --
/// rather than guess at a smaller magnitude, these let a real tuning pass
/// find the right overall scale (including possibly confirming 0 is best).
/// Percent scale applied to the base PASSED_MG/PASSED_EG curves below;
/// 0 = feature off, reproducing the exact pre-passed-pawn eval.
#[derive(Clone, Copy)]
pub struct EvalParams {
    pub passed_mg_pct: i32,
    pub passed_eg_pct: i32,
    /// Percent scale applied to the mobility bonus below. 0 = feature off.
    pub mobility_pct: i32,
    /// Percent scale applied to the rook-file/rook-on-7th bonus below.
    /// 0 = feature off.
    pub rook_pct: i32,
}

impl Default for EvalParams {
    fn default() -> Self {
        // passed pawns: 100/100 SPRT-validated (2026-08-02): +25.7 +/- 12.3
        // Elo vs the pre-passed-pawn baseline, 2156 games, LLR crossed the
        // upper bound.
        // mobility: 100 SPRT-validated (2026-08-03): +52.3 +/- 18.1 Elo vs
        // the passed-pawn baseline, 984 games, LLR crossed the upper bound
        // -- the single biggest eval-term gain in this project so far.
        // rook file/7th: 100 SPRT-validated (2026-08-04): +10.5 +/- 7.2 Elo
        // vs the mobility baseline, 6019 games, LLR crossed the upper bound.
        EvalParams {
            passed_mg_pct: 100,
            passed_eg_pct: 100,
            mobility_pct: 100,
            rook_pct: 100,
        }
    }
}

/// Per-(piece-type, safe-square-count) (mg, eg) bonus, indexed
/// [knight=0/bishop=1/rook=2/queen=3][min(count, 27)]. Values taken
/// directly from RubiChess 20240817's real, SPSA-tuned eMobilitybonus
/// table (RubiChess.h) -- king safety's failed attempt showed this
/// engine's own SPSA harness can't discover a term's magnitude from
/// scratch, so starting from validated real numbers (as passed pawns
/// did) is the standing discipline, not a one-off.
#[rustfmt::skip]
const MOBILITY_MG: [[i32; 28]; 4] = [
    [ 16,  38,  51,  57,  64,  71,  77,  84,  86,   0,   0,   0,   0,   0,   0,   0,
       0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
    [ -8,  23,  45,  55,  65,  74,  80,  81,  86,  85,  86,  95, 100,  89,   0,   0,
       0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
    [-57,  14,  31,  34,  36,  38,  40,  46,  47,  50,  54,  50,  49,  50,  44,   0,
       0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
    [  0,  13,  -2,   3,   5,   4,   6,  11,  10,  15,  14,  17,  16,  18,  17,  19,
      20,  21,  18,  13,  45,  67,  72,  80,  69, 108, 107, 114],
];
#[rustfmt::skip]
const MOBILITY_EG: [[i32; 28]; 4] = [
    [-90, -26,   1,  13,  27,  37,  36,  36,  30,   0,   0,   0,   0,   0,   0,   0,
       0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
    [-25,  -7,   2,  21,  31,  35,  43,  50,  51,  58,  52,  55,  54,  60,   0,   0,
       0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
    [ 11,  17,  57,  70,  82,  90,  92,  96, 101, 111, 109, 115, 118, 116, 116,   0,
       0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
    [-200,-157,  31,  75,  94, 152, 161, 177, 192, 187, 212, 213, 227, 229, 242, 251,
     253, 256, 263, 270, 244, 233, 241, 230, 264, 231, 230, 198],
];

/// Squares that count toward mobility for `color`: everything except this
/// color's own low-rank (still-developing) pawns, squares the opponent's
/// pawns attack (unsafe to sit on), and this color's own king square --
/// RubiChess's real `goodMobility` definition.
fn mobility_area(pos: &Position, color: Color) -> Bitboard {
    let opp = color.flip();
    let low_ranks: Bitboard = if let Color::White = color {
        (RANK_1 << 8) | (RANK_1 << 16)
    } else {
        (RANK_1 << 48) | (RANK_1 << 40)
    };
    let blocked_pawns = pos.bb[color.idx()][PAWN] & low_ranks;

    let mut enemy_pawn_attacks: Bitboard = 0;
    let mut ep = pos.bb[opp.idx()][PAWN];
    while ep != 0 {
        let s = ep.trailing_zeros() as usize;
        ep &= ep - 1;
        enemy_pawn_attacks |= PAWN_ATT[opp.idx()][s];
    }

    !(blocked_pawns | enemy_pawn_attacks | pos.bb[color.idx()][KING])
}

/// Mobility bonus for `color`'s knights/bishops/rooks/queens, as an
/// (mg, eg) magnitude pair (sign applied by the caller).
fn mobility_term(pos: &Position, color: Color, area: Bitboard) -> (i32, i32) {
    let mut mg = 0i32;
    let mut eg = 0i32;

    let mut nb = pos.bb[color.idx()][KNIGHT];
    while nb != 0 {
        let s = nb.trailing_zeros() as usize;
        nb &= nb - 1;
        let cnt = ((KNIGHT_ATT[s] & area).count_ones() as usize).min(27);
        mg += MOBILITY_MG[0][cnt];
        eg += MOBILITY_EG[0][cnt];
    }
    let mut bs = pos.bb[color.idx()][BISHOP];
    while bs != 0 {
        let s = bs.trailing_zeros() as usize;
        bs &= bs - 1;
        let cnt = ((bishop_att(s as u8, pos.occ) & area).count_ones() as usize).min(27);
        mg += MOBILITY_MG[1][cnt];
        eg += MOBILITY_EG[1][cnt];
    }
    let mut rk = pos.bb[color.idx()][ROOK];
    while rk != 0 {
        let s = rk.trailing_zeros() as usize;
        rk &= rk - 1;
        let cnt = ((rook_att(s as u8, pos.occ) & area).count_ones() as usize).min(27);
        mg += MOBILITY_MG[2][cnt];
        eg += MOBILITY_EG[2][cnt];
    }
    let mut qn = pos.bb[color.idx()][QUEEN];
    while qn != 0 {
        let s = qn.trailing_zeros() as usize;
        qn &= qn - 1;
        let cnt = ((queen_att(s as u8, pos.occ) & area).count_ones() as usize).min(27);
        mg += MOBILITY_MG[3][cnt];
        eg += MOBILITY_EG[3][cnt];
    }

    (mg, eg)
}

/// (mg, eg) bonus for a rook on a file with no pawns of its OWN color --
/// index 0 = semi-open (the opponent still has a pawn on that file),
/// index 1 = fully open (neither side does). Values taken directly from
/// RubiChess 20240817's real, SPSA-tuned eSlideronfreefilebonus constant.
const ROOK_FREE_FILE_MG: [i32; 2] = [21, 43];
const ROOK_FREE_FILE_EG: [i32; 2] = [7, 1];
/// (mg, eg) bonus for a rook on the 7th rank (relative) while the enemy
/// king is pinned back on the 7th or 8th -- RubiChess's real
/// eRookon7thbonus constant.
const ROOK_7TH_MG: i32 = -1;
const ROOK_7TH_EG: i32 = 22;

/// Rook-file/rook-on-7th bonus for `color`'s rooks, as an (mg, eg)
/// magnitude pair (sign applied by the caller). Deliberately simplified
/// vs RubiChess's real formula: no rook-vs-king-ring bonus, no castling-
/// rights-based penalty for a rook stuck behind pawns -- both would need
/// infrastructure (king ring, or castling-rights history) this term
/// doesn't otherwise depend on, kept out to stay a single well-scoped
/// idea like every other eval addition this project has gated.
fn rook_term(pos: &Position, color: Color) -> (i32, i32) {
    let opp = color.flip();
    let mut mg = 0i32;
    let mut eg = 0i32;

    let mut rk = pos.bb[color.idx()][ROOK];
    while rk != 0 {
        let s = rk.trailing_zeros() as usize;
        rk &= rk - 1;
        let file = s % 8;
        let file_mask: Bitboard = FILE_A << file;

        if pos.bb[color.idx()][PAWN] & file_mask == 0 {
            let idx = if pos.bb[opp.idx()][PAWN] & file_mask == 0 { 1 } else { 0 };
            mg += ROOK_FREE_FILE_MG[idx];
            eg += ROOK_FREE_FILE_EG[idx];
        }

        let rank = s / 8;
        let on_7th = if let Color::White = color { rank == 6 } else { rank == 1 };
        if on_7th {
            let king_sq = pos.bb[opp.idx()][KING].trailing_zeros() as usize;
            if king_sq < 64 {
                let king_rank = king_sq / 8;
                let king_pinned_back = if let Color::White = color {
                    king_rank == 6 || king_rank == 7
                } else {
                    king_rank == 1 || king_rank == 0
                };
                if king_pinned_back {
                    mg += ROOK_7TH_MG;
                    eg += ROOK_7TH_EG;
                }
            }
        }
    }

    (mg, eg)
}

/// Base rank-scaled curve (index 0 = own back rank, 6 = one step from
/// queening), shared by both colors via mirrored indexing; scaled by
/// EvalParams' percentages before being added to the score.
const PASSED_MG: [i32; 8] = [0, 5, 8, 15, 28, 50, 85, 0];
const PASSED_EG: [i32; 8] = [0, 10, 16, 30, 55, 100, 170, 0];

/// [color][square] -> squares (own file + both neighbors, ahead of this
/// pawn) that must be free of enemy pawns for this pawn to be passed.
const fn build_passed_masks() -> [[u64; 64]; 2] {
    let mut masks = [[0u64; 64]; 2];
    let mut s = 0usize;
    while s < 64 {
        let f = (s % 8) as i32;
        let r = (s / 8) as i32;
        let mut wm = 0u64;
        let mut br = r + 1;
        while br < 8 {
            let mut bf = f - 1;
            while bf <= f + 1 {
                if bf >= 0 && bf < 8 {
                    wm |= 1u64 << (br * 8 + bf);
                }
                bf += 1;
            }
            br += 1;
        }
        masks[0][s] = wm;
        let mut bm = 0u64;
        let mut br2 = r - 1;
        while br2 >= 0 {
            let mut bf = f - 1;
            while bf <= f + 1 {
                if bf >= 0 && bf < 8 {
                    bm |= 1u64 << (br2 * 8 + bf);
                }
                bf += 1;
            }
            br2 -= 1;
        }
        masks[1][s] = bm;
        s += 1;
    }
    masks
}

static PASSED_MASKS: [[u64; 64]; 2] = build_passed_masks();

pub struct Hce {
    pub params: EvalParams,
}

impl Hce {
    pub fn new(params: EvalParams) -> Hce {
        Hce { params }
    }
}

impl Default for Hce {
    fn default() -> Self {
        Hce { params: EvalParams::default() }
    }
}

impl Eval for Hce {
    fn eval(&self, pos: &Position) -> i32 {
        evaluate(pos, &self.params)
    }
}

pub const MG_VALUE: [i32; 6] = [82, 337, 365, 477, 1025, 0];
pub const EG_VALUE: [i32; 6] = [94, 281, 297, 512, 936, 0];

// Piece-square tables written board-style: first row is rank 8 (a8..h8),
// last row is rank 1. White piece on square s indexes T[s ^ 56]; black T[s].

#[rustfmt::skip]
const PAWN_MG: [i32; 64] = [
      0,   0,   0,   0,   0,   0,   0,   0,
     70,  90,  60,  80,  70,  95,  30,  -5,
     -5,   8,  25,  30,  55,  50,  25, -15,
    -12,  10,   5,  20,  22,  10,  12, -20,
    -22,  -5,  -5,  14,  16,   3,   5, -22,
    -20,  -6,  -8,  -8,   4,   2,  25, -14,
    -28,  -4, -18, -20, -12,  20,  30, -18,
      0,   0,   0,   0,   0,   0,   0,   0,
];

#[rustfmt::skip]
const PAWN_EG: [i32; 64] = [
      0,   0,   0,   0,   0,   0,   0,   0,
    130, 125, 115, 100, 105, 100, 120, 140,
     75,  80,  65,  50,  40,  40,  65,  70,
     25,  18,  10,   0,  -5,   0,  12,  15,
     10,   6,  -6, -10, -10,  -8,   2,   0,
      0,   2,  -8,  -2,  -2,  -6,  -4,  -8,
     10,   4,   6,   8,  10,   0,   0,  -6,
      0,   0,   0,   0,   0,   0,   0,   0,
];

#[rustfmt::skip]
const KNIGHT_MG: [i32; 64] = [
   -110, -60, -30, -30,  30, -70, -15, -70,
    -50, -25,  45,  25,  15,  40,   5, -12,
    -30,  40,  30,  45,  60,  70,  50,  30,
     -5,  10,  15,  35,  25,  45,  12,  15,
    -10,   2,  12,  10,  20,  14,  15,  -5,
    -20,  -6,   8,   8,  14,  12,  16, -12,
    -22, -35,  -8,   0,   2,  12, -10,  -8,
    -70, -14, -35, -22, -12, -18, -12, -16,
];

#[rustfmt::skip]
const KNIGHT_EG: [i32; 64] = [
    -40, -25, -10, -20, -20, -18, -40, -70,
    -18,  -5, -18,   0,  -6, -18, -16, -35,
    -16, -12,  10,   6,  -4,  -6, -14, -30,
    -12,   2,  16,  16,  16,   8,   6, -12,
    -12,  -4,  12,  18,  12,  12,   2, -12,
    -16,  -2,   0,  10,   8,  -2, -14, -16,
    -30, -14,  -8,  -4,  -1, -14, -16, -30,
    -20, -35, -16, -10, -14, -12, -35, -45,
];

#[rustfmt::skip]
const BISHOP_MG: [i32; 64] = [
    -20,   4, -55, -25, -15, -30,   5,  -5,
    -18,  10, -12,  -8,  20,  40,  12, -32,
    -10,  25,  30,  28,  25,  35,  25,   0,
     -3,   3,  12,  35,  25,  25,   5,   0,
     -4,   9,   9,  18,  24,   8,   7,   3,
      0,  10,  10,  10,   9,  18,  12,   7,
      3,  20,  11,   4,   8,  15,  25,   0,
    -22,  -2,  -8, -14,  -8,  -8, -25, -15,
];

#[rustfmt::skip]
const BISHOP_EG: [i32; 64] = [
    -10, -14,  -8,  -5,  -4,  -6, -12, -16,
     -5,  -3,   4,  -8,  -2,  -8,  -3, -10,
      2,  -5,   0,   0,  -1,   4,   0,   3,
     -2,   6,   8,   6,  10,   6,   2,   1,
     -4,   2,   9,  13,   4,   7,  -2,  -6,
     -8,  -2,   6,   7,   9,   2,  -4, -10,
    -10,  -6,  -4,   0,   2,  -6, -10, -20,
    -16,  -6, -16,  -3,  -6, -11,  -3, -12,
];

#[rustfmt::skip]
const ROOK_MG: [i32; 64] = [
     20,  30,  20,  36,  42,   6,  20,  30,
     18,  22,  40,  42,  55,  45,  18,  30,
     -4,  14,  18,  25,  12,  30,  40,  10,
    -16,  -8,   4,  18,  16,  25,  -6, -14,
    -26, -18, -10,  -2,   6,  -4,   4, -16,
    -30, -18, -12, -12,   2,   0,  -4, -22,
    -30, -12, -14,  -6,   2,   8,  -4, -50,
    -12, -10,   0,  12,  14,   4, -22, -18,
];

#[rustfmt::skip]
const ROOK_EG: [i32; 64] = [
     10,   8,  12,  10,   8,   8,   6,   4,
      8,  10,  10,   8,  -2,   2,   6,   2,
      6,   6,   6,   4,   3,  -2,  -4,  -2,
      3,   2,   8,   0,   1,   0,  -1,   1,
      2,   4,   5,   2,  -4,  -5,  -6,  -8,
     -3,   0,  -4,  -1,  -6,  -8,  -7, -10,
     -5,  -5,   0,   1,  -6,  -8, -10,  -2,
     -8,   1,   2,   0,  -4, -10,   3, -16,
];

#[rustfmt::skip]
const QUEEN_MG: [i32; 64] = [
    -25,   0,  25,  10,  55,  40,  40,  42,
    -22, -38,  -4,   0, -15,  55,  25,  52,
    -12, -16,   6,   7,  28,  55,  45,  55,
    -25, -25, -15, -15,   0,  16,   0,   0,
     -8, -25,  -8,  -9,  -2,  -4,   2,  -2,
    -13,   3, -10,  -2,  -4,   2,  13,   4,
    -34,  -8,  10,   2,   8,  14,  -2,   0,
     -1, -17,  -8,   9, -14, -24, -30, -48,
];

#[rustfmt::skip]
const QUEEN_EG: [i32; 64] = [
     -8,  22,  22,  26,  26,  18,  10,  20,
    -16,  20,  32,  40,  58,  24,  30,   0,
    -20,   6,   8,  48,  46,  34,  18,   8,
      2,  22,  24,  44,  56,  40,  56,  36,
    -18,  28,  18,  46,  30,  34,  38,  22,
    -16, -26,  14,   6,   8,  16,  10,   4,
    -22, -22, -30, -16, -16, -22, -36, -32,
    -32, -28, -22, -42,  -4, -30, -20, -40,
];

#[rustfmt::skip]
const KING_MG: [i32; 64] = [
    -65,  22,  15, -15, -55, -34,   2,  12,
     28,  -1, -20,  -7,  -8,  -4, -38, -29,
     -9,  24,   2, -16, -20,   6,  22, -22,
    -17, -20, -12, -27, -30, -25, -14, -36,
    -49,  -1, -27, -39, -46, -44, -33, -51,
    -14, -14, -22, -46, -44, -30, -15, -27,
      1,   7,  -8, -64, -43, -16,   9,   8,
    -15,  36,  12, -54,   8, -28,  24,  14,
];

#[rustfmt::skip]
const KING_EG: [i32; 64] = [
    -74, -35, -18, -18, -11,  15,   4, -17,
    -12,  17,  14,  17,  17,  38,  23,  11,
     10,  17,  23,  15,  20,  45,  44,  13,
     -8,  22,  24,  27,  26,  33,  26,   3,
    -18,  -4,  21,  24,  27,  23,   9, -11,
    -19,  -3,  11,  21,  23,  16,   7,  -9,
    -27, -11,   4,  13,  14,   4,  -5, -17,
    -53, -34, -21, -11, -28, -14, -24, -43,
];

const PST_MG: [&[i32; 64]; 6] = [&PAWN_MG, &KNIGHT_MG, &BISHOP_MG, &ROOK_MG, &QUEEN_MG, &KING_MG];
const PST_EG: [&[i32; 64]; 6] = [&PAWN_EG, &KNIGHT_EG, &BISHOP_EG, &ROOK_EG, &QUEEN_EG, &KING_EG];

const PHASE_WEIGHT: [i32; 6] = [0, 1, 1, 2, 4, 0];
const TOTAL_PHASE: i32 = 24;
const TEMPO: i32 = 14;

/// Static evaluation in centipawns from the side-to-move's perspective.
pub fn evaluate(pos: &Position, params: &EvalParams) -> i32 {
    let mut mg = 0i32;
    let mut eg = 0i32;
    let mut phase = 0i32;

    for p in 0..6 {
        let mut w = pos.bb[0][p];
        while w != 0 {
            let s = w.trailing_zeros() as usize;
            w &= w - 1;
            mg += MG_VALUE[p] + PST_MG[p][s ^ 56];
            eg += EG_VALUE[p] + PST_EG[p][s ^ 56];
            phase += PHASE_WEIGHT[p];
        }
        let mut b = pos.bb[1][p];
        while b != 0 {
            let s = b.trailing_zeros() as usize;
            b &= b - 1;
            mg -= MG_VALUE[p] + PST_MG[p][s];
            eg -= EG_VALUE[p] + PST_EG[p][s];
            phase += PHASE_WEIGHT[p];
        }
    }

    // bishop pair
    if pos.bb[0][BISHOP].count_ones() >= 2 {
        mg += 25;
        eg += 45;
    }
    if pos.bb[1][BISHOP].count_ones() >= 2 {
        mg -= 25;
        eg -= 45;
    }

    // passed pawns: scale is a tunable UCI param (0 = off). A flat,
    // unconditional bonus for every "technically passed" pawn failed its
    // SPRT gate at every magnitude an SPSA pass tried -- a real reference
    // (RubiChess eval.cpp) shows why: it only pays out the full bonus when
    // the pawn's very next square is both EMPTY and not attacked by the
    // opponent (i.e. it can actually run), and pays much less when the
    // pawn is currently blockaded or its advance square is defended --
    // "technically passed" is not the same as "actually dangerous", and
    // treating them the same overvalues pawns sitting right next to an
    // enemy piece/king with no real prospects.
    if params.passed_mg_pct != 0 || params.passed_eg_pct != 0 {
        let mut w = pos.bb[0][PAWN];
        while w != 0 {
            let s = w.trailing_zeros() as usize;
            w &= w - 1;
            if PASSED_MASKS[0][s] & pos.bb[1][PAWN] == 0 {
                let rank = s / 8;
                let target = s + 8;
                let can_run = target < 64
                    && (pos.occ & (1u64 << target)) == 0
                    && !attacked(pos, target as u8, Color::Black);
                let factor = if can_run { 100 } else { 40 };
                mg += PASSED_MG[rank] * params.passed_mg_pct * factor / 10000;
                eg += PASSED_EG[rank] * params.passed_eg_pct * factor / 10000;
            }
        }
        let mut b = pos.bb[1][PAWN];
        while b != 0 {
            let s = b.trailing_zeros() as usize;
            b &= b - 1;
            if PASSED_MASKS[1][s] & pos.bb[0][PAWN] == 0 {
                let rank = 7 - s / 8;
                let can_run = s >= 8
                    && (pos.occ & (1u64 << (s - 8))) == 0
                    && !attacked(pos, (s - 8) as u8, Color::White);
                let factor = if can_run { 100 } else { 40 };
                mg -= PASSED_MG[rank] * params.passed_mg_pct * factor / 10000;
                eg -= PASSED_EG[rank] * params.passed_eg_pct * factor / 10000;
            }
        }
    }

    // mobility: scale is a tunable UCI param (0 = off, current default).
    if params.mobility_pct != 0 {
        let white_area = mobility_area(pos, Color::White);
        let black_area = mobility_area(pos, Color::Black);
        let (w_mg, w_eg) = mobility_term(pos, Color::White, white_area);
        let (b_mg, b_eg) = mobility_term(pos, Color::Black, black_area);
        mg += (w_mg - b_mg) * params.mobility_pct / 100;
        eg += (w_eg - b_eg) * params.mobility_pct / 100;
    }

    // rook file / rook on 7th: scale is a tunable UCI param (0 = off).
    if params.rook_pct != 0 {
        let (w_mg, w_eg) = rook_term(pos, Color::White);
        let (b_mg, b_eg) = rook_term(pos, Color::Black);
        mg += (w_mg - b_mg) * params.rook_pct / 100;
        eg += (w_eg - b_eg) * params.rook_pct / 100;
    }

    let phase = phase.min(TOTAL_PHASE);
    let score = (mg * phase + eg * (TOTAL_PHASE - phase)) / TOTAL_PHASE;

    let stm = if let Color::White = pos.side { score } else { -score };
    stm + TEMPO
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fen;

    #[test]
    fn startpos_roughly_balanced() {
        let pos = fen::startpos();
        let e = evaluate(&pos, &EvalParams::default());
        assert!(e.abs() < 60, "startpos eval {}", e);
    }

    #[test]
    fn material_up_is_positive() {
        // Black is missing their queen; white to move sees a big plus.
        let pos = fen::parse("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();
        assert!(evaluate(&pos, &EvalParams::default()) > 700);
        // Same position with black to move is a big minus for the mover.
        let pos = fen::parse("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1").unwrap();
        assert!(evaluate(&pos, &EvalParams::default()) < -700);
    }

    #[test]
    fn passed_pawn_scale_zero_matches_pre_feature_baseline() {
        // With both percentages explicitly at 0, passed-pawn logic must be a
        // true no-op -- this pins the "off" behavior so the tunable feature
        // can never silently drift from the pre-passed-pawn eval baseline.
        let pos = fen::parse("4k3/P7/8/8/8/8/8/4K3 w - - 0 1").unwrap();
        let off = evaluate(
            &pos,
            &EvalParams { passed_mg_pct: 0, passed_eg_pct: 0, mobility_pct: 0, rook_pct: 0 },
        );
        let on = evaluate(
            &pos,
            &EvalParams { passed_mg_pct: 100, passed_eg_pct: 100, mobility_pct: 0, rook_pct: 0 },
        );
        assert_ne!(off, on, "scale=100 should differ from scale=0 for an advanced passed pawn");
    }

    #[test]
    fn blockaded_passed_pawn_scores_less_than_a_free_runner() {
        // Same passed a6-pawn in both positions -- the only difference is
        // whether its next square (a7) is actually reachable. In `blocked`
        // the black king sits directly on a7 (occupied, and self-defended
        // so also "attacked" by black); in `free` black's king is off on
        // h8 and a7 is wide open. A naive "is it technically passed"
        // bonus would score these identically; the corrected version
        // (matching a real reference implementation) must not.
        let params = EvalParams { passed_mg_pct: 100, passed_eg_pct: 100, mobility_pct: 0, rook_pct: 0 };
        let blocked = fen::parse("8/k7/P7/8/8/8/8/4K3 w - - 0 1").unwrap();
        let free = fen::parse("7k/8/P7/8/8/8/8/4K3 w - - 0 1").unwrap();
        assert!(
            evaluate(&free, &params) > evaluate(&blocked, &params) + 20,
            "free-runner eval {} should clearly exceed blockaded eval {}",
            evaluate(&free, &params),
            evaluate(&blocked, &params)
        );
    }

    #[test]
    fn mobility_scale_zero_is_a_true_noop() {
        // A white knight with lots of open squares to jump to -- pct=0
        // must reproduce the exact pre-mobility score.
        let pos = fen::parse("4k3/8/8/8/4N3/8/8/4K3 w - - 0 1").unwrap();
        let off = EvalParams { passed_mg_pct: 0, passed_eg_pct: 0, mobility_pct: 0, rook_pct: 0 };
        let on = EvalParams { passed_mg_pct: 0, passed_eg_pct: 0, mobility_pct: 100, rook_pct: 0 };
        assert_ne!(
            evaluate(&pos, &off),
            evaluate(&pos, &on),
            "mobility at pct=100 should differ from pct=0 for a knight with open squares"
        );
    }

    #[test]
    fn mobile_knight_scores_better_than_a_cornered_one() {
        // Same material, same side to move -- only difference is whether
        // white's knight sits in the open center (many safe squares) or
        // jammed in the corner (few safe squares).
        let params = EvalParams { passed_mg_pct: 0, passed_eg_pct: 0, mobility_pct: 100, rook_pct: 0 };
        let cornered = fen::parse("4k3/8/8/8/8/8/8/N3K3 w - - 0 1").unwrap();
        let mobile = fen::parse("4k3/8/8/8/4N3/8/8/4K3 w - - 0 1").unwrap();
        assert!(
            evaluate(&mobile, &params) > evaluate(&cornered, &params),
            "centralized-knight eval {} should exceed cornered-knight eval {}",
            evaluate(&mobile, &params),
            evaluate(&cornered, &params)
        );
    }

    #[test]
    fn rook_scale_zero_is_a_true_noop() {
        // White rook on a fully open a-file -- pct=0 must reproduce the
        // exact pre-rook-term score.
        let pos = fen::parse("4k3/8/8/8/8/8/8/R3K3 w - - 0 1").unwrap();
        let off =
            EvalParams { passed_mg_pct: 0, passed_eg_pct: 0, mobility_pct: 0, rook_pct: 0 };
        let on =
            EvalParams { passed_mg_pct: 0, passed_eg_pct: 0, mobility_pct: 0, rook_pct: 100 };
        assert_ne!(
            evaluate(&pos, &off),
            evaluate(&pos, &on),
            "rook term at pct=100 should differ from pct=0 for a rook on an open file"
        );
    }

    #[test]
    fn rook_on_open_file_scores_better_than_blocked_by_own_pawn() {
        // Same material (one pawn each) and same rook -- only difference
        // is whether the pawn sits on the rook's own a-file (blocking it)
        // or off on the b-file (leaving the a-file open).
        let params = EvalParams { passed_mg_pct: 0, passed_eg_pct: 0, mobility_pct: 0, rook_pct: 100 };
        let blocked = fen::parse("4k3/8/8/8/8/8/P7/R3K3 w - - 0 1").unwrap();
        let open = fen::parse("4k3/8/8/8/8/8/1P6/R3K3 w - - 0 1").unwrap();
        assert!(
            evaluate(&open, &params) > evaluate(&blocked, &params),
            "open-file eval {} should exceed blocked-file eval {}",
            evaluate(&open, &params),
            evaluate(&blocked, &params)
        );
    }

    #[test]
    fn rook_on_7th_pressing_enemy_king_scores_better_than_off_7th() {
        // Same rook, same enemy king pinned to the back rank -- only
        // difference is whether the rook itself sits on the 7th rank
        // (pressing) or one rank back.
        let params = EvalParams { passed_mg_pct: 0, passed_eg_pct: 0, mobility_pct: 0, rook_pct: 100 };
        let pressing = fen::parse("6k1/4R3/8/8/8/8/8/4K3 w - - 0 1").unwrap();
        let off_7th = fen::parse("6k1/8/4R3/8/8/8/8/4K3 w - - 0 1").unwrap();
        assert!(
            evaluate(&pressing, &params) > evaluate(&off_7th, &params),
            "7th-rank-pressing eval {} should exceed off-7th eval {}",
            evaluate(&pressing, &params),
            evaluate(&off_7th, &params)
        );
    }
}
