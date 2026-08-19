//! CPU-oriented Chessformer input contract and persona/search router.
//!
//! This is deliberately model-agnostic infrastructure. It fixes the 64-square
//! tokenization, dynamic geometric relation keys, continuous-Elo context, and
//! safe backend-routing semantics before any trained transformer asset exists.
//! A later `UNCHFORM` loader can implement a small encoder-only model without
//! changing datasets or persona decisions.

use crate::adapt::Mode;
use crate::board::*;
use crate::threat_features::attacks_from;

pub const BOARD_TOKENS: usize = 64;
pub const EMPTY_TOKEN: u8 = 0;
pub const NO_EP_FILE: u8 = 8;

/// Discrete square token consumed by the proposed tiny Chessformer.
///
/// `piece` is perspective-relative: 0 empty, 1..6 own P..K, 7..12 opponent
/// P..K. Black-to-move positions are rank-flipped so the mover always advances
/// toward increasing ranks. Global state is repeated per token to keep the
/// inference format simple and friendly to int8 embedding lookups.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SquareToken {
    pub piece: u8,
    pub square: u8,
    pub castling: u8,
    pub ep_file: u8,
    pub halfmove_bucket: u8,
    pub elo_context: u8,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChessformerInput {
    pub tokens: [SquareToken; BOARD_TOKENS],
    pub flipped: bool,
}

/// Bit flags for a dynamic source/destination attention bias.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GeometryRelation {
    pub file_delta: i8,
    pub rank_delta: i8,
    pub chebyshev_distance: u8,
    pub flags: u16,
}

impl GeometryRelation {
    pub const SAME_RANK: u16 = 1 << 0;
    pub const SAME_FILE: u16 = 1 << 1;
    pub const SAME_DIAGONAL: u16 = 1 << 2;
    pub const KNIGHT_GEOMETRY: u16 = 1 << 3;
    pub const KING_NEIGHBOR: u16 = 1 << 4;
    pub const SOURCE_ATTACKS_TARGET: u16 = 1 << 5;
    pub const TARGET_ATTACKS_SOURCE: u16 = 1 << 6;
    pub const TARGET_OCCUPIED: u16 = 1 << 7;
    pub const SAME_OWNER: u16 = 1 << 8;

    pub fn has(self, flag: u16) -> bool {
        self.flags & flag != 0
    }
}

fn normalized_castling(pos: &Position) -> u8 {
    if pos.side == Color::White {
        pos.castling & 0x0f
    } else {
        let mut rights = 0;
        if pos.castling & BK != 0 {
            rights |= WK;
        }
        if pos.castling & BQ != 0 {
            rights |= WQ;
        }
        if pos.castling & WK != 0 {
            rights |= BK;
        }
        if pos.castling & WQ != 0 {
            rights |= BQ;
        }
        rights
    }
}

#[inline]
fn orient_square(square: usize, flip: bool) -> usize {
    if flip {
        square ^ 56
    } else {
        square
    }
}

#[inline]
fn original_square(oriented: usize, flip: bool) -> usize {
    orient_square(oriented, flip)
}

/// Encode one position into exactly 64 board-square tokens.
pub fn encode_position(pos: &Position, target_elo: i32) -> ChessformerInput {
    let flip = pos.side == Color::Black;
    let mover = pos.side;
    let castling = normalized_castling(pos);
    let ep_file = if pos.ep == NO_EP {
        NO_EP_FILE
    } else {
        file_of(pos.ep)
    };
    let halfmove_bucket = (pos.halfmove / 8).min(15) as u8;
    let elo_context = (((target_elo.clamp(100, 3650) - 100) * 255) / 3550) as u8;
    let empty = SquareToken {
        piece: EMPTY_TOKEN,
        square: 0,
        castling,
        ep_file,
        halfmove_bucket,
        elo_context,
    };
    let mut tokens = [empty; BOARD_TOKENS];
    for (oriented, token) in tokens.iter_mut().enumerate() {
        let original = original_square(oriented, flip);
        let piece = match pos.piece_on(original as u8) {
            Some((color, kind)) => 1 + kind as u8 + if color == mover { 0 } else { 6 },
            None => EMPTY_TOKEN,
        };
        *token = SquareToken {
            piece,
            square: oriented as u8,
            castling,
            ep_file,
            halfmove_bucket,
            elo_context,
        };
    }
    ChessformerInput {
        tokens,
        flipped: flip,
    }
}

/// Dynamic geometry key for Geometric-Attention-Bias-style lookup/MLP input.
pub fn geometric_relation(
    pos: &Position,
    source_oriented: usize,
    target_oriented: usize,
) -> GeometryRelation {
    debug_assert!(source_oriented < 64 && target_oriented < 64);
    let flip = pos.side == Color::Black;
    let source = original_square(source_oriented, flip);
    let target = original_square(target_oriented, flip);
    let sf = (source_oriented & 7) as i8;
    let sr = (source_oriented >> 3) as i8;
    let tf = (target_oriented & 7) as i8;
    let tr = (target_oriented >> 3) as i8;
    let df = tf - sf;
    let dr = tr - sr;
    let adf = df.unsigned_abs();
    let adr = dr.unsigned_abs();
    let mut flags = 0u16;
    if dr == 0 {
        flags |= GeometryRelation::SAME_RANK;
    }
    if df == 0 {
        flags |= GeometryRelation::SAME_FILE;
    }
    if adf == adr {
        flags |= GeometryRelation::SAME_DIAGONAL;
    }
    if (adf == 1 && adr == 2) || (adf == 2 && adr == 1) {
        flags |= GeometryRelation::KNIGHT_GEOMETRY;
    }
    if adf <= 1 && adr <= 1 && source != target {
        flags |= GeometryRelation::KING_NEIGHBOR;
    }
    let source_piece = pos.piece_on(source as u8);
    let target_piece = pos.piece_on(target as u8);
    if let Some((color, piece)) = source_piece {
        if attacks_from(pos, color, piece, source) & (1u64 << target) != 0 {
            flags |= GeometryRelation::SOURCE_ATTACKS_TARGET;
        }
    }
    if let Some((color, piece)) = target_piece {
        flags |= GeometryRelation::TARGET_OCCUPIED;
        if attacks_from(pos, color, piece, target) & (1u64 << source) != 0 {
            flags |= GeometryRelation::TARGET_ATTACKS_SOURCE;
        }
    }
    if matches!((source_piece, target_piece), (Some((a, _)), Some((b, _))) if a == b) {
        flags |= GeometryRelation::SAME_OWNER;
    }
    GeometryRelation {
        file_delta: df,
        rank_delta: dr,
        chebyshev_distance: adf.max(adr),
        flags,
    }
}

/// Search architecture selected once per move by the persona layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HybridBackend {
    /// Conventional NNUE alpha-beta; authoritative for engines, GMs, defense,
    /// tactical conversion, unknown opponents, and all model-missing cases.
    AlphaBeta,
    /// Chessformer policy orders root/top-ply moves; alpha-beta remains the
    /// value authority and verifies every chosen move.
    PolicyGuidedAlphaBeta,
    /// Sample a human policy at the target Elo, then apply an alpha-beta safety
    /// veto for forced mate and catastrophic tactical loss.
    HumanPolicyGuarded,
}

impl HybridBackend {
    pub fn name(self) -> &'static str {
        match self {
            Self::AlphaBeta => "ALPHABETA",
            Self::PolicyGuidedAlphaBeta => "POLICY_GUIDED_AB",
            Self::HumanPolicyGuarded => "HUMAN_POLICY_GUARDED",
        }
    }
}

/// Conservative routing contract for the future trained Chessformer asset.
///
/// This function has no effect on production play until a caller explicitly
/// wires a verified model. It codifies the safety boundary now so a missing or
/// uncertain model can never weaken play against an engine.
pub fn choose_backend(
    model_available: bool,
    adaptive: bool,
    fixed_strength: bool,
    target_elo: i32,
    mode: Mode,
    confident_human: bool,
    engine_or_uncertain: bool,
) -> HybridBackend {
    if !model_available
        || !adaptive
        || engine_or_uncertain
        || matches!(mode, Mode::Full | Mode::Punish | Mode::Defend)
        || target_elo >= 2300
    {
        return HybridBackend::AlphaBeta;
    }
    if fixed_strength || (confident_human && mode == Mode::Match && target_elo <= 2100) {
        return HybridBackend::HumanPolicyGuarded;
    }
    if confident_human && matches!(mode, Mode::Match | Mode::Clinch) {
        return HybridBackend::PolicyGuidedAlphaBeta;
    }
    HybridBackend::AlphaBeta
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fen;

    #[test]
    fn black_to_move_is_normalized_as_the_mover() {
        let pos = fen::parse("4k3/p7/8/8/8/8/7P/4K3 b - - 3 20").unwrap();
        let input = encode_position(&pos, 1500);
        assert!(input.flipped);
        // Black pawn a7 is mover-owned and rank-flips to a2.
        assert_eq!(input.tokens[8].piece, 1 + PAWN as u8);
        // White pawn h2 becomes the opponent pawn on h7.
        assert_eq!(input.tokens[55].piece, 7 + PAWN as u8);
    }

    #[test]
    fn geometry_captures_dynamic_piece_attacks() {
        let pos = fen::parse("4k3/8/8/3p4/2B5/8/8/4K3 w - - 0 1").unwrap();
        let relation = geometric_relation(&pos, 26, 35); // c4 -> d5
        assert_eq!((relation.file_delta, relation.rank_delta), (1, 1));
        assert!(relation.has(GeometryRelation::SAME_DIAGONAL));
        assert!(relation.has(GeometryRelation::SOURCE_ATTACKS_TARGET));
        assert!(relation.has(GeometryRelation::TARGET_OCCUPIED));
        assert!(!relation.has(GeometryRelation::SAME_OWNER));
    }

    #[test]
    fn router_never_uses_transformer_for_engines_or_gms() {
        assert_eq!(
            choose_backend(true, true, false, 1800, Mode::Match, false, true),
            HybridBackend::AlphaBeta
        );
        assert_eq!(
            choose_backend(true, true, false, 2450, Mode::Match, true, false),
            HybridBackend::AlphaBeta
        );
    }

    #[test]
    fn router_uses_guarded_human_policy_only_in_safe_contexts() {
        assert_eq!(
            choose_backend(true, true, false, 1500, Mode::Match, true, false),
            HybridBackend::HumanPolicyGuarded
        );
        assert_eq!(
            choose_backend(true, true, false, 2200, Mode::Clinch, true, false),
            HybridBackend::PolicyGuidedAlphaBeta
        );
        assert_eq!(
            choose_backend(false, true, false, 1500, Mode::Match, true, false),
            HybridBackend::AlphaBeta
        );
    }
}
