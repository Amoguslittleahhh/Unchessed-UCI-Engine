//! Original, opt-in tactical and terminal heads for SFNNv16-comparable research.
//!
//! This module uses public architectural principles only: sparse legal-state
//! signals, bounded integer-friendly inference, and separate score, WDL, and
//! terminal outputs. It contains no Stockfish source, weights, or feature rows.

use crate::board::{Color, Position, BISHOP, KNIGHT, MK_EP, QUEEN, ROOK};
use crate::movegen::{in_check, legal, pinned_blockers};

const SCORE_LIMIT_CP: i32 = 900;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TacticalFeatures {
    pub legal_moves: u16,
    pub checking_moves: u8,
    pub forcing_captures: u8,
    pub king_escape_count: u8,
    pub pinned_piece_count: u8,
    pub promotion_moves: u8,
    pub low_material: bool,
    pub side_in_check: bool,
    /// 0 non-terminal, 1 checkmate, 2 stalemate.
    pub terminal_class: u8,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TacticalHeads {
    pub bounded_static_cp: i32,
    pub tactical_delta_cp: i32,
    pub terminal_per_mille: u16,
    pub mate_distance: i8,
    pub confidence_per_mille: u16,
    pub features: TacticalFeatures,
}

/// Extract all tactical signals from one legal move list. This explicit path is
/// kept out of the normal search evaluator because legal generation is expensive.
pub fn extract_features(pos: &Position) -> TacticalFeatures {
    let moves = legal(pos);
    let side_in_check = in_check(pos);
    let non_pawn_material = (0..2)
        .flat_map(|c| {
            [KNIGHT, BISHOP, ROOK, QUEEN]
                .into_iter()
                .map(move |p| pos.bb[c][p].count_ones())
        })
        .sum::<u32>();
    let mut out = TacticalFeatures {
        legal_moves: moves.len as u16,
        low_material: non_pawn_material <= 4,
        side_in_check,
        pinned_piece_count: pinned_blockers(pos, pos.side).0.count_ones().min(255) as u8,
        ..TacticalFeatures::default()
    };
    for &mv in moves.as_slice() {
        let next = pos.make(mv);
        if in_check(&next) {
            out.checking_moves = out.checking_moves.saturating_add(1);
        }
        let capture = mv.kind() == MK_EP || pos.piece_on(mv.to()).is_some();
        if capture && (in_check(&next) || pos.piece_on(mv.to()).map_or(false, |(_, p)| p >= ROOK)) {
            out.forcing_captures = out.forcing_captures.saturating_add(1);
        }
        if mv.is_promo() {
            out.promotion_moves = out.promotion_moves.saturating_add(1);
        }
        if side_in_check
            && pos
                .piece_on(mv.from())
                .is_some_and(|(_, p)| p == crate::board::KING)
        {
            out.king_escape_count = out.king_escape_count.saturating_add(1);
        }
    }
    if moves.len == 0 {
        out.terminal_class = if side_in_check { 1 } else { 2 };
    }
    out
}

/// Apply an independently authored gated tactical head to an existing score.
/// Terminal, mate-distance, and confidence heads remain separate outputs.
pub fn heads_from_score(base_stm_cp: i32, pos: &Position) -> TacticalHeads {
    let features = extract_features(pos);
    let escape_term = if features.side_in_check {
        24 - i32::from(features.king_escape_count) * 7
    } else {
        0
    };
    let pressure = i32::from(features.checking_moves) * 15
        + i32::from(features.forcing_captures) * 6
        + i32::from(features.promotion_moves) * 9
        + i32::from(features.pinned_piece_count) * 3
        + escape_term;
    let gate = if features.terminal_class != 0 {
        1000
    } else if features.side_in_check || features.checking_moves > 0 || features.low_material {
        620
    } else {
        180
    };
    let delta = (pressure * gate / 1000).clamp(-SCORE_LIMIT_CP, SCORE_LIMIT_CP);
    let terminal = match features.terminal_class {
        1 => 1000,
        2 => 0,
        _ => (u16::from(features.checking_moves) * 55
            + u16::from(features.forcing_captures) * 25
            + if features.side_in_check { 120 } else { 0 })
        .min(900),
    };
    let confidence = (350
        + u16::from(features.checking_moves) * 90
        + u16::from(features.forcing_captures) * 40
        + if features.side_in_check { 180 } else { 0 }
        + if features.low_material { 60 } else { 0 })
    .min(1000);
    TacticalHeads {
        bounded_static_cp: base_stm_cp.saturating_add(delta),
        tactical_delta_cp: delta,
        terminal_per_mille: terminal,
        mate_distance: if features.terminal_class == 1 {
            0
        } else if features.checking_moves > 0 {
            1
        } else {
            -1
        },
        confidence_per_mille: confidence,
        features,
    }
}

pub fn white_score(base_stm_cp: i32, pos: &Position) -> (i32, TacticalHeads) {
    let heads = heads_from_score(base_stm_cp, pos);
    (
        if pos.side == Color::White {
            heads.bounded_static_cp
        } else {
            -heads.bounded_static_cp
        },
        heads,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fen;

    #[test]
    fn quiet_start_position_has_no_terminal_claim() {
        let f = extract_features(&fen::startpos());
        assert!(!f.side_in_check && f.terminal_class == 0 && f.legal_moves > 0);
    }

    #[test]
    fn mate_and_stalemate_are_separate_terminal_classes() {
        let mate = fen::parse("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1").unwrap();
        let stale = fen::parse("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1").unwrap();
        assert_eq!(extract_features(&mate).terminal_class, 1);
        assert_eq!(extract_features(&stale).terminal_class, 2);
    }

    #[test]
    fn forcing_position_gets_a_confidence_signal() {
        let pos = fen::parse("6k1/8/5P2/8/8/8/8/4R1K1 w - - 0 1").unwrap();
        let heads = heads_from_score(0, &pos);
        assert!(heads.confidence_per_mille > 350);
        assert!(heads.tactical_delta_cp.abs() <= SCORE_LIMIT_CP);
    }
}

/// The runtime path is intentionally opt-in until training, untouched-data
/// validation, speed measurement, and a powered fixed-control match are complete.
pub const PROVENANCE: &str =
    "original clean-room tactical heads; research-only; not stable SFNNv16 parity";
