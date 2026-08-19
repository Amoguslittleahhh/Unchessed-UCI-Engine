//! Compact, project-native threat-relation features for the next NNUE format.
//!
//! Stockfish SFNNv10 demonstrated that attacked-piece relations can improve a
//! high-end NNUE. Copying its 79,856 x 1,024 transformer would be a poor fit
//! for Unchessed's latency and model-size budget. This module instead defines
//! a 32,400-dimensional multiset of relative threat relations suitable for a
//! low-rank 32-wide residual accumulator:
//!
//!   attacker class (12) x target class (12) x relative delta (15 x 15)
//!
//! Classes are perspective-relative `(piece_type, own/opponent)`. Squares use
//! the same vertical and king-file orientation as HalfKAv2_hm. Repeated indices
//! are intentional: two geometrically identical relations count twice. At a
//! 32-wide int16 embedding this table occupies ~2.0 MiB and a full refresh is
//! at most 256 x 32 additions, small enough to benchmark before implementing
//! complex dirty-ray updates. It is research infrastructure only; the shipped
//! v3 evaluator does not consume these features yet.

use crate::board::*;
use crate::movegen::{bishop_att, queen_att, rook_att, KING_ATT, KNIGHT_ATT, PAWN_ATT};

pub const THREAT_CLASSES: usize = 12;
pub const THREAT_RELATIONS: usize = 15 * 15;
pub const THREAT_DIMENSIONS: usize = THREAT_CLASSES * THREAT_CLASSES * THREAT_RELATIONS;
pub const THREAT_LATENT_WIDTH: usize = 32;
pub const MAX_ACTIVE_THREATS: usize = 256;

#[derive(Clone)]
pub struct ThreatFeatureList {
    indices: [u16; MAX_ACTIVE_THREATS],
    len: usize,
    overflowed: bool,
}

impl Default for ThreatFeatureList {
    fn default() -> Self {
        Self {
            indices: [0; MAX_ACTIVE_THREATS],
            len: 0,
            overflowed: false,
        }
    }
}

impl ThreatFeatureList {
    pub fn as_slice(&self) -> &[u16] {
        &self.indices[..self.len]
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn overflowed(&self) -> bool {
        self.overflowed
    }

    fn push(&mut self, index: usize) {
        debug_assert!(index < THREAT_DIMENSIONS);
        if self.len == MAX_ACTIVE_THREATS {
            self.overflowed = true;
            return;
        }
        self.indices[self.len] = index as u16;
        self.len += 1;
    }
}

#[inline]
pub fn attacks_from(pos: &Position, color: Color, piece: usize, square: usize) -> Bitboard {
    match piece {
        PAWN => PAWN_ATT[color.idx()][square],
        KNIGHT => KNIGHT_ATT[square],
        BISHOP => bishop_att(square as u8, pos.occ),
        ROOK => rook_att(square as u8, pos.occ),
        QUEEN => queen_att(square as u8, pos.occ),
        KING => KING_ATT[square],
        _ => 0,
    }
}

/// HalfKAv2_hm-compatible square orientation for one perspective.
fn orientation(pos: &Position, perspective: Color) -> impl Fn(usize) -> usize {
    let white = perspective == Color::White;
    let king_raw = pos.king_sq(perspective) as usize;
    let king_vertical = if white { king_raw } else { king_raw ^ 56 };
    let mirror_file = king_vertical % 8 < 4;
    move |square: usize| {
        let vertical = if white { square } else { square ^ 56 };
        if mirror_file {
            vertical ^ 7
        } else {
            vertical
        }
    }
}

#[inline]
fn piece_class(piece: usize, color: Color, perspective: Color) -> usize {
    piece * 2 + usize::from(color != perspective)
}

#[inline]
fn relation_index(
    attacker_class: usize,
    target_class: usize,
    attacker_square: usize,
    target_square: usize,
) -> usize {
    let af = (attacker_square & 7) as i32;
    let ar = (attacker_square >> 3) as i32;
    let tf = (target_square & 7) as i32;
    let tr = (target_square >> 3) as i32;
    let delta_file = (tf - af + 7) as usize;
    let delta_rank = (tr - ar + 7) as usize;
    let relation = delta_rank * 15 + delta_file;
    (attacker_class * THREAT_CLASSES + target_class) * THREAT_RELATIONS + relation
}

/// Enumerate attacks and defenses whose target square is occupied.
///
/// The returned list is a multiset. Callers must reject `overflowed()` rather
/// than silently training/inferencing on truncated features. No legal position
/// in the regression corpus reaches the 256-relation bound.
pub fn active_threat_features(pos: &Position, perspective: Color) -> ThreatFeatureList {
    let orient = orientation(pos, perspective);
    let mut output = ThreatFeatureList::default();

    for attacker_color in [Color::White, Color::Black] {
        for attacker_piece in 0..6 {
            let mut attackers = pos.bb[attacker_color.idx()][attacker_piece];
            while attackers != 0 {
                let from = attackers.trailing_zeros() as usize;
                attackers &= attackers - 1;
                let mut targets = attacks_from(pos, attacker_color, attacker_piece, from) & pos.occ;
                while targets != 0 {
                    let to = targets.trailing_zeros() as usize;
                    targets &= targets - 1;
                    if let Some((target_color, target_piece)) = pos.piece_on(to as u8) {
                        let index = relation_index(
                            piece_class(attacker_piece, attacker_color, perspective),
                            piece_class(target_piece, target_color, perspective),
                            orient(from),
                            orient(to),
                        );
                        output.push(index);
                    }
                }
            }
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fen;

    #[test]
    fn threat_indices_are_bounded_and_stack_resident() {
        let pos = fen::startpos();
        for perspective in [Color::White, Color::Black] {
            let features = active_threat_features(&pos, perspective);
            assert!(!features.is_empty());
            assert!(!features.overflowed());
            assert!(features.len() <= MAX_ACTIVE_THREATS);
            assert!(features
                .as_slice()
                .iter()
                .all(|&index| (index as usize) < THREAT_DIMENSIONS));
        }
    }

    #[test]
    fn bishop_attack_relation_has_expected_relative_encoding() {
        let pos = fen::parse("4k3/8/8/3p4/2B5/8/8/4K3 w - - 0 1").unwrap();
        let features = active_threat_features(&pos, Color::White);
        // White bishop (class 4) on c4 attacks black pawn (class 1) on d5.
        // Relative delta (+1,+1) maps to row 8, column 8 in the 15x15 grid.
        let expected = ((4 * 12 + 1) * 225 + 8 * 15 + 8) as u16;
        assert!(features.as_slice().contains(&expected));
    }

    #[test]
    fn feature_multiset_is_deterministic() {
        let pos =
            fen::parse("r3k2r/ppp2ppp/2n1bn2/3qp3/3P4/2N1BN2/PPP2PPP/R2Q1RK1 b kq - 7 13").unwrap();
        let a = active_threat_features(&pos, Color::White);
        let b = active_threat_features(&pos, Color::White);
        assert_eq!(a.as_slice(), b.as_slice());
        assert!(!a.overflowed());
    }

    #[test]
    #[ignore = "microbenchmark; run explicitly in release mode"]
    fn threat_feature_microbench() {
        let fens = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            "r2q1rk1/pp2bppp/2npbn2/2p1p3/4P3/2NP1N2/PPPQBPPP/R1B2RK1 w - - 4 9",
            "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
            "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
            "r1bq1rk1/pp2bppp/2np1n2/2p1p3/4P3/2PP1N2/PP1N1PPP/R1BQR1K1 w - - 4 9",
            "3r2k1/pp3ppp/2p1b3/4P3/1q1P1Q2/2R5/PP3PPP/4R1K1 w - - 0 22",
            "8/5pk1/4p1p1/3pP2p/3P3P/5KP1/5P2/4R3 w - - 0 40",
            "8/5pk1/4p1p1/3pP2p/3P3P/5KP1/5P2/8 w - - 0 40",
            "r3k2r/ppp2ppp/2n1bn2/3pp3/8/2N1PN2/PPPPBPPP/R3K2R w KQkq - 6 9",
            "r1bq1rk1/ppp2ppp/2np1n2/4p1B1/2B1P3/2NP1N2/PPP2PPP/R2Q1RK1 w - - 0 9",
        ];
        let positions: Vec<_> = fens
            .iter()
            .enumerate()
            .map(|(index, value)| {
                fen::parse(value).unwrap_or_else(|error| panic!("FEN {index}: {error}"))
            })
            .collect();
        let iterations = 20_000usize;
        let mut relations = 0usize;
        for pos in &positions {
            for perspective in [Color::White, Color::Black] {
                let features = active_threat_features(pos, perspective);
                assert!(!features.overflowed());
                relations += features.len();
            }
        }
        let calls = iterations * positions.len() * 2;

        let weights: Vec<i16> = (0..THREAT_DIMENSIONS * THREAT_LATENT_WIDTH)
            .map(|index| ((index.wrapping_mul(17) % 31) as i16) - 15)
            .collect();
        let fused_started = std::time::Instant::now();
        let mut checksum = 0i16;
        for _ in 0..iterations {
            for pos in &positions {
                for perspective in [Color::White, Color::Black] {
                    let features = std::hint::black_box(active_threat_features(pos, perspective));
                    let mut accumulator = [0i16; THREAT_LATENT_WIDTH];
                    for &index in features.as_slice() {
                        let row = &weights[index as usize * THREAT_LATENT_WIDTH
                            ..(index as usize + 1) * THREAT_LATENT_WIDTH];
                        for (value, &weight) in accumulator.iter_mut().zip(row) {
                            *value = value.wrapping_add(weight);
                        }
                    }
                    let accumulator = std::hint::black_box(accumulator);
                    checksum = checksum.wrapping_add(accumulator[0]);
                }
            }
        }
        let fused_elapsed = fused_started.elapsed();
        println!(
            "threat residual prototype: {:.1} ns/call, {:.1} active relations/call ({checksum})",
            fused_elapsed.as_nanos() as f64 / calls as f64,
            relations as f64 / (positions.len() * 2) as f64,
        );
    }
}
