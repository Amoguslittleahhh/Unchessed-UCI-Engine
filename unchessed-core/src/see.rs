//! Static Exchange Evaluation: the net material gain (in centipawns) of a
//! move once all recaptures on the destination square are played out in
//! least-valuable-attacker order, for both sides. Replaces MVV-LVA as the
//! basis for capture ordering and quiescence pruning — MVV-LVA only looks at
//! the immediate exchange, so it misorders any capture that loses material
//! once the full recapture sequence is considered (e.g. QxP defended by a
//! pawn), which SEE gets right.

use crate::board::*;
use crate::eval::MG_VALUE;
use crate::movegen::{bishop_att, rook_att, KING_ATT, KNIGHT_ATT, PAWN_ATT};

/// All pieces (either color) currently attacking `to`, given occupancy `occ`.
/// `occ` is expected to be a subset of the real position's occupancy (pieces
/// are removed from it as the simulated exchange proceeds), so sliding
/// attacks are recomputed fresh each call to pick up any x-ray attacker
/// revealed by a piece that was just "captured".
fn attackers_to(pos: &Position, to: u8, occ: Bitboard) -> Bitboard {
    let mut att = 0u64;
    att |= PAWN_ATT[Color::Black.idx()][to as usize] & pos.bb[Color::White.idx()][PAWN];
    att |= PAWN_ATT[Color::White.idx()][to as usize] & pos.bb[Color::Black.idx()][PAWN];
    att |= KNIGHT_ATT[to as usize] & (pos.bb[Color::White.idx()][KNIGHT] | pos.bb[Color::Black.idx()][KNIGHT]);
    att |= KING_ATT[to as usize] & (pos.bb[Color::White.idx()][KING] | pos.bb[Color::Black.idx()][KING]);
    let diag = pos.bb[Color::White.idx()][BISHOP]
        | pos.bb[Color::White.idx()][QUEEN]
        | pos.bb[Color::Black.idx()][BISHOP]
        | pos.bb[Color::Black.idx()][QUEEN];
    if diag != 0 {
        att |= bishop_att(to, occ) & diag;
    }
    let ortho = pos.bb[Color::White.idx()][ROOK]
        | pos.bb[Color::White.idx()][QUEEN]
        | pos.bb[Color::Black.idx()][ROOK]
        | pos.bb[Color::Black.idx()][QUEEN];
    if ortho != 0 {
        att |= rook_att(to, occ) & ortho;
    }
    att & occ
}

/// Static Exchange Evaluation for pseudo-legal move `m` from the perspective
/// of the side making it: positive means the full capture/recapture sequence
/// nets material, negative means it loses material. Quiet non-promoting
/// moves are worth 0 (no exchange to evaluate). Castling is never an
/// exchange, also 0.
pub fn see(pos: &Position, m: Move) -> i32 {
    if m.kind() == MK_CASTLE {
        return 0;
    }
    let from = m.from();
    let to = m.to();
    let is_ep = m.kind() == MK_EP;

    let orig_pt = match pos.piece_on(from) {
        Some((_, p)) => p,
        None => return 0,
    };

    let mut side = pos.side;
    let mut occ = pos.occ;

    let mut gain = [0i32; 32];
    let mut d = 0usize;
    gain[0] = if is_ep {
        MG_VALUE[PAWN]
    } else {
        pos.piece_on(to).map(|(_, p)| MG_VALUE[p]).unwrap_or(0)
    };

    // value of the piece that ends up sitting on `to` after this move —
    // the promoted piece for a promotion, otherwise the moving piece itself.
    let mut cur_val = if m.is_promo() {
        MG_VALUE[m.promo_piece()]
    } else {
        MG_VALUE[orig_pt]
    };

    if is_ep {
        let cap_sq = if side == Color::White { to - 8 } else { to + 8 };
        occ &= !(1u64 << cap_sq);
    }
    occ &= !(1u64 << from);
    side = side.flip();

    loop {
        d += 1;
        gain[d] = cur_val - gain[d - 1];
        // No early-exit pruning here: a naive "this ply looks bad, stop
        // searching for further attackers" check discards a ply whose
        // *own* capture may still be forced good by a later backward
        // minimax step (e.g. a losing-looking recapture that's rescued by
        // an x-ray piece behind it) — got this wrong once already (see the
        // three_ply_exchange_with_xray_recapture test below), so the
        // forward pass always runs to exhaustion instead. Exchanges are
        // short in practice; the `d >= 31` cap is just a hard safety bound.
        if d >= 31 {
            break;
        }
        let attackers = attackers_to(pos, to, occ);
        let mut found = None;
        for pt in [PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING] {
            let bb = attackers & pos.bb[side.idx()][pt] & occ;
            if bb != 0 {
                found = Some((bb.trailing_zeros() as u8, pt));
                break;
            }
        }
        let (sq, pt) = match found {
            Some(x) => x,
            None => break,
        };
        occ &= !(1u64 << sq);
        cur_val = MG_VALUE[pt];
        side = side.flip();
    }

    // Backward minimax pass: mirrors the reference algorithm's `while (--d)`
    // loop exactly — note this means the body never runs at all when the
    // forward pass only ever reached d==1 (a plain undefended capture with
    // no recapture), so gain[0] is returned unmodified in that case.
    while d > 1 {
        d -= 1;
        gain[d - 1] = -((-gain[d - 1]).max(gain[d]));
    }
    gain[0]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Move;
    use crate::fen;

    fn mv(from: &str, to: &str) -> Move {
        let f = square_from_str(from);
        let t = square_from_str(to);
        Move::new(f, t, MK_NORMAL)
    }

    fn square_from_str(s: &str) -> u8 {
        let bytes = s.as_bytes();
        let file = bytes[0] - b'a';
        let rank = bytes[1] - b'1';
        rank * 8 + file
    }

    #[test]
    fn undefended_capture_nets_full_victim_value() {
        // White rook d1, black knight d4 (undefended), kings out of the way.
        let pos = fen::parse("4k3/8/8/8/3n4/8/8/3RK3 w - - 0 1").unwrap();
        let sc = see(&pos, mv("d1", "d4"));
        assert_eq!(sc, MG_VALUE[KNIGHT]);
    }

    #[test]
    fn losing_capture_pawn_recaptures_rook() {
        // White rook d1 takes knight d4, but a black pawn on c5 recaptures
        // on d4 with nothing else attacking d4 — a losing trade for White.
        let pos = fen::parse("4k3/8/8/2p5/3n4/8/8/3RK3 w - - 0 1").unwrap();
        let sc = see(&pos, mv("d1", "d4"));
        assert_eq!(sc, MG_VALUE[KNIGHT] - MG_VALUE[ROOK]);
    }

    #[test]
    fn three_ply_exchange_with_xray_recapture() {
        // White rook d2 takes knight d4; black pawn c5 recaptures the rook;
        // white queen d1 (behind the rook, revealed as an x-ray attacker
        // once the rook leaves d2) recaptures the pawn. Net for white:
        // +knight(337) - rook(477) + pawn(82) = -58. Both sides prefer to
        // continue the trade at each step (verified by hand in the
        // implementation notes), so this exercises both the x-ray reveal
        // and the backward minimax pass across more than one recapture.
        let pos = fen::parse("4k3/8/8/2p5/3n4/8/3R4/K2Q4 w - - 0 1").unwrap();
        let sc = see(&pos, mv("d2", "d4"));
        assert_eq!(sc, MG_VALUE[KNIGHT] - MG_VALUE[ROOK] + MG_VALUE[PAWN]);
    }
}
