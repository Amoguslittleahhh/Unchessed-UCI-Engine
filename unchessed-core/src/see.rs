//! Static Exchange Evaluation: the net material gain (in centipawns) of a
//! move once all recaptures on the destination square are played out in
//! least-valuable-attacker order, for both sides. Replaces MVV-LVA as the
//! basis for capture ordering and quiescence pruning — MVV-LVA only looks at
//! the immediate exchange, so it misorders any capture that loses material
//! once the full recapture sequence is considered (e.g. QxP defended by a
//! pawn), which SEE gets right.

use crate::board::*;
use crate::eval::MG_VALUE;
use crate::movegen::{bishop_att, pinned_blockers, rook_att, KING_ATT, KNIGHT_ATT, PAWN_ATT};

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
/// Both colors' pin information for a position.
///
/// SEE needs this, but it depends only on the position -- not on the move
/// being evaluated -- so a caller scoring many moves from the same node can
/// compute it once with [`Pins::new`] and pass it to [`see_with_pins`]
/// instead of paying for four slider-attack scans per capture.
#[derive(Clone, Copy)]
pub struct Pins {
    white_blockers: Bitboard,
    white_pinners: Bitboard,
    black_blockers: Bitboard,
    black_pinners: Bitboard,
}

/// Lazily-computed [`Pins`] for one node.
///
/// Most nodes score a mix of captures and quiets, and a node whose move list
/// happens to contain no captures or promotions never needs pin information
/// at all. Deferring the scan until the first capture is scored keeps the
/// hoist a pure win instead of adding work to quiet-only nodes.
pub struct LazyPins<'a> {
    pos: &'a Position,
    pins: std::cell::OnceCell<Pins>,
}

impl<'a> LazyPins<'a> {
    #[inline]
    pub fn new(pos: &'a Position) -> LazyPins<'a> {
        LazyPins {
            pos,
            pins: std::cell::OnceCell::new(),
        }
    }

    #[inline]
    pub fn get(&self) -> &Pins {
        self.pins.get_or_init(|| Pins::new(self.pos))
    }
}

impl Pins {
    #[inline]
    pub fn new(pos: &Position) -> Pins {
        let (white_blockers, white_pinners) = pinned_blockers(pos, Color::White);
        let (black_blockers, black_pinners) = pinned_blockers(pos, Color::Black);
        Pins {
            white_blockers,
            white_pinners,
            black_blockers,
            black_pinners,
        }
    }
}

pub fn see(pos: &Position, m: Move) -> i32 {
    see_with_pins(pos, m, &Pins::new(pos))
}

/// [`see`], but reusing pin information already computed for this position.
///
/// `pins` MUST have been built from `pos` -- passing pins from a different
/// position would silently produce a wrong exchange value.
pub fn see_with_pins(pos: &Position, m: Move, pins: &Pins) -> i32 {
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

    // Pin info computed once from the STATIC position (an accepted
    // approximation matching real Stockfish, which does the same rather
    // than recomputing pins as pieces are removed mid-exchange -- see
    // Stockfish commit 242c566, which explicitly documents this tradeoff:
    // some positions get a slightly less accurate SEE in exchange for a
    // cheap, single computation). A pinned piece can't be used to continue
    // an exchange on `to` unless doing so wouldn't actually expose its own
    // king (not checked here, matching Stockfish's own simplification) --
    // we just exclude it outright while its pinner is still on the board.
    let Pins {
        white_blockers,
        white_pinners,
        black_blockers,
        black_pinners,
    } = *pins;

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
        let (blockers, pinners) = if side == Color::White {
            (white_blockers, white_pinners)
        } else {
            (black_blockers, black_pinners)
        };
        let pin_mask = if pinners & occ != 0 { !blockers } else { !0u64 };
        let mut found = None;
        for pt in [PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING] {
            let bb = attackers & pos.bb[side.idx()][pt] & occ & pin_mask;
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

    /// Hoisting the pin scan out of `see` must not change any value.
    ///
    /// `Pins` depends only on the position, so `see` (which builds it
    /// internally) and `see_with_pins` (which receives a shared one) must
    /// agree on every legal move of every position -- including the pinned
    /// case, which is exactly the one the shared struct could break.
    #[test]
    fn shared_pins_match_per_call_pins() {
        let fens = [
            "4k3/8/8/8/3n4/8/8/3RK3 w - - 0 1",
            "4k3/8/8/2p5/3n4/8/8/3RK3 w - - 0 1",
            "4k3/8/8/4b3/3n4/8/8/K2RQ3 w - - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/2pP4/1p2P3/2N2N2/PPQBBPPP/R3K2R w KQkq - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            "rnbqkbnr/pp2pppp/2p5/3p4/3PP3/8/PPP2PPP/RNBQKBNR w KQkq - 0 3",
        ];
        for fen_text in fens {
            let pos = fen::parse(fen_text).unwrap();
            let pins = Pins::new(&pos);
            let lazy = LazyPins::new(&pos);
            for m in crate::movegen::legal(&pos).as_slice() {
                let per_call = see(&pos, *m);
                assert_eq!(
                    per_call,
                    see_with_pins(&pos, *m, &pins),
                    "shared Pins differ on {fen_text}"
                );
                assert_eq!(
                    per_call,
                    see_with_pins(&pos, *m, lazy.get()),
                    "LazyPins differ on {fen_text}"
                );
            }
        }
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
    fn pinned_recapturer_is_excluded_from_the_exchange() {
        // Black bishop e5 sits between white queen e1 and black king e8 on
        // the open e-file -- a genuine pin, so it cannot legally recapture
        // on d4 without exposing its own king, even though it geometrically
        // attacks d4. White rook takes the undefended-except-for-the-pin
        // knight on d4; the correct SEE is a clean, un-recaptured knight
        // gain, not knight-minus-rook (which is what a pin-unaware
        // attacker search would compute).
        let pos = fen::parse("4k3/8/8/4b3/3n4/8/8/K2RQ3 w - - 0 1").unwrap();
        let sc = see(&pos, mv("d1", "d4"));
        assert_eq!(sc, MG_VALUE[KNIGHT]);
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
