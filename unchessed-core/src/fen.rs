//! FEN parsing and serialization.

use crate::board::*;
use crate::movegen::attacked;

pub const START_FEN: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

pub fn startpos() -> Position {
    parse(START_FEN).expect("startpos FEN must parse")
}

pub fn parse(fen: &str) -> Result<Position, String> {
    let mut pos = Position::empty();
    let mut parts = fen.split_whitespace();

    let placement = parts.next().ok_or("empty FEN")?;
    let mut rank: i32 = 7;
    let mut file: i32 = 0;
    for ch in placement.chars() {
        match ch {
            '/' => {
                if file != 8 {
                    return Err(format!("rank {} has {} files, not 8", rank, file));
                }
                rank -= 1;
                file = 0;
                if rank < 0 {
                    return Err("too many ranks".into());
                }
            }
            '1'..='8' => {
                file += ch as i32 - '0' as i32;
                if file > 8 {
                    return Err(format!("rank overflow at '{}'", ch));
                }
            }
            _ => {
                if file > 7 {
                    return Err(format!("rank overflow at '{}'", ch));
                }
                let color = if ch.is_ascii_uppercase() {
                    Color::White
                } else {
                    Color::Black
                };
                let piece = match ch.to_ascii_lowercase() {
                    'p' => PAWN,
                    'n' => KNIGHT,
                    'b' => BISHOP,
                    'r' => ROOK,
                    'q' => QUEEN,
                    'k' => KING,
                    _ => return Err(format!("bad piece '{}'", ch)),
                };
                let s = sq(file as u8, rank as u8);
                let b = 1u64 << s;
                pos.bb[color.idx()][piece] |= b;
                pos.occ_side[color.idx()] |= b;
                pos.occ |= b;
                pos.board[s as usize] = (color.idx() * 6 + piece) as u8;
                file += 1;
            }
        }
    }
    if file != 8 {
        return Err(format!("rank {} has {} files, not 8", rank, file));
    }
    if rank != 0 {
        return Err(format!("{} ranks given, not 8", 8 - rank));
    }

    pos.side = match parts.next().unwrap_or("w") {
        "w" => Color::White,
        "b" => Color::Black,
        other => return Err(format!("bad side '{}'", other)),
    };

    pos.castling = 0;
    if let Some(c) = parts.next() {
        if c != "-" {
            for ch in c.chars() {
                pos.castling |= match ch {
                    'K' => WK,
                    'Q' => WQ,
                    'k' => BK,
                    'q' => BQ,
                    _ => return Err(format!("bad castling '{}'", ch)),
                };
            }
        }
    }
    // A castling right is meaningless (and downstream code assumes it's
    // backed by a real king+rook pair on their home squares -- movegen
    // builds the castling move purely from the rights bit plus path/attack
    // checks, and make_move() unconditionally relocates "the" king and rook
    // there) unless that king and rook actually exist. Reject up front
    // rather than letting a later synthetic castling move panic trying to
    // pick up a piece that was never placed.
    let king_rook_ok = |king_sq: u8, rook_sq: u8, color: Color| {
        pos.piece_on(king_sq) == Some((color, KING)) && pos.piece_on(rook_sq) == Some((color, ROOK))
    };
    if pos.castling & WK != 0 && !king_rook_ok(sq(4, 0), sq(7, 0), Color::White) {
        return Err("castling right K without king/rook on e1/h1".into());
    }
    if pos.castling & WQ != 0 && !king_rook_ok(sq(4, 0), sq(0, 0), Color::White) {
        return Err("castling right Q without king/rook on e1/a1".into());
    }
    if pos.castling & BK != 0 && !king_rook_ok(sq(4, 7), sq(7, 7), Color::Black) {
        return Err("castling right k without king/rook on e8/h8".into());
    }
    if pos.castling & BQ != 0 && !king_rook_ok(sq(4, 7), sq(0, 7), Color::Black) {
        return Err("castling right q without king/rook on e8/a8".into());
    }

    pos.ep = NO_EP;
    if let Some(e) = parts.next() {
        if e != "-" {
            let ep_sq = parse_sq(e).ok_or_else(|| format!("bad ep '{}'", e))?;
            // The ep target is the square a double-moving pawn passed over,
            // so it must sit on rank 3 (mover was Black) or rank 6 (mover
            // was White), and that mover's pawn must actually be standing
            // one rank behind the target -- otherwise movegen can emit an
            // "en passant" capture with no real pawn behind it, which
            // board.rs's make_move() removes unconditionally, corrupting
            // occupancy/hash instead of erroring.
            let (expected_rank, pawn_rank, mover) = match pos.side {
                Color::White => (5u8, 4u8, Color::Black),
                Color::Black => (2u8, 3u8, Color::White),
            };
            if rank_of(ep_sq) != expected_rank {
                return Err(format!("ep square '{}' not on the rank a double move passes over", e));
            }
            let pawn_sq = sq(file_of(ep_sq), pawn_rank);
            if pos.piece_on(pawn_sq) != Some((mover, PAWN)) {
                return Err(format!("ep square '{}' has no pawn to capture", e));
            }
            if pos.piece_on(ep_sq).is_some() {
                return Err(format!("ep square '{}' is occupied", e));
            }
            pos.ep = ep_sq;
        }
    }

    pos.halfmove = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0);
    pos.fullmove = parts.next().and_then(|s| s.parse().ok()).unwrap_or(1);

    if pos.bb[0][KING].count_ones() != 1 || pos.bb[1][KING].count_ones() != 1 {
        return Err("each side needs exactly one king".into());
    }

    // The side NOT to move must not be in check -- otherwise the side to move
    // could simply capture that king, which is impossible in a real game (the
    // opponent would have had to already be in check on their own turn and
    // failed to address it). This also catches the degenerate "adjacent
    // kings" case, since a king itself attacks all its neighboring squares.
    // Without this check, downstream code that assumes every king square is
    // a normal, uncapturable king (e.g. NNUE king-bucket lookups) can be fed
    // a position where the "opponent" king is effectively already gone,
    // which previously caused an out-of-bounds panic instead of a UCI error.
    let non_mover = pos.side.flip();
    let non_mover_king = pos.bb[non_mover.idx()][KING].trailing_zeros() as u8;
    if attacked(&pos, non_mover_king, pos.side) {
        return Err("illegal position: side not to move is in check".into());
    }

    pos.hash = pos.compute_hash();
    Ok(pos)
}

pub fn serialize(pos: &Position) -> String {
    let chars = ['P', 'N', 'B', 'R', 'Q', 'K'];
    let mut out = String::new();
    for r in (0..8u8).rev() {
        let mut empty = 0;
        for f in 0..8u8 {
            match pos.piece_on(sq(f, r)) {
                Some((c, p)) => {
                    if empty > 0 {
                        out.push_str(&empty.to_string());
                        empty = 0;
                    }
                    let ch = chars[p];
                    out.push(if let Color::White = c {
                        ch
                    } else {
                        ch.to_ascii_lowercase()
                    });
                }
                None => empty += 1,
            }
        }
        if empty > 0 {
            out.push_str(&empty.to_string());
        }
        if r > 0 {
            out.push('/');
        }
    }
    out.push(' ');
    out.push(if let Color::White = pos.side { 'w' } else { 'b' });
    out.push(' ');
    if pos.castling == 0 {
        out.push('-');
    } else {
        if pos.castling & WK != 0 {
            out.push('K');
        }
        if pos.castling & WQ != 0 {
            out.push('Q');
        }
        if pos.castling & BK != 0 {
            out.push('k');
        }
        if pos.castling & BQ != 0 {
            out.push('q');
        }
    }
    out.push(' ');
    if pos.ep == NO_EP {
        out.push('-');
    } else {
        out.push_str(&sq_name(pos.ep));
    }
    out.push_str(&format!(" {} {}", pos.halfmove, pos.fullmove));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() {
        let fens = [
            START_FEN,
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        ];
        for f in fens {
            let p = parse(f).unwrap();
            assert_eq!(serialize(&p), f, "roundtrip failed for {}", f);
            assert_eq!(p.hash, p.compute_hash());
        }
    }
}
