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
                rank -= 1;
                file = 0;
                if rank < 0 {
                    return Err("too many ranks".into());
                }
            }
            '1'..='8' => file += ch as i32 - '0' as i32,
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

    pos.ep = NO_EP;
    if let Some(e) = parts.next() {
        if e != "-" {
            pos.ep = parse_sq(e).ok_or_else(|| format!("bad ep '{}'", e))?;
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
