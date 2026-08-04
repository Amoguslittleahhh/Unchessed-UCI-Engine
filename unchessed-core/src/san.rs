//! Standard Algebraic Notation parsing (PGN movetext), resolved against the
//! legal moves of a position. Shared by the datagen tool and the future
//! Game Reviewer PGN CLI.

use crate::board::*;
use crate::movegen::legal;

/// Parse a SAN token like "e4", "Nbd2", "exd5", "O-O", "e8=Q+", "Rxe5#".
pub fn parse_san(pos: &Position, san: &str) -> Option<Move> {
    let s: String = san
        .trim_end_matches(['+', '#', '!', '?'])
        .trim_end_matches("e.p.")
        .trim_end_matches(['+', '#', '!', '?'])
        .to_string();
    if s.is_empty() {
        return None;
    }
    let ml = legal(pos);

    // castling
    if s == "O-O" || s == "0-0" {
        return ml
            .as_slice()
            .iter()
            .copied()
            .find(|m| m.kind() == MK_CASTLE && m.to() > m.from());
    }
    if s == "O-O-O" || s == "0-0-0" {
        return ml
            .as_slice()
            .iter()
            .copied()
            .find(|m| m.kind() == MK_CASTLE && m.to() < m.from());
    }

    let bytes = s.as_bytes();
    let mut i = 0;

    // piece letter
    let piece = match bytes[0] {
        b'N' => {
            i = 1;
            KNIGHT
        }
        b'B' => {
            i = 1;
            BISHOP
        }
        b'R' => {
            i = 1;
            ROOK
        }
        b'Q' => {
            i = 1;
            QUEEN
        }
        b'K' => {
            i = 1;
            KING
        }
        _ => PAWN,
    };

    // promotion suffix "=X" (or a bare trailing piece letter, e.g. "e8Q")
    let mut promo: Option<usize> = None;
    let mut end = bytes.len();
    if end >= 2 && bytes[end - 2] == b'=' {
        promo = match bytes[end - 1] {
            b'N' => Some(KNIGHT),
            b'B' => Some(BISHOP),
            b'R' => Some(ROOK),
            b'Q' => Some(QUEEN),
            _ => return None,
        };
        end -= 2;
    } else if piece == PAWN && end >= 3 && matches!(bytes[end - 1], b'N' | b'B' | b'R' | b'Q') {
        promo = match bytes[end - 1] {
            b'N' => Some(KNIGHT),
            b'B' => Some(BISHOP),
            b'R' => Some(ROOK),
            b'Q' => Some(QUEEN),
            _ => None,
        };
        end -= 1;
    }

    // destination square: last two chars
    if end < i + 2 {
        return None;
    }
    let dest = parse_sq(std::str::from_utf8(&bytes[end - 2..end]).ok()?)?;
    end -= 2;

    // between piece letter and dest: optional disambiguation and 'x'
    let mut dis_file: Option<u8> = None;
    let mut dis_rank: Option<u8> = None;
    for &b in &bytes[i..end] {
        match b {
            b'x' => {}
            b'a'..=b'h' => dis_file = Some(b - b'a'),
            b'1'..=b'8' => dis_rank = Some(b - b'1'),
            _ => return None,
        }
    }

    let mut found: Option<Move> = None;
    let mut queen_promo_fallback: Option<Move> = None;
    for &m in ml.as_slice() {
        if m.to() != dest || m.kind() == MK_CASTLE {
            continue;
        }
        let (_, pt) = pos.piece_on(m.from())?;
        if pt != piece {
            continue;
        }
        if let Some(f) = dis_file {
            if file_of(m.from()) != f {
                continue;
            }
        }
        if let Some(r) = dis_rank {
            if rank_of(m.from()) != r {
                continue;
            }
        }
        match promo {
            Some(p) => {
                if !m.is_promo() || m.promo_piece() != p {
                    continue;
                }
            }
            None => {
                if m.is_promo() {
                    // lenient: bare "e8" style token -> accept the queen promo
                    if m.promo_piece() == QUEEN {
                        queen_promo_fallback = Some(m);
                    }
                    continue;
                }
            }
        }
        if found.is_some() {
            return None; // truly ambiguous SAN: refuse rather than guess
        }
        found = Some(m);
    }
    found.or(queen_promo_fallback)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fen;

    fn replay(sans: &[&str]) -> Position {
        let mut pos = fen::startpos();
        for s in sans {
            let mv = parse_san(&pos, s).unwrap_or_else(|| panic!("failed SAN '{}'", s));
            pos = pos.make(mv);
        }
        pos
    }

    #[test]
    fn basic_game() {
        let pos = replay(&[
            "e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4", "Nf6", "O-O", "Be7",
        ]);
        assert_eq!(
            fen::serialize(&pos).split(' ').next().unwrap(),
            "r1bqk2r/1pppbppp/p1n2n2/4p3/B3P3/5N2/PPPP1PPP/RNBQ1RK1"
        );
    }

    #[test]
    fn disambiguation() {
        // two knights can reach d2: Nbd2 must pick the b1 knight
        let pos = fen::parse("rnbqkbnr/ppp2ppp/8/3pp3/3P4/5N2/PPP1PPPP/RNBQKB1R w KQkq - 0 3")
            .unwrap();
        let m = parse_san(&pos, "Nbd2").unwrap();
        assert_eq!(m.uci(), "b1d2");
        let m = parse_san(&pos, "Nfd2").unwrap();
        assert_eq!(m.uci(), "f3d2");
    }

    #[test]
    fn captures_and_ep() {
        let mut pos = replay(&["e4", "Nf6", "e5", "d5"]);
        // exd6 e.p.
        let m = parse_san(&pos, "exd6").unwrap();
        assert_eq!(m.kind(), MK_EP);
        pos = pos.make(m);
        let m = parse_san(&pos, "cxd6").unwrap();
        assert_eq!(m.uci(), "c7d6");
    }

    #[test]
    fn promotion() {
        let pos = fen::parse("8/P6k/8/8/8/8/7K/8 w - - 0 1").unwrap();
        assert_eq!(parse_san(&pos, "a8=Q").unwrap().uci(), "a7a8q");
        assert_eq!(parse_san(&pos, "a8=N+").unwrap().uci(), "a7a8n");
        assert_eq!(parse_san(&pos, "a8").unwrap().uci(), "a7a8q");
    }

    #[test]
    fn rank_disambiguation() {
        let pos = fen::parse("k7/8/8/8/R7/8/R6K/8 w - - 0 1").unwrap();
        assert_eq!(parse_san(&pos, "R4a3").unwrap().uci(), "a4a3");
        assert_eq!(parse_san(&pos, "R2a3").unwrap().uci(), "a2a3");
        // plain "Ra3" is ambiguous -> refuse
        assert!(parse_san(&pos, "Ra3").is_none());
    }
}
