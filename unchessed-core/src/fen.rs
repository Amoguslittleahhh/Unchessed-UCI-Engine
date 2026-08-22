//! FEN parsing and serialization.

use crate::board::*;

pub const START_FEN: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

pub fn startpos() -> Position {
    parse(START_FEN).expect("startpos FEN must parse")
}

pub fn parse(fen: &str) -> Result<Position, String> {
    let fields: Vec<&str> = fen.split_whitespace().collect();
    if !(4..=6).contains(&fields.len()) {
        return Err(format!("FEN needs 4 to 6 fields, got {}", fields.len()));
    }

    let mut pos = Position::empty();
    let ranks: Vec<&str> = fields[0].split('/').collect();
    if ranks.len() != 8 {
        return Err(format!(
            "piece placement needs 8 ranks, got {}",
            ranks.len()
        ));
    }
    for (row, text) in ranks.iter().enumerate() {
        let rank = 7 - row as u8;
        let mut file = 0u8;
        for ch in text.chars() {
            match ch {
                '1'..='8' => {
                    file = file
                        .checked_add(ch as u8 - b'0')
                        .ok_or("rank width overflow")?;
                    if file > 8 {
                        return Err(format!("rank {} exceeds 8 files", 8 - row));
                    }
                }
                _ => {
                    if file >= 8 {
                        return Err(format!("rank {} exceeds 8 files", 8 - row));
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
                    let square = sq(file, rank);
                    let bit = 1u64 << square;
                    pos.bb[color.idx()][piece] |= bit;
                    pos.occ_side[color.idx()] |= bit;
                    pos.occ |= bit;
                    pos.board[square as usize] = (color.idx() * 6 + piece) as u8;
                    file += 1;
                }
            }
        }
        if file != 8 {
            return Err(format!("rank {} has {} files, expected 8", 8 - row, file));
        }
    }

    pos.side = match fields[1] {
        "w" => Color::White,
        "b" => Color::Black,
        other => return Err(format!("bad side '{}'", other)),
    };

    pos.castling = 0;
    if fields[2] != "-" {
        for ch in fields[2].chars() {
            let right = match ch {
                'K' => WK,
                'Q' => WQ,
                'k' => BK,
                'q' => BQ,
                _ => return Err(format!("bad castling '{}'", ch)),
            };
            if pos.castling & right != 0 {
                return Err(format!("duplicate castling right '{}'", ch));
            }
            pos.castling |= right;
        }
    }

    pos.ep = if fields[3] == "-" {
        NO_EP
    } else {
        parse_sq(fields[3]).ok_or_else(|| format!("bad ep '{}'", fields[3]))?
    };
    pos.halfmove = if fields.len() >= 5 {
        fields[4]
            .parse::<u16>()
            .map_err(|_| format!("bad halfmove '{}'", fields[4]))?
    } else {
        0
    };
    pos.fullmove = if fields.len() >= 6 {
        fields[5]
            .parse::<u16>()
            .map_err(|_| format!("bad fullmove '{}'", fields[5]))?
    } else {
        1
    };
    if pos.fullmove == 0 {
        return Err("fullmove number must be at least 1".into());
    }

    if pos.bb[0][KING].count_ones() != 1 || pos.bb[1][KING].count_ones() != 1 {
        return Err("each side needs exactly one king".into());
    }
    if (pos.bb[0][PAWN] | pos.bb[1][PAWN]) & (0xff | (0xffu64 << 56)) != 0 {
        return Err("pawns may not occupy the first or eighth rank".into());
    }
    for color in [Color::White, Color::Black] {
        if pos.bb[color.idx()][PAWN].count_ones() > 8 || pos.occ_side[color.idx()].count_ones() > 16
        {
            return Err("too many pieces for one side".into());
        }
    }
    let white_king = pos.king_sq(Color::White);
    let black_king = pos.king_sq(Color::Black);
    if file_of(white_king).abs_diff(file_of(black_king)) <= 1
        && rank_of(white_king).abs_diff(rank_of(black_king)) <= 1
    {
        return Err("kings may not be adjacent".into());
    }

    let required = [
        (WK, Color::White, 4, 7, "K"),
        (WQ, Color::White, 4, 0, "Q"),
        (BK, Color::Black, 60, 63, "k"),
        (BQ, Color::Black, 60, 56, "q"),
    ];
    for (right, color, king_square, rook_square, name) in required {
        if pos.castling & right != 0
            && (pos.piece_on(king_square) != Some((color, KING))
                || pos.piece_on(rook_square) != Some((color, ROOK)))
        {
            return Err(format!("castling right '{}' lacks home king/rook", name));
        }
    }

    if pos.ep != NO_EP {
        let expected_rank = if pos.side == Color::White { 5 } else { 2 };
        if rank_of(pos.ep) != expected_rank || pos.piece_on(pos.ep).is_some() {
            return Err("invalid en-passant target rank or occupied target".into());
        }
        let pawn_square = if pos.side == Color::White {
            pos.ep - 8
        } else {
            pos.ep + 8
        };
        if pos.piece_on(pawn_square) != Some((pos.side.flip(), PAWN)) {
            return Err("en-passant target has no double-pushed pawn".into());
        }
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
    out.push(if let Color::White = pos.side {
        'w'
    } else {
        'b'
    });
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
    fn rejects_malformed_state_that_could_create_illegal_moves() {
        for invalid in [
            "4k3/8/8/8/8/8/8/4K2 w - - 0 1", // short rank
            "4k3/8/8/8/8/8/8/4K3 w K - 0 1", // no rook
            "4k3/8/8/8/8/8/4K3/8 w K - 0 1", // king off home
            "4k3/8/8/8/8/8/1P6/4K3 w - a1junk 0 1",
            "4k3/8/8/8/8/8/1P6/4K3 w - a1 0 1",
            "4k3/8/8/8/8/8/4k3/4K3 w - - 0 1", // duplicate/adjacent kings
            "4k3/8/8/8/8/8/8/P3K3 w - - 0 1",  // pawn on rank one
        ] {
            assert!(
                parse(invalid).is_err(),
                "accepted malformed FEN: {}",
                invalid
            );
        }
    }

    #[test]
    fn accepts_four_field_fen_with_strict_defaults() {
        let pos = parse("4k3/8/8/8/8/8/8/4K3 w - -").unwrap();
        assert_eq!(pos.halfmove, 0);
        assert_eq!(pos.fullmove, 1);
    }

    #[test]
    fn uncapturable_en_passant_does_not_change_repetition_hash() {
        let no_capture_ep = parse("4k3/8/8/8/4P3/8/8/4K3 b - e3 0 1").unwrap();
        let no_ep = parse("4k3/8/8/8/4P3/8/8/4K3 b - - 0 1").unwrap();
        assert_eq!(no_capture_ep.hash, no_ep.hash);

        let capturable = parse("4k3/8/8/8/3pP3/8/8/4K3 b - e3 0 1").unwrap();
        let capturable_without_ep = parse("4k3/8/8/8/3pP3/8/8/4K3 b - - 0 1").unwrap();
        assert_ne!(capturable.hash, capturable_without_ep.hash);
    }

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
