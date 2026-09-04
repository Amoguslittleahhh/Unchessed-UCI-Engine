//! Perft: move generation correctness gate.

use crate::board::Position;
use crate::movegen::legal;

pub fn perft(pos: &Position, depth: u32) -> u64 {
    if depth == 0 {
        return 1;
    }
    let ml = legal(pos);
    if depth == 1 {
        return ml.len as u64;
    }
    let mut nodes = 0;
    for &m in ml.as_slice() {
        nodes += perft(&pos.make(m), depth - 1);
    }
    nodes
}

/// Perft split by root move (debugging aid).
pub fn perft_divide(pos: &Position, depth: u32) -> Vec<(String, u64)> {
    let ml = legal(pos);
    let mut out = Vec::new();
    for &m in ml.as_slice() {
        let n = if depth <= 1 {
            1
        } else {
            perft(&pos.make(m), depth - 1)
        };
        out.push((m.uci(), n));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fen;

    fn check(fen: &str, depth: u32, expected: u64) {
        let pos = fen::parse(fen).unwrap();
        assert_eq!(perft(&pos, depth), expected, "perft({}) of {}", depth, fen);
    }

    #[test]
    fn startpos_shallow() {
        check(fen::START_FEN, 1, 20);
        check(fen::START_FEN, 2, 400);
        check(fen::START_FEN, 3, 8_902);
        check(fen::START_FEN, 4, 197_281);
        check(fen::START_FEN, 5, 4_865_609);
    }

    #[test]
    fn kiwipete() {
        let f = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";
        check(f, 1, 48);
        check(f, 2, 2_039);
        check(f, 3, 97_862);
        check(f, 4, 4_085_603);
    }

    #[test]
    fn position3() {
        let f = "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1";
        check(f, 1, 14);
        check(f, 2, 191);
        check(f, 3, 2_812);
        check(f, 4, 43_238);
        check(f, 5, 674_624);
    }

    #[test]
    fn position4() {
        let f = "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1";
        check(f, 1, 6);
        check(f, 2, 264);
        check(f, 3, 9_467);
        check(f, 4, 422_333);
    }

    #[test]
    fn position5() {
        let f = "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8";
        check(f, 1, 44);
        check(f, 2, 1_486);
        check(f, 3, 62_379);
        check(f, 4, 2_103_487);
    }

    #[test]
    fn position6() {
        let f = "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10";
        check(f, 1, 46);
        check(f, 2, 2_079);
        check(f, 3, 89_890);
        check(f, 4, 3_894_594);
    }

    #[test]
    #[ignore = "slow; run with cargo test -- --ignored"]
    fn deep() {
        check(fen::START_FEN, 6, 119_060_324);
        check(
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            5,
            193_690_690,
        );
    }
}
