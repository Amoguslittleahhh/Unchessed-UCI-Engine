use unchessed_core::fen;
use unchessed_core::perft::perft;

fn check(fen_text: &str, expected: &[(u32, u64)]) {
    let pos = fen::parse(fen_text).expect("canonical FEN must parse");
    for &(depth, nodes) in expected {
        assert_eq!(perft(&pos, depth), nodes, "perft depth {depth} for {fen_text}");
    }
}

#[test]
fn canonical_online_perft_positions_match() {
    check(fen::START_FEN, &[(1, 20), (2, 400), (3, 8_902), (4, 197_281), (5, 4_865_609)]);
    check("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
          &[(1, 48), (2, 2_039), (3, 97_862), (4, 4_085_603), (5, 193_690_690)]);
    check("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
          &[(1, 14), (2, 191), (3, 2_812), (4, 43_238), (5, 674_624)]);
    check("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
          &[(1, 6), (2, 264), (3, 9_467), (4, 422_333)]);
    check("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
          &[(1, 44), (2, 1_486), (3, 62_379), (4, 2_103_487)]);
    check("r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
          &[(1, 46), (2, 2_079), (3, 89_890), (4, 3_894_594)]);
}
