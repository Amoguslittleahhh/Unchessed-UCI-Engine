extern crate unchessed_core;
use unchessed_core::{fen, perft::perft};

fn check(name: &str, f: &str, pairs: &[(u32, u64)]) {
    let p = fen::parse(f).expect("FEN parse");
    for &(d, want) in pairs {
        let got = perft(&p, d);
        println!("{name}\tdepth={d}\texpected={want}\tobserved={got}\tpass={}", got == want);
        assert_eq!(got, want, "{name} depth {d}");
    }
}
fn main() {
    check("startpos", fen::START_FEN, &[(1,20),(2,400),(3,8902),(4,197281),(5,4865609),(6,119060324)]);
    check("kiwipete", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", &[(1,48),(2,2039),(3,97862),(4,4085603),(5,193690690)]);
    check("position3", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", &[(1,14),(2,191),(3,2812),(4,43238),(5,674624)]);
    check("position4", "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", &[(1,6),(2,264),(3,9467),(4,422333)]);
    check("position5", "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", &[(1,44),(2,1486),(3,62379),(4,2103487)]);
    check("position6", "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", &[(1,46),(2,2079),(3,89890),(4,3894594)]);
}
