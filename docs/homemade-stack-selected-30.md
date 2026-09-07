# Direct SFNNv16 black-box comparison

The Stockfish value is obtained only through UCI from the official Stockfish 19 binary and published network; no Stockfish source or weights are imported into Unchessed. The homemade value is obtained through `evalbar homemade`.

positions=30
stockfish_depth=8
mae_cp=231.200
median_abs_error_cp=144.500
max_abs_error_cp=1152

| FEN | SFNNv16/Stockfish white cp | Homemade white cp | Delta cp |
|---|---:|---:|---:|
| `r2qk1nr/ppp1nppp/1b1p4/1P2p3/4PP2/2P2B2/P2P2PP/RNBQK2R w KQkq - 1 9` | 42 | 61 | +19 |
| `r2q1rk1/ppp1nppp/1b3n2/1P1p4/3PPP2/5B2/PB4PP/RN1Q1RK1 w - - 0 13` | 73 | 105 | +32 |
| `r2q1rk1/pp1n2pp/1bp3n1/1P1pPp2/3P1P2/2NQ1BP1/PB5P/R4RK1 w - f6 0 17` | 445 | 163 | -282 |
| `r2q1r1k/pp1n2pp/1b4n1/1P1BPp2/3P1P2/B2Q2P1/P6P/R4R1K w - - 3 21` | 417 | 66 | -351 |
| `1r3n1k/p3n1pp/1bB5/1P2PQ2/3q1P2/6P1/P6P/R4R1K w - - 1 25` | 241 | 141 | -100 |
| `2r2n1k/p5pp/1bn5/qP2P3/5P2/5RP1/P1Q4P/1R5K w - - 0 29` | 161 | -155 | -316 |
| `7k/p7/1bP1n1p1/4P2p/2Q2P2/2R3PK/7P/3q4 w - - 2 41` | 11 | 154 | +143 |
| `r1b1k2r/p1p1qppp/2p2n2/3p4/1b1QP3/2N2P2/PPP3PP/R1B1KB1R w KQkq - 0 9` | -22 | 24 | +46 |
| `rr4k1/p1p1qp1p/4bp2/2p5/Q1BpP3/4bP2/PPP3PP/1KNR3R w - - 4 17` | 1 | 2 | +1 |
| `r1b1kb1r/p1q2ppp/2p2n2/nB2p1N1/8/5Q2/PPPP1PPP/RNB1K2R w KQkq - 2 9` | 76 | 26 | -50 |
| `r1b1k2r/p1q1bp2/2p3pp/n2np3/8/2NB1QN1/PPPP1PPP/R1B1K2R w KQkq - 2 13` | 154 | 37 | -117 |
| `r4k1r/pbq1bp2/6p1/nB1pp2p/8/6N1/PPPPQPPP/R1B2RK1 w - - 0 17` | -75 | 55 | +130 |
| `r4k1r/1bq1b3/p4pp1/3p3p/B1nPp3/6N1/PPPBQPPP/R3R1K1 w - - 2 21` | -16 | 157 | +173 |
| `r6r/1bq3k1/3b1pp1/p2p3p/3Pp3/PB6/1PPQ1PPP/R3RNK1 w - - 3 25` | -38 | 7 | +45 |
| `r6r/2q3k1/8/p2b2pp/3Ppb2/P3N3/1PPQ1PP1/R3R1K1 w - - 0 29` | 251 | -137 | -388 |
| `2r4r/8/4Nk2/p5pp/3Pp3/P1b5/1PP1RPP1/R5K1 w - - 5 33` | 493 | 188 | -305 |
| `2r5/8/8/p2k2pp/3P4/P1P1r3/2P2PP1/4R1K1 w - - 0 37` | 450 | -188 | -638 |
| `r1b1k2r/ppp1n1pp/1bnp4/4pP2/4N2q/2P2QN1/PP1P1PPP/R1B1KB1R w KQkq - 1 9` | 206 | 68 | -138 |
| `r1b1k2r/ppp5/1bnp1p1p/4p3/3nN2q/2P2QP1/PP3P1P/R1B1KB1R w KQkq - 0 13` | 160 | -292 | -452 |
| `r3k2r/ppp5/1bnp1p1p/4pb2/8/2P3P1/PP1K1PQP/n1B2B1R w kq - 0 17` | -22 | -168 | -146 |
| `r1b1k1nr/pppp1ppp/2n2q2/b7/2B1P3/1Qp2N2/P4PPP/RNB2RK1 w kq - 2 9` | 55 | -64 | -119 |
| `1rb2rk1/ppppnppp/2n3q1/b3P3/P1B5/1QN2N2/5PPP/R1B1R1K1 w - - 1 13` | -99 | 39 | +138 |
| `1rb1r1k1/ppppnppp/4n3/b3P2q/P7/BQNB1N2/4RPPP/3R2K1 w - - 9 17` | 219 | 70 | -149 |
| `1rb3k1/pppprqpp/4n3/4Pp2/P6R/Q1bB1N2/5PPP/3R2K1 w - - 2 21` | 46 | -307 | -353 |
| `1rb1q2k/ppp1r2p/4p1p1/4PpN1/P6R/2Q5/5PPP/3R2K1 w - - 0 25` | 584 | 65 | -519 |
| `1rbq4/ppQ5/4p1pk/4Pp2/P7/8/5PPP/6K1 w - - 0 29` | 646 | -506 | -1152 |
| `1rb5/p3Q3/1p2p1k1/4Pp2/P7/8/5PPP/6K1 w - - 2 33` | 682 | 306 | -376 |
| `r2qk1nr/ppp2ppp/1bnpb3/4P3/2B1P3/2P2N2/P4PPP/RNBQ1RK1 w kq - 1 9` | 70 | 92 | +22 |
| `r3k1nr/pp1q2p1/1bnpp2p/6B1/4P3/2P2N2/P2N1PPP/R2Q1RK1 w kq - 0 13` | 54 | 116 | +62 |
| `r3k2r/ppbq4/2np1n1p/4p1p1/2N1P3/1QP2NB1/P4PPP/R4RK1 w kq - 4 17` | 198 | 24 | -174 |
