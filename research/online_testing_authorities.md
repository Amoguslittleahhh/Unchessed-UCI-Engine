# Online testing authorities consulted

1. FIDE Handbook, Laws of Chess effective 1 January 2023: https://handbook.fide.com/chapter/e012023
   - Primary rules authority supplied by the user.
   - Covers Basic Rules, Competitive Rules, and Appendices.
   - Relevant clauses include legal moves (Articles 3.1–3.10), game completion (Article 5), illegal positions (Article 3.10.3), repetition and 50/75-move rules (Article 9), and mate precedence (Article 9.6.2).

2. Chess Programming Wiki, Perft Results: https://chessprogramming.org/Perft_Results
   - Established debugging reference with canonical leaf-node counts and positions.
   - Initial position: depth 1 = 20, depth 2 = 400, depth 3 = 8,902, depth 4 = 197,281, depth 5 = 4,865,609, depth 6 = 119,060,324.
   - Includes Kiwipete and additional positions targeting castling, en passant, promotions, checks, captures, and mates.

3. Shredder Chess, Universal Chess Interface overview: https://www.shredderchess.com/chess-features/uci-universal-chess-interface.html
   - Describes UCI as the open protocol connecting chess engines and user interfaces.
   - Used as a secondary protocol authority for handshake and engine/GUI interoperability scope.

4. python-chess core documentation: https://python-chess.readthedocs.io/en/latest/core.html
   - Documents the Board abstraction and APIs including legal move generation, game-over detection, FEN/status, move stack, castling rights, en-passant square, halfmove clock, and fullmove number.
   - Used as an independently maintained executable oracle, version 1.11.2, not as proof of FIDE compliance by itself.

Methodological note: search-result snippets were not used as evidence. The full pages were opened and their extracted content was saved by the browser for local inspection before use in the paper.
