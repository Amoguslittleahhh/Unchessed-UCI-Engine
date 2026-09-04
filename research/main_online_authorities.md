# Refreshed online authorities for the main-branch audit

The current FIDE Handbook landing page was checked on 4 September 2026. It continues to list the document titled **FIDE Laws of Chess taking effect from 1 January 2023** as the applicable Laws entry, while separately listing later 2024--2026 regulations for ratings, titles, general regulations, and event administration. The Laws page states that it was approved by the FIDE General Assembly on 7 August 2022 and applied from 1 January 2023. The current handbook landing page is https://handbook.fide.com/ and the Laws text is https://handbook.fide.com/chapter/e012023.

The FIDE Rules Commission page independently provides the 2023 Laws PDF and changes table: https://rcc.fide.com/2023-laws-of-chess/. This is used as a second official FIDE source, not as a replacement for the handbook text.

The earlier perft authority remains relevant: Chess Programming Wiki, Perft Results, https://chessprogramming.org/Perft_Results. It provides canonical node counts and positions for move-generator validation, including start position, Kiwipete, castling, en-passant, promotions, captures, checks, and mates.

The UCI protocol context remains: Shredder Chess, Universal Chess Interface, https://www.shredderchess.com/chess-features/uci-universal-chess-interface.html.

The independent executable rules oracle remains: python-chess Core documentation, https://python-chess.readthedocs.io/en/latest/core.html. The installed version for this audit will be recorded explicitly and will be used for legal moves, FEN/status, castling, en-passant, halfmove, repetition, checkmate, stalemate, and insufficient-material comparisons.

Scope conclusion: the latest official FIDE Laws entry found is still the 1 January 2023 Laws. 2026 FIDE materials concern general/event/rating/administrative regulations and do not replace the Laws text for chess move legality. They will be treated as separate non-engine-testable administrative context where relevant.

The current FIDE General Rules and Regulations page is https://handbook.fide.com/chapter/GeneralRulesAndRegulations032026. It was approved by FIDE Council on 11 December 2025 and applied from 1 March 2026. Its content is complementary technical/administrative regulation, including definitions such as software bug, defect, compliance, and command-line interface. It does not supersede the 2023 Laws for board move legality or game-state rules. The audit will cite it only for current regulatory context and will not misclassify administrative requirements as engine move-generation tests.
