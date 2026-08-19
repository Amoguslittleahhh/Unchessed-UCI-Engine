# Opening book coverage

## Coverage result

The previous hand-written book contained 60 lines and only 54 distinct ECO
codes (10.8% of A00–E99). The default book now layers:

1. **3,810 named historical opening lines covering all 500 ECO codes** from
   [`lichess-org/chess-openings`](https://github.com/lichess-org/chess-openings),
   pinned at the commit recorded in `books/lichess-openings/SOURCE.txt` and
   released under CC0;
2. **45 curated serious lines** with higher production weights;
3. **15 explicitly risk-graded troll lines** (`tricky`, `dubious`, `meme`);
4. named but offbeat first-move families as the `Random` tier;
5. optional external Polyglot data through `BookFile` for empirical popularity
   from arbitrarily large master/human game corpora.

This is complete **named ECO coverage**, not a claim that every chess game or
every legal move sequence can be enumerated. The legal game tree is vastly too
large. Unnamed novelties and real-world frequencies belong in an external
Polyglot book generated from the desired game corpus.

## Safety policy

- Curated mainline classifications override historical/random data.
- Curated troll classifications override historical data only when the move is
  not protected by a curated mainline.
- Strong, engine, and uncertain opponents receive protected mainlines.
- Effective book depth scales from 6 plies below 800 estimated Elo to the full
  configured 40 plies at 2400+/engine strength, reducing a major fingerprint.
- `Random` historical moves require a confidently human opponent with a safely
  low upper strength bound.
- CLINCH suppresses all troll-tier continuations, including forced `Troll=On`.
- Auto trolls require sufficient human evidence and the existing risk gates.
- `Troll=Off` and known/computer anti-troll locks remain absolute.
- `Troll=On` remains the explicit user override.

## Weights

Historical branch weights count how many named variations continue through a
move. They provide broad diversity, not true play frequency. Curated weights
remain authoritative for the project's preferred repertoire. When a Polyglot
book is loaded, its empirical weights dominate matching moves.

For production-strength popularity, build separate Polyglot books from:

- master games for serious theory;
- rating/time-control-stratified human games for human-like variety;
- a separately curated and reviewed troll corpus.

Do not merge untrusted troll labels into mainline data.

## Audit

Run:

```bash
python tools/check_opening_coverage.py
```

The check verifies:

- exactly 3,810 imported rows;
- all 500 ECO codes A00–E99;
- source/license metadata;
- curated tier counts;
- the previous-vs-current coverage summary.

Rust book tests additionally replay every SAN line through the engine's own
legal move generator, verify random/main classifications, and verify that the
Bongcloud remains a risk-3 troll rather than being promoted by historical data.
