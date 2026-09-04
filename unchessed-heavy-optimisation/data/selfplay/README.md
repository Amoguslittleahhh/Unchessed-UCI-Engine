# Maia-3 self-play, random UCI elo limits against each other

Games played by two **Maia-3** instances whose UCI elo limits are drawn
independently and uniformly from 100-3200 and fixed per side for the
whole game — the "mix of maia 3 with random uci elo limits against each
other". Every move is a sample from the model's elo-conditioned policy
(temperature 1.0, the official UCI's human-like mechanism).

**200 games, 13,076 labeled moves** (`seed 42`), with WhiteElo/BlackElo
headers set to the drawn limits, so the files feed the same
level-conditioned pipeline as the real-human data.

## The model

The real Maia-3, as the official `maia-platform-frontend` ships it:
`maia3_simplified.onnx` (45.7 MB ONNX export, GPL-3.0). Not committed
here — fetch with:

```sh
python3 tools/selfplay_elo_mixer.py fetch-model --out /tmp/maia3-onnx
python3 tools/selfplay_elo_mixer.py play \
    --model /tmp/maia3-onnx/simple-maia3-inference/simple_maia3_inference/maia3_simplified.onnx \
    --games 200 --seed 42 --pgn <out.pgn> --labels <out.jsonl> --report
```

Pinned at `mcognetta/simple-maia3-inference` @
`05ede11a0ec43c7d1c5d55d16ec86fbe5b6a3fcc` (the mirror the official
frontend's ONNX came from). Our driver's 4352-move indexing and
inference were verified entry-by-entry and position-by-position against
the reference implementation (max probability diff ~1e-8).

## Conditioning works — measured

Mean top-1 policy probability by the mover's elo across this set:
**0.323 at 100-199, rising monotonically to ~0.52-0.62 at 1900-3200**.
Low-elo play is diffuse (blunder-rich), high-elo play is concentrated —
the strength signal is real and per-side. (Contrast:
`docs/rating-conditioning-finding.md` shows our own net's scalar rating
input moving the output by ~0.004 logit — this is what working
conditioning looks like.)

## Labels

`maia3-100-3200-labels.jsonl` — one row per move:
`{game, elo_white, elo_black, fen, move_uci, move_ply, side, elo_self,
elo_oppo, top1_prob, ldw}`. `ldw` is (loss, draw, win) from the
side-to-move's perspective. All 13,076 moves replay legal from the
recorded FENs (test: `test_selfplay_elo_mixer.py`).

## Honest limits

- The "simplified" export is single-position: no multi-position history
  input and no clock/time inputs, and move selection omits the official
  UCI's one-ply opponent-response ranking. The elo conditioning is the
  model's real one.
- Maia-3 is a *human-move predictor*, not a rated player: "elo 200" here
  means "the move distribution of a ~200-rated human", which is exactly
  the level-conditioning training signal wanted — but the games are
  model-generated, not human.
- 200 games is a seed set; the generator is the deliverable — rerun with
  `--games 2000 --seed <new>` for scale (measured ~15-25 min per
  200 games on the sandbox's 2-core Xeon).

## Where it fits

- `data/training-elo/` — real humans, same bands; empty at 100-900 and
  3000-3200, exactly where this set is dense.
- `tools/build_level_conditioned_moves.py` — turns either set into
  (FEN, skill-window, move) labels for the level-conditioned retrain
  designed in `docs/research-notes-maia-levels-reverse-engineering.md`.
