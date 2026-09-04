# NNUE round 13 — results (2026-08-30)

Honest results for the recipe-validation ask in
`docs/nnue-v4-retrain-data-scaling-finding.md` (item 1 of
`scripts/research/arena_agent_unarchitectured_v1_runtime_speed_prompt.md`).

## What this round actually did

| Item | Status |
|---|---|
| Work out epoch/step count from val-MAE, not a copied constant | **Done** — early-stop patience 3, export best checkpoint |
| Recover the shipped default's training recipe | **Done** — from `full_pipeline.sh` / `full_pipeline_cloud.sh` |
| Fourth local data point (108M shards) | **Not run** — shards not in git; sandbox is 2 vCPU / 3.8 GB / no GPU |
| Go/no-go on cloud 178M | **NO-GO** until the 108M local SPRT exists; see the recipe |
| Unarchitectured v1 retrain / oracle rating sweep / MinTime | **Not started** — gated as before |
| `UnarchitecturedHint` | **Stays default-off** |

Full writeup: `docs/nnue-v4-training-recipe.md`.

## Trainer change

`tools/train_nnue.py` previously exported only the last epoch and always
ran the requested count. Every diagnostic net in the finding was
therefore a *worse* checkpoint than that run had already produced
(57.4→83.6, 55.3→59.3, 51.1→54.3 cp). The trainer now:

- tracks val-MAE per epoch, clones the best `state_dict` on CPU;
- stops after `EARLY_STOP_PATIENCE` (default 3) epochs without a
  ≥`EARLY_STOP_MIN_DELTA` (default 0.1 cp) improvement;
- writes the best checkpoint, and logs `best-epoch`, `samples-seen`,
  and `stopped-early`;
- keeps the production Adam 1e-3 / 60%–80% step-decay, keyed off the
  cap (if we stop before the first drop, LR never drops — intentional).

Control logic lives in `tools/nnue_train_control.py` so it can be tested
without torch. Architecture, loss, feature scheme, and file format are
unchanged. Default search path is unchanged.

## Verification (this host)

- `python3 -m py_compile tools/train_nnue.py tools/nnue_train_control.py`: ok.
- `pytest tools/test_nnue_train_control.py tools/test_train_nnue.py -q`:
  **12 passed, 1 skipped** (the skip is `test_train_exports_best_epoch_not_last`,
  torch not installed — torch is excluded from `requirements-dev.txt` on
  purpose). Control-loop tests do not need torch.
- `pytest tools/ -q`: **330 passed, 22 skipped, 347 subtests** in 10.67 s.
  Round 12's 337/4 on a torch-present host; the extra skips here are
  torch-gated tests, not new failures. New tests are in the 12 above.
- `python3 tools/rust_bracket_check.py --all`: **21/21 files balanced**.
- No Rust sources changed; `cargo test` not run (no rustc in this
  sandbox this session).
- 108M training: not run (shards not in git). No SPRT. No cloud spend.

## Honest negatives

- The 108M point the prompt asked for was not measured. The recipe is
  defended from the three existing SPRTs, the committed launchers, and
  the last-vs-best export bug; it is not a new Elo number.
- No net is proposed for shipping.
- Weight decay / FEN-skipping / loss changes were considered and
  rejected for this round — no evidence they are the lever.
