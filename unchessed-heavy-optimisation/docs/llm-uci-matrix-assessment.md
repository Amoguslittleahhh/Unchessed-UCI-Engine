# Assessment: the "LLM-UCI Developers" summary matrix

A four-row matrix was proposed mapping problem areas to papers. This records
what checking each citation actually found, and the one change that came out
of it.

**Two of the four arXiv IDs do not cite what the matrix claims.** Verified by
fetching each abstract.

| Row | Claimed | Actual paper at that ID | Verdict |
|---|---|---|---|
| Invalid UCI / illegal moves | arXiv:2604.09123 — "Dynamic Logit Masking" | *Prototype-Regularized Federated Learning for Cross-Domain Aspect Sentiment Triplet Extraction* | **Wrong ID.** Federated sentiment extraction — no chess, no decoding constraints. |
| Training the LLM to play better | arXiv:2501.12948 — "DeepSeek-style GRPO with Engine Rewards" | *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via RL* | **Wrong ID for the claim.** R1 is real and does use GRPO, but it is not chess and has no engine rewards. GRPO originates in arXiv:2402.03300 (DeepSeekMath), already reviewed. |
| State representation | arXiv:2501.17186 — long-context PGN over static FENs | *Complete Chess Games Enable LLM Become A Chess Master* | **Correct.** Reviewed last round. |
| Hybrid search execution | arXiv:2412.12119 — LLM policy + MCTS | *Mastering Board Games by External and Internal Planning with Language Models* | **Correct.** Real DeepMind paper, GM-level chess via MCTS-guided LLM. |

I am not able to tell whether the two bad IDs are typos for real papers or
fabrications. Either way the technique names should not be cited to them.

## Do the correct papers apply here?

Both verified papers are about **LLMs playing chess**. This engine is a Rust
alpha-beta searcher with a 4.2M-param policy net; it has no language model and
no token decoder, so:

- **Row 3 (long-context PGN history)** does not transfer directly. Our model
  does consume a history vector, but it is a 32-wide projection of *time
  control and rating only* (`history_len=0`, see
  `tools/reference_forward_unarchitectured_metal.py`) — there is no move history
  in the input at all. Adding real history would be a format change plus a
  retrain, which the SPRT record does not justify.
- **Row 4 (LLM policy + MCTS)** is the architecture this project already
  rejected on evidence. Four SPRT batches found the policy hint never trended
  positive, and `docs/unarchitectured-metal-why-the-hint-costs-elo.md` explains
  why: the benefit lands on the cheapest search pass while the cost is charged
  to every move. Swapping alpha-beta for MCTS to deepen that coupling would be
  a very large change justified by nothing measured here.

## Row 1 turned out to be worth checking anyway

The row is mis-cited, but the underlying concern — *can the model's output
ever produce a move that is not legal?* — is a real one, so I traced it.

**The Rust runtime is structurally immune to the LLM version of this
problem.** It never scores illegal moves in the first place: logits are
computed by iterating `input.legal_actions` directly
(`unarchitectured_metal_runtime.rs`), so the output vector only ever contains real legal
moves. There is no masking step to get wrong. `validate` additionally rejects
an empty action list, more than 218 actions, and any action outside
`64*64*5`. The Python reference *does* use `masked_fill(~legal_mask, -1e4)`,
because it evaluates a padded 218-wide tensor — but that is reference-only
scaffolding, not the shipped path.

**One real gap did surface, one layer up.** In `uci.rs`, hints were built with:

```rust
legal_moves.iter().zip(hint.output.logits.iter())
```

`zip` stops at the shorter side. If the logit vector and the move list ever
disagreed in length, this would silently emit a *partial* ranking — some legal
moves unscored, and if the orders had diverged, scores attached to the wrong
moves. Nothing downstream could detect it; the search would just receive a
plausible-looking ranking that is wrong.

That cannot happen today: `HintKey` includes the entire `legal_actions`
vector, and `latest_exact` only returns a hint on an exact key match. But that
is an invariant maintained **in a different module, by code with no obligation
to keep maintaining it**, and the consuming site asserted nothing.

The fix is small: check the lengths where the assumption is used, and on
mismatch drop the hint and search unhinted rather than order on bad data. It
reports as `length-mismatch` through the existing info string, so it is
visible rather than silent.

This is defence in depth, not a live bug fix. No behaviour changes on any
path that runs today.

## Status

No model, training, or search behaviour changed. `UnarchitecturedHint`
remains default-off and `runtime_safety_suite` remains false — nothing here
bears on either.
