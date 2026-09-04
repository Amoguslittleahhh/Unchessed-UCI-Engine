# Independent Audit — 2026-09-04

## Scope

The optimisation copy was audited independently across UCI command lifecycle, search limits, root-hint handling, persona transitions, numerical sampling, cache sizing, and portable-build assumptions. The audit used source inspection, targeted regression design, Rust tests, release validation, UCI smoke behavior, and Clippy as a static signal.

## Defects fixed

### Fail-open `searchmoves` restriction

When `go searchmoves` contained no legal move, the root search previously fell back to the complete legal move list. This violated the caller's explicit restriction and could allow an unintended move after a stale or malformed command. The search now returns an empty line set, and the UCI layer emits `bestmove 0000`. A regression test covers this behavior.

### Non-finite policy-prior propagation

The persona sampler accepted arbitrary `MovePrior` outputs. NaN, infinity, negative weights, or extreme positive values could make the sampling total non-finite or distort the distribution. A bounded `safe_prior_weight` now converts malformed values to a finite default and clamps valid weights to `[0.1, 100.0]`. A numerical regression test covers NaN, infinity, negative, oversized, and missing entries.

## Reviewed and retained behavior

Stop-token reset and worker joining are correctly performed before `position`, `go`, `setoption`, `ucinewgame`, and `quit`. Ponder searches are deferred until `ponderhit` and discarded on stop or position changes. Evaluator changes clear the TT. Opponent observations are keyed to actual historical plies. The internal root-hint firewall retains alpha-beta and legal move authority.

The cache-aware default is intentionally conservative: explicit Hash overrides automatic sizing, and the host-specific build flag remains removed from the portable artifact. Clippy currently reports a large pre-existing baseline of style lints in tests and call sites; these are not correctness failures and were not mass-reformatted as part of this targeted audit.

## Validation target

The corrected copy must pass the focused new tests, the full workspace test suite, release build-and-test script, UCI smoke checks, and repository diff hygiene before publication.
