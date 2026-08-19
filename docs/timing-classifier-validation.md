# Timing-signal validation

## Production decision

The public-data replication does **not** validate clock timing as a standalone
human/BOT classifier. The production feature remains restricted to its existing
safe role: clock regularity can only modulate an independently ceiling-level
move-quality signal; timing cannot initiate engine classification, cannot enable
trolling, and cannot alter fixed `UCI_Elo` behavior.

The deterministic numerical result is in
[`data/timing-validation/result.md`](../data/timing-validation/result.md). On the
matched account-level sample, higher lag-1 timing autocorrelation produced AUC
**0.413** with an account-bootstrap 95% interval of **[0.260, 0.575]**. At the
runtime threshold of 0.45, sensitivity was **0.200** and the unmarked-class
false-positive rate was **0.333**. The strict leave-one-account-out scalar
logistic experiment reached AUC **0.565**, but it is exploratory, may learn the
opposite sign, and is not shipped.

These are negative/insufficient results, not evidence that BOT timing is
human-like in general. The interval is wide, the clocks are rounded to seconds,
and the matched set contains only 20 affirmative BOT accounts.

## Data and labels

The primary source is the [Lichess open database](https://database.lichess.org/),
whose exports are CC0. Lichess PGNs mark accounts using the Bot API with
`WhiteTitle "BOT"` or `BlackTitle "BOT"`. That title is the positive label.

The comparison class comes from separate samples of rated standard Lichess
games and excludes `BOT` titles. It is named **unmarked**, not human: an absent
BOT title does not prove that a player is human and cannot exclude assistance
or undeclared automation. This deliberate naming prevents the validation report
from overstating its ground truth.

The exact transport repositories, commits, selected globs, byte counts, and
content checksums are pinned in
[`data/timing-validation/source-manifest.json`](../data/timing-validation/source-manifest.json).
Those repositories are treated only as mirrors; their project licenses are not
used as the data license. Extraction accepts only games whose `Site` is an
official `https://lichess.org/` game URL, so the source-data basis remains the
Lichess CC0 export. The raw PGNs are not copied into this repository. The
committed JSONL contains only pseudonymous account/game hashes and aggregate
clock features—no usernames, moves, or PGN text.

Snapshot scale before matching:

- 390 valid BOT game-perspectives from 39 affirmative BOT accounts;
- 2,204 valid unmarked game-perspectives from 2,141 accounts;
- 2,594 derived records total.

## Feature parity with the engine

For each player move after the first recorded move:

1. infer time used as `previous_clock + increment - clock_after`;
2. reject non-positive inferred durations, matching the runtime's rejection of
   zero-millisecond observations;
3. compute `log(time_used / clock_before_move)`;
4. retain the latest 32 positive observations;
5. compute Pearson lag-1 autocorrelation between observations `0..n-1` and
   `1..n` when at least six values and nonzero variance exist.

The autocorrelation formula mirrors
`OpponentModel::timing_autocorrelation`. Unit tests cover the formula, recursive
PGN-variation removal, BOT/unmarked selection, pseudonymisation, matching caps,
and tied AUC behavior.

There are two unavoidable parity gaps:

- monthly/API PGNs in this snapshot round `%clk` to one second, while the UCI
  adapter can receive millisecond clocks;
- PGNs do not expose the runtime's `legal moves > 8` and search-derived
  difficulty filter, so extraction cannot reproduce `position_had_choice`
  exactly without replay plus engine analysis.

These gaps are reported as limitations rather than silently approximated away.
The historical Lichess universal dumps offer centisecond `%clkc` values and are
supported by the parser, but a suitably labeled, balanced BOT sample was not
available in this local replication.

## Leakage-resistant protocol

Validation is deterministic under seed `20260819`:

- match BOT and unmarked game-perspectives one-to-one;
- require the exact same `base+increment` control;
- require the same 200-point source-rating cell and choose the nearest available
  rating within that cell without consulting the timing score;
- cap an account at eight total records and three records in one matching cell;
- aggregate matched records to one median score per account for primary
  statistics;
- bootstrap whole account scores, never individual moves;
- permute labels at the account level;
- leave exactly one account out for every fold of the exploratory scalar
  logistic calibration.

The sequence-level AUC is retained only as a descriptive diagnostic. It is not
treated as an independent-sample result because multiple games can come from
the same BOT account.

The configured gates in `config/timing_validation.json` require enough matched
BOT accounts, a positive account-AUC lower confidence bound, and a low
upper-confidence-bound false-positive rate. They fail. These versioned gates
are useful for future snapshots but are not claimed as an externally
preregistered protocol.

## Reproduction

Clone the transport snapshots at the pinned commits, then run:

```bash
python3 tools/timing_classifier_validation.py extract \
  --config config/timing_validation.json \
  --output data/timing-validation/records.jsonl \
  --summary data/timing-validation/extraction-summary.json \
  --bot-pgn /path/to/c0br4/lichess/game_records/*.pgn \
  --unmarked-pgn \
    /path/to/go-strength/chess/candidating/*.pgn \
    /path/to/additional/unmarked/*.pgn

python3 tools/timing_classifier_validation.py validate \
  --config config/timing_validation.json \
  --records data/timing-validation/records.jsonl \
  --manifest data/timing-validation/source-manifest.json \
  --json data/timing-validation/report.json \
  --markdown data/timing-validation/result.md
```

CI runs the unit tests and regenerates both reports in `--check` mode. Use
`--strict-gates` in an external promotion job when a failed standalone gate
should make that job fail; regular CI intentionally accepts the current honest
negative result.

## What this validation does not establish

- It does not identify cheating or engine assistance.
- It does not validate transfer to unseen platforms, accounts, engine families,
  or time controls.
- It does not estimate one-Elo strength precision.
- It does not validate timing as the cause of a ceiling-strength signal.
- It does not justify increasing timing's runtime weight or enabling timing-only
  classification.
- It does not reproduce the earlier move-level/account-level research dataset;
  it is an independent public-data snapshot with its own provenance.

A stronger follow-up needs centisecond or millisecond clocks, affirmative human
labels, more BOT families, account- and family-disjoint future-month holdouts,
and exact replay of the runtime choice/difficulty filter. Until that evidence
exists, the conservative production rule remains mandatory.
