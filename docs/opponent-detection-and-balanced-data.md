# Opponent detection and balanced Elo data

This document defines the phased production design for estimating opponents
from roughly 100 Elo through the strongest engines. It deliberately separates
**identity**, **agent type**, **playing strength**, and **persona style**.

## What can and cannot be inferred

A deployed engine cannot reliably distinguish 3000 from 3600 using an analysis
oracle weaker than both. Above its measurement ceiling it should report
`>= ceiling`, play at full strength, and disable automatic trolling. Exact
high-engine ratings come from metadata or an externally calibrated engine
league, not from pretending centipawn loss can measure beyond the oracle.

Likewise, exact 100-vs-200 distinctions are only valid when the source platform
has enough stable, non-provisional players at those levels. Sparse tails should
use wider uncertainty intervals rather than false precision.

Human-versus-engine detection from moves alone is probabilistic and can be
adversarially defeated. `uncertain` is a required output, not an error.

## Four independent decisions

The long-term state should be represented as four orthogonal dimensions:

```text
StrengthPolicy: Full | Fixed(Elo) | Adaptive(Elo posterior)
OpponentClass:  Human | KnownEngine | UnknownComputer | SuspectedEngine | Uncertain
Situation:      Normal | Convert | Defend | Clinch
StylePolicy:    HumanPolicy | Forcing | Simplifying | Venom
```

A style must never bypass the strength policy. In particular,
`UCI_LimitStrength` always means play **at** `UCI_Elo`.

## Implemented foundation

The current first phase adds:

- persistent `OpponentDescriptor` metadata that survives `ucinewgame`;
- separate known-engine identity and declared playing strength;
- a 3,551-bucket posterior with one integer-Elo bucket from 100 through 3650;
  credible intervals remain mandatory because granularity is not precision;
- separate human/engine probabilities and an explicit `uncertain` class;
- a game-latched engine classification to prevent threshold flapping;
- conservative Auto-troll gating: known/computer/uncertain opponents cannot be
  auto-trolled;
- recent-sample volatility independent of convergence from the 1500 prior;
- fixed-strength precedence over every persona;
- deferred low-clock observations rather than silently discarding them;
- bounded fixes for forced-reply CLINCH scoring and mixed-depth negative losses.

The heuristic posterior is infrastructure, not a claim of production-grade
calibration. A learned likelihood model should replace it after balanced data
is available.

## Contextual persona response policy

Persona switching now consumes the board phase, check state, legal-move count,
current evaluation, previous evaluation, queen presence, opponent class, and
strength policy. Priority is:

1. fixed `UCI_Elo` → MATCH selector at the requested strength;
2. clearly worse or under pressure in check → DEFEND;
3. unrestricted/behaviorally detected engine in a non-losing position → FULL;
4. forced mate, fresh tactical error, +140cp live swing, winning endgame, or
   stable large lead versus a weaker opponent → PUNISH/convert;
5. confidently human, late, queen-rich, drawish game → CLINCH;
6. otherwise → conservative MATCH.

CLINCH is deliberately withheld from engines and uncertain opponents because
human-only traps are not an appropriate response there. Queenless winning
endgames convert through PUNISH; queenless equal endings remain MATCH. A sharp
downward eval swing breaks PUNISH/CLINCH hysteresis, while stable conversion
holds PUNISH. UCI logs include both the transition reason and move-selection
reason.

## Runtime evidence model

For each informative opponent move, collect:

- best and played move scores;
- legal move count and top-2/top-5 score gaps;
- policy entropy and played-move human-policy likelihood;
- tactical/check/forced/book status;
- game phase and material;
- think time, remaining time, increment, and position difficulty;
- metadata identity and declared current playing strength.

Book and forced moves should contribute very little rating/type evidence.
Accuracy alone is weak engine evidence: GMs can play engine-best moves. Timing
regularity is only a modulator, because real humans premove more often than bot
accounts and timing classifiers fail on unseen accounts. Only sustained
measurement-ceiling move quality can classify an undeclared engine; regular
clock allocation may lower the evidence threshold but can never fire alone.
The committed account-disjoint public-data replication does not validate a
standalone timing classifier (AUC 0.413, 95% CI 0.260–0.575); its BOT labels are
affirmative, while its comparison accounts are honestly labeled `unmarked`,
not verified human. See `docs/timing-classifier-validation.md`.

## Safe decision policy

Until classification is confident:

- no Auto trolling;
- use the upper credible strength bound, capped by engine capability;
- unrestricted/suspected engines receive full strength, while a known engine with an explicit low current rating can be conservatively matched but remains anti-troll locked;
- `UCI_LimitStrength` ignores detection for strength and remains fixed;
- explicit `Troll=On` remains a user override.

A limited Stockfish can have:

```text
identity = KnownEngine(Stockfish)
declared/observed playing strength ~= 1500
auto-troll lock = true
```

Identity and current strength are not the same fact.

## Balanced data, not natural-distribution data

Raw public games cluster around middle ratings. The default manifest now uses
one exact integer-Elo cell for every source rating from 100 through 3650:

```text
exact Elo × time control × color × result
```

This prevents middle ratings from consuming tail quotas. Empty/sparse exact-Elo
cells remain honestly sparse—they are never filled by duplicating games. Model
accuracy still comes from continuous parameter sharing and credible intervals,
not from pretending every exact rating has abundant data.

Position extraction then balances:

```text
game phase × position difficulty × tacticality × policy entropy
```

The repository includes `tools/build_balanced_manifest.py` and
`config/elo_sampling.json` for the metadata-stage reservoir. The manifest stores
source byte ranges and the selected player perspective without copying PGNs.

### Sampling requirements

- fixed quota per cell;
- cap per player per cell;
- cap positions per game and enforce ply spacing;
- deduplicate positions and repeated openings;
- keep platform/time-control rating pools separate or learn explicit mappings;
- do not duplicate rare tail records to fill a quota;
- widen sparse tail bands or use a continuous Elo-conditioned shared model;
- keep human and engine datasets separate.

### Engine data

Generate controlled games from multiple engine families, versions, node/time
limits, skill settings, humanized engines, openings, and timing models. Do not
trust `UCI_Elo` labels directly. Calibrate each configuration in an anchored
match league and use the measured effective strength.

### Human/engine classifier data

Match human and engine examples by effective strength, time control, phase,
and difficulty. Otherwise the classifier learns the invalid shortcut
`strong = engine`.

Hard negatives must include:

- GMs and titled humans in preparation and tactics;
- weak/limited Stockfish, Leela, Maia, and human-policy bots;
- engines with randomized delays;
- human time scrambles;
- improving, declining, and sandbagging profiles.

## Split and evaluation discipline

Use player-disjoint human splits, engine-family-disjoint engine splits,
game-disjoint splits, future-month holdouts, and opening-family holdouts where
possible.

Always report both:

1. **balanced macro metrics** across every Elo/type/time cell;
2. **natural-traffic metrics** for expected production behavior.

Required metrics include posterior calibration and interval coverage, per-band
rating error, GM false-engine rate, weak-engine false-human rate, time to safe
classification, Auto-troll false-positive rate, and fixed-Elo stability across
clock tiers.

## Next phase

1. Package and checksum the real Maia policy asset.
2. Train a continuous-Elo, time-control-conditioned human policy.
3. Train difficulty-conditioned strength likelihoods from a stronger offline
   oracle.
4. Train a separate human/engine classifier with hard negatives.
5. Run in shadow mode: log predictions without changing moves.
6. Enable behavior only after per-band calibration and false-positive gates
   pass.
