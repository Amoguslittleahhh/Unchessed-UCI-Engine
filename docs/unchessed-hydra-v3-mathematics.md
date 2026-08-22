# Unchessed Hydra Aegis v3: certified adaptive compute

> **Experimental lineage document.** This is not a product version. The canonical architecture is [Unarchitectured v1](unarchitectured-v1.md).

## 1. Scope and status

Aegis v2 defined a useful target, but most of it was still a mathematical
architecture. Aegis v3 is deliberately an **implementation-and-reliability
revision**, not merely a larger network. It closes four concrete gaps:

1. a position-only/full binary evaluator wastes the cheap direct-threat
   residual when x-ray/topology refresh is unnecessary;
2. heteroscedastic variance models data noise but does not by itself measure
   epistemic disagreement;
3. fixed confidence thresholds have no empirical coverage guarantee; and
4. the old 104-byte policy record cannot represent underpromotion identity,
   history, time class, player-disjoint splits, WDL, or counterfactual regret.

The repository now contains:

- exact Rust direct, x-ray, and pawn/king topology feature extractors;
- a conservative three-tier Rust router and conformal pruning bounds;
- an A100 trainer for all three XT groups and dual fast heads;
- a frozen, validated 64-byte-header/160-byte-record data ABI;
- game/player split-leakage auditing;
- board-only versus temporal-policy cache contracts;
- calibrated elastic-exit and alpha-beta policy-veto contracts; and
- deterministic architecture calculations and tests.

It still contains **no trained Aegis v3 model**, no `UNCHAEG3` exporter, and no
integrated v3 evaluator. Calculated FLOPs and memory below are not benchmarked
latency, holdout accuracy, NPS, Elo, or SPRT evidence.

## 2. Architecture

```text
                         board state
                              |
               256-wide HalfKAv2 accumulator
                  /           |             \
       fast dual heads   direct relations   x-ray + topology
             |                 |                    |
       mu, aleatoric,      direct mu,u          full value
       epistemic u              \                 /
                  calibrated three-stage XT router
                              |
                     alpha-beta authority
                              |
       board-only Chessformer trunk (value/guide semantics)
                 /                               \
       guide policy adapter             private human adapter
                                        Elo + time + 8 plies
                 \                               /
                 legal-move gather + safety veto
```

The two invariants are:

- search correctness never depends on a learned uncertainty being correct; and
- history/player context can alter human move imitation, but cannot alter the
  board-state value used by alpha-beta.

## 3. Three-stage XT-NNUE

For perspective `p`, retain the v2 accumulators

```math
A_P(s,p),\quad A_2(s,p),\quad A_3(s,p),\quad A_T(s,p),
```

for position, direct relations, x-ray triples, and pawn/king topology. V3 has
three heads:

```math
H_0(A_P) \rightarrow \{(\mu_j,\log\sigma_j^2)\}_{j=1}^{2},
```

```math
H_1(A_P,A_2) \rightarrow (\mu_D,\log\sigma_D^2),
```

```math
H_2(A_P,A_2,A_3,A_T) \rightarrow v_F.
```

The intermediate direct tier is important: direct relations are frequent,
cheap, and useful in ordinary positions, while ray closure and topology updates
are less attractive at every low-value leaf.

### 3.1 Aleatoric and epistemic uncertainty

Two independently initialized fast heads are bootstrap-trained. Define

```math
\bar\mu = \frac{1}{2}(\mu_1+\mu_2),
```

```math
\sigma_{aleatoric}^2 = \frac{1}{2}\sum_j \exp(\log\sigma_j^2),
```

```math
\sigma_{epistemic}^2 = \frac{1}{2}\sum_j(\mu_j-\bar\mu)^2,
```

```math
\sigma_{total} = \sqrt{\sigma_{aleatoric}^2+\sigma_{epistemic}^2}.
```

The heteroscedastic objective for a member is

```math
L_{fast,j}=
\frac{(\operatorname{stopgrad}(v_F)-\mu_j)^2}{2\sigma_j^2}
+\frac{1}{2}\log\sigma_j^2.
```

This still does not make the raw variance calibrated. It only gives the
calibration phase a useful ranking signal.

### 3.2 Conformal score bounds

On a calibration set disjoint from training and final testing, collect
normalized nonconformity scores by material phase and search-depth bucket:

```math
r^- = \frac{\bar\mu-v_F}{\max(1,\sigma_{total})},\qquad
r^+ = \frac{v_F-\bar\mu}{\max(1,\sigma_{total})}.
```

Let `q-` and `q+` be one-sided empirical quantiles at the required coverage.
Runtime bounds are

```math
LCB=\bar\mu-q^-\sigma_{total},\qquad
UCB=\bar\mu+q^+\sigma_{total}.
```

Fail-high pruning may use the fast path only if

```math
LCB \ge \beta + margin,
```

and fail-low only if

```math
UCB \le \alpha - margin.
```

A missing, non-finite, stale, or under-covering calibration table disables the
shortcut. Root, PV, check, mate-score, and explicitly tactical nodes always use
the full evaluator.

### 3.3 Three-stage routing

```text
root / PV / check / tactical pressure -> full
fast uncertainty <= tau0             -> position-only
otherwise evaluate direct relations
  direct uncertainty <= tau1         -> direct tier
  otherwise                           -> full
```

The defaults `tau0=0.10` and `tau1=0.18` are experiment seeds, not validated
production thresholds.

## 4. Exact relation schemas

### 4.1 Direct relations

Unchanged from v2:

```math
12\ attacker\ classes \times 12\ target\ classes \times 15\times15
=32400.
```

### 4.2 X-ray hyperedges

For every bishop, rook, and queen ray, the extractor records the first and
second occupied squares:

```math
x=(attacker,blocker,target,direction).
```

Friendly blockers are retained because they distinguish batteries and
discovered attacks. The second occupied square terminates the ray; extraction
never looks through a third piece.

```math
idx_x=(((12a+b)12+t)8+d),\qquad |X|=12^3\cdot8=13824.
```

Direction is canonicalized with the same vertical and king-file orientation as
HalfKAv2. The Rust implementation uses fixed stack storage and reports overflow
rather than truncating.

### 4.3 Pawn/king topology

For every pawn and both kings, use a canonical 3-file by 4-rank window. The
pre-hash key is

```text
bits  0..11  perspective-owned pawn occupancy
bits 12..23  opponent pawn occupancy
bits 24..25  own pawn / enemy pawn / own king / enemy king
bits 26..31  passed, isolated, connected, lever, doubled, blocked
```

A fixed two-round 32-bit integer finalizer selects one of 4,096 rows. Hash
collisions are therefore intentional model sharing, not accidental format
behavior. The Rust runtime and A100 trainer implement the same key and hash.

## 5. Board-state and human-history separation

V2 repeated Elo context in every square token. If a value head consumes that
trunk, identical chess states can receive different values depending on whom
the engine thinks it is playing. V3 prohibits that coupling.

The board trunk receives only:

- 64 perspective-normalized piece/square tokens;
- castling rights;
- exact en-passant state;
- halfmove bucket; and
- geometric attention relations.

A private human-policy adapter receives:

- continuous target Elo;
- time class;
- the last eight normalized moves, including move kind and promotion bits; and
- a private rank-16 adapter on policy projections.

Gradients from this adapter do not enter the value head. Engine/GM play still
uses authoritative alpha-beta; a guide adapter may order legal moves but cannot
replace search value.

### 5.1 Exact cache keys

Board-value cache:

```text
(full position hash, exact EP square, halfmove bucket, model UUID)
```

The exact EP square is included because repetition hashing intentionally omits
pseudo-uncapturable EP targets while the token encoder exposes EP state.

Human-policy cache adds:

```text
(Elo context, time class, all 8 history slots, history length, persona)
```

History is stored in the key, not collapsed to an unchecked digest. Ordinary
language-model KV caching remains invalid for this bidirectional board encoder.

## 6. Frozen `UNCHD3R0` data ABI

### 6.1 Header: 64 bytes

| Offset | Width | Field |
|---:|---:|---|
| 0 | 8 | ASCII magic `UNCHD3R0` |
| 8 | 2 | version 3 |
| 10 | 2 | header bytes = 64 |
| 12 | 2 | record bytes = 160 |
| 14 | 2 | mandatory semantic flags |
| 16 | 4 | endian marker `0x01020304` |
| 20 | 8 | record count |
| 28 | 32 | SHA-256 of canonical schema descriptor |
| 60 | 4 | CRC32 of bytes 0..59 |

### 6.2 Record: 160 bytes

| Offset | Width | Field |
|---:|---:|---|
| 0 | 96 | 12 mover-normalized `u64` bitboards |
| 96 | 2 | selected source/destination move |
| 98 | 1 | promotion: none/N/B/R/Q |
| 99 | 1 | side-to-move WDL |
| 100 | 2 | mover rating |
| 102 | 1 | normalized castling rights |
| 103 | 1 | EP file or `0xff` |
| 104 | 1 | halfmove clock, saturated |
| 105 | 1 | bullet/blitz/rapid/classical/unknown |
| 106 | 1 | semantic presence/special-move flags |
| 107 | 1 | history length |
| 108 | 16 | eight full `u16` normalized prior moves |
| 124 | 8 | privacy-preserving game hash |
| 132 | 8 | privacy-preserving player hash |
| 140 | 2 | teacher score |
| 142 | 2 | common-budget teacher best move |
| 144 | 2 | teacher best score |
| 146 | 2 | teacher selected-move score |
| 148 | 2 | game ply |
| 150 | 4 | remaining milliseconds or `0xffffffff` |
| 154 | 4 | increment milliseconds or `0xffffffff` |
| 158 | 2 | reserved zero |

Promotion identity no longer collapses all underpromotions into the same
source/destination label. Regret is reconstructed as

```math
r(m)=\max(0,V(best)-V(m)).
```

Game/player hashes support mandatory disjoint split auditing without storing
raw account names. A deployment must generate them with a secret keyed hash on
the ingestion host; unhashed identities must not enter training shards.

The tool `tools/aegis_v3_data.py` atomically writes shards, verifies schema,
CRC, lengths and record invariants, and rejects game/player overlap between
train and validation sets.

## 7. Elastic Chessformer reliability

The 2/128, 4/192, and 8/256 exits remain. V3 adds a coverage certificate in
basis points for each early exit. A cheap exit is allowed only when both its
uncertainty threshold and its frozen holdout coverage pass. Failure escalates
to the next exit; it never downgrades compute.

For human-policy direct play, alpha-beta applies a final veto:

```text
forced mate found by search          -> use alpha-beta move
candidate permits forced mate        -> use alpha-beta move
candidate common-budget loss > limit -> use alpha-beta move
otherwise                            -> accept policy move
```

This is the safety boundary, not a training regularizer.

## 8. V3 training sequence on A100

1. Train the full XT head on board-state labels.
2. Distill direct and dual fast heads from the full head.
3. Freeze model weights; fit phase/depth conformal tables on a separate
   calibration set.
4. Measure final value, calibration, and out-of-distribution metrics on a third
   untouched set.
5. Train the board-only Chessformer trunk and full exit.
6. Train elastic exits with sandwich/random-exit distillation.
7. Train private human and guide policy adapters. History gradients are
   policy-only.
8. Quantization-aware fine-tune, export, and compare float/int8 drift.
9. Add scalar Rust inference before SIMD.
10. Run isolated integrated-NPS and paired-game gates for each compute tier and
    backend.

The v3 XT trainer requires all three paths explicitly:

```bash
python tools/train_nnue_xt_v3_a100.py train \
  --train /data/nnue/train-*.bin \
  --calibration /data/nnue/calibration-*.bin \
  --validation /data/nnue/final-holdout-*.bin \
  --output /data/checkpoints/aegis-v3-xt.pt
```

The three sets may not be aliases or random slices of overlapping games.

## 9. Calculated budget

From `config/unchessed_hydra_v3.json`:

| Component | Calculated budget |
|---|---:|
| XT runtime | 12.83 MiB |
| XT state per ply | 1,280 bytes |
| phase/depth conformal table | 512 bytes |
| Chessformer parameters | 4,668,583 |
| Chessformer int8 target | 4.45 MiB |
| private policy adapters | 49,152 parameters |
| private history adapter | 13,120 parameters |
| 2/128 exit | 45.9 MFLOP |
| 4/192 exit | 170.0 MFLOP |
| 8/256 exit | 549.0 MFLOP |
| legal-only versus dense policy dots | 18.79x reduction |
| v3 data record | 160 bytes |

The model increase over v2 is mostly the two private low-rank policy adapters,
history projection, and extra uncertainty heads. Incremental accumulator state
remains 1,280 bytes per ply.

### 9.1 Standalone extractor measurement

A release Rust microbenchmark over eight positions and both perspectives
measured a full direct+x-ray+topology refresh plus synthetic width-32/16/16
int16 accumulation at **2,264.2 ns/call** on the sandbox's Intel Xeon 2.60 GHz
CPU. Mean active counts were 38.6 direct relations, 11.2 x-rays, and 15.1
topologies. This is a shared-host standalone measurement—not dirty-update cost,
integrated NPS, trained inference latency, accuracy, or Elo. Full provenance is
in `benchmarks/hydra-v3/feature-microbenchmark.json`.

## 10. Promotion gates

Aegis v3 stays default-off until all applicable gates pass:

- Rust/GPU direct, x-ray, and topology feature indices agree on frozen fixtures;
- dirty updates exactly equal full refresh through random legal trees;
- fast/direct/full float heads have separately reported holdout metrics;
- uncertainty-error correlation is positive and conformal coverage meets its
  frozen target in every promoted phase/depth bucket;
- calibration and final test games/players are disjoint;
- underpromotion, EP, castling, rule-fifty, and repetition suites pass;
- history changes human policy but cannot change board value;
- early exits meet coverage and deployment-CPU latency gates independently;
- quantized drift is reported before enabling runtime inference;
- integrated engine NPS is measured separately from standalone model latency;
- alpha-beta safety veto has zero forced-mate regressions; and
- each strength-changing mode passes an isolated paired-game SPRT.

V3's intended breakthrough is not an unsupported accuracy claim. It is the
combination of richer tactical structure, a usable middle compute tier,
epistemic plus aleatoric uncertainty, empirical coverage, and strict separation
between human imitation and search authority.
