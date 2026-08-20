# Unchessed Hydra Aegis v4: legal-set prediction and proof-aware search

## 1. What v4 changes

Aegis v3 made XT-NNUE adaptive and gave uncertainty a calibration contract. Its
remaining policy path was still incomplete:

- the old training record contained only the played move, not the legal set;
- a 4,096 source/destination target collapsed four underpromotions;
- no trainer implemented the 2/128, 4/192, and 8/256 elastic paths;
- policy confidence did not say how costly each alternative move might be;
- conformal coverage could be mistaken for permission to prune legal moves; and
- future dirty hypergraph kernels had no exact multiset-delta oracle.

V4 is a concrete policy/search revision. It adds a frozen promotion-aware legal
set, per-action regret labels and predictions, a true elastic A100 trainer,
exact Rust legal-action generation, a full-refresh-backed hypergraph-delta
oracle, and a search contract that **always preserves full legal fallback**.

Implemented does not mean trained. The repository still has no trained Aegis
v4 checkpoint, quantized package, Rust neural inference, integrated NPS result,
Elo estimate, or SPRT result.

## 2. Architecture

```text
                 three-stage XT-NNUE (v3, unchanged)
                              |
                    alpha-beta value authority
                              |
             exact promotion-aware legal move set
                              |
          shared nested Chessformer trunk, 64 tokens
             /                 |                 \
       2 layers/128       4 layers/192       8 layers/256
             \                 |                 /
       shared legal policy + evidential WDL + regret head
                   /                         \
      private human adapter            private guide adapter
      Elo/time/8-ply history            board/search semantics
                   \                         /
            conformal-regret candidate ordering
                              |
              search priority, never move deletion
```

V4 deliberately does not scale the trunk beyond eight layers or width 256.
The advance is a more faithful action space and safer search interface, not an
unsupported assumption that a larger transformer is stronger.

## 3. Exact action vocabulary

A policy action is

```math
a(m)=from(m)+64\,to(m)+4096\,promotion(m),
```

where

```text
promotion = 0 ordinary, 1 knight, 2 bishop, 3 rook, 4 queen.
```

Therefore

```math
|A|=64\cdot64\cdot5=20480.
```

Only legal actions are evaluated. The Rust generator obtains fully legal moves
from the engine move generator, maps black-to-move positions to mover-relative
coordinates, preserves all four promotion classes, sorts actions, and rejects
rather than truncates a set above the theoretical maximum of 218.

For legal set `L(s)`, policy logits are

```math
\ell_m=\frac{q_{from(m)}^T k_{to(m)}}{\sqrt d}+b_{promotion(m)},
\qquad m\in L(s).
```

Dense policy needs 4,096 source/destination dots before promotions. Legal-only
policy needs at most 218, a calculated 18.79-fold reduction in dot products.

## 4. Shared nested exits

V4 implements one once-for-all network:

```text
exit 0: first 2 blocks, first 128 channels
exit 1: first 4 blocks, first 192 channels
exit 2: all 8 blocks, all 256 channels
```

Every embedding, QKV matrix, attention projection, gated FFN, normalization,
and head uses prefix channels. Policy, WDL, and regret heads are shared across
exits rather than duplicated. This lowers the calculated package to 4,222,905
parameters while forcing shallow and full paths to use the same semantics.

The A100 trainer executes all three paths per batch. Full-exit outputs teach the
smaller exits:

```math
L_{exit}^{policy}=KL(stopgrad(\pi_F)\|\pi_e),
```

```math
L_{exit}^{value}=KL(stopgrad(p_F)\|p_e),
```

```math
L_{exit}^{repr}=1-\cos(h_e,stopgrad(h_F[:d_e])).
```

## 5. Private policy conditioning

The board trunk receives piece, square, castling, exact EP, and halfmove state.
It does not receive player identity, target Elo, time class, or move history.

The private human/guide adapters receive:

- continuous target Elo;
- bullet/blitz/rapid/classical/unknown class;
- eight prior moves, newest first;
- source, destination, promotion identity, and history position; and
- a rank-16 low-rank residual on body/source/target policy projections.

For projection `W` and persona `z`:

```math
y=Wx+B_z A_z(x+h_{history})/16.
```

Only policy tensors consume `h_history`. WDL and legal regret use the board
representation. This preserves transposition-consistent search value while
allowing human move prediction to depend on recent play.

## 6. Evidential WDL

Every exit predicts evidence

```math
e_k=softplus(z_k),\quad \alpha_k=e_k+1,
```

```math
p_k=\alpha_k/\sum_j\alpha_j,\quad u=3/\sum_j\alpha_j.
```

Training uses the evidential squared-error/variance objective plus annealed KL
toward the uniform Dirichlet prior. Exit thresholds are fitted on a calibration
split and are not tuned against the final holdout.

## 7. Legal regret distribution

For each legal move, a 32-wide pair representation is

```math
r_m=tanh(R_f h_{from(m)}+R_t h_{to(m)}+e_{promotion(m)}).
```

The head predicts nonnegative mean regret and log scale:

```math
(\hat r_m,\log s_m)=H_r(r_m),\qquad \hat r_m=softplus(z_m).
```

Given a common-budget teacher regret `r_m`, train

```math
L_{regret}=\frac{(r_m-\hat r_m)^2}{2s_m^2}+\log s_m.
```

Teacher policy is reconstructed only over legal moves:

```math
\pi_T(m|s)=\frac{\exp(-r_m/\tau_r)}
{\sum_{j\in L(s)}\exp(-r_j/\tau_r)}.
```

The default research temperature is 90 cp. Human records use the actually
played legal action; guide records use all common-budget regrets. Identities are
not silently blended in one target.

## 8. Conformal-regret candidate sets

On a calibration split, form upper nonconformity

```math
c_m=\frac{r_m-\hat r_m}{\max(\epsilon,s_m)}.
```

Let `q_0.995` be the empirical upper quantile. Runtime regret bounds are

```math
U_m=\hat r_m+q_{0.995}s_m.
```

A priority set may select

```math
S=\{m:U_m\le 80\text{ cp}\},\qquad |S|\le16.
```

This is an anytime ordering device:

1. search `S` first;
2. use its result to improve alpha-beta bounds and ordering;
3. search every remaining legal move required by ordinary alpha-beta.

The implemented Rust plan always sets

```text
full_legal_fallback_required = true
noncandidate_pruning_allowed = false
```

because 99.5% empirical coverage is not a proof about a particular chess
position. Engines, GMs, unknown opponents, FULL, PUNISH, and DEFEND remain on
alpha-beta authority regardless of policy confidence.

## 9. `UNCHD4R0` data ABI

### 9.1 Header

The header remains 64 bytes and contains:

```text
8-byte magic UNCHD4R0
u16 version 4
u16 header width 64
u16 record width 1088
u16 mandatory semantic flags
u32 endian marker 0x01020304
u64 record count
32-byte schema SHA-256
u32 CRC32 over bytes 0..59
```

### 9.2 Record

A record is 1,088 bytes:

| Offset | Bytes | Meaning |
|---:|---:|---|
| 0 | 160 | complete validated v3 semantic record |
| 160 | 2 | legal action count, 1..218 |
| 162 | 2 | exact played target action |
| 164 | 2 | exact teacher-best action or `0xffff` |
| 166 | 1 | human or guide policy kind |
| 167 | 1 | per-action-regret presence flag |
| 168 | 436 | 218 sorted unique `u16` legal action slots |
| 604 | 436 | 218 signed regret slots |
| 1040 | 48 | reserved zero |

Unused action slots are `0xffff`; unused/unlabelled regrets are `0x7fff`.
Teacher-best must be legal and have zero regret. The played move must exactly
match the v3 source/destination and promotion fields.

The 6.8-fold expansion over v3 is intentional. V4 trades data volume for an
unambiguous legal set and complete regret supervision. At 1,088 bytes, one GiB
holds 986,895 records.

## 10. Data generation and privacy

The Rust generator now emits human `UNCHD4R0` shards directly:

```bash
cargo run --release -p unchessed-datagen -- \
  policy-v4 human-train.aegis4 00112233445566778899aabbccddeeff \
  5000000 0.25 lichess-*.pgn
```

The 128-bit key feeds SipHash-2-4 pseudonyms for player and game identities.
Use the same private key for train/calibration/validation mining so overlap can
be detected; changing it between splits defeats the audit. The key is not
written to the shard. Use a private ingestion machine, disable shell history or
provide the command through a protected job system, and never commit the key or
raw account identifiers.

Generated human records include exact legal actions, promotion identity, WDL,
history, rating, time class, and increment. They intentionally do not fabricate
teacher regrets. Guide/regret shards require a separate common-budget teacher
annotation job. Calibration and final holdout must contain labelled actions if
regret coverage is to be claimed.

`tools/aegis_v4_data.py` verifies headers, schema hash, every legal set, target
membership, regret sentinels, teacher-best consistency, and game/player split
disjointness.

## 11. Exact hypergraph delta oracle

V3 extracted all relation groups but did not define executable incremental
deltas. V4 adds a full-refresh-backed multiset oracle:

```math
\Delta^- = R(s)\setminus R(s'),\qquad
\Delta^+ = R(s')\setminus R(s),
```

with duplicate multiplicity preserved independently for direct, x-ray, and
pawn topology relations. The implementation uses fixed stack arrays, sorted
multiset difference, and explicit overflow.

This oracle is not yet the optimized dirty closure. Its purpose is stricter:
every future local dirty updater must reproduce the oracle exactly over normal,
castling, en-passant, promotion, king-orientation, and random legal-tree tests.

## 12. A100 training

```bash
export TRAIN_POLICY_V4='/data/v4/train-*.aegis4'
export CAL_POLICY_V4='/data/v4/calibration-*.aegis4'
export VAL_POLICY_V4='/data/v4/final-holdout-*.aegis4'
export OUTPUT_DIR=/data/checkpoints/aegis-v4-chessformer
scripts/training/a100_hydra_v4_chessformer_train.sh
```

The launcher validates every shard and performs pairwise player/game leakage
audits before touching CUDA. The trainer uses BF16 autocast, TF32, fused AdamW,
pinned transfers, optional `torch.compile`, atomic current/best checkpoints,
and separate calibration/final metrics.

The branch training loss is

```math
L=0.35L_{human-policy}+0.20L_{guide-policy}
 +0.15L_{EDL}+0.15L_{regret}
 +0.10L_{exit}+0.05L_{representation}.
```

These branch weights are distinct from the ultimate joint Hydra objective in
the architecture config.

## 13. Calculated budget

| Component | Calculated value |
|---|---:|
| XT runtime | 12.83 MiB |
| XT state per ply | 1,280 bytes |
| Shared-head Chessformer | 4,222,905 parameters |
| Int8 storage target | 4.03 MiB |
| Legal regret head | 16,610 parameters |
| 2/128 exit | 47.0 MFLOP |
| 4/192 exit | 171.6 MFLOP |
| 8/256 exit | 551.1 MFLOP |
| Dense-to-legal policy dot reduction | 18.79x |
| V4 record | 1,088 bytes |
| Records/GiB | 986,895 |

These are architecture calculations, not measured model latency or strength.

## 14. Current verification and missing evidence

Implemented and locally testable without a model:

- exact Rust legal action generation and underpromotion separation;
- safe candidate-set ordering with mandatory full fallback;
- full-refresh-backed relation delta reconstruction;
- Rust human-v4 PGN generation;
- cross-language Rust-writer/Python-reader compatibility;
- v4 record validation and split auditing;
- elastic/evidential/legal-regret A100 trainer source;
- deterministic architecture reports.

Still required before production:

- execute trainer self-check and full jobs on the A100;
- generate common-budget guide/regret and disjoint calibration shards;
- report all three exits' policy/WDL/regret metrics;
- fit exit and regret conformal tables without test leakage;
- quantize and export `UNCHAEG4`;
- scalar then SIMD Rust neural inference;
- optimize dirty relation updates against the exact oracle;
- measure integrated NPS and deployment CPU latency; and
- run isolated paired-game SPRTs for each strength-changing mode.
