# Unchessed Hydra v1: mathematical architecture

> **Experimental lineage document.** This is not a product version. The canonical architecture is [Unchessed Apex v1](unchessed-apex-v1.md).

## Status and objective

Hydra is a proposed unified neural base with two execution branches:

- **XT-NNUE:** an incrementally updateable value evaluator for alpha-beta;
- **Chessformer:** a geometry-aware root policy for human emulation and move
  ordering.

The branches are trained jointly and packaged together, but they are not trusted
equally at runtime. Alpha-beta plus XT-NNUE remains the tactical/value authority.
Chessformer proposes, conditions, and orders moves. No accuracy, latency, or Elo
improvement is claimed before trained weights and game gates exist.

The architecture budget is executable:

```bash
python tools/hydra_architecture_report.py \
  --config config/unchessed_hydra_v1.json \
  --json benchmarks/hydra-v1/report.json \
  --markdown benchmarks/hydra-v1/result.md --check
```

## 1. Notation

Let a chess state be

```math
s = (B, c, C, e, h, m),
```

where `B` is the 64-square board, `c` the side to move, `C` castling rights,
`e` en-passant state, `h` the rule-fifty counter, and `m` contextual metadata
such as target Elo and persona.

For perspective `p`:

- `P(s,p)` is the multiset of active HalfKAv2_hm piece-square features;
- `R(s,p)` is the multiset of occupied-target attack/defense relations;
- `K_p` is the oriented king bucket;
- `N_R = |R(s,p)|`;
- `Q_A = 255` is the proposed accumulator quantization scale;
- `d_P = 256` is positional width;
- `d_R = 32` is threat-residual width.

The Chessformer board is a matrix

```math
X(s,E,z) in R^(64 x d),
```

where `E` is target Elo and `z` is persona/time context.

## 2. XT-NNUE positional accumulator

The positional accumulator remains a sparse sum:

```math
a_P(s,p) = b_P + sum_{i in P(s,p)} W^P_i,
```

with `W^P in R^(22528 x 256)`.

For a move from state `s` to `s'`, define removed and added positional feature
multisets `D_P^-` and `D_P^+`. Unless the perspective king changes bucket:

```math
a_P(s',p) = a_P(s,p)
           - sum_{i in D_P^-} W^P_i
           + sum_{i in D_P^+} W^P_i.
```

If the king bucket/orientation changes, the exact full sum is recomputed. This
is the already validated incremental invariant and remains the reference path.

## 3. Factorized threat residual

### 3.1 Relation index

Every occupied-target relation is represented by

```math
r = (a, t, delta),
```

where:

- `a in {0,...,11}` is attacker piece type plus perspective ownership;
- `t in {0,...,11}` is target piece type plus ownership;
- `delta = (df,dr)` with each component in `[-7,7]`.

The flat inference index is

```math
idx(a,t,df,dr)
 = ((12a + t) * 225) + 15(dr + 7) + (df + 7).
```

Thus the materialized inference table has

```math
12 * 12 * 225 = 32400
```

rows.

### 3.2 CP-factorized training parameterization

Training a completely independent vector for each relation wastes statistical
strength. Hydra trains the relation embedding with a rank-`q` CP/Tucker-style
factorization plus a hashed residual:

```math
u_r = A_a elementwise B_t elementwise C_delta,
```

```math
e_r = D^T u_r + H_{hash(r)}.
```

Dimensions are:

```math
A in R^(12 x q),
B in R^(12 x q),
C in R^(225 x q),
D in R^(q x 32),
H in R^(4096 x 32),
q = 16.
```

The structured factor uses only

```math
q(12 + 12 + 225 + 32) = 4496
```

parameters. The hashed residual contributes 131,072 parameters and captures
exceptions without allocating a dense million-parameter optimizer state.

At export, all 32,400 vectors are materialized once:

```math
E^R_r = Quantize_8(e_r),
```

so CPU inference performs one indexed row addition per active relation and no
factorization arithmetic.

### 3.3 Residual accumulator

```math
a_R(s,p) = b_R + sum_{r in R(s,p)} E^R_r.
```

A normalized companion scalar is retained:

```math
rho(s,p) = log(1 + N_R) / log(257).
```

This lets the head distinguish one strong tactical relationship from many
ordinary defended-piece relationships without normalizing away count entirely.

### 3.4 Dirty relationship update

Let `D_R^-` and `D_R^+` be relations removed and added by a move. Then

```math
a_R(s',p) = a_R(s,p)
           - sum_{r in D_R^-} E^R_r
           + sum_{r in D_R^+} E^R_r.
```

The dirty closure is the union of:

```math
D = {from, to, captured, ep-captured}
    union leapersIncident(D)
    union firstSlidersOnRays(D)
    union kings.
```

Relationships with an attacker or target in the closure are regenerated. If a
line-of-sight dependency cannot be proven contained, inference falls back to a
full residual refresh. Full-refresh and dirty-update equality is a hard test,
not an approximate condition.

## 4. Per-channel quantization

A single global scale wastes int8 range because feature channels have different
variance. Hydra uses one power-of-two scale per threat channel:

```math
s_j = 2^(round(log2(max_r |e_{rj}| / 127))),
```

```math
q_{rj} = clip(round(e_{rj}/s_j), -127, 127).
```

The runtime accumulator is int16:

```math
z_j = sum_r q_{rj},
```

and dequantization is folded into the first head layer:

```math
W'_{kj} = W_{kj} * s_j.
```

For QAT, the straight-through estimator is

```math
FakeQuant(x) = x + stopgrad(s * round(x/s) - x).
```

Clipping penalties are explicit:

```math
L_clip = mean(max(0, |x| - 127s)^2).
```

The positional transformer remains int16 at scale 255. Threat rows are int8
and sign-extended during accumulation. The architecture must prove that every
legal active-feature combination remains in int16 range.

## 5. Quantization error budget

For one channel with at most `n` active relations and rounding error at most
`s_j/2` per row:

```math
|epsilon_j| <= n * s_j / 2.
```

For the first affine head with weights `w_j`, a conservative output error is

```math
|epsilon_out| <= sum_j |w_j| * |epsilon_j|.
```

This bound is pessimistic but useful for rejecting unsafe scales before a game
is played. Empirical drift is reported separately as mean, p95, p99, and
maximum centipawn error on a frozen suite.

## 6. XT-NNUE interaction head

For side-to-move and non-side-to-move accumulators, define:

```math
p = concat(SCReLU(a_P^stm), CReLU(a_P^stm),
           SCReLU(a_P^nstm), CReLU(a_P^nstm)),
```

```math
t = concat(CReLU(a_R^stm), CReLU(a_R^nstm), rho_stm, rho_nstm).
```

A cheap cross gate lets explicit threats modulate positional channels:

```math
g = sigmoid(G t),
```

```math
p_hat = p elementwise (1 + alpha * g),
```

where `alpha` is constrained to `[0,0.25]`. The semantic head input is

```math
h_0 = concat(p_hat, t) in R^1090.
```

It is padded with 30 structural zeros to a SIMD-friendly width:

```math
h_tilde_0 = concat(h_0, 0_30) in R^1120.
```

For material-phase stack `k`:

```math
h_1 = W^k_1 h_tilde_0 + b^k_1,    dimension 16,
```

```math
h_2 = concat(SCReLU(h_1), CReLU(h_1)), dimension 32,
```

```math
h_3 = CReLU(W^k_2 h_2 + b^k_2),  dimension 32,
```

```math
v_N(s) = W^k_3 h_3 + b^k_3.
```

Only one of eight stack heads is evaluated. Stack selection is a deterministic
function of non-pawn material and phase, not a learned branch at inference.

## 7. Chessformer square tokens

For oriented square `i`, token input is

```math
x_i = E_piece[p_i] + E_square[i] + E_castle[C]
    + E_ep[e] + M_E(phi(E)) + E_persona[z].
```

`phi(E)` is a Fourier Elo embedding:

```math
phi(E) = [sin(2^k pi e), cos(2^k pi e)] for k=0,...,15,
```

with normalized Elo

```math
e = clip((E - 100)/3550, 0, 1).
```

This replaces separate rating-bucket networks with a smooth conditioning
function while preserving the ability to learn nonlinear rating effects.

## 8. Dynamic Geometric Attention Bias

Let `X` be the 64 token vectors. A compressed board context is

```math
c = LN(GELU(W_2 vec(W_1 X))).
```

For layer `l`, head `h`, and template `r`, mixture coefficients are

```math
lambda_{lhr} = [W^lambda_l c]_{hr}.
```

Learned dynamic bias is

```math
B^dyn_{lh} = sum_{r=1}^{32} lambda_{lhr} T_r,
```

where each `T_r in R^(64 x 64)`.

Hydra adds deterministic chess geometry templates `G_j(s)`:

- same rank, file, and diagonal;
- knight and king geometry;
- source attacks target;
- target attacks source;
- occupied target;
- same owner;
- clear ray and blocker count;
- pawn direction by perspective.

The final bias is

```math
B_{lh}(s) = B^dyn_{lh}(s) + sum_j theta_{lhj} G_j(s).
```

Attention is

```math
A_{lh} = softmax(Q_{lh} K_{lh}^T / sqrt(d_h) + B_{lh} + M),
```

where `M` contains any structural mask. This separates exact chess geometry
from learned global adaptation and should reduce the amount of data needed to
rediscover legal movement topology.

## 9. Elo/persona low-rank adapters

Instead of one full model per persona, every selected dense matrix may receive
a low-rank conditional update:

```math
W(E,z) = W_0 + U diag(g(E,z)) V^T,
```

where rank is 8 or 16 and

```math
g(E,z) = sigmoid(M concat(phi(E), E_persona[z])).
```

Only policy and the last two encoder blocks receive adapters initially. This
keeps the common tactical representation shared while allowing human Elo/style
variation near the output.

## 10. Policy, promotions, and value

For encoded square states `y_i`, source and destination vectors are

```math
q_i = W_q y_i,
k_j = W_k y_j.
```

Base move logit:

```math
ell_{ij} = q_i^T k_j / sqrt(d).
```

A complete legal move includes promotion class `u`:

```math
ell(i,j,u) = ell_{ij} + b_u^T y_j + b_castle + b_ep.
```

Illegal moves receive negative infinity before softmax. The value head predicts
three logits:

```math
p_C(s) = softmax(W_v mean_i(y_i) + b_v)
       = (P_loss, P_draw, P_win).
```

Expected scalar value is

```math
v_C(s) = P_win - P_loss.
```

Policy entropy is

```math
H_pi(s) = -sum_m pi(m|s) log pi(m|s).
```

It is used for time allocation and confidence, never as standalone evidence of
human or engine identity.

## 11. Unified Hydra training objective

Hydra alternates human-policy, engine-policy, and value batches. The full loss
is

```math
L = 0.30 L_XT-WDL
  + 0.15 L_teacher-score
  + 0.25 L_human-policy
  + 0.10 L_teacher-policy
  + 0.08 L_value-consistency
  + 0.04 L_representation
  + 0.05 L_quantization
  + 0.03 L_calibration.
```

The coefficients sum to one and are configuration, not hidden code constants.

### 11.1 XT WDL and robust score

```math
q_N = sigmoid(v_N / tau_N),
```

```math
L_XT-WDL = |q_N - y_WDL|^(5/2).
```

Use a Huber score target rather than raw MSE:

```math
L_teacher-score = Huber(v_N - v_teacher; delta=100 cp).
```

### 11.2 Human and teacher policy

```math
L_human-policy = -w(E,m) log pi_C(m_human|s,E,z),
```

with larger `w` for castling, en passant, and promotion.

For teacher distribution `pi_T` and temperature `T_p`:

```math
L_teacher-policy = T_p^2 KL(softmax(l_T/T_p) || softmax(l_C/T_p)).
```

### 11.3 Cross-branch value consistency

Convert XT scalar evaluation to a WDL distribution `p_N`. Then use symmetric
Jensen-Shannon consistency:

```math
m = (p_N + p_C)/2,
```

```math
L_value-consistency = 0.5 KL(p_N || m) + 0.5 KL(p_C || m).
```

Gradients may be stopped through the more accurate teacher branch during early
stages, then enabled symmetrically after both branches stabilize.

### 11.4 Representation alignment

Let

```math
r_N = LN(P_N concat(a_P^stm-a_P^nstm, a_R^stm-a_R^nstm)),
```

```math
r_C = LN(P_C mean_i(y_i)).
```

Use cosine alignment with variance preservation:

```math
L_representation = 1 - cosine(r_N,r_C)
                 + beta sum_j max(0, sigma_min - std(r_*j))^2.
```

The variance term prevents both branches from satisfying alignment by
collapsing to a constant vector.

### 11.5 Quantization distillation

Evaluate float and fake-quantized XT branches on the same state:

```math
L_quantization = |v_float - v_quant| / 100
               + KL(p_float || p_quant)
               + lambda_clip L_clip.
```

### 11.6 Calibration

For WDL class `y`:

```math
L_calibration = Brier(p_C,y) + Brier(p_N,y).
```

Calibration is reported by Elo, time control, phase, and tactical difficulty.

## 12. Alpha-beta policy integration

Transformer policy contributes a bounded move-order bonus, not a score:

```math
bonus(m,d) = lambda_0 exp(-d/tau_d)
             * clip(log(pi(m)+epsilon) - mean_log_pi,
                    -b_max, b_max).
```

Final move priority is lexicographic:

```text
TT move
winning captures / promotions
forced tactical moves
policy-guided quiets + history + continuation history
remaining quiets
losing captures
```

The policy cannot move a quiet move above a verified winning capture merely by
assigning high probability.

## 13. Human-policy safety veto

For policy candidate `m`, alpha-beta verifies a common-node candidate set. Let

```math
Delta(m) = V_AB(best) - V_AB(m).
```

The intended human loss budget is sampled from the existing heavy-tail model:

```math
mu(E) = 300 exp(-E/900),
```

```math
L ~ LogNormal(log(mu(E)) - sigma^2/2, sigma^2), sigma=1.1.
```

A move is eligible only if:

```math
not forced_mate_loss(m),
SEE_guard(m) passes,
Delta(m) <= min(L, L_max(E,phase)),
```

and the search depth/node budget is common across candidates. If no candidate
passes, use the alpha-beta best move.

## 14. Risk-aware persona routing

Hard safety constraints are applied first:

```text
known/suspected engine -> alpha-beta
uncertain identity     -> alpha-beta
FULL/PUNISH/DEFEND      -> alpha-beta
target Elo >= 2300      -> alpha-beta
missing model           -> alpha-beta
```

Only then may the router maximize expected utility. For backend `b` and
opponent posterior `q(o)`:

```math
J(b) = sum_o q(o) U(b,o) - lambda CVaR_alpha(loss_b).
```

```math
b_star = argmax_b J(b).
```

The CVaR term penalizes rare catastrophic failures of direct policy play. It is
estimated from held-out games, not hand-selected. Backend choice is latched per
move/game context to prevent oscillation.

## 15. Entropy-aware time allocation

Chessformer may modulate, but never trigger, time usage:

```math
complexity = clip(H_pi / log(N_legal), 0, 1),
```

```math
budget' = budget_AB * (1 + eta * (complexity - c_0)),
```

with `eta <= 0.25` and existing hard clock caps. Engine identity classification
is not an input to this equation except through the already selected backend.

## 16. Training curriculum on A100

### Stage A: separate stabilization

1. Load/freeze the current v3 positional transformer.
2. Train CP threat factors, hash residual, and new heads.
3. Train Chessformer human policy on player-disjoint data.
4. Train a separate guide policy from alpha-beta/MCTS teacher distributions.

### Stage B: cross-distillation

1. Unfreeze the positional table at 0.1x learning rate.
2. Enable value consistency with stop-gradient through the better holdout
   branch.
3. Enable representation alignment after 10% warmup.
4. Alternate human:teacher:value batches at 4:2:2 ratio.

### Stage C: quantization-aware fine tuning

1. Fake-quantize positional int16 scale 255.
2. Fake-quantize materialized threat rows to per-channel int8.
3. Fake-quantize dense heads to int8/int32.
4. Fine tune with quantization distillation at one tenth base learning rate.

### Stage D: export and search

1. Materialize all factorized threat rows.
2. Export one `UNCHHYD1` package with section hashes.
3. Validate scalar Rust inference against PyTorch.
4. Add SIMD kernels and exact equality tests.
5. Benchmark alpha-beta, policy-guided AB, and human-guarded policy separately.

## 17. Runtime package

Proposed package sections:

```text
magic UNCHHYD1
schema/version and model UUID
shared normalization and condition metadata
XT positional int16 table + channel scales
XT materialized threat int8 table + channel scales
8 quantized XT heads
Chessformer int8 weights and GAB templates
human adapter weights
teacher/guide adapter weights
calibration tables by Elo/time/phase
per-section SHA-256
```

The engine may load only XT sections in reviewer mode. The adapter loads the
Chessformer sections when a verified model is available. Missing sections cause
an explicit alpha-beta fallback.

## 18. Calculated architecture budget

From `config/unchessed_hydra_v1.json`:

- XT positional table: 11.00 MiB;
- materialized int8 threat table: 0.99 MiB;
- XT total approximate runtime storage: 12.13 MiB;
- XT per-ply dual-perspective state: 1,152 bytes;
- Chessformer: 4,188,744 parameters;
- Chessformer int8 target: 3.99 MiB;
- Chessformer root forward: approximately 547.8 MFLOP.

The 547.8 MFLOP result mathematically rules out per-node Chessformer inference.
It supports root-only caching, policy guidance to limited depth, and possible
future early-exit distillation. These are calculated operations, not measured
A100 or CPU latency.

## 19. Required ablations

Every row below is a separate experiment:

| Baseline | Intervention |
|---|---|
| v3 NNUE | new quantized head only |
| previous row | flat threat table |
| previous row | CP factor training |
| previous row | hashed residual |
| previous row | dirty threat update |
| previous row | per-channel int8 threat export |
| Chessformer absolute positions | deterministic geometry templates |
| previous row | dynamic GAB |
| previous row | Elo adapters |
| previous row | human policy training |
| pure alpha-beta | root policy ordering |
| previous row | limited-depth policy ordering |
| policy direct | alpha-beta safety veto |

Do not bundle these into one SPRT. Loss, latency, and Elo attribution otherwise
becomes impossible.

## 20. Promotion criteria

Hydra is eligible for production only if:

1. XT float holdout improves over frozen v3 with bootstrap interval excluding
   zero;
2. quantized XT mean drift <= 3 cp and maximum <= 25 cp on the frozen suite;
3. dirty and full threat accumulators agree exactly;
4. full alpha-beta NPS loss is <= 10%, or Elo compensates in paired SPRT;
5. Chessformer policy is player-disjoint calibrated across Elo/time bands;
6. safety veto has zero forced-mate regressions;
7. engine/GM/uncertain routing always selects alpha-beta;
8. fixed-Elo strength remains stable across clock tiers;
9. each runtime mode passes a separate game gate;
10. model UUID, data manifest, code commit, config, and section hashes are
    packaged together.

The mathematical design is intended to create testable gains, not to rename an
untested network a breakthrough.
