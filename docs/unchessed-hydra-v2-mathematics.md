# Unchessed Hydra Aegis v2: adaptive hypergraph neural search

> **Experimental lineage document.** This is not a product version. The canonical architecture is [Unchessed Apex v1](unchessed-apex-v1.md).

## 1. Why v1 is not enough

Hydra v1 established a sound split between incremental XT-NNUE value and a
root-only Chessformer policy. Its calculated limits identify the next research
problems:

1. direct attacker/target pairs do not explicitly encode pins, skewers,
   batteries, pawn chains, or king-zone topology;
2. a full threat refresh at every evaluated node is still too expensive;
3. the fixed eight-layer Chessformer costs about 548 MFLOP per root call;
4. ordinary softmax confidence cannot distinguish ignorance from strong
   contradictory evidence;
5. human-policy, engine-policy, value, quantization, and calibration gradients
   can destructively interfere in one shared training run; and
6. the two branches align only through losses rather than a common semantic
   concept space.

Aegis v2 addresses all six without making the transformer authoritative over
alpha-beta.

## 2. Architecture summary

```text
                    shared 64-concept bank
                    /                    \
XT-NNUE fast/full hypergraph       elastic Chessformer
    /           \                  exits 2 / 4 / 8
fast value+u   full value          policy + evidential WDL
    \           /                         |
     uncertainty-safe alpha-beta + persona router
```

The central improvement is **adaptive computation with calibrated uncertainty**:
cheap paths handle ordinary positions; full hypergraph/transformer paths are
mandatory for tactical, PV, in-check, or uncertain positions.

Research foundations used as hypotheses, not copied results:

- evidential Dirichlet uncertainty:
  <https://proceedings.neurips.cc/paper_files/paper/2018/file/a981f2b708044d6fb4a71a1463242520-Paper.pdf>
- gradient surgery for multi-task conflict:
  <https://proceedings.neurips.cc/paper/2020/file/3fe78a8acf5fda99de95303940a2420c-Paper.pdf>
- uncertainty-guided multi-depth distillation:
  <https://arxiv.org/abs/2602.16160>
- Chessformer geometry and square-token policy:
  <https://arxiv.org/abs/2605.19091>

## 3. Multi-resolution XT hypergraph

For perspective `p`, Aegis uses three relationship multisets:

```math
R_2(s,p) = direct occupied-target attacks and defenses,
```

```math
R_3(s,p) = x-ray triples (attacker, blocker, behind-target),
```

```math
R_P(s,p) = local pawn/king topology patterns.
```

The full residual is

```math
a_H(s,p) = b_H
 + sum_{r in R_2} E^2_r
 + sum_{r in R_3} E^3_r
 + sum_{r in R_P} E^P_r.
```

Widths are 32, 16, and 16 respectively. Together with the positional width
256, dual-perspective state is:

```math
2 * (256 + 32 + 16 + 16) * sizeof(int16) = 1280 bytes/ply.
```

### 3.1 X-ray hyperedges

A direct pair cannot distinguish a defended blocker from a pinned blocker. An
x-ray feature is

```math
r_3 = (a,b,t,g),
```

where `a`, `b`, and `t` are perspective-relative attacker, first blocker, and
behind-target classes, while `g` is one of eight ray/orientation classes.

```math
idx_3 = (((12a + b)12 + t)8 + g),
```

with

```math
12^3 * 8 = 13824
```

possible rows. A row activates only when the squares are collinear and `b` is
the first occupied square from `a`, with `t` the next occupied square.

Training uses a rank-factorized hyperedge:

```math
u_3 = A_a elementwise B_b elementwise C_t elementwise G_g,
```

```math
e^3 = D_3^T u_3 + H^3_hash(r_3).
```

This directly represents pins, skewers, discovered attacks, batteries, and
king-line shielding.

### 3.2 Pawn/king topology

For every pawn and king zone, encode a mover-normalized local 3-file by 4-rank
window as a 12-bit occupancy pattern for own/enemy pawns plus state flags:

```math
idx_P = hash(window_own, window_enemy, king_zone, passed, backward) mod 4096.
```

A 4,096 by 16 int8 table adds only 64 KiB after export. Unlike pairwise threats,
this gives the evaluator explicit connected, isolated, passed, lever, shelter,
and storm context.

### 3.3 Dirty update closure

For a move with dirty squares `D_0`, define:

```math
D_1 = D_0 union kings union leapersIncident(D_0),
```

```math
D_2 = D_1 union firstTwoOccupiedSquaresOnEveryRay(D_1),
```

```math
D_P = pawnWindowsIntersecting(D_0).
```

Only hyperedges incident to these closures are removed/recreated. If castling,
promotion, en passant, or a ray ambiguity escapes the bounded closure, perform
an exact full refresh. Truncation is forbidden.

## 4. Two-stage XT evaluation

A cheap head uses only positional accumulators and outputs value plus positive
uncertainty scale:

```math
(v_f, log sigma_f^2) = H_f(a_P^stm, a_P^nstm).
```

The full head adds all hypergraph accumulators:

```math
v_F = H_F(a_P^stm, a_P^nstm, a_H^stm, a_H^nstm).
```

Full evaluation is mandatory when:

```text
root OR PV OR in-check OR sigma_f > tau_node.
```

Otherwise the fast head may be used. The implemented pure router currently
encodes this conservative contract.

### 4.1 Heteroscedastic fast-head loss

Train the cheap head to predict the full head with Gaussian negative
log-likelihood:

```math
L_fast = (v_F - v_f)^2 / (2 sigma_f^2)
       + 0.5 log sigma_f^2.
```

The uncertainty is therefore penalized for being both too small on errors and
too large everywhere.

### 4.2 Safe pruning bounds

When alpha-beta considers a pruning condition, use a one-sided conservative
bound rather than raw fast value. For fail-high pruning:

```math
LCB_f = v_f - k(d,node) sigma_f.
```

Prune only if

```math
LCB_f >= beta + margin.
```

For fail-low pruning use

```math
UCB_f = v_f + k(d,node) sigma_f
```

and require `UCB_f <= alpha - margin`. Root, PV, check, and mate-score regions
never use fast uncertainty to skip the full path.

## 5. Elastic Matryoshka Chessformer

One transformer contains nested width/depth subnetworks:

```text
exit A: layer 2, width 128
exit B: layer 4, width 192
exit C: layer 8, width 256
```

The first `w` channels of every projection form the width-`w` subnetwork.
Training uses a sandwich rule on every batch:

1. full layer-8 width-256 path;
2. smallest layer-2 width-128 path;
3. one random intermediate path.

The full output teaches every smaller path.

### 5.1 Exit distillation

For exit `e` and full exit `F`:

```math
L_exit^policy(e) = T^2 KL(stopgrad(pi_F^T) || pi_e^T),
```

```math
L_exit^value(e) = JS(stopgrad(p_F), p_e),
```

```math
L_exit^repr(e) = 1 - cosine(M_e h_e, stopgrad(h_F)).
```

```math
L_exit = sum_e omega_e
  (L_exit^policy + L_exit^value + lambda_r L_exit^repr).
```

Random-depth training prevents shallow exits from becoming untrained
attachments to a deep-only backbone.

## 6. Evidential WDL and calibrated exits

Instead of softmax logits, the value head predicts nonnegative evidence:

```math
e_k = softplus(z_k),
```

```math
alpha_k = e_k + 1,
```

```math
S = sum_{k=1}^3 alpha_k,
```

```math
P_k = alpha_k / S,
```

```math
u = 3 / S.
```

With zero evidence, `u=1`; large consistent evidence lowers uncertainty. This
follows the standard Dirichlet evidential construction, but must be calibrated
on held-out chess data rather than assumed correct.

Exit policy:

```text
layer 2 when u <= 0.08 and policy calibration passes
layer 4 when u <= 0.16
layer 8 otherwise
in-check -> layer 8
alpha-beta backend -> no transformer call
```

The Rust contract now exposes these choices. Thresholds are configuration and
require conformal/holdout calibration.

### 6.1 Evidential loss

For one-hot WDL target `y`:

```math
L_EDL = sum_k [(y_k - alpha_k/S)^2
        + alpha_k(S-alpha_k)/(S^2(S+1))]
        + lambda KL(Dir(alpha_tilde) || Dir(1)).
```

Incorrect evidence is annealed toward the uniform no-evidence prior. Report
accuracy, NLL, Brier score, ECE, uncertainty-error correlation, and
out-of-distribution behavior.

## 7. Legal-only attention policy

Dense policy computes 4,096 source/destination dot products even though a legal
position has at most 218 legal moves. The encoder still outputs all 64 square
vectors, but runtime gathers only legal move pairs:

```math
ell_m = q_from(m)^T k_to(m) / sqrt(d) + b_special(m).
```

For `N_legal` moves, policy dot cost changes from

```math
4096d
```

to

```math
N_legal d.
```

At the legal maximum this is an 18.79x reduction in policy dot products. Move
generation is already required by alpha-beta, so no extra legality network is
needed.

## 8. Shared semantic concept bank

Aegis adds `M=64` concepts of width 32:

```math
C in R^(64 x 32).
```

XT hypergraph features produce concept evidence:

```math
z_N = softmax(C P_N a_H).
```

Chessformer square states produce:

```math
z_C = softmax(C P_C mean_i(y_i)).
```

Rather than matching coordinates directly, align concept distributions with an
entropy-regularized transport cost:

```math
L_OT = min_{Pi >= 0}
       <Pi, D> + epsilon sum_ij Pi_ij(log Pi_ij - 1),
```

subject to

```math
Pi 1 = z_N,
Pi^T 1 = z_C.
```

`D_ij` is learned concept distance with diagonal preference. The transport plan
allows similar but non-identical concepts to align instead of forcing arbitrary
coordinate equality.

Auxiliary concept labels may include pin, fork, x-ray, passed pawn, king
exposure, material phase, and only-move status. They regularize concepts but are
not required at runtime.

## 9. Multi-task gradient conflict control

Human imitation and engine-optimal policy can point shared parameters in
opposite directions. For task gradients `g_i` and `g_j`, conflict exists when

```math
g_i dot g_j < 0.
```

On shared concept/GAB parameters only, project the conflicting component:

```math
g_i' = g_i - (g_i dot g_j / ||g_j||^2) g_j.
```

Private human adapters, guide adapters, and XT tables are not projected. This is
a module-scoped PCGrad rule: it avoids paying multiple backward/gradient copies
for every parameter while protecting the truly shared representation.

Log the cosine matrix among all tasks. PCGrad itself is an ablation; it is not
assumed better until held-out metrics improve.

## 10. Symmetry and transposition consistency

### 10.1 Color/board equivariance

For legal symmetry transform `g` and move transform `g(m)`:

```math
L_sym^value = |v(g(s)) - v(s)|,
```

```math
L_sym^policy = KL(g(pi(.|s)) || pi(.|g(s))).
```

Use color rotation and legal file mirror where castling/en-passant metadata is
transformed exactly.

### 10.2 Transposition consistency

For two move sequences reaching identical full state hash `s_a=s_b`:

```math
L_trans = |v(s_a)-v(s_b)| + KL(pi_a || pi_b).
```

This catches accidental history leakage. If explicit move history is later
added for human modeling, its effect must be isolated from the board-state
value head.

## 11. Counterfactual-regret policy targets

Raw teacher visit counts can overfit one search configuration. For legal move
`m`, compute common-budget alpha-beta regret:

```math
r(m) = max(0, V(best) - V(m)).
```

Teacher policy is

```math
pi_T(m|s) proportional to
exp(-r(m)/tau_r) * (n(m)+n_0)^gamma.
```

`tau_r` controls tolerance for near-equal moves and `gamma` retains search
confidence. Human target can be combined at data level, not by silently mixing
identities:

```math
pi_persona = (1-lambda(E,z)) pi_human
           + lambda(E,z) pi_T.
```

For average-human direct play, `lambda` is low and alpha-beta applies the safety
veto. For guide mode, `lambda` is high. Engine/GM mode bypasses policy selection.

## 12. Risk-aware adaptive computation

Let the expected cost and tactical loss risk of compute path `a` be `C(a)` and
`R(a)`. Select:

```math
a_star = argmin_a C(a) + lambda_risk CVaR_0.1(R(a))
```

subject to hard constraints:

```text
root/PV/check -> full XT residual
engine/GM/uncertain -> alpha-beta
in-check Chessformer -> layer 8
missing model -> alpha-beta
mate region -> full evaluator and full search safeguards
```

The learned/evidential router is subordinate to these constraints.

## 13. Why ordinary KV caching is not used

Language-model KV caching is exact because causal earlier-token hidden states do
not change when a later token is appended. A chess move changes square tokens
inside a bidirectional 64-token encoder; after one attention layer, every token
may change. Reusing old layer K/V tensors would therefore be approximate and
can silently corrupt policy.

Aegis uses only:

- whole-position transposition caching keyed by full Zobrist state, model UUID,
  Elo condition, and persona;
- exact reuse for identical states;
- early exit for cheaper new states.

Any incremental bidirectional approximation must be a separately measured
model, never called an exact cache.

## 14. Updated joint objective

```math
L = 0.22 L_XT-full-WDL
  + 0.08 L_XT-fast-distill
  + 0.12 L_teacher-score
  + 0.18 L_human-policy
  + 0.08 L_teacher-policy
  + 0.08 L_evidential-WDL
  + 0.06 L_value-consistency
  + 0.05 L_concept-transport
  + 0.05 L_quantization
  + 0.04 L_symmetry
  + 0.04 L_calibration.
```

The coefficients sum to one. Exit-distillation losses are included inside
`L_evidential-WDL` and the two policy losses for each sampled subnetwork.

## 15. Calculated budget

From `config/unchessed_hydra_v2.json`:

| Component | Calculated budget |
|---|---:|
| XT full runtime | 12.55 MiB |
| XT dual-perspective state per ply | 1,280 bytes |
| Chessformer parameters | 4,606,296 |
| Chessformer int8 target | 4.39 MiB |
| Legal-only policy dot reduction | 18.79x |
| layer-2 width-128 exit | 44.3 MFLOP |
| layer-4 width-192 exit | 167.7 MFLOP |
| layer-8 width-256 exit | 545.8 MFLOP |

The shallow exit is about 12.3x fewer calculated FLOPs than full depth. These
are operation counts, not measured CPU latency.

## 16. A100 training curriculum

### Phase 1: experts separately

- train fast XT positional value/uncertainty;
- train full direct/x-ray/pawn hypergraph residual;
- train full-depth Chessformer human and guide adapters;
- freeze future/player-disjoint holdouts before tuning.

### Phase 2: once-for-all elastic training

For each Chessformer batch evaluate full, smallest, and one random subnetwork.
Distill full outputs into shallow exits. Sample widths/layers with a sandwich
rule and maintain separate Batch/RMS normalization behavior if required.

### Phase 3: joint concepts and PCGrad

Enable the shared concept bank and transport loss. Measure gradient cosines for
at least one epoch before enabling module-scoped PCGrad. Reject it if either
human or guide validation regresses.

### Phase 4: evidential calibration

Train evidence regularization with annealing. Fit exit thresholds on a separate
calibration split; do not use the test holdout. Validate out-of-distribution
positions, malformed-state rejection upstream, rare promotions, and tactical
only-move sets.

### Phase 5: QAT and export

Materialize int8 relation/hyperedge tables, train quantization distillation,
export `UNCHAES2`, validate scalar Rust, then add SIMD kernels.

## 17. New mandatory ablations

1. direct threats versus direct+x-ray;
2. x-ray versus x-ray+pawn topology;
3. always-full XT versus uncertainty-gated XT;
4. softmax WDL versus evidential WDL;
5. fixed 8-layer versus elastic exits;
6. dense 4,096 policy versus legal-only policy;
7. coordinate alignment versus concept optimal transport;
8. ordinary multi-task gradients versus module-scoped PCGrad;
9. no symmetry loss versus symmetry+transposition consistency;
10. visit-count teacher versus regret-calibrated teacher.

## 18. Promotion gates

Aegis v2 must satisfy all Hydra v1 gates plus:

- x-ray and pawn feature dirty updates exactly match full refresh;
- uncertainty rises monotonically with fast-head absolute error on holdout;
- uncertainty-gated pruning has zero mate/only-move regressions;
- each early exit has separately reported accuracy and calibration;
- threshold calibration and final test use disjoint datasets;
- shallow-exit latency is measured on the deployment CPU;
- legal-only policy exactly matches dense policy logits for legal moves;
- concept bank does not collapse (effective rank and activation entropy gates);
- PCGrad is enabled only if both human and guide objectives benefit;
- no approximate bidirectional cache is presented as exact;
- every compute mode receives a separate paired game gate.

Hydra Aegis v2 is a stronger architecture because it spends computation where
uncertainty and tactics justify it, while preserving hard alpha-beta safety
boundaries.
