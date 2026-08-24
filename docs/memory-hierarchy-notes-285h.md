# Memory hierarchy on a Core Ultra 9 285H — what "fast memory" actually means

Follow-up to `docs/tuning-core-ultra-9-285h-and-low-end.md`, correcting a
natural misreading of the "24 MB L3" figure and covering what 32 GB of
LPDDR5X-7467 does and does not buy.

## No — you have ~50 MB of on-chip memory, not 24 MB

24 MB is only the **shared last level**. The full on-chip SRAM on Arrow Lake-H:

| Level | Size | Scope |
|---|---:|---|
| L0d (per P-core) | 48 KB | private — Intel renamed the old L1d |
| **L1 (per P-core)** | **192 KB** | private — a *new* mid-level cache, ~9-cycle |
| L2, P-cores | 6 × 3 MB = **18 MB** | **private per core** |
| L2, E-clusters | 2 × 4 MB = **8 MB** | shared within each 4-core cluster |
| **L2 total** | **26 MB** | |
| L3 (Smart Cache) | **24 MB** | shared by all cores |
| **Total on-chip** | **~50 MB** | |

Two things worth internalizing:

**The 18 MB of P-core L2 is private, not shared.** Each P-core gets its own
3 MB. At `Threads = 16` that is 3 MB *per thread* for per-thread hot state —
the accumulator stack, killers, history tables — not 3 MB divided sixteen ways.
For a Lazy SMP engine, where each thread has a largely independent working set,
this is a genuinely good architecture.

**Lion Cove added an L1 that didn't exist before.** The old L1d was renamed L0d
(48 KB), and a new 192 KB L1 sits between it and L2 at ~9 cycles. So the ladder
is deeper than on older Intel parts, and mid-sized working sets degrade more
gracefully.

## Measured latency ladder (why this matters at all)

Pointer-chase (random cyclic permutation, defeats prefetchers), measured on the
sandbox Xeon. **The absolute numbers are that host's; the *shape* is universal:**

| Working set | ns/access |
|---:|---:|
| 32 KB | 1.6 |
| 256 KB | 4.5 |
| 1 MB | 6.6 |
| 4 MB | 44.3 |
| 16 MB | 145.8 |
| 64 MB | 161.1 |
| 256 MB | 183.2 |

**~115x** from L1 to RAM. On your 285H the cliff arrives *earlier* than on this
Xeon (24 MB L3 vs 54 MB), with an extra step at ~3 MB where private L2 ends.

## The number that should decide your roadmap

At `Threads = 16`, sharing a 24 MB L3:

| | f32 net (today) | int16 net (proposed) |
|---|---:|---:|
| NNUE weights (shared once, read-only) | 23.1 MB | 11.5 MB |
| **L3 left for everything else** | **0.9 MB** | **12.5 MB** |

**The current f32 network leaves ~0.9 MB of L3** for the entire transposition
table working set *and* 16 threads' data. That is effectively nothing — the TT
is being served from RAM at ~150 ns a probe.

int16 quantization leaves **12.5 MB**, which is a usable TT working set.

This reframes the priority. In
`docs/performance-ceiling-and-gpu-viability.md` I recommended int16 mainly for
arithmetic throughput (~5x on the output layer). On *your* chip the **cache
argument is the stronger one**: it is the difference between a TT that lives in
L3 and one that lives in RAM.

One clarification, since it cuts the other way: the NNUE weights are read-only
and **shared by all threads** — loading them once does not multiply by 16. It is
the TT and per-thread state that contend for what's left.

## Your 32 GB @ 7467 MT/s

**Bandwidth is genuinely excellent** — roughly **119 GB/s** on a 128-bit
LPDDR5X bus, about 2–3x a typical dual-channel desktop DDR5 setup.

**But chess search is latency-bound, not bandwidth-bound**, so this helps less
than it looks:

- A TT probe is a **dependent random read** of 8–16 bytes. You cannot start the
  next node until it returns. That is pure latency.
- Bandwidth wins when you stream large contiguous blocks. A pointer chase never
  saturates the bus — it is idle, waiting.
- LPDDR5X also tends to have *slightly worse* absolute latency than desktop DDR5
  despite far better bandwidth.

So: **32 GB means capacity is never your constraint.** It does *not* mean a huge
`Hash` is free. Any TT larger than L3 is served at ~150 ns+ per probe no matter
how fast the RAM is rated.

### Practical Hash guidance

| Setting | When |
|---|---|
| **128–256 MB** | Blitz/rapid and normal play. Hot entries stay in L3; the rest spill to (fast) RAM. |
| 512 MB – 1 GB | Long analysis, where the higher **hit rate** outweighs slower probes. |
| 2048 MB | Rarely worth it at fast time controls — you pay ~150 ns per probe to avoid re-searches you may not have needed. |

You have the RAM for any of these. The question is never capacity; it is probe
latency versus hit rate, and that trade depends on time control.

Earlier measurement backing this (from
`docs/tuning-core-ultra-9-285h-and-low-end.md`): probe cost was 2.5 ns at
1–16 MB, 7.2 ns at 64 MB, 14.9 ns at 256 MB on the sandbox host — i.e. bigger is
measurably slower per probe, and only pays if it converts into enough extra hits.

## What this changes

Nothing in the recommended order changes, but the *reason* for item 4 gets
stronger:

1. `Threads` auto-detect — done (`9192b03`).
2. **Compile and test the round-1 SIMD work** — still never compiled here.
3. Dirty-plane `apply_diff` — ~77 ns/move, safe refactor.
4. **int16 retrain** — now justified by cache residency (0.9 MB → 12.5 MB of
   free L3), not just by arithmetic throughput.
5. Search-side items once eval stops dominating.

## Caveats

- The cache-size figures are from vendor/architecture sources for Arrow Lake-H,
  not read off your machine. Worth confirming locally with `wmic cpu get L2CacheSize,L3CacheSize`
  (Windows), `lscpu` (Linux), or CPU-Z, since OEM configurations vary.
- The latency ladder was measured on the sandbox Xeon. The shape transfers; the
  absolute values do not.
- The 119 GB/s figure is theoretical peak for a 128-bit LPDDR5X-7467 bus.
  If your machine is configured with a 64-bit bus instead, halve it. Real
  achievable bandwidth is typically 70–85% of peak.
