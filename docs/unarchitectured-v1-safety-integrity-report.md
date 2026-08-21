# Unarchitectured v1 safety-integrity report

Date: 2026-08-21

## Verdict

| Layer | Result |
|---|---|
| Safety policy parsing | PASS |
| Numerical/loss/gradient controller | PASS after one false-positive fix |
| Validation CUSUM and patience | PASS |
| Atomic heartbeat schema | PASS |
| External watchdog process termination | PASS with real subprocess fault injection |
| Silent-success detection | PASS after fail-closed fix |
| Stale/future heartbeat rejection | PASS |
| Dataset size/balance/dedup checks | PASS |
| Game/player/position split isolation | PASS |
| Chronological split ordering | PASS |
| Provenance file SHA-256 binding | PASS after integrity fix |
| Guide regret-label completeness | PASS after integrity fix |
| Architecture/config/profile consistency | PASS |
| Rust/config/GPU feature schema | PASS |
| Runtime readiness | Correctly **BLOCKED** |
| Real CUDA/NCCL behavior | UNVERIFIED locally |

The safety system behaves fail-closed in every locally executable fault test.
It cannot guarantee against driver, hardware, kernel, or PyTorch faults that
have not been exercised on the selected Verda node.

## Fault injections executed

### Numerical controller

- finite decreasing losses continue;
- NaN loss aborts;
- non-finite gradient aborts;
- pre-clip gradient norm above policy aborts;
- 20x loss-to-EMA spike aborts;
- negative heteroscedastic losses do not false-trigger;
- large negative magnitude spikes still abort;
- validation non-improvement triggers patience stop;
- validation degradation CUSUM triggers early stop.

### Watchdog subprocess integration

Real child process groups were launched for these tests:

1. valid heartbeat followed by clean exit → exit 0, no incident;
2. clean child exit without any heartbeat → fail-closed exit 125 and
   `missing_heartbeat` incident;
3. stale heartbeat plus sleeping child → process-group termination, exit 124,
   GPU diagnostic attempt, and durable incident;
4. malformed or future heartbeat timestamps → treated as stale/invalid;
5. nonzero child exit → `child_failed` incident.

The launcher deletes stale heartbeat/incident files before each phase, so a
previous run cannot satisfy current liveness.

## Integrity bugs found and fixed

### S-01 — Negative-loss false abort

The loss-spike detector previously used `max(epsilon, EMA)`. Some valid
heteroscedastic objectives can be negative, making the allowed magnitude nearly
zero and causing false aborts. It now compares absolute loss against absolute
EMA.

### S-02 — Silent zero-exit bypass

A child could exit with code zero before writing a heartbeat, and the watchdog
would accept success. Clean exit now requires a structurally valid heartbeat;
otherwise it returns 125 and writes an incident.

### S-03 — Wall-clock heartbeat bypass

Heartbeat age relied only on wall time. Future/NTP-skewed timestamps could look
fresh indefinitely. Heartbeats now contain both Unix and monotonic time;
implausible future values fail closed.

### S-04 — Guide records could lack regret labels

The data gate counted `policy_kind=guide` without requiring teacher regrets.
Guide records must now be regret-labelled, and tune/final splits must each meet
a minimum labelled-guide fraction.

### S-05 — Provenance dates were not content-bound

A dated JSON could claim valid chronology without proving which files it
covered. Every split now declares shard basename and SHA-256; the gate streams
and compares actual files before accepting dates.

### S-06 — Date interval validation was incomplete

The gate checked train-end/tune-start and tune-end/final-start but not each
split's start ≤ end. It now validates the complete ordered interval chain.

### S-07 — Canonical architecture depended on experimental student config

Unarchitectured v1 now carries a complete runtime-student contract and canonical
student implementation config. The architecture audit rejects shape, exit,
head, loss, action-vocabulary, or profile drift.

### S-08 — Python package checks were weaker than Rust

Python now enforces the same shape × dtype byte count, scale, flag, alignment,
name, bounds, and checksum constraints as the Rust loader. Large artifact hashes
are streamed to avoid memory amplification.

## Safety ordering before GPU spend

The canonical launcher executes:

1. semantic runtime capability gate;
2. canonical architecture cross-config audit;
3. Rust/config/GPU feature audit;
4. full dataset and provenance gate;
5. record-level ABI validation;
6. player/game leakage audit;
7. Verda hardware preflight;
8. reduced model self-check;
9. per-rank optimizer-inclusive VRAM probe;
10. externally supervised DDP training;
11. tuning-only calibration;
12. untouched final holdout;
13. export, strict package inspection, and tensor drift validation.

Any failed stage prevents later stages.

## Verification totals

- Python: **107 passed, 0 failed, 1 dependency-gated skip**
- Rust normal workspace: **123 passed, 0 failed, 3 ignored**
- Rust deep/ignored tests: **3 passed separately**
- Clippy `-D warnings`: PASS
- Architecture audit: PASS
- Feature schema audit: PASS
- Package corruption tests: PASS
- Shell syntax: PASS
- JSON parse: PASS

## Remaining limits

1. No local PyTorch/NumPy/CUDA/NCCL runtime was available.
2. Multi-rank abort propagation is source-tested but not exercised on real
   NCCL hardware.
3. No forced GPU OOM was injected on Verda.
4. No filesystem-full, NVMe disconnect, host reboot, or spot eviction test was
   run.
5. Full scalar and quantized neural forward remain absent, so runtime readiness
   correctly remains false.
6. A compromised training process could intentionally forge heartbeats; the
   watchdog protects against hangs/crashes, not a hostile child.

## Recommended first Verda safety drill

Before model training, use a cheap one-GPU instance to deliberately test:

- stale heartbeat termination;
- child SIGKILL and nonzero incident capture;
- controlled CUDA OOM;
- disk-full checkpoint failure;
- resume from latest atomic checkpoint;
- four epochs of intentionally worsening validation; and
- NCCL rank failure on a two-GPU node.

Do not start a full 8-GPU Oracle run until those drills produce expected
incidents and leave the best checkpoint intact.
