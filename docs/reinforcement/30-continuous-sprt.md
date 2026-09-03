# 30 — Continuous SPRT infrastructure (self-hosted, no campaign)

**Investigation ID:** `continuous-sprt-infrastructure`
**Tier:** 1 (research/design only)
**Status:** Complete; no implementation, match campaign, cloud job, default change, or engine-strength claim was made.

## Executive conclusion

The repository has the statistical post-processing needed to interpret paired match results, but every historical SPRT has been manually orchestrated. A small, self-hosted, continuous test service would make future validation cheaper and more reproducible without adopting the operational and security burden of the full Stockfish Fishtest deployment.

**Recommendation: pursue infrastructure design, defer implementation until explicitly approved, and do not start a match campaign as part of this item.** The appropriate first version is a single-controller, pull-worker queue with an append-only run manifest, immutable engine artifacts, deterministic opening-pair assignment, PGN/log/result upload, and a server-side pentanomial/GSPRT decision step. The controller should use an established game manager (preferably Fastchess, or Cutechess CLI if that is the validated local choice), while retaining the repository's `tools/pentanomial_sprt.py` as an independently testable reporting/checking path. It should initially run one trusted worker on one host; additional workers can be added only after provenance, duplicate-task handling, and worker-quality checks are demonstrated.

This is infrastructure, not evidence that any candidate is stronger. A run must still be separately proposed, reviewed, and approved under the master brief's real paired-game SPRT rule.

## What was inspected

I read the master brief and existing reinforcement documents `00`–`12`, plus the later numbered investigations present in this checkout. The brief explicitly identifies continuous SPRT as Tier 1 item 22 and says that historical SPRTs were manually orchestrated. It also requires paired real games before any search/default conclusion, preservation of defaults, and an explicit distinction between verified and assumed facts.

Repository inspection covered the Rust workspace (`unchessed-core`, `unchessed-adapter`, `unchessed-reviewer`, and `unchessed-datagen`), the UCI entry points, `tools/pentanomial_sprt.py`, the development requirements, and repository references to cutechess/Fastchess/SPRT. The adapter and reviewer are ordinary UCI binaries launched through `unchessed_core::uci::run`; the workspace has no checked-in Fishtest server, worker, Docker deployment, CI match workflow, opening-book manifest, or installed `cutechess-cli`/`fastchess` binary in this environment. `cargo` is available, but no match manager was available on `PATH` during inspection. The working tree already contained unrelated modifications and untracked reinforcement documents; this report did not alter them.

`tools/pentanomial_sprt.py` accepts either five pentanomial pair counts or a PGN plus engine name. It derives paired outcomes from a Cutechess `-repeat -games 2` style PGN, reports trinomial and pentanomial Elo/intervals, normalized Elo, LOS, and an SPRT LLR with configurable `elo0`, `elo1`, `alpha`, and `beta`. This is useful as a review/audit tool, but it is not a queue, match runner, artifact builder, duplicate detector, or durable state machine. Its PGN parser also assumes the two games of a pair are adjacent and drops an unpaired trailing game; a service must enforce and record that pairing contract rather than silently relying on arbitrary PGN ordering.

## External research and what it means here

The official [Stockfish Fishtest overview][1] describes Fishtest as a distributed task queue: developers submit tests, workers download the book/game manager/engine sources, play batches, and upload results; the server queues work, computes statistics, updates ongoing results, and stops statistically significant tests. The [official Fishtest repository README][2] confirms the split into `server/` and `worker/` components and that workers compile or run the two engine versions and upload game results.

The official [Fishtest mathematics documentation][3] states that production Fishtest uses pentanomial pair outcomes and GSPRT rather than a basic trinomial-only workflow. It also documents worker-quality checking and explains that pentanomial analysis can save testing resources relative to trinomial analysis. Those are authoritative descriptions of Stockfish's system; they are not evidence that this repository has implemented GSPRT or that a candidate has any Elo gain. A small Unchessed service should begin with a clearly specified fixed SPRT model or an audited existing implementation, and should not label a calculation “Fishtest-equivalent” until its statistical model, stopping boundaries, and error properties are tested.

[Fastchess's upstream README][4] documents concurrent engine matches, paired/repeated openings, time controls, PGN output, UCI compliance checking, and a command-line interface. The [Cutechess CLI manual][5] documents concurrency, repeated openings, PGN input/output, explicit random seed, recovery, node/depth/time controls, and engine options. Either manager can therefore be the execution boundary. Fastchess is the more attractive new dependency for a Linux-first service because its current upstream project is actively maintained and MIT licensed; Cutechess remains a defensible fallback where existing PGN conventions or a validated local binary require it. The choice must be frozen in a lockfile/container digest and recorded in each run manifest.

## Proposed safe architecture

### Components

| Component | Responsibility | Deliberate boundary |
|---|---|---|
| **Controller/API** | Accept a reviewed run specification, validate it, enqueue immutable tasks, aggregate results, and expose run status | No arbitrary shell commands from clients; only allowlisted manager/build commands and typed fields |
| **SQLite state store** | Runs, tasks, workers, leases, result hashes, decisions, and audit events | Single writer/controller first; WAL backups and migrations; no distributed database initially |
| **Artifact store** | Engine binaries, source revision, opening book, manager binary, logs, PGNs, JSON summaries | Content-addressed files keyed by SHA-256; retention policy and read-only completed runs |
| **Worker** | Pull a leased task, verify manifest hashes, run the exact manager command in a sandbox, upload result bundle | Worker never chooses hypotheses or edits results; lease expiry permits retry with an idempotency key |
| **Statistical reducer** | Validate pair counts and metadata, recompute statistics, apply stopping policy, mark accepted/rejected/inconclusive | Server-side recomputation is authoritative; worker summaries are untrusted hints |
| **Review UI/CLI** | Display manifest, progress, WDL/pentanomial counts, LLR, boundaries, failures, and artifacts | No “promote default” button; promotion remains a separate human-reviewed change |

The minimum deployment is one controller plus one worker on a private Linux host, with SSH or a private network and no public worker endpoint. A second host can run a worker through outbound HTTPS polling. Workers should run as an unprivileged user in a dedicated directory, with CPU/memory/process/time limits, no repository write access, and no credentials other than a revocable worker token. A container is useful for dependency isolation, but it is not a security boundary by itself; the host should still be patched and firewall-restricted.

### Run lifecycle

1. A maintainer creates a run specification naming the baseline and candidate source revisions or binary digests, compiler/toolchain image, target binary (`unchessed-adapter` or `unchessed-reviewer`), UCI options, threads/hash, time control, book digest, pair count limit, `elo0`/`elo1`, alpha/beta, adjudication policy, and manager version.
2. The controller resolves and hashes every input, builds both engines in a clean reproducible environment, runs UCI compliance and smoke checks, and freezes a signed/hashed manifest. A run cannot enter `queued` unless the two binaries, manager, book, toolchain, and configuration are all present.
3. The scheduler creates tasks containing a disjoint range of opening-pair IDs and a deterministic seed. Each task has an idempotency key derived from run ID, opening range, side assignment, and engine digests. Workers pull, lease, execute, and upload a result bundle.
4. The worker records stdout/stderr, manager log, PGN, termination reason, game count, pair IDs, engine-side assignment, CPU identity class, and SHA-256 hashes. It must fail closed on crash, timeout, illegal move, protocol error, missing pair, duplicate pair, or manifest mismatch; partial output is quarantined rather than counted.
5. The reducer accepts only complete, non-duplicated, manifest-matching pairs. It recomputes W/D/L and the five pair bins from PGN or a structured result record, compares the worker summary, updates the LLR and confidence information, and stops dispatch when a predeclared boundary is crossed. It must retain all raw evidence and state whether the stopping result is upper, lower, or inconclusive at the configured limit.
6. Completion produces a signed immutable report containing the exact command, hashes, counts, statistical parameters, decision, failures, and links to PGN/logs. A human may then decide whether a separate candidate review or Tier 3 campaign is warranted; infrastructure completion never changes a UCI default.

### Reproducibility and pairing requirements

Use a fixed, versioned opening corpus with stable IDs and cryptographic digest. Prefer two games per opening with colors reversed, rather than independently random openings. Record the seed even when the book is deterministic. Do not mix time controls, hardware classes, thread counts, hash sizes, adjudication settings, engine options, or books inside one statistical run. If hardware heterogeneity is later allowed, stratify and report it, and do not silently pool incompatible conditions.

The engine build must identify repository commit, dirty-tree status, Rust/Cargo version, target triple, compiler flags, binary digest, and relevant model-file digest. The model file must be copied into the artifact store and opened read-only by the worker. The manager version and exact command line must be recorded. `ucinewgame` behavior, process restart policy, `Threads`, `Hash`, and any `Move Overhead`/analysis options must be explicit. A preflight should run both binaries through UCI initialization, `isready`, a start-position search, and a small `go nodes` or fixed-time check before games.

A task lease must be idempotent: retrying a lost worker cannot double-count a pair. The controller should accept one canonical result for each pair key and preserve later duplicates as rejected evidence. Completion should be monotonic (`queued → leased → uploaded → validated → counted`) with explicit `failed`, `expired`, and `quarantined` states. Backups must include the SQLite database and artifact manifests, and a restore drill should be part of acceptance testing.

## Statistical policy

The repository's existing script supports the conventional configuration surface, but the service should make the statistical contract explicit in every run. A recommended initial policy is a paired game design, five pentanomial bins, predeclared `elo0` and `elo1`, alpha and beta, a maximum pair/game limit, and a minimum preflight count. The controller should calculate the LLR centrally and stop only at the configured upper/lower boundary or maximum. It should report the observed estimate and interval separately from the sequential decision.

Do not repeatedly restart runs after peeking, change bounds after results arrive, or merge failed/incompatible games into a later run. A continuous queue may contain many independent candidate runs, but it must not turn one candidate into an unregistered sequence of optional-stopping experiments. For parameter tuning, create an explicit family-level policy and correction/confirmation plan; a stream of many “try another coordinate” jobs is not automatically valid evidence.

The initial reducer can call or reimplement the tested mathematics in `tools/pentanomial_sprt.py`, but the implementation should be cross-checked against hand-calculated fixtures and known simulation cases before production use. Fishtest's GSPRT and normalized-Elo methods are a possible later upgrade, not a reason to silently substitute a different test now. The report must say “SPRT with declared parameters” unless GSPRT has actually been implemented and verified.

Worker-quality controls should begin modestly: reject malformed bundles, compare independent recomputation, monitor crashes/timeouts and result-rate anomalies, and quarantine a worker that repeatedly disagrees with server counts. If multiple untrusted public workers are ever accepted, add signed worker registration, per-worker contribution logs, duplicate/overlap checks, cross-worker consistency tests, and a documented statistical outlier policy. The full Fishtest chi-squared worker-quality machinery should not be copied piecemeal without tests.

## Cost and operational sizing

These are planning estimates, not invoices. A single existing Linux workstation costs **$0 incremental infrastructure** and is the recommended first deployment; it consumes local CPU and disk. A small always-on controller VM with 1–2 vCPU, 2–4 GB RAM, 20–40 GB SSD, private networking, and backups is typically in the **low single-digit to roughly $15/month** range depending on provider and region. A separate 4–8 vCPU worker is commonly **roughly $10–$40/month** on an inexpensive VM; burst compute can be cheaper or much more expensive depending on CPU generation and sustained-use pricing. Object storage for compressed PGN/log artifacts is usually negligible at this scale (often **under $1–$5/month** for tens of GB), excluding egress and backup fees.

The major cost is games, not the queue. At 1,000 nodes per move and a fast time control, throughput can vary by orders of magnitude with CPU, position complexity, manager overhead, and draw/adjudication settings. Therefore the service must measure games/hour in a calibration run and estimate campaign cost from that measurement; it must not promise an Elo result or duration from a generic VPS specification. No such throughput measurement was run here because no match manager was installed and this item explicitly separates infrastructure design from a campaign.

A practical staged budget is: **Stage 0, $0**, local controller schema, fixtures, dry-run scheduler, and mocked worker; **Stage 1, $0–$15/month**, one private controller and one trusted worker after explicit approval; **Stage 2, variable**, additional workers only if the owner accepts the CPU/cloud budget and the first deployment passes integrity and restore tests. No cloud spend is authorized by this report.

## Implementation boundary and acceptance gates (future work only)

Implementation should be a separate acknowledged plan. It should not include a candidate patch, engine default change, or automatic campaign. Before the first real match, require: deterministic manifest creation; clean-build and binary-hash verification; UCI smoke/compliance checks; opening-pair and side-swap fixture tests; lease/retry/duplicate tests; PGN/result recomputation fixtures; SPRT boundary fixtures; crash/timeout quarantine; database backup/restore; and a dry run that produces no counted games.

Before accepting a real worker, require an isolated worker test, token revocation test, no-repository-write test, resource limits, and an end-to-end one-task run whose result is manually inspected. Before accepting multiple workers, require overlapping-task rejection, cross-worker result consistency, and a documented quality policy. Before any candidate campaign, obtain the explicit approval required for Tier 3, publish the fixed run specification, and preserve the repository defaults. A campaign's final conclusion must be based on the exact paired-game design and not on a simulation, fixed-position test, or worker telemetry alone.

## Verified versus assumed

| Item | Status |
|---|---|
| Tier 1 item 22 asks for lightweight self-hosted continuous SPRT research | **Verified** in the master brief. |
| Repository contains `tools/pentanomial_sprt.py` with PGN/pentanomial/SPRT reporting | **Verified** by source inspection. |
| UCI binaries are `unchessed-adapter` and `unchessed-reviewer` backed by `unchessed_core::uci::run` | **Verified** by source inspection. |
| Full Fishtest server/worker is present in this repository | **Not verified; repository search found none.** |
| Cutechess CLI or Fastchess is installed in this environment | **Not verified; neither was found on `PATH`.** |
| Official Fishtest uses a distributed queue, workers, pentanomial model, GSPRT, and worker-quality controls | **Verified from official documentation** [1][3]. |
| Fastchess supports current concurrent match orchestration and UCI compliance checking | **Verified from upstream README** [4]; not executed here. |
| Exact games/hour, monthly cloud price, or campaign duration for Unchessed | **Not measured/assumed; must be calibrated on the target host.** |
| Any candidate is stronger, or any campaign should start | **No; not tested and not recommended by this report.** |

## Final disposition

**Pursue as default-preserving infrastructure design; defer implementation pending acknowledgement; do not launch a match campaign.** Start with a private one-controller/one-worker service and immutable manifests rather than a full Fishtest fork. Reuse the repository's pentanomial checker as a verification oracle, but keep the server-side reducer authoritative and explicitly distinguish ordinary SPRT from Fishtest's GSPRT. The service's success criterion is trustworthy, repeatable evidence production—not faster permission to change defaults.

## References

[1]: https://official-stockfish.github.io/docs/fishtest-wiki/Home.html "Official Stockfish Fishtest overview"
[2]: https://github.com/official-stockfish/fishtest "Official Stockfish Fishtest repository"
[3]: https://official-stockfish.github.io/docs/fishtest-wiki/Fishtest-Mathematics.html "Official Stockfish Fishtest statistical methods"
[4]: https://github.com/Disservin/fastchess "Fastchess upstream README"
[5]: https://manpages.ubuntu.com/manpages/xenial/man6/cutechess-cli.6.html "Cutechess CLI manual"

**Report file:** `/home/ubuntu/Unchessed-UCI-Engine/docs/reinforcement/30-continuous-sprt.md`

**Work performed:** documentation/research only. No infrastructure was deployed, no match manager was run, no game result was counted, no cloud resource was purchased, and no engine behavior/default was changed.
