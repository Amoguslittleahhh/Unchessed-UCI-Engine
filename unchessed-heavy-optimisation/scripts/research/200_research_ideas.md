# Unchessed AI — 200 research ideas

A broad brainstorm across everything touching the project, not just the
narrower 84+11-item backlog in `remaining_research_topics.md` (which stays
the authoritative source for "why it matters / how to test" detail on the
core NNUE/search/persona items). This list trades depth for breadth —
one line each — and reaches further into adjacent territory: tooling,
deployment, community, benchmarking, licensing, and speculative directions
that haven't come up before. Numbering restarts at 1 and is independent of
the other backlog file's numbering.

## NNUE / eval architecture (1-20)

1. Ablate mirroring vs own-king vs factorization individually (also in the other backlog, restated here for completeness)
2. Sweep king-bucket count (16/32/64) at fixed accumulator width
3. Sweep accumulator width (256/512/768/1024) at fixed bucket count
4. Try a dual small+big net setup now that quantization is on the roadmap
5. Implement quantization-aware training (int16 first, then int8)
6. **Implement basic incremental accumulator updates — confirmed missing, top priority**
7. Sweep the WDL loss exponent (1.5-3.5)
8. Try HalfKP as a simpler baseline to sanity-check HalfKAv2_hm's gains
9. Try a two-layer output head instead of plain SCReLU
10. Multi-perspective (STM + non-STM) simultaneous training
11. Add a piece-value regularization term to the loss
12. Try phase-interpolated output-bucket weighting (cheaper than a dual net)
13. Confirm no further symmetry exists beyond horizontal mirroring
14. Try label smoothing on the WDL targets
15. Benchmark eval throughput (Mnps) before/after each architecture change, tracked over time
16. Investigate whether SCReLU's clipping range is well-calibrated for this net's output scale
17. Try training with mixed precision (bf16) to speed up experimentation iteration
18. Explore whether a learned per-bucket bias term helps king-bucket boundary positions
19. Test whether feature factorization helps independently of mirroring (isolate from #1)
20. Investigate whether adding a "material phase" scalar as an explicit input feature helps

## Training data / labeling (21-35)

21. Audit self-play opening diversity/entropy against a reference distribution
22. Deduplicate near-identical positions across shards
23. Blend human Lichess positions into NNUE training, not just self-play
24. Build a pipeline to reuse real logged UCI games as training data once volume exists
25. Check W/D/L ratio skew in the training set vs. target distribution
26. Establish a frozen, never-retrained holdout set for cross-generation validation
27. Use Syzygy tablebases to correct endgame labels
28. Try curriculum learning (easy positions first)
29. Mine adversarial positions (largest disagreement between NNUE versions)
30. Quantify HCE labeler noise vs. a deeper/stronger reference evaluator
31. Bootstrap NNUE training labels from NNUE itself instead of HCE (with a frozen-holdout collapse check)
32. Try oversampling tactically sharp positions specifically
33. Try oversampling positions with unusual material imbalances
34. Investigate whether shard order (curriculum ordering across files, not just within) matters
35. Build a data-quality dashboard (label consistency, position diversity, phase balance) as a standing tool

## Search (36-55)

36. Implement aspiration windows
37. Implement razoring
38. Implement futility pruning
39. Implement ProbCut
40. Implement singular extensions
41. Implement multi-cut pruning
42. Implement internal iterative deepening/reduction
43. Implement late move pruning (distinct from LMR)
44. Implement correction history
45. Implement continuation history
46. Implement Lazy SMP multi-threading
47. Implement NNUE-eval-volatility-driven time allocation
48. Retune LMR reduction tables specifically for NNUE's cost profile (vs when they were likely tuned around HCE)
49. Retune null-move reduction (R value) for NNUE
50. Investigate move-count-based pruning threshold tuning
51. Try a tapered/interpolated eval-scaling approach across game phases in search constants
52. Investigate whether TT replacement strategy is well-tuned (always-replace vs depth-preferred)
53. Try a second, smaller TT dedicated to pawn structure/king safety caching
54. Benchmark search node efficiency (nodes/Elo) before/after each new heuristic, not just raw Elo
55. Investigate whether quiescence search depth/extension rules need retuning post-NNUE

## Movegen / core engine (56-65)

56. Expand the perft test suite beyond startpos/Kiwipete (the well-known supplementary positions)
57. Audit whether magic bitboards are used for sliding-piece attacks
58. Profile movegen's share of total per-node time (decide if SIMD movegen work is even worth it)
59. Audit Zobrist hashing collision rate at realistic Hash sizes
60. Build targeted tests for draw-detection edge cases (repetition across null-move, 50-move + TT interaction)
61. Verify null-move pruning correctly disables in zugzwang/sparse-endgame positions
62. Scope (but don't necessarily build) Chess960/Fischer Random support
63. Investigate whether check-evasion movegen has any measurable slow path worth optimizing
64. Audit castling-rights bookkeeping for edge cases (rook captured before moving, etc.)
65. Stress-test underpromotion handling specifically (perft + eval consistency)

## Opening book (66-75)

66. Build book-learning from own game results (reweight lines by observed win rate)
67. Expand the troll-tier line set
68. Weight book move selection by active persona (sharper lines for CLINCH)
69. Auto-generate a larger book from a masters PGN corpus
70. Tune BookDepth per persona
71. Audit Polyglot/embedded book merge and fallback logic at boundary cases
72. Audit ECO name/code accuracy across all embedded lines
73. Investigate book usage rate calibration against real per-Elo book-exit statistics
74. Explore adding a "surprise weapon" mode that intentionally avoids the most-played line in a category
75. Investigate whether book move selection should factor in opponent's own book depth (inferred from prior games)

## Adapter / persona system (76-95)

76. Speed up opponent-Elo convergence via better priors
77. Build policy-net-based engine-tell detection (KL divergence signal)
78. SPRT-sweep persona hysteresis thresholds
79. Sweep the Contempt default value
80. Sweep the troll bail-out eval-gate threshold
81. Validate the CLINCH "narrow-safe-path" metric against real outcomes
82. Tune DEFEND's resignation/save-rate behavior
83. Refine sandbagger-detection uncertainty-widening rate
84. Build structured persona-transition logging/analysis tooling
85. Audit UCI_Opponent engine-name recognition completeness
86. Explore smoother (non-discrete) blending near persona transition boundaries
87. Investigate whether a fifth persona (e.g. "TEACH" — deliberately instructive play) is worth adding
88. Explore adjusting persona thresholds based on time control (blitz vs classical behavior differences)
89. Investigate opponent rating drift mid-game (does a human's real strength change under time pressure, and should the model track that?)
90. Build a persona A/B testing harness for comparing threshold configurations
91. Explore whether MATCH mode should factor in opponent's declared UCI_Opponent title even when unverified
92. Investigate CLINCH mode's interaction with 3-fold repetition avoidance specifically
93. Explore a "coach mode" where the engine narrates suggested improvements live, not just post-game
94. Investigate whether persona switching frequency itself is a detectable tell (ties to anti-fingerprinting)
95. Explore weighting persona transitions by game importance/context if that metadata is ever available

## Policy net (96-110)

96. Add time-pressure (clock fraction) as a policy net input feature
97. Expand to finer/more rating buckets
98. Try non-linear blending between adjacent buckets
99. Try a conv/resnet policy architecture
100. Investigate the <1300 en-passant accuracy gap against real ground-truth decline rates
101. Test whether more training data (beyond 19.9M positions) still improves accuracy
102. Calibrate policy net separately for bullet/blitz/classical game speeds
103. Explore a joint policy+value net
104. Test policy net consistency specifically in check-evasion-heavy positions
105. Calibrate blunder-rate output against real per-rating Lichess statistics before building a calibration loss
106. Explore per-player style transfer (LoRA-style adapters per famous player)
107. Investigate whether policy net accuracy correlates with game phase (opening vs middlegame vs endgame)
108. Explore adding a "confidence" output alongside move probability (how sure is the net about this rating band)
109. Test whether policy net predictions degrade for unusual/rare openings outside the training distribution
110. Explore transformer-based policy architectures (Chessformer-style) as a direct upgrade path

## Reviewer / analysis tool (111-125)

111. Define the move classification scheme (brilliant/good/inaccuracy/mistake/blunder)
112. Define the accuracy % formula (win-probability-based)
113. Build critical-moment detection
114. Design the PGN batch review CLI
115. Add opening-phase-aware classification leniency
116. Add best-alternative-move display
117. Build game-level summary stats (accuracy trend, phase breakdown)
118. Cross-validate Reviewer output against the policy net's blunder-rate calibration
119. Build puzzle-rush/tactics-trainer mode reusing the Reviewer's classification logic
120. Explore exporting Reviewer output as an annotated PGN (NAG codes + comments) for use in other tools
121. Build a "critical mistakes only" summary mode for quick post-game review
122. Explore game-phase-specific accuracy weighting (endgame precision matters differently than opening theory)
123. Add support for reviewing a whole tournament/match series with aggregate stats
124. Explore visualizing accuracy over time as a simple text-based sparkline in CLI output
125. Investigate whether Reviewer output should flag "book deviations" specifically as their own category

## Testing / regression discipline (126-140)

126. Build a fuzzing harness for the UCI protocol parser
127. Combine perft, feature-encoder consistency, and UCI fuzzing into one harness
128. Build a dedicated Hash=1 / extreme-TT-pressure stress test
129. Build a multi-threaded search race-condition audit (once Lazy SMP lands)
130. Wire perft regression into CI as an automatic gate, not a manual step
131. Measure the pre-SPRT smoke test's actual detection floor empirically (replay against known regressions)
132. Build a fixed labeled-position eval regression test (flag if evals drift beyond tolerance)
133. Build an automated "new eval vs last-promoted eval" comparison on a fixed position set
134. Investigate whether a lightweight static-analysis pass on NNUE weight files could catch obviously malformed exports before deployment
135. Build a long-running (24h+) stability/memory-leak test as a standing CI job, not ad hoc
136. Investigate property-based testing (proptest-style) for movegen/search invariants in Rust
137. Build a benchmark suite tracking nodes/sec and Elo-per-node over time across versions
138. Investigate whether differential fuzzing (comparing against another open-source engine's movegen) would catch anything perft misses
139. Build a "does this compile and pass the smoke test" pre-commit hook
140. Explore snapshot-testing UCI output for known positions to catch protocol regressions

## Calibration / deployment (141-150)

141. Empirically calibrate UCI_Elo against a rating ladder of known-strength opponents
142. Audit UCI_LimitStrength consistency across all four personas
143. Check contempt/draw-aversion behavior under UCI_LimitStrength specifically
144. Calibrate book usage rate against real per-Elo statistics
145. Audit UCI option discoverability across multiple GUIs, not just En Croissant
146. Investigate packaging a signed/notarized build to avoid Windows Smart App Control friction
147. Explore cross-platform builds (Linux, macOS) and what CI matrix that requires
148. Investigate a minimal WASM build for browser-based play/demo purposes
149. Explore Docker packaging for reproducible CI/testing environments
150. Investigate whether a lightweight installer/updater is worth building for non-technical users

## Longer-horizon / exploratory, beyond the 14 already answered (151-170)

151. Investigate a genuinely different search paradigm (MCTS hybrid) as a long-shot alternative to pure alpha-beta
152. Explore whether a small value network could complement NNUE for search-time position complexity estimation
153. Investigate distillation from a much larger offline-trained model into the deployed small NNUE (the transformer-oracle idea generalized)
    - **[PARTIALLY ANSWERED 2026-08-28]** the oracle's *training-method* sub-question (DiffusionBlocks block-wise training for the memory wall) is answered in `docs/research-notes-diffusionblocks-2506.14202.md`: defer/drop at 9M–270M scales — standard backprop fits one 80 GB card, and the method has no published regression results. The oracle itself (train → label → retrain NNUE) remains open, owner-gated.
154. Explore adding Chess960 support properly scoped as a real project, not just a maybe
155. Explore endgame tablebase probing during search (not just label correction)
156. Explore a hybrid rule-based endgame module for 8+-piece positions tablebases don't cover
157. Explore neural-network-based move ordering
158. Explore neural-network-based pruning with a verify-and-prune safety net
159. Explore opponent-style-conditioned evaluation (not just move selection)
160. Explore a long-term self-play league with non-transitivity tracking
161. Explore an explainability layer (rule-based + optional LLM hybrid)
162. Explore cross-player style transfer via LoRA-style policy adapters
163. Explore an anti-fingerprinting audit and mitigation pass
164. Explore pondering support
165. Explore MultiPV-weighted move selection inside the adapter
166. Explore a puzzle-rush/tactics-trainer mode
167. Explore game-phase-aware accumulator refresh (once incremental updates exist as a baseline)
168. Investigate whether a small on-device "opponent modeling" transformer (à la ChessMimic/1e4.ai) could replace or augment the current Bayesian Elo estimator
169. Explore whether Unchessed's persona system concept could generalize to other games (a portable "opponent-adaptive AI" framework)
170. Investigate publishing an academic-style writeup of the persona system itself, since the IEEE doc notes it has no direct precedent in the literature

## UCI protocol / engine tooling (171-180)

171. Implement `go ponder`/`ponderhit` properly (full UCI compliance)
172. Audit full UCI option spec compliance against the protocol document (min/max/default declarations)
173. Explore supporting `setoption` validation with clearer error messages for malformed values
174. Build a UCI protocol conformance test suite beyond the existing 9-step smoke test
175. Explore supporting the `UCI_ShowCurrLine`/`UCI_ShowRefutations` optional protocol extensions
176. Investigate whether MultiPV output formatting matches what common GUIs expect exactly
177. Explore adding a `debug` UCI command handler for structured diagnostic output
178. Investigate NNUE file format versioning/migration strategy for future architecture changes
179. Build a standalone NNUE file inspector CLI tool (dump architecture/metadata without loading into the engine)
180. Explore adding build-time feature flags (e.g. a "lite" build without policy net for smaller binary size) and investigate reproducible/deterministic builds for release verification

## Documentation / developer experience (181-190)

181. Write a CONTRIBUTING.md covering the SPRT/smoke-test workflow for anyone else who joins the project
182. Document the NNUE training pipeline end-to-end as a standalone guide (currently split across README + scripts)
183. Write an architecture decision record (ADR) log for major choices (v3→v4, HCE fallback, persona design)
184. Build a glossary of project-specific terms (CLINCH, troll tier, HeuristicPrior, etc.) for new contributors
185. Document the WSL/cloud training environment setup as a reproducible guide
186. Write a "how to read the SPRT logs" guide for anyone unfamiliar with the pentanomial model
187. Document the exact meaning/derivation of each magic constant (ENGINE_CEILING, EVAL_CLAMP, SCALE) in one place
188. Build a changelog that tracks Elo-relevant changes specifically, separate from general commit history
189. Write a troubleshooting guide for common UCI GUI integration issues
190. Document the persona system's design rationale as a standalone explainer (why hysteresis, why four personas specifically)

## Benchmarking / community / ecosystem (191-200)

191. Run Unchessed against a broader field of open-source engines at fixed Elo estimates for external calibration
192. Submit to a public rating list (CCRL-style) once strength is stable, for third-party validation
193. Explore publishing the human policy net's move-prediction results as a standalone research note (the Maia comparison table is genuinely interesting on its own)
194. Explore open-sourcing the NNUE training pipeline as a reusable tool for other from-scratch engine projects
195. Investigate whether the opponent-adaptive persona system is patentable/publishable as novel prior art (defensive publication to prevent others from patenting it)
196. Explore community feedback channels (a Discord/forum) for engine testers once it's more mature
197. Investigate partnering with a chess platform for beta-testing the adaptive engine against real human opponents at scale
198. Explore a public leaderboard of "which persona am I facing" — a fun demo showing the live Bayesian Elo estimate to spectators
199. Investigate licensing strategy (currently no LICENSE file per the GitHub API response) before any public release or collaboration
200. Write a postmortem/retrospective on the v3 regression and smoke-test fix as a case study, useful both internally and as a shareable engineering story
