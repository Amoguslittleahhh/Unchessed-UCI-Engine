# Research request: is DiffusionBlocks worth it for a transformer labeling oracle?

## Context

Unchessed AI is a from-scratch Rust UCI chess engine
(github.com/Amoguslittleahhh/Unchessed-UCI-Engine). Its NNUE evaluator is
small (5,767,937 params: HalfKAv2_hm features, 256-wide accumulator, single
linear output head) and trains on 108M HCE-labeled self-play positions.
Incremental accumulator updates were recently implemented and verified
(commit `00d0941` on `main`) — that item is done, not what this prompt is
about.

An existing, already-scoped-but-unstarted backlog item (see
`scripts/research/remaining_research_topics.md`, item 84, and
`200_research_ideas.md`, item 153) proposes: train a large transformer as a
one-time labeling oracle, generate improved labels for the existing
108M-position dataset (or a subset), then retrain the small deployed NNUE
on those labels instead of the current HCE-generated ones. The transformer
itself is never deployed for inference — only used once, offline, to
produce a better training signal. This is distinct from (and does not
replace) the separately-planned NNUE-bootstrapping idea (using NNUE's own
eval as the labeler), which is a different, already-answered item.

This idea has never been prototyped or costed out. It's speculative.

## What prompted this prompt

A newly-published ICLR 2026 paper, **DiffusionBlocks** (Shing, Koyama,
Akiba — Sakana AI / U. Tokyo, arXiv:2506.14202), reinterprets residual
connections in transformer networks as Euler steps of a diffusion process,
enabling each block of a deep network to be trained independently via a
local score-matching objective instead of full end-to-end backpropagation.
Demonstrated on 12-24 layer ViT/DiT/autoregressive-transformer/masked-
diffusion models, it cuts training memory by a factor of B (the number of
blocks) while matching end-to-end backprop's quality, and additionally
collapses recurrent-depth models' multi-iteration training into a single
forward pass. Code: github.com/SakanaAI/DiffusionBlocks.

This project has hit real GPU memory constraints before (documented
elsewhere in the repo: an A100 40GB vs 80GB sizing decision for the NNUE
training pipeline, where 40GB was ruled out after measuring ~44.5GB actual
peak usage for a *tiny* NNUE run). A transformer oracle large enough to be
a meaningfully better labeler than the current HCE self-play pipeline
would be a much bigger, much more memory-hungry training job. DiffusionBlocks'
memory reduction is the specific reason this idea might now be worth
costing out rather than staying speculative indefinitely.

## What I want researched

1. **Is DiffusionBlocks actually applicable here?** It requires a network
   built from residual/skip-connected blocks with matching input/output
   dimensions per block (stated limitation: doesn't yet extend to
   architectures like U-Net where dimensions change across the network).
   Would a chess-position-evaluating transformer oracle (board state in,
   scalar or WDL-style value out) fit this constraint naturally, or would
   the input/output framing need real rework to fit DiffusionBlocks'
   denoising formulation? Chess positions aren't naturally "noised" the
   way images or text are — is there a sensible mapping (e.g. treating the
   label itself, not the board, as the diffused quantity; or some other
   framing), or does the analogy break down for this specific task?
2. **Scale the memory savings against a realistic oracle size.** Pick 2-3
   plausible transformer sizes for a chess-position value/label oracle
   (e.g. small ~10-30M params, medium ~100-300M params, referencing what
   real chess transformer projects have used — Chessformer, the 1e4.ai/
   ChessMimic project, or similar public work if you can find concrete
   parameter counts and training memory figures for comparable board-
   evaluation or move-prediction transformers). For each size, estimate
   training memory with standard end-to-end backprop vs. with
   DiffusionBlocks at a few block counts (B=2,3,4), and say whether the
   B=1 (no DiffusionBlocks) case would already fit on a single A100 80GB
   or would need DiffusionBlocks (or multi-GPU, or wouldn't fit at all).
3. **What does DiffusionBlocks cost in engineering effort and risk**,
   specifically for a one-time-use oracle (not something that needs to be
   maintained long-term or retrained repeatedly)? Given the oracle is used
   once to generate labels and then discarded, is the memory-savings-for-
   architectural-complexity tradeoff still worth it, or is it more
   sensible to just rent a bigger GPU for a one-off training run instead
   of adopting a less-standard training technique for a single use? Be
   honest if the answer is "just rent an H200 and use normal backprop" —
   the goal here is a real cost-benefit call, not padding a case for using
   the new paper's technique.
4. **Concrete recommendation**: worth prototyping, worth deferring until
   a specific larger oracle size makes memory savings load-bearing, or
   not worth pursuing at all for this project's actual scale. If "worth
   prototyping," sketch the smallest useful experiment (what to build,
   what to measure, what would confirm/deny it's worth going further).

## What NOT to do

Don't assume the answer is yes because the paper is new and interesting.
The prior pattern on this project's research requests has been: some
brainstormed ideas turn out genuinely useful, most don't, and the honest
answer for a lot of "here's a shiny new technique" prompts is "doesn't
apply at this project's scale" or "not worth the engineering cost for a
one-time job." Give that answer plainly if that's where the evidence
points.

## Output format

Four sections matching the four questions above. Keep it to what's
actually load-bearing for a decision — no padding to hit a length target,
no fabricated citations or made-up benchmark figures. If you can't find a
real comparable transformer size/memory figure for a chess-adjacent model,
say so rather than inventing one.
