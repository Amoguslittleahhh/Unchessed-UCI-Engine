#!/usr/bin/env python3
"""Minimal IEEE-two-column PDF writer (stdlib only). Helvetica metrics."""
from __future__ import annotations

import zlib
from pathlib import Path

W, H = 612, 792  # US Letter
MARGIN_T, MARGIN_B, MARGIN_X = 56, 50, 48
GUTTER = 16
COL_W = (W - 2 * MARGIN_X - GUTTER) / 2
LEADING = 11
SIZE = 9
TITLE_SIZE = 14
H1 = 11


def escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


class Page:
    def __init__(self):
        self.ops = []

    def text(self, x, y, s, size=SIZE, leading=None):
        self.ops.append(
            f"BT /F1 {size} Tf {x:.2f} {y:.2f} Td ({escape(s)}) Tj ET"
        )

    def rule(self, x, y, w, sw=0.6):
        self.ops.append(f"{sw} w {x:.2f} {y:.2f} m {x+w:.2f} {y:.2f} l S")

    def raw(self) -> bytes:
        stream = "\n".join(self.ops).encode("latin-1", "replace")
        return stream


def wrap(text, width_pt, size=SIZE):
    # Helvetica average ~0.5 em
    max_chars = max(8, int(width_pt / (size * 0.48)))
    words = text.split()
    lines, cur = [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if len(t) <= max_chars:
            cur = t
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines or [""]


def build(path: Path):
    title = "Fail-Closed Cloud Training of a HalfKAv2_hm NNUE under Live Persona Adaptation: Speed Choices that Do Not Spend Elo"
    authors = "Unchessed AI Technical Report TR-2026-08-31b  |  IEEE-style preprint  |  31 August 2026"
    abstract = (
        "We specify the exact A100 training stack Unchessed will run in the cloud: "
        "GPU-resident shards, batch 131072, TF32, bfloat16 autocast, fused Adam, "
        "cuDNN benchmark, early-stop on val-MAE, best-checkpoint export, and a "
        "fail-closed preflight that refuses to start unless Adaptive/persona stays "
        "on, UnarchitecturedHint stays off, and the operator types "
        "GO_CLOUD=I_ACCEPT_SPRT_GATES. Speed knobs do not change the WDL 2.5 loss "
        "or the feature scheme. Quality is defined as SPRT, not as a small printed "
        "val-MAE. Pipeline certainty is 100% for those gates; Elo is not."
    )
    sections = [
        ("I. INTRODUCTION",
         "Billing on an A100 is wall-clock. The previous cloud launcher still labelled "
         "the run v3, could export a last-checkpoint net, and could start without a go "
         "token. This revision makes the launcher v4, fail-closed, persona-on, and as "
         "fast as Ampere allows without touching the loss. We do not claim 100% Elo. "
         "We claim 100% pipeline certainty: a billed run cannot start unless preflight "
         "passes, and Adaptive cannot be turned off by that run."),
        ("II. NON-NEGOTIABLE QUALITY",
         "Loss remains |sigmoid(raw)-t|^2.5 with t = 0.7 sigmoid(cp/400) + 0.3 (wdl/2), "
         "the Stockfish nnue-pytorch recipe already used here. Features remain HalfKAv2_hm "
         "with factorization; export is UNCHNNUE v4 (22528 x 256 + 8-bucket head). "
         "The epoch argument is a cap of 15; early-stop patience 3 and min-delta 0.1 cp "
         "export the best checkpoint, which is the round-13 fix for three SPRTs that "
         "had shipped a worse last epoch. Adaptive stays default true. UnarchitecturedHint "
         "stays default false. A net is promoted only after SPRT versus unchessed-nnue.bin "
         "at tc=10+0.1 with Adaptive on. That is the quality bar. A small val-MAE is not."),
        ("III. SPEED CHOICES (WHY THESE, NOT OTHERS)",
         "GPU-resident data. One host-to-device copy of the 104-byte records; shuffle is "
         "randperm on device. This removes the PCIe round-trip that made CPU training the "
         "slow path on the WSL box. Batch 131072 is the production A100 size already in "
         "full_pipeline_cloud.sh. The net is tiny relative to 80 GB; occupancy of EmbeddingBag "
         "is the limit, so larger batches are the first-order speed lever. TF32 tensor cores "
         "are the Ampere default for matmul; EmbeddingBag integer indices are unchanged and "
         "exported weights stay IEEE-754 f32. bf16 autocast covers forward and loss; Adam "
         "keeps fp32 master weights. No GradScaler: bf16 shares fp32's exponent. Indices are "
         "int64 and sit outside autocast. Opt-out with USE_AMP=0 if a future kernel miscasts. "
         "Fused Adam is one CUDA kernel for the moment update, with TypeError fallback. "
         "cuDNN benchmark is kept for the Linear head. torch.compile is OFF by default: "
         "EmbeddingBag plus dynamic nnz has compiled poorly in the field. "
         "zero_grad(set_to_none=True) avoids writing zeros across 11M+ embedding rows every "
         "step. Early-stop is the cheapest quality win: the 100-epoch v3 rerun paid for 35 "
         "useless epochs after the 53.6 cp minimum. None of these knobs change the loss."),
        ("IV. FAIL-CLOSED PREFLIGHT",
         "tools/nnue_cloud_runtime.py returns a blocking error list if PERSONA_ACTIVE is not "
         "1, if UNARCH_HINT is 1, if record count is below 1000 or above 500 million, or if "
         "the go token is required and missing. scripts/nnue-pipeline/cloud_train_v4.sh "
         "exports REQUIRE_CLOUD_GO=1. An empty GO_CLOUD in the monthly pipeline now fails "
         "training instead of silently starting a v3 run. That is the 100% certainty this "
         "paper is allowed to claim: the billed process cannot begin in a persona-off, "
         "hint-on, or un-acknowledged configuration."),
        ("V. PERSONA STAYS ACTIVE",
         "MATCH / PUNISH / CLINCH / DEFEND live in adapt.rs and read search eval_cp, not "
         "trainer val-MAE. Training a stronger net with Adaptive left at its UCI default "
         "is the configuration the SPRT must use. Turning Adaptive off for a cleaner Elo "
         "number is a different experiment and is not this launcher. Simulation in the "
         "companion report showed mode-flip rate falling from 30.5% at 80 cp eval MAE to "
         "4.5% at 10 cp: a more accurate evaluator stabilises persona, it does not disable it."),
        ("VI. WHAT THIS DOES NOT CLAIM",
         "Sub-20 cp val-MAE on 5000-node HCE labels remains unreachable (Gaussian Bayes "
         "floor about 56 cp; repo best 51.1 cp at 27M unique). Speed knobs do not lower "
         "that floor. Cloud 178M is still gated on a local 108M best-checkpoint SPRT for "
         "Elo. This paper only makes that run fast and un-sabotageable if someone accepts "
         "the gate. Proven real-world results remain the committed SPRTs: NNUE +107.1 Elo "
         "versus HCE, incremental accumulators +68.6 Elo, HalfKA v3 -70.3 Elo, data-scale "
         "ladder -796 / -383 / -307 Elo versus the shipped net."),
        ("VII. REPRODUCTION",
         "GO_CLOUD=I_ACCEPT_SPRT_GATES DEVICE=cuda BATCH_SIZE=131072 "
         "scripts/nnue-pipeline/cloud_train_v4.sh out.bin shard*.bin "
         "Then SPRT with Adaptive=true. Do not promote on val-MAE alone."),
        ("REFERENCES",
         "[1] Stockfish / nnue-pytorch, HalfKAv2_hm feature scheme. "
         "[2] D. Tan and A. Watkinson Medina, Study of the Proper NNUE Dataset, arXiv:2412.17948. "
         "[3] Unchessed AI, NNUE v4 full-scale training recipe, docs/nnue-v4-training-recipe.md. "
         "[4] Unchessed AI, Centipawn validation-loss floors and persona coupling, "
         "docs/ieee-low-cp-val-mae-and-persona.md. "
         "[5] Ruoss et al., Grandmaster-level chess without search, arXiv:2402.04494."),
    ]

    pages = []
    page = Page()
    y = H - MARGIN_T
    # title block full width
    for line in wrap(title, W - 2 * MARGIN_X, TITLE_SIZE):
        page.text(MARGIN_X, y, line, size=TITLE_SIZE)
        y -= 16
    y -= 4
    page.text(MARGIN_X, y, authors, size=8)
    y -= 10
    page.rule(MARGIN_X, y, W - 2 * MARGIN_X)
    y -= 14
    page.text(MARGIN_X, y, "Abstract—", size=9)
    y -= LEADING
    for line in wrap(abstract, W - 2 * MARGIN_X, SIZE):
        page.text(MARGIN_X, y, line)
        y -= LEADING
    y -= 8
    page.text(MARGIN_X, y, "Index Terms—NNUE, A100, TF32, mixed precision, SPRT, UCI Adaptive, cloud preflight.", size=8)
    y -= 16
    page.rule(MARGIN_X, y, W - 2 * MARGIN_X)
    y -= 18

    col = 0
    col_x = [MARGIN_X, MARGIN_X + COL_W + GUTTER]
    cy = y

    def new_page():
        nonlocal page, cy, col
        pages.append(page)
        page = Page()
        col = 0
        cy = H - MARGIN_T

    def ensure(space=LEADING):
        nonlocal cy, col
        if cy - space < MARGIN_B:
            if col == 0:
                col = 1
                cy = y if len(pages) == 0 else (H - MARGIN_T)
                # after first page, columns start at top
                if pages:
                    cy = H - MARGIN_T
            else:
                new_page()
                cy = H - MARGIN_T

    # After title, two-col body uses current y as first-page column top
    first_col_top = cy

    def emit_line(s, size=SIZE, lead=LEADING):
        nonlocal cy
        ensure(lead)
        x = col_x[col] if pages or col == 1 or cy <= first_col_top else col_x[col]
        # simplify: always use col_x[col]
        x = col_x[col]
        if not pages:
            # first page columns start at first_col_top
            pass
        page.text(x, cy, s, size=size)
        cy -= lead

    for h, body in sections:
        ensure(LEADING * 2)
        emit_line(h, size=H1, lead=14)
        page.rule(col_x[col], cy + 10, COL_W, 0.4)
        for line in wrap(body, COL_W, SIZE):
            emit_line(line)

    pages.append(page)

    # footer page numbers
    for i, p in enumerate(pages, 1):
        p.text(W / 2 - 20, 28, f"{i} / {len(pages)}", size=8)
        p.text(MARGIN_X, 28, "Unchessed AI  TR-2026-08-31b", size=7)

    # assemble PDF
    objects = []
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    # pages obj filled later
    objects.append(b"<< /Type /Pages >>")  # placeholder
    objects.append(
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"
    )
    content_ids = []
    page_ids = []
    for p in pages:
        stream = p.raw()
        compressed = zlib.compress(stream)
        content = (
            f"<< /Filter /FlateDecode /Length {len(compressed)} >>\nstream\n".encode()
            + compressed
            + b"\nendstream"
        )
        objects.append(content)
        cid = len(objects)
        content_ids.append(cid)
        page_dict = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {W} {H}] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {cid} 0 R >>"
        ).encode()
        objects.append(page_dict)
        page_ids.append(len(objects))

    kids = " ".join(f"{i} 0 R" for i in page_ids)
    objects[1] = f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>".encode()

    buf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for i, obj in enumerate(objects, 1):
        offsets.append(len(buf))
        buf += f"{i} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref = len(buf)
    buf += f"xref\n0 {len(objects)+1}\n".encode()
    buf += b"0000000000 65535 f \n"
    for off in offsets[1:]:
        buf += f"{off:010d} 00000 n \n".encode()
    buf += (
        f"trailer << /Size {len(objects)+1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n"
    ).encode()
    path.write_bytes(buf)
    print(f"wrote {path} ({len(buf)} bytes, {len(pages)} pages)")


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    build(root / "docs" / "ieee-cloud-nnue-speed-quality.pdf")
