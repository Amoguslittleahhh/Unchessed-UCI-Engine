"""Cloud-training runtime flags for train_nnue.py.

Pure Python, no torch. The trainer imports this so the A100 path is
explicit, testable, and fail-closed: a missing GO token or a persona-off
request does not silently start a billed run.

Speed flags (TF32, bf16 AMP, fused Adam, cudnn.benchmark) do not change
the WDL^{2.5} loss, the HalfKAv2_hm features, or Adaptive defaults.
Quality gates (best-ckpt, early-stop, SPRT-after) are the same recipe as
docs/nnue-v4-training-recipe.md.
"""
from __future__ import annotations

import os

# Operator must type this exact string. Anything else is a no-go.
CLOUD_GO_TOKEN = "I_ACCEPT_SPRT_GATES"

DEFAULTS = {
    "ALLOW_TF32": "1",
    "USE_AMP": "1",  # bf16 autocast on CUDA; ignored on CPU
    "FUSED_ADAM": "1",
    "CUDNN_BENCHMARK": "1",
    "TORCH_COMPILE": "0",  # EmbeddingBag compile is not guaranteed; opt-in
    "JSONL_METRICS": "1",
    "PERSONA_ACTIVE": "1",
    "UNARCH_HINT": "0",
    "BATCH_SIZE_CUDA": "131072",
    "BATCH_SIZE_CPU": "65536",
    "EPOCH_CAP": "15",
    "EARLY_STOP_PATIENCE": "3",
    "EARLY_STOP_MIN_DELTA": "0.1",
}


def env_bool(name, default="0"):
    v = os.environ.get(name, default).strip().lower()
    return v in ("1", "true", "yes", "on")


def cloud_flags():
    """Resolved flag set. Does not check GO token (see preflight)."""
    return {
        "allow_tf32": env_bool("ALLOW_TF32", DEFAULTS["ALLOW_TF32"]),
        "use_amp": env_bool("USE_AMP", DEFAULTS["USE_AMP"]),
        "fused_adam": env_bool("FUSED_ADAM", DEFAULTS["FUSED_ADAM"]),
        "cudnn_benchmark": env_bool("CUDNN_BENCHMARK", DEFAULTS["CUDNN_BENCHMARK"]),
        "torch_compile": env_bool("TORCH_COMPILE", DEFAULTS["TORCH_COMPILE"]),
        "jsonl_metrics": env_bool("JSONL_METRICS", DEFAULTS["JSONL_METRICS"]),
        "persona_active": env_bool("PERSONA_ACTIVE", DEFAULTS["PERSONA_ACTIVE"]),
        "unarch_hint": env_bool("UNARCH_HINT", DEFAULTS["UNARCH_HINT"]),
        "epoch_cap": int(os.environ.get("EPOCH_CAP", DEFAULTS["EPOCH_CAP"])),
        "go_token_ok": os.environ.get("GO_CLOUD", "") == CLOUD_GO_TOKEN,
    }


def preflight_errors(n_records, device_type, go_required=True):
    """Return a list of blocking errors. Empty list = may train.

    Fail-closed rules:
      - persona must stay on
      - UnarchitecturedHint must stay off (not part of this train)
      - GO token required when go_required (cloud launcher)
      - record count must be enough to train and not exceed 500M safety cap
      - AMP/TF32 only claimed on CUDA
    """
    errors = []
    flags = cloud_flags()
    if not flags["persona_active"]:
        errors.append("PERSONA_ACTIVE must be 1 — Adaptive stays on; do not train a persona-off net as default")
    if flags["unarch_hint"]:
        errors.append("UNARCH_HINT must be 0 — UnarchitecturedHint stays default-off until its own SPRT")
    if go_required and not flags["go_token_ok"]:
        errors.append(
            f"GO_CLOUD is not {CLOUD_GO_TOKEN!r} — refusing to start a billed run"
        )
    if n_records < 1000:
        errors.append(f"only {n_records} records — refusing to train")
    if n_records > 500_000_000:
        errors.append(f"{n_records} records exceed SAFE_MAX_RECORDS=500000000")
    if flags["use_amp"] and device_type != "cuda":
        # not an error: trainer must ignore AMP on CPU
        pass
    if flags["epoch_cap"] < 1:
        errors.append(f"EPOCH_CAP must be >= 1, got {flags['epoch_cap']}")
    return errors


def apply_torch_speed(torch_mod, device):
    """Best-effort CUDA speed knobs. Safe no-ops on CPU / old torch."""
    flags = cloud_flags()
    notes = []
    if getattr(device, "type", None) != "cuda":
        notes.append("cpu: speed knobs skipped")
        return notes
    try:
        if flags["allow_tf32"]:
            torch_mod.backends.cuda.matmul.allow_tf32 = True
            torch_mod.backends.cudnn.allow_tf32 = True
            if hasattr(torch_mod, "set_float32_matmul_precision"):
                torch_mod.set_float32_matmul_precision("high")
            notes.append("tf32 on")
        if flags["cudnn_benchmark"]:
            torch_mod.backends.cudnn.benchmark = True
            notes.append("cudnn.benchmark on")
    except Exception as exc:  # pragma: no cover - defensive
        notes.append(f"speed knobs partial: {exc}")
    return notes
