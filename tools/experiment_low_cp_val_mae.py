#!/usr/bin/env python3
"""Stdlib-only experiments: how low can NNUE val-MAE go, and does that
break the adapter persona?

No torch / numpy. Reproduces the project's published diagnostic numbers as
constants, then simulates the *mechanisms* that bound val-MAE (label noise,
objective mismatch, clipping, quiet filters, WDL mix) and the *independence*
of persona mode selection from the training loss.

See docs/ieee-low-cp-val-mae-persona.pdf (generated markdown sibling).
"""
from __future__ import annotations

import json
import math
import os
import random
import statistics
import sys
from pathlib import Path

SEED = 20260831
N = 80_000  # synthetic positions per condition

# --- committed empirical facts from this repo (not invented) ---
REPO_DIAGNOSTICS = [
    {"unique": 959_102, "epochs": 20, "final_mae": 83.6, "best_mae": 57.4, "sprt_elo": -796.5},
    {"unique": 9_000_000, "epochs": 8, "final_mae": 59.3, "best_mae": 55.3, "sprt_elo": -383.5},
    {"unique": 27_000_000, "epochs": 8, "final_mae": 54.3, "best_mae": 51.1, "sprt_elo": -307.1},
]
V3_BEST_MAE = 53.6  # nnue-shards-safe/v3_research_brief.md epoch 13-14
SHIPPED_NNUE_SPRT_ELO = 107.1  # vs HCE
PERSONA_THRESHOLDS = {
    "defend_enter": -180,
    "defend_exit": -80,
    "punish_eval": 60,
    "punish_hold": 200,
    "clinch_enter_abs": 60,
    "clinch_hold_abs": 100,
    "blunder_cp": 180,
}


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def wdl_p25(pred_logit: float, target: float) -> float:
    d = sigmoid(pred_logit) - target
    return (d * d) * math.sqrt(abs(d) + 1e-8)


def mae_cp(pred_cp: float, label_cp: float) -> float:
    return abs(pred_cp - label_cp)


def sample_true_cp(rng: random.Random) -> float:
    # Mixture: mostly quiet games around 0, fat tails for material.
    if rng.random() < 0.72:
        return rng.gauss(0.0, 90.0)
    return rng.gauss(0.0, 380.0)


def label_from_true(true_cp: float, noise_sigma: float, rng: random.Random) -> float:
    return true_cp + rng.gauss(0.0, noise_sigma)


def clip(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def fit_constant_predictor(labels: list[float]) -> float:
    s = sorted(labels)
    return s[len(s) // 2]


def oracle_mae(true_cp: list[float], labels: list[float]) -> float:
    """MAE if the net recovered the *true* value, vs noisy labels."""
    return statistics.fmean(abs(t - y) for t, y in zip(true_cp, labels))


def bayes_mae_gaussian(sigma: float) -> float:
    return sigma * math.sqrt(2.0 / math.pi)


def train_linear_one_feature(xs: list[float], ys: list[float], steps=80, lr=2e-4):
    """Scalar affine y ≈ a*x + b, L1 (MAE) or used as stand-in for 'capacity'."""
    a, b = 0.0, 0.0
    n = len(xs)
    for _ in range(steps):
        ga = gb = 0.0
        for x, y in zip(xs, ys):
            pred = a * x + b
            g = 1.0 if pred > y else -1.0
            ga += g * x
            gb += g
        a -= lr * ga / n
        b -= lr * gb / n
    return a, b


def evaluate_recipe(name: str, noise_sigma: float, clip_abs: float | None,
                    quiet_max_true: float | None, wdl_blend: float,
                    train_on: str, rng: random.Random) -> dict:
    """Generate synthetic (true, label) pairs and measure floors.

    train_on: 'label' (fit labels — what we actually do) or 'true' (oracle).
    """
    true = []
    labels = []
    for _ in range(N):
        t = sample_true_cp(rng)
        if quiet_max_true is not None and abs(t) > quiet_max_true:
            continue
        y = label_from_true(t, noise_sigma, rng)
        if clip_abs is not None:
            y = clip(y, -clip_abs, clip_abs)
            t_for_true = clip(t, -clip_abs, clip_abs)
        else:
            t_for_true = t
        true.append(t_for_true)
        labels.append(y)
    n = len(true)
    if n < 1000:
        raise SystemExit(f"{name}: too few samples after filters ({n})")

    # Bayes floor vs labels if we recovered truth
    floor_vs_labels = oracle_mae(true, labels)
    # Constant (median) predictor — no-capacity baseline
    med = fit_constant_predictor(labels)
    const_mae = statistics.fmean(abs(med - y) for y in labels)

    # Affine fit of *true* onto labels: the best a linear readout of the
    # generating value can do (capacity plenty for 1-D).
    a, b = train_linear_one_feature(true, labels)
    fit_mae = statistics.fmean(abs(a * t + b - y) for t, y in zip(true, labels))

    # WDL-space training target vs cp MAE: predict sigmoid^{-1}(target)*400
    # using the blended target the trainer actually uses.
    wdl_mae = 0.0
    wdl_loss = 0.0
    for t, y in zip(true, labels):
        target = 0.7 * sigmoid(y / 400.0) + 0.3 * (0.5 + 0.5 * math.tanh(t / 600.0))
        # "perfect WDL fit": raw = logit(target)
        p = clip(target, 1e-6, 1 - 1e-6)
        raw = math.log(p / (1.0 - p))
        pred_cp = raw * 400.0
        wdl_mae += abs(pred_cp - y)
        wdl_loss += wdl_p25(raw, target)
    wdl_mae /= n
    wdl_loss /= n

    gauss_floor = bayes_mae_gaussian(noise_sigma)

    return {
        "name": name,
        "n": n,
        "noise_sigma_cp": noise_sigma,
        "clip_abs": clip_abs,
        "quiet_max_true": quiet_max_true,
        "wdl_blend": wdl_blend,
        "gaussian_bayes_mae_cp": round(gauss_floor, 2),
        "oracle_true_vs_noisy_label_mae_cp": round(floor_vs_labels, 2),
        "constant_predictor_mae_cp": round(const_mae, 2),
        "affine_true_fit_mae_cp": round(fit_mae, 2),
        "perfect_wdl_fit_mae_vs_cp_label": round(wdl_mae, 2),
        "perfect_wdl_p25_loss": round(wdl_loss, 6),
        "hits_sub_20": fit_mae < 20.0,
        "hits_sub_10": fit_mae < 10.0,
        "wdl_fit_hits_sub_20": wdl_mae < 20.0,
    }


def persona_sensitivity(rng: random.Random) -> dict:
    """Persona decide_mode uses search eval in cp, not NNUE training loss.

    Inject eval noise of various MAE and count mode-flip rate vs a noiseless
    oracle eval. This is the coupling the user asked about: can we drive
    val-MAE down without disabling MATCH/PUNISH/CLINCH/DEFEND?
    """
    modes = ("Match", "Punish", "Clinch", "Defend", "Full")

    def decide(eval_cp: int, last_blunder: bool, fullmove: int, prev: str,
               engine_suspect: bool, adaptive: bool) -> str:
        if not adaptive:
            return "Full"
        if engine_suspect:
            return "Full"
        if eval_cp < -180 or (prev == "Defend" and eval_cp < -80):
            return "Defend"
        punish = (last_blunder and eval_cp > 60) or eval_cp > 250
        if punish or (prev == "Punish" and eval_cp > 200):
            return "Punish"
        clinch_enter = fullmove > 28 and abs(eval_cp) < 60
        clinch_hold = prev == "Clinch" and abs(eval_cp) < 100
        if clinch_enter or clinch_hold:
            return "Clinch"
        return "Match"

    maes = [5, 10, 20, 50, 80, 150]
    out = []
    n_pos = 20_000
    for mae in maes:
        # Laplace noise has MAE = b (scale).
        flips = 0
        match_kept = 0
        match_total = 0
        for _ in range(n_pos):
            true_eval = int(rng.gauss(20, 220))
            last_blunder = rng.random() < 0.08
            fullmove = rng.randint(8, 60)
            prev = rng.choice(modes[:4])
            engine_suspect = rng.random() < 0.04
            adaptive = True
            gold = decide(true_eval, last_blunder, fullmove, prev, engine_suspect, adaptive)
            noisy = int(true_eval + rng.gauss(0, mae * math.sqrt(math.pi / 2)))
            got = decide(noisy, last_blunder, fullmove, prev, engine_suspect, adaptive)
            if got != gold:
                flips += 1
            if gold == "Match":
                match_total += 1
                if got == "Match":
                    match_kept += 1
        out.append({
            "eval_mae_cp": mae,
            "mode_flip_rate": round(flips / n_pos, 4),
            "match_retention": round(match_kept / max(match_total, 1), 4),
            "n": n_pos,
        })
    return {
        "note": "Persona reads search eval, not trainer val-MAE. Lower eval MAE "
                "reduces accidental DEFEND/PUNISH/CLINCH flips; MATCH stays the default.",
        "thresholds": PERSONA_THRESHOLDS,
        "by_mae": out,
        "adaptive_default": True,
        "unarchitectured_hint_default": False,
    }


def scaling_extrapolation() -> dict:
    """Log-linear fit of best-MAE vs unique positions from repo SPRTs."""
    xs = [math.log(r["unique"]) for r in REPO_DIAGNOSTICS]
    ys = [r["best_mae"] for r in REPO_DIAGNOSTICS]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    slope = num / den
    intercept = my - slope * mx

    def pred(unique: int) -> float:
        return intercept + slope * math.log(unique)

    targets = {
        "108M (local shards)": 108_000_000,
        "178M (cloud corpus)": 178_000_000,
        "500M": 500_000_000,
        "2e9 (nnue-pytorch superbatch scale)": 2_000_000_000,
    }
    forecast = {k: round(pred(v), 2) for k, v in targets.items()}
    # When would this naive fit hit 20cp / 10cp?
    def unique_for(mae: float) -> float:
        if slope >= 0:
            return float("inf")
        return math.exp((mae - intercept) / slope)

    return {
        "fit": "best_mae = a + b*ln(unique)",
        "a": round(intercept, 3),
        "b": round(slope, 3),
        "forecast_best_mae": forecast,
        "unique_for_20cp": round(unique_for(20.0)),
        "unique_for_10cp": round(unique_for(10.0)),
        "warning": "Diminishing returns already measured (9M→27M recovered only 4.2cp "
                   "best-MAE). Linear-in-log extrapolation is optimistic; do not budget "
                   "cloud spend on hitting 10–20cp by data volume alone.",
    }


def main():
    rng = random.Random(SEED)
    recipes = []
    # Current production-like: ~50-80cp residual from shallow HCE labels.
    recipes.append(evaluate_recipe(
        "baseline_hce_label_noise_70", noise_sigma=70, clip_abs=None,
        quiet_max_true=None, wdl_blend=0.7, train_on="label", rng=rng))
    recipes.append(evaluate_recipe(
        "quiet_filter_sigma35", noise_sigma=35, clip_abs=None,
        quiet_max_true=800, wdl_blend=0.7, train_on="label", rng=rng))
    recipes.append(evaluate_recipe(
        "deeper_teacher_sigma20", noise_sigma=20, clip_abs=None,
        quiet_max_true=800, wdl_blend=0.7, train_on="label", rng=rng))
    recipes.append(evaluate_recipe(
        "stockfish_teacher_sigma12", noise_sigma=12, clip_abs=None,
        quiet_max_true=600, wdl_blend=0.7, train_on="label", rng=rng))
    recipes.append(evaluate_recipe(
        "clip_600_sigma20", noise_sigma=20, clip_abs=600,
        quiet_max_true=800, wdl_blend=0.7, train_on="label", rng=rng))
    recipes.append(evaluate_recipe(
        "clip_400_sigma12_quiet", noise_sigma=12, clip_abs=400,
        quiet_max_true=400, wdl_blend=0.7, train_on="label", rng=rng))
    recipes.append(evaluate_recipe(
        "pathological_overfit_sigma3", noise_sigma=3, clip_abs=200,
        quiet_max_true=200, wdl_blend=0.7, train_on="label", rng=rng))

    persona = persona_sensitivity(rng)
    scaling = scaling_extrapolation()

    # Can WDL-perfect fit report high cp MAE? Yes — that's objective mismatch.
    mismatch = {
        "finding": "A net that drives p=2.5 WDL loss to ~0 still reports large "
                   "val-MAE in centipawns because target = 0.7*sigmoid(cp/400)+0.3*wdl "
                   "is not invertible to the search score. Direct L1-on-cp is the "
                   "objective that actually minimises val-MAE; it is not what "
                   "tools/train_nnue.py trains.",
        "trainer_loss": "|sigmoid(raw)-target|^2.5",
        "reported_metric": "MAE(raw*400, search_score_cp)",
        "implication_for_sub_20": "Switching the *reported* metric without changing "
                                  "labels cannot produce a 10–20cp net that is also "
                                  "strong; it can only make the number look smaller "
                                  "(clipping) or larger (WDL mismatch).",
    }

    go_nogo = {
        "sub_20_cp_with_current_hce_5000_node_labels": "NO-GO",
        "reason": "Bayes floor of ~56cp (sigma=70) exceeds the target; repo best "
                  "measured val-MAE is 51.1cp at 27M positions.",
        "path_that_can_hit_20cp": [
            "Regenerate labels with a much stronger teacher (Stockfish/self-distill) "
            "so label noise sigma ≲ 20cp on quiet positions.",
            "Keep M1/M2 quiet filters (arXiv:2412.17948; already in unchessed-datagen).",
            "Train L1 or Huber on cp (or report WDL loss as the primary metric).",
            "Export best checkpoint (already in train_nnue.py after round 13).",
            "Do not disable Adaptive; persona is search-side and stays on.",
        ],
        "persona_remains_active": True,
        "unarchitectured_hint_stays_off": True,
        "cloud_178M_same_recipe": "still NO-GO per docs/nnue-v4-training-recipe.md",
    }

    report = {
        "seed": SEED,
        "n_per_recipe": N,
        "repo_diagnostics": REPO_DIAGNOSTICS,
        "v3_best_mae_cp": V3_BEST_MAE,
        "shipped_nnue_sprt_elo_vs_hce": SHIPPED_NNUE_SPRT_ELO,
        "recipes": recipes,
        "persona": persona,
        "scaling": scaling,
        "objective_mismatch": mismatch,
        "go_nogo": go_nogo,
    }

    out_dir = Path(__file__).resolve().parents[1] / "artifacts"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "low-cp-val-mae-persona-experiments.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "wrote": str(out_path),
        "recipes": [{k: r[k] for k in (
            "name", "oracle_true_vs_noisy_label_mae_cp", "affine_true_fit_mae_cp",
            "perfect_wdl_fit_mae_vs_cp_label", "hits_sub_20", "hits_sub_10")}
            for r in recipes],
        "persona_flip_at_10cp": persona["by_mae"][1],
        "persona_flip_at_80cp": persona["by_mae"][4],
        "unique_for_20cp": scaling["unique_for_20cp"],
        "go": go_nogo["sub_20_cp_with_current_hce_5000_node_labels"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
