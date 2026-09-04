#!/usr/bin/env python3
"""Persona-stability vs SPRT-Elo correlation (stdlib).

Replays AR(1) search-eval traces that match the noise levels measured on
this engine (50–80 cp val-MAE, MultiPV jitter) through:

  * the *legacy* decide_mode hysteresis (adapt.rs before PersonaState)
  * the *stable* PersonaState (EMA α=0.35, dwell=2, CLINCH pad, emergencies)

Then scores each policy in paired simulated games against a Maia-like
opponent whose move quality tracks the time spent in MATCH vs FULL/PUNISH.
Pearson r between injected engine strength and observed game score is the
SPRT-correlation metric: an unstable persona injects mode noise that
decouples eval gains from W/D/L.

Seed 20260901. Writes artifacts/persona-stability-sprt.json.
"""
from __future__ import annotations

import json
import math
import random
import statistics
from pathlib import Path

SEED = 20260901
N_GAMES = 500
PLIES = 40
STRENGTHS = [0.0, 0.25, 0.5, 0.75, 1.0]  # 0 = noisy eval, 1 = +100 Elo-equivalent


def decide_legacy(eval_cp, fullmove, prev, blunder, engine_suspect, adaptive=True):
    if not adaptive:
        return "Full"
    if engine_suspect:
        return "Full"
    if eval_cp < -180 or (prev == "Defend" and eval_cp < -80):
        return "Defend"
    punish = (blunder and eval_cp > 60) or eval_cp > 250
    if punish or (prev == "Punish" and eval_cp > 200):
        return "Punish"
    clinch_enter = fullmove > 28 and abs(eval_cp) < 60
    clinch_hold = prev == "Clinch" and abs(eval_cp) < 100
    if clinch_enter or clinch_hold:
        return "Clinch"
    return "Match"


class PersonaState:
    ALPHA = 0.35
    DWELL = 2
    DEFEND_EMERGENCY = -220

    def __init__(self):
        self.mode = "Match"
        self.ema = 0.0
        self.init = False
        self.dwell = 0
        self.candidate = "Match"

    def update(self, raw, fullmove, blunder, engine_suspect, confidence=200):
        if not self.init:
            self.ema = float(raw)
            self.init = True
        else:
            self.ema = self.ALPHA * raw + (1.0 - self.ALPHA) * self.ema
        smoothed = int(round(self.ema))
        pad = max(0, min(40, confidence // 20))
        if engine_suspect:
            self.mode = "Full"
            self.dwell = 0
            self.candidate = "Full"
            return self.mode
        if smoothed < self.DEFEND_EMERGENCY:
            self.mode = "Defend"
            self.dwell = 0
            self.candidate = "Defend"
            return self.mode
        if blunder and smoothed > 60:
            self.mode = "Punish"
            self.dwell = 0
            self.candidate = "Punish"
            return self.mode
        proposed = decide_legacy(smoothed, fullmove, self.mode, False, False)
        if proposed == "Clinch" and self.mode != "Clinch" and abs(smoothed) + pad >= 60:
            proposed = "Match"
        if proposed == self.mode:
            self.dwell = 0
            self.candidate = proposed
            return self.mode
        if proposed == self.candidate:
            self.dwell += 1
        else:
            self.candidate = proposed
            self.dwell = 1
        if self.dwell >= self.DWELL:
            self.mode = proposed
            self.dwell = 0
        return self.mode


def ar1_game(rng, noise_sigma, true_drift=0.0):
    """Middlegame search scores: mean-reverting with real swings through
    the CLINCH/DEFEND/PUNISH bands (not an opening that never leaves MATCH)."""
    v = rng.gauss(20.0, 80.0)
    evals = []
    blunders = []
    for ply in range(PLIES):
        # slow wander + occasional 150cp swings (tactics the net mis-scores)
        if rng.random() < 0.08:
            v += rng.choice((-1, 1)) * rng.uniform(80, 220)
        v = 0.88 * v + true_drift * 12.0 + rng.gauss(0.0, 22.0)
        obs = v + rng.gauss(0.0, noise_sigma)
        evals.append(int(obs))
        blunders.append(rng.random() < 0.07)
    return evals, blunders


def run_legacy(evals, blunders):
    prev = "Match"
    modes = []
    for i, e in enumerate(evals):
        fullmove = 22 + i // 2
        prev = decide_legacy(e, fullmove, prev, blunders[i], False)
        modes.append(prev)
    return modes


def run_stable(evals, blunders, confidence=200):
    s = PersonaState()
    modes = []
    for i, e in enumerate(evals):
        fullmove = 22 + i // 2
        modes.append(s.update(e, fullmove, blunders[i], False, confidence))
    return modes


def flip_rate(modes):
    if len(modes) < 2:
        return 0.0
    return sum(a != b for a, b in zip(modes, modes[1:])) / (len(modes) - 1)


def match_share(modes):
    return sum(m == "Match" for m in modes) / len(modes)


def expected_score(modes, strength, rng):
    """Maia-like opponent ~1500. FULL/PUNISH convert; MATCH plays level;
    CLINCH is slightly worse vs accurate defence; DEFEND is resistance."""
    # strength 0..1 scales conversion when we leave MATCH.
    p = 0.5
    for m in modes:
        if m == "Punish":
            p += 0.012 * (0.4 + 0.6 * strength)
        elif m == "Full":
            p += 0.010 * (0.5 + 0.5 * strength)
        elif m == "Clinch":
            p += 0.002 * strength - 0.004  # accidental clinch costs
        elif m == "Defend":
            p += 0.001
        else:
            p += 0.001 * strength
    p = max(0.05, min(0.95, p))
    # one game outcome
    r = rng.random()
    if r < p * p:
        return 1.0
    if r > 1.0 - (1.0 - p) * (1.0 - p):
        return 0.0
    return 0.5


def pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0 or dy == 0:
        return 0.0
    return num / (dx * dy)


def sprt_llr(wins, losses, draws, elo0=0.0, elo1=5.0):
    """2-player pentanomial-style LLR using a 3-nomial (W/D/L) approximation."""
    n = wins + losses + draws
    if n == 0:
        return 0.0

    def expected_p(elo):
        # logistic: P(win)+0.5 P(draw) with drawish chess ~0.32 draws
        s = 1.0 / (1.0 + 10 ** (-elo / 400.0))
        draw = 0.32
        win = (s - 0.5 * draw)
        loss = 1.0 - win - draw
        return max(1e-6, win), max(1e-6, draw), max(1e-6, loss)

    w0, d0, l0 = expected_p(elo0)
    w1, d1, l1 = expected_p(elo1)
    ll = wins * math.log(w1 / w0) + draws * math.log(d1 / d0) + losses * math.log(l1 / l0)
    return ll


def main():
    rng = random.Random(SEED)
    # Noise at current best-net MAE (~50) vs after EMA (effective ~32)
    noise = 50.0
    by_policy = {"legacy": [], "stable": []}
    strength_series = {"legacy": [], "stable": []}
    score_series = {"legacy": [], "stable": []}

    for strength in STRENGTHS:
        for _ in range(N_GAMES):
            evals, blunders = ar1_game(rng, noise, true_drift=8.0 * strength)
            for name, modes in (
                ("legacy", run_legacy(evals, blunders)),
                ("stable", run_stable(evals, blunders)),
            ):
                sc = expected_score(modes, strength, rng)
                by_policy[name].append({
                    "strength": strength,
                    "flips": flip_rate(modes),
                    "match_share": match_share(modes),
                    "score": sc,
                    "modes": {
                        m: modes.count(m) / len(modes)
                        for m in ("Match", "Punish", "Clinch", "Defend", "Full")
                    },
                })
                strength_series[name].append(strength)
                score_series[name].append(sc)

    def summarise(name):
        rows = by_policy[name]
        r = pearson(strength_series[name], score_series[name])
        r_flip = pearson([x["flips"] for x in rows], [x["score"] for x in rows])
        wins = sum(1 for x in rows if x["score"] == 1)
        losses = sum(1 for x in rows if x["score"] == 0)
        draws = sum(1 for x in rows if x["score"] == 0.5)
        return {
            "mean_flip_rate": round(statistics.fmean(x["flips"] for x in rows), 4),
            "mean_match_share": round(statistics.fmean(x["match_share"] for x in rows), 4),
            "mean_clinch_share": round(
                statistics.fmean(x["modes"]["Clinch"] for x in rows), 4
            ),
            "mean_punish_share": round(
                statistics.fmean(x["modes"]["Punish"] for x in rows), 4
            ),
            "mean_score": round(statistics.fmean(x["score"] for x in rows), 4),
            "pearson_strength_vs_score": round(r, 4),
            "pearson_flip_vs_score": round(r_flip, 4),
            "wdl": {"W": wins, "D": draws, "L": losses},
            "sprt_llr_elo0_0_elo1_5": round(sprt_llr(wins, losses, draws), 3),
            "n_games": len(rows),
        }

    # Same traces, paired: how often does stable disagree with legacy?
    rng2 = random.Random(SEED)
    disagree = 0
    n_ply = 0
    for _ in range(200):
        evals, blunders = ar1_game(rng2, noise)
        a, b = run_legacy(evals, blunders), run_stable(evals, blunders)
        disagree += sum(x != y for x, y in zip(a, b))
        n_ply += len(a)

    out = {
        "seed": SEED,
        "n_games_per_strength": N_GAMES,
        "strengths": STRENGTHS,
        "eval_noise_sigma_cp": noise,
        "plies": PLIES,
        "legacy": summarise("legacy"),
        "stable": summarise("stable"),
        "flip_rate_reduction": round(
            1.0 - summarise("stable")["mean_flip_rate"] / max(1e-9, summarise("legacy")["mean_flip_rate"]),
            4,
        ),
        "pearson_lift": round(
            summarise("stable")["pearson_strength_vs_score"]
            - summarise("legacy")["pearson_strength_vs_score"],
            4,
        ),
        "ply_disagreement_rate": round(disagree / n_ply, 4),
        "note": "Simulation of search-eval traces, not cutechess. Real SPRT "
                "still required before claiming Elo. Adaptive stays on.",
        "engine_contract": {
            "adaptive_default": True,
            "unarchitectured_hint_default": False,
            "ema_alpha": PersonaState.ALPHA,
            "dwell": PersonaState.DWELL,
        },
    }
    dest = Path(__file__).resolve().parents[1] / "artifacts" / "persona-stability-sprt.json"
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: out[k] for k in (
        "legacy", "stable", "flip_rate_reduction", "pearson_lift", "ply_disagreement_rate"
    )}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
