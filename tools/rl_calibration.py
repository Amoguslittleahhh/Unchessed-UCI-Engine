#!/usr/bin/env python3
"""Bounded, deterministic CPU calibration for RL plumbing.

This is deliberately a toy transition system, not chess and not a strength
experiment. It checks the lowest-cost interfaces needed before considering a
self-play implementation: legal-action masks, deterministic trajectories,
replay records, terminal outcomes, and a value-only learning sanity update.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class ReplayRecord:
    episode: int
    step: int
    state: int
    side: int
    legal_actions: tuple[int, ...]
    action: int
    outcome: float


class ToySelfPlay:
    """Small acyclic game with explicit legal actions and deterministic ties."""

    def __init__(self, seed: int) -> None:
        self.rng = random.Random(seed)

    @staticmethod
    def legal_actions(state: int) -> tuple[int, ...]:
        if state >= 6:
            return ()
        return (0, 1) if state in (0, 1, 2, 3) else (0,)

    @staticmethod
    def transition(state: int, action: int) -> int:
        if action not in ToySelfPlay.legal_actions(state):
            raise ValueError(f"illegal action {action} at state {state}")
        return state + (2 if action == 1 else 1)

    @staticmethod
    def terminal_outcome(state: int) -> float:
        # States 6 and 7 are wins for side-to-move's preceding player; 8 is draw.
        return 1.0 if state in (6, 7) else 0.0

    def episode(self, episode: int, max_steps: int = 8) -> list[ReplayRecord]:
        state, side, records = 0, 1, []
        for step in range(max_steps):
            legal = self.legal_actions(state)
            if not legal:
                break
            action = legal[self.rng.randrange(len(legal))]
            nxt = self.transition(state, action)
            if not self.legal_actions(nxt):
                outcome = self.terminal_outcome(nxt) * side
                records.append(ReplayRecord(episode, step, state, side, legal, action, outcome))
                return records
            records.append(ReplayRecord(episode, step, state, side, legal, action, 0.0))
            state, side = nxt, -side
        return records


def generate(seed: int, games: int) -> tuple[list[ReplayRecord], dict[str, int]]:
    game = ToySelfPlay(seed)
    replay: list[ReplayRecord] = []
    violations = 0
    completed = 0
    for episode in range(games):
        records = game.episode(episode)
        if records:
            completed += 1
        for record in records:
            if record.action not in record.legal_actions:
                violations += 1
        replay.extend(records)
    return replay, {"games_completed": completed, "legal_mask_violations": violations}


def state_features(state: int, side: int, width: int = 14) -> list[float]:
    features = [0.0] * width
    offset = 0 if side > 0 else 7
    features[offset + min(state, 6)] = 1.0
    return features


def loss(weights: list[float], samples: Iterable[tuple[int, float]]) -> float:
    values = list(samples)
    if not values:
        return 0.0
    return sum((sum(w * x for w, x in zip(weights, state_features(s))) - y) ** 2 for s, y in values) / len(values)


def train_value(replay: list[ReplayRecord], updates: int, learning_rate: float) -> tuple[float, float]:
    # Alternate records rather than whole episodes: this keeps the tiny toy
    # split balanced across reachable states while remaining deterministic.
    train = [(r.state, r.side, r.outcome) for i, r in enumerate(replay) if i % 2 == 0]
    holdout = [(r.state, r.side, r.outcome) for i, r in enumerate(replay) if i % 2 == 1]
    if not train or not holdout:
        raise ValueError("calibration requires at least two games for train/holdout")
    weights = [0.0] * 14
    before = sum((sum(w * x for w, x in zip(weights, state_features(s, side))) - y) ** 2 for s, side, y in holdout) / len(holdout)
    for _ in range(updates):
        for state, side, target in train:
            features = state_features(state, side)
            prediction = sum(w * x for w, x in zip(weights, features))
            error = prediction - target
            for i, feature in enumerate(features):
                weights[i] -= learning_rate * 2.0 * error * feature
    after = sum((sum(w * x for w, x in zip(weights, state_features(s, side))) - y) ** 2 for s, side, y in holdout) / len(holdout)
    return before, after


def run(seed: int, games: int, updates: int, learning_rate: float) -> dict[str, object]:
    replay, counts = generate(seed, games)
    before, after = train_value(replay, updates, learning_rate)
    replay_json = json.dumps([asdict(r) for r in replay], sort_keys=True, separators=(",", ":"))
    replay_hash = hashlib.sha256(replay_json.encode()).hexdigest()
    return {
        "schema": "unchessed.rl-calibration.v1",
        "status": "completed",
        "experiment": "toy_value_only_self_play_plumbing",
        "seed": seed,
        "games_requested": games,
        "updates": updates,
        "learning_rate": learning_rate,
        "replay_records": len(replay),
        "replay_sha256": replay_hash,
        "heldout_loss_before": before,
        "heldout_loss_after": after,
        "heldout_loss_decreased": after < before,
        **counts,
        "not_chess": True,
        "no_elo_claim": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--updates", type=int, default=25)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if args.games < 2 or args.updates < 1 or not 0 < args.learning_rate < 1:
        parser.error("require games >= 2, updates >= 1, and 0 < learning-rate < 1")
    report = run(args.seed, args.games, args.updates, args.learning_rate)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.json:
        args.json.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if report["legal_mask_violations"] == 0 and report["heldout_loss_decreased"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
