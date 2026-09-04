#!/usr/bin/env python3
"""Move-prediction pretrain probe (dual-elo conditioned, legal-only).

Trains a small NumPy MLP on the shards produced by
tools/pretrain_move_dataset.py with the pretrain objective:
cross-entropy over the position's LEGAL actions (20480 = 4096 moves x
5 promotion classes, the Unarchitectured v1 student action
vocabulary), conditioned on (board, elo_self, elo_oppo). This is the
sandbox-runnable proof-of-concept for the two-stage retrain plan
(docs/move-prediction-pretrain-plan.md): stage 1 pretrain = this
objective over the whole mixed corpus; the probe shows whether the
objective actually makes the dual-elo conditioning *informative*.

The gate metric is the rating-conditioning sweep from
docs/rating-conditioning-finding.md, re-run for the probe model: on a
fixed set of held-out positions (opponent elo fixed at 1500, mover
elo swept 600 -> 3200), count how many positions change their
predicted top-1 action between the extremes. The canonical v1
finding was 0/200 (rating input inert); a working pretrain must
show a substantial flip count with the right directional behavior
(high-elo play more concentrated).

Model: [768 board bits + castling one-hot + elo_self/100 +
elo_oppo/100] -> hidden ReLU layers -> 20480 logits, masked to the
legal set (softmax over legal only), L2 + Adam. Deterministic for a
seed. No GPU needed; a 10k-row shard trains in minutes on a laptop.

Outputs: --report JSON (per-band top-1 accuracy / mean top-1
probability, the sweep, baselines) + --checkpoint npz (weights).

Usage:
  python3 tools/pretrain_move_dataset.py --labels \
      data/selfplay/maia3-100-3200-labels.jsonl --out /tmp/pretrain \
      --val-games 50
  python3 tools/pretrain_move_predictor.py --data /tmp/pretrain \
      --epochs 15 --report /tmp/pretrain/report.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

CASTLING_BITS = 4


def load_shard(npz_path: Path) -> dict:
    z = np.load(str(npz_path))
    return {k: z[k] for k in z.files}


def features(arrays: dict) -> np.ndarray:
    """[768 board bits + 4 castling bits + 2 elo] float32."""
    bb = arrays["bitboards"]
    bits = np.zeros((bb.shape[0], 768), dtype=np.float32)
    for plane in range(12):
        v = bb[:, plane]
        for bit in range(64):
            bits[:, plane * 64 + bit] = ((v >> bit) & 1).astype(np.float32)
    elos = np.stack([arrays["elo_self"] / 100.0,
                     arrays["elo_oppo"] / 100.0], axis=1).astype(np.float32)
    cast = np.stack([((arrays["castling"] >> i) & 1).astype(np.float32)
                     for i in range(CASTLING_BITS)], axis=1)
    return np.concatenate([bits, cast, elos], axis=1)


def log_softmax_masked(logits: np.ndarray, legal: np.ndarray,
                       legal_count: np.ndarray) -> np.ndarray:
    """log p over legal actions only (0 elsewhere). legal: (N, K)."""
    n = logits.shape[0]
    out = np.full_like(logits, -1e9, dtype=np.float64)
    for i in range(n):
        k = int(legal_count[i])
        acts = legal[i, :k].astype(np.int64)
        x = logits[i, acts].astype(np.float64)
        m = x.max()
        out[i, acts] = x - m - np.log(np.exp(x - m).sum())
    return out


class MLP:
    def __init__(self, widths: list[int], seed: int,
                 l2: float = 1e-4):
        rng = np.random.default_rng(seed)
        self.W = []
        self.b = []
        for i in range(len(widths) - 1):
            fan = widths[i]
            w = (rng.standard_normal((widths[i + 1], fan))
                 * np.sqrt(2.0 / fan)).astype(np.float32)
            self.W.append(w)
            self.b.append(np.zeros(widths[i + 1], dtype=np.float32))
        self.l2 = l2

    def forward(self, x: np.ndarray):
        acts = [x]
        h = x
        for i in range(len(self.W) - 1):
            h = np.maximum(h @ self.W[i].T + self.b[i], 0.0)
            acts.append(h)
        logits = h @ self.W[-1].T + self.b[-1]
        return logits, acts

    def step(self, x, lr, m, v, mb, vb, t):
        """One Adam step for masked cross-entropy (context set via
        set_batch_context: legal actions, legal counts, targets)."""
        logits, acts = self.forward(x)
        lsm = log_softmax_masked(logits,
                                 self._legal_cache[0], self._legal_cache[1])
        n = logits.shape[0]
        grad_logits = np.exp(lsm)
        # -d/dx log p_y = p - onehot(y)
        for i in range(n):
            grad_logits[i, self._y_cache[i]] -= 1.0
        loss = float(np.mean([-lsm[i, self._y_cache[i]] for i in range(n)]))
        grad_logits *= 1.0 / n
        gW = [None] * len(self.W)
        gb = [None] * len(self.W)
        d = grad_logits
        # d enters as d/d(last layer output); for each layer i, its
        # own ReLU (layer i's post-activation = acts[i+1]) is applied
        # BEFORE layer i's gradients, then pushed through W[i].
        for i in range(len(self.W) - 1, -1, -1):
            if i < len(self.W) - 1:
                d = d * (acts[i + 1] > 0)
            gW[i] = (d.T @ acts[i]).astype(np.float32) + self.l2 * self.W[i]
            gb[i] = d.sum(axis=0).astype(np.float32)
            if i > 0:
                d = d @ self.W[i]
        for i in range(len(self.W)):
            m[i] = 0.9 * m[i] + 0.1 * gW[i]
            v[i] = 0.999 * v[i] + 0.998 * gW[i] ** 2
            mhat = m[i] / (1 - 0.9 ** t)
            vhat = v[i] / (1 - 0.999 ** t)
            self.W[i] -= lr * mhat / (np.sqrt(vhat) + 1e-8)
            mb[i] = 0.9 * mb[i] + 0.1 * gb[i]
            vb[i] = 0.999 * vb[i] + 0.998 * gb[i] ** 2
            mbhat = mb[i] / (1 - 0.9 ** t)
            vbhat = vb[i] / (1 - 0.999 ** t)
            self.b[i] -= lr * mbhat / (np.sqrt(vbhat) + 1e-8)
        return loss

    def set_batch_context(self, legal, legal_count, targets):
        self._legal_cache = (legal, legal_count)
        self._y_cache = targets


def batch_step(model, x, legal, legal_count, targets, lr, m, v, mb, vb,
               t) -> float:
    model.set_batch_context(legal, legal_count, targets)
    return model.step(x, lr, m, v, mb, vb, t)


def top1_at(model: MLP, feat_row: np.ndarray, legal_row: np.ndarray,
            k: int) -> tuple[int, float]:
    feat = feat_row if feat_row.ndim == 2 else feat_row[None, :]
    logits, _ = model.forward(feat)
    logits = logits[0].astype(np.float64)
    acts = legal_row[:k].astype(np.int64)
    x = logits[acts]
    m = x.max()
    p = np.exp(x - m) / np.exp(x - m).sum()
    j = int(p.argmax())
    return int(acts[j]), float(p[j])


def diagnose(model: MLP, val: dict, sweep_positions: int = 200,
             seed: int = 0) -> dict:
    x = features(val)
    legal, lc, y = val["legal"], val["legal_count"], val["action"]
    n = x.shape[0]
    # per-row top-1
    correct = 0
    top1_probs = np.zeros(n)
    for i in range(n):
        a, p = top1_at(model, x[i], legal[i], int(lc[i]))
        top1_probs[i] = p
        correct += (a == int(y[i]))
    # per 100-elo band over the mover's elo
    bands = {}
    for i in range(n):
        band = int(val["elo_self"][i]) // 100 * 100
        bands.setdefault(band, []).append(i)
    per_band = {}
    for band in sorted(bands):
        idx = bands[band]
        base = float(np.mean([1.0 / int(lc[i]) for i in idx]))
        per_band[f"{band}-{band + 99}"] = {
            "n": len(idx),
            "top1_accuracy": round(
                sum(top1_at(model, x[i], legal[i], int(lc[i]))[0]
                    == int(y[i]) for i in idx) / len(idx), 4),
            "mean_top1_prob": round(float(np.mean([
                top1_at(model, x[i], legal[i], int(lc[i]))[1]
                for i in idx])), 4),
            "baseline_top1_prob": round(base, 4),
        }
    # overall baselines
    baseline_acc = float(np.mean([1.0 / int(lc[i]) for i in range(n)]))

    # ---- the rating-conditioning sweep (the gate) ----
    rng = np.random.default_rng(seed)
    n_pos = min(sweep_positions, n)
    sel = rng.choice(n, size=n_pos, replace=False)
    elos = list(range(600, 3201, 100))
    top1_by_pos_elo = {}
    prob_by_pos_elo = {}
    for i in sel:
        row = x[i].copy()
        acts = []
        probs = []
        for e in elos:
            row[768 + 4] = e / 100.0  # elo_self
            a, p = top1_at(model, row[None, :], legal[i], int(lc[i]))
            acts.append(a)
            probs.append(p)
        top1_by_pos_elo[int(i)] = acts
        prob_by_pos_elo[int(i)] = probs
    flips = sum(
        len(set(v)) > 1 for v in top1_by_pos_elo.values())
    flips_extremes = sum(
        v[0] != v[-1] for v in top1_by_pos_elo.values())
    prob_low = float(np.mean([v[0] for v in prob_by_pos_elo.values()]))
    prob_high = float(np.mean([v[-1] for v in prob_by_pos_elo.values()]))
    return {
        "val_rows": n,
        "top1_accuracy": round(correct / n, 4),
        "baseline_top1_accuracy": round(baseline_acc, 4),
        "mean_top1_prob": round(float(np.mean(top1_probs)), 4),
        "per_band": per_band,
        "conditioning_sweep": {
            "positions": n_pos,
            "elo_values": elos,
            "note": ("opponent elo fixed at the position's own "
                     "elo_oppo; mover elo swept 600..3200 step 100; "
                     "flip = any change of predicted top-1 action "
                     "across the sweep (the v1 finding was 0/200)"),
            "positions_flipped_any": flips,
            "positions_flipped_extremes": flips_extremes,
            "mean_top1_prob_at_600": round(prob_low, 4),
            "mean_top1_prob_at_3200": round(prob_high, 4),
        },
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="dir from "
                   "pretrain_move_dataset.py")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--width", type=int, default=256,
                   help="hidden width (two layers)")
    p.add_argument("--seed", type=int, default=20260828)
    p.add_argument("--sweep-positions", type=int, default=200)
    p.add_argument("--report", type=Path)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--max-train-rows", type=int, default=0,
                   help="cap train rows (0 = all; for quick runs)")
    args = p.parse_args()

    data = Path(args.data)
    train = load_shard(data / "shard-train.npz")
    val = load_shard(data / "shard-val.npz")
    xt = features(train)
    xh = features(val)
    if args.max_train_rows:
        xt, xh = xt[:args.max_train_rows], xh[:args.max_train_rows]
        for k in train:
            if isinstance(train[k], np.ndarray):
                train[k] = train[k][:args.max_train_rows]
        for k in val:
            if isinstance(val[k], np.ndarray):
                val[k] = val[k][:args.max_train_rows]
    n = xt.shape[0]
    print(f"train {n} rows, val {xh.shape[0]} rows, hidden "
          f"{args.width}x2", flush=True)

    model = MLP([xt.shape[1], args.width, args.width, 20480],
                seed=args.seed)
    m = [np.zeros_like(w) for w in model.W]
    v = [np.zeros_like(w) for w in model.W]
    mb = [np.zeros_like(b) for b in model.b]
    vb = [np.zeros_like(b) for b in model.b]
    best = (float("inf"), None, None)
    patience_left = 5
    rng = np.random.default_rng(args.seed + 1)
    t0 = time.time()
    for epoch in range(args.epochs):
        perm = rng.permutation(n)
        ce_sum = 0.0
        nb = 0
        for s in range(0, n, args.batch):
            idx = perm[s:s + args.batch]
            ce = batch_step(model, xt[idx], train["legal"][idx],
                            train["legal_count"][idx], train["action"][idx],
                            args.lr, m, v, mb, vb,
                            epoch * 100 + s // args.batch + 1)
            ce_sum += ce
            nb += 1
        # val every epoch (small val set: cheap)
        val_ce = 0.0
        nv = xh.shape[0]
        for s in range(0, nv, args.batch):
            idx = slice(s, min(s + args.batch, nv))
            cnt = min(args.batch, nv - s)
            logits, _ = model.forward(xh[idx])
            lsm = log_softmax_masked(logits, val["legal"][idx],
                                     val["legal_count"][idx])
            val_ce += -sum(lsm[i, int(val["action"][s + i])]
                           for i in range(cnt))
        val_ce /= nv
        el = time.time() - t0
        print(f"epoch {epoch + 1}/{args.epochs} train_ce="
              f"{ce_sum / nb:.4f} val_ce={val_ce:.4f} "
              f"({el:.0f}s)", flush=True)
        if val_ce < best[0] - 1e-4:
            best = (val_ce, [w.copy() for w in model.W],
                    [b.copy() for b in model.b])
            patience_left = 5
        else:
            patience_left -= 1
        if patience_left <= 0:
            print("early stop", flush=True)
            break
    model.W, model.b = best[1], best[2]

    print("running diagnostics (per-band + conditioning sweep)...",
          flush=True)
    report = diagnose(model, val, sweep_positions=args.sweep_positions,
                      seed=args.seed)
    report.update({"tool": "tools/pretrain_move_predictor.py",
                   "data": str(data), "epochs_run": args.epochs,
                   "best_val_ce": round(best[0], 4),
                   "architecture": (f"mlp {xt.shape[1]}->{args.width}"
                                    f"->{args.width}->20480 (numpy, "
                                    f"legal-only softmax)"),
                   "elapsed_s": round(time.time() - t0, 1),
                   "note": ("sandbox probe of the move-prediction "
                            "pretrain objective; the full pretrain "
                            "runs the same objective on the mixed "
                            "1M-5M corpus via the v1 A100 pipeline")})
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
        print(f"report: {args.report}")
    if args.checkpoint:
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(args.checkpoint),
                            **{f"W{i}": w for i, w in enumerate(model.W)},
                            **{f"b{i}": b for i, b in enumerate(model.b)})
        print(f"checkpoint: {args.checkpoint}")
    print(json.dumps(report["conditioning_sweep"], indent=2))
    print(f"top1 accuracy {report['top1_accuracy']} vs baseline "
          f"{report['baseline_top1_accuracy']}; mean top1 prob "
          f"{report['mean_top1_prob']}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
