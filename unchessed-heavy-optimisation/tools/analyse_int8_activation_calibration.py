#!/usr/bin/env python3
"""Decide whether calibrated int8 *activations* can pass the 5e-3 parity gate.

Why
---

`docs/unarchitectured-metal-runtime-optimization.md` lists calibrated int8
activations as the first item of "Remaining performance work". The runtime
already retains int8 *weights* and quantizes activations to **int16**. Going
to int8 activations would let AVX2 use `maddubs`-style 8x8->16 products with
roughly twice the lanes per instruction, which is the single largest
remaining arithmetic win available without changing the model.

It has been tried once and rejected: per-token symmetric quantization missed
the frozen Python parity gate (start-position first logit off by about
`1.01e-2`, versus the required `5e-3`). Affine and small-group variants also
failed at least one component. The doc is explicit that a different
calibration scheme *might* close the gap but is **unproven**.

Writing AVX2 kernels to find out is the expensive way to answer that. The
arithmetic is decidable offline: quantization error is a property of the
weights and activations, not of the instruction selection. This tool
simulates each candidate scheme in float against the real exported
checkpoint and measures the resulting drift on the exact quantities the Rust
gates check.

Two questions are separated deliberately, because conflating them is what
makes "int8 activations failed" sound like a dead end when it isn't
necessarily one:

1. **Per-scheme error.** How much drift does each calibration scheme cause?
2. **Per-site sensitivity.** *Which* matmul sites contribute that drift? If
   the error is concentrated in a few sites, a mixed-precision split (int8
   where it is safe, int16 where it is not) may capture most of the speed
   at a fraction of the error -- a strategy no prior round evaluated.

What this measures
------------------

For each scheme, every activation feeding a quantizable matmul is passed
through a quantize/dequantize round trip that reproduces exactly what the
integer kernel would see, then the forward pass continues in float. The
result is compared against the unmodified reference forward on the same
checkpoint.

Schemes:

- `per_token_symmetric`  -- the already-rejected baseline, reproduced here so
  the tool is validated against a known answer rather than only producing
  new numbers.
- `per_tensor_static`    -- one scale per site, calibrated over the corpus.
- `per_channel_symmetric`-- one scale per input channel.
- `per_group_symmetric`  -- one scale per group of channels (`--group-size`).
- `percentile`           -- per-token, but clipping to a percentile of the
  magnitude distribution instead of the max, so a single outlier can't
  stretch the scale and crush every other value to a few levels.

Reported per scheme: max absolute logit drift, mean drift, whether the best
move is preserved, and the value-head components the Rust tests also check.
The pass/fail line is the project's real gate, `5e-3`, on logits.

Usage
-----

    python tools/analyse_int8_activation_calibration.py \
        --package artifacts/unarchitectured-metal-final.unmetal \
        --out benchmarks/unarchitectured-metal/int8-activation-calibration.json

Requires `torch` (same dependency as the reference forward it builds on).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

if any(a in ("-h", "--help") for a in sys.argv[1:]):
    print(__doc__)
    raise SystemExit(0)

PARITY_GATE = 5e-3
INT8_MAX = 127.0

SCHEMES = (
    "per_token_symmetric",
    "per_tensor_static",
    "per_channel_symmetric",
    "per_group_symmetric",
    "percentile",
)


def quantize_dequantize(x, scheme, group_size=32, percentile=99.9, static_scale=None):
    """Round-trip `x` through int8 exactly as the integer kernel would see it.

    Returns the dequantized tensor. Keeping this in float lets the rest of the
    forward pass run normally, so the measured drift is attributable to the
    quantization of this one site and nothing else.
    """
    import torch

    if scheme == "per_token_symmetric":
        scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / INT8_MAX
    elif scheme == "per_tensor_static":
        if static_scale is None:
            static_scale = x.abs().amax().clamp(min=1e-12) / INT8_MAX
        scale = static_scale
    elif scheme == "per_channel_symmetric":
        # One scale per input channel, shared across tokens. A channel whose
        # values are uniformly small keeps its resolution instead of being
        # crushed by a loud neighbouring channel.
        reduce_dims = tuple(range(x.dim() - 1))
        scale = x.abs().amax(dim=reduce_dims, keepdim=True).clamp(min=1e-12) / INT8_MAX
    elif scheme == "per_group_symmetric":
        *lead, channels = x.shape
        pad = (-channels) % group_size
        if pad:
            x_pad = torch.nn.functional.pad(x, (0, pad))
        else:
            x_pad = x
        grouped = x_pad.reshape(*lead, -1, group_size)
        gscale = grouped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / INT8_MAX
        q = torch.clamp(torch.round(grouped / gscale), -127, 127)
        out = (q * gscale).reshape(*lead, -1)
        return out[..., :channels]
    elif scheme == "percentile":
        flat = x.abs().reshape(*x.shape[:-1], -1)
        clip = torch.quantile(flat.float(), percentile / 100.0, dim=-1, keepdim=True)
        clip = clip.clamp(min=1e-12).to(x.dtype)
        scale = clip / INT8_MAX
        q = torch.clamp(torch.round(x / scale), -127, 127)
        return q * scale
    else:
        raise ValueError(f"unknown scheme {scheme}")

    q = torch.clamp(torch.round(x / scale), -127, 127)
    return q * scale


def patched_forward(w, batch, config, layers, width, scheme, sites, cost=None, **kw):
    """Run the reference forward with int8 activation simulation at `sites`.

    `sites` names which matmul inputs get quantized, so per-site sensitivity
    can be measured independently of the whole-model number. When `cost` is a
    dict it is filled with each site's multiply-accumulate count, so the
    speed value of a subset of sites can be weighed against its error.
    """
    import torch
    import torch.nn.functional as F

    import reference_forward_unarchitectured_metal as ref

    real_linear = F.linear
    counter = {"n": 0}

    def counting_linear(inp, weight, bias=None):
        counter["n"] += 1
        if cost is not None:
            macs = int(inp.numel()) * int(weight.shape[0])
            cost[counter["n"]] = macs
        if sites == "all" or counter["n"] in sites:
            inp = quantize_dequantize(inp, scheme, **kw)
        return real_linear(inp, weight, bias)

    F.linear = counting_linear
    try:
        out = ref.forward(w, batch, config, layers=layers, width=width)
    finally:
        F.linear = real_linear
    return out, counter["n"]


def start_position_batch():
    """The exact fixture the Rust parity test freezes."""
    import torch

    pieces = [0] * 64
    back = [4, 2, 3, 5, 6, 3, 2, 4]
    for f in range(8):
        pieces[f] = back[f]
        pieces[8 + f] = 1
        pieces[48 + f] = 7
        pieces[56 + f] = 6 + back[f]

    actions = []
    for f in range(8):
        src = 8 + f
        actions.append(src | ((src + 8) << 6))
        actions.append(src | ((src + 16) << 6))
    actions += [1 | (16 << 6), 1 | (18 << 6), 6 | (21 << 6), 6 | (23 << 6)]
    actions.sort()
    legal_count = len(actions)
    actions = actions + [0xFFFF] * (218 - legal_count)

    return {
        "pieces": torch.tensor([pieces], dtype=torch.long),
        "castling": torch.tensor([15], dtype=torch.long),
        "ep_file": torch.tensor([8], dtype=torch.long),
        "halfmove_bucket": torch.tensor([0], dtype=torch.long),
        "rating": torch.tensor([2700], dtype=torch.long),
        "time_class": torch.tensor([2], dtype=torch.long),
        "policy_kind": torch.tensor([1], dtype=torch.long),
        "safe_actions": torch.tensor([actions], dtype=torch.long),
        "legal_mask": torch.tensor([[i < legal_count for i in range(218)]]),
    }, legal_count


def midgame_batch():
    """The second frozen fixture: 1.e4 e5, mover = White."""
    import torch
    import chess

    from unarchitectured_metal_position_encoding import encode_position

    board = chess.Board()
    board.push_san("e4")
    board.push_san("e5")
    enc = encode_position(board)
    actions = list(enc["legal_actions"])
    n = len(actions)
    padded = actions + [0xFFFF] * (218 - n)
    return {
        "pieces": torch.tensor([enc["pieces"]], dtype=torch.long),
        "castling": torch.tensor([enc["castling"]], dtype=torch.long),
        "ep_file": torch.tensor([enc["ep_file"]], dtype=torch.long),
        "halfmove_bucket": torch.tensor([enc["halfmove_bucket"]], dtype=torch.long),
        "rating": torch.tensor([2700], dtype=torch.long),
        "time_class": torch.tensor([2], dtype=torch.long),
        "policy_kind": torch.tensor([1], dtype=torch.long),
        "safe_actions": torch.tensor([padded], dtype=torch.long),
        "legal_mask": torch.tensor([[i < n for i in range(218)]]),
    }, n


def compare(baseline, candidate, legal_count):
    """Drift on exactly the quantities the Rust gates check."""
    import torch

    base_logits = baseline["logits"][0][:legal_count]
    cand_logits = candidate["logits"][0][:legal_count]
    diff = (base_logits - cand_logits).abs()

    base_best = int(torch.argmax(base_logits))
    cand_best = int(torch.argmax(cand_logits))

    ev = (baseline["evidence"][0] - candidate["evidence"][0]).abs().max().item()
    rep = (
        (baseline["representation"][0] - candidate["representation"][0])
        .abs()
        .max()
        .item()
    )
    return {
        "max_logit_drift": diff.max().item(),
        "mean_logit_drift": diff.mean().item(),
        "max_evidence_drift": ev,
        "max_representation_drift": rep,
        "best_move_preserved": base_best == cand_best,
        "baseline_best_index": base_best,
        "candidate_best_index": cand_best,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Measure whether calibrated int8 activations can pass 5e-3."
    )
    parser.add_argument(
        "--package", default=str(ROOT / "artifacts" / "unarchitectured-metal-final.unmetal")
    )
    parser.add_argument("--out", default=None)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--percentile", type=float, default=99.9)
    parser.add_argument(
        "--per-site",
        action="store_true",
        help="Also measure each matmul site individually (slower).",
    )
    parser.add_argument(
        "--mixed",
        action="store_true",
        help="Search for the largest int8 subset that still passes the gate.",
    )
    parser.add_argument(
        "--validate",
        type=int,
        default=0,
        metavar="N",
        help="Re-check the chosen assignment on N unseen corpus positions.",
    )
    parser.add_argument(
        "--calibrate",
        type=int,
        default=0,
        metavar="N",
        help="Also calibrate the assignment against N corpus positions "
        "(taken from the end of the corpus, disjoint from --validate).",
    )
    args = parser.parse_args()

    import reference_forward_unarchitectured_metal as ref

    w = ref.read_package(args.package)
    config = {"d_model": 256, "heads": 8, "history_width": 32, "policy_adapter_rank": 16}

    fixtures = {"start_position": start_position_batch(), "midgame_e4e5": midgame_batch()}

    report = {
        "package": str(Path(args.package).name),
        "parity_gate": PARITY_GATE,
        "group_size": args.group_size,
        "percentile": args.percentile,
        "fixtures": {},
    }

    for fixture_name, (batch, legal_count) in fixtures.items():
        baseline = ref.forward(w, batch, config, layers=8, width=256)
        entry = {"legal_count": legal_count, "schemes": {}}

        for scheme in SCHEMES:
            kw = {}
            if scheme == "per_group_symmetric":
                kw["group_size"] = args.group_size
            if scheme == "percentile":
                kw["percentile"] = args.percentile
            cand, n_sites = patched_forward(
                w, batch, config, 8, 256, scheme, "all", **kw
            )
            result = compare(baseline, cand, legal_count)
            result["passes_gate"] = result["max_logit_drift"] <= PARITY_GATE
            result["matmul_sites"] = n_sites
            entry["schemes"][scheme] = result

        if args.mixed:
            # Per-site drift alone doesn't tell you whether a mixed split is
            # viable, because errors from separate sites accumulate through
            # the residual stream. Sort sites by measured individual drift,
            # then admit them to int8 cheapest-first, re-measuring the real
            # combined drift at each step. The last configuration that still
            # passes the gate is the honest answer for how much of the model
            # can use int8 activations at all.
            cost = {}
            _, total = patched_forward(
                w, batch, config, 8, 256, "per_channel_symmetric", set(), cost=cost
            )
            total_macs = sum(cost.values())
            singles = []
            for site in range(1, total + 1):
                cand, _ = patched_forward(
                    w, batch, config, 8, 256, "per_channel_symmetric", {site}
                )
                r = compare(baseline, cand, legal_count)
                singles.append((r["max_logit_drift"], site))
            singles.sort()

            admitted, steps, best, best_sites = set(), [], None, []
            for drift, site in singles:
                admitted.add(site)
                cand, _ = patched_forward(
                    w, batch, config, 8, 256, "per_channel_symmetric", set(admitted)
                )
                r = compare(baseline, cand, legal_count)
                covered = sum(cost[s] for s in admitted) / total_macs
                step = {
                    "sites_int8": len(admitted),
                    "mac_fraction_int8": covered,
                    "max_logit_drift": r["max_logit_drift"],
                    "passes_gate": r["max_logit_drift"] <= PARITY_GATE,
                    "best_move_preserved": r["best_move_preserved"],
                }
                steps.append(step)
                if step["passes_gate"]:
                    best = step
                    best_sites = sorted(admitted)
            entry["mixed_precision"] = {
                "total_sites": total,
                "steps": steps,
                "best_passing": best,
                "best_passing_sites": best_sites if best else [],
                "site_macs": {str(k): v for k, v in cost.items()},
                "total_macs": total_macs,
            }

        if args.per_site:
            _, total = patched_forward(
                w, batch, config, 8, 256, "per_token_symmetric", set()
            )
            per_site = {}
            for site in range(1, total + 1):
                cand, _ = patched_forward(
                    w, batch, config, 8, 256, "per_channel_symmetric", {site}
                )
                r = compare(baseline, cand, legal_count)
                per_site[str(site)] = {
                    "max_logit_drift": r["max_logit_drift"],
                    "best_move_preserved": r["best_move_preserved"],
                }
            entry["per_site_per_channel"] = per_site

        report["fixtures"][fixture_name] = entry

    if args.mixed:
        # A shipped kernel uses ONE site assignment for every position, so the
        # per-fixture winners above are not deployable on their own. Intersect
        # them and re-verify that the shared set still passes on every fixture
        # independently -- a set that passes each fixture only when tuned to it
        # is not a result, it's overfitting to two positions.
        sets = [
            set(e["mixed_precision"]["best_passing_sites"])
            for e in report["fixtures"].values()
        ]
        # Intersecting per-fixture winners is the obvious move and it does NOT
        # work: each set was chosen against its own fixture, so their overlap
        # can still exceed the gate on one of them. Do the search properly
        # instead -- admit sites cheapest-first against the WORST drift over
        # all fixtures at once, which is the quantity a shipped kernel must
        # actually satisfy.
        # Choosing the assignment against only the two frozen fixtures overfits
        # badly (measured: 80/150 unseen positions over gate). Calibrate against
        # a sample of real corpus positions as well, so the chosen set has to
        # survive the variety it will actually meet at runtime.
        calib = dict(fixtures)
        if args.calibrate:
            import chess as _chess
            import torch as _torch

            from unarchitectured_metal_position_encoding import (
                encode_position as _encode,
            )

            _corpus = ROOT / "artifacts" / "unarchitectured-metal-calibration-corpus.jsonl"
            _rows = []
            with _corpus.open() as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    if "fen" in rec:
                        _rows.append(rec)
            # Take calibration positions from the END of the corpus so the
            # holdout (which reads from the front) stays genuinely disjoint.
            for rec in _rows[-args.calibrate :]:
                board = _chess.Board(rec["fen"])
                enc = _encode(board)
                acts = list(enc["legal_actions"])
                n = len(acts)
                if n == 0:
                    continue
                padded = acts + [0xFFFF] * (218 - n)
                calib[f"corpus:{rec['fen']}"] = (
                    {
                        "pieces": _torch.tensor([enc["pieces"]], dtype=_torch.long),
                        "castling": _torch.tensor([enc["castling"]], dtype=_torch.long),
                        "ep_file": _torch.tensor([enc["ep_file"]], dtype=_torch.long),
                        "halfmove_bucket": _torch.tensor(
                            [enc["halfmove_bucket"]], dtype=_torch.long
                        ),
                        "rating": _torch.tensor([2700], dtype=_torch.long),
                        "time_class": _torch.tensor([2], dtype=_torch.long),
                        "policy_kind": _torch.tensor([1], dtype=_torch.long),
                        "safe_actions": _torch.tensor([padded], dtype=_torch.long),
                        "legal_mask": _torch.tensor([[i < n for i in range(218)]]),
                    },
                    n,
                )

        baselines = {
            name: ref.forward(w, b, config, layers=8, width=256)
            for name, (b, _) in calib.items()
        }

        def worst_drift(site_set):
            worst = 0.0
            ok = True
            for name, (b, lc) in calib.items():
                cand, _ = patched_forward(
                    w, b, config, 8, 256, "per_channel_symmetric", set(site_set)
                )
                r = compare(baselines[name], cand, lc)
                worst = max(worst, r["max_logit_drift"])
                ok = ok and r["best_move_preserved"]
            return worst, ok

        order = []
        for site in range(1, 51):
            d, _ = worst_drift({site})
            order.append((d, site))
        order.sort()

        admitted, shared, joint_steps = set(), [], []
        for _, site in order:
            trial = admitted | {site}
            d, ok = worst_drift(trial)
            if d <= PARITY_GATE:
                admitted = trial
                shared = sorted(admitted)
                joint_steps.append(
                    {"sites_int8": len(shared), "worst_drift": d, "best_moves_kept": ok}
                )
        report["joint_greedy_steps"] = joint_steps

        joint = {}
        for fixture_name, (batch, legal_count) in fixtures.items():
            baseline = ref.forward(w, batch, config, layers=8, width=256)
            cand, _ = patched_forward(
                w, batch, config, 8, 256, "per_channel_symmetric", set(shared)
            )
            r = compare(baseline, cand, legal_count)
            r["passes_gate"] = r["max_logit_drift"] <= PARITY_GATE
            joint[fixture_name] = r
        cost = {
            int(k): v
            for k, v in list(report["fixtures"].values())[0]["mixed_precision"][
                "site_macs"
            ].items()
        }
        total_macs = list(report["fixtures"].values())[0]["mixed_precision"][
            "total_macs"
        ]
        report["shared_assignment"] = {
            "sites_int8": shared,
            "site_count": len(shared),
            "mac_fraction_int8": sum(cost[s] for s in shared) / total_macs,
            "per_fixture": joint,
            "passes_all_fixtures": all(v["passes_gate"] for v in joint.values()),
        }

    if args.mixed and args.validate:
        # The assignment above was chosen against two positions. Two positions
        # cannot establish that it holds in general, and an assignment tuned on
        # its own test set is worthless. Re-measure the frozen set on unseen
        # corpus positions and report the worst case -- if the gate fails here,
        # the assignment is overfit and the finding is negative.
        import chess

        from unarchitectured_metal_position_encoding import encode_position

        corpus = ROOT / "artifacts" / "unarchitectured-metal-calibration-corpus.jsonl"
        shared = set(report["shared_assignment"]["sites_int8"])
        rows = []
        with corpus.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                # Line 1 is a provenance manifest, not a position.
                if "fen" in rec:
                    rows.append(rec)

        import torch

        worst = 0.0
        changed = 0
        checked = 0
        failures = []
        for rec in rows[: args.validate]:
            board = chess.Board(rec["fen"])
            enc = encode_position(board)
            acts = list(enc["legal_actions"])
            n = len(acts)
            if n == 0:
                continue
            padded = acts + [0xFFFF] * (218 - n)
            b = {
                "pieces": torch.tensor([enc["pieces"]], dtype=torch.long),
                "castling": torch.tensor([enc["castling"]], dtype=torch.long),
                "ep_file": torch.tensor([enc["ep_file"]], dtype=torch.long),
                "halfmove_bucket": torch.tensor(
                    [enc["halfmove_bucket"]], dtype=torch.long
                ),
                "rating": torch.tensor([2700], dtype=torch.long),
                "time_class": torch.tensor([2], dtype=torch.long),
                "policy_kind": torch.tensor([1], dtype=torch.long),
                "safe_actions": torch.tensor([padded], dtype=torch.long),
                "legal_mask": torch.tensor([[i < n for i in range(218)]]),
            }
            base = ref.forward(w, b, config, layers=8, width=256)
            cand, _ = patched_forward(
                w, b, config, 8, 256, "per_channel_symmetric", shared
            )
            r = compare(base, cand, n)
            checked += 1
            if r["max_logit_drift"] > worst:
                worst = r["max_logit_drift"]
            if not r["best_move_preserved"]:
                changed += 1
            if r["max_logit_drift"] > PARITY_GATE:
                failures.append(
                    {"fen": rec["fen"], "max_logit_drift": r["max_logit_drift"]}
                )
        report["holdout_validation"] = {
            "positions_checked": checked,
            "worst_max_logit_drift": worst,
            "positions_over_gate": len(failures),
            "best_move_changes": changed,
            "generalizes": len(failures) == 0,
            "worst_failures": sorted(
                failures, key=lambda f: -f["max_logit_drift"]
            )[:5],
        }

    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n")
        print(f"wrote {out}")

    for fixture_name, entry in report["fixtures"].items():
        print(f"\n=== {fixture_name} (legal_count={entry['legal_count']}) ===")
        print(f"{'scheme':<24} {'max drift':>12} {'gate':>7} {'best move':>10}")
        for scheme, r in entry["schemes"].items():
            print(
                f"{scheme:<24} {r['max_logit_drift']:>12.3e} "
                f"{'PASS' if r['passes_gate'] else 'FAIL':>7} "
                f"{'kept' if r['best_move_preserved'] else 'CHANGED':>10}"
            )

    sa = report.get("shared_assignment")
    if sa:
        print(
            f"\n=== shared int8 assignment ({sa['site_count']} of 50 sites, "
            f"{sa['mac_fraction_int8']*100:.1f}% of MACs) ==="
        )
        for fixture_name, r in sa["per_fixture"].items():
            print(
                f"{fixture_name:<24} {r['max_logit_drift']:>12.3e} "
                f"{'PASS' if r['passes_gate'] else 'FAIL':>7} "
                f"{'kept' if r['best_move_preserved'] else 'CHANGED':>10}"
            )
        print(f"deployable: {sa['passes_all_fixtures']}")

    hv = report.get("holdout_validation")
    if hv:
        print(
            f"\n=== holdout ({hv['positions_checked']} unseen corpus positions) ==="
        )
        print(f"worst drift        {hv['worst_max_logit_drift']:.3e}")
        print(f"positions >  5e-3  {hv['positions_over_gate']}")
        print(f"best-move changes  {hv['best_move_changes']}")
        print(f"generalizes:       {hv['generalizes']}")


if __name__ == "__main__":
    main()
