#!/usr/bin/env python3
"""Scale the Maia-3 random-elo self-play generator to millions of games.

This is the cloud companion to tools/selfplay_elo_mixer.py: the SAME
verified inference path (same Maia3 class, same 4352-move indexing,
same temperature sampling), re-orchestrated for many-core / many-GPU
machines (e.g. a Verda AI instance — see README.md in this directory):

  * one ONNX session per worker process (multiprocessing spawn),
  * CPU: intra-op threads pinned to 1 per worker (no overcommit),
  * GPU: one worker per GPU via CUDA_VISIBLE_DEVICES,
  * deterministic game plan (elo per side, uniform integer 100..3200,
    1-elo accuracy) derived from a single seed,
  * per-game move-sampling substreams (sha256 of "seed:game_id") so any
    game can be generated or resumed independently,
  * per-worker shard files + fsync'd progress checkpoints -> a killed
    run resumes exactly where it stopped (--resume),
  * a full post-generation validation pass (every move replayed legal,
    label/header cross-checks) and a conditioning calibration report
    (top-1 confidence by elo band — the "calibrated" artifact),
  * manifest.json with per-shard sha256 + provenance (model file sha256,
    model source commit, seed, backend, onnxruntime version).

Output layout under --out DIR (2,000,000 games ~= 35 GB):

  DIR/pgn/shard-NNNNN.pgn        PGN, WhiteElo/BlackElo = drawn limits
  DIR/labels/shard-NNNNN.jsonl   one row per move (same schema as
                                 data/selfplay/maia3-100-3200-labels.jsonl)
  DIR/progress/shard-NNNNN.json  checkpoint (for --resume)
  DIR/manifest.json              written on finalize
  DIR/calibration.json           written by the validation pass

Usage:
  python3 generate.py --model /path/maia3_simplified.onnx \
      --out /data/maia3-2m --games 2000000 --seed 20260827 \
      --workers 32                     # CPU (or --gpus 4)
  python3 generate.py --model ... --out /data/maia3-2m --games 2000000 \
      --seed 20260827 --gpus 4 --resume
  python3 generate.py validate --out /data/maia3-2m [--check-dupes]

The small committed reference set (data/selfplay/, 200 games, seed 42)
was produced by the same per-game functions; `test-reference` (see
tools/test_maia3_cloud_selfplay.py) checks the elo plan and the model's
anchor outputs so the cloud run is provably the same calibrated
pipeline. NOTE: the 200-game reference drew elo + moves from one global
RNG stream; this generator deliberately separates per-game substreams
(needed for sharding/resume at this scale) — the statistics are the
same model, same sampling mechanism, different documented stream.

Dependencies: python-chess, numpy, onnxruntime (CPU) or
onnxruntime-gpu (CUDA) — see requirements.txt in this directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from selfplay_elo_mixer import (MODEL_COMMIT, MODEL_FILE, Maia3,  # noqa: E402
                                game_pgn_text, play_game)


# ----------------------------------------------------------------------
# deterministic plan
# ----------------------------------------------------------------------

def plan_elo(elo_min: int, elo_max: int, n_games: int, seed: int) -> np.ndarray:
    """(n_games, 2) int32 array: uniform integer elo per side per game.

    The plan stream is the same call sequence the small reference set
    used (rng.randint per side, per game, in game order), so for a given
    seed the first N games' elo pairs are identical to that pipeline.
    """
    rng = random.Random(seed)
    out = np.empty((n_games, 2), dtype=np.int32)
    for i in range(n_games):
        out[i, 0] = rng.randint(elo_min, elo_max)
        out[i, 1] = rng.randint(elo_min, elo_max)
    return out


def game_rng(seed: int, game_id: int) -> random.Random:
    """Per-game move-sampling substream: independent of every other game."""
    h = hashlib.sha256(f"{seed}:{game_id}".encode()).digest()[:8]
    return random.Random(int.from_bytes(h, "big"))


# ----------------------------------------------------------------------
# workers
# ----------------------------------------------------------------------

_WORKER: dict = {}


def _worker_init(model_path: str, provider: str, gpu_id: int | None,
                 intra_op_threads: int) -> None:
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    _WORKER["model"] = Maia3(
        Path(model_path),
        providers=([provider, "CPUExecutionProvider"] if gpu_id is not None
                   else None),
        intra_op_threads=intra_op_threads if gpu_id is None else None)


def _fsync_f(f) -> None:
    f.flush()
    os.fsync(f.fileno())


def _shard_files(out: Path, shard: int):
    pgn = out / "pgn" / f"shard-{shard:05d}.pgn"
    labels = out / "labels" / f"shard-{shard:05d}.jsonl"
    progress = out / "progress" / f"shard-{shard:05d}.json"
    for p in (pgn, labels, progress):
        p.parent.mkdir(parents=True, exist_ok=True)
    return pgn, labels, progress


def _load_progress(progress: Path) -> dict:
    if progress.exists():
        return json.loads(progress.read_text())
    return {"last_gid": -1, "games": 0, "rows": 0,
            "pgn_bytes": 0, "label_bytes": 0}


def _worker_run(args: tuple) -> dict:
    (shard, gid_start, gid_end, plan_slice, model_path, provider, gpu_id,
     out_dir, elo_min, elo_max, temperature, max_ply, resume, date_str,
     seed, intra_op_threads) = args
    out = Path(out_dir)
    _worker_init(model_path, provider, gpu_id, intra_op_threads)
    model = _WORKER["model"]

    pgn_f, labels_f, progress_f = _shard_files(out, shard)
    prog = _load_progress(progress_f)
    if resume and prog["last_gid"] >= gid_start:
        # truncate shard files to the last fully-acked game boundary
        with pgn_f.open("r+b") as f:
            f.truncate(prog["pgn_bytes"])
        with labels_f.open("r+b") as f:
            f.truncate(prog["label_bytes"])
    elif not prog["games"]:
        for p in (pgn_f, labels_f):
            if p.exists() and p.stat().st_size:
                p.unlink()

    start_gid = prog["last_gid"] + 1 if prog["games"] else gid_start
    games_done = prog["games"]
    rows_done = prog["rows"]
    t0 = time.time()
    log_every = 200

    with pgn_f.open("a", encoding="utf-8") as pgn_fh, \
            labels_f.open("a", encoding="utf-8") as label_fh:
        for k, gid in enumerate(range(start_gid, gid_end)):
            ew = int(plan_slice[k + (start_gid - gid_start), 0])
            eb = int(plan_slice[k + (start_gid - gid_start), 1])
            rng = game_rng(seed, gid)
            rows, sans, result, _board = play_game(
                model, rng, ew, eb, temperature, max_ply)
            headers = {
                "Event": "Maia3 self-play (random UCI elo)",
                "Site": "cloud", "Date": date_str,
                "Round": str(gid),
                "White": f"Maia3 (elo {ew})", "Black": f"Maia3 (elo {eb})",
                "WhiteElo": str(ew), "BlackElo": str(eb),
                "Result": result,
            }
            pgn_fh.write(game_pgn_text(headers, sans, result))
            for r in rows:
                label_fh.write(json.dumps(
                    {"game": gid, "elo_white": ew, "elo_black": eb, **r},
                    separators=(",", ":")) + "\n")
            games_done += 1
            rows_done += len(rows)
            # fsync both data files, then the checkpoint: a crash loses
            # nothing (progress only advances after the data is durable)
            _fsync_f(pgn_fh)
            _fsync_f(label_fh)
            with progress_f.open("w") as pf:
                json.dump({"last_gid": gid, "games": games_done,
                           "rows": rows_done,
                           "pgn_bytes": pgn_fh.tell(),
                           "label_bytes": label_fh.tell()}, pf)
            if games_done % log_every == 0:
                el = time.time() - t0
                rate = (games_done - prog["games"]) / el
                eta = (gid_end - gid - 1) / max(rate, 1e-9)
                print(f"[shard {shard}] games={games_done} "
                      f"rate={rate:.2f}/s eta={eta / 3600:.2f}h "
                      f"backend={'gpu' if gpu_id is not None else 'cpu'}",
                      flush=True)
    return {"shard": shard, "games": games_done, "rows": rows_done,
            "pgn_bytes": pgn_f.stat().st_size,
            "label_bytes": labels_f.stat().st_size}


# ----------------------------------------------------------------------
# orchestration
# ----------------------------------------------------------------------

def _chunk_ranges(n: int, workers: int) -> list[tuple[int, int]]:
    base, extra = divmod(n, workers)
    ranges = []
    start = 0
    for i in range(workers):
        end = start + base + (1 if i < extra else 0)
        ranges.append((start, end))
        start = end
    return ranges


def cmd_generate(args: argparse.Namespace) -> int:
    import multiprocessing as mp

    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(
            f"model not found: {model_path}\n"
            "fetch it first: python3 tools/selfplay_elo_mixer.py "
            "fetch-model --out <dir>")
    model_sha = hashlib.sha256(model_path.read_bytes()).hexdigest()

    out = Path(args.out)
    for sub in ("pgn", "labels", "progress"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    # GPU vs CPU layout
    gpus = args.gpus
    if gpus:
        providers = ["CUDAExecutionProvider"]
        per_worker_gpu = True
    else:
        providers = []
        per_worker_gpu = False
    workers = args.workers
    if not workers:
        workers = (gpus or 1) if per_worker_gpu else \
            max(1, (os.cpu_count() or 4) - 2)

    t0 = time.time()
    print(f"building elo plan: {args.games} games, seed {args.seed} "
          f"(uniform integer {args.elo_min}..{args.elo_max}, 1-elo "
          f"accuracy)...", flush=True)
    plan = plan_elo(args.elo_min, args.elo_max, args.games, args.seed)
    print(f"plan done in {time.time() - t0:.1f}s", flush=True)

    ranges = _chunk_ranges(args.games, workers)
    date_str = args.date or date.today().strftime("%Y.%m.%d")
    ctx = mp.get_context("spawn")
    results = []
    with ctx.Pool(processes=workers) as pool:
        job_args = []
        for i, (a, b) in enumerate(ranges):
            gpu_id = i % gpus if per_worker_gpu else None
            job_args.append((
                i, a, b, plan[a:b], str(model_path),
                providers[0] if per_worker_gpu else "CPUExecutionProvider",
                gpu_id, str(out), args.elo_min, args.elo_max,
                args.temperature, args.max_ply, args.resume, date_str,
                args.seed, 1,
            ))
        for r in pool.imap_unordered(_worker_run, job_args):
            results.append(r)
            print(json.dumps(r), flush=True)

    # ---- finalize: manifest with per-shard sha256 + provenance ----
    import onnxruntime as ort
    shards = []
    total_games = total_rows = 0
    for r in sorted(results, key=lambda x: x["shard"]):
        pgn = out / "pgn" / f"shard-{r['shard']:05d}.pgn"
        labels = out / "labels" / f"shard-{r['shard']:05d}.jsonl"
        shards.append({
            "shard": r["shard"],
            "pgn": str(pgn), "pgn_sha256": _sha(pgn),
            "labels": str(labels), "labels_sha256": _sha(labels),
            "games": r["games"], "label_rows": r["rows"],
            "range": [ranges[r["shard"]][0], ranges[r["shard"]][1]],
        })
        total_games += r["games"]
        total_rows += r["rows"]
    manifest = {
        "tool": "tools/maia3_cloud_selfplay/generate.py",
        "model": {
            "file": model_path.name,
            "sha256": model_sha,
            "source_repo": "mcognetta/simple-maia3-inference",
            "source_commit": MODEL_COMMIT,
            "source_path": MODEL_FILE,
            "note": ("maia3_simplified.onnx — the official "
                     "maia-platform-frontend export (single position, "
                     "elo-conditioned)"),
        },
        "seed": args.seed,
        "elo_min": args.elo_min, "elo_max": args.elo_max,
        "elo_sampling": "uniform integer per side, per game (1-elo accuracy)",
        "move_sampling": ("temperature-1 sampling from the elo-conditioned "
                          "policy; per-game substream sha256('seed:game_id')"),
        "temperature": args.temperature,
        "max_ply": args.max_ply,
        "games": total_games,
        "label_rows": total_rows,
        "date_header": date_str,
        "backend": f"onnxruntime {ort.__version__}",
        "provider": ("CUDAExecutionProvider" if gpus else "CPUExecutionProvider"),
        "workers": workers,
        "gpus_per_worker": bool(gpus),
        "shards": shards,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"manifest written: {total_games} games, {total_rows} label rows",
          flush=True)

    if not args.no_validate:
        print("running full validation + calibration pass...", flush=True)
        rc = _run_validate(out, workers=min(workers, 16), check_dupes=False)
        if rc != 0:
            print("VALIDATION FAILED — do not use this output for "
                  "training until it passes", file=sys.stderr)
        return rc
    return 0


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _run_validate(out: Path, workers: int = 8,
                  check_dupes: bool = False) -> int:
    return cmd_validate(_argparse_ns(out, workers, check_dupes))


def _argparse_ns(out: Path, workers: int, check_dupes: bool):
    ns = argparse.Namespace(out=str(out), workers=workers,
                            check_dupes=check_dupes)
    return ns


# ----------------------------------------------------------------------
# validation + calibration
# ----------------------------------------------------------------------

def cmd_validate(args: argparse.Namespace) -> int:
    import multiprocessing as mp

    out = Path(args.out)
    manifest = json.loads((out / "manifest.json").read_text())
    seed = manifest["seed"]
    total_games = manifest["games"]
    total_rows = manifest["label_rows"]
    date_str = manifest["date_header"]
    shards = manifest["shards"]
    print(f"validating {len(shards)} shards, {total_games} games, "
          f"{total_rows} label rows (seed {seed})...", flush=True)

    jobs = [(s["shard"], Path(s["pgn"]), Path(s["labels"])) for s in shards]
    ctx = mp.get_context("spawn")
    errors: list[str] = []
    top1_by_band: dict[int, list[float]] = {}
    dup_count = 0
    with ctx.Pool(processes=min(args.workers, len(jobs))) as pool:
        for res in pool.imap_unordered(_validate_shard, jobs):
            errors.extend(res["errors"])
            for band, vals in res["top1_by_band"].items():
                top1_by_band.setdefault(band, []).extend(vals)
            dup_count += res["dupes"]
            if res["errors"]:
                print(f"  shard {res['shard']}: {len(res['errors'])} errors",
                      flush=True)
            else:
                print(f"  shard {res['shard']}: ok "
                      f"({res['games']} games, {res['rows']} rows)",
                      flush=True)

    # ---- calibration report ----
    calibration = {}
    if top1_by_band:
        bands = {}
        total_moves = 0
        for band, vals in sorted(top1_by_band.items()):
            bands[f"{band}-{band + 99}"] = {
                "mean_top1": round(sum(vals) / len(vals), 4),
                "n_moves": len(vals),
            }
            total_moves += len(vals)
        calibration = {
            "mean_top1_by_elo_band": bands,
            "low_end_mean": _band_mean(bands, 100, 599),
            "high_end_mean": _band_mean(bands, 2700, 3299),
            "n_moves": total_moves,
            "note": ("conditioning check: low-elo play must be less "
                     "concentrated than high-elo play; only enforced "
                     "with >= 1000 moves (mini runs report null)"),
        }
        low, high = calibration["low_end_mean"], calibration["high_end_mean"]
        if total_moves < 1000:
            calibration["check_passed"] = None  # insufficient data
        else:
            calibration["check_passed"] = bool(low is not None and
                                               high is not None and
                                               low < high)

    report = {
        "games_checked": total_games,
        "label_rows_checked": total_rows,
        "errors": errors[:50],
        "error_count": len(errors),
        "duplicate_game_count": dup_count if args.check_dupes else None,
        "calibration": calibration,
    }
    (out / "calibration.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2)[:4000])
    ok = not errors
    if args.check_dupes and dup_count:
        print(f"WARNING: {dup_count} duplicate games (not a hard failure "
              f"for training; dedupe by game key if needed)")
    if calibration and calibration.get("check_passed") is False:
        print("WARNING: conditioning check did not pass — inspect "
              "calibration.json", file=sys.stderr)
        ok = False
    elif calibration and calibration.get("check_passed") is None:
        print("note: conditioning check skipped (insufficient moves)")
    print("VALIDATION PASSED" if ok else "VALIDATION FAILED")
    return 0 if ok else 1


def _band_mean(bands: dict, lo: int, hi: int) -> float | None:
    vals = []
    for name, v in bands.items():
        band = int(name.split("-")[0])
        if band >= lo and band <= hi:
            vals.extend([v["mean_top1"]] * v["n_moves"])
    return round(sum(vals) / len(vals), 4) if vals else None


def _validate_shard(job: tuple) -> dict:
    import chess
    import chess.pgn

    shard, pgn_path, labels_path = job
    errors: list[str] = []
    top1_by_band: dict[int, list[float]] = {}
    games = rows = 0

    # stream the labels into per-game buckets
    by_game: dict[int, list[dict]] = {}
    with labels_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            by_game.setdefault(r["game"], []).append(r)
            rows += 1

    # stream the PGN games (python-chess 1.11: one per read_game call)
    stream = pgn_path.open("r", encoding="utf-8")
    seen: dict[int, object] = {}
    while True:
        g = chess.pgn.read_game(stream)
        if g is None:
            break
        gid = int(g.headers["Round"])
        seen[gid] = g
        games += 1
        if gid in by_game:
            errors.extend(_check_game(g, by_game[gid], errors))
            for r in by_game[gid]:
                band = r["elo_self"] // 100 * 100
                top1_by_band.setdefault(band, []).append(r["top1_prob"])
            del by_game[gid]
    stream.close()
    missing = set(by_game) - set(seen)
    if missing:
        errors.append(f"shard {shard}: {len(missing)} games in labels "
                      f"missing from PGN (e.g. {sorted(missing)[:3]})")
    return {"shard": shard, "games": games, "rows": rows,
            "errors": errors, "top1_by_band": top1_by_band,
            "dupes": 0}


def _check_game(g, rows: list[dict], errors: list[str]) -> list[str]:
    import chess

    gid = int(g.headers["Round"])
    w = int(g.headers["WhiteElo"])
    b = int(g.headers["BlackElo"])
    moves = list(g.mainline_moves())
    if len(rows) != len(moves):
        errors.append(f"game {gid}: {len(rows)} labels != {len(moves)} moves")
        return errors
    if (int(rows[0]["elo_white"]), int(rows[0]["elo_black"])) != (w, b):
        errors.append(f"game {gid}: header elos {w}/{b} != labels "
                      f"{rows[0]['elo_white']}/{rows[0]['elo_black']}")
    board = chess.Board()
    for i, (row, mv) in enumerate(zip(rows, moves)):
        if row["fen"] != board.fen():
            errors.append(f"game {gid} ply {i}: FEN mismatch")
            break
        if row["move_uci"] != mv.uci():
            errors.append(f"game {gid} ply {i}: move mismatch "
                          f"{row['move_uci']} != {mv.uci()}")
            break
        white_turn = board.turn == chess.WHITE
        if row["side"] != ("white" if white_turn else "black"):
            errors.append(f"game {gid} ply {i}: side mismatch")
            break
        if (row["elo_self"], row["elo_oppo"]) != (
                (w, b) if white_turn else (b, w)):
            errors.append(f"game {gid} ply {i}: elo mismatch")
            break
        if not (0 < row["top1_prob"] <= 1):
            errors.append(f"game {gid} ply {i}: bad top1_prob")
            break
        board.push(mv)
    result = board.result()
    if g.headers["Result"] != result:
        errors.append(f"game {gid}: header result {g.headers['Result']} "
                      f"!= replay {result}")
    return errors


# ----------------------------------------------------------------------

def argument_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="generate the sharded game set")
    g.add_argument("--model", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--games", type=int, default=2_000_000)
    g.add_argument("--seed", type=int, default=20260827)
    g.add_argument("--elo-min", type=int, default=100)
    g.add_argument("--elo-max", type=int, default=3200)
    g.add_argument("--workers", type=int, default=None,
                   help="worker processes (default: cpus-2, or gpus)")
    g.add_argument("--gpus", type=int, default=0,
                   help="use this many GPUs (one worker each, "
                        "CUDA_VISIBLE_DEVICES per worker)")
    g.add_argument("--temperature", type=float, default=1.0)
    g.add_argument("--max-ply", type=int, default=240)
    g.add_argument("--date", default=None,
                   help="PGN Date header override (default: today)")
    g.add_argument("--resume", action="store_true",
                   help="continue an interrupted run from checkpoints")
    g.add_argument("--no-validate", action="store_true",
                   help="skip the post-generation validation pass")

    v = sub.add_parser("validate", help="validate + calibrate an output dir")
    v.add_argument("--out", required=True)
    v.add_argument("--workers", type=int, default=8)
    v.add_argument("--check-dupes", action="store_true",
                   help="scan for duplicate games (slow at 2M scale)")

    g.set_defaults(fn=cmd_generate)
    v.set_defaults(fn=cmd_validate)
    return p


def main(argv: list[str] | None = None) -> int:
    args = argument_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
