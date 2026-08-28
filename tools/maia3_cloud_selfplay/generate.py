#!/usr/bin/env python3
"""Mixed-engine random-elo self-play generator for many-core cloud machines.

The cloud companion to tools/selfplay_elo_mixer.py, scaled to millions of
games and extended from Maia-3-only to a pool of four engines, each with
the strength mechanism it actually supports (verified from source/binary):

  maia3      in-process ONNX, native UCI_Elo conditioning (100..3200).
             The elo conditioning is the model's real one — the calibrated
             gradient measured in data/selfplay/ (0.32 -> 0.62 top-1).
  stockfish  UCI subprocess, native UCI_Elo 1320..3190 (Stockfish 18,
             verified in sf/src/search.h and live `uci` output).
  lc0        UCI subprocess. LC0 has NO UCI_Elo option (verified in
             v0.32.1 source), so its strength ladder is the thinking
             budget (movetime), monotone in elo, approximate in value.
  rubichess  UCI subprocess. RubiChess has NO UCI_Elo option (verified in
             its option list), so its strength ladder is LimitNps (the
             engine's own NPS cap), monotone in elo, approximate in value.

Games are drawn from the pool: each side independently gets (engine,
elo) — engine uniform over the pool, elo uniform integer inside that
engine's supported range (intersected with --elo-min/--elo-max). Games
played by engines without a native elo limit carry
EloQuality "approximate" in PGN headers and label rows so downstream
level-conditioned training can filter or down-weight them; maia3 games
are "calibrated" and stockfish games "native".

Scale-out design (the heavily-optimized parts):
  * resident per-worker engine pool: every UCI engine process is spawned
    ONCE at worker init and reused for all games in the shard (no
    per-game spawn), maia3 keeps one onnxruntime session per worker,
  * setoption caching: UCI_Elo / LimitNps are re-sent only when they
    change for that engine instance,
  * one reader thread per engine process (line queue) — the worker
    never deadlocks on interleaved I/O,
  * buffered PGN/label writes with fsync every --fsync-every games
    (default 10) + a byte-accurate checkpoint: a crash loses at most
    that many games and --resume truncates exactly to the last
    fsync'd game boundary,
  * deterministic game plan (seeded, per side: engine then elo) and
    per-game move-sampling substreams (sha256 of "seed:game_id"), so
    any game can be regenerated or resumed independently,
  * a full post-generation validation pass (every move replayed legal,
    headers/labels/engine cross-checks) + a conditioning calibration
    report over the maia3 rows,
  * manifest.json with per-shard sha256 + full engine provenance
    (pinned git refs, binary/net sha256, engine id strings).

Output layout under --out DIR (5,000,000 games ~= 85 GB):

  DIR/pgn/shard-NNNNN.pgn        PGN, WhiteElo/BlackElo/WhiteEngine/...
  DIR/labels/shard-NNNNN.jsonl   one row per move (top1_prob/ldw are
                                 null for non-maia3 moves)
  DIR/progress/shard-NNNNN.json  checkpoint (for --resume)
  DIR/manifest.json              written on finalize
  DIR/calibration.json           written by the validation pass

Usage (Verda AI CPU.180V.720G — see README.md in this directory):
  python3 generate.py fetch-engines --engines-dir /data/engines
  python3 generate.py generate --model /path/maia3_simplified.onnx \
      --engines-dir /data/engines --out /data/mixed-5m --games 5000000 \
      --seed 20260827 --workers 90
  python3 generate.py validate --out /data/mixed-5m [--check-dupes]

Backward compatibility: with a single-engine pool (--engines maia3) the
elo plan stream is byte-identical to the original 2M generator's
plan_elo stream, so that configuration reproduces the same game plan
for a given seed.

Dependencies: python-chess, numpy, onnxruntime (CPU) or
onnxruntime-gpu (CUDA) — see requirements.txt in this directory.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import queue
import random
import subprocess
import sys
import threading
import time
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from selfplay_elo_mixer import (MODEL_COMMIT, MODEL_FILE, Maia3,  # noqa: E402
                                game_pgn_text)


# ----------------------------------------------------------------------
# pinned engine provenance (verified 2026-08-28: git clones, builds,
# live UCI handshakes where the net host allowed it)
# ----------------------------------------------------------------------

PINNED_ENGINES = {
    "stockfish": {
        "repo": "https://github.com/official-stockfish/Stockfish",
        "ref": "cb3d4ee9b47d0c5aae855b12379378ea1439675c",  # tag sf_18
        "binary_rel": "src/stockfish",
        "build_rel": "src",
        "build_cmd": "make -j{jobs} build ARCH={arch} COMP=gcc COMPCXX=g++",
        "net_name": "nn-c288c895ea92.nnue",
        "net_url": ("https://tests.stockfishchess.org/api/nn/"
                    "nn-c288c895ea92.nnue"),
        "note": ("Stockfish 18 is NNUE-only; the net is not committed to "
                 "the source repo (0-byte placeholders), fetch it from "
                 "the official tests host at setup time"),
    },
    "rubichess": {
        "repo": "https://github.com/Matthies/RubiChess",
        "ref": "29ddf13d3862fb3b7c8182c2e8f44b9c7fa577e9",  # master 2026-08-27
        "binary_rel": "src/RubiChess",
        "build_rel": "src",
        "build_cmd": "make -j{jobs} compile EVALFILE=",
        "nnue_repo": "https://github.com/Matthies/NN",
        "nnue_ref": "e0b3b43f23a8cbec98eaebb8f374c011f86b0175",
        "net_name": "nn-da9c99e92a-20260819.nnue",
        "net_option": "NNUENetpath",  # UCI option (no --nnue CLI exists)
        "note": ("built without an embedded net; the NNUE file is "
                 "fetched from Matthies/NN and loaded via the NNUENetpath "
                 "UCI option; without a net the generator forces "
                 "Use_NNUE=false (classical eval) — a failed net load "
                 "with Use_NNUE=true hangs the search"),
    },
    "lc0": {
        "repo": "https://github.com/LeelaChessZero/lc0",
        "ref": "fd71a2d921b689c5f479d3227c3806c8e272d9c5",  # tag v0.32.1
        "binary_rel": "build/release/lc0",
        "build_rel": ".",
        "build_cmd": "PATH={venv_bin}:$PATH ./build.sh -Dblas=true",
        "net_repo": "https://github.com/sergiovieri/lc0-training",
        "net_ref": "77325dbab3093ffe6d37d300729af94a7e3f21fd",
        "net_name": "net40.pb.gz",
        "net_rel": "networks/net40.pb.gz",
        "net_option": "WeightsFile",  # UCI option (src/neural/shared_params.cc)
        "net_sha256": "9d020b15c3e05ff1f46c384d08a1c4fdc235f582981d"
                      "05db185dfc1147cd2e9a",  # verified 2026-08-28
        "note": ("no prebuilt Linux release exists (Windows/macOS/Android "
                 "only) — built from source; needs meson + ninja + a "
                 "C++20 compiler (pip install meson ninja)"),
    },
}

# strength ladders for engines without a native UCI_Elo.
# Monotone non-decreasing in the target elo: the step value of the
# largest ladder elo <= target (clamped at both ends). The exact elo
# values are approximate (uncalibrated) but the ordering of strength
# is guaranteed by the monotone lever.
LC0_MOVETIME_LADDER = ((600, 2), (800, 3), (1100, 5), (1400, 8),
                       (1700, 15), (2000, 30), (2300, 60), (2600, 120),
                       (2900, 250), (3200, 500))
RC_NPS_LADDER = ((600, 100), (800, 300), (1100, 800), (1400, 2000),
                 (1700, 5000), (2000, 12000), (2300, 30000),
                 (2600, 80000), (2900, 200000), (3200, 2147483647))


def ladder_value(ladder: tuple, elo: int) -> int:
    val = ladder[0][1]
    for step_elo, step_val in ladder:
        if elo >= step_elo:
            val = step_val
    return val


@dataclasses.dataclass(frozen=True)
class EngineProfile:
    id: str
    display: str
    kind: str            # "maia3" (in-process ONNX) | "uci" (subprocess)
    elo_min: int
    elo_max: int
    elo_quality: str     # calibrated | native | approximate
    strength: str        # policy-conditioning | elo-option |
                         # movetime-ladder | nps-ladder
    move_ms: int = 100   # fixed thinking budget for non-ladder UCI engines


ENGINE_PROFILES = [
    EngineProfile("maia3", "Maia3", "maia3", 100, 3200,
                  "calibrated", "policy-conditioning"),
    EngineProfile("stockfish", "Stockfish", "uci", 1320, 3190,
                  "native", "elo-option", move_ms=100),
    EngineProfile("lc0", "LC0", "uci", 600, 3200,
                  "approximate", "movetime-ladder", move_ms=15),
    EngineProfile("rubichess", "RubiChess", "uci", 600, 3200,
                  "approximate", "nps-ladder", move_ms=250),
]
PROFILE_BY_ID = {p.id: p for p in ENGINE_PROFILES}


def parse_engines(spec: str) -> list[EngineProfile]:
    ids = [s.strip() for s in spec.split(",") if s.strip()]
    out = []
    for i in ids:
        if i not in PROFILE_BY_ID:
            raise SystemExit(f"unknown engine {i!r} "
                             f"(choose from {', '.join(PROFILE_BY_ID)})")
        out.append(PROFILE_BY_ID[i])
    if len({p.id for p in out}) != len(out):
        raise SystemExit("duplicate engine ids in --engines")
    return out


def effective_range(p: EngineProfile, elo_min: int, elo_max: int):
    lo, hi = max(elo_min, p.elo_min), min(elo_max, p.elo_max)
    return lo, hi


# ----------------------------------------------------------------------
# deterministic plan
# ----------------------------------------------------------------------

def plan_elo(elo_min: int, elo_max: int, n_games: int, seed: int) -> np.ndarray:
    """(n_games, 2) int32 array: uniform integer elo per side per game.

    Kept from the original 2M generator: the plan stream is the same
    call sequence the small reference set used (rng.randint per side,
    per game, in game order), so for a given seed the first N games'
    elo pairs are identical to that pipeline.
    """
    rng = random.Random(seed)
    out = np.empty((n_games, 2), dtype=np.int32)
    for i in range(n_games):
        out[i, 0] = rng.randint(elo_min, elo_max)
        out[i, 1] = rng.randint(elo_min, elo_max)
    return out


def plan_games(pool: list[EngineProfile], n_games: int, seed: int,
               elo_min: int, elo_max: int) -> np.ndarray:
    """(n_games, 4) int32 array: [engine_w, elo_w, engine_b, elo_b].

    Single-engine pool: legacy stream (identical to plan_elo for the
    same seed — no engine rng call, so the call sequence is unchanged).
    Mixed pool: per side, rng.choice(engine) then rng.randint(elo).
    """
    eff = {i: effective_range(p, elo_min, elo_max) for i, p in
           enumerate(pool)}
    empty = [i for i, (lo, hi) in eff.items() if lo > hi]
    if empty:
        names = ", ".join(pool[i].id for i in empty)
        raise SystemExit(
            f"engine(s) {names} have no overlap with "
            f"--elo-min/--elo-max; narrow the pool or widen the bounds")
    out = np.empty((n_games, 4), dtype=np.int32)
    rng = random.Random(seed)
    if len(pool) == 1:
        lo, hi = eff[0]
        for i in range(n_games):
            out[i, 0] = 0
            out[i, 1] = rng.randint(lo, hi)
            out[i, 2] = 0
            out[i, 3] = rng.randint(lo, hi)
    else:
        idxs = list(range(len(pool)))
        for i in range(n_games):
            ew = rng.choice(idxs)
            out[i, 0] = ew
            lo, hi = eff[ew]
            out[i, 1] = rng.randint(lo, hi)
            eb = rng.choice(idxs)
            out[i, 2] = eb
            lo, hi = eff[eb]
            out[i, 3] = rng.randint(lo, hi)
    return out


def game_rng(seed: int, game_id: int) -> random.Random:
    """Per-game move-sampling substream: independent of every other game."""
    h = hashlib.sha256(f"{seed}:{game_id}".encode()).digest()[:8]
    return random.Random(int.from_bytes(h, "big"))


def side_move_ms(p: EngineProfile, elo: int, args: argparse.Namespace) -> int:
    if p.strength == "movetime-ladder":
        return ladder_value(LC0_MOVETIME_LADDER, elo)
    if p.strength == "nps-ladder":
        return args.rc_ms  # the NPS cap does the limiting; time is a
                           # budget for the cap to bind
    return args.uci_ms


# ----------------------------------------------------------------------
# UCI subprocess engine (resident, reused across all games of a shard)
# ----------------------------------------------------------------------

class UciEngine:
    def __init__(self, name: str, cmd: list[str], cwd: str | None = None,
                 extra_options: dict | None = None, timeout: int = 120):
        self.name = name
        self.cmd = cmd
        self.cwd = cwd
        self.extra_options = dict(extra_options or {})
        self.timeout = timeout
        self.last_options: dict[str, str] = {}
        self.proc = None
        self.q: queue.Queue = queue.Queue()
        self.tail: list[str] = []
        self.reader = None
        self._spawn()
        self.send("uci")
        self.wait("uciok", self.timeout)
        self.set_options({"Threads": "1", **self.extra_options})
        self.ready()

    def _spawn(self) -> None:
        self.proc = subprocess.Popen(
            self.cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1, cwd=self.cwd)
        self.q = queue.Queue()
        self.tail = []
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self) -> None:
        assert self.proc is not None
        for line in self.proc.stdout:
            self.tail.append(line.rstrip("\n"))
            if len(self.tail) > 200:
                self.tail.pop(0)
            self.q.put(line)

    def send(self, s: str) -> None:
        if self.proc is None or self.proc.poll() is not None:
            raise RuntimeError(f"{self.name}: engine process is dead")
        self.proc.stdin.write(s + "\n")
        self.proc.stdin.flush()

    def wait(self, pat: str, timeout: float) -> str:
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                line = self.q.get(timeout=0.05)
            except queue.Empty:
                if self.proc is not None and self.proc.poll() is not None:
                    raise RuntimeError(
                        f"{self.name}: process exited rc="
                        f"{self.proc.returncode}; tail: "
                        f"{' | '.join(self.tail[-5:])}")
                continue
            if pat in line:
                return line
        raise TimeoutError(f"{self.name}: no {pat!r} within {timeout}s; "
                           f"tail: {' | '.join(self.tail[-5:])}")

    def ready(self) -> None:
        self.send("isready")
        self.wait("readyok", self.timeout)

    def set_options(self, opts: dict) -> None:
        for k, v in opts.items():
            if self.last_options.get(k) != v:
                self.send(f"setoption name {k} value {v}")
                self.last_options[k] = v

    def new_game(self) -> None:
        self.send("ucinewgame")
        self.ready()

    def bestmove(self, fen: str, ms: int) -> str | None:
        self.send(f"position fen {fen}")
        self.send(f"go movetime {ms}")
        try:
            line = self.wait("bestmove", max(30.0, ms / 1000.0 * 10 + 30))
        except TimeoutError:
            return None
        mv = line.split()[1]
        return None if mv == "(none)" else mv

    def reinit(self) -> None:
        try:
            if self.proc is not None:
                self.proc.kill()
                self.proc.wait(timeout=5)
        except Exception:  # noqa: BLE001
            pass
        self.last_options = {}
        self._spawn()
        self.send("uci")
        self.wait("uciok", self.timeout)
        self.set_options({"Threads": "1", **self.extra_options})
        self.ready()


def uci_engine_command(pid: str, engines_dir: Path) -> tuple[list[str], str | None, dict]:
    """Resolve (cmd, cwd, extra_options) for a UCI engine.

    Prefers engines.json (written by fetch-engines); falls back to the
    well-known layout. Raises SystemExit with the fix instructions.
    """
    pinned = PINNED_ENGINES[pid]
    info = {}
    eng_json = engines_dir / "engines.json"
    if eng_json.exists():
        info = json.loads(eng_json.read_text()).get(pid, {})
    base = engines_dir / pid
    binary = info.get("binary") or str(base / pinned.get("binary_rel", pid).split("/")[-1])
    if not Path(binary).exists():
        raise SystemExit(
            f"engine binary not found for {pid}: {binary}\n"
            f"run: python3 generate.py fetch-engines --engines-dir "
            f"{engines_dir} --engines {pid}")
    net = info.get("net")
    cmd = [binary]
    cwd = str(base)
    extra: dict = {}
    if pid == "stockfish":
        net_path = net or str(base / pinned["net_name"])
        if not Path(net_path).exists():
            raise SystemExit(
                f"stockfish net not found: {net_path} (Stockfish 18 is "
                f"NNUE-only and refuses to run without its net)\n"
                f"run: python3 generate.py fetch-engines --engines-dir "
                f"{engines_dir} --engines stockfish")
        extra["EvalFile"] = str(net_path)
    elif pid == "rubichess":
        if net:
            extra[pinned["net_option"]] = str(net)
        else:
            # built without an embedded net: force the classical eval,
            # otherwise Use_NNUE=true (the default) hangs the search
            extra["Use_NNUE"] = "false"
    elif pid == "lc0":
        net_path = net or str(base / pinned["net_name"])
        if not Path(net_path).exists():
            raise SystemExit(
                f"lc0 net not found: {net_path} (lc0 cannot move "
                f"without a network)\n"
                f"run: python3 generate.py fetch-engines --engines-dir "
                f"{engines_dir} --engines lc0")
        extra[pinned["net_option"]] = str(net_path)
    return cmd, cwd, extra


# ----------------------------------------------------------------------
# workers
# ----------------------------------------------------------------------

_WORKER: dict = {}


def _worker_init(args: tuple) -> None:
    (pool, model_path, provider, gpu_id, intra_op_threads,
     engines_dir) = args
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    if any(p.kind == "maia3" for p in pool):
        _WORKER["model"] = Maia3(
            Path(model_path),
            providers=([provider, "CPUExecutionProvider"] if gpu_id is not None
                       else None),
            intra_op_threads=intra_op_threads if gpu_id is None else None)
    # one engine instance PER (engine, side): the two sides of a game
    # can draw the same engine at different elos, and UCI options are
    # per-process, so a shared instance would corrupt the setup
    _WORKER["engines"] = {}
    for p in pool:
        if p.kind != "uci":
            continue
        cmd, cwd, extra = uci_engine_command(p.id, Path(engines_dir))
        for side in ("w", "b"):
            _WORKER["engines"][(p.id, side)] = UciEngine(
                f"{p.id}-{side}", cmd, cwd=cwd, extra_options=extra)


def _side_profile(pool: list, idx: int) -> EngineProfile:
    return pool[int(idx)]


def _play_game_mixed(pool, plan_row, seed, gid, temperature, max_ply,
                     args) -> tuple[list[dict], list[str], str]:
    """Play one game between two (engine, elo) sides. Returns
    (label rows, sans, result). Raises EngineAbort on unrecoverable
    engine failure (caller counts + skips the game)."""
    import chess

    model = _WORKER.get("model")
    engines = _WORKER.get("engines", {})
    pw = pool[int(plan_row[0])]
    eb_w = int(plan_row[1])
    pb = pool[int(plan_row[2])]
    eb_b = int(plan_row[3])
    rng = game_rng(seed, gid)

    # per-side setup (set once per game) — keyed by side so both
    # sides can safely hold the same engine profile at different elos
    sides = {
        "w": (pw, eb_w),
        "b": (pb, eb_b),
    }
    for side, (prof, elo) in sides.items():
        if prof.kind == "uci":
            eng = engines[(prof.id, side)]
            if prof.strength == "elo-option":
                eng.set_options({"UCI_LimitStrength": "true",
                                 "UCI_Elo": str(elo)})
            elif prof.strength == "nps-ladder":
                eng.set_options({"LimitNps":
                                 str(ladder_value(RC_NPS_LADDER, elo))})
            eng.new_game()

    board = chess.Board()
    rows: list[dict] = []
    sans: list[str] = []
    while not board.is_game_over() and board.ply() < max_ply:
        white_turn = board.turn == chess.WHITE
        side_key = "w" if white_turn else "b"
        prof, elo = sides[side_key]
        elo_self = elo
        elo_oppo = eb_w if not white_turn else eb_b
        fen = board.fen()
        if prof.kind == "maia3":
            move_probs, ldw, top1 = model.probs(fen, elo_self, elo_oppo)
            weights = [p ** temperature if temperature > 0 else 1.0
                       for p in move_probs.values()]
            uci = rng.choices(list(move_probs), weights=weights)[0]
            row_extra = {"engine": "maia3", "elo_quality": "calibrated",
                         "top1_prob": round(top1, 4),
                         "ldw": [round(x, 4) for x in ldw]}
        else:
            eng = engines[(prof.id, side_key)]
            uci = eng.bestmove(fen, side_move_ms(prof, elo, args))
            if uci is None:
                eng.reinit()
                uci = eng.bestmove(fen, side_move_ms(prof, elo, args))
            if uci is None:
                raise EngineAbort(f"{prof.id} failed twice from {fen[:40]}")
            row_extra = {"engine": prof.id, "elo_quality": prof.elo_quality,
                         "top1_prob": None, "ldw": None}
        mv = chess.Move.from_uci(uci)
        assert mv in board.legal_moves, (fen, uci, prof.id)
        rows.append({
            "game": gid, "elo_white": eb_w, "elo_black": eb_b,
            "fen": fen, "move_uci": uci, "move_ply": board.ply() + 1,
            "side": "white" if white_turn else "black",
            "elo_self": elo_self, "elo_oppo": elo_oppo, **row_extra,
        })
        sans.append(board.san(mv))
        board.push(mv)
    return rows, sans, board.result()


class EngineAbort(Exception):
    pass


def _worker_run(job: tuple) -> dict:
    (shard, gid_start, gid_end, plan_slice, pool_ids, model_path,
     provider, gpu_id, out_dir, temperature, max_ply, resume, date_str,
     seed, intra_op_threads, engines_dir, uci_ms, rc_ms,
     fsync_every, mixed) = job
    args_ns = argparse.Namespace(uci_ms=uci_ms, rc_ms=rc_ms)
    pool = [PROFILE_BY_ID[i] for i in pool_ids]
    out = Path(out_dir)
    _worker_init((pool, model_path, provider, gpu_id, intra_op_threads,
                  engines_dir))

    pgn_f, labels_f, progress_f = _shard_files(out, shard)
    prog = _load_progress(progress_f)
    if resume and prog["last_gid"] >= gid_start:
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
    skipped = prog.get("skipped", 0)
    t0 = time.time()
    last_log = t0
    pending = 0

    with pgn_f.open("a", encoding="utf-8") as pgn_fh, \
            labels_f.open("a", encoding="utf-8") as label_fh:
        for k, gid in enumerate(range(start_gid, gid_end)):
            row = plan_slice[k + (start_gid - gid_start)]
            try:
                rows, sans, result = _play_game_mixed(
                    pool, row, seed, gid, temperature, max_ply, args_ns)
            except EngineAbort as exc:
                skipped += 1
                print(f"[shard {shard}] skipping game {gid}: {exc}",
                      file=sys.stderr, flush=True)
                continue
            ew = int(row[1])
            eb = int(row[3])
            pw = pool[int(row[0])]
            pb = pool[int(row[2])]
            headers = {
                "Event": ("Mixed engine self-play (random UCI elo)"
                          if mixed else "Maia3 self-play (random UCI elo)"),
                "Site": "cloud", "Date": date_str, "Round": str(gid),
                "White": f"{pw.display} (elo {ew})",
                "Black": f"{pb.display} (elo {eb})",
                "WhiteEngine": pw.id, "BlackEngine": pb.id,
                "WhiteElo": str(ew), "BlackElo": str(eb),
                "WhiteEloQuality": pw.elo_quality,
                "BlackEloQuality": pb.elo_quality,
                "Result": result,
            }
            pgn_fh.write(game_pgn_text(headers, sans, result))
            for r in rows:
                label_fh.write(json.dumps(r, separators=(",", ":")) + "\n")
            games_done += 1
            rows_done += len(rows)
            pending += 1
            # batch fsync: the checkpoint only advances after the data
            # is durable; a crash loses at most fsync_every games
            if pending >= fsync_every:
                _fsync_f(pgn_fh)
                _fsync_f(label_fh)
                pending = 0
                with progress_f.open("w") as pf:
                    json.dump({"last_gid": gid, "games": games_done,
                               "rows": rows_done, "skipped": skipped,
                               "pgn_bytes": pgn_fh.tell(),
                               "label_bytes": label_fh.tell()}, pf)
            now = time.time()
            if now - last_log >= 30:
                el = now - t0
                rate = (games_done - prog["games"]) / el
                eta = (gid_end - gid - 1) / max(rate, 1e-9)
                print(f"[shard {shard}] games={games_done} "
                      f"rate={rate:.2f}/s eta={eta / 3600:.2f}h "
                      f"skipped={skipped}", flush=True)
                last_log = now
        if pending:  # final flush INSIDE the with block (files open)
            _fsync_f(pgn_fh)
            _fsync_f(label_fh)
            with progress_f.open("w") as pf:
                json.dump({"last_gid": gid_end - 1 if gid_end > start_gid
                           else prog["last_gid"], "games": games_done,
                           "rows": rows_done, "skipped": skipped,
                           "pgn_bytes": pgn_fh.tell(),
                           "label_bytes": label_fh.tell()}, pf)
    return {"shard": shard, "games": games_done, "rows": rows_done,
            "skipped": skipped,
            "pgn_bytes": pgn_f.stat().st_size,
            "label_bytes": labels_f.stat().st_size}


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
    return {"last_gid": -1, "games": 0, "rows": 0, "skipped": 0,
            "pgn_bytes": 0, "label_bytes": 0}


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


def _preflight(pool, model_path, engines_dir, gpus) -> dict:
    """One real move per engine before burning hours on the box."""
    import chess

    report = {}
    if any(p.kind == "maia3" for p in pool):
        m = Maia3(Path(model_path),
                  providers=(["CUDAExecutionProvider",
                              "CPUExecutionProvider"] if gpus else None),
                  intra_op_threads=1)
        _p, _l, t1 = m.probs(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            1500, 1500)
        report["maia3"] = f"ok (startpos top1 {t1:.4f})"
    for p in pool:
        if p.kind != "uci":
            continue
        cmd, cwd, extra = uci_engine_command(p.id, Path(engines_dir))
        e = UciEngine(p.id, cmd, cwd=cwd, extra_options=extra, timeout=60)
        e.send("ucinewgame")
        e.ready()
        e.send("position startpos")
        e.send("go movetime 30")
        mv = e.wait("bestmove", 60).split()[1]
        if mv == "(none)":
            raise SystemExit(f"preflight failed: {p.id} returned (none)")
        report[p.id] = f"ok (startpos bestmove {mv})"
        try:
            e.proc.kill()
        except Exception:  # noqa: BLE001
            pass
    return report


def cmd_generate(args: argparse.Namespace) -> int:
    import multiprocessing as mp

    pool = parse_engines(args.engines)
    mixed = len(pool) > 1

    model_path = Path(args.model)
    have_maia3 = any(p.kind == "maia3" for p in pool)
    model_sha = None
    if have_maia3:
        if not model_path.exists():
            raise SystemExit(
                f"model not found: {model_path}\n"
                "fetch it first: python3 tools/selfplay_elo_mixer.py "
                "fetch-model --out <dir>")
        model_sha = hashlib.sha256(model_path.read_bytes()).hexdigest()

    out = Path(args.out)
    for sub in ("pgn", "labels", "progress"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    gpus = args.gpus
    workers = args.workers
    if not workers:
        workers = (gpus or 1) if gpus else max(1, (os.cpu_count() or 4) - 2)

    t0 = time.time()
    print(f"building plan: {args.games} games, seed {args.seed}, pool "
          f"[{', '.join(p.id for p in pool)}] "
          f"(elo bounds {args.elo_min}..{args.elo_max})...", flush=True)
    plan = plan_games(pool, args.games, args.seed, args.elo_min,
                      args.elo_max)
    print(f"plan done in {time.time() - t0:.1f}s", flush=True)

    print("preflight: one real move per engine...", flush=True)
    preflight = _preflight(pool, model_path, args.engines_dir, gpus)
    for k, v in preflight.items():
        print(f"  {k}: {v}", flush=True)

    ranges = _chunk_ranges(args.games, workers)
    date_str = args.date or date.today().strftime("%Y.%m.%d")
    ctx = mp.get_context("spawn")
    results = []
    with ctx.Pool(processes=workers) as poolmp:
        job_args = []
        for i, (a, b) in enumerate(ranges):
            gpu_id = i % gpus if gpus else None
            job_args.append((
                i, a, b, plan[a:b], [p.id for p in pool],
                str(model_path) if have_maia3 else "",
                "CUDAExecutionProvider" if gpus else "CPUExecutionProvider",
                gpu_id, str(out), args.temperature, args.max_ply,
                args.resume, date_str, args.seed, args.maia_threads,
                args.engines_dir, args.uci_ms, args.rc_ms,
                args.fsync_every, mixed,
            ))
        for r in poolmp.imap_unordered(_worker_run, job_args):
            results.append(r)
            print(json.dumps(r), flush=True)

    _finalize_manifest(out, pool, args, model_sha, model_path, workers,
                       gpus, ranges, results, preflight, mixed, date_str)

    if not args.no_validate:
        print("running full validation + calibration pass...", flush=True)
        rc = _run_validate(out, workers=min(workers, 16), check_dupes=False)
        if rc != 0:
            print("VALIDATION FAILED — do not use this output for "
                  "training until it passes", file=sys.stderr)
        return rc
    return 0


def _finalize_manifest(out, pool, args, model_sha, model_path, workers,
                       gpus, ranges, results, preflight, mixed,
                       date_str) -> None:
    import onnxruntime as ort

    shards = []
    total_games = total_rows = total_skipped = 0
    for r in sorted(results, key=lambda x: x["shard"]):
        pgn = out / "pgn" / f"shard-{r['shard']:05d}.pgn"
        labels = out / "labels" / f"shard-{r['shard']:05d}.jsonl"
        shards.append({
            "shard": r["shard"],
            "pgn": str(pgn), "pgn_sha256": _sha(pgn),
            "labels": str(labels), "labels_sha256": _sha(labels),
            "games": r["games"], "label_rows": r["rows"],
            "skipped": r["skipped"],
            "range": [ranges[r["shard"]][0], ranges[r["shard"]][1]],
        })
        total_games += r["games"]
        total_rows += r["rows"]
        total_skipped += r["skipped"]

    engines = []
    for p in pool:
        e = {"id": p.id, "display": p.display, "kind": p.kind,
             "strength": p.strength, "elo_min": p.elo_min,
             "elo_max": p.elo_max, "elo_quality": p.elo_quality,
             "preflight": preflight.get(p.id)}
        if p.kind == "uci":
            pinned = PINNED_ENGINES[p.id]
            e["provenance"] = {k: pinned[k] for k in pinned}
            base = Path(args.engines_dir) / p.id
            bin_path = base / Path(pinned["binary_rel"]).name
            if bin_path.exists():
                e["provenance"]["binary_sha256"] = _sha(bin_path)
            net = str(base / pinned["net_name"])
            if Path(net).exists():
                e["provenance"]["net_sha256"] = _sha(Path(net))
        engines.append(e)

    manifest = {
        "tool": "tools/maia3_cloud_selfplay/generate.py",
        "mode": "mixed-engine" if mixed else "maia3-only",
        "engines": engines,
        "model": None,
        "seed": args.seed,
        "elo_min": args.elo_min, "elo_max": args.elo_max,
        "elo_sampling": ("per side: engine uniform over pool, then elo "
                         "uniform integer inside the engine's supported "
                         "range (1-elo accuracy); single-engine pools "
                         "keep the legacy plan_elo stream"),
        "strength_mechanics": {
            "maia3": "native UCI_Elo model conditioning (calibrated)",
            "stockfish": "native UCI_Elo (1320..3190) + fixed movetime",
            "lc0": "movetime ladder (LC0 has no UCI_Elo option)",
            "rubichess": "LimitNps ladder (RubiChess has no UCI_Elo "
                         "option) + fixed movetime",
            "approximate": "lc0/rubichess elo labels are monotone but "
                           "uncalibrated — filter by EloQuality for "
                           "level-conditioned training if needed",
        },
        "move_sampling": ("maia3: temperature-1 sampling from the "
                          "elo-conditioned policy, per-game substream "
                          "sha256('seed:game_id'); uci engines: bestmove "
                          "of go movetime <budget>"),
        "temperature": args.temperature,
        "max_ply": args.max_ply,
        "uci_ms": args.uci_ms, "rc_ms": args.rc_ms,
        "games": total_games,
        "skipped_games": total_skipped,
        "label_rows": total_rows,
        "date_header": date_str,
        "backend": f"onnxruntime {ort.__version__}" if any(
            p.kind == "maia3" for p in pool) else "uci-subprocess-only",
        "maia_threads": args.maia_threads,
        "provider": ("CUDAExecutionProvider" if gpus else
                     "CPUExecutionProvider"),
        "workers": workers,
        "gpus": bool(gpus),
        "fsync_every": args.fsync_every,
        "shards": shards,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    if model_sha:
        manifest["model"] = {
            "file": model_path.name,
            "sha256": model_sha,
            "source_repo": "mcognetta/simple-maia3-inference",
            "source_commit": MODEL_COMMIT,
            "source_path": MODEL_FILE,
            "note": ("maia3_simplified.onnx — the official "
                     "maia-platform-frontend export (single position, "
                     "elo-conditioned)"),
        }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"manifest written: {total_games} games, {total_rows} label "
          f"rows, {total_skipped} skipped", flush=True)


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
    shards = manifest["shards"]
    print(f"validating {len(shards)} shards, {total_games} games, "
          f"{total_rows} label rows (seed {seed})...", flush=True)

    jobs = [(s["shard"], Path(s["pgn"]), Path(s["labels"])) for s in shards]
    ctx = mp.get_context("spawn")
    errors: list[str] = []
    top1_by_band: dict[int, list[float]] = {}
    engine_stats: dict[str, dict] = {}
    dup_count = 0
    with ctx.Pool(processes=min(args.workers, len(jobs))) as pool:
        for res in pool.imap_unordered(_validate_shard, jobs):
            errors.extend(res["errors"])
            for band, vals in res["top1_by_band"].items():
                top1_by_band.setdefault(band, []).extend(vals)
            for name, st in res["engine_stats"].items():
                tgt = engine_stats.setdefault(
                    name, {"moves": 0, "games_sided": 0})
                tgt["moves"] += st["moves"]
                tgt["games_sided"] += st["games_sided"]
            dup_count += res["dupes"]
            if res["errors"]:
                print(f"  shard {res['shard']}: {len(res['errors'])} errors",
                      flush=True)
            else:
                print(f"  shard {res['shard']}: ok "
                      f"({res['games']} games, {res['rows']} rows)",
                      flush=True)

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
            "note": ("conditioning check over maia3 rows only: "
                     "low-elo play must be less concentrated than "
                     "high-elo play; only enforced with >= 1000 moves "
                     "(mini runs report null)"),
        }
        low, high = calibration["low_end_mean"], calibration["high_end_mean"]
        if total_moves < 1000:
            calibration["check_passed"] = None
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
        "engine_stats": engine_stats,
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
        print("note: conditioning check skipped (insufficient maia3 "
              "moves)")
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
    engine_stats: dict[str, dict] = {}
    games = rows = 0

    by_game: dict[int, list[dict]] = {}
    with labels_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            by_game.setdefault(r["game"], []).append(r)
            rows += 1
            en = r.get("engine", "maia3")
            st = engine_stats.setdefault(en, {"moves": 0, "games_sided": 0})
            st["moves"] += 1

    stream = pgn_path.open("r", encoding="utf-8")
    seen: dict[int, object] = {}
    while True:
        g = chess.pgn.read_game(stream)
        if g is None:
            break
        gid = int(g.headers["Round"])
        seen[gid] = g
        games += 1
        for side in ("White", "Black"):
            en = g.headers.get(f"{side}Engine", "maia3")
            st = engine_stats.setdefault(en, {"moves": 0, "games_sided": 0})
            st["games_sided"] += 1
        if gid in by_game:
            errors.extend(_check_game(g, by_game[gid], errors))
            for r in by_game[gid]:
                if r.get("engine") == "maia3" and r.get("top1_prob") is not None:
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
            "engine_stats": engine_stats, "dupes": 0}


def _check_game(g, rows: list[dict], errors: list[str]) -> list[str]:
    import chess

    gid = int(g.headers["Round"])
    w = int(g.headers["WhiteElo"])
    b = int(g.headers["BlackElo"])
    we = g.headers.get("WhiteEngine", "maia3")
    be = g.headers.get("BlackEngine", "maia3")
    wq = g.headers.get("WhiteEloQuality", "calibrated")
    bq = g.headers.get("BlackEloQuality", "calibrated")
    moves = list(g.mainline_moves())
    if len(rows) != len(moves):
        errors.append(f"game {gid}: {len(rows)} labels != {len(moves)} moves")
        return errors
    if (int(rows[0]["elo_white"]), int(rows[0]["elo_black"])) != (w, b):
        errors.append(f"game {gid}: header elos {w}/{b} != labels "
                      f"{rows[0]['elo_white']}/{rows[0]['elo_black']}")
    board = chess.Board()
    for i, (row, mv) in enumerate(zip(rows, moves)):
        white_turn = board.turn == chess.WHITE
        expect_engine = we if white_turn else be
        expect_quality = wq if white_turn else bq
        if row["fen"] != board.fen():
            errors.append(f"game {gid} ply {i}: FEN mismatch")
            break
        if row["move_uci"] != mv.uci():
            errors.append(f"game {gid} ply {i}: move mismatch "
                          f"{row['move_uci']} != {mv.uci()}")
            break
        if row["side"] != ("white" if white_turn else "black"):
            errors.append(f"game {gid} ply {i}: side mismatch")
            break
        if (row["elo_self"], row["elo_oppo"]) != (
                (w, b) if white_turn else (b, w)):
            errors.append(f"game {gid} ply {i}: elo mismatch")
            break
        if row.get("engine", "maia3") != expect_engine:
            errors.append(f"game {gid} ply {i}: engine mismatch "
                          f"{row.get('engine')} != {expect_engine}")
            break
        if row.get("elo_quality") != expect_quality:
            errors.append(f"game {gid} ply {i}: elo quality mismatch")
            break
        if expect_engine == "maia3":
            if not (0 < row["top1_prob"] <= 1):
                errors.append(f"game {gid} ply {i}: bad top1_prob")
                break
        elif row.get("top1_prob") is not None or row.get("ldw") is not None:
            errors.append(f"game {gid} ply {i}: engine row must have "
                          f"null top1_prob/ldw")
            break
        board.push(mv)
    result = board.result()
    if g.headers["Result"] != result:
        errors.append(f"game {gid}: header result {g.headers['Result']} "
                      f"!= replay {result}")
    return errors


# ----------------------------------------------------------------------
# fetch-engines: pinned clone + build + nets + self-test
# ----------------------------------------------------------------------

def _run(cmd: list[str] | str, cwd: str | None = None, env: dict | None = None,
         log: Path | None = None, check: bool = True) -> None:
    full = dict(os.environ)
    if env:
        full.update(env)
    lf_ctx = log.open("a") if log else open(os.devnull, "w")
    with lf_ctx as lf:
        lf.write(f"$ {cmd if isinstance(cmd, str) else ' '.join(cmd)}\n")
        lf.flush()
        rc = subprocess.call(cmd, cwd=cwd, env=full, stdout=lf,
                             stderr=subprocess.STDOUT, shell=isinstance(cmd, str))
    if check and rc != 0:
        tail = log.read_text().splitlines()[-20:] if log else []
        raise SystemExit(f"command failed rc={rc}: {cmd}\n" +
                         "\n".join(tail))


def cmd_fetch_engines(args: argparse.Namespace) -> int:
    base = Path(args.engines_dir)
    base.mkdir(parents=True, exist_ok=True)
    jobs = [j.strip() for j in args.engines.split(",") if j.strip()]
    unknown = [j for j in jobs if j not in PINNED_ENGINES]
    if unknown:
        raise SystemExit(f"unknown engine(s) {unknown}; available: "
                         f"{', '.join(PINNED_ENGINES)}")
    for j in jobs:
        _fetch_one(j, base, args.jobs, base / f"fetch-{j}.log")
    info = {}
    eng_json = base / "engines.json"
    if eng_json.exists():
        info = json.loads(eng_json.read_text())
    (base / "engines.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info, indent=2))
    print("fetch-engines complete. Run a generate preflight "
          "(it plays one move per engine) before the full run.")
    return 0


def _fetch_one(pid: str, base: Path, jobs: int, log: Path) -> None:
    pinned = PINNED_ENGINES[pid]
    d = base / pid
    d.mkdir(parents=True, exist_ok=True)
    src = d / "src"

    print(f"[{pid}] clone/checkout {pinned['ref'][:12]}...", flush=True)
    if not (src / ".git").exists():
        _run(["git", "clone", "-q", "--depth", "1", pinned["repo"],
              str(src)], log=log)
    _run(["git", "-C", str(src), "fetch", "-q", "--depth", "1", "origin",
          pinned["ref"]], log=log)
    _run(["git", "-C", str(src), "checkout", "-q", pinned["ref"]], log=log)
    if pid == "lc0":
        _run(["git", "-C", str(src), "submodule", "update", "--init",
              "--depth", "1"], log=log)

    print(f"[{pid}] building (this is the slow part)...", flush=True)
    build = pinned["build_cmd"].format(
        jobs=jobs, arch=os.environ.get("SF_ARCH", "x86-64-ssse3"),
        venv_bin=_venv_hint())
    _run(build, cwd=str(src / pinned["build_rel"]), log=log)

    binary = src / pinned["binary_rel"]
    if not binary.exists():
        raise SystemExit(f"[{pid}] build did not produce {binary}")

    net = _fetch_net(pid, pinned, d, src, log)

    print(f"[{pid}] self-test (one real move)...", flush=True)
    cmd, cwd, extra = _resolved_command(pid, binary, net)
    e = UciEngine(pid, cmd, cwd=cwd, extra_options=extra, timeout=180)
    e.send("ucinewgame")
    e.ready()
    e.send("position startpos")
    e.send("go movetime 30")
    line = e.wait("bestmove", 120)
    mv = line.split()[1]
    if mv == "(none)":
        raise SystemExit(f"[{pid}] self-test returned (none) — net "
                         f"problem?")
    print(f"[{pid}] self-test ok: startpos bestmove {mv}", flush=True)
    try:
        e.proc.kill()
    except Exception:  # noqa: BLE001
        pass

    eng_json = base / "engines.json"
    info = {}
    if eng_json.exists():
        info = json.loads(eng_json.read_text())
    info[pid] = {
        "binary": str(binary),
        "binary_sha256": _sha(binary),
        "net": str(net) if net else None,
        "net_sha256": _sha(net) if net else None,
        "source_ref": pinned["ref"],
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "self_test_bestmove": mv,
    }
    eng_json.write_text(json.dumps(info, indent=2) + "\n")


def _resolved_command(pid: str, binary: Path, net) -> tuple:
    # nets load via UCI options (NNUENetpath / WeightsFile), not CLI
    pinned = PINNED_ENGINES[pid]
    if pid == "stockfish":
        return [str(binary)], str(binary.parent), {"EvalFile": str(net)}
    extra = {}
    if pid == "rubichess":
        if net:
            extra[pinned["net_option"]] = str(net)
        else:
            extra["Use_NNUE"] = "false"
    if pid == "lc0":
        extra[pinned["net_option"]] = str(net)
    return [str(binary)], str(binary.parent), extra


def _venv_hint() -> str:
    venv = os.environ.get("VENV_BIN", "")
    return venv if venv else str(Path(sys.executable).parent)


def _fetch_net(pid: str, pinned: dict, d: Path, src: Path, log: Path) -> Path:
    net_path = d / pinned["net_name"]
    if pid == "stockfish":
        if not (net_path.exists() and net_path.stat().st_size > 1_000_000):
            print(f"[{pid}] downloading net from {pinned['net_url']} "
                  f"(official tests host — needs normal internet access, "
                  f"not the sandbox egress list)...", flush=True)
            _run(["curl", "-sL", "--max-time", "600", "-o", str(net_path),
                  pinned["net_url"]], log=log)
        if not (net_path.exists() and net_path.stat().st_size > 1_000_000):
            raise SystemExit(
                f"[{pid}] net download failed or too small: {net_path}\n"
                f"fetch it manually from {pinned['net_url']} and place "
                f"it at {net_path}")
        return net_path
    if pid == "rubichess":
        nnd = d / "NN"
        if not (nnd / ".git").exists():
            _run(["git", "clone", "-q", "--depth", "1",
                  pinned["nnue_repo"], str(nnd)], log=log)
        _run(["git", "-C", str(nnd), "fetch", "-q", "--depth", "1",
              "origin", pinned["nnue_ref"]], log=log)
        _run(["git", "-C", str(nnd), "checkout", "-q",
              pinned["nnue_ref"]], log=log)
        net = nnd / pinned["net_name"]
        if not net.exists():
            raise SystemExit(f"[{pid}] net {net} not in pinned NN repo")
        if not net_path.exists():
            net_path.write_bytes(net.read_bytes())
        return net_path
    if pid == "lc0":
        nrd = d / "net-repo"
        if not (nrd / ".git").exists():
            _run(["git", "clone", "-q", "--depth", "1", pinned["net_repo"],
                  str(nrd)], log=log)
        _run(["git", "-C", str(nrd), "fetch", "-q", "--depth", "1",
              "origin", pinned["net_ref"]], log=log)
        _run(["git", "-C", str(nrd), "checkout", "-q",
              pinned["net_ref"]], log=log)
        net = nrd / pinned["net_rel"]
        if not net.exists():
            raise SystemExit(f"[{pid}] net {net} not in pinned repo")
        if not net_path.exists():
            net_path.write_bytes(net.read_bytes())
        if pinned.get("net_sha256"):
            got = _sha(net_path)[:16]
            if not pinned["net_sha256"].startswith(got[:16]) and \
                    not got.startswith(pinned["net_sha256"][:16]):
                print(f"[{pid}] WARNING: net sha256 prefix mismatch "
                      f"({got} vs {pinned['net_sha256'][:16]})",
                      file=sys.stderr)
        return net_path
    return net_path


# ----------------------------------------------------------------------

def argument_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="generate the sharded game set")
    g.add_argument("--model", default="",
                   help="maia3 ONNX path (required when maia3 is in "
                        "--engines)")
    g.add_argument("--out", required=True)
    g.add_argument("--games", type=int, default=5_000_000)
    g.add_argument("--seed", type=int, default=20260827)
    g.add_argument("--engines", default="maia3,stockfish,lc0,rubichess",
                   help=f"comma list from {', '.join(PROFILE_BY_ID)}")
    g.add_argument("--elo-min", type=int, default=100)
    g.add_argument("--elo-max", type=int, default=3200)
    g.add_argument("--workers", type=int, default=None,
                   help="worker processes (default: cpus-2; on a "
                        "180-vCPU box use ~90 for the mixed pool)")
    g.add_argument("--gpus", type=int, default=0,
                   help="use this many GPUs for maia3 (one worker each)")
    g.add_argument("--temperature", type=float, default=1.0)
    g.add_argument("--max-ply", type=int, default=240)
    g.add_argument("--date", default=None,
                   help="PGN Date header override (default: today)")
    g.add_argument("--engines-dir", default="engines",
                   help="directory with engine binaries/nets (fetch-"
                        "engines target; engines.json inside is used "
                        "for exact paths)")
    g.add_argument("--uci-ms", type=int, default=100,
                   help="movetime budget for stockfish (default 100)")
    g.add_argument("--rc-ms", type=int, default=250,
                   help="movetime budget for rubichess so its NPS cap "
                        "binds (default 250)")
    g.add_argument("--maia-threads", type=int, default=1,
                   help="onnxruntime intra-op threads per maia3 worker; "
                        "default 1: measured 29.5 vs 26.8 ms/ply (1.10x) "
                        "for 1 vs 2 threads on one CPU, so 2 threads "
                        "costs a core and buys nothing measurable")
    g.add_argument("--fsync-every", type=int, default=10,
                   help="fsync + checkpoint every N games (default 10; "
                        "a crash loses at most N games per shard)")
    g.add_argument("--resume", action="store_true",
                   help="continue an interrupted run from checkpoints")
    g.add_argument("--no-validate", action="store_true",
                   help="skip the post-generation validation pass")

    v = sub.add_parser("validate", help="validate + calibrate an output dir")
    v.add_argument("--out", required=True)
    v.add_argument("--workers", type=int, default=8)
    v.add_argument("--check-dupes", action="store_true",
                   help="scan for duplicate games (slow at 5M scale)")

    f = sub.add_parser("fetch-engines",
                       help="pinned clone + build + nets + self-test")
    f.add_argument("--engines-dir", default="engines")
    f.add_argument("--engines", default="stockfish,rubichess,lc0",
                   help="comma list to fetch (default all UCI engines)")
    f.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) // 2),
                   help="build parallelism for make")

    g.set_defaults(fn=cmd_generate)
    v.set_defaults(fn=cmd_validate)
    f.set_defaults(fn=cmd_fetch_engines)
    return p


def main(argv: list[str] | None = None) -> int:
    args = argument_parser().parse_args(argv)
    if args.cmd == "generate" and not args.model and \
            "maia3" in [s.strip() for s in args.engines.split(",")]:
        raise SystemExit("--model is required when maia3 is in --engines")
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
