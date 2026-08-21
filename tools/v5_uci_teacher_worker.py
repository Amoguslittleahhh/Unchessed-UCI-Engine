#!/usr/bin/env python3
"""Annotate UNCHD4R0 legal sets with exact common-budget UCI regrets.

Each legal action is searched independently with the same node limit. By
default the engine hash is cleared before every action, avoiding action-order TT
leakage. This is intentionally expensive gold-label generation for a many-core
CPU machine, not the fast human-PGN mining pass.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import select
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Iterator

from a100_common import sha256_file
from aegis_v3_data import FLAG_TEACHER
from aegis_v4_data import (
    AegisV4Record,
    HEADER_BYTES,
    LEGAL_FLAG_REGRETS,
    MAX_LEGAL_ACTIONS,
    POLICY_GUIDE,
    RECORD_BYTES,
    parse_header,
    write_shard,
)

MATE_SCORE = 30_000


def action_to_uci(action: int) -> str:
    if not 0 <= action < 64 * 64 * 5:
        raise ValueError(f"action {action} outside policy vocabulary")
    source = action & 63
    target = (action >> 6) & 63
    promotion = action >> 12

    def square_name(square):
        return chr(ord("a") + (square & 7)) + chr(ord("1") + (square >> 3))

    suffix = "" if promotion == 0 else " nbrq"[promotion]
    return square_name(source) + square_name(target) + suffix


def record_to_fen(record: AegisV4Record) -> str:
    pieces = "PNBRQKpnbrqk"
    board = ["" for _ in range(64)]
    for plane, bitboard in enumerate(record.base.bitboards):
        value = bitboard
        while value:
            lsb = value & -value
            square = lsb.bit_length() - 1
            if board[square]:
                raise ValueError("overlapping bitboards in teacher record")
            board[square] = pieces[plane]
            value ^= lsb
    ranks = []
    for rank in range(7, -1, -1):
        text = ""
        empty = 0
        for file in range(8):
            piece = board[rank * 8 + file]
            if piece:
                if empty:
                    text += str(empty)
                    empty = 0
                text += piece
            else:
                empty += 1
        if empty:
            text += str(empty)
        ranks.append(text)
    rights = ""
    for bit, name in ((1, "K"), (2, "Q"), (4, "k"), (8, "q")):
        if record.base.castling & bit:
            rights += name
    rights = rights or "-"
    ep = "-" if record.base.ep_file == 0xFF else chr(ord("a") + record.base.ep_file) + "6"
    fullmove = max(1, record.base.ply // 2 + 1)
    return f"{'/'.join(ranks)} w {rights} {ep} {record.base.halfmove} {fullmove}"


def parse_score(tokens: list[str]) -> int | None:
    try:
        index = tokens.index("score")
        kind = tokens[index + 1]
        value = int(tokens[index + 2])
    except (ValueError, IndexError):
        return None
    # A lower/upper bound is not a common-budget action value. Accepting one
    # as exact silently corrupts regret labels.
    if "lowerbound" in tokens[index + 3 :] or "upperbound" in tokens[index + 3 :]:
        return None
    if kind == "cp":
        return max(-MATE_SCORE, min(MATE_SCORE, value))
    if kind == "mate":
        sign = 1 if value > 0 else -1
        return sign * (MATE_SCORE - min(abs(value), 1000))
    return None


class UciEngine:
    def __init__(
        self,
        command: list[str],
        threads: int,
        hash_mb: int,
        timeout: float,
        clear_hash_per_action: bool,
        options: list[tuple[str, str]],
    ):
        self.timeout = timeout
        self.clear_hash_per_action = clear_hash_per_action
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=False,
            bufsize=0,
        )
        try:
            self.send("uci")
            self.read_until("uciok")
            self.set_option("Threads", str(threads))
            self.set_option("Hash", str(hash_mb))
            self.set_option("Ponder", "false")
            for name, value in options:
                self.set_option(name, value)
            self.ready()
        except Exception:
            self.close()
            raise

    def send(self, command: str) -> None:
        if self.process.stdin is None:
            raise RuntimeError("UCI engine stdin is closed")
        self.process.stdin.write((command + "\n").encode("utf-8"))
        self.process.stdin.flush()

    def read_line(self, deadline: float) -> str:
        if self.process.stdout is None:
            raise RuntimeError("UCI engine stdout is closed")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("UCI engine response timeout")
        readable, _, _ = select.select([self.process.stdout], [], [], remaining)
        if not readable:
            raise TimeoutError("UCI engine response timeout")
        line = self.process.stdout.readline()
        if line == b"":
            stderr = self.process.stderr.read() if self.process.stderr else b""
            raise RuntimeError(
                f"UCI engine exited unexpectedly: {stderr[-2000:].decode('utf-8', errors='replace')}"
            )
        return line.decode("utf-8", errors="replace").strip()

    def read_until(self, marker: str) -> list[str]:
        deadline = time.monotonic() + self.timeout
        lines = []
        while True:
            line = self.read_line(deadline)
            lines.append(line)
            if line == marker or line.startswith(marker + " "):
                return lines

    def set_option(self, name: str, value: str) -> None:
        suffix = "" if value == "" else f" value {value}"
        self.send(f"setoption name {name}{suffix}")

    def ready(self) -> None:
        self.send("isready")
        self.read_until("readyok")

    def analyse_action(self, fen: str, move: str, nodes: int) -> int:
        if self.clear_hash_per_action:
            self.set_option("Clear Hash", "")
            self.ready()
        self.send(f"position fen {fen}")
        self.send(f"go nodes {nodes} searchmoves {move}")
        deadline = time.monotonic() + self.timeout
        latest_score = None
        while True:
            line = self.read_line(deadline)
            tokens = line.split()
            if tokens[:1] == ["info"]:
                score = parse_score(tokens)
                if score is not None:
                    latest_score = score
            if tokens[:1] == ["bestmove"]:
                if latest_score is None:
                    raise RuntimeError(f"teacher returned bestmove without score for {move}")
                return latest_score

    def close(self) -> None:
        if self.process.poll() is None:
            try:
                self.send("quit")
                self.process.wait(timeout=5)
            except (BrokenPipeError, subprocess.TimeoutExpired):
                self.process.kill()
                self.process.wait()
        for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
            if stream is not None and not stream.closed:
                stream.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()


def read_record_range(path: str | Path, start: int, count: int | None) -> Iterator[AegisV4Record]:
    path = Path(path)
    with path.open("rb") as handle:
        total = parse_header(handle.read(HEADER_BYTES))
        if start < 0 or start > total:
            raise ValueError(f"start {start} outside shard with {total} records")
        end = total if count is None else min(total, start + count)
        handle.seek(HEADER_BYTES + start * RECORD_BYTES)
        for index in range(start, end):
            payload = handle.read(RECORD_BYTES)
            try:
                yield AegisV4Record.unpack(payload)
            except ValueError as error:
                raise ValueError(f"{path}: record {index}: {error}") from error


def annotate_record(record: AegisV4Record, engine: UciEngine, nodes: int) -> AegisV4Record:
    fen = record_to_fen(record)
    actions = record.legal_actions[: record.legal_count]
    scores = [engine.analyse_action(fen, action_to_uci(action), nodes) for action in actions]
    best_score = max(scores)
    best_index = min(index for index, score in enumerate(scores) if score == best_score)
    best_action = actions[best_index]
    regrets = tuple(min(32_766, max(0, best_score - score)) for score in scores)
    padded_regrets = regrets + (32_767,) * (MAX_LEGAL_ACTIONS - len(regrets))
    selected_index = actions.index(record.target_action)
    base = dataclasses.replace(
        record.base,
        flags=record.base.flags | FLAG_TEACHER,
        teacher_score=max(-32_768, min(32_767, best_score)),
        best_move=best_action & 0x0FFF,
        best_score=max(-32_768, min(32_767, best_score)),
        move_score=max(-32_768, min(32_767, scores[selected_index])),
    )
    annotated = dataclasses.replace(
        record,
        base=base,
        teacher_best_action=best_action,
        policy_kind=POLICY_GUIDE,
        legal_flags=record.legal_flags | LEGAL_FLAG_REGRETS,
        legal_regrets=padded_regrets,
    )
    annotated.validate()
    return annotated


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=path.name, delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def parse_option(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("UCI option must be NAME=VALUE")
    return tuple(value.split("=", 1))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--engine-arg", action="append", default=[])
    parser.add_argument("--input", required=True)
    parser.add_argument("--input-sha256")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int)
    parser.add_argument("--nodes-per-action", type=int, default=5000)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--hash-mb", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--option", type=parse_option, action="append", default=[])
    parser.add_argument("--asset", action="append", default=[])
    parser.add_argument("--no-clear-hash", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    manifest_path = args.manifest or Path(str(args.output) + ".json")
    if args.resume and args.output.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("output_sha256") == sha256_file(args.output):
            print(f"resume: verified existing {args.output}")
            return
    if args.nodes_per_action <= 0 or args.threads <= 0 or args.hash_mb <= 0:
        raise SystemExit("nodes, threads, and hash must be positive")
    resolved_engine = shutil.which(args.engine) or args.engine
    command = [resolved_engine, *args.engine_arg]
    started = time.monotonic()
    legal_actions = 0
    emitted = 0
    with UciEngine(
        command,
        args.threads,
        args.hash_mb,
        args.timeout,
        not args.no_clear_hash,
        args.option,
    ) as engine:
        def generated():
            nonlocal legal_actions, emitted
            for record in read_record_range(args.input, args.start, args.count):
                annotated = annotate_record(record, engine, args.nodes_per_action)
                legal_actions += annotated.legal_count
                emitted += 1
                yield annotated

        write_shard(args.output, generated())
    elapsed = time.monotonic() - started
    manifest = {
        "schema": 1,
        "input": str(Path(args.input).resolve()),
        "input_sha256": args.input_sha256 or sha256_file(args.input),
        "output": str(args.output.resolve()),
        "output_sha256": sha256_file(args.output),
        "engine": str(Path(resolved_engine).resolve()),
        "engine_sha256": sha256_file(resolved_engine),
        "engine_args": args.engine_arg,
        "uci_options": dict(args.option),
        "assets": [
            {
                "path": str(Path(path).resolve()),
                "sha256": sha256_file(path),
                "bytes": Path(path).stat().st_size,
            }
            for path in args.asset
        ],
        "start": args.start,
        "count_requested": args.count,
        "records": emitted,
        "legal_actions": legal_actions,
        "nodes_per_action": args.nodes_per_action,
        "threads": args.threads,
        "hash_mb": args.hash_mb,
        "clear_hash_per_action": not args.no_clear_hash,
        "elapsed_seconds": elapsed,
        "records_per_second": emitted / max(elapsed, 1e-9),
        "actions_per_second": legal_actions / max(elapsed, 1e-9),
    }
    atomic_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
