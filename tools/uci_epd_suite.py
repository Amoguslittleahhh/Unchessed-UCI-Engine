#!/usr/bin/env python3
"""Run a licensed/public EPD best-move suite against a UCI engine.

This runner does not distribute test positions. It accepts EPD ``bm`` and
``am`` operations whose moves are encoded as UCI/LAN coordinates (for example
``e2e4`` or ``a7a8q``). Proprietary ChessBase/Fritz suites must be lawfully
exported by their owner; SAN-only answers must be converted to coordinates by
the licensed GUI before use.

A solved-position count is a regression diagnostic, not an Elo estimate. Use
controlled engine matches and SPRT for strength claims.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import queue
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

UCI_MOVE_RE = re.compile(r"^[a-h][1-8][a-h][1-8][qrbn]?$", re.IGNORECASE)
INFO_VALUE_RE = re.compile(r"\b(depth|nodes|time)\s+(\d+)")
SCORE_RE = re.compile(r"\bscore\s+(cp|mate)\s+(-?\d+)")


@dataclass(frozen=True)
class Position:
    fen: str
    identifier: str
    best_moves: tuple[str, ...]
    avoid_moves: tuple[str, ...]
    line_number: int


@dataclass
class SearchResult:
    bestmove: str
    depth: int = 0
    nodes: int = 0
    time_ms: int = 0
    score_kind: str | None = None
    score: int | None = None


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_operations(text: str) -> list[str]:
    operations: list[str] = []
    current: list[str] = []
    quoted = False
    escaped = False
    for character in text:
        if escaped:
            current.append(character)
            escaped = False
        elif character == "\\" and quoted:
            current.append(character)
            escaped = True
        elif character == '"':
            current.append(character)
            quoted = not quoted
        elif character == ";" and not quoted:
            value = "".join(current).strip()
            if value:
                operations.append(value)
            current = []
        else:
            current.append(character)
    value = "".join(current).strip()
    if value:
        operations.append(value)
    return operations


def unquote(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] == '"':
        return value[1:-1].replace('\\"', '"').replace("\\\\", "\\")
    return value


def move_tokens(value: str) -> tuple[str, ...]:
    tokens = []
    for token in value.replace(",", " ").split():
        cleaned = token.strip().rstrip("+#?!").lower()
        if cleaned:
            tokens.append(cleaned)
    return tuple(tokens)


def parse_epd_line(line: str, line_number: int) -> Position | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    fields = stripped.split(maxsplit=4)
    if len(fields) < 4:
        raise ValueError(f"line {line_number}: EPD needs four FEN fields")
    board, side, castling, en_passant = fields[:4]
    operations_text = fields[4] if len(fields) == 5 else ""
    operations: dict[str, str] = {}
    for operation in split_operations(operations_text):
        name, separator, value = operation.partition(" ")
        operations[name.lower()] = value.strip() if separator else ""
    best_moves = move_tokens(operations.get("bm", ""))
    avoid_moves = move_tokens(operations.get("am", ""))
    identifier = unquote(operations.get("id", f"line-{line_number}"))
    # EPD supplies the four position fields; UCI position requires six-field FEN.
    fen = f"{board} {side} {castling} {en_passant} 0 1"
    return Position(fen, identifier, best_moves, avoid_moves, line_number)


def load_epd(path: Path) -> list[Position]:
    positions = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, 1):
            position = parse_epd_line(line, line_number)
            if position is not None:
                positions.append(position)
    if not positions:
        raise ValueError(f"{path}: no EPD positions")
    return positions


def answers_are_uci(position: Position) -> bool:
    answers = (*position.best_moves, *position.avoid_moves)
    return bool(answers) and all(UCI_MOVE_RE.fullmatch(move) for move in answers)


def solved(position: Position, bestmove: str) -> bool | None:
    if not answers_are_uci(position):
        return None
    move = bestmove.lower()
    if position.best_moves and move not in position.best_moves:
        return False
    if position.avoid_moves and move in position.avoid_moves:
        return False
    return True


class UciEngine:
    def __init__(self, command: Sequence[str], timeout: float):
        self.command = list(command)
        self.timeout = timeout
        self.process = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self.name = "unknown"
        self.author = "unknown"
        self.output: queue.Queue[str | None] = queue.Queue()
        self.reader = threading.Thread(target=self._pump_stdout, daemon=True)
        self.reader.start()

    def _pump_stdout(self) -> None:
        if self.process.stdout is not None:
            for line in self.process.stdout:
                self.output.put(line.rstrip("\r\n"))
        self.output.put(None)

    def send(self, command: str) -> None:
        if self.process.stdin is None:
            raise RuntimeError("engine stdin closed")
        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()

    def read_until(self, prefix: str) -> list[str]:
        deadline = time.monotonic() + self.timeout
        lines = []
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise TimeoutError(f"engine did not emit {prefix} within {self.timeout}s")
            try:
                value = self.output.get(timeout=remaining)
            except queue.Empty as error:
                raise TimeoutError(
                    f"engine did not emit {prefix} within {self.timeout}s"
                ) from error
            if value is None:
                stderr = self.process.stderr.read() if self.process.stderr else ""
                raise RuntimeError(f"engine exited before {prefix}: {stderr[-1000:]}")
            lines.append(value)
            if value.startswith(prefix):
                return lines

    def handshake(self, options: dict[str, str]) -> None:
        self.send("uci")
        for line in self.read_until("uciok"):
            if line.startswith("id name "):
                self.name = line[8:]
            elif line.startswith("id author "):
                self.author = line[10:]
        for name, value in options.items():
            self.send(f"setoption name {name} value {value}")
        self.send("isready")
        self.read_until("readyok")

    def search(self, position: Position, go_command: str) -> SearchResult:
        self.send("ucinewgame")
        self.send("isready")
        self.read_until("readyok")
        self.send(f"position fen {position.fen}")
        self.send(go_command)
        lines = self.read_until("bestmove ")
        result = SearchResult(bestmove="0000")
        for line in lines:
            if line.startswith("bestmove "):
                result.bestmove = line.split()[1]
            elif line.startswith("info "):
                for name, value in INFO_VALUE_RE.findall(line):
                    setattr(result, "time_ms" if name == "time" else name, int(value))
                score = SCORE_RE.search(line)
                if score:
                    result.score_kind = score.group(1)
                    result.score = int(score.group(2))
        return result

    def close(self) -> None:
        if self.process.poll() is None:
            try:
                self.send("quit")
                self.process.wait(timeout=2)
            except (BrokenPipeError, subprocess.TimeoutExpired):
                self.process.kill()
        if self.process.stdin:
            self.process.stdin.close()
        if self.process.stdout:
            self.process.stdout.close()
        if self.process.stderr:
            self.process.stderr.close()


def go_command(args: argparse.Namespace) -> str:
    if args.movetime is not None:
        return f"go movetime {args.movetime}"
    if args.depth is not None:
        return f"go depth {args.depth}"
    return f"go nodes {args.nodes}"


def run(args: argparse.Namespace) -> dict:
    positions = load_epd(args.epd)
    if args.limit:
        positions = positions[: args.limit]
    command = [str(args.engine), *args.engine_arg]
    engine = UciEngine(command, args.timeout)
    options = {
        "Threads": str(args.threads),
        "Hash": str(args.hash),
        "MultiPV": "1",
    }
    started = time.monotonic()
    rows = []
    try:
        engine.handshake(options)
        if not args.allow_adaptive and "adapter" in engine.name.lower():
            raise ValueError("adaptive/player-facing engine refused; use unchessed-reviewer")
        for index, position in enumerate(positions, 1):
            result = engine.search(position, go_command(args))
            status = solved(position, result.bestmove)
            rows.append(
                {
                    "index": index,
                    "id": position.identifier,
                    "line": position.line_number,
                    "bestmove": result.bestmove,
                    "expected_best": list(position.best_moves),
                    "expected_avoid": list(position.avoid_moves),
                    "solved": status,
                    "depth": result.depth,
                    "nodes": result.nodes,
                    "time_ms": result.time_ms,
                    "score_kind": result.score_kind,
                    "score": result.score,
                }
            )
            print(
                f"[{index}/{len(positions)}] {position.identifier}: {result.bestmove} "
                + ("PASS" if status is True else "FAIL" if status is False else "UNSCORED")
            )
    finally:
        engine.close()
    elapsed = time.monotonic() - started
    scored = [row for row in rows if row["solved"] is not None]
    passed = sum(row["solved"] is True for row in scored)
    return {
        "schema": 1,
        "engine": {
            "name": engine.name,
            "author": engine.author,
            "command": command,
            "binary_sha256": file_sha256(args.engine),
            "threads": args.threads,
            "hash_mb": args.hash,
            "multipv": 1,
        },
        "suite": {
            "path_name": args.epd.name,
            "sha256": file_sha256(args.epd),
            "positions": len(rows),
            "answer_format": "UCI coordinate moves only",
        },
        "limit": go_command(args),
        "summary": {
            "scored": len(scored),
            "unscored_non_uci_or_missing_answers": len(rows) - len(scored),
            "solved": passed,
            "score_percent": 100.0 * passed / len(scored) if scored else None,
            "wall_seconds": round(elapsed, 6),
            "reported_search_ms": sum(row["time_ms"] for row in rows),
            "reported_nodes": sum(row["nodes"] for row in rows),
        },
        "warning": "test-suite score is a regression diagnostic, not an Elo estimate",
        "positions": rows,
    }


def markdown(report: dict) -> str:
    summary = report["summary"]
    score = "n/a" if summary["score_percent"] is None else f"{summary['score_percent']:.2f}%"
    return "\n".join(
        [
            "# UCI EPD suite result",
            "",
            f"- Engine: `{report['engine']['name']}`",
            f"- Engine SHA-256: `{report['engine']['binary_sha256']}`",
            f"- Suite: `{report['suite']['path_name']}`",
            f"- Suite SHA-256: `{report['suite']['sha256']}`",
            f"- Limit: `{report['limit']}`",
            f"- Scored: {summary['scored']}",
            f"- Solved: {summary['solved']} ({score})",
            f"- Unscored answers: {summary['unscored_non_uci_or_missing_answers']}",
            f"- Wall time: {summary['wall_seconds']:.3f}s",
            "",
            f"> {report['warning']}.",
            "",
        ]
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--engine-arg", action="append", default=[])
    parser.add_argument("--epd", type=Path, required=True)
    limits = parser.add_mutually_exclusive_group(required=True)
    limits.add_argument("--movetime", type=int)
    limits.add_argument("--depth", type=int)
    limits.add_argument("--nodes", type=int)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--hash", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--allow-adaptive", action="store_true")
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.threads < 1 or args.hash < 1:
        raise ValueError("Threads and Hash must be positive")
    report = run(args)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(markdown(report), encoding="utf-8")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, TimeoutError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
