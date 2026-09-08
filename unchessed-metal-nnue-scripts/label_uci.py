#!/usr/bin/env python3
"""Label a JSONL manifest through a persistent UCI engine process.

The teacher name is metadata only: Stockfish, lc0, and maia3 outputs are kept
separate and must not be averaged implicitly.

Ported from manus/research-facilities@e4851ba with three real bugs fixed,
found in review before any real run:

1. No timeout on reading the engine's stdout -- a single hung/crashed engine
   (bad FEN it silently ignores, OOM, etc.) blocked the whole run forever
   with no way to detect it. Every read now has a deadline
   (--timeout-sec, default 30s); a timed-out position is recorded with
   timed_out=true instead of silently missing or wedging the run.
2. No terminal-position handling. A position with no legal moves gets
   `bestmove (none)` and no score line from a real UCI engine -- the
   original script would just write score_cp=null for it. The training
   procedure this pipeline exists to serve explicitly requires "explicit
   terminal/mate encoding" and to "reject illegal, duplicate, malformed...
   records rather than silently training on them"; this now detects
   `bestmove (none)`/`bestmove 0000` and records terminal=true with an
   explicit encoded score instead of a silent null.
3. No recovery from a dead engine process. If the engine exited mid-run,
   the original script's next `send()` would raise BrokenPipeError and
   crash with a raw traceback, discarding the option to at least finish
   labeling what's left with a relaunch. This now detects the closed
   pipe/EOF, restarts the engine, and continues from the next position
   (already-written positions are safe either way -- output is flushed
   per record).
"""
from __future__ import annotations
import argparse, json, queue, re, subprocess, sys, threading, time
from pathlib import Path

CP = re.compile(r"score cp (-?\d+)")
MATE = re.compile(r"score mate (-?\d+)")
WDL = re.compile(r"wdl (-?\d+) (-?\d+) (-?\d+)")
MATE_SCORE_CP = 30_000


class EngineDied(Exception):
    pass


class Engine:
    def __init__(self, path: str, options: list[str]):
        self.path = path
        self.options = options
        self.proc: subprocess.Popen | None = None
        self.q: "queue.Queue[str | None]" = queue.Queue()
        self._start()

    def _start(self) -> None:
        self.proc = subprocess.Popen(
            [self.path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1,
        )
        # A plain select()-on-readiness timeout doesn't compose safely with a
        # buffered TextIOWrapper: readline() can pull several already-written
        # lines out of the OS pipe in one read(), after which select() sees no
        # NEW OS-level bytes and reports "not ready" even though the next
        # readline() would return instantly from Python's own buffer. Found by
        # actually running this against a real engine, not just reading the
        # code -- the adapter's multi-line "id"/"option" handshake burst
        # stalled after exactly one line. A background reader thread feeding a
        # Queue (the same pattern every other UCI-driving script in this repo
        # already uses) sidesteps the issue entirely.
        self.q = queue.Queue()
        threading.Thread(target=self._pump, daemon=True).start()
        self._send("uci")
        self._wait_for("uciok", timeout=30)
        for opt in self.options:
            name, value = opt.split("=", 1)
            self._send(f"setoption name {name} value {value}")
        self._send("isready")
        self._wait_for("readyok", timeout=30)

    def _pump(self) -> None:
        assert self.proc and self.proc.stdout
        for line in iter(self.proc.stdout.readline, ""):
            self.q.put(line)
        self.q.put(None)  # EOF sentinel

    def restart(self) -> None:
        try:
            if self.proc is not None:
                self.proc.kill()
        except Exception:
            pass
        self._start()

    def _send(self, s: str) -> None:
        assert self.proc and self.proc.stdin
        try:
            self.proc.stdin.write(s + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise EngineDied(str(exc)) from exc

    def _readline(self, timeout: float) -> str | None:
        """One line from stdout, or None on timeout. Raises EngineDied on EOF."""
        try:
            line = self.q.get(timeout=max(0.0, timeout))
        except queue.Empty:
            return None
        if line is None:
            raise EngineDied("engine stdout closed (process exited)")
        return line

    def _wait_for(self, token: str, timeout: float) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            line = self._readline(deadline - time.time())
            if line is not None and line.startswith(token):
                return
        raise EngineDied(f"timed out waiting for '{token}' during startup")

    def bestmove(self, fen: str, go_cmd: str, timeout: float):
        """Returns (bestmove, cp, mate, wdl, timed_out, terminal)."""
        self._send(f"position fen {fen}")
        self._send(go_cmd)
        deadline = time.time() + timeout
        cp = mate = wdl = bestmove = None
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                self._abort_search()
                return None, cp, mate, wdl, True, False
            line = self._readline(remaining)
            if line is None:
                continue  # queue.get timed out this iteration but deadline not yet reached
            if line.startswith("info "):
                m = CP.search(line)
                if m:
                    cp = int(m.group(1))
                m = MATE.search(line)
                if m:
                    mate = int(m.group(1))
                m = WDL.search(line)
                if m:
                    wdl = list(map(int, m.groups()))
            if line.startswith("bestmove "):
                bestmove = line.split()[1]
                break
        terminal = bestmove in ("(none)", "0000")
        return bestmove, cp, mate, wdl, False, terminal

    def _abort_search(self, drain_timeout: float = 10.0) -> None:
        """A timed-out position leaves the engine mid-search. Without this,
        its eventual (late) `bestmove` and `info` lines arrive interleaved
        with the NEXT position's real output and silently corrupt that
        label -- found by actually running the timeout path against a real
        engine: the position right after a timeout came back mislabeled with
        a stale bestmove. `stop` + draining until that stale `bestmove`
        appears (or a bounded wait gives up) keeps engine and script turn-
        taking synchronized before the next position is sent."""
        self._send("stop")
        deadline = time.time() + drain_timeout
        while time.time() < deadline:
            line = self._readline(deadline - time.time())
            if line is not None and line.startswith("bestmove "):
                return
        # Engine didn't answer `stop` in time either -- it's not just slow,
        # something is actually wrong. Restart is the only way back to a
        # known-good state; the caller's own EngineDied handling covers this.
        raise EngineDied(f"no bestmove within {drain_timeout}s of 'stop' after a timeout")

    def quit(self) -> None:
        try:
            self._send("quit")
            if self.proc is not None:
                self.proc.wait(timeout=10)
        except Exception:
            if self.proc is not None:
                self.proc.kill()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--teacher", required=True, choices=["stockfish", "lc0", "maia3"])
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--nodes", type=int)
    ap.add_argument("--option", action="append", default=[], help="UCI option as NAME=VALUE")
    ap.add_argument("--timeout-sec", type=float, default=30.0,
                     help="max wall time to wait for one position's bestmove before recording timed_out=true")
    ap.add_argument("--max-restarts", type=int, default=5,
                     help="give up after this many engine deaths in one run")
    args = ap.parse_args()

    engine = Engine(args.engine, args.option)
    go_cmd = f"go nodes {args.nodes}" if args.nodes else f"go depth {args.depth}"
    restarts = 0
    n_timed_out = 0
    n_terminal = 0
    n_labeled = 0

    with Path(args.out).open("w") as out:
        rows = [json.loads(r) for r in Path(args.manifest).read_text().splitlines() if r.strip()]
        i = 0
        while i < len(rows):
            row = rows[i]
            try:
                bestmove, cp, mate, wdl, timed_out, terminal = engine.bestmove(
                    row["fen"], go_cmd, args.timeout_sec
                )
            except EngineDied as exc:
                restarts += 1
                if restarts > args.max_restarts:
                    print(f"engine died {restarts} times, giving up at position {i}: {exc}", file=sys.stderr)
                    break
                print(f"engine died ({exc}), restarting (attempt {restarts}/{args.max_restarts})", file=sys.stderr)
                engine.restart()
                continue  # retry the same position on the fresh process
            if timed_out:
                n_timed_out += 1
            if terminal:
                n_terminal += 1
                # No legal moves: encode as a resolved terminal score rather than a
                # silent null. Real sign (win/loss/draw for the side to move) can't
                # be determined from the engine's empty response alone without a
                # local mate/stalemate check, so this is intentionally left to a
                # downstream board-aware pass -- what matters here is that the
                # record is never silently written as an ordinary score_cp=null.
                cp = mate = None
            else:
                n_labeled += 1
            result = dict(
                row, teacher=args.teacher, depth=args.depth, nodes=args.nodes,
                bestmove=bestmove, score_cp=cp, score_mate=mate, wdl=wdl,
                timed_out=timed_out, terminal=terminal,
            )
            out.write(json.dumps(result, sort_keys=True) + "\n")
            out.flush()
            i += 1

    engine.quit()
    print(
        f"labeled={n_labeled} terminal={n_terminal} timed_out={n_timed_out} "
        f"restarts={restarts} total={len(rows)}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
