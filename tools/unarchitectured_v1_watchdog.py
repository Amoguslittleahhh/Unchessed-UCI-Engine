#!/usr/bin/env python3
"""Supervise an Unarchitectured v1 job without human monitoring."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path

from unarchitectured_v1_safety import TrainingSafetyController, atomic_json


def gpu_snapshot() -> dict:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            capture_output=True,
            timeout=15,
            check=True,
        )
        return {"nvidia_smi": completed.stdout.strip().splitlines()}
    except Exception as error:  # watchdog diagnostics must never crash supervision
        return {"nvidia_smi_error": repr(error)}


def terminate_group(process: subprocess.Popen, grace: float) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def supervise(command, policy, heartbeat: Path, incident: Path) -> int:
    config = policy["training"]
    process = subprocess.Popen(command, start_new_session=True)
    started = time.monotonic()
    heartbeat_seen = False
    last_payload = None
    try:
        while process.poll() is None:
            now = time.monotonic()
            if heartbeat.exists():
                heartbeat_seen = True
                try:
                    last_payload = json.loads(heartbeat.read_text(encoding="utf-8"))
                    age = time.monotonic() - float(last_payload["monotonic_time"])
                    wall_age = time.time() - float(last_payload["unix_time"])
                    future_limit = config["maximum_future_clock_skew_seconds"]
                    if age < -future_limit or wall_age < -future_limit:
                        raise ValueError("heartbeat timestamp is implausibly in the future")
                except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
                    age = float("inf")
                    last_payload = {"heartbeat_error": repr(error)}
                if age > config["heartbeat_timeout_seconds"]:
                    reason = f"heartbeat stale for {age:.1f}s"
                    terminate_group(process, config["terminate_grace_seconds"])
                    atomic_json(
                        incident,
                        {
                            "schema": 1,
                            "action": "terminated",
                            "reason": reason,
                            "last_heartbeat": last_payload,
                            "gpu": gpu_snapshot(),
                            "command": command,
                        },
                    )
                    return 124
            elif now - started > config["startup_timeout_seconds"]:
                reason = "no heartbeat before startup timeout"
                terminate_group(process, config["terminate_grace_seconds"])
                atomic_json(
                    incident,
                    {
                        "schema": 1,
                        "action": "terminated",
                        "reason": reason,
                        "gpu": gpu_snapshot(),
                        "command": command,
                    },
                )
                return 124
            time.sleep(config["poll_interval_seconds"])
        returncode = int(process.returncode or 0)
        if not heartbeat_seen and heartbeat.exists():
            try:
                last_payload = json.loads(heartbeat.read_text(encoding="utf-8"))
                heartbeat_seen = (
                    last_payload.get("architecture") == "Unarchitectured v1"
                    and "unix_time" in last_payload
                    and "monotonic_time" in last_payload
                )
            except (OSError, ValueError, json.JSONDecodeError):
                heartbeat_seen = False
        if returncode == 0 and not heartbeat_seen:
            returncode = 125
            atomic_json(
                incident,
                {
                    "schema": 1,
                    "action": "missing_heartbeat",
                    "reason": "child exited successfully without proving liveness",
                    "gpu": gpu_snapshot(),
                    "command": command,
                },
            )
        elif returncode:
            atomic_json(
                incident,
                {
                    "schema": 1,
                    "action": "child_failed",
                    "returncode": returncode,
                    "last_heartbeat": last_payload,
                    "gpu": gpu_snapshot(),
                    "command": command,
                },
            )
        return returncode
    except BaseException:
        terminate_group(process, config["terminate_grace_seconds"])
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy", default="config/unarchitectured_v1_safety.json"
    )
    parser.add_argument("--heartbeat", required=True, type=Path)
    parser.add_argument("--incident", required=True, type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise SystemExit("watchdog requires a command after --")
    policy = json.loads(Path(args.policy).read_text(encoding="utf-8"))
    TrainingSafetyController(policy)  # validate policy before starting paid work
    raise SystemExit(supervise(command, policy, args.heartbeat, args.incident))


if __name__ == "__main__":
    main()
