#!/usr/bin/env python3
"""NUMA-aware, resumable orchestration for 180-core Aegis v5 teacher labeling.

The orchestrator splits UNCHD4R0 human shards into deterministic ranges and
runs one persistent UCI teacher per pinned CPU set. It never shares a core set
between concurrent workers, validates memory/hash budgets before launch, and
writes per-task plus aggregate SHA-256 provenance manifests.
"""

from __future__ import annotations

import argparse
import dataclasses
import glob
import hashlib
import json
import os
import queue
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path

from a100_common import sha256_file
from aegis_v4_data import HEADER_BYTES, parse_header


@dataclasses.dataclass(frozen=True)
class Task:
    task_id: str
    input: str
    input_sha256: str
    start: int
    count: int
    output: str
    manifest: str
    log: str


def atomic_json(path: Path, payload: object) -> None:
    import tempfile

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=path.name, delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def parse_cpu_list(text: str) -> list[int]:
    output = []
    for field in text.strip().split(","):
        if not field:
            continue
        if "-" in field:
            first, last = (int(value) for value in field.split("-", 1))
            output.extend(range(first, last + 1))
        else:
            output.append(int(field))
    return sorted(set(output))


def available_cores() -> list[int]:
    if hasattr(os, "sched_getaffinity"):
        return sorted(os.sched_getaffinity(0))
    return list(range(os.cpu_count() or 1))


def physical_core_representatives(cores: list[int]) -> list[int]:
    available = set(cores)
    representatives = []
    seen: set[tuple[int, ...]] = set()
    for core in cores:
        path = Path(f"/sys/devices/system/cpu/cpu{core}/topology/thread_siblings_list")
        siblings = tuple(
            value for value in parse_cpu_list(path.read_text()) if value in available
        ) if path.exists() else (core,)
        if siblings not in seen:
            seen.add(siblings)
            representatives.append(min(siblings))
    return sorted(representatives)


def numa_groups(cores: list[int]) -> list[list[int]]:
    available = set(cores)
    groups = []
    for path in sorted(Path("/sys/devices/system/node").glob("node[0-9]*")):
        cpulist = path / "cpulist"
        if cpulist.exists():
            group = [value for value in parse_cpu_list(cpulist.read_text()) if value in available]
            if group:
                groups.append(group)
    return groups or [cores]


def allocate_core_sets(
    cores: list[int], cores_per_worker: int, requested_workers: int, use_numa: bool
) -> list[list[int]]:
    if cores_per_worker <= 0 or requested_workers <= 0:
        raise ValueError("cores_per_worker and requested_workers must be positive")
    groups = numa_groups(cores) if use_numa else [cores]
    chunks_by_node = [
        [group[index : index + cores_per_worker] for index in range(0, len(group), cores_per_worker)]
        for group in groups
    ]
    chunks_by_node = [
        [chunk for chunk in chunks if len(chunk) == cores_per_worker]
        for chunks in chunks_by_node
    ]
    output = []
    cursor = 0
    while len(output) < requested_workers and any(chunks_by_node):
        node = cursor % len(chunks_by_node)
        if chunks_by_node[node]:
            output.append(chunks_by_node[node].pop(0))
        cursor += 1
        if cursor > requested_workers * max(1, len(chunks_by_node)) * 4:
            break
    if len(output) < requested_workers:
        raise ValueError(
            f"requested {requested_workers} workers x {cores_per_worker} cores, "
            f"but only {len(output)} complete pinned sets are available"
        )
    return output


def memory_total_mb() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemTotal:"):
            return int(line.split()[1]) // 1024
    raise RuntimeError("could not read MemTotal")


def shard_count(path: Path) -> int:
    with path.open("rb") as handle:
        return parse_header(handle.read(HEADER_BYTES))


def load_config(path: str | Path) -> dict:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("schema") != 1:
        raise ValueError("v5 datagen config schema 1 required")
    return config


def config_sha256(config: dict) -> str:
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def expand_inputs(patterns: list[str]) -> list[Path]:
    paths = sorted({Path(path).resolve() for pattern in patterns for path in glob.glob(pattern)})
    if not paths:
        raise ValueError("input_globs matched no shards")
    return paths


def topology(config: dict) -> dict:
    cpu = config["cpu"]
    cores = available_cores()
    if cpu.get("physical_cores_only", False):
        cores = physical_core_representatives(cores)
    reserve = int(cpu.get("reserve_cores", 0))
    if reserve:
        if reserve >= len(cores):
            raise ValueError("reserve_cores leaves no worker cores")
        cores = cores[:-reserve]
    requested_cores = int(cpu["requested_cores"])
    if len(cores) < requested_cores:
        raise ValueError(
            f"configuration requires {requested_cores} cores, affinity exposes {len(cores)}"
        )
    cores = cores[:requested_cores]
    cores_per_worker = int(cpu["cores_per_worker"])
    workers = requested_cores // cores_per_worker
    sets = allocate_core_sets(cores, cores_per_worker, workers, cpu.get("numa_aware", True))
    teacher = config["teacher"]
    if int(teacher["threads"]) != cores_per_worker:
        raise ValueError("teacher threads must equal cores_per_worker to prevent oversubscription")
    estimated_mb = workers * (
        int(teacher["hash_mb"]) + int(teacher.get("estimated_process_mb", 64))
    )
    allowed_mb = int(memory_total_mb() * float(cpu.get("maximum_memory_fraction", 0.80)))
    if estimated_mb > allowed_mb:
        raise ValueError(
            f"estimated worker memory {estimated_mb:,} MiB exceeds configured allowance {allowed_mb:,} MiB"
        )
    return {
        "available_affinity_cores": available_cores(),
        "selected_cores": cores,
        "core_sets": sets,
        "workers": workers,
        "cores_per_worker": cores_per_worker,
        "numa_groups": numa_groups(cores),
        "memory_total_mb": memory_total_mb(),
        "estimated_worker_memory_mb": estimated_mb,
        "allowed_worker_memory_mb": allowed_mb,
    }


def build_plan(config: dict) -> dict:
    topo = topology(config)
    data = config["data"]
    output_dir = Path(data["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)
    records_per_task = int(data["records_per_task"])
    if records_per_task <= 0:
        raise ValueError("records_per_task must be positive")
    tasks = []
    inputs = []
    for input_path in expand_inputs(data["input_globs"]):
        count = shard_count(input_path)
        digest = sha256_file(input_path)
        inputs.append({"path": str(input_path), "records": count, "sha256": digest})
        for start in range(0, count, records_per_task):
            task_count = min(records_per_task, count - start)
            identity = f"{input_path}|{digest}|{start}|{task_count}|{config_sha256(config)}"
            task_id = hashlib.sha256(identity.encode()).hexdigest()[:20]
            stem = f"guide-{input_path.stem}-{start:012d}-{start + task_count:012d}-{task_id}"
            tasks.append(
                dataclasses.asdict(
                    Task(
                        task_id=task_id,
                        input=str(input_path),
                        input_sha256=digest,
                        start=start,
                        count=task_count,
                        output=str(output_dir / f"{stem}.aegis4"),
                        manifest=str(output_dir / f"{stem}.json"),
                        log=str(output_dir / "logs" / f"{stem}.log"),
                    )
                )
            )
    return {
        "schema": 1,
        "created_unix": int(time.time()),
        "config": config,
        "config_sha256": config_sha256(config),
        "topology": topo,
        "inputs": inputs,
        "tasks": tasks,
        "records": sum(task["count"] for task in tasks),
    }


def task_command(plan: dict, task: dict) -> list[str]:
    teacher = plan["config"]["teacher"]
    worker = Path(__file__).with_name("v5_uci_teacher_worker.py")
    command = [
        sys.executable,
        str(worker),
        "--engine",
        teacher["engine"],
        "--input",
        task["input"],
        "--input-sha256",
        task["input_sha256"],
        "--output",
        task["output"],
        "--manifest",
        task["manifest"],
        "--start",
        str(task["start"]),
        "--count",
        str(task["count"]),
        "--nodes-per-action",
        str(teacher["nodes_per_action"]),
        "--threads",
        str(teacher["threads"]),
        "--hash-mb",
        str(teacher["hash_mb"]),
        "--timeout",
        str(teacher.get("timeout_seconds", 120)),
        "--resume",
    ]
    for value in teacher.get("engine_args", []):
        command.extend(("--engine-arg", str(value)))
    for name, value in teacher.get("options", {}).items():
        command.extend(("--option", f"{name}={value}"))
    for asset in teacher.get("assets", []):
        command.extend(("--asset", str(asset)))
    if not teacher.get("clear_hash_per_action", True):
        command.append("--no-clear-hash")
    return command


def completed_task(task: dict) -> bool:
    output = Path(task["output"])
    manifest = Path(task["manifest"])
    if not output.exists() or not manifest.exists():
        return False
    try:
        metadata = json.loads(manifest.read_text(encoding="utf-8"))
        return metadata.get("output_sha256") == sha256_file(output)
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def run_plan(plan: dict, dry_run: bool = False) -> dict:
    core_sets = plan["topology"]["core_sets"]
    pending = [task for task in plan["tasks"] if not completed_task(task)]
    if dry_run:
        return {
            "workers": len(core_sets),
            "pending": len(pending),
            "example": shlex.join(task_command(plan, pending[0])) if pending else None,
        }
    work_queue: queue.Queue = queue.Queue()
    for task in pending:
        work_queue.put(task)
    failures = []
    completed = 0
    lock = threading.Lock()
    stop = threading.Event()

    def slot(core_set):
        nonlocal completed
        while not stop.is_set():
            try:
                task = work_queue.get_nowait()
            except queue.Empty:
                return
            command = task_command(plan, task)
            env = os.environ.copy()
            env.update(
                {
                    "OMP_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "RAYON_NUM_THREADS": "1",
                }
            )

            def pin_child():
                os.sched_setaffinity(0, set(core_set))

            Path(task["log"]).parent.mkdir(parents=True, exist_ok=True)
            with open(task["log"], "w", encoding="utf-8") as log:
                process = subprocess.run(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=env,
                    preexec_fn=pin_child if hasattr(os, "sched_setaffinity") else None,
                )
            with lock:
                if process.returncode:
                    failures.append(
                        {
                            "task_id": task["task_id"],
                            "returncode": process.returncode,
                            "log": task["log"],
                        }
                    )
                    stop.set()
                else:
                    completed += 1
            work_queue.task_done()

    threads = [threading.Thread(target=slot, args=(core_set,), daemon=False) for core_set in core_sets]
    started = time.monotonic()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    result = {
        "workers": len(core_sets),
        "already_complete": len(plan["tasks"]) - len(pending),
        "completed_this_run": completed,
        "remaining": work_queue.qsize(),
        "failures": failures,
        "elapsed_seconds": time.monotonic() - started,
    }
    if failures:
        raise RuntimeError(json.dumps(result, indent=2))
    return result


def status(plan: dict) -> dict:
    complete = sum(completed_task(task) for task in plan["tasks"])
    return {
        "tasks": len(plan["tasks"]),
        "complete": complete,
        "pending": len(plan["tasks"]) - complete,
        "records_planned": plan["records"],
    }


def finalize(plan: dict) -> dict:
    missing = [task["task_id"] for task in plan["tasks"] if not completed_task(task)]
    if missing:
        raise ValueError(f"cannot finalize: {len(missing)} tasks are incomplete")
    task_manifests = [json.loads(Path(task["manifest"]).read_text()) for task in plan["tasks"]]
    return {
        "schema": 1,
        "config_sha256": plan["config_sha256"],
        "engine_sha256": sorted({item["engine_sha256"] for item in task_manifests}),
        "assets": sorted(
            {
                (asset["path"], asset["sha256"], asset["bytes"])
                for item in task_manifests
                for asset in item.get("assets", [])
            }
        ),
        "input_shards": plan["inputs"],
        "output_shards": [
            {
                "path": item["output"],
                "sha256": item["output_sha256"],
                "records": item["records"],
                "legal_actions": item["legal_actions"],
            }
            for item in task_manifests
        ],
        "records": sum(item["records"] for item in task_manifests),
        "legal_actions": sum(item["legal_actions"] for item in task_manifests),
        "worker_seconds": sum(item["elapsed_seconds"] for item in task_manifests),
        "clear_hash_per_action": all(item["clear_hash_per_action"] for item in task_manifests),
        "nodes_per_action": sorted({item["nodes_per_action"] for item in task_manifests}),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    topology_parser = subparsers.add_parser("topology")
    topology_parser.add_argument("--config", required=True)
    plan_parser = subparsers.add_parser("plan")
    plan_parser.add_argument("--config", required=True)
    plan_parser.add_argument("--output", required=True, type=Path)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--plan", required=True, type=Path)
    run_parser.add_argument("--dry-run", action="store_true")
    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("--plan", required=True, type=Path)
    finalize_parser = subparsers.add_parser("finalize")
    finalize_parser.add_argument("--plan", required=True, type=Path)
    finalize_parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if args.command == "topology":
        result = topology(load_config(args.config))
    elif args.command == "plan":
        result = build_plan(load_config(args.config))
        atomic_json(args.output, result)
    else:
        plan = json.loads(args.plan.read_text(encoding="utf-8"))
        if args.command == "run":
            result = run_plan(plan, args.dry_run)
        elif args.command == "status":
            result = status(plan)
        else:
            result = finalize(plan)
            atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
