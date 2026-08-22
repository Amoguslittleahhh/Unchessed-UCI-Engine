#!/usr/bin/env python3
"""Create a checksum-verified Unchessed release bundle.

Usage:
  python tools/package_release.py --target-dir target/release --output release
  python tools/package_release.py ... --require-policy
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_binary(directory, name):
    for candidate in (directory / name, directory / f"{name}.exe"):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"missing release binary {name} in {directory}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-dir", default="target/release")
    parser.add_argument("--output", required=True)
    parser.add_argument("--nnue", default="unchessed-nnue.bin")
    parser.add_argument("--policy", default="unchessed-maia.bin")
    parser.add_argument("--require-policy", action="store_true")
    args = parser.parse_args()

    target = Path(args.target_dir)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    sources = [
        find_binary(target, "unchessed-adapter"),
        find_binary(target, "unchessed-reviewer"),
        find_binary(target, "unchessed-datagen"),
        Path(args.nnue),
    ]
    policy = Path(args.policy)
    if policy.is_file():
        sources.append(policy)
    elif args.require_policy:
        raise FileNotFoundError(
            f"required policy sidecar {policy} is absent; refusing heuristic-only package"
        )

    manifest = {
        "format": 1,
        "policy": "included" if policy in sources else "absent (heuristic fallback)",
        "files": {},
    }
    for source in sources:
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = output / source.name
        shutil.copy2(source, destination)
        manifest["files"][destination.name] = {
            "bytes": destination.stat().st_size,
            "sha256": sha256(destination),
        }
    (output / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(output / "MANIFEST.json")


if __name__ == "__main__":
    main()
