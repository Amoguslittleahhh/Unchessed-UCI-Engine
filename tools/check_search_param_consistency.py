#!/usr/bin/env python3
"""Check that the advertised UCI option surface matches the code defaults.

The engine's tunables live in two places that must agree:

  * the Rust defaults: `SearchParams::default()` (search.rs),
    `EvalParams::default()` (eval.rs), and the `Options` literal (uci.rs);
  * the UCI advertisement + `setoption` handler in uci.rs (the default a
    GUI sees, the min/max it offers, and the clamp actually applied).

A drift between the two means the shipped behavior no longer matches the
config an SPRT campaign was run with (the SPSA/tuning work order in
docs/parameter-calibration-audit.md relies on advertised bounds being the
tuning range). This tool cross-checks all three for every tunable:

  1. advertised spin default == struct default (field mapped by name);
  2. advertised default within advertised [min, max];
  3. advertised [min, max] == the handler's `v.clamp(lo, hi)`;
  4. check-option default (ProbcutSeeFilter) == struct bool default;
  5. UnarchitecturedMinTime: advertised default == the Options literal in
     uci.rs, and the handler clamp == advertised min/max.

Stdlib only; runs from a fresh clone. Exit 0 when everything matches, 1 on
any drift (each drift named on stdout).

Usage:
  python3 tools/check_search_param_consistency.py [--repo PATH] [--json]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# UCI option name (lowercase, as matched in the setoption handler) ->
# struct field in SearchParams::default() / EvalParams::default().
SEARCH_PARAM_FIELDS = {
    "rfpmargin": "rfp_margin",
    "nullmovebase": "nm_base",
    "nullmovedivisor": "nm_divisor",
    "lmrmindepth": "lmr_min_depth",
    "lmrminmovenumber": "lmr_min_movenum",
    "lmrbigmovenumber": "lmr_big_movenum",
    "aspirationdelta": "aspiration_delta",
    "aspirationmindepth": "aspiration_min_depth",
    "probcutmargin": "probcut_margin",
    "probcutreduction": "probcut_reduction",
    "probcutmindepth": "probcut_min_depth",
    "futilitymargin": "futility_margin",
    "futilitymaxdepth": "futility_max_depth",
}
EVAL_PARAM_FIELDS = {
    "passedpawnmgpct": "passed_mg_pct",
    "passedpawnegpct": "passed_eg_pct",
    "mobilitypct": "mobility_pct",
    "rookpct": "rook_pct",
    "knightoutpostpct": "knight_outpost_pct",
}
CHECK_PARAM_FIELDS = {
    "probcutseefilter": "probcut_see_filter",
}
MINTIME_OPTION = "unarchitecturedmintime"
MINTIME_FIELD = "unarchitectured_min_time_ms"


def find_repo_root(start: Path) -> Path:
    for d in [start, *start.parents]:
        if (d / "unchessed-core" / "src" / "uci.rs").is_file():
            return d
    raise FileNotFoundError("cannot locate unchessed-core/src/uci.rs from " + str(start))


def parse_default_block(src: str, impl_name: str) -> dict[str, int | bool]:
    """Extract `field: literal` pairs from `impl Default for <impl_name>`."""
    m = re.search(r"impl Default for " + impl_name + r"\s*\{", src)
    if not m:
        raise ValueError(f"impl Default for {impl_name} not found")
    body = src[m.end():]
    depth = 1
    for i, ch in enumerate(body):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                body = body[:i]
                break
    out: dict[str, int | bool] = {}
    for fm in re.finditer(r"^\s*(\w+)\s*:\s*(\d[\d_]*)\s*,", body, re.M):
        out[fm.group(1)] = int(fm.group(2).replace("_", ""))
    for bm in re.finditer(r"^\s*(\w+)\s*:\s*(true|false)\s*,", body, re.M):
        out[bm.group(1)] = bm.group(2) == "true"
    return out


def parse_uci_advertised(src: str) -> dict[str, dict]:
    """name -> {'type': 'spin'|'check', 'default': ..., 'min': ..., 'max': ...}"""
    out: dict[str, dict] = {}
    for m in re.finditer(
        r"option name (\w+) type spin default (\d+) min (\d+) max (\d+)", src
    ):
        out[m.group(1).lower()] = {
            "type": "spin",
            "default": int(m.group(2)),
            "min": int(m.group(3)),
            "max": int(m.group(4)),
        }
    for m in re.finditer(r"option name (\w+) type check default (true|false)", src):
        out[m.group(1).lower()] = {
            "type": "check",
            "default": m.group(2) == "true",
        }
    return out


def parse_handler_clamps(src: str) -> dict[str, tuple[int, int]]:
    """lowercase handler key -> (clamp lo, clamp hi).

    The clamp sits inside a nested `if let Ok(v) = ... { }` block, so a
    brace-excluding scan between the handler's `{` and the call would stop
    at the `if`'s own brace; instead scan a window after each handler key.
    """
    wanted = set(SEARCH_PARAM_FIELDS) | set(EVAL_PARAM_FIELDS) | {MINTIME_OPTION}
    out: dict[str, tuple[int, int]] = {}
    for m in re.finditer(r'"(\w+)"\s*=>', src):
        key = m.group(1).lower()
        if key not in wanted:
            continue
        window = src[m.end(): m.end() + 300]
        cm = re.search(r"\.clamp\(\s*(\d[\d_]*)\s*,\s*(\d[\d_]*)\s*\)", window)
        if cm:
            out[key] = (int(cm.group(1).replace("_", "")), int(cm.group(2).replace("_", "")))
    return out


def parse_mintime_struct_default(src: str) -> int | None:
    m = re.search(r"unarchitectured_min_time_ms\s*:\s*(\d[\d_]*)", src)
    return int(m.group(1).replace("_", "")) if m else None


def check(repo: Path) -> dict:
    search_rs = (repo / "unchessed-core/src/search.rs").read_text()
    eval_rs = (repo / "unchessed-core/src/eval.rs").read_text()
    uci_rs = (repo / "unchessed-core/src/uci.rs").read_text()

    search_defaults = parse_default_block(search_rs, "SearchParams")
    eval_defaults = parse_default_block(eval_rs, "EvalParams")
    advertised = parse_uci_advertised(uci_rs)
    clamps = parse_handler_clamps(uci_rs)
    mintime_struct = parse_mintime_struct_default(uci_rs)

    params: dict[str, dict] = {}
    failures: list[str] = []

    for name, field in {**SEARCH_PARAM_FIELDS, **EVAL_PARAM_FIELDS}.items():
        struct_defaults = search_defaults if field in search_defaults else eval_defaults
        rec: dict = {"field": field}
        adv = advertised.get(name)
        if adv is None:
            failures.append(f"{name}: no UCI advertisement found")
            params[name] = rec
            continue
        rec["advertised"] = adv
        struct_val = struct_defaults.get(field)
        rec["struct_default"] = struct_val
        if struct_val is None:
            failures.append(f"{name}: struct field {field} not found in Default impl")
            params[name] = rec
            continue
        if adv["default"] != struct_val:
            failures.append(
                f"{name}: advertised default {adv['default']} != struct default {struct_val}"
            )
        if not (adv["min"] <= adv["default"] <= adv["max"]):
            failures.append(
                f"{name}: advertised default {adv['default']} outside [{adv['min']}, {adv['max']}]"
            )
        clamp = clamps.get(name)
        rec["clamp"] = clamp
        if clamp is None:
            failures.append(f"{name}: no v.clamp() found in setoption handler")
        elif tuple(clamp) != (adv["min"], adv["max"]):
            failures.append(
                f"{name}: handler clamp {clamp} != advertised [{adv['min']}, {adv['max']}]"
            )
        elif not (clamp[0] <= struct_val <= clamp[1]):
            failures.append(f"{name}: struct default {struct_val} outside handler clamp {clamp}")
        params[name] = rec

    for name, field in CHECK_PARAM_FIELDS.items():
        rec: dict = {"field": field}
        adv = advertised.get(name)
        struct_val = search_defaults.get(field)
        rec["advertised"] = adv
        rec["struct_default"] = struct_val
        if adv is None or adv["type"] != "check":
            failures.append(f"{name}: UCI check option not found")
        elif adv["default"] != struct_val:
            failures.append(
                f"{name}: advertised default {adv['default']} != struct default {struct_val}"
            )
        params[name] = rec

    # UnarchitecturedMinTime: advertised vs uci.rs Options literal vs clamp.
    rec: dict = {"field": MINTIME_FIELD}
    adv = advertised.get(MINTIME_OPTION)
    rec["advertised"] = adv
    rec["struct_default"] = mintime_struct
    if adv is None:
        failures.append(f"{MINTIME_OPTION}: no UCI advertisement found")
    elif mintime_struct is None:
        failures.append(f"{MINTIME_OPTION}: Options literal default not found in uci.rs")
    elif adv["default"] != mintime_struct:
        failures.append(
            f"{MINTIME_OPTION}: advertised default {adv['default']} != "
            f"Options literal {mintime_struct}"
        )
    clamp = clamps.get(MINTIME_OPTION)
    rec["clamp"] = clamp
    if adv is not None:
        if clamp is None:
            failures.append(f"{MINTIME_OPTION}: no v.clamp() found in setoption handler")
        elif tuple(clamp) != (adv["min"], adv["max"]):
            failures.append(
                f"{MINTIME_OPTION}: handler clamp {clamp} != advertised "
                f"[{adv['min']}, {adv['max']}]"
            )
    params[MINTIME_OPTION] = rec

    return {
        "repo": str(repo),
        "checked": len(params),
        "params": params,
        "failures": failures,
    }


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--repo", type=Path, default=None, help="repo root (default: search upward from cwd)")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = argument_parser().parse_args(argv)
    repo = args.repo or find_repo_root(Path.cwd())
    try:
        result = check(repo)
    except (FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        for name, rec in result["params"].items():
            adv = rec.get("advertised") or {}
            if adv.get("type") == "spin":
                spec = f"default={adv.get('default')} advertised=[{adv.get('min')}..{adv.get('max')}] clamp={rec.get('clamp')}"
            else:
                spec = f"default={adv.get('default')} struct={rec.get('struct_default')}"
            status = "OK   "
            if any(name in f for f in result["failures"]):
                status = "DRIFT"
            print(f"{status} {name:24s} ({rec.get('field')}): {spec}")
        for f in result["failures"]:
            print(f"DRIFT {f}")
        print(
            f"{result['checked']} options checked, {len(result['failures'])} drift(s)"
        )
    return 1 if result["failures"] else 0


if __name__ == "__main__":
    sys.exit(main())
