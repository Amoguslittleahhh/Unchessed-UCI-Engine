#!/usr/bin/env bash
set -euo pipefail
source "${HOME}/.cargo/env"
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

if [[ $# -ne 1 || ! -s "$1" ]]; then
  echo "usage: $0 /absolute/path/to/unchessed-nnue.bin" >&2
  echo "Refusing to benchmark the hand-crafted fallback; pass the exact NNUE file explicitly." >&2
  exit 2
fi
EVAL_FILE=$(realpath "$1")
EVAL_SHA256=$(sha256sum "$EVAL_FILE" | awk '{print $1}')
PORTABLE_TARGET="$ROOT/target-bench-portable"
V3_TARGET="$ROOT/target-bench-v3"
mkdir -p "$ROOT/benchmarks/results"

cargo build --release --workspace --target-dir "$PORTABLE_TARGET" >/dev/null
RUSTFLAGS='-C target-cpu=x86-64-v3' cargo build --release --workspace --target-dir "$V3_TARGET" >/dev/null

PORTABLE="$PORTABLE_TARGET/release/unchessed-adapter"
V3="$V3_TARGET/release/unchessed-adapter"
OUT="$ROOT/benchmarks/results/portable-v3-$(date +%Y%m%d-%H%M%S).tsv"
printf 'build\thash_mb\tfen\teval_file\teval_sha256\tnodes\ttime_ms\tnps\tmax_rss_kb\tbestmove\n' > "$OUT"

FENS=(
  'startpos'
  'fen r3k2r/p1ppqpb1/bn2pnp1/2pP4/1p2P3/2N2N2/PPQBBPPP/R3K2R w KQkq - 0 1'
  'fen 4rrk1/pp1b1ppp/2n1p3/2qpP3/3N4/2P1B3/PPQ2PPP/R4RK1 w - - 0 16'
  'fen 8/5pk1/6p1/3p4/3P1P2/5KP1/8/8 w - - 0 40'
)
HASHES=(4 8 16 32 64)
for build in portable v3; do
  exe="$PORTABLE"
  [[ "$build" == v3 ]] && exe="$V3"
  for hash in "${HASHES[@]}"; do
    for fen in "${FENS[@]}"; do
      if [[ "$fen" == startpos ]]; then
        pos='position startpos'
      else
        pos="position $fen"
      fi
      tmp=$(mktemp)
      {
        /usr/bin/time -f 'RSS=%M' sh -c "{ printf 'uci\\nsetoption name Threads value 1\\nsetoption name Hash value $hash\\nsetoption name EvalFile value $EVAL_FILE\\nsetoption name Adaptive value false\\nsetoption name OwnBook value false\\nisready\\n$pos\\ngo nodes 500000\\n'; sleep 2; printf 'quit\\n'; } | '$exe'"
      } >"$tmp" 2>&1 || true
      info=$(grep '^info .*nodes ' "$tmp" | tail -1 || true)
      nodes=$(printf '%s\n' "$info" | sed -n 's/.* nodes \([0-9][0-9]*\) .*/\1/p')
      time_ms=$(printf '%s\n' "$info" | sed -n 's/.* time \([0-9][0-9]*\) .*/\1/p')
      nps=$(printf '%s\n' "$info" | sed -n 's/.* nps \([0-9][0-9]*\) .*/\1/p')
      rss=$(sed -n 's/^RSS=//p' "$tmp" | tail -1 || true)
      best=$(grep '^bestmove ' "$tmp" | tail -1 | awk '{print $2}' || true)
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$build" "$hash" "$fen" "$EVAL_FILE" "$EVAL_SHA256" "${nodes:-0}" "${time_ms:-0}" "${nps:-0}" "${rss:-0}" "${best:-0000}" >> "$OUT"
      rm -f "$tmp"
    done
  done
done
printf '%s\n' "$OUT"
