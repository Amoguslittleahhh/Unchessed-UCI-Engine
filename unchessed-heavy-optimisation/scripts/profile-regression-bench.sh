#!/usr/bin/env bash
set -euo pipefail
source "${HOME}/.cargo/env"
ROOT=/home/ubuntu/unchessed-research
SRC="$ROOT/unchessed-heavy-optimisation"
MAIN=/home/ubuntu/unchessed-main-inspect
WORK=/tmp/unchessed-profile-control
rm -rf "$WORK"
cp -a "$SRC" "$WORK"
python3 - "$WORK/Cargo.toml" <<'PY'
from pathlib import Path
p = Path(__import__('sys').argv[1])
s = p.read_text()
s = s.replace('lto = "fat"\n', 'lto = true\n')
s = s.replace('panic = "abort"\n', '').replace('strip = "symbols"\n', '').replace('incremental = false\n', '')
p.write_text(s)
PY
cargo build --release --manifest-path "$MAIN/Cargo.toml" -p unchessed-adapter >/tmp/main-profile-build.log
cargo build --release --manifest-path "$SRC/Cargo.toml" -p unchessed-adapter >/tmp/heavy-profile-build.log
cargo build --release --manifest-path "$WORK/Cargo.toml" -p unchessed-adapter >/tmp/plain-profile-build.log
printf 'variant\tdepth\tnodes\tnps\thashfull\n'
for spec in \
  "main:$MAIN/target/release/unchessed-adapter:$MAIN/unchessed-nnue.bin" \
  "heavy-profile:$SRC/target/release/unchessed-adapter:$SRC/unchessed-nnue.bin" \
  "plain-profile:$WORK/target/release/unchessed-adapter:$SRC/unchessed-nnue.bin"; do
  IFS=: read -r variant bin net <<<"$spec"
  out=$(mktemp)
  {
    printf 'uci\n';
    printf 'setoption name Threads value 1\n';
    printf 'setoption name Hash value 256\n';
    printf 'setoption name EvalFile value %s\n' "$net";
    printf 'setoption name Adaptive value false\n';
    printf 'setoption name OwnBook value false\n';
    printf 'isready\n';
    printf 'position startpos\n';
    printf 'go nodes 3000000\n';
    sleep 15;
    printf 'quit\n';
  } | "$bin" 2>/dev/null | tee "$out" >/dev/null
  line=$(grep 'info depth' "$out" | tail -1 || true)
  depth=$(awk '{for(i=1;i<=NF;i++)if($i=="depth"){print $(i+1)}}' <<<"$line")
  nodes=$(awk '{for(i=1;i<=NF;i++)if($i=="nodes"){print $(i+1)}}' <<<"$line")
  nps=$(awk '{for(i=1;i<=NF;i++)if($i=="nps"){print $(i+1)}}' <<<"$line")
  hashfull=$(awk '{for(i=1;i<=NF;i++)if($i=="hashfull"){print $(i+1)}}' <<<"$line")
  printf '%s\t%s\t%s\t%s\t%s\n' "$variant" "${depth:-NA}" "${nodes:-NA}" "${nps:-NA}" "${hashfull:-NA}"
  rm -f "$out"
done
printf '\nprofile-control-cargo:\n'
sed -n '1,20p' "$WORK/Cargo.toml"
