#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BUILD="$ROOT/build"
TEX="unchessed-research-guide.tex"
PDF="unchessed-research-guide.pdf"

if ! command -v pdflatex >/dev/null 2>&1; then
  echo "pdflatex is required (TeX Live 2024 or newer recommended)." >&2
  exit 1
fi

mkdir -p "$BUILD"
cd "$ROOT"
for pass in 1 2 3; do
  pdflatex \
    -interaction=nonstopmode \
    -halt-on-error \
    -file-line-error \
    -output-directory="$BUILD" \
    "$TEX"
done
cp "$BUILD/$PDF" "$ROOT/$PDF"
echo "Built $ROOT/$PDF"
