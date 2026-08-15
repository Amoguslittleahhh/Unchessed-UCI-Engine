#!/bin/bash
# Startup script for the labeling-only CPU box (no GPU, no training here) --
# trimmed from cloud_startup.sh: skips the CUDA-enabled PyTorch install
# entirely (unneeded, ~1GB+ of downloads/installs saved), since this box's
# only job is running unchessed-datagen's labeling step.
set -x
exec > ~/startup.log 2>&1

apt-get update
apt-get install -y build-essential pkg-config libssl-dev zstd curl git

if ! command -v cargo > /dev/null 2>&1; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
fi
source "$HOME/.cargo/env"

curl -sSL https://raw.githubusercontent.com/verda-cloud/verda-cli/main/scripts/install.sh | sh

git clone https://github.com/Amoguslittleahhh/Unchessed-UCI-Engine.git ~/unchessed-kingsafety-src
cd ~/unchessed-kingsafety-src
cargo build --release -p unchessed-datagen

mkdir -p ~/unchessed-ai/data/maia-data/fresh
mkdir -p ~/unchessed-ai/data/maia-data/nnue

touch ~/STARTUP_DONE
echo "Startup setup complete (labeling-only, no torch/CUDA installed)." >> ~/startup.log
