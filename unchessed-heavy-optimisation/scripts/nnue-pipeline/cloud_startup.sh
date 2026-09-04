#!/bin/bash
# Verda startup script: runs automatically on first boot. Sets up everything
# full_pipeline_cloud.sh expects to already be in place, so the box is ready
# to run the pipeline the moment you SSH in -- no manual setup step needed.
# Logs to ~/startup.log; touches ~/STARTUP_DONE when finished (check for
# that file before running the pipeline, in case this is still mid-setup).
set -x
exec > ~/startup.log 2>&1

apt-get update
apt-get install -y build-essential pkg-config libssl-dev zstd curl git python3-venv python3-pip

# Rust toolchain (needed to build unchessed-datagen)
if ! command -v cargo > /dev/null 2>&1; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
fi
source "$HOME/.cargo/env"

# Verda CLI (for the manual `verda vm delete` step later, and VERDA_INSTANCE_ID lookups)
curl -sSL https://raw.githubusercontent.com/verda-cloud/verda-cli/main/scripts/install.sh | sh

# Clone + build the engine repo. Path matches what full_pipeline_cloud.sh
# expects as TRAIN_SRC/BIN.
git clone https://github.com/Amoguslittleahhh/Unchessed-UCI-Engine.git ~/unchessed-kingsafety-src
cd ~/unchessed-kingsafety-src
cargo build --release -p unchessed-datagen

# Python venv with CUDA-enabled torch, matching VENV in full_pipeline_cloud.sh
python3 -m venv ~/unchessed-ai/data/maia-venv
~/unchessed-ai/data/maia-venv/bin/pip install --upgrade pip
~/unchessed-ai/data/maia-venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu128
~/unchessed-ai/data/maia-venv/bin/pip install numpy

# Data directories full_pipeline_cloud.sh writes into (FRESH/NNUE_DIR/results)
mkdir -p ~/unchessed-ai/data/maia-data/fresh
mkdir -p ~/unchessed-ai/data/maia-data/nnue
mkdir -p ~/unchessed-ai/results/nnue_training

touch ~/STARTUP_DONE
echo "Startup setup complete." >> ~/startup.log
