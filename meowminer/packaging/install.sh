#!/bin/bash
# Rig-side installer invoked by mfarm's miner-downloader.sh with the install
# dir as $1 (default /opt/mfarm/miners/meowminer). Expects the bundle's wheels
# alongside this script.
set -euo pipefail

DEST="${1:-/opt/mfarm/miners/meowminer}"
HERE="$(cd "$(dirname "$0")" && pwd)"

mkdir -p "$DEST"
python3 -m venv "$DEST/venv"
"$DEST/venv/bin/pip" install --upgrade pip -q
# torch from PyPI (CUDA build); pinned major to avoid surprise ABI bumps
"$DEST/venv/bin/pip" install -q "torch>=2.2,<3" numpy blake3
"$DEST/venv/bin/pip" install -q "$HERE"/*.whl
ln -sf "$DEST/venv/bin/meowminer" "$DEST/meowminer"

echo "meowminer installed: $("$DEST/meowminer" --version)"
