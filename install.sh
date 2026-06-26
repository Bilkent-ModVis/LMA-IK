#!/usr/bin/env bash
# Vanilla-install setup for LMA-IK on macOS (11+) and Ubuntu (20.04+).
# Creates a Python 3.11 virtual environment in ./.venv and installs all
# dependencies declared in pyproject.toml. No paths need to be edited.
#
# CPU and Apple-Silicon (MPS) wheels of PyTorch install automatically from
# PyPI. For an NVIDIA CUDA build, install the matching torch wheel afterwards
# following https://pytorch.org/get-started/locally/.
set -euo pipefail
cd "$(dirname "$0")"

if command -v uv >/dev/null 2>&1; then
    uv venv --python 3.11 .venv
    uv pip install --python .venv -e ".[notebooks]"
else
    python3.11 -m venv .venv
    ./.venv/bin/python -m pip install --upgrade pip
    ./.venv/bin/python -m pip install -e ".[notebooks]"
fi

echo
echo "Setup complete. Reproduce the representative result with:"
echo "    ./.venv/bin/python reproduce.py"
