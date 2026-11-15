#!/usr/bin/env bash
# Configure, build and test a preset. Default: dev-cuda.
set -euo pipefail

PRESET="${1:-dev-cuda}"
cmake --preset "${PRESET}"
cmake --build --preset "${PRESET}" -j
ctest --preset "${PRESET}"