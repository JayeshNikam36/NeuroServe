#!/usr/bin/env bash
# Release gate, run inside the dev container from a fresh clone:
#   bash scripts/dev.sh bash scripts/release_check.sh
# 1. configure, build and run every test (CPU, GPU, integration, Python) with the dev-cuda preset;
# 2. start the server on the real model and run scripts/release_smoke.py against it.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "== 1/2 build and every test (dev-cuda)"
bash scripts/build.sh dev-cuda

echo "== 2/2 live server (configs/dev.yaml)"
PYTHONPATH=python python3 -m neuroserve.server --config configs/dev.yaml > /tmp/release-server.log 2>&1 &
SERVER_PID=$!
trap 'kill "${SERVER_PID}" 2>/dev/null || true' EXIT
for _ in $(seq 1 120); do
    if python3 -c "import httpx; httpx.get('http://127.0.0.1:8000/health').raise_for_status()" 2>/dev/null; then
        break
    fi
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "the server exited during startup:"
        tail -20 /tmp/release-server.log
        exit 1
    fi
    sleep 1
done
PYTHONPATH=python python3 scripts/release_smoke.py
echo "== release check passed"
