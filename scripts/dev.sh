#!/usr/bin/env bash
# Usage:
#   bash scripts/dev.sh --build            # (re)build image, then open a shell
#   bash scripts/dev.sh                    # open a shell in the container
#   bash scripts/dev.sh <command...>       # run a command in the container
#   NS_PUBLISH_PORT=8000 bash scripts/dev.sh <command...>   # also expose port 8000 on 127.0.0.1
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="neuroserve-dev:latest"

if [[ "${1:-}" == "--build" ]]; then
    docker build \
        -f "${ROOT}/docker/dev.Dockerfile" \
        --build-arg USER_UID="$(id -u)" \
        --build-arg USER_GID="$(id -g)" \
        -t "${IMAGE}" \
        "${ROOT}/docker"
    shift
fi

# Interactive flags only when attached to a terminal (so the script also works from other scripts and CI).
TTY_FLAGS=()
if [[ -t 0 && -t 1 ]]; then
    TTY_FLAGS=(-it)
fi

# Mount local model files read-only at /models when present.
MODEL_MOUNT=()
if [[ -d "${HOME}/models" ]]; then
    MODEL_MOUNT=(-v "${HOME}/models:/models:ro")
fi

# Publish a port to the host, on localhost only, when NS_PUBLISH_PORT is set.
PORT_ARGS=()
if [[ -n "${NS_PUBLISH_PORT:-}" ]]; then
    PORT_ARGS=(-p "127.0.0.1:${NS_PUBLISH_PORT}:${NS_PUBLISH_PORT}")
fi

if [[ $# -eq 0 ]]; then
    set -- bash
fi

exec docker run --rm "${TTY_FLAGS[@]}" --gpus all \
    "${MODEL_MOUNT[@]}" "${PORT_ARGS[@]}" \
    -v "${ROOT}:/workspace" -w /workspace \
    "${IMAGE}" "$@"
