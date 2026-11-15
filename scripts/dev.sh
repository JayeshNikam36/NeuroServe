#!/usr/bin/env bash
# Usage:
#   bash scripts/dev.sh --build            # (re)build image, then open a shell
#   bash scripts/dev.sh                    # open a shell in the container
#   bash scripts/dev.sh <command...>       # run a command in the container
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

if [[ $# -eq 0 ]]; then
    set -- bash
fi

exec docker run --rm "${TTY_FLAGS[@]}" --gpus all \
    -v "${ROOT}:/workspace" -w /workspace \
    "${IMAGE}" "$@"