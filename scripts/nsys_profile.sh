#!/usr/bin/env bash
# Nsight Systems timeline of the server under the step-profile scenarios (docs/profiling.md):
#   bash scripts/dev.sh bash scripts/nsys_profile.sh [scenarios]      (default: decode,prefill)
# Starts the server from configs/dev.yaml under `nsys profile` (CUDA API, kernels, NVTX ranges of
# the engine step's phases), runs neuroserve.bench.step_profile against it, stops it, and exports
# summaries. Output: benchmarks/results/raw/nsys-<time>/ (neuroserve.nsys-rep opens in the Nsight
# Systems GUI; stats_*.csv; steps/ is the step profile of the same run).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH=python

NSYS="${NSYS:-$(command -v nsys || true)}"
if [[ -z "${NSYS}" ]]; then
    echo "nsys (Nsight Systems CLI) not found. In the dev container: apt-get update && apt-get install -y" \
         "cuda-nsight-systems-12-4, or set NSYS=/path/to/nsys. See docs/profiling.md." >&2
    exit 2
fi
SCENARIOS="${1:-decode,prefill}"
PORT="${NS_PROFILE_PORT:-8101}"
OUT="benchmarks/results/raw/nsys-$(date +%Y%m%d-%H%M%S)"
mkdir -p "${OUT}"
echo "== nsys: $("${NSYS}" --version)"

# The dev config on its own port, with room to queue a whole scenario at once.
python3 - "${OUT}/config.yaml" "${PORT}" <<'PY'
import sys
import yaml
cfg = yaml.safe_load(open("configs/dev.yaml", encoding="utf-8"))
cfg.setdefault("server", {}).update(host="127.0.0.1", port=int(sys.argv[2]))
cfg.setdefault("engine", {}).update(queue_timeout_ms=0, max_queue=1024)
yaml.safe_dump(cfg, open(sys.argv[1], "w", encoding="utf-8"), sort_keys=False)
PY

"${NSYS}" profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none --force-overwrite=true \
    --output="${OUT}/neuroserve" python3 -m neuroserve.server --config "${OUT}/config.yaml" \
    > "${OUT}/server.log" 2>&1 &
NSYS_PID=$!
stop_server() {  # SIGINT to the server (it shuts down cleanly; nsys then writes the report), never to nsys
    local pid
    for pid in $(pgrep -f "neuroserve.server --config ${OUT}/config.yaml" || true); do
        if [[ "${pid}" != "${NSYS_PID}" ]]; then
            kill -INT "${pid}" 2>/dev/null || true
        fi
    done
}
trap stop_server EXIT

for _ in $(seq 1 300); do
    if python3 -c "import httpx; httpx.get('http://127.0.0.1:${PORT}/health').raise_for_status()" 2>/dev/null; then
        break
    fi
    if ! kill -0 "${NSYS_PID}" 2>/dev/null; then
        echo "the server exited during startup:" >&2
        tail -20 "${OUT}/server.log" >&2
        exit 1
    fi
    sleep 1
done

echo "== step profile under nsys (${SCENARIOS})"
python3 -m neuroserve.bench.step_profile --url "http://127.0.0.1:${PORT}" --config "${OUT}/config.yaml" \
    --scenarios "${SCENARIOS}" --out-dir "${OUT}/steps"

echo "== stopping the server; nsys writes the report"
stop_server
trap - EXIT
wait "${NSYS_PID}" || true
if [[ ! -f "${OUT}/neuroserve.nsys-rep" ]]; then
    echo "no report written; see ${OUT}/server.log" >&2
    exit 1
fi

"${NSYS}" stats --force-export=true --report cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum,nvtx_sum \
    --format csv --output "${OUT}/stats" "${OUT}/neuroserve.nsys-rep" > /dev/null
echo "== top kernels by GPU time"
"${NSYS}" stats --report cuda_gpu_kern_sum --format column "${OUT}/neuroserve.nsys-rep" 2>/dev/null | head -20 || true
echo "== NVTX ranges (engine step phases)"
"${NSYS}" stats --report nvtx_sum --format column "${OUT}/neuroserve.nsys-rep" 2>/dev/null | head -20 || true
echo "== CUDA API calls"
"${NSYS}" stats --report cuda_api_sum --format column "${OUT}/neuroserve.nsys-rep" 2>/dev/null | head -14 || true
echo "== wrote ${OUT}"
