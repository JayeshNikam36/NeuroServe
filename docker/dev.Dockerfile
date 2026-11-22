# syntax=docker/dockerfile:1
ARG CUDA_IMAGE=nvidia/cuda:12.4.1-devel-ubuntu22.04
FROM ${CUDA_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential ninja-build git ca-certificates curl \
        python3 python3-dev python3-pip python3-venv gdb \
        g++-12 \
    && rm -rf /var/lib/apt/lists/*

# g++-12: only for the tsan-cpu preset. GCC 11's ThreadSanitizer cannot see pthread_cond_clockwait
# (used by condition_variable::wait_for) and reports false "double lock" errors.
# Ubuntu 22.04 ships CMake 3.22, which can't set C++20 for CUDA. Pin a modern CMake.
# pytest/httpx/openai: tests (openai: the official client, for streaming tests and demos).
# tokenizers/jinja2: tokenization. fastapi/uvicorn/pydantic/pyyaml: API server.
# nvidia-ml-py: NVML bindings (pynvml), for the benchmarks' GPU clock/temperature/power sampler.
RUN python3 -m pip install --no-cache-dir \
        "cmake==3.30.5" "pytest==8.3.3" "httpx==0.27.2" "openai==1.55.3" \
        "tokenizers==0.20.3" "jinja2==3.1.4" \
        "fastapi==0.115.5" "uvicorn==0.32.1" "pydantic==2.10.3" "pyyaml==6.0.2" \
        "nvidia-ml-py==12.560.30" \
    && cmake --version

# 2023.4 (CUDA 12.4's own) cannot import traces of this multi-threaded server ("Wrong event order");
# a newer standalone release profiles CUDA 12.4 programs and imports them.
# Nsight Systems CLI (nsys): engine-step timelines with CUDA kernels and NVTX ranges
# (scripts/nsys_profile.sh, docs/profiling.md). Linked into /usr/local/bin if the package does not.
RUN apt-get update && apt-get install -y --no-install-recommends "$(apt-cache pkgnames nsight-systems-2024 | sort -V | tail -n1)" \
    && rm -rf /var/lib/apt/lists/* \
    && (command -v nsys || ln -s "$(ls -d /opt/nvidia/nsight-systems/*/bin/nsys | head -n1)" /usr/local/bin/nsys) \
    && nsys --version

# Non-root user matching the host UID/GID, so files created in the
# mounted repo are owned by you, not root.
ARG USERNAME=dev
ARG USER_UID=1000
ARG USER_GID=1000
RUN groupadd --gid ${USER_GID} ${USERNAME} \
    && useradd --uid ${USER_UID} --gid ${USER_GID} -m -s /bin/bash ${USERNAME}

USER ${USERNAME}
WORKDIR /workspace
