# syntax=docker/dockerfile:1
ARG CUDA_IMAGE=nvidia/cuda:12.4.1-devel-ubuntu22.04
FROM ${CUDA_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential ninja-build git ca-certificates \
        python3 python3-dev python3-pip python3-venv gdb \
    && rm -rf /var/lib/apt/lists/*

# Ubuntu 22.04 ships CMake 3.22, which can't set C++20 for CUDA. Pin a modern CMake.
# pytest: Python tests. tokenizers + jinja2: tokenization and chat templates.
RUN python3 -m pip install --no-cache-dir \
        "cmake==3.30.5" "pytest==8.3.3" "tokenizers==0.20.3" "jinja2==3.1.4" \
    && cmake --version

# Non-root user matching the host UID/GID, so files created in the
# mounted repo are owned by you, not root.
ARG USERNAME=dev
ARG USER_UID=1000
ARG USER_GID=1000
RUN groupadd --gid ${USER_GID} ${USERNAME} \
    && useradd --uid ${USER_UID} --gid ${USER_GID} -m -s /bin/bash ${USERNAME}

USER ${USERNAME}
WORKDIR /workspace
