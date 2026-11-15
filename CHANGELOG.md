# Changelog

All notable changes are listed here. Versions follow [Semantic Versioning](https://semver.org/);
before 1.0, minor versions may change the API.

## [0.0.1] — Phase 1: Foundation

### Added
- OpenAI-compatible HTTP server: `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/health`.
- Greedy generation on the GPU through cuda-llm-engine v1.0.2 (FP16 Llama, paged KV cache).
- Hugging Face-exact chat templating, stop tokens, and detokenization.
- Python package `neuroserve` with pybind11 bindings to the C++ runtime.
- Structured JSON logging with request ids; prompts not logged by default.
- OpenAI-format errors ([docs/errors.md](docs/errors.md)); unhealthy state after GPU errors.
- Docker dev environment, CMake presets, CI, and tests (C++, CUDA, integration, Python).

### Known limitations
- Greedy decoding only, `n=1`, no streaming, no batching across requests.
