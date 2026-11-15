# ADR-0001: Repository layout and engine integration

- **Status:** Accepted
- **Date:** 2026-11-16

## Context
NeuroServe builds a serving, scheduling and observability layer on top of
cuda-llm-engine (v1.0.1), which has its own history, tags and published
benchmark results. We need a layout that keeps the engine reviewable on its
own, ties every NeuroServe build to an exact engine version, and lets the
engine evolve when NeuroServe needs kernel changes.

## Options considered
1. **Copy the engine source into NeuroServe.** Simple, but loses history and
   creates two diverging copies.
2. **Merge the engine's history into NeuroServe (monorepo).** One repo, but the
   engine's v1.0 results stop being a standalone, citable artifact.
3. **Git submodule pinned to a tag.** Engine stays independent; NeuroServe
   records the exact commit it was built and benchmarked against.

## Decision
Option 3. The engine lives at `third_party/cuda-llm-engine`, pinned to tag
`v1.0.1`, using an HTTPS URL so anyone can clone. NeuroServe code goes in its
own top-level directories (`runtime/`, `cuda/`, `bindings/`, `server/`,
`frontend/`, `tests/`, `benchmarks/`), created only when their first tested
component exists.

## Consequences
- Clones require `--recurse-submodules` (documented in the README).
- Engine changes are made and tagged in the engine repo first, then the pin is
  moved in a dedicated NeuroServe commit.
- Every benchmark result records both the NeuroServe and engine commit hashes.