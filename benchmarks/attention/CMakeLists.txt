# Prefill attention kernels (Day 48 on, docs/prefill-attention.md): time per layer with CUDA events.
#   ./build/dev-cuda/benchmarks/attention/ns_attention_bench --out benchmarks/results/day49-attention-tiled.md
add_executable(ns_attention_bench attention_bench.cu)
target_include_directories(ns_attention_bench PRIVATE ${PROJECT_SOURCE_DIR}/runtime/backend/src)
target_link_libraries(ns_attention_bench PRIVATE neuroserve::kernels neuroserve::backend cuda_engine::core)
