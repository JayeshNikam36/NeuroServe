# CUDA Graphs for decode, A/B (Day 57, docs/cuda-graphs.md): decode step time through the eager,
# graph-eager and graph paths, interleaved, with NVML clocks.
#   ./build/dev-cuda/benchmarks/decode_graph/ns_decode_graph_bench --out benchmarks/results/day57-cuda-graphs.md
add_executable(ns_decode_graph_bench decode_graph_bench.cpp)
target_link_libraries(ns_decode_graph_bench PRIVATE neuroserve::backend CUDA::nvml)
