# Block-size sweep (Day 42): decode attention time, capacity and waste at 8, 16 and 32-token blocks.
#   ./build/dev-cuda/benchmarks/block_size/ns_block_size_sweep --out benchmarks/results/day42-block-size.md
add_executable(ns_block_size_sweep block_size_sweep.cu)
target_link_libraries(ns_block_size_sweep PRIVATE cuda_engine::core)
