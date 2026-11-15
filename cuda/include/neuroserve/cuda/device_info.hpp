#pragma once

#include <cstddef>
#include <string>

namespace neuroserve::cuda {

// NeuroServe needs Ampere (8.0) or newer: async copies and BF16 are used in later phases.
inline constexpr int kMinComputeCapabilityMajor = 8;

struct DeviceInfo {
    int index = -1;
    std::string name;
    std::string uuid;                  // "GPU-xxxxxxxx-...", same format as `nvidia-smi -L`
    int pci_bus_id = -1;
    int cc_major = 0;
    int cc_minor = 0;
    int sm_count = 0;
    std::size_t total_mem_bytes = 0;
    std::size_t free_mem_bytes = 0;    // at the moment of the query
    std::size_t l2_cache_bytes = 0;
    std::size_t shared_mem_per_block_optin_bytes = 0;
    int memory_bus_width_bits = 0;
    int driver_version = 0;            // highest CUDA version the driver supports, e.g. 13000
    int runtime_version = 0;           // CUDA runtime this binary uses, e.g. 12040
};

struct InitResult {
    DeviceInfo info;
    double init_ms = 0.0;              // time to create the CUDA context
};

// Number of CUDA devices; 0 if none or the driver is unusable. Optional error text explains why.
int device_count(std::string* error = nullptr);

// Properties of one device. Throws std::out_of_range for a bad index, CudaError on CUDA failure.
DeviceInfo query_device(int index);

// Selects the device, creates its CUDA context, checks the minimum compute capability.
InitResult initialize_device(int index = 0);

// 12040 -> "12.4"
std::string format_cuda_version(int version);

// Multi-line human-readable report.
std::string summary(const DeviceInfo& info);

// One-line JSON object (used later in benchmark result files).
std::string to_json(const DeviceInfo& info);

}  // namespace neuroserve::cuda