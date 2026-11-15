#include "neuroserve/cuda/device_info.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>

#include "neuroserve/cuda/check.hpp"

namespace neuroserve::cuda {
namespace {

std::string format_uuid(const cudaUUID_t& uuid) {
    static constexpr char kHex[] = "0123456789abcdef";
    std::string out = "GPU-";
    for (int i = 0; i < 16; ++i) {
        if (i == 4 || i == 6 || i == 8 || i == 10) {
            out += '-';
        }
        const auto byte = static_cast<unsigned char>(uuid.bytes[i]);
        out += kHex[byte >> 4];
        out += kHex[byte & 0x0F];
    }
    return out;
}

double to_gib(std::size_t bytes) { return static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0); }
double to_mib(std::size_t bytes) { return static_cast<double>(bytes) / (1024.0 * 1024.0); }

std::string json_escape(const std::string& s) {
    std::ostringstream out;
    for (const char c : s) {
        switch (c) {
            case '"':  out << "\\\""; break;
            case '\\': out << "\\\\"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                        << static_cast<int>(static_cast<unsigned char>(c)) << std::dec;
                } else {
                    out << c;
                }
        }
    }
    return out.str();
}

void check_index(int index, int count) {
    if (index < 0 || index >= count) {
        throw std::out_of_range("CUDA device index " + std::to_string(index) +
                                " is out of range (" + std::to_string(count) +
                                " device(s) found)");
    }
}

}  // namespace

int device_count(std::string* error) {
    int count = 0;
    const cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        if (error != nullptr) {
            *error = std::string(cudaGetErrorName(err)) + ": " + cudaGetErrorString(err);
        }
        (void)cudaGetLastError();  // clear it so later CUDA calls are not affected
        return 0;
    }
    return count;
}

DeviceInfo query_device(int index) {
    check_index(index, device_count());

    cudaDeviceProp prop{};
    NS_CUDA_CHECK(cudaGetDeviceProperties(&prop, index));

    DeviceInfo info;
    info.index = index;
    info.name = prop.name;
    info.uuid = format_uuid(prop.uuid);
    info.pci_bus_id = prop.pciBusID;
    info.cc_major = prop.major;
    info.cc_minor = prop.minor;
    info.sm_count = prop.multiProcessorCount;
    info.total_mem_bytes = prop.totalGlobalMem;
    info.l2_cache_bytes = static_cast<std::size_t>(prop.l2CacheSize);
    info.shared_mem_per_block_optin_bytes = prop.sharedMemPerBlockOptin;
    info.memory_bus_width_bits = prop.memoryBusWidth;

    // Free memory needs the device's context: switch to it, measure, switch back.
    int previous = 0;
    NS_CUDA_CHECK(cudaGetDevice(&previous));
    NS_CUDA_CHECK(cudaSetDevice(index));
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    NS_CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    NS_CUDA_CHECK(cudaSetDevice(previous));
    info.free_mem_bytes = free_bytes;

    NS_CUDA_CHECK(cudaDriverGetVersion(&info.driver_version));
    NS_CUDA_CHECK(cudaRuntimeGetVersion(&info.runtime_version));
    return info;
}

InitResult initialize_device(int index) {
    std::string error;
    const int count = device_count(&error);
    if (count == 0) {
        throw std::runtime_error("no CUDA-capable GPU detected" +
                                 (error.empty() ? std::string() : " (" + error + ")"));
    }
    check_index(index, count);

    const auto start = std::chrono::steady_clock::now();
    NS_CUDA_CHECK(cudaSetDevice(index));
    NS_CUDA_CHECK(cudaFree(nullptr));  // forces context creation now, not on the first real call
    const auto end = std::chrono::steady_clock::now();

    InitResult result;
    result.info = query_device(index);
    result.init_ms = std::chrono::duration<double, std::milli>(end - start).count();

    if (result.info.cc_major < kMinComputeCapabilityMajor) {
        throw std::runtime_error("GPU '" + result.info.name + "' has compute capability " +
                                 std::to_string(result.info.cc_major) + "." +
                                 std::to_string(result.info.cc_minor) +
                                 "; NeuroServe requires " +
                                 std::to_string(kMinComputeCapabilityMajor) + ".0 or newer");
    }
    return result;
}

std::string format_cuda_version(int version) {
    return std::to_string(version / 1000) + "." + std::to_string((version % 1000) / 10);
}

std::string summary(const DeviceInfo& d) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(2);
    out << "GPU " << d.index << ": " << d.name << "\n"
        << "  UUID:               " << d.uuid << "\n"
        << "  Compute capability: " << d.cc_major << "." << d.cc_minor
        << " (" << d.sm_count << " SMs)\n"
        << "  Memory:             " << to_gib(d.free_mem_bytes) << " GiB free / "
        << to_gib(d.total_mem_bytes) << " GiB total, " << d.memory_bus_width_bits << "-bit bus\n"
        << "  L2 cache:           " << to_mib(d.l2_cache_bytes) << " MiB\n"
        << "  Shared mem/block:   " << d.shared_mem_per_block_optin_bytes / 1024 << " KiB (opt-in max)\n"
        << "  PCI bus:            " << d.pci_bus_id << "\n"
        << "  Driver / runtime:   CUDA " << format_cuda_version(d.driver_version)
        << " / CUDA " << format_cuda_version(d.runtime_version);
    return out.str();
}

std::string to_json(const DeviceInfo& d) {
    std::ostringstream out;
    out << "{\"index\":" << d.index
        << ",\"name\":\"" << json_escape(d.name) << "\""
        << ",\"uuid\":\"" << d.uuid << "\""
        << ",\"pci_bus_id\":" << d.pci_bus_id
        << ",\"compute_capability\":\"" << d.cc_major << "." << d.cc_minor << "\""
        << ",\"sm_count\":" << d.sm_count
        << ",\"total_mem_bytes\":" << d.total_mem_bytes
        << ",\"free_mem_bytes\":" << d.free_mem_bytes
        << ",\"l2_cache_bytes\":" << d.l2_cache_bytes
        << ",\"shared_mem_per_block_optin_bytes\":" << d.shared_mem_per_block_optin_bytes
        << ",\"memory_bus_width_bits\":" << d.memory_bus_width_bits
        << ",\"driver_version\":\"" << format_cuda_version(d.driver_version) << "\""
        << ",\"runtime_version\":\"" << format_cuda_version(d.runtime_version) << "\"}";
    return out.str();
}

}  // namespace neuroserve::cuda