#include <gtest/gtest.h>

#include <stdexcept>
#include <string>

#include "neuroserve/cuda/check.hpp"
#include "neuroserve/cuda/device_info.hpp"
#include "neuroserve/cuda/smoke.hpp"

namespace nc = neuroserve::cuda;

TEST(DeviceInfo, AtLeastOneGpu) {
    std::string error;
    EXPECT_GE(nc::device_count(&error), 1) << error;
}

TEST(DeviceInfo, Device0PropertiesAreSane) {
    const nc::DeviceInfo d = nc::query_device(0);
    EXPECT_FALSE(d.name.empty());
    EXPECT_EQ(d.uuid.rfind("GPU-", 0), 0u);
    EXPECT_EQ(d.uuid.size(), 40u);  // "GPU-" + 32 hex digits + 4 dashes
    EXPECT_GE(d.cc_major, nc::kMinComputeCapabilityMajor);
    EXPECT_GT(d.sm_count, 0);
    EXPECT_GT(d.total_mem_bytes, std::size_t{1} << 30);
    EXPECT_GT(d.free_mem_bytes, 0u);
    EXPECT_LE(d.free_mem_bytes, d.total_mem_bytes);
    EXPECT_GT(d.memory_bus_width_bits, 0);
}

TEST(DeviceInfo, DriverSupportsRuntime) {
    const nc::DeviceInfo d = nc::query_device(0);
    EXPECT_GE(d.driver_version, d.runtime_version)
        << "driver CUDA " << nc::format_cuda_version(d.driver_version)
        << " is older than runtime CUDA " << nc::format_cuda_version(d.runtime_version);
}

TEST(DeviceInfo, BinaryArchMatchesDevice) {
    const nc::DeviceInfo d = nc::query_device(0);
    EXPECT_EQ(nc::compiled_arch(), d.cc_major * 100 + d.cc_minor * 10);
}

TEST(DeviceInfo, InvalidIndexThrowsOutOfRange) {
    EXPECT_THROW(nc::query_device(-1), std::out_of_range);
    EXPECT_THROW(nc::query_device(99), std::out_of_range);
}

TEST(DeviceInfo, InitializeDeviceReportsTiming) {
    const nc::InitResult r = nc::initialize_device(0);
    EXPECT_EQ(r.info.index, 0);
    EXPECT_GE(r.init_ms, 0.0);
}

TEST(DeviceInfo, FormatCudaVersion) {
    EXPECT_EQ(nc::format_cuda_version(12040), "12.4");
    EXPECT_EQ(nc::format_cuda_version(13000), "13.0");
    EXPECT_EQ(nc::format_cuda_version(11080), "11.8");
}

TEST(DeviceInfo, JsonHasKeyFields) {
    const std::string json = nc::to_json(nc::query_device(0));
    for (const char* key : {"\"name\"", "\"uuid\"", "\"compute_capability\"", "\"total_mem_bytes\"",
                            "\"free_mem_bytes\"", "\"driver_version\"", "\"runtime_version\""}) {
        EXPECT_NE(json.find(key), std::string::npos) << "missing " << key;
    }
}

TEST(CudaCheck, ThrowsCudaErrorWithCodeAndExpression) {
    try {
        NS_CUDA_CHECK(cudaSetDevice(9999));
        FAIL() << "expected CudaError";
    } catch (const nc::CudaError& e) {
        EXPECT_EQ(e.code(), cudaErrorInvalidDevice);
        EXPECT_NE(std::string(e.what()).find("cudaSetDevice(9999)"), std::string::npos) << e.what();
    }
}