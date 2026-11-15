#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <string>
#include <vector>

#include "neuroserve/backend/cuda_llm_engine_backend.hpp"
#include "neuroserve/core/generate.hpp"
#include "neuroserve/core/inference_backend.hpp"
#include "neuroserve/core/version.hpp"
#include "neuroserve/cuda/check.hpp"
#include "neuroserve/cuda/device_info.hpp"

namespace py = pybind11;
namespace ns = neuroserve;

namespace {

py::dict device_info(int index) {
    const ns::cuda::DeviceInfo d = ns::cuda::query_device(index);
    py::dict out;
    out["index"] = d.index;
    out["name"] = d.name;
    out["uuid"] = d.uuid;
    out["compute_capability"] = std::to_string(d.cc_major) + "." + std::to_string(d.cc_minor);
    out["sm_count"] = d.sm_count;
    out["total_mem_bytes"] = d.total_mem_bytes;
    out["free_mem_bytes"] = d.free_mem_bytes;
    out["driver_version"] = ns::cuda::format_cuda_version(d.driver_version);
    out["runtime_version"] = ns::cuda::format_cuda_version(d.runtime_version);
    return out;
}

// Python-facing engine. Owns one backend; heavy calls run with the GIL released.
class Engine {
public:
    explicit Engine(const ns::CudaLlmEngineConfig& config)
        : backend_(std::make_unique<ns::CudaLlmEngineBackend>(config)) {}

    std::vector<ns::TokenId> generate(const std::vector<ns::TokenId>& prompt_ids, int max_new_tokens) {
        return ns::generate_greedy(*backend_, prompt_ids, max_new_tokens);
    }

    py::dict model_info() const {
        const ns::ModelInfo& i = backend_->model_info();
        py::dict out;
        out["name"] = i.name;
        out["n_layers"] = i.n_layers;
        out["n_kv_heads"] = i.n_kv_heads;
        out["head_dim"] = i.head_dim;
        out["vocab_size"] = i.vocab_size;
        out["max_prompt_tokens"] = i.max_prompt_tokens;
        out["kv_bytes_per_token"] = i.kv_bytes_per_token;
        return out;
    }

    py::dict kv_stats() const {
        const ns::KvCacheStats s = backend_->kv_stats();
        py::dict out;
        out["page_size_tokens"] = s.page_size_tokens;
        out["total_pages"] = s.total_pages;
        out["free_pages"] = s.free_pages;
        out["max_pages_per_sequence"] = s.max_pages_per_sequence;
        return out;
    }

private:
    std::unique_ptr<ns::CudaLlmEngineBackend> backend_;
};

}  // namespace

PYBIND11_MODULE(_core, m) {
    m.doc() = "NeuroServe native runtime (C++/CUDA)";
    m.attr("__version__") = std::string(ns::version());

    // C++ exception types -> Python exception classes (std::invalid_argument -> ValueError is built in).
    py::register_exception<ns::KvCacheExhausted>(m, "KvCacheExhausted", PyExc_RuntimeError);
    py::register_exception<ns::cuda::CudaError>(m, "CudaError", PyExc_RuntimeError);

    m.def("device_info", &device_info, py::arg("index") = 0,
          "Properties of a CUDA device, as a dict.");

    py::class_<Engine>(m, "Engine", "Loads a model onto the GPU and generates tokens (greedy).")
        .def(py::init([](const std::string& model_dir, int max_prompt_tokens, int num_pages,
                         int max_batch, int max_pages_per_sequence, int device_index) {
                 ns::CudaLlmEngineConfig config;
                 config.model_dir = model_dir;
                 config.max_prompt_tokens = max_prompt_tokens;
                 config.num_pages = num_pages;
                 config.max_batch = max_batch;
                 config.max_pages_per_sequence = max_pages_per_sequence;
                 config.device_index = device_index;
                 py::gil_scoped_release release;  // loading 2.5 GB of weights takes seconds
                 return std::make_unique<Engine>(config);
             }),
             py::arg("model_dir"), py::kw_only(),
             py::arg("max_prompt_tokens") = 512, py::arg("num_pages") = 2048,
             py::arg("max_batch") = 8, py::arg("max_pages_per_sequence") = 64,
             py::arg("device_index") = 0)
        .def("generate", &Engine::generate,
             py::arg("prompt_ids"), py::arg("max_new_tokens") = 32,
             py::call_guard<py::gil_scoped_release>(),
             "Greedy generation: returns exactly max_new_tokens token ids.")
        .def("model_info", &Engine::model_info)
        .def("kv_stats", &Engine::kv_stats);
}