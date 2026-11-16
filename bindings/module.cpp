#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "neuroserve/backend/cuda_llm_engine_backend.hpp"
#include "neuroserve/core/engine_loop.hpp"
#include "neuroserve/core/inference_backend.hpp"
#include "neuroserve/core/log.hpp"
#include "neuroserve/core/scheduler.hpp"
#include "neuroserve/core/version.hpp"
#include "neuroserve/cuda/check.hpp"
#include "neuroserve/cuda/device_info.hpp"

namespace py = pybind11;
namespace ns = neuroserve;
namespace core = neuroserve::core;

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

// Python-facing engine. Owns one backend and the engine loop that drives it: every call from
// Python is submitted to the loop's engine thread, so any number of Python threads can generate
// at the same time and their requests are batched together. Heavy calls run with the GIL released.
class Engine {
public:
    Engine(const ns::CudaLlmEngineConfig& config, std::unique_ptr<core::SchedulingPolicy> policy)
        : backend_(std::make_unique<ns::CudaLlmEngineBackend>(config)),
          loop_(std::make_unique<core::EngineLoop>(*backend_, loop_options(config), std::move(policy))) {}

    // Members are destroyed in reverse order: the loop (and its thread) stops before the backend goes.

    core::GenerationResult run(std::vector<ns::TokenId> prompt_ids, int max_new_tokens,
                               std::vector<ns::TokenId> stop_token_ids, std::string client_request_id) {
        core::GenerationParams params;
        params.prompt = std::move(prompt_ids);
        params.max_new_tokens = max_new_tokens;
        params.stop_token_ids = std::move(stop_token_ids);
        params.client_request_id = std::move(client_request_id);
        return loop_->submit(std::move(params)).wait();
    }

    std::vector<ns::TokenId> generate(std::vector<ns::TokenId> prompt_ids, int max_new_tokens) {
        return run(std::move(prompt_ids), max_new_tokens, {}, {}).tokens;
    }

    py::dict model_info() const {
        const ns::ModelInfo& i = backend_->model_info();  // immutable after loading: safe from any thread
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
        const ns::KvCacheStats s = loop_->kv_stats();  // the loop's snapshot: never touches the backend
        py::dict out;
        out["page_size_tokens"] = s.page_size_tokens;
        out["total_pages"] = s.total_pages;
        out["free_pages"] = s.free_pages;
        out["max_pages_per_sequence"] = s.max_pages_per_sequence;
        return out;
    }

    py::dict stats() const {
        const core::EngineLoopStats s = loop_->stats();
        py::dict out;
        out["submitted"] = s.submitted;
        out["rejected"] = s.rejected;
        out["completed"] = s.completed;
        out["failed"] = s.failed;
        out["cancelled"] = s.cancelled;
        out["decode_steps"] = s.decode_steps;
        out["pending"] = s.pending;
        out["active"] = s.active;
        out["max_batch_seen"] = s.max_batch_seen;
        out["policy"] = std::string(loop_->policy_name());
        return out;
    }

private:
    static core::EngineLoopOptions loop_options(const ns::CudaLlmEngineConfig& config) {
        core::EngineLoopOptions options;
        options.max_batch = config.max_batch;
        return options;
    }

    std::unique_ptr<ns::CudaLlmEngineBackend> backend_;
    std::unique_ptr<core::EngineLoop> loop_;
};

}  // namespace

PYBIND11_MODULE(_core, m) {
    m.doc() = "NeuroServe native runtime (C++/CUDA)";
    m.attr("__version__") = std::string(ns::version());

    // C++ exception types -> Python exception classes (std::invalid_argument -> ValueError is built in).
    py::register_exception<ns::KvCacheExhausted>(m, "KvCacheExhausted", PyExc_RuntimeError);
    py::register_exception<ns::cuda::CudaError>(m, "CudaError", PyExc_RuntimeError);
    py::register_exception<core::EngineStopped>(m, "EngineStopped", PyExc_RuntimeError);

    m.def("set_log_level", [](const std::string& level) { core::log::set_level(core::log::parse_level(level)); },
          py::arg("level"), "Level of the C++ runtime's JSON log lines (stderr): DEBUG, INFO, WARNING or ERROR.");

    py::class_<core::GenerationResult>(m, "GenerationResult", "Result of one completed request.")
        .def_readonly("request_id", &core::GenerationResult::request_id)
        .def_readonly("tokens", &core::GenerationResult::tokens)
        .def_property_readonly("finish_reason", [](const core::GenerationResult& r) {
            return std::string(core::to_string(r.finish_reason));
        })
        .def_readonly("prompt_tokens", &core::GenerationResult::prompt_tokens)
        .def_readonly("queue_ms", &core::GenerationResult::queue_ms)
        .def_readonly("ttft_ms", &core::GenerationResult::ttft_ms)
        .def_readonly("e2e_ms", &core::GenerationResult::e2e_ms)
        .def("__repr__", [](const core::GenerationResult& r) {
            return "<GenerationResult " + r.request_id + " tokens=" + std::to_string(r.tokens.size()) +
                   " finish_reason=" + std::string(core::to_string(r.finish_reason)) + ">";
        });

    m.def("device_info", &device_info, py::arg("index") = 0,
          "Properties of a CUDA device, as a dict.");

    py::class_<Engine>(m, "Engine", "Loads a model onto the GPU and generates tokens (greedy).")
        .def(py::init([](const std::string& model_dir, int max_prompt_tokens, int num_pages,
                         int max_batch, int max_pages_per_sequence, int device_index,
                         const std::string& scheduling_policy) {
                 auto policy = core::make_policy(scheduling_policy);  // unknown name: ValueError, before loading
                 ns::CudaLlmEngineConfig config;
                 config.model_dir = model_dir;
                 config.max_prompt_tokens = max_prompt_tokens;
                 config.num_pages = num_pages;
                 config.max_batch = max_batch;
                 config.max_pages_per_sequence = max_pages_per_sequence;
                 config.device_index = device_index;
                 py::gil_scoped_release release;  // loading 2.5 GB of weights takes seconds
                 return std::make_unique<Engine>(config, std::move(policy));
             }),
             py::arg("model_dir"), py::kw_only(),
             py::arg("max_prompt_tokens") = 512, py::arg("num_pages") = 2048,
             py::arg("max_batch") = 8, py::arg("max_pages_per_sequence") = 64,
             py::arg("device_index") = 0, py::arg("scheduling_policy") = "fcfs")
        .def("run", &Engine::run,
             py::arg("prompt_ids"), py::arg("max_new_tokens") = 32, py::kw_only(),
             py::arg("stop_token_ids") = std::vector<ns::TokenId>{}, py::arg("client_request_id") = "",
             py::call_guard<py::gil_scoped_release>(),
             "Submits one request to the engine loop and waits for it. Stops on (and includes) the first "
             "stop token. Thread-safe: concurrent calls are batched.")
        .def("generate", &Engine::generate,
             py::arg("prompt_ids"), py::arg("max_new_tokens") = 32,
             py::call_guard<py::gil_scoped_release>(),
             "Greedy generation without stop tokens: returns exactly max_new_tokens token ids.")
        .def("model_info", &Engine::model_info)
        .def("kv_stats", &Engine::kv_stats)
        .def("stats", &Engine::stats, "Engine loop counters: queue, batch and request outcomes.");
}
