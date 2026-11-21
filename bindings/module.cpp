#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <atomic>
#include <chrono>
#include <iterator>
#include <memory>
#include <optional>
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

// One streaming request, as Python sees it. read() and result() block with the GIL released.
// The engine thread never calls into Python: it appends tokens to the request's buffer in C++,
// and a Python thread reads them from there.
class TokenStream {
public:
    explicit TokenStream(core::RequestHandle handle) : handle_(std::move(handle)) {}

    const std::string& request_id() const { return handle_.id(); }

    // New tokens, oldest first; [] if `timeout_s` passed first; None once the request has
    // finished and every token has been read (then call result()).
    std::optional<std::vector<ns::TokenId>> read(std::optional<double> timeout_s) {
        if (finished_) {
            return std::nullopt;
        }
        const auto wait = timeout_s ? std::chrono::milliseconds(static_cast<long long>(*timeout_s * 1000.0))
                                    : std::chrono::milliseconds(1000);
        for (;;) {
            auto batch = handle_.read_tokens(wait);
            finished_ = batch.finished;
            if (!batch.tokens.empty()) {
                return std::move(batch.tokens);  // the next read() returns None if this was the end
            }
            if (finished_) {
                return std::nullopt;
            }
            if (timeout_s) {
                return std::vector<ns::TokenId>{};
            }
        }
    }

    // Blocks until the request finishes; returns its result or raises its error.
    core::GenerationResult result() const { return handle_.wait(); }

private:
    core::RequestHandle handle_;
    std::atomic<bool> finished_{false};
};

// Python-facing engine. Owns one backend and the engine loop that drives it: every call from
// Python is submitted to the loop's engine thread, so any number of Python threads can generate
// at the same time and their requests are batched together. Heavy calls run with the GIL released.
class Engine {
public:
    Engine(const ns::CudaLlmEngineConfig& config, core::EngineLoopOptions loop_options,
           std::unique_ptr<core::SchedulingPolicy> policy)
        : policy_options_(loop_options.policy_options),
          backend_(std::make_unique<ns::CudaLlmEngineBackend>(config)),
          loop_(std::make_unique<core::EngineLoop>(*backend_, std::move(loop_options), std::move(policy))) {}

    // Members are destroyed in reverse order: the loop (and its thread) stops before the backend goes.

    core::GenerationResult run(std::vector<ns::TokenId> prompt_ids, int max_new_tokens,
                               std::vector<ns::TokenId> stop_token_ids, std::string client_request_id, int priority) {
        core::GenerationParams params;
        params.prompt = std::move(prompt_ids);
        params.max_new_tokens = max_new_tokens;
        params.stop_token_ids = std::move(stop_token_ids);
        params.client_request_id = std::move(client_request_id);
        params.priority = priority;
        return loop_->submit(std::move(params)).wait();
    }

    std::unique_ptr<TokenStream> stream(std::vector<ns::TokenId> prompt_ids, int max_new_tokens,
                       std::vector<ns::TokenId> stop_token_ids, std::string client_request_id, int priority) {
        core::GenerationParams params;
        params.prompt = std::move(prompt_ids);
        params.max_new_tokens = max_new_tokens;
        params.stop_token_ids = std::move(stop_token_ids);
        params.client_request_id = std::move(client_request_id);
        params.priority = priority;
        params.stream = true;
        return std::make_unique<TokenStream>(loop_->submit(std::move(params)));  // a bad request raises here
    }

    std::vector<ns::TokenId> generate(std::vector<ns::TokenId> prompt_ids, int max_new_tokens) {
        return run(std::move(prompt_ids), max_new_tokens, {}, {}, core::kNormalPriority).tokens;
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
        out["kv_cache_key"] = i.kv_cache_key;
        return out;
    }

    py::dict kv_stats() const {
        const ns::KvCacheStats s = loop_->kv_stats();  // the loop's snapshot: never touches the backend
        py::dict out;
        out["page_size_tokens"] = s.page_size_tokens;
        out["total_pages"] = s.total_pages;
        out["free_pages"] = s.free_pages;
        out["max_pages_per_sequence"] = s.max_pages_per_sequence;
        out["shared_pages"] = s.shared_pages;
        out["indexed_pages"] = s.indexed_pages;
        out["prefix_lookups"] = s.prefix_lookups;
        out["prefix_lookup_pages"] = s.prefix_lookup_pages;
        out["prefix_hit_pages"] = s.prefix_hit_pages;
        out["cached_pages"] = s.cached_pages;
        out["evicted_pages"] = s.evicted_pages;
        return out;
    }

    bool cancel(const std::string& request_id) { return loop_->cancel_request(request_id); }

    void set_scheduling_policy(const std::string& name) {
        loop_->set_policy(core::make_policy(name, policy_options_));  // unknown name: ValueError; next step on
    }

    py::dict decisions(std::uint64_t since, std::size_t limit) const {
        const core::DecisionPage page = loop_->decisions(since, limit);
        py::list data;
        for (const auto& e : page.decisions) {
            py::dict numbers;
            for (const auto& [name, value] : e.numbers) {
                numbers[py::str(name)] = value;
            }
            py::dict d;
            d["seq"] = e.seq;
            d["step"] = e.step;
            d["time_ms"] = e.unix_ms;
            d["request_id"] = e.request_id;
            d["client_request_id"] = e.client_request_id;
            d["action"] = std::string(core::to_string(e.action));
            d["reason"] = e.reason;
            d["policy"] = e.policy;
            d["numbers"] = numbers;
            data.append(d);
        }
        py::dict out;
        out["data"] = data;
        out["next_since"] = page.next_since;
        out["dropped"] = page.dropped;
        return out;
    }

    py::dict latency() const {
        const core::LatencyStats l = loop_->latency();
        const auto summary = [](const core::LatencySummary& s) {
            py::dict d;
            d["count"] = s.count;
            d["mean"] = s.mean;
            d["p50"] = s.p50;
            d["p95"] = s.p95;
            d["p99"] = s.p99;
            d["max"] = s.max;
            return d;
        };
        py::dict out;
        out["queue_ms"] = summary(l.queue_ms);
        out["ttft_ms"] = summary(l.ttft_ms);
        out["itl_ms"] = summary(l.itl_ms);
        out["e2e_ms"] = summary(l.e2e_ms);
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
        out["timed_out"] = s.timed_out;
        out["preemptions"] = s.preemptions;
        out["decode_steps"] = s.decode_steps;
        out["prefill_tokens"] = s.prefill_tokens;
        out["decode_tokens"] = s.decode_tokens;
        out["max_step_tokens"] = s.max_step_tokens;
        out["pending"] = s.pending;
        out["active"] = s.active;
        out["max_batch_seen"] = s.max_batch_seen;
        out["policy"] = std::string(loop_->policy_name());
        return out;
    }

private:
    const core::PolicyOptions policy_options_;  // also used by set_scheduling_policy
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
    py::register_exception<core::RequestCancelled>(m, "RequestCancelled", PyExc_RuntimeError);
    py::register_exception<core::QueueFull>(m, "QueueFull", PyExc_RuntimeError);
    py::register_exception<core::QueueTimeout>(m, "QueueTimeout", PyExc_RuntimeError);

    py::tuple policies(std::size(core::kPolicyNames));
    for (std::size_t i = 0; i < std::size(core::kPolicyNames); ++i) {
        policies[i] = std::string(core::kPolicyNames[i]);
    }
    m.attr("scheduling_policies") = policies;  // the names engine.scheduling_policy accepts

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
        .def_readonly("itl_ms", &core::GenerationResult::itl_ms)
        .def_readonly("max_itl_ms", &core::GenerationResult::max_itl_ms)
        .def("__repr__", [](const core::GenerationResult& r) {
            return "<GenerationResult " + r.request_id + " tokens=" + std::to_string(r.tokens.size()) +
                   " finish_reason=" + std::string(core::to_string(r.finish_reason)) + ">";
        });

    py::class_<TokenStream>(m, "TokenStream", "A streaming request: read() its tokens as they are generated.")
        .def_property_readonly("request_id", &TokenStream::request_id)
        .def("read", &TokenStream::read, py::arg("timeout") = py::none(), py::call_guard<py::gil_scoped_release>(),
             "New token ids, oldest first. Waits up to `timeout` seconds (forever if None); returns [] if "
             "nothing arrived in time, and None once the request has finished and every token was read.")
        .def("result", &TokenStream::result, py::call_guard<py::gil_scoped_release>(),
             "Waits for the request to finish; returns its GenerationResult or raises its error.");

    m.def("device_info", &device_info, py::arg("index") = 0,
          "Properties of a CUDA device, as a dict.");

    py::class_<Engine>(m, "Engine", "Loads a model onto the GPU and generates tokens (greedy).")
        .def(py::init([](const std::string& model_dir, int max_prompt_tokens, int num_pages,
                         int max_batch, int max_batch_tokens, int max_pages_per_sequence, int device_index,
                         const std::string& scheduling_policy, int priority_aging_ms, int max_queue,
                         int queue_timeout_ms, int decision_log_capacity, bool prefix_caching,
                         const std::string& kv_admission) {
                 core::EngineLoopOptions loop_options;
                 loop_options.max_batch = max_batch;
                 loop_options.max_batch_tokens = max_batch_tokens;
                 loop_options.policy_options.priority_aging_ms = priority_aging_ms;
                 loop_options.max_queue = max_queue;
                 loop_options.queue_timeout_ms = queue_timeout_ms;
                 loop_options.decision_log_capacity = decision_log_capacity;
                 loop_options.admission = core::parse_admission(kv_admission);  // ValueError if unknown
                 // Bad options raise ValueError here, before the weights are loaded.
                 if (max_queue < 1) {
                     throw std::invalid_argument("max_queue must be at least 1");
                 }
                 if (queue_timeout_ms < 0) {
                     throw std::invalid_argument("queue_timeout_ms must be 0 (no timeout) or more");
                 }
                 if (decision_log_capacity < 1) {
                     throw std::invalid_argument("decision_log_capacity must be at least 1");
                 }
                 auto policy = core::make_policy(scheduling_policy, loop_options.policy_options);
                 ns::CudaLlmEngineConfig config;
                 config.model_dir = model_dir;
                 config.max_prompt_tokens = max_prompt_tokens;
                 config.num_pages = num_pages;
                 config.max_batch = max_batch;
                 config.max_pages_per_sequence = max_pages_per_sequence;
                 config.device_index = device_index;
                 config.prefix_caching = prefix_caching;
                 py::gil_scoped_release release;  // loading 2.5 GB of weights takes seconds
                 return std::make_unique<Engine>(config, std::move(loop_options), std::move(policy));
             }),
             py::arg("model_dir"), py::kw_only(),
             py::arg("max_prompt_tokens") = 512, py::arg("num_pages") = 2048,
             py::arg("max_batch") = 8, py::arg("max_batch_tokens") = 512, py::arg("max_pages_per_sequence") = 64,
             py::arg("device_index") = 0, py::arg("scheduling_policy") = "fcfs",
             py::arg("priority_aging_ms") = 1000, py::arg("max_queue") = 1024, py::arg("queue_timeout_ms") = 0,
             py::arg("decision_log_capacity") = 4096, py::arg("prefix_caching") = false,
             py::arg("kv_admission") = "on-demand")
        .def("run", &Engine::run,
             py::arg("prompt_ids"), py::arg("max_new_tokens") = 32, py::kw_only(),
             py::arg("stop_token_ids") = std::vector<ns::TokenId>{}, py::arg("client_request_id") = "",
             py::arg("priority") = core::kNormalPriority,
             py::call_guard<py::gil_scoped_release>(),
             "Submits one request to the engine loop and waits for it. Stops on (and includes) the first "
             "stop token. priority: 0 low, 1 normal, 2 high (used by the 'priority' policy). "
             "Thread-safe: concurrent calls are batched.")
        .def("stream", &Engine::stream,
             py::arg("prompt_ids"), py::arg("max_new_tokens") = 32, py::kw_only(),
             py::arg("stop_token_ids") = std::vector<ns::TokenId>{}, py::arg("client_request_id") = "",
             py::arg("priority") = core::kNormalPriority,
             "Submits one request and returns a TokenStream at once; read its tokens while it runs. "
             "Same arguments and validation as run().")
        .def("generate", &Engine::generate,
             py::arg("prompt_ids"), py::arg("max_new_tokens") = 32,
             py::call_guard<py::gil_scoped_release>(),
             "Greedy generation without stop tokens: returns exactly max_new_tokens token ids.")
        .def("cancel", &Engine::cancel, py::arg("request_id"),
             "Cancels an unfinished request by its id (ns-...): it stops at the next engine step, its KV "
             "pages are freed, and its caller gets RequestCancelled. False if no unfinished request has this id.")
        .def("model_info", &Engine::model_info)
        .def("kv_stats", &Engine::kv_stats)
        .def("stats", &Engine::stats, "Engine loop counters: queue, batch and request outcomes.")
        .def("decisions", &Engine::decisions, py::arg("since") = 0, py::arg("limit") = 100,
             "Why requests were admitted, deferred, rejected or preempted, with the numbers each decision was "
             "made on: {'data': [...], 'next_since': cursor, 'dropped': missed}. Pass next_since back as since.")
        .def("latency", &Engine::latency,
             "Latency since start, per metric (queue_ms, ttft_ms, itl_ms, e2e_ms): count, mean, p50, p95, p99, "
             "max. Fixed-bucket histograms: percentiles are within one bucket (about 9%) of the exact value.")
        .def("set_scheduling_policy", &Engine::set_scheduling_policy, py::arg("name"),
             "Switches the scheduling policy from the next engine step on (A/B comparisons).");
}
