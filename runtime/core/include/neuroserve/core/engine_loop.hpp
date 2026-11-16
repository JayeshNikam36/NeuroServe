#pragma once

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <exception>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "neuroserve/core/inference_backend.hpp"
#include "neuroserve/core/request.hpp"

namespace neuroserve::core {

// What a caller asks for.
struct GenerationParams {
    std::vector<TokenId> prompt;
    int max_new_tokens = 16;
    std::vector<TokenId> stop_token_ids;  // generation ends on (and includes) the first of these
    std::string client_request_id;        // caller's id (e.g. HTTP X-Request-ID), only for logs
};

enum class FinishReason : std::uint8_t { kStop, kLength };
std::string_view to_string(FinishReason reason) noexcept;

// What a caller gets back when the request completes.
struct GenerationResult {
    std::string request_id;
    std::vector<TokenId> tokens;  // generated tokens, including the stop token if one was hit
    FinishReason finish_reason = FinishReason::kLength;
    int prompt_tokens = 0;
    double queue_ms = 0.0;  // first QUEUED -> ADMITTED
    double ttft_ms = 0.0;   // RECEIVED -> first token (time to first token)
    double e2e_ms = 0.0;    // RECEIVED -> COMPLETED
};

// The request was cancelled because the engine is stopping, or submit() was called after stop().
class EngineStopped : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// The caller's side of one request. Copyable; all copies refer to the same request.
class RequestHandle {
public:
    struct State;  // shared with the engine thread; defined in engine_loop.cpp

    explicit RequestHandle(std::shared_ptr<State> state);

    const std::string& id() const noexcept;

    // Waits up to `timeout`; true once the request has finished (successfully or not).
    bool wait_for(std::chrono::milliseconds timeout) const;

    // Blocks until the request finishes. Returns the result, or rethrows the request's error
    // (the backend's exception for FAILED, EngineStopped for CANCELLED).
    GenerationResult wait() const;

private:
    std::shared_ptr<State> state_;
};

struct EngineLoopOptions {
    int max_batch = 8;  // most sequences in one decode step
};

// Counters for monitoring and tests. A consistent snapshot, safe to read from any thread.
struct EngineLoopStats {
    std::uint64_t submitted = 0;  // accepted into the queue
    std::uint64_t rejected = 0;   // refused by submit() (invalid, or engine stopped)
    std::uint64_t completed = 0;
    std::uint64_t failed = 0;
    std::uint64_t cancelled = 0;
    std::uint64_t decode_steps = 0;
    std::size_t pending = 0;         // waiting in the queue
    std::size_t active = 0;          // admitted and generating
    std::size_t max_batch_seen = 0;  // largest decode batch so far
};

// Runs generation for many requests on one dedicated engine thread.
//
//   caller threads ── submit() ──► pending queue ──► engine thread: admit ► prefill ► decode steps
//   caller threads ◄── RequestHandle::wait() ◄──────── result / error per request
//
// Only the engine thread touches the backend. Callers only enqueue and wait.
// Each step decodes all active requests together (one decode_step call for the whole batch),
// and a request finishes on the step that produces its stop token or its last allowed token.
// Logs one JSON line per finished request (logger "neuroserve.engine"); never logs token ids.
// Admission reserves KV pages for a request's whole length up front, so the cache cannot run out
// mid-generation (Day 14 replaces this with on-demand allocation and preemption).
class EngineLoop {
public:
    // Starts the engine thread. The backend must outlive the loop.
    explicit EngineLoop(InferenceBackend& backend, EngineLoopOptions options = {});
    ~EngineLoop();  // stop()

    EngineLoop(const EngineLoop&) = delete;
    EngineLoop& operator=(const EngineLoop&) = delete;

    // Thread-safe. Validates and enqueues; never waits for generation.
    // Throws std::invalid_argument for a bad request and EngineStopped after stop().
    RequestHandle submit(GenerationParams params);

    // Thread-safe and idempotent. Cancels queued and running requests, then joins the engine thread.
    void stop();
    bool stop_requested() const;

    EngineLoopStats stats() const;

    // KV-cache usage as of the engine thread's last change. Safe from any thread; up to date for a
    // request's pages by the time its wait() returns.
    KvCacheStats kv_stats() const;

private:
    struct Entry;
    using EntryPtr = std::unique_ptr<Entry>;

    void validate(const GenerationParams& params) const;
    int pages_for(const GenerationParams& params) const noexcept;

    // Engine thread only.
    void run() noexcept;
    void loop();
    void admit_locked(std::vector<EntryPtr>& admitted);
    void start(EntryPtr entry);
    void decode_active();
    bool append_token(Entry& entry, TokenId token);
    void complete(EntryPtr entry) noexcept;
    void fail(EntryPtr entry, std::exception_ptr error) noexcept;
    void cancel(Entry& entry, std::exception_ptr error) noexcept;
    void release(Entry& entry) noexcept;
    void cancel_all() noexcept;
    void refresh_kv_snapshot() noexcept;

    InferenceBackend& backend_;
    const EngineLoopOptions options_;
    const ModelInfo model_info_;
    int page_size_ = 0;
    int total_pages_ = 0;
    int max_sequence_tokens_ = 0;
    RequestIdGenerator ids_;

    mutable std::mutex mu_;  // guards everything down to stats_
    std::condition_variable cv_;
    bool stop_ = false;
    std::deque<EntryPtr> pending_;
    EngineLoopStats stats_;
    KvCacheStats kv_snapshot_;

    std::vector<EntryPtr> active_;  // engine thread only
    int reserved_pages_ = 0;        // engine thread only

    std::mutex join_mu_;
    std::thread thread_;  // last member: started after everything above is initialised
};

}  // namespace neuroserve::core
