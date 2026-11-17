#pragma once

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_set>
#include <vector>

#include "neuroserve/core/decision_log.hpp"
#include "neuroserve/core/inference_backend.hpp"
#include "neuroserve/core/latency_histogram.hpp"
#include "neuroserve/core/request.hpp"
#include "neuroserve/core/scheduler.hpp"

namespace neuroserve::core {

// What a caller asks for.
struct GenerationParams {
    std::vector<TokenId> prompt;
    int max_new_tokens = 16;
    std::vector<TokenId> stop_token_ids;  // generation ends on (and includes) the first of these
    std::string client_request_id;        // caller's id (e.g. HTTP X-Request-ID), only for logs
    int priority = kNormalPriority;       // 0 low, 1 normal, 2 high; used by the "priority" policy
    bool stream = false;                  // tokens can be read while it runs (RequestHandle::read_tokens)
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
    double itl_ms = 0.0;      // mean inter-token latency: time between consecutive new tokens (0 with < 2)
    double max_itl_ms = 0.0;  // the longest gap between two new tokens (a preemption shows up here)
};

// The request was cancelled because the engine is stopping, or submit() was called after stop().
class EngineStopped : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// The request was cancelled by EngineLoop::cancel_request (e.g. the client disconnected).
class RequestCancelled : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// submit() refused the request: max_queue requests are already waiting. Retry later.
class QueueFull : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// The request waited longer than queue_timeout_ms without being admitted and was REJECTED.
class QueueTimeout : public std::runtime_error {
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

    // Streaming (GenerationParams::stream). Waits up to `timeout` for tokens not read yet and
    // returns them in order (none if the timeout passed first). `finished` is true once the request
    // has finished and every token has been read; wait() then returns the result or rethrows the
    // error. Each token is returned once, even if a preemption makes the engine generate it again.
    // Meant for one reader per request. Throws std::logic_error for a request without stream.
    struct TokenBatch {
        std::vector<TokenId> tokens;
        bool finished = false;
    };
    TokenBatch read_tokens(std::chrono::milliseconds timeout) const;

private:
    std::shared_ptr<State> state_;
};

// What one engine step did (for the on_step hook, tests and metrics).
struct StepStats {
    std::size_t prefill_requests = 0;  // requests admitted and prefilled in this step
    int prefill_tokens = 0;            // their prompt tokens
    int decode_tokens = 0;             // one per request decoded in this step
    std::size_t preempted = 0;
};

struct EngineLoopOptions {
    int max_batch = 8;             // most sequences in one decode step
    int max_batch_tokens = 512;    // most tokens one step processes (prefill + decode); see scheduler.hpp
    std::string policy = "fcfs";   // scheduling policy by name (see make_policy)
    PolicyOptions policy_options{};  // settings for the policy named in `policy`
    int max_queue = 1024;          // most requests waiting in the queue; submit() throws QueueFull beyond it
    int queue_timeout_ms = 0;      // a request not admitted within this time is REJECTED (QueueTimeout); 0 = never
    int decision_log_capacity = 4096;  // explanations kept for decisions() (the newest; older ones are dropped)
    // Called on the engine thread after every step that did work. Must be fast and must not throw.
    std::function<void(const StepStats&)> on_step{};
};

// Latency of completed requests (queue, TTFT, e2e: one sample per request) and of every gap
// between two new tokens (ITL: one sample per token after the first). A consistent snapshot.
struct LatencyStats {
    LatencySummary queue_ms;
    LatencySummary ttft_ms;
    LatencySummary itl_ms;
    LatencySummary e2e_ms;
};

// Counters for monitoring and tests. A consistent snapshot, safe to read from any thread.
struct EngineLoopStats {
    std::uint64_t submitted = 0;  // accepted into the queue
    std::uint64_t rejected = 0;   // refused by submit() (invalid, queue full, or engine stopped)
    std::uint64_t timed_out = 0;  // REJECTED after waiting longer than queue_timeout_ms
    std::uint64_t completed = 0;
    std::uint64_t failed = 0;
    std::uint64_t cancelled = 0;
    std::uint64_t preemptions = 0;  // times a running request was preempted (and later restarted)
    std::uint64_t decode_steps = 0;
    std::uint64_t prefill_tokens = 0;  // prompt tokens processed (including restarts after preemption)
    std::uint64_t decode_tokens = 0;
    int max_step_tokens = 0;           // most tokens any single step processed
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
// A step prefills the requests admitted in it and decodes one token for every request that was
// already running; its token count is bounded by max_batch_tokens.
// KV pages are allocated on demand as requests grow. Once per step a SchedulingPolicy
// (scheduler.hpp) decides which running requests to preempt when the batch cannot grow, and which
// queued requests to admit; the loop checks each decision and applies it. A preempted request
// frees its pages, goes back to the front of the queue and later restarts from its prompt
// (greedy decoding regenerates the same tokens).
class EngineLoop {
public:
    // Starts the engine thread. The backend must outlive the loop. Without an explicit `policy`,
    // the one named in options.policy is created (std::invalid_argument for an unknown name).
    explicit EngineLoop(InferenceBackend& backend, EngineLoopOptions options = {},
                        std::unique_ptr<SchedulingPolicy> policy = nullptr);
    ~EngineLoop();  // stop()

    EngineLoop(const EngineLoop&) = delete;
    EngineLoop& operator=(const EngineLoop&) = delete;

    // Thread-safe. Validates and enqueues; never waits for generation.
    // Throws std::invalid_argument for a bad request and EngineStopped after stop().
    RequestHandle submit(GenerationParams params);

    // Thread-safe. Cancels one request by its id (ns-...): at the start of the next engine step it
    // leaves the queue or the running batch, its sequence and KV pages are freed, and its caller gets
    // RequestCancelled (a stream ends after the tokens generated so far). Returns false if no
    // unfinished request has this id.
    bool cancel_request(const std::string& request_id);

    // Thread-safe and idempotent. Cancels queued and running requests, then joins the engine thread.
    void stop();
    bool stop_requested() const;

    EngineLoopStats stats() const;

    // Thread-safe. Latency percentiles since the engine started (fixed-bucket histograms).
    LatencyStats latency() const;

    // Thread-safe. Why requests were admitted, deferred, rejected or preempted: up to `limit`
    // explanations newer than `since` (0 = from the oldest kept), oldest first.
    DecisionPage decisions(std::uint64_t since, std::size_t limit) const;

    // Thread-safe. Name of the policy currently deciding.
    std::string policy_name() const;

    // Thread-safe. Replaces the scheduling policy from the next engine step on (for A/B comparisons
    // on a running engine). Requests already admitted are not affected.
    void set_policy(std::unique_ptr<SchedulingPolicy> policy);

    // KV-cache usage as of the engine thread's last change. Safe from any thread; up to date for a
    // request's pages by the time its wait() returns.
    KvCacheStats kv_stats() const;

private:
    struct Entry;
    using EntryPtr = std::unique_ptr<Entry>;

    void validate(const GenerationParams& params) const;
    int pages_for_tokens(long long tokens) const noexcept;  // KV pages that hold this many tokens

    // Engine thread only.
    void run() noexcept;
    void loop();
    void apply_cancellations();
    void expire_queued();  // queue_timeout_ms: reject requests that waited too long for admission
    void schedule_locked(int free_pages, std::vector<EntryPtr>& admitted, std::vector<EntryPtr>& preempted);
    void preempt(std::vector<EntryPtr> victims);
    void start(EntryPtr entry, std::vector<EntryPtr>& joining);
    void decode_active();
    bool append_token(Entry& entry, TokenId token);
    void publish_token(Entry& entry, TokenId token);  // stream: hand a new token to the reader
    void flush_itl_samples_locked();  // itl_samples_ -> itl_hist_; caller holds mu_
    // Decision log; caller holds mu_.
    void explain_locked(const SchedulerInput& input, const ScheduleDecision& decision);
    void record_locked(const Entry& entry, DecisionAction action, std::string reason,
                       std::vector<std::pair<std::string, std::int64_t>> numbers);
    void complete(EntryPtr entry) noexcept;
    void fail(EntryPtr entry, std::exception_ptr error) noexcept;
    void cancel(Entry& entry, std::exception_ptr error) noexcept;
    void reject(Entry& entry, std::exception_ptr error) noexcept;  // a queued request that was never admitted
    void release(Entry& entry) noexcept;
    void cancel_all() noexcept;
    void refresh_kv_snapshot() noexcept;

    InferenceBackend& backend_;
    const EngineLoopOptions options_;
    const ModelInfo model_info_;
    std::unique_ptr<SchedulingPolicy> policy_;  // used only on the engine thread (swapped under mu_)
    int page_size_ = 0;
    int total_pages_ = 0;
    int max_sequence_tokens_ = 0;
    RequestIdGenerator ids_;

    mutable std::mutex mu_;  // guards everything down to stats_
    std::condition_variable cv_;
    bool stop_ = false;
    std::uint64_t next_order_ = 0;  // arrival order of submitted requests
    std::unique_ptr<SchedulingPolicy> next_policy_;  // set_policy(): taken at the next step
    std::string policy_name_;
    std::unordered_set<std::string> unfinished_;      // ids of submitted requests not finished yet
    DecisionLog decision_log_;  // guarded by mu_
    std::uint64_t step_ = 0;    // scheduling passes so far (the "step" of an explanation)
    LatencyHistogram queue_hist_;  // guarded by mu_, like the stats
    LatencyHistogram ttft_hist_;
    LatencyHistogram itl_hist_;
    LatencyHistogram e2e_hist_;
    std::vector<double> itl_samples_;  // engine thread only: gaps of this step, added to itl_hist_ under mu_
    std::unordered_set<std::string> cancel_requests_;  // cancel_request(): applied at the next step
    std::deque<EntryPtr> pending_;
    EngineLoopStats stats_;
    KvCacheStats kv_snapshot_;

    std::vector<EntryPtr> active_;  // engine thread only; in admission order

    std::mutex join_mu_;
    std::thread thread_;  // last member: started after everything above is initialised
};

}  // namespace neuroserve::core
