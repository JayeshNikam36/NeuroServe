#include "neuroserve/core/engine_loop.hpp"

#include <algorithm>
#include <iterator>
#include <type_traits>
#include <utility>

#include "neuroserve/core/kv_block_manager.hpp"
#include "neuroserve/core/log.hpp"
#include "neuroserve/core/nvtx.hpp"

namespace neuroserve::core {

struct RequestHandle::State {
    std::string id;       // set before the handle is shared; never changes
    bool stream = false;  // set before the handle is shared; never changes
    std::mutex mu;
    std::condition_variable cv;
    bool done = false;
    GenerationResult result;
    std::exception_ptr error;
    std::vector<TokenId> streamed;  // stream only: every token generated so far, in order
    std::size_t read = 0;           // stream only: how many of them read_tokens() has returned
};

struct EngineLoop::Entry {
    Entry(std::string request_id, GenerationParams generation_params)
        : request(std::move(request_id)),
          params(std::move(generation_params)),
          state(std::make_shared<RequestHandle::State>()) {
        state->id = request.id();
        state->stream = params.stream;
    }

    Request request;
    GenerationParams params;
    std::shared_ptr<RequestHandle::State> state;
    std::uint64_t order = 0;  // arrival order; preempted requests re-enter the queue in this order
    // What a prefill must compute: the prompt, then every token generated so far. A preempted
    // request keeps its tokens and resumes from them (recompute, Day 38): nothing is generated twice.
    std::vector<TokenId> prefill_input() const {
        std::vector<TokenId> input = params.prompt;
        input.insert(input.end(), tokens.begin(), tokens.end());
        return input;
    }
    int cached = 0;           // tokens in the KV cache (prompt + fed-back generated tokens)
    int waited_steps = 0;     // engine steps spent in the queue since it was (re)queued
    SeqId seq = 0;
    bool has_seq = false;
    int own_pages = 0;        // pages only this request holds: what preempting it would give back
    std::vector<TokenId> tokens;
    Request::Clock::time_point first_token_at{};
    FinishReason finish = FinishReason::kLength;
    // Inter-token latency, over new tokens only: tokens regenerated after a preemption are not
    // timed again (the client already has them), but the stall before the next new token is.
    std::size_t timed_tokens = 0;
    Request::Clock::time_point last_token_at{};
    double itl_sum_ms = 0.0;
    double itl_max_ms = 0.0;
    std::string last_defer_reason;  // a deferral is logged when its reason changes, not every step
};

namespace {

constexpr std::string_view kLogger = "neuroserve.engine";

std::string describe(const std::exception_ptr& error) {
    try {
        std::rethrow_exception(error);
    } catch (const std::exception& e) {
        return e.what();
    } catch (...) {
        return "unknown error";
    }
}

double ms_between(Request::Clock::time_point from, Request::Clock::time_point to) {
    return std::chrono::duration<double, std::milli>(to - from).count();
}

void deliver_result(RequestHandle::State& state, GenerationResult result) {
    {
        std::lock_guard lock(state.mu);
        state.result = std::move(result);
        state.done = true;
    }
    state.cv.notify_all();
}

void deliver_error(RequestHandle::State& state, std::exception_ptr error) {
    {
        std::lock_guard lock(state.mu);
        state.error = std::move(error);
        state.done = true;
    }
    state.cv.notify_all();
}

}  // namespace

std::string_view to_string(FinishReason reason) noexcept {
    switch (reason) {
        case FinishReason::kStop:   return "stop";
        case FinishReason::kLength: return "length";
    }
    return "unknown";
}

// ---- RequestHandle ---------------------------------------------------------------------------

RequestHandle::RequestHandle(std::shared_ptr<State> state) : state_(std::move(state)) {}

const std::string& RequestHandle::id() const noexcept { return state_->id; }

bool RequestHandle::wait_for(std::chrono::milliseconds timeout) const {
    std::unique_lock lock(state_->mu);
    return state_->cv.wait_for(lock, timeout, [this] { return state_->done; });
}

GenerationResult RequestHandle::wait() const {
    std::unique_lock lock(state_->mu);
    state_->cv.wait(lock, [this] { return state_->done; });
    if (state_->error) {
        std::rethrow_exception(state_->error);
    }
    return state_->result;
}

RequestHandle::TokenBatch RequestHandle::read_tokens(std::chrono::milliseconds timeout) const {
    std::unique_lock lock(state_->mu);
    if (!state_->stream) {
        throw std::logic_error("read_tokens: request " + state_->id + " was not submitted with stream = true");
    }
    state_->cv.wait_for(lock, timeout, [this] { return state_->read < state_->streamed.size() || state_->done; });
    TokenBatch batch;
    batch.tokens.assign(state_->streamed.begin() + static_cast<std::ptrdiff_t>(state_->read), state_->streamed.end());
    state_->read = state_->streamed.size();
    batch.finished = state_->done;  // every token is published before the request finishes
    return batch;
}

// ---- EngineLoop: caller side -----------------------------------------------------------------

EngineLoop::EngineLoop(InferenceBackend& backend, EngineLoopOptions options,
                       std::unique_ptr<SchedulingPolicy> policy)
    : backend_(backend),
      options_(std::move(options)),
      model_info_(backend.model_info()),
      policy_(policy ? std::move(policy) : make_policy(options_.policy, options_.policy_options)),
      decision_log_(static_cast<std::size_t>(std::max(options_.decision_log_capacity, 0))) {
    if (options_.max_batch < 1) {
        throw std::invalid_argument("max_batch must be at least 1");
    }
    if (options_.max_batch_tokens < 1) {
        throw std::invalid_argument("max_batch_tokens must be at least 1");
    }
    if (options_.max_queue < 1) {
        throw std::invalid_argument("max_queue must be at least 1");
    }
    if (options_.queue_timeout_ms < 0) {
        throw std::invalid_argument("queue_timeout_ms must be 0 (no timeout) or more");
    }
    const KvCacheStats kv = backend.kv_stats();
    page_size_ = kv.page_size_tokens;
    total_pages_ = kv.total_pages;
    max_sequence_tokens_ = kv.max_pages_per_sequence * kv.page_size_tokens;
    if (page_size_ < 1 || total_pages_ < 1 || max_sequence_tokens_ < 1) {
        throw std::invalid_argument("the backend reports an empty KV cache");
    }
    kv_snapshot_ = kv;
    policy_name_ = std::string(policy_->name());
    log::write(log::Level::kInfo, kLogger, "engine loop started",
               {{"backend", std::string(backend.name())}, {"policy", std::string(policy_->name())},
                {"max_batch", options_.max_batch}, {"max_batch_tokens", options_.max_batch_tokens},
                {"kv_admission", std::string(admission_name(options_.admission))},
                {"total_pages", total_pages_}, {"page_size_tokens", page_size_}});
    thread_ = std::thread([this] { run(); });
}

EngineLoop::~EngineLoop() { stop(); }

int EngineLoop::pages_for_tokens(long long tokens) const noexcept {
    return static_cast<int>((tokens + page_size_ - 1) / page_size_);
}

void EngineLoop::validate(const GenerationParams& params) const {
    if (params.prompt.empty()) {
        throw std::invalid_argument("the prompt must not be empty");
    }
    if (params.max_new_tokens < 1) {
        throw std::invalid_argument("max_new_tokens must be at least 1, got " +
                                    std::to_string(params.max_new_tokens));
    }
    if (params.priority < kLowPriority || params.priority > kHighPriority) {
        throw std::invalid_argument("priority must be 0 (low), 1 (normal) or 2 (high), got " +
                                    std::to_string(params.priority));
    }
    const auto prompt_len = static_cast<long long>(params.prompt.size());
    if (prompt_len > options_.max_batch_tokens) {  // its prefill could never fit in a step
        throw std::invalid_argument("the prompt has " + std::to_string(prompt_len) +
                                    " tokens; max_batch_tokens is " + std::to_string(options_.max_batch_tokens));
    }
    if (model_info_.max_prompt_tokens > 0 && prompt_len > model_info_.max_prompt_tokens) {
        throw std::invalid_argument("the prompt has " + std::to_string(prompt_len) +
                                    " tokens; the limit is " + std::to_string(model_info_.max_prompt_tokens));
    }
    if (model_info_.vocab_size > 0) {
        for (const TokenId token : params.prompt) {
            if (token < 0 || token >= model_info_.vocab_size) {
                throw std::invalid_argument("token id " + std::to_string(token) +
                                            " is outside the vocabulary (0.." +
                                            std::to_string(model_info_.vocab_size - 1) + ")");
            }
        }
    }
    if (prompt_len + params.max_new_tokens > max_sequence_tokens_) {
        throw std::invalid_argument("prompt plus max_new_tokens is " +
                                    std::to_string(prompt_len + params.max_new_tokens) +
                                    " tokens; a sequence holds at most " + std::to_string(max_sequence_tokens_));
    }
    // Every request must fit alone, so the oldest running request can always finish.
    if (pages_for_tokens(prompt_len + params.max_new_tokens) > total_pages_) {
        throw std::invalid_argument("the request needs more KV pages than the cache has");
    }
}

RequestHandle EngineLoop::submit(GenerationParams params) {
    auto entry = std::make_unique<Entry>(ids_.next(), std::move(params));
    try {
        validate(entry->params);
    } catch (const std::exception& e) {
        entry->request.transition_to(RequestState::kRejected);
        log::write(log::Level::kDebug, kLogger, "request rejected",
                   {{"engine_request_id", entry->request.id()},
                    {"request_id", entry->params.client_request_id}, {"error", std::string(e.what())}});
        std::lock_guard lock(mu_);
        ++stats_.rejected;
        record_locked(*entry, DecisionAction::kReject, "invalid",
                      {{"prompt_tokens", static_cast<std::int64_t>(entry->params.prompt.size())},
                       {"max_new_tokens", entry->params.max_new_tokens}});
        throw;
    }
    entry->request.transition_to(RequestState::kValidated);
    RequestHandle handle(entry->state);
    {
        std::lock_guard lock(mu_);
        if (stop_) {
            entry->request.transition_to(RequestState::kRejected);
            ++stats_.rejected;
            record_locked(*entry, DecisionAction::kReject, "engine_stopped", {});
            throw EngineStopped("the engine is stopped");
        }
        if (pending_.size() >= static_cast<std::size_t>(options_.max_queue)) {
            entry->request.transition_to(RequestState::kRejected);
            ++stats_.rejected;
            record_locked(*entry, DecisionAction::kReject, "queue_full",
                          {{"queued", static_cast<std::int64_t>(pending_.size())}, {"max_queue", options_.max_queue}});
            log::write(log::Level::kInfo, kLogger, "request rejected",
                       {{"engine_request_id", entry->request.id()}, {"request_id", entry->params.client_request_id},
                        {"reason", "queue full"}, {"max_queue", options_.max_queue}});
            throw QueueFull("the queue is full (" + std::to_string(pending_.size()) + " waiting, the limit is " +
                            std::to_string(options_.max_queue) + ")");
        }
        // From here on only the engine thread touches the entry; the mutex hands it over safely.
        entry->request.transition_to(RequestState::kQueued);
        entry->order = next_order_++;
        unfinished_.insert(entry->request.id());
        pending_.push_back(std::move(entry));
        ++stats_.submitted;
        stats_.pending = pending_.size();
    }
    cv_.notify_one();
    return handle;
}

void EngineLoop::stop() {
    {
        std::lock_guard lock(mu_);
        stop_ = true;
    }
    cv_.notify_all();
    std::lock_guard join_lock(join_mu_);  // two threads calling stop() must not both join
    if (thread_.joinable()) {
        thread_.join();
    }
}

bool EngineLoop::cancel_request(const std::string& request_id) {
    {
        std::lock_guard lock(mu_);
        if (stop_ || unfinished_.count(request_id) == 0) {
            return false;
        }
        cancel_requests_.insert(request_id);
    }
    cv_.notify_one();
    return true;
}

bool EngineLoop::stop_requested() const {
    std::lock_guard lock(mu_);
    return stop_;
}

EngineLoopStats EngineLoop::stats() const {
    std::lock_guard lock(mu_);
    return stats_;
}

std::string EngineLoop::policy_name() const {
    std::lock_guard lock(mu_);
    return policy_name_;
}

void EngineLoop::set_policy(std::unique_ptr<SchedulingPolicy> policy) {
    if (!policy) {
        throw std::invalid_argument("set_policy: no policy");
    }
    std::lock_guard lock(mu_);
    next_policy_ = std::move(policy);
}

KvCacheStats EngineLoop::kv_stats() const {
    std::lock_guard lock(mu_);
    return kv_snapshot_;
}

namespace {

using ProfileClock = std::chrono::steady_clock;
static_assert(std::is_same_v<ProfileClock, Request::Clock>, "ms_between (above) takes the profile's time points");

void add_to(StepRecord& sum, const StepRecord& r) {
    sum.housekeeping_ms += r.housekeeping_ms;
    sum.schedule_ms += r.schedule_ms;
    sum.preempt_ms += r.preempt_ms;
    sum.prefill_ms += r.prefill_ms;
    sum.decode_ms += r.decode_ms;
    sum.host_ms += r.host_ms;
    sum.invariants_ms += r.invariants_ms;
    sum.total_ms += r.total_ms;
    sum.prefill_requests += r.prefill_requests;
    sum.prefill_tokens += r.prefill_tokens;
    sum.decode_batch += r.decode_batch;
}

}  // namespace

StepProfile EngineLoop::step_profile(bool with_records) const {
    std::lock_guard lock(mu_);
    StepProfile out = profile_;
    out.elapsed_ms = ms_between(profile_origin_, ProfileClock::now());
    if (with_records) {
        out.records.reserve(profile_ring_.size());
        // Oldest first: once the ring is full, the oldest record is the one about to be overwritten.
        const std::size_t first = profile_ring_.size() < static_cast<std::size_t>(options_.step_profile_capacity)
                                      ? 0
                                      : profile_next_;
        for (std::size_t i = 0; i < profile_ring_.size(); ++i) {
            out.records.push_back(profile_ring_[(first + i) % profile_ring_.size()]);
        }
    }
    return out;
}

void EngineLoop::reset_step_profile() {
    std::lock_guard lock(mu_);
    profile_origin_ = ProfileClock::now();
    profile_ = StepProfile{};
    profile_ring_.clear();
    profile_next_ = 0;
}

void EngineLoop::record_step_locked(ProfileClock::time_point waited_from, ProfileClock::time_point started,
                                    StepRecord record) {
    if (started < profile_origin_) {
        return;  // the profile was reset during this step: it belongs to neither profile
    }
    profile_.idle_ms += ms_between(std::max(waited_from, profile_origin_), started);
    ++profile_.steps;
    if (record.prefill_requests > 0 || record.decode_batch > 0) {
        ++profile_.busy_steps;
    }
    record.start_ms = ms_between(profile_origin_, started);
    add_to(profile_.totals, record);
    const auto capacity = static_cast<std::size_t>(std::max(0, options_.step_profile_capacity));
    if (capacity == 0) {
        return;
    }
    if (profile_ring_.size() < capacity) {
        profile_ring_.push_back(record);
    } else {
        profile_ring_[profile_next_] = record;
        profile_next_ = (profile_next_ + 1) % capacity;
        ++profile_.dropped;
    }
}

// ---- EngineLoop: engine thread ---------------------------------------------------------------

void EngineLoop::run() noexcept {
    try {
        loop();
    } catch (...) {
        // Only a bug gets here (e.g. a policy breaking a limit). cancel_all() still answers every
        // waiting caller, and the engine refuses new work.
        log::write(log::Level::kError, kLogger, "engine loop crashed; cancelling all requests",
                   {{"error", describe(std::current_exception())}});
    }
    cancel_all();
}

void EngineLoop::loop() {
    for (;;) {
        std::vector<EntryPtr> admitted;
        std::vector<EntryPtr> preempted;
        const auto waited_from = ProfileClock::now();
        {
            std::unique_lock lock(mu_);
            // Sleep only when there is nothing to do; while requests are active, keep stepping.
            cv_.wait(lock, [this] { return stop_ || !pending_.empty() || !active_.empty(); });
            if (stop_) {
                return;
            }
        }
        // The step's phases, timed on the host (docs/profiling.md) and marked for Nsight Systems.
        const NvtxRange step_range("ns.step");
        const auto started = ProfileClock::now();
        StepRecord timing;
        step_prefill_ms_ = 0.0;
        step_decode_ms_ = 0.0;
        apply_cancellations();  // first: cancelled requests free their pages before this step's decisions
        expire_queued();
        const int free_pages = backend_.kv_stats().free_pages;  // backend calls: outside the lock
        refresh_own_pages();
        auto mark = ProfileClock::now();
        timing.housekeeping_ms = ms_between(started, mark);
        {
            const NvtxRange range("ns.schedule");
            std::lock_guard lock(mu_);
            schedule_locked(free_pages, admitted, preempted);
        }
        auto now = ProfileClock::now();
        timing.schedule_ms = ms_between(mark, now);
        mark = now;
        StepStats step;
        step.preempted = preempted.size();
        preempt(std::move(preempted));  // first: admissions may count on the pages it frees
        now = ProfileClock::now();
        timing.preempt_ms = ms_between(mark, now);
        // Prefill the admitted requests; they join the decode batch from the next step on.
        std::vector<EntryPtr> joining;
        for (const auto& entry : admitted) {
            ++step.prefill_requests;
            step.prefill_tokens += static_cast<int>(entry->params.prompt.size() + entry->tokens.size());
        }
        if (!admitted.empty()) {
            start_all(std::move(admitted), joining);
        }
        step.decode_tokens = static_cast<int>(active_.size());
        if (!active_.empty()) {
            decode_active();
        }
        for (auto& entry : joining) {
            active_.push_back(std::move(entry));
        }
        if (step.prefill_requests > 0 || step.decode_tokens > 0) {
            {
                std::lock_guard lock(mu_);
                stats_.prefill_tokens += static_cast<std::uint64_t>(step.prefill_tokens);
                stats_.decode_tokens += static_cast<std::uint64_t>(step.decode_tokens);
                stats_.max_step_tokens = std::max(stats_.max_step_tokens, step.prefill_tokens + step.decode_tokens);
                flush_itl_samples_locked();
            }
            if (options_.on_step) {
                options_.on_step(step);
            }
        }
        refresh_kv_snapshot();
        const auto checks_from = ProfileClock::now();
        if (options_.check_invariants) {
            const NvtxRange range("ns.invariants");
            check_kv_bookkeeping();  // a violation ends the loop (run() logs it and cancels everything)
        }
        const auto finished = ProfileClock::now();
        timing.invariants_ms = ms_between(checks_from, finished);
        timing.prefill_ms = step_prefill_ms_;
        timing.decode_ms = step_decode_ms_;
        timing.total_ms = ms_between(started, finished);
        timing.host_ms = std::max(0.0, timing.total_ms - timing.housekeeping_ms - timing.schedule_ms -
                                           timing.preempt_ms - timing.prefill_ms - timing.decode_ms -
                                           timing.invariants_ms);
        timing.prefill_requests = static_cast<int>(step.prefill_requests);
        timing.prefill_tokens = step.prefill_tokens;
        timing.decode_batch = step.decode_tokens;
        std::lock_guard lock(mu_);
        stats_.active = active_.size();
        record_step_locked(waited_from, started, timing);
    }
}

void EngineLoop::refresh_own_pages() {
    for (const auto& entry : active_) {
        entry->own_pages = backend_.sequence_pages(entry->seq).exclusive;
    }
}

void EngineLoop::check_kv_bookkeeping() const {
    backend_.check_kv_invariants();
    // I11: the loop plans with the same numbers the backend holds. Each running request's cached
    // tokens must be its sequence's length and cover its pages, and the used pages must be the
    // requests' own pages plus the shared ones, each counted once.
    long long own = 0;
    for (const auto& entry : active_) {
        if (!entry->has_seq) {
            throw KvInvariantViolation("I12: running request " + entry->request.id() + " has no sequence");
        }
        const int length = backend_.sequence_length(entry->seq);
        if (length != entry->cached) {
            throw KvInvariantViolation("I11: request " + entry->request.id() + " has " + std::to_string(entry->cached) +
                                       " cached tokens in the loop's view but " + std::to_string(length) +
                                       " in the backend");
        }
        const SequencePages pages = backend_.sequence_pages(entry->seq);
        if (pages.held != pages_for_tokens(entry->cached)) {
            throw KvInvariantViolation("I11: request " + entry->request.id() + " holds " + std::to_string(pages.held) +
                                       " pages for " + std::to_string(entry->cached) + " cached tokens, which need " +
                                       std::to_string(pages_for_tokens(entry->cached)));
        }
        own += pages.exclusive;
    }
    // I12: finished, failed, cancelled and preempted requests hold nothing.
    const int live = backend_.live_sequences();
    if (live >= 0 && static_cast<std::size_t>(live) != active_.size()) {
        throw KvInvariantViolation("I12: the backend holds " + std::to_string(live) + " sequences for " +
                                   std::to_string(active_.size()) + " running requests");
    }
    {
        std::lock_guard lock(mu_);
        for (const auto& entry : pending_) {
            if (entry->has_seq) {
                throw KvInvariantViolation("I12: queued request " + entry->request.id() + " still holds a sequence");
            }
        }
    }
    const KvCacheStats kv = backend_.kv_stats();
    const int used = kv.total_pages - kv.free_pages;
    if (used != own + kv.shared_pages) {
        throw KvInvariantViolation("I11: the running requests hold " + std::to_string(own) + " pages of their own and " +
                                   std::to_string(kv.shared_pages) + " shared, the backend reports " +
                                   std::to_string(used) + " used");
    }
}

void EngineLoop::apply_cancellations() {
    std::vector<EntryPtr> cancelled;
    {
        std::lock_guard lock(mu_);
        if (cancel_requests_.empty()) {
            return;
        }
        const auto wanted = [this](const EntryPtr& entry) { return cancel_requests_.count(entry->request.id()) > 0; };
        for (auto& entry : pending_) {
            if (wanted(entry)) {
                cancelled.push_back(std::move(entry));
            }
        }
        for (auto& entry : active_) {
            if (wanted(entry)) {
                cancelled.push_back(std::move(entry));
            }
        }
        cancel_requests_.clear();  // ids that finished in the meantime have nothing left to cancel
        std::erase_if(pending_, [](const EntryPtr& entry) { return !entry; });
        std::erase_if(active_, [](const EntryPtr& entry) { return !entry; });
        stats_.pending = pending_.size();
        stats_.active = active_.size();
    }
    const auto error = std::make_exception_ptr(RequestCancelled("request cancelled"));
    for (auto& entry : cancelled) {
        log::write(log::Level::kInfo, kLogger, "request cancelled",
                   {{"engine_request_id", entry->request.id()}, {"request_id", entry->params.client_request_id},
                    {"state", std::string(to_string(entry->request.state()))},
                    {"generated_tokens", entry->tokens.size()}});
        cancel(*entry, error);  // frees the sequence and its KV pages, then answers the caller
    }
}

void EngineLoop::expire_queued() {
    if (options_.queue_timeout_ms <= 0) {
        return;
    }
    const auto now = Request::Clock::now();
    const auto limit = std::chrono::milliseconds(options_.queue_timeout_ms);
    std::vector<EntryPtr> expired;
    {
        std::lock_guard lock(mu_);
        for (auto& entry : pending_) {
            // Only requests never admitted: a preempted request already streamed tokens and is
            // restarted as soon as there is room, so it is not rejected halfway.
            const auto queued = entry->request.entered_at(RequestState::kQueued);
            if (entry->request.preemption_count() == 0 && queued && now - *queued >= limit) {
                const auto waited = std::chrono::duration_cast<std::chrono::milliseconds>(now - *queued);
                record_locked(*entry, DecisionAction::kReject, "queue_timeout",
                              {{"waited_ms", waited.count()}, {"queue_timeout_ms", options_.queue_timeout_ms}});
                expired.push_back(std::move(entry));
            }
        }
        if (expired.empty()) {
            return;
        }
        std::erase_if(pending_, [](const EntryPtr& entry) { return !entry; });
        stats_.pending = pending_.size();
    }
    for (auto& entry : expired) {
        const auto waited = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - *entry->request.entered_at(RequestState::kQueued));
        log::write(log::Level::kInfo, kLogger, "request rejected",
                   {{"engine_request_id", entry->request.id()}, {"request_id", entry->params.client_request_id},
                    {"reason", "queue timeout"}, {"waited_ms", waited.count()},
                    {"queue_timeout_ms", options_.queue_timeout_ms}});
        reject(*entry, std::make_exception_ptr(QueueTimeout(
                           "the request waited " + std::to_string(waited.count()) + " ms in the queue (limit " +
                           std::to_string(options_.queue_timeout_ms) + " ms)")));
    }
}

void EngineLoop::record_locked(const Entry& entry, DecisionAction action, std::string reason,
                               std::vector<std::pair<std::string, std::int64_t>> numbers) {
    Explanation e;
    e.step = step_;
    e.unix_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::system_clock::now().time_since_epoch()).count();
    e.request_id = entry.request.id();
    e.client_request_id = entry.params.client_request_id;
    e.action = action;
    e.reason = std::move(reason);
    e.policy = policy_name_;
    e.numbers = std::move(numbers);
    decision_log_.add(std::move(e));
}

// Explains one scheduling decision with the numbers it was made on. The reasons are derived by the
// engine, the same way for every policy: replay the decision against the step's resources and,
// for each request the policy did not admit, name the first resource that was short. If none was,
// the policy itself held the request back (its ordering rule or its prefill cap).
void EngineLoop::explain_locked(const SchedulerInput& input, const ScheduleDecision& decision) {
    using N = std::int64_t;
    N growth = 0;
    for (const auto& r : input.running) {
        growth += r.next_step_pages;
    }
    N pages_left = input.free_pages;
    N kept = static_cast<N>(input.running.size());
    for (const std::size_t i : decision.preempt) {
        const auto& r = input.running[i];
        record_locked(*active_[i], DecisionAction::kPreempt, "kv_pressure",
                      {{"free_pages", input.free_pages}, {"pages_needed_to_grow", growth},
                       {"pages_held", r.pages}, {"generated_tokens", r.generated_tokens}, {"priority", r.priority}});
        pages_left += r.pages;
        growth -= r.next_step_pages;
        --kept;
    }
    pages_left -= growth;
    N slots_left = input.max_batch - kept;
    N budget_left = input.max_step_tokens - kept;  // every running request decodes one token

    std::vector<bool> admitted(input.queued.size(), false);
    for (const std::size_t i : decision.admit) {
        const auto& q = input.queued[i];
        record_locked(*pending_[i], DecisionAction::kAdmit, "admitted",
                      {{"prompt_tokens", q.prompt_tokens}, {"pages_needed", q.pages}, {"free_pages", pages_left},
                       {"slots_free", slots_left}, {"token_budget_left", budget_left},
                       {"waited_steps", q.waited_steps}, {"queue_position", static_cast<N>(i)}, {"priority", q.priority}});
        pending_[i]->last_defer_reason.clear();
        admitted[i] = true;
        pages_left -= q.pages;
        budget_left -= q.prompt_tokens;
        --slots_left;
    }
    for (std::size_t i = 0; i < input.queued.size(); ++i) {
        if (admitted[i]) {
            continue;
        }
        const auto& q = input.queued[i];
        std::string reason = !decision.preempt.empty() ? "kv_pressure"      // no admissions while memory is short
                           : slots_left <= 0           ? "batch_full"
                           : q.pages > pages_left      ? "kv_pages"
                           : q.prompt_tokens > budget_left ? "token_budget"
                                                           : "held_by_policy";
        Entry& entry = *pending_[i];
        if (reason == entry.last_defer_reason) {
            continue;  // already explained; logging it every step would flood the log
        }
        entry.last_defer_reason = reason;
        record_locked(entry, DecisionAction::kDefer, std::move(reason),
                      {{"prompt_tokens", q.prompt_tokens}, {"pages_needed", q.pages}, {"free_pages", pages_left},
                       {"slots_free", slots_left}, {"token_budget_left", budget_left},
                       {"waited_steps", q.waited_steps}, {"queue_position", static_cast<N>(i)}, {"priority", q.priority}});
    }
}

DecisionPage EngineLoop::decisions(std::uint64_t since, std::size_t limit) const {
    std::lock_guard lock(mu_);
    return decision_log_.since(since, limit);
}

void EngineLoop::schedule_locked(int free_pages, std::vector<EntryPtr>& admitted,
                                 std::vector<EntryPtr>& preempted) {
    if (next_policy_) {  // set_policy() was called: switch between steps
        policy_ = std::move(next_policy_);
        log::write(log::Level::kInfo, kLogger, "scheduling policy switched",
                   {{"from", policy_name_}, {"to", std::string(policy_->name())}});
        policy_name_ = std::string(policy_->name());
    }
    if (pending_.empty() && active_.empty()) {
        return;
    }
    ++step_;
    // The policy decides; the loop checks the decision and applies it.
    SchedulerInput input;
    input.max_batch = options_.max_batch;
    input.free_pages = free_pages;
    // Worst-case admission reserves every running request's remaining growth up to prompt +
    // max_new_tokens: only what is left is free for admissions. This step's growth stays in
    // free_pages, where the policy counts it as next_step_pages, so it is not reserved twice.
    const bool worst_case = options_.admission == KvAdmission::kWorstCase;
    if (worst_case) {
        long long reserved = 0;
        for (const auto& entry : active_) {
            const long long whole = static_cast<long long>(entry->params.prompt.size()) + entry->params.max_new_tokens;
            const int growth = pages_for_tokens(entry->cached + 1) - pages_for_tokens(entry->cached);
            reserved += pages_for_tokens(whole) - pages_for_tokens(entry->cached) - growth;
        }
        input.free_pages = static_cast<int>(std::max<long long>(0, free_pages - reserved));
    }
    input.max_step_tokens = options_.max_batch_tokens;
    const auto now = Request::Clock::now();
    const auto age_ms = [now](const Entry& entry) {  // since arrival; preemption does not reset it
        const auto received = *entry.request.entered_at(RequestState::kReceived);
        return static_cast<long long>(std::chrono::duration_cast<std::chrono::milliseconds>(now - received).count());
    };
    input.queued.reserve(pending_.size());
    for (const auto& entry : pending_) {
        // A resumed request's prefill covers its prompt and the tokens it kept: all of its pages.
        // Its prefill counts as at most one step's budget, so it can always be admitted (alone, at
        // worst); that step then exceeds max_batch_tokens by the recompute (chunked prefill,
        // Days 52-53, will spread it). A new request's prompt is within the budget (submit checks).
        const auto input_tokens = static_cast<long long>(entry->params.prompt.size() + entry->tokens.size());
        const auto counted = std::min<long long>(input_tokens, options_.max_batch_tokens);
        // On demand it needs its prefill's pages; worst case, every page it can ever hold.
        const long long whole = static_cast<long long>(entry->params.prompt.size()) + entry->params.max_new_tokens;
        input.queued.push_back({entry->request.id(), static_cast<int>(counted), entry->params.max_new_tokens,
                                pages_for_tokens(worst_case ? whole : input_tokens), entry->waited_steps,
                                entry->params.priority, age_ms(*entry)});
    }
    input.running.reserve(active_.size());
    for (const auto& entry : active_) {
        // A shared page is not freed when one of its holders is preempted, so the policy counts only
        // the request's own pages. Growth is always a new page of its own (shared pages are full).
        const int growth = pages_for_tokens(entry->cached + 1) - pages_for_tokens(entry->cached);
        input.running.push_back({entry->request.id(), entry->own_pages, growth,
                                 static_cast<int>(entry->tokens.size()), entry->params.max_new_tokens,
                                 entry->params.priority, age_ms(*entry)});
    }
    const ScheduleDecision decision = policy_->schedule(input);
    check_decision(input, decision);  // a faulty policy stops the engine loudly (see run())
    explain_locked(input, decision);

    for (const std::size_t index : decision.preempt) {
        preempted.push_back(std::move(active_[index]));  // leaves an empty slot, removed below
    }
    std::erase_if(active_, [](const EntryPtr& entry) { return !entry; });
    for (const std::size_t index : decision.admit) {
        EntryPtr& slot = pending_[index];
        slot->request.transition_to(RequestState::kAdmitted);
        admitted.push_back(std::move(slot));
    }
    std::erase_if(pending_, [](const EntryPtr& entry) { return !entry; });
    for (auto& entry : pending_) {
        ++entry->waited_steps;  // passed over this step
    }
    stats_.pending = pending_.size();
}

void EngineLoop::preempt(std::vector<EntryPtr> victims) {
    if (victims.empty()) {
        return;
    }
    for (auto& entry : victims) {
        const auto kept = entry->tokens.size();
        release(*entry);  // frees the sequence; its indexed pages stay CACHED for the resumption
        entry->cached = 0;  // its tokens are kept: it resumes from prompt + tokens, never regenerating
        entry->waited_steps = 0;
        entry->last_defer_reason.clear();  // back in the queue: its next deferral is news again
        entry->request.transition_to(RequestState::kPreempted);
        entry->request.transition_to(RequestState::kQueued);
        log::write(log::Level::kInfo, kLogger, "request preempted",
                   {{"engine_request_id", entry->request.id()}, {"request_id", entry->params.client_request_id},
                    {"kept_tokens", kept}});
    }
    // Back to the front of the queue, in arrival order: they were waiting before anything queued now.
    std::sort(victims.begin(), victims.end(),
              [](const EntryPtr& a, const EntryPtr& b) { return a->order < b->order; });
    std::lock_guard lock(mu_);
    stats_.preemptions += victims.size();
    pending_.insert(pending_.begin(), std::make_move_iterator(victims.begin()),
                    std::make_move_iterator(victims.end()));
    stats_.pending = pending_.size();
}

void EngineLoop::start_all(std::vector<EntryPtr> admitted, std::vector<EntryPtr>& joining) {
    // A sequence each. A resumed request recomputes its prompt and the tokens it already generated
    // (the client has them); the prefill's output is its next new token.
    std::vector<EntryPtr> ready;
    std::vector<std::vector<TokenId>> inputs;
    for (auto& entry : admitted) {
        try {
            entry->seq = backend_.create_sequence();
            entry->has_seq = true;
            entry->request.transition_to(RequestState::kPrefill);
            inputs.push_back(entry->prefill_input());
            ready.push_back(std::move(entry));
        } catch (...) {
            fail(std::move(entry), std::current_exception());  // only this request fails
        }
    }
    if (ready.empty()) {
        return;
    }
    std::vector<PrefillItem> items(ready.size());
    for (std::size_t i = 0; i < ready.size(); ++i) {
        items[i].seq = ready[i]->seq;
        items[i].prompt = inputs[i];
    }
    const auto called = ProfileClock::now();
    try {
        const NvtxRange range("ns.prefill");
        backend_.prefill_batch(items);  // all together when the backend can (docs/prefill-attention.md)
    } catch (...) {
        const auto error = std::current_exception();  // the backend could not try: every request fails
        for (auto& item : items) {
            item.error = error;
        }
    }
    step_prefill_ms_ += ms_between(called, ProfileClock::now());

    for (std::size_t i = 0; i < ready.size(); ++i) {
        EntryPtr entry = std::move(ready[i]);
        if (items[i].error) {
            fail(std::move(entry), items[i].error);  // only this request fails
            continue;
        }
        try {
            entry->cached = static_cast<int>(inputs[i].size());
            if (entry->first_token_at == Request::Clock::time_point{}) {
                entry->first_token_at = Request::Clock::now();  // a restarted request keeps its first token time
            }
            if (append_token(*entry, items[i].first)) {
                complete(std::move(entry));  // stop token or max_new_tokens == 1: no decode needed
                continue;
            }
            entry->request.transition_to(RequestState::kDecode);
            joining.push_back(std::move(entry));
        } catch (...) {
            fail(std::move(entry), std::current_exception());
        }
    }
}

void EngineLoop::decode_active() {
    std::vector<SeqId> seqs;
    std::vector<TokenId> last_tokens;
    seqs.reserve(active_.size());
    last_tokens.reserve(active_.size());
    for (const auto& entry : active_) {
        seqs.push_back(entry->seq);
        last_tokens.push_back(entry->tokens.back());
    }

    std::vector<TokenId> next;
    try {
        const NvtxRange range("ns.decode");
        const auto called = ProfileClock::now();
        next = backend_.decode_step(seqs, last_tokens);
        step_decode_ms_ += ms_between(called, ProfileClock::now());
        if (next.size() != seqs.size()) {
            throw std::logic_error("decode_step returned " + std::to_string(next.size()) +
                                   " tokens for " + std::to_string(seqs.size()) + " sequences");
        }
    } catch (...) {
        // The backend cannot say which sequence caused the error, so the whole batch fails.
        // The loop itself keeps serving later requests.
        const auto error = std::current_exception();
        log::write(log::Level::kError, kLogger, "decode step failed; failing its batch",
                   {{"batch_size", seqs.size()}, {"error", describe(error)}});
        auto failed = std::move(active_);
        active_.clear();
        for (auto& entry : failed) {
            fail(std::move(entry), error);
        }
        return;
    }

    {
        std::lock_guard lock(mu_);
        ++stats_.decode_steps;
        stats_.max_batch_seen = std::max(stats_.max_batch_seen, seqs.size());
    }
    for (auto& entry : active_) {
        ++entry->cached;  // the fed-back token is now in the KV cache
    }

    std::vector<EntryPtr> still_running;
    still_running.reserve(active_.size());
    for (std::size_t i = 0; i < active_.size(); ++i) {
        if (append_token(*active_[i], next[i])) {
            complete(std::move(active_[i]));  // finished on this step: no wasted decode work
        } else {
            still_running.push_back(std::move(active_[i]));
        }
    }
    active_ = std::move(still_running);
}

bool EngineLoop::append_token(Entry& entry, TokenId token) {
    entry.tokens.push_back(token);
    if (entry.tokens.size() > entry.timed_tokens) {  // a new token, not one regenerated after a preemption
        const auto now = Request::Clock::now();
        if (entry.timed_tokens > 0) {
            const double gap = ms_between(entry.last_token_at, now);
            entry.itl_sum_ms += gap;
            entry.itl_max_ms = std::max(entry.itl_max_ms, gap);
            itl_samples_.push_back(gap);
        }
        entry.last_token_at = now;
        entry.timed_tokens = entry.tokens.size();
    }
    if (entry.params.stream) {
        publish_token(entry, token);
    }
    const auto& stops = entry.params.stop_token_ids;
    if (std::find(stops.begin(), stops.end(), token) != stops.end()) {
        entry.finish = FinishReason::kStop;
        return true;
    }
    if (static_cast<int>(entry.tokens.size()) >= entry.params.max_new_tokens) {
        entry.finish = FinishReason::kLength;
        return true;
    }
    return false;
}

void EngineLoop::publish_token(Entry& entry, TokenId token) {
    const std::size_t index = entry.tokens.size() - 1;
    bool regenerated_differently = false;
    {
        std::lock_guard lock(entry.state->mu);
        auto& streamed = entry.state->streamed;
        if (index == streamed.size()) {
            streamed.push_back(token);
        } else {
            // Unreachable since Day 38: a resumed request recomputes the tokens its reader has and
            // only generates new ones. Kept as a check: a token published twice is logged.
            regenerated_differently = streamed[index] != token;
        }
    }
    entry.state->cv.notify_all();
    if (regenerated_differently) {
        log::write(log::Level::kError, kLogger, "restarted request generated a different token than it streamed",
                   {{"engine_request_id", entry.request.id()}, {"index", index}});
    }
}

void EngineLoop::flush_itl_samples_locked() {
    for (const double gap : itl_samples_) {
        itl_hist_.record(gap);
    }
    itl_samples_.clear();
}

LatencyStats EngineLoop::latency() const {
    std::lock_guard lock(mu_);
    return {summarize(queue_hist_), summarize(ttft_hist_), summarize(itl_hist_), summarize(e2e_hist_)};
}

void EngineLoop::release(Entry& entry) noexcept {
    if (entry.has_seq) {
        try {
            backend_.free_sequence(entry.seq);
        } catch (...) {
            // Nothing more can be done for this sequence; the request still finishes.
        }
        entry.has_seq = false;
    }
    refresh_kv_snapshot();  // before the caller is answered, so it sees its pages returned
}

void EngineLoop::refresh_kv_snapshot() noexcept {
    try {
        const KvCacheStats kv = backend_.kv_stats();
        std::lock_guard lock(mu_);
        kv_snapshot_ = kv;
    } catch (...) {
        // Keep the previous snapshot; it is only used for monitoring.
    }
}

void EngineLoop::complete(EntryPtr entry) noexcept {
    release(*entry);
    entry->request.transition_to(RequestState::kCompleted);

    const Request& request = entry->request;
    const auto received = *request.entered_at(RequestState::kReceived);
    GenerationResult result;
    result.request_id = request.id();
    result.finish_reason = entry->finish;
    result.prompt_tokens = static_cast<int>(entry->params.prompt.size());
    result.queue_ms = ms_between(*request.entered_at(RequestState::kQueued),
                                 *request.entered_at(RequestState::kAdmitted));
    result.ttft_ms = ms_between(received, entry->first_token_at);
    result.e2e_ms = ms_between(received, request.last_transition_at());
    if (entry->timed_tokens > 1) {
        result.itl_ms = entry->itl_sum_ms / static_cast<double>(entry->timed_tokens - 1);
        result.max_itl_ms = entry->itl_max_ms;
    }
    result.tokens = std::move(entry->tokens);

    log::write(log::Level::kInfo, kLogger, "request completed",
               {{"engine_request_id", result.request_id}, {"request_id", entry->params.client_request_id},
                {"priority", entry->params.priority},
                {"prompt_tokens", result.prompt_tokens}, {"completion_tokens", result.tokens.size()},
                {"finish_reason", std::string(to_string(result.finish_reason))},
                {"queue_ms", result.queue_ms}, {"ttft_ms", result.ttft_ms}, {"itl_ms", result.itl_ms},
                {"max_itl_ms", result.max_itl_ms}, {"e2e_ms", result.e2e_ms}});
    {
        std::lock_guard lock(mu_);
        ++stats_.completed;
        unfinished_.erase(entry->request.id());
        queue_hist_.record(result.queue_ms);
        ttft_hist_.record(result.ttft_ms);
        e2e_hist_.record(result.e2e_ms);
        flush_itl_samples_locked();  // so latency() includes this request's gaps once its caller has the result
    }
    deliver_result(*entry->state, std::move(result));
}

void EngineLoop::fail(EntryPtr entry, std::exception_ptr error) noexcept {
    release(*entry);
    entry->request.transition_to(RequestState::kFailed);
    log::write(log::Level::kError, kLogger, "request failed",
               {{"engine_request_id", entry->request.id()}, {"request_id", entry->params.client_request_id},
                {"error", describe(error)}});
    {
        std::lock_guard lock(mu_);
        ++stats_.failed;
        unfinished_.erase(entry->request.id());
    }
    deliver_error(*entry->state, std::move(error));
}

void EngineLoop::reject(Entry& entry, std::exception_ptr error) noexcept {
    entry.request.transition_to(RequestState::kRejected);  // never admitted: no sequence, no pages
    {
        std::lock_guard lock(mu_);
        ++stats_.timed_out;
        unfinished_.erase(entry.request.id());
    }
    deliver_error(*entry.state, std::move(error));
}

void EngineLoop::cancel(Entry& entry, std::exception_ptr error) noexcept {
    release(entry);
    entry.request.transition_to(RequestState::kCancelled);
    {
        std::lock_guard lock(mu_);
        ++stats_.cancelled;
        unfinished_.erase(entry.request.id());
    }
    deliver_error(*entry.state, std::move(error));
}

void EngineLoop::cancel_all() noexcept {
    std::deque<EntryPtr> pending;
    {
        std::lock_guard lock(mu_);
        stop_ = true;  // also when the loop died from a bug: later submits are refused
        pending.swap(pending_);
        stats_.pending = 0;
    }
    const auto error = std::make_exception_ptr(EngineStopped("request cancelled: the engine is stopping"));
    for (auto& entry : pending) {
        cancel(*entry, error);
    }
    for (auto& entry : active_) {
        cancel(*entry, error);
    }
    const auto cancelled = pending.size() + active_.size();
    active_.clear();
    {
        std::lock_guard lock(mu_);
        stats_.active = 0;
    }
    log::write(log::Level::kInfo, kLogger, "engine loop stopped", {{"cancelled_requests", cancelled}});
}

}  // namespace neuroserve::core
