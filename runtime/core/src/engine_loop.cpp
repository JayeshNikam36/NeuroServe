#include "neuroserve/core/engine_loop.hpp"

#include <algorithm>
#include <iterator>
#include <utility>

#include "neuroserve/core/log.hpp"

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
    int cached = 0;           // tokens in the KV cache (prompt + fed-back generated tokens)
    int waited_steps = 0;     // engine steps spent in the queue since it was (re)queued
    SeqId seq = 0;
    bool has_seq = false;
    std::vector<TokenId> tokens;
    Request::Clock::time_point first_token_at{};
    FinishReason finish = FinishReason::kLength;
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
      policy_(policy ? std::move(policy) : make_policy(options_.policy, options_.policy_options)) {
    if (options_.max_batch < 1) {
        throw std::invalid_argument("max_batch must be at least 1");
    }
    if (options_.max_batch_tokens < 1) {
        throw std::invalid_argument("max_batch_tokens must be at least 1");
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
        throw;
    }
    entry->request.transition_to(RequestState::kValidated);
    RequestHandle handle(entry->state);
    {
        std::lock_guard lock(mu_);
        if (stop_) {
            entry->request.transition_to(RequestState::kRejected);
            ++stats_.rejected;
            throw EngineStopped("the engine is stopped");
        }
        // From here on only the engine thread touches the entry; the mutex hands it over safely.
        entry->request.transition_to(RequestState::kQueued);
        entry->order = next_order_++;
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
        {
            std::unique_lock lock(mu_);
            // Sleep only when there is nothing to do; while requests are active, keep stepping.
            cv_.wait(lock, [this] { return stop_ || !pending_.empty() || !active_.empty(); });
            if (stop_) {
                return;
            }
        }
        const int free_pages = backend_.kv_stats().free_pages;  // backend call: outside the lock
        {
            std::lock_guard lock(mu_);
            schedule_locked(free_pages, admitted, preempted);
        }
        StepStats step;
        step.preempted = preempted.size();
        preempt(std::move(preempted));  // first: admissions may count on the pages it frees
        // Prefill the admitted requests; they join the decode batch from the next step on.
        std::vector<EntryPtr> joining;
        for (auto& entry : admitted) {
            ++step.prefill_requests;
            step.prefill_tokens += static_cast<int>(entry->params.prompt.size());
            start(std::move(entry), joining);
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
            }
            if (options_.on_step) {
                options_.on_step(step);
            }
        }
        refresh_kv_snapshot();
        std::lock_guard lock(mu_);
        stats_.active = active_.size();
    }
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
    // The policy decides; the loop checks the decision and applies it.
    SchedulerInput input;
    input.max_batch = options_.max_batch;
    input.free_pages = free_pages;
    input.max_step_tokens = options_.max_batch_tokens;
    const auto now = Request::Clock::now();
    const auto age_ms = [now](const Entry& entry) {  // since arrival; preemption does not reset it
        const auto received = *entry.request.entered_at(RequestState::kReceived);
        return static_cast<long long>(std::chrono::duration_cast<std::chrono::milliseconds>(now - received).count());
    };
    input.queued.reserve(pending_.size());
    for (const auto& entry : pending_) {
        const auto prompt = static_cast<long long>(entry->params.prompt.size());
        input.queued.push_back({entry->request.id(), static_cast<int>(prompt), entry->params.max_new_tokens,
                                pages_for_tokens(prompt), entry->waited_steps,  // prefill; decodes next step
                                entry->params.priority, age_ms(*entry)});
    }
    input.running.reserve(active_.size());
    for (const auto& entry : active_) {
        const int held = pages_for_tokens(entry->cached);
        input.running.push_back({entry->request.id(), held, pages_for_tokens(entry->cached + 1) - held,
                                 static_cast<int>(entry->tokens.size()), entry->params.max_new_tokens,
                                 entry->params.priority, age_ms(*entry)});
    }
    const ScheduleDecision decision = policy_->schedule(input);
    check_decision(input, decision);  // a faulty policy stops the engine loudly (see run())

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
        const auto discarded = entry->tokens.size();
        release(*entry);  // frees the sequence and its KV pages
        entry->tokens.clear();  // restart from the prompt: greedy decoding regenerates the same tokens
        entry->cached = 0;
        entry->waited_steps = 0;
        entry->request.transition_to(RequestState::kPreempted);
        entry->request.transition_to(RequestState::kQueued);
        log::write(log::Level::kInfo, kLogger, "request preempted",
                   {{"engine_request_id", entry->request.id()}, {"request_id", entry->params.client_request_id},
                    {"discarded_tokens", discarded}});
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

void EngineLoop::start(EntryPtr entry, std::vector<EntryPtr>& joining) {
    try {
        entry->seq = backend_.create_sequence();
        entry->has_seq = true;
        entry->request.transition_to(RequestState::kPrefill);
        const TokenId first = backend_.prefill(entry->seq, entry->params.prompt);
        entry->cached = static_cast<int>(entry->params.prompt.size());
        if (entry->first_token_at == Request::Clock::time_point{}) {
            entry->first_token_at = Request::Clock::now();  // a restarted request keeps its first token time
        }
        if (append_token(*entry, first)) {
            complete(std::move(entry));  // stop token or max_new_tokens == 1: no decode needed
            return;
        }
        entry->request.transition_to(RequestState::kDecode);
        joining.push_back(std::move(entry));
    } catch (...) {
        fail(std::move(entry), std::current_exception());  // only this request fails
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
        next = backend_.decode_step(seqs, last_tokens);
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
            // A preempted request is generating again from its prompt. Its reader already has
            // this token; greedy decoding must reproduce it exactly.
            regenerated_differently = streamed[index] != token;
        }
    }
    entry.state->cv.notify_all();
    if (regenerated_differently) {
        log::write(log::Level::kError, kLogger, "restarted request generated a different token than it streamed",
                   {{"engine_request_id", entry.request.id()}, {"index", index}});
    }
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
    result.tokens = std::move(entry->tokens);

    log::write(log::Level::kInfo, kLogger, "request completed",
               {{"engine_request_id", result.request_id}, {"request_id", entry->params.client_request_id},
                {"priority", entry->params.priority},
                {"prompt_tokens", result.prompt_tokens}, {"completion_tokens", result.tokens.size()},
                {"finish_reason", std::string(to_string(result.finish_reason))},
                {"queue_ms", result.queue_ms}, {"ttft_ms", result.ttft_ms}, {"e2e_ms", result.e2e_ms}});
    {
        std::lock_guard lock(mu_);
        ++stats_.completed;
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
    }
    deliver_error(*entry->state, std::move(error));
}

void EngineLoop::cancel(Entry& entry, std::exception_ptr error) noexcept {
    release(entry);
    entry.request.transition_to(RequestState::kCancelled);
    {
        std::lock_guard lock(mu_);
        ++stats_.cancelled;
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
