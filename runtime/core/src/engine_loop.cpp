#include "neuroserve/core/engine_loop.hpp"

#include <algorithm>
#include <utility>

#include "neuroserve/core/log.hpp"

namespace neuroserve::core {

struct RequestHandle::State {
    std::string id;  // set before the handle is shared; never changes
    std::mutex mu;
    std::condition_variable cv;
    bool done = false;
    GenerationResult result;
    std::exception_ptr error;
};

struct EngineLoop::Entry {
    Entry(std::string request_id, GenerationParams generation_params)
        : request(std::move(request_id)),
          params(std::move(generation_params)),
          state(std::make_shared<RequestHandle::State>()) {
        state->id = request.id();
    }

    Request request;
    GenerationParams params;
    std::shared_ptr<RequestHandle::State> state;
    int pages = 0;          // KV pages reserved for the whole request
    bool reserved = false;  // pages are counted in reserved_pages_
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

// ---- EngineLoop: caller side -----------------------------------------------------------------

EngineLoop::EngineLoop(InferenceBackend& backend, EngineLoopOptions options)
    : backend_(backend), options_(options), model_info_(backend.model_info()) {
    if (options_.max_batch < 1) {
        throw std::invalid_argument("max_batch must be at least 1");
    }
    const KvCacheStats kv = backend.kv_stats();
    page_size_ = kv.page_size_tokens;
    total_pages_ = kv.total_pages;
    max_sequence_tokens_ = kv.max_pages_per_sequence * kv.page_size_tokens;
    if (page_size_ < 1 || total_pages_ < 1 || max_sequence_tokens_ < 1) {
        throw std::invalid_argument("the backend reports an empty KV cache");
    }
    kv_snapshot_ = kv;
    log::write(log::Level::kInfo, kLogger, "engine loop started",
               {{"backend", std::string(backend.name())}, {"max_batch", options_.max_batch},
                {"total_pages", total_pages_}, {"page_size_tokens", page_size_}});
    thread_ = std::thread([this] { run(); });
}

EngineLoop::~EngineLoop() { stop(); }

int EngineLoop::pages_for(const GenerationParams& params) const noexcept {
    const auto tokens = static_cast<long long>(params.prompt.size()) + params.max_new_tokens;
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
    const auto prompt_len = static_cast<long long>(params.prompt.size());
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
    if (pages_for(params) > total_pages_) {
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
    entry->pages = pages_for(entry->params);
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

KvCacheStats EngineLoop::kv_stats() const {
    std::lock_guard lock(mu_);
    return kv_snapshot_;
}

// ---- EngineLoop: engine thread ---------------------------------------------------------------

void EngineLoop::run() noexcept {
    try {
        loop();
    } catch (...) {
        // Only a bug gets here. cancel_all() still answers every waiting caller.
    }
    cancel_all();
}

void EngineLoop::loop() {
    for (;;) {
        std::vector<EntryPtr> admitted;
        {
            std::unique_lock lock(mu_);
            // Sleep only when there is nothing to do; while requests are active, keep stepping.
            cv_.wait(lock, [this] { return stop_ || !pending_.empty() || !active_.empty(); });
            if (stop_) {
                return;
            }
            admit_locked(admitted);
        }
        for (auto& entry : admitted) {
            start(std::move(entry));
        }
        if (!active_.empty()) {
            decode_active();
        }
        refresh_kv_snapshot();
        std::lock_guard lock(mu_);
        stats_.active = active_.size();
    }
}

void EngineLoop::admit_locked(std::vector<EntryPtr>& admitted) {
    // First come, first served. Stop at the first request that does not fit, so no request
    // can be overtaken forever by smaller ones (Day 13 moves this decision into a policy).
    while (!pending_.empty()) {
        Entry& next = *pending_.front();
        const auto running = active_.size() + admitted.size();
        if (running >= static_cast<std::size_t>(options_.max_batch) ||
            reserved_pages_ + next.pages > total_pages_) {
            break;
        }
        reserved_pages_ += next.pages;
        next.reserved = true;
        next.request.transition_to(RequestState::kAdmitted);
        admitted.push_back(std::move(pending_.front()));
        pending_.pop_front();
    }
    stats_.pending = pending_.size();
}

void EngineLoop::start(EntryPtr entry) {
    try {
        entry->seq = backend_.create_sequence();
        entry->has_seq = true;
        entry->request.transition_to(RequestState::kPrefill);
        const TokenId first = backend_.prefill(entry->seq, entry->params.prompt);
        entry->first_token_at = Request::Clock::now();
        if (append_token(*entry, first)) {
            complete(std::move(entry));  // stop token or max_new_tokens == 1: no decode needed
            return;
        }
        entry->request.transition_to(RequestState::kDecode);
        active_.push_back(std::move(entry));
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

void EngineLoop::release(Entry& entry) noexcept {
    if (entry.has_seq) {
        try {
            backend_.free_sequence(entry.seq);
        } catch (...) {
            // Nothing more can be done for this sequence; the request still finishes.
        }
        entry.has_seq = false;
    }
    if (entry.reserved) {
        reserved_pages_ -= entry.pages;
        entry.reserved = false;
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
