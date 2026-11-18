#pragma once

#include <atomic>
#include <future>

#include "fake_backend.hpp"

namespace neuroserve::test_support {

// Holds the engine thread inside its first prefill until open(), so that every request submitted
// meanwhile is queued when the engine goes on. Tests in which requests must compete for pages (or
// run at the same time) then do not depend on how fast the engine thread happens to run: under a
// loaded CI machine it could otherwise finish one request before the next is even submitted.
// Opens itself when destroyed, so a failing test cannot leave the engine stuck.
class FirstPrefillGate {
public:
    explicit FirstPrefillGate(FakeBackend& backend) {
        backend.on_prefill = [this] {
            if (first_.exchange(false)) {
                opened_.wait();
            }
        };
    }
    ~FirstPrefillGate() { open(); }
    FirstPrefillGate(const FirstPrefillGate&) = delete;
    FirstPrefillGate& operator=(const FirstPrefillGate&) = delete;

    void open() {
        if (!open_called_.exchange(true)) {
            open_.set_value();
        }
    }

private:
    std::atomic<bool> first_{true};
    std::atomic<bool> open_called_{false};
    std::promise<void> open_;
    std::shared_future<void> opened_ = open_.get_future().share();
};

}  // namespace neuroserve::test_support
