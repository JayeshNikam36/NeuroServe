#pragma once

#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "neuroserve/core/scheduler.hpp"

// Golden scheduler traces: a trace file describes a workload (limits and request arrivals) and the
// step-by-step log a policy must produce for it. The simulator below replays the workload with the
// same rules as EngineLoop, so policies can be tested deterministically without threads or a GPU:
//   * KV pages are allocated on demand: a request holds ceil(cached tokens / page_size) pages;
//   * each step: preempted requests free their pages and go back to the front of the queue (in
//     arrival order) to restart from their prompt; admitted requests are prefilled (first token)
//     and join the decode batch from the next step; one batched decode adds a token to every
//     request that was already running, which finishes at max_new tokens and frees its pages;
//   * a step's tokens (admitted prompt tokens + one per decoded request) must fit max_batch_tokens;
//   * the simulator fails if a step would need more pages than exist, like the real backend.
//
// File format (lines starting with '#' are comments):
//   policy fcfs
//   limits max_batch=4 total_pages=64 page_size=16 max_batch_tokens=512   (the last one is optional)
//   arrive step=0 id=a prompt=12 max_new=3
//   --- expected
//   step 0 queued=[a] running=[] preempt=[] admit=[a] finished=[] tokens=12+0 free_pages=63
namespace neuroserve::test_support {

struct TraceSpec {
    std::string policy = "fcfs";
    int max_batch = 0;
    int total_pages = 0;
    int page_size = 16;
    int max_batch_tokens = 1 << 30;  // effectively unlimited unless the trace sets it
    struct Arrival {
        int step = 0;
        std::string id;
        int prompt = 0;
        int max_new = 0;
    };
    std::vector<Arrival> arrivals;
};

struct TraceFile {
    std::string header;    // everything up to and including the "--- expected" line
    TraceSpec spec;
    std::string expected;  // the expected log
};

inline std::string read_file(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("cannot open " + path);
    }
    std::stringstream buffer;
    buffer << in.rdbuf();
    return buffer.str();
}

// Reads "key=value" from a token, checking the key.
inline int int_field(const std::string& token, const std::string& key) {
    if (token.rfind(key + "=", 0) != 0) {
        throw std::runtime_error("expected " + key + "=..., got '" + token + "'");
    }
    return std::stoi(token.substr(key.size() + 1));
}

inline std::string text_field(const std::string& token, const std::string& key) {
    if (token.rfind(key + "=", 0) != 0) {
        throw std::runtime_error("expected " + key + "=..., got '" + token + "'");
    }
    return token.substr(key.size() + 1);
}

inline TraceFile parse_trace(const std::string& text) {
    const std::string marker = "--- expected\n";
    const auto split = text.find(marker);
    if (split == std::string::npos) {
        throw std::runtime_error("trace has no '--- expected' line");
    }
    TraceFile trace;
    trace.header = text.substr(0, split + marker.size());
    trace.expected = text.substr(split + marker.size());

    std::istringstream lines(text.substr(0, split));
    std::string line;
    while (std::getline(lines, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        std::istringstream words(line);
        std::string kind;
        words >> kind;
        std::vector<std::string> tokens;
        for (std::string token; words >> token;) {
            tokens.push_back(token);
        }
        if (kind == "policy" && tokens.size() == 1) {
            trace.spec.policy = tokens[0];
        } else if (kind == "limits" && (tokens.size() == 3 || tokens.size() == 4)) {
            trace.spec.max_batch = int_field(tokens[0], "max_batch");
            trace.spec.total_pages = int_field(tokens[1], "total_pages");
            trace.spec.page_size = int_field(tokens[2], "page_size");
            if (tokens.size() == 4) {
                trace.spec.max_batch_tokens = int_field(tokens[3], "max_batch_tokens");
            }
        } else if (kind == "arrive" && tokens.size() == 4) {
            trace.spec.arrivals.push_back({int_field(tokens[0], "step"), text_field(tokens[1], "id"),
                                           int_field(tokens[2], "prompt"), int_field(tokens[3], "max_new")});
        } else {
            throw std::runtime_error("cannot parse trace line: " + line);
        }
    }
    if (trace.spec.max_batch < 1 || trace.spec.total_pages < 1 || trace.spec.page_size < 1) {
        throw std::runtime_error("trace needs a 'limits' line with positive values");
    }
    return trace;
}

inline std::string id_list(const std::vector<std::string>& ids) {
    std::string out = "[";
    for (std::size_t i = 0; i < ids.size(); ++i) {
        out += (i ? " " : "") + ids[i];
    }
    return out + "]";
}

// Replays the workload with `policy` and returns the step log (one line per non-idle step).
inline std::string simulate(const TraceSpec& spec, neuroserve::core::SchedulingPolicy& policy) {
    struct Request {
        std::string id;
        int order = 0;
        int prompt = 0;
        int max_new = 0;
        int cached = 0;     // tokens in the KV cache
        int generated = 0;
    };
    const auto pages_for = [&](int tokens) { return (tokens + spec.page_size - 1) / spec.page_size; };
    std::vector<Request> queued;
    std::vector<Request> running;  // in admission order
    int used_pages = 0;
    int next_order = 0;
    std::size_t arrived = 0;
    std::string log;

    for (int step = 0; step < 100000; ++step) {
        for (const auto& a : spec.arrivals) {
            if (a.step == step) {
                queued.push_back({a.id, next_order++, a.prompt, a.max_new, 0, 0});
                ++arrived;
            }
        }
        if (queued.empty() && running.empty()) {
            if (arrived == spec.arrivals.size()) {
                return log;
            }
            continue;  // idle step: nothing to log
        }

        neuroserve::core::SchedulerInput input;
        input.max_batch = spec.max_batch;
        input.free_pages = spec.total_pages - used_pages;
        input.max_step_tokens = spec.max_batch_tokens;
        std::vector<std::string> queued_ids;
        std::vector<std::string> running_ids;
        for (const auto& r : queued) {
            queued_ids.push_back(r.id);
            input.queued.push_back({r.id, r.prompt, r.max_new, pages_for(r.prompt)});
        }
        for (const auto& r : running) {
            running_ids.push_back(r.id);
            const int held = pages_for(r.cached);
            input.running.push_back({r.id, held, pages_for(r.cached + 1) - held, r.generated, r.max_new});
        }
        const auto decision = policy.schedule(input);
        neuroserve::core::check_decision(input, decision);

        // Preempt: free the pages, restart later from the prompt, back to the front of the queue.
        std::vector<std::string> preempted_ids;
        std::vector<Request> victims;
        std::vector<bool> preempted(running.size(), false);
        for (const std::size_t index : decision.preempt) {
            preempted[index] = true;
            preempted_ids.push_back(running[index].id);
        }
        std::vector<Request> still_running;
        for (std::size_t i = 0; i < running.size(); ++i) {
            if (preempted[i]) {
                used_pages -= pages_for(running[i].cached);
                victims.push_back({running[i].id, running[i].order, running[i].prompt, running[i].max_new, 0, 0});
            } else {
                still_running.push_back(running[i]);
            }
        }
        running = std::move(still_running);

        // Admit: prefill produces the first token; the request decodes from the next step on.
        std::vector<std::string> admitted_ids;
        std::vector<std::string> finished_ids;
        std::vector<Request> joining;
        int prefill_tokens = 0;
        std::vector<bool> taken(queued.size(), false);
        for (const std::size_t index : decision.admit) {
            Request r = queued[index];
            taken[index] = true;
            admitted_ids.push_back(r.id);
            prefill_tokens += r.prompt;
            r.cached = r.prompt;
            r.generated = 1;
            used_pages += pages_for(r.cached);
            if (r.generated >= r.max_new) {
                finished_ids.push_back(r.id);
                used_pages -= pages_for(r.cached);
            } else {
                joining.push_back(r);
            }
        }
        std::vector<Request> still_queued;
        for (std::size_t i = 0; i < queued.size(); ++i) {
            if (!taken[i]) {
                still_queued.push_back(queued[i]);
            }
        }
        std::sort(victims.begin(), victims.end(), [](const Request& a, const Request& b) { return a.order < b.order; });
        queued = victims;
        queued.insert(queued.end(), still_queued.begin(), still_queued.end());

        // One batched decode step for the requests that were already running.
        const int decode_tokens = static_cast<int>(running.size());
        if (prefill_tokens + decode_tokens > spec.max_batch_tokens) {
            throw std::runtime_error("step " + std::to_string(step) + " exceeded max_batch_tokens");
        }
        still_running.clear();
        for (auto& r : running) {
            used_pages += pages_for(r.cached + 1) - pages_for(r.cached);
            ++r.cached;
            if (++r.generated >= r.max_new) {
                finished_ids.push_back(r.id);
                used_pages -= pages_for(r.cached);
            } else {
                still_running.push_back(r);
            }
        }
        running = std::move(still_running);
        running.insert(running.end(), joining.begin(), joining.end());
        if (used_pages > spec.total_pages) {
            throw std::runtime_error("step " + std::to_string(step) + " needed more KV pages than exist");
        }

        log += "step " + std::to_string(step) + " queued=" + id_list(queued_ids) + " running=" +
               id_list(running_ids) + " preempt=" + id_list(preempted_ids) + " admit=" + id_list(admitted_ids) +
               " finished=" + id_list(finished_ids) + " tokens=" + std::to_string(prefill_tokens) + "+" +
               std::to_string(decode_tokens) + " free_pages=" + std::to_string(spec.total_pages - used_pages) + "\n";
    }
    throw std::runtime_error("trace did not finish within 100000 steps (a policy that never admits?)");
}

}  // namespace neuroserve::test_support
