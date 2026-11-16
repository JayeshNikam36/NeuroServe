#pragma once

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "neuroserve/core/scheduler.hpp"

// Golden scheduler traces: a trace file describes a workload (limits and request arrivals) and the
// step-by-step log a policy must produce for it. The simulator below replays the workload with the
// same rules as EngineLoop, so policies can be tested deterministically without threads or a GPU:
//   * a request reserves ceil((prompt + max_new) / page_size) KV pages while it runs;
//   * the step it is admitted, prefill produces its first token and the batched decode its second;
//   * every later step's decode adds one token; it finishes at max_new tokens and frees its pages.
//
// File format (lines starting with '#' are comments):
//   policy fcfs
//   limits max_batch=4 total_pages=64 page_size=16
//   arrive step=0 id=a prompt=12 max_new=3
//   --- expected
//   step 0 queued=[a] running=[] admit=[a] finished=[] free_pages=63
namespace neuroserve::test_support {

struct TraceSpec {
    std::string policy = "fcfs";
    int max_batch = 0;
    int total_pages = 0;
    int page_size = 16;
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
        } else if (kind == "limits" && tokens.size() == 3) {
            trace.spec.max_batch = int_field(tokens[0], "max_batch");
            trace.spec.total_pages = int_field(tokens[1], "total_pages");
            trace.spec.page_size = int_field(tokens[2], "page_size");
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
        int prompt = 0;
        int max_new = 0;
        int pages = 0;
        int generated = 0;
    };
    std::vector<Request> queued;
    std::vector<Request> running;
    int free_pages = spec.total_pages;
    std::size_t next_arrival = 0;
    std::string log;

    for (int step = 0; step < 100000; ++step) {
        for (const auto& a : spec.arrivals) {
            if (a.step == step) {
                const int pages = (a.prompt + a.max_new + spec.page_size - 1) / spec.page_size;
                queued.push_back({a.id, a.prompt, a.max_new, pages, 0});
                ++next_arrival;
            }
        }
        if (queued.empty() && running.empty()) {
            if (next_arrival == spec.arrivals.size()) {
                return log;
            }
            continue;  // idle step: nothing to log
        }

        std::vector<std::string> queued_ids;
        std::vector<std::string> running_ids;
        neuroserve::core::SchedulerInput input;
        input.max_batch = spec.max_batch;
        input.free_pages = free_pages;
        for (const auto& r : queued) {
            queued_ids.push_back(r.id);
            input.queued.push_back({r.id, r.prompt, r.max_new, r.pages});
        }
        for (const auto& r : running) {
            running_ids.push_back(r.id);
            input.running.push_back({r.id, r.pages, r.generated, r.max_new});
        }

        std::vector<std::string> admitted_ids;
        std::vector<std::string> finished_ids;
        if (!queued.empty()) {
            const auto decision = policy.schedule(input);
            neuroserve::core::check_decision(input, decision);
            std::vector<bool> taken(queued.size(), false);
            for (const std::size_t index : decision.admit) {
                Request r = queued[index];
                taken[index] = true;
                admitted_ids.push_back(r.id);
                free_pages -= r.pages;
                r.generated = 1;  // prefill produces the first token
                if (r.generated >= r.max_new) {
                    finished_ids.push_back(r.id);
                    free_pages += r.pages;
                } else {
                    running.push_back(r);
                }
            }
            std::vector<Request> still_queued;
            for (std::size_t i = 0; i < queued.size(); ++i) {
                if (!taken[i]) {
                    still_queued.push_back(queued[i]);
                }
            }
            queued = std::move(still_queued);
        }

        // One batched decode step for everything running, including requests admitted this step.
        std::vector<Request> still_running;
        for (auto& r : running) {
            if (++r.generated >= r.max_new) {
                finished_ids.push_back(r.id);
                free_pages += r.pages;
            } else {
                still_running.push_back(r);
            }
        }
        running = std::move(still_running);

        log += "step " + std::to_string(step) + " queued=" + id_list(queued_ids) + " running=" +
               id_list(running_ids) + " admit=" + id_list(admitted_ids) + " finished=" + id_list(finished_ids) +
               " free_pages=" + std::to_string(free_pages) + "\n";
    }
    throw std::runtime_error("trace did not finish within 100000 steps (a policy that never admits?)");
}

}  // namespace neuroserve::test_support
