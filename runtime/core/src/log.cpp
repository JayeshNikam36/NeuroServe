#include "neuroserve/core/log.hpp"

#include <atomic>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <mutex>
#include <stdexcept>
#include <utility>

namespace neuroserve::core::log {

namespace {

std::atomic<int> g_level{static_cast<int>(Level::kWarning)};
std::mutex g_sink_mutex;  // serialises whole lines, so lines from different threads never interleave
Sink g_sink;               // empty: stderr

void append_escaped(std::string& out, std::string_view text) {
    out += '"';
    for (const char c : text) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned>(static_cast<unsigned char>(c)));
                    out += buf;
                } else {
                    out += c;  // UTF-8 bytes pass through unchanged
                }
        }
    }
    out += '"';
}

void append_value(std::string& out, const Field::Value& value) {
    if (const auto* i = std::get_if<std::int64_t>(&value)) {
        out += std::to_string(*i);
    } else if (const auto* d = std::get_if<double>(&value)) {
        if (!std::isfinite(*d)) {
            out += "null";  // JSON has no NaN or infinity
        } else {
            char buf[32];
            std::snprintf(buf, sizeof(buf), "%.10g", *d);
            out += buf;
        }
    } else if (const auto* b = std::get_if<bool>(&value)) {
        out += *b ? "true" : "false";
    } else {
        append_escaped(out, std::get<std::string>(value));
    }
}

void append_timestamp(std::string& out, std::chrono::system_clock::time_point time) {
    const auto since_epoch = time.time_since_epoch();
    const auto seconds = std::chrono::duration_cast<std::chrono::seconds>(since_epoch);
    const auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(since_epoch - seconds).count();
    const std::time_t t = static_cast<std::time_t>(seconds.count());
    std::tm utc{};
    gmtime_r(&t, &utc);
    char buf[40];
    const auto n = std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S", &utc);
    std::snprintf(buf + n, sizeof(buf) - n, ".%03d+00:00", static_cast<int>(millis));
    out += '"';
    out += buf;
    out += '"';
}

}  // namespace

Level parse_level(std::string_view name) {
    if (name == "DEBUG") return Level::kDebug;
    if (name == "INFO") return Level::kInfo;
    if (name == "WARNING") return Level::kWarning;
    if (name == "ERROR") return Level::kError;
    throw std::invalid_argument("unknown log level '" + std::string(name) +
                                "' (expected DEBUG, INFO, WARNING or ERROR)");
}

std::string_view level_name(Level level) noexcept {
    switch (level) {
        case Level::kDebug:   return "DEBUG";
        case Level::kInfo:    return "INFO";
        case Level::kWarning: return "WARNING";
        case Level::kError:   return "ERROR";
    }
    return "UNKNOWN";
}

void set_level(Level level) noexcept { g_level.store(static_cast<int>(level), std::memory_order_relaxed); }

Level level() noexcept { return static_cast<Level>(g_level.load(std::memory_order_relaxed)); }

bool enabled(Level level) noexcept {
    return static_cast<int>(level) >= g_level.load(std::memory_order_relaxed);
}

void set_sink(Sink sink) {
    std::lock_guard lock(g_sink_mutex);
    g_sink = std::move(sink);
}

std::string format_line(Level level, std::string_view logger, std::string_view message,
                        std::initializer_list<Field> fields, std::chrono::system_clock::time_point time) {
    std::string out;
    out.reserve(160);
    out += "{\"ts\": ";
    append_timestamp(out, time);
    out += ", \"level\": ";
    append_escaped(out, level_name(level));
    out += ", \"logger\": ";
    append_escaped(out, logger);
    out += ", \"msg\": ";
    append_escaped(out, message);
    for (const Field& field : fields) {
        out += ", ";
        append_escaped(out, field.key);
        out += ": ";
        append_value(out, field.value);
    }
    out += '}';
    return out;
}

void write(Level level, std::string_view logger, std::string_view message, std::initializer_list<Field> fields) {
    if (!enabled(level)) {
        return;
    }
    const std::string line = format_line(level, logger, message, fields, std::chrono::system_clock::now());
    std::lock_guard lock(g_sink_mutex);
    if (g_sink) {
        g_sink(line);
    } else {
        std::fwrite(line.data(), 1, line.size(), stderr);
        std::fputc('\n', stderr);
        std::fflush(stderr);
    }
}

}  // namespace neuroserve::core::log
