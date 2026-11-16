#pragma once

#include <chrono>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <string>
#include <string_view>
#include <type_traits>
#include <variant>

// Structured logging for the C++ runtime: one JSON object per line, the same shape as the Python
// server's log lines ({"ts", "level", "logger", "msg", ...fields}), so both can be searched
// together. Thread-safe. Lines go to stderr unless a sink is installed.
namespace neuroserve::core::log {

enum class Level : int { kDebug = 10, kInfo = 20, kWarning = 30, kError = 40 };  // Python's numbers

// "DEBUG", "INFO", "WARNING", "ERROR" (Python logging names). Throws std::invalid_argument otherwise.
Level parse_level(std::string_view name);
std::string_view level_name(Level level) noexcept;

void set_level(Level level) noexcept;  // default: kWarning
Level level() noexcept;
bool enabled(Level level) noexcept;

// One key/value pair of a log line. Values: integers, floating point, bool, or text.
struct Field {
    using Value = std::variant<std::int64_t, double, bool, std::string>;

    template <typename T>
    Field(std::string_view field_key, const T& field_value) : key(field_key) {
        if constexpr (std::is_same_v<T, bool>) {
            value = field_value;
        } else if constexpr (std::is_integral_v<T>) {
            value = static_cast<std::int64_t>(field_value);
        } else if constexpr (std::is_floating_point_v<T>) {
            value = static_cast<double>(field_value);
        } else {
            value = std::string(field_value);
        }
    }

    std::string key;
    Value value;
};

// Receives each finished line, without the trailing newline. nullptr restores the stderr sink.
using Sink = std::function<void(std::string_view line)>;
void set_sink(Sink sink);

// Formats one line (exposed for tests).
std::string format_line(Level level, std::string_view logger, std::string_view message,
                        std::initializer_list<Field> fields, std::chrono::system_clock::time_point time);

// Writes one line if `level` is enabled.
void write(Level level, std::string_view logger, std::string_view message,
           std::initializer_list<Field> fields = {});

}  // namespace neuroserve::core::log
