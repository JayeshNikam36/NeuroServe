#include "neuroserve/core/kv_memory.hpp"

#include <cctype>
#include <map>
#include <stdexcept>
#include <string>

namespace neuroserve::core {
namespace {

// A minimal reader for config.json: the top-level object's scalar values, as text. Nested objects
// and arrays are skipped whole; strings are unescaped only as far as config.json needs (\" and \\).
class TopLevel {
public:
    explicit TopLevel(std::string_view json) : s_(json) {
        skip_ws();
        expect('{');
        skip_ws();
        if (peek() == '}') {
            return;
        }
        while (true) {
            skip_ws();
            const std::string key = string();
            skip_ws();
            expect(':');
            skip_ws();
            if (peek() == '{' || peek() == '[') {
                skip_nested();
            } else if (peek() == '"') {
                values_[key] = {string(), true};
            } else {
                values_[key] = {scalar(), false};
            }
            skip_ws();
            if (peek() == ',') {
                ++i_;
                continue;
            }
            expect('}');
            return;
        }
    }

    struct Value {
        std::string text;
        bool is_string = false;
    };
    const Value* find(const std::string& key) const {
        const auto it = values_.find(key);
        return it == values_.end() ? nullptr : &it->second;
    }

private:
    char peek() const { return i_ < s_.size() ? s_[i_] : '\0'; }
    void skip_ws() {
        while (i_ < s_.size() && std::isspace(static_cast<unsigned char>(s_[i_]))) {
            ++i_;
        }
    }
    void expect(char c) {
        if (peek() != c) {
            throw std::invalid_argument(std::string("config.json: expected '") + c + "' at offset " + std::to_string(i_));
        }
        ++i_;
    }
    std::string string() {
        expect('"');
        std::string out;
        while (i_ < s_.size() && s_[i_] != '"') {
            if (s_[i_] == '\\' && i_ + 1 < s_.size()) {
                ++i_;
            }
            out += s_[i_++];
        }
        expect('"');
        return out;
    }
    std::string scalar() {  // a number, true, false or null
        const std::size_t start = i_;
        while (i_ < s_.size() && s_[i_] != ',' && s_[i_] != '}' && !std::isspace(static_cast<unsigned char>(s_[i_]))) {
            ++i_;
        }
        if (i_ == start) {
            throw std::invalid_argument("config.json: expected a value at offset " + std::to_string(start));
        }
        return std::string(s_.substr(start, i_ - start));
    }
    void skip_nested() {  // from '{' or '[' to its match, strings included
        int depth = 0;
        do {
            const char c = peek();
            if (c == '\0') {
                throw std::invalid_argument("config.json: unterminated object or array");
            }
            if (c == '"') {
                string();
                continue;
            }
            depth += (c == '{' || c == '[') ? 1 : (c == '}' || c == ']') ? -1 : 0;
            ++i_;
        } while (depth > 0);
    }

    std::string_view s_;
    std::size_t i_ = 0;
    std::map<std::string, Value> values_;
};

int positive_int(const TopLevel& c, const std::string& key) {
    const auto* v = c.find(key);
    if (v == nullptr) {
        throw std::invalid_argument("config.json: no \"" + key + "\"");
    }
    std::size_t used = 0;
    long long n = 0;
    try {
        n = v->is_string ? 0 : std::stoll(v->text, &used);
    } catch (const std::exception&) {
        used = 0;
    }
    if (v->is_string || used != v->text.size() || n <= 0 || n > 1'000'000) {
        throw std::invalid_argument("config.json: \"" + key + "\" must be a positive integer, got " + v->text);
    }
    return static_cast<int>(n);
}

}  // namespace

ModelShape parse_model_config(std::string_view json) {
    const TopLevel c(json);
    ModelShape shape;
    shape.num_layers = positive_int(c, "num_hidden_layers");
    const int heads = positive_int(c, "num_attention_heads");
    shape.num_kv_heads = c.find("num_key_value_heads") ? positive_int(c, "num_key_value_heads") : heads;
    if (c.find("head_dim")) {
        shape.head_dim = positive_int(c, "head_dim");
    } else {
        const int hidden = positive_int(c, "hidden_size");
        if (hidden % heads != 0) {
            throw std::invalid_argument("config.json: hidden_size is not a multiple of num_attention_heads");
        }
        shape.head_dim = hidden / heads;
    }
    if (heads % shape.num_kv_heads != 0) {
        throw std::invalid_argument("config.json: num_attention_heads is not a multiple of num_key_value_heads");
    }
    if (const auto* dtype = c.find("torch_dtype")) {
        shape.dtype = dtype->text;
    }
    return shape;
}

int dtype_bytes(std::string_view dtype) {
    if (dtype == "float16" || dtype == "fp16" || dtype == "bfloat16" || dtype == "bf16") {
        return 2;
    }
    if (dtype == "float32" || dtype == "fp32") {
        return 4;
    }
    throw std::invalid_argument("unknown dtype '" + std::string(dtype) + "'");
}

std::int64_t kv_bytes_per_token(const ModelShape& shape, std::string_view kv_dtype) {
    return std::int64_t{2} * shape.num_layers * shape.num_kv_heads * shape.head_dim * dtype_bytes(kv_dtype);
}

std::int64_t kv_bytes_per_block(const ModelShape& shape, std::string_view kv_dtype, int block_size) {
    return kv_bytes_per_token(shape, kv_dtype) * block_size;
}

std::string_view admission_name(KvAdmission mode) noexcept {
    return mode == KvAdmission::kWorstCase ? "worst-case" : "on-demand";
}

KvAdmission parse_admission(std::string_view name) {
    if (name == "on-demand") {
        return KvAdmission::kOnDemand;
    }
    if (name == "worst-case") {
        return KvAdmission::kWorstCase;
    }
    throw std::invalid_argument("unknown KV admission mode '" + std::string(name) + "' (on-demand, worst-case)");
}

}  // namespace neuroserve::core
