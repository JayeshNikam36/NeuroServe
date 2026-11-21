#include <gtest/gtest.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "neuroserve/backend/cuda_llm_engine_backend.hpp"

namespace fs = std::filesystem;
namespace core = neuroserve::core;
using neuroserve::CudaLlmEngineBackend;
using neuroserve::CudaLlmEngineConfig;
using neuroserve::KvCacheExhausted;
using neuroserve::SeqId;
using neuroserve::TokenId;

namespace {

constexpr int kNumPrompts = 8;

std::string env_or(const char* name, const char* fallback) {
    const char* value = std::getenv(name);
    return value != nullptr ? value : fallback;
}

std::vector<TokenId> read_i32(const fs::path& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("cannot open " + path.string());
    }
    f.seekg(0, std::ios::end);
    const auto bytes = static_cast<std::size_t>(f.tellg());
    f.seekg(0);
    std::vector<TokenId> values(bytes / sizeof(TokenId));
    f.read(reinterpret_cast<char*>(values.data()),
           static_cast<std::streamsize>(values.size() * sizeof(TokenId)));
    return values;
}

struct Reference {
    std::vector<TokenId> prompt;
    std::vector<TokenId> greedy;
};

}  // namespace

class CudaLlmEngineBackendTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        const std::string model_dir = env_or("NS_MODEL_DIR", "/models/llama-3.2-1b-fp16");
        const std::string ref_dir = env_or("NS_REF_DIR", "/models/llama-3.2-1b-ref");
        if (!fs::exists(fs::path(model_dir) / "weights.bin") ||
            !fs::exists(fs::path(ref_dir) / "prompt_00_ids.i32")) {
            skip_reason_ = "model or reference files not found (NS_MODEL_DIR=" + model_dir +
                           ", NS_REF_DIR=" + ref_dir + ")";
            return;
        }
        CudaLlmEngineConfig config;
        config.model_dir = model_dir;
        config.max_prompt_tokens = 512;
        config.num_pages = 512;  // 8192 tokens, 256 MiB of KV
        config.max_batch = 8;
        config.max_pages_per_sequence = 64;
        backend_ = std::make_unique<CudaLlmEngineBackend>(config);
        model_dir_ = model_dir;

        for (int i = 0; i < kNumPrompts; ++i) {
            char stem[32];
            std::snprintf(stem, sizeof(stem), "prompt_%02d", i);
            refs_.push_back({read_i32(fs::path(ref_dir) / (std::string(stem) + "_ids.i32")),
                             read_i32(fs::path(ref_dir) / (std::string(stem) + "_greedy.i32"))});
        }
    }

    static void TearDownTestSuite() {
        backend_.reset();
        refs_.clear();
    }

    void SetUp() override {
        if (!backend_) {
            GTEST_SKIP() << skip_reason_;
        }
    }

    void expect_no_leaked_pages() const {
        const auto kv = backend_->kv_stats();
        EXPECT_EQ(kv.free_pages, kv.total_pages) << "KV pages leaked";
    }

    static inline std::unique_ptr<CudaLlmEngineBackend> backend_;
    static inline std::vector<Reference> refs_;
    static inline std::string skip_reason_;
    static inline std::string model_dir_;
};

TEST_F(CudaLlmEngineBackendTest, ModelInfoMatchesLlama32_1B) {
    const auto& info = backend_->model_info();
    EXPECT_EQ(info.n_layers, 16);
    EXPECT_EQ(info.n_kv_heads, 8);
    EXPECT_EQ(info.head_dim, 64);
    EXPECT_EQ(info.vocab_size, 128256);
    EXPECT_EQ(info.kv_bytes_per_token, 32768u);  // 2 x 16 x 8 x 64 x 2 bytes
    EXPECT_EQ(backend_->kv_stats().page_size_tokens, 16);
    expect_no_leaked_pages();
}

// The root of every prefix hash chain names this model, these weights, FP16 K/V, 16-token pages and
// Llama 3's RoPE (docs/kv-cache.md, "Hashing full blocks").
TEST_F(CudaLlmEngineBackendTest, KvCacheKeyIsTheRootOfTheModelsIdentity) {
    const core::KvCacheIdentity& id = backend_->kv_identity();
    EXPECT_EQ(id.model_id, "llama-3.2-1b");
    EXPECT_EQ(id.dtype, "fp16");
    EXPECT_EQ(id.block_size, 16);
    EXPECT_EQ(id.rope.theta, 500000.0);
    EXPECT_EQ(id.rope.scaling_type, "llama3");
    EXPECT_EQ(id.weights_digest, core::file_fingerprint((fs::path(model_dir_) / "weights.bin").string()) + ":" +
                                     core::file_fingerprint((fs::path(model_dir_) / "index.tsv").string()));

    const std::string& key = backend_->model_info().kv_cache_key;
    EXPECT_EQ(key, core::hash_hex(core::PrefixHasher(id).root()));
    ASSERT_EQ(key.size(), 16u);
    const auto is_hex = [](char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'); };
    EXPECT_TRUE(std::all_of(key.begin(), key.end(), is_hex)) << key;
}

TEST_F(CudaLlmEngineBackendTest, SinglePromptMatchesHuggingFace) {
    const Reference& ref = refs_.at(0);
    const SeqId id = backend_->create_sequence();

    std::vector<TokenId> out{backend_->prefill(id, ref.prompt)};
    while (out.size() < ref.greedy.size()) {
        const SeqId ids[] = {id};
        const TokenId last[] = {out.back()};
        out.push_back(backend_->decode_step(ids, last).at(0));
    }

    EXPECT_EQ(out, ref.greedy);
    EXPECT_EQ(backend_->sequence_length(id),
              static_cast<int>(ref.prompt.size() + ref.greedy.size() - 1));
    backend_->free_sequence(id);
    expect_no_leaked_pages();
}

TEST_F(CudaLlmEngineBackendTest, AllPromptsDecodedTogetherMatchHuggingFace) {
    std::vector<SeqId> ids;
    std::vector<std::vector<TokenId>> outs(kNumPrompts);
    for (int i = 0; i < kNumPrompts; ++i) {
        ids.push_back(backend_->create_sequence());
        outs[i].push_back(backend_->prefill(ids[i], refs_[i].prompt));
    }

    const std::size_t n_new = refs_[0].greedy.size();
    for (std::size_t step = 1; step < n_new; ++step) {
        std::vector<TokenId> last;
        for (const auto& o : outs) {
            last.push_back(o.back());
        }
        const std::vector<TokenId> next = backend_->decode_step(ids, last);
        ASSERT_EQ(next.size(), ids.size());
        for (int i = 0; i < kNumPrompts; ++i) {
            outs[i].push_back(next[i]);
        }
    }

    for (int i = 0; i < kNumPrompts; ++i) {
        EXPECT_EQ(outs[i], refs_[i].greedy) << "prompt " << i << " (" << refs_[i].prompt.size() << " tokens)";
    }
    for (const SeqId id : ids) {
        backend_->free_sequence(id);
    }
    expect_no_leaked_pages();
}

TEST_F(CudaLlmEngineBackendTest, RejectsBadRequestsWithoutSideEffects) {
    const int free_before = backend_->kv_stats().free_pages;
    const std::vector<TokenId>& good = refs_[0].prompt;

    EXPECT_THROW(backend_->prefill(9999, good), std::invalid_argument);  // unknown sequence

    const SeqId id = backend_->create_sequence();
    EXPECT_THROW(backend_->prefill(id, std::vector<TokenId>{}), std::invalid_argument);         // empty
    EXPECT_THROW(backend_->prefill(id, std::vector<TokenId>(600, 1000)), std::length_error);    // too long
    EXPECT_THROW(backend_->prefill(id, std::vector<TokenId>{1, -5}), std::invalid_argument);    // bad token
    {
        const SeqId ids[] = {id};
        const TokenId toks[] = {1};
        EXPECT_THROW(backend_->decode_step(ids, toks), std::logic_error);  // not prefilled yet
    }
    EXPECT_EQ(backend_->sequence_length(id), 0);

    ASSERT_NO_THROW(backend_->prefill(id, good));
    EXPECT_THROW(backend_->prefill(id, good), std::logic_error);  // already prefilled
    {
        const SeqId ids[] = {id, id};
        const TokenId toks[] = {1, 1};
        EXPECT_THROW(backend_->decode_step(ids, toks), std::invalid_argument);  // duplicate
    }
    {
        const SeqId ids[] = {id};
        const TokenId toks[] = {1, 2};
        EXPECT_THROW(backend_->decode_step(ids, toks), std::invalid_argument);  // size mismatch
    }

    backend_->free_sequence(id);
    EXPECT_THROW(backend_->free_sequence(id), std::invalid_argument);  // double free
    EXPECT_EQ(backend_->kv_stats().free_pages, free_before);
}

TEST_F(CudaLlmEngineBackendTest, ReportsKvExhaustionInsteadOfCrashing) {
    const std::vector<TokenId> prompt(400, 1000);  // exactly 25 pages
    const int pages_per_prompt = 25;

    std::vector<SeqId> live;
    while (backend_->kv_stats().free_pages >= pages_per_prompt) {
        const SeqId id = backend_->create_sequence();
        backend_->prefill(id, prompt);
        live.push_back(id);
    }

    const int free_before = backend_->kv_stats().free_pages;
    const SeqId extra = backend_->create_sequence();
    EXPECT_THROW(backend_->prefill(extra, prompt), KvCacheExhausted);
    EXPECT_EQ(backend_->kv_stats().free_pages, free_before);  // nothing allocated
    EXPECT_EQ(backend_->sequence_length(extra), 0);

    backend_->free_sequence(extra);
    for (const SeqId id : live) {
        backend_->free_sequence(id);
    }
    expect_no_leaked_pages();
}
TEST_F(CudaLlmEngineBackendTest, SharedPrefixPagesAreReusedAndFreedByTheirLastHolder) {
    // A 40-token prompt (the reference prompts, concatenated): pages 0-1 full, page 2 partial.
    std::vector<TokenId> prompt;
    for (const Reference& r : refs_) {
        prompt.insert(prompt.end(), r.prompt.begin(), r.prompt.end());
    }
    ASSERT_GE(prompt.size(), 40u);
    prompt.resize(40);
    const auto decode_alone = [&](SeqId id, TokenId token) {
        const SeqId ids[] = {id};
        const TokenId tokens[] = {token};
        return backend_->decode_step(ids, tokens).at(0);
    };

    // The truth: the prompt alone, then 8 more tokens.
    const SeqId solo = backend_->create_sequence();
    std::vector<TokenId> truth{backend_->prefill(solo, prompt)};
    while (truth.size() < 9) {
        truth.push_back(decode_alone(solo, truth.back()));
    }
    backend_->free_sequence(solo);
    const int free_before = backend_->kv_stats().free_pages;

    const SeqId a = backend_->create_sequence();
    ASSERT_EQ(backend_->prefill(a, prompt), truth[0]);
    const SeqId b = backend_->create_sequence();
    backend_->share_prefix(b, a, 2);  // b starts with a's two full pages: 32 cached positions
    EXPECT_EQ(backend_->sequence_length(b), 32);
    EXPECT_EQ(backend_->sequence_pages(a).exclusive, 1);
    EXPECT_EQ(backend_->sequence_pages(b).exclusive, 0);
    EXPECT_EQ(backend_->kv_stats().shared_pages, 2);
    EXPECT_EQ(backend_->kv_stats().free_pages, free_before - 3) << "sharing took no page";
    backend_->check_kv_invariants();

    // b feeds the 8 uncached prompt tokens itself: its K/V for positions 32-39 go to a page of its
    // own, and its attention reads the shared pages. Its next token must be the prompt's (the path
    // Day 34's "prefill from a cached offset" will take; a mismatch here would be the ADR-0006
    // near-tie risk between the prefill and decode code paths, not a counting bug).
    TokenId next = 0;
    for (std::size_t i = 32; i < prompt.size(); ++i) {
        next = decode_alone(b, prompt[i]);
    }
    EXPECT_EQ(next, truth[0]) << "reused prefix + decoded suffix vs the whole prompt prefilled";
    EXPECT_EQ(backend_->sequence_pages(b).held, 3);
    EXPECT_EQ(backend_->sequence_pages(b).exclusive, 1);

    // b's writes never touched the shared pages: a continues exactly as the prompt alone did.
    std::vector<TokenId> continued{truth[0]};
    while (continued.size() < truth.size()) {
        continued.push_back(decode_alone(a, continued.back()));
    }
    EXPECT_EQ(continued, truth);

    backend_->free_sequence(a);  // the shared pages stay: b still lists them
    EXPECT_EQ(backend_->kv_stats().shared_pages, 0);
    EXPECT_EQ(backend_->sequence_pages(b).exclusive, 3);
    EXPECT_EQ(backend_->kv_stats().free_pages, free_before - 3);
    backend_->check_kv_invariants();
    backend_->free_sequence(b);  // the last holder: everything is free
    expect_no_leaked_pages();
    backend_->check_kv_invariants();
}

// ---- Prefix caching (Day 34): prefill from a cached offset must reproduce Hugging Face exactly.
// Llama-3.2-1B-Instruct's chat prompts all start with the same system header, so one request that
// holds its pages turns every other chat prompt into a prefix hit. Each reference case runs twice at
// batch size 1: once with nothing to reuse, once with the header's pages reused, its prefill
// starting at the first uncached token (positions and RoPE from there). Both must equal the
// Hugging Face greedy tokens.
namespace {

// The integer arrays named `key` in the reference JSON, in order (the file is ours: flat cases of
// "name", "prompt_ids" and "greedy_ids"; no JSON library needed).
std::vector<std::vector<TokenId>> json_int_arrays(const std::string& text, const std::string& key) {
    std::vector<std::vector<TokenId>> out;
    const std::string needle = "\"" + key + "\"";
    for (std::size_t at = text.find(needle); at != std::string::npos; at = text.find(needle, at + 1)) {
        const std::size_t open = text.find('[', at);
        const std::size_t close = text.find(']', open);
        std::vector<TokenId> values;
        std::size_t i = open + 1;
        while (i < close) {
            while (i < close && (text[i] < '0' || text[i] > '9') && text[i] != '-') {
                ++i;
            }
            if (i >= close) {
                break;
            }
            std::size_t used = 0;
            values.push_back(static_cast<TokenId>(std::stol(text.substr(i, close - i), &used)));
            i += used;
        }
        out.push_back(std::move(values));
    }
    return out;
}

}  // namespace

class CudaPrefixCacheTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        const std::string model_dir = env_or("NS_INSTRUCT_MODEL_DIR", "/models/llama-3.2-1b-instruct-fp16");
        const std::string ref = env_or("NS_INSTRUCT_REF", "/models/llama-3.2-1b-ref/instruct_greedy.json");
        if (!fs::exists(fs::path(model_dir) / "weights.bin") || !fs::exists(ref)) {
            skip_reason_ = "instruct weights or reference not found (" + model_dir + ", " + ref + ")";
            return;
        }
        std::ifstream in(ref);
        const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        prompts_ = json_int_arrays(text, "prompt_ids");
        greedy_ = json_int_arrays(text, "greedy_ids");
        CudaLlmEngineConfig config;
        config.model_dir = model_dir;
        config.model_name = "llama-3.2-1b-instruct";
        config.num_pages = 1024;          // 16 requests of 520 + 32 tokens without the cache: 560 pages
        config.max_prompt_tokens = 640;   // a 512-token system prompt and a user turn
        config.max_batch = 16;
        config.prefix_caching = true;
        backend_ = std::make_unique<CudaLlmEngineBackend>(config);
    }

    static void TearDownTestSuite() { backend_.reset(); }

    void SetUp() override {
        if (!backend_) {
            GTEST_SKIP() << skip_reason_;
        }
        backend_->evict_cached_pages();  // each test starts from an empty cache (CACHED pages, Day 37)
    }

    // Greedy generation at batch size 1; `hit_pages` gets how many pages its prefill reused.
    static std::vector<TokenId> generate(const std::vector<TokenId>& prompt, std::size_t n, std::int64_t& hit_pages) {
        const std::int64_t hits_before = backend_->kv_stats().prefix_hit_pages;
        const SeqId id = backend_->create_sequence();
        std::vector<TokenId> out{backend_->prefill(id, prompt)};
        hit_pages = backend_->kv_stats().prefix_hit_pages - hits_before;
        while (out.size() < n) {
            const SeqId ids[] = {id};
            const TokenId last[] = {out.back()};
            out.push_back(backend_->decode_step(ids, last).at(0));
        }
        backend_->free_sequence(id);
        return out;
    }

    static inline std::unique_ptr<CudaLlmEngineBackend> backend_;
    static inline std::vector<std::vector<TokenId>> prompts_;
    static inline std::vector<std::vector<TokenId>> greedy_;
    static inline std::string skip_reason_;
};

TEST_F(CudaPrefixCacheTest, PrefixHitsReproduceHuggingFaceExactly) {
    ASSERT_EQ(prompts_.size(), 5u);
    ASSERT_EQ(greedy_.size(), 5u);
    std::int64_t total_hits = 0;
    for (std::size_t c = 0; c < prompts_.size(); ++c) {
        std::int64_t hits = 0;
        backend_->evict_cached_pages();  // nothing to reuse: the earlier cases' pages are CACHED
        EXPECT_EQ(generate(prompts_[c], greedy_[c].size(), hits), greedy_[c]) << "case " << c << " without a hit";
        EXPECT_EQ(hits, 0) << "nothing to reuse";

        // A holder keeps case 0's pages (the shared system header) live; it is never decoded.
        const SeqId holder = backend_->create_sequence();
        backend_->prefill(holder, prompts_[0]);
        const std::vector<TokenId> out = generate(prompts_[c], greedy_[c].size(), hits);
        backend_->free_sequence(holder);
        std::cout << "case " << c << ": " << prompts_[c].size() << " prompt tokens, " << hits * 16
                  << " reused, prefill from position " << hits * 16 << "\n";
        EXPECT_EQ(out, greedy_[c]) << "case " << c << " with " << hits << " pages reused";
        total_hits += hits;
        backend_->check_kv_invariants();
    }
    EXPECT_GT(total_hits, 0) << "no case shared a full page with the holder: nothing was tested";
    const auto kv = backend_->kv_stats();
    EXPECT_EQ(kv.free_pages, kv.total_pages) << "KV pages leaked";
    EXPECT_EQ(kv.indexed_pages, kv.cached_pages) << "released pages stay indexed as CACHED, nothing else";
}

TEST_F(CudaPrefixCacheTest, AnIdenticalPromptReusesEveryPageButTheLastAndMatches) {
    // The longest reuse: the same prompt as the holder, all its full pages but the one holding the
    // last token (whose logits must be computed). The prefill then computes only the last 1..16 tokens.
    const std::vector<TokenId>& prompt = prompts_.at(0);
    const SeqId holder = backend_->create_sequence();
    backend_->prefill(holder, prompt);
    std::int64_t hits = 0;
    const std::vector<TokenId> out = generate(prompt, greedy_.at(0).size(), hits);
    backend_->free_sequence(holder);
    EXPECT_EQ(hits, static_cast<std::int64_t>((prompt.size() - 1) / 16));
    EXPECT_EQ(out, greedy_.at(0));
}

TEST_F(CudaPrefixCacheTest, CopyOnWriteKeepsBothRequestsCorrect) {
    // Copy-on-write (Day 35): a prompt that fills whole pages, reused by an identical request. The
    // page holding the last prompt token is copied, not shared, and the prefill computes only that
    // token, writing into the copy. Both requests must then match Hugging Face: the new one, and
    // the holder whose page the copy came from.
    std::size_t c = 0;
    while (c < prompts_.size() && prompts_[c].size() % 16 != 0) {
        ++c;
    }
    if (c == prompts_.size()) {
        GTEST_SKIP() << "no reference prompt fills whole pages";
    }
    const std::vector<TokenId>& prompt = prompts_[c];
    const std::vector<TokenId>& expected = greedy_[c];
    const SeqId holder = backend_->create_sequence();
    std::vector<TokenId> held{backend_->prefill(holder, prompt)};

    std::int64_t hits = 0;
    const std::vector<TokenId> out = generate(prompt, expected.size(), hits);
    std::cout << "case " << c << ": " << prompt.size() << " prompt tokens, " << hits
              << " pages reused (the last one copied), prefill of 1 token\n";
    EXPECT_EQ(hits, static_cast<std::int64_t>(prompt.size() / 16));
    EXPECT_EQ(out, expected) << "the request that prefilled one token into the copy";

    while (held.size() < expected.size()) {  // the holder decodes only now: its pages must be intact
        const SeqId ids[] = {holder};
        const TokenId last[] = {held.back()};
        held.push_back(backend_->decode_step(ids, last).at(0));
    }
    EXPECT_EQ(held, expected) << "the holder whose page was copied";
    backend_->free_sequence(holder);
    backend_->check_kv_invariants();
    const auto kv = backend_->kv_stats();
    EXPECT_EQ(kv.free_pages, kv.total_pages) << "KV pages leaked";
}

// The shared-prefix end-to-end test (Day 36, KV-007): 16 requests with the same 512-token system
// prompt and their own user turns. Run once without the cache and once with it, on one backend,
// with every step shaped the same (all 16 prefilled, then decoded together in one batch of 16), so
// that the cache is the only difference (outputs are deterministic but not batch-invariant,
// ADR-0006). The outputs must be identical; the hit rate and the reused tokens are reported.
TEST_F(CudaPrefixCacheTest, SixteenRequestsWithA512TokenSystemPromptMatchTheNoCacheRun) {
    constexpr int kRequests = 16;
    constexpr std::size_t kNewTokens = 32;
    std::vector<TokenId> system = prompts_.at(0);  // the chat header, then a long instruction
    for (TokenId t = 0; system.size() < 512; ++t) {
        system.push_back(1000 + (t * 37) % 20000);
    }
    std::vector<std::vector<TokenId>> prompts;
    for (int r = 0; r < kRequests; ++r) {
        std::vector<TokenId> p = system;
        for (int i = 0; i < 8; ++i) {
            p.push_back(30000 + r * 100 + i);  // the request's own user turn
        }
        prompts.push_back(std::move(p));
    }

    // All 16 prefilled in order, then decoded together; returns each request's tokens and sets
    // `reused` to the prompt tokens taken from the cache.
    auto run = [&](bool cache, std::int64_t& reused) {
        backend_->set_prefix_caching(cache);
        const std::int64_t hits_before = backend_->kv_stats().prefix_hit_pages;
        std::vector<SeqId> ids;
        std::vector<std::vector<TokenId>> out(kRequests);
        for (int r = 0; r < kRequests; ++r) {
            ids.push_back(backend_->create_sequence());
            out[static_cast<std::size_t>(r)].push_back(backend_->prefill(ids.back(), prompts[static_cast<std::size_t>(r)]));
        }
        reused = (backend_->kv_stats().prefix_hit_pages - hits_before) * 16;
        while (out[0].size() < kNewTokens) {  // one batch of 16, every step
            std::vector<TokenId> last;
            for (const auto& o : out) {
                last.push_back(o.back());
            }
            const std::vector<TokenId> next = backend_->decode_step(ids, last);
            for (int r = 0; r < kRequests; ++r) {
                out[static_cast<std::size_t>(r)].push_back(next.at(static_cast<std::size_t>(r)));
            }
        }
        backend_->check_kv_invariants();
        for (const SeqId id : ids) {
            backend_->free_sequence(id);
        }
        const auto kv = backend_->kv_stats();
        EXPECT_EQ(kv.free_pages, kv.total_pages) << "KV pages leaked";
        return out;
    };

    std::int64_t reused_without = 0;
    std::int64_t reused_with = 0;
    const auto start = backend_->kv_stats();
    const auto without = run(false, reused_without);
    const auto mid = backend_->kv_stats();
    const auto with = run(true, reused_with);
    const auto end = backend_->kv_stats();
    backend_->set_prefix_caching(true);

    EXPECT_EQ(reused_without, 0);
    EXPECT_EQ(mid.prefix_lookups, start.prefix_lookups) << "no lookups with the cache off";
    const std::int64_t lookup_pages = end.prefix_lookup_pages - mid.prefix_lookup_pages;
    const std::int64_t hit_pages = end.prefix_hit_pages - mid.prefix_hit_pages;
    EXPECT_EQ(end.prefix_lookups - mid.prefix_lookups, kRequests);
    EXPECT_EQ(hit_pages, (kRequests - 1) * 32) << "every request after the first reuses the 32 system pages";
    EXPECT_EQ(reused_with, (kRequests - 1) * 512);
    const std::int64_t prompt_tokens = kRequests * static_cast<std::int64_t>(prompts[0].size());
    std::cout << "shared prefix: " << kRequests << " requests x " << prompts[0].size() << " prompt tokens; hit rate "
              << hit_pages << "/" << lookup_pages << " full pages ("
              << (lookup_pages > 0 ? 100.0 * static_cast<double>(hit_pages) / static_cast<double>(lookup_pages) : 0.0)
              << "%); " << reused_with << " of " << prompt_tokens << " prompt tokens reused, "
              << prompt_tokens - reused_with << " prefilled\n";

    int differing = 0;
    for (int r = 0; r < kRequests; ++r) {
        const auto& a = without[static_cast<std::size_t>(r)];
        const auto& b = with[static_cast<std::size_t>(r)];
        if (a != b) {
            ++differing;
            const auto first = std::mismatch(a.begin(), a.end(), b.begin()).first - a.begin();
            ADD_FAILURE() << "request " << r << ": first difference at generated token " << first << " of " << a.size();
        }
    }
    std::cout << "outputs identical to the no-cache run: " << kRequests - differing << "/" << kRequests << "\n";
}

// CACHED pages and LRU eviction (Day 37, KV-008): a request's pages outlive it and are reused by a
// later request with no overlap in time; under memory pressure the least recently released pages
// are evicted first, their pages rewritten by other requests, and outputs stay equal to Hugging Face.
TEST_F(CudaPrefixCacheTest, CachedPagesAreReusedAfterTheirRequestAndEvictedOldestFirst) {
    std::int64_t hits = 0;
    EXPECT_EQ(generate(prompts_.at(0), greedy_.at(0).size(), hits), greedy_.at(0));
    EXPECT_EQ(hits, 0);
    EXPECT_GT(backend_->kv_stats().cached_pages, 0) << "its pages stay CACHED";
    EXPECT_EQ(generate(prompts_.at(1), greedy_.at(1).size(), hits), greedy_.at(1));
    EXPECT_GT(hits, 0) << "case 1 reuses case 0's system header pages, although case 0 has finished";

    // Fill the pool with other prompts: once no page is FREE, each new page evicts the least
    // recently released CACHED one, cases 0 and 1 first.
    const std::int64_t evicted_before = backend_->kv_stats().evicted_pages;
    for (int f = 0; f < 27; ++f) {  // 27 x 40 pages > 1024
        std::vector<TokenId> filler;
        for (int i = 0; i < 640; ++i) {
            filler.push_back(2000 + f * 700 + i);
        }
        const SeqId id = backend_->create_sequence();
        backend_->prefill(id, filler);
        backend_->free_sequence(id);
    }
    const auto kv = backend_->kv_stats();
    std::cout << "evicted " << kv.evicted_pages - evicted_before << " pages; " << kv.cached_pages
              << " cached, " << kv.free_pages << " of " << kv.total_pages << " reclaimable\n";
    EXPECT_GT(kv.evicted_pages, evicted_before);
    backend_->check_kv_invariants();

    EXPECT_EQ(generate(prompts_.at(0), greedy_.at(0).size(), hits), greedy_.at(0)) << "after its pages were evicted";
    EXPECT_EQ(hits, 0) << "the oldest pages were evicted first";
    EXPECT_EQ(generate(prompts_.at(0), greedy_.at(0).size(), hits), greedy_.at(0)) << "from its own CACHED pages";
    EXPECT_GT(hits, 0);
    backend_->check_kv_invariants();
}

