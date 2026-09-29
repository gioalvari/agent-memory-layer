#include "memorylayer/config.h"
#include <cassert>
#include <cstring>
#include <iostream>
#include <sys/wait.h>
#include <unistd.h>

template <std::size_t N>
void assert_parse_fails(const char* (&argv)[N]) {
    const pid_t child = fork();
    assert(child >= 0);
    if (child == 0) {
        memorylayer::parse_args(static_cast<int>(N), const_cast<char**>(argv));
        _exit(0);
    }
    int status = 0;
    assert(waitpid(child, &status, 0) == child);
    assert(WIFEXITED(status));
    assert(WEXITSTATUS(status) == 1);
}

void test_defaults() {
    const char* argv[] = {"memory-layer", "--embedding-model", "test.gguf"};
    auto cfg = memorylayer::parse_args(3, const_cast<char**>(argv));
    assert(cfg.backend_url == "http://localhost:8080");
    assert(cfg.port == 8800);
    assert(cfg.embedding_model_path == "test.gguf");
    assert(cfg.db_path == "memories.sqlite");
    assert(cfg.top_k == 5);
    assert(cfg.decay_days == 30);
    assert(cfg.decay_mode == memorylayer::DecayMode::Tiebreak);
    assert(cfg.target_coverage == 0.0f);
    assert(cfg.gpu_layers == 99);
    assert(cfg.max_inject_tokens == 2048);
    assert(cfg.memory_line_chars == 400);
    // Default top-k memories at the default line length fit the default budget.
    assert(memorylayer::required_inject_tokens(cfg.top_k, cfg.memory_line_chars) <=
           cfg.max_inject_tokens);
    std::cout << "test_defaults PASSED\n";
}

void test_retrieval_options() {
    const char* valid[] = {"memory-layer", "--embedding-model", "m.gguf",
                           "--decay-mode", "legacy", "--target-coverage", "0.95"};
    const auto cfg = memorylayer::parse_args(7, const_cast<char**>(valid));
    assert(cfg.decay_mode == memorylayer::DecayMode::Legacy);
    assert(cfg.target_coverage == 0.95f);

    // Without an explicit budget, legacy 0.95 (k=26) raises the 2048 default.
    assert(!cfg.max_inject_tokens_explicit);
    assert(cfg.max_inject_tokens == memorylayer::required_inject_tokens(26, 400));
    assert(cfg.max_inject_tokens > 2048);

    const char* default_fits[] = {"memory-layer", "--embedding-model", "m.gguf",
                                  "--target-coverage", "0.9"};
    const auto fits = memorylayer::parse_args(5, const_cast<char**>(default_fits));
    assert(fits.max_inject_tokens == 2048);

    const char* high_target[] = {"memory-layer", "--embedding-model", "m.gguf",
                                 "--target-coverage", "0.95"};
    const auto high = memorylayer::parse_args(5, const_cast<char**>(high_target));
    assert(high.max_inject_tokens == memorylayer::required_inject_tokens(16, 400));

    const char* long_lines[] = {"memory-layer", "--embedding-model", "m.gguf",
                                "--target-coverage", "0.9", "--memory-line-chars", "800"};
    const auto raised = memorylayer::parse_args(7, const_cast<char**>(long_lines));
    assert(raised.memory_line_chars == 800);
    assert(raised.max_inject_tokens == memorylayer::required_inject_tokens(8, 800));

    const char* explicit_budget[] = {"memory-layer", "--embedding-model", "m.gguf",
                                     "--target-coverage", "0.95", "--decay-mode", "legacy",
                                     "--max-inject-tokens", "1000"};
    const auto kept = memorylayer::parse_args(9, const_cast<char**>(explicit_budget));
    assert(kept.max_inject_tokens_explicit);
    assert(kept.max_inject_tokens == 1000);

    const char* bad_line_chars[] = {"memory-layer", "--embedding-model", "m.gguf",
                                    "--memory-line-chars", "0"};
    assert_parse_fails(bad_line_chars);

    const char* invalid_mode[] = {"memory-layer", "--embedding-model", "m.gguf",
                                  "--decay-mode", "recent"};
    assert_parse_fails(invalid_mode);
    const char* invalid_coverage[] = {"memory-layer", "--embedding-model", "m.gguf",
                                      "--target-coverage", "0.85"};
    assert_parse_fails(invalid_coverage);
    const char* non_numeric_coverage[] = {"memory-layer", "--embedding-model", "m.gguf",
                                          "--target-coverage", "high"};
    assert_parse_fails(non_numeric_coverage);
    std::cout << "test_retrieval_options PASSED\n";
}

void test_custom_values() {
    const char* argv[] = {
        "memory-layer",
        "--backend", "http://myserver:9090",
        "--port", "9999",
        "--embedding-model", "custom.gguf",
        "--db", "custom.db",
        "--top-k", "3",
        "--decay-days", "7",
        "--dedup-threshold", "0.85",
        "--max-memories-per-agent", "500",
        "--gpu-layers", "32"
    };
    auto cfg = memorylayer::parse_args(19, const_cast<char**>(argv));
    assert(cfg.backend_url == "http://myserver:9090");
    assert(cfg.port == 9999);
    assert(cfg.embedding_model_path == "custom.gguf");
    assert(cfg.db_path == "custom.db");
    assert(cfg.top_k == 3);
    assert(cfg.decay_days == 7);
    assert(cfg.dedup_threshold == 0.85f);
    assert(cfg.max_memories_per_agent == 500);
    assert(cfg.gpu_layers == 32);
    std::cout << "test_custom_values PASSED\n";
}

void test_inject_mode() {
    const char* d[] = {"memory-layer", "--embedding-model", "m.gguf"};
    assert(memorylayer::parse_args(3, const_cast<char**>(d)).inject_mode == "system");
    const char* s[] = {"memory-layer", "--embedding-model", "m.gguf", "--inject-mode", "suffix"};
    assert(memorylayer::parse_args(5, const_cast<char**>(s)).inject_mode == "suffix");
    const char* sticky[] = {"memory-layer", "--embedding-model", "m.gguf",
                            "--inject-mode", "sticky", "--sticky-cache-entries", "12"};
    auto sticky_cfg = memorylayer::parse_args(7, const_cast<char**>(sticky));
    assert(sticky_cfg.inject_mode == "sticky");
    assert(sticky_cfg.sticky_cache_entries == 12);
    std::cout << "test_inject_mode PASSED\n";
}

int main() {
    test_inject_mode();
    test_retrieval_options();
    test_defaults();
    test_custom_values();
    std::cout << "All config tests PASSED\n";
    return 0;
}
