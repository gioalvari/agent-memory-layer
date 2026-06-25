#include "memorylayer/config.h"
#include <cassert>
#include <cstring>
#include <iostream>

void test_defaults() {
    const char* argv[] = {"memory-layer", "--embedding-model", "test.gguf"};
    auto cfg = memorylayer::parse_args(3, const_cast<char**>(argv));
    assert(cfg.backend_url == "http://localhost:8080");
    assert(cfg.port == 8800);
    assert(cfg.embedding_model_path == "test.gguf");
    assert(cfg.db_path == "memories.sqlite");
    assert(cfg.top_k == 5);
    assert(cfg.decay_days == 30);
    assert(cfg.gpu_layers == 99);
    std::cout << "test_defaults PASSED\n";
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

int main() {
    test_defaults();
    test_custom_values();
    std::cout << "All config tests PASSED\n";
    return 0;
}
