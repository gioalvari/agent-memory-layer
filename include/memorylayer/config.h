#pragma once
#include <string>
#include <cstdint>

namespace memorylayer {

struct Config {
    std::string backend_url = "http://localhost:8080";
    int port = 8800;
    std::string embedding_model_path;
    std::string db_path = "memories.sqlite";
    int top_k = 5;
    int decay_days = 30;
    float dedup_threshold = 0.92f;
    int max_memories_per_agent = 1000;
    int max_memories_global = 5000;
    int gpu_layers = 99;
    float min_score_threshold = 0.3f;
    float agent_boost = 1.2f;
    int max_inject_tokens = 2048;  // Max tokens for injected memory context
};

Config parse_args(int argc, char* argv[]);

} // namespace memorylayer
