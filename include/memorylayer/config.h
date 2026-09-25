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
    std::string admin_token;       // If set, /admin/* requires Authorization: Bearer <token>
    int memory_ttl_days = 0;       // Delete memories older than N days (0 = disabled)
    float similarity_threshold = 0.3f; // Alias for min_score_threshold (CLI: --similarity-threshold)
    int max_context_tokens = 8192; // Max total context tokens; guards against injection overflow
    // Where retrieved memories go: "system" appends them to the system prompt;
    // "suffix" prepends them to the last user message so the system prompt and
    // conversation history stay byte-identical (backend prefix/KV cache reuse).
    std::string inject_mode = "system";
};

Config parse_args(int argc, char* argv[]);

} // namespace memorylayer
