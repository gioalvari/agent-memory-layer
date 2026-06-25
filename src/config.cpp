#include "memorylayer/config.h"
#include <cstring>
#include <cstdlib>
#include <iostream>

namespace memorylayer {

static void print_usage() {
    std::cerr << "Usage: memory-layer --embedding-model <path> [options]\n"
              << "\nRequired:\n"
              << "  --embedding-model <path>    Path to GGUF embedding model\n"
              << "\nOptions:\n"
              << "  --backend <url>             LLM backend URL (default: http://localhost:8080)\n"
              << "  --port <n>                  Proxy listen port (default: 8800)\n"
              << "  --db <path>                 SQLite database path (default: memories.sqlite)\n"
              << "  --top-k <n>                 Memories to inject (default: 5)\n"
              << "  --decay-days <n>            Temporal decay half-life (default: 30)\n"
              << "  --dedup-threshold <f>       Deduplication cosine threshold (default: 0.92)\n"
              << "  --max-memories-per-agent <n> Max memories per agent (default: 1000)\n"
              << "  --gpu-layers <n>            GPU layers for embedding model (default: 99)\n"
              << "  --max-inject-tokens <n>     Max tokens for injected memory context (default: 2048)\n"
              << "  --admin-token <token>       Bearer token required for /admin/* endpoints (default: none)\n";
}

Config parse_args(int argc, char* argv[]) {
    Config cfg;

    for (int i = 1; i < argc; i++) {
        const char* arg = argv[i];

        if (strcmp(arg, "--backend") == 0 && i + 1 < argc) {
            cfg.backend_url = argv[++i];
        } else if (strcmp(arg, "--port") == 0 && i + 1 < argc) {
            cfg.port = std::atoi(argv[++i]);
        } else if (strcmp(arg, "--embedding-model") == 0 && i + 1 < argc) {
            cfg.embedding_model_path = argv[++i];
        } else if (strcmp(arg, "--db") == 0 && i + 1 < argc) {
            cfg.db_path = argv[++i];
        } else if (strcmp(arg, "--top-k") == 0 && i + 1 < argc) {
            cfg.top_k = std::atoi(argv[++i]);
        } else if (strcmp(arg, "--decay-days") == 0 && i + 1 < argc) {
            cfg.decay_days = std::atoi(argv[++i]);
        } else if (strcmp(arg, "--dedup-threshold") == 0 && i + 1 < argc) {
            cfg.dedup_threshold = std::atof(argv[++i]);
        } else if (strcmp(arg, "--max-memories-per-agent") == 0 && i + 1 < argc) {
            cfg.max_memories_per_agent = std::atoi(argv[++i]);
        } else if (strcmp(arg, "--gpu-layers") == 0 && i + 1 < argc) {
            cfg.gpu_layers = std::atoi(argv[++i]);
        } else if (strcmp(arg, "--max-inject-tokens") == 0 && i + 1 < argc) {
            cfg.max_inject_tokens = std::atoi(argv[++i]);
        } else if (strcmp(arg, "--admin-token") == 0 && i + 1 < argc) {
            cfg.admin_token = argv[++i];
        } else if (strcmp(arg, "--help") == 0 || strcmp(arg, "-h") == 0) {
            print_usage();
            std::exit(0);
        }
    }

    if (cfg.embedding_model_path.empty()) {
        std::cerr << "Error: --embedding-model is required\n\n";
        print_usage();
        std::exit(1);
    }

    return cfg;
}

} // namespace memorylayer
