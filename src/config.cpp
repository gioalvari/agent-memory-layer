#include "memorylayer/config.h"
#include <algorithm>
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
               << "  --decay-mode <mode>         tiebreak | legacy (default: tiebreak)\n"
               << "  --target-coverage <p>       0.8 | 0.9 | 0.95 calibrated retrieval coverage\n"
               << "  --adaptive-k                With --target-coverage: k per query from its top-1 similarity\n"
              << "  --dedup-threshold <f>       Deduplication cosine threshold (default: 0.92)\n"
              << "  --max-memories-per-agent <n> Max memories per agent (default: 1000)\n"
              << "  --gpu-layers <n>            GPU layers for embedding model (default: 99)\n"
              << "  --max-inject-tokens <n>     Max tokens for injected memory context (default: 2048;\n"
              << "                              raised to fit all k memories with --target-coverage)\n"
              << "  --memory-line-chars <n>     Bytes kept per user/assistant side of a memory (default: 400)\n"
              << "  --admin-token <token>       Bearer token required for /admin/* endpoints (default: none)\n"
              << "  --memory-ttl-days <n>       Auto-delete memories older than N days (default: 0=disabled)\n"
              << "  --similarity-threshold <f>  Minimum cosine similarity to inject a memory (default: 0.3)\n"
              << "  --max-context-tokens <n>    Guard: max total context tokens (default: 8192)\n"
               << "  --inject-mode <mode>        system | suffix | sticky (default: system)\n"
               << "  --sticky-cache-entries <n>  Sticky conversation blocks to retain (default: 4096)\n";
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
        } else if (strcmp(arg, "--decay-mode") == 0 && i + 1 < argc) {
            const std::string mode = argv[++i];
            if (mode == "tiebreak") {
                cfg.decay_mode = DecayMode::Tiebreak;
            } else if (mode == "legacy") {
                cfg.decay_mode = DecayMode::Legacy;
            } else {
                std::cerr << "Error: --decay-mode must be 'tiebreak' or 'legacy'\n\n";
                print_usage();
                std::exit(1);
            }
        } else if (strcmp(arg, "--target-coverage") == 0 && i + 1 < argc) {
            char* end = nullptr;
            const char* value = argv[++i];
            cfg.target_coverage = std::strtof(value, &end);
            if (*value == '\0' || *end != '\0') {
                std::cerr << "Error: --target-coverage must be 0.8, 0.9, or 0.95\n\n";
                print_usage();
                std::exit(1);
            }
        } else if (strcmp(arg, "--adaptive-k") == 0) {
            cfg.adaptive_k = true;
        } else if (strcmp(arg, "--dedup-threshold") == 0 && i + 1 < argc) {
            cfg.dedup_threshold = std::atof(argv[++i]);
        } else if (strcmp(arg, "--max-memories-per-agent") == 0 && i + 1 < argc) {
            cfg.max_memories_per_agent = std::atoi(argv[++i]);
        } else if (strcmp(arg, "--gpu-layers") == 0 && i + 1 < argc) {
            cfg.gpu_layers = std::atoi(argv[++i]);
        } else if (strcmp(arg, "--max-inject-tokens") == 0 && i + 1 < argc) {
            cfg.max_inject_tokens = std::atoi(argv[++i]);
            cfg.max_inject_tokens_explicit = true;
        } else if (strcmp(arg, "--memory-line-chars") == 0 && i + 1 < argc) {
            cfg.memory_line_chars = std::atoi(argv[++i]);
        } else if (strcmp(arg, "--admin-token") == 0 && i + 1 < argc) {
            cfg.admin_token = argv[++i];
        } else if (strcmp(arg, "--memory-ttl-days") == 0 && i + 1 < argc) {
            cfg.memory_ttl_days = std::atoi(argv[++i]);
        } else if (strcmp(arg, "--similarity-threshold") == 0 && i + 1 < argc) {
            cfg.similarity_threshold = static_cast<float>(std::atof(argv[++i]));
            cfg.min_score_threshold  = cfg.similarity_threshold;
        } else if (strcmp(arg, "--max-context-tokens") == 0 && i + 1 < argc) {
            cfg.max_context_tokens = std::atoi(argv[++i]);
        } else if (strcmp(arg, "--inject-mode") == 0 && i + 1 < argc) {
            cfg.inject_mode = argv[++i];
        } else if (strcmp(arg, "--sticky-cache-entries") == 0 && i + 1 < argc) {
            cfg.sticky_cache_entries = std::atoi(argv[++i]);
        } else if (strcmp(arg, "--help") == 0 || strcmp(arg, "-h") == 0) {
            print_usage();
            std::exit(0);
        }
    }

    if (cfg.inject_mode != "system" && cfg.inject_mode != "suffix" &&
        cfg.inject_mode != "sticky") {
        std::cerr << "Error: --inject-mode must be 'system', 'suffix', or 'sticky'\n\n";
        print_usage();
        std::exit(1);
    }

    if (cfg.sticky_cache_entries < 0) {
        std::cerr << "Error: --sticky-cache-entries must be non-negative\n\n";
        print_usage();
        std::exit(1);
    }

    if (cfg.decay_days <= 0) {
        std::cerr << "Error: --decay-days must be positive\n\n";
        print_usage();
        std::exit(1);
    }

    if (!is_valid_target_coverage(cfg.target_coverage)) {
        std::cerr << "Error: --target-coverage must be 0.8, 0.9, or 0.95\n\n";
        print_usage();
        std::exit(1);
    }

    if (cfg.memory_line_chars <= 0) {
        std::cerr << "Error: --memory-line-chars must be positive\n\n";
        print_usage();
        std::exit(1);
    }

    if (cfg.adaptive_k && cfg.target_coverage == 0.0f) {
        std::cerr << "Error: --adaptive-k requires --target-coverage\n\n";
        print_usage();
        std::exit(1);
    }

    if (cfg.target_coverage > 0.0f && !cfg.max_inject_tokens_explicit) {
        const int required = required_inject_tokens(
            max_calibrated_top_k(cfg.target_coverage, cfg.decay_mode, cfg.adaptive_k),
            cfg.memory_line_chars);
        cfg.max_inject_tokens = std::max(cfg.max_inject_tokens, required);
    }

    if (cfg.embedding_model_path.empty()) {
        std::cerr << "Error: --embedding-model is required\n\n";
        print_usage();
        std::exit(1);
    }

    return cfg;
}

} // namespace memorylayer
