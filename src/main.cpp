#include "memorylayer/config.h"
#include "memorylayer/memory_store.h"
#include "memorylayer/embedding.h"
#include "memorylayer/proxy.h"
#include "memorylayer/logger.h"
#include <iostream>
#include <csignal>

static memorylayer::MemoryProxy* g_proxy = nullptr;

static void signal_handler(int) {
    LOG_INFO("main", "Shutting down...");
    if (g_proxy) g_proxy->stop();
}

int main(int argc, char* argv[]) {
    auto cfg = memorylayer::parse_args(argc, argv);

    LOG_INFO("main", "=== Agent Memory Layer ===");
    LOG_INFO("main", "Embedding model: " + cfg.embedding_model_path);
    LOG_INFO("main", "Database: " + cfg.db_path);
    LOG_INFO("main", "Backend: " + cfg.backend_url);
    LOG_INFO("main", "Port: " + std::to_string(cfg.port));
    std::string k_desc = "effective_k=" + std::to_string(cfg.top_k);
    if (cfg.target_coverage > 0.0f && cfg.adaptive_k) {
        const auto* table = memorylayer::adaptive_k_table(cfg.target_coverage, cfg.decay_mode);
        const auto k_at = [&](float top1) {
            return std::to_string(
                memorylayer::adaptive_top_k(cfg.target_coverage, cfg.decay_mode, top1));
        };
        k_desc = "adaptive_k=" + k_at(table->edge_low) + "/" + k_at(table->edge_high) + "/" +
                 k_at(1.0f) +
                 " (top-1 similarity <=" + std::to_string(table->edge_low) + ", <=" +
                 std::to_string(table->edge_high) + ", above)";
    } else if (cfg.target_coverage > 0.0f) {
        k_desc = "effective_k=" +
                 std::to_string(memorylayer::calibrated_top_k(cfg.target_coverage, cfg.decay_mode));
    }
    LOG_INFO("main", "Retrieval: decay_mode=" +
             std::string(cfg.decay_mode == memorylayer::DecayMode::Tiebreak ? "tiebreak" : "legacy") +
             ", " + k_desc +
             ", max_inject_tokens=" + std::to_string(cfg.max_inject_tokens) +
             ", memory_line_chars=" + std::to_string(cfg.memory_line_chars));

    memorylayer::MemoryStore store(cfg.db_path, cfg.max_memories_per_agent, cfg.max_memories_global);
    LOG_INFO("main", "SQLite store initialized");

    memorylayer::EmbeddingWorker embedder(cfg.embedding_model_path, cfg.gpu_layers);
    if (!embedder.is_ready()) {
        LOG_ERROR("main", "Failed to load embedding model");
        return 1;
    }
    LOG_INFO("main", "Embedding worker ready (dim=" + std::to_string(embedder.dimension()) + ")");

    memorylayer::MemoryProxy proxy(cfg, store, embedder);
    g_proxy = &proxy;

    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    const bool ok = proxy.run();

    return ok ? 0 : 1;
}
