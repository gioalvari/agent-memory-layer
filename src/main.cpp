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
    const int effective_k = cfg.target_coverage > 0.0f
        ? memorylayer::calibrated_top_k(cfg.target_coverage, cfg.decay_mode)
        : cfg.top_k;
    LOG_INFO("main", "Retrieval: decay_mode=" +
             std::string(cfg.decay_mode == memorylayer::DecayMode::Tiebreak ? "tiebreak" : "legacy") +
             ", effective_k=" + std::to_string(effective_k));

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
