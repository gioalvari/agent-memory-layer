#include "memorylayer/config.h"
#include "memorylayer/memory_store.h"
#include "memorylayer/embedding.h"
#include "memorylayer/proxy.h"
#include <iostream>
#include <csignal>

static memorylayer::MemoryProxy* g_proxy = nullptr;

static void signal_handler(int) {
    std::cout << "\n[main] Shutting down...\n";
    if (g_proxy) g_proxy->stop();
}

int main(int argc, char* argv[]) {
    auto cfg = memorylayer::parse_args(argc, argv);

    std::cout << "=== Agent Memory Layer ===\n";
    std::cout << "[main] Embedding model: " << cfg.embedding_model_path << "\n";
    std::cout << "[main] Database: " << cfg.db_path << "\n";
    std::cout << "[main] Backend: " << cfg.backend_url << "\n";
    std::cout << "[main] Port: " << cfg.port << "\n";

    memorylayer::MemoryStore store(cfg.db_path, cfg.max_memories_per_agent, cfg.max_memories_global);
    std::cout << "[main] SQLite store initialized\n";

    memorylayer::EmbeddingWorker embedder(cfg.embedding_model_path, cfg.gpu_layers);
    if (!embedder.is_ready()) {
        std::cerr << "[main] ERROR: Failed to load embedding model\n";
        return 1;
    }
    std::cout << "[main] Embedding worker ready (dim=" << embedder.dimension() << ")\n";

    memorylayer::MemoryProxy proxy(cfg, store, embedder);
    g_proxy = &proxy;

    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    proxy.run();

    return 0;
}
