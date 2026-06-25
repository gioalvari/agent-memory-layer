#pragma once
#include "memorylayer/config.h"
#include "memorylayer/memory_store.h"
#include "memorylayer/embedding.h"
#include "json.hpp"
#include "httplib.h"
#include <string>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>

namespace memorylayer {

struct SaveJob {
    std::string agent_id;
    std::string user_text;
    std::string assist_text;
};

class MemoryProxy {
public:
    MemoryProxy(const Config& cfg, MemoryStore& store, EmbeddingWorker& embedder);
    ~MemoryProxy();

    void run();
    void stop();
    void shutdown();

private:
    const Config& cfg_;
    MemoryStore& store_;
    EmbeddingWorker& embedder_;

    std::atomic<bool> stop_{false};
    httplib::Server* svr_ptr_ = nullptr;

    std::thread saver_thread_;
    std::queue<SaveJob> save_queue_;
    std::mutex save_mutex_;
    std::condition_variable save_cv_;

    void saver_loop();
    void enqueue_save(const std::string& agent_id,
                      const std::string& user_text,
                      const std::string& assist_text);

    std::string retrieve_and_inject(const std::string& agent_id,
                                    const std::string& user_text,
                                    nlohmann::json& messages);

    std::string extract_last_user_message(const nlohmann::json& messages) const;
    std::string extract_conversation_context(const nlohmann::json& messages, int max_turns = 3) const;

    // Debug: last injection info
    struct DebugInjection {
        std::string agent_id;
        std::string query;
        std::string injected_context;
        double timestamp = 0;
    };
    DebugInjection last_injection_;
    std::mutex debug_mutex_;
};

} // namespace memorylayer
