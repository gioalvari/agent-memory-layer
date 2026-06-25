#pragma once
#include "memorylayer/config.h"
#include "memorylayer/memory_store.h"
#include "memorylayer/embedding.h"
#include "json.hpp"
#include "httplib.h"
#include <string>
#include <thread>
#include <queue>
#include <deque>
#include <mutex>
#include <condition_variable>
#include <atomic>

namespace memorylayer {

struct SaveJob {
    std::string agent_id;
    std::string user_text;   // full context stored in DB (human-readable)
    std::string assist_text;
    std::string embed_text;  // last user message — what gets embedded for retrieval
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
    std::atomic<httplib::Server*> svr_ptr_{nullptr};

    std::thread saver_thread_;
    std::queue<SaveJob> save_queue_;
    std::mutex save_mutex_;
    std::condition_variable save_cv_;

    void saver_loop();
    void enqueue_save(const std::string& agent_id,
                      const std::string& user_text,
                      const std::string& assist_text,
                      const std::string& embed_text = "");

    std::atomic<int64_t> save_drop_count_{0};

    std::string retrieve_and_inject(const std::string& agent_id,
                                    const std::string& user_text,
                                    nlohmann::json& messages);

    std::string extract_last_user_message(const nlohmann::json& messages) const;
    std::string extract_conversation_context(const nlohmann::json& messages, int max_turns = 3) const;
    // Returns false and sets 401 on res if admin token is configured and does not match.
    bool check_admin_auth(const httplib::Request& req, httplib::Response& res) const;

    // Debug: injection history ring buffer (newest first, capped at kInjectionRingSize).
    struct DebugInjection {
        std::string agent_id;
        std::string query;
        std::string injected_context;
        double timestamp = 0;
    };
    static constexpr int kInjectionRingSize = 10;
    std::deque<DebugInjection> injection_ring_;
    std::mutex debug_mutex_;
};

} // namespace memorylayer
