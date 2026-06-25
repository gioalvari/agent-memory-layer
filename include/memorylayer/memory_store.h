#pragma once
#include <string>
#include <vector>
#include <optional>
#include <cstdint>
#include <mutex>

struct sqlite3;

namespace memorylayer {

struct Memory {
    int64_t id;
    std::string agent_id;
    double created_at;
    double updated_at;
    std::string user_text;
    std::string assist_text;
    std::vector<float> user_emb;
    std::vector<float> assist_emb;
    int access_count;
};

struct ScoredMemory {
    Memory memory;
    float score;
};

class MemoryStore {
public:
    explicit MemoryStore(const std::string& db_path, int max_per_agent = 1000, int max_global = 5000);
    ~MemoryStore();

    MemoryStore(const MemoryStore&) = delete;
    MemoryStore& operator=(const MemoryStore&) = delete;

    int64_t insert(const std::string& agent_id,
                   const std::string& user_text,
                   const std::string& assist_text,
                   const std::vector<float>& user_emb,
                   const std::vector<float>& assist_emb);

    std::vector<ScoredMemory> search(const std::vector<float>& query_emb,
                                     const std::string& agent_id,
                                     int top_k,
                                     float min_score,
                                     int decay_days,
                                     float agent_boost) const;

    int64_t find_duplicate(const std::vector<float>& user_emb, float threshold, const std::string& agent_id = "") const;

    void touch(int64_t memory_id);
    void evict(const std::string& agent_id);
    int count(const std::string& agent_id) const;

    std::vector<Memory> list_memories(const std::string& agent_id, int limit = 50, int offset = 0) const;
    bool remove(int64_t memory_id);
    struct Stats {
        int total_memories;
        int total_agents;
        std::vector<std::pair<std::string, int>> per_agent_counts;
    };
    Stats get_stats() const;

private:
    sqlite3* db_ = nullptr;
    int max_per_agent_;
    int max_global_;
    void init_schema();

    // RAM vector cache for fast search
    struct CachedEntry {
        int64_t id;
        std::string agent_id;
        double created_at;
        std::string user_text;
        std::string assist_text;
        int access_count;
        int user_emb_offset;   // offset into embeddings_ vector
        int assist_emb_offset; // offset into embeddings_ vector
    };
    mutable std::vector<CachedEntry> cache_;
    mutable std::vector<float> embeddings_;  // flat array of all embeddings
    int emb_dim_ = 0;
    mutable std::mutex cache_mutex_;

    void load_cache();
    void recompact_embeddings();
    void add_to_cache(int64_t id, const std::string& agent_id, double created_at,
                      const std::string& user_text, const std::string& assist_text,
                      int access_count,
                      const std::vector<float>& user_emb, const std::vector<float>& assist_emb);
};

} // namespace memorylayer
