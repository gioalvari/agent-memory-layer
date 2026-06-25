#pragma once
#include <string>
#include <vector>
#include <optional>
#include <cstdint>

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

    int64_t find_duplicate(const std::vector<float>& user_emb, float threshold) const;

    void touch(int64_t memory_id);
    void evict(const std::string& agent_id);
    int count(const std::string& agent_id) const;

private:
    sqlite3* db_ = nullptr;
    int max_per_agent_;
    int max_global_;
    void init_schema();
};

} // namespace memorylayer
