#pragma once
#include "memorylayer/retrieval.h"
#include "memorylayer/temporal.h"
#include <string>
#include <vector>
#include <optional>
#include <cstdint>
#include <mutex>

struct sqlite3;

namespace memorylayer {

// Text stays in memories; rows only retain byte ranges into that text plus the
// vector. This avoids duplicating potentially large conversation turns.
struct MemorySentence {
    int idx = 0;
    int start_byte = 0;
    int end_byte = 0;
    std::vector<float> embedding;
};

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
    std::vector<MemorySentence> user_sentences;
    std::vector<MemorySentence> assist_sentences;
};

struct ScoredMemory {
    Memory memory;
    float score;
    float similarity = 0.0f;  // raw max(cos user, cos assistant), no boost or decay
};

// Converts arbitrary user input into a safe FTS5 MATCH expression. Each
// alphanumeric token is quoted and combined with OR; an empty result means
// lexical retrieval should be skipped.
std::string sanitize_fts_query(const std::string& query);

struct HybridSearchResult {
    std::vector<ScoredMemory> memories;
    float dense_top_similarity = 0.0f;
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
                    const std::vector<float>& assist_emb,
                    std::optional<double> created_at = std::nullopt);

    std::vector<ScoredMemory> search(const std::vector<float>& query_emb,
                                     const std::string& agent_id,
                                     int top_k,
                                       float min_score,
                                       int decay_days,
                                       float agent_boost,
                                       DecayMode decay_mode = DecayMode::Tiebreak,
                                       std::optional<TemporalWindow> temporal_window = std::nullopt,
                                       float temporal_boost = 0.0f,
                                       std::optional<double> now_unix = std::nullopt) const;

    // score is the RRF fusion score in hybrid results; similarity remains the
    // raw dense cosine (zero for lexical-only hits).
    HybridSearchResult search_hybrid(const std::vector<float>& query_emb,
                                     const std::string& user_query,
                                     const std::string& agent_id,
                                     int top_k,
                                     float min_score,
                                     int decay_days,
                                      float agent_boost,
                                      int rrf_k,
                                      DecayMode decay_mode = DecayMode::Tiebreak,
                                      std::optional<TemporalWindow> temporal_window = std::nullopt,
                                      float temporal_boost = 0.0f,
                                      std::optional<double> now_unix = std::nullopt) const;

    int64_t find_duplicate(const std::vector<float>& user_emb, float threshold, const std::string& agent_id = "") const;

    void touch(int64_t memory_id);
    void evict(const std::string& agent_id);
    void purge_expired(int ttl_days);
    int count(const std::string& agent_id) const;

    std::vector<Memory> list_memories(const std::string& agent_id, int limit = 50, int offset = 0) const;
    bool remove(int64_t memory_id);
    void store_sentences(int64_t memory_id, char side,
                         const std::vector<MemorySentence>& sentences);
    void load_sentences(std::vector<ScoredMemory>& memories) const;
    int sentence_count(int64_t memory_id) const;
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

    // RAM vector cache for fast search. mentioned_days contains UTC epoch days
    // extracted at save time and supports temporal ranking without DB reads.
    struct CachedEntry {
        int64_t id;
        std::string agent_id;
        double created_at;
        std::string user_text;
        std::string assist_text;
        std::vector<int64_t> mentioned_days;
        int access_count;
        int user_emb_offset;   // offset into embeddings_ vector
        int assist_emb_offset; // offset into embeddings_ vector
    };
    mutable std::vector<CachedEntry> cache_;
    mutable std::vector<float> embeddings_;  // flat array of all embeddings
    int emb_dim_ = 0;
    mutable int holes_count_ = 0;   // entries erased from cache_ but not yet recompacted
    mutable std::mutex cache_mutex_;
    mutable std::mutex db_mutex_;

    void load_cache();
    void recompact_embeddings();
    bool should_recompact() const;  // caller must hold cache_mutex_
    void add_to_cache(int64_t id, const std::string& agent_id, double created_at,
                      const std::string& user_text, const std::string& assist_text,
                       int access_count,
                       const std::vector<float>& user_emb, const std::vector<float>& assist_emb,
                       std::vector<int64_t> mentioned_days);
    void backfill_memory_dates();
    void store_memory_dates(int64_t memory_id, const std::vector<int64_t>& days);
};

} // namespace memorylayer
