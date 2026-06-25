#include "memorylayer/memory_store.h"
#include "sqlite3.h"
#include <stdexcept>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <ctime>
#include <iostream>
#include <map>

namespace memorylayer {

MemoryStore::MemoryStore(const std::string& db_path, int max_per_agent, int max_global)
    : max_per_agent_(max_per_agent), max_global_(max_global) {
    int rc = sqlite3_open(db_path.c_str(), &db_);
    if (rc != SQLITE_OK) {
        throw std::runtime_error("Failed to open database: " + std::string(sqlite3_errmsg(db_)));
    }
    sqlite3_exec(db_, "PRAGMA journal_mode=WAL;", nullptr, nullptr, nullptr);
    sqlite3_exec(db_, "PRAGMA synchronous=NORMAL;", nullptr, nullptr, nullptr);
    init_schema();
    load_cache();
}

MemoryStore::~MemoryStore() {
    if (db_) sqlite3_close(db_);
}

void MemoryStore::init_schema() {
    const char* sql = R"(
        CREATE TABLE IF NOT EXISTS memories (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_id     TEXT,
            created_at   REAL NOT NULL,
            updated_at   REAL NOT NULL,
            user_text    TEXT NOT NULL,
            assist_text  TEXT NOT NULL,
            user_emb     BLOB NOT NULL,
            assist_emb   BLOB NOT NULL,
            access_count INTEGER DEFAULT 1
        );
        CREATE INDEX IF NOT EXISTS idx_memories_agent ON memories(agent_id);
        CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at DESC);
    )";
    char* err = nullptr;
    sqlite3_exec(db_, sql, nullptr, nullptr, &err);
    if (err) {
        std::string msg = err;
        sqlite3_free(err);
        throw std::runtime_error("Schema init failed: " + msg);
    }
}

void MemoryStore::load_cache() {
    const char* sql = "SELECT id, agent_id, created_at, user_text, assist_text, user_emb, assist_emb, access_count FROM memories ORDER BY id";
    sqlite3_stmt* stmt = nullptr;
    int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        std::cerr << "[store] load_cache prepare failed: " << sqlite3_errmsg(db_) << "\n";
        return;
    }

    while (sqlite3_step(stmt) == SQLITE_ROW) {
        CachedEntry entry;
        entry.id = sqlite3_column_int64(stmt, 0);
        const char* aid = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        entry.agent_id = aid ? aid : "";
        entry.created_at = sqlite3_column_double(stmt, 2);
        const char* ut = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3));
        entry.user_text = ut ? ut : "";
        const char* at = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 4));
        entry.assist_text = at ? at : "";
        entry.access_count = sqlite3_column_int(stmt, 7);

        const float* u_blob = reinterpret_cast<const float*>(sqlite3_column_blob(stmt, 5));
        int u_size = sqlite3_column_bytes(stmt, 5) / static_cast<int>(sizeof(float));
        const float* a_blob = reinterpret_cast<const float*>(sqlite3_column_blob(stmt, 6));
        int a_size = sqlite3_column_bytes(stmt, 6) / static_cast<int>(sizeof(float));

        if (emb_dim_ == 0 && u_size > 0) emb_dim_ = u_size;

        entry.user_emb_offset = static_cast<int>(embeddings_.size());
        embeddings_.insert(embeddings_.end(), u_blob, u_blob + u_size);
        entry.assist_emb_offset = static_cast<int>(embeddings_.size());
        embeddings_.insert(embeddings_.end(), a_blob, a_blob + a_size);

        cache_.push_back(std::move(entry));
    }
    sqlite3_finalize(stmt);
}

void MemoryStore::add_to_cache(int64_t id, const std::string& agent_id, double created_at,
                               const std::string& user_text, const std::string& assist_text,
                               int access_count,
                               const std::vector<float>& user_emb, const std::vector<float>& assist_emb) {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    if (emb_dim_ == 0 && !user_emb.empty()) emb_dim_ = static_cast<int>(user_emb.size());

    CachedEntry entry;
    entry.id = id;
    entry.agent_id = agent_id;
    entry.created_at = created_at;
    entry.user_text = user_text;
    entry.assist_text = assist_text;
    entry.access_count = access_count;
    entry.user_emb_offset = static_cast<int>(embeddings_.size());
    embeddings_.insert(embeddings_.end(), user_emb.begin(), user_emb.end());
    entry.assist_emb_offset = static_cast<int>(embeddings_.size());
    embeddings_.insert(embeddings_.end(), assist_emb.begin(), assist_emb.end());
    cache_.push_back(std::move(entry));
}

static double now_unix() {
    return static_cast<double>(std::time(nullptr));
}

int64_t MemoryStore::insert(const std::string& agent_id,
                            const std::string& user_text,
                            const std::string& assist_text,
                            const std::vector<float>& user_emb,
                            const std::vector<float>& assist_emb) {
    const char* sql = R"(
        INSERT INTO memories (agent_id, created_at, updated_at, user_text, assist_text, user_emb, assist_emb)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    )";
    sqlite3_stmt* stmt = nullptr;
    int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        std::cerr << "[store] insert prepare failed: " << sqlite3_errmsg(db_) << "\n";
        return -1;
    }

    if (agent_id.empty()) {
        sqlite3_bind_null(stmt, 1);
    } else {
        sqlite3_bind_text(stmt, 1, agent_id.c_str(), -1, SQLITE_TRANSIENT);
    }

    double ts = now_unix();
    sqlite3_bind_double(stmt, 2, ts);
    sqlite3_bind_double(stmt, 3, ts);
    sqlite3_bind_text(stmt, 4, user_text.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 5, assist_text.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_blob(stmt, 6, user_emb.data(), user_emb.size() * sizeof(float), SQLITE_TRANSIENT);
    sqlite3_bind_blob(stmt, 7, assist_emb.data(), assist_emb.size() * sizeof(float), SQLITE_TRANSIENT);

    rc = sqlite3_step(stmt);
    if (rc != SQLITE_DONE) {
        std::cerr << "[store] insert step failed: " << sqlite3_errmsg(db_) << "\n";
        sqlite3_finalize(stmt);
        return -1;
    }
    int64_t id = sqlite3_last_insert_rowid(db_);
    sqlite3_finalize(stmt);

    add_to_cache(id, agent_id, ts, user_text, assist_text, 1, user_emb, assist_emb);

    return id;
}

int MemoryStore::count(const std::string& agent_id) const {
    const char* sql = agent_id.empty()
        ? "SELECT COUNT(*) FROM memories WHERE agent_id IS NULL"
        : "SELECT COUNT(*) FROM memories WHERE agent_id = ?";

    sqlite3_stmt* stmt = nullptr;
    int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        std::cerr << "[store] count prepare failed: " << sqlite3_errmsg(db_) << "\n";
        return 0;
    }
    if (!agent_id.empty()) {
        sqlite3_bind_text(stmt, 1, agent_id.c_str(), -1, SQLITE_TRANSIENT);
    }

    int result = 0;
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        result = sqlite3_column_int(stmt, 0);
    }
    sqlite3_finalize(stmt);
    return result;
}

void MemoryStore::touch(int64_t memory_id) {
    const char* sql = "UPDATE memories SET updated_at = ?, access_count = access_count + 1 WHERE id = ?";
    sqlite3_stmt* stmt = nullptr;
    int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        std::cerr << "[store] touch prepare failed: " << sqlite3_errmsg(db_) << "\n";
        return;
    }
    sqlite3_bind_double(stmt, 1, now_unix());
    sqlite3_bind_int64(stmt, 2, memory_id);
    rc = sqlite3_step(stmt);
    if (rc != SQLITE_DONE) {
        std::cerr << "[store] touch step failed: " << sqlite3_errmsg(db_) << "\n";
    }
    sqlite3_finalize(stmt);

    // Update cache
    std::lock_guard<std::mutex> lock(cache_mutex_);
    for (auto& entry : cache_) {
        if (entry.id == memory_id) {
            entry.access_count++;
            break;
        }
    }
}

void MemoryStore::evict(const std::string& agent_id) {
    int limit = agent_id.empty() ? max_global_ : max_per_agent_;
    int current = count(agent_id);
    if (current <= limit) return;

    int to_delete = current - limit;

    // Get IDs that will be deleted
    const char* id_sql = agent_id.empty()
        ? "SELECT id FROM memories WHERE agent_id IS NULL ORDER BY created_at ASC LIMIT ?"
        : "SELECT id FROM memories WHERE agent_id = ? ORDER BY created_at ASC LIMIT ?";

    sqlite3_stmt* id_stmt = nullptr;
    int rc = sqlite3_prepare_v2(db_, id_sql, -1, &id_stmt, nullptr);
    if (rc != SQLITE_OK) {
        std::cerr << "[store] evict id-select prepare failed: " << sqlite3_errmsg(db_) << "\n";
        return;
    }
    if (agent_id.empty()) {
        sqlite3_bind_int(id_stmt, 1, to_delete);
    } else {
        sqlite3_bind_text(id_stmt, 1, agent_id.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_int(id_stmt, 2, to_delete);
    }

    std::vector<int64_t> ids_to_remove;
    while (sqlite3_step(id_stmt) == SQLITE_ROW) {
        ids_to_remove.push_back(sqlite3_column_int64(id_stmt, 0));
    }
    sqlite3_finalize(id_stmt);

    // Delete from SQLite
    const char* sql = agent_id.empty()
        ? "DELETE FROM memories WHERE id IN (SELECT id FROM memories WHERE agent_id IS NULL ORDER BY created_at ASC LIMIT ?)"
        : "DELETE FROM memories WHERE id IN (SELECT id FROM memories WHERE agent_id = ? ORDER BY created_at ASC LIMIT ?)";

    sqlite3_stmt* stmt = nullptr;
    rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        std::cerr << "[store] evict delete prepare failed: " << sqlite3_errmsg(db_) << "\n";
        return;
    }
    if (agent_id.empty()) {
        sqlite3_bind_int(stmt, 1, to_delete);
    } else {
        sqlite3_bind_text(stmt, 1, agent_id.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_int(stmt, 2, to_delete);
    }
    rc = sqlite3_step(stmt);
    if (rc != SQLITE_DONE) {
        std::cerr << "[store] evict delete step failed: " << sqlite3_errmsg(db_) << "\n";
    }
    sqlite3_finalize(stmt);

    // Remove from cache
    std::lock_guard<std::mutex> lock(cache_mutex_);
    cache_.erase(
        std::remove_if(cache_.begin(), cache_.end(),
            [&ids_to_remove](const CachedEntry& e) {
                return std::find(ids_to_remove.begin(), ids_to_remove.end(), e.id) != ids_to_remove.end();
            }),
        cache_.end());
}

static float cosine_similarity(const float* a, const float* b, int dim) {
    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
    for (int i = 0; i < dim; i++) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    float denom = std::sqrt(norm_a) * std::sqrt(norm_b);
    return denom > 0.0f ? dot / denom : 0.0f;
}

std::vector<ScoredMemory> MemoryStore::search(const std::vector<float>& query_emb,
                                               const std::string& agent_id,
                                               int top_k,
                                               float min_score,
                                               int decay_days,
                                               float agent_boost) const {
    std::lock_guard<std::mutex> lock(cache_mutex_);

    double now = now_unix();
    double decay_hours_max = 24.0 * decay_days;
    int dim = static_cast<int>(query_emb.size());

    std::vector<ScoredMemory> results;

    for (const auto& entry : cache_) {
        // Filter: include global memories (empty agent_id) and agent-specific ones
        if (!agent_id.empty() && !entry.agent_id.empty() && entry.agent_id != agent_id) {
            continue;
        }
        if (agent_id.empty() && !entry.agent_id.empty()) {
            continue;
        }

        const float* u_ptr = embeddings_.data() + entry.user_emb_offset;
        const float* a_ptr = embeddings_.data() + entry.assist_emb_offset;

        float cos_user = cosine_similarity(query_emb.data(), u_ptr, dim);
        float cos_assist = cosine_similarity(query_emb.data(), a_ptr, dim);
        float cos_max = std::max(cos_user, cos_assist);

        double age_hours = (now - entry.created_at) / 3600.0;
        float decay = std::max(0.5f, 1.0f - static_cast<float>(age_hours / decay_hours_max));

        float score = cos_max * decay;

        if (!entry.agent_id.empty() && entry.agent_id == agent_id) {
            score *= agent_boost;
        }

        if (score >= min_score) {
            Memory mem;
            mem.id = entry.id;
            mem.agent_id = entry.agent_id;
            mem.created_at = entry.created_at;
            mem.updated_at = entry.created_at; // approximation; not stored in cache
            mem.user_text = entry.user_text;
            mem.assist_text = entry.assist_text;
            mem.user_emb.assign(u_ptr, u_ptr + dim);
            mem.assist_emb.assign(a_ptr, a_ptr + dim);
            mem.access_count = entry.access_count;
            results.push_back({std::move(mem), score});
        }
    }

    std::sort(results.begin(), results.end(),
              [](const ScoredMemory& a, const ScoredMemory& b) { return a.score > b.score; });

    if (static_cast<int>(results.size()) > top_k) {
        results.resize(top_k);
    }

    return results;
}

int64_t MemoryStore::find_duplicate(const std::vector<float>& user_emb, float threshold, const std::string& agent_id) const {
    std::lock_guard<std::mutex> lock(cache_mutex_);

    int dim = static_cast<int>(user_emb.size());
    int64_t best_id = -1;
    float best_cos = 0.0f;

    for (const auto& entry : cache_) {
        // Scope to agent: include global memories and those matching agent_id
        if (!agent_id.empty() && !entry.agent_id.empty() && entry.agent_id != agent_id) {
            continue;
        }
        if (agent_id.empty() && !entry.agent_id.empty()) {
            continue;
        }

        const float* u_ptr = embeddings_.data() + entry.user_emb_offset;
        // Check dimension matches
        int entry_dim = entry.assist_emb_offset - entry.user_emb_offset;
        if (entry_dim != dim) continue;

        float cos = cosine_similarity(user_emb.data(), u_ptr, dim);
        if (cos > best_cos) {
            best_cos = cos;
            best_id = entry.id;
        }
    }

    return best_cos >= threshold ? best_id : -1;
}

std::vector<Memory> MemoryStore::list_memories(const std::string& agent_id, int limit, int offset) const {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    std::vector<Memory> results;

    int skipped = 0;
    for (auto it = cache_.rbegin(); it != cache_.rend() && static_cast<int>(results.size()) < limit; ++it) {
        bool match = agent_id.empty() || it->agent_id == agent_id;
        if (!match) continue;
        if (skipped < offset) { skipped++; continue; }

        Memory m;
        m.id = it->id;
        m.agent_id = it->agent_id;
        m.created_at = it->created_at;
        m.user_text = it->user_text;
        m.assist_text = it->assist_text;
        m.access_count = it->access_count;
        results.push_back(std::move(m));
    }
    return results;
}

bool MemoryStore::remove(int64_t memory_id) {
    const char* sql = "DELETE FROM memories WHERE id = ?";
    sqlite3_stmt* stmt = nullptr;
    int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) return false;
    sqlite3_bind_int64(stmt, 1, memory_id);
    sqlite3_step(stmt);
    int changes = sqlite3_changes(db_);
    sqlite3_finalize(stmt);

    if (changes > 0) {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        cache_.erase(std::remove_if(cache_.begin(), cache_.end(),
            [memory_id](const CachedEntry& e) { return e.id == memory_id; }), cache_.end());
    }
    return changes > 0;
}

MemoryStore::Stats MemoryStore::get_stats() const {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    Stats stats;
    stats.total_memories = static_cast<int>(cache_.size());

    std::map<std::string, int> counts;
    for (const auto& entry : cache_) {
        std::string key = entry.agent_id.empty() ? "(global)" : entry.agent_id;
        counts[key]++;
    }
    stats.total_agents = static_cast<int>(counts.size());
    stats.per_agent_counts.assign(counts.begin(), counts.end());
    return stats;
}

} // namespace memorylayer
