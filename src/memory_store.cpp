#include "memorylayer/memory_store.h"
#include "sqlite3.h"
#include <stdexcept>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <ctime>

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
    sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);

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

    sqlite3_step(stmt);
    int64_t id = sqlite3_last_insert_rowid(db_);
    sqlite3_finalize(stmt);
    return id;
}

int MemoryStore::count(const std::string& agent_id) const {
    const char* sql = agent_id.empty()
        ? "SELECT COUNT(*) FROM memories WHERE agent_id IS NULL"
        : "SELECT COUNT(*) FROM memories WHERE agent_id = ?";

    sqlite3_stmt* stmt = nullptr;
    sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
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
    sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    sqlite3_bind_double(stmt, 1, now_unix());
    sqlite3_bind_int64(stmt, 2, memory_id);
    sqlite3_step(stmt);
    sqlite3_finalize(stmt);
}

void MemoryStore::evict(const std::string& agent_id) {
    int limit = agent_id.empty() ? max_global_ : max_per_agent_;
    int current = count(agent_id);
    if (current <= limit) return;

    int to_delete = current - limit;

    const char* sql = agent_id.empty()
        ? "DELETE FROM memories WHERE id IN (SELECT id FROM memories WHERE agent_id IS NULL ORDER BY created_at ASC LIMIT ?)"
        : "DELETE FROM memories WHERE id IN (SELECT id FROM memories WHERE agent_id = ? ORDER BY created_at ASC LIMIT ?)";

    sqlite3_stmt* stmt = nullptr;
    sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (agent_id.empty()) {
        sqlite3_bind_int(stmt, 1, to_delete);
    } else {
        sqlite3_bind_text(stmt, 1, agent_id.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_int(stmt, 2, to_delete);
    }
    sqlite3_step(stmt);
    sqlite3_finalize(stmt);
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
    const char* sql = agent_id.empty()
        ? "SELECT id, agent_id, created_at, updated_at, user_text, assist_text, user_emb, assist_emb, access_count FROM memories WHERE agent_id IS NULL"
        : "SELECT id, agent_id, created_at, updated_at, user_text, assist_text, user_emb, assist_emb, access_count FROM memories WHERE agent_id IS NULL OR agent_id = ?";

    sqlite3_stmt* stmt = nullptr;
    sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (!agent_id.empty()) {
        sqlite3_bind_text(stmt, 1, agent_id.c_str(), -1, SQLITE_TRANSIENT);
    }

    double now = now_unix();
    double decay_hours_max = 24.0 * decay_days;
    int dim = static_cast<int>(query_emb.size());

    std::vector<ScoredMemory> results;

    while (sqlite3_step(stmt) == SQLITE_ROW) {
        Memory mem;
        mem.id = sqlite3_column_int64(stmt, 0);

        const char* aid = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        mem.agent_id = aid ? aid : "";
        mem.created_at = sqlite3_column_double(stmt, 2);
        mem.updated_at = sqlite3_column_double(stmt, 3);

        const char* ut = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 4));
        const char* at_text = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 5));
        mem.user_text = ut ? ut : "";
        mem.assist_text = at_text ? at_text : "";

        const float* u_blob = reinterpret_cast<const float*>(sqlite3_column_blob(stmt, 6));
        int u_size = sqlite3_column_bytes(stmt, 6) / static_cast<int>(sizeof(float));
        mem.user_emb.assign(u_blob, u_blob + u_size);

        const float* a_blob = reinterpret_cast<const float*>(sqlite3_column_blob(stmt, 7));
        int a_size = sqlite3_column_bytes(stmt, 7) / static_cast<int>(sizeof(float));
        mem.assist_emb.assign(a_blob, a_blob + a_size);

        mem.access_count = sqlite3_column_int(stmt, 8);

        float cos_user = cosine_similarity(query_emb.data(), mem.user_emb.data(), dim);
        float cos_assist = cosine_similarity(query_emb.data(), mem.assist_emb.data(), dim);
        float cos_max = std::max(cos_user, cos_assist);

        double age_hours = (now - mem.created_at) / 3600.0;
        float decay = std::max(0.5f, 1.0f - static_cast<float>(age_hours / decay_hours_max));

        float score = cos_max * decay;

        if (!mem.agent_id.empty() && mem.agent_id == agent_id) {
            score *= agent_boost;
        }

        if (score >= min_score) {
            results.push_back({std::move(mem), score});
        }
    }
    sqlite3_finalize(stmt);

    std::sort(results.begin(), results.end(),
              [](const ScoredMemory& a, const ScoredMemory& b) { return a.score > b.score; });

    if (static_cast<int>(results.size()) > top_k) {
        results.resize(top_k);
    }

    return results;
}

int64_t MemoryStore::find_duplicate(const std::vector<float>& user_emb, float threshold) const {
    const char* sql = "SELECT id, user_emb FROM memories";
    sqlite3_stmt* stmt = nullptr;
    sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);

    int dim = static_cast<int>(user_emb.size());
    int64_t best_id = -1;
    float best_cos = 0.0f;

    while (sqlite3_step(stmt) == SQLITE_ROW) {
        int64_t id = sqlite3_column_int64(stmt, 0);
        const float* blob = reinterpret_cast<const float*>(sqlite3_column_blob(stmt, 1));
        int size = sqlite3_column_bytes(stmt, 1) / static_cast<int>(sizeof(float));
        if (size != dim) continue;

        float cos = cosine_similarity(user_emb.data(), blob, dim);
        if (cos > best_cos) {
            best_cos = cos;
            best_id = id;
        }
    }
    sqlite3_finalize(stmt);

    return best_cos >= threshold ? best_id : -1;
}

} // namespace memorylayer
