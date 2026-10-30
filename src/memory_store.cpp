#include "memorylayer/memory_store.h"
#include "sqlite3.h"
#include <stdexcept>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <ctime>
#include <iostream>
#include <map>
#include <cctype>
#include <unordered_map>
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#endif

namespace memorylayer {

static double now_unix();
static float dot_product(const float* a, const float* b, int dim);

std::string sanitize_fts_query(const std::string& query) {
    std::vector<std::string> terms;
    std::string term;
    for (unsigned char ch : query) {
        if (std::isalnum(ch)) {
            term += static_cast<char>(ch);
        } else if (!term.empty()) {
            terms.push_back(std::move(term));
            term.clear();
        }
    }
    if (!term.empty()) terms.push_back(std::move(term));

    std::string result;
    for (const auto& value : terms) {
        if (!result.empty()) result += " OR ";
        result += '"' + value + '"';
    }
    return result;
}

MemoryStore::MemoryStore(const std::string& db_path, int max_per_agent, int max_global)
    : max_per_agent_(max_per_agent), max_global_(max_global) {
    int rc = sqlite3_open(db_path.c_str(), &db_);
    if (rc != SQLITE_OK) {
        throw std::runtime_error("Failed to open database: " + std::string(sqlite3_errmsg(db_)));
    }
    sqlite3_exec(db_, "PRAGMA journal_mode=WAL;", nullptr, nullptr, nullptr);
    sqlite3_exec(db_, "PRAGMA synchronous=NORMAL;", nullptr, nullptr, nullptr);
    sqlite3_exec(db_, "PRAGMA foreign_keys=ON;", nullptr, nullptr, nullptr);
    init_schema();
    backfill_memory_dates();
    load_cache();
    load_facts_cache();
}

MemoryStore::~MemoryStore() {
    if (db_) sqlite3_close(db_);
}

void MemoryStore::init_schema() {
    std::lock_guard<std::recursive_mutex> db_lock(db_mutex_);
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

        CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
            user_text, assist_text, content=''
        );

        CREATE TRIGGER IF NOT EXISTS memories_fts_insert AFTER INSERT ON memories BEGIN
            INSERT INTO memories_fts(rowid, user_text, assist_text)
            VALUES (new.id, new.user_text, new.assist_text);
        END;
        CREATE TRIGGER IF NOT EXISTS memories_fts_delete AFTER DELETE ON memories BEGIN
            INSERT INTO memories_fts(memories_fts, rowid, user_text, assist_text)
            VALUES ('delete', old.id, old.user_text, old.assist_text);
        END;
        CREATE TRIGGER IF NOT EXISTS memories_fts_update
        AFTER UPDATE OF user_text, assist_text ON memories BEGIN
            INSERT INTO memories_fts(memories_fts, rowid, user_text, assist_text)
            VALUES ('delete', old.id, old.user_text, old.assist_text);
            INSERT INTO memories_fts(rowid, user_text, assist_text)
            VALUES (new.id, new.user_text, new.assist_text);
        END;
        CREATE TABLE IF NOT EXISTS memory_dates (
            memory_id INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
            day       INTEGER NOT NULL,
            PRIMARY KEY (memory_id, day)
        );
        CREATE INDEX IF NOT EXISTS idx_memory_dates_day ON memory_dates(day);
        CREATE TABLE IF NOT EXISTS memory_sentences (
            memory_id  INTEGER NOT NULL,
            side       TEXT NOT NULL CHECK(side IN ('u', 'a')),
            idx        INTEGER NOT NULL,
            start_byte INTEGER NOT NULL,
            end_byte   INTEGER NOT NULL,
            embedding  BLOB NOT NULL,
            PRIMARY KEY (memory_id, side, idx),
            FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_memory_sentences_memory
            ON memory_sentences(memory_id);
        CREATE TABLE IF NOT EXISTS facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT, agent_id TEXT, subject TEXT NOT NULL,
            attribute TEXT NOT NULL, value TEXT NOT NULL, key TEXT NOT NULL,
            source_memory_id INTEGER, created_at REAL NOT NULL, superseded_by INTEGER,
            embedding BLOB NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_facts_active ON facts(agent_id, key) WHERE superseded_by IS NULL;
    )";
    char* err = nullptr;
    sqlite3_exec(db_, sql, nullptr, nullptr, &err);
    if (err) {
        std::string msg = err;
        sqlite3_free(err);
        throw std::runtime_error("Schema init failed: " + msg);
    }

    int memory_count = 0;
    int fts_count = 0;
    sqlite3_stmt* count_stmt = nullptr;
    if (sqlite3_prepare_v2(db_, "SELECT COUNT(*) FROM memories", -1, &count_stmt, nullptr) == SQLITE_OK &&
        sqlite3_step(count_stmt) == SQLITE_ROW) {
        memory_count = sqlite3_column_int(count_stmt, 0);
    }
    sqlite3_finalize(count_stmt);
    count_stmt = nullptr;
    if (sqlite3_prepare_v2(db_, "SELECT COUNT(*) FROM memories_fts", -1, &count_stmt, nullptr) == SQLITE_OK &&
        sqlite3_step(count_stmt) == SQLITE_ROW) {
        fts_count = sqlite3_column_int(count_stmt, 0);
    }
    sqlite3_finalize(count_stmt);
    if (memory_count != fts_count) {
        const char* backfill = R"(
            INSERT INTO memories_fts(memories_fts) VALUES ('delete-all');
            INSERT INTO memories_fts(rowid, user_text, assist_text)
            SELECT id, user_text, assist_text FROM memories;
        )";
        err = nullptr;
        sqlite3_exec(db_, backfill, nullptr, nullptr, &err);
        if (err) {
            std::string msg = err;
            sqlite3_free(err);
            throw std::runtime_error("FTS backfill failed: " + msg);
        }
    }
}

void MemoryStore::store_memory_dates(int64_t memory_id, const std::vector<int64_t>& days) {
    std::lock_guard<std::recursive_mutex> db_lock(db_mutex_);
    const char* sql = "INSERT OR IGNORE INTO memory_dates (memory_id, day) VALUES (?, ?)";
    sqlite3_stmt* stmt = nullptr;
    if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK) return;
    for (const int64_t day : days) {
        sqlite3_bind_int64(stmt, 1, memory_id);
        sqlite3_bind_int64(stmt, 2, day);
        sqlite3_step(stmt);
        sqlite3_reset(stmt);
    }
    sqlite3_finalize(stmt);
}

void MemoryStore::backfill_memory_dates() {
    std::lock_guard<std::recursive_mutex> lock(db_mutex_);
    const char* sql = R"(
        SELECT m.id, m.created_at, m.user_text, m.assist_text
        FROM memories m
        WHERE NOT EXISTS (SELECT 1 FROM memory_dates d WHERE d.memory_id = m.id)
    )";
    sqlite3_stmt* stmt = nullptr;
    if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK) return;
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        const int64_t id = sqlite3_column_int64(stmt, 0);
        const double created_at = sqlite3_column_double(stmt, 1);
        const char* user = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
        const char* assist = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3));
        std::vector<int64_t> days = extract_mentioned_days(user ? user : "", created_at);
        const auto assistant_days = extract_mentioned_days(assist ? assist : "", created_at);
        days.insert(days.end(), assistant_days.begin(), assistant_days.end());
        std::sort(days.begin(), days.end());
        days.erase(std::unique(days.begin(), days.end()), days.end());
        store_memory_dates(id, days);
    }
    sqlite3_finalize(stmt);
}

std::string normalize_fact_component(const std::string& value) {
    std::string output;
    bool space = true;
    for (unsigned char ch : value) {
        if (std::isspace(ch)) { space = true; continue; }
        if (space && !output.empty()) output += ' ';
        output += static_cast<char>(std::tolower(ch));
        space = false;
    }
    return output;
}

std::string make_fact_key(const std::string& subject, const std::string& attribute) {
    return normalize_fact_component(subject) + "|" + normalize_fact_component(attribute);
}

void MemoryStore::load_facts_cache() {
    std::lock_guard<std::recursive_mutex> lock(db_mutex_);
    const char* sql = "SELECT id, agent_id, subject, attribute, value, key, source_memory_id, created_at, superseded_by, embedding FROM facts ORDER BY id";
    sqlite3_stmt* stmt = nullptr;
    if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK) return;
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        Fact fact;
        fact.id = sqlite3_column_int64(stmt, 0);
        const auto text = [stmt](int col) { const char* v = reinterpret_cast<const char*>(sqlite3_column_text(stmt, col)); return std::string(v ? v : ""); };
        fact.agent_id = text(1); fact.subject = text(2); fact.attribute = text(3); fact.value = text(4); fact.key = text(5);
        fact.source_memory_id = sqlite3_column_type(stmt, 6) == SQLITE_NULL ? -1 : sqlite3_column_int64(stmt, 6);
        fact.created_at = sqlite3_column_double(stmt, 7);
        if (sqlite3_column_type(stmt, 8) != SQLITE_NULL) fact.superseded_by = sqlite3_column_int64(stmt, 8);
        const float* blob = reinterpret_cast<const float*>(sqlite3_column_blob(stmt, 9));
        const int size = sqlite3_column_bytes(stmt, 9) / static_cast<int>(sizeof(float));
        if (blob && size > 0) fact.embedding.assign(blob, blob + size);
        facts_cache_.push_back(std::move(fact));
    }
    sqlite3_finalize(stmt);
}

int64_t MemoryStore::insert_fact(const std::string& agent_id, const std::string& subject,
                                 const std::string& attribute, const std::string& value,
                                 int64_t source_memory_id, const std::vector<float>& embedding) {
    if (embedding.empty()) return -1;
    const std::string key = make_fact_key(subject, attribute);
    const std::string normalized_value = normalize_fact_component(value);
    std::lock_guard<std::recursive_mutex> lock(db_mutex_);
    int64_t old_id = -1;
    for (const auto& fact : facts_cache_) {
        if (!fact.superseded_by && fact.agent_id == agent_id && fact.key == key) {
            if (normalize_fact_component(fact.value) == normalized_value) return fact.id;
            old_id = fact.id;
            break;
        }
    }
    const char* sql = "INSERT INTO facts (agent_id, subject, attribute, value, key, source_memory_id, created_at, embedding) VALUES (?, ?, ?, ?, ?, ?, ?, ?)";
    sqlite3_stmt* stmt = nullptr;
    if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK) return -1;
    if (agent_id.empty()) sqlite3_bind_null(stmt, 1); else sqlite3_bind_text(stmt, 1, agent_id.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 2, subject.c_str(), -1, SQLITE_TRANSIENT); sqlite3_bind_text(stmt, 3, attribute.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 4, value.c_str(), -1, SQLITE_TRANSIENT); sqlite3_bind_text(stmt, 5, key.c_str(), -1, SQLITE_TRANSIENT);
    if (source_memory_id > 0) sqlite3_bind_int64(stmt, 6, source_memory_id); else sqlite3_bind_null(stmt, 6);
    const double created_at = now_unix();
    sqlite3_bind_double(stmt, 7, created_at); sqlite3_bind_blob(stmt, 8, embedding.data(), embedding.size() * sizeof(float), SQLITE_TRANSIENT);
    if (sqlite3_step(stmt) != SQLITE_DONE) { sqlite3_finalize(stmt); return -1; }
    const int64_t id = sqlite3_last_insert_rowid(db_); sqlite3_finalize(stmt);
    if (old_id > 0) {
        sqlite3_stmt* update = nullptr;
        if (sqlite3_prepare_v2(db_, "UPDATE facts SET superseded_by = ? WHERE id = ?", -1, &update, nullptr) == SQLITE_OK) {
            sqlite3_bind_int64(update, 1, id); sqlite3_bind_int64(update, 2, old_id); sqlite3_step(update); sqlite3_finalize(update);
        }
        for (auto& fact : facts_cache_) if (fact.id == old_id) fact.superseded_by = id;
    }
    facts_cache_.push_back({id, agent_id, subject, attribute, value, key, source_memory_id, created_at, std::nullopt, embedding});
    return id;
}

std::vector<ScoredFact> MemoryStore::search_facts(const std::vector<float>& query_emb, const std::string& agent_id, int top_k, float min_score) const {
    std::vector<float> q = query_emb; float norm = 0.0f;
    for (float v : q) norm += v * v; norm = std::sqrt(norm); if (norm > 0) for (float& v : q) v /= norm;
    std::vector<ScoredFact> results;
    std::lock_guard<std::recursive_mutex> lock(db_mutex_);
    for (const auto& fact : facts_cache_) {
        if (fact.superseded_by || fact.embedding.size() != q.size()) continue;
        if ((!agent_id.empty() && !fact.agent_id.empty() && fact.agent_id != agent_id) || (agent_id.empty() && !fact.agent_id.empty())) continue;
        const float score = dot_product(q.data(), fact.embedding.data(), static_cast<int>(q.size()));
        if (score >= min_score) results.push_back({fact, score});
    }
    std::sort(results.begin(), results.end(), [](const auto& a, const auto& b) { return a.similarity > b.similarity; });
    if (static_cast<int>(results.size()) > top_k) results.resize(top_k);
    return results;
}

std::vector<Fact> MemoryStore::list_facts(const std::string& agent_id, bool include_superseded, int limit) const {
    std::lock_guard<std::recursive_mutex> lock(db_mutex_); std::vector<Fact> results;
    for (auto it = facts_cache_.rbegin(); it != facts_cache_.rend() && static_cast<int>(results.size()) < limit; ++it) {
        if (!agent_id.empty() && it->agent_id != agent_id) continue;
        if (!include_superseded && it->superseded_by) continue;
        results.push_back(*it);
    }
    return results;
}

bool MemoryStore::remove_fact(int64_t fact_id) {
    std::lock_guard<std::recursive_mutex> lock(db_mutex_); sqlite3_stmt* stmt = nullptr;
    if (sqlite3_prepare_v2(db_, "DELETE FROM facts WHERE id = ?", -1, &stmt, nullptr) != SQLITE_OK) return false;
    sqlite3_bind_int64(stmt, 1, fact_id); sqlite3_step(stmt); const bool removed = sqlite3_changes(db_) > 0; sqlite3_finalize(stmt);
    if (removed) facts_cache_.erase(std::remove_if(facts_cache_.begin(), facts_cache_.end(), [fact_id](const Fact& fact) { return fact.id == fact_id; }), facts_cache_.end());
    return removed;
}

void MemoryStore::load_cache() {
    std::lock_guard<std::recursive_mutex> lock(db_mutex_);
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

        sqlite3_stmt* dates = nullptr;
        if (sqlite3_prepare_v2(db_, "SELECT day FROM memory_dates WHERE memory_id = ?", -1,
                               &dates, nullptr) == SQLITE_OK) {
            sqlite3_bind_int64(dates, 1, entry.id);
            while (sqlite3_step(dates) == SQLITE_ROW) {
                entry.mentioned_days.push_back(sqlite3_column_int64(dates, 0));
            }
            sqlite3_finalize(dates);
        }

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
                                const std::vector<float>& user_emb, const std::vector<float>& assist_emb,
                                std::vector<int64_t> mentioned_days) {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    if (emb_dim_ == 0 && !user_emb.empty()) emb_dim_ = static_cast<int>(user_emb.size());

    CachedEntry entry;
    entry.id = id;
    entry.agent_id = agent_id;
    entry.created_at = created_at;
    entry.user_text = user_text;
    entry.assist_text = assist_text;
    entry.mentioned_days = std::move(mentioned_days);
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
                             const std::vector<float>& assist_emb,
                             std::optional<double> created_at) {
    std::lock_guard<std::recursive_mutex> lock(db_mutex_);
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

    const double ts = created_at.value_or(now_unix());
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

    std::vector<int64_t> days = extract_mentioned_days(user_text, ts);
    const auto assistant_days = extract_mentioned_days(assist_text, ts);
    days.insert(days.end(), assistant_days.begin(), assistant_days.end());
    std::sort(days.begin(), days.end());
    days.erase(std::unique(days.begin(), days.end()), days.end());
    store_memory_dates(id, days);
    add_to_cache(id, agent_id, ts, user_text, assist_text, 1, user_emb, assist_emb,
                 std::move(days));

    return id;
}

int MemoryStore::count(const std::string& agent_id) const {
    std::lock_guard<std::recursive_mutex> lock(db_mutex_);
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
    std::lock_guard<std::recursive_mutex> db_lock(db_mutex_);
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
    std::lock_guard<std::recursive_mutex> db_lock(db_mutex_);
    int limit = agent_id.empty() ? max_global_ : max_per_agent_;
    int current = count(agent_id);
    if (current <= limit) return;

    int to_delete = current - limit;

    // LRU-weighted eviction: score = updated_at + 86400 * access_count
    // Low-access, stale memories are evicted first; frequently-touched ones survive longer.
    const char* id_sql = agent_id.empty()
        ? "SELECT id FROM memories WHERE agent_id IS NULL"
          " ORDER BY (updated_at + 86400.0 * CAST(access_count AS REAL)) ASC LIMIT ?"
        : "SELECT id FROM memories WHERE agent_id = ?"
          " ORDER BY (updated_at + 86400.0 * CAST(access_count AS REAL)) ASC LIMIT ?";

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

    sqlite3_stmt* dates_stmt = nullptr;
    if (sqlite3_prepare_v2(db_, "DELETE FROM memory_dates WHERE memory_id = ?", -1,
                           &dates_stmt, nullptr) == SQLITE_OK) {
        for (const int64_t id : ids_to_remove) {
            sqlite3_bind_int64(dates_stmt, 1, id);
            sqlite3_step(dates_stmt);
            sqlite3_reset(dates_stmt);
        }
        sqlite3_finalize(dates_stmt);
    }
    sqlite3_stmt* stmt = nullptr;
    if (sqlite3_prepare_v2(db_, "DELETE FROM memories WHERE id = ?", -1, &stmt, nullptr) !=
        SQLITE_OK) {
        std::cerr << "[store] evict delete prepare failed: " << sqlite3_errmsg(db_) << "\n";
        return;
    }
    for (const int64_t id : ids_to_remove) {
        sqlite3_bind_int64(stmt, 1, id);
        if (sqlite3_step(stmt) != SQLITE_DONE) {
            std::cerr << "[store] evict delete step failed: " << sqlite3_errmsg(db_) << "\n";
        }
        sqlite3_reset(stmt);
    }
    sqlite3_finalize(stmt);

    sqlite3_stmt* facts_stmt = nullptr;
    if (sqlite3_prepare_v2(db_, "DELETE FROM facts WHERE source_memory_id = ?", -1,
                           &facts_stmt, nullptr) == SQLITE_OK) {
        for (const int64_t id : ids_to_remove) {
            sqlite3_bind_int64(facts_stmt, 1, id);
            sqlite3_step(facts_stmt);
            sqlite3_reset(facts_stmt);
        }
        sqlite3_finalize(facts_stmt);
        facts_cache_.erase(std::remove_if(facts_cache_.begin(), facts_cache_.end(),
            [&ids_to_remove](const Fact& fact) {
                return std::find(ids_to_remove.begin(), ids_to_remove.end(),
                                 fact.source_memory_id) != ids_to_remove.end();
            }), facts_cache_.end());
    }

    // Remove from cache and trigger lazy recompaction
    std::lock_guard<std::mutex> lock(cache_mutex_);
    cache_.erase(
        std::remove_if(cache_.begin(), cache_.end(),
            [&ids_to_remove](const CachedEntry& e) {
                return std::find(ids_to_remove.begin(), ids_to_remove.end(), e.id) != ids_to_remove.end();
            }),
        cache_.end());
    holes_count_ += static_cast<int>(ids_to_remove.size());
    if (should_recompact()) {
        recompact_embeddings();
        holes_count_ = 0;
    }
}

// Stored embeddings are L2-normalized by EmbeddingWorker. Cosine similarity on
// unit vectors equals dot product, eliminating norm computations per comparison.
// Callers must normalize the query before use (see search() / find_duplicate()).
static float dot_product(const float* a, const float* b, int dim) {
#ifdef __APPLE__
    float result = 0.0f;
    vDSP_dotpr(a, 1, b, 1, &result, static_cast<vDSP_Length>(dim));
    return result;
#else
    float result = 0.0f;
    for (int i = 0; i < dim; i++) result += a[i] * b[i];
    return result;
#endif
}

std::vector<ScoredMemory> MemoryStore::search(const std::vector<float>& query_emb,
                                               const std::string& agent_id,
                                               int top_k,
                                               float min_score,
                                               int decay_days,
                                                float agent_boost,
                                                DecayMode decay_mode,
                                                std::optional<TemporalWindow> temporal_window,
                                                float temporal_boost,
                                                std::optional<double> supplied_now) const {
    std::lock_guard<std::mutex> lock(cache_mutex_);

    // Normalize the query once so dot_product == cosine similarity against unit stored vectors.
    std::vector<float> q = query_emb;
    {
        float norm = 0.0f;
        for (float v : q) norm += v * v;
        norm = std::sqrt(norm);
        if (norm > 0.0f) for (float& v : q) v /= norm;
    }

    const double now = supplied_now.value_or(now_unix());
    int dim = static_cast<int>(q.size());

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

        float cos_user = dot_product(q.data(), u_ptr, dim);
        float cos_assist = dot_product(q.data(), a_ptr, dim);
        float cos_max = std::max(cos_user, cos_assist);

        const double age_hours = (now - entry.created_at) / 3600.0;
        const bool same_agent = !entry.agent_id.empty() && entry.agent_id == agent_id;
        const float threshold_score = similarity_score(cos_max, same_agent, agent_boost);
        const float score = rank_score(cos_max, age_hours, decay_days, same_agent,
                                        agent_boost, decay_mode);

        // Keep threshold semantics on the cosine scale; tiebreak recency must
        // not make an otherwise below-threshold memory eligible for injection.
        if (threshold_score >= min_score) {
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
            bool temporal_match = false;
            if (temporal_window) {
                const int64_t created_day = utc_day(entry.created_at);
                temporal_match = created_day >= temporal_window->start_day &&
                                 created_day < temporal_window->end_day;
                if (!temporal_match) {
                    temporal_match = std::any_of(entry.mentioned_days.begin(),
                                                 entry.mentioned_days.end(),
                                                 [&temporal_window](int64_t day) {
                        return day >= temporal_window->start_day &&
                               day < temporal_window->end_day;
                    });
                }
            }
            results.push_back({std::move(mem), score + (temporal_match ? temporal_boost : 0.0f),
                               cos_max});
        }
    }

    std::sort(results.begin(), results.end(),
              [](const ScoredMemory& a, const ScoredMemory& b) { return a.score > b.score; });

    if (static_cast<int>(results.size()) > top_k) {
        results.resize(top_k);
    }

    return results;
}

HybridSearchResult MemoryStore::search_hybrid(const std::vector<float>& query_emb,
                                              const std::string& user_query,
                                              const std::string& agent_id,
                                              int top_k,
                                              float min_score,
                                              int decay_days,
                                               float agent_boost,
                                               int rrf_k,
                                               DecayMode decay_mode,
                                               std::optional<TemporalWindow> temporal_window,
                                               float temporal_boost,
                                               std::optional<double> supplied_now) const {
    const int candidate_count = std::max(50, 4 * top_k);
    auto dense = search(query_emb, agent_id, candidate_count, min_score, decay_days,
                        agent_boost, decay_mode, temporal_window, temporal_boost,
                        supplied_now);
    HybridSearchResult result;
    if (!dense.empty()) result.dense_top_similarity = dense.front().similarity;

    const std::string match_query = sanitize_fts_query(user_query);
    if (match_query.empty()) {
        if (static_cast<int>(dense.size()) > top_k) dense.resize(top_k);
        result.memories = std::move(dense);
        return result;
    }

    std::vector<int64_t> lexical_ids;
    const char* lexical_sql = R"(
        SELECT m.id FROM memories_fts
        JOIN memories AS m ON m.id = memories_fts.rowid
        WHERE memories_fts MATCH ?
          AND ((? = '' AND m.agent_id IS NULL)
               OR (? != '' AND (m.agent_id IS NULL OR m.agent_id = ?)))
        ORDER BY bm25(memories_fts)
        LIMIT ?
    )";
    std::lock_guard<std::recursive_mutex> db_lock(db_mutex_);
    sqlite3_stmt* stmt = nullptr;
    if (sqlite3_prepare_v2(db_, lexical_sql, -1, &stmt, nullptr) == SQLITE_OK) {
        sqlite3_bind_text(stmt, 1, match_query.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 2, agent_id.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 3, agent_id.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 4, agent_id.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_int(stmt, 5, candidate_count);
        while (sqlite3_step(stmt) == SQLITE_ROW) {
            lexical_ids.push_back(sqlite3_column_int64(stmt, 0));
        }
    }
    sqlite3_finalize(stmt);

    std::unordered_map<int64_t, ScoredMemory> by_id;
    std::unordered_map<int64_t, float> fused_scores;
    for (size_t rank = 0; rank < dense.size(); ++rank) {
        const auto id = dense[rank].memory.id;
        by_id.emplace(id, dense[rank]);
        fused_scores[id] += 1.0f / static_cast<float>(rrf_k + rank + 1);
    }

    {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        for (size_t rank = 0; rank < lexical_ids.size(); ++rank) {
            const auto id = lexical_ids[rank];
            fused_scores[id] += 1.0f / static_cast<float>(rrf_k + rank + 1);
            if (by_id.find(id) != by_id.end()) continue;
            const auto found = std::find_if(cache_.begin(), cache_.end(),
                                            [id](const CachedEntry& entry) { return entry.id == id; });
            if (found == cache_.end()) continue;
            const int dim = found->assist_emb_offset - found->user_emb_offset;
            const float* user_ptr = embeddings_.data() + found->user_emb_offset;
            const float* assist_ptr = embeddings_.data() + found->assist_emb_offset;
            Memory memory{found->id, found->agent_id, found->created_at, found->created_at,
                          found->user_text, found->assist_text,
                          std::vector<float>(user_ptr, user_ptr + dim),
                          std::vector<float>(assist_ptr, assist_ptr + dim), found->access_count};
            by_id.emplace(id, ScoredMemory{std::move(memory), 0.0f, 0.0f});
        }
    }

    result.memories.reserve(fused_scores.size());
    for (const auto& [id, score] : fused_scores) {
        auto found = by_id.find(id);
        if (found == by_id.end()) continue;
        found->second.score = score;
        result.memories.push_back(std::move(found->second));
    }
    std::sort(result.memories.begin(), result.memories.end(),
              [](const ScoredMemory& left, const ScoredMemory& right) {
                  return left.score > right.score;
              });
    if (static_cast<int>(result.memories.size()) > top_k) result.memories.resize(top_k);
    return result;
}

int64_t MemoryStore::find_duplicate(const std::vector<float>& user_emb, float threshold, const std::string& agent_id) const {
    std::lock_guard<std::mutex> lock(cache_mutex_);

    // Normalize query once for dot-product similarity against unit stored vectors.
    std::vector<float> q = user_emb;
    {
        float norm = 0.0f;
        for (float v : q) norm += v * v;
        norm = std::sqrt(norm);
        if (norm > 0.0f) for (float& v : q) v /= norm;
    }
    int dim = static_cast<int>(q.size());
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

        float cos = dot_product(q.data(), u_ptr, dim);
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
    std::lock_guard<std::recursive_mutex> db_lock(db_mutex_);
    sqlite3_stmt* dates_stmt = nullptr;
    if (sqlite3_prepare_v2(db_, "DELETE FROM memory_dates WHERE memory_id = ?", -1,
                           &dates_stmt, nullptr) == SQLITE_OK) {
        sqlite3_bind_int64(dates_stmt, 1, memory_id);
        sqlite3_step(dates_stmt);
        sqlite3_finalize(dates_stmt);
    }
    sqlite3_stmt* facts_stmt = nullptr;
    if (sqlite3_prepare_v2(db_, "DELETE FROM facts WHERE source_memory_id = ?", -1,
                           &facts_stmt, nullptr) == SQLITE_OK) {
        sqlite3_bind_int64(facts_stmt, 1, memory_id);
        sqlite3_step(facts_stmt);
        sqlite3_finalize(facts_stmt);
        facts_cache_.erase(std::remove_if(facts_cache_.begin(), facts_cache_.end(),
            [memory_id](const Fact& fact) { return fact.source_memory_id == memory_id; }),
            facts_cache_.end());
    }
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
        holes_count_++;
        if (should_recompact()) {
            recompact_embeddings();
            holes_count_ = 0;
        }
    }
    return changes > 0;
}

void MemoryStore::store_sentences(int64_t memory_id, char side,
                                  const std::vector<MemorySentence>& sentences) {
    if (sentences.empty() || (side != 'u' && side != 'a')) return;
    std::lock_guard<std::recursive_mutex> lock(db_mutex_);
    const char* sql = R"(
        INSERT OR REPLACE INTO memory_sentences
            (memory_id, side, idx, start_byte, end_byte, embedding)
        VALUES (?, ?, ?, ?, ?, ?)
    )";
    sqlite3_stmt* stmt = nullptr;
    if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK) return;
    for (const auto& sentence : sentences) {
        if (sentence.embedding.empty()) continue;
        sqlite3_bind_int64(stmt, 1, memory_id);
        sqlite3_bind_text(stmt, 2, &side, 1, SQLITE_TRANSIENT);
        sqlite3_bind_int(stmt, 3, sentence.idx);
        sqlite3_bind_int(stmt, 4, sentence.start_byte);
        sqlite3_bind_int(stmt, 5, sentence.end_byte);
        sqlite3_bind_blob(stmt, 6, sentence.embedding.data(),
                          static_cast<int>(sentence.embedding.size() * sizeof(float)),
                          SQLITE_TRANSIENT);
        if (sqlite3_step(stmt) != SQLITE_DONE) {
            std::cerr << "[store] store_sentences step failed: " << sqlite3_errmsg(db_) << "\n";
        }
        sqlite3_reset(stmt);
        sqlite3_clear_bindings(stmt);
    }
    sqlite3_finalize(stmt);
}

void MemoryStore::load_sentences(std::vector<ScoredMemory>& memories) const {
    if (memories.empty()) return;
    std::lock_guard<std::recursive_mutex> lock(db_mutex_);
    const char* sql = R"(
        SELECT side, idx, start_byte, end_byte, embedding
        FROM memory_sentences WHERE memory_id = ? ORDER BY side, idx
    )";
    sqlite3_stmt* stmt = nullptr;
    if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK) return;
    for (auto& scored : memories) {
        sqlite3_bind_int64(stmt, 1, scored.memory.id);
        while (sqlite3_step(stmt) == SQLITE_ROW) {
            MemorySentence sentence;
            sentence.idx = sqlite3_column_int(stmt, 1);
            sentence.start_byte = sqlite3_column_int(stmt, 2);
            sentence.end_byte = sqlite3_column_int(stmt, 3);
            const auto* blob = static_cast<const float*>(sqlite3_column_blob(stmt, 4));
            const int bytes = sqlite3_column_bytes(stmt, 4);
            if (blob && bytes > 0) {
                sentence.embedding.assign(blob, blob + bytes / static_cast<int>(sizeof(float)));
                (sqlite3_column_text(stmt, 0)[0] == 'u' ? scored.memory.user_sentences
                                                         : scored.memory.assist_sentences)
                    .push_back(std::move(sentence));
            }
        }
        sqlite3_reset(stmt);
        sqlite3_clear_bindings(stmt);
    }
    sqlite3_finalize(stmt);
}

int MemoryStore::sentence_count(int64_t memory_id) const {
    std::lock_guard<std::recursive_mutex> lock(db_mutex_);
    sqlite3_stmt* stmt = nullptr;
    if (sqlite3_prepare_v2(db_, "SELECT COUNT(*) FROM memory_sentences WHERE memory_id = ?",
                           -1, &stmt, nullptr) != SQLITE_OK) return 0;
    sqlite3_bind_int64(stmt, 1, memory_id);
    int result = sqlite3_step(stmt) == SQLITE_ROW ? sqlite3_column_int(stmt, 0) : 0;
    sqlite3_finalize(stmt);
    return result;
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

bool MemoryStore::should_recompact() const {
    // Caller must hold cache_mutex_
    if (emb_dim_ == 0 || holes_count_ == 0) return false;
    // Trigger when orphaned data exceeds 4MB
    size_t orphaned_bytes = static_cast<size_t>(holes_count_) * emb_dim_ * 2 * sizeof(float);
    if (orphaned_bytes >= 4UL * 1024 * 1024) return true;
    // Or when fragmentation exceeds 25% of live entries
    if (!cache_.empty() && holes_count_ > static_cast<int>(cache_.size()) / 4) return true;
    return false;
}

void MemoryStore::recompact_embeddings() {    // Caller must hold cache_mutex_
    if (cache_.empty()) {
        embeddings_.clear();
        return;
    }

    std::vector<float> new_embeddings;
    new_embeddings.reserve(cache_.size() * emb_dim_ * 2);

    for (auto& entry : cache_) {
        const float* u_ptr = embeddings_.data() + entry.user_emb_offset;
        const float* a_ptr = embeddings_.data() + entry.assist_emb_offset;

        int new_user_offset = static_cast<int>(new_embeddings.size());
        new_embeddings.insert(new_embeddings.end(), u_ptr, u_ptr + emb_dim_);

        int new_assist_offset = static_cast<int>(new_embeddings.size());
        new_embeddings.insert(new_embeddings.end(), a_ptr, a_ptr + emb_dim_);

        entry.user_emb_offset = new_user_offset;
        entry.assist_emb_offset = new_assist_offset;
    }

    embeddings_ = std::move(new_embeddings);
}

void MemoryStore::purge_expired(int ttl_days) {
    if (ttl_days <= 0) return;
    std::lock_guard<std::recursive_mutex> db_lock(db_mutex_);

    const double cutoff = static_cast<double>(std::time(nullptr)) - ttl_days * 86400.0;

    // Collect IDs to remove first (avoid long-lock)
    std::vector<int64_t> to_delete;
    {
        const char* sql = "SELECT id FROM memories WHERE created_at < ?;";
        sqlite3_stmt* stmt = nullptr;
        if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) == SQLITE_OK) {
            sqlite3_bind_double(stmt, 1, cutoff);
            while (sqlite3_step(stmt) == SQLITE_ROW) {
                to_delete.push_back(sqlite3_column_int64(stmt, 0));
            }
            sqlite3_finalize(stmt);
        }
    }

    if (to_delete.empty()) return;

    // Delete from DB
    const char* del_sql = "DELETE FROM memories WHERE id = ?;";
    sqlite3_stmt* del_stmt = nullptr;
    if (sqlite3_prepare_v2(db_, del_sql, -1, &del_stmt, nullptr) == SQLITE_OK) {
        sqlite3_stmt* dates_stmt = nullptr;
        sqlite3_prepare_v2(db_, "DELETE FROM memory_dates WHERE memory_id = ?", -1,
                           &dates_stmt, nullptr);
        for (int64_t id : to_delete) {
            if (dates_stmt) {
                sqlite3_bind_int64(dates_stmt, 1, id);
                sqlite3_step(dates_stmt);
                sqlite3_reset(dates_stmt);
            }
            sqlite3_bind_int64(del_stmt, 1, id);
            sqlite3_step(del_stmt);
            sqlite3_reset(del_stmt);
        }
        sqlite3_stmt* facts_stmt = nullptr;
        if (sqlite3_prepare_v2(db_, "DELETE FROM facts WHERE source_memory_id = ?", -1,
                               &facts_stmt, nullptr) == SQLITE_OK) {
            for (const int64_t id : to_delete) {
                sqlite3_bind_int64(facts_stmt, 1, id);
                sqlite3_step(facts_stmt);
                sqlite3_reset(facts_stmt);
            }
            sqlite3_finalize(facts_stmt);
            facts_cache_.erase(std::remove_if(facts_cache_.begin(), facts_cache_.end(),
                [&to_delete](const Fact& fact) {
                    return std::find(to_delete.begin(), to_delete.end(), fact.source_memory_id) !=
                           to_delete.end();
                }), facts_cache_.end());
        }
        if (dates_stmt) sqlite3_finalize(dates_stmt);
        sqlite3_finalize(del_stmt);
    }

    // Invalidate RAM cache entries
    std::lock_guard<std::mutex> lock(cache_mutex_);
    for (auto& entry : cache_) {
        for (int64_t id : to_delete) {
            if (entry.id == id) {
                entry.id = -1;  // mark as hole
                ++holes_count_;
                break;
            }
        }
    }
}

} // namespace memorylayer
