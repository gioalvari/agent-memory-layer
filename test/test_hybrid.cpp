#include "memorylayer/memory_store.h"
#include "sqlite3.h"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

namespace {

std::vector<float> make_embedding(std::initializer_list<float> values) {
    std::vector<float> embedding(values);
    embedding.resize(768, 0.0f);
    return embedding;
}

void remove_database(const char* path) {
    std::remove(path);
    std::remove((std::string(path) + "-wal").c_str());
    std::remove((std::string(path) + "-shm").c_str());
}

void test_sanitize_fts_query() {
    assert(memorylayer::sanitize_fts_query("ticket ZX-4471") ==
           "\"ticket\" OR \"ZX\" OR \"4471\"");
    assert(memorylayer::sanitize_fts_query("\" OR * NEAR(foo, bar)") ==
           "\"OR\" OR \"NEAR\" OR \"foo\" OR \"bar\"");
    assert(memorylayer::sanitize_fts_query("--- !!!") == "");
    assert(memorylayer::sanitize_fts_query("\xF0\x9F\xA7\xA0 quote\"") == "\"quote\"");
    std::cout << "test_sanitize_fts_query PASSED\n";
}

void test_fts_sync_on_insert_delete_evict_and_dedup_touch() {
    const char* path = "test_hybrid_sync.sqlite";
    remove_database(path);
    memorylayer::MemoryStore store(path, 1, 5000);
    const auto low = make_embedding({0.0f, 1.0f});
    const auto query = make_embedding({1.0f, 0.0f});

    const int64_t first = store.insert("agent", "ticket ZX-4471", "first answer", low, low);
    auto found = store.search_hybrid(query, "ZX-4471", "agent", 5, 0.9f, 30, 1.0f, 60);
    assert(found.memories.size() == 1);
    assert(found.memories.front().memory.id == first);
    assert(found.memories.front().similarity == 0.0f);  // lexical-only hit

    store.touch(first);  // Dedup merge currently updates metadata only; index remains valid.
    found = store.search_hybrid(query, "ZX-4471", "agent", 5, 0.9f, 30, 1.0f, 60);
    assert(found.memories.size() == 1);

    assert(store.remove(first));
    found = store.search_hybrid(query, "ZX-4471", "agent", 5, 0.9f, 30, 1.0f, 60);
    assert(found.memories.empty());

    const int64_t evicted = store.insert("agent", "ticket EVICT-1", "old", low, low);
    const int64_t kept = store.insert("agent", "ticket KEEP-2", "new", low, low);
    store.touch(kept);
    store.evict("agent");
    found = store.search_hybrid(query, "EVICT-1", "agent", 5, 0.9f, 30, 1.0f, 60);
    assert(found.memories.empty());
    found = store.search_hybrid(query, "KEEP-2", "agent", 5, 0.9f, 30, 1.0f, 60);
    assert(found.memories.size() == 1 && found.memories.front().memory.id == kept);
    (void)evicted;
    remove_database(path);
    std::cout << "test_fts_sync_on_insert_delete_evict_and_dedup_touch PASSED\n";
}

void test_fts_backfill_migration() {
    const char* path = "test_hybrid_backfill.sqlite";
    remove_database(path);
    sqlite3* db = nullptr;
    assert(sqlite3_open(path, &db) == SQLITE_OK);
    const char* schema = R"(
        CREATE TABLE memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT, agent_id TEXT, created_at REAL NOT NULL,
            updated_at REAL NOT NULL, user_text TEXT NOT NULL, assist_text TEXT NOT NULL,
            user_emb BLOB NOT NULL, assist_emb BLOB NOT NULL, access_count INTEGER DEFAULT 1
        );
    )";
    assert(sqlite3_exec(db, schema, nullptr, nullptr, nullptr) == SQLITE_OK);
    const auto emb = make_embedding({0.0f, 1.0f});
    sqlite3_stmt* stmt = nullptr;
    assert(sqlite3_prepare_v2(db,
        "INSERT INTO memories(agent_id, created_at, updated_at, user_text, assist_text, user_emb, assist_emb) "
        "VALUES(NULL, 0, 0, 'legacy ABC-123', 'answer', ?, ?)", -1, &stmt, nullptr) == SQLITE_OK);
    sqlite3_bind_blob(stmt, 1, emb.data(), static_cast<int>(emb.size() * sizeof(float)), SQLITE_TRANSIENT);
    sqlite3_bind_blob(stmt, 2, emb.data(), static_cast<int>(emb.size() * sizeof(float)), SQLITE_TRANSIENT);
    assert(sqlite3_step(stmt) == SQLITE_DONE);
    sqlite3_finalize(stmt);
    sqlite3_close(db);

    memorylayer::MemoryStore store(path);
    auto found = store.search_hybrid(emb, "ABC-123", "", 5, 0.9f, 30, 1.0f, 60);
    assert(found.memories.size() == 1);
    assert(found.memories.front().memory.user_text == "legacy ABC-123");
    remove_database(path);
    std::cout << "test_fts_backfill_migration PASSED\n";
}

void test_rrf_fusion_and_dense_only_compatibility() {
    const char* path = "test_hybrid_rrf.sqlite";
    remove_database(path);
    memorylayer::MemoryStore store(path);
    const auto dense = make_embedding({1.0f, 0.0f});
    const auto lexical = make_embedding({0.0f, 1.0f});
    store.insert("agent", "semantic result", "unrelated", dense, dense);
    const int64_t exact_id = store.insert("agent", "ticket ZX-4471", "exact", lexical, lexical);
    const int64_t shared = store.insert("agent", "ticket ZX-4471 supporting", "shared", dense, dense);

    const auto regular = store.search(dense, "agent", 3, 0.3f, 30, 1.0f);
    const auto legacy_again = store.search(dense, "agent", 3, 0.3f, 30, 1.0f);
    assert(regular.size() == legacy_again.size());
    for (size_t i = 0; i < regular.size(); ++i) {
        assert(regular[i].memory.id == legacy_again[i].memory.id);
        assert(regular[i].score == legacy_again[i].score);
    }

    const auto hybrid = store.search_hybrid(dense, "ticket ZX-4471", "agent", 3,
                                            0.3f, 30, 1.0f, 1);
    assert(hybrid.memories.size() == 3);
    assert(hybrid.memories.front().memory.id == shared);
    assert(hybrid.memories.front().score > hybrid.memories.back().score);
    const auto exact = std::find_if(hybrid.memories.begin(), hybrid.memories.end(),
                                    [exact_id](const memorylayer::ScoredMemory& memory) {
                                        return memory.memory.id == exact_id;
                                    });
    assert(exact != hybrid.memories.end());
    assert(exact->similarity == 0.0f);
    remove_database(path);
    std::cout << "test_rrf_fusion_and_dense_only_compatibility PASSED\n";
}

}  // namespace

int main() {
    test_sanitize_fts_query();
    test_fts_sync_on_insert_delete_evict_and_dedup_touch();
    test_fts_backfill_migration();
    test_rrf_fusion_and_dense_only_compatibility();
    std::cout << "All hybrid tests PASSED (4 tests)\n";
    return 0;
}
