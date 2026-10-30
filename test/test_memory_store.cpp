#include "memorylayer/memory_store.h"
#include <cassert>
#include <iostream>
#include <cstdio>
#include <vector>
#include <sqlite3.h>

static std::vector<float> make_emb(float val) {
    return std::vector<float>(768, val);
}

void test_insert_and_count() {
    std::remove("test_store.sqlite");
    memorylayer::MemoryStore store("test_store.sqlite");

    assert(store.count("agent1") == 0);
    assert(store.count("") == 0);

    store.insert("agent1", "hello", "world", make_emb(0.1f), make_emb(0.2f));
    assert(store.count("agent1") == 1);
    assert(store.count("") == 0);

    store.insert("", "global q", "global a", make_emb(0.3f), make_emb(0.4f));
    assert(store.count("") == 1);

    store.insert("agent1", "second", "response", make_emb(0.5f), make_emb(0.6f));
    assert(store.count("agent1") == 2);

    std::remove("test_store.sqlite");
    std::cout << "test_insert_and_count PASSED\n";
}

void test_eviction() {
    std::remove("test_evict.sqlite");
    memorylayer::MemoryStore store("test_evict.sqlite", 3, 5000);

    for (int i = 0; i < 5; i++) {
        store.insert("agent1", "q" + std::to_string(i), "a" + std::to_string(i),
                     make_emb(0.1f * i), make_emb(0.2f * i));
    }
    assert(store.count("agent1") == 5);

    store.evict("agent1");
    assert(store.count("agent1") == 3);

    std::remove("test_evict.sqlite");
    std::cout << "test_eviction PASSED\n";
}

// LRU eviction: a memory touched many times must survive over a never-touched newer one.
void test_eviction_lru_preserves_hot_memories() {
    std::remove("test_evict_lru.sqlite");
    memorylayer::MemoryStore store("test_evict_lru.sqlite", 2, 5000);

    // Insert 3 memories: q0, q1 are old but q0 will be touched repeatedly
    int64_t id0 = store.insert("agent1", "hot memory", "a0", make_emb(0.1f), make_emb(0.2f));
    store.insert("agent1", "cold memory 1", "a1", make_emb(0.3f), make_emb(0.4f));
    store.insert("agent1", "cold memory 2", "a2", make_emb(0.5f), make_emb(0.6f));

    // Bump access_count of id0 many times — it should survive eviction
    for (int i = 0; i < 10; i++) store.touch(id0);

    // Evict down to limit=2: the two least-used should be removed (cold ones)
    store.evict("agent1");
    assert(store.count("agent1") == 2);

    // Verify that id0 (hot) survived
    auto memories = store.list_memories("agent1", 10, 0);
    bool hot_survived = false;
    for (const auto& m : memories) {
        if (m.id == id0) { hot_survived = true; break; }
    }
    assert(hot_survived && "Hot memory (high access_count) should survive LRU eviction");

    std::remove("test_evict_lru.sqlite");
    std::cout << "test_eviction_lru_preserves_hot_memories PASSED\n";
}

void test_temporal_boost_and_date_cleanup() {
    std::remove("test_temporal_store.sqlite");
    memorylayer::MemoryStore store("test_temporal_store.sqlite");
    const double now = 1709596800.0;  // 2024-03-05 UTC
    const std::vector<float> query = {1.0f, 0.0f};
    const std::vector<float> stronger = {1.0f, 0.0f};
    const std::vector<float> weaker = {0.8f, 0.6f};
    const int64_t old_id = store.insert("agent", "old", "no date", stronger, stronger,
                                        now - 20 * 86400.0);
    const int64_t dated_id = store.insert("agent", "new", "Planned for March 3, 2024.",
                                          weaker, weaker, now - 20 * 86400.0);

    const memorylayer::TemporalWindow window{memorylayer::utc_day(now) - 2,
                                               memorylayer::utc_day(now) - 1};
    const auto unboosted = store.search(query, "agent", 2, 0.3f, 30, 1.0f,
                                        memorylayer::DecayMode::Tiebreak, window, 0.0f, now);
    const auto boosted = store.search(query, "agent", 2, 0.3f, 30, 1.0f,
                                      memorylayer::DecayMode::Tiebreak, window, 0.3f, now);
    assert(unboosted.size() == 2 && unboosted[0].memory.id == old_id);
    assert(boosted.size() == 2 && boosted[0].memory.id == dated_id);
    // Boost ranks only threshold-eligible candidates; the 0.8 match stays excluded.
    const auto thresholded = store.search(query, "agent", 2, 0.9f, 30, 1.0f,
                                          memorylayer::DecayMode::Tiebreak, window, 1.0f, now);
    assert(thresholded.size() == 1 && thresholded[0].memory.id == old_id);

    assert(store.remove(dated_id));
    sqlite3* db = nullptr;
    assert(sqlite3_open("test_temporal_store.sqlite", &db) == SQLITE_OK);
    sqlite3_stmt* stmt = nullptr;
    assert(sqlite3_prepare_v2(db, "SELECT COUNT(*) FROM memory_dates WHERE memory_id = ?", -1,
                              &stmt, nullptr) == SQLITE_OK);
    sqlite3_bind_int64(stmt, 1, dated_id);
    assert(sqlite3_step(stmt) == SQLITE_ROW && sqlite3_column_int(stmt, 0) == 0);
    sqlite3_finalize(stmt);
    sqlite3_close(db);
    std::remove("test_temporal_store.sqlite");
    std::cout << "test_temporal_boost_and_date_cleanup PASSED\n";
}

void test_sentence_rows_delete_with_memory() {
    std::remove("test_sentence_delete.sqlite");
    memorylayer::MemoryStore store("test_sentence_delete.sqlite");
    const int64_t id = store.insert("agent", "question text long enough", "answer text long enough",
                                    make_emb(0.1f), make_emb(0.2f));
    store.store_sentences(id, 'u', {{0, 0, 8, {1.0f, 0.0f}}});
    store.store_sentences(id, 'a', {{0, 0, 6, {0.0f, 1.0f}}});
    assert(store.sentence_count(id) == 2);
    assert(store.remove(id));
    assert(store.sentence_count(id) == 0);
    std::remove("test_sentence_delete.sqlite");
    std::cout << "test_sentence_rows_delete_with_memory PASSED\n";
}

void test_sentence_rows_delete_with_eviction() {
    std::remove("test_sentence_evict.sqlite");
    memorylayer::MemoryStore store("test_sentence_evict.sqlite", 1, 5000);
    const int64_t first = store.insert("agent", "first question", "first answer",
                                       make_emb(0.1f), make_emb(0.2f));
    const int64_t second = store.insert("agent", "second question", "second answer",
                                        make_emb(0.3f), make_emb(0.4f));
    store.store_sentences(first, 'u', {{0, 0, 5, {1.0f}}});
    store.store_sentences(second, 'u', {{0, 0, 6, {1.0f}}});
    store.evict("agent");
    assert(store.count("agent") == 1);
    assert((store.sentence_count(first) == 0) != (store.sentence_count(second) == 0));
    std::remove("test_sentence_evict.sqlite");
    std::cout << "test_sentence_rows_delete_with_eviction PASSED\n";
}

int main() {
    test_insert_and_count();
    test_eviction();
    test_eviction_lru_preserves_hot_memories();
    test_temporal_boost_and_date_cleanup();
    test_sentence_rows_delete_with_memory();
    test_sentence_rows_delete_with_eviction();
    test_temporal_boost_and_date_cleanup();
    std::cout << "All memory_store tests PASSED\n";
    return 0;
}
