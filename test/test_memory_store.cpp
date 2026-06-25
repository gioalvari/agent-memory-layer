#include "memorylayer/memory_store.h"
#include <cassert>
#include <iostream>
#include <cstdio>
#include <vector>

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

int main() {
    test_insert_and_count();
    test_eviction();
    test_eviction_lru_preserves_hot_memories();
    std::cout << "All memory_store tests PASSED\n";
    return 0;
}
