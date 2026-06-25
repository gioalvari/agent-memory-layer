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

int main() {
    test_insert_and_count();
    test_eviction();
    std::cout << "All memory_store tests PASSED\n";
    return 0;
}
