#include "memorylayer/memory_store.h"
#include <cassert>
#include <iostream>
#include <cstdio>
#include <vector>

static std::vector<float> make_vec(std::initializer_list<float> vals, int pad_to = 768) {
    std::vector<float> v(vals);
    v.resize(pad_to, 0.0f);
    return v;
}

void test_search_returns_most_similar() {
    std::remove("test_cosine.sqlite");
    memorylayer::MemoryStore store("test_cosine.sqlite");

    auto emb_a = make_vec({1.0f, 0.0f, 0.0f});
    auto emb_b = make_vec({0.0f, 1.0f, 0.0f});
    auto emb_c = make_vec({0.7f, 0.7f, 0.0f});

    store.insert("", "question A", "answer A", emb_a, emb_a);
    store.insert("", "question B", "answer B", emb_b, emb_b);
    store.insert("", "question C", "answer C", emb_c, emb_c);

    auto query = make_vec({0.9f, 0.1f, 0.0f});
    auto results = store.search(query, "", 2, 0.0f, 9999, 1.0f);

    assert(results.size() == 2);
    assert(results[0].memory.user_text == "question A");
    assert(results[1].memory.user_text == "question C");
    assert(results[0].score > results[1].score);

    std::remove("test_cosine.sqlite");
    std::cout << "test_search_returns_most_similar PASSED\n";
}

void test_dual_embedding_matches_assist() {
    std::remove("test_dual.sqlite");
    memorylayer::MemoryStore store("test_dual.sqlite");

    auto user_emb = make_vec({1.0f, 0.0f, 0.0f});
    auto assist_emb = make_vec({0.0f, 1.0f, 0.0f});
    store.insert("", "unrelated question", "relevant answer", user_emb, assist_emb);

    auto query = make_vec({0.0f, 0.9f, 0.1f});
    auto results = store.search(query, "", 5, 0.0f, 9999, 1.0f);

    assert(results.size() == 1);
    assert(results[0].score > 0.5f);
    std::cout << "test_dual_embedding_matches_assist PASSED\n";

    std::remove("test_dual.sqlite");
}

void test_dedup_finds_similar() {
    std::remove("test_dedup.sqlite");
    memorylayer::MemoryStore store("test_dedup.sqlite");

    auto emb = make_vec({1.0f, 0.0f, 0.0f});
    store.insert("", "hello", "world", emb, emb);

    auto similar = make_vec({0.99f, 0.01f, 0.0f});
    int64_t dup = store.find_duplicate(similar, 0.99f);
    assert(dup > 0);

    auto different = make_vec({0.0f, 1.0f, 0.0f});
    int64_t no_dup = store.find_duplicate(different, 0.99f);
    assert(no_dup == -1);

    std::remove("test_dedup.sqlite");
    std::cout << "test_dedup_finds_similar PASSED\n";
}

int main() {
    test_search_returns_most_similar();
    test_dual_embedding_matches_assist();
    test_dedup_finds_similar();
    std::cout << "All cosine tests PASSED\n";
    return 0;
}
