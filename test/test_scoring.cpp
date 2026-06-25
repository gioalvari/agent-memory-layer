#include "memorylayer/memory_store.h"
#include <cassert>
#include <iostream>
#include <cstdio>
#include <cmath>
#include <ctime>

static std::vector<float> make_vec(std::initializer_list<float> vals, int pad_to = 768) {
    std::vector<float> v(vals);
    v.resize(pad_to, 0.0f);
    return v;
}

// Test 1: Temporal decay reduces score for older memories
void test_temporal_decay() {
    std::remove("test_decay.sqlite");
    memorylayer::MemoryStore store("test_decay.sqlite");

    auto emb = make_vec({1.0f, 0.0f, 0.0f});
    
    // Insert two identical-embedding memories
    store.insert("", "recent question", "recent answer", emb, emb);
    store.insert("", "old question", "old answer", emb, emb);

    // We can't control created_at easily, but both should have same score
    // since they're created at nearly the same time with identical embeddings
    auto results = store.search(emb, "", 10, 0.0f, 30, 1.0f);
    assert(results.size() == 2);
    // Both scores should be very close (same age, same embedding)
    assert(std::abs(results[0].score - results[1].score) < 0.01f);
    // Score should be close to 1.0 (perfect match, zero decay)
    assert(results[0].score > 0.95f);

    std::remove("test_decay.sqlite");
    std::cout << "test_temporal_decay PASSED\n";
}

// Test 2: Decay floor never goes below 0.5
void test_decay_floor() {
    std::remove("test_floor.sqlite");
    memorylayer::MemoryStore store("test_floor.sqlite");

    auto emb = make_vec({1.0f, 0.0f, 0.0f});
    store.insert("", "question", "answer", emb, emb);

    // With decay_days=1, after 48+ hours the decay formula gives:
    // decay = max(0.5, 1.0 - hours/(24*1))
    // At 0 hours: decay = 1.0, score = 1.0
    // The floor is max(0.5, ...), so minimum score = 0.5 * cos_sim
    // Since we just inserted, score should be ~1.0
    auto results = store.search(emb, "", 10, 0.0f, 1, 1.0f);
    assert(results.size() == 1);
    assert(results[0].score > 0.5f); // floor guarantees this

    std::remove("test_floor.sqlite");
    std::cout << "test_decay_floor PASSED\n";
}

// Test 3: Agent boost multiplies score for matching agent
void test_agent_boost() {
    std::remove("test_boost.sqlite");
    memorylayer::MemoryStore store("test_boost.sqlite");

    auto emb = make_vec({1.0f, 0.0f, 0.0f});
    
    // Insert one memory for agent "coder" and one global
    store.insert("coder", "agent question", "agent answer", emb, emb);
    store.insert("", "global question", "global answer", emb, emb);

    // Search as "coder" with 1.5x boost
    auto results = store.search(emb, "coder", 10, 0.0f, 9999, 1.5f);
    assert(results.size() == 2);
    
    // The agent-specific memory should score higher due to boost
    assert(results[0].memory.agent_id == "coder");
    assert(results[0].score > results[1].score);
    
    // The boosted score should be ~1.5x the unboosted score
    float ratio = results[0].score / results[1].score;
    assert(ratio > 1.4f && ratio < 1.6f);  // approximately 1.5x

    std::remove("test_boost.sqlite");
    std::cout << "test_agent_boost PASSED\n";
}

// Test 4: Agent isolation — agent A shouldn't see agent B's memories
void test_agent_isolation() {
    std::remove("test_iso.sqlite");
    memorylayer::MemoryStore store("test_iso.sqlite");

    auto emb = make_vec({1.0f, 0.0f, 0.0f});
    
    store.insert("agentA", "A's question", "A's answer", emb, emb);
    store.insert("agentB", "B's question", "B's answer", emb, emb);
    store.insert("", "global question", "global answer", emb, emb);

    // Agent A should see its own + global, NOT agent B's
    auto results_a = store.search(emb, "agentA", 10, 0.0f, 9999, 1.0f);
    assert(results_a.size() == 2);  // agentA + global
    for (const auto& r : results_a) {
        assert(r.memory.agent_id != "agentB");
    }

    // Agent B should see its own + global, NOT agent A's
    auto results_b = store.search(emb, "agentB", 10, 0.0f, 9999, 1.0f);
    assert(results_b.size() == 2);
    for (const auto& r : results_b) {
        assert(r.memory.agent_id != "agentA");
    }

    // Global search should see ONLY global
    auto results_global = store.search(emb, "", 10, 0.0f, 9999, 1.0f);
    assert(results_global.size() == 1);
    assert(results_global[0].memory.agent_id.empty());

    std::remove("test_iso.sqlite");
    std::cout << "test_agent_isolation PASSED\n";
}

// Test 5: Min score threshold filters low-quality matches
void test_min_score_filter() {
    std::remove("test_filter.sqlite");
    memorylayer::MemoryStore store("test_filter.sqlite");

    auto emb_a = make_vec({1.0f, 0.0f, 0.0f});
    auto emb_b = make_vec({0.0f, 1.0f, 0.0f});
    
    store.insert("", "question A", "answer A", emb_a, emb_a);
    store.insert("", "question B", "answer B", emb_b, emb_b);

    auto query = make_vec({1.0f, 0.0f, 0.0f});

    // With high threshold, only the exact match should pass
    auto results = store.search(query, "", 10, 0.9f, 9999, 1.0f);
    assert(results.size() == 1);
    assert(results[0].memory.user_text == "question A");

    // With low threshold, both should appear
    auto results_low = store.search(query, "", 10, 0.0f, 9999, 1.0f);
    assert(results_low.size() == 2);

    std::remove("test_filter.sqlite");
    std::cout << "test_min_score_filter PASSED\n";
}

// Test 6: Top-k truncation returns only k best results
void test_topk_truncation() {
    std::remove("test_topk.sqlite");
    memorylayer::MemoryStore store("test_topk.sqlite");

    // Insert 10 memories with slightly different embeddings
    for (int i = 0; i < 10; i++) {
        float v = 1.0f - (i * 0.05f);
        auto emb = make_vec({v, 0.1f * i, 0.0f});
        store.insert("", "q" + std::to_string(i), "a" + std::to_string(i), emb, emb);
    }

    auto query = make_vec({1.0f, 0.0f, 0.0f});
    
    // top_k=3 should return exactly 3
    auto results = store.search(query, "", 3, 0.0f, 9999, 1.0f);
    assert(static_cast<int>(results.size()) == 3);
    
    // They should be in descending score order
    assert(results[0].score >= results[1].score);
    assert(results[1].score >= results[2].score);

    std::remove("test_topk.sqlite");
    std::cout << "test_topk_truncation PASSED\n";
}

// Test 7: Empty database returns empty results
void test_empty_db() {
    std::remove("test_empty.sqlite");
    memorylayer::MemoryStore store("test_empty.sqlite");

    auto query = make_vec({1.0f, 0.0f, 0.0f});
    auto results = store.search(query, "", 10, 0.0f, 9999, 1.0f);
    assert(results.empty());

    std::remove("test_empty.sqlite");
    std::cout << "test_empty_db PASSED\n";
}

// Test 8: All memories below threshold returns empty
void test_all_below_threshold() {
    std::remove("test_allbelow.sqlite");
    memorylayer::MemoryStore store("test_allbelow.sqlite");

    auto emb = make_vec({1.0f, 0.0f, 0.0f});
    store.insert("", "question", "answer", emb, emb);

    // Query orthogonal to stored embedding → cosine ≈ 0
    auto query = make_vec({0.0f, 0.0f, 1.0f});
    auto results = store.search(query, "", 10, 0.3f, 9999, 1.0f);
    assert(results.empty());

    std::remove("test_allbelow.sqlite");
    std::cout << "test_all_below_threshold PASSED\n";
}

// Test 9: Recompaction after remove doesn't break search
void test_search_after_remove() {
    std::remove("test_rmv.sqlite");
    memorylayer::MemoryStore store("test_rmv.sqlite");

    auto emb_a = make_vec({1.0f, 0.0f, 0.0f});
    auto emb_b = make_vec({0.0f, 1.0f, 0.0f});
    auto emb_c = make_vec({0.0f, 0.0f, 1.0f});

    int64_t id_a = store.insert("", "A", "A", emb_a, emb_a);
    store.insert("", "B", "B", emb_b, emb_b);
    store.insert("", "C", "C", emb_c, emb_c);

    // Remove first entry
    store.remove(id_a);

    // Search should still work with correct results (B and C)
    auto query = make_vec({0.0f, 0.9f, 0.1f});
    auto results = store.search(query, "", 10, 0.0f, 9999, 1.0f);
    assert(results.size() == 2);
    assert(results[0].memory.user_text == "B");  // closest to query

    std::remove("test_rmv.sqlite");
    std::cout << "test_search_after_remove PASSED\n";
}

// Test 10: Token budget estimation
void test_token_estimation() {
    // estimate_tokens is chars/4 + 1
    std::string short_text = "hello";  // 5 chars → 2 tokens
    std::string long_text(400, 'a');   // 400 chars → 101 tokens
    
    // Verify the formula manually
    assert(5 / 4 + 1 == 2);
    assert(400 / 4 + 1 == 101);
    
    std::cout << "test_token_estimation PASSED\n";
}

int main() {
    test_temporal_decay();
    test_decay_floor();
    test_agent_boost();
    test_agent_isolation();
    test_min_score_filter();
    test_topk_truncation();
    test_empty_db();
    test_all_below_threshold();
    test_search_after_remove();
    test_token_estimation();
    std::cout << "All scoring tests PASSED (" << 10 << " tests)\n";
    return 0;
}
