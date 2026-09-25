#include "memorylayer/sticky.h"

#include <cassert>
#include <iostream>
#include <string>
#include <unordered_set>

using json = nlohmann::json;

namespace {

memorylayer::ScoredMemory memory(int64_t id, double created_at = 1704067200.0) {
    memorylayer::ScoredMemory result;
    result.memory.id = id;
    result.memory.created_at = created_at;
    result.memory.user_text = "Question " + std::to_string(id);
    result.memory.assist_text = "Answer " + std::to_string(id);
    result.score = 1.0f - static_cast<float>(id) / 100.0f;
    return result;
}

json initial_messages() {
    return json::array({
        {{"role", "system"}, {"content", "System"}},
        {{"role", "user"}, {"content", "First question"}},
    });
}

void test_key_stability_and_sensitivity() {
    json messages = initial_messages();
    const uint64_t first = memorylayer::sticky_message_key("agent", messages, 1);
    messages.push_back({{"role", "assistant"}, {"content", "First answer"}});
    messages.push_back({{"role", "user"}, {"content", "Second question"}});
    assert(memorylayer::sticky_message_key("agent", messages, 1) == first);
    assert(memorylayer::sticky_message_key("other-agent", messages, 1) != first);
    messages[1]["content"] = "Changed question";
    assert(memorylayer::sticky_message_key("agent", messages, 1) != first);
    std::cout << "test_key_stability_and_sensitivity PASSED\n";
}

void test_lru_eviction() {
    memorylayer::StickyCache cache(2);
    cache.put(1, {"one", {1}});
    cache.put(2, {"two", {2}});
    assert(cache.get(1)->block == "one");
    cache.put(3, {"three", {3}});
    assert(cache.get(1).has_value());
    assert(!cache.get(2).has_value());
    assert(cache.get(3)->block == "three");
    assert(cache.size() == 2);
    std::cout << "test_lru_eviction PASSED\n";
}

void test_history_reinjection_preserves_prefix() {
    memorylayer::StickyCache cache(8);
    json turn_one = initial_messages();
    const uint64_t first_key = memorylayer::sticky_message_key("a0", turn_one, 1);
    const std::string block = "<memory context>\n- [2024-01-01] remembered\n</memory context>";
    memorylayer::inject_sticky_block(turn_one, 1, block);
    cache.put(first_key, {block, {42}});
    const std::string sent_turn_one_prefix = turn_one.dump();

    json turn_two = initial_messages();
    turn_two.push_back({{"role", "assistant"}, {"content", "First answer"}});
    turn_two.push_back({{"role", "user"}, {"content", "Second question"}});
    const auto result = memorylayer::apply_sticky_history(turn_two, "a0", cache);

    assert(result.reused_blocks == 1);
    assert(result.shown_memory_ids == std::unordered_set<int64_t>{42});
    json turn_two_prefix = json::array({turn_two[0], turn_two[1]});
    assert(turn_two_prefix.dump() == sent_turn_one_prefix);
    std::cout << "test_history_reinjection_preserves_prefix PASSED\n";
}

void test_excluded_ids_filtered() {
    const std::vector<memorylayer::ScoredMemory> memories = {
        memory(1), memory(2), memory(3), memory(4),
    };
    const std::unordered_set<int64_t> excluded = {1, 3};
    const auto filtered = memorylayer::filter_excluded_memories(memories, excluded, 2);
    assert(filtered.size() == 2);
    assert(filtered[0].memory.id == 2);
    assert(filtered[1].memory.id == 4);
    std::cout << "test_excluded_ids_filtered PASSED\n";
}

void test_deterministic_formatting() {
    const auto formatted = memorylayer::format_sticky_memory_context_budgeted(
        {memory(7)}, 2048);
    assert(!formatted.block.empty());
    assert(formatted.block.find("2024-01-01") != std::string::npos);
    assert(formatted.block.find("ago") == std::string::npos);
    assert(formatted.block.find("just now") == std::string::npos);
    assert(formatted.memory_ids == std::vector<int64_t>{7});
    std::cout << "test_deterministic_formatting PASSED\n";
}

} // namespace

int main() {
    test_key_stability_and_sensitivity();
    test_lru_eviction();
    test_history_reinjection_preserves_prefix();
    test_excluded_ids_filtered();
    test_deterministic_formatting();
    std::cout << "All sticky tests PASSED\n";
    return 0;
}
