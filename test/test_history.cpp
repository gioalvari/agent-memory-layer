#include "memorylayer/history.h"

#include <cassert>
#include <iostream>
#include <string>

using json = nlohmann::json;

namespace {

json message(const std::string& role, const std::string& content) {
    return {{"role", role}, {"content", content}};
}

json conversation() {
    return json::array({
        message("system", "System instructions"),
        message("user", std::string(80, 'a')),
        message("assistant", std::string(80, 'b')),
        message("user", std::string(80, 'c')),
        message("assistant", std::string(80, 'd')),
        message("user", "latest")
    });
}

void test_budget_off_preserves_messages() {
    const json messages = conversation();
    const auto result = memorylayer::trim_history(messages, 0, 0.5);
    assert(!result.trimmed);
    assert(result.messages == messages);
    std::cout << "test_budget_off_preserves_messages PASSED\n";
}

void test_under_budget_preserves_messages() {
    const json messages = conversation();
    const int budget = memorylayer::estimate_non_system_message_tokens(messages);
    const auto result = memorylayer::trim_history(messages, budget, 0.5);
    assert(!result.trimmed);
    assert(result.messages == messages);
    std::cout << "test_under_budget_preserves_messages PASSED\n";
}

void test_over_budget_trims_to_hysteresis_target() {
    const auto result = memorylayer::trim_history(conversation(), 80, 0.5);
    assert(result.trimmed);
    assert(result.dropped_messages == 4);
    assert(result.messages.size() == 2);
    assert(result.messages[0]["role"] == "system");
    assert(result.messages[1]["role"] == "user");
    assert(result.messages[1]["content"] == "latest");
    assert(memorylayer::estimate_non_system_message_tokens(result.messages) <= 40);
    std::cout << "test_over_budget_trims_to_hysteresis_target PASSED\n";
}

void test_tool_call_and_tool_messages_stay_together() {
    json messages = json::array({
        message("user", "old request"),
        {{"role", "assistant"}, {"content", nullptr},
         {"tool_calls", json::array({{{"id", "call-1"}}})}},
        {{"role", "tool"}, {"tool_call_id", "call-1"}, {"content", "old result"}},
        message("user", "latest request")
    });

    const auto result = memorylayer::trim_history(messages, 1, 0.5);
    assert(result.trimmed);
    assert(result.dropped_messages == 3);
    assert(result.messages.size() == 1);
    assert(result.messages[0]["role"] == "user");
    assert(result.messages[0]["content"] == "latest request");
    std::cout << "test_tool_call_and_tool_messages_stay_together PASSED\n";
}

void test_last_user_is_kept_when_over_budget() {
    json messages = json::array({
        message("user", "old request"),
        message("assistant", "old answer"),
        message("user", std::string(1000, 'z'))
    });
    const auto result = memorylayer::trim_history(messages, 1, 0.5);
    assert(result.trimmed);
    assert(result.messages.size() == 1);
    assert(result.messages[0]["role"] == "user");
    assert(result.messages[0]["content"] == std::string(1000, 'z'));
    std::cout << "test_last_user_is_kept_when_over_budget PASSED\n";
}

void test_trimming_is_deterministic() {
    const json messages = conversation();
    const auto first = memorylayer::trim_history(messages, 80, 0.5);
    const auto second = memorylayer::trim_history(messages, 80, 0.5);
    assert(first.trimmed == second.trimmed);
    assert(first.dropped_messages == second.dropped_messages);
    assert(first.messages == second.messages);
    std::cout << "test_trimming_is_deterministic PASSED\n";
}

} // namespace

int main() {
    test_budget_off_preserves_messages();
    test_under_budget_preserves_messages();
    test_over_budget_trims_to_hysteresis_target();
    test_tool_call_and_tool_messages_stay_together();
    test_last_user_is_kept_when_over_budget();
    test_trimming_is_deterministic();
    std::cout << "All history tests PASSED (6 tests)\n";
    return 0;
}
