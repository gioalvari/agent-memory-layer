#include "memorylayer/injector.h"
#include <cassert>
#include <iostream>
#include <ctime>

using json = nlohmann::json;

void test_format_memory_line() {
    memorylayer::ScoredMemory sm;
    sm.memory.user_text = "How do I sort a vector in C++?";
    sm.memory.assist_text = "Use std::sort with iterators. Example: std::sort(v.begin(), v.end())";
    sm.memory.created_at = static_cast<double>(std::time(nullptr)) - 7200;
    sm.score = 0.85f;

    double now = static_cast<double>(std::time(nullptr));
    auto line = memorylayer::format_memory_line(sm, now);

    assert(line.find("[2h ago]") != std::string::npos);
    assert(line.find("sort a vector") != std::string::npos);
    assert(line.find("std::sort") != std::string::npos);
    std::cout << "test_format_memory_line PASSED\n";
}

void test_format_memory_context_empty() {
    std::vector<memorylayer::ScoredMemory> empty;
    auto ctx = memorylayer::format_memory_context(empty, 0.0);
    assert(ctx.empty());
    std::cout << "test_format_memory_context_empty PASSED\n";
}

void test_inject_with_existing_system() {
    json messages = json::array({
        {{"role", "system"}, {"content", "You are a helpful assistant."}},
        {{"role", "user"}, {"content", "What is C++?"}}
    });

    std::string ctx = "<memory context>\nRelevant past interactions:\n- [1h ago] test memory\n</memory context>";
    memorylayer::inject_memories(messages, ctx);

    std::string sys = messages[0]["content"].get<std::string>();
    assert(sys.find("You are a helpful assistant.") != std::string::npos);
    assert(sys.find("<memory context>") != std::string::npos);
    std::cout << "test_inject_with_existing_system PASSED\n";
}

void test_inject_without_system() {
    json messages = json::array({
        {{"role", "user"}, {"content", "Hello"}}
    });

    std::string ctx = "<memory context>\ntest\n</memory context>";
    memorylayer::inject_memories(messages, ctx);

    assert(messages.size() == 2);
    assert(messages[0]["role"] == "system");
    assert(messages[0]["content"].get<std::string>().find("<memory context>") != std::string::npos);
    assert(messages[1]["role"] == "user");
    std::cout << "test_inject_without_system PASSED\n";
}

void test_no_injection_on_empty_context() {
    json messages = json::array({
        {{"role", "user"}, {"content", "Hello"}}
    });

    memorylayer::inject_memories(messages, "");
    assert(messages.size() == 1);
    std::cout << "test_no_injection_on_empty_context PASSED\n";
}

int main() {
    test_format_memory_line();
    test_format_memory_context_empty();
    test_inject_with_existing_system();
    test_inject_without_system();
    test_no_injection_on_empty_context();
    std::cout << "All injector tests PASSED\n";
    return 0;
}
