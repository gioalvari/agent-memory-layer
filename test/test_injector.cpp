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

void test_inject_suffix_keeps_prefix_stable() {
    json messages = json::array({
        {{"role", "system"}, {"content", "Shared prompt."}},
        {{"role", "user"}, {"content", "First"}},
        {{"role", "assistant"}, {"content", "Answer"}},
        {{"role", "user"}, {"content", "Second"}}
    });
    json original = messages;

    memorylayer::inject_memories(messages, "<memory context>\nm\n</memory context>",
                                 memorylayer::InjectMode::Suffix);

    assert(messages.size() == 4);
    for (int i = 0; i < 3; i++) assert(messages[i] == original[i]);
    std::string last = messages[3]["content"].get<std::string>();
    assert(last.rfind("<memory context>", 0) == 0);
    assert(last.size() >= 6 && last.substr(last.size() - 6) == "Second");
    std::cout << "test_inject_suffix_keeps_prefix_stable PASSED\n";
}

void test_inject_suffix_structured_content() {
    json messages = json::array({
        {{"role", "system"}, {"content", "S"}},
        {{"role", "user"}, {"content", json::array({{{"type", "text"}, {"text", "hi"}}})}}
    });
    memorylayer::inject_memories(messages, "<memory context>x</memory context>",
                                 memorylayer::InjectMode::Suffix);
    assert(messages.size() == 3);
    assert(messages[0]["content"] == "S");
    assert(messages[1]["role"] == "system");
    assert(messages[2]["role"] == "user");
    std::cout << "test_inject_suffix_structured_content PASSED\n";
}

void test_inject_suffix_without_user_falls_back() {
    json messages = json::array({{{"role", "system"}, {"content", "S"}}});
    memorylayer::inject_memories(messages, "<memory context>x</memory context>",
                                 memorylayer::InjectMode::Suffix);
    assert(messages.size() == 1);
    assert(messages[0]["content"].get<std::string>().find("<memory context>") != std::string::npos);
    std::cout << "test_inject_suffix_without_user_falls_back PASSED\n";
}

void test_parse_inject_mode() {
    assert(memorylayer::parse_inject_mode("suffix") == memorylayer::InjectMode::Suffix);
    assert(memorylayer::parse_inject_mode("system") == memorylayer::InjectMode::System);
    assert(memorylayer::parse_inject_mode("bogus") == memorylayer::InjectMode::System);
    std::cout << "test_parse_inject_mode PASSED\n";
}

int main() {
    test_inject_suffix_keeps_prefix_stable();
    test_inject_suffix_structured_content();
    test_inject_suffix_without_user_falls_back();
    test_parse_inject_mode();
    test_format_memory_line();
    test_format_memory_context_empty();
    test_inject_with_existing_system();
    test_inject_without_system();
    test_no_injection_on_empty_context();
    std::cout << "All injector tests PASSED\n";
    return 0;
}
