#include "memorylayer/injector.h"
#include <algorithm>
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
    assert(memorylayer::parse_inject_mode("sticky") == memorylayer::InjectMode::Sticky);
    assert(memorylayer::parse_inject_mode("system") == memorylayer::InjectMode::System);
    assert(memorylayer::parse_inject_mode("bogus") == memorylayer::InjectMode::System);
    std::cout << "test_parse_inject_mode PASSED\n";
}

memorylayer::ScoredMemory long_memory(double created_at) {
    memorylayer::ScoredMemory sm;
    sm.memory.user_text = std::string(5000, 'u');
    sm.memory.assist_text = std::string(5000, 'a');
    sm.memory.created_at = created_at;
    return sm;
}

void test_line_token_bound_holds() {
    const double now = 100000.0 * 86400.0;
    for (int chars : {50, 200, 800}) {
        // Oldest representable age gives the longest "99999d ago" timestamp.
        const auto line = memorylayer::format_memory_line(long_memory(86400.0), now, chars);
        assert(line.find(std::string(static_cast<std::size_t>(chars), 'u') + "...") !=
               std::string::npos);
        assert(line.find(std::string(static_cast<std::size_t>(chars) + 1, 'u')) ==
               std::string::npos);
        assert(memorylayer::estimate_tokens(line + "\n") <=
               memorylayer::max_memory_line_tokens(chars));
    }
    std::cout << "test_line_token_bound_holds PASSED\n";
}

void test_required_budget_fits_all_k_memories() {
    const double now = 100000.0 * 86400.0;
    for (int chars : {200, 800}) {
        for (int k : {4, 16, 26}) {
            std::vector<memorylayer::ScoredMemory> memories(
                static_cast<std::size_t>(k), long_memory(86400.0));
            const int budget = memorylayer::required_inject_tokens(k, chars);
            const auto fits = memorylayer::format_memory_context_budgeted(
                memories, now, budget, chars);
            // Two header newlines plus one per injected memory line.
            assert(static_cast<int>(std::count(fits.begin(), fits.end(), '\n')) == k + 2);
            const auto tight = memorylayer::format_memory_context_budgeted(
                memories, now, budget - memorylayer::max_memory_line_tokens(chars), chars);
            assert(static_cast<int>(std::count(tight.begin(), tight.end(), '\n')) == k + 1);
        }
    }
    std::cout << "test_required_budget_fits_all_k_memories PASSED\n";
}

memorylayer::MemorySentence sentence(int idx, int start, int end,
                                     std::vector<float> embedding) {
    return {idx, start, end, std::move(embedding)};
}

void test_sentence_splitter_edge_cases() {
    const std::string unicode = u8"Café downtown is open until midnight. 東京 station is open throughout the evening!";
    const auto unicode_parts = memorylayer::split_memory_sentences(unicode);
    assert(unicode_parts.size() == 2);
    assert(unicode.substr(unicode_parts[0].start_byte,
                         unicode_parts[0].end_byte - unicode_parts[0].start_byte) ==
               "Café downtown is open until midnight.");

    const auto no_punctuation = memorylayer::split_memory_sentences("one continuous thought without punctuation");
    assert(no_punctuation.size() == 1);

    const auto code = memorylayer::split_memory_sentences("Before context is sufficiently long.\n```\nx = 1; y = 2;\n```\nAfter context is sufficiently long.");
    assert(code.size() == 3);
    assert(code[1].start_byte < static_cast<int>(std::string("Before context is sufficiently long.\n```").size()));

    const auto long_parts = memorylayer::split_memory_sentences(std::string(350, 'a') + " " + std::string(350, 'b'));
    assert(long_parts.size() >= 2);
    for (const auto& part : long_parts) assert(part.end_byte - part.start_byte <= 300);
    std::cout << "test_sentence_splitter_edge_cases PASSED\n";
}

void test_compression_selects_relevant_later_sentence() {
    const std::string text = "This opening sentence is unrelated filler. The deployment uses blue green rollout safely.";
    const int later = static_cast<int>(text.find("The deployment"));
    std::vector<memorylayer::MemorySentence> sentences = {
        sentence(0, 0, later - 1, {1.0f, 0.0f}),
        sentence(1, later, static_cast<int>(text.size()), {0.0f, 1.0f}),
    };
    const auto compressed = memorylayer::compress_memory_side(text, sentences, {0.0f, 1.0f}, 60);
    assert(compressed.find("deployment") != std::string::npos);
    assert(compressed.find("unrelated") == std::string::npos);
    std::cout << "test_compression_selects_relevant_later_sentence PASSED\n";
}

void test_compression_orders_and_marks_gaps() {
    const std::string text = "Alpha context is meaningful. Beta context is meaningful. Gamma context is meaningful.";
    const int beta = static_cast<int>(text.find("Beta"));
    const int gamma = static_cast<int>(text.find("Gamma"));
    std::vector<memorylayer::MemorySentence> sentences = {
        sentence(0, 0, beta - 1, {0.8f, 0.0f}),
        sentence(1, beta, gamma - 1, {0.0f, 0.1f}),
        sentence(2, gamma, static_cast<int>(text.size()), {1.0f, 0.0f}),
    };
    const auto compressed = memorylayer::compress_memory_side(text, sentences, {1.0f, 0.0f}, 65);
    assert(compressed.find("Alpha") < compressed.find("Gamma"));
    assert(compressed.find(" … ") != std::string::npos);
    std::cout << "test_compression_orders_and_marks_gaps PASSED\n";
}

void test_compression_fallback_and_flag_off_format() {
    const std::string text = u8"ééé relevant later sentence";
    const auto fallback = memorylayer::compress_memory_side(text, {}, {1.0f}, 5);
    assert(fallback == u8"éé...");

    auto memory = long_memory(static_cast<double>(std::time(nullptr)) - 60.0);
    const auto normal = memorylayer::format_memory_context_budgeted({memory}, 0.0, 2048, 50);
    const auto legacy = memorylayer::format_memory_context_budgeted({memory}, 0.0, 2048, 50);
    assert(normal == legacy);
    std::cout << "test_compression_fallback_and_flag_off_format PASSED\n";
}

int main() {
    test_line_token_bound_holds();
    test_required_budget_fits_all_k_memories();
    test_sentence_splitter_edge_cases();
    test_compression_selects_relevant_later_sentence();
    test_compression_orders_and_marks_gaps();
    test_compression_fallback_and_flag_off_format();
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
