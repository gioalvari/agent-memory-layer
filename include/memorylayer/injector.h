#pragma once
#include "memorylayer/memory_store.h"
#include "json.hpp"
#include <string>
#include <vector>

namespace memorylayer {

std::string format_memory_line(const ScoredMemory& mem, double now_unix, int max_chars = 200);
std::string format_memory_context(const std::vector<ScoredMemory>& memories, double now_unix);
int estimate_tokens(const std::string& text);
std::string format_memory_context_budgeted(const std::vector<ScoredMemory>& memories,
                                           double now_unix, int max_tokens);
enum class InjectMode {
    System,  // append to the system prompt (inserted if missing)
    Suffix,  // prepend to the last user message; earlier messages are untouched
    Sticky,  // suffix placement; sticky history is handled by the proxy cache
};

// Parses "system" / "suffix" / "sticky"; anything else maps to System.
InjectMode parse_inject_mode(const std::string& mode);

void inject_memories(nlohmann::json& messages, const std::string& memory_context,
                     InjectMode mode = InjectMode::System);

} // namespace memorylayer
