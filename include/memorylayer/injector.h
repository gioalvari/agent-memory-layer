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
void inject_memories(nlohmann::json& messages, const std::string& memory_context);

} // namespace memorylayer
