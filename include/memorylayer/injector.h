#pragma once
#include "memorylayer/memory_store.h"
#include "memorylayer/retrieval.h"
#include "json.hpp"
#include <string>
#include <vector>

namespace memorylayer {

std::string format_memory_line(const ScoredMemory& mem, double now_unix,
                               int max_chars = kDefaultMemoryLineChars);
std::string format_memory_context(const std::vector<ScoredMemory>& memories, double now_unix);
int estimate_tokens(const std::string& text);
std::string format_memory_context_budgeted(const std::vector<ScoredMemory>& memories,
                                            double now_unix, int max_tokens,
                                            int line_chars = kDefaultMemoryLineChars);
// Deterministic UTF-8-safe sentence spans used for storage and compression.
std::vector<MemorySentence> split_memory_sentences(const std::string& text);
std::string compress_memory_side(const std::string& text,
                                 const std::vector<MemorySentence>& sentences,
                                 const std::vector<float>& query_embedding,
                                 int max_chars);
std::string format_memory_context_budgeted_compressed(
    const std::vector<ScoredMemory>& memories, const std::vector<float>& query_embedding,
    double now_unix, int max_tokens, int line_chars = kDefaultMemoryLineChars);
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
