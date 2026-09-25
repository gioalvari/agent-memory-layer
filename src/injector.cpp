#include "memorylayer/injector.h"
#include <cmath>
#include <sstream>

namespace memorylayer {

static std::string format_time_ago(double seconds_ago) {
    if (seconds_ago < 60) return "just now";
    if (seconds_ago < 3600) return std::to_string(static_cast<int>(seconds_ago / 60)) + "m ago";
    if (seconds_ago < 86400) return std::to_string(static_cast<int>(seconds_ago / 3600)) + "h ago";
    return std::to_string(static_cast<int>(seconds_ago / 86400)) + "d ago";
}

static std::string truncate(const std::string& s, int max_chars) {
    if (static_cast<int>(s.size()) <= max_chars) return s;
    return s.substr(0, max_chars) + "...";
}

std::string format_memory_line(const ScoredMemory& mem, double now_unix, int max_chars) {
    double age_seconds = now_unix - mem.memory.created_at;
    std::string time_str = format_time_ago(age_seconds);
    std::string user_part = truncate(mem.memory.user_text, max_chars);
    std::string assist_part = truncate(mem.memory.assist_text, max_chars);
    return "- [" + time_str + "] " + user_part + " Response: " + assist_part;
}

std::string format_memory_context(const std::vector<ScoredMemory>& memories, double now_unix) {
    if (memories.empty()) return "";

    std::ostringstream oss;
    oss << "<memory context>\nRelevant past interactions:\n";
    for (const auto& sm : memories) {
        oss << format_memory_line(sm, now_unix) << "\n";
    }
    oss << "</memory context>";
    return oss.str();
}

int estimate_tokens(const std::string& text) {
    // Heuristic: ~4 characters per token (conservative for English text)
    return static_cast<int>(text.size()) / 4 + 1;
}

std::string format_memory_context_budgeted(const std::vector<ScoredMemory>& memories,
                                           double now_unix, int max_tokens) {
    if (memories.empty()) return "";

    std::string header = "<memory context>\nRelevant past interactions:\n";
    std::string footer = "</memory context>";
    int budget = max_tokens - estimate_tokens(header) - estimate_tokens(footer);

    if (budget <= 0) return "";

    std::ostringstream oss;
    oss << header;

    int used_tokens = 0;
    int included = 0;
    for (const auto& sm : memories) {
        std::string line = format_memory_line(sm, now_unix) + "\n";
        int line_tokens = estimate_tokens(line);

        if (used_tokens + line_tokens > budget) break;

        oss << line;
        used_tokens += line_tokens;
        included++;
    }

    if (included == 0) return "";

    oss << footer;
    return oss.str();
}

InjectMode parse_inject_mode(const std::string& mode) {
    if (mode == "suffix") return InjectMode::Suffix;
    if (mode == "sticky") return InjectMode::Sticky;
    return InjectMode::System;
}

void inject_memories(nlohmann::json& messages, const std::string& memory_context,
                     InjectMode mode) {
    if (memory_context.empty()) return;

    if (mode == InjectMode::Suffix || mode == InjectMode::Sticky) {
        // Only the final user turn changes, so every token before it (system
        // prompt + history) is identical to the previous request and stays
        // reusable in the backend's prefix cache.
        for (auto it = messages.rbegin(); it != messages.rend(); ++it) {
            if ((*it).value("role", "") != "user") continue;
            auto& content = (*it)["content"];
            if (content.is_string()) {
                content = memory_context + "\n\n" + content.get<std::string>();
            } else {
                // Structured (multi-part) content: add the block as its own
                // system message right before the user turn.
                auto pos = std::next(it).base();
                nlohmann::json block = {{"role", "system"}, {"content", memory_context}};
                messages.insert(pos, block);
            }
            return;
        }
        // No user message: fall through to system-prompt injection.
    }

    if (!messages.empty() && messages[0].value("role", "") == "system") {
        std::string content = messages[0]["content"].get<std::string>();
        content += "\n\n" + memory_context;
        messages[0]["content"] = content;
    } else {
        nlohmann::json sys_msg = {{"role", "system"}, {"content", memory_context}};
        messages.insert(messages.begin(), sys_msg);
    }
}

} // namespace memorylayer
