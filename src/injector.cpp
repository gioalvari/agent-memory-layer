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

void inject_memories(nlohmann::json& messages, const std::string& memory_context) {
    if (memory_context.empty()) return;

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
