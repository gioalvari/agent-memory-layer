#include "memorylayer/injector.h"
#include "memorylayer/temporal.h"
#include <cmath>
#include <algorithm>
#include <cctype>
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

static bool is_utf8_continuation(unsigned char c) {
    return (c & 0xC0) == 0x80;
}

static int utf8_safe_end(const std::string& text, int end) {
    end = std::min(end, static_cast<int>(text.size()));
    while (end > 0 && end < static_cast<int>(text.size()) &&
           is_utf8_continuation(static_cast<unsigned char>(text[end]))) {
        --end;
    }
    return end;
}

static std::string truncate_utf8(const std::string& text, int max_chars) {
    if (static_cast<int>(text.size()) <= max_chars) return text;
    return text.substr(0, utf8_safe_end(text, max_chars)) + "...";
}

static void trim_span(const std::string& text, int& start, int& end) {
    while (start < end && std::isspace(static_cast<unsigned char>(text[start]))) ++start;
    while (end > start && std::isspace(static_cast<unsigned char>(text[end - 1]))) --end;
}

static bool is_bullet_line(const std::string& text, int start, int end) {
    while (start < end && (text[start] == ' ' || text[start] == '\t')) ++start;
    if (start >= end) return false;
    if (text[start] == '-' || text[start] == '*' || text[start] == '+') return true;
    return std::isdigit(static_cast<unsigned char>(text[start])) && start + 1 < end &&
           text[start + 1] == '.';
}

static void split_long_span(const std::string& text, int start, int end,
                            std::vector<MemorySentence>& out) {
    constexpr int kMaxSentenceBytes = 300;
    while (end - start > kMaxSentenceBytes) {
        int split = utf8_safe_end(text, start + kMaxSentenceBytes);
        int whitespace = split;
        while (whitespace > start &&
               !std::isspace(static_cast<unsigned char>(text[whitespace - 1]))) {
            --whitespace;
        }
        if (whitespace > start) split = whitespace;
        int part_start = start;
        int part_end = split;
        trim_span(text, part_start, part_end);
        if (part_start < part_end) out.push_back({0, part_start, part_end, {}});
        start = split;
        while (start < end && std::isspace(static_cast<unsigned char>(text[start]))) ++start;
    }
    trim_span(text, start, end);
    if (start < end) out.push_back({0, start, end, {}});
}

std::vector<MemorySentence> split_memory_sentences(const std::string& text) {
    std::vector<MemorySentence> raw;
    int line_start = 0;
    bool in_fenced_code = false;
    int code_start = -1;
    for (int pos = 0; pos <= static_cast<int>(text.size()); ++pos) {
        if (pos != static_cast<int>(text.size()) && text[pos] != '\n') continue;
        int line_end = pos;
        int start = line_start;
        trim_span(text, start, line_end);
        const bool fence = start + 3 <= line_end && text.compare(start, 3, "```") == 0;
        if (in_fenced_code) {
            if (fence) {
                split_long_span(text, code_start, line_end, raw);
                in_fenced_code = false;
                code_start = -1;
            }
        } else if (fence) {
            in_fenced_code = true;
            code_start = start;
        } else if (start < line_end && is_bullet_line(text, start, line_end)) {
            split_long_span(text, start, line_end, raw);
        } else {
            int sentence_start = start;
            for (int i = start; i < line_end; ++i) {
                if (text[i] != '.' && text[i] != '!' && text[i] != '?') continue;
                int sentence_end = i + 1;
                split_long_span(text, sentence_start, sentence_end, raw);
                sentence_start = sentence_end;
            }
            split_long_span(text, sentence_start, line_end, raw);
        }
        line_start = pos + 1;
    }
    if (in_fenced_code) split_long_span(text, code_start, static_cast<int>(text.size()), raw);

    // Very short fragments rarely carry useful meaning alone; retain their text
    // by merging them into the closest preceding span, or the next span at the
    // start of a message.
    std::vector<MemorySentence> merged;
    for (auto sentence : raw) {
        if (sentence.end_byte - sentence.start_byte < 20 && !merged.empty()) {
            merged.back().end_byte = sentence.end_byte;
        } else {
            merged.push_back(std::move(sentence));
        }
    }
    if (merged.size() > 1 && merged.front().end_byte - merged.front().start_byte < 20) {
        merged[1].start_byte = merged.front().start_byte;
        merged.erase(merged.begin());
    }
    for (int i = 0; i < static_cast<int>(merged.size()); ++i) merged[i].idx = i;
    return merged;
}

static float cosine(const std::vector<float>& query, const std::vector<float>& candidate) {
    if (query.empty() || query.size() != candidate.size()) return -1.0f;
    float dot = 0.0f;
    float q_norm = 0.0f;
    float c_norm = 0.0f;
    for (std::size_t i = 0; i < query.size(); ++i) {
        dot += query[i] * candidate[i];
        q_norm += query[i] * query[i];
        c_norm += candidate[i] * candidate[i];
    }
    return q_norm > 0.0f && c_norm > 0.0f ? dot / std::sqrt(q_norm * c_norm) : -1.0f;
}

static std::string join_selected(const std::string& text,
                                 const std::vector<const MemorySentence*>& selected) {
    std::ostringstream result;
    for (std::size_t i = 0; i < selected.size(); ++i) {
        if (i > 0) {
            result << (selected[i]->idx == selected[i - 1]->idx + 1 ? " " : " … ");
        }
        result << text.substr(selected[i]->start_byte,
                              selected[i]->end_byte - selected[i]->start_byte);
    }
    return result.str();
}

std::string compress_memory_side(const std::string& text,
                                 const std::vector<MemorySentence>& sentences,
                                 const std::vector<float>& query_embedding,
                                 int max_chars) {
    if (static_cast<int>(text.size()) <= max_chars) return text;
    if (sentences.empty() || query_embedding.empty()) return truncate_utf8(text, max_chars);

    std::vector<const MemorySentence*> ranked;
    for (const auto& sentence : sentences) {
        if (sentence.start_byte >= 0 && sentence.end_byte <= static_cast<int>(text.size()) &&
            sentence.start_byte < sentence.end_byte && !sentence.embedding.empty()) {
            ranked.push_back(&sentence);
        }
    }
    std::sort(ranked.begin(), ranked.end(), [&query_embedding](const auto* a, const auto* b) {
        const float a_score = cosine(query_embedding, a->embedding);
        const float b_score = cosine(query_embedding, b->embedding);
        return a_score == b_score ? a->idx < b->idx : a_score > b_score;
    });

    std::vector<const MemorySentence*> selected;
    for (const auto* sentence : ranked) {
        auto candidate = selected;
        candidate.push_back(sentence);
        std::sort(candidate.begin(), candidate.end(),
                  [](const auto* a, const auto* b) { return a->idx < b->idx; });
        if (static_cast<int>(join_selected(text, candidate).size()) <= max_chars) {
            selected = std::move(candidate);
        }
    }
    return selected.empty() ? truncate_utf8(text, max_chars) : join_selected(text, selected);
}

std::string format_memory_line(const ScoredMemory& mem, double now_unix, int max_chars,
                               bool absolute_dates) {
    double age_seconds = now_unix - mem.memory.created_at;
    std::string time_str = format_time_ago(age_seconds);
    std::string user_part = truncate(mem.memory.user_text, max_chars);
    std::string assist_part = truncate(mem.memory.assist_text, max_chars);
    const std::string timestamp = absolute_dates
        ? utc_date(mem.memory.created_at) + ", " + time_str
        : time_str;
    return "- [" + timestamp + "] " + user_part + " Response: " + assist_part;
}

std::string format_memory_context(const std::vector<ScoredMemory>& memories, double now_unix,
                                  bool absolute_dates) {
    if (memories.empty()) return "";

    std::ostringstream oss;
    oss << kMemoryContextHeader;
    for (const auto& sm : memories) {
        oss << format_memory_line(sm, now_unix, kDefaultMemoryLineChars, absolute_dates) << "\n";
    }
    oss << kMemoryContextFooter;
    return oss.str();
}

int estimate_tokens(const std::string& text) {
    // Heuristic: ~4 characters per token (conservative for English text)
    return static_cast<int>(text.size()) / 4 + 1;
}

std::string format_memory_context_budgeted(const std::vector<ScoredMemory>& memories,
                                            double now_unix, int max_tokens,
                                            int line_chars, bool absolute_dates) {
    if (memories.empty()) return "";

    const std::string header(kMemoryContextHeader);
    const std::string footer(kMemoryContextFooter);
    int budget = max_tokens - estimate_tokens(header) - estimate_tokens(footer);

    if (budget <= 0) return "";

    std::ostringstream oss;
    oss << header;

    int used_tokens = 0;
    int included = 0;
    for (const auto& sm : memories) {
        std::string line = format_memory_line(sm, now_unix, line_chars, absolute_dates) + "\n";
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

std::string format_memory_context_budgeted_compressed(
    const std::vector<ScoredMemory>& memories, const std::vector<float>& query_embedding,
    double now_unix, int max_tokens, int line_chars, bool absolute_dates) {
    if (memories.empty()) return "";
    const std::string header(kMemoryContextHeader);
    const std::string footer(kMemoryContextFooter);
    const int budget = max_tokens - estimate_tokens(header) - estimate_tokens(footer);
    if (budget <= 0) return "";
    std::ostringstream oss;
    oss << header;
    int used_tokens = 0;
    int included = 0;
    for (const auto& sm : memories) {
        const std::string user = compress_memory_side(sm.memory.user_text,
                                                      sm.memory.user_sentences,
                                                      query_embedding, line_chars);
        const std::string assist = compress_memory_side(sm.memory.assist_text,
                                                        sm.memory.assist_sentences,
                                                        query_embedding, line_chars);
        const std::string timestamp = absolute_dates
            ? utc_date(sm.memory.created_at) + ", " +
                  format_time_ago(now_unix - sm.memory.created_at)
            : format_time_ago(now_unix - sm.memory.created_at);
        const std::string line = "- [" + timestamp + "] " +
                                  user + " Response: " + assist + "\n";
        const int line_tokens = estimate_tokens(line);
        if (used_tokens + line_tokens > budget) break;
        oss << line;
        used_tokens += line_tokens;
        ++included;
    }
    if (included == 0) return "";
    oss << footer;
    return oss.str();
}

std::string format_fact_memory_context_budgeted(const std::vector<ScoredFact>& facts,
                                                 const std::vector<ScoredMemory>& memories,
                                                 double now_unix, int max_tokens, int line_chars,
                                                 const std::vector<float>* query_embedding,
                                                 bool absolute_dates) {
    if (facts.empty()) {
        return query_embedding
            ? format_memory_context_budgeted_compressed(memories, *query_embedding, now_unix,
                                                        max_tokens, line_chars, absolute_dates)
            : format_memory_context_budgeted(memories, now_unix, max_tokens, line_chars,
                                             absolute_dates);
    }
    const std::string header(kMemoryContextHeader);
    const std::string footer(kMemoryContextFooter);
    int budget = max_tokens - estimate_tokens(header) - estimate_tokens(footer);
    if (budget <= 0) return "";
    std::ostringstream oss;
    oss << header << "Known facts (latest values):\n";
    int used = estimate_tokens("Known facts (latest values):\n");
    for (const auto& scored : facts) {
        const auto& fact = scored.fact;
        std::string line = "- " + fact.subject + " " + fact.attribute + ": " + fact.value +
                           " (as of " + format_time_ago(now_unix - fact.created_at) + ")\n";
        const int tokens = estimate_tokens(line);
        if (used + tokens > budget) break;
        oss << line;
        used += tokens;
    }
    if (!memories.empty()) {
        oss << "Relevant conversation memories:\n";
        used += estimate_tokens("Relevant conversation memories:\n");
        for (const auto& memory : memories) {
            ScoredMemory formatted = memory;
            if (query_embedding) {
                formatted.memory.user_text = compress_memory_side(
                    memory.memory.user_text, memory.memory.user_sentences, *query_embedding,
                    line_chars);
                formatted.memory.assist_text = compress_memory_side(
                    memory.memory.assist_text, memory.memory.assist_sentences, *query_embedding,
                    line_chars);
            }
            const std::string line = format_memory_line(formatted, now_unix, line_chars,
                                                        absolute_dates) + "\n";
            const int tokens = estimate_tokens(line);
            if (used + tokens > budget) break;
            oss << line;
            used += tokens;
        }
    }
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
