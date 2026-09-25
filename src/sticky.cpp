#include "memorylayer/sticky.h"

#include <algorithm>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <utility>

namespace memorylayer {

namespace {

constexpr uint64_t kFnvOffsetBasis = 14695981039346656037ULL;
constexpr uint64_t kFnvPrime = 1099511628211ULL;

void fnv_append(uint64_t& hash, const std::string& value) {
    for (unsigned char byte : value) {
        hash ^= byte;
        hash *= kFnvPrime;
    }
}

std::string truncate(const std::string& text, int max_chars) {
    if (static_cast<int>(text.size()) <= max_chars) return text;
    return text.substr(0, max_chars) + "...";
}

std::string utc_date(double timestamp) {
    const std::time_t time = static_cast<std::time_t>(timestamp);
    std::tm utc{};
    gmtime_r(&time, &utc);
    std::ostringstream output;
    output << std::put_time(&utc, "%Y-%m-%d");
    return output.str();
}

std::string sticky_memory_line(const ScoredMemory& memory, int max_chars = 200) {
    return "- [" + utc_date(memory.memory.created_at) + "] " +
           truncate(memory.memory.user_text, max_chars) + " Response: " +
           truncate(memory.memory.assist_text, max_chars);
}

int estimate_tokens(const std::string& text) {
    return static_cast<int>(text.size()) / 4 + 1;
}

} // namespace

StickyCache::StickyCache(std::size_t capacity) : capacity_(capacity) {}

std::optional<StickyCacheEntry> StickyCache::get(uint64_t key) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto found = entries_.find(key);
    if (found == entries_.end()) return std::nullopt;

    lru_.splice(lru_.begin(), lru_, found->second.order);
    return found->second.entry;
}

void StickyCache::put(uint64_t key, StickyCacheEntry entry) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto found = entries_.find(key);
    if (found != entries_.end()) {
        found->second.entry = std::move(entry);
        lru_.splice(lru_.begin(), lru_, found->second.order);
        return;
    }

    lru_.push_front(key);
    entries_.emplace(key, Item{std::move(entry), lru_.begin()});
    if (entries_.size() > capacity_) {
        const uint64_t evicted_key = lru_.back();
        lru_.pop_back();
        entries_.erase(evicted_key);
    }
}

std::size_t StickyCache::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return entries_.size();
}

uint64_t sticky_message_key(const std::string& agent_id,
                            const nlohmann::json& original_messages,
                            std::size_t user_index) {
    uint64_t hash = kFnvOffsetBasis;
    fnv_append(hash, agent_id);
    fnv_append(hash, "\x1e");
    for (std::size_t index = 0; index <= user_index; ++index) {
        const auto& message = original_messages.at(index);
        fnv_append(hash, message.value("role", ""));
        fnv_append(hash, "\x1f");
        if (message.contains("content")) {
            fnv_append(hash, message.at("content").dump());
        } else {
            fnv_append(hash, "null");
        }
        fnv_append(hash, "\x1e");
    }
    return hash;
}

std::optional<std::size_t> last_user_message_index(const nlohmann::json& messages) {
    for (std::size_t index = messages.size(); index > 0; --index) {
        if (messages[index - 1].value("role", "") == "user") return index - 1;
    }
    return std::nullopt;
}

void inject_sticky_block(nlohmann::json& messages, std::size_t user_index,
                         const std::string& block) {
    if (block.empty()) return;
    auto& content = messages.at(user_index)["content"];
    if (content.is_string()) {
        content = block + "\n\n" + content.get<std::string>();
        return;
    }

    nlohmann::json system_block = {{"role", "system"}, {"content", block}};
    messages.insert(messages.begin() + static_cast<nlohmann::json::difference_type>(user_index),
                    std::move(system_block));
}

StickyHistoryResult apply_sticky_history(nlohmann::json& messages,
                                         const std::string& agent_id,
                                         StickyCache& cache) {
    StickyHistoryResult result;
    const auto last_user = last_user_message_index(messages);
    if (!last_user) return result;

    struct CachedBlock {
        std::size_t user_index;
        StickyCacheEntry entry;
    };
    std::vector<CachedBlock> blocks;
    for (std::size_t index = 0; index < *last_user; ++index) {
        if (messages[index].value("role", "") != "user") continue;
        const auto entry = cache.get(sticky_message_key(agent_id, messages, index));
        if (!entry) continue;
        result.shown_memory_ids.insert(entry->memory_ids.begin(), entry->memory_ids.end());
        if (!entry->block.empty()) {
            blocks.push_back({index, *entry});
            ++result.reused_blocks;
        }
    }

    for (auto it = blocks.rbegin(); it != blocks.rend(); ++it) {
        inject_sticky_block(messages, it->user_index, it->entry.block);
    }
    return result;
}

std::vector<ScoredMemory> filter_excluded_memories(
    const std::vector<ScoredMemory>& memories,
    const std::unordered_set<int64_t>& excluded_ids, int top_k) {
    std::vector<ScoredMemory> filtered;
    for (const auto& memory : memories) {
        if (excluded_ids.find(memory.memory.id) != excluded_ids.end()) continue;
        filtered.push_back(memory);
        if (static_cast<int>(filtered.size()) == top_k) break;
    }
    return filtered;
}

StickyFormattedContext format_sticky_memory_context_budgeted(
    const std::vector<ScoredMemory>& memories, int max_tokens) {
    StickyFormattedContext result;
    if (memories.empty()) return result;

    const std::string header = "<memory context>\nRelevant past interactions:\n";
    const std::string footer = "</memory context>";
    const int budget = max_tokens - estimate_tokens(header) - estimate_tokens(footer);
    if (budget <= 0) return result;

    std::ostringstream output;
    output << header;
    int used_tokens = 0;
    for (const auto& memory : memories) {
        const std::string line = sticky_memory_line(memory) + "\n";
        const int line_tokens = estimate_tokens(line);
        if (used_tokens + line_tokens > budget) break;
        output << line;
        used_tokens += line_tokens;
        result.memory_ids.push_back(memory.memory.id);
    }
    if (result.memory_ids.empty()) return result;

    output << footer;
    result.block = output.str();
    return result;
}

} // namespace memorylayer
