#pragma once

#include "memorylayer/memory_store.h"
#include "json.hpp"
#include <cstddef>
#include <cstdint>
#include <list>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace memorylayer {

struct StickyCacheEntry {
    std::string block;
    std::vector<int64_t> memory_ids;
};

class StickyCache {
public:
    explicit StickyCache(std::size_t capacity);

    std::optional<StickyCacheEntry> get(uint64_t key);
    void put(uint64_t key, StickyCacheEntry entry);
    std::size_t size() const;

private:
    struct Item {
        StickyCacheEntry entry;
        std::list<uint64_t>::iterator order;
    };

    std::size_t capacity_;
    std::list<uint64_t> lru_;
    std::unordered_map<uint64_t, Item> entries_;
    mutable std::mutex mutex_;
};

// Hashes agent_id and original messages through user_index, inclusive. Message
// content is serialized with json::dump() so structured content is stable too.
uint64_t sticky_message_key(const std::string& agent_id,
                            const nlohmann::json& original_messages,
                            std::size_t user_index);

std::optional<std::size_t> last_user_message_index(const nlohmann::json& messages);

// Applies the suffix-style block directly to a specific user message.
void inject_sticky_block(nlohmann::json& messages, std::size_t user_index,
                         const std::string& block);

struct StickyHistoryResult {
    std::size_t reused_blocks = 0;
    std::unordered_set<int64_t> shown_memory_ids;
};

// Re-injects cached blocks for every historical user turn. The supplied
// messages must still be in their original, client-provided form.
StickyHistoryResult apply_sticky_history(nlohmann::json& messages,
                                         const std::string& agent_id,
                                         StickyCache& cache);

std::vector<ScoredMemory> filter_excluded_memories(
    const std::vector<ScoredMemory>& memories,
    const std::unordered_set<int64_t>& excluded_ids, int top_k);

struct StickyFormattedContext {
    std::string block;
    std::vector<int64_t> memory_ids;
};

// Uses UTC calendar dates rather than relative times, keeping a cached block
// byte-identical when it is re-injected in a later request.
StickyFormattedContext format_sticky_memory_context_budgeted(
    const std::vector<ScoredMemory>& memories, int max_tokens);

} // namespace memorylayer
