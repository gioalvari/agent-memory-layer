#pragma once

#include <functional>
#include <optional>
#include <string>

namespace memorylayer {

// Returns a model-token count when a tokenizer is available. Returning nullopt
// asks count_tokens() to use the deterministic character fallback instead.
using TokenCounter = std::function<std::optional<int>(const std::string&)>;

inline int heuristic_token_count(const std::string& text) {
    return static_cast<int>(text.size()) / 4 + 1;
}

inline int count_tokens(const std::string& text, const TokenCounter& counter = {}) {
    if (counter) {
        const auto count = counter(text);
        if (count && *count >= 0) return *count;
    }
    return heuristic_token_count(text);
}

} // namespace memorylayer
