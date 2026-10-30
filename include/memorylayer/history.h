#pragma once

#include "json.hpp"

namespace memorylayer {

struct HistoryTrimResult {
    nlohmann::json messages;
    int dropped_messages = 0;
    bool trimmed = false;
};

// Approximate the tokens used by non-system chat messages. The estimate uses
// the same simple character heuristic as memory injection budgeting.
int estimate_non_system_message_tokens(const nlohmann::json& messages);

// Retain all system messages and a suffix of complete conversation turns. A
// retained suffix always begins with a user message, so assistant tool calls
// and their following tool messages remain together in the same turn.
HistoryTrimResult trim_history(const nlohmann::json& messages, int budget,
                               double trim_ratio);

} // namespace memorylayer
