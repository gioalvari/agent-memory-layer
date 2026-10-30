#include "memorylayer/history.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace memorylayer {

namespace {

int estimate_message_tokens(const nlohmann::json& message) {
    return static_cast<int>(message.dump().size()) / 4 + 1;
}

} // namespace

int estimate_non_system_message_tokens(const nlohmann::json& messages) {
    int tokens = 0;
    for (const auto& message : messages) {
        if (message.value("role", "") != "system") {
            tokens += estimate_message_tokens(message);
        }
    }
    return tokens;
}

HistoryTrimResult trim_history(const nlohmann::json& messages, int budget,
                               double trim_ratio) {
    HistoryTrimResult result{messages};
    if (budget <= 0 || estimate_non_system_message_tokens(messages) <= budget) {
        return result;
    }

    std::vector<std::size_t> user_starts;
    for (std::size_t index = 0; index < messages.size(); ++index) {
        if (messages[index].value("role", "") == "user") {
            user_starts.push_back(index);
        }
    }
    if (user_starts.empty()) return result;

    const int target = std::max(0, static_cast<int>(std::floor(budget * trim_ratio)));
    std::size_t cut_index = user_starts.back();
    for (const std::size_t candidate : user_starts) {
        int retained_tokens = 0;
        for (std::size_t index = candidate; index < messages.size(); ++index) {
            if (messages[index].value("role", "") != "system") {
                retained_tokens += estimate_message_tokens(messages[index]);
            }
        }
        if (retained_tokens <= target) {
            cut_index = candidate;
            break;
        }
    }

    nlohmann::json retained = nlohmann::json::array();
    int dropped = 0;
    for (std::size_t index = 0; index < messages.size(); ++index) {
        const bool system = messages[index].value("role", "") == "system";
        if (system || index >= cut_index) {
            retained.push_back(messages[index]);
        } else {
            ++dropped;
        }
    }

    if (dropped == 0) return result;
    result.messages = std::move(retained);
    result.dropped_messages = dropped;
    result.trimmed = true;
    return result;
}

} // namespace memorylayer
