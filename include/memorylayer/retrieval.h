#pragma once

#include <algorithm>
#include <string_view>

namespace memorylayer {

enum class DecayMode {
    Tiebreak,
    Legacy,
};

constexpr float kRecencyTiebreakWeight = 1e-3f;

inline float recency_decay(double age_hours, int decay_days) {
    const double horizon_hours = 24.0 * static_cast<double>(decay_days);
    const float linear_decay = horizon_hours > 0.0
        ? 1.0f - static_cast<float>(age_hours / horizon_hours)
        : 0.5f;
    return std::max(0.5f, std::min(1.0f, linear_decay));
}

inline float similarity_score(float cos_max, bool same_agent, float agent_boost) {
    return cos_max * (same_agent ? agent_boost : 1.0f);
}

inline float rank_score(float cos_max, double age_hours, int decay_days,
                        bool same_agent, float agent_boost, DecayMode mode) {
    const float similarity = similarity_score(cos_max, same_agent, agent_boost);
    const float recency = recency_decay(age_hours, decay_days);
    if (mode == DecayMode::Legacy) return similarity * recency;
    return similarity + kRecencyTiebreakWeight * recency;
}

// Split-conformal rank quantiles, ceil((1 - alpha)(n + 1))-th best-evidence
// rank, calibrated once on all 467 labeled LongMemEval-S questions
// (research/conformal/src/conformal_retrieval/followup.py). Marginal
// at-least-one-evidence coverage, specific to nomic-embed-text.
inline int calibrated_top_k(float target_coverage, DecayMode mode) {
    if (target_coverage == 0.8f) return mode == DecayMode::Legacy ? 7 : 4;
    if (target_coverage == 0.9f) return mode == DecayMode::Legacy ? 15 : 8;
    if (target_coverage == 0.95f) return mode == DecayMode::Legacy ? 26 : 16;
    return 0;
}

// 400 bytes per side: on LongMemEval-S with Qwen2.5-7B, five 400-byte memories
// answer 53.7% of short-answer questions versus 40.2% at 200 bytes, and five
// worst-case lines still fit the 2048-token default injection budget.
constexpr int kDefaultMemoryLineChars = 400;
constexpr std::string_view kMemoryContextHeader =
    "<memory context>\nRelevant past interactions:\n";
constexpr std::string_view kMemoryContextFooter = "</memory context>";
// Longest timestamp either formatter emits: "99999d ago" or "YYYY-MM-DD".
constexpr int kMaxMemoryTimestampChars = 10;

// Mirrors estimate_tokens(): bytes / 4 + 1.
constexpr int estimate_tokens_for_bytes(int bytes) { return bytes / 4 + 1; }

// Upper bound on the estimated tokens of one injected memory line,
// "- [<time>] <user...> Response: <assistant...>\n", where each side is
// truncated to line_chars bytes plus "...".
constexpr int max_memory_line_tokens(int line_chars) {
    return estimate_tokens_for_bytes(3 + kMaxMemoryTimestampChars + 2 +
                                     2 * (line_chars + 3) + 11 + 1);
}

// Injection budget that fits k memory lines of any length without dropping one.
constexpr int required_inject_tokens(int k, int line_chars) {
    return k * max_memory_line_tokens(line_chars) +
           estimate_tokens_for_bytes(static_cast<int>(kMemoryContextHeader.size())) +
           estimate_tokens_for_bytes(static_cast<int>(kMemoryContextFooter.size()));
}

inline bool is_valid_target_coverage(float target_coverage) {
    return target_coverage == 0.0f || calibrated_top_k(target_coverage, DecayMode::Tiebreak) != 0;
}

} // namespace memorylayer
