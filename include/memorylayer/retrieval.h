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

// Mondrian rank conformal on the serving-time confidence of a query: the raw
// (unboosted) cosine of its top-ranked memory. Tercile edges and one rank
// quantile per tercile, calibrated on the same 467 questions (about 155 per
// tercile). Queries whose best match is weak get a deeper k, confident ones a
// shallower k, so coverage is roughly equal across terciles rather than only
// on average. adaptive_top_k() never goes below the global calibrated k: with
// a 4-memory tercile k, confident queries lost 5-6 points of end-to-end
// accuracy, and the floored set still contains both conformal sets.
struct AdaptiveKTable {
    float edge_low;
    float edge_high;
    int k[3];  // low, mid, high top-1 similarity
};

inline const AdaptiveKTable* adaptive_k_table(float target_coverage, DecayMode mode) {
    static constexpr AdaptiveKTable kTiebreak[] = {
        {0.6318194f, 0.7150962f, {9, 3, 2}},
        {0.6318194f, 0.7150962f, {21, 7, 4}},
        {0.6318194f, 0.7150962f, {49, 14, 5}},
    };
    static constexpr AdaptiveKTable kLegacy[] = {
        {0.6065812f, 0.6967925f, {20, 4, 3}},
        {0.6065812f, 0.6967925f, {35, 8, 5}},
        {0.6065812f, 0.6967925f, {69, 17, 7}},
    };
    const AdaptiveKTable* table = mode == DecayMode::Legacy ? kLegacy : kTiebreak;
    if (target_coverage == 0.8f) return &table[0];
    if (target_coverage == 0.9f) return &table[1];
    if (target_coverage == 0.95f) return &table[2];
    return nullptr;
}

// Tercile index: 0 when top1 <= edge_low, 1 when <= edge_high, else 2.
inline int adaptive_bin(const AdaptiveKTable& table, float top1_similarity) {
    if (top1_similarity <= table.edge_low) return 0;
    if (top1_similarity <= table.edge_high) return 1;
    return 2;
}

inline int adaptive_top_k(float target_coverage, DecayMode mode, float top1_similarity) {
    const AdaptiveKTable* table = adaptive_k_table(target_coverage, mode);
    if (!table) return 0;
    return std::max(table->k[adaptive_bin(*table, top1_similarity)],
                    calibrated_top_k(target_coverage, mode));
}

// Deepest k a coverage policy can request: the calibrated k, or the
// low-confidence tercile's k when adaptive.
inline int max_calibrated_top_k(float target_coverage, DecayMode mode, bool adaptive) {
    if (!adaptive) return calibrated_top_k(target_coverage, mode);
    const AdaptiveKTable* table = adaptive_k_table(target_coverage, mode);
    return table ? table->k[0] : 0;
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
