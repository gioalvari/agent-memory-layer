#include "memorylayer/retrieval.h"

#include <cassert>
#include <cmath>
#include <iostream>

// Values below were calculated once with decay_tiebreak and production in
// research/conformal/src/conformal_retrieval/study.py.
void test_rank_score_python_parity() {
    using memorylayer::DecayMode;
    using memorylayer::rank_score;

    assert(std::abs(rank_score(0.90f, 0.0, 30, false, 1.2f,
                               DecayMode::Tiebreak) - 0.901f) < 1e-6f);
    assert(std::abs(rank_score(0.91f, 480.0, 30, false, 1.2f,
                               DecayMode::Tiebreak) - 0.9105f) < 1e-6f);
    assert(std::abs(rank_score(0.85f, 240.0, 30, false, 1.2f,
                               DecayMode::Legacy) - 0.5666667f) < 1e-6f);
    assert(std::abs(rank_score(0.85f, 0.0, 30, true, 1.2f,
                               DecayMode::Legacy) - 1.02f) < 1e-6f);
    assert(std::abs(rank_score(0.91f, 480.0, 30, false, 1.2f,
                               DecayMode::Legacy) - 0.455f) < 1e-6f);
    std::cout << "test_rank_score_python_parity PASSED\n";
}

void test_tiebreak_scoring_behavior() {
    using memorylayer::DecayMode;
    using memorylayer::rank_score;

    // After twenty days, legacy recency outweighs a 0.01 cosine advantage;
    // tiebreak preserves the higher cosine rank.
    const float older_tiebreak = rank_score(0.90f, 480.0, 30, false, 1.2f,
                                             DecayMode::Tiebreak);
    const float newer_tiebreak = rank_score(0.89f, 0.0, 30, false, 1.2f,
                                             DecayMode::Tiebreak);
    const float older_legacy = rank_score(0.90f, 480.0, 30, false, 1.2f,
                                           DecayMode::Legacy);
    const float newer_legacy = rank_score(0.89f, 0.0, 30, false, 1.2f,
                                           DecayMode::Legacy);
    assert(older_tiebreak > newer_tiebreak);
    assert(older_legacy < newer_legacy);

    const float recent_tie = rank_score(0.90f, 0.0, 30, false, 1.2f,
                                         DecayMode::Tiebreak);
    const float old_tie = rank_score(0.90f, 960.0, 30, false, 1.2f,
                                      DecayMode::Tiebreak);
    assert(recent_tie > old_tie);

    // Recency is deliberately excluded from the threshold-scale score.
    const float threshold = 0.90f;
    const float old_rank = rank_score(0.90f, 960.0, 30, false, 1.2f,
                                       DecayMode::Tiebreak);
    assert(old_rank >= threshold);
    assert(memorylayer::similarity_score(0.90f, false, 1.2f) >= threshold);
    assert(memorylayer::similarity_score(0.8999f, false, 1.2f) < threshold);
    std::cout << "test_tiebreak_scoring_behavior PASSED\n";
}

void test_target_coverage_mapping() {
    using memorylayer::DecayMode;
    assert(memorylayer::calibrated_top_k(0.8f, DecayMode::Tiebreak) == 4);
    assert(memorylayer::calibrated_top_k(0.9f, DecayMode::Tiebreak) == 8);
    assert(memorylayer::calibrated_top_k(0.95f, DecayMode::Tiebreak) == 16);
    assert(memorylayer::calibrated_top_k(0.8f, DecayMode::Legacy) == 7);
    assert(memorylayer::calibrated_top_k(0.9f, DecayMode::Legacy) == 15);
    assert(memorylayer::calibrated_top_k(0.95f, DecayMode::Legacy) == 26);
    std::cout << "test_target_coverage_mapping PASSED\n";
}

int main() {
    test_rank_score_python_parity();
    test_tiebreak_scoring_behavior();
    test_target_coverage_mapping();
    std::cout << "All retrieval tests PASSED\n";
    return 0;
}
