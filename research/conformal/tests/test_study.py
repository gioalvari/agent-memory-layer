"""Tests for pure conformal retrieval study functions."""

from __future__ import annotations

import numpy as np

from conformal_retrieval.study import (
    DECAY_DAYS,
    DECAY_FLOOR_08_SCORE,
    DECAY_TIEBREAK_SCORE,
    PRODUCTION_SCORE,
    RAW_SCORE,
    EmbeddedQuestion,
    Memory,
    Question,
    conformal_threshold,
    construct_set,
    decay,
    pair_memories,
    rank_quantile,
    score_memories,
)


def test_conformal_quantile_finite_sample_edges() -> None:
    """The lower quantile follows the specified finite-sample convention."""
    scores = [0.2, 0.8, 0.4]
    assert conformal_threshold(scores, 0.0) == float("-inf")
    assert conformal_threshold(scores, 0.24) == float("-inf")
    assert conformal_threshold(scores, 0.25) == 0.2
    assert conformal_threshold(scores, 0.50) == 0.4
    assert conformal_threshold(scores, 1.0) == 0.8


def test_construct_set_includes_the_threshold() -> None:
    """Thresholding retains all ties at the score boundary."""
    assert np.array_equal(construct_set(np.asarray([0.3, 0.2, 0.3]), 0.3), [0, 2])


def test_rank_quantile_uses_upper_finite_sample_order_statistic() -> None:
    """Rank conformal uses the corrected upper order statistic."""
    ranks = [1, 2, 4, 9]
    assert rank_quantile(ranks, 0.20) == 9
    assert rank_quantile(ranks, 0.40) == 4
    assert rank_quantile(ranks, 0.80) == 1


def test_score_variants_preserve_or_change_ranking_as_specified() -> None:
    """Raw, recency tie-break, and decay-floor scores follow their formulas."""
    question = Question(
        question_id="synthetic",
        question_type="multi-session",
        text="query",
        abstention=False,
        memories=(
            Memory("near", "near", 0.0, True, False),
            Memory("old", "old", 24.0 * 60.0, False, False),
        ),
    )
    embedded = EmbeddedQuestion(
        question=question,
        query_embedding=np.asarray([1.0, 0.0]),
        user_embeddings=np.asarray([[0.8, 0.6], [0.8, 0.6]]),
        assistant_embeddings=np.asarray([[0.0, 1.0], [0.0, 1.0]]),
    )
    scores = score_memories(embedded)
    assert scores[RAW_SCORE].tolist() == [0.8, 0.8]
    assert scores[DECAY_TIEBREAK_SCORE][0] > scores[DECAY_TIEBREAK_SCORE][1]
    assert np.allclose(scores[PRODUCTION_SCORE], [0.96, 0.48])
    assert np.allclose(scores[DECAY_FLOOR_08_SCORE], [0.96, 0.768])


def test_pair_memories_and_fallback_evidence() -> None:
    """User-assistant pairs receive turn labels or answer-session fallback labels."""
    record = {
        "question_id": "synthetic",
        "question_type": "single-session-user",
        "question": "query",
        "question_date": "2023/05/30 (Tue) 23:40",
        "answer_session_ids": ["answer"],
        "haystack_dates": ["2023/05/29 (Mon) 23:40", "2023/05/28 (Sun) 23:40"],
        "haystack_session_ids": ["answer", "other"],
        "haystack_sessions": [
            [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}],
            [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}],
        ],
    }
    paired = pair_memories(record)
    assert len(paired.memories) == 2
    assert paired.memories[0].evidence
    assert paired.memories[0].fallback_evidence
    assert not paired.memories[1].evidence
    assert paired.memories[0].age_hours == 24.0


def test_decay_matches_cpp_formula() -> None:
    """Decay is linear until its production floor."""
    assert decay(0.0) == 1.0
    assert decay(24.0 * DECAY_DAYS / 2.0) == 0.5
    assert decay(24.0 * DECAY_DAYS) == 0.5
    assert decay(24.0 * DECAY_DAYS * 10.0) == 0.5
    assert Memory("u", "a", 0.0, False, False).age_hours == 0.0
