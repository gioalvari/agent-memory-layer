"""Tests for the follow-up conformal policies and production token model."""

from __future__ import annotations

import math

import numpy as np
import pytest

from conformal_retrieval.followup import (
    CONTEXT_OVERHEAD,
    Profile,
    budget_policy,
    crc_k,
    format_time_ago,
    gap_policy,
    line_tokens,
    mondrian_edges,
    outcome,
    rank_policy,
    truncated_bytes,
    upper_quantile,
    visible_prefix,
)
from conformal_retrieval.study import Memory


def make_profile(
    scores: list[float], tokens: list[int], evidence: list[bool]
) -> Profile:
    """Build a profile whose inputs are already in ranked order."""
    flags = np.asarray(evidence)
    positions = np.flatnonzero(flags)
    return Profile(
        question_type="t",
        sorted_scores=np.asarray(scores),
        cumulative_tokens=np.cumsum(tokens),
        recall=np.cumsum(flags) / flags.sum(),
        best_rank=int(positions[0] + 1),
        last_rank=int(positions[-1] + 1),
        evidence_count=int(flags.sum()),
        top1_similarity=float(scores[0]),
    )


def test_upper_quantile_uses_finite_sample_rank_and_infinity() -> None:
    """The quantile is the ceil((1-a)(n+1))-th value, or inf when too small."""
    assert upper_quantile([3.0, 1.0, 2.0, 9.0], 0.2) == 9.0
    assert upper_quantile([3.0, 1.0, 2.0, 9.0], 0.4) == 3.0
    assert upper_quantile([1.0, 2.0], 0.1) == math.inf


def test_crc_k_matches_the_risk_bound() -> None:
    """CRC picks the first k with (n * mean loss + 1) / (n + 1) <= alpha."""
    losses = np.asarray([[1.0, 0.5, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    assert crc_k(losses, 0.5) == 2
    assert crc_k(losses, 0.2) == 3
    assert crc_k(losses, 0.1) == 3


def test_truncation_and_line_tokens_mirror_the_proxy() -> None:
    """Truncation appends '...' and the line uses bytes / 4 + 1."""
    assert truncated_bytes("a" * 200) == 200
    assert truncated_bytes("a" * 201) == 203
    assert truncated_bytes("é" * 150) == 203
    memory = Memory("u" * 500, "a" * 10, 49.0, True, False)
    line = "- [2d ago] " + "u" * 203 + " Response: " + "a" * 10 + "\n"
    assert line_tokens(memory) == len(line) // 4 + 1
    assert line_tokens(memory, 1000) > line_tokens(memory)


def test_format_time_ago_boundaries() -> None:
    """Relative ages match the C++ minute, hour, and day formatting."""
    assert format_time_ago(59) == "just now"
    assert format_time_ago(3599) == "59m ago"
    assert format_time_ago(86399) == "23h ago"
    assert format_time_ago(86400 * 12.5) == "12d ago"


def test_visible_prefix_is_byte_truncated_and_lowercased() -> None:
    """Visibility uses the same byte limit and ignores split code points."""
    assert visible_prefix("ABé", 3) == "ab"


def test_outcome_counts_coverage_recall_and_overhead() -> None:
    """A size-2 set covers the first of two evidence memories."""
    profile = make_profile([0.9, 0.8, 0.7], [10, 20, 30], [False, True, True])
    result = outcome(profile, 2)
    assert result["coverage"] == 1.0
    assert result["recall"] == 0.5
    assert result["all"] == 0.0
    assert result["tokens"] == 30 + CONTEXT_OVERHEAD
    assert outcome(profile, 0)["tokens"] == 0.0
    assert outcome(profile, 10)["size"] == 3.0


def test_policies_calibrate_rank_budget_and_gap() -> None:
    """Each policy reaches the calibration quantile of its own score."""
    profiles = [
        make_profile([0.9, 0.8, 0.5], [10, 10, 10], [True, False, False]),
        make_profile([0.9, 0.7, 0.6], [10, 10, 10], [False, True, False]),
        make_profile([0.9, 0.8, 0.4], [10, 10, 10], [False, False, True]),
    ]
    size_of, k = rank_policy(profiles, 0.25)
    assert k == 3.0 and size_of(profiles[0]) == 3
    size_of, budget = budget_policy(profiles, 0.25)
    assert budget == 30 + CONTEXT_OVERHEAD
    assert [size_of(p) for p in profiles] == [3, 3, 3]
    size_of, gap = gap_policy(profiles, 0.25)
    assert gap == pytest.approx(0.5)
    assert size_of(profiles[1]) == 3


def test_mondrian_edges_split_into_terciles() -> None:
    """Two interior edges split the feature distribution into three bins."""
    edges = mondrian_edges([0.0, 1.0, 2.0, 3.0])
    assert len(edges) == 2
    assert list(np.searchsorted(edges, [0.0, 1.5, 3.0])) == [0, 1, 2]
