"""Tests for the end-to-end prompt construction and grading."""

from __future__ import annotations

import numpy as np

from conformal_retrieval.endtoend import (
    FOOTER,
    HEADER,
    Setting,
    build_messages,
    is_correct,
    memory_block,
    normalize,
    paired_bootstrap,
    request_key,
    select,
    time_ago,
    truncate,
)
from conformal_retrieval.study import EmbeddedQuestion, Memory, Question


def question() -> EmbeddedQuestion:
    """Return three memories ranked 0 > 2 > 1 whose second is evidence."""
    memories = (
        Memory("first", "a", 1.0, False, False),
        Memory("second", "b", 48.0, True, False),
        Memory("third", "c", 2.0, False, False),
    )
    return EmbeddedQuestion(
        question=Question("q", "single-session-user", "Where?", False, memories),
        query_embedding=np.asarray([1.0, 0.0]),
        user_embeddings=np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
        assistant_embeddings=np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
    )


def test_truncate_and_time_ago_mirror_the_proxy() -> None:
    """Byte truncation appends '...' and ages use the proxy units."""
    assert truncate("abc", 3) == "abc"
    assert truncate("abcd", 3) == "abc..."
    assert truncate("aé", 2) == "a..."
    assert time_ago(7200) == "2h ago"
    assert time_ago(3 * 86400) == "3d ago"


def test_memory_block_format() -> None:
    """The block matches the system-mode header, lines and footer."""
    block = memory_block([Memory("u" * 5, "a", 48.0, True, False)], 3)
    assert block == HEADER + "- [2d ago] uuu... Response: a\n" + FOOTER
    assert memory_block([], 200) == ""


def test_select_ranks_and_oracle_keeps_evidence_only() -> None:
    """Top-k follows the tiebreak score; the oracle keeps only evidence."""
    item = question()
    top = select(item, Setting("k2", 2, 200))
    assert [memory.user_text for memory in top] == ["first", "third"]
    oracle = select(item, Setting("oracle", 0, 200, oracle=True))
    assert [memory.user_text for memory in oracle] == ["second"]


def test_build_messages_places_memories_in_the_system_prompt() -> None:
    """No-memory requests omit the block; others append it after the date."""
    item = question()
    plain = build_messages(item, Setting("none", 0, 0), "2023-05-30")
    assert HEADER not in plain[0]["content"]
    assert "Current date: 2023-05-30." in plain[0]["content"]
    assert plain[1] == {"role": "user", "content": "Where?"}
    with_memory = build_messages(item, Setting("k1", 1, 200), "2023-05-30")
    assert with_memory[0]["content"].endswith(FOOTER)
    assert request_key("m", plain) != request_key("m", with_memory)


def test_grading_matches_whole_normalized_words() -> None:
    """Punctuation and case are ignored, partial numbers are not matches."""
    assert normalize("  The $750, Paid!  ") == "the $750 paid"
    assert is_correct("$750", "You paid $750.")
    assert is_correct("@jessica_poole", "Her handle is @Jessica_Poole.")
    assert not is_correct("1", "You watched 15 videos.")
    assert not is_correct("", "anything")


def test_paired_bootstrap_difference() -> None:
    """The mean paired difference is exact and inside its interval."""
    result = paired_bootstrap(
        [True, True, False, True],
        [False, True, False, False],
        np.random.default_rng(0),
    )
    assert result["diff"] == 0.5
    assert result["low"] <= 0.5 <= result["high"]
