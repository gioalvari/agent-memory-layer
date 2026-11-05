"""Tests for pure real-proxy validation helpers."""

from __future__ import annotations

import numpy as np

from conformal_retrieval.proxy_check import compare, memory_lines, request_key, unit


def test_unit_writes_normalized_float32_bytes() -> None:
    """Stored vectors are L2-normalized float32 blobs."""
    stored = np.frombuffer(unit(np.asarray([3.0, 4.0])), dtype=np.float32)
    assert np.allclose(stored, [0.6, 0.8])


def test_request_key_uses_date_and_question() -> None:
    """The key remains stable when injected memory context changes."""
    messages = [
        {"role": "system", "content": "Prompt\nCurrent date: 2023-01-01."},
        {"role": "user", "content": "Question"},
    ]
    injected = [
        {
            "role": "system",
            "content": "Prompt\nCurrent date: 2023-01-01.\n\n<memory context>",
        },
        {"role": "user", "content": "Question"},
    ]

    assert request_key(messages) == request_key(injected)


def test_compare_distinguishes_order_from_set_equality() -> None:
    """A reordered memory context preserves the set but not exact prompt match."""
    expected = [
        {
            "role": "system",
            "content": "Prompt\n- [1h ago] first\n- [2h ago] second",
        }
    ]
    actual = [
        {
            "role": "system",
            "content": "Prompt\n- [2h ago] second\n- [1h ago] first",
        }
    ]

    assert memory_lines(expected[0]["content"]) == ["first", "second"]
    assert compare(expected, actual) == {
        "exact": False,
        "same_lines": False,
        "same_set": True,
        "expected_lines": 2,
        "actual_lines": 2,
    }
