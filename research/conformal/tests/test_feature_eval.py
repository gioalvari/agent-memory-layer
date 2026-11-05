"""Tests for pure feature-evaluation helpers."""

from __future__ import annotations

import numpy as np

from conformal_retrieval.feature_eval import (
    FACT_HEADER,
    MEMORY_FOOTER,
    MEMORY_HEADER,
    context_tokens,
    evidence_recall,
    fact_count,
    injected_lines,
    output_name,
    request_key,
    summarize_rows,
)
from conformal_retrieval.study import EmbeddedQuestion, Memory, Question


def item() -> EmbeddedQuestion:
    """Return a question with one evidence memory."""
    memories = (
        Memory("evidence user", "evidence assistant", 1.0, True, False),
        Memory("other user", "other assistant", 2.0, False, False),
    )
    return EmbeddedQuestion(
        Question("question", "temporal", "What happened?", False, memories),
        np.asarray([1.0, 0.0]),
        np.asarray([[1.0, 0.0], [0.0, 1.0]]),
        np.asarray([[1.0, 0.0], [0.0, 1.0]]),
    )


def test_request_key_survives_injection_and_separates_same_questions() -> None:
    """The key ignores injected context but incorporates the current date."""
    bare = [
        {"role": "system", "content": "Prompt\nCurrent date: 2023-01-01."},
        {"role": "user", "content": "Same question"},
    ]
    injected = [
        {
            "role": "system",
            "content": (
                "Prompt\nCurrent date: 2023-01-01.\n\n"
                f"{MEMORY_HEADER}\n- [1h ago] memory\n{MEMORY_FOOTER}"
            ),
        },
        {"role": "user", "content": "Same question"},
    ]
    different_date = [
        {"role": "system", "content": "Prompt\nCurrent date: 2023-01-02."},
        {"role": "user", "content": "Same question"},
    ]

    assert request_key(bare) == request_key(injected)
    assert request_key(bare) != request_key(different_date)


def test_context_parsing_counts_only_injected_content() -> None:
    """Memory lines, context tokens, and fact bullets follow proxy formatting."""
    system = (
        "Prompt\n"
        f"{MEMORY_HEADER}\n"
        "- [1h ago] user Response: assistant\n"
        f"{FACT_HEADER}\n"
        "- durable fact\n"
        f"{MEMORY_FOOTER}"
    )

    assert injected_lines(system) == ["user Response: assistant"]
    assert (
        context_tokens(system)
        == len(system[system.find(MEMORY_HEADER) :].encode()) // 4 + 1
    )
    assert context_tokens("Prompt only") == 0
    assert fact_count(system) == 1


def test_evidence_recall_and_safe_output_name() -> None:
    """Evidence matching returns coverage and fractional recall."""
    question = item()
    assert evidence_recall(question, ["evidence user Response: unrelated"]) == (
        True,
        1.0,
    )
    assert evidence_recall(question, ["other user Response: other assistant"]) == (
        False,
        0.0,
    )
    assert output_name("a/b c") == "a_b_c"


def test_summarize_rows_reports_paired_answer_accuracy() -> None:
    """Overall and type summaries include deterministic paired QA comparison."""
    baseline = [
        {
            "question_id": "a",
            "question_type": "type",
            "answerable": True,
            "session_recall": True,
            "evidence_recall": 1.0,
            "injected_tokens": 10,
            "injected_lines": 1,
            "facts": 0,
            "correct": False,
        }
    ]
    rows = [{**baseline[0], "correct": True}]

    summary = summarize_rows(rows, baseline)["overall"]
    assert summary["qa_accuracy"] == 1.0
    assert summary["qa_vs_baseline"]["diff"] == 1.0
    assert summary["session_recall_at_k"] == 1.0
