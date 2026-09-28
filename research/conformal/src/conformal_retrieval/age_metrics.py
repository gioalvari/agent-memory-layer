"""Produce evidence-age coverage summaries from cached study embeddings."""

from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

from conformal_retrieval.study import (
    ALPHAS,
    PRODUCTION_SCORE,
    REPETITIONS,
    RESULTS_PATH,
    SEED,
    Question,
    conformal_threshold,
    embedded_questions,
    pair_memories,
    score_memories,
    selected_metrics,
)

AGE_PATH = RESULTS_PATH.with_name("conformal_age_metrics.json")


def age_bucket(question: Question) -> str:
    """Assign a question to the youngest evidence-memory age bucket."""
    age_days = min(
        memory.age_hours / 24.0 for memory in question.memories if memory.evidence
    )
    if age_days <= 7:
        return "<=7d"
    if age_days <= 30:
        return "8-30d"
    return ">30d"


def main() -> None:
    """Calculate production-conformal and top-five metrics by evidence age."""
    records = json.loads(
        RESULTS_PATH.with_name("longmemeval_s_subset.json").read_text()
    )
    questions = embedded_questions([pair_memories(record) for record in records])
    answerable = [
        question for question in questions if not question.question.abstention
    ]
    scores = {
        question.question.question_id: score_memories(question)[PRODUCTION_SCORE]
        for question in answerable
    }
    values: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    rng = np.random.default_rng(SEED)
    for _ in range(REPETITIONS):
        indices = rng.permutation(len(answerable))
        calibration = [answerable[int(index)] for index in indices[: len(indices) // 2]]
        test = [answerable[int(index)] for index in indices[len(indices) // 2 :]]
        for alpha in ALPHAS:
            threshold = conformal_threshold(
                [
                    float(
                        np.max(
                            scores[question.question.question_id][
                                np.asarray(
                                    [
                                        memory.evidence
                                        for memory in question.question.memories
                                    ]
                                )
                            ]
                        )
                    )
                    for question in calibration
                ],
                alpha,
            )
            for method in ("conformal_production", "top_5"):
                grouped: dict[str, list[dict[str, float | bool]]] = defaultdict(list)
                for question in test:
                    score = scores[question.question.question_id]
                    selected = (
                        np.flatnonzero(score >= threshold)
                        if method == "conformal_production"
                        else np.argsort(-score)[:5]
                    )
                    grouped[age_bucket(question.question)].append(
                        selected_metrics(question, selected)
                    )
                for bucket, rows in grouped.items():
                    values[method][str(alpha)][bucket].append(
                        float(np.mean([row["any"] for row in rows]))
                    )
    output = {
        method: {
            alpha: {
                bucket: {"coverage": float(np.mean(repetitions)), "n": len(repetitions)}
                for bucket, repetitions in buckets.items()
            }
            for alpha, buckets in alphas.items()
        }
        for method, alphas in values.items()
    }
    AGE_PATH.write_text(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
