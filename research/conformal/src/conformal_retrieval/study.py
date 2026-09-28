"""Dataset construction, scoring, and split-conformal evaluation."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import logging
import math
import sqlite3
import statistics
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = ROOT / "data"
RAW_DIR = Path("/tmp/conformal")
SOURCE_DATA = RAW_DIR / "longmemeval-cleaned" / "longmemeval_s_cleaned.json"
CACHE_PATH = DATA_DIR / "embedding_cache.sqlite"
QUESTIONS_PATH = DATA_DIR / "longmemeval_s_subset.json"
RESULTS_PATH = DATA_DIR / "conformal_results.json"
PARETO_PATH = DATA_DIR / "conformal_pareto.csv"
EMBEDDING_URL = "http://127.0.0.1:8490/v1/embeddings"
SEED = 20260925
SUBSET_SIZE = 500
REPETITIONS = 200
ALPHAS = (0.05, 0.10, 0.20)
TOP_KS = tuple(range(1, 11)) + (15, 20, 30)
DECAY_DAYS = 30
AGENT_BOOST = 1.2
RAW_SCORE = "raw"
PRODUCTION_SCORE = "production"
DECAY_TIEBREAK_SCORE = "decay_tiebreak"
DECAY_FLOOR_08_SCORE = "decay_floor_0.8"
SCORE_VARIANTS = (
    RAW_SCORE,
    PRODUCTION_SCORE,
    DECAY_TIEBREAK_SCORE,
    DECAY_FLOOR_08_SCORE,
)
# The model's embedded metadata caps the 8192 server request at the production
# worker's 2048-token context. This conservative character equivalent keeps
# OpenAI-compatible endpoint inputs below that hard ceiling.
MAX_EMBED_CHARS = 1500


@dataclass(frozen=True)
class Memory:
    """One retrievable user-assistant interaction."""

    user_text: str
    assistant_text: str
    age_hours: float
    evidence: bool
    fallback_evidence: bool


@dataclass(frozen=True)
class Question:
    """A benchmark query and its per-query memory haystack."""

    question_id: str
    question_type: str
    text: str
    abstention: bool
    memories: tuple[Memory, ...]


@dataclass(frozen=True)
class EmbeddedQuestion:
    """A benchmark query whose texts have cached embeddings."""

    question: Question
    query_embedding: np.ndarray
    user_embeddings: np.ndarray
    assistant_embeddings: np.ndarray


def parse_date(value: str) -> dt.datetime:
    """Parse the LongMemEval timestamp format.

    Parameters
    ----------
    value
        A date such as ``2023/05/30 (Tue) 23:40``.

    Returns
    -------
    datetime.datetime
        Naive timestamp in the dataset's common timezone.
    """
    return dt.datetime.strptime(
        value.replace(" (", " ").replace(")", ""), "%Y/%m/%d %a %H:%M"
    )


def pair_memories(record: dict[str, Any]) -> Question:
    """Pair user turns with following assistant turns and label evidence.

    Parameters
    ----------
    record
        A LongMemEval-S record.

    Returns
    -------
    Question
        Query metadata and its paired memories.
    """
    question_date = parse_date(str(record["question_date"]))
    answer_sessions = {str(item) for item in record.get("answer_session_ids", [])}
    sessions = record["haystack_sessions"]
    session_dates = record["haystack_dates"]
    session_ids = record["haystack_session_ids"]
    has_answer = any(
        bool(turn.get("has_answer")) for session in sessions for turn in session
    )
    use_fallback = not has_answer and bool(answer_sessions)
    memories: list[Memory] = []
    for session, date_text, session_id in zip(sessions, session_dates, session_ids):
        age_hours = (
            question_date - parse_date(str(date_text))
        ).total_seconds() / 3600.0
        for current, following in zip(session, session[1:]):
            if current.get("role") != "user" or following.get("role") != "assistant":
                continue
            evidence = bool(current.get("has_answer")) or bool(
                following.get("has_answer")
            )
            fallback_evidence = use_fallback and str(session_id) in answer_sessions
            memories.append(
                Memory(
                    user_text=str(current.get("content", "")),
                    assistant_text=str(following.get("content", "")),
                    age_hours=age_hours,
                    evidence=evidence or fallback_evidence,
                    fallback_evidence=fallback_evidence,
                )
            )
    return Question(
        question_id=str(record["question_id"]),
        question_type=str(record["question_type"]),
        text=str(record["question"]),
        abstention=str(record["question_id"]).endswith("_abs"),
        memories=tuple(memories),
    )


def decay(age_hours: float, decay_days: int = DECAY_DAYS, floor: float = 0.5) -> float:
    """Return the C++ production recency multiplier.

    Parameters
    ----------
    age_hours
        Hours between a memory session and its query.
    decay_days
        Linear-decay horizon in days.

    Returns
    -------
    float
        The multiplier, floored at ``floor``.
    """
    return max(floor, 1.0 - age_hours / (24.0 * decay_days))


def normalize(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize one vector or a matrix of row vectors."""
    denominator = np.linalg.norm(vectors, axis=-1, keepdims=True)
    result: np.ndarray = np.divide(
        vectors, denominator, out=np.zeros_like(vectors), where=denominator != 0
    )
    return result


def score_memories(question: EmbeddedQuestion) -> dict[str, np.ndarray]:
    """Calculate every requested score variant for a question.

    The tie-break variant preserves cosine ordering except for scores within
    0.001, allowing recency to resolve otherwise near-identical candidates.
    """
    query = normalize(question.query_embedding)
    users = normalize(question.user_embeddings)
    assistants = normalize(question.assistant_embeddings)
    raw = np.maximum(users @ query, assistants @ query)
    recency = np.asarray(
        [decay(memory.age_hours) for memory in question.question.memories]
    )
    production = raw * recency * AGENT_BOOST
    floor_08 = np.asarray(
        [decay(memory.age_hours, floor=0.8) for memory in question.question.memories]
    )
    return {
        RAW_SCORE: raw,
        PRODUCTION_SCORE: production,
        DECAY_TIEBREAK_SCORE: raw + 1e-3 * recency,
        DECAY_FLOOR_08_SCORE: raw * floor_08 * AGENT_BOOST,
    }


def conformal_threshold(scores: Sequence[float], alpha: float) -> float:
    """Compute the lower split-conformal score threshold.

    Parameters
    ----------
    scores
        Calibration best-evidence scores.
    alpha
        Target miscoverage rate.

    Returns
    -------
    float
        The ``floor(alpha * (n + 1))``-th smallest score, or negative infinity.
    """
    index = min(math.floor(alpha * (len(scores) + 1)), len(scores))
    if index < 1:
        return float("-inf")
    return float(np.sort(np.asarray(scores))[index - 1])


def construct_set(scores: np.ndarray, threshold: float) -> np.ndarray:
    """Return the indices meeting a score threshold."""
    return np.flatnonzero(scores >= threshold)


def rank_quantile(best_ranks: Sequence[int], alpha: float) -> int:
    """Return the split-conformal rank limit for best-evidence ranks."""
    index = math.ceil((1.0 - alpha) * (len(best_ranks) + 1)) - 1
    index = min(max(index, 0), len(best_ranks) - 1)
    return int(np.sort(np.asarray(best_ranks))[index])


def token_count(memory: Memory) -> int:
    """Estimate injected tokens using production's chars-over-four heuristic."""
    return len(memory.user_text + memory.assistant_text) // 4 + 1


def cache_key(text: str) -> str:
    """Return the stable SHA-256 key used by the embedding cache."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class EmbeddingCache:
    """SQLite-backed, text-hash keyed embedding cache."""

    def __init__(self, path: Path) -> None:
        """Open or create the local cache at ``path``."""
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS embeddings "
            "(key TEXT PRIMARY KEY, vector BLOB NOT NULL, dim INTEGER NOT NULL)"
        )

    def get(self, text: str) -> np.ndarray | None:
        """Return a cached vector for ``text``, when available."""
        row = self.connection.execute(
            "SELECT vector, dim FROM embeddings WHERE key = ?", (cache_key(text),)
        ).fetchone()
        if row is None:
            return None
        vector, dimension = row
        return np.frombuffer(vector, dtype=np.float32).copy().reshape(int(dimension))

    def put_many(self, texts: Sequence[str], vectors: Sequence[np.ndarray]) -> None:
        """Persist vectors for their corresponding texts."""
        rows = [
            (cache_key(text), vector.astype(np.float32).tobytes(), int(vector.size))
            for text, vector in zip(texts, vectors)
        ]
        self.connection.executemany(
            "INSERT OR REPLACE INTO embeddings (key, vector, dim) VALUES (?, ?, ?)",
            rows,
        )
        self.connection.commit()

    def close(self) -> None:
        """Close the underlying database connection."""
        self.connection.close()


def request_embeddings(texts: Sequence[str]) -> list[np.ndarray]:
    """Request a batch of embeddings from the local llama-server endpoint."""
    request = urllib.request.Request(
        EMBEDDING_URL,
        # The source worker has no explicit truncation and its 2048-token llama
        # context would return an empty embedding for an overlong input. The
        # OpenAI-compatible endpoint ignores its truncate field for this model,
        # so conservatively prefix-truncate before submission and disclose the
        # unavoidable difference in the report.
        data=json.dumps(
            {
                "input": [text[:MAX_EMBED_CHARS] for text in texts],
                "model": "nomic-embed-text",
            }
        ).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            payload = json.loads(response.read())
    except urllib.error.URLError as error:
        raise RuntimeError(
            f"Embedding endpoint unavailable at {EMBEDDING_URL}"
        ) from error
    return [np.asarray(item["embedding"], dtype=np.float32) for item in payload["data"]]


def embed_texts(texts: Iterable[str]) -> dict[str, np.ndarray]:
    """Fetch all unique texts, first using the persistent embedding cache."""
    unique = list(dict.fromkeys(texts))
    cache = EmbeddingCache(CACHE_PATH)
    result: dict[str, np.ndarray] = {}
    missing: list[str] = []
    for text in unique:
        cached = cache.get(text)
        if cached is None:
            missing.append(text)
        else:
            result[text] = cached
    LOGGER.info("Embedding cache: %d hit(s), %d missing", len(result), len(missing))
    batch_size = 16
    for start in range(0, len(missing), batch_size):
        batch = missing[start : start + batch_size]
        vectors = request_embeddings(batch)
        if len(vectors) != len(batch):
            raise RuntimeError("Embedding API returned a different number of vectors")
        cache.put_many(batch, vectors)
        result.update(zip(batch, vectors))
        LOGGER.info(
            "Embedded %d/%d uncached texts",
            min(start + len(batch), len(missing)),
            len(missing),
        )
    cache.close()
    return result


def deterministic_subset(
    records: list[dict[str, Any]], size: int
) -> list[dict[str, Any]]:
    """Select a stable, hash-ranked subset without depending on file order."""
    return sorted(
        records,
        key=lambda record: hashlib.sha256(
            (str(SEED) + str(record["question_id"])).encode()
        ).hexdigest(),
    )[:size]


def embedded_questions(questions: Sequence[Question]) -> list[EmbeddedQuestion]:
    """Load cached embeddings and assemble arrays per benchmark question."""
    texts = [question.text for question in questions]
    texts.extend(
        text
        for question in questions
        for memory in question.memories
        for text in (memory.user_text, memory.assistant_text)
    )
    vectors = embed_texts(texts)
    return [
        EmbeddedQuestion(
            question=question,
            query_embedding=vectors[question.text],
            user_embeddings=np.asarray(
                [vectors[memory.user_text] for memory in question.memories]
            ),
            assistant_embeddings=np.asarray(
                [vectors[memory.assistant_text] for memory in question.memories]
            ),
        )
        for question in questions
    ]


def selected_metrics(
    question: EmbeddedQuestion, selected: np.ndarray
) -> dict[str, float | bool]:
    """Calculate coverage, set size, and token metrics for one selected set."""
    evidence = np.asarray([memory.evidence for memory in question.question.memories])
    selected_memories = [question.question.memories[int(index)] for index in selected]
    return {
        "any": bool(np.any(evidence[selected])),
        "all": bool(np.all(np.isin(np.flatnonzero(evidence), selected))),
        "all_eligible": bool(np.count_nonzero(evidence) >= 2),
        "size": float(len(selected)),
        "tokens": float(sum(token_count(memory) for memory in selected_memories)),
        "empty": len(selected) == 0,
    }


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    """Return a two-sided 95% Wilson confidence interval."""
    if total == 0:
        return (float("nan"), float("nan"))
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + z**2 / total
    center = (proportion + z**2 / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt(proportion * (1.0 - proportion) / total + z**2 / (4.0 * total**2))
        / denominator
    )
    return center - margin, center + margin


def summarize(rows: Sequence[dict[str, float | bool]]) -> dict[str, float]:
    """Aggregate metrics across a single test fold."""

    def values(name: str) -> list[float]:
        return [float(row[name]) for row in rows]

    any_values = values("any")
    all_rows = [row for row in rows if bool(row["all_eligible"])]
    return {
        "coverage": float(statistics.mean(any_values)),
        "all_coverage": float(statistics.mean([float(row["all"]) for row in all_rows]))
        if all_rows
        else float("nan"),
        "mean_size": float(statistics.mean(values("size"))),
        "median_size": float(statistics.median(values("size"))),
        "p90_size": float(np.quantile(values("size"), 0.9)),
        "mean_tokens": float(statistics.mean(values("tokens"))),
        "empty_rate": float(statistics.mean(values("empty"))),
        "wilson_low": wilson_interval(int(sum(any_values)), len(rows))[0],
        "wilson_high": wilson_interval(int(sum(any_values)), len(rows))[1],
    }


def evaluate_method(
    calibration: Sequence[EmbeddedQuestion],
    test: Sequence[EmbeddedQuestion],
    alpha: float,
    method: str,
    score_variant: str,
    scores_by_id: Mapping[str, Mapping[str, np.ndarray]],
) -> tuple[dict[str, float], dict[str, dict[str, float]], int | None]:
    """Fit one score-specific method and evaluate its test questions."""
    cal_scores = [
        scores_by_id[item.question.question_id][score_variant] for item in calibration
    ]
    test_scores = [
        scores_by_id[item.question.question_id][score_variant] for item in test
    ]
    if method == "absolute":
        threshold = conformal_threshold(
            [
                float(np.max(score[evidence]))
                for score, question in zip(cal_scores, calibration)
                if (
                    evidence := np.asarray(
                        [memory.evidence for memory in question.question.memories]
                    )
                ).any()
            ],
            alpha,
        )
        selections = [construct_set(score, threshold) for score in test_scores]
        k: int | None = None
    elif method == "rank":
        ranks: list[int] = []
        for score, question in zip(cal_scores, calibration):
            evidence = np.asarray(
                [memory.evidence for memory in question.question.memories]
            )
            order = np.argsort(-score, kind="stable")
            ranks.append(int(np.min(np.flatnonzero(evidence[order])) + 1))
        k = rank_quantile(ranks, alpha)
        selections = [np.argsort(-score, kind="stable")[:k] for score in test_scores]
    elif method.startswith("top_"):
        k = int(method.removeprefix("top_"))
        selections = [np.argsort(-score, kind="stable")[:k] for score in test_scores]
    else:
        raise ValueError(f"Unknown method: {method}")
    rows = [
        selected_metrics(question, selected)
        for question, selected in zip(test, selections)
    ]
    groups: dict[str, list[dict[str, float | bool]]] = defaultdict(list)
    for question, row in zip(test, rows):
        groups[question.question.question_type].append(row)
    return (
        summarize(rows),
        {name: summarize(group) for name, group in groups.items()},
        k,
    )


def mean_and_range(values: Sequence[float]) -> dict[str, float]:
    """Return a mean and percentile range over repeated split results."""
    return {
        "mean": float(np.mean(values)),
        "p05": float(np.quantile(values, 0.05)),
        "p95": float(np.quantile(values, 0.95)),
    }


def auc(labels: Sequence[bool], scores: Sequence[float]) -> float:
    """Compute AUROC using average ranks, without an additional dependency."""
    positive = sum(labels)
    negative = len(labels) - positive
    if positive == 0 or negative == 0:
        return float("nan")
    order = np.argsort(np.asarray(scores))
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    sorted_scores = np.asarray(scores)[order]
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return float(
        (np.sum(ranks[np.asarray(labels)]) - positive * (positive + 1) / 2.0)
        / (positive * negative)
    )


def evidence_age_bucket(question: EmbeddedQuestion) -> str:
    """Return the youngest labeled-evidence age bucket for one question."""
    ages = [
        memory.age_hours / 24.0
        for memory in question.question.memories
        if memory.evidence
    ]
    youngest = min(ages)
    if youngest <= 7:
        return "<=7d"
    if youngest <= 30:
        return "8-30d"
    return ">30d"


def direct_top_k_analysis(
    questions: Sequence[EmbeddedQuestion],
    scores_by_id: Mapping[str, Mapping[str, np.ndarray]],
) -> dict[str, Any]:
    """Report full-sample score-variant coverage by type and evidence age."""
    analysis: dict[str, Any] = {"by_type": {}, "by_age": {}}
    for score_variant in SCORE_VARIANTS:
        analysis["by_type"][score_variant] = {}
        analysis["by_age"][score_variant] = {}
        for k in (5, 10):
            rows_by_type: dict[str, list[dict[str, float | bool]]] = defaultdict(list)
            rows_by_age: dict[str, list[dict[str, float | bool]]] = defaultdict(list)
            for question in questions:
                score = scores_by_id[question.question.question_id][score_variant]
                selected = np.argsort(-score, kind="stable")[:k]
                row = selected_metrics(question, selected)
                rows_by_type[question.question.question_type].append(row)
                rows_by_age[evidence_age_bucket(question)].append(row)
            for name, rows in rows_by_type.items():
                analysis["by_type"][score_variant].setdefault(name, {"n": len(rows)})[
                    f"top_{k}"
                ] = summarize(rows)
            for name, rows in rows_by_age.items():
                analysis["by_age"][score_variant].setdefault(name, {"n": len(rows)})[
                    f"top_{k}"
                ] = summarize(rows)
    return analysis


def mondrian_rank_evaluation(
    calibration: Sequence[EmbeddedQuestion],
    test: Sequence[EmbeddedQuestion],
    alpha: float,
    score_variant: str,
    scores_by_id: Mapping[str, Mapping[str, np.ndarray]],
) -> dict[str, tuple[dict[str, float], int]]:
    """Evaluate type-conditional rank calibration where 30 labels are available."""
    calibration_by_type: dict[str, list[EmbeddedQuestion]] = defaultdict(list)
    test_by_type: dict[str, list[EmbeddedQuestion]] = defaultdict(list)
    for question in calibration:
        calibration_by_type[question.question.question_type].append(question)
    for question in test:
        test_by_type[question.question.question_type].append(question)
    results: dict[str, tuple[dict[str, float], int]] = {}
    for question_type, cal_questions in calibration_by_type.items():
        if len(cal_questions) < 30 or not test_by_type[question_type]:
            continue
        ranks: list[int] = []
        for question in cal_questions:
            score = scores_by_id[question.question.question_id][score_variant]
            evidence = np.asarray(
                [memory.evidence for memory in question.question.memories]
            )
            order = np.argsort(-score, kind="stable")
            ranks.append(int(np.min(np.flatnonzero(evidence[order])) + 1))
        k = rank_quantile(ranks, alpha)
        rows = []
        for question in test_by_type[question_type]:
            score = scores_by_id[question.question.question_id][score_variant]
            rows.append(
                selected_metrics(question, np.argsort(-score, kind="stable")[:k])
            )
        results[question_type] = (summarize(rows), k)
    return results


def run_study(questions: Sequence[EmbeddedQuestion]) -> dict[str, Any]:
    """Execute repeated split-conformal and fixed-rank evaluations."""
    answerable = [
        question
        for question in questions
        if not question.question.abstention
        and any(memory.evidence for memory in question.question.memories)
    ]
    abstentions = [question for question in questions if question.question.abstention]
    unlabeled = [
        question
        for question in questions
        if not question.question.abstention
        and not any(memory.evidence for memory in question.question.memories)
    ]
    scores_by_id = {
        question.question.question_id: score_memories(question)
        for question in questions
    }
    methods_by_variant = {
        score_variant: (
            ["absolute", "rank"]
            if score_variant in {RAW_SCORE, PRODUCTION_SCORE}
            else ["rank"]
        )
        + [f"top_{k}" for k in TOP_KS]
        for score_variant in SCORE_VARIANTS
    }
    observations: dict[str, dict[str, dict[str, list[dict[str, float]]]]] = {
        variant: {method: {str(alpha): [] for alpha in ALPHAS} for method in methods}
        for variant, methods in methods_by_variant.items()
    }
    k_observations: dict[str, dict[str, dict[str, list[float]]]] = {
        variant: {method: {str(alpha): [] for alpha in ALPHAS} for method in methods}
        for variant, methods in methods_by_variant.items()
    }
    mondrian_observations: dict[str, dict[str, dict[str, list[dict[str, float]]]]] = (
        defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    )
    mondrian_k_observations: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    rng = np.random.default_rng(SEED)
    for repetition in range(REPETITIONS):
        indices = rng.permutation(len(answerable))
        calibration = [answerable[int(index)] for index in indices[: len(indices) // 2]]
        test = [answerable[int(index)] for index in indices[len(indices) // 2 :]]
        for alpha in ALPHAS:
            for score_variant, methods in methods_by_variant.items():
                for method in methods:
                    aggregate, _, k = evaluate_method(
                        calibration, test, alpha, method, score_variant, scores_by_id
                    )
                    observations[score_variant][method][str(alpha)].append(aggregate)
                    if k is not None:
                        k_observations[score_variant][method][str(alpha)].append(
                            float(k)
                        )
                for question_type, (metrics, k) in mondrian_rank_evaluation(
                    calibration, test, alpha, score_variant, scores_by_id
                ).items():
                    mondrian_observations[score_variant][question_type][
                        str(alpha)
                    ].append(metrics)
                    mondrian_k_observations[score_variant][question_type][
                        str(alpha)
                    ].append(float(k))
        LOGGER.info("Completed split %d/%d", repetition + 1, REPETITIONS)
    results: dict[str, Any] = {
        "methods": {},
        "mondrian_rank": {},
        "metadata": {},
        "stratified_top_k": direct_top_k_analysis(answerable, scores_by_id),
    }
    for score_variant, by_method in observations.items():
        results["methods"][score_variant] = {}
        for method, by_alpha in by_method.items():
            results["methods"][score_variant][method] = {}
            for alpha_text, repetition_metrics in by_alpha.items():
                summary = {
                    metric: mean_and_range(
                        [item[metric] for item in repetition_metrics]
                    )
                    for metric in repetition_metrics[0]
                }
                if k_values := k_observations[score_variant][method][alpha_text]:
                    summary["k"] = mean_and_range(k_values)
                    summary["k"]["median"] = float(statistics.median(k_values))
                results["methods"][score_variant][method][alpha_text] = summary
    for score_variant, by_type in mondrian_observations.items():
        results["mondrian_rank"][score_variant] = {}
        for question_type, by_alpha in by_type.items():
            results["mondrian_rank"][score_variant][question_type] = {
                "n": sum(
                    question.question.question_type == question_type
                    for question in answerable
                ),
                "alphas": {},
            }
            for alpha_text, values in by_alpha.items():
                summary = {
                    metric: mean_and_range([item[metric] for item in values])
                    for metric in values[0]
                }
                k_values = mondrian_k_observations[score_variant][question_type][
                    alpha_text
                ]
                summary["k"] = mean_and_range(k_values)
                summary["k"]["median"] = float(statistics.median(k_values))
                results["mondrian_rank"][score_variant][question_type]["alphas"][
                    alpha_text
                ] = summary
    max_scores = [
        float(np.max(scores_by_id[question.question.question_id][PRODUCTION_SCORE]))
        for question in questions
    ]
    labels = [question.question.abstention for question in questions]
    abstention_stats = {
        "n": len(abstentions),
        "answerable_n": len(answerable),
        "abstention_max_score_median": float(
            np.median([score for score, label in zip(max_scores, labels) if label])
        ),
        "answerable_max_score_median": float(
            np.median([score for score, label in zip(max_scores, labels) if not label])
        ),
        "abstention_auroc_higher_score_is_answerable": auc(
            [not label for label in labels], max_scores
        ),
    }
    for alpha in ALPHAS:
        conformal_stats = []
        for repetition in range(REPETITIONS):
            # Recreate the threshold using the same deterministic split stream.
            split_rng = np.random.default_rng(SEED)
            for _ in range(repetition + 1):
                split_indices = split_rng.permutation(len(answerable))
            calibration = [
                answerable[int(index)]
                for index in split_indices[: len(split_indices) // 2]
            ]
            threshold = conformal_threshold(
                [
                    float(
                        np.max(
                            scores_by_id[question.question.question_id][
                                PRODUCTION_SCORE
                            ][
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
            conformal_stats.extend(
                [
                    bool(
                        np.any(
                            scores_by_id[question.question.question_id][
                                PRODUCTION_SCORE
                            ]
                            >= threshold
                        )
                    )
                    for question in abstentions
                ]
            )
        abstention_stats[f"nonempty_rate_alpha_{alpha}"] = float(
            np.mean(conformal_stats)
        )
    results["abstention"] = abstention_stats
    results["metadata"] = {
        "seed": SEED,
        "repetitions": REPETITIONS,
        "answerable_questions": len(answerable),
        "abstention_questions": len(abstentions),
        "unlabeled_questions": len(unlabeled),
        "score_variants": list(SCORE_VARIANTS),
    }
    return results


def dataset_stats(questions: Sequence[Question]) -> dict[str, Any]:
    """Calculate required dataset-label and haystack statistics."""
    answerable = [question for question in questions if not question.abstention]
    labeled_answerable = [
        question
        for question in answerable
        if any(memory.evidence for memory in question.memories)
    ]
    memories = [len(question.memories) for question in questions]
    evidence = [
        sum(memory.evidence for memory in question.memories)
        for question in labeled_answerable
    ]
    fallback_questions = sum(
        any(memory.fallback_evidence for memory in question.memories)
        for question in labeled_answerable
    )
    ages = [
        min(memory.age_hours / 24.0 for memory in question.memories if memory.evidence)
        for question in answerable
        if any(memory.evidence for memory in question.memories)
    ]
    return {
        "questions": len(questions),
        "answerable_questions": len(answerable),
        "labeled_answerable_questions": len(labeled_answerable),
        "unlabeled_answerable_questions": len(answerable) - len(labeled_answerable),
        "abstention_questions": len(questions) - len(answerable),
        "memories_mean": float(statistics.mean(memories)),
        "memories_median": float(statistics.median(memories)),
        "evidence_mean_answerable": float(statistics.mean(evidence)),
        "evidence_median_answerable": float(statistics.median(evidence)),
        "fallback_questions": fallback_questions,
        "fallback_share_answerable": fallback_questions / len(answerable),
        "evidence_age_buckets": {
            "<=7d": sum(age <= 7 for age in ages),
            "8-30d": sum(7 < age <= 30 for age in ages),
            ">30d": sum(age > 30 for age in ages),
        },
        "question_types": dict(
            Counter(question.question_type for question in labeled_answerable)
        ),
    }


def write_pareto(results: dict[str, Any]) -> None:
    """Write coverage-versus-token plot data for conformal and top-k methods."""
    with PARETO_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["method", "alpha", "coverage", "mean_tokens"]
        )
        writer.writeheader()
        for score_variant, methods in results["methods"].items():
            for method, by_alpha in methods.items():
                for alpha, metrics in by_alpha.items():
                    writer.writerow(
                        {
                            "method": f"{score_variant}_{method}",
                            "alpha": alpha,
                            "coverage": metrics["coverage"]["mean"],
                            "mean_tokens": metrics["mean_tokens"]["mean"],
                        }
                    )


def main() -> None:
    """Run data selection, embedding, and the conformal study."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset-size", type=int, default=SUBSET_SIZE)
    arguments = parser.parse_args()
    if not SOURCE_DATA.exists():
        raise FileNotFoundError(f"Download LongMemEval-S to {SOURCE_DATA}")
    records = json.loads(SOURCE_DATA.read_text())
    subset = deterministic_subset(records, arguments.subset_size)
    QUESTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    QUESTIONS_PATH.write_text(json.dumps(subset))
    questions = [pair_memories(record) for record in subset]
    stats = dataset_stats(questions)
    LOGGER.info("Dataset stats: %s", stats)
    embeddings = embedded_questions(questions)
    results = run_study(embeddings)
    results["dataset"] = stats
    RESULTS_PATH.write_text(json.dumps(results, indent=2, allow_nan=True))
    write_pareto(results)
    LOGGER.info("Wrote %s and %s", RESULTS_PATH, PARETO_PATH)


if __name__ == "__main__":
    main()
