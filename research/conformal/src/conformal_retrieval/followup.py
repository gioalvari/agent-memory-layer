"""Follow-up conformal studies on the proxy's actual injected context.

The first study reported tokens for untruncated memories. The proxy formats each
memory as one line whose user and assistant parts are truncated to 200 bytes, so
this module re-measures every policy with production-formatted token counts and
adds four analyses:

* deployed calibration: one finite-sample quantile on all labeled questions;
* token-budget conformal: calibrate the injection budget instead of ``k``;
* adaptive sets that use only serving-time features (relative score gap and
  Mondrian bins on the top-1 similarity);
* conformal risk control of the expected evidence recall.
"""

from __future__ import annotations

import json
import logging
import math
import statistics
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from conformal_retrieval.study import (
    DATA_DIR,
    DECAY_TIEBREAK_SCORE,
    PRODUCTION_SCORE,
    QUESTIONS_PATH,
    RAW_SCORE,
    REPETITIONS,
    SEED,
    EmbeddedQuestion,
    Memory,
    embedded_questions,
    pair_memories,
    rank_quantile,
    score_memories,
)

LOGGER = logging.getLogger(__name__)
FOLLOWUP_PATH = DATA_DIR / "conformal_followup.json"
ALPHAS = (0.05, 0.10, 0.20)
LINE_MAX_BYTES = 200
HEADER = "<memory context>\nRelevant past interactions:\n"
FOOTER = "</memory context>"
MONDRIAN_BINS = 3
VARIANTS = {"tiebreak": DECAY_TIEBREAK_SCORE, "legacy": PRODUCTION_SCORE}
VISIBILITY_LIMITS = (200, 400, 800, 1600)
GRID_KS = (2, 4, 8, 16)


def estimate_tokens(text_bytes: int) -> int:
    """Mirror the proxy's ``size() / 4 + 1`` token heuristic on a byte length."""
    return text_bytes // 4 + 1


def truncated_bytes(text: str, max_bytes: int = LINE_MAX_BYTES) -> int:
    """Return the byte length after the proxy's ``substr(0, n) + "..."``."""
    size = len(text.encode("utf-8"))
    return size if size <= max_bytes else max_bytes + 3


def format_time_ago(age_seconds: float) -> str:
    """Mirror the proxy's relative timestamp used in system and suffix modes."""
    if age_seconds < 60:
        return "just now"
    if age_seconds < 3600:
        return f"{int(age_seconds / 60)}m ago"
    if age_seconds < 86400:
        return f"{int(age_seconds / 3600)}h ago"
    return f"{int(age_seconds / 86400)}d ago"


def line_tokens(memory: Memory, max_bytes: int = LINE_MAX_BYTES) -> int:
    """Estimate the tokens of one injected ``- [age] user Response: assistant``."""
    size = (
        len("- [" + format_time_ago(memory.age_hours * 3600.0) + "] ")
        + truncated_bytes(memory.user_text, max_bytes)
        + len(" Response: ")
        + truncated_bytes(memory.assistant_text, max_bytes)
        + 1
    )
    return estimate_tokens(size)


CONTEXT_OVERHEAD = estimate_tokens(len(HEADER)) + estimate_tokens(len(FOOTER))


def upper_quantile(values: Sequence[float], alpha: float) -> float:
    """Return the ``ceil((1 - alpha)(n + 1))``-th smallest calibration score.

    Parameters
    ----------
    values
        Nonconformity scores, larger meaning harder to cover.
    alpha
        Target miscoverage rate.

    Returns
    -------
    float
        The split-conformal quantile, or infinity when ``n`` is too small.
    """
    index = math.ceil((1.0 - alpha) * (len(values) + 1))
    if index > len(values):
        return float("inf")
    return float(np.sort(np.asarray(values, dtype=float))[index - 1])


def crc_k(losses: np.ndarray, alpha: float) -> int:
    """Return the smallest ``k`` whose conformal risk bound is at most ``alpha``.

    Parameters
    ----------
    losses
        Matrix ``(n, K)`` of losses in ``[0, 1]``, non-increasing along ``k``.
    alpha
        Target expected loss.

    Returns
    -------
    int
        One-based ``k`` with ``(n * mean(L_k) + 1) / (n + 1) <= alpha``; ``K``
        when no smaller value satisfies the bound.
    """
    n = losses.shape[0]
    bounds = (n * losses.mean(axis=0) + 1.0) / (n + 1.0)
    feasible = np.flatnonzero(bounds <= alpha)
    return int(feasible[0] + 1) if feasible.size else int(losses.shape[1])


def mondrian_edges(features: Sequence[float], bins: int = MONDRIAN_BINS) -> np.ndarray:
    """Return interior quantile edges that split ``features`` into equal bins."""
    edges: np.ndarray = np.quantile(
        np.asarray(features), np.linspace(0, 1, bins + 1)[1:-1]
    )
    return edges


@dataclass(frozen=True)
class Profile:
    """Label-derived and serving-time quantities for one ranked haystack."""

    question_type: str
    sorted_scores: np.ndarray
    cumulative_tokens: np.ndarray
    recall: np.ndarray
    best_rank: int
    last_rank: int
    evidence_count: int
    top1_similarity: float

    @property
    def top1(self) -> float:
        """Return the raw cosine of the top-ranked memory, known at serving time.

        This is the unboosted ``max(cos(query, user), cos(query, assistant))``
        of the first memory in rank order, which the proxy can read before it
        decides how many memories to inject.
        """
        return self.top1_similarity

    @property
    def tokens_to_first(self) -> int:
        """Return the line tokens needed to include the first evidence."""
        return int(self.cumulative_tokens[self.best_rank - 1])

    @property
    def relative_gap(self) -> float:
        """Return the top-1 minus best-evidence score gap."""
        return float(self.sorted_scores[0] - self.sorted_scores[self.best_rank - 1])

    def size_for_budget(self, budget: float) -> int:
        """Return the memories whose cumulative line tokens fit ``budget``."""
        return int(np.searchsorted(self.cumulative_tokens, budget, side="right"))

    def size_for_gap(self, gap: float) -> int:
        """Return the memories scoring at least ``top1 - gap``."""
        return int(np.count_nonzero(self.sorted_scores >= self.sorted_scores[0] - gap))


def build_profile(
    question: EmbeddedQuestion,
    scores: np.ndarray,
    tokens: np.ndarray,
    similarity: np.ndarray | None = None,
) -> Profile:
    """Rank one haystack and precompute its coverage and token curves.

    ``similarity`` is the raw cosine used for the serving-time top-1 feature;
    it defaults to ``scores``.
    """
    order = np.argsort(-scores, kind="stable")
    raw = scores if similarity is None else similarity
    evidence = np.asarray([memory.evidence for memory in question.question.memories])[
        order
    ]
    positions = np.flatnonzero(evidence)
    return Profile(
        question_type=question.question.question_type,
        sorted_scores=scores[order],
        cumulative_tokens=np.cumsum(tokens[order]),
        recall=np.cumsum(evidence) / evidence.sum(),
        best_rank=int(positions[0] + 1),
        last_rank=int(positions[-1] + 1),
        evidence_count=int(evidence.sum()),
        top1_similarity=float(raw[order[0]]),
    )


def outcome(profile: Profile, size: int) -> dict[str, float]:
    """Return coverage, recall, and injected tokens for the top ``size`` lines."""
    size = min(size, len(profile.sorted_scores))
    return {
        "coverage": float(profile.best_rank <= size),
        "recall": float(profile.recall[size - 1]) if size else 0.0,
        "all": float(profile.last_rank <= size),
        "size": float(size),
        "tokens": float(profile.cumulative_tokens[size - 1] + CONTEXT_OVERHEAD)
        if size
        else 0.0,
    }


def aggregate(
    rows: Sequence[Mapping[str, float]], profiles: Sequence[Profile]
) -> dict[str, float]:
    """Aggregate per-question outcomes of one test fold."""
    multi = [
        row["all"]
        for row, profile in zip(rows, profiles)
        if profile.evidence_count >= 2
    ]
    tokens = [row["tokens"] for row in rows]
    return {
        "coverage": float(np.mean([row["coverage"] for row in rows])),
        "recall": float(np.mean([row["recall"] for row in rows])),
        "all_coverage": float(np.mean(multi)) if multi else float("nan"),
        "mean_size": float(np.mean([row["size"] for row in rows])),
        "mean_tokens": float(np.mean(tokens)),
        "p95_tokens": float(np.quantile(tokens, 0.95)),
        "max_tokens": float(np.max(tokens)),
    }


Policy = Callable[[Sequence[Profile], float], tuple[Callable[[Profile], int], float]]


def rank_policy(
    calibration: Sequence[Profile], alpha: float
) -> tuple[Callable[[Profile], int], float]:
    """Calibrate one global ``k`` on best-evidence ranks."""
    k = rank_quantile([profile.best_rank for profile in calibration], alpha)
    return (lambda _profile: k), float(k)


def budget_policy(
    calibration: Sequence[Profile], alpha: float
) -> tuple[Callable[[Profile], int], float]:
    """Calibrate the line-token budget that reaches the first evidence."""
    budget = upper_quantile([p.tokens_to_first for p in calibration], alpha)
    return (lambda profile: profile.size_for_budget(budget)), budget + CONTEXT_OVERHEAD


def gap_policy(
    calibration: Sequence[Profile], alpha: float
) -> tuple[Callable[[Profile], int], float]:
    """Calibrate a score window below each query's own top-1 similarity."""
    gap = upper_quantile([p.relative_gap for p in calibration], alpha)
    return (lambda profile: profile.size_for_gap(gap)), gap


def mondrian_policy(
    calibration: Sequence[Profile], alpha: float
) -> tuple[Callable[[Profile], int], float]:
    """Calibrate one rank ``k`` per top-1-similarity tercile."""
    edges = mondrian_edges([profile.top1 for profile in calibration])
    ranks: dict[int, list[int]] = defaultdict(list)
    for profile in calibration:
        ranks[int(np.searchsorted(edges, profile.top1))].append(profile.best_rank)
    ks = {bin_: rank_quantile(values, alpha) for bin_, values in ranks.items()}
    fallback = max(ks.values())
    return (
        lambda profile: ks.get(int(np.searchsorted(edges, profile.top1)), fallback)
    ), float("nan")


def mondrian_floor_policy(
    calibration: Sequence[Profile], alpha: float
) -> tuple[Callable[[Profile], int], float]:
    """Use ``max(tercile k, global k)``: never shallower than the global policy.

    The set contains both the Mondrian and the global rank set, so it keeps the
    marginal guarantee of each.
    """
    mondrian, _ = mondrian_policy(calibration, alpha)
    global_k = rank_quantile([profile.best_rank for profile in calibration], alpha)
    return (lambda profile: max(mondrian(profile), global_k)), float(global_k)


def crc_policy(
    calibration: Sequence[Profile], alpha: float
) -> tuple[Callable[[Profile], int], float]:
    """Choose ``k`` controlling expected missed evidence (1 - recall)."""
    width = max(len(profile.recall) for profile in calibration)
    losses = np.asarray(
        [
            1.0 - np.pad(p.recall, (0, width - len(p.recall)), mode="edge")
            for p in calibration
        ]
    )
    k = crc_k(losses, alpha)
    return (lambda _profile: k), float(k)


POLICIES: dict[str, Policy] = {
    "rank": rank_policy,
    "token_budget": budget_policy,
    "relative_gap": gap_policy,
    "mondrian_top1": mondrian_policy,
    "mondrian_floor": mondrian_floor_policy,
    "crc_recall": crc_policy,
}

BINNED_POLICIES = ("rank", "mondrian_top1", "mondrian_floor")


def summarize_runs(values: Sequence[float]) -> dict[str, float]:
    """Return mean, median, and 5--95% range across repeated splits."""
    finite = [value for value in values if not math.isnan(value)]
    if not finite:
        return {"mean": float("nan"), "median": float("nan")}
    return {
        "mean": float(np.mean(finite)),
        "median": float(statistics.median(finite)),
        "p05": float(np.quantile(finite, 0.05)),
        "p95": float(np.quantile(finite, 0.95)),
    }


def repeated_splits(
    profiles: Sequence[Profile], repetitions: int = REPETITIONS
) -> dict[str, dict[str, dict[str, Any]]]:
    """Evaluate every policy on the same 50:50 splits as the first study."""
    runs: dict[str, dict[str, list[dict[str, float]]]] = {
        name: {str(alpha): [] for alpha in ALPHAS} for name in POLICIES
    }
    bins: dict[str, dict[str, list[dict[str, float]]]] = {
        name: {str(alpha): [] for alpha in ALPHAS} for name in BINNED_POLICIES
    }
    rng = np.random.default_rng(SEED)
    for _ in range(repetitions):
        indices = rng.permutation(len(profiles))
        calibration = [profiles[int(i)] for i in indices[: len(indices) // 2]]
        test = [profiles[int(i)] for i in indices[len(indices) // 2 :]]
        edges = mondrian_edges([profile.top1 for profile in calibration])
        for alpha in ALPHAS:
            for name, policy in POLICIES.items():
                size_of, parameter = policy(calibration, alpha)
                rows = [outcome(profile, size_of(profile)) for profile in test]
                metrics = aggregate(rows, test)
                metrics["parameter"] = parameter
                runs[name][str(alpha)].append(metrics)
                if name in BINNED_POLICIES:
                    bins[name][str(alpha)].append(per_bin(test, rows, edges))
    summary: dict[str, dict[str, dict[str, Any]]] = {}
    for name, by_alpha in runs.items():
        summary[name] = {}
        for alpha_text, metrics_list in by_alpha.items():
            summary[name][alpha_text] = {
                metric: summarize_runs([item[metric] for item in metrics_list])
                for metric in metrics_list[0]
            }
    for name, by_alpha_bins in bins.items():
        summary[f"{name}_bins"] = {
            alpha_text: {
                key: summarize_runs([item[key] for item in items]) for key in items[0]
            }
            for alpha_text, items in by_alpha_bins.items()
        }
    return summary


def per_bin(
    test: Sequence[Profile], rows: Sequence[Mapping[str, float]], edges: np.ndarray
) -> dict[str, float]:
    """Return test coverage and mean size by top-1 bin for one split."""
    grouped: dict[int, list[Mapping[str, float]]] = defaultdict(list)
    for profile, row in zip(test, rows):
        grouped[int(np.searchsorted(edges, profile.top1))].append(row)
    result: dict[str, float] = {}
    for bin_ in range(MONDRIAN_BINS):
        items = grouped.get(bin_, [])
        result[f"bin{bin_}_coverage"] = (
            float(np.mean([row["coverage"] for row in items])) if items else math.nan
        )
        result[f"bin{bin_}_size"] = (
            float(np.mean([row["size"] for row in items])) if items else math.nan
        )
    return result


def deployed_calibration(profiles: Sequence[Profile]) -> dict[str, Any]:
    """Calibrate on every labeled question, as a shipped constant would be."""
    result: dict[str, Any] = {"n": len(profiles)}
    for alpha in ALPHAS:
        k = rank_quantile([profile.best_rank for profile in profiles], alpha)
        tokens = [outcome(profile, k)["tokens"] for profile in profiles]
        _, budget = budget_policy(profiles, alpha)
        _, crc = crc_policy(profiles, alpha)
        result[str(alpha)] = {
            "rank_k": k,
            "rank_k_tokens_mean": float(np.mean(tokens)),
            "rank_k_tokens_max": float(np.max(tokens)),
            "token_budget": budget,
            "crc_recall_k": int(crc),
            "mondrian_top1": deployed_mondrian(profiles, alpha),
        }
    return result


def deployed_mondrian(profiles: Sequence[Profile], alpha: float) -> dict[str, Any]:
    """Return top-1 tercile edges and per-tercile rank quantiles on all data."""
    edges = mondrian_edges([profile.top1 for profile in profiles])
    groups: dict[int, list[Profile]] = defaultdict(list)
    for profile in profiles:
        groups[int(np.searchsorted(edges, profile.top1))].append(profile)
    ks = [
        rank_quantile([p.best_rank for p in groups[b]], alpha)
        for b in range(MONDRIAN_BINS)
    ]
    sizes = [ks[int(np.searchsorted(edges, profile.top1))] for profile in profiles]
    return {
        "edges": [float(edge) for edge in edges],
        "k": ks,
        "n": [len(groups[b]) for b in range(MONDRIAN_BINS)],
        "mean_k": float(np.mean(sizes)),
    }


def visible_prefix(text: str, max_bytes: int) -> str:
    """Return the lowercased text the model sees after byte truncation."""
    return text.encode()[:max_bytes].decode(errors="ignore").lower()


def truncation_stats(
    questions: Sequence[EmbeddedQuestion], answers: Mapping[str, str]
) -> dict[str, Any]:
    """Measure how often line truncation hides the labeled evidence.

    The answer-visibility probe only uses short answers (at most 40 characters)
    that occur verbatim in an evidence turn; it is a lower bound on the problem,
    not a full audit, because paraphrased answers are not counted.
    """
    long_user = long_assistant = total = 0
    occurrences: list[list[str]] = []
    for question in questions:
        answer = answers[question.question.question_id].strip().lower()
        short = 0 < len(answer) <= 40
        texts: list[str] = []
        for memory in question.question.memories:
            if not memory.evidence:
                continue
            total += 1
            long_user += len(memory.user_text.encode()) > LINE_MAX_BYTES
            long_assistant += len(memory.assistant_text.encode()) > LINE_MAX_BYTES
            texts.extend(
                text
                for text in (memory.user_text, memory.assistant_text)
                if short and answer in text.lower()
            )
        if texts:
            occurrences.append([answer, *texts])
    visibility = {
        str(limit): float(
            np.mean(
                [
                    any(item[0] in visible_prefix(text, limit) for text in item[1:])
                    for item in occurrences
                ]
            )
        )
        for limit in VISIBILITY_LIMITS
    }
    return {
        "evidence_memories": total,
        "user_over_200_bytes": long_user / total,
        "assistant_over_200_bytes": long_assistant / total,
        "short_answer_questions": len(occurrences),
        "short_answer_visible_by_limit": visibility,
    }


def short_answer(question: EmbeddedQuestion, answers: Mapping[str, str]) -> str | None:
    """Return a short answer found verbatim in an evidence turn, if any."""
    answer = answers[question.question.question_id].strip().lower()
    if not 0 < len(answer) <= 40:
        return None
    for memory in question.question.memories:
        if memory.evidence and (
            answer in memory.user_text.lower()
            or answer in memory.assistant_text.lower()
        ):
            return answer
    return None


def visibility_grid(
    questions: Sequence[EmbeddedQuestion], answers: Mapping[str, str], variant: str
) -> dict[str, dict[str, float]]:
    """Return answer visibility and tokens for retrieval depth x line length.

    A question counts as visible when its short answer appears in the truncated
    text of any injected memory, so this measures what the model can actually
    read rather than whether an evidence-labeled memory was retrieved.
    """
    grid: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"visible": [], "tokens": []}
    )
    for question in questions:
        answer = short_answer(question, answers)
        if answer is None:
            continue
        order = np.argsort(-score_memories(question)[variant], kind="stable")
        for limit in VISIBILITY_LIMITS:
            for k in GRID_KS:
                selected = [question.question.memories[int(i)] for i in order[:k]]
                cell = grid[f"k{k}_bytes{limit}"]
                cell["visible"].append(
                    float(
                        any(
                            answer in visible_prefix(text, limit)
                            for memory in selected
                            for text in (memory.user_text, memory.assistant_text)
                        )
                    )
                )
                cell["tokens"].append(
                    float(sum(line_tokens(m, limit) for m in selected))
                    + CONTEXT_OVERHEAD
                )
    return {
        name: {
            "n": float(len(values["visible"])),
            "visible": float(np.mean(values["visible"])),
            "mean_tokens": float(np.mean(values["tokens"])),
        }
        for name, values in grid.items()
    }


def token_accounting(profiles: Sequence[Profile]) -> dict[str, float]:
    """Compare production line tokens with the first study's raw-text counts."""
    per_line = np.concatenate(
        [np.diff(profile.cumulative_tokens, prepend=0) for profile in profiles]
    )
    return {
        "line_tokens_mean": float(per_line.mean()),
        "line_tokens_max": float(per_line.max()),
        "context_overhead": float(CONTEXT_OVERHEAD),
    }


def main() -> None:
    """Load cached embeddings and write ``data/conformal_followup.json``."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    records = json.loads(QUESTIONS_PATH.read_text())
    answers = {str(record["question_id"]): str(record["answer"]) for record in records}
    questions = embedded_questions([pair_memories(record) for record in records])
    answerable = [
        question
        for question in questions
        if not question.question.abstention
        and any(memory.evidence for memory in question.question.memories)
    ]
    LOGGER.info("Loaded %d answerable questions", len(answerable))
    results: dict[str, Any] = {
        "metadata": {"seed": SEED, "repetitions": REPETITIONS, "n": len(answerable)},
        "truncation": truncation_stats(answerable, answers),
    }
    for label, variant in VARIANTS.items():
        profiles = []
        for question in answerable:
            tokens = np.asarray([line_tokens(m) for m in question.question.memories])
            scores = score_memories(question)
            profiles.append(
                build_profile(
                    question,
                    scores[variant],
                    tokens,
                    similarity=scores[RAW_SCORE],
                )
            )
        results[label] = {
            "tokens": token_accounting(profiles),
            "deployed": deployed_calibration(profiles),
            "visibility_grid": visibility_grid(answerable, answers, variant),
            "splits": repeated_splits(profiles),
        }
        LOGGER.info("Finished %s", label)
    FOLLOWUP_PATH.write_text(json.dumps(results, indent=2, allow_nan=True))
    LOGGER.info("Wrote %s", FOLLOWUP_PATH)


if __name__ == "__main__":
    main()
