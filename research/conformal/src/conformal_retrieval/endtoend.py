"""End-to-end check: does a local model answer better with fewer, longer lines?

The follow-up study showed that 200-byte truncation often hides the answer
from the injected context. This module asks a local instruct model the
LongMemEval-S questions whose short answer appears verbatim in an evidence
turn, with memories formatted exactly as the proxy's default ``system`` mode
does, for several retrieval-depth x line-length settings. Grading is
deterministic: the normalized gold answer must occur in the normalized reply.

Run a llama-server first, for example::

    llama-server -m Qwen2.5-7B-Instruct-Q4_K_M.gguf --port 8491 -c 32768 -np 4

Responses are cached in SQLite, so an interrupted run resumes where it stopped.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sqlite3
import string
import threading
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from conformal_retrieval.followup import short_answer
from conformal_retrieval.study import (
    DATA_DIR,
    DECAY_TIEBREAK_SCORE,
    QUESTIONS_PATH,
    SEED,
    EmbeddedQuestion,
    Memory,
    embedded_questions,
    pair_memories,
    parse_date,
    score_memories,
)

LOGGER = logging.getLogger(__name__)
CHAT_URL = "http://127.0.0.1:8491/v1/chat/completions"
CACHE_PATH = DATA_DIR / "endtoend_cache.sqlite"
RESULTS_PATH = DATA_DIR / "endtoend_results.json"
HEADER = "<memory context>\nRelevant past interactions:\n"
FOOTER = "</memory context>"
SYSTEM_PROMPT = (
    "You are a helpful assistant with memory of earlier conversations with "
    "the user. Answer the user's question in one short sentence. If the "
    "memories do not contain the answer, say you don't know."
)
MAX_TOKENS = 64
BOOTSTRAP = 2000
REFERENCE = "k8_b200"


@dataclass(frozen=True)
class Setting:
    """One injected-context configuration."""

    name: str
    k: int
    line_bytes: int
    oracle: bool = False


SETTINGS = (
    Setting("none", 0, 0),
    Setting("k5_b200", 5, 200),
    Setting("k8_b200", 8, 200),
    Setting("k16_b200", 16, 200),
    Setting("k4_b400", 4, 400),
    Setting("k5_b400", 5, 400),
    Setting("k8_b400", 8, 400),
    Setting("k4_b800", 4, 800),
    Setting("k8_b800", 8, 800),
    Setting("k2_b1600", 2, 1600),
    Setting("oracle_b1600", 0, 1600, oracle=True),
)


def truncate(text: str, max_bytes: int) -> str:
    """Mirror the proxy's byte ``substr(0, n) + "..."`` truncation."""
    encoded = text.encode()
    if len(encoded) <= max_bytes:
        return text
    return encoded[:max_bytes].decode(errors="ignore") + "..."


def time_ago(age_seconds: float) -> str:
    """Mirror the proxy's relative timestamp."""
    if age_seconds < 60:
        return "just now"
    if age_seconds < 3600:
        return f"{int(age_seconds / 60)}m ago"
    if age_seconds < 86400:
        return f"{int(age_seconds / 3600)}h ago"
    return f"{int(age_seconds / 86400)}d ago"


def memory_block(memories: Sequence[Memory], line_bytes: int) -> str:
    """Format memories as the proxy's system-mode memory context block."""
    if not memories:
        return ""
    lines = [
        f"- [{time_ago(memory.age_hours * 3600.0)}] "
        f"{truncate(memory.user_text, line_bytes)} Response: "
        f"{truncate(memory.assistant_text, line_bytes)}\n"
        for memory in memories
    ]
    return HEADER + "".join(lines) + FOOTER


def select(question: EmbeddedQuestion, setting: Setting) -> list[Memory]:
    """Return the memories injected for ``setting`` in rank order."""
    order = np.argsort(-score_memories(question)[DECAY_TIEBREAK_SCORE], kind="stable")
    ranked = [question.question.memories[int(index)] for index in order]
    if setting.oracle:
        return [memory for memory in ranked if memory.evidence]
    return ranked[: setting.k]


def build_messages(
    question: EmbeddedQuestion, setting: Setting, question_date: str
) -> list[dict[str, str]]:
    """Build the chat request the proxy would forward to the backend."""
    system = f"{SYSTEM_PROMPT}\nCurrent date: {question_date}."
    block = memory_block(select(question, setting), setting.line_bytes)
    if block:
        system += "\n\n" + block
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": question.question.text},
    ]


_PUNCTUATION = str.maketrans("", "", string.punctuation.replace("$", ""))


def normalize(text: str) -> str:
    """Lowercase, drop punctuation except ``$`` and collapse whitespace."""
    return re.sub(r"\s+", " ", text.lower().translate(_PUNCTUATION)).strip()


def is_correct(answer: str, reply: str) -> bool:
    """Return whether the normalized gold answer occurs as whole words."""
    gold = normalize(answer)
    return bool(gold) and f" {gold} " in f" {normalize(reply)} "


def paired_bootstrap(
    a: Sequence[bool], b: Sequence[bool], rng: np.random.Generator
) -> dict[str, float]:
    """Return the mean paired difference ``a - b`` and a 95% bootstrap range."""
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    samples = rng.integers(0, len(diff), size=(BOOTSTRAP, len(diff)))
    means = diff[samples].mean(axis=1)
    return {
        "diff": float(diff.mean()),
        "low": float(np.quantile(means, 0.025)),
        "high": float(np.quantile(means, 0.975)),
    }


class ReplyCache:
    """SQLite cache of model replies keyed by the full request."""

    def __init__(self, path: Path) -> None:
        """Open or create the cache at ``path``."""
        self.lock = threading.Lock()
        self.connection = sqlite3.connect(path, check_same_thread=False)
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS replies (key TEXT PRIMARY KEY, reply TEXT,"
            " prompt_tokens INTEGER)"
        )

    def get(self, key: str) -> tuple[str, int] | None:
        """Return a cached reply and its prompt token count."""
        with self.lock:
            row = self.connection.execute(
                "SELECT reply, prompt_tokens FROM replies WHERE key = ?", (key,)
            ).fetchone()
        return None if row is None else (str(row[0]), int(row[1]))

    def put(self, key: str, reply: str, prompt_tokens: int) -> None:
        """Store one reply."""
        with self.lock:
            self.connection.execute(
                "INSERT OR REPLACE INTO replies VALUES (?, ?, ?)",
                (key, reply, prompt_tokens),
            )
            self.connection.commit()


def request_key(model: str, messages: Sequence[Mapping[str, str]]) -> str:
    """Return a stable key for one deterministic chat request."""
    payload = json.dumps({"model": model, "messages": messages}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def chat(messages: Sequence[Mapping[str, str]]) -> tuple[str, int]:
    """Send one greedy chat completion to the local server."""
    body = {
        "messages": list(messages),
        "temperature": 0,
        "seed": SEED,
        "max_tokens": MAX_TOKENS,
    }
    request = urllib.request.Request(
        CHAT_URL,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            payload = json.loads(response.read())
    except urllib.error.URLError as error:
        raise RuntimeError(f"Chat endpoint unavailable at {CHAT_URL}") from error
    reply = str(payload["choices"][0]["message"]["content"] or "")
    return reply, int(payload["usage"]["prompt_tokens"])


def main() -> None:
    """Ask every setting and write ``data/endtoend_results.json``."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="label stored with results")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    arguments = parser.parse_args()
    records = json.loads(QUESTIONS_PATH.read_text())
    answers = {str(r["question_id"]): str(r["answer"]) for r in records}
    dates = {
        str(r["question_id"]): parse_date(str(r["question_date"])).strftime("%Y-%m-%d")
        for r in records
    }
    questions = [
        question
        for question in embedded_questions([pair_memories(r) for r in records])
        if not question.question.abstention
        and short_answer(question, answers) is not None
    ]
    if arguments.limit:
        questions = questions[: arguments.limit]
    LOGGER.info("Evaluating %d questions x %d settings", len(questions), len(SETTINGS))
    cache = ReplyCache(CACHE_PATH)

    def run(item: tuple[EmbeddedQuestion, Setting]) -> dict[str, Any]:
        question, setting = item
        qid = question.question.question_id
        messages = build_messages(question, setting, dates[qid])
        key = request_key(arguments.model, messages)
        cached = cache.get(key)
        reply, prompt_tokens = cached if cached is not None else chat(messages)
        if cached is None:
            cache.put(key, reply, prompt_tokens)
        return {
            "question_id": qid,
            "question_type": question.question.question_type,
            "setting": setting.name,
            "correct": is_correct(answers[qid], reply),
            "prompt_tokens": prompt_tokens,
        }

    jobs = [(question, setting) for setting in SETTINGS for question in questions]
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=arguments.workers) as pool:
        for index, row in enumerate(pool.map(run, jobs), start=1):
            rows.append(row)
            if index % 100 == 0:
                LOGGER.info("Completed %d/%d requests", index, len(jobs))
    by_setting: dict[str, list[dict[str, Any]]] = {s.name: [] for s in SETTINGS}
    for row in rows:
        by_setting[row["setting"]].append(row)
    for items in by_setting.values():
        items.sort(key=lambda row: row["question_id"])
    rng = np.random.default_rng(SEED)
    reference = [row["correct"] for row in by_setting[REFERENCE]]
    summary: dict[str, Any] = {}
    for setting in SETTINGS:
        items = by_setting[setting.name]
        correct = [row["correct"] for row in items]
        summary[setting.name] = {
            "k": setting.k,
            "line_bytes": setting.line_bytes,
            "accuracy": float(np.mean(correct)),
            "prompt_tokens_mean": float(np.mean([r["prompt_tokens"] for r in items])),
            f"vs_{REFERENCE}": paired_bootstrap(correct, reference, rng),
            "by_type": {
                name: float(
                    np.mean([r["correct"] for r in items if r["question_type"] == name])
                )
                for name in sorted({r["question_type"] for r in items})
            },
        }
    RESULTS_PATH.write_text(
        json.dumps(
            {
                "model": arguments.model,
                "questions": len(questions),
                "max_tokens": MAX_TOKENS,
                "settings": summary,
            },
            indent=2,
        )
    )
    LOGGER.info("Wrote %s", RESULTS_PATH)


if __name__ == "__main__":
    main()
