"""Evaluate new retrieval features with the real ``memory-layer`` binary.

The evaluator preloads each LongMemEval-S haystack into its own proxy namespace,
then records the actual request forwarded by the proxy.  This makes retrieval
metrics independent of an answer model while preserving the C++ implementation's
ranking, FTS, temporal parsing, injection budget, and prompt formatting.

``--compress-memories`` and ``--extract-facts`` depend on artifacts produced by
the asynchronous save worker.  They are evaluated by replaying the source turns
through the proxy before asking the benchmark question.  The replay is serial so
the worker has a chance to persist sentence embeddings and extracted facts.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import sqlite3
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from collections import defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np

from conformal_retrieval.endtoend import (
    MAX_TOKENS,
    SYSTEM_PROMPT,
    is_correct,
    paired_bootstrap,
)
from conformal_retrieval.followup import short_answer
from conformal_retrieval.proxy_check import SCHEMA, unit, wait_for
from conformal_retrieval.study import (
    QUESTIONS_PATH,
    SEED,
    EmbeddedQuestion,
    embedded_questions,
    pair_memories,
    parse_date,
)

LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[4]
RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
DEFAULT_PROXY_PORT = 8802
DEFAULT_SHIM_PORT = 8493
MEMORY_HEADER = "<memory context>"
MEMORY_FOOTER = "</memory context>"
FACT_HEADER = "Known facts (latest values):"


@dataclass(frozen=True)
class FeatureRun:
    """One feature configuration evaluated against the proxy."""

    name: str
    flags: tuple[str, ...]
    replay: bool = False
    facts: bool = False


RUNS = (
    FeatureRun("baseline", ()),
    FeatureRun("hybrid", ("--hybrid-search",)),
    FeatureRun("temporal", ("--temporal-boost", "0.15", "--absolute-dates")),
    FeatureRun("compress", ("--compress-memories",), replay=True),
    FeatureRun(
        "hybrid_temporal_compress",
        (
            "--hybrid-search",
            "--temporal-boost",
            "0.15",
            "--absolute-dates",
            "--compress-memories",
        ),
        replay=True,
    ),
    FeatureRun("facts", ("--extract-facts",), replay=True, facts=True),
)


def build_database(
    path: Path, questions: Sequence[EmbeddedQuestion], now: float
) -> None:
    """Build a preloaded database and its FTS tables for direct retrieval runs."""
    connection = sqlite3.connect(path)
    connection.executescript(SCHEMA)
    connection.executescript(
        """
        CREATE VIRTUAL TABLE memories_fts USING fts5(user_text, assist_text);
        CREATE TRIGGER memories_fts_insert AFTER INSERT ON memories BEGIN
            INSERT INTO memories_fts(rowid, user_text, assist_text)
            VALUES (new.id, new.user_text, new.assist_text);
        END;
        """
    )
    rows = []
    for item in questions:
        for index, memory in enumerate(item.question.memories):
            created = now - memory.age_hours * 3600.0
            rows.append(
                (
                    item.question.question_id,
                    created,
                    created,
                    memory.user_text,
                    memory.assistant_text,
                    unit(item.user_embeddings[index]),
                    unit(item.assistant_embeddings[index]),
                )
            )
    connection.executemany(
        "INSERT INTO memories (agent_id, created_at, updated_at, user_text,"
        " assist_text, user_emb, assist_emb) VALUES (?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    connection.commit()
    connection.close()


def request_key(messages: Sequence[Mapping[str, Any]]) -> str:
    """Return a stable key that distinguishes a benchmark request.

    The proxy appends retrieved context to the system message before forwarding
    it, so use its unchanged current-date line instead of the full system
    prompt.  Questions can share their text, while the date-plus-question pair
    remains stable before and after injection.
    """
    system = str(messages[0]["content"])
    date_line = next(line for line in system.splitlines() if "Current date" in line)
    return date_line + "\n" + str(messages[-1]["content"])


class Recorder:
    """Record proxy requests and optionally synthesize deterministic backend replies."""

    def __init__(self, backend: str | None) -> None:
        """Set the upstream backend, or use a deterministic fake backend."""
        self.backend = backend
        self.lock = threading.Lock()
        self.records: dict[str, dict[str, Any]] = {}

    def handle(self, body: bytes) -> bytes:
        """Record a request and return its backend response."""
        payload = json.loads(body)
        if self.backend is None:
            reply = {
                "choices": [{"message": {"content": ""}}],
                "usage": {"prompt_tokens": 0},
            }
            raw = json.dumps(reply).encode()
        else:
            request = urllib.request.Request(
                self.backend,
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=600) as response:
                raw = response.read()
        decoded = json.loads(raw)
        with self.lock:
            self.records[request_key(payload["messages"])] = {
                "messages": payload["messages"],
                "reply": str(decoded["choices"][0]["message"]["content"] or ""),
                "prompt_tokens": int(decoded.get("usage", {}).get("prompt_tokens", 0)),
            }
        return raw


def serve_recorder(recorder: Recorder, port: int) -> ThreadingHTTPServer:
    """Serve the recorder on a background thread."""

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 - standard library interface
            body = self.rfile.read(int(self.headers["Content-Length"]))
            try:
                raw, status = recorder.handle(body), 200
            except Exception as error:  # noqa: BLE001 - return proxy-compatible error
                LOGGER.warning("Recorder forwarding failed: %s", error)
                raw, status = json.dumps({"error": str(error)}).encode(), 502
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            """Silence request logs."""

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def post_proxy(
    port: int,
    agent_id: str,
    messages: Sequence[Mapping[str, str]],
    model: str,
) -> None:
    """Send one OpenAI-compatible request through the proxy."""
    body = {
        "model": model,
        "messages": list(messages),
        "temperature": 0,
        "seed": SEED,
        "max_tokens": MAX_TOKENS,
    }
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "X-Agent-Id": agent_id},
        method="POST",
    )
    last_error: urllib.error.HTTPError | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=900) as response:
                response.read()
            return
        except urllib.error.HTTPError as error:
            last_error = error
            if error.code != 500 or attempt == 2:
                raise
            error.read()
            time.sleep(0.25 * (attempt + 1))
    raise RuntimeError("unreachable") from last_error


def benchmark_messages(question: EmbeddedQuestion, date: str) -> list[dict[str, str]]:
    """Build the bare request that the real proxy enriches."""
    return [
        {"role": "system", "content": f"{SYSTEM_PROMPT}\nCurrent date: {date}."},
        {"role": "user", "content": question.question.text},
    ]


def replay_question(port: int, item: EmbeddedQuestion, model: str) -> None:
    """Save all source turns through the proxy, waiting for the saver to drain."""
    agent_id = item.question.question_id
    for memory in item.question.memories:
        post_proxy(
            port,
            agent_id,
            [
                {"role": "user", "content": memory.user_text},
                {"role": "assistant", "content": memory.assistant_text},
            ],
            model,
        )


def wait_for_rows(database: Path, expected: int, include_sentences: bool) -> None:
    """Wait for background saving and sentence extraction to reach ``expected`` rows."""
    deadline = time.monotonic() + max(120.0, expected * 2.0)
    # Short memories may legitimately produce no sentence rows, so for the
    # sentence table wait until the count is stable rather than >= expected.
    last_sentences, stable_since = -1, time.monotonic()
    memories = last_memories = -1
    while time.monotonic() < deadline:
        try:
            connection = sqlite3.connect(database)
            memories = int(
                connection.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            )
            sentences = int(
                connection.execute("SELECT COUNT(*) FROM memory_sentences").fetchone()[
                    0
                ]
            )
            connection.close()
            if (memories, sentences) != (last_memories, last_sentences):
                last_memories, last_sentences = memories, sentences
                stable_since = time.monotonic()
            # Deduplication can merge identical turns, so counts may settle
            # slightly below ``expected``; accept a count that stopped moving.
            settled = memories > 0 and time.monotonic() - stable_since >= 5.0
            if settled or (
                memories >= expected
                and (not include_sentences or sentences >= expected)
            ):
                return
        except sqlite3.Error:
            pass
        time.sleep(0.25)
    raise RuntimeError(
        f"Proxy save worker did not persist {expected} memories "
        f"(memories={memories}, sentences={last_sentences})"
    )


def restore_replay_ages(database: Path, questions: Sequence[EmbeddedQuestion]) -> None:
    """Restore synthetic LongMemEval ages after asynchronous source-turn replay."""
    now = time.time()
    connection = sqlite3.connect(database)
    try:
        for item in questions:
            rows = connection.execute(
                "SELECT id FROM memories WHERE agent_id = ? ORDER BY id",
                (item.question.question_id,),
            ).fetchall()
            if len(rows) != len(item.question.memories):
                raise RuntimeError(
                    f"Expected {len(item.question.memories)} saved turns for "
                    f"{item.question.question_id}, found {len(rows)}"
                )
            connection.executemany(
                "UPDATE memories SET created_at = ?, updated_at = ? WHERE id = ?",
                [
                    (
                        now - memory.age_hours * 3600.0,
                        now - memory.age_hours * 3600.0,
                        row[0],
                    )
                    for row, memory in zip(rows, item.question.memories)
                ],
            )
        connection.commit()
    finally:
        connection.close()


def injected_lines(system: str) -> list[str]:
    """Return memory lines from an injected system prompt."""
    return [
        line.split("] ", 1)[1]
        for line in system.splitlines()
        if line.startswith("- [") and "] " in line
    ]


def context_tokens(system: str) -> int:
    """Mirror the proxy's character-over-four token heuristic for injected context."""
    start = system.find(MEMORY_HEADER)
    if start < 0:
        return 0
    return len(system[start:].encode()) // 4 + 1


def fact_count(system: str) -> int:
    """Count fact bullets in an injected context."""
    if FACT_HEADER not in system:
        return 0
    return sum(
        1
        for line in system.splitlines()
        if line.startswith("- ") and not line.startswith("- [")
    )


def evidence_recall(item: EmbeddedQuestion, lines: Sequence[str]) -> tuple[bool, float]:
    """Return any-evidence recall and fraction of evidence memories represented."""
    evidence = [memory for memory in item.question.memories if memory.evidence]
    if not evidence:
        return False, 0.0
    matched = sum(
        any(memory.user_text in line or memory.assistant_text in line for line in lines)
        for memory in evidence
    )
    return matched > 0, matched / len(evidence)


def summarize_rows(
    rows: Sequence[Mapping[str, Any]], baseline: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Aggregate metrics overall and per LongMemEval type."""
    baseline_by_id = {str(row["question_id"]): row for row in baseline}
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    groups["overall"] = list(rows)
    for row in rows:
        groups[str(row["question_type"])].append(row)
    summary: dict[str, Any] = {}
    rng = np.random.default_rng(SEED)
    for name, items in sorted(groups.items()):
        answerable = [item for item in items if item["answerable"]]
        values: dict[str, Any] = {
            "questions": len(items),
            "session_recall_at_k": float(
                np.mean([item["session_recall"] for item in items])
            ),
            "evidence_recall": float(
                np.mean([item["evidence_recall"] for item in items])
            ),
            "injected_tokens_mean": float(
                np.mean([item["injected_tokens"] for item in items])
            ),
            "injected_lines_mean": float(
                np.mean([item["injected_lines"] for item in items])
            ),
            "facts_mean": float(np.mean([item["facts"] for item in items])),
            "qa_questions": len(answerable),
        }
        if answerable:
            correct = [bool(item["correct"]) for item in answerable]
            reference = [
                bool(baseline_by_id[str(item["question_id"])]["correct"])
                for item in answerable
            ]
            values["qa_accuracy"] = float(np.mean(correct))
            values["qa_vs_baseline"] = paired_bootstrap(correct, reference, rng)
        else:
            values["qa_accuracy"] = None
            values["qa_vs_baseline"] = None
        summary[name] = values
    return summary


def output_name(model: str) -> str:
    """Make an output-safe model label."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", model)


def run_feature(
    run: FeatureRun,
    questions: Sequence[EmbeddedQuestion],
    dates: Mapping[str, str],
    answers: Mapping[str, str],
    binary: Path,
    embedding_model: Path,
    backend: str | None,
    model: str,
    workers: int,
    proxy_port: int,
    shim_port: int,
) -> list[dict[str, Any]]:
    """Run one proxy feature configuration and return per-question observations."""
    if run.facts and backend is None:
        raise RuntimeError("facts evaluation requires --backend")
    recorder = Recorder(backend)
    server = serve_recorder(recorder, shim_port)
    with tempfile.TemporaryDirectory() as directory:
        temp = Path(directory)
        database = temp / "memories.sqlite"
        if not run.replay:
            build_database(database, questions, time.time())
        command = [
            str(binary),
            "--embedding-model",
            str(embedding_model),
            "--backend",
            f"http://127.0.0.1:{shim_port}",
            "--port",
            str(proxy_port),
            "--db",
            str(database),
            "--similarity-threshold",
            "-1",
            "--inject-mode",
            "system",
            "--top-k",
            "5",
            *run.flags,
        ]
        log_path = temp / "proxy.log"
        with log_path.open("w") as log:
            proxy = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            try:
                # Startup backfills (FTS5, dates, sentences) over a large prebuilt
                # database can take minutes, so allow a generous health timeout.
                started = time.monotonic()
                wait_for(f"http://127.0.0.1:{proxy_port}/health", timeout=3600.0)
                LOGGER.info(
                    "%s proxy healthy after %.1fs", run.name, time.monotonic() - started
                )
                if run.replay:
                    expected_memories = 0
                    for index, item in enumerate(questions, start=1):
                        replay_question(proxy_port, item, model)
                        expected_memories += len(item.question.memories)
                        wait_for_rows(database, expected_memories, True)
                        if index % 10 == 0:
                            LOGGER.info(
                                "%s replay: %d/%d", run.name, index, len(questions)
                            )
                    restore_replay_ages(database, questions)
                else:
                    with ThreadPoolExecutor(max_workers=workers) as pool:
                        futures = [
                            pool.submit(
                                post_proxy,
                                proxy_port,
                                item.question.question_id,
                                benchmark_messages(
                                    item, dates[item.question.question_id]
                                ),
                                model,
                            )
                            for item in questions
                        ]
                        for index, future in enumerate(futures, start=1):
                            future.result()
                            if index % 50 == 0:
                                LOGGER.info(
                                    "%s retrieval: %d/%d",
                                    run.name,
                                    index,
                                    len(questions),
                                )
                if run.replay:
                    for item in questions:
                        post_proxy(
                            proxy_port,
                            item.question.question_id,
                            benchmark_messages(item, dates[item.question.question_id]),
                            model,
                        )
            finally:
                proxy.terminate()
                proxy.wait(timeout=60)
                shutil.copy(log_path, f"/tmp/feature-eval-{run.name}.log")
                server.shutdown()
                server.server_close()
    rows: list[dict[str, Any]] = []
    for item in questions:
        key = request_key(benchmark_messages(item, dates[item.question.question_id]))
        record = recorder.records.get(key)
        if record is None:
            raise RuntimeError(
                f"No benchmark request recorded for {item.question.question_id}"
            )
        system = str(record["messages"][0]["content"])
        lines = injected_lines(system)
        session_recall, recalled_fraction = evidence_recall(item, lines)
        answerable = short_answer(item, answers) is not None
        rows.append(
            {
                "question_id": item.question.question_id,
                "question_type": item.question.question_type,
                "answerable": answerable,
                "session_recall": session_recall,
                "evidence_recall": recalled_fraction,
                "injected_tokens": context_tokens(system),
                "injected_lines": len(lines),
                "facts": fact_count(system),
                "correct": is_correct(
                    answers[item.question.question_id], record["reply"]
                )
                if backend is not None and answerable
                else None,
            }
        )
    return rows


def main() -> None:
    """Evaluate selected real-proxy retrieval features and write a JSON report."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--embedding-model", type=Path, required=True)
    parser.add_argument("--model", default="retrieval_only")
    parser.add_argument(
        "--backend", default="", help="chat backend URL; empty means fake"
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--facts-limit", type=int, default=8)
    parser.add_argument("--runs", default=",".join(run.name for run in RUNS))
    parser.add_argument("--proxy-port", type=int, default=DEFAULT_PROXY_PORT)
    parser.add_argument("--shim-port", type=int, default=DEFAULT_SHIM_PORT)
    arguments = parser.parse_args()
    selected_names = [name for name in arguments.runs.split(",") if name]
    selected = [run for run in RUNS if run.name in selected_names]
    unknown = set(selected_names) - {run.name for run in RUNS}
    if unknown:
        parser.error(f"unknown runs: {sorted(unknown)}")
    if "baseline" not in selected_names:
        parser.error("--runs must include baseline")
    records = json.loads(QUESTIONS_PATH.read_text())
    answers = {str(record["question_id"]): str(record["answer"]) for record in records}
    dates = {
        str(record["question_id"]): parse_date(str(record["question_date"])).strftime(
            "%Y-%m-%d"
        )
        for record in records
    }
    questions = [
        item
        for item in embedded_questions([pair_memories(record) for record in records])
        if not item.question.abstention
    ]
    if arguments.limit:
        questions = questions[: arguments.limit]
    backend = arguments.backend or None
    rows_by_run: dict[str, list[dict[str, Any]]] = {}
    for run in selected:
        run_questions = questions[: arguments.facts_limit] if run.facts else questions
        LOGGER.info("Evaluating %s on %d questions", run.name, len(run_questions))
        rows_by_run[run.name] = run_feature(
            run,
            run_questions,
            dates,
            answers,
            arguments.binary,
            arguments.embedding_model,
            backend,
            arguments.model,
            arguments.workers,
            arguments.proxy_port,
            arguments.shim_port,
        )
    baseline_rows = rows_by_run["baseline"]
    settings: dict[str, Any] = {}
    for run in selected:
        rows = rows_by_run[run.name]
        reference = (
            baseline_rows[: len(rows)]
            if len(rows) != len(baseline_rows)
            else baseline_rows
        )
        settings[run.name] = {
            "flags": list(run.flags),
            "questions": len(rows),
            "replay": run.replay,
            "by_type": summarize_rows(rows, reference),
        }
    RESULTS_DIR.mkdir(exist_ok=True)
    output = RESULTS_DIR / f"feature_eval_{output_name(arguments.model)}.json"
    output.write_text(
        json.dumps(
            {
                "model": arguments.model,
                "embedding_model": str(arguments.embedding_model),
                "backend": backend,
                "questions": len(questions),
                "top_k": 5,
                "runs": settings,
            },
            indent=2,
        )
    )
    LOGGER.info("Wrote %s", output)


if __name__ == "__main__":
    main()
